"""
Pipeline de tradução em lotes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

from .anti_hallucination import anti_hallucination_filter
from .cache_utils import (
    cache_exists,
    chunk_hash,
    detect_model_collapse,
    is_duplicate_reuse_safe,
    is_near_duplicate,
    load_cache,
    save_cache,
    set_cache_base_dir,
)
from .config import AppConfig
from .debug_run import DebugRunWriter
from .glossary_utils import format_manual_pairs_for_translation, select_terms_for_chunk
from .language_guardrails import detect_residual_source_language, source_leak_segments
from .languages import (
    compile_term_pattern,
    detect_source_language,
    normalize_source_language,
    source_language_name,
)
from .llm_backend import LLMBackend
from .postprocess import normalize_dialogue_quotes
from .postprocess_translation import postprocess_translation
from .preprocess import (
    chunk_for_translation,
    chunk_for_translation_with_offsets,
    paragraphs_from_text,
    preprocess_text,
)
from .qa import count_quote_lines, has_curly_quote_balance_regression, needs_retry
from .quote_fix import (
    collapse_repeated_curly_quotes,
    count_curly_quotes,
    fix_unbalanced_quotes,
    repair_missing_open_quotes_per_paragraph,
)
from .refine import has_suspicious_repetition  # reuse guardrail
from .repair import REPAIR_PIPELINE_VERSION, repair_prompt_fingerprint, repair_translation_chunk
from .sanitizer import SanitizationReport, log_report, sanitize_translation_output
from .section_splitter import SECTION_PATTERN, split_into_sections
from .text_postprocess import apply_custom_normalizers, apply_structural_normalizers
from .utils import timed

PT_HEADING_RE = re.compile(
    r"^#?\s*(pr[oó]logo|cap[ií]tulo\s+[0-9一二三四五六七八九十百零〇]+(?::[^\n]*)?|ep[ií]logo|p[oó]s[- ]?escrito|posf[aá]cio)\s*$",
    re.IGNORECASE,
)
TRANSLATE_PIPELINE_VERSION = "41"
TRANSLATE_START_MARKER_RE = r"###\s*TEXTO_TRADUZ(?:IDO|DO)?_INICIO"
TRANSLATE_END_MARKER_RE = r"###\s*TEXTO_TRADUZ(?:IDO|DO)?_FIM"
TRANSLATE_ANY_MARKER_RE = r"###\s*TEXTO_TRADUZ[A-Z_]*"
SCENE_SEPARATOR_RE = re.compile(r"^\s*(?:\*\s*){3,}\s*$|^\s*[—–-]{3,}\s*$", re.MULTILINE)


def _remove_premature_curly_close(source_text: str, translated_text: str) -> str:
    """Remove um fechamento precoce quando a estrutura da fonte torna isso inequívoco.

    Alguns modelos fecham uma fala antes de uma risada/interjeição e repetem o
    fechamento no fim do mesmo parágrafo. Só removemos o primeiro fechamento
    quando a fonte está balanceada, a saída preservou todas as aberturas e tem
    exatamente um fechamento extra. Assim não mascaramos uma abertura perdida.
    """
    source_open, source_close = count_curly_quotes(source_text)
    target_open, target_close = count_curly_quotes(translated_text)
    if (
        not source_open
        or source_open != source_close
        or target_open != source_open
        or target_close != source_close + 1
    ):
        return translated_text

    depth = 0
    unmatched_close = None
    for idx, char in enumerate(translated_text):
        if char == "“":
            depth += 1
        elif char == "”":
            if depth:
                depth -= 1
            else:
                unmatched_close = idx
                break
    if unmatched_close is None:
        return translated_text

    paragraph_start = translated_text.rfind("\n\n", 0, unmatched_close) + 2
    previous_close = translated_text.rfind("”", paragraph_start, unmatched_close)
    if previous_close < paragraph_start:
        return translated_text
    between = translated_text[previous_close + 1 : unmatched_close]
    if "“" in between or "\n\n" in between or len(re.sub(r"\W+", "", between)) < 8:
        return translated_text

    return translated_text[:previous_close] + translated_text[previous_close + 1 :]


def _normalize_chunk_dialogue_quotes(source_text: str, translated_text: str) -> str:
    """Normaliza aspas retas da saída quando a estrutura da fonte é inequívoca.

    Alguns modelos fecham uma fala corretamente e deixam uma aspa reta extra no
    fim da narração seguinte. Só removemos esse marcador quando a fonte usa
    aspas curvas balanceadas e a saída tem exatamente uma aspa reta a mais.
    """
    if not translated_text:
        return translated_text

    translated_text = _remove_premature_curly_close(source_text, translated_text)

    source_open, source_close = count_curly_quotes(source_text)
    if (
        not source_open
        or source_open != source_close
        or "“" in translated_text
        or "”" in translated_text
    ):
        return normalize_dialogue_quotes(translated_text)

    positions = [
        idx
        for idx, char in enumerate(translated_text)
        if char == '"'
        and not (idx > 0 and translated_text[idx - 1].isdigit())
        and not (idx + 1 < len(translated_text) and translated_text[idx + 1].isdigit())
    ]
    expected_quotes = source_open + source_close
    if len(positions) == expected_quotes + 1 and not translated_text[positions[-1] + 1 :].strip():
        translated_text = translated_text[: positions[-1]] + translated_text[positions[-1] + 1 :]

    return normalize_dialogue_quotes(translated_text)


def _repair_residual_source_segments(
    translated_text: str,
    *,
    backend: LLMBackend,
    cfg: AppConfig,
    logger: logging.Logger,
    glossary_text: str | None,
    allow_adaptation: bool,
    source_language: str,
    temperature: float,
    label: str,
) -> tuple[str, int, int]:
    """Retraduz apenas segmentos residuais da origem, sem reescrever o chunk inteiro."""
    source_name = source_language_name(source_language)
    repaired = translated_text
    attempts_total = 0
    replacements = 0
    blocks: list[str] = []
    seen_blocks: set[str] = set()
    for segment in source_leak_segments(translated_text, source_language):
        if not segment.strip():
            continue
        position = translated_text.find(segment)
        if position < 0:
            continue
        block_start = translated_text.rfind("\n\n", 0, position) + 2
        block_end = translated_text.find("\n\n", position)
        if block_end < 0:
            block_end = len(translated_text)
        block = translated_text[block_start:block_end].strip()
        if not block or block in seen_blocks:
            continue
        seen_blocks.add(block)
        blocks.append(block)

    for segment_index, segment in enumerate(blocks, start=1):
        if segment not in repaired:
            continue
        prompt = build_translation_prompt(
            segment,
            context=None,
            glossary_text=glossary_text,
            allow_adaptation=allow_adaptation,
            chunk_profile="dialogue",
            source_language=source_language,
        )
        prompt += (
            f"\n\nATENÇÃO: Este segmento ainda contém {source_name}. "
            "Traduza-o integralmente para PT-BR, preservando aspas, hesitações e pontuação. "
            f"Não devolva texto em {source_name} nem explicações."
        )
        previous_temperature = backend.temperature
        backend.temperature = temperature
        try:
            raw_text, _clean_text, attempts, _report = _call_with_retry(
                backend=backend,
                prompt=prompt,
                cfg=cfg,
                logger=logger,
                label=f"{label}-{segment_index}",
            )
        except Exception as exc:
            logger.warning("Fallback de idioma residual falhou no bloco %d: %s", segment_index, exc)
            continue
        finally:
            backend.temperature = previous_temperature
        attempts_total += attempts

        candidate = _strip_translate_markers(_parse_translation_output(raw_text))
        candidate, _report = sanitize_translation_output(
            candidate, logger=logger, fail_on_contamination=False
        )
        candidate = anti_hallucination_filter(
            orig=segment, llm_raw=raw_text, cleaned=candidate, mode="translate"
        )
        candidate = postprocess_translation(candidate, segment)
        candidate = _normalize_chunk_dialogue_quotes(segment, candidate)
        candidate, _ = collapse_repeated_curly_quotes(candidate)
        candidate_whole = repaired.replace(segment, candidate, 1)
        candidate_has_source, _reason = detect_residual_source_language(candidate, source_language)
        invalid_reason = ""
        if not candidate.strip():
            invalid_reason = "empty"
        elif candidate_has_source:
            invalid_reason = "residual_source_language"
        elif has_curly_quote_balance_regression(repaired, candidate_whole):
            invalid_reason = f"quote_balance {count_curly_quotes(repaired)}->{count_curly_quotes(candidate_whole)}"
        elif len(candidate.strip()) < len(segment.strip()) * 0.45:
            invalid_reason = "too_short"
        if invalid_reason:
            logger.warning(
                "Fallback de idioma residual não produziu substituição válida no segmento %d: %s.",
                segment_index,
                invalid_reason,
            )
            continue
        repaired = candidate_whole
        replacements += 1

    return repaired, attempts_total, replacements


def translation_prompt_fingerprint(*, allow_adaptation: bool, source_language: str = "en") -> str:
    """Calcula a assinatura da política de prompt usada na tradução."""
    template = build_translation_prompt(
        "{chunk}",
        context="{context}",
        glossary_text="{glossary}",
        allow_adaptation=allow_adaptation,
        chunk_profile="mixed",
        source_language=source_language,
    )
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _extract_last_sentence(text: str) -> str:
    """Extrai a ultima frase simples (delimitada por .!?) e limpa marcadores."""
    cleaned = re.sub(TRANSLATE_ANY_MARKER_RE, "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    for part in reversed(parts):
        candidate = part.strip().strip("#").strip()
        if candidate:
            return candidate
    return ""


def _split_context_paragraphs(text: str) -> list[str]:
    """Divide contexto parágrafos."""
    return [part.strip() for part in re.split(r"\n\s*\n+", text or "") if part.strip()]


def _tail_paragraphs_from_entries(
    entries: list[dict[str, str]], key: str, max_paragraphs: int, max_chars: int
) -> list[str]:
    """Seleciona os últimos parágrafos das traduções anteriores."""
    if max_paragraphs <= 0 or max_chars <= 0:
        return []
    selected: list[str] = []
    remaining = max_chars
    for entry in reversed(entries):
        paragraphs = _split_context_paragraphs(entry.get(key, ""))
        for paragraph in reversed(paragraphs):
            if len(selected) >= max_paragraphs or remaining <= 0:
                break
            clean = re.sub(r"\s+", " ", paragraph).strip()
            if not clean:
                continue
            if len(clean) > remaining:
                clean = clean[-remaining:].lstrip()
            selected.append(clean)
            remaining -= len(clean)
        if len(selected) >= max_paragraphs or remaining <= 0:
            break
    return list(reversed(selected))


def build_recent_translation_context(
    entries: list[dict[str, str]],
    *,
    max_paragraphs: int = 3,
    max_chars: int = 1200,
    include_pt: bool = True,
) -> str:
    """Monta recente tradução contexto."""
    source_paragraphs = _tail_paragraphs_from_entries(entries, "source", max_paragraphs, max_chars)
    target_paragraphs = (
        _tail_paragraphs_from_entries(entries, "target", max_paragraphs, max_chars)
        if include_pt
        else []
    )
    if not source_paragraphs and not target_paragraphs:
        return ""
    parts: list[str] = []
    if source_paragraphs:
        parts.append("ORIGINAL ANTERIOR (somente contexto, nao traduzir):")
        parts.append("\n\n".join(source_paragraphs))
    if target_paragraphs:
        parts.append(
            "TRADUCAO PT-BR ANTERIOR (somente consistencia de tom/termos; nao reescrever):"
        )
        parts.append("\n\n".join(target_paragraphs))
    return "\n\n".join(parts).strip()


def classify_translation_chunk(text: str) -> str:
    """Classifica tradução chunk."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return "narration"
    dialogue_lines = 0
    dialogue_chars = 0
    for line in lines:
        is_dialogue = bool(re.match(r'^(?:["“”]|[—–-]\s)', line))
        is_dialogue = is_dialogue or bool(re.search(r'["“][^"”]{2,}["”]', line))
        if is_dialogue:
            dialogue_lines += 1
            dialogue_chars += len(line)
    total_chars = sum(len(line) for line in lines) or 1
    line_ratio = dialogue_lines / max(len(lines), 1)
    char_ratio = dialogue_chars / total_chars
    if line_ratio >= 0.40 or char_ratio >= 0.45:
        return "dialogue"
    if line_ratio <= 0.15 and char_ratio <= 0.20:
        return "narration"
    return "mixed"


def starts_with_scene_boundary(text: str) -> bool:
    """Verifica se o trecho começa em um limite de cena."""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return bool(SCENE_SEPARATOR_RE.fullmatch(stripped))
    return False


def ends_with_scene_boundary(text: str) -> bool:
    """Verifica se o trecho termina em um limite de cena."""
    for line in reversed((text or "").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        return bool(SCENE_SEPARATOR_RE.fullmatch(stripped))
    return False


def _source_specific_translation_rules(source_language: str) -> str:
    """Retorna cuidados de tradução relevantes para o idioma de origem."""
    language = normalize_source_language(source_language)
    rules = {
        "en": (
            "- Evite calques sintáticos e converta possessivos para construções naturais em PT-BR.\n"
            '- Exemplo: "Mara\'s group" deve ser traduzido como "grupo da Mara".'
        ),
        "ja": (
            "- Preserve honoríficos quando forem relevantes à relação entre personagens.\n"
            "- O japonês frequentemente omite sujeito, número e gênero; use o contexto e não invente "
            "informações ausentes."
        ),
        "ko": (
            "- Preserve a intenção dos níveis de formalidade e dos honoríficos em fala natural.\n"
            "- Não explicite sujeito ou gênero quando o coreano e o contexto não os determinarem."
        ),
        "zh": (
            "- Trate títulos, parentesco e formas de tratamento de modo consistente com o glossário.\n"
            "- Resolva sujeitos elípticos pelo contexto sem acrescentar gênero ou informação narrativa."
        ),
    }
    return rules.get(
        language,
        "- Respeite ambiguidades do idioma de origem e evite estruturas calcadas na sua gramática.",
    )


def build_translation_prompt(
    chunk: str,
    context: str | None = None,
    glossary_text: str | None = None,
    allow_adaptation: bool = False,
    chunk_profile: str = "mixed",
    source_language: str = "en",
) -> str:
    """Monta o prompt de tradução com idioma, contexto e glossário opcional."""
    language = normalize_source_language(source_language)
    source_name = source_language_name(language)
    source_rules = _source_specific_translation_rules(language)

    context_block = ""
    if context:
        context_block = (
            "CONTEXTO RECENTE (NÃO TRADUZIR NEM REESCREVER; use apenas para coerência):\n"
            f'"{context.strip()}"\n\n'
        )

    glossary_block = ""
    if glossary_text:
        glossary_block = (
            "SIGA EXATAMENTE AS TRADUÇÕES DO GLOSSÁRIO ABAIXO.\n"
            "Não crie variantes, não altere nomes próprios e não adicione explicações.\n"
            f"{glossary_text}\n\n"
        )

    adaptation_block = ""
    if allow_adaptation:
        adaptation_block = (
            "\n- Em piadas e trocadilhos sem equivalente literal, preserve o efeito narrativo, "
            "o registro e a intenção sem acrescentar uma nova piada.\n"
        )

    profile = (chunk_profile or "mixed").strip().lower()
    if profile == "dialogue":
        profile_block = """
FOCO DO TRECHO: DIÁLOGO.
- Priorize fala natural em PT-BR, preservando intenção, tensão, hesitações e personalidade.
- Não deixe a fala excessivamente formal se o original for casual.
- Preserve insultos, medo, humor e gírias relevantes para a voz do personagem.
- Preserve o estilo de marcação de diálogo do trecho de entrada.
"""
    elif profile == "narration":
        profile_block = """
FOCO DO TRECHO: NARRAÇÃO/DESCRIÇÃO.
- Priorize fluidez literária, clareza espacial, continuidade de ação e tempo verbal consistente.
- Evite calques do idioma de origem e excesso de oralidade na narração.
- Preserve o tom emocional da cena sem embelezar além do original.
"""
    else:
        profile_block = """
FOCO DO TRECHO: MISTO.
- Diferencie naturalmente narração e falas: narração fluida, diálogos vivos.
- Preserve transições entre ação, pensamento e fala sem reorganizar parágrafos.
"""

    return f"""
Você é um tradutor literário profissional, especializado em {source_name} → português brasileiro.
Traduza o texto integralmente, sem alterar eventos, ordem narrativa, personalidade ou conteúdo.

REGRAS:
1. Preserve fielmente o sentido, o tom e o registro.
2. Não resuma, não pule frases e não adicione conteúdo.
3. Mantenha nomes e termos conforme o glossário.
4. Preserve número, pessoa, referentes e ambiguidades; não troque singular por plural nem o inverso.
5. Use reticências somente quando elas existirem no original.
6. Remova apenas ruído inequívoco de OCR ou PDF.
7. Produza português brasileiro natural, sem explicar suas escolhas.

CUIDADOS DO IDIOMA DE ORIGEM:
{source_rules}
{adaptation_block}
{profile_block}
REVISÃO SILENCIOSA OBRIGATÓRIA:
- Confira se cada frase tem correspondente e se sujeito, referente, número e gênero não mudaram.
- Não use "vocês" quando a origem se refere a uma única pessoa.
- Evite calques como "tomar ações" quando uma construção natural, como "agir", preserva o sentido.
- Corrija concordância, regência e tempo verbal em PT-BR.
- Não inclua checklist, notas ou comentários na saída.

FORMATO DE SAÍDA:
### TEXTO_TRADUZIDO_INICIO

<texto traduzido>
### TEXTO_TRADUZIDO_FIM

Nada antes ou depois dos marcadores.

{glossary_block}{context_block}TEXTO A SER TRADUZIDO:
\"\"\"{chunk}\"\"\""""


def _parse_translation_output(raw: str) -> str:
    """Extrai bloco entre TEXTO_TRADUZIDO_INICIO/FIM; se ausentes, retorna o texto util."""
    match = re.search(
        rf"{TRANSLATE_START_MARKER_RE}\s*(.*?)(?:{TRANSLATE_END_MARKER_RE}\s*|$)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate

    end_only = re.search(rf"^(.*?){TRANSLATE_END_MARKER_RE}", raw, flags=re.IGNORECASE | re.DOTALL)
    if end_only:
        candidate = end_only.group(1).strip()
        if candidate:
            return candidate

    cleaned = raw.strip()
    if cleaned:
        return cleaned
    raise ValueError("Saida vazia apos tentar extrair texto traduzido.")


def source_heading_to_pt(title: str | None) -> str | None:
    """Converte um título estrutural conhecido em heading Markdown em PT-BR."""
    if not title:
        return None
    clean = str(title).strip().strip("#").strip()
    if not clean or clean.casefold() == "full text":
        return None

    if re.fullmatch(
        r"(?:prologue|pr[oó]logo|prolog|序章|前言|プロローグ|프롤로그)",
        clean,
        flags=re.IGNORECASE,
    ):
        return "# Prólogo"
    if re.fullmatch(
        r"(?:epilogue|ep[ií]logo|epilog|終章|终章|エピローグ|에필로그)",
        clean,
        flags=re.IGNORECASE,
    ):
        return "# Epílogo"
    if re.fullmatch(
        r"(?:afterword|posf[aá]cio|p[oó]s[- ]?escrito|あとがき|後書き|后记|後記|후기)",
        clean,
        flags=re.IGNORECASE,
    ):
        return "# Pós-escrito"

    number_pattern = r"(\d+|[一二三四五六七八九十百零〇]+)"
    latin_match = re.fullmatch(
        rf"(?:chapter|cap[ií]tulo|chapitre|kapitel|capitolo|glava|глава)\s+"
        rf"{number_pattern}(?:\s*[:：–—-]\s*(.*))?",
        clean,
        flags=re.IGNORECASE,
    )
    asian_match = re.fullmatch(
        rf"第\s*{number_pattern}\s*[章話话巻卷](?:\s*[:：–—-]?\s*(.*))?",
        clean,
    )
    korean_match = re.fullmatch(
        r"제\s*(\d+|[일이삼사오육칠팔구십백]+)\s*장(?:\s*[:：–—-]?\s*(.*))?",
        clean,
    )
    match = latin_match or asian_match or korean_match
    if not match:
        return None
    number = match.group(1)
    subtitle = (match.group(2) or "").strip()
    if subtitle:
        suffix = f": {subtitle}"
    elif re.search(r"[:：]\s*$", clean):
        suffix = ":"
    else:
        suffix = ""
    return f"# Capítulo {number}{suffix}"


def ensure_section_heading(output: str, source_title: str | None) -> tuple[str, bool]:
    """Garante que a primeira saída da seção preserve seu heading estrutural."""
    heading = source_heading_to_pt(source_title)
    if not heading:
        return output, False
    lines = output.splitlines()
    first_idx = next((idx for idx, line in enumerate(lines) if line.strip()), None)
    if first_idx is not None:
        first = lines[first_idx].strip()
        if PT_HEADING_RE.match(first):
            return output, False
    stripped = output.strip()
    if not stripped:
        return heading, True
    return f"{heading}\n\n{stripped}", True


def _strip_translate_markers(text: str) -> str:
    """Remove qualquer linha/bloco com marcadores TEXTO_TRADUZIDO_ remanescentes."""
    lines = []
    for ln in text.splitlines():
        if re.search(TRANSLATE_ANY_MARKER_RE, ln, flags=re.IGNORECASE):
            continue
        lines.append(ln)
    cleaned = "\n".join(lines)
    cleaned = re.sub(
        rf"{TRANSLATE_START_MARKER_RE}.*?({TRANSLATE_END_MARKER_RE})?",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def _split_dialogue_blocks(chunk: str) -> list[str]:
    """
    Divide o chunk em blocos menores priorizando linhas de diálogo (aspas/travessão).
    """
    blocks: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        """Consolida o bloco acumulado antes de continuar o processamento."""
        nonlocal buffer
        if buffer:
            joined = "\n".join(buffer).strip()
            if joined:
                blocks.append(joined)
            buffer = []

    for raw_line in chunk.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith(('"', "“", "”", "’", "-", "—")):
            flush()
            blocks.append(line)
        else:
            buffer.append(line)
    flush()
    return [b for b in blocks if b.strip()]


def _is_stub_chunk(text: str) -> bool:
    """Indica se simulado chunk."""
    stripped = text.strip()
    if not stripped:
        return True
    if SECTION_PATTERN.fullmatch(stripped.lstrip("#").strip()):
        return True
    return False


def _is_short_dialogue_line(line: str) -> bool:
    """Indica se curtos diálogo linha."""
    ln = line.strip()
    if not ln:
        return False
    if re.match(r"^[\"“].+[\"”]$", ln):
        return True
    words = ln.split()
    if not words or len(words) > 6:
        return False
    if re.search(r"[?!…\.][\"”]?$", ln):
        return True
    return False


def _separate_short_dialogues(text: str) -> str:
    """Insere linha em branco entre falas curtíssimas consecutivas para preservar parágrafos."""
    lines = text.splitlines()
    new_lines: list[str] = []
    prev_short = False
    for ln in lines:
        current_short = _is_short_dialogue_line(ln)
        if current_short and prev_short:
            if new_lines and new_lines[-1].strip() != "":
                new_lines.append("")
        new_lines.append(ln)
        prev_short = current_short if ln.strip() else False
    return "\n".join(new_lines)


def _build_chunk_glossary(
    manual_terms: list[dict] | None,
    chunk: str,
    *,
    match_limit: int,
    fallback_limit: int,
    logger: logging.Logger,
    chunk_index: int,
) -> tuple[str | None, int, int, list[dict]]:
    """
    Seleciona termos do glossário que aparecem no chunk.
    Retorna (glossary_text, matched_count, total_injetados).
    """
    if not manual_terms:
        return None, 0, 0, []
    selected, matched = select_terms_for_chunk(
        manual_terms,
        chunk,
        match_limit=match_limit,
        fallback_limit=fallback_limit,
    )
    glossary_text = format_manual_pairs_for_translation(selected, limit=len(selected) or None)
    injected = len(selected)
    if injected:
        logger.debug(
            "Glossario (chunk %d): injetados=%d matched=%d",
            chunk_index,
            injected,
            matched,
        )
    return glossary_text or None, matched, injected, selected


def _glossary_chunk_manifest(
    *,
    glossary_text: str | None,
    matched_count: int,
    injected_count: int,
    selected_terms: list[dict],
    enforced_replacements: dict[str, int] | None = None,
) -> dict:
    """Monta o registro dos termos de glossário aplicados ao chunk."""
    if matched_count > 0:
        selection_mode = "matched"
    elif selected_terms:
        selection_mode = "fallback"
    elif glossary_text:
        selection_mode = "static"
    else:
        selection_mode = "none"

    terms = []
    for term in selected_terms:
        terms.append(
            {
                "key": str(term.get("key", "")).strip(),
                "pt": str(term.get("pt", "")).strip(),
                "category": term.get("category"),
                "enforce": bool(term.get("enforce")),
            }
        )

    return {
        "enabled": bool(glossary_text),
        "selection_mode": selection_mode,
        "matched_count": matched_count,
        "injected_count": injected_count if glossary_text else 0,
        "fallback_used": selection_mode == "fallback",
        "terms": terms,
        "prompt_hash": chunk_hash(glossary_text) if glossary_text else None,
        "prompt_chars": len(glossary_text or ""),
        "enforced_replacements": enforced_replacements or {},
    }


def enforce_canonical_terms(text: str, terms: list[dict]) -> tuple[str, dict]:
    """Substitui termos de aplicação obrigatória pelos equivalentes canônicos em português."""
    if not text or not terms:
        return text, {}
    replacements: dict[str, int] = {}
    for term in terms:
        if not term:
            continue
        pt = str(term.get("pt", "")).strip()
        if not pt:
            continue
        target_replacements = term.get("target_replacements") or {}
        if isinstance(target_replacements, dict):
            for alias, replacement in target_replacements.items():
                alias_s = str(alias).strip()
                replacement_s = str(replacement).strip()
                if (
                    not alias_s
                    or not replacement_s
                    or alias_s.casefold() == replacement_s.casefold()
                ):
                    continue
                pattern = compile_term_pattern(alias_s)
                text, count = pattern.subn(replacement_s, text)
                if count:
                    replacements[alias_s] = replacements.get(alias_s, 0) + count
        variants: list[str] = []
        key = str(term.get("key", "")).strip()
        canonical_same_as_source = key.casefold() == pt.casefold()
        if term.get("enforce"):
            if key and not canonical_same_as_source:
                variants.append(key)
                aliases = term.get("source_aliases") or term.get("aliases") or []
                if isinstance(aliases, list):
                    variants.extend(str(a).strip() for a in aliases if str(a).strip())
        bad_aliases = term.get("bad_aliases") or term.get("forbidden_aliases") or []
        if isinstance(bad_aliases, str):
            bad_aliases = [bad_aliases]
        if isinstance(bad_aliases, list):
            variants.extend(str(a).strip() for a in bad_aliases if str(a).strip())
        ordered_variants: list[str] = []
        seen_variants: set[str] = set()
        for variant in sorted(variants, key=len, reverse=True):
            marker = variant.casefold()
            if not variant or marker == pt.casefold() or marker in seen_variants:
                continue
            seen_variants.add(marker)
            ordered_variants.append(variant)
        for variant in ordered_variants:
            pattern = compile_term_pattern(variant)
            text, count = pattern.subn(pt, text)
            if count:
                replacements[variant] = replacements.get(variant, 0) + count
    return text, replacements


def translate_document(
    pdf_text: str,
    backend: LLMBackend,
    cfg: AppConfig,
    logger: logging.Logger,
    source_slug: str | None = None,
    progress_path: Path | None = None,
    resume_manifest: dict | None = None,
    glossary_text: str | None = None,
    glossary_manual_terms: list[dict] | None = None,
    debug_translation: bool = False,
    parallel_workers: int = 1,
    debug_chunks: bool = False,
    already_preprocessed: bool = False,
    split_by_sections: bool | None = None,
    allow_adaptation: bool | None = None,
    translation_repair: bool | None = None,
    fail_on_chunk_error: bool | None = None,
    debug_run: DebugRunWriter | None = None,
    source_language: str | None = None,
) -> str:
    """Executa pré-processamento opcional, divisão em chunks, tradução e sanitização."""
    set_cache_base_dir(cfg.output_dir)
    split_flag = cfg.split_by_sections if split_by_sections is None else split_by_sections
    allow_adapt_flag = (
        cfg.translate_allow_adaptation if allow_adaptation is None else allow_adaptation
    )
    repair_enabled = (
        getattr(cfg, "use_translation_repair", True)
        if translation_repair is None
        else translation_repair
    )
    fail_on_error = cfg.fail_on_chunk_error if fail_on_chunk_error is None else fail_on_chunk_error
    if not hasattr(backend, "temperature"):
        backend.temperature = cfg.translate_temperature
    clean = (
        pdf_text
        if already_preprocessed
        else preprocess_text(pdf_text, logger, skip_front_matter=cfg.skip_front_matter)
    )
    requested_source_language = source_language or cfg.source_language
    resolved_source_language = detect_source_language(clean, requested_source_language)
    logger.info(
        "Idioma de origem: %s (%s).",
        source_language_name(resolved_source_language),
        resolved_source_language,
    )
    clean = _separate_short_dialogues(clean)
    clean, source_quote_repairs = repair_missing_open_quotes_per_paragraph(
        clean, logger=logger, label="source"
    )
    clean, source_quote_boundary_fixed = fix_unbalanced_quotes(clean, logger=logger, label="source")
    source_quote_boundary_fixed = source_quote_boundary_fixed or bool(source_quote_repairs)
    if source_quote_boundary_fixed:
        logger.info("Fronteiras de aspas do texto-fonte foram restauradas antes da tradução.")
    doc_hash = chunk_hash(clean)
    sections = split_into_sections(clean) if split_flag else [{"title": "Full Text", "body": clean}]
    if split_flag:
        total_sections = len(sections)
        heading_starts = [
            s.get("start_idx", 0)
            for s in sections
            if s.get("title") and s.get("title") != "Full Text"
        ]
        first_heading = min(heading_starts) if heading_starts else 0
        late_first = first_heading > len(clean) * 0.05
        long_text = len(clean) > 50000
        if total_sections <= 2 and (late_first or long_text):
            logger.warning(
                "split_by_sections fallback: disabling section split (sections=%d, first_heading=%d len=%d)",
                total_sections,
                first_heading,
                len(clean),
            )
            sections = [
                {"title": "Full Text", "body": clean, "start_idx": 0, "end_idx": len(clean)}
            ]
            split_flag = False
    chunk_records = []
    original_paragraphs_total = 0
    sections_debug = []
    for sidx, sec in enumerate(sections, start=1):
        paragraphs = paragraphs_from_text(sec["body"])
        original_paragraphs_total += len(paragraphs)
        if debug_run:
            sec_chunks = chunk_for_translation_with_offsets(
                paragraphs, max_chars=cfg.translate_chunk_chars, logger=logger
            )
        else:
            sec_chunks = [
                (c, None, None)
                for c in chunk_for_translation(
                    paragraphs, max_chars=cfg.translate_chunk_chars, logger=logger
                )
            ]
        section_start = sec.get("start_idx")
        section_end = sec.get("end_idx")
        if debug_run:
            sections_debug.append(
                {
                    "title": sec.get("title", ""),
                    "start_idx": section_start,
                    "end_idx": section_end,
                    "chars": len(sec.get("body", "")),
                    "is_toc_stub": _is_stub_chunk(sec.get("body", "")),
                }
            )
        for ch, start_offset, end_offset in sec_chunks:
            global_start = (
                (section_start + start_offset)
                if section_start is not None and start_offset is not None
                else None
            )
            global_end = (
                (section_start + end_offset)
                if section_start is not None and end_offset is not None
                else None
            )
            chunk_records.append(
                {
                    "section": sidx,
                    "title": sec.get("title", ""),
                    "text": ch,
                    "start_offset": global_start,
                    "end_offset": global_end,
                }
            )
    sanitized_records: list[dict] = []
    for i, rec in enumerate(chunk_records):
        text = rec.get("text", "")
        if _is_stub_chunk(text):
            if i + 1 < len(chunk_records):
                chunk_records[i + 1]["text"] = (
                    text.strip() + "\n\n" + chunk_records[i + 1]["text"]
                ).strip()
                if rec.get("start_offset") is not None:
                    chunk_records[i + 1]["start_offset"] = rec.get("start_offset")
                logger.warning(
                    "chunk_toc_stub: merged chunk %d (title=%s, len=%d) into next chunk",
                    i + 1,
                    rec.get("title", ""),
                    len(text.strip()),
                )
                continue
            if not sanitized_records:
                logger.warning(
                    "chunk_toc_stub: keeping last stub chunk %d (title=%s, len=%d) because it is the only chunk",
                    i + 1,
                    rec.get("title", ""),
                    len(text.strip()),
                )
            else:
                logger.warning(
                    "chunk_toc_stub: dropped last stub chunk %d (title=%s, len=%d)",
                    i + 1,
                    rec.get("title", ""),
                    len(text.strip()),
                )
                continue
        sanitized_records.append(rec)
    chunk_records = sanitized_records
    chunks = [c["text"] for c in chunk_records]
    if debug_run:
        debug_run.write_json("30_split_chunk/sections.json", sections_debug)
        for idx, rec in enumerate(chunk_records, start=1):
            if debug_run.should_write_chunk(idx):
                debug_run.append_jsonl(
                    "30_split_chunk/chunks.jsonl",
                    {
                        "chunk_index": idx,
                        "section_index": rec.get("section"),
                        "section_title": rec.get("title", ""),
                        "start_offset": rec.get("start_offset"),
                        "end_offset": rec.get("end_offset"),
                        "input_hash": debug_run.sha256_text(rec.get("text", "")),
                        "chars_in": len(rec.get("text", "")),
                    },
                )
    max_chunk_len = max((len(c["text"]) for c in chunk_records), default=0)
    logger.info(
        "Iniciando traducao: %d chunks (alvo=%d, max_observado=%d)",
        len(chunks),
        cfg.translate_chunk_chars,
        max_chunk_len,
    )
    if parallel_workers > 1:
        logger.info("Context chaining ativo; paralelismo ajustado para 1 na tradução.")
        parallel_workers = 1
    state_path = Path(cfg.output_dir) / "state_traducao.json"
    dialogue_guardrails_mode = getattr(cfg, "translate_dialogue_guardrails", "strict")
    dialogue_split_fallback = getattr(cfg, "translate_dialogue_split_fallback", True)
    dialogue_retry_temps = getattr(cfg, "translate_dialogue_retry_temps", None) or [
        cfg.translate_temperature,
        0.25,
        0.10,
    ]
    context_paragraphs = max(0, int(getattr(cfg, "translate_context_paragraphs", 3) or 0))
    context_chars = max(0, int(getattr(cfg, "translate_context_chars", 1200) or 0))
    context_include_pt = bool(getattr(cfg, "translate_context_include_pt", True))
    glossary_match_limit = getattr(cfg, "translate_glossary_match_limit", 80)
    glossary_fallback_limit = getattr(cfg, "translate_glossary_fallback_limit", 30)
    try:
        state_payload = {
            "input_file": source_slug or "document",
            "hash": doc_hash,
            "timestamp": datetime.now().isoformat(),
            "total_chunks": len(chunks),
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

    if cfg.dump_chunks and chunks:
        slug = source_slug or "document"
        debug_path = Path(cfg.output_dir) / f"{slug}_chunks_debug.md"
        total = len(chunks)
        parts = []
        for idx, chunk in enumerate(chunks, start=1):
            parts.append(f"=== CHUNK {idx}/{total} ===")
            parts.append(chunk)
            parts.append("")  # linha em branco entre chunks
        debug_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
        logger.info("Chunks salvos em %s", debug_path)

    translated_chunks: List[str] = []
    total_chunks = len(chunks)
    translated_ok: set[int] = set()
    failed_chunks: set[int] = set()
    chunk_outputs: dict[int, str] = {}
    processed_indices: set[int] = set()
    cache_hits = 0
    fallbacks = 0
    collapse_detected = 0
    duplicate_reuse = 0
    seen_chunks: list[tuple[str, str]] = []
    contamination_count = 0
    error_count = 0
    orig_chars_total = 0
    sanitized_chars_total = 0
    normalization_totals = {"dialogue_splits": 0, "triple_quotes_removed": 0}
    paragraph_mismatch: dict[str, int] | None = None
    chunk_metrics: list[dict] = []
    repair_metrics: list[dict] = []
    repair_attempted_total = 0
    repair_changed_total = 0
    repair_cache_hits_total = 0
    repair_suspect_total = 0
    repair_elapsed_total = 0.0

    glossary_hash = chunk_hash(glossary_text) if glossary_text else None
    manual_glossary_hash = (
        chunk_hash(json.dumps(glossary_manual_terms, ensure_ascii=False, sort_keys=True))
        if glossary_manual_terms
        else None
    )
    chunk_hashes = [chunk_hash(c) for c in chunks]
    prompt_hash = translation_prompt_fingerprint(
        allow_adaptation=allow_adapt_flag, source_language=resolved_source_language
    )
    current_cache_signature = {
        "backend": getattr(backend, "backend", None),
        "model": getattr(backend, "model", None),
        "num_predict": getattr(backend, "num_predict", None),
        "temperature": getattr(backend, "temperature", None),
        "repeat_penalty": getattr(backend, "repeat_penalty", None),
        "translate_chunk_chars": cfg.translate_chunk_chars,
        "glossary_hash": glossary_hash,
        "manual_glossary_hash": manual_glossary_hash,
        "doc_hash": doc_hash,
        "source": source_slug or "",
        "source_language": resolved_source_language,
        "allow_adaptation": allow_adapt_flag,
        "translation_repair": bool(repair_enabled),
        "repair_prompt_hash": repair_prompt_fingerprint(resolved_source_language)
        if repair_enabled
        else None,
        "repair_pipeline_version": REPAIR_PIPELINE_VERSION if repair_enabled else None,
        "translate_context_paragraphs": context_paragraphs,
        "translate_context_chars": context_chars,
        "translate_context_include_pt": context_include_pt,
        "source_quote_boundary_fixed": source_quote_boundary_fixed,
        "split_by_sections": split_flag,
        "dialogue_guardrails_mode": dialogue_guardrails_mode,
        "prompt_hash": prompt_hash,
        "pipeline_version": TRANSLATE_PIPELINE_VERSION,
    }

    def _is_cache_compatible(data: dict) -> bool:
        """Verifica se a entrada de cache corresponde à política atual."""
        meta = data.get("metadata")
        if not isinstance(meta, dict):
            return False
        return all(meta.get(k) == v for k, v in current_cache_signature.items())

    if resume_manifest:
        manifest_doc_hash = resume_manifest.get("doc_hash")
        if manifest_doc_hash and manifest_doc_hash != doc_hash:
            logger.warning(
                "Manifesto de progresso pertence a outro documento (doc_hash diferente); ignorando resume.",
            )
            resume_manifest = None

    if resume_manifest:
        manifest_total = resume_manifest.get("total_chunks")
        if isinstance(manifest_total, int) and manifest_total != total_chunks:
            logger.warning(
                "Manifesto indica %d chunks, mas chunking atual gerou %d; usando chunking atual.",
                manifest_total,
                total_chunks,
            )
        raw_chunks = resume_manifest.get("chunks") or {}
        raw_hashes = resume_manifest.get("chunk_hashes") or {}
        if isinstance(raw_chunks, dict):
            for key, val in raw_chunks.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                if isinstance(val, str):
                    expected_hash = (
                        raw_hashes.get(str(idx)) if isinstance(raw_hashes, dict) else None
                    )
                    current_hash = chunk_hashes[idx - 1] if 0 < idx <= len(chunk_hashes) else None
                    if expected_hash and current_hash and expected_hash != current_hash:
                        logger.warning(
                            "Manifesto resume ignorado para chunk %d: hash diferente do input atual.",
                            idx,
                        )
                        continue
                    chunk_outputs[idx] = val

        raw_translated = resume_manifest.get("translated_chunks") or []
        for idx in raw_translated:
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            if idx_int in chunk_outputs:
                translated_ok.add(idx_int)
            else:
                logger.warning(
                    "Manifesto marca chunk %s como traduzido, mas nao ha conteudo salvo; retraduzindo.",
                    idx_int,
                )

        raw_failed = resume_manifest.get("failed_chunks") or []
        for idx in raw_failed:
            try:
                failed_chunks.add(int(idx))
            except (TypeError, ValueError):
                continue

    def _write_progress() -> None:
        """Grava progresso."""
        if progress_path is None:
            return
        data = {
            "total_chunks": total_chunks,
            "translated_chunks": sorted(translated_ok),
            "failed_chunks": sorted(failed_chunks),
            "timestamp": datetime.now().isoformat(),
            "doc_hash": doc_hash,
            "chunk_hashes": {str(i + 1): h for i, h in enumerate(chunk_hashes)},
            "chunks": {str(idx): text for idx, text in chunk_outputs.items()},
        }
        try:
            progress_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover - I/O edge case
            logger.warning("Falha ao gravar manifesto de progresso em %s: %s", progress_path, exc)

    _write_progress()

    previous_context: str | None = None
    current_section: int | None = None
    context_entries: list[dict[str, str]] = []
    debug_dir = Path(cfg.output_dir) / "debug_traducao"
    translate_manifest_chunks: list[dict] = []

    debug_file = None
    debug_file_path: Path | None = None
    if debug_chunks:
        debug_dir_path = Path(cfg.output_dir)
        debug_dir_path.mkdir(parents=True, exist_ok=True)
        base = source_slug or "document"
        debug_file_path = debug_dir_path / f"{base}_pt_chunks_debug.jsonl"
        debug_file = debug_file_path.open("w", encoding="utf-8")

    def _write_chunk_debug(entry: dict) -> None:
        """Grava chunk depuração."""
        if debug_file:
            debug_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _count_quotes(txt: str) -> int:
        """Conta aspas."""
        return sum(1 for ch in txt if ch in {'"', "“", "”", "'"})

    for idx, chunk_info in enumerate(chunk_records, start=1):
        chunk = chunk_info["text"]
        suspect_output = False
        suspect_reason = ""
        if _is_stub_chunk(chunk):
            logger.warning(
                "chunk_toc_stub: skipping chunk %d/%d (title=%s, len=%d)",
                idx,
                total_chunks,
                chunk_info.get("title", ""),
                len(chunk.strip()),
            )
            translated_ok.add(idx)
            processed_indices.add(idx)
            chunk_outputs[idx] = ""
            _write_progress()
            if debug_run and debug_run.should_write_chunk(idx):
                debug_stage_dir = debug_run.stage_dir("40_translate") / "debug_traducao"
                outputs_payload = {
                    "debug_original": debug_run.rel_path(
                        debug_stage_dir / f"chunk{idx:03d}_original_en.txt"
                    ),
                    "debug_context": debug_run.rel_path(
                        debug_stage_dir / f"chunk{idx:03d}_context.txt"
                    ),
                    "debug_glossary": debug_run.rel_path(
                        debug_stage_dir / f"chunk{idx:03d}_glossary.txt"
                    ),
                    "debug_llm_raw": debug_run.rel_path(
                        debug_stage_dir / f"chunk{idx:03d}_llm_raw.txt"
                    ),
                    "debug_final": debug_run.rel_path(
                        debug_stage_dir / f"chunk{idx:03d}_final_pt.txt"
                    ),
                    "output_hash": debug_run.sha256_text(""),
                }
                translate_manifest_chunks.append(
                    {
                        "chunk_index": idx,
                        "section_index": chunk_info.get("section"),
                        "section_title": chunk_info.get("title", ""),
                        "start_offset": chunk_info.get("start_offset"),
                        "end_offset": chunk_info.get("end_offset"),
                        "input_hash": debug_run.sha256_text(chunk),
                        "chars_in": len(chunk),
                        "context_hash": debug_run.sha256_text(previous_context)
                        if previous_context
                        else None,
                        "from_cache": False,
                        "from_duplicate": False,
                        "llm_attempts": 0,
                        "retry_reasons": [],
                        "suspect_output": False,
                        "suspect_reason": "",
                        "contamination_detected": False,
                        "sanitization_ratio": None,
                        "dialogue": {
                            "input_quotes": _count_quotes(chunk),
                            "output_quotes": 0,
                            "input_quote_lines": count_quote_lines(chunk),
                            "output_quote_lines": 0,
                            "possible_omission": False,
                            "dialogue_splits": 0,
                        },
                        "normalizers": {"triple_quotes_removed": 0, "dialogue_splits": 0},
                        "lengths": {"chars_out": 0, "ratio_out_in": 0.0},
                        "glossary": _glossary_chunk_manifest(
                            glossary_text=None,
                            matched_count=0,
                            injected_count=0,
                            selected_terms=[],
                        ),
                        "outputs": outputs_payload,
                        "errors": None,
                    }
                )
            continue
        chunk_glossary_text, glossary_matched, glossary_injected, chunk_terms = (
            _build_chunk_glossary(
                glossary_manual_terms,
                chunk,
                match_limit=glossary_match_limit,
                fallback_limit=glossary_fallback_limit,
                logger=logger,
                chunk_index=idx,
            )
        )
        if not chunk_glossary_text:
            chunk_glossary_text = glossary_text
        section_id = chunk_info.get("section")
        if current_section is None or section_id != current_section:
            context_entries = []
            current_section = section_id
        if starts_with_scene_boundary(chunk):
            context_entries = []
        # Define a janela de contexto de tradução, carregando os blocos mais recentes
        # para dar referência consistente de estilo ao LLM.
        previous_context = build_recent_translation_context(
            context_entries,
            max_paragraphs=context_paragraphs,
            max_chars=context_chars,
            include_pt=context_include_pt,
        )

        # Gera o hash (assinatura) deste trecho para checar se ele já foi traduzido antes
        chunk_hash_val = chunk_hash(chunk)
        start_offset = chunk_info.get("start_offset")
        end_offset = chunk_info.get("end_offset")
        from_cache = False
        from_duplicate = False
        llm_attempts = 0
        raw_text: str | None = None
        sanitizer_report = None
        error_message: str | None = None
        parsed_clean: str | None = None
        normalizer_stats = {"dialogue_splits": 0, "triple_quotes_removed": 0}
        cache_payload: dict | None = None
        cache_raw_output: str | None = None
        collapse_fallback = False
        retry_reasons: list[str] = []
        context_used = previous_context
        chunk_profile = classify_translation_chunk(chunk)
        sanitization_ratio: float | None = None
        glossary_enforced: dict[str, int] = {}
        repair_attempted = False
        repair_changed = False
        repair_used_cache = False
        repair_llm_attempts = 0
        repair_issues: list[dict[str, str]] = []
        repair_retry_reasons: list[str] = []
        repair_suspect = False
        repair_suspect_reason = ""
        repair_elapsed_seconds = 0.0
        pre_repair_text = ""

        # Tenta carregar a tradução deste chunk do cache para pular processamento
        if cache_exists("translate", chunk_hash_val):
            data = load_cache("translate", chunk_hash_val)
            meta_ok = _is_cache_compatible(data)
            if not meta_ok:
                logger.debug(
                    "Cache de tradução ignorado: assinatura diferente de backend/model/num_predict."
                )
            else:
                cached = data.get("final_output")
                if cached:
                    logger.info(
                        "Reusando cache de tradução para chunk trad-%d/%d", idx, total_chunks
                    )
                    parsed_clean = cached
                    translated_chunks.append(cached)
                    translated_ok.add(idx)
                    chunk_outputs[idx] = cached
                    processed_indices.add(idx)
                    cache_hits += 1
                    from_cache = True
                    _write_progress()

        if parsed_clean is None:
            if idx in translated_ok and idx in chunk_outputs:
                logger.info("Reusando traducao salva para chunk trad-%d/%d", idx, total_chunks)
                parsed_clean = chunk_outputs[idx]
                translated_chunks.append(chunk_outputs[idx])
                processed_indices.add(idx)
                _write_progress()
            else:
                reused_dup = False
                for prev_chunk, prev_final in seen_chunks:
                    if is_near_duplicate(prev_chunk, chunk) and is_duplicate_reuse_safe(
                        prev_chunk, chunk
                    ):
                        logger.info(
                            "Chunk %d marcado como duplicado de um anterior; reuso habilitado.", idx
                        )
                        parsed_clean = prev_final
                        translated_chunks.append(prev_final)
                        translated_ok.add(idx)
                        chunk_outputs[idx] = prev_final
                        processed_indices.add(idx)
                        duplicate_reuse += 1
                        from_duplicate = True
                        _write_progress()
                        reused_dup = True
                        break
                if not reused_dup:
                    base_prompt = build_translation_prompt(
                        chunk,
                        context=previous_context,
                        glossary_text=chunk_glossary_text,
                        allow_adaptation=allow_adapt_flag,
                        chunk_profile=chunk_profile,
                        source_language=resolved_source_language,
                    )
                    prompt = base_prompt
                    try:
                        attempt = 0
                        retry_reason = ""
                        last_retry_reason = ""
                        suspect_output = False
                        suspect_reason = ""
                        while attempt < cfg.max_retries:
                            prev_temp = backend.temperature
                            temp_for_attempt = dialogue_retry_temps[
                                min(attempt, len(dialogue_retry_temps) - 1)
                            ]
                            backend.temperature = temp_for_attempt
                            try:
                                raw_text, _clean_text, llm_attempts, sanitizer_report = (
                                    _call_with_retry(
                                        backend=backend,
                                        prompt=prompt,
                                        cfg=cfg,
                                        logger=logger,
                                        label=f"trad-{idx}/{len(chunks)}",
                                    )
                                )
                            finally:
                                backend.temperature = prev_temp
                            parsed = _parse_translation_output(raw_text)
                            parsed_raw = _strip_translate_markers(parsed)
                            parsed_clean, report = sanitize_translation_output(
                                parsed_raw, logger=logger, fail_on_contamination=False
                            )
                            sanitizer_report = report
                            log_report(report, logger, prefix=f"trad-parse-{idx}")
                            if debug_translation and report.contamination_detected:
                                debug_dir.mkdir(parents=True, exist_ok=True)
                                attempt_tag = attempt + 1
                                base = f"chunk{idx:03d}_attempt{attempt_tag}"
                                (debug_dir / f"{base}_raw.txt").write_text(
                                    parsed_raw, encoding="utf-8"
                                )
                                (debug_dir / f"{base}_clean.txt").write_text(
                                    parsed_clean, encoding="utf-8"
                                )
                            if not parsed_clean.strip():
                                raise ValueError("Traducao vazia apos parsing/sanitizacao.")
                            raw_candidate = anti_hallucination_filter(
                                orig=chunk, llm_raw=raw_text, cleaned=parsed_raw, mode="translate"
                            )
                            parsed_clean = anti_hallucination_filter(
                                orig=chunk, llm_raw=raw_text, cleaned=parsed_clean, mode="translate"
                            )
                            raw_candidate = postprocess_translation(raw_candidate, chunk)
                            parsed_clean = postprocess_translation(parsed_clean, chunk)
                            raw_candidate = _normalize_chunk_dialogue_quotes(chunk, raw_candidate)
                            parsed_clean = _normalize_chunk_dialogue_quotes(chunk, parsed_clean)
                            raw_candidate, _ = collapse_repeated_curly_quotes(raw_candidate)
                            parsed_clean, _ = collapse_repeated_curly_quotes(parsed_clean)
                            sanitized_ratio = (
                                len(parsed_clean.strip()) / max(len(parsed_raw.strip()), 1)
                                if parsed_raw.strip()
                                else 1.0
                            )
                            sanitization_ratio = sanitized_ratio
                            # Conta aspas na entrada e na saída para avaliar se diálogos foram perdidos ou inventados
                            input_quotes_count = _count_quotes(chunk)
                            output_quotes_count = _count_quotes(parsed_clean)
                            input_quote_lines = count_quote_lines(chunk)
                            output_quote_lines = count_quote_lines(parsed_clean)

                            # Compara as métricas para determinar se a tradução é aceitável ou precisa de retry
                            clean_retry, clean_reason = needs_retry(
                                chunk,
                                parsed_clean,
                                input_quotes=input_quotes_count,
                                output_quotes=output_quotes_count,
                                input_quote_lines=input_quote_lines,
                                output_quote_lines=output_quote_lines,
                                contamination_detected=bool(report.contamination_detected),
                                sanitization_ratio=sanitized_ratio,
                            )
                            narrative_ratio = len(parsed_clean.strip()) / max(len(chunk.strip()), 1)
                            if (
                                not clean_retry
                                and input_quotes_count == 0
                                and narrative_ratio < 0.7
                            ):
                                clean_retry = True
                                clean_reason = "narrative_ratio_low"
                            residual_source, residual_source_reason = (
                                detect_residual_source_language(
                                    parsed_clean, resolved_source_language
                                )
                            )
                            if not clean_retry and residual_source:
                                clean_retry = True
                                clean_reason = residual_source_reason
                            raw_retry, _ = needs_retry(
                                chunk,
                                raw_candidate,
                                input_quotes=input_quotes_count,
                                output_quotes=_count_quotes(raw_candidate),
                                input_quote_lines=input_quote_lines,
                                output_quote_lines=count_quote_lines(raw_candidate),
                                contamination_detected=False,
                                sanitization_ratio=1.0,
                            )
                            prefer_raw = (
                                report.contamination_detected
                                and not raw_retry
                                and (sanitized_ratio < 0.95 or "omissao_dialogo" in clean_reason)
                            )
                            retry = (
                                clean_retry
                                or (report.contamination_detected and sanitized_ratio < 0.95)
                                or (
                                    report.contamination_detected
                                    and "omissao_dialogo" in clean_reason
                                )
                            )
                            retry_reason = (
                                clean_reason
                                if clean_retry
                                else (
                                    "sanitizacao_agressiva"
                                    if sanitized_ratio < 0.95
                                    else retry_reason
                                )
                            )
                            guardrail_triggered = False
                            guardrail_reason = ""
                            if dialogue_guardrails_mode != "off":
                                guard_ratio = 0.5 if dialogue_guardrails_mode == "relaxed" else 0.4
                                if input_quotes_count >= 4 and output_quotes_count < max(
                                    1, int(input_quotes_count * guard_ratio)
                                ):
                                    guardrail_triggered = True
                                    guardrail_reason = f"omissao_dialogo_guardrail_quotes ({output_quotes_count}/{input_quotes_count})"
                                elif input_quote_lines >= 2 and output_quote_lines < max(
                                    1, int(input_quote_lines * guard_ratio)
                                ):
                                    guardrail_triggered = True
                                    guardrail_reason = f"omissao_dialogo_guardrail_linhas ({output_quote_lines}/{input_quote_lines})"
                            if guardrail_triggered:
                                retry = True
                                retry_reason = (
                                    guardrail_reason or retry_reason or "omissao_dialogo_guardrail"
                                )
                            if retry:
                                retry_log_reason = retry_reason or guardrail_reason or "retry"
                                last_retry_reason = retry_log_reason
                                retry_reasons.append(retry_log_reason)
                                logger.warning(
                                    "QA retry traducao chunk %d/%d: reason=%s attempt=%d/%d",
                                    idx,
                                    len(chunks),
                                    retry_log_reason,
                                    attempt + 1,
                                    cfg.max_retries,
                                )
                            if not retry:
                                # se sanitização falhou mas raw está ok, preferir raw
                                if prefer_raw:
                                    parsed_clean = raw_candidate
                                break
                            attempt += 1
                            is_dialogue_retry = (
                                "omissao_dialogo" in (retry_reason or "") or guardrail_triggered
                            )
                            fallback_done = False
                            if (
                                attempt >= cfg.max_retries
                                and "residual_" in (retry_reason or "")
                                and dialogue_split_fallback
                            ):
                                fallback_text, fallback_attempts, fallback_replacements = (
                                    _repair_residual_source_segments(
                                        parsed_clean,
                                        backend=backend,
                                        cfg=cfg,
                                        logger=logger,
                                        glossary_text=chunk_glossary_text,
                                        allow_adaptation=allow_adapt_flag,
                                        source_language=resolved_source_language,
                                        temperature=dialogue_retry_temps[-1],
                                        label=f"trad-residual-{idx}/{len(chunks)}",
                                    )
                                )
                                llm_attempts += fallback_attempts
                                residual_after_fallback, _ = detect_residual_source_language(
                                    fallback_text, resolved_source_language
                                )
                                if fallback_replacements and not residual_after_fallback:
                                    parsed_clean = fallback_text
                                    retry = False
                                    fallback_done = True
                                    retry_reasons.append("residual_source_targeted_fallback")
                                    logger.info(
                                        "Fallback de idioma residual no chunk %d/%d: segmentos=%d",
                                        idx,
                                        len(chunks),
                                        fallback_replacements,
                                    )
                            if (
                                not fallback_done
                                and attempt >= cfg.max_retries
                                and is_dialogue_retry
                                and dialogue_split_fallback
                            ):
                                blocks = _split_dialogue_blocks(chunk) or [chunk]
                                logger.warning(
                                    "Fallback de split de dialogos no chunk %d/%d (%d blocos).",
                                    idx,
                                    len(chunks),
                                    len(blocks),
                                )
                                block_outputs: list[str] = []
                                for b_idx, block in enumerate(blocks, start=1):
                                    block_glossary, _, _, _ = _build_chunk_glossary(
                                        glossary_manual_terms,
                                        block,
                                        match_limit=glossary_match_limit,
                                        fallback_limit=glossary_fallback_limit,
                                        logger=logger,
                                        chunk_index=idx,
                                    )
                                    if not block_glossary:
                                        block_glossary = chunk_glossary_text
                                    block_prompt = build_translation_prompt(
                                        block,
                                        context=None,
                                        glossary_text=block_glossary,
                                        allow_adaptation=allow_adapt_flag,
                                        chunk_profile=classify_translation_chunk(block),
                                        source_language=resolved_source_language,
                                    )
                                    block_prompt += "\n\nATENÇÃO: NENHUMA fala pode ser omitida. Traduza exatamente este bloco preservando todas as aspas e travessões. Não resuma."
                                    prev_temp_block = backend.temperature
                                    backend.temperature = dialogue_retry_temps[-1]
                                    try:
                                        block_raw, _block_clean, _, block_report = _call_with_retry(
                                            backend=backend,
                                            prompt=block_prompt,
                                            cfg=cfg,
                                            logger=logger,
                                            label=f"trad-split-{idx}-{b_idx}",
                                        )
                                    finally:
                                        backend.temperature = prev_temp_block
                                    block_parsed = _parse_translation_output(block_raw)
                                    block_parsed_raw = _strip_translate_markers(block_parsed)
                                    try:
                                        block_clean, block_clean_report = (
                                            sanitize_translation_output(
                                                block_parsed_raw,
                                                logger=logger,
                                                fail_on_contamination=False,
                                            )
                                        )
                                    except ValueError:
                                        logger.warning(
                                            "Fallback split: block %d/%d sanitized to empty; keeping raw.",
                                            b_idx,
                                            len(blocks),
                                        )
                                        block_clean = block_parsed_raw or block
                                        block_clean_report = None
                                    sanitizer_report = block_clean_report or sanitizer_report
                                    block_clean = anti_hallucination_filter(
                                        orig=block,
                                        llm_raw=block_raw,
                                        cleaned=block_clean,
                                        mode="translate",
                                    )
                                    block_clean = postprocess_translation(block_clean, block)
                                    block_outputs.append(block_clean)
                                parsed_clean = "\n\n".join(block_outputs).strip()
                                retry = False
                                fallback_done = True
                            if fallback_done:
                                break
                            if attempt >= cfg.max_retries:
                                suspect_output = True
                                suspect_reason = (
                                    last_retry_reason
                                    or retry_reason
                                    or guardrail_reason
                                    or "max_retries_exceeded"
                                )
                                if prefer_raw:
                                    parsed_clean = raw_candidate
                                logger.warning(
                                    "Max retries atingido no chunk %d/%d; mantendo ultima saida (reason=%s)",
                                    idx,
                                    len(chunks),
                                    suspect_reason,
                                )
                                break
                            logger.warning(
                                "QA retry traducao chunk %d/%d: %s (tentativa %d/%d)",
                                idx,
                                len(chunks),
                                retry_reason,
                                attempt + 1,
                                cfg.max_retries,
                            )
                            if "omissao_dialogo" in retry_reason:
                                prompt = (
                                    base_prompt
                                    + "\n\nATENÇÃO: Você omitiu falas. Refaça traduzindo TODAS as frases e mantendo cada fala entre aspas exatamente uma vez. Não resuma. Não remova risos/interjeições."
                                )
                            elif "omissao_paragrafos" in retry_reason:
                                prompt = (
                                    base_prompt
                                    + "\n\nATENÇÃO: Você fundiu ou omitiu parágrafos. Refaça traduzindo TODO o conteúdo, mantendo cada parágrafo da fonte separado e na mesma ordem. Não una fala e narração. Não resuma."
                                )
                            elif "residual_" in retry_reason:
                                prompt = (
                                    base_prompt
                                    + f"\n\nATENÇÃO: Sua saída manteve frases em {source_language_name(resolved_source_language)}. Refaça traduzindo todo o conteúdo para português brasileiro e preserve apenas nomes próprios previstos no glossário."
                                )
                            elif (
                                "unbalanced_quotes" in retry_reason
                                or "extra_curly_quotes" in retry_reason
                            ):
                                prompt = (
                                    base_prompt
                                    + "\n\nATENÇÃO: Você alterou a estrutura de aspas. Refaça mantendo cada fala entre aspas exatamente uma vez; não feche uma fala antes de risos, interjeições ou o fim da mesma fala."
                                )
                            elif "truncado" in retry_reason:
                                prompt = (
                                    base_prompt
                                    + "\n\nATENÇÃO: Sua saída foi truncada. Refaça incluindo TODO o conteúdo."
                                )
                            else:
                                prompt = (
                                    base_prompt
                                    + "\n\nATENÇÃO: Sua saída anterior veio truncada ou repetitiva. Refaça e inclua TODO o conteúdo. Não resuma."
                                )

                        parsed_clean = postprocess_translation(parsed_clean, chunk)
                        parsed_clean = _normalize_chunk_dialogue_quotes(chunk, parsed_clean)
                        terms_for_enforcement = chunk_terms if glossary_matched > 0 else []
                        parsed_clean, enforced = enforce_canonical_terms(
                            parsed_clean, terms_for_enforcement
                        )
                        glossary_enforced = enforced
                        if enforced:
                            logger.info(
                                "Glossario enforcement chunk %d/%d: %s",
                                idx,
                                len(chunks),
                                ", \n".join(
                                    f"{k}->{enforced[k]}" for k in list(enforced.keys())[:5]
                                ),
                            )
                        # correção de aspas curvas
                        opens_q, closes_q = count_curly_quotes(parsed_clean)
                        if has_curly_quote_balance_regression(chunk, parsed_clean):
                            parsed_clean, fixed = fix_unbalanced_quotes(
                                parsed_clean, logger=logger, label=f"trad-{idx}"
                            )
                            if fixed:
                                opens_q, closes_q = count_curly_quotes(parsed_clean)
                        orig_len = len(chunk.strip())
                        cleaned_len = len(parsed_clean.strip())
                        if orig_len and cleaned_len < orig_len * 0.5:
                            logger.error(
                                "Traducao suspeita: chunk %d/%d ficou com %d%% do tamanho original apos sanitizacao.",
                                idx,
                                len(chunks),
                                int((cleaned_len / orig_len) * 100) if orig_len else 0,
                            )
                            marker = f"[CHUNK_TRADUCAO_SUSPEITO_{idx}] "
                            parsed_clean = (
                                f"{marker}{parsed_clean}" if parsed_clean.strip() else marker
                            )
                        elif orig_len and cleaned_len < orig_len * 0.7:
                            logger.warning(
                                "Traducao suspeita: chunk %d/%d muito menor que o original; mantendo traducao mesmo assim.",
                                idx,
                                len(chunks),
                            )
                        if has_suspicious_repetition(parsed_clean):
                            logger.warning(
                                "Traducao com repeticao suspeita; chunk %d/%d marcado para revisao.",
                                idx,
                                len(chunks),
                            )
                        if detect_model_collapse(
                            parsed_clean, original_len=len(chunk), mode="translate"
                        ):
                            logger.warning(
                                "Colapso detectado no chunk %d/%d; usando texto original do chunk.",
                                idx,
                                len(chunks),
                            )
                            collapse_detected += 1
                            parsed_clean = chunk
                            collapse_fallback = True
                        if repair_enabled and not collapse_fallback:
                            pre_repair_text = parsed_clean
                            repair_result = repair_translation_chunk(
                                source_text=chunk,
                                translated_text=parsed_clean,
                                backend=backend,
                                logger=logger,
                                glossary_text=chunk_glossary_text,
                                glossary_terms=chunk_terms,
                                max_attempts=max(1, min(cfg.max_retries, 2)),
                                source_language=resolved_source_language,
                                cache_metadata={
                                    "chunk_index": idx,
                                    "source": source_slug or "",
                                    "doc_hash": doc_hash,
                                    "manual_glossary_hash": manual_glossary_hash,
                                },
                            )
                            repair_attempted = repair_result.attempted
                            repair_changed = repair_result.changed
                            repair_used_cache = repair_result.used_cache
                            repair_llm_attempts = repair_result.llm_attempts
                            repair_issues = repair_result.issues
                            repair_retry_reasons = repair_result.retry_reasons
                            repair_suspect = repair_result.suspect_output
                            repair_suspect_reason = repair_result.suspect_reason
                            repair_elapsed_seconds = repair_result.elapsed_seconds
                            if repair_suspect:
                                suspect_output = True
                                if not suspect_reason:
                                    suspect_reason = repair_suspect_reason
                            if repair_attempted:
                                logger.info(
                                    "Repair traducao chunk %d/%d: changed=%s issues=%s",
                                    idx,
                                    len(chunks),
                                    repair_changed,
                                    ",".join(issue.get("type", "") for issue in repair_issues[:5]),
                                )
                                parsed_clean = repair_result.text
                                parsed_clean = postprocess_translation(parsed_clean, chunk)
                                parsed_clean, repair_enforced = enforce_canonical_terms(
                                    parsed_clean, terms_for_enforcement
                                )
                                if repair_enforced:
                                    for key, value in repair_enforced.items():
                                        glossary_enforced[key] = (
                                            glossary_enforced.get(key, 0) + value
                                        )
                                opens_q, closes_q = count_curly_quotes(parsed_clean)
                                if has_curly_quote_balance_regression(chunk, parsed_clean):
                                    parsed_clean, _ = fix_unbalanced_quotes(
                                        parsed_clean, logger=logger, label=f"trad-repair-{idx}"
                                    )
                        translated_chunks.append(parsed_clean)
                        translated_ok.add(idx)
                        failed_chunks.discard(idx)
                        chunk_outputs[idx] = parsed_clean
                        processed_indices.add(idx)
                        seen_chunks.append((chunk, parsed_clean))
                        cache_raw_output = raw_text
                        cache_payload = {
                            "chunk_index": idx,
                            "mode": "translate",
                            "source": source_slug or "",
                            "doc_hash": doc_hash,
                            "backend": getattr(backend, "backend", None),
                            "model": getattr(backend, "model", None),
                            "num_predict": getattr(backend, "num_predict", None),
                            "temperature": getattr(backend, "temperature", None),
                            "repeat_penalty": getattr(backend, "repeat_penalty", None),
                            "translate_chunk_chars": cfg.translate_chunk_chars,
                            "glossary_hash": glossary_hash,
                            "manual_glossary_hash": manual_glossary_hash,
                            "allow_adaptation": allow_adapt_flag,
                            "translation_repair": bool(repair_enabled),
                            "repair_prompt_hash": repair_prompt_fingerprint()
                            if repair_enabled
                            else None,
                            "repair_pipeline_version": REPAIR_PIPELINE_VERSION
                            if repair_enabled
                            else None,
                            "translate_context_paragraphs": context_paragraphs,
                            "translate_context_chars": context_chars,
                            "translate_context_include_pt": context_include_pt,
                            "split_by_sections": split_flag,
                            "dialogue_guardrails_mode": dialogue_guardrails_mode,
                            "prompt_hash": prompt_hash,
                            "pipeline_version": TRANSLATE_PIPELINE_VERSION,
                        }
                    except Exception as exc:
                        # debug de falha
                        fail_dir = Path(cfg.output_dir) / "debug_translate_chunks" / "failed"
                        try:
                            fail_dir.mkdir(parents=True, exist_ok=True)
                            attempt_tag = llm_attempts or 0
                            if raw_text:
                                (
                                    fail_dir / f"chunk{idx:03d}_attempt{attempt_tag}_raw.txt"
                                ).write_text(raw_text, encoding="utf-8")
                            (fail_dir / f"chunk{idx:03d}_prompt.txt").write_text(
                                prompt, encoding="utf-8"
                            )
                            (fail_dir / f"chunk{idx:03d}_context.txt").write_text(
                                context_used or "", encoding="utf-8"
                            )
                            error_payload = {
                                "chunk_index": idx,
                                "label": f"trad-{idx}/{len(chunks)}",
                                "error": str(exc),
                                "stack": traceback.format_exc(),
                            }
                            (fail_dir / f"chunk{idx:03d}_error.json").write_text(
                                json.dumps(error_payload, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                        failed_chunks.add(idx)
                        error_message = str(exc)
                        llm_attempts = getattr(exc, "attempts", llm_attempts)
                        if sanitizer_report is None and hasattr(exc, "last_report"):
                            sanitizer_report = getattr(exc, "last_report")
                        raise RuntimeError(
                            f"Falha ao traduzir chunk {idx}/{len(chunks)}: {exc}"
                        ) from exc
                    finally:
                        _write_progress()

        final_output = parsed_clean if parsed_clean is not None else chunk_outputs.get(idx, "")
        final_output, normalizer_stats = apply_structural_normalizers(final_output)
        final_output = apply_custom_normalizers(final_output, convert_quote_dialogues=False)
        normalization_totals["dialogue_splits"] += normalizer_stats.get("dialogue_splits", 0)
        normalization_totals["triple_quotes_removed"] += normalizer_stats.get(
            "triple_quotes_removed", 0
        )
        chunk_outputs[idx] = final_output
        _write_progress()
        final_stripped = final_output.strip()
        is_unusable_placeholder = final_stripped.startswith(
            ("[CHUNK_NAO_PROCESSADO", "[CHUNK_TRANSLATION_REJECTED")
        )
        if ends_with_scene_boundary(chunk) or is_unusable_placeholder:
            context_entries = []
        elif final_stripped:
            context_entries.append({"source": chunk, "target": final_output})
            max_entries = max(context_paragraphs + 2, 3)
            if len(context_entries) > max_entries:
                context_entries = context_entries[-max_entries:]
        if debug_translation and idx <= 5:
            debug_dir.mkdir(parents=True, exist_ok=True)
            base = f"chunk{idx:03d}"
            (debug_dir / f"{base}_original_en.txt").write_text(chunk, encoding="utf-8")
            (debug_dir / f"{base}_context.txt").write_text(context_used or "", encoding="utf-8")
            (debug_dir / f"{base}_glossary.txt").write_text(
                chunk_glossary_text or "", encoding="utf-8"
            )
            if raw_text:
                (debug_dir / f"{base}_llm_raw.txt").write_text(raw_text, encoding="utf-8")
            (debug_dir / f"{base}_final_pt.txt").write_text(final_output, encoding="utf-8")
        if debug_run and debug_run.should_write_chunk(idx):
            debug_stage_dir = debug_run.stage_dir("40_translate") / "debug_traducao"
            debug_stage_dir.mkdir(parents=True, exist_ok=True)
            base = f"chunk{idx:03d}"
            debug_run.write_text(
                debug_run.rel_path(debug_stage_dir / f"{base}_original_en.txt"),
                chunk,
            )
            debug_run.write_text(
                debug_run.rel_path(debug_stage_dir / f"{base}_context.txt"),
                context_used or "",
            )
            debug_run.write_text(
                debug_run.rel_path(debug_stage_dir / f"{base}_glossary.txt"),
                chunk_glossary_text or "",
            )
            if raw_text is not None:
                raw_hash = debug_run.sha256_text(raw_text)
                if not debug_run.store_llm_raw:
                    llm_payload = f"[[OMITTED]]\n[[SHA256:{raw_hash}]]\n"
                elif debug_run.max_chars_per_file and len(raw_text) > debug_run.max_chars_per_file:
                    truncated = raw_text[: debug_run.max_chars_per_file]
                    llm_payload = f"{truncated}\n\n[[TRUNCATED]]\n[[SHA256:{raw_hash}]]\n"
                else:
                    llm_payload = raw_text
                debug_run.write_text(
                    debug_run.rel_path(debug_stage_dir / f"{base}_llm_raw.txt"),
                    llm_payload,
                    allow_truncate=False,
                )
            debug_run.write_text(
                debug_run.rel_path(debug_stage_dir / f"{base}_final_pt.txt"),
                final_output,
            )
        if cache_payload is not None:
            save_cache(
                "translate",
                chunk_hash_val,
                raw_output=cache_raw_output,
                final_output=final_output,
                metadata=cache_payload,
            )
        orig_len_for_stats = len(chunk)
        orig_chars_total += orig_len_for_stats
        sanitized_chars_total += len(final_output)
        if sanitizer_report and sanitizer_report.contamination_detected:
            contamination_count += 1
        if error_message:
            error_count += 1

        cleaned_ratio = (
            (len(final_output.strip()) / max(len(chunk.strip()), 1)) if chunk.strip() else 0.0
        )
        too_short = cleaned_ratio < 0.60
        too_long = cleaned_ratio > 1.80
        suspicious = has_suspicious_repetition(final_output)
        residual_source, residual_source_reason = detect_residual_source_language(
            final_output, resolved_source_language
        )
        if residual_source:
            suspect_output = True
            if not suspect_reason:
                suspect_reason = residual_source_reason
        orig_quotes = _count_quotes(chunk)
        translated_quotes = _count_quotes(final_output)
        possible_omission = False
        if orig_quotes >= 4 and translated_quotes <= max(1, int(orig_quotes * 0.4)):
            possible_omission = True
            logger.warning(
                "Possível omissão de falas no chunk %d/%d (aspas %d -> %d).",
                idx,
                total_chunks,
                orig_quotes,
                translated_quotes,
            )
        reject_reasons: list[str] = []
        if cleaned_ratio > getattr(cfg, "translate_max_ratio", 1.8):
            reject_reasons.append(f"ratio_out_in_{cleaned_ratio:.2f}")
        if fail_on_error and suspect_output:
            reject_reasons.append(suspect_reason or "suspect_output")
        rejected_output = bool(reject_reasons)
        if rejected_output:
            failed_chunks.add(idx)
            translated_ok.discard(idx)
            placeholder = f"[CHUNK_TRANSLATION_REJECTED_{idx}] ({';'.join(reject_reasons)})"
            translated_chunks[-1] = placeholder
            final_output = placeholder
            chunk_outputs[idx] = placeholder
            logger.error(
                "Chunk %d/%d rejeitado por guardrail: %s (ratio=%.2f)",
                idx,
                total_chunks,
                ";".join(reject_reasons),
                cleaned_ratio,
            )
            if fail_on_error:
                raise RuntimeError(
                    f"Tradução rejeitada no chunk {idx}/{total_chunks}: {', '.join(reject_reasons)}"
                )

        chunk_metrics.append(
            {
                "chunk_index": idx,
                "chars_in": len(chunk),
                "chars_out": len(final_output),
                "ratio_out_in": round(cleaned_ratio, 3),
                "from_cache": from_cache,
                "from_duplicate": from_duplicate,
                "llm_attempts": llm_attempts,
                "chunk_profile": chunk_profile,
                "context_chars": len(context_used or ""),
                "too_short": too_short,
                "too_long": too_long,
                "suspicious_repetition": suspicious,
                "source_language": resolved_source_language,
                "residual_source_language": residual_source,
                "residual_source_language_reason": residual_source_reason,
                "residual_english": residual_source if resolved_source_language == "en" else False,
                "residual_english_reason": residual_source_reason
                if resolved_source_language == "en"
                else "",
                "possible_omission": possible_omission,
                "repair_attempted": repair_attempted,
                "repair_changed": repair_changed,
                "repair_used_cache": repair_used_cache,
                "repair_llm_attempts": repair_llm_attempts,
                "repair_issues": repair_issues,
                "repair_retry_reasons": repair_retry_reasons,
                "repair_suspect": repair_suspect,
                "repair_suspect_reason": repair_suspect_reason,
                "repair_elapsed_seconds": round(repair_elapsed_seconds, 3),
                "pre_repair_hash": chunk_hash(pre_repair_text) if pre_repair_text else None,
                "dialogue_splits": normalizer_stats.get("dialogue_splits", 0),
                "triple_quotes_removed": normalizer_stats.get("triple_quotes_removed", 0),
                "suspect_output": suspect_output,
                "suspect_reason": suspect_reason,
                "collapse_fallback": collapse_fallback,
                "rejected_output": rejected_output,
                "reject_reason": ";".join(reject_reasons),
                "glossary_matched": glossary_matched,
                "glossary_injected": glossary_injected,
                "glossary_fallback_used": bool(chunk_terms and glossary_matched == 0),
                "glossary_enforced_replacements": glossary_enforced,
            }
        )
        repair_metrics.append(
            {
                "chunk_index": idx,
                "enabled": bool(repair_enabled),
                "attempted": repair_attempted,
                "changed": repair_changed,
                "used_cache": repair_used_cache,
                "llm_attempts": repair_llm_attempts,
                "issues": repair_issues,
                "retry_reasons": repair_retry_reasons,
                "suspect_output": repair_suspect,
                "suspect_reason": repair_suspect_reason,
                "elapsed_seconds": round(repair_elapsed_seconds, 3),
            }
        )
        if repair_attempted:
            repair_attempted_total += 1
        if repair_changed:
            repair_changed_total += 1
        if repair_used_cache:
            repair_cache_hits_total += 1
        if repair_suspect:
            repair_suspect_total += 1
        repair_elapsed_total += repair_elapsed_seconds

        report_dict = {
            "contamination_detected": bool(sanitizer_report.contamination_detected)
            if sanitizer_report
            else False,
            "removed_lines_count": getattr(sanitizer_report, "removed_lines_count", 0)
            if sanitizer_report
            else 0,
            "collapsed_repetitions": getattr(sanitizer_report, "collapsed_repetitions", 0)
            if sanitizer_report
            else 0,
            "leading_noise_removed": getattr(sanitizer_report, "leading_noise_removed", False)
            if sanitizer_report
            else False,
            "removed_think_blocks": getattr(sanitizer_report, "removed_think_blocks", 0)
            if sanitizer_report
            else 0,
        }

        if debug_chunks:
            entry = {
                "chunk_index": idx,
                "original_start_offset": start_offset,
                "original_end_offset": end_offset,
                "original_text": chunk,
                "original_chars": orig_len_for_stats,
                "original_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                "from_cache": from_cache,
                "from_duplicate": from_duplicate,
                "llm_attempts": llm_attempts,
                "llm_raw_output": raw_text,
                "sanitized_output": final_output,
                "sanitized_chars": len(final_output),
                "sanitized_hash": hashlib.sha256(final_output.encode("utf-8")).hexdigest(),
                "sanitizer_report": report_dict,
                "normalizer_stats": normalizer_stats,
                "glossary": _glossary_chunk_manifest(
                    glossary_text=chunk_glossary_text,
                    matched_count=glossary_matched,
                    injected_count=glossary_injected,
                    selected_terms=chunk_terms,
                    enforced_replacements=glossary_enforced,
                ),
                "error": error_message,
            }
            _write_chunk_debug(entry)
        if debug_run and debug_run.should_write_chunk(idx):
            orig_quotes = _count_quotes(chunk)
            translated_quotes = _count_quotes(final_output)
            input_quote_lines = count_quote_lines(chunk)
            output_quote_lines = count_quote_lines(final_output)
            possible_omission = False
            if orig_quotes >= 4 and translated_quotes <= max(1, int(orig_quotes * 0.4)):
                possible_omission = True
            cleaned_ratio = (
                (len(final_output.strip()) / max(len(chunk.strip()), 1)) if chunk.strip() else 0.0
            )
            output_hash = debug_run.sha256_text(final_output)
            debug_stage_dir = debug_run.stage_dir("40_translate") / "debug_traducao"
            outputs_payload = {
                "debug_original": debug_run.rel_path(
                    debug_stage_dir / f"chunk{idx:03d}_original_en.txt"
                ),
                "debug_context": debug_run.rel_path(
                    debug_stage_dir / f"chunk{idx:03d}_context.txt"
                ),
                "debug_glossary": debug_run.rel_path(
                    debug_stage_dir / f"chunk{idx:03d}_glossary.txt"
                ),
                "debug_llm_raw": debug_run.rel_path(
                    debug_stage_dir / f"chunk{idx:03d}_llm_raw.txt"
                ),
                "debug_final": debug_run.rel_path(debug_stage_dir / f"chunk{idx:03d}_final_pt.txt"),
                "output_hash": output_hash,
            }
            if repair_attempted:
                repair_stage_dir = debug_run.stage_dir("45_repair") / "debug_repair"
                repair_stage_dir.mkdir(parents=True, exist_ok=True)
                debug_run.write_text(
                    debug_run.rel_path(repair_stage_dir / f"chunk{idx:03d}_original_en.txt"),
                    chunk,
                )
                debug_run.write_text(
                    debug_run.rel_path(repair_stage_dir / f"chunk{idx:03d}_before_pt.txt"),
                    pre_repair_text or final_output,
                )
                debug_run.write_text(
                    debug_run.rel_path(repair_stage_dir / f"chunk{idx:03d}_after_pt.txt"),
                    final_output,
                )
            translate_manifest_chunks.append(
                {
                    "chunk_index": idx,
                    "section_index": chunk_info.get("section"),
                    "section_title": chunk_info.get("title", ""),
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "input_hash": debug_run.sha256_text(chunk),
                    "chars_in": len(chunk),
                    "context_hash": debug_run.sha256_text(context_used) if context_used else None,
                    "from_cache": from_cache,
                    "from_duplicate": from_duplicate,
                    "llm_attempts": llm_attempts,
                    "chunk_profile": chunk_profile,
                    "retry_reasons": retry_reasons,
                    "suspect_output": suspect_output,
                    "suspect_reason": suspect_reason,
                    "contamination_detected": bool(sanitizer_report.contamination_detected)
                    if sanitizer_report
                    else False,
                    "sanitization_ratio": sanitization_ratio,
                    "dialogue": {
                        "input_quotes": orig_quotes,
                        "output_quotes": translated_quotes,
                        "input_quote_lines": input_quote_lines,
                        "output_quote_lines": output_quote_lines,
                        "possible_omission": possible_omission,
                        "dialogue_splits": normalizer_stats.get("dialogue_splits", 0),
                    },
                    "normalizers": {
                        "triple_quotes_removed": normalizer_stats.get("triple_quotes_removed", 0),
                        "dialogue_splits": normalizer_stats.get("dialogue_splits", 0),
                    },
                    "lengths": {
                        "chars_out": len(final_output),
                        "ratio_out_in": round(cleaned_ratio, 3),
                    },
                    "glossary": _glossary_chunk_manifest(
                        glossary_text=chunk_glossary_text,
                        matched_count=glossary_matched,
                        injected_count=glossary_injected,
                        selected_terms=chunk_terms,
                        enforced_replacements=glossary_enforced,
                    ),
                    "repair": {
                        "enabled": bool(repair_enabled),
                        "attempted": repair_attempted,
                        "changed": repair_changed,
                        "used_cache": repair_used_cache,
                        "llm_attempts": repair_llm_attempts,
                        "issues": repair_issues,
                        "retry_reasons": repair_retry_reasons,
                        "suspect_output": repair_suspect,
                        "suspect_reason": repair_suspect_reason,
                        "pre_repair_hash": debug_run.sha256_text(pre_repair_text)
                        if pre_repair_text
                        else None,
                    },
                    "outputs": outputs_payload,
                    "errors": None
                    if not error_message
                    else {
                        "message": error_message,
                    },
                }
            )
    logger.info(
        "Resumo da traducao: total=%d sucesso=%d erro=%d",
        total_chunks,
        len(translated_ok),
        len(failed_chunks),
    )

    if len(processed_indices) != total_chunks:
        logger.error(
            "Inconsistencia: apenas %d/%d chunks registraram alguma saida.",
            len(processed_indices),
            total_chunks,
        )

    missing_outputs = [i for i in range(1, total_chunks + 1) if i not in chunk_outputs]
    if missing_outputs:
        logger.error(
            "Chunks sem saida detectados apos traducao: %s; placeholders inseridos.",
            missing_outputs,
        )
        for midx in missing_outputs:
            placeholder = f"[CHUNK_NAO_PROCESSADO_{midx}]"
            chunk_outputs[midx] = placeholder
            failed_chunks.add(midx)
        _write_progress()

    heading_fixes = 0
    seen_output_sections: set[int] = set()
    for idx in range(1, total_chunks + 1):
        chunk_info = chunk_records[idx - 1] if idx - 1 < len(chunk_records) else {}
        section_id = chunk_info.get("section")
        if section_id is None or section_id in seen_output_sections:
            continue
        seen_output_sections.add(section_id)
        fixed_output, changed = ensure_section_heading(
            chunk_outputs.get(idx, ""),
            str(chunk_info.get("title", "")),
        )
        if changed:
            chunk_outputs[idx] = fixed_output
            heading_fixes += 1

    ordered_outputs = [
        chunk_outputs.get(i, f"[CHUNK_NAO_PROCESSADO_{i}]") for i in range(1, total_chunks + 1)
    ]
    translated_chunks = ordered_outputs

    result = "\n\n".join(ordered_outputs).strip()
    if not result:
        raise ValueError("Traducao resultou em texto vazio.")

    translated_paragraphs = [p for p in result.split("\n\n") if p.strip()]
    if len(translated_paragraphs) < original_paragraphs_total:
        logger.error(
            "Paragrafos ausentes apos traducao: original=%d traduzido=%d",
            original_paragraphs_total,
            len(translated_paragraphs),
        )
        paragraph_mismatch = {
            "original": original_paragraphs_total,
            "translated": len(translated_paragraphs),
        }

    if debug_chunks:
        reduction_pct = (
            (sanitized_chars_total / orig_chars_total * 100) if orig_chars_total else 0.0
        )
        avg_orig = (orig_chars_total / total_chunks) if total_chunks else 0.0
        avg_san = (sanitized_chars_total / total_chunks) if total_chunks else 0.0
        logger.info(
            "Debug chunks resumo: total=%d cache=%d dup=%d contaminados=%d erros=%d",
            total_chunks,
            cache_hits,
            duplicate_reuse,
            contamination_count,
            error_count,
        )
        logger.info(
            "Debug chunks tamanhos: orig_med=%.1f san_med=%.1f (%.1f%% do original)",
            avg_orig,
            avg_san,
            reduction_pct,
        )
        if debug_file_path:
            logger.info("Arquivo de debug de chunks: %s", debug_file_path)

    result, final_report = sanitize_translation_output(
        result, logger=logger, fail_on_contamination=False
    )
    opens_final, closes_final = count_curly_quotes(result)
    if opens_final != closes_final:
        result, _ = fix_unbalanced_quotes(result, logger=logger, label="trad-final")
    log_report(final_report, logger, prefix="trad-final")
    try:
        version = (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        version = "unknown"
    status = "failed" if failed_chunks else "ok"
    report = {
        "mode": "translate",
        "status": status,
        "input": source_slug or "",
        "total_chunks": total_chunks,
        "cache_hits": cache_hits,
        "fallbacks": fallbacks,
        "failed_chunks": len(failed_chunks),
        "collapse_detected": collapse_detected,
        "duplicates_reused": duplicate_reuse,
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": version,
        "effective_translate_chunk_chars": cfg.translate_chunk_chars,
        "max_chunk_chars_observed": max_chunk_len,
        "dialogue_splits": normalization_totals.get("dialogue_splits", 0),
        "triple_quotes_removed": normalization_totals.get("triple_quotes_removed", 0),
        "section_heading_fixes": heading_fixes,
        "translation_repair_enabled": bool(repair_enabled),
        "repair_attempted_chunks": repair_attempted_total,
        "repair_changed_chunks": repair_changed_total,
        "repair_cache_hits": repair_cache_hits_total,
        "repair_suspect_chunks": repair_suspect_total,
        "repair_elapsed_seconds": round(repair_elapsed_total, 3),
        "translate_context_paragraphs": context_paragraphs,
        "translate_context_chars": context_chars,
        "translate_context_include_pt": context_include_pt,
    }
    if paragraph_mismatch:
        report["paragraph_mismatch"] = paragraph_mismatch
    try:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        slug = (source_slug or "document").replace("\\", "_").replace("/", "_")
        report_path = Path(cfg.output_dir) / f"{slug}_translate_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics_payload = {
            "total_chunks": total_chunks,
            "cache_hits": cache_hits,
            "duplicates_reused": duplicate_reuse,
            "fallbacks": fallbacks,
            "failed_chunks": len(failed_chunks),
            "collapse_detected": collapse_detected,
            "chunks": chunk_metrics,
            "effective_translate_chunk_chars": cfg.translate_chunk_chars,
            "max_chunk_chars_observed": max_chunk_len,
            "dialogue_splits": normalization_totals.get("dialogue_splits", 0),
            "triple_quotes_removed": normalization_totals.get("triple_quotes_removed", 0),
            "section_heading_fixes": heading_fixes,
            "translation_repair_enabled": bool(repair_enabled),
            "repair_attempted_chunks": repair_attempted_total,
            "repair_changed_chunks": repair_changed_total,
            "repair_cache_hits": repair_cache_hits_total,
            "repair_suspect_chunks": repair_suspect_total,
            "repair_elapsed_seconds": round(repair_elapsed_total, 3),
            "translate_context_paragraphs": context_paragraphs,
            "translate_context_chars": context_chars,
            "translate_context_include_pt": context_include_pt,
            "source_quote_boundary_fixed": source_quote_boundary_fixed,
        }
        metrics_path = Path(cfg.output_dir) / f"{slug}_translate_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        repair_report = {
            "mode": "translation_repair",
            "enabled": bool(repair_enabled),
            "input": source_slug or "",
            "total_chunks": total_chunks,
            "attempted_chunks": repair_attempted_total,
            "changed_chunks": repair_changed_total,
            "cache_hits": repair_cache_hits_total,
            "suspect_chunks": repair_suspect_total,
            "elapsed_seconds": round(repair_elapsed_total, 3),
            "pipeline_version": REPAIR_PIPELINE_VERSION,
            "prompt_hash": repair_prompt_fingerprint() if repair_enabled else None,
            "timestamp": datetime.now().isoformat(),
        }
        repair_report_path = Path(cfg.output_dir) / f"{slug}_repair_report.json"
        repair_report_path.write_text(
            json.dumps(repair_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        repair_metrics_path = Path(cfg.output_dir) / f"{slug}_repair_metrics.json"
        repair_metrics_path.write_text(
            json.dumps(
                {
                    **repair_report,
                    "chunks": repair_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
    if debug_file:
        debug_file.close()
        if debug_file_path:
            logger.info("Arquivo de debug de chunks: %s", debug_file_path)
    if debug_run:
        translate_manifest = {
            "run_id": debug_run.run_id,
            "stage": "translate",
            "source_slug": source_slug or "",
            "input_kind": debug_run.input_kind,
            "input_paths": {
                "preprocessed": debug_run.preprocessed_rel,
                "desquebrado": debug_run.desquebrado_rel,
            },
            "chunking": {
                "split_by_sections": split_flag,
                "translate_chunk_chars": cfg.translate_chunk_chars,
                "total_sections": len(sections),
                "total_chunks": total_chunks,
            },
            "cache_signature": {
                "backend": getattr(backend, "backend", None),
                "model": getattr(backend, "model", None),
                "num_predict": getattr(backend, "num_predict", None),
                "temperature": getattr(backend, "temperature", None),
                "repeat_penalty": getattr(backend, "repeat_penalty", None),
                "translate_chunk_chars": cfg.translate_chunk_chars,
                "translate_context_paragraphs": context_paragraphs,
                "translate_context_chars": context_chars,
                "translate_context_include_pt": context_include_pt,
                "glossary_hash": glossary_hash,
                "manual_glossary_hash": manual_glossary_hash,
            },
            "glossary": {
                "enabled": bool(glossary_text or glossary_manual_terms),
                "manual_terms_total": len(glossary_manual_terms or []),
                "match_limit": glossary_match_limit,
                "fallback_limit": glossary_fallback_limit,
                "glossary_hash": glossary_hash,
                "manual_glossary_hash": manual_glossary_hash,
            },
            "chunks": translate_manifest_chunks,
            "totals": {
                "cache_hits": cache_hits,
                "duplicate_reuse": duplicate_reuse,
                "fallbacks": fallbacks,
                "collapse_detected": collapse_detected,
                "contamination_count": contamination_count,
                "error_count": error_count,
                "orig_chars_total": orig_chars_total,
                "sanitized_chars_total": sanitized_chars_total,
                "section_heading_fixes": heading_fixes,
            },
        }
        debug_run.write_manifest("translate", translate_manifest)
        repair_manifest = {
            "run_id": debug_run.run_id,
            "stage": "repair",
            "source_slug": source_slug or "",
            "enabled": bool(repair_enabled),
            "pipeline_version": REPAIR_PIPELINE_VERSION,
            "prompt_hash": repair_prompt_fingerprint() if repair_enabled else None,
            "chunks": repair_metrics,
            "totals": {
                "total_chunks": total_chunks,
                "attempted_chunks": repair_attempted_total,
                "changed_chunks": repair_changed_total,
                "cache_hits": repair_cache_hits_total,
                "suspect_chunks": repair_suspect_total,
                "elapsed_seconds": round(repair_elapsed_total, 3),
            },
        }
        debug_run.write_manifest("repair", repair_manifest)
    if failed_chunks:
        msg = (
            f"Traducao finalizada com falhas: {len(failed_chunks)}/{total_chunks} chunks nao foram traduzidos. "
            "Placeholders foram inseridos; consulte debug_translate_chunks/failed."
        )
        if fail_on_error:
            raise RuntimeError(msg)
        logger.error(msg)
    return result


def _call_with_retry(
    backend: LLMBackend,
    prompt: str,
    cfg: AppConfig,
    logger: logging.Logger,
    label: str,
) -> tuple[str, str, int, SanitizationReport | None]:
    """Chama o backend com novas tentativas e sanitização conservadora da tradução."""
    delay = cfg.initial_backoff
    last_error: Exception | None = None
    last_report: SanitizationReport | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            latency, response = timed(backend.generate, prompt)
            text, report = sanitize_translation_output(
                response.text, logger=logger, fail_on_contamination=False
            )
            last_report = report
            log_report(report, logger, prefix=label)
            if not text.strip():
                raise ValueError("Texto vazio apos sanitizacao.")
            if report.contamination_detected:
                logger.warning(
                    "%s: contaminacao detectada; texto limpo sera usado (%d chars)",
                    label,
                    len(text),
                )
            logger.info("%s ok (%.2fs, %d chars)", label, latency, len(text))
            return response.text, text, attempt, report
        except Exception as exc:
            last_error = exc
            logger.warning("%s falhou (tentativa %d/%d): %s", label, attempt, cfg.max_retries, exc)
            if attempt < cfg.max_retries:
                time.sleep(delay)
                delay *= cfg.backoff_factor
    err = RuntimeError(f"{label} falhou apos {cfg.max_retries} tentativas: {last_error}")
    setattr(err, "attempts", cfg.max_retries)
    setattr(err, "last_report", last_report)
    raise err
