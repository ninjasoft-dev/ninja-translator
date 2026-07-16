"""
Refinamento capítulo a capítulo de arquivos Markdown.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .advanced_preprocess import clean_text as advanced_clean
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
from .cleanup import cleanup_before_refine, detect_glued_dialogues, detect_obvious_dupes
from .config import AppConfig
from .debug_run import DebugRunWriter
from .desquebrar import normalize_md_paragraphs
from .glossary_utils import (
    DEFAULT_GLOSSARY_PROMPT_LIMIT,
    GlossaryState,
    apply_suggestions_to_state,
    format_glossary_for_prompt,
    normalize_key,
    parse_glossary_suggestions,
    save_dynamic_glossary,
    select_terms_for_target_text,
    split_refined_and_suggestions,
)
from .language_guardrails import detect_residual_source_language
from .languages import normalize_source_language, source_language_name
from .llm_backend import LLMBackend
from .preprocess import chunk_for_refine, paragraphs_from_text
from .qa import (
    has_curly_quote_balance_regression,
    has_curly_quote_count_regression,
    has_malformed_quote_boundary,
    needs_retry,
)
from .quote_fix import (
    collapse_repeated_curly_quotes,
    count_curly_quotes,
    fix_blank_lines_inside_quotes,
    fix_unbalanced_quotes,
)
from .sanitizer import sanitize_refine_output
from .text_postprocess import (
    apply_custom_normalizers,
    apply_structural_normalizers,
    fix_dialogue_artifacts,
)
from .utils import ensure_dir, read_text, timed, write_text

REFINE_PIPELINE_VERSION = "17"


def refine_prompt_fingerprint(source_language: str = "en") -> str:
    """Calcula a assinatura da política de prompt usada no refino."""
    template = build_refine_prompt(
        "{section}",
        glossary_enabled=True,
        glossary_block="{glossary}",
        source_language=source_language,
    )
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _cache_signature_from(cfg: AppConfig, backend: LLMBackend, source_language: str = "en") -> dict:
    """Obtém a assinatura de compatibilidade armazenada no cache."""
    return {
        "backend": getattr(backend, "backend", None),
        "model": getattr(backend, "model", None),
        "num_predict": getattr(backend, "num_predict", None),
        "temperature": getattr(backend, "temperature", None),
        "repeat_penalty": getattr(backend, "repeat_penalty", None),
        "guardrails": getattr(cfg, "refine_guardrails", None),
        "source_language": normalize_source_language(source_language),
        "prompt_hash": refine_prompt_fingerprint(source_language),
        "pipeline_version": REFINE_PIPELINE_VERSION,
    }


def _glossary_hash(glossary_state: GlossaryState | None) -> str | None:
    """Calcula a assinatura estável do glossário aplicado ao refino."""
    if not glossary_state:
        return None
    try:
        payload = json.dumps(glossary_state.combined_index, ensure_ascii=False, sort_keys=True)
    except Exception:
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_cache_compatible(data: dict, signature: dict) -> bool:
    """Verifica se a entrada de cache corresponde à política atual."""
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        return False
    return all(meta.get(k) == v for k, v in signature.items())


@dataclass
class RefineStats:
    """Reúne as métricas produzidas durante o refino."""

    total_blocks: int = 0
    success_blocks: int = 0
    error_blocks: int = 0


@dataclass
class RefineProgress:
    """Representa o progresso persistido de uma execução de refino."""

    total_blocks: int
    refined_blocks: set[int]
    error_blocks: set[int]
    chunk_outputs: Dict[int, str]
    progress_path: Path | None


_CURRENT_STATS: RefineStats | None = None
_CURRENT_PROGRESS: RefineProgress | None = None
_GLOBAL_BLOCK_INDEX: int = 0


@contextmanager
def processing_context(stats: RefineStats, progress: RefineProgress | None):
    """Controla o ciclo de vida e as métricas de uma etapa de refino."""
    global _CURRENT_STATS, _CURRENT_PROGRESS, _GLOBAL_BLOCK_INDEX
    prev_stats = _CURRENT_STATS
    prev_progress = _CURRENT_PROGRESS
    prev_counter = _GLOBAL_BLOCK_INDEX
    _CURRENT_STATS = stats
    _CURRENT_PROGRESS = progress
    _GLOBAL_BLOCK_INDEX = 0
    try:
        yield
    finally:
        _CURRENT_STATS = prev_stats
        _CURRENT_PROGRESS = prev_progress
        _GLOBAL_BLOCK_INDEX = prev_counter


def _next_block_index() -> int:
    """Determina o próximo bloco pendente a partir do progresso salvo."""
    global _GLOBAL_BLOCK_INDEX
    _GLOBAL_BLOCK_INDEX += 1
    return _GLOBAL_BLOCK_INDEX


def _write_guardrail_debug_file(
    base_dir: Path,
    section_index: int,
    chunk_index: int,
    block_index: int,
    reasons: list[str],
    guardrails_mode: str,
    collapse_flag: bool,
    collapse_details: dict | None = None,
) -> None:
    """Grava detalhes de uma rejeição quando a depuração está ativa."""
    if not reasons and not collapse_flag:
        return
    debug_dir = Path(base_dir) / "debug_refine_guardrails"
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "section_index": section_index,
        "chunk_index": chunk_index,
        "block_index": block_index,
        "guardrails_mode": guardrails_mode,
        "collapse_detected": collapse_flag,
        "reasons": reasons or [],
        "collapse_details": collapse_details or {},
        "timestamp": datetime.now().isoformat(),
    }
    path = debug_dir / f"refine_chunk{chunk_index:03d}_block{block_index:04d}_guardrails.json"
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def has_suspicious_repetition(text: str, min_repeats: int = 3) -> bool:
    """
    Sinais fortes de repetição/loop.
    Requer pelo menos 2 dos 3 sinais:
    - bloco >=120 chars repetido
    - diversidade lexical muito baixa
    - sentença/parágrafo idêntico repetido >= min_repeats
    """
    if not text or not text.strip():
        return False
    signals = 0
    if re.search(r"(.{120,}?)(?:\s+\1){1,}", text, flags=re.DOTALL):
        signals += 1
    tokens = text.split()
    if len(tokens) >= 40:
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        if unique_ratio < 0.25:
            signals += 1
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    counts: Dict[str, int] = {}
    for s in sentences:
        counts[s] = counts.get(s, 0) + 1
    if any(c >= min_repeats for c in counts.values()):
        signals += 1
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    para_counts: Dict[str, int] = {}
    for p in paragraphs:
        para_counts[p] = para_counts.get(p, 0) + 1
    if any(c >= min_repeats for c in para_counts.values()):
        signals += 1
    return signals >= 2


def has_meta_noise(text: str) -> bool:
    """Detecta meta-texto óbvio que não deve aparecer na saída final."""
    lower = text.lower()
    markers = [
        "as an ai",
        "as a language model",
        "sou um modelo de linguagem",
        "como um modelo de linguagem",
        "<think>",
        "</think>",
    ]
    return any(m in lower for m in markers)


def _count_paragraphs(text: str) -> int:
    """Conta parágrafos."""
    return len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()])


def _count_leading_quote_dialogues(text: str) -> int:
    """Conta diálogos iniciados por aspas."""
    return sum(1 for line in text.splitlines() if line.strip().startswith(('"', "“", "”")))


def _count_leading_dash_dialogues(text: str) -> int:
    """Conta iniciais travessão diálogos."""
    return sum(1 for line in text.splitlines() if line.strip().startswith("—"))


def _count_nonblank_lines(text: str) -> int:
    """Conta as linhas não vazias do trecho."""
    return sum(1 for line in text.splitlines() if line.strip())


def _dialogue_or_paragraph_regression(
    original: str,
    cleaned: str,
    *,
    allowed_paragraph_increase: int = 0,
    allowed_paragraph_decrease: int = 0,
    allowed_line_increase: int = 0,
    allowed_line_decrease: int = 0,
) -> dict:
    """Detecta perdas estruturais de diálogos ou parágrafos no refino."""
    original_quote_lines = _count_leading_quote_dialogues(original)
    cleaned_quote_lines = _count_leading_quote_dialogues(cleaned)
    original_dash_lines = _count_leading_dash_dialogues(original)
    cleaned_dash_lines = _count_leading_dash_dialogues(cleaned)
    original_paragraphs = _count_paragraphs(original)
    cleaned_paragraphs = _count_paragraphs(cleaned)
    original_nonblank_lines = _count_nonblank_lines(original)
    cleaned_nonblank_lines = _count_nonblank_lines(cleaned)

    introduced_dash_dialogues = (
        original_dash_lines == 0 and original_quote_lines > 0 and cleaned_dash_lines > 0
    )
    introduced_quote_dialogues = (
        original_quote_lines == 0 and original_dash_lines > 0 and cleaned_quote_lines > 0
    )
    lost_dash_dialogues = original_dash_lines >= 2 and cleaned_dash_lines < max(
        1, original_dash_lines - 1
    )
    lost_quote_dialogues = original_quote_lines >= 2 and cleaned_quote_lines < max(
        1, original_quote_lines - 1
    )

    paragraph_structure_changed = False
    if original_paragraphs >= 4:
        max_paragraphs = original_paragraphs + max(0, allowed_paragraph_increase)
        min_paragraphs = max(1, original_paragraphs - max(0, allowed_paragraph_decrease))
        paragraph_structure_changed = (
            cleaned_paragraphs > max_paragraphs or cleaned_paragraphs < min_paragraphs
        )

    line_structure_changed = False
    if original_nonblank_lines >= 2:
        max_lines = original_nonblank_lines + max(0, allowed_line_increase)
        min_lines = max(1, original_nonblank_lines - max(0, allowed_line_decrease))
        line_structure_changed = (
            cleaned_nonblank_lines > max_lines or cleaned_nonblank_lines < min_lines
        )

    return {
        "dialogue_style_changed": introduced_dash_dialogues
        or introduced_quote_dialogues
        or lost_dash_dialogues
        or lost_quote_dialogues,
        "paragraph_structure_changed": paragraph_structure_changed,
        "line_structure_changed": line_structure_changed,
        "original_quote_lines": original_quote_lines,
        "cleaned_quote_lines": cleaned_quote_lines,
        "original_dash_lines": original_dash_lines,
        "cleaned_dash_lines": cleaned_dash_lines,
        "original_paragraphs": original_paragraphs,
        "cleaned_paragraphs": cleaned_paragraphs,
        "original_nonblank_lines": original_nonblank_lines,
        "cleaned_nonblank_lines": cleaned_nonblank_lines,
    }


def sanitize_refine_chunk_output(
    text: str,
    original: str,
    logger: logging.Logger | None = None,
    label: str | None = None,
) -> tuple[str, bool, dict]:
    """Sanitiza um chunk refinado sem reestruturar seus parágrafos."""
    stats = {
        "blank_lines_fixed": 0,
        "dialogue_splits": 0,
        "repeated_curly_quotes_fixed": 0,
    }
    cleaned = re.sub(r'"""+\s*$', "", text, flags=re.MULTILINE)
    cleaned, repeated_curly_quotes_fixed = collapse_repeated_curly_quotes(cleaned)
    stats["repeated_curly_quotes_fixed"] = repeated_curly_quotes_fixed
    cleaned, fixes = fix_blank_lines_inside_quotes(cleaned, logger=logger, label=label)
    stats["blank_lines_fixed"] = fixes
    cleaned, count_split = re.subn(r"”\s+“", "”\n\n“", cleaned)
    stats["dialogue_splits"] = count_split
    cleaned, tag_joins = re.subn(
        r"”\s*\n\s*\n\s*(?=(perguntou|disse|respondeu|murmurou|exclamou)\b)",
        "” ",
        cleaned,
        flags=re.IGNORECASE,
    )
    stats["dialogue_tag_joins"] = tag_joins

    artifacts = '"""' in cleaned
    input_malformed_quote_boundary = has_malformed_quote_boundary(original)
    malformed_quote_boundary = has_malformed_quote_boundary(cleaned)
    introduced_malformed_quote_boundary = (
        malformed_quote_boundary and not input_malformed_quote_boundary
    )
    structure_info = _dialogue_or_paragraph_regression(
        original,
        cleaned,
        allowed_paragraph_increase=count_split,
        allowed_paragraph_decrease=tag_joins + fixes,
        allowed_line_increase=count_split,
        allowed_line_decrease=tag_joins + fixes,
    )
    opens_curly, closes_curly = count_curly_quotes(cleaned)
    regression_dialogue = ("”\n\n“" in original) and ("” “" in cleaned)
    quotes_balanced = not has_curly_quote_balance_regression(original, cleaned)
    introduced_extra_curly_quotes = has_curly_quote_count_regression(original, cleaned)
    soft_retry = False
    ok = True
    if (
        artifacts
        or regression_dialogue
        or structure_info["dialogue_style_changed"]
        or structure_info["paragraph_structure_changed"]
        or structure_info["line_structure_changed"]
    ):
        ok = False
    elif not quotes_balanced or introduced_extra_curly_quotes:
        soft_retry = True
    elif introduced_malformed_quote_boundary:
        # O modelo pode equilibrar a contagem global de aspas e ainda iniciar
        # uma fala com um fechamento espúrio (”“Fala). Refaça uma vez antes
        # de recorrer ao chunk original.
        soft_retry = True
    return (
        cleaned,
        ok,
        {
            "artifacts": artifacts,
            "quotes_balanced": quotes_balanced,
            "introduced_extra_curly_quotes": introduced_extra_curly_quotes,
            "malformed_quote_boundary": malformed_quote_boundary,
            "introduced_malformed_quote_boundary": introduced_malformed_quote_boundary,
            "regression_dialogue": regression_dialogue,
            "soft_retry": soft_retry,
            **structure_info,
            **stats,
        },
    )


def save_refine_debug_files(
    output_dir: Path,
    section_index: int,
    chunk_index: int,
    original_text: str,
    llm_raw: str,
    final_text: str,
    logger: logging.Logger,
) -> None:
    """Grava artefatos de depuração do refino por chunk."""
    ensure_dir(output_dir)
    base = f"sec{section_index:03d}_chunk{chunk_index:03d}"

    def _write(name: str, content: str) -> None:
        """Persiste o estado intermediário de forma atômica."""
        path = output_dir / f"{base}_{name}.txt"
        path.write_text(content, encoding="utf-8")

    _write("original", original_text)
    _write("llm_raw", llm_raw)
    _write("final", final_text)
    logger.info("Debug refine salvo: %s_* em %s", base, output_dir)


def _write_progress(progress: RefineProgress | None, logger: logging.Logger) -> None:
    """Grava progresso."""
    if progress is None or progress.progress_path is None:
        return
    data = {
        "total_blocks": progress.total_blocks,
        "refined_blocks": sorted(progress.refined_blocks),
        "error_blocks": sorted(progress.error_blocks),
        "timestamp": datetime.now().isoformat(),
        "chunks": {str(idx): text for idx, text in progress.chunk_outputs.items()},
    }
    try:
        progress.progress_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # pragma: no cover - I/O edge case
        logger.warning("Falha ao gravar manifesto de refine em %s: %s", progress.progress_path, exc)


def _prepare_progress(
    progress_path: Path,
    resume_manifest: dict | None,
    total_blocks: int,
    logger: logging.Logger,
) -> RefineProgress:
    """Prepara progresso."""
    refined: set[int] = set()
    errored: set[int] = set()
    chunk_outputs: Dict[int, str] = {}
    manifest_total = None

    data = resume_manifest
    if isinstance(data, dict):
        manifest_total = data.get("total_blocks")
        if isinstance(manifest_total, int) and manifest_total != total_blocks:
            logger.warning(
                "Manifesto indica %d blocos, mas chunking atual gerou %d; usando chunking atual.",
                manifest_total,
                total_blocks,
            )
        raw_chunks = data.get("chunks") or {}
        if isinstance(raw_chunks, dict):
            for key, val in raw_chunks.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                if isinstance(val, str):
                    chunk_outputs[idx] = val

        raw_refined = data.get("refined_blocks") or []
        for idx in raw_refined:
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            if idx_int in chunk_outputs:
                refined.add(idx_int)
            else:
                logger.warning(
                    "Manifesto marca bloco %s como refinado, mas não há conteúdo salvo; refinando novamente.",
                    idx_int,
                )

        raw_error = data.get("error_blocks") or []
        for idx in raw_error:
            try:
                errored.add(int(idx))
            except (TypeError, ValueError):
                continue

    return RefineProgress(
        total_blocks=total_blocks,
        refined_blocks=refined,
        error_blocks=errored,
        chunk_outputs=chunk_outputs,
        progress_path=progress_path,
    )


def build_refine_prompt(
    section: str,
    glossary_enabled: bool = False,
    glossary_block: str | None = None,
    source_language: str = "en",
) -> str:
    """Monta o prompt de revisão da tradução em português."""
    source_name = source_language_name(source_language)
    glossary_section = ""
    if glossary_enabled and glossary_block:
        glossary_section = (
            f"\nUse como referencia (sem adicionar explicacoes) o glossario a seguir:\n"
            f"{glossary_block}\n"
        )

    prompt = f"""
FORMATO CRÍTICO: preserve exatamente o marcador de diálogo do texto de entrada. Se ele usa aspas curvas, mantenha aspas curvas em todas as falas; não use travessões. Uma resposta que troque aspas por travessões será descartada.
Comece a resposta diretamente com `### TEXTO_REFINADO_INICIO` e termine com `### TEXTO_REFINADO_FIM`. Não escreva introdução, explicações, notas, separadores `***` nem lista de ajustes.

Você é um EDITOR PROFISSIONAL DE LIGHT NOVELS, responsável por transformar um texto traduzido para o português brasileiro em uma versão natural, fluida, coerente, com tom literário e qualidade de publicação.
Não altere absolutamente nada da história, dos eventos, das falas, da linha do tempo ou do conteúdo original. Apenas melhore a escrita.

REGRAS DE PRESERVAÇÃO:
- Preserve nomes próprios e honoríficos (-san, -kun, etc.) exatamente como no texto de entrada; não invente nem remova.
- Preserve o estilo de marcação de diálogo do texto de entrada (aspas curvas e/ou travessões). Não converta travessões em aspas nem vice-versa.
- Se uma fala começa com aspas, ela deve continuar começando com aspas; se começa com travessão, deve continuar com travessão.
- Nunca misture travessão e aspas na mesma fala por reformatacao. Exemplo proibido: — Fala", disse ele.
- Preserve a quantidade e a ordem dos parágrafos. Não transforme cada frase em um parágrafo separado.
- Não insira quebras de linha simples dentro de um parágrafo; mantenha o parágrafo em uma linha quando ele vier em uma linha.
- Não altere apelidos/insultos; apenas corrija gramática, pontuação e fluidez.
- Nunca remova conteúdo; não resuma; não pule linhas; não introduza aspas triplas.
- Se um trecho já estiver bom, mantenha-o como está. O refine deve ser mínimo e local, não uma reescrita completa.

OBJETIVOS DO EDITOR:

1. Naturalizar o português brasileiro, transformando frases literais em frases fluidas, claras e naturais.
2. Corrigir gagueiras mal traduzidas, ajustando para formas naturais em PT-BR:

   * "H-he..." → "E-ele..." ou apenas "Ele..."
3. Corrigir construções calcadas no idioma de origem e falsos cognatos.
4. Remover ruídos de OCR/PDF como aspas triplas \"\"\" e caracteres soltos.
5. Melhorar ritmo, coerência e pontuação de diálogos.
6. Remover calques literais:

   * "O que é com essa atitude de superioridade?!" → "Que atitude é essa, todo se achando?!"
7. Padronizar fluidez narrativa.
8. Remover repetições consecutivas ou quase idênticas geradas na tradução/refine (falas ou narrativas), mantendo apenas uma ocorrência completa e bem formatada.
9. Reunir trechos que foram colados na mesma linha por erro (ex.: “Mmm?” “Por que você…”) devolvendo fluxo natural de diálogo, sem alterar sentido.
  10. Manter consistência de gênero/narrador (masculino/feminino) conforme o original; não inverter narrador masculino.
  11. Se alguma frase, fala ou trecho narrativo ainda estiver em {source_name}, traduza esse trecho para português brasileiro natural, preservando apenas nomes próprios, honoríficos e termos canônicos do glossário.
  12. Corrigir concordância verbal e nominal, possessivos literais e frases sem verbo principal. Cada período deve permanecer gramatical e compreensível sozinho, sem inventar informação para completar lacunas.
  13. Quando a tradução trouxer duas aspas curvas coladas ou um fechamento de aspa sem abertura na mesma fala, restaure apenas a pontuação necessária e preserve o conteúdo.
  14. Preserve o tempo verbal narrativo do trecho: uma cena narrada no passado deve continuar no passado.
 15. Traduza vocabulário comum em {source_name}, salvo quando o glossário o marcar explicitamente como termo canônico.
  16. Corrija regência e complementos calcados no idioma de origem. Exemplos: "convencer alguém a me acreditar" deve virar "convencer alguém a acreditar em mim"; "tem X para considerar" deve virar "também é preciso considerar X".

PROIBIÇÕES ABSOLUTAS:

* Não resumir.
* Não cortar falas.
* Não alterar eventos.
* Não adicionar conteúdo.
* Não mudar tom ou personalidade dos personagens.
* Não reorganizar parágrafos.
* Não dividir parágrafos.
* Não inserir quebras de linha apenas por estilo.
  * Não fundir parágrafos ou alterar segmentação original.
  * Não converter o padrão de diálogo do trecho.
  * Não deixar frases em {source_name} quando o restante do trecho está em português.
  * Não conservar um erro gramatical apenas por estar presente na tradução de entrada.

FORMATO DE SAÍDA:
Retorne apenas:

### TEXTO_REFINADO_INICIO

<texto refinado>
### TEXTO_REFINADO_FIM

Nada antes ou depois dos marcadores.
Comece a resposta exatamente com `### TEXTO_REFINADO_INICIO`; não escreva introdução, separadores `***`, explicações, lista de ajustes ou notas.

{glossary_section}Texto para revisao (PT-BR):
\"\"\"{section}\"\"\"
"""
    return prompt


def split_markdown_sections(md_text: str) -> List[Tuple[str, str]]:
    """
    Divide o Markdown em seções por headings `#` até `######`.

    Retorna lista de tuplas (título, corpo). Se não houver headings,
    retorna uma única seção com título vazio. Preserva prefixo antes do primeiro heading.
    """
    pattern = re.compile(r"^#{1,6}\s+.+$", flags=re.MULTILINE)
    matches = list(pattern.finditer(md_text))
    sections: List[Tuple[str, str]] = []

    if not matches:
        return [("", md_text.strip())]

    # Prefixo antes do primeiro heading
    prefix = md_text[: matches[0].start()].strip()
    if prefix:
        sections.append(("", prefix))

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md_text)
        title = match.group().strip()
        body = md_text[start:end].strip()
        sections.append((title, body))
    return sections


def refine_section(
    title: str,
    body: str,
    backend: LLMBackend,
    cfg: AppConfig,
    logger: logging.Logger,
    index: int,
    total: int,
    glossary_state: GlossaryState | None = None,
    glossary_prompt_limit: int = DEFAULT_GLOSSARY_PROMPT_LIMIT,
    debug_refine: bool = False,
    metrics: dict | None = None,
    seen_chunks: list | None = None,
    debug_writer: Callable[[dict], None] | None = None,
    debug_run: DebugRunWriter | None = None,
    manifest_chunks: list[dict] | None = None,
    source_language: str = "en",
) -> str:
    """
    Executa o processo de refinamento em uma única seção (capítulo) do texto,
    dividindo-a em chunks menores e processando cada um iterativamente.
    """
    resolved_source_language = normalize_source_language(source_language)
    if resolved_source_language == "auto":
        resolved_source_language = "en"
    if metrics is None:
        metrics = {}
    if seen_chunks is None:
        seen_chunks = []
    paragraphs = paragraphs_from_text(body)
    chunks = chunk_for_refine(paragraphs, max_chars=cfg.refine_chunk_chars, logger=logger)
    logger.info("Refinando seção %s (%d chunks)", title or f"#{index}", len(chunks))
    refined_parts: List[str] = []
    stats = _CURRENT_STATS
    progress = _CURRENT_PROGRESS
    glossary_block = None
    cache_signature = _cache_signature_from(cfg, backend, resolved_source_language)

    for c_idx, chunk in enumerate(chunks, start=1):
        block_idx = _next_block_index()
        record_chunk = bool(
            debug_run and manifest_chunks is not None and debug_run.should_write_chunk(block_idx)
        )
        if stats:
            stats.total_blocks += 1
        guard_mode = getattr(cfg, "refine_guardrails", "strict")
        h = chunk_hash(chunk)
        block_metrics = metrics.setdefault("block_metrics", [])
        fallback_reasons: list[str] = []
        retry_reasons: list[str] = []
        llm_attempts = 0
        from_cache = False
        from_duplicate = False
        error_message: str | None = None

        def _maybe_write_debug_files(
            original_text: str, llm_raw_text: str | None, final_text: str
        ) -> None:
            """Grava os artefatos detalhados somente quando a depuração está ativa."""
            if not debug_run or not debug_run.should_write_chunk(block_idx):
                return
            debug_stage_dir = debug_run.stage_dir("60_refine") / "debug_refine"
            debug_stage_dir.mkdir(parents=True, exist_ok=True)
            debug_run.write_text(
                debug_run.rel_path(debug_stage_dir / f"chunk{block_idx:03d}_original_pt.txt"),
                original_text,
            )
            debug_run.write_text(
                debug_run.rel_path(debug_stage_dir / f"chunk{block_idx:03d}_context.txt"),
                "",
            )
            if llm_raw_text is not None:
                raw_hash = debug_run.sha256_text(llm_raw_text)
                if not debug_run.store_llm_raw:
                    llm_payload = f"[[OMITTED]]\n[[SHA256:{raw_hash}]]\n"
                elif (
                    debug_run.max_chars_per_file
                    and len(llm_raw_text) > debug_run.max_chars_per_file
                ):
                    truncated = llm_raw_text[: debug_run.max_chars_per_file]
                    llm_payload = f"{truncated}\n\n[[TRUNCATED]]\n[[SHA256:{raw_hash}]]\n"
                else:
                    llm_payload = llm_raw_text
                debug_run.write_text(
                    debug_run.rel_path(debug_stage_dir / f"chunk{block_idx:03d}_llm_raw.txt"),
                    llm_payload,
                    allow_truncate=False,
                )
            debug_run.write_text(
                debug_run.rel_path(debug_stage_dir / f"chunk{block_idx:03d}_final_pt.txt"),
                final_text,
            )

        def _apply_normalizers(text: str) -> tuple[str, dict]:
            """Aplica os normalizadores determinísticos à saída refinada."""
            normalized, norm_stats = apply_structural_normalizers(text)
            normalized = apply_custom_normalizers(normalized, convert_quote_dialogues=False)
            metrics["dialogue_splits"] = metrics.get("dialogue_splits", 0) + norm_stats.get(
                "dialogue_splits", 0
            )
            metrics["triple_quotes_removed"] = metrics.get(
                "triple_quotes_removed", 0
            ) + norm_stats.get("triple_quotes_removed", 0)
            return normalized, norm_stats

        def record_block(
            final_text: str,
            *,
            used_fallback: bool = False,
            from_cache: bool = False,
            from_duplicate: bool = False,
            collapse: bool = False,
            normalizer_stats: dict | None = None,
            guardrail_reasons: list[str] | None = None,
        ) -> None:
            """Registra bloco."""
            ratio = (len(final_text.strip()) / max(len(chunk.strip()), 1)) if chunk.strip() else 0.0
            residual_source, residual_source_reason = detect_residual_source_language(
                final_text, resolved_source_language
            )
            block_metrics.append(
                {
                    "block_index": block_idx,
                    "chars_in": len(chunk),
                    "chars_out": len(final_text),
                    "ratio_out_in": round(ratio, 3),
                    "used_fallback": used_fallback,
                    "guardrails_mode": guard_mode,
                    "suspicious_repetition": has_suspicious_repetition(final_text),
                    "source_language": resolved_source_language,
                    "residual_source_language": residual_source,
                    "residual_source_language_reason": residual_source_reason,
                    "residual_english": residual_source
                    if resolved_source_language == "en"
                    else False,
                    "residual_english_reason": residual_source_reason
                    if resolved_source_language == "en"
                    else "",
                    "from_cache": from_cache,
                    "from_duplicate": from_duplicate,
                    "collapse_detected": collapse,
                    "dialogue_splits": (normalizer_stats or {}).get("dialogue_splits", 0),
                    "triple_quotes_removed": (normalizer_stats or {}).get(
                        "triple_quotes_removed", 0
                    ),
                    "guardrail_reasons": guardrail_reasons or [],
                }
            )

        llm_raw: str | None = None
        for prev_chunk, prev_final in seen_chunks:
            if is_near_duplicate(prev_chunk, chunk) and is_duplicate_reuse_safe(prev_chunk, chunk):
                logger.info(
                    "Chunk ref-%d/%d-%d/%d marcado como duplicado; reuso habilitado.",
                    index,
                    total,
                    c_idx,
                    len(chunks),
                )
                normalized_dup, norm_stats = _apply_normalizers(prev_final)
                refined_parts.append(normalized_dup)
                metrics["duplicates"] = metrics.get("duplicates", 0) + 1
                from_duplicate = True
                if stats:
                    stats.success_blocks += 1
                if progress:
                    progress.refined_blocks.add(block_idx)
                    progress.error_blocks.discard(block_idx)
                    progress.chunk_outputs[block_idx] = normalized_dup
                _write_progress(progress, logger)
                record_block(normalized_dup, from_duplicate=True, normalizer_stats=norm_stats)
                if debug_writer:
                    debug_writer(
                        {
                            "para_index": block_idx,
                            "original_text": chunk,
                            "original_chars": len(chunk),
                            "refined_text": normalized_dup,
                            "refined_chars": len(normalized_dup),
                            "llm_raw_output": None,
                            "sanitizer_report": None,
                            "normalizer_stats": norm_stats,
                        }
                    )
                _maybe_write_debug_files(chunk, None, normalized_dup)
                if record_chunk:
                    debug_stage_dir = debug_run.stage_dir("60_refine") / "debug_refine"
                    outputs_payload = {
                        "debug_original": debug_run.rel_path(
                            debug_stage_dir / f"chunk{block_idx:03d}_original_pt.txt"
                        ),
                        "debug_context": debug_run.rel_path(
                            debug_stage_dir / f"chunk{block_idx:03d}_context.txt"
                        ),
                        "debug_llm_raw": debug_run.rel_path(
                            debug_stage_dir / f"chunk{block_idx:03d}_llm_raw.txt"
                        ),
                        "debug_final": debug_run.rel_path(
                            debug_stage_dir / f"chunk{block_idx:03d}_final_pt.txt"
                        ),
                        "output_hash": debug_run.sha256_text(normalized_dup),
                    }
                    manifest_chunks.append(
                        {
                            "chunk_index": block_idx,
                            "section_index": index,
                            "section_title": title or "",
                            "input_hash": debug_run.sha256_text(chunk),
                            "chars_in": len(chunk),
                            "context_hash": None,
                            "from_cache": False,
                            "from_duplicate": True,
                            "llm_attempts": 0,
                            "retry_reasons": [],
                            "suspect_output": False,
                            "suspect_reason": "",
                            "contamination_detected": False,
                            "sanitization_ratio": None,
                            "normalizers": {
                                "triple_quotes_removed": norm_stats.get("triple_quotes_removed", 0),
                                "dialogue_splits": norm_stats.get("dialogue_splits", 0),
                            },
                            "lengths": {
                                "chars_out": len(normalized_dup),
                                "ratio_out_in": round(
                                    len(normalized_dup.strip()) / max(len(chunk.strip()), 1),
                                    3,
                                )
                                if chunk.strip()
                                else 0.0,
                            },
                            "outputs": outputs_payload,
                            "errors": None,
                        }
                    )
                continue
        if cache_exists("refine", h):
            data = load_cache("refine", h)
            if not _is_cache_compatible(data, cache_signature):
                logger.debug(
                    "Cache de refine ignorado: assinatura diferente de backend/model/num_predict."
                )
            else:
                cached = data.get("final_output")
                if cached:
                    logger.info(
                        "Reusando cache de refine para bloco ref-%d/%d-%d/%d",
                        index,
                        total,
                        c_idx,
                        len(chunks),
                    )
                    normalized_cached, norm_stats = _apply_normalizers(cached)
                    refined_parts.append(normalized_cached)
                    metrics["cache_hits"] = metrics.get("cache_hits", 0) + 1
                    from_cache = True
                    if stats:
                        stats.success_blocks += 1
                    if progress:
                        progress.refined_blocks.add(block_idx)
                        progress.error_blocks.discard(block_idx)
                        progress.chunk_outputs[block_idx] = normalized_cached
                    _write_progress(progress, logger)
                    record_block(normalized_cached, from_cache=True, normalizer_stats=norm_stats)
                    if debug_writer:
                        debug_writer(
                            {
                                "para_index": block_idx,
                                "original_text": chunk,
                                "original_chars": len(chunk),
                                "refined_text": normalized_cached,
                                "refined_chars": len(normalized_cached),
                                "llm_raw_output": None,
                                "sanitizer_report": None,
                                "normalizer_stats": norm_stats,
                            }
                        )
                    _maybe_write_debug_files(chunk, None, normalized_cached)
                    if record_chunk:
                        debug_stage_dir = debug_run.stage_dir("60_refine") / "debug_refine"
                        outputs_payload = {
                            "debug_original": debug_run.rel_path(
                                debug_stage_dir / f"chunk{block_idx:03d}_original_pt.txt"
                            ),
                            "debug_context": debug_run.rel_path(
                                debug_stage_dir / f"chunk{block_idx:03d}_context.txt"
                            ),
                            "debug_llm_raw": debug_run.rel_path(
                                debug_stage_dir / f"chunk{block_idx:03d}_llm_raw.txt"
                            ),
                            "debug_final": debug_run.rel_path(
                                debug_stage_dir / f"chunk{block_idx:03d}_final_pt.txt"
                            ),
                            "output_hash": debug_run.sha256_text(normalized_cached),
                        }
                        manifest_chunks.append(
                            {
                                "chunk_index": block_idx,
                                "section_index": index,
                                "section_title": title or "",
                                "input_hash": debug_run.sha256_text(chunk),
                                "chars_in": len(chunk),
                                "context_hash": None,
                                "from_cache": True,
                                "from_duplicate": False,
                                "llm_attempts": 0,
                                "retry_reasons": [],
                                "suspect_output": False,
                                "suspect_reason": "",
                                "contamination_detected": False,
                                "sanitization_ratio": None,
                                "normalizers": {
                                    "triple_quotes_removed": norm_stats.get(
                                        "triple_quotes_removed", 0
                                    ),
                                    "dialogue_splits": norm_stats.get("dialogue_splits", 0),
                                },
                                "lengths": {
                                    "chars_out": len(normalized_cached),
                                    "ratio_out_in": round(
                                        len(normalized_cached.strip()) / max(len(chunk.strip()), 1),
                                        3,
                                    )
                                    if chunk.strip()
                                    else 0.0,
                                },
                                "outputs": outputs_payload,
                                "errors": None,
                            }
                        )
                    continue
        if (
            progress
            and block_idx in progress.refined_blocks
            and block_idx in progress.chunk_outputs
        ):
            logger.info(
                "Reusando refinamento salvo para bloco ref-%d/%d-%d/%d",
                index,
                total,
                c_idx,
                len(chunks),
            )
            normalized_cached, norm_stats = _apply_normalizers(progress.chunk_outputs[block_idx])
            refined_parts.append(normalized_cached)
            if stats:
                stats.success_blocks += 1
            _write_progress(progress, logger)
            progress.chunk_outputs[block_idx] = normalized_cached
            record_block(normalized_cached, normalizer_stats=norm_stats)
            if debug_writer:
                reused = normalized_cached
                debug_writer(
                    {
                        "para_index": block_idx,
                        "original_text": chunk,
                        "original_chars": len(chunk),
                        "refined_text": reused,
                        "refined_chars": len(reused),
                        "llm_raw_output": None,
                        "sanitizer_report": None,
                        "normalizer_stats": norm_stats,
                    }
                )
            _maybe_write_debug_files(chunk, None, normalized_cached)
            if record_chunk:
                debug_stage_dir = debug_run.stage_dir("60_refine") / "debug_refine"
                outputs_payload = {
                    "debug_original": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_original_pt.txt"
                    ),
                    "debug_context": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_context.txt"
                    ),
                    "debug_llm_raw": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_llm_raw.txt"
                    ),
                    "debug_final": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_final_pt.txt"
                    ),
                    "output_hash": debug_run.sha256_text(normalized_cached),
                }
                manifest_chunks.append(
                    {
                        "chunk_index": block_idx,
                        "section_index": index,
                        "section_title": title or "",
                        "input_hash": debug_run.sha256_text(chunk),
                        "chars_in": len(chunk),
                        "context_hash": None,
                        "from_cache": False,
                        "from_duplicate": False,
                        "llm_attempts": 0,
                        "retry_reasons": [],
                        "suspect_output": False,
                        "suspect_reason": "",
                        "contamination_detected": False,
                        "sanitization_ratio": None,
                        "normalizers": {
                            "triple_quotes_removed": norm_stats.get("triple_quotes_removed", 0),
                            "dialogue_splits": norm_stats.get("dialogue_splits", 0),
                        },
                        "lengths": {
                            "chars_out": len(normalized_cached),
                            "ratio_out_in": round(
                                len(normalized_cached.strip()) / max(len(chunk.strip()), 1),
                                3,
                            )
                            if chunk.strip()
                            else 0.0,
                        },
                        "outputs": outputs_payload,
                        "errors": None,
                    }
                )
            continue
        if glossary_state:
            selected_terms, _ = select_terms_for_target_text(
                list(glossary_state.combined_index.values()),
                chunk,
                match_limit=glossary_prompt_limit,
            )
            selected_index = {
                normalize_key(str(term.get("key", ""))): term
                for term in selected_terms
                if str(term.get("key", "")).strip()
            }
            glossary_block = format_glossary_for_prompt(selected_index, glossary_prompt_limit)
        else:
            glossary_block = None
        prompt = build_refine_prompt(
            chunk,
            glossary_enabled=bool(glossary_block),
            glossary_block=glossary_block,
            source_language=resolved_source_language,
        )
        logger.debug("Refinando seção com %d caracteres...", len(chunk))
        try:
            attempt = 0
            while True:
                llm_raw, response_text = _call_with_retry(
                    backend=backend,
                    prompt=prompt,
                    cfg=cfg,
                    logger=logger,
                    label=f"ref-{index}/{total}-{c_idx}/{len(chunks)}",
                    max_retries=1,
                )
                llm_attempts += 1
                refined_candidate = response_text
                if glossary_state:
                    raw_without_suggestions, suggestion_block = split_refined_and_suggestions(
                        llm_raw
                    )
                    if suggestion_block is not None:
                        refined_candidate = sanitize_refine_output(raw_without_suggestions)
                    suggestions = parse_glossary_suggestions(suggestion_block or "")
                    if suggestions:
                        updated = apply_suggestions_to_state(glossary_state, suggestions, logger)
                        if updated:
                            save_dynamic_glossary(glossary_state, logger)

                collapse_flag = False
                collapse_reasons = None
                collapse_details = None
                fallback_reasons = []
                used_fallback = False
                if guard_mode == "off":
                    refined_text = refined_candidate
                    if (
                        not refined_text.strip()
                        or has_meta_noise(refined_text)
                        or has_meta_noise(llm_raw or "")
                    ):
                        used_fallback = True
                        fallback_reasons.append("empty_or_meta_guardrail")
                    else:
                        collapse_flag, collapse_reasons, collapse_details = detect_model_collapse(
                            refined_text,
                            original_len=len(chunk),
                            mode="refine",
                            return_reasons=True,
                        )
                        if collapse_flag:
                            used_fallback = True
                            fallback_reasons.append("collapse_detector")
                elif guard_mode == "relaxed":
                    filtered_text = anti_hallucination_filter(
                        orig=chunk,
                        llm_raw=llm_raw,
                        cleaned=refined_candidate,
                        mode="refine",
                    )
                    if (
                        filtered_text == chunk
                        and refined_candidate.strip()
                        and refined_candidate.strip() != chunk.strip()
                    ):
                        refined_text = refined_candidate
                    else:
                        refined_text = filtered_text
                    severe_issue = False
                    if not refined_text.strip():
                        severe_issue = True
                        fallback_reasons.append("empty_after_guardrail")
                    elif has_meta_noise(refined_text) or has_meta_noise(llm_raw or ""):
                        severe_issue = True
                        fallback_reasons.append("meta_noise")
                    else:
                        collapse_flag, collapse_reasons, collapse_details = detect_model_collapse(
                            refined_text,
                            original_len=len(chunk),
                            mode="refine",
                            return_reasons=True,
                        )
                        if collapse_flag:
                            severe_issue = True
                            fallback_reasons.append("collapse_detector")
                    if severe_issue:
                        used_fallback = True
                else:  # strict
                    refined_text = anti_hallucination_filter(
                        orig=chunk,
                        llm_raw=llm_raw,
                        cleaned=refined_candidate,
                        mode="refine",
                    )
                    if refined_text == chunk and refined_candidate.strip() != chunk.strip():
                        used_fallback = True
                        fallback_reasons.append("anti_hallucination_filter")
                    if not refined_text.strip():
                        used_fallback = True
                        fallback_reasons.append("empty_after_guardrail")
                    else:
                        collapse_flag, collapse_reasons, collapse_details = detect_model_collapse(
                            refined_text,
                            original_len=len(chunk),
                            mode="refine",
                            return_reasons=True,
                        )
                        if collapse_flag:
                            used_fallback = True
                            fallback_reasons.append("collapse_detector")

                fmt_soft_retry = False
                if not used_fallback:
                    sanitized_refined, ok_fmt, fmt_info = sanitize_refine_chunk_output(
                        refined_text, chunk, logger=logger, label=f"ref-{index}-{c_idx}"
                    )
                    fmt_soft_retry = fmt_info.get("soft_retry", False)
                    if not ok_fmt and not fmt_soft_retry:
                        used_fallback = True
                        fallback_reasons.append(f"format_validation:{fmt_info}")
                        if debug_refine:
                            fail_dir = cfg.output_dir / "debug_refine_failed"
                            fail_dir.mkdir(parents=True, exist_ok=True)
                            (fail_dir / f"ref_{index}_{c_idx}_raw.txt").write_text(
                                refined_text, encoding="utf-8"
                            )
                            (fail_dir / f"ref_{index}_{c_idx}_rejected.txt").write_text(
                                sanitized_refined, encoding="utf-8"
                            )
                        logger.warning(
                            "Refine chunk %d/%d-%d/%d rejeitado por formatacao (ok_fmt=%s info=%s); usando fallback.",
                            index,
                            total,
                            c_idx,
                            len(chunks),
                            ok_fmt,
                            fmt_info,
                        )
                    else:
                        refined_text = sanitized_refined

                retry, retry_reason = needs_retry(chunk, refined_text)
                residual_source, residual_source_reason = detect_residual_source_language(
                    refined_text, resolved_source_language
                )
                if not used_fallback and residual_source:
                    retry = True
                    retry_reason = residual_source_reason
                if fmt_soft_retry:
                    retry = True
                    retry_reason = retry_reason or "soft_quote_balance"
                    logger.warning(
                        "Refine chunk %d/%d-%d/%d com aspas desbalanceadas; retry suave.",
                        index,
                        total,
                        c_idx,
                        len(chunks),
                    )
                attempt += 1
                if retry and attempt < cfg.max_retries:
                    if retry_reason:
                        retry_reasons.append(retry_reason)
                    logger.warning(
                        "QA retry refine chunk %d/%d-%d/%d: %s (tentativa %d/%d)",
                        index,
                        total,
                        c_idx,
                        len(chunks),
                        retry_reason,
                        attempt + 1,
                        cfg.max_retries,
                    )
                    if "omissao_dialogo" in retry_reason:
                        prompt = (
                            prompt
                            + "\n\nATENÇÃO: Você omitiu falas. Refaça traduzindo TODAS as frases e mantendo cada fala entre aspas exatamente uma vez. Não resuma. Não remova risos/interjeições."
                        )
                    elif "residual_" in retry_reason:
                        prompt = (
                            prompt
                            + f"\n\nATENÇÃO: Ainda há frases em {source_language_name(resolved_source_language)}. Refaça mantendo a mesma estrutura de parágrafos e traduzindo essas frases para português brasileiro natural. Preserve apenas nomes próprios, honoríficos e termos do glossário."
                        )
                    elif "truncado" in retry_reason:
                        prompt = (
                            prompt
                            + "\n\nATENÇÃO: Sua saída foi truncada. Refaça incluindo TODO o conteúdo."
                        )
                    elif "malformed_quote_boundary" in retry_reason:
                        prompt = (
                            prompt
                            + "\n\nATENÇÃO: Há uma fala iniciada por fechamento e abertura de aspas colados. Refaça preservando cada fala com aspas corretas, sem `”“` e sem alterar parágrafos."
                        )
                    else:
                        prompt = (
                            prompt
                            + "\n\nATENÇÃO: sua saída anterior veio truncada ou repetitiva. Refaça mantendo TODO o conteúdo. Não resuma."
                        )
                    continue
                # fim do loop de retry
                break

            if fmt_soft_retry and not used_fallback:
                used_fallback = True
                fallback_reasons.append("format_validation_unresolved")

            if used_fallback:
                refined_text = chunk
                metrics["fallbacks"] = metrics.get("fallbacks", 0) + 1
                if collapse_flag:
                    metrics["collapse"] = metrics.get("collapse", 0) + 1
            else:
                ratio = len(refined_text.strip()) / max(len(chunk.strip()), 1)
                if ratio < 0.8 or ratio > 1.8:
                    logger.info(
                        "Refine: divergência de tamanho aceita (mode=%s, ratio=%.2f) no chunk %d/%d da seção %s.",
                        guard_mode,
                        ratio,
                        c_idx,
                        len(chunks),
                        title or f"#{index}",
                    )
                if has_suspicious_repetition(refined_text):
                    logger.warning(
                        "Refinador devolveu texto com repetição suspeita; aceitando (mode=%s) no chunk %d/%d da seção %s.",
                        guard_mode,
                        c_idx,
                        len(chunks),
                        title or f"#{index}",
                    )
            refined_text, norm_stats = _apply_normalizers(refined_text)
            final_residual_source, final_residual_source_reason = detect_residual_source_language(
                refined_text, resolved_source_language
            )
            if final_residual_source:
                logger.warning(
                    "Refine chunk %d/%d-%d/%d ainda contém possível idioma de origem residual: %s",
                    index,
                    total,
                    c_idx,
                    len(chunks),
                    final_residual_source_reason,
                )
            if used_fallback or collapse_flag:
                reasons_payload = fallback_reasons
                if collapse_reasons:
                    reasons_payload = reasons_payload + [f"collapse:{r}" for r in collapse_reasons]
                if final_residual_source:
                    reasons_payload = reasons_payload + [final_residual_source_reason]
                _write_guardrail_debug_file(
                    cfg.output_dir,
                    section_index=index,
                    chunk_index=c_idx,
                    block_index=block_idx,
                    reasons=reasons_payload,
                    guardrails_mode=guard_mode,
                    collapse_flag=collapse_flag,
                    collapse_details=collapse_details,
                )
            # Debug opcional: salva até os 5 primeiros chunks
            if debug_refine and block_idx <= 5:
                debug_dir = cfg.output_dir / "debug_refine"
                save_refine_debug_files(
                    output_dir=debug_dir,
                    section_index=index,
                    chunk_index=c_idx,
                    original_text=chunk,
                    llm_raw=llm_raw,
                    final_text=refined_text,
                    logger=logger,
                )
            logger.debug("Seção refinada com %d caracteres.", len(refined_text))
            refined_parts.append(refined_text)
            if stats:
                stats.success_blocks += 1
            if progress:
                progress.refined_blocks.add(block_idx)
                progress.error_blocks.discard(block_idx)
                progress.chunk_outputs[block_idx] = refined_text
            seen_chunks.append((chunk, refined_text))
            record_block(
                refined_text,
                used_fallback=used_fallback,
                collapse=collapse_flag,
                normalizer_stats=norm_stats,
                guardrail_reasons=fallback_reasons if (used_fallback or collapse_flag) else None,
            )
            save_cache(
                "refine",
                h,
                raw_output=llm_raw,
                final_output=refined_text,
                metadata={
                    "chunk_index": c_idx,
                    "section_index": index,
                    "mode": "refine",
                    "backend": getattr(backend, "backend", None),
                    "model": getattr(backend, "model", None),
                    "num_predict": getattr(backend, "num_predict", None),
                    "temperature": getattr(backend, "temperature", None),
                    "repeat_penalty": getattr(backend, "repeat_penalty", None),
                    "guardrails": getattr(cfg, "refine_guardrails", None),
                    "source_language": resolved_source_language,
                    "prompt_hash": refine_prompt_fingerprint(resolved_source_language),
                    "pipeline_version": REFINE_PIPELINE_VERSION,
                },
            )
            if debug_writer:
                debug_writer(
                    {
                        "para_index": block_idx,
                        "original_text": chunk,
                        "original_chars": len(chunk),
                        "refined_text": refined_text,
                        "refined_chars": len(refined_text),
                        "llm_raw_output": llm_raw,
                        "sanitizer_report": None,
                        "normalizer_stats": norm_stats,
                    }
                )
            if record_chunk:
                suspect_output = bool(used_fallback or collapse_flag or final_residual_source)
                suspect_reason = ""
                if used_fallback and fallback_reasons:
                    suspect_reason = ";".join(fallback_reasons)
                elif collapse_flag:
                    suspect_reason = "collapse_detector"
                elif final_residual_source:
                    suspect_reason = final_residual_source_reason
                debug_stage_dir = debug_run.stage_dir("60_refine") / "debug_refine"
                _maybe_write_debug_files(chunk, llm_raw, refined_text)
                outputs_payload = {
                    "debug_original": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_original_pt.txt"
                    ),
                    "debug_context": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_context.txt"
                    ),
                    "debug_llm_raw": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_llm_raw.txt"
                    ),
                    "debug_final": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_final_pt.txt"
                    ),
                    "output_hash": debug_run.sha256_text(refined_text),
                }
                manifest_chunks.append(
                    {
                        "chunk_index": block_idx,
                        "section_index": index,
                        "section_title": title or "",
                        "input_hash": debug_run.sha256_text(chunk),
                        "chars_in": len(chunk),
                        "context_hash": None,
                        "from_cache": from_cache,
                        "from_duplicate": from_duplicate,
                        "llm_attempts": llm_attempts,
                        "retry_reasons": retry_reasons,
                        "suspect_output": suspect_output,
                        "suspect_reason": suspect_reason,
                        "source_language": resolved_source_language,
                        "residual_source_language": final_residual_source,
                        "residual_source_language_reason": final_residual_source_reason,
                        "residual_english": final_residual_source
                        if resolved_source_language == "en"
                        else False,
                        "residual_english_reason": final_residual_source_reason
                        if resolved_source_language == "en"
                        else "",
                        "contamination_detected": False,
                        "sanitization_ratio": None,
                        "normalizers": {
                            "triple_quotes_removed": norm_stats.get("triple_quotes_removed", 0),
                            "dialogue_splits": norm_stats.get("dialogue_splits", 0),
                        },
                        "lengths": {
                            "chars_out": len(refined_text),
                            "ratio_out_in": round(
                                len(refined_text.strip()) / max(len(chunk.strip()), 1),
                                3,
                            )
                            if chunk.strip()
                            else 0.0,
                        },
                        "outputs": outputs_payload,
                        "errors": None,
                    }
                )
        except RuntimeError as exc:
            logger.warning(
                "Chunk ref-%d/%d-%d/%d falhou; usando texto original. Erro: %s",
                index,
                total,
                c_idx,
                len(chunks),
                exc,
            )
            error_message = str(exc)
            fallback_text, norm_stats = _apply_normalizers(chunk)
            refined_parts.append(fallback_text)
            if stats:
                stats.error_blocks += 1
            if progress:
                progress.error_blocks.add(block_idx)
                progress.chunk_outputs[block_idx] = fallback_text
            metrics["fallbacks"] = metrics.get("fallbacks", 0) + 1
            _write_guardrail_debug_file(
                cfg.output_dir,
                section_index=index,
                chunk_index=c_idx,
                block_index=block_idx,
                reasons=["exception"],
                guardrails_mode=guard_mode,
                collapse_flag=False,
            )
            record_block(
                fallback_text,
                used_fallback=True,
                normalizer_stats=norm_stats,
                guardrail_reasons=["exception"],
            )
            if debug_writer:
                debug_writer(
                    {
                        "para_index": block_idx,
                        "original_text": chunk,
                        "original_chars": len(chunk),
                        "refined_text": fallback_text,
                        "refined_chars": len(fallback_text),
                        "llm_raw_output": llm_raw,
                        "sanitizer_report": None,
                        "normalizer_stats": norm_stats,
                    }
                )
            _maybe_write_debug_files(chunk, llm_raw, fallback_text)
            if record_chunk:
                debug_stage_dir = debug_run.stage_dir("60_refine") / "debug_refine"
                outputs_payload = {
                    "debug_original": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_original_pt.txt"
                    ),
                    "debug_context": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_context.txt"
                    ),
                    "debug_llm_raw": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_llm_raw.txt"
                    ),
                    "debug_final": debug_run.rel_path(
                        debug_stage_dir / f"chunk{block_idx:03d}_final_pt.txt"
                    ),
                    "output_hash": debug_run.sha256_text(fallback_text),
                }
                manifest_chunks.append(
                    {
                        "chunk_index": block_idx,
                        "section_index": index,
                        "section_title": title or "",
                        "input_hash": debug_run.sha256_text(chunk),
                        "chars_in": len(chunk),
                        "context_hash": None,
                        "from_cache": from_cache,
                        "from_duplicate": from_duplicate,
                        "llm_attempts": llm_attempts,
                        "retry_reasons": retry_reasons,
                        "suspect_output": True,
                        "suspect_reason": "exception",
                        "contamination_detected": False,
                        "sanitization_ratio": None,
                        "normalizers": {
                            "triple_quotes_removed": norm_stats.get("triple_quotes_removed", 0),
                            "dialogue_splits": norm_stats.get("dialogue_splits", 0),
                        },
                        "lengths": {
                            "chars_out": len(fallback_text),
                            "ratio_out_in": round(
                                len(fallback_text.strip()) / max(len(chunk.strip()), 1),
                                3,
                            )
                            if chunk.strip()
                            else 0.0,
                        },
                        "outputs": outputs_payload,
                        "errors": {"message": error_message},
                    }
                )
        finally:
            _write_progress(progress, logger)

    refined_section = "\n\n".join(refined_parts).strip()
    if title:
        return f"{title}\n\n{refined_section}"
    return refined_section


def refine_markdown_file(
    input_path: Path,
    output_path: Path,
    backend: LLMBackend,
    cfg: AppConfig,
    logger: logging.Logger,
    progress_path: Path | None = None,
    resume_manifest: dict | None = None,
    normalize_paragraphs: bool = False,
    glossary_state: GlossaryState | None = None,
    glossary_prompt_limit: int = DEFAULT_GLOSSARY_PROMPT_LIMIT,
    debug_refine: bool = False,
    parallel_workers: int = 1,
    preprocess_advanced: bool = False,
    debug_chunks: bool = False,
    cleanup_mode: str = "off",
    debug_run: DebugRunWriter | None = None,
    source_language: str | None = None,
) -> None:
    """
    Ponto de entrada principal para o refinamento (polimento e revisão) de um arquivo Markdown traduzido.
    Orquestra a divisão do arquivo, processamento paralelo (ou sequencial) e junção final.
    """
    set_cache_base_dir(cfg.output_dir)
    resolved_source_language = normalize_source_language(source_language or cfg.source_language)
    if resolved_source_language == "auto":
        resolved_source_language = "en"
        logger.info("Idioma de origem não informado no refine; usando inglês para o QA residual.")
    raw_md = read_text(input_path)
    md_text = raw_md
    if preprocess_advanced:
        md_text = advanced_clean(md_text)
    if normalize_paragraphs:
        md_text = normalize_md_paragraphs(md_text)
    cleanup_preview_hash_before = chunk_hash(md_text)

    cleanup_applied = False
    cleanup_stats: dict = {}
    cleanup_mode = cleanup_mode if cleanup_mode in ("off", "auto", "on") else "off"
    trigger_cleanup = False
    if cleanup_mode == "on":
        trigger_cleanup = True
    elif cleanup_mode == "auto":
        if detect_obvious_dupes(raw_md) or detect_glued_dialogues(raw_md):
            trigger_cleanup = True
        elif getattr(cfg, "refine_guardrails", "strict") == "strict" and (
            detect_obvious_dupes(raw_md) or detect_glued_dialogues(raw_md)
        ):
            trigger_cleanup = True
    if trigger_cleanup:
        md_text, cleanup_stats = cleanup_before_refine(md_text)
        cleanup_applied = True
        pre_refine_path = output_path.with_name(f"{output_path.stem}_pre_refine_cleanup.md")
        pre_refine_path.write_text(md_text, encoding="utf-8")
        if debug_run:
            debug_run.pre_refine_rel = (
                f"50_cleanup_pre_refine/{debug_run.slug}_pre_refine_cleanup.md"
            )
            debug_run.write_text(debug_run.pre_refine_rel, md_text)
    cleanup_preview_hash_after = chunk_hash(md_text)
    if debug_run:
        cleanup_report = {
            "cleanup_mode": cleanup_mode,
            "cleanup_applied": cleanup_applied,
            "stats": cleanup_stats,
            "hash_before": cleanup_preview_hash_before,
            "hash_after": cleanup_preview_hash_after,
        }
        debug_run.write_cleanup_report(cleanup_report)

    doc_hash = chunk_hash(md_text)
    sections = split_markdown_sections(md_text)
    logger.info("Arquivo %s: %d seções detectadas", input_path.name, len(sections))
    logger.info("Refine guardrails mode: %s", getattr(cfg, "refine_guardrails", "strict"))
    stats = RefineStats()
    metrics: dict[str, int | list | dict | bool | str] = {
        "cache_hits": 0,
        "fallbacks": 0,
        "collapse": 0,
        "duplicates": 0,
        "dialogue_splits": 0,
        "triple_quotes_removed": 0,
        "block_metrics": [],
    }
    metrics["source_language"] = resolved_source_language
    metrics["cleanup_mode"] = cleanup_mode
    metrics["cleanup_applied"] = cleanup_applied
    metrics["cleanup_stats"] = cleanup_stats
    metrics["cleanup_preview_hash_before"] = cleanup_preview_hash_before
    metrics["cleanup_preview_hash_after"] = cleanup_preview_hash_after
    seen_chunks: list[tuple[str, str]] = []
    refine_manifest_chunks: list[dict] = []

    # Pré-computa total de blocos para progress
    total_blocks = 0
    max_refine_chunk_len = 0
    for _, body in sections:
        paragraphs = paragraphs_from_text(body)
        chunks = chunk_for_refine(paragraphs, max_chars=cfg.refine_chunk_chars, logger=logger)
        total_blocks += len(chunks)
        if chunks:
            max_refine_chunk_len = max(max_refine_chunk_len, max(len(c) for c in chunks))
    metrics["effective_refine_chunk_chars"] = cfg.refine_chunk_chars
    metrics["max_chunk_chars_observed"] = max_refine_chunk_len

    if progress_path is None:
        progress_path = output_path.with_name(f"{output_path.stem}_progress.json")

    state_path = output_path.parent / "state_refine.json"
    debug_file = None
    debug_file_path: Path | None = None

    try:
        state_payload = {
            "input_file": str(input_path),
            "hash": doc_hash,
            "timestamp": datetime.now().isoformat(),
            "total_chunks": total_blocks,
            "refine_guardrails": getattr(cfg, "refine_guardrails", "strict"),
        }
        state_path.write_text(
            json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

    progress = _prepare_progress(
        progress_path=progress_path,
        resume_manifest=resume_manifest,
        total_blocks=total_blocks,
        logger=logger,
    )
    _write_progress(progress, logger)

    if debug_chunks:
        debug_file_path = output_path.with_name(f"{output_path.stem}_chunks_debug.jsonl")
        debug_file = debug_file_path.open("w", encoding="utf-8")

    def _write_chunk_debug(entry: dict) -> None:
        """Grava chunk depuração."""
        if debug_file:
            debug_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    refined_sections: List[str] = []
    with processing_context(stats, progress):
        for idx, (title, body) in enumerate(sections, start=1):
            refined_sections.append(
                refine_section(
                    title=title,
                    body=body,
                    backend=backend,
                    cfg=cfg,
                    logger=logger,
                    index=idx,
                    total=len(sections),
                    glossary_state=glossary_state,
                    glossary_prompt_limit=glossary_prompt_limit,
                    debug_refine=debug_refine,
                    metrics=metrics,
                    seen_chunks=seen_chunks,
                    debug_writer=_write_chunk_debug if debug_chunks else None,
                    debug_run=debug_run,
                    manifest_chunks=refine_manifest_chunks if debug_run else None,
                    source_language=resolved_source_language,
                )
            )

    final_md = "\n\n".join(refined_sections).strip()
    if not final_md:
        raise ValueError(f"Refine produziu texto vazio para {input_path}")

    final_md = sanitize_refine_output(final_md)
    final_md, dialogue_stats = fix_dialogue_artifacts(final_md)
    final_md, _ = apply_structural_normalizers(final_md)
    final_md = apply_custom_normalizers(final_md, convert_quote_dialogues=False)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Pós-processo de diálogo (refine-final): %s", dialogue_stats)
    opens_q, closes_q = count_curly_quotes(final_md)
    if opens_q != closes_q:
        final_md, _ = fix_unbalanced_quotes(final_md, logger=logger, label="refine-final")

    write_text(output_path, final_md)
    if glossary_state:
        save_dynamic_glossary(glossary_state, logger)
    try:
        version = (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        version = "unknown"
    report = {
        "mode": "refine",
        "input": str(input_path),
        "total_chunks": stats.total_blocks,
        "cache_hits": metrics.get("cache_hits", 0),
        "fallbacks": metrics.get("fallbacks", 0),
        "collapse_detected": metrics.get("collapse", 0),
        "duplicates_reused": metrics.get("duplicates", 0),
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": version,
        "refine_guardrails": getattr(cfg, "refine_guardrails", "strict"),
        "effective_refine_chunk_chars": cfg.refine_chunk_chars,
        "max_chunk_chars_observed": metrics.get("max_chunk_chars_observed", 0),
        "dialogue_splits": metrics.get("dialogue_splits", 0),
        "triple_quotes_removed": metrics.get("triple_quotes_removed", 0),
    }
    try:
        slug_report = Path(input_path).stem
        report_path = output_path.parent / f"{slug_report}_refine_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        refine_metrics = {
            "total_blocks": stats.total_blocks,
            "cache_hits": metrics.get("cache_hits", 0),
            "duplicates": metrics.get("duplicates", 0),
            "fallbacks": metrics.get("fallbacks", 0),
            "collapse": metrics.get("collapse", 0),
            "blocks": metrics.get("block_metrics", []),
            "guardrails_mode": getattr(cfg, "refine_guardrails", "strict"),
            "cleanup_mode": metrics.get("cleanup_mode", "off"),
            "cleanup_applied": metrics.get("cleanup_applied", False),
            "cleanup_stats": metrics.get("cleanup_stats", {}),
            "cleanup_preview_hash_before": metrics.get("cleanup_preview_hash_before"),
            "cleanup_preview_hash_after": metrics.get("cleanup_preview_hash_after"),
            "effective_refine_chunk_chars": cfg.refine_chunk_chars,
            "max_chunk_chars_observed": metrics.get("max_chunk_chars_observed", 0),
            "dialogue_splits": metrics.get("dialogue_splits", 0),
            "triple_quotes_removed": metrics.get("triple_quotes_removed", 0),
        }
        slug = Path(input_path).stem
        metrics_path = output_path.parent / f"{slug}_refine_metrics.json"
        metrics_path.write_text(
            json.dumps(refine_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
    if debug_file:
        debug_file.close()
        if debug_file_path:
            logger.info("Arquivo de debug de refine: %s", debug_file_path)
    if debug_run:
        pre_refine_rel = debug_run.pre_refine_rel if cleanup_applied else None
        refine_manifest = {
            "run_id": debug_run.run_id,
            "stage": "refine",
            "source_slug": debug_run.slug,
            "input_paths": {
                "pt_before_refine": debug_run.pt_output_rel,
                "pre_refine_cleanup": pre_refine_rel,
            },
            "refine": {
                "cleanup_before_refine": cleanup_mode,
                "cleanup_applied": cleanup_applied,
                "split_markdown_sections": True,
                "refine_chunk_chars": cfg.refine_chunk_chars,
                "total_sections": len(sections),
                "total_chunks": stats.total_blocks,
            },
            "cache_signature": {
                "backend": getattr(backend, "backend", None),
                "model": getattr(backend, "model", None),
                "num_predict": getattr(backend, "num_predict", None),
                "temperature": getattr(backend, "temperature", None),
                "repeat_penalty": getattr(backend, "repeat_penalty", None),
                "refine_chunk_chars": cfg.refine_chunk_chars,
                "glossary_hash": _glossary_hash(glossary_state),
            },
            "chunks": refine_manifest_chunks,
            "totals": {
                "cache_hits": metrics.get("cache_hits", 0),
                "duplicate_reuse": metrics.get("duplicates", 0),
                "contamination_count": 0,
                "error_count": stats.error_blocks,
            },
        }
        debug_run.write_manifest("refine", refine_manifest)
    logger.info(
        "Refine concluído: %s (blocos: total=%d sucesso=%d placeholders=%d)",
        output_path.name,
        stats.total_blocks,
        stats.success_blocks,
        stats.error_blocks,
    )


def _call_with_retry(
    backend: LLMBackend,
    prompt: str,
    cfg: AppConfig,
    logger: logging.Logger,
    label: str,
    max_retries: int | None = None,
) -> tuple[str, str]:
    """Executa com nova tentativa."""
    delay = cfg.initial_backoff
    last_error: Exception | None = None
    attempts = max_retries if max_retries is not None else cfg.max_retries
    for attempt in range(1, attempts + 1):
        try:
            latency, response = timed(backend.generate, prompt)
            raw_text = response.text
            text = sanitize_refine_output(raw_text)
            if not text.strip():
                raise ValueError("Texto vazio após sanitização do refine.")
            logger.info("%s ok (%.2fs, %d chars)", label, latency, len(text))
            return raw_text, text
        except Exception as exc:
            last_error = exc
            logger.warning("%s falhou (tentativa %d/%d): %s", label, attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(delay)
                delay *= cfg.backoff_factor
    raise RuntimeError(f"{label} falhou após {attempts} tentativas: {last_error}")
