"""Reconstrução de parágrafos com modo determinístico ou apoiado por LLM."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .cache_utils import (
    cache_exists,
    chunk_hash,
    load_cache,
    save_cache,
    set_cache_base_dir,
)
from .config import AppConfig
from .desquebrar_safe import safe_reflow
from .llm_backend import LLMBackend
from .preprocess import paragraphs_from_text
from .utils import chunk_by_paragraphs, timed

DESQUEBRAR_PROMPT = """
UNA APENAS AS QUEBRAS DE LINHA ERRADAS DO TEXTO ENTRE AS MARCAS ABAIXO.
NAO REESCREVA, NAO TRADUZA, NAO RESUMA, NAO ADICIONE NADA.
NAO TROQUE PALAVRAS, NAO MUDE PONTUACAO, NAO MUDE NENHUM TERMO.
RETORNE SOMENTE O TEXTO CORRIGIDO, SEM CABECALHOS OU COMENTARIOS.

TEXTO:
\"\"\"{chunk}\"\"\""""

ELLIPSIS_RE = re.compile(r"\.\.\.|…")
SAFE_DIALOGUE_BREAK_PATTERNS = (
    (re.compile(r"”{2,}\s*“"), "”\n\n“"),
    (re.compile(r"”\s*“"), "”\n\n“"),
)
QUOTE_LINE_TOKENS = {'"', "“", "”", "'''", '"""'}
QUOTE_CHARS = {'"', "“", "”"}
HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
STUTTER_SPACE_RE = re.compile(r"\b([A-Za-zÀ-ÿ])-\s+([A-Za-zÀ-ÿ])")
POSTPROCESS_VERSION = 1


@dataclass
class DesquebrarStats:
    """Reúne as métricas produzidas durante a reconstrução de parágrafos."""

    total_chunks: int = 0
    cache_hits: int = 0
    fallbacks: int = 0
    dialogue_splits: int = 0
    hyphen_linewrap_count: int = 0
    stutter_space_count: int = 0
    stray_quote_lines: int = 0
    hardwrap_joins_count: int = 0
    internal_hyphen_fixed_count: int = 0
    scene_separators_fixed_count: int = 0
    blocks: list[dict] | None = None


def _count_alnum(text: str) -> int:
    """Conta os caracteres alfanuméricos do texto."""
    return sum(1 for ch in text if ch.isalnum())


def _count_ellipses(text: str) -> int:
    """Conta ocorrências de reticências em suas formas aceitas."""
    return len(ELLIPSIS_RE.findall(text))


def _has_lonely_quote_line(text: str) -> bool:
    """Verifica se existem linhas formadas apenas por aspas."""
    return any(line.strip() in QUOTE_LINE_TOKENS for line in text.splitlines())


def _strip_triple_quote_wrapper(text: str) -> str:
    """Remove um invólucro de aspas triplas criado pelo modelo."""
    stripped = text.strip()
    if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) >= 6:
        return stripped[3:-3].strip()
    return stripped


def _remove_stray_quote_lines(text: str) -> tuple[str, int]:
    """Remove linhas isoladas formadas apenas por aspas."""
    lines = text.splitlines()
    kept: list[str] = []
    removed = 0
    for line in lines:
        if line.strip() in QUOTE_LINE_TOKENS:
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


def _isolate_asterisks(text: str) -> str:
    """Isola separadores de cena formados por asteriscos."""
    lines = text.splitlines()
    output: list[str] = []

    def append_blank() -> None:
        """Acrescenta uma única linha em branco quando necessário."""
        if output and output[-1] != "":
            output.append("")

    for line in lines:
        if line.strip() == "***":
            append_blank()
            output.append("***")
            output.append("")
            continue
        output.append(line)

    compact: list[str] = []
    for line in output:
        if line == "":
            if not compact or compact[-1] != "":
                compact.append("")
        else:
            compact.append(line)
    return "\n".join(compact)


def _count_quotes(text: str) -> int:
    """Conta aspas."""
    return sum(1 for ch in text if ch in QUOTE_CHARS)


def _has_quote_inflation(orig: str, output: str) -> bool:
    """Detecta aumento anormal na quantidade de aspas."""
    orig_count = _count_quotes(orig)
    out_count = _count_quotes(output)
    if out_count <= orig_count:
        return False
    return out_count >= orig_count * 1.5 and (out_count - orig_count) >= 4


def postprocess_llm_output(text: str) -> tuple[str, dict]:
    """
    Remove artefatos indesejados da saída do LLM no processo de desquebra (ex.: linhas extras de aspas,
    hifenizações residuais e marcadores de markdown) retornando o texto limpo e métricas.
    """
    cleaned = _strip_triple_quote_wrapper(text)
    cleaned, stray_quote_lines = _remove_stray_quote_lines(cleaned)
    cleaned, hyphen_linewrap_count = HYPHEN_LINEBREAK_RE.subn(r"\1-\2", cleaned)
    cleaned, stutter_space_count = STUTTER_SPACE_RE.subn(r"\1-\2", cleaned)
    cleaned = _isolate_asterisks(cleaned)
    return (
        cleaned,
        {
            "hyphen_linewrap_count": hyphen_linewrap_count,
            "stutter_space_count": stutter_space_count,
            "stray_quote_lines": stray_quote_lines,
        },
    )


def validate_desquebrar_output(orig: str, output: str) -> tuple[bool, list[str]]:
    """Valida a saída reconstruída e rejeita perdas ou corrupções de conteúdo."""
    reasons: list[str] = []
    if _has_lonely_quote_line(output):
        reasons.append("lonely_quote_line")
    if _count_ellipses(output) != _count_ellipses(orig):
        reasons.append("ellipsis_mismatch")
    orig_alnum = _count_alnum(orig)
    out_alnum = _count_alnum(output)
    if orig_alnum and out_alnum < orig_alnum * 0.99:
        reasons.append("alnum_loss")
    return not reasons, reasons


def deterministic_unbreak(text: str) -> str:
    """Reconstrói apenas linhas pertencentes ao mesmo parágrafo."""
    if not text:
        return text
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = normalized.split("\n\n")
    rebuilt: list[str] = []
    for para in paragraphs:
        if para.strip() == "":
            rebuilt.append("")
            continue
        lines = para.split("\n")
        acc = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if acc:
                    rebuilt.append(acc.strip())
                    acc = ""
                rebuilt.append(stripped)
                continue
            if acc:
                if acc.endswith("-") and stripped[:1].islower():
                    acc = acc[:-1] + stripped
                else:
                    acc = f"{acc} {stripped}"
            else:
                acc = stripped
        if acc or (not rebuilt or rebuilt[-1] != ""):
            rebuilt.append(acc.strip())
    return "\n\n".join(rebuilt).strip()


def normalize_dialogue_breaks_source_safe(text: str) -> tuple[str, dict]:
    """Normaliza quebras de diálogo sem criar linhas formadas apenas por aspas."""
    if not text:
        return text, {"dialogue_splits": 0}
    cleaned = text
    total = 0
    for pattern, replacement in SAFE_DIALOGUE_BREAK_PATTERNS:
        cleaned, count = pattern.subn(replacement, cleaned)
        total += count
    return cleaned, {"dialogue_splits": total}


def build_desquebrar_prompt(chunk: str) -> str:
    """
    Constrói o prompt instruindo o LLM a unir parágrafos quebrados do PDF
    sem traduzir e sem alterar o conteúdo.
    """
    return DESQUEBRAR_PROMPT.format(chunk=chunk)


def normalize_scene_separators(text: str) -> tuple[str, int]:
    """
    Garante que "***" fique isolado por linhas em branco.
    """
    lines = text.split("\n")
    output: list[str] = []
    fixes = 0

    def append_blank():
        """Acrescenta uma única linha em branco quando necessário."""
        nonlocal fixes
        if output and output[-1] != "":
            output.append("")
            fixes += 1

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "***":
            append_blank()
            output.append("***")
            if i + 1 < len(lines) and lines[i + 1].strip() != "":
                output.append("")
                fixes += 1
            i += 1
            continue
        output.append(line)
        i += 1

    # compact blanks (no mais que 2 seguidos)
    compact: list[str] = []
    prev_blank = False
    for ln in output:
        if ln == "":
            if not prev_blank:
                compact.append("")
            prev_blank = True
        else:
            compact.append(ln)
            prev_blank = False
    return "\n".join(compact).strip(), fixes


def normalize_hardwrap_joins(text: str) -> tuple[str, int]:
    """
    Junta quebras de linha internas em frases quando seguro.
    """
    if not text:
        return text, 0

    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")
    joined_paras: list[str] = []
    joins = 0

    def is_block_start(line: str) -> bool:
        """Indica se bloco início."""
        stripped = line.lstrip()
        return stripped.startswith(('"', "“", "”", "-", "—", "#", "***"))

    END_TOKENS = (".", "?", "!", "…", ":", '"', "”", ")", "]")

    for p_idx, para in enumerate(paragraphs):
        next_para = paragraphs[p_idx + 1].strip() if p_idx + 1 < len(paragraphs) else ""
        scene_after = next_para == "***"
        lines = para.split("\n")
        if len(lines) <= 1:
            joined_paras.append(para.strip())
            continue
        i = 0
        new_lines: list[str] = []
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            if i + 1 >= len(lines):
                new_lines.append(stripped)
                break
            nxt = lines[i + 1]
            nxt_stripped = nxt.strip()
            if not nxt_stripped:
                new_lines.append(stripped)
                i += 2
                continue

            j = i + 2
            while j < len(lines) and not lines[j].strip():
                j += 1
            has_scene_ahead = j < len(lines) and lines[j].strip() == "***"

            if (
                not stripped.endswith(END_TOKENS)
                and not is_block_start(stripped)
                and not is_block_start(nxt_stripped)
                and nxt_stripped[:1].islower()
                and not has_scene_ahead
                and not (scene_after and i == len(lines) - 2)
            ):
                new_lines.append(f"{stripped} {nxt_stripped}")
                joins += 1
                i += 2
                continue

            new_lines.append(stripped)
            i += 1
        joined_paras.append("\n".join(new_lines))

    return "\n\n".join(p for p in joined_paras if p.strip()), joins


def normalize_internal_hyphen_by_dominance(text: str) -> tuple[str, dict]:
    """
    Remove hifenização interna somente quando a forma sem hífen domina no texto.
    """
    pattern = re.compile(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b")
    matches = pattern.findall(text)
    if not matches:
        return text, {}

    blocked_prefixes = {"demi", "half", "anti", "non", "pre", "post", "re"}
    blocked_suffixes = {"san", "sama", "kun", "chan"}

    counts: dict[str, dict[str, int]] = {}
    for left, right in matches:
        hyph = f"{left}-{right}"
        plain = f"{left}{right}"
        data = counts.setdefault(hyph, {"hyph": 0, "plain": text.count(plain)})
        data["hyph"] += 1

    replacements: dict[str, str] = {}
    for left, right in matches:
        hyph = f"{left}-{right}"
        if hyph in replacements:
            continue
        plain = f"{left}{right}"
        data = counts.get(hyph, {"hyph": 0, "plain": 0})
        if data["plain"] < 3 or data["hyph"] > 2:
            continue
        if len(left) <= 1 or len(right) <= 1:
            continue
        if right.lower() in blocked_suffixes:
            continue
        if left.lower() in blocked_prefixes:
            continue
        replacements[hyph] = plain

    if not replacements:
        return text, {}

    def _sub(match: re.Match[str]) -> str:
        """Aplica a substituição atual e atualiza as métricas relacionadas."""
        token = match.group(0)
        return replacements.get(token, token)

    new_text, subs = pattern.subn(_sub, text)
    fixed_counts = {k: counts[k]["hyph"] for k in replacements}
    return new_text, {"total": subs, "details": fixed_counts}


def desquebrar_text(
    text: str,
    cfg: AppConfig,
    logger: logging.Logger,
    backend: LLMBackend,
    chunk_chars: int | None = None,
) -> tuple[str, DesquebrarStats]:
    """
    Normaliza quebras de linha com LLM respeitando chunking seguro.

    Retorna (texto_desquebrado, stats).
    """
    set_cache_base_dir(cfg.output_dir)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return "", DesquebrarStats(total_chunks=0, cache_hits=0, fallbacks=0, blocks=[])
    paragraphs = paragraphs_from_text(normalized)
    if len(paragraphs) <= 1:
        paragraphs = [ln.strip() for ln in normalized.splitlines() if ln.strip()]

    max_chars = chunk_chars or cfg.desquebrar_chunk_chars
    chunks = chunk_by_paragraphs(paragraphs, max_chars=max_chars, logger=logger, label="desquebrar")
    total_chunks = len(chunks)
    stats = DesquebrarStats(total_chunks=total_chunks, blocks=[])

    outputs: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        h = chunk_hash(chunk)
        from_cache = False
        if cache_exists("desquebrar", h):
            data = load_cache("desquebrar", h)
            meta_ok = False
            meta = data.get("metadata")
            expected = {
                "backend": getattr(backend, "backend", None),
                "model": getattr(backend, "model", None),
                "num_predict": getattr(backend, "num_predict", None),
                "temperature": getattr(backend, "temperature", None),
                "chunk_chars": max_chars,
                "repeat_penalty": getattr(backend, "repeat_penalty", None),
                "postprocess_version": POSTPROCESS_VERSION,
            }
            if isinstance(meta, dict):
                meta_ok = all(meta.get(k) == v for k, v in expected.items())
            if not meta_ok:
                logger.debug("Cache de desquebrar ignorado: assinatura diferente.")
            cached = data.get("final_output") if meta_ok else None
            if cached:
                logger.info("desq-%d/%d cache_hit", idx, total_chunks)
                outputs.append(cached)
                stats.cache_hits += 1
                from_cache = True

        if from_cache:
            stats.blocks.append(
                {
                    "chunk_index": idx,
                    "chars_in": len(chunk),
                    "chars_out": len(outputs[-1]) if outputs else 0,
                    "from_cache": True,
                    "fallback": False,
                }
            )
            continue

        prompt = build_desquebrar_prompt(chunk)
        try:
            latency, response = timed(backend.generate, prompt)
            cleaned, post_stats = postprocess_llm_output(response.text)
            stats.hyphen_linewrap_count += post_stats["hyphen_linewrap_count"]
            stats.stutter_space_count += post_stats["stutter_space_count"]
            stats.stray_quote_lines += post_stats["stray_quote_lines"]
            cleaned = cleaned.strip()
            if not cleaned:
                raise ValueError("Resposta vazia do desquebrar.")
            if post_stats["stray_quote_lines"] > 0:
                fallback_text = safe_reflow(chunk)
                outputs.append(fallback_text)
                stats.fallbacks += 1
                logger.warning(
                    "desq-%d/%d qa fallback (stray_quote_lines=%d)",
                    idx,
                    total_chunks,
                    post_stats["stray_quote_lines"],
                )
                stats.blocks.append(
                    {
                        "chunk_index": idx,
                        "chars_in": len(chunk),
                        "chars_out": len(fallback_text),
                        "latency": latency,
                        "from_cache": False,
                        "fallback": True,
                        "fallback_reason": "qa_stray_quote_lines",
                        "stray_quote_lines": post_stats["stray_quote_lines"],
                    }
                )
                continue
            if _has_quote_inflation(chunk, cleaned):
                fallback_text = safe_reflow(chunk)
                outputs.append(fallback_text)
                stats.fallbacks += 1
                logger.warning("desq-%d/%d qa fallback (quote_inflation)", idx, total_chunks)
                stats.blocks.append(
                    {
                        "chunk_index": idx,
                        "chars_in": len(chunk),
                        "chars_out": len(fallback_text),
                        "latency": latency,
                        "from_cache": False,
                        "fallback": True,
                        "fallback_reason": "qa_quote_inflation",
                    }
                )
                continue
            is_valid, reasons = validate_desquebrar_output(chunk, cleaned)
            if not is_valid:
                fallback_text = deterministic_unbreak(chunk)
                outputs.append(fallback_text)
                stats.fallbacks += 1
                reason_str = ",".join(reasons) or "invalid_output"
                logger.warning(
                    "desq-%d/%d invalid output; usando fallback deterministico (reasons=%s)",
                    idx,
                    total_chunks,
                    reason_str,
                )
                stats.blocks.append(
                    {
                        "chunk_index": idx,
                        "chars_in": len(chunk),
                        "chars_out": len(fallback_text),
                        "latency": latency,
                        "from_cache": False,
                        "fallback": True,
                        "fallback_reason": reason_str,
                    }
                )
                continue
            outputs.append(cleaned)
            logger.info(
                "desq-%d/%d ok (%.2fs, %d chars)",
                idx,
                total_chunks,
                latency,
                len(cleaned),
            )
            stats.blocks.append(
                {
                    "chunk_index": idx,
                    "chars_in": len(chunk),
                    "chars_out": len(cleaned),
                    "latency": latency,
                    "from_cache": False,
                    "fallback": False,
                }
            )
            save_cache(
                "desquebrar",
                h,
                raw_output=response.text,
                final_output=cleaned,
                metadata={
                    "chunk_index": idx,
                    "mode": "desquebrar",
                    "model": getattr(backend, "model", None),
                    "backend": getattr(backend, "backend", None),
                    "num_predict": getattr(backend, "num_predict", None),
                    "temperature": getattr(backend, "temperature", None),
                    "chunk_chars": max_chars,
                    "repeat_penalty": getattr(backend, "repeat_penalty", None),
                    "postprocess_version": POSTPROCESS_VERSION,
                },
            )
        except Exception as exc:  # pragma: no cover - network/LLM failure path
            logger.warning(
                "Bloco %d do desquebrar falhou; usando fallback deterministico. Erro: %s",
                idx,
                exc,
            )
            fallback_text = deterministic_unbreak(chunk)
            outputs.append(fallback_text)
            stats.fallbacks += 1
            logger.info("desq-%d/%d fallback", idx, total_chunks)
            stats.blocks.append(
                {
                    "chunk_index": idx,
                    "chars_in": len(chunk),
                    "chars_out": len(fallback_text),
                    "from_cache": False,
                    "fallback": True,
                    "fallback_reason": "exception",
                    "error": str(exc),
                }
            )

    combined = "\n\n".join(outputs).strip()
    combined, scene_fixes = normalize_scene_separators(combined)
    combined, hardwrap_joins = normalize_hardwrap_joins(combined)
    combined, hyphen_stats = normalize_internal_hyphen_by_dominance(combined)
    combined, norm_stats = normalize_dialogue_breaks_source_safe(combined)
    combined = normalize_wrapped_lines(combined)
    stats.dialogue_splits = norm_stats.get("dialogue_splits", 0)
    stats.scene_separators_fixed_count = scene_fixes
    stats.hardwrap_joins_count = hardwrap_joins
    stats.internal_hyphen_fixed_count = hyphen_stats.get("total", 0)
    logger.info(
        (
            "desquebrar metrics: hyphen_linewrap=%d stutter_space=%d stray_quote_lines=%d "
            "hardwrap_joins=%d internal_hyphen_fixes=%d scene_fixes=%d"
        ),
        stats.hyphen_linewrap_count,
        stats.stutter_space_count,
        stats.stray_quote_lines,
        stats.hardwrap_joins_count,
        stats.internal_hyphen_fixed_count,
        stats.scene_separators_fixed_count,
    )
    return combined, stats


def normalize_wrapped_lines(text: str) -> str:
    """Reconstrói quebras indevidas na narração sem alterar falas ou títulos."""
    if not text:
        return text

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    new_lines: list[str] = []

    def is_block_start(value: str) -> bool:
        """Indica se bloco início."""
        stripped = value.lstrip()
        return stripped.startswith(("'", '"', "\u201c", "\u201d", "-", "\u2014", "#"))

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            i += 1
            continue

        if is_block_start(stripped):
            new_lines.append(line)
            i += 1
            continue

        ends_ok = stripped.endswith((".", "?", "!", "\u2026", ":", "\u201d", '"'))
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            nxt_stripped = nxt.strip()
            if not nxt_stripped:
                new_lines.append(line)
                i += 1
                continue

            j = i + 2
            while j < len(lines) and not lines[j].strip():
                j += 1
            has_scene_ahead = j < len(lines) and lines[j].strip() == "***"

            nxt_block = is_block_start(nxt_stripped)
            starts_lower = nxt_stripped[:1].islower()
            if (not ends_ok) and starts_lower and not nxt_block and not has_scene_ahead:
                new_lines.append(f"{stripped} {nxt_stripped}")
                i += 2
                continue

        new_lines.append(line)
        i += 1

    return "\n".join(new_lines)


def desquebrar_stats_to_dict(stats: DesquebrarStats | None, cfg: AppConfig) -> dict:
    """Serializa as métricas de desquebra em um dicionário."""
    if stats is None:
        return {}
    return {
        "total_chunks": stats.total_chunks,
        "cache_hits": stats.cache_hits,
        "fallbacks": stats.fallbacks,
        "dialogue_splits": stats.dialogue_splits,
        "hyphen_linewrap_count": stats.hyphen_linewrap_count,
        "stutter_space_count": stats.stutter_space_count,
        "stray_quote_lines": stats.stray_quote_lines,
        "hardwrap_joins_count": stats.hardwrap_joins_count,
        "internal_hyphen_fixed_count": stats.internal_hyphen_fixed_count,
        "scene_separators_fixed_count": stats.scene_separators_fixed_count,
        "blocks": stats.blocks or [],
        "effective_desquebrar_chunk_chars": cfg.desquebrar_chunk_chars,
        "backend": getattr(cfg, "desquebrar_backend", None),
        "model": getattr(cfg, "desquebrar_model", None),
    }


def normalize_md_paragraphs(md_text: str) -> str:
    """
    Normaliza parágrafos juntando linhas internas, preservando blocos especiais.
    """
    if not md_text:
        return md_text

    text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    normalized: list[str] = []
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush_buffer() -> None:
        """Consolida o conteúdo acumulado no bloco de saída."""
        nonlocal buffer
        if buffer:
            normalized.append(" ".join(buffer).strip())
            buffer = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if in_fence:
            normalized.append(raw_line)
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_buffer()
            in_fence = True
            fence_marker = stripped[:3]
            normalized.append(raw_line)
            continue

        if stripped == "":
            flush_buffer()
            normalized.append("")
            continue

        if (
            re.match(r"^#{1,6}\s", stripped)
            or re.match(r"^>\s", stripped)
            or re.match(r"^[-*+]\s", stripped)
            or re.match(r"^\d+\.\s", stripped)
        ):
            flush_buffer()
            normalized.append(stripped)
            continue

        if buffer:
            if buffer[-1].endswith("-"):
                buffer[-1] = buffer[-1][:-1]
                buffer.append(stripped.lstrip())
            else:
                buffer.append(stripped)
        else:
            buffer.append(stripped)

    flush_buffer()

    compact: list[str] = []
    prev_blank = False
    for ln in normalized:
        if ln == "":
            if not prev_blank:
                compact.append("")
            prev_blank = True
        else:
            compact.append(ln)
            prev_blank = False

    return "\n".join(compact).strip()
