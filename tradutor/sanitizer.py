"""
Sanitização agressiva contra alucinações e ruídos dos modelos.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

# Meta mínimos para tradução (evitar cortar falas legítimas).
META_PATTERNS_TRANSLATE = [
    r"as an ai language model",
    r"i am an ai",
    r"\bi cannot\b",
    r"\bi can't\b",
    r"como um modelo de linguagem",
    r"eu sou apenas",
    r"^\s*sorry[,:]?\s+(but\s+)?i (can't|cannot)\b",
    r"^\s*desculp\w*[,:]?\s+(mas\s+)?(n[aã]o posso|n[aã]o consigo|eu n[aã]o)\b",
]

# Conjunto agressivo (refine/strict); remove "desculp" genérico.
META_PATTERNS_STRICT = [
    r"parece que voc[eˆ] est[ a]",
    r"como um modelo de linguagem",
    r"n[a\u00c6]o posso",
    r"n[a\u00c6]o sou capaz",
    r"n\u00c6o posso ajudar",
    r"eu sou apenas",
    r"como um assistente",
    r"as an ai language model",
    r"i am an ai",
    r"i cannot provide",
    r"i'm just an ai",
    r"as a language model",
    r"^\s*mudanc(a|\u2021)as e justificativas[:]?.*$",
    r"^\s*alterac(ao|\u00c6o|oes|\u00e4es) realizadas[:]?.*$",
    r"^\s*(nesta|nessa) revis(ao|\u00c6o).*$",
    r"^\s*(justificativa|racionalidade|rationale).*$",
    r"^\s*em resumo.*$",
    r"^\s*resumo[: ].*$",
]

REFINE_MARKER_BLOCK_RE = re.compile(
    r"###\s*TEXTO_REFINADO_INICIO\s*(.*?)\s*###\s*TEXTO_REFINADO_FIM",
    flags=re.IGNORECASE | re.DOTALL,
)
REFINE_DELIMITER_RE = re.compile(r"(?m)^\s*(?:\*{3,}|-{3,})\s*$")
REFINE_META_PREAMBLE_RE = re.compile(
    r"(?:aqui\s+est[áa]|segue\s+(?:a\s+)?revis[ãa]o|revis[ãa]o\s+(?:do|deste)|"
    r"here\s+is|below\s+is).{0,180}(?:texto|revis[ãa]o|ajuste)|"
    r"(?:principais\s+)?(?:ajustes|altera[çc][õo]es|mudan[çc]as)\s+(?:feitos|realizadas|e\s+notas)",
    flags=re.IGNORECASE | re.DOTALL,
)
REFINE_BODY_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{1,2}\s*)?"
    r"(?:texto\s+(?:revisado|refinado)|revis[ãa]o\s+do\s+texto)"
    r"(?:\s*\([^\n)]*\))?\s*:?[\s*]*$"
)
REFINE_META_SECTION_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{1,2}\s*)?"
    r"(?:(?:principais\s+)?(?:ajustes|altera[çc][õo]es|mudan[çc]as|notas|observa[çc][õo]es)"
    r"(?:\s+(?:feitos|realizadas|da\s+revis[ãa]o))?)\s*:?[\s*]*$"
)


def _extract_delimited_refine_text(text: str) -> str:
    """Extrai apenas o texto quando o modelo cerca a revisão com meta-comentários.

    Alguns modelos locais ignoram os marcadores pedidos e respondem com uma
    introdução, `***`, o texto revisado, outro `***` e uma lista de mudanças.
    O recorte só é aplicado quando a moldura contém metatexto inequívoco; um
    separador de cena legítimo da obra, isoladamente, permanece intacto.
    """
    marker_match = REFINE_MARKER_BLOCK_RE.search(text)
    if marker_match:
        candidate = marker_match.group(1).strip()
        if candidate:
            prefix = text[: marker_match.start()].strip()
            # Um título Markdown pode ficar fora dos marcadores sem ser
            # metacomentário do modelo. Preserve apenas esse prefixo estrito.
            prefix_lines = [line.strip() for line in prefix.splitlines() if line.strip()]
            if prefix_lines and all(re.fullmatch(r"#{1,6}\s+.+", line) for line in prefix_lines):
                return f"{prefix}\n\n{candidate}"
            return candidate

    body_heading = REFINE_BODY_HEADING_RE.search(text)
    if body_heading:
        candidate = text[body_heading.end() :]
        meta_section = REFINE_META_SECTION_RE.search(candidate)
        if meta_section:
            candidate = candidate[: meta_section.start()]
        # Separadores introduzidos antes de uma seção de notas também não são
        # conteúdo da novel.
        candidate = re.sub(r"(?:\n\s*(?:\*{3,}|-{3,})\s*)+$", "", candidate).strip()
        if candidate:
            return candidate

    delimiters = list(REFINE_DELIMITER_RE.finditer(text))
    if len(delimiters) < 2:
        return text

    preamble = text[: delimiters[0].start()]
    postamble = text[delimiters[1].end() :]
    if not (REFINE_META_PREAMBLE_RE.search(preamble) or REFINE_META_PREAMBLE_RE.search(postamble)):
        return text

    candidate = text[delimiters[0].end() : delimiters[1].start()].strip()
    return candidate or text


@dataclass
class SanitizationReport:
    """Reúne as alterações e alertas produzidos pela sanitização."""

    removed_think_blocks: int = 0
    removed_meta_lines: int = 0
    removed_repeated_lines: int = 0
    removed_repeated_paragraphs: int = 0
    removed_empty_lines: int = 0
    contamination_detected: bool = False
    leading_noise_removed: bool = False
    removed_lines_count: int = 0
    collapsed_repetitions: int = 0


def _remove_think_blocks(text: str) -> Tuple[str, int]:
    """Remove blocos internos de raciocínio expostos pelo modelo."""
    pattern = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
    new_text, count = pattern.subn("", text)
    return new_text, count


def _remove_meta_lines(text: str, patterns: List[str]) -> Tuple[str, int, bool]:
    """Remove linhas de metatexto que não pertencem à tradução."""
    lines = text.splitlines()
    kept: List[str] = []
    removed = 0
    contamination = False
    for line in lines:
        lowered = line.lower()
        if any(re.search(pat, lowered) for pat in patterns):
            removed += 1
            contamination = True
            continue
        kept.append(line)
    return "\n".join(kept), removed, contamination


def _collapse_repeated_lines(text: str) -> Tuple[str, int]:
    """Reduz repetições consecutivas da mesma linha."""
    lines = text.splitlines()
    kept: List[str] = []
    removed = 0
    prev = None
    for line in lines:
        if prev is not None and line.strip() and prev.strip() == line.strip():
            removed += 1
            continue
        kept.append(line)
        prev = line
    return "\n".join(kept), removed


def _collapse_repeated_paragraphs(text: str) -> Tuple[str, int]:
    """Reduz repetições consecutivas do mesmo parágrafo."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    kept: List[str] = []
    removed = 0
    prev = None
    for p in paragraphs:
        if prev is not None and p == prev:
            removed += 1
            continue
        kept.append(p)
        prev = p
    return "\n\n".join(kept), removed


def _strip_empty_lines(text: str) -> Tuple[str, int]:
    """Remove linhas vazias da coleção intermediária."""
    lines = text.splitlines()
    kept: List[str] = []
    removed = 0
    for line in lines:
        if line.strip() == "":
            removed += 1
            continue
        kept.append(line.rstrip())
    return "\n".join(kept), removed


def _remove_repeated_sequences(text: str) -> Tuple[str, int]:
    """Remove sequências longas produzidas por loops do modelo."""
    pattern = re.compile(r"(.{50,}?)(?:\s+\1){1,}", flags=re.DOTALL)
    new_text, count = pattern.subn(lambda m: m.group(1), text)
    return new_text, count


def remove_leading_noise(text: str) -> str:
    """Remove ruído evidente no início sem descartar conteúdo narrativo."""
    lines = text.splitlines()
    cleaned: List[str] = []
    started = False

    for line in lines:
        stripped = line.strip()

        if not started:
            if not stripped:
                continue
            if (
                len(stripped) <= 12
                and not re.search(r"[A-Za-zÀ-ÿ0-9]", stripped)
                and not re.search(r"[.!?…]$", stripped)
            ):
                continue

            started = True

        cleaned.append(line)

    return "\n".join(cleaned)


def sanitize_text(
    text: str,
    logger: logging.Logger | None = None,
    fail_on_contamination: bool = True,
    collapse_repeated_lines: bool = True,
    collapse_repeated_paragraphs: bool = True,
    remove_repeated_sequences: bool = True,
    strip_empty_lines: bool = True,
    apply_leading_noise_filter: bool = True,
    meta_patterns: List[str] | None = None,
) -> Tuple[str, SanitizationReport]:
    """Reduz metatexto, loops e ruídos recorrentes em saídas do modelo."""
    report = SanitizationReport()

    text, count = _remove_think_blocks(text)
    report.removed_think_blocks = count

    meta_used = meta_patterns if meta_patterns is not None else META_PATTERNS_STRICT
    text, meta_removed, contamination = _remove_meta_lines(text, meta_used)
    report.removed_meta_lines = meta_removed
    report.contamination_detected = contamination

    repeated_lines = 0
    if collapse_repeated_lines:
        text, repeated_lines = _collapse_repeated_lines(text)
    report.removed_repeated_lines = repeated_lines

    seq_removed = 0
    repeated_paragraphs = 0
    if remove_repeated_sequences:
        text, seq_removed = _remove_repeated_sequences(text)
    if collapse_repeated_paragraphs:
        text, repeated_paragraphs = _collapse_repeated_paragraphs(text)
    report.removed_repeated_paragraphs = seq_removed + repeated_paragraphs

    empty = 0
    if strip_empty_lines:
        text, empty = _strip_empty_lines(text)
    report.removed_empty_lines = empty
    report.collapsed_repetitions = seq_removed + repeated_paragraphs

    before_noise = text
    if apply_leading_noise_filter:
        text = remove_leading_noise(text)
    report.leading_noise_removed = apply_leading_noise_filter and text != before_noise
    # Remove aspas triplas soltas no fim de linha.
    text = re.sub(r'"""\s*$', "", text, flags=re.MULTILINE)
    text = text.replace("<think>", "").replace("</think>", "")
    report.removed_lines_count = (
        report.removed_meta_lines + report.removed_repeated_lines + report.removed_empty_lines
    )

    text = text.strip()
    if not text:
        if logger:
            logger.error("Sanitizacao resultou em texto vazio.")
        raise ValueError("Texto vazio apos sanitizacao.")

    if fail_on_contamination and report.contamination_detected:
        raise ValueError("Contaminacao detectada na saida do modelo.")

    return text, report


def sanitize_translation_output(
    text: str,
    logger: logging.Logger | None = None,
    fail_on_contamination: bool = False,
) -> Tuple[str, SanitizationReport]:
    """Aplica a sanitização conservadora destinada à tradução."""
    return sanitize_text(
        text,
        logger=logger,
        fail_on_contamination=fail_on_contamination,
        collapse_repeated_lines=False,
        collapse_repeated_paragraphs=False,
        remove_repeated_sequences=False,
        strip_empty_lines=False,
        apply_leading_noise_filter=True,
        meta_patterns=META_PATTERNS_TRANSLATE,
    )


def sanitize_refine_output(text: str) -> str:
    """Aplica a sanitização conservadora destinada ao refino."""
    cleaned = _extract_delimited_refine_text(text)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    filtered_lines = []
    for line in cleaned.splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("texto refinado:") or lowered.startswith("refined text:"):
            continue
        filtered_lines.append(line)
    cleaned = "\n".join(filtered_lines)

    cleaned = re.sub(
        r"^\s*###\s*TEXTO_REFINADO_(?:INICIO|FIM)\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    cleaned = re.sub(
        r"###\s*TEXTO_REFINADO_(?:INICIO|FIM)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"###\s*TEXTO_TRADUZ[A-Z_]*INICIO.*?###\s*TEXTO_TRADUZ[A-Z_]*FIM",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"###\s*TEXTO_TRADUZ[A-Z_]*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"===GLOSSARIO_SUGERIDO_INICIO===.*?===GLOSSARIO_SUGERIDO_FIM===",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    start = cleaned.find("===GLOSSARIO_SUGERIDO_INICIO===")
    end = cleaned.find("===GLOSSARIO_SUGERIDO_FIM===")
    if start != -1 and (end == -1 or end < start):
        pre = cleaned[:start]
        pre_rstrip = pre.rstrip()
        if pre_rstrip.endswith('"""'):
            pre = pre_rstrip[:-3]
        cleaned = pre.rstrip()

    return cleaned.strip()


def log_report(report: SanitizationReport, logger: logging.Logger, prefix: str) -> None:
    """Registra o relatório de sanitização com o prefixo da etapa."""
    logger.debug(
        "%s sanitizacao -> think:%d meta:%d rep_linhas:%d rep_parag:%d vazias:%d contam:%s leading_noise:%s colapsos:%d",
        prefix,
        report.removed_think_blocks,
        report.removed_meta_lines,
        report.removed_repeated_lines,
        report.removed_repeated_paragraphs,
        report.removed_empty_lines,
        report.contamination_detected,
        report.leading_noise_removed,
        report.collapsed_repetitions,
    )
