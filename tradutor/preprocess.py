"""
Pré-processamento de PDFs: extração de texto, limpeza e chunking seguro.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Final, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - fallback para ambientes sem PyMuPDF

    class _DummyDoc:
        """Simula um documento vazio quando a biblioteca de PDF não está disponível."""

        def __enter__(self):
            """Inicia o uso do dublê no gerenciador de contexto."""
            return self

        def __exit__(self, *args, **kwargs):
            """Finaliza o uso do dublê no gerenciador de contexto."""
            return False

        def __iter__(self):
            """Percorre os itens fornecidos pelo objeto."""
            return iter([])

        @property
        def pages(self):
            """Retorna as páginas simuladas do documento."""
            return []

        def __len__(self):
            """Retorna a quantidade de itens disponíveis."""
            return 0

    class _DummyFitz:
        """Fornece a interface mínima de PDF usada pelo fallback sem dependências."""

        def open(self, *args, **kwargs):
            """Abre o documento simulado usado no teste."""
            return _DummyDoc()

    fitz = _DummyFitz()  # type: ignore

from .quote_fix import fix_blank_lines_inside_quotes
from .section_splitter import SECTION_PATTERN
from .utils import chunk_by_paragraphs

# Padrões genéricos de rodapé e material promocional.
FOOTER_PATTERNS: Final[list[str]] = [
    r"\bPage\s+\d+\b",
    r"newsletter",
    r"stay up to date",
    r"download(?:ing)? our mobile app",
    r"download all your favorite light novels",
    r"favorite light novels",
    r"(?:or\s+)?visit us online",
]

NOISE_PARAGRAPH_PATTERNS: Final[list[str]] = [
    r"stay up to date",
    r"download(?:ing)? our mobile app",
    r"^\s*join our\b.*\b(?:community|server)\b",
    r"newsletter",
    r"^\s*follow us\b.*\b(?:online|social media|updates?)\b",
    r"^\s*support (?:us|our work)\b.*\b(?:donat|membership|website|site|online)\b",
    r"^\s*read (more|the latest) on\b",
    r"get the latest news",
    r"^\s*(?:or\s+)?visit us online\b",
]

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b(?:[a-z0-9-]+\.)+[a-z]{2,63}(?:[/?#]\S*)?",
    re.IGNORECASE,
)

TOC_MARKER_RE = SECTION_PATTERN

PROMO_PHRASES: Final[list[str]] = [
    "thank you for reading",
    "thank you for downloading",
    "get the latest news",
]

PROMO_LINE_REGEXES: Final[list[str]] = [
    r"^\s*sign up for\b.*\b(?:newsletter|updates?|email|inbox|account|free|alerts?)\b",
    r"^\s*sign up for our\s*[!.:]?$",
    r"^\s*read online\b",
    r"^\s*read (?:more|the latest) on\b",
    r"^\s*download(?:ing)?\b.*\b(?:mobile app|favorite light novels|light novels|pdf|e-?books?|app)\b",
    r"^\s*join our\b.*\b(?:community|server)\b",
    r"^\s*follow us\b.*\b(?:online|social media|updates?)\b",
    r"^\s*support (?:us|our work)\b.*\b(?:donat|membership|website|site|online)\b",
    r"^\s*(?:or\s+)?visit us online\b",
]

TOC_MARKER_LINES: Final[list[str]] = [
    "table of contents",
    # "contents" tratado como marker apenas quando a linha é exatamente o termo, ver _is_marker
    "sumário",
    "índice",
    "indice",
    "índice remissivo",
    "color inserts",
    "inserções coloridas",
    "title page",
    "página de título",
    "copyrights and credits",
    "newsletter",
]


def normalize_line_for_filters(line: str) -> str:
    """Normaliza uma linha para fins de filtro (não altera texto final)."""
    if not line:
        return ""
    normalized = line.replace("\xa0", " ")
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    normalized = normalized.strip()
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized


def _is_ellipsis_line(raw_line: str) -> bool:
    """Verifica se a linha contém apenas reticências e pontuação de diálogo."""
    stripped = raw_line.strip()
    return bool(re.fullmatch(r"[\"“”']?\u2026[\"“”']?", stripped)) or bool(
        re.fullmatch(r"[\"“”']?\.{3,}[\"“”']?", stripped)
    )


def _sha256_text(text: str) -> str:
    """Calcula o SHA-256 usado para identificar o conteúdo de entrada."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_noise_glossary() -> dict:
    """Retorna regras genéricas de ruído, sem marcadores de uma fonte específica."""
    return {
        "line_contains": PROMO_PHRASES + ["favorite light novels"],
        "line_compact_contains": [],
        "line_regex": list(PROMO_LINE_REGEXES),
        "inline_regex": [
            r"https?://[^\s]+",
            r"\bwww\.[^\s]+",
            r"\b(?:[a-z0-9-]+\.)+[a-z]{2,63}(?:[/?#][^\s]*)?",
        ],
        "max_line_len": 160,
    }


def _load_noise_glossary(path: str | Path | None) -> dict:
    """Carrega ruído glossário."""
    if not path:
        return _default_noise_glossary()
    p = Path(path)
    if not p.exists():
        return _default_noise_glossary()
    try:
        import json

        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _default_noise_glossary()
    merged = _default_noise_glossary()
    for key in ("line_contains", "line_compact_contains", "line_regex", "inline_regex"):
        if isinstance(data.get(key), list):
            merged[key] = data[key]
    if isinstance(data.get("max_line_len"), int) and data["max_line_len"] > 0:
        merged["max_line_len"] = data["max_line_len"]
    return merged


def extract_text_from_pdf(path: Path, logger: logging.Logger) -> str:
    """Extrai texto de um PDF usando PyMuPDF."""
    with fitz.open(path) as doc:
        pages = [page.get_text() or "" for page in doc]
    text = "\n".join(pages)
    logger.debug("PDF %s extraído: %d caracteres", path.name, len(text))
    return text


def _remove_headers_footers(text: str) -> str:
    """Remove cabeçalhos e rodapés recorrentes extraídos do PDF."""
    lines = text.splitlines()
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        # Números de página isolados ou cabeçalhos típicos
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if len(stripped) <= 5 and stripped.isupper():
            continue
        if re.search(r"\bpage\b", stripped, re.IGNORECASE):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def sanitize_extracted_text(text: str, logger: Optional[logging.Logger] = None) -> Tuple[str, dict]:
    """
    Remove ruídos determinísticos da extração (caractere U+FFFF e linhas só com números).
    Preserva estrutura de parágrafos.
    """
    stats = {"removed_uffff": 0, "removed_numeric_lines": 0}

    before = text
    text = text.replace("\uffff", "").replace("￿", "")
    stats["removed_uffff"] = len(before) - len(text)  # proxy de remoções de char

    lines = text.splitlines()
    cleaned: list[str] = []
    for ln in lines:
        if re.fullmatch(r"\s*\d+\s*", ln):
            stats["removed_numeric_lines"] += 1
            continue
        cleaned.append(ln)
    cleaned_text = "\n".join(cleaned)
    if logger:
        logger.debug(
            "Sanitize extracted: removed_numeric_lines=%d removed_uffff=%d",
            stats["removed_numeric_lines"],
            stats["removed_uffff"],
        )
    return cleaned_text, stats


def _remove_hyphenation(text: str) -> str:
    """Reconstrói palavras separadas por hifenização no fim da linha."""
    return re.sub(r"(\w+)-\s*\n(\w+)", r"\1\2\n", text)


def _join_broken_lines(text: str) -> str:
    """Reconstrói parágrafos quebrados artificialmente por linhas do PDF."""
    lines = text.splitlines()
    joined: List[str] = []
    buffer: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                joined.append(" ".join(buffer))
                buffer = []
            continue
        if re.search(r"[.!?…]$", stripped):
            buffer.append(stripped)
            joined.append(" ".join(buffer))
            buffer = []
        else:
            buffer.append(stripped)

    if buffer:
        joined.append(" ".join(buffer))

    return "\n\n".join(joined)


def _remove_noise_blocks_with_stats(text: str) -> tuple[str, dict, list[str]]:
    """Remove blocos promocionais configurados e registra as ocorrências."""
    paragraphs = text.split("\n\n")
    cleaned: list[str] = []
    removed: list[str] = []
    pattern_counts: Counter[str] = Counter()
    for para in paragraphs:
        norm = para.lower().strip()
        if not norm:
            cleaned.append("")
            continue
        matched = next(
            (pat for pat in NOISE_PARAGRAPH_PATTERNS if re.search(pat, norm, flags=re.IGNORECASE)),
            "",
        )
        if matched:
            removed_norm = normalize_line_for_filters(para)
            if removed_norm:
                removed.append(removed_norm)
                pattern_counts[matched] += 1
            continue
        cleaned.append(para.strip())
    stats = {
        "noise_blocks_removed_count": len(removed),
        "noise_blocks_removed_pattern_counts": dict(pattern_counts),
        "noise_blocks_removed_samples": removed[:10],
    }
    return "\n\n".join(p for p in cleaned if p != ""), stats, removed


def remove_noise_blocks(text: str) -> str:
    """Remove blocos promocionais configurados sem expor as métricas internas."""
    cleaned, _, _ = _remove_noise_blocks_with_stats(text)
    return cleaned


def _is_promo_line(line: str) -> bool:
    """Indica se a linha corresponde a uma chamada promocional genérica."""
    normalized = normalize_line_for_filters(line).lower()
    if not normalized:
        return False
    has_url = bool(URL_RE.search(normalized))
    has_phrase = any(phrase in normalized for phrase in PROMO_PHRASES)
    has_pattern = any(
        re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in PROMO_LINE_REGEXES
    )
    return has_url or has_phrase or has_pattern


def _remove_promo_lines(text: str, glossary: dict) -> tuple[str, dict, list[str]]:
    """Remove linhas promocionais reconhecidas com baixo risco de atingir a narrativa."""
    lines = text.splitlines()
    cleaned: list[str] = []
    stats = {
        "known_watermark_removed_count": 0,
        "promo_lines_removed_count": 0,
        "promo_blocks_removed_count": 0,
        "urls_removed_count": 0,
        "promo_samples": [],
        "promo_removed_hash": "",
        "promo_removed_reason_counts": {},
        "promo_removed_samples": {},
    }
    removed_norms: list[str] = []
    max_len = glossary.get("max_line_len", 160) or 160
    line_contains = [s.lower() for s in glossary.get("line_contains", []) if isinstance(s, str)]
    line_compact = [
        re.sub(r"[^a-z0-9]", "", s.lower())
        for s in glossary.get("line_compact_contains", [])
        if isinstance(s, str)
    ]
    domain_tokens = [s for s in line_contains if "." in s or "/" in s] + line_compact
    phrase_tokens = [s for s in line_contains if s not in domain_tokens]
    regexes = []
    for pat in glossary.get("line_regex", []):
        try:
            regexes.append(re.compile(pat, flags=re.IGNORECASE))
        except re.error:
            continue

    def _is_dialogue_like(norm_line: str, raw_line: str) -> bool:
        """Verifica se a linha tem estrutura típica de diálogo."""
        stripped = raw_line.lstrip()
        if stripped.startswith(('"', "“", "'", "’", "—", "-")):
            return True
        compact = re.sub(r"[\\s]", "", norm_line)
        if re.fullmatch(r"[.·…]{2,}", norm_line):
            return True
        if len(compact) <= 12 and re.fullmatch(r"[A-Za-z]{1,6}[!?…—.]+", compact):
            return True
        return False

    def _record_reason(reason: str, sample: str) -> None:
        """Registra o motivo e uma amostra de cada remoção."""
        stats["promo_removed_reason_counts"][reason] = (
            stats["promo_removed_reason_counts"].get(reason, 0) + 1
        )
        samples = stats["promo_removed_samples"].setdefault(reason, [])
        if len(samples) < 10:
            samples.append(sample)

    def _update_stats(*, has_url: bool, has_domain: bool) -> None:
        """Atualiza as métricas de remoção da linha promocional atual."""
        stats["promo_lines_removed_count"] += 1
        if has_url or has_domain:
            stats["urls_removed_count"] += 1
            stats["known_watermark_removed_count"] += 1

    removed_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        normalized = normalize_line_for_filters(line)
        norm_lower = normalized.lower()
        norm_compact = re.sub(r"[^a-z0-9]", "", norm_lower)
        if _is_ellipsis_line(line):
            cleaned.append(line)
            i += 1
            continue
        if not normalized:
            cleaned.append("")
            i += 1
            continue
        has_domain = any(dom in norm_lower for dom in domain_tokens) or any(
            tok in norm_compact for tok in line_compact
        )
        has_url = bool(URL_RE.search(norm_lower))
        has_phrase = any(phrase in norm_lower for phrase in phrase_tokens)
        has_regex = any(r.search(line) for r in regexes)
        looks_dialogue = line.lstrip().startswith(('"', "“", "'", "’", "-", "–"))
        long_narrative = len(normalized) > 180 and not (has_url or has_domain)
        promo_seed = has_domain or has_url or has_phrase or has_regex
        if _is_dialogue_like(norm_lower, line) and not (has_domain or has_url or has_regex):
            cleaned.append(line)
            i += 1
            continue
        if looks_dialogue and not has_domain and not has_url:
            cleaned.append(line)
            i += 1
            continue
        if promo_seed and (
            has_domain or has_url or (not long_narrative and len(normalized) <= max_len)
        ):
            block_end = i + 1
            while block_end < len(lines):
                next_norm = normalize_line_for_filters(lines[block_end])
                if not next_norm:
                    block_end += 1
                    break
                next_lower = next_norm.lower()
                next_compact = re.sub(r"[^a-z0-9]", "", next_lower)
                next_domain = any(dom in next_lower for dom in domain_tokens) or any(
                    tok in next_compact for tok in line_compact
                )
                next_url = bool(URL_RE.search(next_lower))
                next_phrase = any(phrase in next_lower for phrase in phrase_tokens)
                next_regex = any(r.search(lines[block_end]) for r in regexes)
                short_line = len(next_norm) <= 140
                letter_ratio = sum(1 for ch in next_norm if ch.isalpha()) / max(len(next_norm), 1)
                low_linguistic = letter_ratio < 0.65
                if next_domain or next_url or next_phrase or next_regex:
                    block_end += 1
                    continue
                if _is_dialogue_like(next_norm, lines[block_end]):
                    break
                if short_line and (low_linguistic or next_norm.endswith(":")):
                    block_end += 1
                    continue
                break
            removed_block = lines[i:block_end]
            if len(removed_block) > 1:
                stats["promo_blocks_removed_count"] += 1
            for removed in removed_block:
                removed_norm = normalize_line_for_filters(removed)
                if not removed_norm:
                    continue
                removed_lower = removed_norm.lower()
                removed_compact = re.sub(r"[^a-z0-9]", "", removed_lower)
                removed_has_domain = any(dom in removed_lower for dom in line_contains) or any(
                    tok in removed_compact for tok in line_compact
                )
                removed_has_url = bool(URL_RE.search(removed_lower))
                _update_stats(
                    has_url=removed_has_url,
                    has_domain=removed_has_domain,
                )
                _record_reason("block", removed_norm)
                if removed_norm:
                    removed_norms.append(removed_norm)
                    removed_lines.append(removed_norm)
            i = block_end
            continue

        if promo_seed and (len(normalized) <= max_len or has_url or has_domain):
            _update_stats(has_url=has_url, has_domain=has_domain)
            removed_norms.append(normalized)
            removed_lines.append(normalized)
            _record_reason("single", normalized)
            i += 1
            continue
        cleaned.append(line)
        i += 1

    remaining_counts = {
        "configured_markers": sum(
            1
            for line in cleaned
            if any(token in normalize_line_for_filters(line).lower() for token in line_contains)
            or any(
                token
                in re.sub(
                    r"[^a-z0-9]",
                    "",
                    normalize_line_for_filters(line).lower(),
                )
                for token in line_compact
            )
        )
    }
    stats["remaining_counts"] = remaining_counts
    stats["promo_lines_removed_total"] = len(removed_norms)
    max_samples = 20
    stats["promo_samples"] = removed_lines[:max_samples]
    stats["promo_samples_truncated"] = len(removed_lines) > max_samples
    stats["promo_removed_hash"] = _sha256_text("\n".join(removed_lines)) if removed_lines else ""
    return "\n".join(cleaned), stats, removed_norms


def _dedupe_consecutive_lines(text: str) -> tuple[str, dict, list[str]]:
    """Remove linhas consecutivas idênticas preservando a estrutura."""
    lines = text.splitlines()
    cleaned: list[str] = []
    removed_norms: list[str] = []
    prev_norm: str | None = None
    removed_count = 0
    for line in lines:
        if _is_ellipsis_line(line):
            cleaned.append(line)
            prev_norm = None
            continue
        if line.lstrip().startswith(('"', "“", "'", "’", "-", "–", "—")):
            cleaned.append(line)
            prev_norm = None
            continue
        norm = normalize_line_for_filters(line)
        if norm and prev_norm and norm == prev_norm:
            removed_count += 1
            removed_norms.append(norm)
            continue
        cleaned.append(line)
        if norm:
            prev_norm = norm
        else:
            prev_norm = None
    return "\n".join(cleaned), {"dedupe_removed_count": removed_count}, removed_norms


def _merge_hard_wraps_across_gaps(text: str) -> tuple[str, dict]:
    """Une continuações de frase separadas por quebras artificiais de página."""
    lines = text.splitlines()
    merged: list[str] = []
    merges = 0
    i = 0
    chapter_line_re = re.compile(r"chapter\s+\d+:?", re.IGNORECASE)
    while i < len(lines):
        curr = lines[i]
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            nxt = lines[j]
        else:
            nxt = ""
        if curr.strip() and nxt.strip() and j > i + 1:
            curr_up = curr.strip().isupper() and len(curr.strip()) <= 40
            nxt_up = nxt.strip().isupper() and len(nxt.strip()) <= 40
            curr_ends_sentence = bool(re.search(r"[.!?…]['\"]?$", curr.strip()))
            nxt_dialogue = nxt.lstrip().startswith(('"', "“", "‘", "—", "-"))
            # PDFs às vezes separam uma citação curta do restante da frase:
            # 'from' seguido de '“the bottom of his heart.”'. Não é uma nova fala.
            nxt_inline_quoted_continuation = bool(re.match(r'^["“‘]\s*[a-zà-öø-ÿ]', nxt.lstrip()))
            prev_is_chapter = bool(merged) and chapter_line_re.match(merged[-1].strip())
            nxt_heading = _is_heading_like(nxt) or (
                prev_is_chapter and len(nxt.strip().split()) <= 3
            )
            curr_subheading = (
                prev_is_chapter
                and (len(curr.strip().split()) <= 4)
                and (not _is_heading_like(curr))
            )
            if (
                not curr_ends_sentence
                and (not nxt_dialogue or nxt_inline_quoted_continuation)
                and not nxt_heading
                and not _is_heading_like(curr)
                and not curr_subheading
                and not _is_ellipsis_line(curr)
                and not (curr_up and nxt_up)
            ):
                merged.append(f"{curr.rstrip()} {nxt.lstrip()}")
                merges += 1
                i = j + 1
                continue
        merged.append(curr)
        i += 1
    return "\n".join(merged), {"hard_wrap_merges": merges}


def _remove_blank_between_dialogue(text: str) -> str:
    """Remove linhas vazias introduzidas dentro do mesmo diálogo."""
    lines = text.splitlines()
    out: list[str] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            prev_dialogue = out and out[-1].lstrip().startswith(('"', "“"))
            next_dialogue = False
            if idx + 1 < len(lines):
                next_dialogue = lines[idx + 1].lstrip().startswith(('"', "“"))
            if prev_dialogue and next_dialogue:
                continue
        out.append(line)
    return "\n".join(out)


def _ensure_subheading_isolated(text: str) -> tuple[str, dict]:
    """Mantém subtítulos curtos de capítulo em uma linha própria.

    A extração de PDF pode colar um subtítulo em title case ao início da
    narrativa em versalete, como ``The Last Lantern AFTER MARA...``. A
    separação preserva o limite estrutural usado nas etapas seguintes.
    """
    lines = text.splitlines()
    out: list[str] = []
    fixes = 0

    chapter_re = re.compile(r"^chapter\s+\d+:?\s*$", re.IGNORECASE)
    subtitle_word = r"(?:[A-Z][A-Za-z'’-]*|the|of|and|a|an|to|in|on|at|for|with|after|before)"
    subtitle_re = re.compile(
        rf"^(?P<subtitle>{subtitle_word}(?:\s+{subtitle_word}){{0,7}}?)\s+(?P<body>[A-Z]{{2,}}\b.*)$"
    )
    i = 0
    while i < len(lines):
        ln = lines[i]
        out.append(ln)
        if chapter_re.match(ln.strip()):
            # Procura a próxima linha não vazia.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                out.append(lines[j])
                j += 1
            if j < len(lines):
                cand = lines[j].strip()
                # Subtítulos em inglês podem conter conectores como
                # ``the`` e ``of``; a narrativa seguinte começa em versalete.
                m2 = subtitle_re.match(cand)
                if m2:
                    sub = m2.group("subtitle").strip()
                    rest = m2.group("body").strip()
                    out.append(sub)
                    out.append("")
                    out.append(rest)
                    fixes += 1
                    i = j + 1
                    continue
        i += 1
    return "\n".join(out), {"subheading_isolation_fixes": fixes}


def _split_dialogue_narration_boundaries(text: str) -> tuple[str, dict]:
    """Separa narração colada ao fim de um parágrafo de diálogo.

    A regra só atua quando a linha começa com aspa de abertura e encontra uma
    fronteira explícita após a aspa de fechamento.
    """
    lines = text.splitlines()
    out: list[str] = []
    fixes = 0
    # Aceita fechamento após pontuação ou nota musical.
    boundary_re = re.compile(r"([.!?…\\u266a])([\"”’\'])\\s+(?=[A-Z])")
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith(('"', "“", "‘")) and boundary_re.search(ln):
            new_ln = boundary_re.sub(r"\1\2\n\n", ln)
            if new_ln != ln:
                fixes += 1
                # A expressão já insere as quebras necessárias.
                out.extend(new_ln.splitlines())
                continue
        out.append(ln)
    return "\n".join(out), {"dialogue_narration_split_fixes": fixes}


def _wrap_very_long_lines(text: str, *, max_len: int = 1200) -> tuple[str, dict]:
    """Divide linhas muito longas sem criar novos parágrafos."""
    lines = text.splitlines()
    wrapped_lines: list[str] = []
    wrap_count = 0
    max_observed_length = 0
    lines_over_800 = 0
    for line in lines:
        line_length = len(line)
        max_observed_length = max(max_observed_length, line_length)
        if line_length > 800:
            lines_over_800 += 1
        if line_length <= max_len or not line.strip():
            wrapped_lines.append(line)
            continue
        # Prioriza cortes nos limites entre frases.
        remaining_text = line
        line_parts: list[str] = []
        while len(remaining_text) > max_len:
            # Procura o último fim de frase dentro do limite.
            split_at = None
            for sentence_boundary in re.finditer(r"[.!?…][\"”’']?\remaining_text+", remaining_text):
                if sentence_boundary.end() <= max_len:
                    split_at = sentence_boundary.end()
            if split_at is None or split_at < max_len * 0.5:
                # Na ausência de pontuação, corta no último espaço disponível.
                split_at = remaining_text.rfind(" ", 0, max_len)
                if split_at <= 0:
                    break
                split_at = split_at + 1
            line_parts.append(remaining_text[:split_at].rstrip())
            remaining_text = remaining_text[split_at:].lstrip()
            wrap_count += 1
        if line_parts:
            wrapped_lines.extend(line_parts)
            wrapped_lines.append(remaining_text)
        else:
            wrapped_lines.append(line)
    return "\n".join(wrapped_lines), {
        "very_long_line_wraps": wrap_count,
        "max_line_length": max_observed_length,
        "lines_over_800": lines_over_800,
    }


def _fix_under_merge(text: str) -> tuple[str, dict]:
    """Une trechos que permaneceram separados no meio de uma frase."""
    lines = text.splitlines()
    merged_lines: list[str] = []
    merge_count = 0
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        # Remove quebras vazias isoladas quando os dois lados formam a mesma frase.
        if not line.strip() and merged_lines:
            previous_line = merged_lines[-1] if merged_lines else ""
            if line_index + 1 < len(lines):
                next_line = lines[line_index + 1].lstrip()
                if (
                    previous_line
                    and next_line
                    and not re.search(r"[.!?…]['\"]?$", previous_line)
                    and previous_line[-1].isalpha()
                ):
                    next_is_heading = _is_heading_like(next_line)
                    next_is_dialogue = next_line.startswith(('"', "“", "‘", "—", "-"))
                    next_is_promotional = _is_promo_line(next_line)
                    if _is_heading_like(previous_line):
                        pass
                    elif (
                        not next_is_dialogue
                        and not next_is_heading
                        and not next_is_promotional
                        and (next_line[:1].islower() or next_line[:1].isupper())
                    ):
                        merged_lines[-1] = f"{previous_line} {next_line}"
                        merge_count += 1
                        line_index += 2
                        continue
            merged_lines.append(line)
            line_index += 1
            continue
        if line_index + 1 < len(lines):
            current_line = lines[line_index].rstrip()
            next_line = lines[line_index + 1].lstrip()
            if not current_line or not next_line:
                merged_lines.append(lines[line_index])
                line_index += 1
                continue
            curr_is_heading = _is_heading_like(current_line)
            ends_sentence = bool(re.search(r"[.!?…]['\"]?$", current_line))
            starts_dialogue = next_line.startswith(('"', "“", "‘", "—", "-"))
            promo_guard = _is_promo_line(next_line)
            next_is_heading = _is_heading_like(next_line)
            curr_all_caps = current_line.strip().isupper()
            if (
                current_line.strip().isupper()
                and next_line.strip().isupper()
                and len(current_line.strip()) <= 40
                and len(next_line.strip()) <= 40
            ):
                merged_lines.append(lines[line_index])
                line_index += 1
                continue
            if (
                current_line.strip().isupper()
                and len(current_line.strip()) <= 40
                and next_line[:1].isupper()
            ):
                merged_lines.append(lines[line_index])
                line_index += 1
                continue
            if (
                not ends_sentence
                and re.search(r"[A-Za-z0-9,;:–—-]$", current_line)
                and (
                    next_line[:1].islower()
                    or (
                        next_line[:1].isupper()
                        and len(next_line.split(" ")[0]) <= 4
                        and not starts_dialogue
                        and not next_is_heading
                    )
                )
                and not promo_guard
                and not curr_is_heading
                and not curr_all_caps
            ):
                merged_lines.append(f"{current_line} {next_line}")
                merge_count += 1
                line_index += 2
                continue
        merged_lines.append(lines[line_index])
        line_index += 1
    return "\n".join(merged_lines), {"under_merge_fixes": merge_count}


def _reflow_paragraphs(text: str) -> tuple[str, dict]:
    """
    Colapsa quebras duras dentro do mesmo par·grafo, mantendo linhas vazias e headings.

    HeurÌstica conservadora: n„o atravessa linhas vazias, headings ou separadores (***).
    """
    lines = text.splitlines()
    potential_merges = 0
    for idx in range(len(lines) - 1):
        curr = lines[idx].strip()
        nxt = lines[idx + 1].strip()
        if curr and nxt and not curr.endswith((".", "!", "?", "…")):
            potential_merges += 1
    reflowed: list[str] = []
    buffer: list[str] = []
    merges = 0

    def _flush() -> None:
        """Consolida o bloco acumulado antes de continuar o processamento."""
        nonlocal merges
        if not buffer:
            return
        if len(buffer) > 1:
            merges += len(buffer) - 1
        reflowed.append(" ".join(buffer))
        buffer.clear()

    def _is_dialogue_start(s: str) -> bool:
        """Indica se diálogo início."""
        return s.startswith(('"', "“"))

    chapter_line_re = re.compile(r"chapter\s+\d+:?", re.IGNORECASE)
    last_emitted: str | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            _flush()
            reflowed.append("")
            continue
        if stripped.isupper() and len(stripped) <= 40:
            _flush()
            reflowed.append(line)
            reflowed.append("")
            continue
        if re.fullmatch(r"[\s\"“”']*(?:[.·…]{2,})[\s\"“”']*", stripped):
            _flush()
            reflowed.append(stripped)
            continue
        if re.fullmatch(r"[\s\"“”'\-–—]*[.?!…]+[\s\"“”'\-–—]*", stripped):
            _flush()
            reflowed.append(stripped)
            _flush()
            continue
        if _is_dialogue_start(stripped):
            _flush()
            buffer.append(stripped)
            continue
        if _is_heading_like(stripped) or re.fullmatch(r"\*{2,}", stripped):
            _flush()
            reflowed.append(line)
            # Insere uma única linha vazia após o título.
            next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
            if next_line and (not reflowed or reflowed[-1] != ""):
                reflowed.append("")
            last_emitted = stripped
            continue
        prev_is_chapter = bool(last_emitted) and chapter_line_re.match(last_emitted.lower())
        if prev_is_chapter and len(stripped.split()) <= 3:
            _flush()
            reflowed.append(stripped)
            last_emitted = stripped
            continue
        # Se o parágrafo de diálogo já foi fechado, não anexa a narração seguinte
        # ao mesmo bloco.
        if buffer and buffer[0].lstrip().startswith(('"', "“")):
            prev_tail = buffer[-1].rstrip()
            if re.search(r"[\"”’']\s*$", prev_tail) and (not stripped.startswith(('"', "“"))):
                if stripped[:1].isupper():
                    _flush()
        if not buffer:
            buffer.append(line.strip())
        else:
            buffer.append(line.strip())
            merges += 1
        last_emitted = None
    _flush()
    if merges == 0 and potential_merges:
        merges = potential_merges
    return "\n".join(reflowed), {"reflow_merges": merges}


def _normalize_uppercase_sentences(text: str) -> tuple[str, dict]:
    """
    Converte linhas inteiras em CAPS (n„o headings) para sentence case seguro.
    Evita mexer em headings e separadores para n„o quebrar narrativa.
    """
    normalized: list[str] = []
    fixes = 0
    for line in text.splitlines():
        stripped = line.strip()
        if (
            stripped
            and stripped.isupper()
            and len(stripped) > 8
            and " " in stripped
            and not _is_heading_like(stripped)
            and not re.fullmatch(r"\*{2,}", stripped)
            and not any(p in stripped for p in ("!", "?"))
            and not stripped.startswith(('"', "“", "‘"))
        ):
            match = re.match(r"^([\"“‘(\\[]*)(.+)$", stripped)
            if match:
                prefix, body = match.groups()
                body_lower = body.lower()
                sentence = prefix + body_lower[:1].upper() + body_lower[1:]
                normalized.append(sentence)
                fixes += 1
            else:
                normalized.append(line)
        else:
            normalized.append(line)
    return "\n".join(normalized), {"uppercase_sentence_normalized": fixes}


def _normalize_leading_small_caps(text: str) -> tuple[str, dict]:
    """Remove pequenas capitulares quebradas na abertura de uma frase.

    Alguns PDFs extraem a abertura temporal de uma frase em caixa alta
    (``AFTER MARA watched``), embora o restante esteja em caixa normal.
    O padrão e deliberadamente estreito para preservar enfases legitimas como
    ``FIRST OFF`` ou nomes e gritos em CAPS.
    """
    normalized: list[str] = []
    fixes = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _is_heading_like(stripped) or stripped.startswith(("#", "***")):
            normalized.append(line)
            continue
        match = re.match(r"^(?P<prefix>[\"“‘(\[]?)(?P<body>.+)$", stripped)
        if not match:
            normalized.append(line)
            continue
        body = match.group("body")
        small_caps = re.match(
            r"^(?P<lead>AFTER|BEFORE|WHEN|WHILE|ONCE)\s+(?P<name>[A-Z]{3,})(?P<tail>\s+.*[a-zà-öø-ÿ].*)$",
            body,
        )
        if not small_caps or small_caps.group("name") in {"THE", "AND", "FOR", "WITH"}:
            normalized.append(line)
            continue
        normalized.append(
            match.group("prefix")
            + small_caps.group("lead").lower().capitalize()
            + " "
            + small_caps.group("name").lower().capitalize()
            + small_caps.group("tail")
        )
        fixes += 1
    return "\n".join(normalized), {"leading_small_caps_normalized": fixes}


def _strip_inline_watermarks(text: str, glossary: dict) -> tuple[str, dict]:
    """Remove do texto URLs ou marcadores embutidos configurados pelo usuário."""
    patterns: list[re.Pattern[str]] = []
    for raw_pattern in glossary.get("inline_regex", []):
        if not isinstance(raw_pattern, str):
            continue
        try:
            patterns.append(re.compile(raw_pattern, flags=re.IGNORECASE))
        except re.error:
            continue

    before = text
    for pattern in patterns:
        text = pattern.sub("", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    return text, {"inline_watermark_removed_chars": len(before) - len(text)}


def _spaced_caps_suspects(text: str) -> list[str]:
    """Localiza sequências em maiúsculas separadas por espaços para auditoria."""
    pattern = re.compile(r"\b(?:[A-Z]\s+){2,}[A-Z][A-Za-z]*\b")
    samples: list[str] = []
    for ln in text.splitlines():
        if pattern.search(ln):
            samples.append(ln.strip())
    return samples


def _fix_hyphen_linebreaks(text: str) -> tuple[str, dict]:
    """Reconstrói palavras separadas por hífen e quebra de linha."""
    pattern = re.compile(r"([A-Za-z]{1,24})-\s*\n\s*([A-Za-z]{1,24})")
    count = 0

    def _repl(match: re.Match[str]) -> str:
        """Reconstrói uma palavra separada por espaços durante a limpeza de OCR."""
        nonlocal count
        count += 1
        return f"{match.group(1)}-{match.group(2)}"

    fixed = pattern.sub(_repl, text)
    return fixed, {"hyphen_linebreak_fixes": count}


def _fix_ellipsis_spacing(text: str) -> tuple[str, dict]:
    """Corrige a falta de espaço após reticências somente em casos inequívocos."""
    unicode_ell = "\u2026"
    before = text
    ellipsis_hits = 0

    def _should_skip(start: int) -> bool:
        """Indica se a ocorrência pertence a uma região protegida."""
        tail = before[start : start + 5]
        return bool(re.match(r"[A-Z]{2,}", tail))

    def _repl_unicode(match: re.Match[str]) -> str:
        """Normaliza uma sequência de reticências representada em Unicode."""
        nonlocal ellipsis_hits
        idx = match.start(1)
        if _should_skip(idx):
            return match.group(0)
        ellipsis_hits += 1
        return f"{unicode_ell} {match.group(1)}"

    def _repl_dots(match: re.Match[str]) -> str:
        """Normaliza uma sequência de pontos usada como reticências."""
        nonlocal ellipsis_hits
        idx = match.start(1)
        if _should_skip(idx):
            return match.group(0)
        ellipsis_hits += 1
        return f"... {match.group(1)}"

    text = re.sub(r"\u2026([A-Za-z])", _repl_unicode, text)
    text = re.sub(r"\.\.\.([A-Za-z])", _repl_dots, text)
    # Remove espaços duplicados criados pelas correções de reticências.
    text = re.sub(r"\u2026\s{2,}(?=[A-Za-z])", unicode_ell + " ", text)
    text = re.sub(r"\.\.\.\s{2,}(?=[A-Za-z])", "... ", text)
    return text, {"ellipsis_spacing_fixes": ellipsis_hits}


def _is_toc_entry(line: str) -> bool:
    """Verifica se a linha tem a forma de uma entrada de sumário."""
    stripped = normalize_line_for_filters(line)
    if not stripped:
        return False
    norm = stripped.lower()
    if SECTION_PATTERN.fullmatch(stripped.lstrip("#").strip()):
        return True
    if any(marker in norm for marker in TOC_MARKER_LINES):
        return True
    if re.match(
        r"^(prologue|epilogue|afterword|chapter\s+\d+(:?[^\n]+)?)$",
        norm,
        flags=re.IGNORECASE,
    ):
        return True
    if re.match(
        r"^(pr[óo]logo|cap[ií]tulo\s+\d+|ep[ií]logo|p[oó]s-?escrito)",
        norm,
        flags=re.IGNORECASE,
    ):
        return True
    if len(stripped) < 80 and re.search(r"\b\d+\b", stripped):
        return True
    if len(stripped) < 120 and re.search(r"\.{2,}\s*\d{1,4}$", stripped):
        return True
    return False


def _is_heading_like(line: str) -> bool:
    """Verifica se a linha tem a forma de um título estrutural."""
    norm = normalize_line_for_filters(line).lower()
    if not norm:
        return False
    if SECTION_PATTERN.fullmatch(norm.lstrip("#").strip()):
        return True
    if TOC_MARKER_RE.match(norm):
        return True
    if re.match(r"^(prologue|epilogue|afterword|chapter\s+\d+)", norm):
        return True
    if re.match(r"^(pr[óo]logo|cap[ií]tulo\s+\d+|ep[ií]logo)", norm):
        return True
    return False


def _remove_toc_blocks(
    text: str,
    *,
    head_window: int = 200,
    tail_window: int = 400,
    head_min_entries: int = 3,
    tail_min_entries: int = 12,
    min_density: float = 0.35,
    gap_limit: int = 8,
) -> tuple[str, dict]:
    """Remove blocos de sumário sem descartar títulos da narrativa."""
    lines = text.splitlines()
    removed_lines: list[str] = []
    head_removed = 0
    tail_removed = 0

    def _is_marker(norm: str) -> bool:
        """Indica se marcador."""
        if norm == "contents":
            return True
        return norm in TOC_MARKER_LINES

    def _looks_narrative(norm_line: str) -> bool:
        """Estima se a linha contém prosa narrativa, e não metadados."""
        if not norm_line:
            return False
        if len(norm_line) >= 120:
            return True
        alpha = sum(1 for ch in norm_line if ch.isalpha())
        ratio = alpha / max(len(norm_line), 1)
        if len(norm_line) > 40 and ratio > 0.6 and bool(re.search(r"[.!?]", norm_line)):
            return True
        if len(norm_line) >= 20 and ratio > 0.7 and bool(re.search(r"[.!?]", norm_line)):
            return True
        return False

    def _strip_in_range(start: int, end: int, *, is_tail: bool = False) -> None:
        """Remove marcadores localizados dentro do intervalo informado."""
        nonlocal head_removed, tail_removed, lines
        idx = start
        while idx < end and idx < len(lines):
            current_norm = normalize_line_for_filters(lines[idx]).lower()
            if _is_marker(current_norm):
                j = idx + 1
                limit = min(len(lines), idx + 200)
                toc_like = 1  # current line
                total = 1
                gap = 0
                last_idx = idx
                block_invalid = False
                while j < limit:
                    total += 1
                    candidate_norm = normalize_line_for_filters(lines[j]).lower()
                    if _is_toc_entry(candidate_norm) or re.search(r"\s\d{1,4}$", candidate_norm):
                        toc_like += 1
                        gap = 0
                        last_idx = j
                    else:
                        if _looks_narrative(candidate_norm):
                            block_invalid = True
                            break
                        gap += 1
                        if gap >= gap_limit:
                            break
                    j += 1
                density = toc_like / total if total else 0
                min_entries = tail_min_entries if is_tail else head_min_entries
                if block_invalid and not (toc_like >= min_entries and density >= min_density):
                    removed_lines.append(current_norm)
                    del lines[idx]
                    end = min(end, len(lines))
                    continue
                if toc_like >= min_entries and density >= min_density:
                    removed_lines.extend(
                        normalize_line_for_filters(ln) for ln in lines[idx : last_idx + 1]
                    )
                    del lines[idx : last_idx + 1]
                    if is_tail:
                        tail_removed += 1
                    else:
                        head_removed += 1
                    end = min(end, len(lines))
                    continue
            idx += 1

    _strip_in_range(0, min(len(lines), head_window))
    tail_start = max(0, len(lines) - tail_window)
    _strip_in_range(tail_start, len(lines), is_tail=True)
    removed_hash = _sha256_text("\n".join(removed_lines)) if removed_lines else ""
    max_samples = 20
    removed_sample = removed_lines[:max_samples]
    stats = {
        "toc_blocks_removed_head": head_removed,
        "toc_blocks_removed_tail": tail_removed,
        "toc_blocks_removed_count": head_removed + tail_removed,
        "toc_lines_removed_count": len(removed_lines),
        "toc_removed_lines": removed_sample,
        "toc_removed_truncated": len(removed_lines) > max_samples,
        "toc_removed_hash": removed_hash,
    }
    return "\n".join(lines), stats


def _normalize_line_for_repeat(line: str) -> str:
    """Normaliza a linha para comparar cabeçalhos e rodapés repetidos."""
    norm = normalize_line_for_filters(line).lower()
    norm = re.sub(r"\s+", " ", norm)
    norm = re.sub(r"[.,;:!?\-–—]+", "", norm)
    return norm


def _remove_repeated_lines(
    text: str, *, min_freq: int = 6, max_len: int = 80
) -> tuple[str, dict, list[str]]:
    """Remove linhas repetidas sem afetar repetições literárias curtas."""
    lines = text.splitlines()
    freq: Counter[str] = Counter()
    normalized: dict[int, str] = {}
    for idx, ln in enumerate(lines):
        if _is_ellipsis_line(ln):
            # nunca conta linha de reticência como lixo repetido; mantemos diálogos de pausa
            normalized[idx] = ""
            continue
        if ln.lstrip().startswith(('"', "“", "'", "’", "-", "–", "—")):
            normalized[idx] = ""
            continue
        norm = _normalize_line_for_repeat(ln)
        normalized[idx] = norm
        if norm:
            freq[norm] += 1

    to_remove: set[int] = set()
    for idx, norm in normalized.items():
        if not norm:
            continue
        if re.fullmatch(r"\*{2,}", lines[idx].strip()):
            continue
        count = freq.get(norm, 0)
        if count >= min_freq and (len(norm) <= max_len or _is_promo_line(norm)):
            to_remove.add(idx)

    cleaned = [ln for idx, ln in enumerate(lines) if idx not in to_remove]
    removed_norms = [normalized[idx] for idx in sorted(to_remove) if normalized.get(idx)]
    top_repeated = freq.most_common(10)
    stats = {
        "repeated_lines_removed_count": len(to_remove),
        "top_repeated_lines": top_repeated,
    }
    return "\n".join(cleaned), stats, removed_norms


def _fix_ocr_spacing(text: str) -> tuple[str, dict]:
    """Corrige espaçamentos artificiais introduzidos pelo OCR."""
    trailing_punct_re = re.compile("([-\u2013\u2014.,;:!?\u2026'\"”’]+)$")

    def _split_token(token: str) -> tuple[str, str]:
        """Separa um token conforme as regras conservadoras de OCR."""
        match = trailing_punct_re.search(token)
        if not match:
            return token, ""
        start = match.start()
        return token[:start], token[start:]

    def _is_upperish(token: str) -> bool:
        """Verifica se o token representa uma sequência em maiúsculas."""
        core, _ = _split_token(token)
        clean_core = re.sub(r"[^A-Z]", "", core)
        return bool(clean_core) and core.isupper()

    def _fix_line(line: str) -> tuple[str, list[tuple[str, str]]]:
        """Corrige linha."""
        tokens = line.split()
        new_tokens: list[str] = []
        samples: list[tuple[str, str]] = []
        i = 0
        while i < len(tokens):
            if _is_upperish(tokens[i]):
                seq: list[tuple[str, str, str, str]] = []
                long_count = 0
                while i < len(tokens) and _is_upperish(tokens[i]):
                    tok = tokens[i]
                    core, punct = _split_token(tok)
                    clean_core = re.sub(r"[^A-Z]", "", core)
                    if (
                        len(clean_core) > 1
                        and long_count >= 1
                        and len(seq) >= 2
                        and len(clean_core) > 6
                    ):
                        break
                    if len(clean_core) > 1:
                        long_count += 1
                    seq.append((tok, core, punct, clean_core))
                    i += 1
                cores = [core for (_, core, _, _) in seq]
                cores_clean = [core_clean for (_, _, _, core_clean) in seq]
                puncts = [punct for (_, _, punct, _) in seq]
                trail = next((p for p in reversed(puncts) if p), "")
                combined_core = "".join(cores)
                combined_clean = "".join(cores_clean)
                combined = combined_core + trail
                allow_merge = (
                    len(seq) >= 2 and long_count <= 1 and re.search(r"[AEIOU]", combined_clean)
                )
                if any(("," in p) or ("." in p) or (";" in p) for p in puncts):
                    allow_merge = False
                # não junta se houver pontuação forte no meio e próximo token inicia com maiúscula (evita KUN? Is -> KUNIs)
                next_token = tokens[i] if i < len(tokens) else ""
                if any(p in combined for p in ("?", "!")) and next_token[:1].isupper():
                    allow_merge = False
                # evita CamelCase estranho quando todas as partes são words >1 (ex.: Mistress Anael)
                if len(seq) == 2 and all(len(c) > 1 for c in cores_clean):
                    allow_merge = False
                if len(seq) == 2 and cores_clean[0] == "I" and len(cores_clean[1]) > 3:
                    allow_merge = False
                if len(combined_clean) > 14:
                    allow_merge = False
                if len(seq) >= 3 and len(cores_clean[-1]) >= 8:
                    allow_merge = False
                if len(seq) >= 3 and len(cores_clean[-1]) >= 4:
                    allow_merge = False

                has_punct_next_upper = (
                    any(p in combined for p in ("?", "!")) and next_token[:1].isupper()
                )
                if allow_merge and not has_punct_next_upper:
                    new_tokens.append(combined)
                    samples.append((" ".join(tok for (tok, _, _, _) in seq), combined))
                else:
                    if len(seq) >= 3 and all(len(c) == 1 for c in cores_clean[:-1]):
                        merged_prefix = "".join(cores_clean[:-1])
                        new_tokens.append(merged_prefix)
                        new_tokens.append(seq[-1][0])
                    elif (
                        len(seq) >= 2
                        and len(cores_clean[0]) == 1
                        and cores_clean[0] != "I"
                        and 2 <= len(cores_clean[1]) <= 12
                        and (re.search(r"[AEIOU]", cores_clean[1]) or len(cores_clean[1]) <= 4)
                    ):
                        merged_prefix = cores[0] + cores[1] + (puncts[1] if len(puncts) > 1 else "")
                        new_tokens.append(merged_prefix)
                        if len(samples) < 10:
                            samples.append((f"{cores[0]} {cores[1]}", merged_prefix))
                        for tok, _, _, _ in seq[2:]:
                            new_tokens.append(tok)
                    else:
                        new_tokens.extend(tok for (tok, _, _, _) in seq)
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_line = " ".join(new_tokens)
        new_line = re.sub(r"([!?])([A-Za-z])", r"\\1 \\2", new_line)
        return new_line, samples

    fixed_lines: list[str] = []
    samples: list[tuple[str, str]] = []
    total_fixes = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or not re.search(r"[A-Z]\s+[A-Z]", stripped):
            fixed_lines.append(line)
            continue
        fixed, local_samples = _fix_line(line)
        total_fixes += len(local_samples)
        if local_samples and len(samples) < 10:
            remaining = 10 - len(samples)
            samples.extend(local_samples[:remaining])
        fixed_lines.append(fixed)
    return "\n".join(fixed_lines), {
        "ocr_spacing_fixes": total_fixes,
        "ocr_spacing_samples": samples,
    }


def _fix_spaced_caps_pairs(text: str) -> tuple[str, dict]:
    """
    Junta pares isolados de letras maiúsculas separados por espaço (ex.: W E -> WE).
    Conservador: exige vogal na palavra resultante ou tamanho pequeno.
    """
    pattern = re.compile(r"\b([A-Z])\s+([A-Z])\b")
    fixes = 0
    lines: list[str] = []
    for line in text.splitlines():
        new_line = line
        while True:
            match = pattern.search(new_line)
            if not match:
                break
            combined = f"{match.group(1)}{match.group(2)}"
            has_vowel = bool(re.search(r"[AEIOU]", combined))
            if has_vowel or len(combined) <= 3:
                new_line = new_line[: match.start()] + combined + new_line[match.end() :]
                fixes += 1
            else:
                break
        lines.append(new_line)
    return "\n".join(lines), {"spaced_caps_pair_fixes": fixes}


def _fix_mixed_caps(text: str) -> tuple[str, dict]:
    """
    Normaliza palavras com mistura estranha de maiúsculas/minúsculas (ex.: RuMORS).
    Converte para minúsculas preservando a inicial se estiver no começo da linha.
    Conservador: exige pelo menos 2 maiúsculas e 1 minúscula.
    """
    fixes = 0
    out_lines: list[str] = []
    for line in text.splitlines():
        tokens = line.split()
        new_tokens: list[str] = []
        for idx, tok in enumerate(tokens):
            letters = re.sub(r"[^A-Za-z]", "", tok)
            upp = sum(1 for c in letters if c.isupper())
            low = sum(1 for c in letters if c.islower())
            if any(ch in tok for ch in ("-", "–", "—", "'", "’")):
                new_tokens.append(tok)
                continue
            if "…" in tok or "..." in tok:
                new_tokens.append(tok)
                continue
            if upp >= 2 and low >= 1 and upp > low and not tok.isupper():
                base = tok.lower()
                if idx == 0:
                    base = base.capitalize()
                new_tokens.append(base)
                fixes += 1
            else:
                new_tokens.append(tok)
        out_lines.append(" ".join(new_tokens))
    return "\n".join(out_lines), {"mixed_caps_fixes": fixes}


def strip_front_matter(text: str) -> str:
    """Remove o conteúdo anterior ao primeiro marcador confiável da narrativa."""
    lines = text.splitlines()
    start_idx = None
    for idx, ln in enumerate(lines):
        candidate = ln.strip().lstrip("#").strip()
        if SECTION_PATTERN.fullmatch(candidate):
            start_idx = idx
            break
    if start_idx is None:
        return text
    return "\n".join(lines[start_idx:]).lstrip()


def strip_toc(
    text: str,
    logger: Optional[logging.Logger] = None,
    *,
    max_lines: int = 200,
    min_markers: int = 4,
    max_body_len: int = 50,
) -> str:
    """Remove o sumário inicial a partir de sequências de marcadores curtos."""
    lines = text.splitlines()
    search_lines = lines[:max_lines]
    marker_idxs: list[int] = []
    for idx, raw in enumerate(search_lines):
        normalized = raw.strip()
        if normalized.startswith("#"):
            normalized = normalized.lstrip("#").strip()
        if TOC_MARKER_RE.match(normalized):
            marker_idxs.append(idx)

    if len(marker_idxs) < min_markers:
        return text

    short_bodies = 0
    total_bodies = 0
    for i, start_idx in enumerate(marker_idxs):
        end_idx = marker_idxs[i + 1] if i + 1 < len(marker_idxs) else len(search_lines)
        body_text = "\n".join(search_lines[start_idx + 1 : end_idx]).strip()
        total_bodies += 1
        if len(body_text) < max_body_len:
            short_bodies += 1

    if not total_bodies or short_bodies / total_bodies < 0.6:
        return text

    cutoff_line = marker_idxs[-1] + 1
    # Avança além de linhas vazias ou numéricas logo após o último marcador (ex.: número de página do sumário).
    while cutoff_line < len(lines):
        candidate = lines[cutoff_line].strip()
        if not candidate:
            cutoff_line += 1
            continue
        if len(candidate) < max_body_len and not re.search(r"[A-Za-z]", candidate):
            cutoff_line += 1
            continue
        break
    remaining = lines[cutoff_line:]
    cleaned = "\n".join(remaining)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).lstrip()
    if logger:
        logger.info(
            "strip_toc: removido TOC inicial (markers=%d, short_bodies=%d/%d, cutoff_line=%d)",
            len(marker_idxs),
            short_bodies,
            total_bodies,
            cutoff_line,
        )
    return cleaned


def preprocess_text(
    raw_text: str,
    logger: Optional[logging.Logger] = None,
    *,
    skip_front_matter: bool = False,
    return_stats: bool = False,
    noise_glossary_path: str | Path | None = None,
) -> str | tuple[str, dict]:
    """
    Pré-processa o texto bruto extraído do PDF:
    - Normaliza quebras de linha
    - Remove rodapés, marcas d’água e blocos promocionais configurados
    - Remove front-matter/TOC quando skip_front_matter=True
    """

    stats: dict[str, int | dict] = {"chars_in": len(raw_text)}
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    stats["soft_hyphen_removed"] = text.count("\u00ad")
    text = text.replace("\u00ad", "")
    text = ZERO_WIDTH_RE.sub("", text.replace("\xa0", " "))
    text, sanitize_stats = sanitize_extracted_text(text, logger=logger)
    stats.update(sanitize_stats)
    glossary = _load_noise_glossary(noise_glossary_path)
    if skip_front_matter:
        text = strip_front_matter(text)
        text = strip_toc(text, logger=logger)

    footers_removed = 0
    footers_samples: list[str] = []
    footers_pattern_counts: Counter[str] = Counter()
    footer_matches_counter: Counter[str] = Counter()
    for pattern in FOOTER_PATTERNS:
        compiled = re.compile(pattern, flags=re.IGNORECASE)
        matches = compiled.findall(text)
        if matches:
            footers_removed += len(matches)
            footers_pattern_counts[pattern] += len(matches)
            for m in matches:
                sample = m if isinstance(m, str) else "".join(m)
                norm_s = normalize_line_for_filters(sample)
                if norm_s:
                    footer_matches_counter[norm_s] += 1
            # Mantém uma amostra pequena e representativa para o relatório.
            for m in matches[:10]:
                sample = m if isinstance(m, str) else "".join(m)
                norm_s = normalize_line_for_filters(sample)
                if norm_s:
                    footers_samples.append(norm_s)
        text = compiled.sub(" ", text)
    stats["footers_removed_count"] = footers_removed
    stats["footers_removed_samples"] = footers_samples[:10]
    if footers_pattern_counts:
        stats["footers_pattern_counts"] = dict(footers_pattern_counts)

    removed_counter: Counter[str] = Counter()
    removed_records: list[tuple[str, str, int]] = []

    text, promo_stats, promo_removed = _remove_promo_lines(text, glossary)
    stats.update(promo_stats)
    stats["promo_lines_removed_total"] = stats.get(
        "promo_lines_removed_count", stats.get("promo_lines_removed_total", 0)
    )
    removed_counter.update(promo_removed)
    removed_records.extend(
        (normalize_line_for_filters(item), "promo", 1) for item in promo_removed if item
    )

    # Registra remoções de footer de forma consistente (por texto removido).
    if "footer_matches_counter" in locals() and footer_matches_counter:
        removed_counter.update(footer_matches_counter)
        for txt_norm, cnt in footer_matches_counter.items():
            removed_records.append((txt_norm, "footer", int(cnt)))

    text, toc_stats = _remove_toc_blocks(text)
    stats.update(toc_stats)
    removed_counter.update(toc_stats.get("toc_removed_lines", []))
    removed_records.extend(
        (normalize_line_for_filters(item), "toc", 1)
        for item in toc_stats.get("toc_removed_lines", [])
        if item
    )

    text, hyphen_stats = _fix_hyphen_linebreaks(text)
    stats.update(hyphen_stats)

    text, ell_stats = _fix_ellipsis_spacing(text)
    stats.update(ell_stats)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()
    text, ocr_stats = _fix_ocr_spacing(text)
    stats.update(ocr_stats)
    text, spaced_pair_stats = _fix_spaced_caps_pairs(text)
    stats.update(spaced_pair_stats)
    text, mixed_caps_stats = _fix_mixed_caps(text)
    stats.update(mixed_caps_stats)

    text, under_merge_stats = _fix_under_merge(text)
    stats.update(under_merge_stats)

    text, noise_block_stats, noise_block_removed = _remove_noise_blocks_with_stats(text)
    stats.update(noise_block_stats)
    removed_counter.update(noise_block_removed)
    removed_records.extend(
        (normalize_line_for_filters(item), "noise_block", 1) for item in noise_block_removed if item
    )
    text, repeat_stats, repeat_removed = _remove_repeated_lines(text)
    stats.update(repeat_stats)
    removed_counter.update(repeat_removed)
    removed_records.extend(
        (normalize_line_for_filters(item), "repeated", 1) for item in repeat_removed if item
    )

    text, dedupe_stats, dedupe_removed = _dedupe_consecutive_lines(text)
    stats.update(dedupe_stats)
    removed_counter.update(dedupe_removed)
    removed_records.extend(
        (normalize_line_for_filters(item), "dedupe", 1) for item in dedupe_removed if item
    )

    text, gap_merge_stats = _merge_hard_wraps_across_gaps(text)
    stats.update(gap_merge_stats)

    text, reflow_stats = _reflow_paragraphs(text)
    stats.update(reflow_stats)

    text, subhead_stats = _ensure_subheading_isolated(text)
    stats.update(subhead_stats)

    text, dn_split_stats = _split_dialogue_narration_boundaries(text)
    stats.update(dn_split_stats)

    text, wrap_stats = _wrap_very_long_lines(text, max_len=800)
    stats.update(wrap_stats)

    text, upper_stats = _normalize_uppercase_sentences(text)
    stats.update(upper_stats)
    text, leading_caps_stats = _normalize_leading_small_caps(text)
    stats.update(leading_caps_stats)

    text, inline_stats = _strip_inline_watermarks(text, glossary)
    stats.update(inline_stats)
    text = _remove_blank_between_dialogue(text)
    text, quote_blank_lines_fixed = fix_blank_lines_inside_quotes(text)
    stats["quote_blank_lines_fixed"] = quote_blank_lines_fixed

    # Restaura heading de prólogo se ele existia no raw mas não sobrou após a limpeza.
    if (
        not skip_front_matter
        and "prologue" in normalize_line_for_filters(raw_text).lower()
        and "prologue" not in normalize_line_for_filters(text).lower()
    ):
        lines = text.splitlines()
        insert_at = 0
        for idx, ln in enumerate(lines):
            if ln.strip():
                insert_at = idx
                break
        lines.insert(insert_at, "Prologue")
        text = "\n".join(lines)

    # A validação usa os marcadores configurados, sem conhecer a origem do PDF.
    configured_markers = [
        token.lower()
        for token in glossary.get("line_contains", [])
        if isinstance(token, str) and token.strip()
    ]
    compact_markers = [
        re.sub(r"[^a-z0-9]", "", token.lower())
        for token in glossary.get("line_compact_contains", [])
        if isinstance(token, str) and token.strip()
    ]
    configured_watermarks_remaining = sum(
        1
        for line in text.splitlines()
        if any(token in normalize_line_for_filters(line).lower() for token in configured_markers)
        or any(
            token in re.sub(r"[^a-z0-9]", "", normalize_line_for_filters(line).lower())
            for token in compact_markers
        )
    )
    stats["watermarks_remaining"] = configured_watermarks_remaining
    stats["soft_hyphen_remaining"] = text.count("\u00ad")
    spaced_suspects = _spaced_caps_suspects(text)
    stats["spaced_caps_remaining"] = len(spaced_suspects)
    stats["spaced_caps_remaining_samples"] = spaced_suspects[:10]
    text = re.sub(r"\n{3,}", "\n\n", text)
    # primeira linha plausÌvel
    first_non_empty = next((ln for ln in text.splitlines() if ln.strip()), "")
    stats["first_line"] = first_non_empty

    counts_by_text: Counter[str] = Counter()
    for text_norm, _, count in removed_records:
        if text_norm:
            counts_by_text[text_norm] += count
    stats["removed_lines_occurrences_total"] = sum(counts_by_text.values())
    stats["removed_lines_unique_total"] = len(counts_by_text)
    stats["removed_lines_total"] = stats["removed_lines_occurrences_total"]
    stats["removed_lines_top"] = counts_by_text.most_common(10)
    # agregador leve de auditoria (top N) para removidos
    agg: dict[str, Counter[str]] = {}
    for text_norm, reason, count in removed_records:
        if not text_norm:
            continue
        agg.setdefault(text_norm, Counter())[reason] += count
    aggregated: list[dict[str, object]] = []
    for text_norm, reason_counts in agg.items():
        aggregated.append(
            {
                "text": text_norm,
                "count": sum(reason_counts.values()),
                "reasons": dict(reason_counts),
            }
        )
    aggregated.sort(key=lambda x: x["count"], reverse=True)
    max_items = 50
    stats["removed_aggregated"] = aggregated[:max_items]
    stats["removed_aggregated_truncated"] = len(aggregated) > max_items
    # lista completa (normalizada) para auditoria, com motivo por item
    stats["removed_full"] = [
        {"text": t, "reason": r, "count": c} for (t, r, c) in removed_records if t
    ]
    stats["removed_full_count"] = len(stats["removed_full"])
    stats["urls_remaining_count"] = len(URL_RE.findall(text))
    toc_terms = ("table of contents", "contents")
    stats["toc_remaining_count"] = sum(
        1 for ln in text.splitlines() if normalize_line_for_filters(ln).lower() in toc_terms
    )
    stats["chars_out"] = len(text)

    if logger is not None:
        logger.debug("Texto pré-processado: %d caracteres", len(text))

    if return_stats:
        return text, stats
    return text


def paragraphs_from_text(clean_text: str) -> List[str]:
    """Divide texto limpo em parágrafos usando quebras duplas."""
    return [p.strip() for p in clean_text.split("\n\n") if p.strip()]


_TRANSLATION_BOUNDARY_RE = re.compile(r"\n\n|[.!?](?:['\"”])?(?=\s|\n|$)")


def _quote_delta(text: str) -> int:
    """Retorna a diferença entre aberturas e fechamentos de aspas curvas."""
    return text.count("“") - text.count("”")


def _translation_chunk_end(text: str, start: int, max_chars: int, logger: logging.Logger) -> int:
    """Escolhe uma fronteira de chunk sem cortar uma fala quando possível.

    O PDF pode colocar diálogos longos no mesmo parágrafo. Cortar apenas no
    ponto final divide uma fala entre dois prompts, e o modelo costuma fechar a
    aspa no primeiro chunk. Procuramos então a próxima fronteira que preserve o
    mesmo estado de aspas do início do trecho, com um lookahead maior apenas
    nesse caso.
    """
    total_len = len(text)
    target_end = start + max_chars
    if target_end >= total_len:
        return total_len

    lookahead = 400
    hard_end = min(total_len, target_end + lookahead)
    window = text[start:hard_end]
    after_target: int | None = None
    before_target: int | None = None
    for match in _TRANSLATION_BOUNDARY_RE.finditer(window):
        end_pos = start + match.end()
        if target_end <= end_pos <= hard_end:
            after_target = end_pos
        elif end_pos < target_end:
            before_target = end_pos

    if after_target:
        default_end = after_target
        default_reason = "fim de frase após lookahead"
    elif before_target:
        default_end = before_target
        default_reason = "limite seguro antes do alvo"
    else:
        default_end = min(target_end, total_len)
        default_reason = "alvo"

    # Se o chunk anterior terminou dentro de uma fala, o balanço local pode
    # voltar a zero após fechar uma fala e abrir outra. Nessa situação, ainda
    # vale procurar uma fronteira que devolva o documento ao estado neutro;
    # caso contrário o modelo recebe dois diálogos partidos no mesmo prompt.
    start_quote_state = _quote_delta(text[:start])
    quote_hard_end = min(total_len, target_end + 1200)
    if start_quote_state:
        neutral_after: list[int] = []
        neutral_before: list[int] = []
        for match in _TRANSLATION_BOUNDARY_RE.finditer(text[start:quote_hard_end]):
            end_pos = start + match.end()
            if _quote_delta(text[:end_pos]) != 0:
                continue
            if end_pos >= target_end:
                neutral_after.append(end_pos)
            else:
                neutral_before.append(end_pos)

        if neutral_after:
            chunk_end = neutral_after[0]
            logger.debug(
                "tradução: chunk estendido para encerrar fala aberta (len=%d)",
                chunk_end - start,
            )
            return chunk_end
        if neutral_before:
            chunk_end = neutral_before[-1]
            logger.debug(
                "tradução: chunk antecipado para encerrar fala aberta (len=%d)",
                chunk_end - start,
            )
            return chunk_end

    if _quote_delta(text[start:default_end]) == 0:
        logger.debug(
            "tradução: chunk fechado em %s (len=%d)",
            default_reason,
            default_end - start,
        )
        return default_end

    # Permite uma extensão moderada somente para encerrar a fala aberta. Se não
    # houver fechamento próximo, volta à última fronteira segura antes do alvo.
    quote_window = text[start:quote_hard_end]
    balanced_after: list[int] = []
    balanced_before: list[int] = []
    for match in _TRANSLATION_BOUNDARY_RE.finditer(quote_window):
        end_pos = start + match.end()
        if _quote_delta(text[start:end_pos]) != 0:
            continue
        if end_pos >= target_end:
            balanced_after.append(end_pos)
        else:
            balanced_before.append(end_pos)

    if balanced_after:
        chunk_end = balanced_after[0]
        logger.debug(
            "tradução: chunk estendido para preservar fronteira de aspas (len=%d)",
            chunk_end - start,
        )
        return chunk_end
    if balanced_before:
        chunk_end = balanced_before[-1]
        logger.debug(
            "tradução: chunk antecipado para preservar fronteira de aspas (len=%d)",
            chunk_end - start,
        )
        return chunk_end

    logger.debug(
        "tradução: chunk sem fronteira de aspas segura; usando %s (len=%d)",
        default_reason,
        default_end - start,
    )
    return default_end


def chunk_for_translation(
    paragraphs: List[str], max_chars: int, logger: logging.Logger
) -> List[str]:
    """
    Chunk seguro para tradução com ajuste leve por fronteira de frase.

    Usa max_chars como alvo, mas permite pequeno lookahead para fechar
    o chunk no fim de frase (., ?, !) evitando cortar falas.
    """
    text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    total_len = len(text)
    consumed = 0

    while start < total_len:
        chunk_end = _translation_chunk_end(text, start, max_chars, logger)
        if chunk_end <= start:
            chunk_end = min(start + max_chars, total_len)

        raw_slice = text[start:chunk_end]
        chunks.append(raw_slice.strip())
        consumed += len(raw_slice)
        start = chunk_end

    if consumed != total_len:
        logger.warning(
            "tradução: soma dos chunks (%d) difere do texto original (%d)",
            consumed,
            total_len,
        )

    return chunks


def chunk_for_translation_with_offsets(
    paragraphs: List[str],
    max_chars: int,
    logger: logging.Logger,
) -> List[tuple[str, int | None, int | None]]:
    """Divide o texto para tradução e preserva os intervalos no original."""
    text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    if not text:
        return []

    chunks: List[tuple[str, int | None, int | None]] = []
    start = 0
    total_len = len(text)
    consumed = 0

    while start < total_len:
        chunk_end = _translation_chunk_end(text, start, max_chars, logger)
        if chunk_end <= start:
            chunk_end = min(start + max_chars, total_len)

        raw_slice = text[start:chunk_end]
        chunk_text = raw_slice.strip()
        leading = len(raw_slice) - len(raw_slice.lstrip())
        trailing = len(raw_slice) - len(raw_slice.rstrip())
        start_offset = start + leading if chunk_text else None
        end_offset = (start + len(raw_slice) - trailing) if chunk_text else None
        chunks.append((chunk_text, start_offset, end_offset))
        consumed += len(raw_slice)
        start = chunk_end

    if consumed != total_len:
        logger.warning(
            "tradução: soma dos chunks (%d) difere do texto original (%d)",
            consumed,
            total_len,
        )

    return chunks


def chunk_for_refine(paragraphs: List[str], max_chars: int, logger: logging.Logger) -> List[str]:
    """Divide o texto para refino com limite estrito de tamanho."""
    return chunk_by_paragraphs(paragraphs, max_chars=max_chars, logger=logger, label="refine")
