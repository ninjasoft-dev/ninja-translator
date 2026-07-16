"""Limpeza estrutural determinística anterior ao refino."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def _normalize_spaces(text: str) -> str:
    """Normaliza espaços horizontais sem alterar parágrafos."""
    return " ".join(text.split())


def _normalize_for_dupe(text: str) -> str:
    """Normaliza texto para comparacao aproximada."""
    cleaned = _normalize_spaces(text.strip())
    cleaned = cleaned.strip('"\u201c\u201d\u2018\u2019')
    cleaned = re.sub(r"[\u2019\u2018\u00b4\u0060]", "'", cleaned)
    cleaned = re.sub(r"[\u201c\u201d]", '"', cleaned)
    cleaned = re.sub(r"\.{3,}", "...", cleaned)
    return cleaned.lower()


def _is_near_duplicate(a: str | None, b: str | None, threshold: float = 0.96) -> bool:
    """Checagem forte de duplicidade (quase identico)."""
    if not a or not b:
        return False
    if a == b:
        return True
    if abs(len(a) - len(b)) > max(8, int(max(len(a), len(b)) * 0.2)):
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _is_fuzzy_duplicate(a: str | None, b: str | None, threshold: float = 0.9) -> bool:
    """Verifica duplicidade aproximada por contenção ou pequena variação textual."""
    if not a or not b:
        return False
    if a == b:
        return True
    ratio = SequenceMatcher(None, a, b).ratio()
    if ratio >= threshold:
        return True
    len_ratio = min(len(a), len(b)) / max(len(a), len(b))
    if len_ratio >= 0.75 and (a in b or b in a):
        return True
    return False


def dedupe_adjacent_lines(text: str) -> tuple[str, dict]:
    """Remove linhas e parágrafos consecutivos idênticos ou quase idênticos."""

    def _is_protected_short(line: str) -> bool:
        """Indica se protegidos curtos."""
        stripped = line.strip()
        if len(stripped) > 25:
            return False
        alnum = sum(1 for ch in stripped if ch.isalnum())
        alnum_ratio = (alnum / len(stripped)) if stripped else 1
        single_word = len(stripped.split()) == 1
        starts_dash = stripped.startswith(("—", "-"))
        punct_heavy = alnum_ratio < 0.6
        return single_word or starts_dash or punct_heavy

    lines = text.splitlines()
    deduped: list[str] = []
    prev_norm: str | None = None
    prev_short = False
    run_len = 0
    lines_removed = 0
    for ln in lines:
        norm = _normalize_for_dupe(ln)
        is_short_protected = _is_protected_short(ln)
        if norm and _is_fuzzy_duplicate(prev_norm, norm, threshold=0.94):
            run_len = run_len + 1 if prev_short == is_short_protected else 2
            prev_short = is_short_protected
            # Em falas curtas e onomatopeias, só remove repetições claramente acidentais.
            if is_short_protected and run_len < 4:
                deduped.append(ln)
                continue
            # Mantém a versão mais completa quando uma linha é prefixo da seguinte.
            if deduped and len(_normalize_spaces(deduped[-1])) < len(_normalize_spaces(ln)):
                deduped[-1] = ln
            lines_removed += 1
            prev_norm = norm if norm else None
            continue
        deduped.append(ln)
        prev_norm = norm if norm else None
        prev_short = is_short_protected
        run_len = 1

    paragraphs = "\n".join(deduped).split("\n\n")
    final_paragraphs: list[str] = []
    prev_para_norm: str | None = None
    blocks_removed = 0
    for para in paragraphs:
        norm_para = _normalize_for_dupe(para)
        if norm_para and _is_fuzzy_duplicate(prev_para_norm, norm_para, threshold=0.9):
            # Mantem o mais completo (mais longo) entre os dois
            if final_paragraphs and len(_normalize_spaces(final_paragraphs[-1])) < len(
                _normalize_spaces(para)
            ):
                final_paragraphs[-1] = para
            blocks_removed += 1
            continue
        final_paragraphs.append(para)
        prev_para_norm = norm_para if norm_para else None

    return "\n\n".join(final_paragraphs), {
        "lines_removed": lines_removed,
        "blocks_removed": blocks_removed,
    }


_GLUED_PATTERNS = (
    re.compile(r"([.!?…][\"\u201d])\s+(?=[\"\u201c])"),
    re.compile(r"([.!?…])\s+(?=\u2014\s*)"),
)


def fix_glued_dialogues(text: str) -> tuple[str, dict]:
    """
    Insere quebras conservadoras quando duas falas estao coladas.
    """
    fixed_lines: list[str] = []
    breaks_inserted = 0
    for ln in text.splitlines():
        if ln.lstrip().startswith("#"):
            fixed_lines.append(ln)
            continue
        new_line = ln
        for pattern in _GLUED_PATTERNS:
            new_line, count = pattern.subn(r"\1\n", new_line)
            breaks_inserted += count
        fixed_lines.append(new_line)
    return "\n".join(fixed_lines), {"breaks_inserted": breaks_inserted}


def dedupe_prefix_lines(text: str) -> tuple[str, dict]:
    """Remove uma linha truncada quando a seguinte contém sua versão completa."""
    lines = text.splitlines()
    cleaned: list[str] = []
    removed = 0
    blocked = 0

    def _ends_open(ln: str) -> bool:
        """Indica se termina com aberto."""
        return not re.search(r"[.!?.:;]['\")\]]?\s*$", ln)

    idx = 0
    total = len(lines)
    while idx < total:
        current = lines[idx]
        nxt = lines[idx + 1] if idx + 1 < total else None
        if nxt is not None:
            cur_strip = current.strip()
            nxt_strip = nxt.strip()
            cur_norm = _normalize_spaces(cur_strip)
            nxt_norm = _normalize_spaces(nxt_strip)
            if nxt_strip.startswith("“") or re.fullmatch(r"\d+", cur_strip):
                blocked += 1
            elif (
                cur_norm
                and nxt_norm
                and len(nxt_norm) > len(cur_norm)
                and nxt_norm.startswith(cur_norm)
                and _ends_open(cur_norm)
            ):
                removed += 1
                idx += 1
                continue
        cleaned.append(current)
        idx += 1

    return "\n".join(cleaned), {
        "prefix_lines_removed": removed,
        "prefix_merges_blocked": blocked,
    }


def _split_fragments(para: str) -> list[str]:
    """Divide o parágrafo em fragmentos curtos para comparar repetições internas."""
    if not para.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[\"“\-\w])", para.strip())
    return [p.strip() for p in parts if p.strip()]


def _is_short_fragment(fragment: str) -> bool:
    """Indica se curtos fragmento."""
    stripped = fragment.strip()
    if not stripped:
        return False
    if len(stripped) > 25:
        return False
    words = stripped.replace("—", "").split()
    return len(words) <= 2


def dedupe_adjacent_fragments(text: str) -> tuple[str, dict]:
    """Remove sequências repetidas de frases ou falas no mesmo parágrafo."""
    paragraphs = text.split("\n\n")
    removed = 0
    cleaned_paras: list[str] = []
    for para in paragraphs:
        if "\n" in para:
            cleaned_paras.append(para.strip())
            continue
        frags = _split_fragments(para)
        if not frags:
            cleaned_paras.append(para.strip())
            continue
        # Não deduplicar parágrafos compostos só por falas/onomatopeias curtas
        if all(len(f.strip()) <= 25 and len(f.split()) <= 2 for f in frags):
            cleaned_paras.append(" ".join(frags).strip())
            continue
        filtered: list[str] = []
        idx = 0
        total = len(frags)
        while idx < total:
            removed_block = False
            # tenta remover blocos repetidos de tamanho 3,2,1
            for k in (3, 2, 1):
                if idx + k <= total and len(filtered) >= k:
                    prev_block = _normalize_for_dupe(" ".join(filtered[-k:]))
                    next_block = _normalize_for_dupe(" ".join(frags[idx : idx + k]))
                    if _is_fuzzy_duplicate(prev_block, next_block, threshold=0.9):
                        if k < 3 and all(_is_short_fragment(f) for f in frags[idx : idx + k]):
                            continue
                        removed += k
                        idx += k
                        removed_block = True
                        break
            if removed_block:
                continue
            filtered.append(frags[idx])
            idx += 1
        cleaned_paras.append(" ".join(filtered).strip())

    return "\n\n".join([p for p in cleaned_paras if p != ""]), {"fragments_removed": removed}


def cleanup_before_refine(md: str) -> tuple[str, dict]:
    """Executa as correções determinísticas anteriores ao refino."""
    prefix_cleaned, stats_prefix = dedupe_prefix_lines(md)
    deduped, stats_dedupe = dedupe_adjacent_lines(prefix_cleaned)
    fixed, stats_glued = fix_glued_dialogues(deduped)
    frag_cleaned, stats_frag = dedupe_adjacent_fragments(fixed)
    combined = {
        "lines_removed": stats_dedupe.get("lines_removed", 0),
        "blocks_removed": stats_dedupe.get("blocks_removed", 0),
        "breaks_inserted": stats_glued.get("breaks_inserted", 0),
        "prefix_lines_removed": stats_prefix.get("prefix_lines_removed", 0),
        "prefix_merges_blocked": stats_prefix.get("prefix_merges_blocked", 0),
        "fragments_removed": stats_frag.get("fragments_removed", 0),
    }
    return frag_cleaned, combined


def detect_obvious_dupes(md: str) -> bool:
    """
    Heuristica leve para detectar duplicacoes adjacentes.
    Dispara se encontrar pelo menos uma repeticao consecutiva.
    """
    lines = [ln for ln in md.splitlines() if _normalize_spaces(ln)]
    prev: str | None = None
    for ln in lines:
        norm = _normalize_for_dupe(ln)
        if _is_near_duplicate(prev, norm):
            return True
        prev = norm
    return False


def detect_glued_dialogues(md: str) -> bool:
    """
    Heuristica leve para detectar falas coladas.
    """
    return any(pattern.search(md) for pattern in _GLUED_PATTERNS)
