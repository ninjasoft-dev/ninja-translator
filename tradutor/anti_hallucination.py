"""Proteções contra alucinações aplicadas à tradução e ao refino."""

from __future__ import annotations

import re
from typing import List


def detect_language_anomaly(text: str, mode: str = "refine") -> bool:
    """Detecta idioma anomalia."""
    if not text:
        return True
    lower = text.lower()
    cjk_blocks = re.findall(r"[\u4e00-\u9fff]{6,}", text)
    if cjk_blocks:
        return True
    # Não use cognatos isolados como `esta` ou `porque`: ambos são válidos em
    # PT-BR e faziam o refinador descartar saídas saudáveis. Mantenha apenas
    # sinais estrangeiros inequívocos.
    french_es = ["mon ami", "bonjour", "ma ch", "très", "siempre"]
    if any(pat in lower for pat in french_es):
        return True
    if mode != "translate":
        english_words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
        if english_words:
            common_en = {
                "the",
                "and",
                "with",
                "from",
                "this",
                "that",
                "here",
                "there",
                "you",
                "your",
                "their",
            }
            en_hits = sum(1 for w in english_words if w.lower() in common_en)
            english_ratio = en_hits / max(len(english_words), 1)
            pt_markers = [
                " que ",
                " de ",
                " para ",
                " não",
                " uma ",
                " um ",
                " com ",
                " ao ",
                " na ",
                " no ",
            ]
            has_pt_markers = any(marker in f" {lower} " for marker in pt_markers)
            if english_ratio > 0.25 and not has_pt_markers:
                return True
    markers = [
        "as an ai",
        "here is the refined text",
        "<think>",
        "</think>",
        "assistant:",
        "user:",
        "como um modelo de linguagem",
    ]
    if any(pat in lower for pat in markers):
        return True
    return False


def detect_repetition_anomaly(text: str) -> bool:
    """Detecta repetição anomalia."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    counts = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    if any(c >= 3 for c in counts.values()):
        return True
    # Contagem bruta de palavras marca qualquer texto PT-BR normal como
    # anômalo: artigos e preposições aparecem dezenas de vezes num chunk.
    # Só considere repetição global quando um token relevante domina uma
    # fração expressiva de uma saída suficientemente longa.
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*", text.casefold())
    ignored = {
        "a",
        "o",
        "as",
        "os",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "e",
        "é",
        "em",
        "na",
        "no",
        "nas",
        "nos",
        "um",
        "uma",
        "que",
        "se",
        "por",
        "para",
        "com",
        "ao",
        "aos",
        "à",
        "às",
        "eu",
        "ela",
        "ele",
        "eles",
        "elas",
        "não",
        "mais",
    }
    relevant = [word for word in words if len(word) >= 5 and word not in ignored]
    if len(relevant) >= 40:
        counts: dict[str, int] = {}
        for word in relevant:
            counts[word] = counts.get(word, 0) + 1
        if max(counts.values(), default=0) >= 12 and max(counts.values()) / len(relevant) >= 0.12:
            return True
    return False


def detect_structure_anomaly(text: str) -> bool:
    """Detecta estrutura anomalia."""
    lower = text.lower()
    if (
        "### texto_traduzido_inicio".lower() in lower
        and "### texto_traduzido_fim".lower() not in lower
    ):
        return True
    if (
        "### texto_refinado_inicio".lower() in lower
        and "### texto_refinado_fim".lower() not in lower
    ):
        return True
    bad_markers = ["<think>", "assistant:", "user:", "===glossario_s", "```"]
    if any(bm in lower for bm in bad_markers):
        return True
    return False


def _extract_entities(text: str) -> List[str]:
    """Extrai entidades nomeadas simples para comparação entre textos."""
    return [m.group() for m in re.finditer(r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÄÖÜ][\wÁÉÍÓÚÂÊÔÃÕÄÖÜ-]{2,}\b", text)]


def detect_semantic_drift(orig: str, llm: str) -> bool:
    """Detecta indícios de desvio semântico entre origem e tradução."""
    orig_entities = _extract_entities(orig)
    llm_entities = _extract_entities(llm)
    if not orig_entities:
        return False
    shared = len(set(orig_entities) & set(llm_entities))
    if shared / max(len(orig_entities), 1) < 0.6:
        return True
    if len(llm_entities) > len(orig_entities) + 5:
        return True
    return False


def detect_entity_mutation(orig: str, candidate: str) -> bool:
    """Detecta deformações de nomes recorrentes, como `Mara` -> `Maramente`.

    O teste é deliberadamente estreito: só considera entidades que aparecem ao
    menos duas vezes no original e um token novo que conserva o mesmo prefixo
    de três letras, mas foi estendido pelo modelo.
    """
    source_counts: dict[str, int] = {}
    for entity in _extract_entities(orig):
        normalized = entity.casefold()
        source_counts[normalized] = source_counts.get(normalized, 0) + 1
    stable_entities = [
        entity for entity, count in source_counts.items() if count >= 2 and len(entity) >= 4
    ]
    if not stable_entities:
        return False

    for token in _extract_entities(candidate):
        normalized = token.casefold()
        if normalized in source_counts:
            continue
        for entity in stable_entities:
            if normalized.startswith(entity[:3]) and len(normalized) >= len(entity) + 2:
                return True
    return False


INLINE_SLASH_TOKEN_RE = re.compile(r"(?<![\w/])[A-Za-zÀ-ÿ]{1,}/[A-Za-zÀ-ÿ]{1,}(?![\w/])")


def detect_inline_slash_mutation(orig: str, candidate: str) -> bool:
    """Detecta tokens inventados como `do/a` que não existiam no original."""
    original_tokens = {match.group(0).casefold() for match in INLINE_SLASH_TOKEN_RE.finditer(orig)}
    return any(
        match.group(0).casefold() not in original_tokens
        for match in INLINE_SLASH_TOKEN_RE.finditer(candidate)
    )


def sanitize_llm_output(llm_raw: str) -> str:
    """Sanitiza llm saída."""
    cleaned = llm_raw
    cleaned = cleaned.replace("Here is the refined text:", "")
    cleaned = cleaned.replace("Texto refinado:", "")
    cleaned = cleaned.replace("Here is the text:", "")
    cleaned = re.sub(r"```.+?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\s{3,}", " ", cleaned)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def anti_hallucination_filter(orig: str, llm_raw: str, cleaned: str, mode: str) -> str:
    """Aplica as proteções e preserva a tradução anterior quando detecta uma anomalia."""
    # Na tradução, o fallback não devolve a origem porque isso recolocaria
    # conteúdo ainda não traduzido na saída; apenas sanitizamos.
    if mode == "translate":
        return sanitize_llm_output(cleaned)

    safe = sanitize_llm_output(cleaned)
    if not safe.strip():
        return orig
    if detect_structure_anomaly(llm_raw):
        return orig
    if detect_entity_mutation(orig, safe) or detect_inline_slash_mutation(orig, safe):
        return orig
    if detect_language_anomaly(safe, mode=mode):
        return orig
    if len(safe.strip()) < max(20, int(len(orig.strip()) * 0.2)):
        return orig
    if detect_repetition_anomaly(safe):
        return orig
    return safe
