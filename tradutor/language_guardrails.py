from __future__ import annotations

import re

from .languages import normalize_source_language

ENGLISH_LEAK_WORDS = frozenset(
    """
    about after again against all also although always among another around asked
    because before behind believe between brought called calmly came choose choosing
    could desire directly does doesn't doing done enough every everyone everything
    felt from gave gives going gone have haven't having however into itself knew
    know knowing made make makes might must neither never nothing once only other
    perhaps replied said says should something still strongly such than that their
    theirs them themselves then there these they they're this those though through
    told toward under until upon wanted wasn't were weren't what whatever when
    where whether which while who whom whose with within without would wouldn't
    your you're yours
    """.split()
)
PORTUGUESE_ANCHOR_WORDS = frozenset(
    """
    agora ainda alguém algum alguma alguns algumas antes aqui assim até caso com
    como da das de dela dele deles depois do dos ela elas ele eles em então era
    essa esse isso mais mas mesmo minha muito na não nas nem no nos para pela pelo
    por porque quando que quem sem seu sua talvez também tinha uma você vocês
    """.split()
)
ENGLISH_CONTRACTION_RE = re.compile(
    r"\b(?:i['’]m|i['’]ll|i['’]ve|you['’]re|you['’]ll|they['’]re|we['’]re|"
    r"it['’]s|that['’]s|can['’]t|won['’]t|don['’]t|doesn['’]t|didn['’]t|"
    r"wouldn['’]t|shouldn['’]t|couldn['’]t|haven['’]t|weren['’]t|wasn['’]t)\b",
    re.IGNORECASE,
)
ENGLISH_POSSESSIVE_RE = re.compile(r"\b[A-Z][A-Za-z]+['’]s\b")
# Formas plurais inglesas que não têm uso natural em PT-BR. Mantemos a lista
# curta para não sinalizar empréstimos técnicos legítimos ou nomes próprios.
SINGLE_TOKEN_ENGLISH_LEAKS = frozenset(
    {
        "af",
        "arright",
        "boost",
        "buff",
        "buffs",
        "kys",
        "selves",
        "they",
        "though",
        "uh",
        "uhh",
    }
)
SHORT_ENGLISH_LEAKS = frozenset({"i see"})
MIXED_ENGLISH_ARTIFACT_RE = re.compile(
    r"(?<![A-Za-zÀ-ÿ])(?:I(?:-(?=[a-zà-ÿ])|\s+(?=[a-zà-ÿ]))|Y-you\b)",
    re.IGNORECASE,
)


def english_leak_segments(text: str) -> list[str]:
    """Detecta segmentos longos que ainda parecem estar em ingles."""
    if not text:
        return []
    flagged: list[str] = []
    single_token_match = re.search(
        rf"\b(?:{'|'.join(re.escape(word) for word in sorted(SINGLE_TOKEN_ENGLISH_LEAKS))})\b",
        text,
        flags=re.IGNORECASE,
    )
    if single_token_match:
        return [single_token_match.group(0)]
    short_phrase_match = re.search(
        rf"\b(?:{'|'.join(re.escape(phrase) for phrase in sorted(SHORT_ENGLISH_LEAKS))})\b",
        text,
        flags=re.IGNORECASE,
    )
    if short_phrase_match:
        return [short_phrase_match.group(0)]
    mixed_artifact_match = MIXED_ENGLISH_ARTIFACT_RE.search(text)
    if mixed_artifact_match:
        return [mixed_artifact_match.group(0)]
    segments = re.split(r"(?:\n+|(?<=[.!?])\s+)", text)
    for raw_segment in segments:
        segment = raw_segment.strip()
        if len(segment) < 35:
            continue
        words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", segment)
        if len(words) < 6:
            continue
        normalized = [word.lower().replace("’", "'") for word in words]
        english_hits = sum(1 for word in normalized if word in ENGLISH_LEAK_WORDS)
        english_hits += len(ENGLISH_CONTRACTION_RE.findall(segment)) * 2
        english_hits += len(ENGLISH_POSSESSIVE_RE.findall(segment))
        portuguese_hits = sum(1 for word in normalized if word in PORTUGUESE_ANCHOR_WORDS)
        dense_english = (
            len(words) >= 10 and english_hits >= 5 and english_hits / max(len(words), 1) >= 0.25
        )
        dominant_english = english_hits >= 4 and english_hits >= portuguese_hits + 2
        if dense_english or dominant_english:
            flagged.append(segment)
    return flagged


def detect_residual_english(text: str) -> tuple[bool, str]:
    """Detecta palavras e trechos em inglês que restaram na tradução."""
    segments = english_leak_segments(text)
    if not segments:
        return False, ""
    preview = re.sub(r"\s+", " ", segments[0]).strip()
    if len(preview) > 80:
        preview = preview[:77].rstrip() + "..."
    return True, f"residual_english:{preview}"


_SOURCE_SCRIPT_PATTERNS = {
    "ja": re.compile(r"(?:[ぁ-ゖァ-ヺ]+|[一-龯々〆ヵヶ]{2,})"),
    "ko": re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]{2,}"),
    "zh": re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}"),
    "ru": re.compile(r"[\u0400-\u04ff]{3,}"),
    "ar": re.compile(r"[\u0600-\u06ff]{3,}"),
    "th": re.compile(r"[\u0e00-\u0e7f]{3,}"),
}
_SOURCE_LANGUAGE_WORDS = {
    "es": frozenset({"el", "los", "las", "una", "con", "para", "pero", "estaba"}),
    "fr": frozenset({"le", "les", "une", "des", "avec", "pour", "mais", "était"}),
    "de": frozenset({"der", "die", "das", "und", "mit", "für", "nicht", "aber"}),
    "it": frozenset({"il", "gli", "che", "con", "per", "non", "sono", "della"}),
    "pl": frozenset({"się", "nie", "jest", "ale", "przez", "jego", "był", "była"}),
    "nl": frozenset({"het", "een", "van", "met", "voor", "niet", "maar", "zijn"}),
    "tr": frozenset({"bir", "ve", "ile", "için", "değil", "ama", "olan", "daha"}),
    "id": frozenset({"yang", "dan", "dengan", "untuk", "tidak", "tetapi", "dari", "ada"}),
    "vi": frozenset({"và", "của", "một", "không", "nhưng", "cho", "với", "được"}),
}


def source_leak_segments(text: str, source_language: str) -> list[str]:
    """Localiza trechos que ainda parecem estar no idioma de origem."""
    language = normalize_source_language(source_language)
    if not text or language == "auto":
        return []
    if language == "en":
        return english_leak_segments(text)

    script_pattern = _SOURCE_SCRIPT_PATTERNS.get(language)
    if script_pattern:
        return [match.group(0) for match in script_pattern.finditer(text)]

    markers = _SOURCE_LANGUAGE_WORDS.get(language)
    if not markers:
        return []
    flagged: list[str] = []
    for raw_segment in re.split(r"(?:\n+|(?<=[.!?])\s+)", text):
        words = re.findall(r"[^\W\d_]+", raw_segment.casefold(), flags=re.UNICODE)
        if sum(word in markers for word in words) >= 2:
            flagged.append(raw_segment.strip())
    return flagged


def detect_residual_source_language(text: str, source_language: str) -> tuple[bool, str]:
    """Informa se a saída ainda contém material provável do idioma de origem."""
    language = normalize_source_language(source_language)
    if language == "en":
        return detect_residual_english(text)
    segments = source_leak_segments(text, language)
    if not segments:
        return False, ""
    preview = re.sub(r"\s+", " ", segments[0]).strip()
    if len(preview) > 80:
        preview = preview[:77].rstrip() + "..."
    return True, f"residual_source_language:{language}:{preview}"


def residual_issue_type(source_language: str) -> str:
    """Mantém o identificador legado de inglês nos relatórios existentes."""
    language = normalize_source_language(source_language)
    return "residual_english" if language == "en" else "residual_source_language"
