"""Catálogo e detecção leve dos idiomas de origem aceitos pelo pipeline."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    """Descreve um idioma de origem e os nomes aceitos na configuração."""

    code: str
    name_pt: str
    aliases: tuple[str, ...] = ()


LANGUAGES: tuple[LanguageSpec, ...] = (
    LanguageSpec("en", "inglês", ("english", "ingles")),
    LanguageSpec("ja", "japonês", ("japanese", "japones", "jp")),
    LanguageSpec("ko", "coreano", ("korean", "kr")),
    LanguageSpec("zh", "chinês", ("chinese", "chines", "mandarin", "cn", "zh-cn", "zh-tw")),
    LanguageSpec("es", "espanhol", ("spanish", "espanol")),
    LanguageSpec("fr", "francês", ("french", "frances")),
    LanguageSpec("de", "alemão", ("german", "alemao")),
    LanguageSpec("it", "italiano", ("italian",)),
    LanguageSpec("ru", "russo", ("russian",)),
    LanguageSpec("ar", "árabe", ("arabic", "arabe")),
    LanguageSpec("pl", "polonês", ("polish", "polones")),
    LanguageSpec("nl", "holandês", ("dutch", "holandes")),
    LanguageSpec("tr", "turco", ("turkish",)),
    LanguageSpec("id", "indonésio", ("indonesian", "indonesio")),
    LanguageSpec("vi", "vietnamita", ("vietnamese",)),
    LanguageSpec("th", "tailandês", ("thai", "tailandes")),
)
SUPPORTED_SOURCE_LANGUAGE_CODES = ("auto", *(language.code for language in LANGUAGES))

_LANGUAGE_BY_CODE = {language.code: language for language in LANGUAGES}
_LANGUAGE_BY_ALIAS = {
    alias.casefold(): language.code
    for language in LANGUAGES
    for alias in (language.code, language.name_pt, *language.aliases)
}

_LATIN_LANGUAGE_MARKERS = {
    "en": frozenset({"the", "and", "that", "with", "from", "this", "was", "were", "have"}),
    "es": frozenset({"el", "los", "las", "una", "que", "con", "para", "pero", "como", "estaba"}),
    "fr": frozenset({"le", "les", "une", "des", "que", "avec", "pour", "mais", "dans", "était"}),
    "de": frozenset({"der", "die", "das", "und", "mit", "für", "nicht", "aber", "war", "ist"}),
    "it": frozenset({"il", "gli", "una", "che", "con", "per", "non", "ma", "era", "sono"}),
    "pl": frozenset({"się", "nie", "jest", "jak", "ale", "dla", "przez", "jego", "jej", "był"}),
    "nl": frozenset({"het", "een", "van", "met", "voor", "niet", "maar", "was", "zijn", "dat"}),
    "tr": frozenset({"bir", "ve", "ile", "için", "değil", "ama", "bu", "olan", "olarak", "daha"}),
    "id": frozenset(
        {"yang", "dan", "dengan", "untuk", "tidak", "tetapi", "ini", "itu", "dari", "ada"}
    ),
    "vi": frozenset({"và", "của", "một", "không", "nhưng", "cho", "với", "đã", "được", "trong"}),
}


def normalize_source_language(value: str | None) -> str:
    """Normaliza código ou nome de idioma e rejeita opções desconhecidas."""
    normalized = (value or "auto").strip().casefold().replace("_", "-")
    if normalized == "auto":
        return normalized
    code = _LANGUAGE_BY_ALIAS.get(normalized)
    if code:
        return code
    supported = ", ".join(SUPPORTED_SOURCE_LANGUAGE_CODES)
    raise ValueError(f"Idioma de origem não suportado: {value!r}. Opções: {supported}.")


def source_language_name(code: str) -> str:
    """Retorna o nome em português usado em logs e prompts."""
    normalized = normalize_source_language(code)
    if normalized == "auto":
        return "detecção automática"
    return _LANGUAGE_BY_CODE[normalized].name_pt


def detect_source_language(text: str, configured_language: str = "auto") -> str:
    """Resolve o idioma configurado ou estima o idioma predominante do texto.

    A heurística cobre alfabetos de identificação segura e palavras funcionais
    frequentes. Em textos curtos ou ambíguos, ``--source-language`` é a opção
    recomendada.
    """
    configured = normalize_source_language(configured_language)
    if configured != "auto":
        return configured

    sample = text[:50_000]
    if re.search(r"[ぁ-ゖァ-ヺ]", sample):
        return "ja"
    if re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", sample):
        return "ko"
    if re.search(r"[฀-๿]", sample):
        return "th"
    if re.search(r"[؀-ۿ]", sample):
        return "ar"
    if re.search(r"[Ѐ-ӿ]", sample):
        return "ru"
    if re.search(r"[㐀-䶿一-鿿]", sample):
        return "zh"

    words = re.findall(r"[^\W\d_]+", sample.casefold(), flags=re.UNICODE)
    scores = {
        code: sum(word in markers for word in words)
        for code, markers in _LATIN_LANGUAGE_MARKERS.items()
    }
    best_code, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score >= 2:
        return best_code

    # Remove acentos apenas para reconhecer o caso legado mais comum sem uma
    # dependência pesada de detecção estatística.
    ascii_sample = "".join(
        char for char in unicodedata.normalize("NFKD", sample) if not unicodedata.combining(char)
    ).casefold()
    if re.search(r"\b(?:the|and|with|from|that)\b", ascii_sample):
        return "en"
    return "en"


_UNSEGMENTED_SCRIPT_RE = re.compile(
    r"[ぁ-ゖァ-ヺ\u3400-\u4dbf\u4e00-\u9fff가-힣ㄱ-ㅎㅏ-ㅣ\u0e00-\u0e7f]"
)


def compile_term_pattern(term: str, *, case_sensitive: bool = False) -> re.Pattern[str]:
    """Compila uma busca adequada a idiomas com ou sem separação por espaços."""
    normalized = re.sub(r"\s+", " ", term.strip())
    flags = 0 if case_sensitive else re.IGNORECASE
    escaped = re.escape(normalized)
    if _UNSEGMENTED_SCRIPT_RE.search(normalized):
        return re.compile(escaped, flags)
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags)
