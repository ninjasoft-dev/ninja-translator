"""Normalização conservadora de títulos, capítulos e marcadores de cena."""

from __future__ import annotations

import re

_SIMPLE_CHAPTER_RE = re.compile(
    r"^(?:#\s*)?cap[ií]tulo\s+(?P<number>\d+):?\s*$",
    re.IGNORECASE,
)
_MARKDOWN_CHAPTER_WITH_SUBTITLE_RE = re.compile(
    r"^#\s*cap[ií]tulo\s+(?P<number>\d+):\s+(?P<subtitle>\S.+?)\s*$",
    re.IGNORECASE,
)
_MARKDOWN_SIMPLE_CHAPTER_RE = re.compile(
    r"^#\s*cap[ií]tulo\s+(?P<number>\d+):?\s*$",
    re.IGNORECASE,
)
_TIME_LABEL_RE = re.compile(
    r"^(?P<name>[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
    r"(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+){1,3})\s+"
    r"(?P<label>ALGUM TEMPO ANTES(?:,\s*há um tempo)?…?)$"
)
_LEADING_CAPS_RE = re.compile(
    r"^(?P<lead>DEPOIS QUE|APÓS|QUANDO|ENQUANTO)\s+"
    r"(?P<name>[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ'’-]{2,})"
    r"(?P<tail>\s+.*[a-zà-öø-ÿ].*)$"
)


def _normalize_leading_small_caps(line: str) -> str:
    """Restaura a caixa de uma abertura temporal extraída em versalete."""
    match = _LEADING_CAPS_RE.match(line)
    if not match:
        return line
    lead = match.group("lead").lower().capitalize()
    name = match.group("name").lower().capitalize()
    return f"{lead} {name}{match.group('tail')}"


def normalize_structure(text: str) -> str:
    """Recupera estruturas inequívocas sem reorganizar a narrativa."""
    lines = text.splitlines()
    normalized: list[str] = []
    heading_pattern = re.compile(
        r"^(?P<head>(?:pr[oó]logo|cap[ií]tulo\s+[^\s].*?|"
        r"ep[ií]logo|interl[úu]dio))(?P<rest>.*)$",
        re.IGNORECASE,
    )

    def add_blank_line() -> None:
        """Acrescenta uma única linha vazia ao resultado."""
        if not normalized or normalized[-1] != "":
            normalized.append("")

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            add_blank_line()
            index += 1
            continue

        titled_chapter = _MARKDOWN_CHAPTER_WITH_SUBTITLE_RE.match(stripped)
        if titled_chapter:
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            duplicate = (
                _MARKDOWN_SIMPLE_CHAPTER_RE.match(lines[next_index].strip())
                if next_index < len(lines)
                else None
            )
            if duplicate and duplicate.group("number") == titled_chapter.group("number"):
                normalized.append(
                    f"# Capítulo {titled_chapter.group('number')}: "
                    f"{titled_chapter.group('subtitle')}"
                )
                add_blank_line()
                index = next_index + 1
                continue

        stripped = _normalize_leading_small_caps(stripped)

        if stripped.startswith("***") and stripped != "***":
            normalized.append("***")
            add_blank_line()
            normalized.append(stripped[3:].strip())
            add_blank_line()
            index += 1
            continue

        chapter_match = _SIMPLE_CHAPTER_RE.match(stripped)
        if chapter_match:
            subtitle, subtitle_index = _next_chapter_subtitle(lines, index + 1)
            chapter_number = chapter_match.group("number")
            if subtitle:
                normalized.append(f"# Capítulo {chapter_number}: {subtitle}")
                add_blank_line()
                index = subtitle_index + 1
                continue
            normalized.append(f"# Capítulo {chapter_number}:")
            add_blank_line()
            index += 1
            continue

        time_label = _split_character_time_label(stripped)
        if time_label:
            name, label = time_label
            normalized.append(f"## {name}")
            add_blank_line()
            normalized.append(label)
            add_blank_line()
            index += 1
            continue

        heading_match = heading_pattern.match(stripped.rstrip(":"))
        if heading_match:
            normalized.append(heading_match.group("head").strip())
            add_blank_line()
            remaining_text = heading_match.group("rest").strip()
            if remaining_text:
                normalized.append(remaining_text)
                add_blank_line()
            index += 1
            continue

        normalized.append(stripped)
        index += 1

    cleaned: list[str] = []
    previous_was_blank = False
    for line in normalized:
        is_blank = line == ""
        if is_blank and previous_was_blank:
            continue
        cleaned.append(line)
        previous_was_blank = is_blank

    return "\n".join(cleaned).strip()


def _next_chapter_subtitle(lines: list[str], start: int) -> tuple[str | None, int]:
    """Localiza um subtítulo curto logo após o cabeçalho de capítulo."""
    index = start
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return None, index

    candidate = lines[index].strip()
    if not _looks_like_chapter_subtitle(candidate):
        return None, index

    next_index = index + 1
    while next_index < len(lines) and not lines[next_index].strip():
        next_index += 1
    if next_index >= len(lines):
        return None, index
    return candidate, index


def _looks_like_chapter_subtitle(line: str) -> bool:
    """Indica se a linha tem formato conservador de subtítulo."""
    if not line or line.startswith(("#", '"', "“", "—")):
        return False
    if line.endswith((".", "!", "?")) or line.isupper():
        return False
    words = line.split()
    return (
        2 <= len(words) <= 8
        and len(line) <= 90
        and bool(re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9'’,:;—\- ]+", line))
        and line[:1].isupper()
    )


def _split_character_time_label(line: str) -> tuple[str, str] | None:
    """Separa o nome do personagem de um marcador temporal colado."""
    match = _TIME_LABEL_RE.match(line)
    if not match:
        return None
    return match.group("name"), "Algum tempo antes…"
