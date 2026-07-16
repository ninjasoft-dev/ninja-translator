"""Edição determinística opcional aplicada após o refino."""

from __future__ import annotations

import re
from typing import Any

Change = dict[str, object]
EditorInfo = dict[str, Any]


def _record_change(
    changes: list[Change],
    before: str,
    after: str,
    line_number: int,
    reason: str,
    mode: str,
) -> None:
    """Registra uma alteração somente quando o conteúdo da linha mudou."""
    if before == after:
        return
    changes.append(
        {
            "before": before,
            "after": after,
            "line": line_number,
            "reason": reason,
            "mode": mode,
        }
    )


def editor_lite(text: str) -> tuple[str, EditorInfo]:
    """Corrige espaçamento e repetições acidentais sem reescrever o texto."""
    output_lines: list[str] = []
    changes: list[Change] = []
    repeated_word = re.compile(r"\b(\w+)\s+\1\b", flags=re.IGNORECASE)

    for line_number, line in enumerate(text.splitlines(), start=1):
        original = line
        line = repeated_word.sub(r"\1", line)
        line = re.sub(r"\s+([,.;!?])", r"\1", line)
        line = re.sub(r"[ \t]{2,}", " ", line)
        _record_change(
            changes,
            original,
            line,
            line_number,
            "espaçamento ou repetição acidental",
            "editor-lite",
        )
        output_lines.append(line)

    return "\n".join(output_lines), {"changes": len(changes), "detail": changes}


def editor_consistency(text: str, memory: dict[str, Any] | None = None) -> tuple[str, EditorInfo]:
    """Aplica substituições de consistência fornecidas pela memória editorial."""
    memory = memory or {}
    configured_replacements = memory.get("replacements", {})
    replacements = configured_replacements if isinstance(configured_replacements, dict) else {}
    output_lines: list[str] = []
    changes: list[Change] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        original = line
        for source, target in replacements.items():
            if not isinstance(source, str) or not isinstance(target, str) or not source:
                continue
            line = re.sub(re.escape(source), target, line, flags=re.IGNORECASE)
        if memory.get("past_preference"):
            line = re.sub(
                r"\b[eE]ra como se (ele|ela) é\b",
                r"era como se \1 fosse",
                line,
            )
        _record_change(
            changes,
            original,
            line,
            line_number,
            "padronização definida na memória editorial",
            "editor-consistency",
        )
        output_lines.append(line)

    memory["changes"] = int(memory.get("changes", 0)) + len(changes)
    return "\n".join(output_lines), {"changes": len(changes), "detail": changes}


def editor_voice(
    text: str, character_map: dict[str, dict[str, str]] | None = None
) -> tuple[str, EditorInfo]:
    """Uniformiza apenas pontuação e espaços em falas iniciadas por travessão."""
    character_map = character_map or {}
    output_lines: list[str] = []
    changes: list[Change] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        original = line
        if line.lstrip().startswith("—"):
            line = re.sub(r"\.{4,}", "…", line)
            line = re.sub(r"[ \t]{2,}", " ", line)
        _record_change(
            changes,
            original,
            line,
            line_number,
            "ritmo e espaçamento de fala",
            "editor-voice",
        )
        output_lines.append(line)

    return "\n".join(output_lines), {
        "changes": len(changes),
        "detail": changes,
        "character_map": character_map,
    }


def editor_strict(text: str) -> tuple[str, EditorInfo]:
    """Remove vícios de linguagem conhecidos sem alterar eventos da narrativa."""
    output_lines: list[str] = []
    changes: list[Change] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        original = line
        line = re.sub(r"\btipo,\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(
            r"\bcomo se (ele|ela) fosse tipo\b",
            r"como se \1 fosse",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"\b(muito\s+){2,}", "muito ", line, flags=re.IGNORECASE)
        line = re.sub(r"[ \t]{2,}", " ", line)
        line = re.sub(r"\s+([,.;!?])", r"\1", line)
        _record_change(
            changes,
            original,
            line,
            line_number,
            "redução de vícios de linguagem",
            "editor-strict",
        )
        output_lines.append(line)

    return "\n".join(output_lines), {"changes": len(changes), "detail": changes}


def editor_pipeline(text: str, flags: dict[str, bool]) -> tuple[str, list[Change]]:
    """Executa, na ordem, os modos editoriais habilitados."""
    changes: list[Change] = []
    current_text = text
    memory: dict[str, Any] = {}

    if flags.get("lite"):
        current_text, info = editor_lite(current_text)
        changes.extend(info.get("detail", []))
    if flags.get("consistency"):
        current_text, info = editor_consistency(current_text, memory)
        changes.extend(info.get("detail", []))
    if flags.get("voice"):
        current_text, info = editor_voice(current_text)
        changes.extend(info.get("detail", []))
    if flags.get("strict"):
        current_text, info = editor_strict(current_text)
        changes.extend(info.get("detail", []))

    return current_text, changes
