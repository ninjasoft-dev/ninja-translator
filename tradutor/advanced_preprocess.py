"""
Pré-processamento avançado opcional para limpeza de texto antes de traduzir/refinar.
Determinístico e conservador: não altera ordem nem diálogos.
"""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Limpa texto."""
    if not text:
        return text
    cleaned = text
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Remove marcas de formatação incomuns.
    cleaned = re.sub(r"[■◆◆◇♢◆■]+", "", cleaned)
    cleaned = cleaned.replace("<lf>", "").replace("<LF>", "")
    # desfaz hifenização de quebra de linha
    cleaned = re.sub(r"(\w+)-\s*\n(\w+)", r"\1\2\n", cleaned)
    # agrupa linhas quebradas de diálogo simples: "— algo\ncontinuação"
    lines = cleaned.splitlines()
    buffer = []
    result = []
    for ln in lines:
        if ln.strip() == "":
            if buffer:
                result.append(" ".join(buffer).strip())
                buffer = []
            result.append("")
            continue
        if (
            buffer
            and not buffer[-1].endswith((".", "!", "?", "—"))
            and not ln.lstrip().startswith("—")
        ):
            buffer.append(ln.strip())
        else:
            if buffer:
                result.append(" ".join(buffer).strip())
            buffer = [ln.strip()]
    if buffer:
        result.append(" ".join(buffer).strip())
    cleaned = "\n".join(result)
    return cleaned.strip()
