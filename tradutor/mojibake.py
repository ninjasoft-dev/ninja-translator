from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

# Sequências em codepoints para evitar introduzir mojibake literal no código.
MOJIBAKE_CODEPOINTS: list[tuple[int, ...]] = [
    (0xD4, 0xC7, 0xA3),  # cp1252 decode de U+201C (“) vira D4 C7 A3
    (0xD4, 0xC7, 0x98),  # cp1252 decode de U+201D (”) vira D4 C7 98
    (0xD4, 0xC7, 0xAA),  # cp1252 decode de U+2026 (…) vira D4 C7 AA
    (0xC3, 0xA2, 0x20AC),  # prefixo comum de UTF-8 mojibake (bytes c3 a2 82 ac)
    (0x251C,),  # caractere U+251C (box drawings) usado como sentinela de range quebrado
    (0x251C, 0xC7),  # U+251C seguido de C7 (sequência típica de mojibake)
    (0x251C, 0xD1),  # U+251C seguido de D1 (sequência típica de mojibake)
]

MOJIBAKE_TOKENS: list[str] = ["".join(chr(c) for c in seq) for seq in MOJIBAKE_CODEPOINTS]


def scan_paths(paths: Iterable[Path], tokens: Sequence[str] | None = None) -> list[str]:
    """Retorna lista de mensagens de erro encontradas ao varrer por mojibake/UTF-8."""
    tokens = list(tokens) if tokens is not None else MOJIBAKE_TOKENS
    errors: list[str] = []
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"Arquivo nao e UTF-8: {path} ({exc})")
            continue
        for token in tokens:
            if token and token in content:
                errors.append(f"Encontrado mojibake '{token}' em {path}")
                break
    return errors
