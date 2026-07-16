from __future__ import annotations

from pathlib import Path

import pytest

from tradutor.mojibake import MOJIBAKE_TOKENS


def _iter_source_files() -> list[Path]:
    """Percorre os arquivos-fonte incluídos na auditoria de codificação."""
    root = Path(__file__).resolve().parents[1]
    sources: set[Path] = set()
    patterns = ("*.py", "*.md", "*.json")
    for rel in ("tradutor", "tests"):
        base = root / rel
        for pattern in patterns:
            sources.update(base.rglob(pattern))
    return sorted(sources)


@pytest.mark.parametrize("path", _iter_source_files())
def test_source_files_are_utf8_and_free_of_mojibake(path: Path) -> None:
    """Valida as regras de conteúdo válido na validação de codificação."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        pytest.fail(f"Arquivo nao e UTF-8: {path} ({exc})")
    for token in MOJIBAKE_TOKENS:
        assert token not in content, f"Encontrado mojibake '{token}' em {path}"
