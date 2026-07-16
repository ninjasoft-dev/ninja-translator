from __future__ import annotations

from tradutor import desquebrar


def test_quote_chars_includes_curly_quotes() -> None:
    """Confirma o registro correto de aspas e estrutura de diálogos na reconstrução de parágrafos."""
    assert "“" in desquebrar.QUOTE_CHARS
    assert "”" in desquebrar.QUOTE_CHARS


def test_ellipsis_re_matches_unicode_and_ascii() -> None:
    """Valida as regras de reticências na reconstrução de parágrafos."""
    assert desquebrar.ELLIPSIS_RE.search("...")
    assert desquebrar.ELLIPSIS_RE.search("…")


def test_remove_stray_quote_lines_removes_curly_quotes() -> None:
    """Valida a remoção segura de aspas e estrutura de diálogos na reconstrução de parágrafos."""
    text = "\n".join(
        [
            "primeira linha",
            "“",
            "linha do meio",
            "”",
            "ultima linha",
        ]
    )
    cleaned, removed = desquebrar._remove_stray_quote_lines(text)
    assert removed == 2
    assert "“" not in cleaned
    assert "”" not in cleaned
    assert "primeira linha" in cleaned
    assert "ultima linha" in cleaned


def test_count_quotes_handles_straight_and_curly() -> None:
    """Conta de modo uniforme aspas retas e curvas."""
    text = '“texto” "texto"'
    assert desquebrar._count_quotes(text) == 4
