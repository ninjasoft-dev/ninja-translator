from tradutor.desquebrar import normalize_wrapped_lines


def test_normalize_wrapped_lines_joins_lowercase_followup() -> None:
    """Valida a normalização de conteúdo válido na reconstrução de linhas."""
    raw = "He looked at her\nwith a faint smile."
    out = normalize_wrapped_lines(raw)
    assert "her with a faint smile." in out
    assert "\n" not in out


def test_normalize_wrapped_lines_keeps_dialogue_and_headings() -> None:
    """Confirma a preservação de aspas e estrutura de diálogos na reconstrução de linhas."""
    raw = "“Eh?”\ncontinuou.\n# Heading\nnext line"
    out = normalize_wrapped_lines(raw)
    lines = out.splitlines()
    assert lines[0].startswith("“Eh?”")
    assert lines[1].startswith("continuou.")
    assert lines[2].startswith("# Heading")
