import textwrap

from tradutor.preprocess import preprocess_text


def test_preprocess_removes_watermarks_but_keeps_text() -> None:
    """Confirma que o pré-processamento distingue o caso válido do artefato que deve corrigir."""
    raw = textwrap.dedent(
        """
        They say nothing good can ever come from revenge.
        Page 190
        Sample Group | releases.example
        Foul Goddess…
        I will have my revenge.
        Page 191
        Sample Group | releases.example
        """
    ).strip()

    cleaned = preprocess_text(raw, logger=None)

    assert "Page 190" not in cleaned
    assert "Sample Group" not in cleaned
    assert "releases" not in cleaned

    assert "revenge" in cleaned
    assert "Foul Goddess" in cleaned
    assert "I will have my revenge" in cleaned
