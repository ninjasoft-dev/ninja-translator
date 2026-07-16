from tradutor.translate import _separate_short_dialogues


def test_separate_short_dialogues_inserts_blank_lines_between_short_lines() -> None:
    """Valida a normalização de conteúdo válido na separação de diálogos curtos."""
    raw = "\n".join(
        [
            "“Eh?”",
            "“Hm?”",
            "“…Hmm,” sighed the Baron.",
            "“Goddess…?”",
        ]
    )
    normalized = _separate_short_dialogues(raw)
    paragraphs = [p for p in normalized.split("\n\n") if p.strip()]
    assert len(paragraphs) == 4
    assert paragraphs[0].strip() == "“Eh?”"
    assert paragraphs[1].strip() == "“Hm?”"
    assert paragraphs[2].strip() == "“…Hmm,” sighed the Baron."
    assert paragraphs[3].strip() == "“Goddess…?”"
