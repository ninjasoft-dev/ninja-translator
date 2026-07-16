from tradutor.refine import split_markdown_sections


def test_split_markdown_sections_preserves_prefix_and_headings() -> None:
    """Confirma a preservação de conteúdo válido no refino."""
    md = """Preamble text that should stay.

# Capítulo 1: O começo
Corpo do capitulo 1.

## Capítulo 2: Continuação
Corpo do capitulo 2."""
    sections = split_markdown_sections(md)
    assert len(sections) == 3
    assert sections[0][0] == ""
    assert "Preamble text" in sections[0][1]
    assert sections[1][0].startswith("# Capítulo 1")
    assert "Corpo do capitulo 1" in sections[1][1]
    assert sections[2][0].startswith("## Capítulo 2")
    assert "Corpo do capitulo 2" in sections[2][1]
