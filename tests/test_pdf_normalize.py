from tradutor.pdf import _inline_markdown_to_html, normalize_markdown_for_pdf


def test_normalize_markdown_for_pdf_converts_br_and_paragraphs():
    """Valida a normalização de conteúdo válido na geração de PDF."""
    text = "Linha 1<br/>\nLinha 2\n\nLinha 3<br>Continua"
    parts = normalize_markdown_for_pdf(text)
    assert parts == ["Linha 1", "Linha 2", "Linha 3\nContinua"]


def test_inline_markdown_to_html_preserves_simple_tags():
    """Confirma a preservação de conteúdo válido na geração de PDF."""
    src = "**negrito** e _italico_"
    html = _inline_markdown_to_html(src)
    assert "<b>negrito</b>" in html
    assert "<i>italico</i>" in html
