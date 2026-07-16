import logging

from tradutor.preprocess import paragraphs_from_text
from tradutor.utils import chunk_by_paragraphs, setup_logging


def test_chunking_preserves_text_length_and_content() -> None:
    """Confirma a preservação de integridade do conteúdo na divisão em chunks."""
    text = (
        "Sentence one. Another sentence follows.\n\n"
        "Second paragraph has two sentences. And a final one."
    )
    paragraphs = paragraphs_from_text(text)
    logger = setup_logging(logging.DEBUG)

    chunks = chunk_by_paragraphs(paragraphs, max_chars=40, logger=logger, label="test-chunk")

    reconstructed = "".join(chunks)
    expected = "\n\n".join(paragraphs)

    assert reconstructed == expected
    assert sum(len(c) for c in chunks) == len(expected)
