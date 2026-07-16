import logging

from tradutor.preprocess import preprocess_text
from tradutor.utils import setup_logging


def test_preprocess_removes_toc_when_skip_front_matter_enabled() -> None:
    """Valida a remoção segura de sumários e conteúdo narrativo no pré-processamento."""
    logger = setup_logging(level=logging.ERROR)
    toc = """Prologue
1
Chapter 1
5
Chapter 2
9
Afterword
18

"""
    real = "Real story starts here.\nAnd continues without TOC."
    raw = toc + real

    cleaned = preprocess_text(raw, logger=logger, skip_front_matter=True)

    assert cleaned.startswith("Real story starts here.")
    assert "Chapter 1" not in cleaned.splitlines()[0]
