import logging

from tradutor.preprocess import strip_toc
from tradutor.utils import setup_logging


def test_strip_toc_removes_initial_summary() -> None:
    """Valida a remoção segura de sumários e conteúdo narrativo no comportamento testado."""
    logger = setup_logging(level=logging.ERROR)
    toc = """Prologue
1
Chapter 1
5
Chapter 2
9
Chapter 3
12
Afterword
18
"""
    real_text = "This is the real story start.\nMore narrative that should stay."
    raw = toc + "\n" + real_text

    cleaned = strip_toc(raw, logger=logger, max_lines=50, min_markers=4, max_body_len=50)

    assert cleaned.startswith("This is the real story start.")
    assert "Prologue\n1" not in cleaned
