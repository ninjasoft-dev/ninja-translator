import logging
import re

from tradutor.preprocess import (
    chunk_for_translation,
    paragraphs_from_text,
    preprocess_text,
)
from tradutor.section_splitter import split_into_sections
from tradutor.utils import setup_logging


def test_no_stub_chunks_after_toc_removal() -> None:
    """Valida as regras de sumários e conteúdo narrativo no comportamento testado."""
    logger = setup_logging(level=logging.ERROR)
    raw = """Prologue
Chapter 1
Chapter 2
Chapter 3

Prologue
The actual story begins here with enough narrative content to exceed the stub threshold. """
    raw += (
        "It keeps going with dialogue and description so the chunk length comfortably surpasses two hundred characters, "
        "preventing the guardrail from treating it as a stub."
    )

    clean = preprocess_text(raw, logger=logger, skip_front_matter=True)
    sections = split_into_sections(clean)
    assert sections, "Expected at least one section after TOC removal."

    chunks: list[str] = []
    for sec in sections:
        paragraphs = paragraphs_from_text(sec["body"])
        chunks.extend(chunk_for_translation(paragraphs, max_chars=240, logger=logger))

    assert chunks, "Chunk list should not be empty."
    first = chunks[0].strip()
    assert not re.fullmatch(
        r"#\\s*(Prologue|Chapter\\s+\\d+(?::[^\\n]+)?|Epilogue|Afterword)\\s*",
        first,
        flags=re.IGNORECASE,
    )
    assert len(first) >= 200
