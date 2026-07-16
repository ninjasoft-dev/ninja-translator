import logging

from tradutor.preprocess import (
    _quote_delta,
    _translation_chunk_end,
    chunk_for_translation,
    chunk_for_translation_with_offsets,
    preprocess_text,
)


def test_preprocess_isolates_multiword_chapter_subtitle_from_dropcap_narration() -> None:
    """Valida as regras de títulos estruturais no pré-processamento."""
    raw = "Chapter 1:\nAfter the Last Trial AFTER MARA watched Theo freeze."

    cleaned, stats = preprocess_text(raw, return_stats=True)

    assert "Chapter 1:" in cleaned
    assert "After the Last Trial" in cleaned
    assert "After Mara watched Theo freeze." in cleaned
    assert "After the Last Trial AFTER MARA" not in cleaned
    assert stats["subheading_isolation_fixes"] == 1


def test_preprocess_normalizes_leading_small_caps_from_pdf() -> None:
    """Valida a normalização de artefatos de extração e OCR no pré-processamento."""
    raw = "Chapter 1:\n\nAfter the Last Trial\n\nAFTER MARA watched Theo freeze."

    cleaned, stats = preprocess_text(raw, return_stats=True)

    assert "After Mara watched Theo freeze." in cleaned
    assert stats["leading_small_caps_normalized"] == 1


def test_preprocess_collapses_blank_line_inside_open_quote() -> None:
    """Valida a normalização de aspas e estrutura de diálogos no pré-processamento."""
    raw = "“The speaker continues here.\n\nThis is still the same quote.”"

    cleaned, stats = preprocess_text(raw, return_stats=True)

    assert "continues here. This is still" in cleaned
    assert stats["quote_blank_lines_fixed"] == 1


def test_preprocess_joins_inline_quoted_continuation_after_pdf_gap() -> None:
    """Valida a normalização de conteúdo válido no pré-processamento."""
    raw = "It is possible that he believes that from\n\n“the bottom of his heart.”"

    cleaned, stats = preprocess_text(raw, return_stats=True)

    assert "from “the bottom of his heart.”" in cleaned
    assert stats["hard_wrap_merges"] == 1


def test_translation_chunks_do_not_split_open_curly_dialogue() -> None:
    """Evita encerrar um chunk no meio de um diálogo com aspas curvas."""
    paragraphs = [
        "Introdução curta.",
        "“" + "Uma frase de diálogo. " * 12 + "Fim da fala.”",
        "Encerramento curto.",
    ]
    logger = logging.getLogger("chunk-quote-test")

    chunks = chunk_for_translation(paragraphs, max_chars=120, logger=logger)
    chunks_with_offsets = chunk_for_translation_with_offsets(
        paragraphs, max_chars=120, logger=logger
    )

    assert len(chunks) == len(chunks_with_offsets)
    assert any(len(chunk) > 120 for chunk in chunks)
    assert all(chunk.count("“") == chunk.count("”") for chunk in chunks)
    assert [chunk for chunk, _start, _end in chunks_with_offsets] == chunks


def test_translation_chunk_closes_dialogue_opened_before_chunk_boundary() -> None:
    """Valida as regras de aspas e estrutura de diálogos no pré-processamento."""
    text = (
        "“Fala iniciada antes da fronteira. "
        "Ainda aberta e agora encerrada.” Narração intermediária. “"
        + ("Uma frase adicional. " * 32)
        + "Encerramento.” Depois."
    )
    start = text.index("Ainda aberta")

    end = _translation_chunk_end(
        text, start, max_chars=90, logger=logging.getLogger("chunk-cross-boundary")
    )

    assert _quote_delta(text[:start]) == 1
    assert _quote_delta(text[:end]) == 0
    assert text[start:end].endswith("Encerramento.”")
