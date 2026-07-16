import logging

from tradutor.sanitizer import sanitize_translation_output


def test_sanitize_translation_keeps_dialogue_apology() -> None:
    """Confirma a preservação de aspas e estrutura de diálogos na sanitização."""
    text = '“Desculpe…” disse Kayako.\n"O que você está pedindo desculpas?"'
    cleaned, report = sanitize_translation_output(text, logger=logging.getLogger("sanitizer-test"))
    assert "Desculpe" in cleaned
    assert "desculpas" in cleaned
    assert report.contamination_detected is False
