from tradutor.qa import needs_retry


def test_needs_retry_when_dialogue_quotes_missing() -> None:
    """Confirma a detecção de problemas em aspas e estrutura de diálogos nas verificações de qualidade."""
    input_text = '"A"\n"B"\n"C"\n"D"\n'
    output_text = '"A"\n"B"\n'
    retry, reason = needs_retry(input_text, output_text)
    assert retry
    assert "omissao_dialogo" in reason


def test_needs_retry_when_sanitization_aggressive() -> None:
    """Confirma a detecção de problemas em conteúdo válido nas verificações de qualidade."""
    input_text = "Hello\nWorld"
    output_text = "Hello"
    retry, reason = needs_retry(
        input_text,
        output_text,
        contamination_detected=True,
        sanitization_ratio=0.8,
    )
    assert retry
    assert "sanitizacao" in reason or "output truncado" in reason
