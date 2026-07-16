from tradutor.cleanup import cleanup_before_refine


def _cleanup(text: str) -> str:
    """Executa a limpeza com os parâmetros comuns deste módulo."""
    cleaned, _ = cleanup_before_refine(text)
    return cleaned


def test_dedupe_keeps_three_crack():
    """Confirma a preservação de repetições indevidas na limpeza determinística."""
    text = "Crack!\nCrack!\nCrack!"
    cleaned = _cleanup(text)
    assert cleaned.count("Crack!") == 3


def test_dedupe_keeps_double_question_dash():
    """Confirma a preservação de repetições indevidas na limpeza determinística."""
    text = "— ?\n— ?"
    cleaned = _cleanup(text)
    assert cleaned.count("— ?") == 2


def test_dedupe_keeps_double_ellipsis_dash():
    """Confirma a preservação de reticências na limpeza determinística."""
    text = "— …\n— …"
    cleaned = _cleanup(text)
    assert cleaned.count("— …") == 2


def test_dedupe_keeps_short_fragments_in_paragraph():
    """Confirma a preservação de linhas e limites de parágrafo na limpeza determinística."""
    text = "Crack! Crack! Crack!"
    cleaned = _cleanup(text)
    assert cleaned.count("Crack!") == 3


def test_dedupe_keeps_short_dialogue_fragments():
    """Confirma a preservação de aspas e estrutura de diálogos na limpeza determinística."""
    text = "— ? — ?"
    cleaned = _cleanup(text)
    assert cleaned.count("— ?") == 2
