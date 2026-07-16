from tradutor.cache_utils import detect_model_collapse


def test_stopwords_do_not_trigger_collapse():
    """Confirma que o detector de colapso distingue o caso válido do artefato que deve corrigir."""
    text = "Eu acho que isso é de de de de de de de de de de de de alguma forma correto e de novo."
    assert detect_model_collapse(text, mode="refine") is False


def test_repeated_token_run_triggers_collapse():
    """Confirma a detecção de problemas em repetições indevidas no detector de colapso."""
    text = "sim sim sim sim sim sim sim sim"
    assert detect_model_collapse(text, mode="refine") is True


def test_repeated_lines_triggers_collapse():
    """Confirma a detecção de problemas em repetições indevidas no detector de colapso."""
    text = (
        "Linha longa repetida aqui\nLinha longa repetida aqui\nLinha longa repetida aqui\ncontinua"
    )
    assert detect_model_collapse(text, mode="refine") is True


def test_short_repeated_lines_do_not_trigger():
    """Confirma que o detector de colapso distingue o caso válido do artefato que deve corrigir."""
    text = "curta\ncurta\ncurta\nseguindo"
    assert detect_model_collapse(text, mode="refine") is False


def test_accent_heavy_pt_does_not_collapse():
    """Confirma a preservação de repetições indevidas no detector de colapso."""
    phrase = "Ação e emoção são parte da sua decisão, coração difícil e útil para você."
    text = " ".join([phrase] * 10)  # muitas ocorrências com acento
    assert detect_model_collapse(text, mode="refine") is False


def test_bad_ratio_triggers_collapse_refine():
    """Confirma a detecção de problemas em repetições indevidas no detector de colapso."""
    text = "curto demais"
    assert detect_model_collapse(text, original_len=200, mode="refine") is True
