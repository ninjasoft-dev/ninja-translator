from tradutor.refine import has_suspicious_repetition


def test_repetition_detector_not_trigger_on_normal_text() -> None:
    """Confirma a detecção de problemas em repetições indevidas no detector de repetição."""
    text = "Ele disse algo para Mara. Ela respondeu calmamente. O narrador descreve a cena em detalhes naturais sem loops."
    assert has_suspicious_repetition(text) is False


def test_repetition_detector_triggers_on_loops() -> None:
    """Confirma a detecção de problemas em repetições indevidas no detector de repetição."""
    para = "Bram riu alto e ergueu a mão de Mara."
    text = f"{para}\n\n{para}\n\n{para}"
    assert has_suspicious_repetition(text) is True
