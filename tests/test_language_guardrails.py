from tradutor.language_guardrails import detect_residual_english, english_leak_segments


def test_detect_residual_english_long_sentence() -> None:
    """Detecta uma frase longa que permaneceu integralmente em inglês."""
    text = "“I have no desire to die,” replied Lina calmly, choosing not to answer directly."

    detected, reason = detect_residual_english(text)

    assert detected
    assert reason.startswith("residual_english:")


def test_detect_residual_english_known_single_game_term() -> None:
    """Detecta um termo técnico conhecido quando isolado em inglês."""
    detected, reason = detect_residual_english("Ela perdeu os buffs e recuou.")

    assert detected
    assert reason == "residual_english:buffs"


def test_detect_residual_english_plural_self() -> None:
    """Detecta a forma inglesa plural usada no meio do texto traduzido."""
    detected, reason = detect_residual_english(
        "As máscaras eram selves temporários para um propósito."
    )

    assert detected
    assert reason == "residual_english:selves"


def test_detect_residual_english_slang_abbreviation() -> None:
    """Detecta uma abreviação informal que permaneceu em inglês."""
    detected, reason = detect_residual_english("Ela é super desconfiada AF.")

    assert detected
    assert reason == "residual_english:AF"


def test_detect_residual_english_single_word_connector() -> None:
    """Detecta um conector inglês isolado entre frases em português."""
    detected, reason = detect_residual_english("Ela é arrogante, though.")

    assert detected
    assert reason == "residual_english:though"


def test_detect_residual_english_stylized_alright() -> None:
    """Reconhece uma grafia estilizada da interjeição inglesa."""
    detected, reason = detect_residual_english("“Arright!”, disse Itsuki.")

    assert detected
    assert reason == "residual_english:Arright"


def test_detect_residual_english_hybrid_pronoun() -> None:
    """Detecta um pronome híbrido criado na tradução."""
    detected, reason = detect_residual_english(
        "Ninguém o encarava com raiva — they todos pareciam felizes."
    )

    assert detected
    assert reason == "residual_english:they"


def test_detect_residual_english_boost() -> None:
    """Detecta um termo de ação deixado indevidamente em inglês."""
    detected, reason = detect_residual_english("Eles precisavam de um boost extra.")

    assert detected
    assert reason == "residual_english:boost"


def test_detect_residual_english_short_phrase() -> None:
    """Detecta uma expressão curta que permaneceu em inglês."""
    detected, reason = detect_residual_english("“I see”, disse Iara.")

    assert detected
    assert reason == "residual_english:I see"


def test_detect_residual_english_mixed_pronoun_artifact() -> None:
    """Detecta um artefato que combina pronome inglês e texto em português."""
    detected, reason = detect_residual_english("I não me importo de voltar para casa.")

    assert detected
    assert reason == "residual_english:I"


def test_detect_residual_english_mixed_pronoun_before_any_portuguese_word() -> None:
    """Detecta o pronome inglês mesmo antes de uma palavra portuguesa incomum."""
    detected, reason = detect_residual_english("Uau—I aposto que ela consegue.")

    assert detected
    assert reason == "residual_english:I"


def test_detect_residual_english_interjection() -> None:
    """Detecta uma interjeição inglesa no início da fala."""
    detected, reason = detect_residual_english("Uhh… sei disso.")

    assert detected
    assert reason == "residual_english:Uhh"


def test_detect_residual_english_short_interjection() -> None:
    """Detecta uma interjeição inglesa formada por uma única palavra."""
    detected, reason = detect_residual_english("Uh… sei disso.")

    assert detected
    assert reason == "residual_english:Uh"


def test_does_not_flag_natural_portuguese_with_names() -> None:
    """Preserva frases naturais em português que contêm nomes próprios."""
    text = "Mara Vale olhou para Lina com calma e respondeu que não pretendia recuar."

    assert english_leak_segments(text) == []
