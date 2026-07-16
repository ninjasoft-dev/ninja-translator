from tradutor.postprocess_translation import postprocess_translation
from tradutor.qa import needs_retry
from tradutor.translate import _normalize_chunk_dialogue_quotes


def test_needs_retry_unbalanced_curly_quotes():
    """Confirma a detecção de problemas em conteúdo válido na tradução."""
    input_text = "“How did you get there?”"
    output_text = "“Como você chegou lá?"

    retry, reason = needs_retry(input_text, output_text)

    assert retry is True
    assert "unbalanced_quotes" in reason


def test_chunk_quote_normalizer_removes_one_spurious_terminal_straight_quote() -> None:
    """Valida a remoção segura de aspas e estrutura de diálogos na tradução."""
    source = "“The dialogue ends here.” The narration continues."
    translated = '"A fala termina aqui." A narração continua."'

    normalized = _normalize_chunk_dialogue_quotes(source, translated)

    assert normalized == "“A fala termina aqui.” A narração continua."


def test_chunk_quote_normalizer_removes_premature_curly_close_before_laughter() -> None:
    """Valida a remoção segura de aspas e estrutura de diálogos na tradução."""
    source = "Aurelia riu. “Speak properly, won't you? Pfft, hee hee! Pathetic!”"
    translated = "Aurelia riu. “Fale direito, pode ser?” Pfft, hee hee! Patética!”"

    normalized = _normalize_chunk_dialogue_quotes(source, translated)

    assert normalized == "Aurelia riu. “Fale direito, pode ser? Pfft, hee hee! Patética!”"


def test_needs_retry_allows_quote_boundary_in_source_chunk() -> None:
    """Confirma que a tradução distingue o caso válido do artefato que deve corrigir."""
    input_text = "From my perspective, this can be resolved.”"
    output_text = "Do meu ponto de vista, isso pode ser resolvido.”"

    retry, reason = needs_retry(input_text, output_text)

    assert retry is False
    assert reason == ""


def test_needs_retry_allows_odd_quote_boundary_when_style_changes() -> None:
    """Confirma que a tradução distingue o caso válido do artefato que deve corrigir."""
    input_text = "“First.”\n\n“Second.”\n\nThird?”"
    output_text = '"Primeiro."\n\n"Segundo."\n\n"Terceiro'

    retry, reason = needs_retry(input_text, output_text)

    assert retry is False
    assert reason == ""


def test_needs_retry_rejects_extra_close_against_source_boundary() -> None:
    """Confirma a detecção de problemas em limites estruturais na tradução."""
    input_text = "From my perspective, this can be resolved.”"
    output_text = "Do meu ponto de vista, isso pode ser resolvido.””"

    retry, reason = needs_retry(input_text, output_text)

    assert retry is True
    assert reason == "unbalanced_quotes"


def test_needs_retry_rejects_extra_balanced_quote_pair() -> None:
    """Confirma a detecção de problemas em aspas e estrutura de diálogos na tradução."""
    input_text = "“Primeira fala.”"
    output_text = "“Primeira fala.”\n\n“Fala inventada.”"

    retry, reason = needs_retry(input_text, output_text)

    assert retry is True
    assert reason == "extra_curly_quotes"


def test_needs_retry_allows_one_quote_pair_that_repairs_internal_source_defect() -> None:
    """Confirma que a tradução distingue o caso válido do artefato que deve corrigir."""
    input_text = "“First speech.”\n\nFrom my perspective, this can be resolved.”\n\n“No, forget it."
    output_text = (
        "“Primeira fala.”\n\n“Do meu ponto de vista, isso pode ser resolvido.”\n\n“Não, esqueça.”"
    )

    retry, reason = needs_retry(input_text, output_text)

    assert retry is False
    assert reason == ""


def test_needs_retry_allows_single_missing_open_quote_repair() -> None:
    """Confirma que a tradução distingue o caso válido do artefato que deve corrigir."""
    input_text = "I believe the class shares that intention.”"
    output_text = "“Acredito que a turma compartilhe essa intenção.”"

    retry, reason = needs_retry(input_text, output_text)

    assert retry is False
    assert reason == ""


def test_needs_retry_extra_short_repetition():
    """Confirma a detecção de problemas em repetições indevidas na tradução."""
    input_text = "Crack!\nSilence."
    output_text = "Crack!\nCrack!\nCrack!\nSilence."

    retry, reason = needs_retry(input_text, output_text)

    assert retry is True
    assert "extra_short_repetition" in reason


def test_needs_retry_allows_short_line_repetition_present_in_source() -> None:
    """Confirma que a tradução distingue o caso válido do artefato que deve corrigir."""
    input_text = "\n\n".join(["“On your knees.”"] * 5)
    output_text = "\n\n".join(["“Ajoelhe-se.”"] * 5)

    retry, reason = needs_retry(input_text, output_text)

    assert retry is False
    assert reason == ""


def test_dash_line_strips_trailing_quote():
    """Valida a remoção segura de aspas e estrutura de diálogos na tradução."""
    samples = [
        ("— Entendido”, Lina me respondeu.", "— Entendido, Lina me respondeu."),
        ("— Oh?!” Aurelia perguntou.", "— Oh?! Aurelia perguntou."),
        ("— Eh?” Aurelia disse.", "— Eh? Aurelia disse."),
        ("— Hm.” Lina comentou.", "— Hm. Lina comentou."),
        ("— …” ele murmurou.", "— … ele murmurou."),
    ]

    for raw, expected in samples:
        cleaned = postprocess_translation(
            raw, en_text=""
        )  # en_text vazio para passar pelo pipeline
        assert cleaned == expected


def test_postprocess_translation_collapses_duplicate_curly_quote() -> None:
    """Valida a normalização de aspas e estrutura de diálogos na tradução."""
    assert postprocess_translation("A fala terminou.””", en_text="") == "A fala terminou.”"
