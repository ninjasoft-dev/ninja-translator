"""Testes dos normalizadores determinísticos aplicados à tradução."""

from tradutor.text_postprocess import apply_custom_normalizers


def test_custom_normalizer_converts_gulp() -> None:
    """Converte a onomatopeia isolada sem tocar na frase seguinte."""
    text = "Gulp.\nThe baron swallowed hard and kept going."
    result = apply_custom_normalizers(text)

    assert result.startswith("Glup.")
    assert "swallowed hard" in result


def test_custom_normalizer_translates_common_english_interjections() -> None:
    """Traduz interjeições residuais frequentes em saídas híbridas."""
    text = "Phew, consegui.\n\nGeez, foi estranho.\n\nHuh, entendi.\n\nUgh, que saco."
    result = apply_custom_normalizers(text)

    assert "Ufa, consegui." in result
    assert "Nossa, foi estranho." in result
    assert "Hã, entendi." in result
    assert "Argh, que saco." in result


def test_custom_normalizer_converts_full_quote_dialogues_to_dashes() -> None:
    """Converte apenas linhas inteiras de diálogo entre aspas."""
    text = '“Hello there.”\nNarration line.\n"Oi!"'
    result = apply_custom_normalizers(text)

    assert result.splitlines() == ["— Hello there.", "Narration line.", "— Oi!"]


def test_custom_normalizer_can_preserve_quote_dialogues() -> None:
    """Mantém aspas quando a conversão para travessão está desabilitada."""
    text = '“Hello there.”\nNarration line.\n"Oi!"'

    result = apply_custom_normalizers(text, convert_quote_dialogues=False)

    assert result == text


def test_custom_normalizer_fixes_poderam() -> None:
    """Corrige a flexão verbal incorreta sem alterar o restante da frase."""
    assert apply_custom_normalizers("Eles poderam vencer.") == "Eles puderam vencer."


def test_custom_normalizer_merges_speech_with_verb() -> None:
    """Reúne fala e atribuição separadas por um parágrafo vazio."""
    result = apply_custom_normalizers("“Oi.”\n\nperguntou João.")

    assert result == "— Oi. perguntou João."


def test_custom_normalizer_merges_dash_attribution_line() -> None:
    """Reúne uma atribuição iniciada por travessão à fala anterior."""
    text = "— Por quê?\n— perguntou Theo, direto."

    result = apply_custom_normalizers(text, convert_quote_dialogues=False)

    assert result == "— Por quê? — perguntou Theo, direto."


def test_custom_normalizer_merges_attribution_after_blank_line() -> None:
    """Ignora linhas vazias entre a fala e uma atribuição inequívoca."""
    text = "— O que você está lendo?\n\n— Mara perguntou enquanto fechava o livro."

    result = apply_custom_normalizers(text, convert_quote_dialogues=False)

    assert result == "— O que você está lendo? — Mara perguntou enquanto fechava o livro."


def test_custom_normalizer_preserves_pronoun_speech() -> None:
    """Não confunde uma nova fala em primeira pessoa com atribuição."""
    text = "— Vai devolver o livro?\n\n— Eu disse que sim!"

    result = apply_custom_normalizers(text, convert_quote_dialogues=False)

    assert result == text


def test_custom_normalizer_merges_pareceu_attribution() -> None:
    """Reconhece atribuições narrativas construídas com 'pareceu'."""
    text = "— E-eu também vou?\n— Theo pareceu chocado."

    result = apply_custom_normalizers(text, convert_quote_dialogues=False)

    assert result == "— E-eu também vou? — Theo pareceu chocado."
