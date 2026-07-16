"""Testes do pós-processamento final semanticamente neutro."""

from tradutor.postprocess import final_pt_postprocess


def test_final_postprocess_normalizes_punctuation_and_quotes() -> None:
    """Corrige pontuação repetida e espaços internos de aspas."""
    text = "“  Olá...  ”” Ela respondeu  !"

    result = final_pt_postprocess(text)

    assert result == "“Olá…” Ela respondeu!"


def test_final_postprocess_converts_safe_straight_quote_pairs() -> None:
    """Converte pares de aspas retas sem alterar marcas de polegada."""
    text = '"Chegamos", disse Mara. A placa media 12" de largura.'

    result = final_pt_postprocess(text)

    assert result == '“Chegamos”, disse Mara. A placa media 12" de largura.'


def test_final_postprocess_converts_line_initial_dialogue_dash() -> None:
    """Padroniza hífen e meia-risca usados no início de falas."""
    text = "- Primeira fala.\n– Segunda fala."

    result = final_pt_postprocess(text)

    assert result == "— Primeira fala.\n— Segunda fala."


def test_final_postprocess_removes_internal_markers() -> None:
    """Remove delimitadores internos sem descartar o conteúdo traduzido."""
    text = "### TEXTO_TRADUZIDO_INICIO\nTexto final.\n### TEXTO_TRADUZIDO_FIM"

    result = final_pt_postprocess(text)

    assert result == "Texto final."


def test_final_postprocess_is_semantically_neutral() -> None:
    """Mantém palavras e nomes que não sejam marcadores estruturais."""
    text = "Mara levou a lanterna antiga até a ponte."

    assert final_pt_postprocess(text) == text


def test_final_postprocess_is_idempotent() -> None:
    """Não acumula novas alterações em execuções repetidas."""
    text = "— Espere...\n\nMara parou."

    once = final_pt_postprocess(text)

    assert final_pt_postprocess(once) == once
