from tradutor.refine import build_refine_prompt


def test_gender_instruction_present() -> None:
    """Valida as regras de consistência de gênero nas regras de gênero."""
    prompt = build_refine_prompt("dummy")
    lower = prompt.lower()
    assert "gênero" in lower
    assert "narrador" in lower
    assert "masculino" in lower


def test_refine_prompt_preserves_dialogue_and_paragraph_style() -> None:
    """Confirma a preservação de aspas e estrutura de diálogos nas regras de gênero."""
    prompt = build_refine_prompt("dummy")
    lower = prompt.lower()
    assert "não converta travessões em aspas" in lower
    assert "não dividir parágrafos" in lower
    assert "não inserir quebras de linha" in lower
    assert "não converter o padrão de diálogo" in lower
