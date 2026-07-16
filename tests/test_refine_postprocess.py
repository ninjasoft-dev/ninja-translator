from tradutor.text_postprocess import fix_dialogue_artifacts


def test_falas_coladas_com_espaco():
    """Valida as regras de aspas e estrutura de diálogos no pós-processamento."""
    text = "“A!” “B!”"
    fixed, stats = fix_dialogue_artifacts(text)
    assert fixed == "“A!”\n\n“B!”"
    assert stats["dialogue_splits"] == 1


def test_falas_coladas_sem_espaco():
    """Valida as regras de aspas e estrutura de diálogos no pós-processamento."""
    text = "“A!””“B!”"
    fixed, _ = fix_dialogue_artifacts(text)
    assert fixed == "“A!”\n\n“B!”"


def test_aspas_coladas_generico():
    """Valida as regras de aspas e estrutura de diálogos no pós-processamento."""
    text = "Algo disse.” “Outra fala..."
    fixed, _ = fix_dialogue_artifacts(text)
    assert fixed.endswith("disse.”\n\n“Outra fala...")


def test_remove_aspas_triplas():
    """Valida a remoção segura de aspas e estrutura de diálogos no pós-processamento."""
    text = 'Algo suficiente."""\n'
    fixed, stats = fix_dialogue_artifacts(text)
    assert fixed == "Algo suficiente.\n"
    assert stats["triple_quotes_removed"] == 1


def test_linha_em_branco_dentro_de_fala():
    """Valida as regras de conteúdo válido no pós-processamento."""
    text = "“Entendo.\n\nQuer dizer que sim.”"
    fixed, stats = fix_dialogue_artifacts(text)
    assert "\n\n" not in fixed
    assert "“Entendo. Quer dizer que sim.”" == fixed
    assert stats["inquote_blank_collapses"] >= 1


def test_paragrafo_fora_de_aspas_permanece():
    """Valida as regras de aspas e estrutura de diálogos no pós-processamento."""
    original = "Paragrafo A.\n\nParagrafo B."
    fixed, _ = fix_dialogue_artifacts(original)
    assert fixed == original


def test_idempotencia():
    """Confirma que o pós-processamento é idempotente."""
    text = "“Oi!” “” “Tchau.”\n\nFora das aspas."
    once, _ = fix_dialogue_artifacts(text)
    twice, _ = fix_dialogue_artifacts(once)
    assert once == twice
