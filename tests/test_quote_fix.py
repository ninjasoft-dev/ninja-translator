from tradutor.quote_fix import (
    count_curly_quotes,
    fix_blank_lines_inside_quotes,
    fix_unbalanced_quotes,
    repair_missing_open_quotes_per_paragraph,
)


def test_fix_unbalanced_quotes_inserts_closing_before_narration():
    """Valida a normalização de conteúdo válido no reparo de aspas."""
    text = (
        "… “Essa coisa de cavaleiro é só para manter as aparências do reino, sabe? "
        "Ele bebeu um gole antes de continuar. “O Rei Caçador de Monstros…”"
    )
    fixed, changed = fix_unbalanced_quotes(text, logger=None, label="test")
    opens, closes = count_curly_quotes(fixed)
    assert changed
    assert opens == closes
    assert "sabe?” Ele bebeu" in fixed
    assert "O Rei Caçador de Monstros" in fixed


def test_fix_no_change_when_balanced():
    """Valida a normalização de conteúdo válido no reparo de aspas."""
    balanced = "“Olá.” Ele disse. “Tchau.”"
    fixed, changed = fix_unbalanced_quotes(balanced, logger=None, label="test2")
    assert not changed
    assert fixed == balanced


def test_repair_missing_open_quote_when_global_counts_are_misleading():
    """Valida as regras de aspas e estrutura de diálogos no reparo de aspas."""
    text = "Fala que perdeu a abertura.”\n\n“Fala já íntegra.”"

    fixed, fixes = repair_missing_open_quotes_per_paragraph(text, logger=None, label="local")

    assert fixes == 1
    assert fixed.startswith("“Fala que perdeu a abertura.”")
    assert count_curly_quotes(fixed) == (2, 2)


def test_repair_missing_open_quote_keeps_paired_inline_quote_unchanged():
    """Confirma a preservação de aspas e estrutura de diálogos no reparo de aspas."""
    text = "A expressão “bem-vindo” continua corretamente delimitada."

    fixed, fixes = repair_missing_open_quotes_per_paragraph(text, logger=None, label="local")

    assert fixes == 0
    assert fixed == text


def test_fix_blank_lines_inside_quotes_collapses():
    """Valida a normalização de conteúdo válido no reparo de aspas."""
    text = "“Ele falou algo.\n\nContinuou a frase.”\n\nFora do diálogo."
    cleaned, fixes = fix_blank_lines_inside_quotes(text, logger=None, label="blank")
    assert fixes == 1
    assert "algo. Continuou" in cleaned
    # Fora das aspas, parágrafo permanece
    assert "\n\nFora do diálogo." in cleaned
