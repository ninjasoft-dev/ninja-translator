from tradutor.text_postprocess import (
    apply_structural_normalizers,
    normalize_dialogue_breaks,
    strip_stray_triple_quotes,
)


def test_normalize_dialogue_breaks_inserts_blank_line():
    """Valida a normalização de aspas e estrutura de diálogos nos normalizadores textuais."""
    text = "“…Shut up. I’ll kill you.” “You say you’ll kill people all the time, little man."
    normalized, stats = normalize_dialogue_breaks(text)
    assert ".”\n\n“" in normalized
    assert stats["dialogue_splits"] > 0
    assert "Shut up." in normalized
    assert "You say you’ll kill people all the time, little man." in normalized


def test_strip_stray_triple_quotes_removes_trailing():
    """Valida a remoção segura de conteúdo válido nos normalizadores textuais."""
    cleaned, stats = strip_stray_triple_quotes('Daren recuou."""')
    assert cleaned == "Daren recuou."
    assert stats["triple_quotes_removed"] == 1


def test_structural_normalizers_preserve_regular_quotes():
    """Confirma a preservação de conteúdo válido nos normalizadores textuais."""
    text = 'Ele disse: "ok" e sorriu.'
    cleaned, stats = apply_structural_normalizers(text)
    assert cleaned == text
    assert stats["dialogue_splits"] == 0
    assert stats["triple_quotes_removed"] == 0


def test_normalize_dialogue_specific_case_gyaaahhh():
    """Valida as regras de aspas e estrutura de diálogos nos normalizadores textuais."""
    text = "“Gyaaahhh!” “De novo emburrado..."
    normalized, stats = normalize_dialogue_breaks(text)
    assert "“Gyaaahhh!”\n\n“De novo emburrado..." in normalized
    assert stats["dialogue_splits"] >= 1


def test_normalize_dialogue_specific_case_kill_you():
    """Valida as regras de aspas e estrutura de diálogos nos normalizadores textuais."""
    text = "... Eu vou te matar.” “Você fica ameaçando ..."
    normalized, stats = normalize_dialogue_breaks(text)
    assert "te matar.”\n\n“Você fica ameaçando" in normalized
    assert stats["dialogue_splits"] >= 1
