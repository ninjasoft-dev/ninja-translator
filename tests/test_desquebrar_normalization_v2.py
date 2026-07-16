from tradutor.desquebrar import (
    normalize_hardwrap_joins,
    normalize_internal_hyphen_by_dominance,
    normalize_scene_separators,
)


def test_hardwrap_join():
    """Reconstrói uma frase dividida por uma quebra rígida de linha."""
    text = "... voice, which\nmysteriously came ..."

    normalized, joins = normalize_hardwrap_joins(text)

    assert "which mysteriously" in normalized
    assert joins == 1


def test_internal_hyphen_dominance_applies():
    """Valida as regras de artefatos de extração e OCR na reconstrução de parágrafos."""
    text = "understand\nunderstand\nunderstand\nunder-stand"

    normalized, stats = normalize_internal_hyphen_by_dominance(text)

    assert "under-stand" not in normalized
    assert "understand" in normalized
    assert stats.get("total", 0) >= 1


def test_internal_hyphen_keeps_legit_compound():
    """Confirma a preservação de artefatos de extração e OCR na reconstrução de parágrafos."""
    text = "demi-humans are here.\nDemihumans are rare."

    normalized, stats = normalize_internal_hyphen_by_dominance(text)

    assert "demi-humans" in normalized
    assert stats.get("total", 0) == 0


def test_internal_hyphen_keeps_honorific():
    """Confirma a preservação de artefatos de extração e OCR na reconstrução de parágrafos."""
    text = "Zine-sama greeted everyone."

    normalized, stats = normalize_internal_hyphen_by_dominance(text)

    assert "Zine-sama" in normalized
    assert stats.get("total", 0) == 0


def test_scene_separator_isolated():
    """Valida as regras de limites estruturais na reconstrução de parágrafos."""
    text = "Line A\n***\nLine B"

    normalized, fixes = normalize_scene_separators(text)

    assert "\n\n***\n\n" in normalized
    assert fixes >= 1
