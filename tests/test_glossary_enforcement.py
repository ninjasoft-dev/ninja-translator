from tradutor.translate import enforce_canonical_terms


def test_enforce_canonical_glossary_term():
    """Substitui pelo termo canônico uma forma marcada como obrigatória."""
    terms = [
        {
            "key": "Lord of the Flies",
            "pt": "Senhor das Moscas",
            "aliases": ["Lord-of-the-Flies"],
            "enforce": True,
        }
    ]
    text = "O Lord of the Flies apareceu. Outro Lord-of-the-Flies caiu."

    normalized, replacements = enforce_canonical_terms(text, terms)

    assert "Senhor das Moscas" in normalized
    assert "Lord of the Flies" not in normalized
    assert replacements.get("Lord of the Flies", 0) >= 1


def test_enforce_bad_alias_without_expanding_valid_aliases():
    """Corrige aliases proibidos sem expandir formas válidas."""
    terms = [
        {
            "key": "Bram",
            "pt": "Bram",
            "aliases": ["Bane"],
            "bad_aliases": ["Banamente"],
        }
    ]
    text = "Banamente riu. Bane-san pegou a garrafa."

    normalized, replacements = enforce_canonical_terms(text, terms)

    assert "Bram riu." in normalized
    assert "Bane-san" in normalized
    assert "Banamente" not in normalized
    assert replacements == {"Banamente": 1}


def test_enforce_does_not_expand_name_alias_when_canonical_is_same_as_source():
    """Confirma a preservação de termos de glossário no tratamento do glossário."""
    terms = [
        {
            "key": "Mara Vale",
            "pt": "Mara Vale",
            "source_aliases": ["Mara", "Mara"],
            "enforce": True,
        }
    ]
    text = "Mara Vale falou com Mara."

    normalized, replacements = enforce_canonical_terms(text, terms)

    assert normalized == text
    assert replacements == {}


def test_enforce_explicit_target_replacement_for_selected_term():
    """Aplica a substituição contextual configurada para o termo selecionado."""
    terms = [
        {
            "key": "Paralyze",
            "pt": "Paralisar",
            "target_replacements": {"Paralizar": "Paralisar"},
        }
    ]

    normalized, replacements = enforce_canonical_terms("Usei Paralizar.", terms)

    assert normalized == "Usei Paralisar."
    assert replacements == {"Paralizar": 1}
