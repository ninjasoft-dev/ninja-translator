from tradutor.glossary_utils import select_terms_for_chunk, select_terms_for_target_text


def test_glossary_word_boundary_matching() -> None:
    """Respeita limites de palavra ao localizar termos do glossário."""
    manual_terms = [{"key": "Art", "pt": "Arte"}]
    chunk_text = "This is a partial mention that should not match."
    selected, matched = select_terms_for_chunk(
        manual_terms, chunk_text, match_limit=80, fallback_limit=0
    )
    assert matched == 0
    assert selected == []


def test_glossary_case_sensitive_source_term_does_not_match_common_word() -> None:
    """Confirma a preservação de termos de glossário no tratamento do glossário."""
    manual_terms = [{"key": "Freeze", "pt": "Congelar", "source_case_sensitive": True}]

    selected, matched = select_terms_for_chunk(
        manual_terms,
        "After she watched his body freeze, she lost consciousness.",
        match_limit=80,
        fallback_limit=0,
    )

    assert matched == 0
    assert selected == []

    selected, matched = select_terms_for_chunk(
        manual_terms,
        "Cael cast Freeze on the undead enemy.",
        match_limit=80,
        fallback_limit=0,
    )

    assert matched == 1
    assert selected == manual_terms


def test_target_text_selection_uses_pt_forms_without_unrelated_fallback() -> None:
    """Confirma o fallback seguro diante de problemas em conteúdo válido no tratamento do glossário."""
    terms = [
        {"key": "Last Trial", "pt": "Confronto Final"},
        {"key": "Freeze", "pt": "Congelar", "bad_aliases": ["Congelamento"]},
        {"key": "Mira", "pt": "Império de Mira", "allowed_target_aliases": ["Mira"]},
    ]

    selected, matched = select_terms_for_target_text(
        terms,
        "Após o Confronto Final, Mira voltou ao acampamento.",
    )

    assert matched == 2
    assert [term["key"] for term in selected] == ["Last Trial", "Mira"]
