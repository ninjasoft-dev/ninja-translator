from tradutor.glossary_audit import audit_glossary_data, is_probably_portuguese_alias


def test_glossary_audit_finds_ambiguous_and_portuguese_source_aliases() -> None:
    """Relata aliases ambíguos e formas em português usadas como origem."""
    data = {
        "terms": [
            {
                "key": "Grey Raven",
                "pt": "Grey Raven",
                "aliases": ["Lord of the Flies", "Senhor das Moscas"],
            },
            {
                "key": "Lord of the Flies",
                "pt": "Senhor das Moscas",
                "aliases": ["Lord-of-the-Flies"],
            },
        ]
    }

    report = audit_glossary_data(data)

    assert report["summary"]["terms"] == 2
    assert report["summary"]["ambiguous_source_aliases"] == 1
    assert report["ambiguous_source_aliases"][0]["value"] == "Lord of the Flies"
    assert report["summary"]["portuguese_source_aliases"] == 1
    assert report["portuguese_source_aliases"][0]["alias"] == "Senhor das Moscas"


def test_portuguese_alias_heuristic_ignores_english_noise() -> None:
    """Valida as regras de ruído e marcas d'água no tratamento do glossário."""
    assert is_probably_portuguese_alias("Guardiões das Quatro Torres")
    assert is_probably_portuguese_alias("Caçador de Grandes Feras")
    assert not is_probably_portuguese_alias("Mya-a-ah")
    assert not is_probably_portuguese_alias("Lord of the Flies")
