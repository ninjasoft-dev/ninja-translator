import json
import logging
from pathlib import Path

from tradutor.glossary_utils import (
    build_glossary_state,
    format_manual_pairs_for_translation,
    resolve_manual_glossary_path,
)


def test_glossary_loader_preserves_enforcement_metadata(tmp_path: Path) -> None:
    """Confirma a preservação de termos de glossário no tratamento do glossário."""
    glossary_path = tmp_path / "glossary.json"
    glossary_path.write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "key": "Suou Kayako",
                        "pt": "Suou Kayako",
                        "category": "personagem",
                        "gender": "feminino",
                        "type": "personagem",
                        "enforce": True,
                        "aliases": ["Kayako Suou", "Kayado"],
                        "source_aliases": ["Kayako Suou"],
                        "bad_aliases": ["Kayado"],
                        "allowed_target_aliases": ["Kayako"],
                        "source_case_sensitive": True,
                        "notes": "Aliada de Mara.",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = build_glossary_state(
        manual_path=glossary_path,
        dynamic_path=None,
        logger=logging.getLogger("test-glossary-loader"),
    )

    assert state is not None
    term = state.manual_terms[0]
    assert term["enforce"] is True
    assert term["gender"] == "feminino"
    assert term["type"] == "personagem"
    assert term["source_aliases"] == ["Kayako Suou"]
    assert term["aliases"] == ["Kayako Suou"]
    assert term["bad_aliases"] == ["Kayado"]
    assert term["allowed_target_aliases"] == ["Kayako"]
    assert term["source_case_sensitive"] is True


def test_translation_glossary_prompt_includes_metadata() -> None:
    """Confirma o registro correto de termos de glossário no tratamento do glossário."""
    block = format_manual_pairs_for_translation(
        [
            {
                "key": "Four Tower Wardens",
                "pt": "Quatro Santos",
                "category": "organização",
                "gender": "masculino plural",
                "enforce": True,
                "notes": "Não usar Quatro Sábios.",
            }
        ],
        limit=None,
    )

    assert "categoria: organização" in block
    assert "genero: masculino plural" in block
    assert "uso obrigatorio" in block
    assert "Não usar Quatro Sábios." in block


def test_translation_glossary_prompt_includes_forbidden_aliases() -> None:
    """Confirma o registro correto de termos de glossário no tratamento do glossário."""
    block = format_manual_pairs_for_translation(
        [
            {
                "key": "Order of Aurelia",
                "pt": "Ordem de Aurelia",
                "bad_aliases": ["Discípulos de Aurelia"],
            }
        ],
        limit=None,
    )

    assert "nao usar: Discípulos de Aurelia" in block


def test_resolve_manual_glossary_path_prefers_explicit_path(tmp_path: Path) -> None:
    """Prioriza o caminho de glossário informado explicitamente."""
    explicit = tmp_path / "manual.json"

    assert resolve_manual_glossary_path(explicit) == explicit
