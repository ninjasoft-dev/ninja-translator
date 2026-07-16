"""Testes da revisão determinística aplicada após tradução e refino."""

from tradutor.post_translation_review import (
    finalize_translation_text,
    review_translation_text,
)
from tradutor.translate import ensure_section_heading, source_heading_to_pt


def test_source_heading_to_pt() -> None:
    """Traduz títulos estruturais conhecidos para Markdown em PT-BR."""
    assert source_heading_to_pt("Chapter 5:") == "# Capítulo 5:"
    assert source_heading_to_pt("Epilogue") == "# Epílogo"
    assert source_heading_to_pt("Afterword") == "# Pós-escrito"
    assert source_heading_to_pt("Full Text") is None


def test_ensure_section_heading_inserts_missing_heading() -> None:
    """Insere o título da seção quando a saída do chunk o omite."""
    text, changed = ensure_section_heading("A porta abriu.", "Chapter 5:")

    assert changed is True
    assert text == "# Capítulo 5:\n\nA porta abriu."


def test_ensure_section_heading_does_not_duplicate_heading() -> None:
    """Preserva a saída quando o cabeçalho esperado já existe."""
    text, changed = ensure_section_heading("# Capítulo 2:\n\nTexto.", "Chapter 2:")

    assert changed is False
    assert text == "# Capítulo 2:\n\nTexto."


def test_review_restores_only_unambiguous_initial_heading() -> None:
    """Restaura prólogo no início, mas não adivinha posições de capítulos."""
    text = "O sino tocou."
    sections = [{"title": "Prologue"}, {"title": "Chapter 1:"}]

    reviewed, report = review_translation_text(text, sections=sections)

    assert reviewed.startswith("# Prólogo\n\nO sino tocou.")
    assert "# Capítulo 1:" not in reviewed
    assert report.heading_fixes == 1


def test_review_applies_generic_editorial_replacements() -> None:
    """Corrige resíduos de idioma sem depender de personagens ou títulos."""
    text = "Eu wish que fosse bipede, não semi-deuses—or meow. Isso é sus AF."

    reviewed, report = review_translation_text(text)

    assert reviewed == (
        "Quem me dera que fosse bípede, não semideuses — ou miau. Isso é suspeita pra caramba."
    )
    assert report.text_replacements


def test_review_fixes_gendered_articles_from_glossary() -> None:
    """Usa o gênero do glossário para ajustar artigos e predicativos."""
    terms = [
        {
            "key": "Bruma",
            "pt": "Bruma",
            "category": "criatura",
            "gender": "feminino",
        }
    ]

    reviewed, _ = review_translation_text(
        "O Bruma avançou. Não confio no Bruma como aliado.",
        glossary_terms=terms,
    )

    assert reviewed == "A Bruma avançou. Não confio na Bruma como aliada."


def test_review_applies_contextual_glossary_replacement() -> None:
    """Prefere uma substituição contextual à troca cega pelo termo canônico."""
    terms = [
        {
            "key": "Last Trial",
            "pt": "Prova Final",
            "target_replacements": {
                "Após a Provação Derradeira": "Após a Prova Final",
            },
            "bad_aliases": ["Last Trial"],
        }
    ]

    reviewed, report = review_translation_text(
        "Após a Provação Derradeira, lembramos do Last Trial.",
        glossary_terms=terms,
    )

    assert reviewed == "Após a Prova Final, lembramos do Prova Final."
    assert report.glossary_replacements


def test_review_collapses_duplicate_canonical_names() -> None:
    """Colapsa expansões duplicadas de nomes canônicos de personagens."""
    terms = [
        {
            "key": "Mara Vale",
            "pt": "Mara Vale",
            "category": "personagem",
            "source_aliases": ["Mara", "Vale"],
        }
    ]

    reviewed, report = review_translation_text(
        "Mara Mara Vale respondeu. Mara Vale Vale saiu.",
        glossary_terms=terms,
    )

    assert reviewed == "Mara Vale respondeu. Mara Vale saiu."
    assert report.text_replacements


def test_finalize_review_normalizes_all_caps_known_names() -> None:
    """Restaura a caixa de entidades em CAPS usando apenas o glossário."""
    terms = [
        {
            "key": "Mara Vale",
            "pt": "Mara Vale",
            "category": "personagem",
            "source_aliases": ["Mara"],
        },
        {"key": "Pip", "pt": "Pip", "category": "criatura"},
    ]

    reviewed, report = finalize_translation_text("MARA falou com PIP.", glossary_terms=terms)

    assert reviewed == "Mara falou com Pip."
    assert report["editorial"]["all_caps_name_replacements"]


def test_review_normalizes_case_only_forbidden_alias() -> None:
    """Corrige a caixa de uma forma proibida sem alterar a canônica já válida."""
    terms = [
        {
            "key": "Fio Limite",
            "pt": "Fio Limite",
            "category": "técnica",
            "bad_aliases": ["fio limite"],
        }
    ]

    reviewed, report = review_translation_text("A técnica fio limite pesa.", glossary_terms=terms)

    assert reviewed == "A técnica Fio Limite pesa."
    assert report.glossary_replacements == {"fio limite->Fio Limite": 1}


def test_finalize_review_repairs_one_dangling_curly_quote() -> None:
    """Fecha uma única aspa curva pendente sem reescrever a fala."""
    reviewed, report = finalize_translation_text(
        "“Primeira fala sem fechamento.\n\n“Segunda fala.”"
    )

    assert reviewed == "“Primeira fala sem fechamento.”\n\n“Segunda fala.”"
    assert report["quote_balance_fixed"] is True


def test_finalize_review_restores_one_missing_open_quote() -> None:
    """Restaura uma abertura perdida quando a posição é inequívoca."""
    reviewed, report = finalize_translation_text("Fala sem abertura.”\n\n“Outra fala.”")

    assert reviewed == "“Fala sem abertura.”\n\n“Outra fala.”"
    assert report["quote_balance_fixed"] is True


def test_finalize_review_collapses_blank_line_inside_quote() -> None:
    """Remove um parágrafo vazio criado dentro da mesma fala."""
    reviewed, report = finalize_translation_text("“Primeira parte.\n\nContinuação da mesma fala.”")

    assert reviewed == "“Primeira parte. Continuação da mesma fala.”"
    assert report["quote_blank_lines_fixed"] == 1
