"""Testes das correções conservadoras de estrutura Markdown."""

from tradutor.structure_normalizer import normalize_structure


def test_heading_with_inline_text_is_split() -> None:
    """Separa o corpo narrativo colado ao título de prólogo."""
    text = "Prólogo MARA VALE ACENOU.\n\nOutro parágrafo."

    result = normalize_structure(text)

    assert [line for line in result.splitlines() if line] == [
        "Prólogo",
        "MARA VALE ACENOU.",
        "Outro parágrafo.",
    ]


def test_normalize_structure_is_idempotent() -> None:
    """Produz o mesmo texto quando aplicada mais de uma vez."""
    text = "Prólogo\n\nMara Vale acenou.\n\nOutro parágrafo."

    once = normalize_structure(text)

    assert normalize_structure(once) == once


def test_chapter_subtitle_is_merged_into_markdown_heading() -> None:
    """Anexa um subtítulo curto ao cabeçalho de capítulo anterior."""
    text = "# Capítulo 1:\n\nA Última Lanterna\n\nDepois que Mara acordou."

    result = normalize_structure(text)

    assert result.startswith("# Capítulo 1: A Última Lanterna")
    assert "Depois que Mara acordou." in result


def test_duplicate_generic_heading_is_removed_after_titled_heading() -> None:
    """Remove o cabeçalho genérico duplicado após um título completo."""
    text = "# Capítulo 1: A Última Lanterna\n\n# Capítulo 1:\n\nDEPOIS QUE MARA acordou."

    result = normalize_structure(text)

    assert result.count("# Capítulo 1:") == 1
    assert "Depois que Mara acordou." in result


def test_leading_small_caps_are_normalized_without_known_names() -> None:
    """Restaura caixa normal em uma abertura temporal com nome genérico."""
    result = normalize_structure("DEPOIS QUE MARA atravessou a ponte.")

    assert result == "Depois que Mara atravessou a ponte."


def test_character_time_label_is_split_into_structure() -> None:
    """Transforma nome e marcador temporal colados em dois blocos."""
    result = normalize_structure("Mara Vale ALGUM TEMPO ANTES, há um tempo…")

    assert result == "## Mara Vale\n\nAlgum tempo antes…"


def test_scene_separator_glued_to_narration_is_isolated() -> None:
    """Separa o marcador de cena do texto colocado na mesma linha."""
    assert normalize_structure("*** A cena continua.") == "***\n\nA cena continua."
