"""Testes das opções usadas para iniciar e documentar a interface desktop."""

import pytest

from tradutor.gui_app import TranslatorApp, resolve_appearance_mode


@pytest.mark.parametrize(
    ("configured_value", "expected_theme"),
    [
        ("light", "light"),
        ("DARK", "dark"),
        ("outro", "dark"),
        (None, "dark"),
    ],
)
def test_resolve_appearance_mode_uses_safe_fallback(
    configured_value: str | None,
    expected_theme: str,
) -> None:
    """Aceita apenas temas suportados e mantém o escuro como padrão."""
    assert resolve_appearance_mode(configured_value) == expected_theme


def test_gui_accepts_explicit_config_path(tmp_path, monkeypatch) -> None:
    """Permite iniciar a interface com uma configuração genérica e isolada."""
    config_path = tmp_path / "capture.yaml"
    config_path.write_text("translate_model: modelo-de-demonstracao\n", encoding="utf-8")
    monkeypatch.setenv("NINJA_TRANSLATOR_CONFIG", str(config_path))

    config = TranslatorApp._load_project_config()

    assert config.translate_model == "modelo-de-demonstracao"
