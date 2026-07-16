from pathlib import Path

from tradutor.config import load_config


def test_load_config_defaults_when_keys_missing(tmp_path: Path) -> None:
    """Valida as regras de configuração e caminhos no carregamento da configuração."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("translate_model: custom-model\n", encoding="utf-8")

    cfg = load_config(cfg_path)

    assert cfg.translate_model == "custom-model"
    assert cfg.use_desquebrar is True
    assert cfg.desquebrar_repeat_penalty == 1.08
    assert cfg.cleanup_before_refine == "auto"


def test_load_config_overrides_new_fields(tmp_path: Path) -> None:
    """Valida as regras de configuração e caminhos no carregamento da configuração."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """\nuse_desquebrar: false\ncleanup_before_refine: on\ndesquebrar_repeat_penalty: 1.5\ntranslate_backend: gemini\n""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.use_desquebrar is False
    assert cfg.cleanup_before_refine == "on"
    assert cfg.desquebrar_repeat_penalty == 1.5
    assert cfg.translate_backend == "gemini"
