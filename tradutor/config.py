"""
Configurações centrais do pipeline de tradução e refino.

Mantém valores padrão em um único lugar para facilitar manutenção e leitura.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

BackendType = Literal["ollama", "gemini", "openai"]
GuardrailsType = Literal["strict", "relaxed", "off"]
DialogueGuardrailsType = Literal["strict", "relaxed", "off"]
OllamaApiMode = Literal["generate", "chat"]
DEFAULT_CONFIG_PATHS = (Path("config.yaml"), Path("config.yml"))
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    """Valores padrão para todo o pipeline."""

    # Diretórios padrão
    data_dir: Path = Path("data")
    output_dir: Path = Path("saida")
    font_dir: Path = Path(".cache/fonts")

    # Modelos e idioma de origem
    source_language: str = "auto"
    translate_backend: BackendType = "ollama"
    translate_model: str = "gemma3:4b"
    refine_backend: BackendType = "ollama"
    refine_model: str = "gemma3:4b"
    desquebrar_backend: BackendType = "ollama"
    desquebrar_model: str = "gemma3:4b"
    dump_chunks: bool = False
    refine_guardrails: GuardrailsType = "strict"
    refine_after_translate: bool = False
    use_desquebrar: bool = True
    desquebrar_mode: Literal["safe", "llm"] = "llm"
    fail_on_chunk_error: bool = False

    # Temperaturas
    translate_temperature: float = 0.15
    refine_temperature: float = 0.30
    desquebrar_temperature: float = 0.08
    translate_repeat_penalty: float = 1.1
    refine_repeat_penalty: float | None = None
    desquebrar_repeat_penalty: float | None = 1.08
    translate_num_ctx: int | None = None
    refine_num_ctx: int | None = None
    desquebrar_num_ctx: int | None = None
    ollama_keep_alive: str | int = "30m"
    ollama_api_mode: OllamaApiMode = "generate"
    ollama_think: bool | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    skip_front_matter: bool = True
    split_by_sections: bool = True
    translate_allow_adaptation: bool = False
    use_translation_repair: bool = True
    translate_context_paragraphs: int = 3
    translate_context_chars: int = 1200
    translate_context_include_pt: bool = True
    translate_dialogue_guardrails: DialogueGuardrailsType = "strict"
    translate_dialogue_retry_temps: list[float] = field(default_factory=list)
    translate_dialogue_split_fallback: bool = True
    translate_glossary_match_limit: int = 80
    translate_glossary_fallback_limit: int = 30
    translate_max_ratio: float = 1.8

    # Comprimento de saída
    translate_num_predict: int = 3072
    refine_num_predict: int = 1024
    desquebrar_num_predict: int = 1024

    # Chunk sizes
    translate_chunk_chars: int = 2400
    refine_chunk_chars: int = 2400
    desquebrar_chunk_chars: int = 2400

    # Tentativas e backoff
    max_retries: int = 3
    initial_backoff: float = 1.5
    backoff_factor: float = 1.8

    # Timeouts
    request_timeout: int = 120

    # Cleanup deterministico antes do refine
    cleanup_before_refine: str | bool = (
        "auto"  # valores: off | auto | on (bool suportado por configs antigas)
    )

    # PDF
    pdf_title_font_size: int = 16
    pdf_heading_font_size: int = 13
    pdf_body_font_size: int = 11
    pdf_enabled: bool = False
    pdf_font_file: str = ""
    pdf_font_size: int = 12
    pdf_font_leading: float = 15.0
    pdf_font_fallbacks: list[str] = field(default_factory=list)
    pdf_margin: int = 48
    pdf_author: str = ""
    pdf_language: str = "pt-BR"

    # Preprocess noise glossary
    preprocess_noise_glossary_path: Path | None = None

    # Debug completo
    debug_max_chunks: int | None = None
    debug_max_chars_per_file: int | None = 200000
    debug_store_llm_raw: bool = True


def ensure_paths(cfg: AppConfig) -> None:
    """Garante que os diretórios principais existam."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.font_dir.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Carrega configurações a partir de YAML, com fallback para valores padrão.
    """
    base = AppConfig()

    path: Path | None = None
    if config_path:
        candidate = Path(config_path)
        if candidate.exists():
            path = candidate
    else:
        for candidate in DEFAULT_CONFIG_PATHS:
            if candidate.exists():
                path = candidate
                break

    if path is None:
        return base

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return base
    except Exception as exc:  # pragma: no cover - I/O edge case
        log.warning("Falha ao ler config %s; usando defaults. Erro: %s", path, exc)
        return base

    if not isinstance(data, dict):
        log.warning("Config %s tem formato inesperado; usando defaults.", path)
        return base

    overrides = {}
    # suporte a bloco pdf_font: {file, size, leading}
    pdf_font_block = data.get("pdf_font")
    if isinstance(pdf_font_block, dict):
        if "file" in pdf_font_block:
            overrides["pdf_font_file"] = pdf_font_block.get("file", "")
        if "size" in pdf_font_block:
            overrides["pdf_font_size"] = pdf_font_block.get("size", base.pdf_font_size)
        if "leading" in pdf_font_block:
            overrides["pdf_font_leading"] = pdf_font_block.get("leading", base.pdf_font_leading)
    for key, value in data.items():
        if key == "pdf_font":
            continue
        if key not in base.__dict__:
            continue
        if key.endswith("_dir"):
            overrides[key] = Path(value)
        else:
            overrides[key] = value

    merged = {**base.__dict__, **overrides}
    # compat: cleanup_before_refine bool -> string
    cleanup_val = merged.get("cleanup_before_refine")
    if isinstance(cleanup_val, bool):
        merged["cleanup_before_refine"] = "on" if cleanup_val else "off"
    return AppConfig(**merged)
