import json
import logging
from pathlib import Path

from tradutor.cache_utils import chunk_hash, set_cache_base_dir
from tradutor.config import AppConfig
from tradutor.refine import refine_markdown_file


class _StubBackend:
    """Conta chamadas feitas após a rejeição de um cache incompatível."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo, os parâmetros de geração e o contador de chamadas."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        """Conta chamadas feitas após a rejeição de um cache incompatível e retorna a resposta configurada."""
        self.calls += 1
        text = "Texto refinado pelo backend."
        return type("Resp", (), {"text": text})


def test_refine_cache_ignores_prompt_signature_mismatch(tmp_path: Path) -> None:
    """Valida as regras de compatibilidade do cache no refino."""
    cfg = AppConfig(output_dir=tmp_path)
    backend = _StubBackend()
    logger = logging.getLogger("refine-cache")

    input_path = tmp_path / "input.md"
    output_path = tmp_path / "output.md"
    input_text = "Texto base para refino."
    input_path.write_text(input_text, encoding="utf-8")

    set_cache_base_dir(tmp_path)
    h = chunk_hash(input_text)
    cache_path = tmp_path / "cache_refine" / f"{h}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "hash": h,
        "raw_output": "RAW",
        "final_output": "CACHED_SHOULD_BE_IGNORED",
        "timestamp": "now",
        "metadata": {
            "backend": backend.backend,
            "model": backend.model,
            "num_predict": backend.num_predict,
            "temperature": backend.temperature,
            "repeat_penalty": backend.repeat_penalty,
            "guardrails": getattr(cfg, "refine_guardrails", "strict"),
            "prompt_hash": "legacy-prompt",
            "pipeline_version": "0",
        },
    }
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8")

    refine_markdown_file(
        input_path=input_path,
        output_path=output_path,
        backend=backend,
        cfg=cfg,
        logger=logger,
        progress_path=None,
        resume_manifest=None,
        normalize_paragraphs=False,
        glossary_state=None,
        debug_refine=False,
        parallel_workers=1,
        preprocess_advanced=False,
        debug_chunks=False,
        cleanup_mode="off",
    )

    output_text = output_path.read_text(encoding="utf-8")
    assert "CACHED_SHOULD_BE_IGNORED" not in output_text
    assert "Texto refinado pelo backend." in output_text
    assert backend.calls >= 1
