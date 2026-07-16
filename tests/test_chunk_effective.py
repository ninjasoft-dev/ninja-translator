import json
import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.llm_backend import LLMResponse
from tradutor.refine import refine_markdown_file
from tradutor.translate import translate_document
from tradutor.utils import setup_logging


class FakeTranslateBackend:
    """Simula uma tradução para medir o tamanho efetivo dos chunks."""

    def generate(self, prompt: str) -> LLMResponse:
        """Retorna uma tradução para medir o tamanho efetivo dos chunks."""
        return LLMResponse(
            text="### TEXTO_TRADUZIDO_INICIO\nTexto traduzido.\n\nOutra linha.\n### TEXTO_TRADUZIDO_FIM",
            latency=0.01,
        )


class FakeRefineBackend:
    """Simula um refino para medir o tamanho efetivo dos chunks."""

    backend = "ollama"
    model = "fake-refine"
    num_predict = 128
    temperature = 0.1
    repeat_penalty = 1.0

    def generate(self, prompt: str) -> LLMResponse:
        """Retorna um refino para medir o tamanho efetivo dos chunks."""
        return LLMResponse(text="Texto refinado simples.", latency=0.01)


def test_translate_metrics_include_effective_chunk(tmp_path: Path) -> None:
    """Confirma o registro do tamanho efetivo dos chunks no relatório de execução."""
    cfg = AppConfig(
        data_dir=tmp_path,
        output_dir=tmp_path,
        translate_chunk_chars=50,
        translate_num_predict=256,
    )
    logger = setup_logging(logging.ERROR)
    translate_document(
        pdf_text="Primeira frase. Segunda frase curta.",
        backend=FakeTranslateBackend(),
        cfg=cfg,
        logger=logger,
        source_slug="sample",
    )
    metrics_path = tmp_path / "sample_translate_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["effective_translate_chunk_chars"] == 50
    assert "max_chunk_chars_observed" in metrics


def test_refine_metrics_include_effective_chunk(tmp_path: Path) -> None:
    """Confirma o registro do tamanho efetivo dos chunks no relatório de execução."""
    cfg = AppConfig(
        data_dir=tmp_path,
        output_dir=tmp_path,
        refine_chunk_chars=40,
    )
    logger = setup_logging(logging.ERROR)
    input_md = tmp_path / "doc_pt.md"
    input_md.write_text("Um paragrafo curto.\n\nOutro paragrafo.", encoding="utf-8")
    output_md = tmp_path / "doc_pt_refinado.md"

    refine_markdown_file(
        input_path=input_md,
        output_path=output_md,
        backend=FakeRefineBackend(),
        cfg=cfg,
        logger=logger,
        cleanup_mode="off",
    )
    metrics_path = tmp_path / "doc_pt_refine_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["effective_refine_chunk_chars"] == 40
    assert "max_chunk_chars_observed" in metrics
