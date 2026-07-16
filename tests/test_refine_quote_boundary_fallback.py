import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.llm_backend import LLMResponse
from tradutor.refine import refine_markdown_file


class _MalformedQuoteBackend:
    """Mantém um limite de aspas malformado para forçar o fallback do refino."""

    backend = "fake"
    model = "fake-refine"
    temperature = 0.2
    num_predict = 256
    repeat_penalty = 1.0

    def generate(self, prompt: str) -> LLMResponse:
        """Retorna uma resposta que mantém um limite de aspas malformado para forçar o fallback do refino."""
        return LLMResponse(
            text="### TEXTO_REFINADO_INICIO\n”“Ah, tudo bem.”\n### TEXTO_REFINADO_FIM",
            latency=0.01,
        )


def test_refine_falls_back_when_malformed_quote_boundary_persists(
    tmp_path: Path,
) -> None:
    """Confirma o fallback seguro diante de problemas em aspas e estrutura de diálogos no refino."""
    input_path = tmp_path / "sample_pt.md"
    output_path = tmp_path / "sample_pt_refinado.md"
    input_path.write_text("“Ah, tudo bem.”", encoding="utf-8")
    cfg = AppConfig(
        output_dir=tmp_path,
        refine_chunk_chars=500,
        max_retries=1,
        refine_guardrails="strict",
    )

    refine_markdown_file(
        input_path=input_path,
        output_path=output_path,
        backend=_MalformedQuoteBackend(),
        cfg=cfg,
        logger=logging.getLogger("refine-quote-boundary"),
        cleanup_mode="off",
    )

    output = output_path.read_text(encoding="utf-8")
    assert output == "“Ah, tudo bem.”"
    assert "”“" not in output
