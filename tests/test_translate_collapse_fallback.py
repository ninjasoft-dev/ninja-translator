import logging
from pathlib import Path

import tradutor.translate as translate_module
from tradutor.config import AppConfig
from tradutor.translate import translate_document


class _StubBackend:
    """Produz uma resposta colapsada para acionar o fallback da tradução."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo e os parâmetros de geração usados pelo dublê."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0

    def generate(self, prompt: str):
        """Retorna uma resposta colapsada para acionar o fallback da tradução."""
        text = "### TEXTO_TRADUZIDO_INICIO\nsaida qualquer\n### TEXTO_TRADUZIDO_FIM"
        return type("Resp", (), {"text": text})


def test_translate_collapse_fallback_uses_original(monkeypatch, tmp_path: Path) -> None:
    """Confirma o fallback seguro diante de problemas em repetições indevidas na tradução."""
    cfg = AppConfig(output_dir=tmp_path, max_retries=1, split_by_sections=False)
    backend = _StubBackend()
    logger = logging.getLogger("collapse-fallback")
    input_text = "Texto narrativo simples sem dialogo."

    monkeypatch.setattr(translate_module, "detect_model_collapse", lambda *a, **k: True)

    result = translate_document(
        pdf_text=input_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        progress_path=None,
        resume_manifest=None,
        glossary_text=None,
        debug_translation=False,
        parallel_workers=1,
        debug_chunks=False,
        already_preprocessed=True,
    )

    assert result.strip() == input_text.strip()
