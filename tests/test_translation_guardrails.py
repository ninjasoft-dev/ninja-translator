import types

import pytest

from tradutor.config import AppConfig
from tradutor.translate import translate_document
from tradutor.utils import setup_logging


class _StubBackend:
    """Fornece respostas curtas para exercitar os limites de aceitação."""

    def __init__(self, output: str):
        """Inicializa a saída configurada e o contador de chamadas mantidos pelo dublê."""
        self.output = output
        self.calls = 0

    def generate(self, prompt: str):
        """Retorna respostas curtas para exercitar os limites de aceitação."""
        self.calls += 1
        return types.SimpleNamespace(text=self.output)


def test_translation_rejects_ratio_when_fail_on_error() -> None:
    """Confirma a detecção de problemas em integridade do conteúdo na tradução."""
    cfg = AppConfig(split_by_sections=False, translate_max_ratio=1.5, fail_on_chunk_error=True)
    logger = setup_logging()
    text = "Hello world."
    # Grande demais em relacao ao input para forcar ratio alto
    huge_output = (
        "### TEXTO_TRADUZIDO_INICIO\n" + ("Hello world. " * 20) + "\n### TEXTO_TRADUZIDO_FIM"
    )
    backend = _StubBackend(huge_output)

    with pytest.raises(RuntimeError):
        translate_document(
            pdf_text=text,
            backend=backend,
            cfg=cfg,
            logger=logger,
            already_preprocessed=True,
        )
    assert backend.calls >= 0


def test_translation_inserts_placeholder_when_rejected() -> None:
    """Valida a normalização de marcadores de controle na tradução."""
    cfg = AppConfig(split_by_sections=False, translate_max_ratio=1.5, fail_on_chunk_error=False)
    logger = setup_logging()
    text = "Hello again."
    huge_output = (
        "### TEXTO_TRADUZIDO_INICIO\n" + ("Hello again. " * 20) + "\n### TEXTO_TRADUZIDO_FIM"
    )
    backend = _StubBackend(huge_output)

    result = translate_document(
        pdf_text=text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        already_preprocessed=True,
    )

    assert "[CHUNK_TRANSLATION_REJECTED_1]" in result
    assert backend.calls >= 0
