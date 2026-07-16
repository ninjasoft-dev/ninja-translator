import logging
import types

from tradutor.config import AppConfig
from tradutor.translate import translate_document
from tradutor.utils import setup_logging


class _StubBackend:
    """Registra chamadas quando a divisão por seções é considerada insegura."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo e os parâmetros de geração usados pelo dublê."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 256
        self.temperature = 0.1
        self.repeat_penalty = 1.0

    def generate(self, prompt: str):
        """Registra chamadas quando a divisão por seções é considerada insegura e retorna a resposta configurada."""
        long_pt = "X" * 4000
        return types.SimpleNamespace(
            text=f"### TEXTO_TRADUZIDO_INICIO\n{long_pt}\n### TEXTO_TRADUZIDO_FIM"
        )


def test_translate_disables_split_when_sections_suspect(caplog) -> None:
    """Valida as regras de conteúdo válido na tradução."""
    cfg = AppConfig(split_by_sections=True, translate_chunk_chars=5000)
    logger = setup_logging(logging.ERROR)
    long_preamble = "A" * 4000
    text = f"{long_preamble}\n\nEpilogue\nThe end."

    with caplog.at_level(logging.WARNING):
        translate_document(
            pdf_text=text,
            backend=_StubBackend(),
            cfg=cfg,
            logger=logger,
            already_preprocessed=True,
        )
    assert any("split_by_sections fallback" in rec.message for rec in caplog.records)
