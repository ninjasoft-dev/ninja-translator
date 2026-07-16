import logging
from pathlib import Path

from tradutor.cache_utils import set_cache_base_dir
from tradutor.config import AppConfig
from tradutor.refine import refine_section


class _ResidualEnglishRefineBackend:
    """Devolve inglês residual na primeira tentativa de refino."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo, os parâmetros de geração e o contador de chamadas."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        """Retorna inglês residual na primeira tentativa de refino."""
        self.calls += 1
        if self.calls == 1:
            text = (
                "### TEXTO_REFINADO_INICIO\n"
                "“I have no desire to die,” replied Lina calmly, choosing not to answer directly.\n"
                "### TEXTO_REFINADO_FIM"
            )
        else:
            text = (
                "### TEXTO_REFINADO_INICIO\n"
                "“Não tenho vontade de morrer”, respondeu Lina com calma, sem responder diretamente.\n"
                "### TEXTO_REFINADO_FIM"
            )
        return type("Resp", (), {"text": text})


def test_refine_retries_on_residual_english_sentence(tmp_path: Path) -> None:
    """Confirma a detecção de problemas em idioma residual no refino."""
    set_cache_base_dir(tmp_path)
    cfg = AppConfig(output_dir=tmp_path, max_retries=2, refine_chunk_chars=2000)
    backend = _ResidualEnglishRefineBackend()
    logger = logging.getLogger("refine-residual-english")
    body = "“I have no desire to die,” replied Lina calmly, choosing not to answer directly."

    result = refine_section(
        title="",
        body=body,
        backend=backend,
        cfg=cfg,
        logger=logger,
        index=1,
        total=1,
    )

    assert "Não tenho vontade de morrer" in result
    assert "I have no desire" not in result
    assert backend.calls >= 2
