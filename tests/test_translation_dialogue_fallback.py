import re

import tradutor.translate as translate
from tradutor.config import AppConfig
from tradutor.utils import setup_logging


class _StubBackend:
    """Traduz blocos de diálogo separadamente durante o fallback."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo e os parâmetros de geração usados pelo dublê."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 128
        self.temperature = 0.1
        self.repeat_penalty = 1.0

    def generate(self, prompt: str):
        """Retorna uma resposta que traduz blocos de diálogo separadamente durante o fallback."""
        raise RuntimeError("Should not call generate directly")


def _extract_block(prompt: str) -> str:
    """Extrai bloco."""
    match = re.search(r"TEXTO A SER TRADUZIDO:\n\"\"\"(.*?)\"\"\"", prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def test_dialogue_split_fallback_translates_all_blocks(monkeypatch) -> None:
    """Confirma o fallback seguro diante de problemas em aspas e estrutura de diálogos na tradução."""
    backend = _StubBackend()
    logger = setup_logging()
    cfg = AppConfig(split_by_sections=False, max_retries=1, translate_chunk_chars=5000)

    def fake_call_with_retry(backend, prompt, cfg, logger, label):
        """Substitui a chamada ao modelo por uma resposta determinística."""
        block_text = _extract_block(prompt)
        raw = f"### TEXTO_TRADUZIDO_INICIO\n{block_text}\n### TEXTO_TRADUZIDO_FIM"
        return raw, block_text, 1, None

    monkeypatch.setattr(translate, "_call_with_retry", fake_call_with_retry)
    monkeypatch.setattr(
        translate, "needs_retry", lambda *a, **k: (True, "omissao_dialogo_guardrail")
    )

    chunk_text = "\n".join(
        [
            '"Eh?"',
            '"Hm?"',
            '"Goddess...?"',
        ]
    )

    result = translate.translate_document(
        pdf_text=chunk_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        already_preprocessed=True,
    )

    assert "Eh?" in result
    assert "Hm?" in result
    assert "Goddess...?" in result
