import logging
import re
import sys

# Dublê mínimo do PyMuPDF para o teste de fumaça não depender da biblioteca externa.
if "fitz" not in sys.modules:

    class _DummyDoc:
        """Representa um documento PDF mínimo para o teste de fumaça."""

        def __enter__(self):
            """Inicia o uso do dublê no gerenciador de contexto."""
            return self

        def __exit__(self, *args, **kwargs):
            """Finaliza o uso do dublê no gerenciador de contexto."""
            return False

        def __iter__(self):
            """Percorre os itens fornecidos pelo objeto."""
            return iter([])

    class _DummyFitz:
        """Fornece a interface mínima de abertura de documentos PDF."""

        def open(self, *args, **kwargs):
            """Abre o documento simulado usado no teste."""
            return _DummyDoc()

    sys.modules["fitz"] = _DummyFitz()

from tradutor.config import AppConfig
from tradutor.llm_backend import LLMResponse
from tradutor.sanitizer import META_PATTERNS_TRANSLATE
from tradutor.translate import translate_document
from tradutor.utils import setup_logging


class FakeBackend:
    """Executa o caminho mínimo de tradução sem servidor externo."""

    def generate(self, prompt: str) -> LLMResponse:
        """Retorna uma resposta que executa o caminho mínimo de tradução sem servidor externo."""
        return LLMResponse(
            text="### TEXTO_TRADUZIDO_INICIO\nPrimeiro paragrafo em portugues.\n\nSegundo paragrafo em portugues, continuando a ideia.\n### TEXTO_TRADUZIDO_FIM",
            latency=0.01,
        )


def test_translate_document_smoke() -> None:
    """Valida as regras de conteúdo válido no fluxo mínimo de tradução."""
    cfg = AppConfig()
    logger = setup_logging(logging.DEBUG)
    pdf_text = (
        "First paragraph in English. It sets the scene and introduces characters.\n\n"
        "Second paragraph continues the story with another short line."
    )

    result = translate_document(
        pdf_text=pdf_text,
        backend=FakeBackend(),
        cfg=cfg,
        logger=logger,
    )

    assert result, "A traducao nao pode ser vazia."
    assert "TEXTO_TRADUZIDO_INICIO" not in result
    assert "TEXTO_TRADUZIDO_FIM" not in result
    lower_result = result.lower()
    assert "<think>" not in lower_result
    for meta in META_PATTERNS_TRANSLATE:
        assert re.search(meta, lower_result) is None, f"Contem meta: {meta}"
    lines = [line for line in result.splitlines() if line.strip()]
    assert len(lines) >= 2, "Deve haver pelo menos dois paragrafos/linhas nao vazias."
