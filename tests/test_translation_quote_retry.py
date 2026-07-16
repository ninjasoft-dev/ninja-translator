import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.translate import translate_document


class _QuoteRetryBackend:
    """Omite aspas na primeira resposta e as restaura na tentativa seguinte."""

    def __init__(self, outputs):
        """Inicializa as saídas sequenciais e o contador usado nas novas tentativas."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 128
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0
        self.outputs = outputs

    def generate(self, prompt: str):
        """Retorna uma resposta que omite aspas na primeira resposta e as restaura na tentativa seguinte."""
        text = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return type("Resp", (), {"text": text})


def test_translation_retry_on_missing_dialogues(tmp_path: Path) -> None:
    """Valida as regras de conteúdo válido na tradução."""
    cfg = AppConfig(output_dir=tmp_path, max_retries=2, split_by_sections=False)
    # A primeira resposta perde aspas; a segunda restaura a estrutura da entrada.
    input_text = '"A"\n"B"\n"C"\n"D"\n'
    bad_output = '### TEXTO_TRADUZIDO_INICIO\n"A"\n"B"\n### TEXTO_TRADUZIDO_FIM'
    good_output = '### TEXTO_TRADUZIDO_INICIO\n"A"\n"B"\n"C"\n"D"\n### TEXTO_TRADUZIDO_FIM'
    backend = _QuoteRetryBackend([bad_output, good_output])
    logger = logging.getLogger("quote-retry")

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

    assert "C" in result and "D" in result
    assert backend.calls >= 2


def test_translation_retry_on_aggressive_sanitization(tmp_path: Path) -> None:
    """Valida as regras de conteúdo válido na tradução."""
    cfg = AppConfig(output_dir=tmp_path, max_retries=2, split_by_sections=False)
    input_text = "Hello.\n\nThis is a paragraph.\n\nAnother line."
    bad_output = "### TEXTO_TRADUZIDO_INICIO\n<think>foo</think>\nHello.\n### TEXTO_TRADUZIDO_FIM"
    good_output = "### TEXTO_TRADUZIDO_INICIO\nOlá.\nEste é um parágrafo.\nOutra linha.\n### TEXTO_TRADUZIDO_FIM"
    backend = _QuoteRetryBackend([bad_output, good_output])
    logger = logging.getLogger("sanitize-retry")

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

    assert "Olá." in result
    assert backend.calls >= 2
