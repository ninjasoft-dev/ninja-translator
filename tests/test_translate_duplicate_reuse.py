import logging
import re
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.translate import translate_document


class _NumberAwareBackend:
    """Produz respostas distintas conforme o número presente no chunk."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo, os parâmetros de geração e o contador de chamadas."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        """Retorna respostas distintas conforme o número presente no chunk."""
        self.calls += 1
        match = re.search(r"TEXTO A SER TRADUZIDO:\n\"\"\"(.*)\"\"\"", prompt, flags=re.DOTALL)
        chunk = match.group(1) if match else prompt
        numbers = re.findall(r"\b(\d+)\b", chunk)
        number = numbers[-1] if numbers else "0"
        text = (
            "### TEXTO_TRADUZIDO_INICIO\n"
            f"Numero detectado: {number}. {chunk}\n"
            "### TEXTO_TRADUZIDO_FIM"
        )
        return type("Resp", (), {"text": text})


def test_translate_near_duplicate_blocks_number_change(tmp_path: Path) -> None:
    """Valida as regras de repetições indevidas na tradução."""
    cfg = AppConfig(output_dir=tmp_path, split_by_sections=True, translate_chunk_chars=120)
    backend = _NumberAwareBackend()
    logger = logging.getLogger("translate-duplicate")

    para_one = "This is a test paragraph with number 100 that should not be reused."
    para_two = "This is a test paragraph with number 200 that should not be reused."
    input_text = f"Chapter 1\n\n{para_one}\n\nChapter 2\n\n{para_two}"

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

    assert "Numero detectado: 100." in result
    assert "Numero detectado: 200." in result
    assert backend.calls >= 2
