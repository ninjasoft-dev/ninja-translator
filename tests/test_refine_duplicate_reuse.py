import logging
import re
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.refine import refine_markdown_file


class _NumberAwareBackend:
    """Produz respostas distintas conforme o número presente no bloco refinado."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo, os parâmetros de geração e o contador de chamadas."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        """Retorna respostas distintas conforme o número presente no bloco refinado."""
        self.calls += 1
        match = re.search(
            r"Texto para revisao \(PT-BR\):\s*\"\"\"(.*)\"\"\"",
            prompt,
            flags=re.DOTALL,
        )
        chunk = match.group(1) if match else prompt
        numbers = re.findall(r"\b(\d+)\b", chunk)
        number = numbers[-1] if numbers else "0"
        return type("Resp", (), {"text": f"Numero refinado: {number}. {chunk}"})


def test_refine_near_duplicate_blocks_number_change(tmp_path: Path) -> None:
    """Valida as regras de repetições indevidas no refino."""
    cfg = AppConfig(output_dir=tmp_path, refine_chunk_chars=120, refine_guardrails="off")
    backend = _NumberAwareBackend()
    logger = logging.getLogger("refine-duplicate")

    input_path = tmp_path / "input.md"
    output_path = tmp_path / "output.md"
    para_one = "Texto base com numero 100 para refino."
    para_two = "Texto base com numero 200 para refino."
    input_path.write_text(
        f"# Capitulo 1\n\n{para_one}\n\n# Capitulo 2\n\n{para_two}",
        encoding="utf-8",
    )

    refine_markdown_file(
        input_path=input_path,
        output_path=output_path,
        backend=backend,
        cfg=cfg,
        logger=logger,
        progress_path=None,
        resume_manifest=None,
        normalize_paragraphs=False,
        glossary_state=None,
        debug_refine=False,
        parallel_workers=1,
        preprocess_advanced=False,
        debug_chunks=False,
        cleanup_mode="off",
    )

    result = output_path.read_text(encoding="utf-8")
    assert "Numero refinado: 100." in result
    assert "Numero refinado: 200." in result
    assert backend.calls >= 2
