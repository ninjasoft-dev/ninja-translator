import types

from tradutor.config import AppConfig
from tradutor.desquebrar import desquebrar_text
from tradutor.desquebrar_safe import safe_reflow
from tradutor.utils import setup_logging


class _StubBackend:
    """Fornece saídas controladas para o pós-processamento da desquebra."""

    def __init__(self, outputs):
        """Inicializa as saídas configuradas e o contador de chamadas mantidos pelo dublê."""
        self.outputs = outputs
        self.calls = 0

    def generate(self, prompt):
        """Retorna saídas controladas para o pós-processamento da desquebra."""
        out = self.outputs[self.calls]
        self.calls += 1
        return types.SimpleNamespace(text=out)


def _run_desquebrar(original, llm_output, **cfg_overrides):
    """Executa desquebrar."""
    cfg = AppConfig(desquebrar_chunk_chars=500, **cfg_overrides)
    logger = setup_logging()
    backend = _StubBackend([llm_output])
    result, stats = desquebrar_text(original, cfg, logger, backend)
    return result, stats, backend


def test_desquebrar_stray_quote_line_vol8_case():
    """Valida as regras de aspas e estrutura de diálogos na reconstrução de parágrafos."""
    original = "“devices.”\n“Oh, what…”"
    llm_output = '“devices.”\n"\n“Oh, what…”.'

    result, stats, backend = _run_desquebrar(original, llm_output)

    assert backend.calls == 1
    assert stats.fallbacks == 1
    assert stats.blocks[0]["fallback_reason"] == "qa_stray_quote_lines"
    assert result == "“devices.”\n\n“Oh, what…”"


def test_desquebrar_fallback_on_stray_quote_line():
    """Confirma o fallback seguro diante de problemas em aspas e estrutura de diálogos na reconstrução de parágrafos."""
    original = "Mechanical devices.\nOh, what a savage prospect…"
    llm_output = 'Mechanical devices.\n"\nOh, what a savage prospect…'

    result, stats, backend = _run_desquebrar(original, llm_output)

    assert backend.calls == 1
    assert stats.fallbacks == 1
    assert stats.stray_quote_lines >= 1
    assert stats.blocks[0]["fallback_reason"] == "qa_stray_quote_lines"
    assert result.replace("\n\n", "\n") == safe_reflow(original)


def test_desquebrar_fixes_stutter_space():
    """Valida a normalização de conteúdo válido na reconstrução de parágrafos."""
    original = "D- do."
    llm_output = "D- do."

    result, stats, _ = _run_desquebrar(original, llm_output)

    assert stats.fallbacks == 0
    assert result == "D-do."


def test_desquebrar_fixes_hyphen_linewrap():
    """Valida a normalização de artefatos de extração e OCR na reconstrução de parágrafos."""
    original = "hang-\nups"
    llm_output = "hang-\nups"

    result, stats, _ = _run_desquebrar(original, llm_output)

    assert stats.fallbacks == 0
    assert result == "hang-ups"


def test_desquebrar_isolates_asterisks():
    """Isola marcadores de cena formados por asteriscos."""
    original = "alpha\n***\nomega"
    llm_output = "alpha\n***\nomega"

    result, stats, _ = _run_desquebrar(original, llm_output)

    assert stats.fallbacks == 0
    assert result == "alpha\n\n***\n\nomega"


def test_desquebrar_hyphen_and_asterisks_postprocess(tmp_path):
    """Valida as regras de artefatos de extração e OCR na reconstrução de parágrafos."""
    original = "D- do\nhang-\nups\nbefore\n***\nafter"
    llm_output = original

    result, stats, _ = _run_desquebrar(original, llm_output, output_dir=tmp_path)

    assert "D-do" in result
    assert "hang-ups" in result
    assert "\nbefore\n\n***\n\nafter" in result
    assert stats.fallbacks == 0
    assert stats.hyphen_linewrap_count >= 1
