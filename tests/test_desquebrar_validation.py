import types

from tradutor.config import AppConfig
from tradutor.desquebrar import desquebrar_text, deterministic_unbreak
from tradutor.utils import setup_logging


class _StubBackend:
    """Fornece saídas válidas ou corrompidas para exercitar a validação."""

    def __init__(self, outputs):
        """Inicializa as saídas configuradas e o contador de chamadas mantidos pelo dublê."""
        self.outputs = outputs
        self.calls = 0

    def generate(self, prompt):
        """Retorna saídas válidas ou corrompidas para exercitar a validação."""
        out = self.outputs[self.calls]
        self.calls += 1
        return types.SimpleNamespace(text=out)


def test_desquebrar_rejects_lonely_quote_and_falls_back():
    """Confirma o fallback seguro diante de problemas em aspas e estrutura de diálogos na reconstrução de parágrafos."""
    cfg = AppConfig(desquebrar_chunk_chars=500)
    logger = setup_logging()
    original = "For now, yes...\nBut we continue."
    bad_output = 'For now, yes...\n"\nBut we continue.'
    backend = _StubBackend([bad_output])

    result, stats = desquebrar_text(original, cfg, logger, backend)

    assert backend.calls == 1
    assert stats.fallbacks == 1
    assert stats.blocks[0]["fallback"] is True
    assert "fallback_reason" in stats.blocks[0]
    collapsed = result.replace("\n\n", " ").strip()
    assert collapsed == "For now, yes... But we continue."


def test_desquebrar_rejects_content_loss():
    """Confirma a detecção de problemas em integridade do conteúdo na reconstrução de parágrafos."""
    cfg = AppConfig(desquebrar_chunk_chars=500)
    logger = setup_logging()
    original = "Nggh…I do think they’ve rather served..."
    bad_output = "Ngg...r served..."
    backend = _StubBackend([bad_output])

    result, stats = desquebrar_text(original, cfg, logger, backend)

    assert backend.calls == 1
    assert stats.fallbacks == 1
    reason = stats.blocks[0].get("fallback_reason", "")
    assert "alnum_loss" in reason
    assert result == deterministic_unbreak(original)
