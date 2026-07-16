import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.translate import translate_document


class _RetryBackend:
    """Falha por truncamento antes de devolver uma tradução completa."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo, os parâmetros de geração e o contador de chamadas."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        """Retorna uma resposta que falha por truncamento antes de devolver uma tradução completa."""
        self.calls += 1
        if self.calls == 1:
            # output truncado forçando retry (ratio baixo)
            text = "### TEXTO_TRADUZIDO_INICIO\nOi.\n### TEXTO_TRADUZIDO_FIM"
        else:
            text = "### TEXTO_TRADUZIDO_INICIO\nTexto completo traduzido.\n### TEXTO_TRADUZIDO_FIM"
        return type("Resp", (), {"text": text})


class _MissingMarkerBackend:
    """Devolve uma tradução válida sem o marcador final esperado."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo e os parâmetros de geração usados pelo dublê."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0

    def generate(self, prompt: str):
        """Retorna uma tradução válida sem o marcador final esperado."""
        text = "### TEXTO_TRADUZIDO_INICIO\nTitulo traduzido sem marcador final"
        return type("Resp", (), {"text": text})


class _NarrativeRetryBackend:
    """Produz uma narrativa curta antes da resposta recuperada."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo, os parâmetros de geração e o contador de chamadas."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        """Retorna uma narrativa curta antes da resposta recuperada."""
        self.calls += 1
        if self.calls == 1:
            text = (
                "### TEXTO_TRADUZIDO_INICIO\n"
                "Texto narrativo reduzido demais para manter a proporcao adequada no chunk.\n"
                "### TEXTO_TRADUZIDO_FIM"
            )
        else:
            text = "### TEXTO_TRADUZIDO_INICIO\nTexto narrativo completo e consistente.\n### TEXTO_TRADUZIDO_FIM"
        return type("Resp", (), {"text": text})


class _TypoMarkerBackend:
    """Usa um marcador de tradução grafado incorretamente."""

    backend = "stub"
    model = "stub"
    num_predict = 10
    temperature = 0.1
    repeat_penalty = 1.0

    def generate(self, prompt: str):
        """Retorna uma resposta que usa um marcador de tradução grafado incorretamente."""
        text = "### TEXTO_TRADUZIDO_INICIO\nTexto traduzido.\n### TEXTO_TRADUZDO_FIM"
        return type("Resp", (), {"text": text})


class _QuotedDialogueBackend:
    """Preserva linhas de diálogo delimitadas por aspas."""

    backend = "stub"
    model = "stub"
    num_predict = 10
    temperature = 0.1
    repeat_penalty = 1.0

    def generate(self, prompt: str):
        """Retorna uma resposta que preserva linhas de diálogo delimitadas por aspas."""
        text = '### TEXTO_TRADUZIDO_INICIO\n"Oi."\n\nNarracao.\n\n"Sim."\n### TEXTO_TRADUZIDO_FIM'
        return type("Resp", (), {"text": text})


class _ResidualEnglishBackend:
    """Mantém uma frase em inglês antes da nova tentativa."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo, os parâmetros de geração e o contador de chamadas."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        """Retorna uma resposta que mantém uma frase em inglês antes da nova tentativa."""
        self.calls += 1
        if self.calls == 1:
            text = (
                "### TEXTO_TRADUZIDO_INICIO\n"
                '"I have no desire to die," replied Lina calmly, choosing not to answer directly.\n'
                "### TEXTO_TRADUZIDO_FIM"
            )
        else:
            text = (
                "### TEXTO_TRADUZIDO_INICIO\n"
                '"Não tenho vontade de morrer", respondeu Lina com calma, sem responder diretamente.\n'
                "### TEXTO_TRADUZIDO_FIM"
            )
        return type("Resp", (), {"text": text})


def test_translate_retries_on_truncated_output(tmp_path: Path) -> None:
    """Confirma a detecção de problemas em conteúdo válido na tradução."""
    cfg = AppConfig(output_dir=tmp_path, max_retries=2, split_by_sections=False)
    backend = _RetryBackend()
    logger = logging.getLogger("retry-test")
    input_text = (
        "This is a longer input text that should be fully present after translation. " * 8
    ).strip()

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

    assert "Texto completo traduzido" in result
    assert backend.calls >= 2


def test_translate_retries_on_low_ratio_narrative(tmp_path: Path) -> None:
    """Confirma a detecção de problemas em integridade do conteúdo na tradução."""
    cfg = AppConfig(output_dir=tmp_path, max_retries=2, split_by_sections=False)
    backend = _NarrativeRetryBackend()
    logger = logging.getLogger("narrative-retry")
    input_text = ("Narrativa sem dialogo com tamanho moderado para teste. " * 2).strip()

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

    assert "Texto narrativo completo" in result
    assert backend.calls >= 2


def test_translate_parses_output_without_end_marker(tmp_path: Path) -> None:
    """Valida as regras de marcadores de controle na tradução."""
    cfg = AppConfig(output_dir=tmp_path, max_retries=1, split_by_sections=False)
    backend = _MissingMarkerBackend()
    logger = logging.getLogger("missing-marker")
    input_text = (
        "Epilogue content that should be passed through without missing markers. " * 5
    ).strip()

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

    assert "Titulo traduzido" in result
    assert "TEXTO_TRADUZIDO" not in result


def test_translate_strips_typo_translation_marker(tmp_path: Path) -> None:
    """Valida a remoção segura de marcadores de controle na tradução."""
    cfg = AppConfig(output_dir=tmp_path, max_retries=1, split_by_sections=False)
    logger = logging.getLogger("typo-marker")

    result = translate_document(
        pdf_text="Short input for marker cleanup.",
        backend=_TypoMarkerBackend(),
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

    assert "Texto traduzido" in result
    assert "TEXTO_TRADUZ" not in result


def test_translate_preserves_quoted_dialogue_lines(tmp_path: Path) -> None:
    """Confirma a preservação de aspas e estrutura de diálogos na tradução."""
    cfg = AppConfig(output_dir=tmp_path, max_retries=1, split_by_sections=False)
    logger = logging.getLogger("quoted-dialogue")

    result = translate_document(
        pdf_text="A short paragraph.\n\nAnother short paragraph.",
        backend=_QuotedDialogueBackend(),
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

    assert "“Oi.”" in result
    assert "“Sim.”" in result
    assert "— Oi." not in result
    assert "— Sim." not in result


def test_translate_retries_on_residual_english_sentence(tmp_path: Path) -> None:
    """Confirma a detecção de problemas em idioma residual na tradução."""
    cfg = AppConfig(output_dir=tmp_path, max_retries=2, split_by_sections=False)
    backend = _ResidualEnglishBackend()
    logger = logging.getLogger("residual-english")
    input_text = '"I have no desire to die," replied Lina calmly, choosing not to answer directly.'

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

    assert "Não tenho vontade de morrer" in result
    assert "I have no desire" not in result
    assert backend.calls >= 2
