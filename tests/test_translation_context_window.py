import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.translate import (
    build_recent_translation_context,
    build_translation_prompt,
    classify_translation_chunk,
    translate_document,
)


class _PromptCaptureBackend:
    """Captura os prompts para verificar o contexto deslizante entre chunks."""

    def __init__(self) -> None:
        """Inicializa os prompts capturados e o contador de chamadas do backend."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.prompts: list[str] = []
        self.calls = 0

    def generate(self, prompt: str):
        """Captura os prompts para verificar o contexto deslizante entre chunks e retorna a resposta configurada."""
        self.calls += 1
        self.prompts.append(prompt)
        text = f"### TEXTO_TRADUZIDO_INICIO\nTexto traduzido do chunk {self.calls} com continuidade suficiente para teste.\n### TEXTO_TRADUZIDO_FIM"
        return type("Resp", (), {"text": text})


def test_build_recent_translation_context_uses_source_and_pt_tail() -> None:
    """Confirma o registro correto de contexto enviado ao modelo na tradução."""
    context = build_recent_translation_context(
        [
            {
                "source": "First source paragraph.",
                "target": "Primeiro parágrafo traduzido.",
            },
            {
                "source": "Second source paragraph.",
                "target": "Segundo parágrafo traduzido.",
            },
        ],
        max_paragraphs=1,
        max_chars=500,
        include_pt=True,
    )

    assert "Second source paragraph." in context
    assert "Segundo parágrafo traduzido." in context
    assert "First source paragraph." not in context


def test_translation_prompt_has_dialogue_specific_rules() -> None:
    """Valida as regras de aspas e estrutura de diálogos na tradução."""
    prompt = build_translation_prompt('"Are you okay?"\n\n"Yes."', chunk_profile="dialogue")

    assert "FOCO DO TRECHO: DIÁLOGO" in prompt
    assert "fala natural em PT-BR" in prompt


def test_translation_prompt_requests_silent_linguistic_review() -> None:
    """Valida as regras de contexto enviado ao modelo na tradução."""
    prompt = build_translation_prompt("The riders could take intense action.")

    assert "REVISÃO SILENCIOSA OBRIGATÓRIA" in prompt
    assert "tomar ações" in prompt


def test_classify_translation_chunk_dialogue_and_narration() -> None:
    """Valida as regras de aspas e estrutura de diálogos na tradução."""
    assert classify_translation_chunk('"Oi."\n\n"Sim."') == "dialogue"
    assert (
        classify_translation_chunk("The wind crossed the empty field under the gray sky.")
        == "narration"
    )


def test_translate_document_sends_sliding_context_to_next_chunk(tmp_path: Path) -> None:
    """Confirma o registro correto de contexto enviado ao modelo na tradução."""
    cfg = AppConfig(
        output_dir=tmp_path,
        max_retries=1,
        split_by_sections=False,
        translate_chunk_chars=120,
        translate_context_paragraphs=2,
        translate_context_chars=500,
        translate_context_include_pt=True,
        use_translation_repair=False,
    )
    backend = _PromptCaptureBackend()
    source = (
        ("First paragraph establishes the emotional state of the scene. " * 6)
        + "Alpha context anchor stays near the tail of the previous chunk. "
        + ("More source text keeps the first chunk large enough to split. " * 8)
        + "\n\n"
        + ("Second paragraph continues the scene and should receive context. " * 8)
    )

    translate_document(
        pdf_text=source,
        backend=backend,
        cfg=cfg,
        logger=logging.getLogger("context-window"),
        source_slug="sample",
        already_preprocessed=True,
        translation_repair=False,
    )

    assert len(backend.prompts) >= 2
    assert "CONTEXTO RECENTE" in backend.prompts[1]
    assert "Alpha context anchor" in backend.prompts[1]
    assert "Texto traduzido do chunk 1" in backend.prompts[1]
