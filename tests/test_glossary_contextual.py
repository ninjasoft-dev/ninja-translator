import json
import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.debug_run import DebugRunWriter
from tradutor.translate import translate_document


class _PromptCaptureBackend:
    """Captura os prompts para conferir a injeção contextual do glossário."""

    def __init__(self) -> None:
        """Inicializa o backend e a coleção de prompts capturados pelo dublê."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 128
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.prompts: list[str] = []

    def generate(self, prompt: str):
        """Captura os prompts para conferir a injeção contextual do glossário e retorna a resposta configurada."""
        self.prompts.append(prompt)
        return type(
            "Resp",
            (),
            {"text": "### TEXTO_TRADUZIDO_INICIO\nTradução do escudo.\n### TEXTO_TRADUZIDO_FIM"},
        )


def test_glossary_injects_only_matching_terms(tmp_path: Path) -> None:
    """Confirma o registro correto de termos de glossário no tratamento do glossário."""
    cfg = AppConfig(output_dir=tmp_path, split_by_sections=False)
    backend = _PromptCaptureBackend()
    logger = logging.getLogger("glossary-context")
    manual_terms = [
        {"key": "Shield", "pt": "Escudo"},
        {"key": "Sword", "pt": "Espada"},
    ]
    input_text = ("The shield was heavy and sturdy. " * 10).strip()

    translate_document(
        pdf_text=input_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        glossary_manual_terms=manual_terms,
    )

    assert backend.prompts, "espera ao menos um prompt enviado ao backend"
    prompt = backend.prompts[0]
    assert "Shield" in prompt and "Escudo" in prompt
    assert "Sword" not in prompt


def test_glossary_matches_aliases(tmp_path: Path) -> None:
    """Seleciona uma entrada quando o chunk contém um de seus aliases."""
    cfg = AppConfig(output_dir=tmp_path, split_by_sections=False)
    backend = _PromptCaptureBackend()
    logger = logging.getLogger("glossary-alias")
    manual_terms = [
        {
            "key": "Magic Sword",
            "pt": "Espada Mágica",
            "aliases": ["Blade of Dawn", "Dawnblade"],
        },
        {"key": "Shield", "pt": "Escudo"},
    ]
    input_text = ("The Blade of Dawn was legendary and revered across the lands. " * 8).strip()

    translate_document(
        pdf_text=input_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        glossary_manual_terms=manual_terms,
    )

    assert backend.prompts, "espera ao menos um prompt enviado ao backend"
    prompt = backend.prompts[0]
    assert "Magic Sword" in prompt and "Espada Mágica" in prompt
    assert "Shield" not in prompt


def test_glossary_fallback_does_not_enforce_terms(tmp_path: Path) -> None:
    """Confirma o fallback seguro diante de problemas em termos de glossário no tratamento do glossário."""

    class _Backend:
        """Devolve respostas fixas nos cenários sem correspondência de glossário."""

        def __init__(self) -> None:
            """Inicializa o backend, o modelo e os parâmetros de geração usados pelo dublê."""
            self.backend = "stub"
            self.model = "stub"
            self.num_predict = 128
            self.temperature = 0.1
            self.repeat_penalty = 1.0

        def generate(self, prompt: str):
            """Retorna respostas fixas nos cenários sem correspondência de glossário."""
            return type(
                "Resp",
                (),
                {"text": "### TEXTO_TRADUZIDO_INICIO\nArt aparece aqui.\n### TEXTO_TRADUZIDO_FIM"},
            )

    cfg = AppConfig(output_dir=tmp_path, split_by_sections=False)
    backend = _Backend()
    logger = logging.getLogger("glossary-fallback")
    manual_terms = [{"key": "Art", "pt": "Arte", "enforce": True}]
    input_text = ("Nothing related to that term in this chunk. " * 5).strip()

    result = translate_document(
        pdf_text=input_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        glossary_manual_terms=manual_terms,
    )

    assert "Arte" not in result
    assert "Art" in result


def test_debug_manifest_records_chunk_glossary(tmp_path: Path) -> None:
    """Confirma o registro correto de termos de glossário no tratamento do glossário."""
    cfg = AppConfig(output_dir=tmp_path, split_by_sections=False)
    backend = _PromptCaptureBackend()
    logger = logging.getLogger("glossary-debug-manifest")
    debug_run = DebugRunWriter.create(
        output_dir=tmp_path,
        slug="sample",
        input_kind="md",
        max_chunks=None,
        max_chars_per_file=5000,
        store_llm_raw=True,
    )
    manual_terms = [
        {"key": "Shield", "pt": "Escudo", "enforce": True},
        {"key": "Sword", "pt": "Espada"},
    ]
    input_text = ("The shield was heavy and sturdy. " * 10).strip()

    translate_document(
        pdf_text=input_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        glossary_manual_terms=manual_terms,
        debug_run=debug_run,
    )

    manifest = json.loads(
        (debug_run.run_dir / "40_translate" / "translate_manifest.json").read_text(encoding="utf-8")
    )
    chunk = manifest["chunks"][0]
    assert manifest["glossary"]["manual_terms_total"] == 2
    assert chunk["glossary"]["selection_mode"] == "matched"
    assert chunk["glossary"]["matched_count"] == 1
    assert chunk["glossary"]["terms"] == [
        {"key": "Shield", "pt": "Escudo", "category": None, "enforce": True}
    ]
    glossary_path = debug_run.run_dir / chunk["outputs"]["debug_glossary"]
    assert "Shield" in glossary_path.read_text(encoding="utf-8")
    assert "Sword" not in glossary_path.read_text(encoding="utf-8")
