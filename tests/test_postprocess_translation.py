import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.postprocess_translation import postprocess_translation
from tradutor.translate import translate_document


class _ParryBackend:
    """Simula uma tradução que contém um falso cognato conhecido."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo e os parâmetros de geração usados pelo dublê."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 128
        self.temperature = 0.1
        self.repeat_penalty = 1.0

    def generate(self, prompt: str):
        """Retorna uma tradução que contém um falso cognato conhecido."""
        return type(
            "Resp",
            (),
            {
                "text": "### TEXTO_TRADUZIDO_INICIO\nEle parriu o golpe rapidamente.\n### TEXTO_TRADUZIDO_FIM"
            },
        )


def test_postprocess_fixes_parry_false_cognate(tmp_path: Path) -> None:
    """Valida a normalização de vocabulário residual no pós-processamento."""
    cfg = AppConfig(output_dir=tmp_path, split_by_sections=False)
    backend = _ParryBackend()
    logger = logging.getLogger("parry-fix")
    input_text = "He parried the incoming blow."

    result = translate_document(
        pdf_text=input_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="sample",
    )

    assert "aparou" in result
    assert "parriu" not in result


class _MixedEnglishArtifactBackend:
    """Simula artefatos híbridos de inglês e português."""

    def __init__(self) -> None:
        """Inicializa o backend, o modelo, os parâmetros de geração e o contador de chamadas."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 128
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        """Retorna artefatos híbridos de inglês e português."""
        self.calls += 1
        return type(
            "Resp",
            (),
            {
                "text": "### TEXTO_TRADUZIDO_INICIO\n… I-isso? I não me importo.\n### TEXTO_TRADUZIDO_FIM"
            },
        )


def test_pre_qa_postprocess_fixes_mixed_english_artifacts(tmp_path: Path) -> None:
    """Valida a normalização de idioma residual no pós-processamento."""
    cfg = AppConfig(output_dir=tmp_path, split_by_sections=False)
    backend = _MixedEnglishArtifactBackend()
    logger = logging.getLogger("mixed-english-pre-qa")

    result = translate_document(
        pdf_text="… Y-yes? I do not care.",
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="mixed-artifact",
    )

    assert "S-sim? Eu não me importo." in result
    assert backend.calls == 1


def test_pre_qa_postprocess_adapts_af_slang() -> None:
    """Valida a normalização de vocabulário residual no pós-processamento."""
    result = postprocess_translation("Ela é super desconfiada AF.")

    assert result == "Ela é super desconfiada pra caramba."


def test_pre_qa_postprocess_translates_single_word_connector() -> None:
    """Valida a normalização de conteúdo válido no pós-processamento."""
    result = postprocess_translation("Ela é arrogante, though.")

    assert result == "Ela é arrogante, porém."


def test_pre_qa_postprocess_translates_arright_interjection() -> None:
    """Valida a normalização de vocabulário residual no pós-processamento."""
    result = postprocess_translation("“Arright!” Itsuki sorriu.")

    assert result == "“Beleza!” Itsuki sorriu."


def test_pre_qa_postprocess_removes_hybrid_english_pronoun() -> None:
    """Valida a remoção segura de idioma residual no pós-processamento."""
    result = postprocess_translation("Ninguém o encarava com raiva — they todos pareciam felizes.")

    assert result == "Ninguém o encarava com raiva — todos pareciam felizes."


def test_pre_qa_postprocess_translates_boost() -> None:
    """Valida a normalização de conteúdo válido no pós-processamento."""
    result = postprocess_translation("Eles precisavam de um boost extra.")

    assert result == "Eles precisavam de um impulso extra."


def test_pre_qa_postprocess_fixes_mixed_pronoun_and_interjection() -> None:
    """Valida a normalização de consistência de gênero no pós-processamento."""
    result = postprocess_translation("Uau—I aposto que ela consegue. Uhh… sei disso. Uh… também.")

    assert result == "Uau—Eu aposto que ela consegue. Ah… sei disso. Ah… também."


def test_pre_qa_postprocess_fixes_english_stutter_before_qa() -> None:
    """Valida a normalização de idioma residual no pós-processamento."""
    result = postprocess_translation("Y-you divindades podem ser interessantes.")

    assert result == "V-vocês, divindades, podem ser interessantes."
