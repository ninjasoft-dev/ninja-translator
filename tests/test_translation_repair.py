import json
import logging
from pathlib import Path

from tradutor.cache_utils import set_cache_base_dir
from tradutor.config import AppConfig
from tradutor.repair import (
    detect_translation_repair_issues,
    repair_translation_chunk,
    validate_repair_candidate,
)
from tradutor.translate import translate_document


class _RepairBackend:
    """Simula o reparo de um chunk com problema objetivo."""

    backend = "stub"
    model = "stub"
    num_predict = 10
    temperature = 0.1
    repeat_penalty = 1.0

    def __init__(self) -> None:
        """Inicializa o contador de chamadas mantidos pelo dublê."""
        self.calls = 0

    def generate(self, prompt: str):
        """Retorna o reparo de um chunk com problema objetivo."""
        self.calls += 1
        text = (
            "### TEXTO_REPARADO_INICIO\n"
            "“Não tenho vontade de morrer”, respondeu Lina com calma, sem responder diretamente.\n"
            "### TEXTO_REPARADO_FIM"
        )
        return type("Resp", (), {"text": text})


class _TranslateThenRepairBackend:
    """Produz primeiro a tradução defeituosa e depois o reparo aceito."""

    backend = "stub"
    model = "stub"
    num_predict = 10
    temperature = 0.1
    repeat_penalty = 1.0

    def __init__(self) -> None:
        """Inicializa o contador de chamadas mantidos pelo dublê."""
        self.calls = 0

    def generate(self, prompt: str):
        """Retorna primeiro a tradução defeituosa e depois o reparo aceito."""
        self.calls += 1
        if self.calls == 1:
            text = (
                "### TEXTO_TRADUZIDO_INICIO\n"
                "“I have no desire to die,” replied Lina calmly, choosing not to answer directly.\n"
                "### TEXTO_TRADUZIDO_FIM"
            )
        else:
            text = (
                "### TEXTO_REPARADO_INICIO\n"
                "“Não tenho vontade de morrer”, respondeu Lina com calma, sem responder diretamente.\n"
                "### TEXTO_REPARADO_FIM"
            )
        return type("Resp", (), {"text": text})


class _AmputatingRepairBackend:
    """Produz um reparo que remove conteúdo para testar sua rejeição."""

    backend = "stub"
    model = "stub"
    num_predict = 10
    temperature = 0.1
    repeat_penalty = 1.0

    def __init__(self) -> None:
        """Inicializa o contador de chamadas mantidos pelo dublê."""
        self.calls = 0

    def generate(self, prompt: str):
        """Retorna um reparo que remove conteúdo para testar sua rejeição."""
        self.calls += 1
        text = (
            "### TEXTO_REPARADO_INICIO\n"
            "“Nika… Nika—me desculpe… Me desculpe por ter demorado tanto pra te encontrar…!”\n"
            "### TEXTO_REPARADO_FIM"
        )
        return type("Resp", (), {"text": text})


def test_detect_translation_repair_issues_flags_residual_english() -> None:
    """Confirma a detecção de problemas em idioma residual na tradução."""
    issues = detect_translation_repair_issues(
        source_text='"I have no desire to die," replied Lina calmly.',
        translated_text='"I have no desire to die," replied Lina calmly, choosing not to answer directly.',
    )

    assert any(issue["type"] == "residual_english" for issue in issues)


def test_repair_translation_chunk_fixes_residual_english(tmp_path: Path) -> None:
    """Valida a normalização de idioma residual na tradução."""
    set_cache_base_dir(tmp_path)
    backend = _RepairBackend()
    result = repair_translation_chunk(
        source_text='"I have no desire to die," replied Lina calmly.',
        translated_text='"I have no desire to die," replied Lina calmly, choosing not to answer directly.',
        backend=backend,
        logger=logging.getLogger("repair-test"),
        max_attempts=1,
    )

    assert result.attempted
    assert result.changed
    assert result.elapsed_seconds >= 0
    assert "Não tenho vontade de morrer" in result.text
    assert backend.calls == 1


def test_translate_document_repairs_chunk_before_final_output(tmp_path: Path) -> None:
    """Valida as regras de conteúdo válido na tradução."""
    set_cache_base_dir(tmp_path)
    cfg = AppConfig(
        output_dir=tmp_path,
        max_retries=1,
        split_by_sections=False,
        use_translation_repair=True,
    )
    backend = _TranslateThenRepairBackend()

    result = translate_document(
        pdf_text='"I have no desire to die," replied Lina calmly, choosing not to answer directly.',
        backend=backend,
        cfg=cfg,
        logger=logging.getLogger("translate-repair-test"),
        source_slug="sample",
        already_preprocessed=True,
    )

    assert "Não tenho vontade de morrer" in result
    assert "I have no desire" not in result
    assert backend.calls >= 2
    repair_metrics = json.loads(
        (tmp_path / "sample_repair_metrics.json").read_text(encoding="utf-8")
    )
    assert repair_metrics["elapsed_seconds"] >= 0
    assert repair_metrics["chunks"][0]["elapsed_seconds"] >= 0


def test_repair_rejects_candidate_that_removes_existing_translation(
    tmp_path: Path,
) -> None:
    """Confirma a detecção de problemas em conteúdo válido na tradução."""
    set_cache_base_dir(tmp_path)
    source = (
        "Opening line with Nika and Bruma.\n\n"
        "More reunion dialogue.\n\n"
        "“I have no desire to die,” replied Lina calmly."
    )
    translated = (
        "“Deixei isso a seu critério, certo? Se você decidiu vir, tudo bem pra mim.”\n\n"
        "O rosto da Nika se iluminou. “Nika está tão feliz de ver você de novo, mestre!”\n\n"
        "O Pip e o Bruma realmente adoram a Nika.\n\n"
        "“I have no desire to die,” replied Lina calmly, choosing not to answer directly."
    )
    result = repair_translation_chunk(
        source_text=source,
        translated_text=translated,
        backend=_AmputatingRepairBackend(),
        logger=logging.getLogger("repair-amputation-test"),
        max_attempts=1,
    )

    assert result.attempted
    assert result.suspect_output
    assert result.text == translated
    assert result.retry_reasons

    validation_reason = validate_repair_candidate(
        source_text=source,
        translated_text=translated,
        candidate_text="\n\n".join(translated.split("\n\n")[1:]),
    )
    assert validation_reason
    assert "repair_removed" in validation_reason
