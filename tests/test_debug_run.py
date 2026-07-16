import json
import logging
import re
import types
from pathlib import Path

import pytest

from tradutor.config import AppConfig
from tradutor.debug_run import DebugRunWriter
from tradutor.llm_backend import LLMResponse
from tradutor.refine import refine_markdown_file
from tradutor.translate import translate_document
from tradutor.utils import read_text, setup_logging, write_text


class FakeTranslateBackend:
    """Produz uma tradução controlada para inspecionar os artefatos de depuração."""

    def generate(self, prompt: str) -> LLMResponse:
        """Retorna uma tradução controlada para inspecionar os artefatos de depuração."""
        return LLMResponse(
            text="### TEXTO_TRADUZIDO_INICIO\nTexto traduzido em português.\n\nSegundo parágrafo traduzido.\n### TEXTO_TRADUZIDO_FIM",
            latency=0.01,
        )


class FakeRefineBackend:
    """Produz um refino controlado para inspecionar os artefatos de depuração."""

    def generate(self, prompt: str) -> LLMResponse:
        """Retorna um refino controlado para inspecionar os artefatos de depuração."""
        return LLMResponse(
            text="### TEXTO_REFINADO_INICIO\nTexto refinado.\n### TEXTO_REFINADO_FIM",
            latency=0.01,
        )


class FakeLLMBackend:
    """Simula o backend compartilhado pelo fluxo completo de depuração."""

    def __init__(
        self,
        backend: str = "fake",
        model: str = "fake",
        temperature=None,
        num_predict=None,
        repeat_penalty=None,
        **kwargs,
    ):
        """Inicializa o backend e os parâmetros de geração usados na depuração."""
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.repeat_penalty = repeat_penalty

    def generate(self, prompt: str) -> LLMResponse:
        """Retorna o backend compartilhado pelo fluxo completo de depuração."""
        if "TEXTO_REFINADO" in prompt:
            text = "### TEXTO_REFINADO_INICIO\nREFINED BLOCK\n### TEXTO_REFINADO_FIM"
        else:
            text = "### TEXTO_TRADUZIDO_INICIO\nTRADUZIDO BLOCO\n### TEXTO_TRADUZIDO_FIM"
        return LLMResponse(text=text, latency=0.0)


def test_debug_run_translate_refine_outputs(tmp_path: Path) -> None:
    """Valida as regras de conteúdo válido na geração de artefatos de depuração."""
    cfg = AppConfig(output_dir=tmp_path)
    logger = setup_logging(logging.DEBUG)
    source_text = "First paragraph in English.\n\nSecond paragraph in English."

    result_plain = translate_document(
        pdf_text=source_text,
        backend=FakeTranslateBackend(),
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        already_preprocessed=True,
    )

    debug_run = DebugRunWriter.create(
        output_dir=tmp_path,
        slug="sample",
        input_kind="md",
        max_chunks=None,
        max_chars_per_file=5000,
        store_llm_raw=True,
    )
    debug_run.preprocessed_rel = "10_preprocess/sample_preprocessed.md"
    debug_run.desquebrado_rel = "20_desquebrar/sample_raw_desquebrado.md"
    debug_run.write_text(debug_run.preprocessed_rel, source_text)
    debug_run.write_text(debug_run.desquebrado_rel, source_text)

    result_debug = translate_document(
        pdf_text=source_text,
        backend=FakeTranslateBackend(),
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        already_preprocessed=True,
        debug_run=debug_run,
    )

    assert result_plain == result_debug

    input_path = tmp_path / "sample_pt.md"
    output_path = tmp_path / "sample_pt_refinado.md"
    write_text(input_path, result_debug)
    debug_run.pt_output_rel = "sample_pt.md"

    refine_markdown_file(
        input_path=input_path,
        output_path=output_path,
        backend=FakeRefineBackend(),
        cfg=cfg,
        logger=logger,
        cleanup_mode="off",
        debug_run=debug_run,
    )

    translate_manifest_path = debug_run.run_dir / "40_translate" / "translate_manifest.json"
    refine_manifest_path = debug_run.run_dir / "60_refine" / "refine_manifest.json"
    assert translate_manifest_path.exists()
    assert refine_manifest_path.exists()

    translate_manifest = json.loads(read_text(translate_manifest_path))
    refine_manifest = json.loads(read_text(refine_manifest_path))
    assert translate_manifest["chunking"]["total_chunks"] == len(translate_manifest["chunks"])
    assert refine_manifest["refine"]["total_chunks"] == len(refine_manifest["chunks"])

    debug_chunk = debug_run.run_dir / "40_translate" / "debug_traducao" / "chunk001_final_pt.txt"
    assert debug_chunk.exists()

    for path_value in translate_manifest["input_paths"].values():
        assert path_value is None or not Path(path_value).is_absolute()
    for chunk in translate_manifest["chunks"]:
        for key in ("debug_original", "debug_context", "debug_llm_raw", "debug_final"):
            assert not Path(chunk["outputs"][key]).is_absolute()
    for key, value in refine_manifest["input_paths"].items():
        if value is not None:
            assert not Path(value).is_absolute()


def test_debug_mode_end_to_end_artifacts(monkeypatch, tmp_path: Path) -> None:
    """Valida as regras de métricas e artefatos na geração de artefatos de depuração."""
    import tradutor.main as main  # noqa: WPS433

    base_para = (
        "This is a long paragraph in English with repeated words to test debug chunking and ensure more than one "
        "chunk is produced cleanly without altering outputs."
    )
    sample_text = "\n\n".join(
        [
            " ".join([base_para] * 5),
            " ".join([base_para] * 4),
            " ".join([base_para] * 3),
        ]
    )
    md_path = tmp_path / "sample.md"
    md_path.write_text(sample_text, encoding="utf-8")

    common_cfg = {
        "data_dir": tmp_path,
        "translate_chunk_chars": 80,
        "refine_chunk_chars": 60,
    }
    base_cfg = AppConfig(output_dir=tmp_path / "base_out", **common_cfg)
    debug_cfg = AppConfig(
        output_dir=tmp_path / "debug_out",
        debug_max_chunks=1,
        debug_max_chars_per_file=50,
        debug_store_llm_raw=False,
        **common_cfg,
    )

    logger = setup_logging(logging.DEBUG)
    monkeypatch.setattr(main, "LLMBackend", FakeLLMBackend)

    base_args = types.SimpleNamespace(
        command="traduz-md",
        input=str(md_path),
        backend="fake",
        model="fake-model",
        num_predict=64,
        request_timeout=30,
        preprocess_advanced=False,
        normalize_paragraphs=False,
        clear_cache=None,
        translate_allow_adaptation=False,
        use_glossary=False,
        manual_glossary=None,
        parallel=1,
        debug=False,
        debug_chunks=False,
        split_by_sections=False,
        fail_on_chunk_error=False,
        resume=False,
        refine=True,
        no_refine=False,
        cleanup_before_refine="off",
        pdf_enabled=False,
    )
    debug_args = types.SimpleNamespace(**{**base_args.__dict__, "debug": True})

    main.run_translate_md(base_args, base_cfg, logger)
    base_output = read_text(base_cfg.output_dir / "sample_pt_refinado.md")
    base_timings = json.loads(read_text(base_cfg.output_dir / "sample_timings.json"))
    assert base_timings["status"] == "success"
    assert base_timings["command"] == "traduz-md"
    assert base_timings["total_elapsed_seconds"] >= 0
    assert "translate" in base_timings["stages"]
    assert "refine" in base_timings["stages"]

    main.run_translate_md(debug_args, debug_cfg, logger)
    debug_output_path = debug_cfg.output_dir / "sample_pt_refinado.md"
    debug_output = read_text(debug_output_path)
    assert debug_output == base_output

    run_root = debug_cfg.output_dir / "debug_runs" / "sample"
    runs = list(run_root.iterdir())
    assert runs, "debug run directory should exist"
    run_dir = runs[0]

    expected_dirs = [
        "00_inputs",
        "10_preprocess",
        "20_desquebrar",
        "30_split_chunk",
        "40_translate",
        "50_cleanup_pre_refine",
        "60_refine",
        "99_reports",
    ]
    for dirname in expected_dirs:
        assert (run_dir / dirname).is_dir()

    translate_manifest = json.loads(read_text(run_dir / "40_translate" / "translate_manifest.json"))
    refine_manifest = json.loads(read_text(run_dir / "60_refine" / "refine_manifest.json"))
    run_summary = json.loads(read_text(run_dir / "99_reports" / "run_summary.json"))

    assert re.match(r"sample/\d{8}_\d{6}$", translate_manifest["run_id"])
    assert translate_manifest["run_id"] == refine_manifest["run_id"] == run_summary["run_id"]
    total_chunks = translate_manifest["chunking"]["total_chunks"]
    assert total_chunks >= 1
    assert len(translate_manifest["chunks"]) == min(
        total_chunks, debug_cfg.debug_max_chunks or total_chunks
    )
    total_refine_chunks = refine_manifest["refine"]["total_chunks"]
    assert len(refine_manifest["chunks"]) == min(
        total_refine_chunks, debug_cfg.debug_max_chunks or total_refine_chunks
    )

    def _assert_rel(path_str: str) -> None:
        """Verifica se o artefato foi registrado com caminho relativo."""
        assert path_str is None or not Path(path_str).is_absolute()
        assert path_str is None or not re.match(r"^[A-Za-z]:", path_str)

    for path_value in translate_manifest["input_paths"].values():
        _assert_rel(path_value)
    for chunk in translate_manifest["chunks"]:
        for key in ("debug_original", "debug_context", "debug_llm_raw", "debug_final"):
            _assert_rel(chunk["outputs"][key])
    for key, value in refine_manifest["input_paths"].items():
        _assert_rel(value)
    for chunk in refine_manifest["chunks"]:
        for key in ("debug_original", "debug_context", "debug_llm_raw", "debug_final"):
            _assert_rel(chunk["outputs"][key])
    for rel_path in run_summary["paths"].values():
        _assert_rel(rel_path)
    for rel_path in run_summary["final_outputs"].values():
        _assert_rel(rel_path)

    chunk_dir = run_dir / "40_translate" / "debug_traducao"
    assert (chunk_dir / "chunk001_original_en.txt").exists()
    assert (chunk_dir / "chunk001_final_pt.txt").exists()
    assert "[[OMITTED]]" in read_text(chunk_dir / "chunk001_llm_raw.txt")
    assert not (chunk_dir / "chunk002_final_pt.txt").exists()
    refine_chunk_dir = run_dir / "60_refine" / "debug_refine"
    assert (refine_chunk_dir / "chunk001_final_pt.txt").exists()
    assert not (refine_chunk_dir / "chunk002_final_pt.txt").exists()

    assert (run_dir / "30_split_chunk" / "sections.json").exists()
    assert (run_dir / "30_split_chunk" / "chunks.jsonl").exists()
    assert (run_dir / "99_reports" / "errors.jsonl").exists()
    debug_timings = json.loads(read_text(run_dir / "99_reports" / "timings.json"))
    assert debug_timings["status"] == "success"
    assert debug_timings["total_elapsed_human"]
    assert run_summary["final_outputs"]["pt"] == "sample_pt.md"
    assert run_summary["final_outputs"]["pt_refinado"] == "sample_pt_refinado.md"


def test_debug_run_writes_summary_on_failure(monkeypatch, tmp_path: Path) -> None:
    """Confirma o registro correto de métricas e artefatos na geração de artefatos de depuração."""
    import tradutor.main as main  # noqa: WPS433

    md_path = tmp_path / "sample.md"
    md_path.write_text("Short text content.", encoding="utf-8")

    cfg = AppConfig(output_dir=tmp_path)
    logger = setup_logging(logging.DEBUG)

    def _boom(*args, **kwargs):
        """Simula uma falha durante a execução."""
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "translate_document", _boom)
    monkeypatch.setattr(main, "LLMBackend", FakeLLMBackend)

    args = types.SimpleNamespace(
        command="traduz-md",
        input=str(md_path),
        backend="fake",
        model="fake-model",
        num_predict=64,
        request_timeout=30,
        preprocess_advanced=False,
        normalize_paragraphs=False,
        clear_cache=None,
        translate_allow_adaptation=False,
        use_glossary=False,
        manual_glossary=None,
        parallel=1,
        debug=True,
        debug_chunks=False,
        split_by_sections=False,
        fail_on_chunk_error=False,
        resume=False,
        no_refine=False,
        cleanup_before_refine="off",
        pdf_enabled=False,
    )

    with pytest.raises(RuntimeError):
        main.run_translate_md(args, cfg, logger)

    run_root = cfg.output_dir / "debug_runs" / "sample"
    runs = list(run_root.iterdir())
    assert runs, "debug run directory should exist on failure"
    run_dir = runs[0]
    assert (run_dir / "99_reports" / "errors.jsonl").exists()
    assert (run_dir / "99_reports" / "timings.json").exists()
    top_level_timings = json.loads(read_text(cfg.output_dir / "sample_timings.json"))
    assert top_level_timings["status"] == "failed"
    assert top_level_timings["failed_stage"] == "translate"
    summary_path = run_dir / "99_reports" / "run_summary.json"
    assert summary_path.exists()
    summary = json.loads(read_text(summary_path))
    assert any("run_aborted_at_stage:translate" == note for note in summary.get("notes", []))
    for rel_path in summary["paths"].values():
        assert not Path(rel_path).is_absolute()
        assert not re.match(r"^[A-Za-z]:", rel_path)
