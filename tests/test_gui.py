"""Testes da integração entre o formulário desktop e a CLI."""

from pathlib import Path

from tradutor.gui import (
    TranslationJob,
    build_cli_command,
    expected_output_paths,
    infer_translation_command,
    required_api_key_environment,
    validate_translation_job,
)


def _job(input_path: Path, **overrides: object) -> TranslationJob:
    """Cria uma execução mínima para os testes de comando."""
    values = {
        "input_path": input_path,
        "source_language": "ja",
        "backend": "ollama",
        "model": "modelo-local",
        "request_timeout": 180,
    }
    values.update(overrides)
    return TranslationJob(**values)


def test_infer_translation_command_by_extension() -> None:
    """Usa extração de PDF somente quando a entrada realmente é um PDF."""
    assert infer_translation_command(Path("volume.pdf")) == "traduz"
    assert infer_translation_command(Path("volume.md")) == "traduz-md"
    assert infer_translation_command(Path("volume.TXT")) == "traduz-md"


def test_build_pdf_command_with_optional_steps(tmp_path: Path) -> None:
    """Traduz as escolhas da GUI para as flags booleanas explícitas da CLI."""
    source = tmp_path / "volume.pdf"
    glossary = tmp_path / "glossario.json"
    source.write_bytes(b"%PDF")
    glossary.write_text("{}", encoding="utf-8")
    job = _job(
        source,
        refine=True,
        repair=False,
        export_pdf=True,
        resume=True,
        debug=True,
        glossary_path=glossary,
        desquebrar_mode="safe",
    )

    command = build_cli_command(job, python_executable="python-test")

    assert command[:4] == ["python-test", "-m", "tradutor.main", "traduz"]
    assert "--refine" in command
    assert "--no-translation-repair" in command
    assert "--pdf-enabled" in command
    assert command[-3:] == ["--use-glossary", "--manual-glossary", str(glossary)]
    assert command[command.index("--desquebrar-mode") + 1] == "safe"


def test_build_markdown_command_omits_pdf_preparation(tmp_path: Path) -> None:
    """Não envia à CLI opções exclusivas da extração de PDF."""
    source = tmp_path / "volume.md"
    source.write_text("# Chapter 1", encoding="utf-8")

    command = build_cli_command(_job(source))

    assert "traduz-md" in command
    assert "--desquebrar-mode" not in command
    assert "--no-refine" in command
    assert "--translation-repair" in command
    assert "--no-pdf-enabled" in command


def test_validate_translation_job_reports_actionable_errors(tmp_path: Path) -> None:
    """Reúne erros de formulário antes de criar um subprocesso."""
    job = _job(
        tmp_path / "inexistente.epub",
        backend="desconhecido",
        model=" ",
        request_timeout=0,
        desquebrar_mode="outro",
    )

    errors = validate_translation_job(job)

    assert len(errors) == 5
    assert any("arquivo de entrada" in error for error in errors)
    assert any("backend" in error for error in errors)


def test_external_backends_expose_only_environment_name() -> None:
    """Mantém credenciais fora do comando e centraliza seus nomes públicos."""
    assert required_api_key_environment("ollama") is None
    assert required_api_key_environment("gemini") == "GEMINI_API_KEY"
    assert required_api_key_environment("openai") == "OPENAI_API_KEY"


def test_expected_output_paths_follow_selected_finish(tmp_path: Path) -> None:
    """Aponta os arquivos principais sem depender dos relatórios auxiliares."""
    job = _job(Path("minha-obra.pdf"), refine=True, export_pdf=True)

    paths = expected_output_paths(job, tmp_path)

    assert paths == (
        tmp_path / "minha-obra_pt.md",
        tmp_path / "minha-obra_pt_refinado.md",
        tmp_path / "pdf" / "minha-obra_pt_refinado.pdf",
    )
