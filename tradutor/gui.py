"""Contrato entre a interface gráfica e a CLI do tradutor."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_INPUT_SUFFIXES = frozenset({".pdf", ".md", ".txt"})
API_KEY_ENVIRONMENTS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


@dataclass(frozen=True)
class TranslationJob:
    """Representa uma execução iniciada pela interface gráfica."""

    input_path: Path
    source_language: str
    backend: str
    model: str
    request_timeout: int
    refine: bool = False
    repair: bool = True
    export_pdf: bool = False
    resume: bool = False
    debug: bool = False
    glossary_path: Path | None = None
    desquebrar_mode: str = "llm"


def infer_translation_command(input_path: Path) -> str:
    """Escolhe o subcomando da CLI conforme o formato da entrada."""
    suffix = input_path.suffix.casefold()
    if suffix == ".pdf":
        return "traduz"
    if suffix in {".md", ".txt"}:
        return "traduz-md"
    raise ValueError(f"Formato de entrada não suportado: {suffix or 'sem extensão'}.")


def required_api_key_environment(backend: str) -> str | None:
    """Retorna a variável de ambiente exigida pelo backend externo."""
    return API_KEY_ENVIRONMENTS.get(backend.casefold())


def validate_translation_job(job: TranslationJob) -> list[str]:
    """Valida os campos que podem ser verificados antes de iniciar a CLI."""
    errors: list[str] = []
    if not job.input_path.is_file():
        errors.append("Selecione um arquivo de entrada existente.")
    elif job.input_path.suffix.casefold() not in SUPPORTED_INPUT_SUFFIXES:
        errors.append("A entrada deve ser um arquivo PDF, Markdown ou TXT.")

    if job.backend not in {"ollama", "gemini", "openai"}:
        errors.append("Selecione um backend de tradução válido.")
    if not job.model.strip():
        errors.append("Informe o modelo que será usado na tradução.")
    if job.request_timeout < 1:
        errors.append("O timeout deve ser maior que zero.")
    if job.desquebrar_mode not in {"llm", "safe"}:
        errors.append("Selecione um modo de preparação de texto válido.")
    if job.glossary_path and not job.glossary_path.is_file():
        errors.append("O arquivo de glossário selecionado não existe.")
    return errors


def build_cli_command(
    job: TranslationJob,
    *,
    python_executable: str | None = None,
) -> list[str]:
    """Monta o comando da CLI sem incluir credenciais ou dados sensíveis."""
    command_name = infer_translation_command(job.input_path)
    command = [
        python_executable or sys.executable,
        "-m",
        "tradutor.main",
        command_name,
        "--input",
        str(job.input_path),
        "--source-language",
        job.source_language,
        "--backend",
        job.backend,
        "--model",
        job.model.strip(),
        "--request-timeout",
        str(job.request_timeout),
        "--refine" if job.refine else "--no-refine",
        "--translation-repair" if job.repair else "--no-translation-repair",
        "--pdf-enabled" if job.export_pdf else "--no-pdf-enabled",
    ]

    if command_name == "traduz":
        command.extend(("--desquebrar-mode", job.desquebrar_mode))
    if job.resume:
        command.append("--resume")
    if job.debug:
        command.extend(("--debug", "--debug-chunks"))
    if job.glossary_path:
        command.extend(("--use-glossary", "--manual-glossary", str(job.glossary_path)))
    return command


def expected_output_paths(job: TranslationJob, output_dir: Path) -> tuple[Path, ...]:
    """Calcula os principais arquivos que uma execução bem-sucedida deve gerar."""
    translated = output_dir / f"{job.input_path.stem}_pt.md"
    paths = [translated]
    final_markdown = translated
    if job.refine:
        final_markdown = output_dir / f"{job.input_path.stem}_pt_refinado.md"
        paths.append(final_markdown)
    if job.export_pdf:
        paths.append(output_dir / "pdf" / f"{final_markdown.stem}.pdf")
    return tuple(paths)


def main() -> None:
    """Inicia a aplicação desktop e apresenta uma mensagem útil sem a dependência visual."""
    try:
        from .gui_app import TranslatorApp
    except ModuleNotFoundError as exc:
        if exc.name != "customtkinter":
            raise
        raise SystemExit(
            "A interface gráfica requer CustomTkinter. "
            "Instale as dependências com: pip install -r requirements.txt"
        ) from exc

    app = TranslatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
