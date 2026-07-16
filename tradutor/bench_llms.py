"""Benchmark para comparar modelos no pipeline de tradução."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import requests

from tradutor.cache_utils import set_cache_base_dir
from tradutor.config import AppConfig, load_config
from tradutor.glossary_utils import (
    build_glossary_state,
    format_manual_pairs_for_translation,
    resolve_manual_glossary_path,
)
from tradutor.languages import (
    SUPPORTED_SOURCE_LANGUAGE_CODES,
    detect_source_language,
)
from tradutor.llm_backend import LLMBackend
from tradutor.pdf_reader import extract_pdf_text
from tradutor.quality_checks import format_quality_cell, run_translation_quality_checks
from tradutor.translate import build_translation_prompt, translate_document
from tradutor.utils import setup_logging


def slugify_model(name: str) -> str:
    """Converte em identificador modelo."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _normalize_base_url(endpoint: str) -> str:
    """Normaliza a URL-base do servidor local de modelos."""
    if endpoint.endswith("/api/generate"):
        return endpoint[: -len("/api/generate")]
    return endpoint.rstrip("/")


def build_backend(model: str, endpoint: str, cfg: AppConfig, logger: logging.Logger) -> LLMBackend:
    """Monta backend."""
    return LLMBackend(
        backend="ollama",
        model=model,
        temperature=cfg.translate_temperature,
        logger=logger,
        base_url=_normalize_base_url(endpoint),
        request_timeout=cfg.request_timeout,
        repeat_penalty=cfg.translate_repeat_penalty,
        num_predict=cfg.translate_num_predict,
        num_ctx=cfg.translate_num_ctx,
        keep_alive=getattr(cfg, "ollama_keep_alive", "30m"),
        api_mode=getattr(cfg, "ollama_api_mode", "generate"),
        think=getattr(cfg, "ollama_think", None),
    )


def call_ollama_single_prompt(
    model: str,
    prompt: str,
    endpoint: str,
    cfg: AppConfig,
    logger: logging.Logger,
) -> tuple[str, float]:
    """Executa ollama único prompt."""
    backend = build_backend(model=model, endpoint=endpoint, cfg=cfg, logger=logger)
    start = time.monotonic()
    try:
        response = backend.generate(prompt)
        elapsed = time.monotonic() - start
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao chamar Ollama para modelo '{model}' em {endpoint}: {exc}"
        ) from exc
    return response.text, elapsed


def call_ollama_pipeline(
    model: str,
    text: str,
    endpoint: str,
    cfg: AppConfig,
    logger: logging.Logger,
    source_slug: str,
    glossary_text: str | None = None,
    glossary_manual_terms: list[dict] | None = None,
    source_language: str = "auto",
) -> tuple[str, float]:
    """Executa o pipeline de tradução com o backend local."""
    backend = build_backend(model=model, endpoint=endpoint, cfg=cfg, logger=logger)
    start = time.monotonic()
    translated = translate_document(
        pdf_text=text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug=f"{source_slug}_{slugify_model(model)}",
        already_preprocessed=False,
        split_by_sections=cfg.split_by_sections,
        allow_adaptation=cfg.translate_allow_adaptation,
        fail_on_chunk_error=False,
        glossary_text=glossary_text,
        glossary_manual_terms=glossary_manual_terms,
        source_language=source_language,
    )
    elapsed = time.monotonic() - start
    return translated, elapsed


def _list_models_via_cli() -> list[str]:
    """
    Usa `ollama list` para obter modelos instalados. Retorna lista vazia em caso de falha.
    """
    cmd_json = ["ollama", "list", "--format", "json"]
    for cmd in (cmd_json, ["ollama", "list"]):
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
        except Exception:
            continue
        output = result.stdout.strip()
        if not output:
            continue
        try:
            data = json.loads(output)
            names = [item["name"] for item in data if isinstance(item, dict) and "name" in item]
            if names:
                return names
        except Exception:
            pass
        names: list[str] = []
        for line in output.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("name"):
                continue
            parts = line.split()
            if parts:
                names.append(parts[0])
        if names:
            return names
    return []


def _list_models_via_api(endpoint: str) -> set[str]:
    """
    Obtém a lista de modelos instalados no Ollama a partir de /api/tags.
    Se falhar, retorna conjunto vazio para nao bloquear a execução.
    """
    tags_url = endpoint.rstrip("/")
    if tags_url.endswith("/generate"):
        tags_url = tags_url.rsplit("/", 1)[0] + "/tags"
    else:
        tags_url = tags_url + "/tags"
    try:
        resp = requests.get(tags_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {m["name"] for m in data.get("models", []) if "name" in m}
    except Exception:
        return set()


def list_installed_models(endpoint: str) -> list[str]:
    """
    Descobre modelos usando `ollama list` (preferencial) ou /api/tags.
    """
    models = _list_models_via_cli()
    if models:
        return models
    return sorted(_list_models_via_api(endpoint))


def read_input(path: Path, max_chars: int) -> str:
    """Lê entrada."""
    if path.suffix.lower() == ".pdf":
        text = extract_pdf_text(path, logger=None)
    else:
        text = path.read_text(encoding="utf-8")
    if max_chars > 0:
        text = text[:max_chars]
    return text.strip()


def write_model_output(
    out_dir: Path,
    slug: str,
    model: str,
    translated: str,
    elapsed: float,
    input_path: Path,
) -> str:
    """Grava modelo saída."""
    model_slug = slugify_model(model)
    out_path = out_dir / f"{slug}_{model_slug}.md"
    header = [
        f"# Benchmark de traducao - {model}",
        f"- Modelo: {model}",
        f"- Arquivo de origem: {input_path}",
        f"- Tempo de resposta: {elapsed:.2f} s",
        "",
    ]
    out_path.write_text("\n".join(header) + "\n" + translated, encoding="utf-8")
    return out_path.name


def write_error_output(
    out_dir: Path, slug: str, model: str, elapsed: float, input_path: Path, error: str
) -> str:
    """Grava erro saída."""
    model_slug = slugify_model(model)
    out_path = out_dir / f"{slug}_{model_slug}_erro.md"
    header = [
        f"# Benchmark de traducao - {model}",
        f"- Modelo: {model}",
        f"- Arquivo de origem: {input_path}",
        f"- Tempo ate falha: {elapsed:.2f} s",
        "- Status: falhou",
        "",
        "## Erro",
        "",
        error,
        "",
    ]
    out_path.write_text("\n".join(header), encoding="utf-8")
    return out_path.name


def write_quality_report(out_dir: Path, slug: str, model: str, report: dict) -> str:
    """Grava qualidade relatório."""
    model_slug = slugify_model(model)
    out_path = out_dir / f"{slug}_{model_slug}_qa.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path.name


def write_summary(
    out_dir: Path,
    slug: str,
    input_path: Path,
    used_chars: int,
    endpoint: str,
    mode: str,
    cfg: AppConfig,
    rows: list[dict[str, str | float]],
    glossary_path: Path | None = None,
    glossary_terms_count: int = 0,
) -> None:
    """Grava sumário."""
    lines = [
        f"# Resumo de benchmark de traducao - {slug}",
        "",
        f"- Arquivo de origem: {input_path}",
        f"- Caracteres usados: {used_chars}",
        f"- Endpoint: {endpoint}",
        f"- Modo: {mode}",
        f"- Chunk chars: {cfg.translate_chunk_chars}",
        f"- Temperatura: {cfg.translate_temperature}",
        f"- num_ctx: {cfg.translate_num_ctx}",
        f"- num_predict: {cfg.translate_num_predict}",
        f"- repeat_penalty: {cfg.translate_repeat_penalty}",
        f"- ollama_api_mode: {cfg.ollama_api_mode}",
        f"- ollama_think: {cfg.ollama_think}",
        f"- Glossário: {glossary_path if glossary_path else 'desativado'}",
        f"- Termos de glossário carregados: {glossary_terms_count}",
        "",
        "| Modelo | Arquivo de saida | QA | Relatório QA | Tempo (s) | Status | Erro |",
        "|--------|------------------|----|--------------|-----------|--------|------|",
    ]
    for row in rows:
        elapsed = float(row["elapsed"])
        lines.append(
            f"| {row['model']} | {row['file']} | {row.get('quality', '')} | {row.get('qa_file', '')} | {elapsed:.2f} | {row['status']} | {row.get('error', '')} |"
        )
    (out_dir / f"resumo_{slug}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Interpreta argumentos."""
    parser = argparse.ArgumentParser(description="Benchmark de traducao com varios modelos Ollama.")
    parser.add_argument(
        "--input",
        required=True,
        help="Arquivo de entrada (.txt, .md ou .pdf)",
    )
    parser.add_argument(
        "--source-language",
        choices=SUPPORTED_SOURCE_LANGUAGE_CODES,
        default="auto",
        help="Idioma de origem; auto usa detecção heurística.",
    )
    parser.add_argument("--models", nargs="*", help="Lista de modelos Ollama a usar")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1500,
        help="Maximo de caracteres do texto de entrada",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmark/traducao",
        help="Diretorio de saida para resultados",
    )
    parser.add_argument(
        "--single-prompt",
        action="store_true",
        help="Modo legado: envia todo o texto em uma unica chamada, sem chunking/retry do pipeline.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override de translate_temperature do config.yaml",
    )
    parser.add_argument("--num-ctx", type=int, help="Override de translate_num_ctx do config.yaml")
    parser.add_argument(
        "--num-predict",
        type=int,
        help="Override de translate_num_predict do config.yaml",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        help="Override de translate_repeat_penalty do config.yaml",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        help="Override de translate_chunk_chars do config.yaml",
    )
    parser.add_argument("--timeout", type=int, help="Override de request_timeout do config.yaml")
    parser.add_argument(
        "--use-glossary",
        action="store_true",
        help="Ativa glossário manual no benchmark e no relatório de QA.",
    )
    parser.add_argument(
        "--manual-glossary",
        help="Arquivo JSON de glossário manual (padrão: glossario/glossario_manual.json ou glossario/glossario_geral.json).",
    )
    parser.add_argument(
        "--auto-glossary-dir",
        help="Diretório opcional com JSONs adicionais de glossário manual.",
    )
    parser.add_argument(
        "--ollama-api-mode",
        choices=["generate", "chat"],
        help="Override de ollama_api_mode do config.yaml.",
    )
    parser.add_argument(
        "--ollama-think",
        choices=["true", "false", "auto"],
        help="Override de ollama_think do config.yaml. Use false para modelos que gastam tokens em thinking.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:11434/api/generate",
        help="Endpoint do Ollama (default http://localhost:11434/api/generate)",
    )
    return parser.parse_args()


def main() -> None:
    """Executa o benchmark dos modelos de tradução selecionados."""
    logger = setup_logging(logging.INFO)
    cfg = load_config()
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Arquivo de entrada não encontrado: {input_path}")

    installed = list_installed_models(args.endpoint)
    if args.models:
        models = args.models
        if installed:
            missing = [m for m in models if m not in installed]
            available = [m for m in models if m in installed]
            if missing:
                print(f"Atencao: ignorando modelos nao instalados: {', '.join(missing)}")
            if available:
                models = available
            elif missing:
                raise SystemExit("Nenhum dos modelos informados esta instalado segundo o Ollama.")
    else:
        models = installed
        if not models:
            raise SystemExit(
                "Nenhum modelo Ollama foi encontrado. Rode `ollama list` para confirmar as instalacoes ou use --models."
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir = out_dir / "_pipeline_state"
    think_override = None
    if args.ollama_think == "true":
        think_override = True
    elif args.ollama_think == "false":
        think_override = False
    elif args.ollama_think == "auto":
        think_override = None
    else:
        think_override = cfg.ollama_think
    cfg = replace(
        cfg,
        output_dir=state_dir,
        translate_temperature=args.temperature
        if args.temperature is not None
        else cfg.translate_temperature,
        translate_num_ctx=args.num_ctx if args.num_ctx is not None else cfg.translate_num_ctx,
        translate_num_predict=args.num_predict
        if args.num_predict is not None
        else cfg.translate_num_predict,
        translate_repeat_penalty=args.repeat_penalty
        if args.repeat_penalty is not None
        else cfg.translate_repeat_penalty,
        translate_chunk_chars=args.chunk_chars
        if args.chunk_chars is not None
        else cfg.translate_chunk_chars,
        request_timeout=args.timeout if args.timeout is not None else cfg.request_timeout,
        ollama_api_mode=args.ollama_api_mode
        if args.ollama_api_mode is not None
        else cfg.ollama_api_mode,
        ollama_think=think_override,
    )
    set_cache_base_dir(cfg.output_dir)

    text = read_input(input_path, max_chars=args.max_chars)
    glossary_path: Path | None = None
    glossary_manual_terms: list[dict] | None = None
    glossary_text: str | None = None
    if args.use_glossary:
        glossary_path = resolve_manual_glossary_path(args.manual_glossary)
        manual_dir = Path(args.auto_glossary_dir) if args.auto_glossary_dir else None
        glossary_state = build_glossary_state(
            manual_path=glossary_path,
            dynamic_path=None,
            logger=logger,
            manual_dir=manual_dir,
        )
        if glossary_state:
            glossary_manual_terms = glossary_state.manual_terms
            glossary_text = format_manual_pairs_for_translation(glossary_manual_terms, limit=30)
            logger.info(
                "Glossário do benchmark carregado: %d termos de %s",
                len(glossary_manual_terms),
                glossary_path,
            )

    slug = input_path.stem.lower()
    resolved_source_language = detect_source_language(text, args.source_language)
    rows: list[dict[str, str | float]] = []

    for model in models:
        started = time.monotonic()
        try:
            if args.single_prompt:
                prompt = build_translation_prompt(
                    text,
                    glossary_text=glossary_text,
                    allow_adaptation=cfg.translate_allow_adaptation,
                    source_language=resolved_source_language,
                )
                translated, elapsed = call_ollama_single_prompt(
                    model=model,
                    prompt=prompt,
                    endpoint=args.endpoint,
                    cfg=cfg,
                    logger=logger,
                )
            else:
                translated, elapsed = call_ollama_pipeline(
                    model=model,
                    text=text,
                    endpoint=args.endpoint,
                    cfg=cfg,
                    logger=logger,
                    source_slug=slug,
                    glossary_text=glossary_text,
                    glossary_manual_terms=glossary_manual_terms,
                    source_language=resolved_source_language,
                )
            fname = write_model_output(out_dir, slug, model, translated, elapsed, input_path)
            quality_report = run_translation_quality_checks(
                text,
                translated,
                glossary_manual_terms,
                source_language=resolved_source_language,
            )
            qa_file = write_quality_report(out_dir, slug, model, quality_report)
            rows.append(
                {
                    "model": model,
                    "file": fname,
                    "qa_file": qa_file,
                    "quality": format_quality_cell(quality_report),
                    "elapsed": elapsed,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            error = str(exc).replace("|", "\\|").replace("\n", " ")
            logger.error("Benchmark de traducao falhou para %s: %s", model, exc)
            fname = write_error_output(out_dir, slug, model, elapsed, input_path, str(exc))
            rows.append(
                {
                    "model": model,
                    "file": fname,
                    "qa_file": "",
                    "quality": "",
                    "elapsed": elapsed,
                    "status": "falhou",
                    "error": error,
                }
            )

    mode = "single-prompt" if args.single_prompt else "pipeline"
    write_summary(
        out_dir,
        slug,
        input_path,
        len(text),
        args.endpoint,
        mode,
        cfg,
        rows,
        glossary_path=glossary_path,
        glossary_terms_count=len(glossary_manual_terms or []),
    )


if __name__ == "__main__":
    main()
