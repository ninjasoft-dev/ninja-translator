"""Benchmark de ponta a ponta para combinações de tradução e refino."""

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
from tradutor.desquebrar_safe import desquebrar_safe
from tradutor.glossary_utils import (
    build_glossary_state,
    format_manual_pairs_for_translation,
    resolve_manual_glossary_path,
)
from tradutor.llm_backend import LLMBackend
from tradutor.pdf_reader import extract_pdf_text
from tradutor.preprocess import preprocess_text
from tradutor.quality_checks import format_quality_cell, run_translation_quality_checks
from tradutor.refine import refine_markdown_file
from tradutor.translate import translate_document
from tradutor.utils import setup_logging


def slugify_model(name: str) -> str:
    """Converte em identificador modelo."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _normalize_base_url(endpoint: str) -> str:
    """Normaliza a URL-base do servidor local de modelos."""
    if endpoint.endswith("/api/generate"):
        return endpoint[: -len("/api/generate")]
    return endpoint.rstrip("/")


def build_backend(
    *,
    model: str,
    endpoint: str,
    cfg: AppConfig,
    logger: logging.Logger,
    stage: str,
) -> LLMBackend:
    """Monta backend."""
    is_refine = stage == "refine"
    return LLMBackend(
        backend="ollama",
        model=model,
        temperature=cfg.refine_temperature if is_refine else cfg.translate_temperature,
        logger=logger,
        base_url=_normalize_base_url(endpoint),
        request_timeout=cfg.request_timeout,
        repeat_penalty=cfg.refine_repeat_penalty if is_refine else cfg.translate_repeat_penalty,
        num_predict=cfg.refine_num_predict if is_refine else cfg.translate_num_predict,
        num_ctx=cfg.refine_num_ctx if is_refine else cfg.translate_num_ctx,
        keep_alive=getattr(cfg, "ollama_keep_alive", "30m"),
        api_mode=getattr(cfg, "ollama_api_mode", "generate"),
        think=getattr(cfg, "ollama_think", None),
    )


def _list_models_via_cli() -> list[str]:
    """Lista os modelos instalados por meio da ferramenta de linha de comando."""
    for cmd in (["ollama", "list", "--format", "json"], ["ollama", "list"]):
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
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
        names = []
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
    """Lista os modelos instalados por meio da API local."""
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
    """Lista instalados modelos."""
    models = _list_models_via_cli()
    if models:
        return models
    return sorted(_list_models_via_api(endpoint))


def resolve_models(requested: list[str] | None, installed: list[str], label: str) -> list[str]:
    """Resolve a seleção de modelos informada na linha de comando."""
    if requested:
        if installed:
            missing = [m for m in requested if m not in installed]
            available = [m for m in requested if m in installed]
            if missing:
                print(f"Atencao: ignorando modelos {label} nao instalados: {', '.join(missing)}")
            if available:
                return available
            raise SystemExit(f"Nenhum modelo {label} informado esta instalado segundo o Ollama.")
        return requested
    if not installed:
        raise SystemExit(
            "Nenhum modelo Ollama foi encontrado. Use --translate-models/--refine-models."
        )
    return installed


def read_and_prepare_input(
    input_path: Path,
    cfg: AppConfig,
    logger: logging.Logger,
    *,
    max_chars: int,
    desquebrar_mode: str,
) -> tuple[str, str]:
    """Lê e prepara o texto de entrada do benchmark."""
    if input_path.suffix.lower() == ".pdf":
        raw_text = extract_pdf_text(input_path, logger)
    else:
        raw_text = input_path.read_text(encoding="utf-8")
    if max_chars > 0:
        raw_text = raw_text[:max_chars]
    pre_text = preprocess_text(
        raw_text,
        logger,
        skip_front_matter=getattr(cfg, "skip_front_matter", False),
        noise_glossary_path=getattr(cfg, "preprocess_noise_glossary_path", None),
    )
    if desquebrar_mode == "safe":
        pre_text = desquebrar_safe(pre_text)
    return raw_text.strip(), pre_text.strip()


def write_quality_report(out_dir: Path, combo_slug: str, report: dict) -> str:
    """Grava qualidade relatório."""
    out_path = out_dir / f"{combo_slug}_qa.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path.name


def write_error_output(out_dir: Path, combo_slug: str, error: str) -> str:
    """Grava erro saída."""
    out_path = out_dir / f"{combo_slug}_erro.md"
    out_path.write_text(f"# Benchmark e2e falhou\n\n{error}\n", encoding="utf-8")
    return out_path.name


def write_summary(
    out_dir: Path,
    slug: str,
    input_path: Path,
    used_chars: int,
    endpoint: str,
    cfg: AppConfig,
    rows: list[dict[str, str | float]],
    *,
    glossary_path: Path | None,
    glossary_terms_count: int,
    desquebrar_mode: str,
) -> None:
    """Grava sumário."""
    ok_rows = [row for row in rows if row["status"] == "ok"]
    ranked = sorted(
        ok_rows,
        key=lambda row: (float(row.get("score", 0.0)), -float(row["total_elapsed"])),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    lines = [
        f"# Resumo de benchmark e2e - {slug}",
        "",
        f"- Arquivo de origem: {input_path}",
        f"- Caracteres usados: {used_chars}",
        f"- Endpoint: {endpoint}",
        f"- Desquebrar: {desquebrar_mode}",
        f"- Temperatura tradução: {cfg.translate_temperature}",
        f"- Temperatura refine: {cfg.refine_temperature}",
        f"- Chunk chars: {cfg.translate_chunk_chars}",
        f"- ollama_api_mode: {cfg.ollama_api_mode}",
        f"- ollama_think: {cfg.ollama_think}",
        f"- Glossário: {glossary_path if glossary_path else 'desativado'}",
        f"- Termos de glossário carregados: {glossary_terms_count}",
    ]
    if best:
        lines.extend(
            [
                f"- Melhor combinação por QA: {best['translate_model']} -> {best['refine_model']} ({best['quality']})",
            ]
        )
    lines.extend(
        [
            "",
            "| Tradutor | Refinador | Final | QA | QA JSON | Tradução (s) | Refine (s) | Total (s) | Status | Erro |",
            "|----------|-----------|-------|----|---------|--------------|------------|-----------|--------|------|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['translate_model']} | {row['refine_model']} | {row.get('final_file', '')} | {row.get('quality', '')} | {row.get('qa_file', '')} | {float(row.get('translate_elapsed', 0.0)):.2f} | {float(row.get('refine_elapsed', 0.0)):.2f} | {float(row.get('total_elapsed', 0.0)):.2f} | {row['status']} | {row.get('error', '')} |"
        )
    (out_dir / f"resumo_e2e_{slug}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Interpreta argumentos."""
    parser = argparse.ArgumentParser(description="Benchmark e2e de combinacoes tradutor/refinador.")
    parser.add_argument(
        "--input",
        required=True,
        help="Arquivo de entrada (.txt, .md ou .pdf)",
    )
    parser.add_argument(
        "--translate-models", nargs="*", help="Modelos Ollama para a etapa de traducao"
    )
    parser.add_argument("--refine-models", nargs="*", help="Modelos Ollama para a etapa de refine")
    parser.add_argument(
        "--same-model-only",
        action="store_true",
        help="Testa apenas pares com o mesmo nome de modelo.",
    )
    parser.add_argument(
        "--limit-combos",
        type=int,
        default=0,
        help="Limita a quantidade de combinacoes executadas",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2500,
        help="Maximo de caracteres do texto de entrada; use 0 para livro completo",
    )
    parser.add_argument("--out-dir", default="benchmark/e2e", help="Diretorio de saida")
    parser.add_argument("--desquebrar-mode", choices=["off", "safe"], default="safe")
    parser.add_argument(
        "--translate-temperature", type=float, help="Override de translate_temperature"
    )
    parser.add_argument("--refine-temperature", type=float, help="Override de refine_temperature")
    parser.add_argument("--num-ctx", type=int, help="Override de translate/refine num_ctx")
    parser.add_argument(
        "--translate-num-predict", type=int, help="Override de translate_num_predict"
    )
    parser.add_argument("--refine-num-predict", type=int, help="Override de refine_num_predict")
    parser.add_argument("--chunk-chars", type=int, help="Override de translate_chunk_chars")
    parser.add_argument("--timeout", type=int, help="Override de request_timeout")
    parser.add_argument("--use-glossary", action="store_true", help="Ativa glossario manual")
    parser.add_argument("--manual-glossary", help="Arquivo JSON de glossario manual")
    parser.add_argument("--auto-glossary-dir", help="Diretorio opcional com JSONs adicionais")
    parser.add_argument(
        "--ollama-api-mode",
        choices=["generate", "chat"],
        help="Override de ollama_api_mode do config.yaml.",
    )
    parser.add_argument(
        "--ollama-think",
        choices=["true", "false", "auto"],
        help="Override de ollama_think do config.yaml.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:11434/api/generate",
        help="Endpoint do Ollama",
    )
    return parser.parse_args()


def main() -> None:
    """Executa o benchmark de tradução e refino de ponta a ponta."""
    logger = setup_logging(logging.INFO)
    cfg = load_config()
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Arquivo de entrada não encontrado: {input_path}")

    think_override = cfg.ollama_think
    if args.ollama_think == "true":
        think_override = True
    elif args.ollama_think == "false":
        think_override = False
    elif args.ollama_think == "auto":
        think_override = None

    cfg = replace(
        cfg,
        translate_temperature=args.translate_temperature
        if args.translate_temperature is not None
        else cfg.translate_temperature,
        refine_temperature=args.refine_temperature
        if args.refine_temperature is not None
        else cfg.refine_temperature,
        translate_num_ctx=args.num_ctx if args.num_ctx is not None else cfg.translate_num_ctx,
        refine_num_ctx=args.num_ctx if args.num_ctx is not None else cfg.refine_num_ctx,
        translate_num_predict=args.translate_num_predict
        if args.translate_num_predict is not None
        else cfg.translate_num_predict,
        refine_num_predict=args.refine_num_predict
        if args.refine_num_predict is not None
        else cfg.refine_num_predict,
        translate_chunk_chars=args.chunk_chars
        if args.chunk_chars is not None
        else cfg.translate_chunk_chars,
        request_timeout=args.timeout if args.timeout is not None else cfg.request_timeout,
        ollama_api_mode=args.ollama_api_mode
        if args.ollama_api_mode is not None
        else cfg.ollama_api_mode,
        ollama_think=think_override,
    )

    installed = list_installed_models(args.endpoint)
    translate_models = resolve_models(args.translate_models, installed, "tradutor")
    refine_models = resolve_models(
        args.refine_models or args.translate_models, installed, "refinador"
    )
    if args.same_model_only:
        combos = [(model, model) for model in translate_models if model in refine_models]
    else:
        combos = [
            (translate_model, refine_model)
            for translate_model in translate_models
            for refine_model in refine_models
        ]
    if args.limit_combos > 0:
        combos = combos[: args.limit_combos]
    if not combos:
        raise SystemExit("Nenhuma combinacao de modelos para executar.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_root = out_dir / "_pipeline_state"
    state_root.mkdir(parents=True, exist_ok=True)

    source_text, prepared_text = read_and_prepare_input(
        input_path,
        cfg,
        logger,
        max_chars=args.max_chars,
        desquebrar_mode=args.desquebrar_mode,
    )
    slug = input_path.stem.lower()
    prepared_path = state_root / f"{slug}_prepared.md"
    prepared_path.write_text(prepared_text, encoding="utf-8")

    glossary_path: Path | None = None
    manual_dir: Path | None = None
    glossary_terms_count = 0
    if args.use_glossary:
        glossary_path = resolve_manual_glossary_path(args.manual_glossary)
        manual_dir = Path(args.auto_glossary_dir) if args.auto_glossary_dir else None
        preview_state = build_glossary_state(glossary_path, None, logger, manual_dir=manual_dir)
        if preview_state:
            glossary_terms_count = len(preview_state.manual_terms)

    translation_runs: dict[str, dict[str, str | float]] = {}
    for translate_model in dict.fromkeys(translate_model for translate_model, _ in combos):
        translate_slug = f"{slug}_{slugify_model(translate_model)}"
        translate_state = state_root / f"{translate_slug}_translate"
        translate_state.mkdir(parents=True, exist_ok=True)
        translate_cfg = replace(cfg, output_dir=translate_state)
        set_cache_base_dir(translate_cfg.output_dir)
        try:
            glossary_state_translate = None
            glossary_text = None
            if args.use_glossary and glossary_path:
                glossary_state_translate = build_glossary_state(
                    glossary_path,
                    None,
                    logger,
                    manual_dir=manual_dir,
                )
                if glossary_state_translate:
                    glossary_text = format_manual_pairs_for_translation(
                        glossary_state_translate.manual_terms, limit=30
                    )

            translate_backend = build_backend(
                model=translate_model,
                endpoint=args.endpoint,
                cfg=translate_cfg,
                logger=logger,
                stage="translate",
            )
            start = time.monotonic()
            translated = translate_document(
                pdf_text=prepared_text,
                backend=translate_backend,
                cfg=translate_cfg,
                logger=logger,
                source_slug=translate_slug,
                already_preprocessed=True,
                split_by_sections=translate_cfg.split_by_sections,
                allow_adaptation=translate_cfg.translate_allow_adaptation,
                fail_on_chunk_error=False,
                glossary_text=glossary_text,
                glossary_manual_terms=glossary_state_translate.manual_terms
                if glossary_state_translate
                else None,
            )
            translate_elapsed = time.monotonic() - start
            translated_path = out_dir / f"{translate_slug}_pt.md"
            translated_path.write_text(translated, encoding="utf-8")
            translation_runs[translate_model] = {
                "translated_path": str(translated_path),
                "translate_elapsed": translate_elapsed,
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            error = str(exc).replace("|", "\\|").replace("\n", " ")
            logger.error("Benchmark e2e falhou na traducao com %s: %s", translate_model, exc)
            error_file = write_error_output(out_dir, translate_slug, str(exc))
            translation_runs[translate_model] = {
                "translated_path": error_file,
                "translate_elapsed": 0.0,
                "status": "falhou",
                "error": error,
            }

    rows: list[dict[str, str | float]] = []
    for translate_model, refine_model in combos:
        combo_slug = f"{slug}_{slugify_model(translate_model)}__{slugify_model(refine_model)}"
        combo_state = state_root / combo_slug
        combo_state.mkdir(parents=True, exist_ok=True)
        combo_cfg = replace(cfg, output_dir=combo_state)
        set_cache_base_dir(combo_cfg.output_dir)
        translation_run = translation_runs.get(translate_model)
        if not translation_run or translation_run.get("status") != "ok":
            rows.append(
                {
                    "translate_model": translate_model,
                    "refine_model": refine_model,
                    "final_file": translation_run.get("translated_path", "")
                    if translation_run
                    else "",
                    "qa_file": "",
                    "score": 0.0,
                    "quality": "",
                    "translate_elapsed": float(translation_run.get("translate_elapsed", 0.0))
                    if translation_run
                    else 0.0,
                    "refine_elapsed": 0.0,
                    "total_elapsed": float(translation_run.get("translate_elapsed", 0.0))
                    if translation_run
                    else 0.0,
                    "status": "falhou",
                    "error": str(translation_run.get("error", "falha na traducao"))
                    if translation_run
                    else "falha na traducao",
                }
            )
            continue

        translated_path = Path(str(translation_run["translated_path"]))
        translate_elapsed = float(translation_run["translate_elapsed"])
        started_refine = time.monotonic()
        try:
            glossary_state = None
            if args.use_glossary and glossary_path:
                glossary_state = build_glossary_state(
                    glossary_path,
                    combo_state / "glossario_dinamico.json",
                    logger,
                    manual_dir=manual_dir,
                )

            refine_backend = build_backend(
                model=refine_model,
                endpoint=args.endpoint,
                cfg=combo_cfg,
                logger=logger,
                stage="refine",
            )
            refined_path = out_dir / f"{combo_slug}_pt_refinado.md"
            start = time.monotonic()
            refine_markdown_file(
                input_path=translated_path,
                output_path=refined_path,
                backend=refine_backend,
                cfg=combo_cfg,
                logger=logger,
                progress_path=combo_state / f"{combo_slug}_refine_progress.json",
                glossary_state=glossary_state,
                cleanup_mode=str(getattr(combo_cfg, "cleanup_before_refine", "off")),
            )
            refined = refined_path.read_text(encoding="utf-8")
            report = run_translation_quality_checks(
                source_text,
                refined,
                glossary_state.manual_terms if glossary_state else None,
            )
            qa_file = write_quality_report(out_dir, combo_slug, report)
            refine_elapsed = time.monotonic() - start
            total_elapsed = translate_elapsed + refine_elapsed
            rows.append(
                {
                    "translate_model": translate_model,
                    "refine_model": refine_model,
                    "final_file": refined_path.name,
                    "qa_file": qa_file,
                    "score": float(report["score"]),
                    "quality": format_quality_cell(report),
                    "translate_elapsed": translate_elapsed,
                    "refine_elapsed": refine_elapsed,
                    "total_elapsed": total_elapsed,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            refine_elapsed = time.monotonic() - started_refine
            total_elapsed = translate_elapsed + refine_elapsed
            error = str(exc).replace("|", "\\|").replace("\n", " ")
            logger.error(
                "Benchmark e2e falhou para %s -> %s: %s",
                translate_model,
                refine_model,
                exc,
            )
            error_file = write_error_output(out_dir, combo_slug, str(exc))
            rows.append(
                {
                    "translate_model": translate_model,
                    "refine_model": refine_model,
                    "final_file": error_file,
                    "qa_file": "",
                    "score": 0.0,
                    "quality": "",
                    "translate_elapsed": translate_elapsed,
                    "refine_elapsed": refine_elapsed,
                    "total_elapsed": total_elapsed,
                    "status": "falhou",
                    "error": error,
                }
            )

    write_summary(
        out_dir,
        slug,
        input_path,
        len(source_text),
        args.endpoint,
        cfg,
        rows,
        glossary_path=glossary_path,
        glossary_terms_count=glossary_terms_count,
        desquebrar_mode=args.desquebrar_mode,
    )


if __name__ == "__main__":
    main()
