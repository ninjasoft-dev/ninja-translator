from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AppConfig

DEBUG_SUBDIRS = (
    "00_inputs",
    "10_preprocess",
    "20_desquebrar",
    "30_split_chunk",
    "40_translate",
    "45_repair",
    "50_cleanup_pre_refine",
    "60_refine",
    "99_reports",
)


def _to_jsonable(value: Any) -> Any:
    """Converte valores internos em estruturas serializáveis como JSON."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _git_sha() -> str | None:
    """Obtém o identificador do commit atual, quando disponível."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
        return sha if sha else None
    except Exception:
        return None


@dataclass
class DebugRunWriter:
    """Organiza os artefatos de depuração produzidos por uma execução."""

    output_dir: Path
    slug: str
    input_kind: str
    timestamp: str
    run_dir: Path
    run_id: str
    max_chunks: int | None = None
    max_chars_per_file: int | None = None
    store_llm_raw: bool = True

    preprocessed_rel: str | None = None
    desquebrado_rel: str | None = None
    pre_refine_rel: str | None = None
    pt_output_rel: str | None = None
    pt_refined_rel: str | None = None

    @classmethod
    def create(
        cls,
        *,
        output_dir: Path,
        slug: str,
        input_kind: str,
        max_chunks: int | None = None,
        max_chars_per_file: int | None = None,
        store_llm_raw: bool = True,
        timestamp: datetime | None = None,
    ) -> "DebugRunWriter":
        """Cria a árvore de diretórios de uma nova execução de depuração."""
        if timestamp is None:
            timestamp = datetime.now()
        stamp = timestamp.strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / "debug_runs" / slug / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        for sub in DEBUG_SUBDIRS:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        run_id = f"{slug}/{stamp}"
        return cls(
            output_dir=output_dir,
            slug=slug,
            input_kind=input_kind,
            timestamp=stamp,
            run_dir=run_dir,
            run_id=run_id,
            max_chunks=max_chunks,
            max_chars_per_file=max_chars_per_file,
            store_llm_raw=store_llm_raw,
        )

    def stage_dir(self, stage: str) -> Path:
        """Cria e retorna o diretório de artefatos de uma etapa."""
        return self.run_dir / stage

    def rel_path(self, path: Path) -> str:
        """Converte um caminho de artefato em referência relativa à execução."""
        return path.relative_to(self.run_dir).as_posix()

    def sha256_text(self, text: str) -> str:
        """Calcula o SHA-256 de um conteúdo textual."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def write_json(self, rel_path: str | Path, payload: dict) -> None:
        """Grava json."""
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_jsonl(self, rel_path: str | Path, payload: dict) -> None:
        """Acrescenta um registro serializado ao arquivo JSONL."""
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_text(self, rel_path: str | Path, text: str, *, allow_truncate: bool = True) -> str:
        """Grava texto."""
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        full_hash = self.sha256_text(text)
        if (
            allow_truncate
            and self.max_chars_per_file is not None
            and len(text) > self.max_chars_per_file
        ):
            text = text[: self.max_chars_per_file]
        path.write_text(text, encoding="utf-8")
        return full_hash

    def should_write_chunk(self, chunk_index: int) -> bool:
        """Indica se deve gravar chunk."""
        if self.max_chunks is None:
            return True
        return chunk_index <= self.max_chunks

    def write_args(self, args: dict, cfg: AppConfig) -> None:
        """Grava argumentos."""
        payload = {
            "args": _to_jsonable(args),
            "config": _to_jsonable(cfg.__dict__),
        }
        self.write_json("00_inputs/args.json", payload)

    def write_backend(self, payload: dict) -> None:
        """Grava backend."""
        self.write_json("00_inputs/backend.json", payload)

    def write_versions(
        self,
        translate_prompt_hash: str | None,
        refine_prompt_hash: str | None,
        repair_prompt_hash: str | None = None,
    ) -> None:
        """Grava as versões das dependências relevantes."""
        version_path = Path(__file__).parent / "VERSION"
        try:
            pipeline_version = version_path.read_text(encoding="utf-8").strip()
        except Exception:
            pipeline_version = "unknown"
        payload = {
            "pipeline_version": pipeline_version,
            "prompt_hashes": {
                "translate": translate_prompt_hash,
                "repair": repair_prompt_hash,
                "refine": refine_prompt_hash,
            },
            "git_sha": _git_sha(),
        }
        self.write_json("00_inputs/versions.json", payload)

    def write_timing(self, timing_data: dict) -> None:
        """Grava as métricas de duração das etapas."""
        self.write_json("99_reports/timings.json", timing_data)

    def write_run_summary(self, payload: dict) -> None:
        """Grava o resumo consolidado da execução."""
        errors_path = self.run_dir / "99_reports" / "errors.jsonl"
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        errors_path.touch(exist_ok=True)
        self.write_json("99_reports/run_summary.json", payload)

    def write_error(self, payload: dict) -> None:
        """Grava erro."""
        self.append_jsonl("99_reports/errors.jsonl", payload)

    def write_preprocess_report(self, payload: dict) -> None:
        """Grava pré-processamento relatório."""
        self.write_json("10_preprocess/preprocess_report.json", payload)

    def write_desquebrar_report(self, payload: dict) -> None:
        """Grava o relatório gerado pela etapa de desquebra."""
        self.write_json("20_desquebrar/desquebrar_report.json", payload)

    def write_cleanup_report(self, payload: dict) -> None:
        """Grava limpeza relatório."""
        self.write_json("50_cleanup_pre_refine/cleanup_report.json", payload)

    def write_manifest(self, stage: str, payload: dict) -> None:
        """Grava manifesto."""
        if stage == "translate":
            self.write_json("40_translate/translate_manifest.json", payload)
        elif stage == "repair":
            self.write_json("45_repair/repair_manifest.json", payload)
        elif stage == "refine":
            self.write_json("60_refine/refine_manifest.json", payload)

    def write_run_metadata(
        self,
        *,
        args: dict,
        cfg: AppConfig,
        translate_prompt_hash: str | None,
        refine_prompt_hash: str | None,
        repair_prompt_hash: str | None = None,
    ) -> None:
        """Grava os metadados necessários para reproduzir a execução."""
        self.write_args(args, cfg)
        self.write_versions(translate_prompt_hash, refine_prompt_hash, repair_prompt_hash)
