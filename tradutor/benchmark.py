"""Benchmark leve para comparar modelos em trechos de ficção."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from sacrebleu import corpus_bleu, corpus_chrf

from .config import load_config
from .llm_backend import LLMBackend
from .translate import translate_document
from .utils import setup_logging, timed


def _load_samples(path: Path) -> List[Dict[str, str]]:
    """Carrega as amostras textuais usadas pelo benchmark."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de amostras não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_benchmark(models: List[Dict]) -> None:
    """
    Executa benchmark de tradução comparando múltiplos modelos.

    models: lista de dicts com chaves name, backend, model, temperature.
    """
    samples_path = Path("tests/benchmark_samples.json")
    samples = _load_samples(samples_path)
    logger = setup_logging(logging.INFO)

    cfg = load_config()
    results = []
    for model_cfg in models:
        backend = LLMBackend(
            backend=model_cfg["backend"],
            model=model_cfg["model"],
            temperature=model_cfg["temperature"],
            logger=logger,
            request_timeout=cfg.request_timeout,
            repeat_penalty=model_cfg.get("repeat_penalty", cfg.translate_repeat_penalty),
            num_predict=model_cfg.get("num_predict", cfg.translate_num_predict),
            num_ctx=getattr(cfg, "translate_num_ctx", None),
            api_mode=getattr(cfg, "ollama_api_mode", "generate"),
            think=getattr(cfg, "ollama_think", None),
        )
        logger.info(
            "Benchmark com LLM: name=%s backend=%s model=%s temp=%.2f chunk=%d timeout=%ds num_predict=%d",
            model_cfg["name"],
            model_cfg["backend"],
            model_cfg["model"],
            model_cfg["temperature"],
            cfg.translate_chunk_chars,
            cfg.request_timeout,
            model_cfg.get("num_predict", cfg.translate_num_predict),
        )

        hypotheses: List[str] = []
        references: List[str] = []
        latencies: List[float] = []

        for sample in samples:
            latency, translation = timed(
                translate_document,
                pdf_text=sample["source"],
                backend=backend,
                cfg=cfg,
                logger=logger,
            )
            latencies.append(latency)
            hypotheses.append(translation)
            references.append(sample["reference"])

        bleu = corpus_bleu(hypotheses, [references]).score
        chrf = corpus_chrf(hypotheses, [references]).score
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        results.append(
            {
                "model_name": model_cfg["name"],
                "backend": model_cfg["backend"],
                "bleu": bleu,
                "chrf": chrf,
                "avg_latency": avg_latency,
            }
        )

    header = f"{'model_name':20} {'backend':10} {'BLEU':>8} {'chrF':>8} {'avg_latency_sec':>16}"
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res['model_name']:20} "
            f"{res['backend']:10} "
            f"{res['bleu']:8.2f} "
            f"{res['chrf']:8.3f} "
            f"{res['avg_latency']:16.2f}"
        )


DEFAULT_MODELS = [
    {
        "name": "modelo-local",
        "backend": "ollama",
        "model": "gemma3:4b",
        "temperature": 0.15,
    }
]


if __name__ == "__main__":
    run_benchmark(DEFAULT_MODELS)
