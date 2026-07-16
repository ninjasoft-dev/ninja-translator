"""Interface de linha de comando para a reconstrução de parágrafos."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tradutor.config import load_config
from tradutor.desquebrar import desquebrar_text
from tradutor.llm_backend import LLMBackend
from tradutor.utils import write_text


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos da ferramenta de desquebra."""
    parser = argparse.ArgumentParser(
        description="Aplica desquebrar em um arquivo de texto/Markdown."
    )
    parser.add_argument("--input", required=True, help="Arquivo de entrada (txt/md).")
    parser.add_argument("--output", help="Arquivo de saída (padrão: <nome>_desquebrado.md).")
    parser.add_argument("--config", help="Caminho opcional para config.yaml.")
    parser.add_argument("--debug", action="store_true", help="Ativa logs detalhados.")
    return parser


def main() -> None:
    """Executa a reconstrução de parágrafos pela linha de comando."""
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    logger = logging.getLogger("desquebrar")
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Arquivo de entrada não encontrado: {inp}")
    output = (
        Path(args.output) if args.output else inp.with_name(f"{inp.stem}_desquebrado{inp.suffix}")
    )

    backend = LLMBackend(
        backend=getattr(cfg, "desquebrar_backend", "ollama"),
        model=getattr(cfg, "desquebrar_model", ""),
        temperature=getattr(cfg, "desquebrar_temperature", 0.0),
        logger=logger,
        request_timeout=getattr(cfg, "request_timeout", 120),
        repeat_penalty=getattr(cfg, "desquebrar_repeat_penalty", None),
        num_predict=getattr(cfg, "desquebrar_num_predict", 1024),
    )

    text = inp.read_text(encoding="utf-8")
    cleaned, _stats = desquebrar_text(
        text,
        cfg,
        logger,
        backend=backend,
        chunk_chars=getattr(cfg, "desquebrar_chunk_chars", 2400),
    )
    write_text(output, cleaned)
    logger.info("Arquivo desquebrado salvo em %s", output)


if __name__ == "__main__":
    main()
