"""
Funções utilitárias compartilhadas.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, List, Sequence, Tuple


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configura o registro de eventos no console."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("tradutor")


def ensure_dir(path: Path) -> None:
    """Cria diretório se não existir."""
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path, encoding: str = "utf-8") -> str:
    """Lê arquivo texto com encoding definido."""
    return path.read_text(encoding=encoding)


def write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Escreve texto garantindo diretório."""
    ensure_dir(path.parent)
    path.write_text(content, encoding=encoding)


def chunk_by_paragraphs(
    paragraphs: Sequence[str],
    max_chars: int,
    logger: logging.Logger,
    label: str,
) -> List[str]:
    """
    Agrupa texto em chunks respeitando parágrafos e limites seguros de frase, sem perda de texto.
    """
    text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    if not text:
        return []

    boundary_re = re.compile(r"\n\n|[.!?][\"'”’)]?(?=\s|\n|$)")
    chunks: List[str] = []
    start = 0
    total_len = len(text)

    while start < total_len:
        max_end = start + max_chars
        if max_end >= total_len:
            chunks.append(text[start:])
            break

        window = text[start:max_end]
        end: int | None = None

        # Preferir o último limite seguro dentro da janela
        for match in boundary_re.finditer(window):
            end = start + match.end()

        if end is not None and end > start:
            chunk_len = end - start
            logger.debug("%s: chunk cortado em limite seguro (len=%d)", label, chunk_len)
        else:
            # Busca próximo limite seguro à frente; pode ultrapassar max_chars para não quebrar frases
            next_match = boundary_re.search(text, pos=max_end)
            if next_match:
                end = next_match.end()
                chunk_len = end - start
                logger.warning(
                    "%s: chunk excede max_chars para respeitar limite seguro (%d > %d)",
                    label,
                    chunk_len,
                    max_chars,
                )
            else:
                end = total_len
                chunk_len = end - start
                logger.warning(
                    "%s: sem limite seguro; consumindo resto (%d chars)",
                    label,
                    chunk_len,
                )

        chunks.append(text[start:end])
        start = end

    sum_len = sum(len(c) for c in chunks)
    if sum_len != total_len:
        logger.warning(
            "%s: soma dos chunks (%d) difere do texto original (%d)",
            label,
            sum_len,
            total_len,
        )

    return chunks


def timed(fn, *args, **kwargs) -> Tuple[float, Any]:
    """Executa função e retorna (segundos, resultado)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def dedent_triple(text: str) -> str:
    """Remove indentação mínima preservando quebras."""
    import textwrap

    return textwrap.dedent(text).strip()
