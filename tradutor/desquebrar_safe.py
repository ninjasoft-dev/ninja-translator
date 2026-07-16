"""Reconstrução determinística e conservadora de texto extraído de PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

# Adicionei de volta a aspa curva de fechamento (”) que o agente removeu
END_PUNCTUATION = {".", "?", "!", '"', "'", ":", "”"}
SHORT_TITLE_LEN = 25


def _is_blank(line: str) -> bool:
    """Verifica se a linha está completamente vazia ou contém apenas espaços."""
    return not line or not line.strip()


def _starts_lowercase(line: str) -> bool:
    """Verifica se o primeiro caractere alfabético está em minúscula."""
    for ch in line.lstrip():
        if ch.isalpha():
            return ch.islower()
        return False
    return False


def _is_dialogue_start(line: str) -> bool:
    """Detecta linhas iniciadas por aspas ou travessão."""
    stripped = line.lstrip()
    # Adicionei de volta a aspa curva de abertura (“)
    return stripped.startswith(('"', "'", "-", "—", "“", "”"))


def _is_title_like(line: str) -> bool:
    """Detecta se uma linha se parece com um título estrutural ou cabeçalho."""
    stripped = line.strip()
    if not stripped:
        return False

    letters = [c for c in stripped if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return True

    if len(stripped) <= SHORT_TITLE_LEN:
        last = stripped[-1]
        if last in ".?!,;:":
            return False

        words = [w for w in stripped.replace("-", " ").split() if w]
        alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
        if not alpha_words:
            return False
        uppercase_heads = [w for w in alpha_words if w[0].isupper()]
        if len(alpha_words) == 1:
            return bool(uppercase_heads)
        required = max(2, len(alpha_words) - 1)
        return len(uppercase_heads) >= required
    return False


def _should_join(current: str, nxt: str) -> bool:
    """
    Decide se a linha atual deve ser concatenada com a linha seguinte (nxt).
    Aplica regras heurísticas para evitar juntar parágrafos distintos.
    """
    # nxt ja deve ser uma linha com texto (o loop principal pula vazios)
    if _is_blank(current) or _is_blank(nxt):
        return False

    cur = current.rstrip()
    nxt_stripped = nxt.lstrip()

    if cur.endswith(tuple(END_PUNCTUATION)):
        return False
    if not _starts_lowercase(nxt_stripped):
        return False
    if _is_dialogue_start(nxt_stripped):
        return False
    if _is_title_like(cur) or _is_title_like(nxt_stripped):
        return False
    return True


def _merge_lines(current: str, nxt: str) -> str:
    """Une duas linhas, resolvendo hifenizações pendentes no fim da linha atual."""
    cur = current.rstrip()
    nxt_clean = nxt.lstrip()
    if cur.endswith("-"):
        cur = cur[:-1]
        return f"{cur}{nxt_clean}"
    return f"{cur} {nxt_clean}"


def safe_reflow(text: str) -> str:
    """Reconstrói parágrafos de forma determinística, preservando diálogos."""
    if not text:
        return text

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    output: list[str] = []
    idx = 0
    total = len(lines)

    while idx < total:
        line = lines[idx]

        # 1. Compressao de linhas vazias
        if _is_blank(line):
            if output and output[-1] != "":
                output.append("")
            idx += 1
            continue

        current = line.strip()

        # 2. Loop de tentativa de juncao (Smart Gap Skip Otimizado pelo Agente)
        while True:
            next_idx = idx + 1
            # Otimização do agente: while na mesma linha para pular vazios
            while next_idx < total and _is_blank(lines[next_idx]):
                next_idx += 1

            if next_idx >= total:
                idx = next_idx
                break

            nxt_line = lines[next_idx]

            if _should_join(current, nxt_line):
                current = _merge_lines(current, nxt_line)
                idx = next_idx  # Pula para a linha que foi consumida e continua o loop
                continue

            break

        output.append(current)
        idx += 1

    return "\n".join(output).strip()


def desquebrar_safe(text: str) -> str:
    """Oferece a reconstrução conservadora de parágrafos como API pública."""
    return safe_reflow(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refinador deterministico (safe_refiner).")
    parser.add_argument("--input", required=True, help="Arquivo de entrada extraido do PDF.")
    parser.add_argument("--output", required=True, help="Arquivo de saida com reflow seguro.")
    args = parser.parse_args()

    raw = Path(args.input).read_text(encoding="utf-8")
    refined = safe_reflow(raw)
    out_path = Path(args.output)
    out_path.write_text(refined, encoding="utf-8")
    print(f"safe_refiner: escrito em {out_path}")
