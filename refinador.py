"""Camada de compatibilidade para a interface moderna de refino."""

from __future__ import annotations

import sys

from tradutor.main import main


def _inject_subcommand(argv: list[str]) -> list[str]:
    """
    Garante que o subcomando 'refina' seja inserido se não houver.

    Mantém compatibilidade com chamadas legadas.
    """
    if len(argv) <= 1:
        return argv + ["refina"]
    if argv[1] in {"traduz", "refina"}:
        return argv
    return argv[:1] + ["refina"] + argv[1:]


if __name__ == "__main__":
    sys.argv = _inject_subcommand(sys.argv)
    main()
