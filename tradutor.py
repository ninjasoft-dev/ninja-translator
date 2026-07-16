"""Camada de compatibilidade para a interface moderna de tradução."""

from __future__ import annotations

import sys

from tradutor.main import main


def _inject_subcommand(argv: list[str]) -> list[str]:
    """
    Garante que o subcomando 'traduz' seja inserido se não houver.

    Assim, chamadas legadas como `python tradutor.py --input ...` continuam funcionando.
    """
    if len(argv) <= 1:
        return argv + ["traduz"]
    if argv[1] in {"traduz", "refina"}:
        return argv

    global_flags = {"--debug"}
    globals_args: list[str] = []
    rest: list[str] = []

    args_iter = iter(argv[1:])
    for arg in args_iter:
        if arg in global_flags:
            globals_args.append(arg)
            continue
        rest.append(arg)

    return [argv[0], *globals_args, "traduz", *rest]


if __name__ == "__main__":
    sys.argv = _inject_subcommand(sys.argv)
    main()
