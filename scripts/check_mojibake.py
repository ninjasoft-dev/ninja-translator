from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradutor.mojibake import MOJIBAKE_TOKENS, scan_paths  # noqa: E402


def main(argv: list[str]) -> int:
    """Verifica arquivos em busca de sequências conhecidas de mojibake."""
    paths = [Path(path) for path in argv[1:]]
    errors = scan_paths(paths, tokens=MOJIBAKE_TOKENS) if paths else []
    for msg in errors:
        sys.stderr.write(msg + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
