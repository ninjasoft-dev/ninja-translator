from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tradutor.glossary_audit import audit_glossary_data, format_audit_report


def main() -> int:
    """Executa a auditoria de glossário pela linha de comando."""
    parser = argparse.ArgumentParser(
        description="Audit glossary aliases and duplicate/conflicting terms."
    )
    parser.add_argument("path", help="Glossary JSON path.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum issues shown in text mode.")
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with status 1 when duplicate keys or ambiguous aliases are found.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    data = json.loads(path.read_text(encoding="utf-8"))
    report = audit_glossary_data(data)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_audit_report(report, limit=args.limit))

    summary = report.get("summary", {})
    has_blocking_issues = bool(
        summary.get("duplicate_keys") or summary.get("ambiguous_source_aliases")
    )
    return 1 if args.fail_on_issues and has_blocking_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
