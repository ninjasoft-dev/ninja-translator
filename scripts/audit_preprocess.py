from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradutor.config import load_config
from tradutor.preprocess import normalize_line_for_filters, preprocess_text

NOISE_TOKENS = (
    "newsletter",
    "table of contents",
    "download all your favorite light novels",
    "favorite light novels",
    "sign up for",
    "thank you for reading",
    "thank you for downloading",
    "visit us online",
    "join our community",
)

URL_RE = re.compile(
    r"(?:https?://|www\.)|(?:\w+\.(?:example|test|invalid)(?:[/?#]|$))",
    re.IGNORECASE,
)
STORY_MARKER_RE = re.compile(
    r"^(?:prologue|chapter\s+(?:\d+|one)\b|epilogue)",
    re.IGNORECASE,
)
TOC_LIKE_RE = re.compile(
    r"^(?:chapter\s+\d+|prologue|epilogue|afterword|color inserts|title page)$",
    re.IGNORECASE,
)


def _compact(text: str) -> str:
    """Compacta espaços e normaliza o texto para comparações tolerantes."""
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def _alpha_ratio(text: str) -> float:
    """Calcula a proporção de caracteres alfabéticos do texto."""
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isalpha()) / len(text)


def _is_obvious_noise(text: str) -> bool:
    """Verifica se a linha corresponde a um ruído configurado conhecido."""
    lowered = text.lower()
    if URL_RE.search(lowered):
        return True
    if any(tok in lowered for tok in NOISE_TOKENS):
        return True
    if TOC_LIKE_RE.fullmatch(lowered.strip(" .:-")):
        return True
    if re.fullmatch(r"\d{1,4}", lowered.strip()):
        return True
    return False


def _story_start_line(lines: list[str], skip_front_matter: bool) -> int:
    """Localiza a primeira linha que aparenta pertencer à narrativa."""
    if not skip_front_matter:
        return 1
    for idx, line in enumerate(lines, start=1):
        if STORY_MARKER_RE.match(normalize_line_for_filters(line)):
            return idx
    return 1


def _raw_missing_candidates(
    raw_text: str, clean_text: str, *, skip_front_matter: bool
) -> list[dict[str, Any]]:
    """Identifica trechos do original que podem ter sido removidos indevidamente."""
    raw_lines = raw_text.splitlines()
    start_line = _story_start_line(raw_lines, skip_front_matter)
    clean_compact = _compact(clean_text)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, line in enumerate(raw_lines, start=1):
        if idx < start_line:
            continue
        norm = normalize_line_for_filters(line)
        if len(norm) < 45:
            continue
        if _alpha_ratio(norm) < 0.55:
            continue
        if _is_obvious_noise(norm):
            continue
        compact = _compact(norm)
        if len(compact) < 35 or compact in clean_compact or compact in seen:
            continue
        seen.add(compact)
        candidates.append(
            {
                "line": idx,
                "text": norm,
                "prev": normalize_line_for_filters(raw_lines[idx - 2]) if idx >= 2 else "",
                "next": normalize_line_for_filters(raw_lines[idx]) if idx < len(raw_lines) else "",
            }
        )
    return candidates


def _suspicious_removed(stats: dict[str, Any]) -> list[dict[str, Any]]:
    """Seleciona remoções que merecem inspeção manual."""
    suspicious: list[dict[str, Any]] = []
    for item in stats.get("removed_full", []):
        text = str(item.get("text", ""))
        reason = str(item.get("reason", ""))
        if len(text) < 35:
            continue
        if _alpha_ratio(text) < 0.55:
            continue
        if _is_obvious_noise(text):
            continue
        if reason in {"promo", "footer"} or text.endswith((".", "!", "?", "”", '"')):
            suspicious.append(item)
    return suspicious


def _write_markdown_report(
    path: Path,
    *,
    input_path: Path,
    output_path: Path,
    stats: dict[str, Any],
    suspicious_removed: list[dict[str, Any]],
    raw_missing: list[dict[str, Any]],
) -> None:
    """Grava markdown relatório."""
    reason_counts = Counter(str(item.get("reason", "")) for item in stats.get("removed_full", []))
    lines = [
        "# Preprocess audit",
        "",
        f"- input: `{input_path}`",
        f"- preprocessed: `{output_path}`",
        f"- chars_in: {stats.get('chars_in')}",
        f"- chars_out: {stats.get('chars_out')}",
        f"- removed_full_count: {stats.get('removed_full_count')}",
        f"- noise_blocks_removed_count: {stats.get('noise_blocks_removed_count')}",
        f"- watermarks_remaining: {stats.get('watermarks_remaining')}",
        f"- urls_remaining_count: {stats.get('urls_remaining_count')}",
        f"- toc_remaining_count: {stats.get('toc_remaining_count')}",
        f"- suspicious_removed_count: {len(suspicious_removed)}",
        f"- raw_missing_candidates_count: {len(raw_missing)}",
        "",
        "## Removed reason counts",
        "",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"- {reason or '<empty>'}: {count}")
    lines.extend(["", "## Suspicious removed", ""])
    for item in suspicious_removed[:50]:
        lines.append(f"- [{item.get('reason')}] {item.get('text')}")
    if not suspicious_removed:
        lines.append("- none")
    lines.extend(["", "## Raw lines not found in clean text", ""])
    for item in raw_missing[:80]:
        lines.append(f"- L{item['line']}: {item['text']}")
        if item.get("prev"):
            lines.append(f"  - prev: {item['prev']}")
        if item.get("next"):
            lines.append(f"  - next: {item['next']}")
    if not raw_missing:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Executa a auditoria do pré-processamento pela linha de comando."""
    parser = argparse.ArgumentParser(description="Audit PDF preprocessing output.")
    parser.add_argument("--input", required=True, help="Raw extracted text file.")
    parser.add_argument(
        "--output-dir", default="saida/preprocess_audit", help="Audit output directory."
    )
    parser.add_argument(
        "--skip-front-matter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use config default when omitted.",
    )
    args = parser.parse_args()

    cfg = load_config()
    skip_front_matter = (
        cfg.skip_front_matter if args.skip_front_matter is None else args.skip_front_matter
    )
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_text = input_path.read_text(encoding="utf-8")
    clean_text, stats = preprocess_text(
        raw_text, return_stats=True, skip_front_matter=skip_front_matter
    )
    stem = input_path.stem.removesuffix("_raw_extracted")
    preprocessed_path = output_dir / f"{stem}_preprocessed.md"
    report_path = output_dir / f"{stem}_preprocess_report.json"
    audit_path = output_dir / f"{stem}_audit.json"
    md_path = output_dir / f"{stem}_audit.md"

    suspicious_removed = _suspicious_removed(stats)
    raw_missing = _raw_missing_candidates(raw_text, clean_text, skip_front_matter=skip_front_matter)
    audit = {
        "input": str(input_path),
        "preprocessed": str(preprocessed_path),
        "skip_front_matter": skip_front_matter,
        "stats_summary": {
            "chars_in": stats.get("chars_in"),
            "chars_out": stats.get("chars_out"),
            "removed_full_count": stats.get("removed_full_count"),
            "noise_blocks_removed_count": stats.get("noise_blocks_removed_count"),
            "watermarks_remaining": stats.get("watermarks_remaining"),
            "urls_remaining_count": stats.get("urls_remaining_count"),
            "toc_remaining_count": stats.get("toc_remaining_count"),
            "first_line": stats.get("first_line"),
        },
        "suspicious_removed": suspicious_removed,
        "raw_missing_candidates": raw_missing,
    }

    preprocessed_path.write_text(clean_text, encoding="utf-8")
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown_report(
        md_path,
        input_path=input_path,
        output_path=preprocessed_path,
        stats=stats,
        suspicious_removed=suspicious_removed,
        raw_missing=raw_missing,
    )

    print(f"preprocessed={preprocessed_path}")
    print(f"report={report_path}")
    print(f"audit={audit_path}")
    print(f"markdown={md_path}")
    print(f"suspicious_removed={len(suspicious_removed)}")
    print(f"raw_missing_candidates={len(raw_missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
