"""
Utilitário para mesclar glossários entre volumes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

DEFAULT_OUTPUT = Path("MASTER_GLOSSARIO.json")


def load_terms(path: Path, logger: logging.Logger) -> List[Dict]:
    """Carrega termos."""
    if not path.exists():
        logger.warning("Arquivo %s não encontrado; ignorando.", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Falha ao ler %s: %s", path, exc)
        return []
    terms = data.get("terms") if isinstance(data, dict) else None
    return terms if isinstance(terms, list) else []


def merge_terms(inputs: List[Path], logger: logging.Logger) -> tuple[List[Dict], List[str]]:
    """Combina termos."""
    merged: Dict[str, Dict] = {}
    conflicts: List[str] = []

    for path in inputs:
        for term in load_terms(path, logger):
            key = str(term.get("key", "")).strip()
            pt = str(term.get("pt", "")).strip()
            if not key or not pt:
                continue
            locked = bool(term.get("locked"))
            existing = merged.get(key)
            if existing:
                if existing.get("locked"):
                    if pt != existing.get("pt"):
                        conflicts.append(
                            f'Conflito: "{key}" -> "{existing.get("pt")}" vs "{pt}" (mantido locked)'
                        )
                    continue
                if locked:
                    merged[key] = {**term, "locked": True}
                else:
                    # mantém o primeiro; se diferente, loga conflito
                    if pt != existing.get("pt"):
                        conflicts.append(
                            f'Conflito: "{key}" -> "{existing.get("pt")}" vs "{pt}" (mantido primeiro)'
                        )
            else:
                merged[key] = {**term, "locked": locked}
    return list(merged.values()), conflicts


def main() -> None:
    """Executa a interface de linha de comando de merge."""
    parser = argparse.ArgumentParser(
        description="Mescla glossários em um único MASTER_GLOSSARIO.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Diretório com arquivos de glossário (*.json)",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Arquivo de saída")
    args = parser.parse_args()

    logger = logging.getLogger("glossario.merge")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        logger.error("Diretório inválido: %s", input_dir)
        raise SystemExit(1)

    files = sorted(input_dir.glob("*.json"))
    if not files:
        logger.error("Nenhum glossário encontrado em %s", input_dir)
        raise SystemExit(1)

    merged_terms, conflicts = merge_terms(files, logger)
    payload = {"terms": merged_terms}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Glossário mesclado salvo em %s (%d termos).", output_path, len(merged_terms))

    if conflicts:
        log_path = Path("saida/glossario_conflicts.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(conflicts), encoding="utf-8")
        logger.info("Conflitos registrados em %s", log_path)


if __name__ == "__main__":
    main()
