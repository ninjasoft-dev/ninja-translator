"""Verificações editoriais opcionais de consistência entre volumes."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _volume_key(name: str) -> str:
    """Extrai uma chave de ordenação numérica do nome do volume."""
    m = re.search(r"vol[\s_-]*(\d+)", name, flags=re.IGNORECASE)
    if m:
        return f"Vol {int(m.group(1)):02d}"
    return name


def load_volume_texts(volume_dir: str) -> Dict[str, str]:
    """Carrega e ordena os textos dos volumes informados."""
    base = Path(volume_dir)
    volumes: Dict[str, str] = {}
    for path in sorted(base.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue
        key = _volume_key(path.stem)
        volumes[key] = content
    return volumes


def load_glossaries(glossario_dir: str, master_glossario: str | None = None) -> Dict[str, Dict]:
    """Carrega os glossários associados aos volumes analisados."""
    base = Path(glossario_dir)
    gloss: Dict[str, Dict] = {}
    for path in sorted(base.glob("glossario_vol*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = _volume_key(path.stem)
        gloss[key] = data
    if master_glossario:
        try:
            data = json.loads(Path(master_glossario).read_text(encoding="utf-8"))
            gloss["MASTER"] = data
        except Exception:
            pass
    return gloss


def build_character_registry(glossarios: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
    """Monta o cadastro consolidado de personagens."""
    registry: Dict[str, Dict[str, Any]] = {}
    for vol_key, data in glossarios.items():
        terms = data.get("terms") if isinstance(data, dict) else None
        if not isinstance(terms, list):
            continue
        for term in terms:
            if not isinstance(term, dict):
                continue
            if str(term.get("category", "")).lower() != "personagem":
                continue
            key = str(term.get("key", "")).strip()
            pt = str(term.get("pt", "")).strip()
            if not key:
                continue
            entry = registry.setdefault(
                key,
                {
                    "key": key,
                    "pt": pt or key,
                    "aliases": set(),
                    "volumes": set(),
                    "gender": None,
                },
            )
            entry["volumes"].add(vol_key)
            aliases = term.get("aliases") or term.get("alias") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            for al in aliases:
                entry["aliases"].add(str(al).strip())
            # inferência simples de gênero a partir de notes
            notes = str(term.get("notes", "")).lower()
            if any(tok in notes for tok in ["heroína", "ela", "sacerdotisa", "princesa"]):
                entry["gender"] = entry["gender"] or "F"
            if any(tok in notes for tok in ["herói", "ele", "guerreiro", "príncipe"]):
                entry["gender"] = entry["gender"] or "M"
    # normalizar sets
    for v in registry.values():
        v["aliases"] = sorted(a for a in v["aliases"] if a)
        v["volumes"] = sorted(v["volumes"])
    return registry


def check_term_consistency(
    volumes: Dict[str, str],
    glossarios: Dict[str, Dict],
    master_glossario: Dict | None = None,
) -> List[Dict]:
    """Verifica termo consistência."""
    issues: List[Dict] = []
    key_map: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    def add_terms(source_key: str, data: Dict):
        """Acrescenta termos normalizados ao conjunto de análise."""
        terms = data.get("terms") if isinstance(data, dict) else None
        if not isinstance(terms, list):
            return
        for term in terms:
            if not isinstance(term, dict):
                continue
            key = str(term.get("key", "")).strip()
            pt = str(term.get("pt", "")).strip()
            if not key or not pt:
                continue
            key_map[key][source_key].add(pt)

    for vol_key, data in glossarios.items():
        add_terms(vol_key, data)
    if master_glossario:
        add_terms("MASTER", master_glossario)

    for key, by_vol in key_map.items():
        all_pts = set()
        for pts in by_vol.values():
            all_pts |= pts
        if len(all_pts) > 1:
            issues.append(
                {
                    "type": "term_inconsistency",
                    "severity": "warning",
                    "term_key": key,
                    "volumes": sorted(by_vol.keys()),
                    "pt_variants": sorted(all_pts),
                    "suggestion": "Padronizar e, se necessário, mover variantes para aliases.",
                }
            )
    return issues


def _count_pronouns(text: str) -> Dict[str, int]:
    """Conta pronomes usados como indícios de gênero no contexto."""
    counts = defaultdict(int)
    for pron in [
        "ele",
        "ela",
        "dele",
        "dela",
        "seu",
        "sua",
        "o guerreiro",
        "a guerreira",
    ]:
        counts[pron] = len(re.findall(rf"\b{pron}\b", text, flags=re.IGNORECASE))
    return counts


def check_gender_consistency(
    volumes: Dict[str, str], character_registry: Dict[str, Dict[str, Any]]
) -> List[Dict]:
    """Verifica gênero consistência."""
    issues: List[Dict] = []
    for name, info in character_registry.items():
        aliases = [name] + list(info.get("aliases", []))
        per_volume: Dict[str, Dict[str, int]] = {}
        for vol_key, text in volumes.items():
            snippet = text
            if not any(
                re.search(rf"\b{re.escape(a)}\b", text, flags=re.IGNORECASE) for a in aliases
            ):
                continue
            per_volume[vol_key] = _count_pronouns(snippet)
        if not per_volume:
            continue
        # Usa o primeiro volume como referência para os seguintes.
        first_vol = sorted(per_volume.keys())[0]
        base = per_volume[first_vol]
        fem = base["ela"] + base["dela"] + base["sua"] + base["a guerreira"]
        masc = base["ele"] + base["dele"] + base["seu"] + base["o guerreiro"]
        expected = "F" if fem >= masc else "M" if masc > fem else None
        for vol_key, cnt in per_volume.items():
            fem_v = cnt["ela"] + cnt["dela"] + cnt["sua"] + cnt["a guerreira"]
            masc_v = cnt["ele"] + cnt["dele"] + cnt["seu"] + cnt["o guerreiro"]
            if expected == "F" and masc_v > fem_v + 2:
                issues.append(
                    {
                        "type": "gender_inconsistency",
                        "severity": "warning",
                        "character": name,
                        "expected": "F",
                        "volume": vol_key,
                        "evidence": {
                            "masculine_pronouns": masc_v,
                            "feminine_pronouns": fem_v,
                        },
                        "suggestion": f"Revisar trechos em {vol_key} onde {name} recebe pronomes masculinos.",
                    }
                )
            if expected == "M" and fem_v > masc_v + 2:
                issues.append(
                    {
                        "type": "gender_inconsistency",
                        "severity": "warning",
                        "character": name,
                        "expected": "M",
                        "volume": vol_key,
                        "evidence": {
                            "masculine_pronouns": masc_v,
                            "feminine_pronouns": fem_v,
                        },
                        "suggestion": f"Revisar trechos em {vol_key} onde {name} recebe pronomes femininos.",
                    }
                )
    return issues


def check_voice_consistency(
    volumes: Dict[str, str], character_registry: Dict[str, Dict[str, Any]]
) -> List[Dict]:
    """Verifica voz consistência."""
    issues: List[Dict] = []
    informal_tokens = {"cara", "mano", "hein", "uh", "ah", "né"}
    formal_tokens = {"vós", "senhor", "senhora", "venerável", "humilde"}

    for name, info in character_registry.items():
        aliases = [name] + list(info.get("aliases", []))
        per_volume_style = {}
        for vol_key, text in volumes.items():
            if not any(
                re.search(rf"\b{re.escape(a)}\b", text, flags=re.IGNORECASE) for a in aliases
            ):
                continue
            inf = sum(text.lower().count(tok) for tok in informal_tokens)
            form = sum(text.lower().count(tok) for tok in formal_tokens)
            per_volume_style[vol_key] = {"informal": inf, "formal": form}
        if len(per_volume_style) < 2:
            continue
        base_vol = sorted(per_volume_style.keys())[0]
        base = per_volume_style[base_vol]
        for vol_key, style in per_volume_style.items():
            if vol_key == base_vol:
                continue
            if base["formal"] > base["informal"] * 2 and style["informal"] > style["formal"] * 2:
                issues.append(
                    {
                        "type": "voice_inconsistency",
                        "severity": "info",
                        "character": name,
                        "volumes": [base_vol, vol_key],
                        "description": f"{name} é formal em {base_vol} e muito mais coloquial em {vol_key}.",
                    }
                )
    return issues


def check_lore_timeline_consistency(volumes: Dict[str, str]) -> List[Dict]:
    """Verifica inconsistências de cronologia e informações de mundo."""
    issues: List[Dict] = []
    patterns = {
        "perdeu_braco": re.compile(r"perdeu o braço direito", flags=re.IGNORECASE),
        "usou_braco": re.compile(r"ergueu o braço direito", flags=re.IGNORECASE),
        "morreu": re.compile(r"\bmorreu\b", flags=re.IGNORECASE),
        "vivo": re.compile(r"\bapareceu vivo\b", flags=re.IGNORECASE),
    }
    hits = defaultdict(list)
    for vol_key, text in volumes.items():
        for tag, pat in patterns.items():
            for m in pat.finditer(text):
                span = text[max(0, m.start() - 40) : m.end() + 40]
                hits[tag].append((vol_key, span.strip()))
    # checagem simples de pares incompatíveis
    if hits["perdeu_braco"] and hits["usou_braco"]:
        issues.append(
            {
                "type": "timeline_inconsistency",
                "severity": "info",
                "entity": "braço direito",
                "description": "Menção a perder braço direito e depois usá-lo.",
                "evidence": {
                    "perdeu": hits["perdeu_braco"],
                    "usou": hits["usou_braco"],
                },
            }
        )
    if hits["morreu"] and hits["vivo"]:
        issues.append(
            {
                "type": "timeline_inconsistency",
                "severity": "info",
                "entity": "personagem",
                "description": "Menção de morte e posterior aparição vivo.",
                "evidence": {"morreu": hits["morreu"], "vivo": hits["vivo"]},
            }
        )
    return issues


def run_intervolume_checks(
    volumes_dir: str,
    glossario_dir: str,
    master_glossario_path: str | None,
    checks: Dict[str, bool],
    output: str,
) -> Dict:
    """Executa o conjunto de verificações entre volumes."""
    volumes = load_volume_texts(volumes_dir)
    glossarios = load_glossaries(glossario_dir, master_glossario=master_glossario_path)
    master_gloss = glossarios.get("MASTER") if "MASTER" in glossarios else None
    registry = build_character_registry(glossarios)

    issues: List[Dict] = []
    if checks.get("terms", True):
        issues.extend(check_term_consistency(volumes, glossarios, master_gloss))
    if checks.get("gender", True):
        issues.extend(check_gender_consistency(volumes, registry))
    if checks.get("voice", True):
        issues.extend(check_voice_consistency(volumes, registry))
    if checks.get("lore", True):
        issues.extend(check_lore_timeline_consistency(volumes))

    report = {
        "volumes": sorted(volumes.keys()),
        "checks_enabled": checks,
        "issues": issues,
    }
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos da auditoria entre volumes."""
    p = argparse.ArgumentParser(description="Consistência inter-volume (QA)")
    p.add_argument("--volumes", required=True, help="Diretório com arquivos .md refinados")
    p.add_argument(
        "--glossario-dir",
        required=True,
        help="Diretório com glossarios por volume (glossario_volXX.json)",
    )
    p.add_argument("--master-glossario", help="Arquivo MASTER_GLOSSARIO.json (opcional)")
    p.add_argument(
        "--output",
        default="saida/consistencia_intervolume.json",
        help="Arquivo de saída do relatório",
    )
    p.add_argument("--no-check-terms", action="store_true")
    p.add_argument("--no-check-gender", action="store_true")
    p.add_argument("--no-check-voice", action="store_true")
    p.add_argument("--no-check-lore", action="store_true")
    return p


def main(argv: List[str] | None = None) -> None:
    """Executa a interface de linha de comando de intervolume."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    checks = {
        "terms": not args.no_check_terms,
        "gender": not args.no_check_gender,
        "voice": not args.no_check_voice,
        "lore": not args.no_check_lore,
    }
    run_intervolume_checks(
        volumes_dir=args.volumes,
        glossario_dir=args.glossario_dir,
        master_glossario_path=args.master_glossario,
        checks=checks,
        output=args.output,
    )
    print(f"Relatório gerado em {args.output}")


if __name__ == "__main__":
    main()
