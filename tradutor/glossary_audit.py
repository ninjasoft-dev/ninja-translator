from __future__ import annotations

import re
from typing import Any

PORTUGUESE_ALIAS_WORDS = {
    "anciaos",
    "ancioes",
    "arma",
    "barao",
    "cacador",
    "cacadores",
    "calice",
    "cavaleiros",
    "cidade",
    "cla",
    "de",
    "dente",
    "dragao",
    "exterminador",
    "exterminadores",
    "filhos",
    "fortaleza",
    "grande",
    "matador",
    "matadores",
    "monstros",
    "olho",
    "quatro",
    "rei",
    "ruinas",
    "sabio",
    "sagrados",
    "santos",
    "senhor",
    "senhora",
    "tigre",
    "tigres",
}


def normalize_text(value: str) -> str:
    """Normaliza texto."""
    return re.sub(r"\s+", " ", value.strip().casefold())


def _strip_accents_for_markers(value: str) -> str:
    """Remove acentos antes de comparar marcadores linguísticos."""
    table = str.maketrans(
        "áàâãäéèêëíìîïóòôõöúùûüçÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ",
        "aaaaaeeeeiiiiooooouuuucAAAAAEEEEIIIIOOOOOUUUUC",
    )
    return value.translate(table)


def as_list(value: Any) -> list[str]:
    """Converte lista para o formato esperado."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def terms_from_data(data: Any) -> list[dict[str, Any]]:
    """Extrai a lista de termos de uma estrutura de glossário."""
    if isinstance(data, dict) and isinstance(data.get("terms"), list):
        return [term for term in data["terms"] if isinstance(term, dict)]
    if isinstance(data, list):
        return [term for term in data if isinstance(term, dict)]
    raise ValueError("Glossary must be a JSON object with 'terms' or a list of terms.")


def source_aliases_for_entry(term: dict[str, Any]) -> list[str]:
    """Reúne o termo original e seus aliases de origem."""
    aliases = as_list(term.get("source_aliases"))
    if aliases:
        return aliases
    return as_list(term.get("aliases"))


def is_probably_portuguese_alias(value: str) -> bool:
    """Estima se um alias de origem está escrito em português."""
    clean = value.strip()
    if not clean:
        return False
    if re.search(r"[áàâãéèêíìîóòôõúùûçÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ]", clean):
        return True
    folded = _strip_accents_for_markers(clean).casefold()
    words = set(re.findall(r"[a-z]+", folded))
    return bool(words & PORTUGUESE_ALIAS_WORDS)


def audit_glossary_data(data: Any) -> dict[str, Any]:
    """Executa as verificações de integridade do glossário."""
    terms = terms_from_data(data)
    duplicate_keys = _group_duplicate_terms(terms, "key")
    duplicate_pt = _group_duplicate_terms(terms, "pt")
    duplicate_aliases_within_term = _duplicate_aliases_within_terms(terms)
    source_alias_mismatches = _source_alias_mismatches(terms)
    portuguese_source_aliases = _portuguese_source_aliases(terms)
    ambiguous_source_aliases = _ambiguous_source_aliases(terms)
    redundant_source_aliases = _redundant_source_aliases(terms)

    return {
        "summary": {
            "terms": len(terms),
            "duplicate_keys": len(duplicate_keys),
            "duplicate_pt_groups": len(duplicate_pt),
            "ambiguous_source_aliases": len(ambiguous_source_aliases),
            "portuguese_source_aliases": len(portuguese_source_aliases),
            "source_alias_mismatches": len(source_alias_mismatches),
            "duplicate_aliases_within_term": len(duplicate_aliases_within_term),
            "redundant_source_aliases": len(redundant_source_aliases),
        },
        "duplicate_keys": duplicate_keys,
        "duplicate_pt_groups": duplicate_pt,
        "ambiguous_source_aliases": ambiguous_source_aliases,
        "portuguese_source_aliases": portuguese_source_aliases,
        "source_alias_mismatches": source_alias_mismatches,
        "duplicate_aliases_within_term": duplicate_aliases_within_term,
        "redundant_source_aliases": redundant_source_aliases,
    }


def format_audit_report(report: dict[str, Any], *, limit: int = 20) -> str:
    """Formata os achados da auditoria de glossário como texto."""
    summary = report.get("summary", {})
    lines = [
        "Glossary audit",
        f"- terms: {summary.get('terms', 0)}",
        f"- duplicate keys: {summary.get('duplicate_keys', 0)}",
        f"- duplicate PT groups: {summary.get('duplicate_pt_groups', 0)}",
        f"- ambiguous source aliases: {summary.get('ambiguous_source_aliases', 0)}",
        f"- Portuguese aliases in source aliases: {summary.get('portuguese_source_aliases', 0)}",
        f"- source_aliases/aliases mismatches: {summary.get('source_alias_mismatches', 0)}",
        f"- duplicate aliases within term: {summary.get('duplicate_aliases_within_term', 0)}",
        f"- redundant source aliases: {summary.get('redundant_source_aliases', 0)}",
    ]
    _append_issue_preview(
        lines, "Ambiguous aliases", report.get("ambiguous_source_aliases", []), limit
    )
    _append_issue_preview(
        lines,
        "Portuguese source aliases",
        report.get("portuguese_source_aliases", []),
        limit,
    )
    _append_issue_preview(lines, "Duplicate keys", report.get("duplicate_keys", []), limit)
    _append_issue_preview(
        lines, "Duplicate PT groups", report.get("duplicate_pt_groups", []), limit
    )
    return "\n".join(lines)


def _append_issue_preview(
    lines: list[str], title: str, issues: list[dict[str, Any]], limit: int
) -> None:
    """Adiciona uma amostra limitada de ocorrências ao relatório."""
    if not issues:
        return
    lines.append("")
    lines.append(f"{title}:")
    for issue in issues[:limit]:
        value = (
            issue.get("value") or issue.get("key") or issue.get("pt") or issue.get("alias") or ""
        )
        refs = issue.get("refs") or issue.get("terms") or []
        rendered_refs = ", ".join(
            str(ref.get("key", ref)) if isinstance(ref, dict) else str(ref) for ref in refs[:4]
        )
        suffix = f" -> {rendered_refs}" if rendered_refs else ""
        lines.append(f"- {value}{suffix}")
    if len(issues) > limit:
        lines.append(f"- ... {len(issues) - limit} more")


def _group_duplicate_terms(terms: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    """Agrupa entradas que declaram o mesmo termo de origem."""
    groups: dict[str, list[dict[str, Any]]] = {}
    display: dict[str, str] = {}
    for idx, term in enumerate(terms):
        value = str(term.get(field, "")).strip()
        if not value:
            continue
        norm = normalize_text(value)
        groups.setdefault(norm, []).append(
            {
                "index": idx,
                "key": str(term.get("key", "")).strip(),
                "pt": str(term.get("pt", "")).strip(),
            }
        )
        display.setdefault(norm, value)
    issues = []
    for norm, refs in sorted(groups.items(), key=lambda item: item[0]):
        if len(refs) > 1:
            issues.append({"value": display.get(norm, norm), "terms": refs})
    return issues


def _duplicate_aliases_within_terms(
    terms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Encontra aliases repetidos dentro da mesma entrada."""
    issues = []
    for idx, term in enumerate(terms):
        aliases = source_aliases_for_entry(term)
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for alias in aliases:
            norm = normalize_text(alias)
            if norm in seen:
                duplicates.append(alias)
            else:
                seen[norm] = alias
        if duplicates:
            issues.append(
                {
                    "index": idx,
                    "key": str(term.get("key", "")).strip(),
                    "aliases": duplicates,
                }
            )
    return issues


def _source_alias_mismatches(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detecta aliases de origem incompatíveis com a entrada."""
    issues = []
    for idx, term in enumerate(terms):
        if "source_aliases" not in term or "aliases" not in term:
            continue
        source_aliases = [normalize_text(alias) for alias in as_list(term.get("source_aliases"))]
        legacy_aliases = [normalize_text(alias) for alias in as_list(term.get("aliases"))]
        if source_aliases != legacy_aliases:
            issues.append(
                {
                    "index": idx,
                    "key": str(term.get("key", "")).strip(),
                    "source_aliases": as_list(term.get("source_aliases")),
                    "aliases": as_list(term.get("aliases")),
                }
            )
    return issues


def _portuguese_source_aliases(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Localiza aliases em português usados como formas de origem."""
    issues = []
    for idx, term in enumerate(terms):
        for alias in source_aliases_for_entry(term):
            if is_probably_portuguese_alias(alias):
                issues.append(
                    {
                        "index": idx,
                        "key": str(term.get("key", "")).strip(),
                        "pt": str(term.get("pt", "")).strip(),
                        "alias": alias,
                    }
                )
    return issues


def _ambiguous_source_aliases(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Localiza aliases de origem associados a mais de um termo."""
    refs_by_alias: dict[str, list[dict[str, Any]]] = {}
    display: dict[str, str] = {}
    for idx, term in enumerate(terms):
        key = str(term.get("key", "")).strip()
        if not key:
            continue
        values = [
            ("key", key),
            *[("source_alias", alias) for alias in source_aliases_for_entry(term)],
        ]
        seen_for_term: set[str] = set()
        for field, value in values:
            norm = normalize_text(value)
            if len(norm) < 3 or norm in seen_for_term:
                continue
            seen_for_term.add(norm)
            refs_by_alias.setdefault(norm, []).append(
                {
                    "index": idx,
                    "key": key,
                    "pt": str(term.get("pt", "")).strip(),
                    "field": field,
                }
            )
            display.setdefault(norm, value)
    issues = []
    for norm, refs in sorted(refs_by_alias.items(), key=lambda item: item[0]):
        keys = {ref["key"] for ref in refs}
        if len(keys) > 1:
            issues.append({"value": display.get(norm, norm), "refs": refs})
    return issues


def _redundant_source_aliases(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Localiza aliases que apenas repetem o termo principal."""
    issues = []
    for idx, term in enumerate(terms):
        key = str(term.get("key", "")).strip()
        pt = str(term.get("pt", "")).strip()
        aliases = source_aliases_for_entry(term)
        redundant = [
            alias
            for alias in aliases
            if normalize_text(alias) in {normalize_text(key), normalize_text(pt)}
        ]
        if redundant:
            issues.append(
                {
                    "index": idx,
                    "key": key,
                    "pt": pt,
                    "aliases": redundant,
                }
            )
    return issues
