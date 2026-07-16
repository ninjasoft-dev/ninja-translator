from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .languages import (
    compile_term_pattern,
    normalize_source_language,
    source_language_name,
)

GlossaryEntry = Dict[str, Any]
GlossaryIndex = Dict[str, GlossaryEntry]
GlossaryPtIndex = Dict[str, GlossaryEntry]

GLOSSARIO_SUGERIDO_INICIO = "===GLOSSARIO_SUGERIDO_INICIO==="
GLOSSARIO_SUGERIDO_FIM = "===GLOSSARIO_SUGERIDO_FIM==="
DEFAULT_GLOSSARY_PROMPT_LIMIT = 100
DEFAULT_MANUAL_GLOSSARY_CANDIDATES = (
    Path("glossario/glossario_manual.json"),
    Path("glossario/glossario_geral.json"),
)


def normalize_key(key: str) -> str:
    """Normaliza a chave do glossário para comparação/índice."""
    return key.strip().lower()


def normalize_value(value: str) -> str:
    """Normaliza textos (key/pt) para comparação insensível a caixa/espaços."""
    return value.strip().lower()


def resolve_manual_glossary_path(path: str | Path | None = None) -> Path:
    """Resolve o glossário manual explícito ou o caminho configurado para a obra."""
    if path:
        return Path(path)
    for candidate in DEFAULT_MANUAL_GLOSSARY_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_MANUAL_GLOSSARY_CANDIDATES[0]


def _is_valid_dynamic_term(candidate: str, logger: logging.Logger) -> bool:
    """Aplica filtros de sanidade para evitar termos dinâmicos descritivos demais."""
    cand = candidate.strip()
    if not cand:
        return False
    if len(cand) > 80:
        logger.info("Ignorando termo dinâmico muito longo: %r", cand)
        return False
    if len(cand.split()) > 6:
        logger.info("Ignorando termo dinâmico com muitas palavras: %r", cand)
        return False
    lowered = f" {cand.lower()} "
    if " que " in lowered or " uma " in lowered or " um " in lowered:
        logger.info("Ignorando termo dinâmico com padrão de frase: %r", cand)
        return False
    return True


def _build_index(terms: List[GlossaryEntry]) -> GlossaryIndex:
    """Monta o índice de termos e aliases do glossário."""
    return {
        normalize_key(str(term.get("key", ""))): term
        for term in terms
        if str(term.get("key", "")).strip()
    }


def _build_manual_pt_index(terms: List[GlossaryEntry]) -> GlossaryPtIndex:
    """Índice auxiliar por campo pt (normalizado) para evitar duplicar conceitos no dinâmico."""
    idx: GlossaryPtIndex = {}
    for term in terms:
        pt_raw = str(term.get("pt", "")).strip()
        if not pt_raw:
            continue
        pt_norm = normalize_value(pt_raw)
        if pt_norm and pt_norm not in idx:
            idx[pt_norm] = term
    return idx


def _string_list(value: Any) -> list[str]:
    """Normaliza uma coleção opcional como lista de strings não vazias."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _as_bool(value: Any) -> bool:
    """Interpreta flags booleanas vindas de JSON sem tratar 'false' como verdadeiro."""
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "sim"}
    return bool(value)


def _merge_indexes(manual_index: GlossaryIndex, dynamic_index: GlossaryIndex) -> GlossaryIndex:
    """Combina índices de glossário preservando a ordem das entradas."""
    merged = dict(manual_index)
    for key, entry in dynamic_index.items():
        if key not in merged:
            merged[key] = entry
    return merged


def _load_terms(path: Path, source: str, logger: logging.Logger) -> List[GlossaryEntry]:
    """Carrega termos."""
    if not path.exists():
        logger.info("Glossário %s não encontrado em %s; prosseguindo com vazio.", source, path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - leitura/parse
        logger.warning("Falha ao ler glossário %s em %s: %s", source, path, exc)
        return []
    raw_terms = data.get("terms") if isinstance(data, dict) else None
    if not isinstance(raw_terms, list):
        logger.warning(
            "Formato inesperado no glossário %s em %s; usando lista vazia.",
            source,
            path,
        )
        return []

    terms: List[GlossaryEntry] = []
    for entry in raw_terms:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key", "")).strip()
        pt = str(entry.get("pt", "")).strip()
        if not key or not pt:
            continue
        source_aliases = _string_list(entry.get("source_aliases") or entry.get("aliases") or [])
        bad_aliases = _string_list(entry.get("bad_aliases") or entry.get("forbidden_aliases") or [])
        allowed_target_aliases = _string_list(
            entry.get("allowed_target_aliases") or entry.get("target_aliases") or []
        )
        raw_target_replacements = entry.get("target_replacements") or {}
        target_replacements = (
            {
                str(alias).strip(): str(replacement).strip()
                for alias, replacement in raw_target_replacements.items()
                if str(alias).strip() and str(replacement).strip()
            }
            if isinstance(raw_target_replacements, dict)
            else {}
        )
        normalized: GlossaryEntry = {
            "key": key,
            "pt": pt,
            "category": entry.get("category"),
            "notes": entry.get("notes"),
            "source": "manual" if source == "manual" else "dynamic",
            "locked": bool(entry.get("locked", source == "manual")),
            # `aliases` permanece como nome compatível para correspondências no texto de origem.
            "aliases": source_aliases,
            "source_aliases": source_aliases,
            "aliases_norm": [normalize_key(a) for a in source_aliases],
            "source_aliases_norm": [normalize_key(a) for a in source_aliases],
            "bad_aliases": bad_aliases,
            "allowed_target_aliases": allowed_target_aliases,
            "target_replacements": target_replacements,
            # Algumas habilidades têm nomes que também são palavras comuns em
            # idioma de origem. Esta flag mantém a busca no original restrita ao uso
            # grafado como nome de habilidade, por exemplo `Freeze` e não
            # `body freeze`.
            "source_case_sensitive": _as_bool(entry.get("source_case_sensitive", False)),
        }
        for field in ("enforce", "gender", "type", "term_type"):
            if field in entry:
                normalized[field] = entry[field]
        terms.append(normalized)
    logger.info("Glossário %s carregado: %d termos.", source, len(terms))
    return terms


def _load_terms_from_dir(dir_path: Path, logger: logging.Logger) -> List[GlossaryEntry]:
    """
    Carrega todos os arquivos *.json de um diretório como glossário manual agregado.
    Mantém apenas a primeira ocorrência de cada key para evitar sobrescritas acidentais.
    """
    if not dir_path.exists():
        logger.info(
            "Diretório de glossário não encontrado em %s; prosseguindo sem auto-glossary.",
            dir_path,
        )
        return []
    if not dir_path.is_dir():
        logger.warning("Caminho de auto-glossary não é um diretório: %s", dir_path)
        return []

    aggregated: List[GlossaryEntry] = []
    seen: set[str] = set()
    for file in sorted(dir_path.glob("*.json")):
        terms = _load_terms(file, "manual", logger)
        for term in terms:
            key_norm = normalize_key(str(term.get("key", "")))
            if not key_norm or key_norm in seen:
                continue
            seen.add(key_norm)
            aggregated.append(term)
    logger.info("Auto-glossary: %d termos carregados de %s", len(aggregated), dir_path)
    return aggregated


@dataclass
class GlossaryState:
    """Mantém os índices e metadados do glossário carregado."""

    manual_terms: List[GlossaryEntry]
    dynamic_terms: List[GlossaryEntry]
    manual_index: GlossaryIndex
    dynamic_index: GlossaryIndex
    combined_index: GlossaryIndex
    dynamic_path: Path | None
    manual_pt_index: GlossaryPtIndex

    def refresh_combined(self) -> None:
        """Recalcula índices combinados a partir das listas atuais."""
        self.manual_index = _build_index(self.manual_terms)
        self.dynamic_index = _build_index(self.dynamic_terms)
        self.manual_pt_index = _build_manual_pt_index(self.manual_terms)
        self.combined_index = _merge_indexes(self.manual_index, self.dynamic_index)


def build_glossary_state(
    manual_path: Path | None,
    dynamic_path: Path | None,
    logger: logging.Logger,
    manual_dir: Path | None = None,
) -> GlossaryState | None:
    """Carrega glossários manual/dinâmico (e auto-glossary opcional) e retorna estado consolidado."""
    if manual_path is None and dynamic_path is None and manual_dir is None:
        return None

    manual_terms: List[GlossaryEntry] = []
    if manual_dir:
        manual_terms.extend(_load_terms_from_dir(manual_dir, logger))
    if manual_path:
        manual_terms.extend(_load_terms(manual_path, "manual", logger))
    dynamic_terms = _load_terms(dynamic_path, "dynamic", logger) if dynamic_path else []

    state = GlossaryState(
        manual_terms=manual_terms,
        dynamic_terms=dynamic_terms,
        manual_index=_build_index(manual_terms),
        dynamic_index=_build_index(dynamic_terms),
        combined_index={},
        dynamic_path=dynamic_path,
        manual_pt_index=_build_manual_pt_index(manual_terms),
    )
    state.combined_index = _merge_indexes(state.manual_index, state.dynamic_index)
    return state


def format_glossary_for_prompt(
    combined_index: GlossaryIndex, limit: int = DEFAULT_GLOSSARY_PROMPT_LIMIT
) -> str:
    """Gera bloco de texto para o prompt a partir do glossário combinado."""
    if not combined_index:
        return ""
    entries = sorted(combined_index.values(), key=lambda e: normalize_key(str(e.get("key", ""))))[
        :limit
    ]
    lines = ["GLOSSÁRIO CANÔNICO (use SEMPRE estas traduções):"]
    for entry in entries:
        key = str(entry.get("key", "")).strip()
        pt = str(entry.get("pt", "")).strip()
        if not key or not pt:
            continue
        category = entry.get("category")
        notes = entry.get("notes")
        line = f"- {key} -> {pt}"
        if category:
            line += f" ({category})"
        if notes:
            line += f" | {notes}"
        lines.append(line)
    return "\n".join(lines)


def format_manual_pairs_for_translation(
    manual_terms: list[GlossaryEntry], limit: int | None = 30
) -> str:
    """Formata pares EN->PT do glossário manual para uso no prompt de tradução."""
    if not manual_terms:
        return ""
    entries = sorted(manual_terms, key=lambda e: normalize_key(str(e.get("key", ""))))
    if limit is not None:
        entries = entries[:limit]
    lines = ["TERMOS CANONICOS (NAO TRADUZIR DIFERENTE DESTO):"]
    for entry in entries:
        en = str(entry.get("key", "")).strip()
        pt = str(entry.get("pt", "")).strip()
        if not en or not pt:
            continue
        line = f'Ingles: "{en}" -> Portugues: "{pt}"'
        hints: list[str] = []
        category = entry.get("category")
        gender = entry.get("gender")
        notes = entry.get("notes")
        if category:
            hints.append(f"categoria: {category}")
        if gender:
            hints.append(f"genero: {gender}")
        if entry.get("enforce"):
            hints.append("uso obrigatorio")
        bad_aliases = _string_list(entry.get("bad_aliases"))
        if bad_aliases:
            hints.append("nao usar: " + ", ".join(bad_aliases[:5]))
        if notes:
            hints.append(str(notes))
        if hints:
            line += " | " + " | ".join(hints)
        lines.append(line)
    return "\n".join(lines)


def split_refined_and_suggestions(text: str) -> Tuple[str, str | None]:
    """
    Separa texto refinado e bloco de glossário sugerido pelos delimitadores.
    Retorna (texto_refinado, bloco_ou_none).
    """
    start = text.find(GLOSSARIO_SUGERIDO_INICIO)
    end = text.find(GLOSSARIO_SUGERIDO_FIM)
    if start == -1 or end == -1 or end < start:
        return text.strip(), None
    refined = text[:start].strip()
    block = text[start + len(GLOSSARIO_SUGERIDO_INICIO) : end].strip()
    return refined, block


def select_terms_for_chunk(
    manual_terms: list[GlossaryEntry],
    chunk_text: str,
    match_limit: int = 80,
    fallback_limit: int = 30,
) -> tuple[list[GlossaryEntry], int]:
    """
    Seleciona termos cujo `key` ou `alias` aparece no chunk.
    Por padrão a busca é case-insensitive; `source_case_sensitive=true` no
    termo mantém a distinção entre um nome canônico e uma palavra comum.
    Retorna (termos_para_prompt, matched_count).
    """
    if not manual_terms:
        return [], 0
    chunk_compact = re.sub(r"\s+", " ", chunk_text)
    chunk_norm = chunk_compact.lower()

    def _matches_term(term_value: str, *, case_sensitive: bool) -> bool:
        """Verifica se o termo aparece no texto segundo suas regras de caixa."""
        term_value = term_value.strip()
        if not term_value:
            return False
        haystack = chunk_compact if case_sensitive else chunk_norm
        return bool(
            compile_term_pattern(term_value, case_sensitive=case_sensitive).search(haystack)
        )

    matches: list[GlossaryEntry] = []
    seen: set[str] = set()
    for term in manual_terms:
        key = str(term.get("key", "")).strip()
        key_norm = normalize_key(key)
        if not key_norm or key_norm in seen:
            continue
        aliases = _string_list(term.get("source_aliases") or term.get("aliases") or [])
        if not aliases:
            aliases = _string_list(
                term.get("source_aliases_norm") or term.get("aliases_norm") or []
            )
        case_sensitive = _as_bool(term.get("source_case_sensitive", False))
        matched = _matches_term(key, case_sensitive=case_sensitive) or any(
            _matches_term(alias, case_sensitive=case_sensitive) for alias in aliases
        )
        if matched:
            matches.append(term)
            seen.add(key_norm)
    matches = sorted(matches, key=lambda e: normalize_key(str(e.get("key", ""))))[:match_limit]
    if matches:
        return matches, len(matches)
    fallback = sorted(manual_terms, key=lambda e: normalize_key(str(e.get("key", ""))))[
        :fallback_limit
    ]
    return fallback, 0


def select_terms_for_target_text(
    terms: list[GlossaryEntry],
    target_text: str,
    match_limit: int = 80,
) -> tuple[list[GlossaryEntry], int]:
    """Seleciona termos relevantes para uma etapa que recebe PT-BR.

    O refinador não deve receber o glossário inteiro: isso ocupa contexto e
    incentiva mudanças em termos que não estão no trecho. A busca considera a
    forma canônica em português e aliases de saída, inclusive formas proibidas
    que precisam ser corrigidas. Não há fallback deliberado quando nada casa.
    """
    if not terms or not target_text:
        return [], 0

    compact = re.sub(r"\s+", " ", target_text.lower())

    def _contains(value: str) -> bool:
        """Verifica se o texto contém o termo respeitando limites de palavra."""
        value_norm = normalize_key(value)
        if not value_norm:
            return False
        return bool(compile_term_pattern(value_norm).search(compact))

    selected: list[GlossaryEntry] = []
    seen: set[str] = set()
    for term in terms:
        key_norm = normalize_key(str(term.get("key", "")))
        if not key_norm or key_norm in seen:
            continue
        variants = [str(term.get("pt", "")).strip()]
        variants.extend(_string_list(term.get("allowed_target_aliases")))
        variants.extend(_string_list(term.get("bad_aliases") or term.get("forbidden_aliases")))
        replacements = term.get("target_replacements") or {}
        if isinstance(replacements, dict):
            variants.extend(str(alias).strip() for alias in replacements if str(alias).strip())
        if any(_contains(variant) for variant in variants):
            selected.append(term)
            seen.add(key_norm)

    selected = sorted(selected, key=lambda entry: normalize_key(str(entry.get("key", ""))))[
        :match_limit
    ]
    return selected, len(selected)


def parse_glossary_suggestions(block: str) -> List[GlossaryEntry]:
    """
    Converte bloco textual de sugestões em lista de entradas.
    Formato esperado:
        key: termo
        pt: tradução
        category: opcional
        notes: opcional
        ---
    """
    if not block:
        return []
    suggestions: List[GlossaryEntry] = []
    current: GlossaryEntry = {}

    def flush_current() -> None:
        """Conclui a entrada de glossário que está sendo interpretada."""
        if current.get("key") and current.get("pt"):
            suggestions.append(
                {
                    "key": str(current["key"]).strip(),
                    "pt": str(current["pt"]).strip(),
                    "category": current.get("category"),
                    "notes": current.get("notes"),
                }
            )

    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "---":
            flush_current()
            current = {}
            continue
        if ":" not in line:
            continue
        field, value = line.split(":", 1)
        field = field.strip().lower()
        value = value.strip()
        if field in {"key", "pt", "category", "notes"}:
            current[field] = value
    flush_current()
    return suggestions


def build_glossary_curation_prompt(glossary_json: str, source_language: str = "auto") -> str:
    """Monta o prompt de curadoria de um glossário literário."""
    language = normalize_source_language(source_language)
    source_name = "idioma de origem" if language == "auto" else source_language_name(language)
    return f"""
Você é um CURADOR PROFISSIONAL DE GLOSSÁRIOS para tradução de ficção literária.
Receba o JSON abaixo e normalize, padronize e enriqueça o glossário para uso em tradução automática.

Objetivos:
- Manter consistência entre volumes; remover inconsistências e redundâncias.
- Padronizar nomes próprios, skills, itens, locais, conceitos.
- Garantir fluidez e naturalidade em PT-BR; adaptar termos culturalmente sensíveis.
- Proteger traduções corretas com "locked": true.
- Criar aliases adicionais e regras linguísticas essenciais.
- Garantir coerência de gênero gramatical, estilo e voz.

Ações obrigatórias por entrada:
- Garantir consistência entre "key", "pt" e "aliases".
- Adicionar "gender": masculino/feminino/neutro quando aplicável.
- Adicionar "type": personagem / criatura / habilidade / conceito / título / local / item / evento / mecânica.
- Expandir "aliases" com variantes úteis em {source_name} e PT-BR.
- Eliminar construções calcadas na gramática de {source_name} nas traduções.
- Manter "locked": true.
- Reorganizar notes para ficarem claras, objetivas e úteis ao tradutor automático.
- Consolidar duplicatas em um único termo com aliases.

Adicionar pseudo-termos (regras gerais) ao final:
- source_grammar_rule
- stuttering_rule
- calque_blocker
- humor_adaptation_rule
- proper_noun_preservation
- ocr_noise_removal

Regras adicionais:
- Não alterar termos já locked=true.
- Preservar humor em trocadilhos e apelidos.
- Manter campos úteis existentes (category/source/key/pt) sincronizados com term_pt quando aplicável.

Formato da saída: JSON válido, mesma estrutura, apenas o JSON (nada fora).

Glossário para curadoria (JSON):
{glossary_json}
"""


def apply_suggestions_to_state(
    state: GlossaryState,
    suggestions: List[GlossaryEntry],
    logger: logging.Logger,
) -> bool:
    """
    Aplica sugestões no glossário dinâmico respeitando prioridade do manual e flags locked.
    Retorna True se houve mudança no glossário dinâmico.
    """
    if not suggestions:
        return False

    changed = False
    for entry in suggestions:
        key_raw = str(entry.get("key", "")).strip()
        pt = str(entry.get("pt", "")).strip()
        # Para o refinador, tratamos key/pt como o mesmo rótulo em PT-BR
        term_pt = pt or key_raw
        if not key_raw or not pt or not term_pt.strip():
            continue
        key_norm = normalize_key(term_pt)
        category = entry.get("category")
        notes = entry.get("notes")

        pt_norm = normalize_value(pt) if pt else ""
        if pt_norm and pt_norm in state.manual_pt_index:
            logger.debug(
                "Ignorando sugestão de glossário para '%s' (pt já definido no manual).",
                key_raw,
            )
            continue

        if key_norm in state.manual_index:
            logger.debug(
                "Ignorando sugestão de glossário para '%s' (definido no manual).",
                key_raw,
            )
            continue

        existing = state.dynamic_index.get(key_norm)
        if existing:
            if existing.get("locked"):
                logger.debug(
                    "Entrada dinâmica '%s' está bloqueada; não será alterada.",
                    existing.get("key"),
                )
                continue
            updated = False
            if term_pt and term_pt != existing.get("pt"):
                existing["pt"] = term_pt
                existing["key"] = term_pt
                updated = True
            if category and category != existing.get("category"):
                existing["category"] = category
                updated = True
            if notes and notes != existing.get("notes"):
                existing["notes"] = notes
                updated = True
            if updated:
                changed = True
                logger.info("Glossário dinâmico atualizado para '%s' -> %s", key_raw, pt)
            continue

        candidate = term_pt.strip()
        if not _is_valid_dynamic_term(candidate, logger):
            continue

        new_entry: GlossaryEntry = {
            "key": candidate,
            "pt": candidate,
            "category": category if category else None,
            "notes": notes if notes else None,
            "source": "dynamic",
            "locked": False,
        }
        state.dynamic_terms.append(new_entry)
        state.dynamic_index[key_norm] = new_entry
        changed = True
        logger.info("Nova entrada adicionada ao glossário dinâmico: %s -> %s", key_raw, pt)

    if changed:
        state.refresh_combined()
    return changed


def save_dynamic_glossary(state: GlossaryState, logger: logging.Logger) -> None:
    """Grava o glossário dinâmico no caminho configurado."""
    if state.dynamic_path is None:
        return
    state.dynamic_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_terms = sorted(state.dynamic_terms, key=lambda t: normalize_key(str(t.get("key", ""))))
    payload = {"terms": sorted_terms}
    try:
        state.dynamic_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "Glossário dinâmico salvo em %s (termos: %d).",
            state.dynamic_path,
            len(sorted_terms),
        )
    except Exception as exc:  # pragma: no cover - I/O edge case
        logger.warning("Falha ao salvar glossário dinâmico em %s: %s", state.dynamic_path, exc)
