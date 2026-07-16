from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .glossary_utils import build_glossary_state
from .postprocess import final_pt_postprocess
from .quality_checks import run_translation_quality_checks
from .quote_fix import fix_blank_lines_inside_quotes, fix_unbalanced_quotes
from .structure_normalizer import normalize_structure
from .translate import ensure_section_heading


@dataclass
class ReviewReport:
    """Reúne as alterações realizadas pela revisão determinística."""

    heading_fixes: int = 0
    glossary_replacements: dict[str, int] = field(default_factory=dict)
    text_replacements: dict[str, int] = field(default_factory=dict)
    all_caps_name_replacements: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converte o objeto para dict."""
        return {
            "heading_fixes": self.heading_fixes,
            "glossary_replacements": self.glossary_replacements,
            "text_replacements": self.text_replacements,
            "all_caps_name_replacements": self.all_caps_name_replacements,
        }


def _sub_word(
    text: str,
    pattern: str,
    repl: str,
    *,
    flags: int = re.IGNORECASE,
    preserve_case: bool = False,
) -> tuple[str, int]:
    """Substitui palavras completas sem alterar trechos internos de outros termos."""
    if not preserve_case:
        return re.subn(pattern, repl, text, flags=flags)

    def _replace(match: re.Match[str]) -> str:
        """Aplica a substituição atual e registra a alteração."""
        found = match.group(0)
        if found.isupper():
            return repl.upper()
        if found[:1].isupper():
            return repl[:1].upper() + repl[1:]
        return repl

    return re.subn(pattern, _replace, text, flags=flags)


def _record(counter: dict[str, int], key: str, count: int) -> None:
    """Registra uma alteração e incrementa sua métrica."""
    if count:
        counter[key] = counter.get(key, 0) + count


def _term_string_list(value: Any) -> list[str]:
    """Normaliza um campo de glossário como lista de strings."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _term_variants_for_article_fix(term: dict) -> list[str]:
    """Reúne variantes válidas do termo para corrigir seus artigos."""
    variants: list[str] = []
    for key_field in ("pt", "key"):
        value = str(term.get(key_field, "")).strip()
        if value:
            variants.append(value)
            variants.extend(
                token for token in value.split() if token[:1].isupper() and len(token) >= 3
            )
    variants.extend(_term_string_list(term.get("source_aliases")))
    variants.extend(_term_string_list(term.get("aliases")))
    variants.extend(_term_string_list(term.get("allowed_target_aliases")))
    out: list[str] = []
    seen: set[str] = set()
    for value in variants:
        clean = value.strip()
        marker = clean.casefold()
        if not clean or marker in seen:
            continue
        seen.add(marker)
        out.append(clean)
    return sorted(out, key=len, reverse=True)


def apply_editorial_replacements(text: str, report: ReviewReport | None = None) -> str:
    """Corrige resíduos recorrentes de tradução sem depender de uma obra."""
    current_report = report or ReviewReport()
    replacements = [
        (r"\bEu wish\b", "Quem me dera", False),
        (r"(?<=\w)—or\b", " — ou", False),
        (r"(?<=\w)—,", " —,", False),
        (r"\bbipede\b", "bípede", True),
        (r"\bsemi-deuses\b", "semideuses", True),
        (r"\bTh-the… y’re…", "El-eles…", False),
        (r"\bsus\s+AF\b", "suspeita pra caramba", False),
        (r"\bmeow\b", "miau", True),
        (r"\bskills\b", "habilidades", True),
        (r"\bskill\b", "habilidade", True),
    ]
    for pattern, replacement, preserve_case in replacements:
        text, count = _sub_word(
            text,
            pattern,
            replacement,
            preserve_case=preserve_case,
        )
        _record(
            current_report.text_replacements,
            f"{pattern}->{replacement}",
            count,
        )
    return text


def apply_gendered_article_fixes(
    text: str, glossary_terms: list[dict], report: ReviewReport | None = None
) -> str:
    """Corrige artigos e predicativos incompatíveis com gênero conhecido."""
    rpt = report or ReviewReport()
    for term in glossary_terms:
        gender = str(term.get("gender", "")).strip().casefold()
        category = (
            str(term.get("category") or term.get("type") or term.get("term_type") or "")
            .strip()
            .casefold()
        )
        if (not gender.startswith(("femin", "mascul"))) or (
            "person" not in category and "personagem" not in category and "criatura" not in category
        ):
            continue
        feminine = gender.startswith("femin")
        for variant in _term_variants_for_article_fix(term):
            escaped = re.escape(variant)
            if feminine:
                article_pairs = [
                    (rf"\bo\s+{escaped}\b", f"a {variant}"),
                    (rf"\bO\s+{escaped}\b", f"A {variant}"),
                    (rf"\bdo\s+{escaped}\b", f"da {variant}"),
                    (rf"\bDo\s+{escaped}\b", f"Da {variant}"),
                    (rf"\bno\s+{escaped}\b", f"na {variant}"),
                    (rf"\bNo\s+{escaped}\b", f"Na {variant}"),
                    (rf"\bao\s+{escaped}\b", f"à {variant}"),
                    (rf"\bAo\s+{escaped}\b", f"À {variant}"),
                ]
            else:
                article_pairs = [
                    (rf"\ba\s+{escaped}\b", f"o {variant}"),
                    (rf"\bA\s+{escaped}\b", f"O {variant}"),
                    (rf"\bda\s+{escaped}\b", f"do {variant}"),
                    (rf"\bDa\s+{escaped}\b", f"Do {variant}"),
                    (rf"\bna\s+{escaped}\b", f"no {variant}"),
                    (rf"\bNa\s+{escaped}\b", f"No {variant}"),
                    (rf"\bà\s+{escaped}\b", f"ao {variant}"),
                    (rf"\bÀ\s+{escaped}\b", f"Ao {variant}"),
                ]
            for pattern, repl in article_pairs:
                text, count = re.subn(pattern, repl, text)
                _record(rpt.text_replacements, f"{pattern}->{repl}", count)
            if feminine:
                text, count = re.subn(rf"\b({escaped}) como aliado\b", r"\1 como aliada", text)
                _record(
                    rpt.text_replacements,
                    f"{variant} como aliado->{variant} como aliada",
                    count,
                )
            else:
                text, count = re.subn(rf"\b({escaped}) como aliada\b", r"\1 como aliado", text)
                _record(
                    rpt.text_replacements,
                    f"{variant} como aliada->{variant} como aliado",
                    count,
                )
    return text


def apply_glossary_bad_aliases(
    text: str, glossary_terms: list[dict], report: ReviewReport | None = None
) -> str:
    """Substitui apenas formas explicitamente proibidas pelo termo canonico."""
    rpt = report or ReviewReport()
    for term in glossary_terms:
        pt = str(term.get("pt", "")).strip()
        if not pt:
            continue
        target_replacements = term.get("target_replacements") or {}
        if isinstance(target_replacements, dict):
            for alias, replacement in target_replacements.items():
                alias_s = str(alias).strip()
                replacement_s = str(replacement).strip()
                if not alias_s or not replacement_s:
                    continue
                pattern = rf"(?<!\w){re.escape(alias_s)}(?!\w)"
                text, count = re.subn(pattern, replacement_s, text, flags=re.IGNORECASE)
                _record(rpt.glossary_replacements, f"{alias_s}->{replacement_s}", count)
        bad_aliases = term.get("bad_aliases") or term.get("forbidden_aliases") or []
        if isinstance(bad_aliases, str):
            bad_aliases = [bad_aliases]
        if not isinstance(bad_aliases, list):
            continue
        for alias in bad_aliases:
            alias_s = str(alias).strip()
            if not alias_s or alias_s == pt:
                continue
            pattern = rf"(?<!\w){re.escape(alias_s)}(?!\w)"
            # Uma forma proibida que só difere por caixa deve preservar essa
            # diferença: `nome arcano` -> `Nome Arcano`, sem reescrever o canônico.
            flags = 0 if alias_s.casefold() == pt.casefold() else re.IGNORECASE
            text, count = re.subn(pattern, pt, text, flags=flags)
            _record(rpt.glossary_replacements, f"{alias_s}->{pt}", count)
    return text


def apply_duplicate_canonical_name_fixes(
    text: str, glossary_terms: list[dict], report: ReviewReport | None = None
) -> str:
    """Colapsa duplicatas causadas por expansao indevida de nomes canonicos."""
    rpt = report or ReviewReport()
    for term in glossary_terms:
        pt = str(term.get("pt", "")).strip()
        if not pt or len(pt.split()) < 2:
            continue
        key = str(term.get("key", "")).strip()
        if key.casefold() != pt.casefold():
            continue
        category = (
            str(term.get("category") or term.get("type") or term.get("term_type") or "")
            .strip()
            .casefold()
        )
        if "person" not in category and "personagem" not in category:
            continue
        parts = pt.split()
        first = parts[0]
        last = parts[-1]
        patterns = [
            (rf"(?<!\w){re.escape(first)}\s+{re.escape(pt)}(?!\w)", pt),
            (rf"(?<!\w){re.escape(pt)}\s+{re.escape(last)}(?!\w)", pt),
            (rf"(?<!\w){re.escape(pt)}\s+{re.escape(pt)}(?!\w)", pt),
        ]
        for _ in range(3):
            changed = False
            for pattern, repl in patterns:
                text, count = re.subn(pattern, repl, text, flags=re.IGNORECASE)
                if count:
                    changed = True
                    _record(rpt.text_replacements, f"{pattern}->{repl}", count)
            if not changed:
                break
    return text


def _is_named_entity_term(term: dict) -> bool:
    """Verifica se a entrada do glossário representa uma entidade nomeada."""
    category = (
        str(term.get("category") or term.get("type") or term.get("term_type") or "")
        .strip()
        .casefold()
    )
    entity_markers = (
        "person",
        "personagem",
        "criatura",
        "local",
        "organiza",
        "organização",
        "raça",
        "apelido",
    )
    return any(marker in category for marker in entity_markers)


def _can_use_name_parts(term: dict) -> bool:
    """Indica se partes isoladas de um nome composto são seguras."""
    category = (
        str(term.get("category") or term.get("type") or term.get("term_type") or "")
        .strip()
        .casefold()
    )
    return (
        "person" in category
        or "personagem" in category
        or "criatura" in category
        or "apelido" in category
    )


def _looks_like_proper_name(value: str) -> bool:
    """Estima se o texto tem a forma de um nome próprio."""
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", value)
    if not words:
        return False
    connectors = {"da", "das", "de", "do", "dos", "e"}
    return all(word[:1].isupper() or word.casefold() in connectors for word in words)


def _named_entity_variants(term: dict) -> list[str]:
    """Gera variantes comparáveis de uma entidade nomeada."""
    variants: list[str] = []
    for key_field in ("pt", "key", "source_aliases", "aliases", "allowed_target_aliases"):
        value = term.get(key_field)
        values = _term_string_list(value)
        if key_field in {"pt", "key"} and isinstance(value, str):
            values = [value.strip()] if value.strip() else []
        variants.extend(values)

    if _can_use_name_parts(term):
        # Partes isoladas são úteis para nomes pessoais como "Mara Vale",
        # mas não para títulos traduzidos como "Irmão Mais Velho do ...":
        # estes fariam palavras comuns em CAPS virarem falsos nomes próprios.
        canonical_pt = str(term.get("pt", "")).strip()
        source_key = str(term.get("key", "")).strip()
        if (
            canonical_pt
            and canonical_pt.casefold() == source_key.casefold()
            and _looks_like_proper_name(canonical_pt)
        ):
            variants.extend(
                part for part in canonical_pt.split() if len(part) >= 3 and part[:1].isupper()
            )

    unique: dict[str, str] = {}
    for value in variants:
        clean = value.strip()
        if len(clean) < 3 or not _looks_like_proper_name(clean):
            continue
        unique.setdefault(clean.casefold(), clean)
    return sorted(unique.values(), key=len, reverse=True)


def normalize_all_caps_entity_names(
    text: str, glossary_terms: list[dict], report: ReviewReport | None = None
) -> str:
    """Restaura a capitalização canônica de entidades extraídas em maiúsculas."""
    rpt = report or ReviewReport()
    for term in glossary_terms:
        if not _is_named_entity_term(term):
            continue
        for variant in _named_entity_variants(term):
            upper = variant.upper()
            if upper == variant:
                continue
            pattern = rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ]){re.escape(upper)}(?![A-Za-zÀ-ÖØ-öø-ÿ])"
            text, count = re.subn(pattern, variant, text)
            _record(rpt.all_caps_name_replacements, f"{upper}->{variant}", count)
    return text


def restore_headings_from_sections(
    text: str, sections: list[dict], report: ReviewReport | None = None
) -> str:
    """Reinsere títulos presentes no mapa de seções e ausentes na tradução."""
    rpt = report or ReviewReport()
    if not sections:
        return text
    paragraphs = re.split(r"\n\s*\n", text.strip())
    section_titles = [
        str(sec.get("title", "")) for sec in sections if str(sec.get("title", "")).strip()
    ]
    section_titles = [title for title in section_titles if title.lower() != "full text"]
    for title in section_titles:
        heading, changed = ensure_section_heading("", title)
        if not heading or not changed:
            continue
        heading_plain = heading.lstrip("#").strip()
        if heading_plain.endswith(":"):
            heading_re = re.compile(
                rf"^#?\s*{re.escape(heading_plain)}(?:\s+\S.*)?\s*$",
                flags=re.IGNORECASE,
            )
        else:
            heading_re = re.compile(rf"^#?\s*{re.escape(heading_plain)}\s*$", flags=re.IGNORECASE)
        if any(
            heading_re.match(p.strip().splitlines()[0].strip()) for p in paragraphs if p.strip()
        ):
            continue
        insert_idx = _guess_heading_insert_index(paragraphs, title)
        if insert_idx is None:
            continue
        paragraphs.insert(insert_idx, heading)
        rpt.heading_fixes += 1
    return "\n\n".join(p.strip() for p in paragraphs if p.strip())


def _guess_heading_insert_index(paragraphs: list[str], source_title: str) -> int | None:
    """Retorna uma posição segura apenas para seções iniciais inequívocas."""
    del paragraphs
    if source_title.strip().casefold() == "prologue":
        return 0
    return None


def review_translation_text(
    text: str,
    *,
    sections: list[dict] | None = None,
    glossary_terms: list[dict] | None = None,
) -> tuple[str, ReviewReport]:
    """Revisa tradução texto."""
    report = ReviewReport()
    reviewed = restore_headings_from_sections(text, sections or [], report)
    reviewed = apply_editorial_replacements(reviewed, report)
    reviewed = apply_glossary_bad_aliases(reviewed, glossary_terms or [], report)
    reviewed = normalize_all_caps_entity_names(reviewed, glossary_terms or [], report)
    reviewed = apply_duplicate_canonical_name_fixes(reviewed, glossary_terms or [], report)
    reviewed = apply_gendered_article_fixes(reviewed, glossary_terms or [], report)
    return reviewed, report


def finalize_translation_text(
    text: str,
    *,
    source_text: str = "",
    sections: list[dict] | None = None,
    glossary_terms: list[dict] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Aplica a revisão final não destrutiva e produz um relatório de QA.

    Esta etapa é usada tanto após a tradução quanto após o refino. Ela combina
    normalização estrutural, regras editoriais determinísticas e checagens de
    qualidade; não chama LLM e não muda o sentido do texto.
    """
    terms = glossary_terms or []
    normalized = normalize_structure(final_pt_postprocess(text))
    reviewed, editorial = review_translation_text(
        normalized, sections=sections, glossary_terms=terms
    )
    reviewed = normalize_structure(final_pt_postprocess(reviewed))
    reviewed, quote_balance_fixed = fix_unbalanced_quotes(reviewed)
    if quote_balance_fixed:
        # ``fix_unbalanced_quotes`` pode inserir o fechamento antes da próxima
        # abertura. Quando essa abertura inicia um novo parágrafo, move o
        # fechamento para o fim da fala anterior antes da limpeza final.
        reviewed = re.sub(r"(?m)([^\n])\n(?:[ \t]*\n)+[ \t]*”(?=“)", r"\1”\n\n", reviewed)
        reviewed = normalize_structure(final_pt_postprocess(reviewed))
    reviewed, quote_blank_lines_fixed = fix_blank_lines_inside_quotes(reviewed)
    if quote_blank_lines_fixed:
        reviewed = normalize_structure(final_pt_postprocess(reviewed))
    quality = run_translation_quality_checks(source_text, reviewed, terms)
    return reviewed, {
        "editorial": editorial.to_dict(),
        "quote_blank_lines_fixed": quote_blank_lines_fixed,
        "quote_balance_fixed": quote_balance_fixed,
        "quality": quality,
    }


def load_sections(path: str | Path | None) -> list[dict]:
    """Carrega seções."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def load_glossary_terms(path: str | Path | None) -> list[dict]:
    """Carrega glossário termos."""
    if not path:
        return []
    import logging

    state = build_glossary_state(Path(path), None, logging.getLogger(__name__), manual_dir=None)
    return state.manual_terms if state else []
