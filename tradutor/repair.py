"""Reparo seletivo da tradução antes do refino."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .cache_utils import (
    cache_exists,
    chunk_hash,
    detect_model_collapse,
    load_cache,
    save_cache,
)
from .language_guardrails import (
    detect_residual_source_language,
    residual_issue_type,
)
from .languages import compile_term_pattern, normalize_source_language, source_language_name
from .llm_backend import LLMBackend
from .postprocess_translation import postprocess_translation
from .qa import count_quote_lines, count_quotes
from .sanitizer import sanitize_refine_output

REPAIR_PIPELINE_VERSION = "8"
REPAIR_START_MARKER_RE = r"###\s*TEXTO_REPARADO_INICIO"
REPAIR_END_MARKER_RE = r"###\s*TEXTO_REPARADO_FIM"


@dataclass
class RepairResult:
    """
    Estrutura de dados que armazena o resultado de uma tentativa de reparo de tradução,
    incluindo se houve mudança, tempo gasto, e problemas residuais.
    """

    text: str
    changed: bool = False
    attempted: bool = False
    used_cache: bool = False
    llm_attempts: int = 0
    issues: list[dict[str, str]] = field(default_factory=list)
    retry_reasons: list[str] = field(default_factory=list)
    suspect_output: bool = False
    suspect_reason: str = ""
    raw_output: str = ""
    elapsed_seconds: float = 0.0


def repair_prompt_fingerprint(source_language: str = "en") -> str:
    """
    Gera um hash unívoco para a estrutura base do prompt de reparo,
    útil para controle de invalidação de cache.
    """
    prompt = build_repair_prompt(
        source_text="{source}",
        translated_text="{translated}",
        issues=[{"type": residual_issue_type(source_language), "detail": "{detail}"}],
        glossary_text="{glossary}",
        source_language=source_language,
    )
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_repair_prompt(
    *,
    source_text: str,
    translated_text: str,
    issues: list[dict[str, str]],
    glossary_text: str | None = None,
    source_language: str = "en",
) -> str:
    """
    Constrói o prompt focado em consertar problemas detectados (resíduos da origem, perdas).
    O LLM recebe orientações estritas para apenas consertar e não reescrever o texto todo.
    """
    language = normalize_source_language(source_language)
    source_name = source_language_name(language)
    issue_lines = "\n".join(
        f"- {item.get('type', 'issue')}: {item.get('detail') or item.get('found') or ''}".rstrip()
        for item in issues
    )
    glossary_block = ""
    if glossary_text:
        glossary_block = (
            f"GLOSSARIO DO CHUNK (seguir exatamente; nao inventar termos):\n{glossary_text}\n\n"
        )
    return f"""
Você é um revisor de tradução de {source_name} para português brasileiro.
Você receberá o texto original, a tradução atual e uma lista objetiva de problemas detectados.

TAREFA:
Corrija SOMENTE os problemas listados. Não reescreva o trecho inteiro se não for necessário.

REGRAS:
- Traduza para PT-BR qualquer frase, fala ou trecho narrativo que ainda esteja em {source_name}.
- Corrija termos do glossário que estejam em forma não canônica.
- Preserve nomes próprios, honoríficos e termos canônicos do glossário.
- Preserve eventos, ordem narrativa, falas e sentido.
- Preserve a ordem e a quantidade de parágrafos sempre que possível.
- Não resuma, não acrescente explicações, não omita conteúdo.
- Não mude o estilo de diálogo do trecho.

PROBLEMAS DETECTADOS:
{issue_lines or "- nenhum problema listado"}

FORMATO DE SAÍDA:
Retorne exclusivamente:

### TEXTO_REPARADO_INICIO

<tradução reparada>
### TEXTO_REPARADO_FIM

{glossary_block}ORIGINAL EM {source_name.upper()}:
\"\"\"{source_text}\"\"\"

TRADUÇÃO ATUAL:
\"\"\"{translated_text}\"\"\""""


def parse_repair_output(raw: str) -> str:
    """
    Extrai o texto contido entre as tags de início e fim geradas pelo LLM no modo reparo.
    Se não achar as tags, utiliza a sanitização fallback do refino.
    """
    match = re.search(
        rf"{REPAIR_START_MARKER_RE}\s*(.*?)(?:{REPAIR_END_MARKER_RE}\s*|$)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return sanitize_refine_output(raw).strip()


def detect_translation_repair_issues(
    *,
    source_text: str,
    translated_text: str,
    glossary_terms: list[dict] | None = None,
    source_language: str = "en",
) -> list[dict[str, str]]:
    """
    Detecta de forma heurística se a tradução apresentou problemas críticos
    que merecem um ciclo de reparo automático (como resquícios do idioma de origem ou perdas drásticas de tamanho).
    """
    issues: list[dict[str, str]] = []
    residual_source, reason = detect_residual_source_language(translated_text, source_language)
    if residual_source:
        issues.append({"type": residual_issue_type(source_language), "detail": reason})

    source_quotes = count_quotes(source_text)
    translated_quotes = count_quotes(translated_text)
    if source_quotes >= 4 and translated_quotes <= max(1, int(source_quotes * 0.4)):
        issues.append(
            {
                "type": "possible_dialogue_omission",
                "detail": f"quotes {source_quotes}->{translated_quotes}",
            }
        )
    source_quote_lines = count_quote_lines(source_text)
    translated_quote_lines = count_quote_lines(translated_text)
    if source_quote_lines >= 2 and translated_quote_lines <= max(1, int(source_quote_lines * 0.4)):
        issues.append(
            {
                "type": "possible_dialogue_line_omission",
                "detail": f"quote_lines {source_quote_lines}->{translated_quote_lines}",
            }
        )

    ratio = (
        len(translated_text.strip()) / max(len(source_text.strip()), 1)
        if source_text.strip()
        else 1.0
    )
    if ratio < 0.55:
        issues.append({"type": "possibly_too_short", "detail": f"ratio {ratio:.2f}"})

    for term in glossary_terms or []:
        key = str(term.get("key", "")).strip()
        pt = str(term.get("pt", "")).strip()
        if not key or not pt or key.casefold() == pt.casefold():
            continue
        if (
            _contains(source_text, key)
            and _contains(translated_text, key)
            and not _contains(translated_text, pt)
        ):
            issues.append({"type": "source_term_leak", "found": key, "detail": f"use {pt}"})
        bad_aliases = term.get("bad_aliases") or term.get("forbidden_aliases") or []
        if isinstance(bad_aliases, str):
            bad_aliases = [bad_aliases]
        if isinstance(bad_aliases, list):
            for alias in bad_aliases:
                alias_s = str(alias).strip()
                if alias_s and _contains(translated_text, alias_s):
                    issues.append({"type": "bad_alias", "found": alias_s, "detail": f"use {pt}"})
    return issues


def validate_repair_candidate(
    *,
    source_text: str,
    translated_text: str,
    candidate_text: str,
    source_language: str = "en",
) -> str | None:
    """Rejeita reparos que parecem ter removido conteúdo já traduzido."""
    candidate = candidate_text.strip()
    current = translated_text.strip()
    if not candidate:
        return "empty_repair"
    if not current:
        return None

    ratio = len(candidate) / max(len(current), 1)
    if len(current) >= 600 and ratio < 0.90:
        return f"repair_removed_content_ratio:{ratio:.2f}"

    current_paragraphs = _paragraph_count(current)
    candidate_paragraphs = _paragraph_count(candidate)
    if current_paragraphs >= 4 and candidate_paragraphs < max(2, int(current_paragraphs * 0.75)):
        return f"repair_removed_paragraphs:{candidate_paragraphs}/{current_paragraphs}"

    current_quote_lines = count_quote_lines(current)
    candidate_quote_lines = count_quote_lines(candidate)
    if current_quote_lines >= 4 and candidate_quote_lines < max(2, int(current_quote_lines * 0.75)):
        return f"repair_removed_dialogue_lines:{candidate_quote_lines}/{current_quote_lines}"

    first_line = _first_meaningful_line(current)
    if (
        first_line
        and len(first_line) >= 30
        and not detect_residual_source_language(first_line, source_language)[0]
    ):
        head = candidate[: max(600, len(first_line) * 3)]
        if first_line not in head:
            return "repair_removed_opening"

    return None


def repair_translation_chunk(
    *,
    source_text: str,
    translated_text: str,
    backend: LLMBackend,
    logger: logging.Logger,
    glossary_text: str | None = None,
    glossary_terms: list[dict] | None = None,
    max_attempts: int = 2,
    cache_metadata: dict[str, Any] | None = None,
    source_language: str = "en",
) -> RepairResult:
    """
    Coordena o ciclo de detecção e, se necessário, reparação de uma tradução imperfeita.
    Faz uso de cache para economizar chamadas no caso de reparos já feitos anteriormente.
    """
    started = time.perf_counter()
    issues = detect_translation_repair_issues(
        source_text=source_text,
        translated_text=translated_text,
        glossary_terms=glossary_terms,
        source_language=source_language,
    )
    if not issues:
        return RepairResult(
            text=translated_text,
            issues=[],
            elapsed_seconds=time.perf_counter() - started,
        )

    metadata = {
        "mode": "repair",
        "pipeline_version": REPAIR_PIPELINE_VERSION,
        "prompt_hash": repair_prompt_fingerprint(source_language),
        "source_language": normalize_source_language(source_language),
        "backend": getattr(backend, "backend", None),
        "model": getattr(backend, "model", None),
        "num_predict": getattr(backend, "num_predict", None),
        "temperature": getattr(backend, "temperature", None),
        "repeat_penalty": getattr(backend, "repeat_penalty", None),
        **(cache_metadata or {}),
    }
    cache_key = chunk_hash(
        json.dumps(
            {
                "source": source_text,
                "translated": translated_text,
                "glossary": glossary_text or "",
                "issues": issues,
                "metadata": metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if cache_exists("repair", cache_key):
        cached = load_cache("repair", cache_key)
        final_output = str(cached.get("final_output") or "")
        if final_output and not validate_repair_candidate(
            source_text=source_text,
            translated_text=translated_text,
            candidate_text=final_output,
            source_language=source_language,
        ):
            return RepairResult(
                text=final_output,
                changed=final_output.strip() != translated_text.strip(),
                attempted=True,
                used_cache=True,
                issues=issues,
                elapsed_seconds=time.perf_counter() - started,
            )

    prompt = build_repair_prompt(
        source_text=source_text,
        translated_text=translated_text,
        issues=issues,
        glossary_text=glossary_text,
        source_language=source_language,
    )
    last_text = translated_text
    raw_output = ""
    retry_reasons: list[str] = []
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            response = backend.generate(prompt)
            raw_output = response.text
        except Exception as exc:
            retry_reasons.append(f"llm_error:{exc}")
            logger.warning(
                "Repair falhou na chamada LLM tentativa %d/%d: %s",
                attempt,
                max_attempts,
                exc,
            )
            continue
        candidate = parse_repair_output(raw_output)
        candidate = postprocess_translation(candidate, source_text)
        if not candidate.strip():
            retry_reasons.append("empty_repair")
        elif detect_model_collapse(candidate, original_len=len(translated_text), mode="refine"):
            retry_reasons.append("collapse_detector")
        else:
            residual_source, residual_reason = detect_residual_source_language(
                candidate, source_language
            )
            if residual_source:
                retry_reasons.append(residual_reason)
            else:
                validation_reason = validate_repair_candidate(
                    source_text=source_text,
                    translated_text=translated_text,
                    candidate_text=candidate,
                    source_language=source_language,
                )
                if validation_reason:
                    retry_reasons.append(validation_reason)
                    candidate = translated_text
                else:
                    save_cache(
                        "repair",
                        cache_key,
                        raw_output=raw_output,
                        final_output=candidate,
                        metadata=metadata,
                    )
                    return RepairResult(
                        text=candidate,
                        changed=candidate.strip() != translated_text.strip(),
                        attempted=True,
                        llm_attempts=attempt,
                        issues=issues,
                        retry_reasons=retry_reasons,
                        raw_output=raw_output,
                        elapsed_seconds=time.perf_counter() - started,
                    )
        last_text = candidate or last_text
        logger.warning(
            "Repair manteve problema na tentativa %d/%d: %s",
            attempt,
            max_attempts,
            retry_reasons[-1] if retry_reasons else "unknown",
        )
        prompt = build_repair_prompt(
            source_text=source_text,
            translated_text=last_text,
            issues=issues,
            glossary_text=glossary_text,
            source_language=source_language,
        )
        prompt += "\n\nATENÇÃO: a tentativa anterior ainda falhou. Corrija apenas os problemas listados, sem reescrever o trecho inteiro."

    suspect_reason = retry_reasons[-1] if retry_reasons else "repair_failed"
    return RepairResult(
        text=translated_text,
        attempted=True,
        llm_attempts=max(1, max_attempts),
        issues=issues,
        retry_reasons=retry_reasons,
        suspect_output=True,
        suspect_reason=suspect_reason,
        raw_output=raw_output,
        elapsed_seconds=time.perf_counter() - started,
    )


def _contains(text: str, needle: str) -> bool:
    """Verifica se o texto contém o termo respeitando limites de palavra."""
    if not text or not needle:
        return False
    return bool(compile_term_pattern(needle).search(text))


def _paragraph_count(text: str) -> int:
    """Conta os parágrafos não vazios de um trecho."""
    return len([part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()])


def _first_meaningful_line(text: str) -> str:
    """Retorna a primeira linha com conteúdo relevante."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""
