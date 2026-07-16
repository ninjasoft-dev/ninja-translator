from __future__ import annotations

import re

from .quote_fix import collapse_repeated_curly_quotes

_BIRTH_CONTEXT = re.compile(
    r"\b(bebe|bebes|bebê|bebês|filh[oa]s?|gravidez|gr[aá]vida|gesta[cç][aã]o|matern|parto)\b",
    re.IGNORECASE,
)
_DASH_TRAIL_QUOTE_RE = re.compile(r"^(—\s.*?)[\"”](?=[,;.\s]|$)")
_DASH_LEAD_QUOTE_RE = re.compile(r"^(—\s*)[\"“]\s*(.+)$")
_DASH_SPEECH_TAG_RE = re.compile(r"^(—\s[^\"”]{0,120}?)[\"”]([\s,;.!?]+[A-Z\u00c0-\u017f].*)$")


def postprocess_translation(pt_text: str, en_text: str | None = None) -> str:
    """
    Ajustes determinísticos pós-tradução para falsos cognatos comuns e artefatos de fala.

    - Se o original contém parry/parried/parrying e a tradução trouxe parriu/parrir/parrindo/etc.,
      substitui por formas de "aparar".
    - Evita interferir em contextos de parto (bebê/gravidez/etc).
    - Remove aspas sobrando em falas que começam com travessão.
    """
    if not pt_text:
        return pt_text

    pt_text, _ = collapse_repeated_curly_quotes(pt_text)

    # Corrige resíduos híbridos do inglês antes da QA por chunk. Se forem
    # deixados para a revisão final, o guardrail os rejeita repetidamente.
    pt_text = re.sub(r"(?<![A-Za-zÀ-ÿ])I-isso\b", "S-sim", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(
        r"(?<![A-Za-zÀ-ÿ])I\s+(?=[a-zà-ÿ])",
        "Eu ",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(r"\buh+\b", "Ah", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(r"\barright\b", "Beleza", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(r"\bboost\b", "impulso", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(r"\bthey\s+todos\b", "todos", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(
        r"\bY-you\s+divindades\s+podem\b",
        "V-vocês, divindades, podem",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(r"\bY-you\b", "V-você", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(
        r"\b(super\s+)?desconfiad([ao])\s+AF\b",
        r"\1desconfiad\2 pra caramba",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(r"\bthough\b", "porém", pt_text, flags=re.IGNORECASE)

    # Parry/parried/parrying -> aparar (somente quando presente no EN e fora de contexto de parto)
    if (
        en_text is not None
        and re.search(r"\bparr(?:y|ied|ying)\b", en_text, flags=re.IGNORECASE)
        and not _BIRTH_CONTEXT.search(pt_text)
    ):

        def _replace(match: re.Match[str]) -> str:
            """Aplica a substituição atual e registra a alteração."""
            suffix = match.group(1).lower()
            if suffix in {"r", "ir"}:
                return "aparar"
            if suffix in {"u", "iu", "ou"}:
                return "aparou"
            if suffix in {"ndo"}:
                return "aparando"
            if suffix in {"ido", "ida", "idos", "idas"}:
                base = "aparad"
                end = suffix[2:]
                return f"{base}{end}"
            if suffix in {"ia", "iam"}:
                return "aparava" if suffix == "ia" else "aparavam"
            return "aparou"

        pattern = re.compile(
            r"\bparr(iu|ir|indo|ido|ida|idos|idas|ia|iam|ou|r)\b", flags=re.IGNORECASE
        )
        pt_text = pattern.sub(_replace, pt_text)

    # Limpeza de travessão + aspas mistas
    fixed_lines: list[str] = []
    for ln in pt_text.splitlines():
        cleaned = ln.strip()
        cleaned = _DASH_TRAIL_QUOTE_RE.sub(r"\1", cleaned)
        cleaned = _DASH_LEAD_QUOTE_RE.sub(r"\1\2", cleaned)
        cleaned = _DASH_SPEECH_TAG_RE.sub(r"\1\2", cleaned)
        fixed_lines.append(cleaned if cleaned != "" else ln)

    return "\n".join(fixed_lines)
