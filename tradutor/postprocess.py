"""Pós-processamento determinístico e semanticamente neutro para PT-BR."""

from __future__ import annotations

import re


def _normalize_straight_dialogue_quotes(text: str) -> str:
    """Converte pares seguros de aspas retas em aspas tipográficas de diálogo."""
    positions = [
        index
        for index, character in enumerate(text)
        if character == '"'
        and not (index > 0 and text[index - 1].isdigit())
        and not (index + 1 < len(text) and text[index + 1].isdigit())
    ]
    if not positions or len(positions) % 2:
        return text

    replacements = {index: "“" if order % 2 == 0 else "”" for order, index in enumerate(positions)}
    return "".join(replacements.get(index, character) for index, character in enumerate(text))


def _normalize_quote_spacing(text: str) -> str:
    """Corrige espaços ao redor de aspas sem modificar palavras."""
    text = re.sub(r"([“\"])[ \t]+", r"\1", text)
    text = re.sub(r"[ \t]+([”\"])", r"\1", text)
    text = re.sub(r"([”\"])(?=[“\"])", r"\1 ", text)
    text = re.sub(r"(?<=[\w.,;:!?…])(?=“[A-Za-zÀ-ÿ])", " ", text)
    return re.sub(r"(?<=[”\"])(?=[A-Za-zÀ-ÿ])", " ", text)


def normalize_dialogue_quotes(text: str) -> str:
    """Normaliza pares seguros de aspas de diálogo."""
    return _normalize_quote_spacing(_normalize_straight_dialogue_quotes(text))


def _normalize_line_initial_dashes(text: str) -> str:
    """Converte hífen ou meia-risca em travessão no início de falas."""
    normalized_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("- ", "– ")):
            indentation = line[: len(line) - len(stripped)]
            line = indentation + "— " + stripped[2:]
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def _separate_narrative_paragraphs(text: str) -> str:
    """Insere linha vazia entre blocos narrativos consecutivos."""
    output_lines: list[str] = []
    previous_nonempty = False
    previous_dialogue = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            previous_nonempty = False
            previous_dialogue = False
            continue

        is_dialogue = stripped.startswith("— ")
        if previous_nonempty and not is_dialogue and not previous_dialogue:
            output_lines.append("")
        output_lines.append(stripped)
        previous_nonempty = True
        previous_dialogue = is_dialogue

    return "\n".join(output_lines).strip()


def final_pt_postprocess(text: str) -> str:
    """Normaliza pontuação, marcadores e parágrafos sem fazer correções lexicais."""
    if not text:
        return text

    cleaned = re.sub(r"\.{3,}", "…", text)
    cleaned = cleaned.replace("--", "—")
    cleaned = re.sub(r"(?m)^([ \t]*)\*{2}(?=[ \t]+\S)", r"\1***", cleaned)
    cleaned = normalize_dialogue_quotes(cleaned)

    # Contrações inglesas podem deixar aspas simples dentro de uma fala já
    # delimitada por aspas curvas; apenas os delimitadores redundantes saem.
    cleaned = re.sub(r"“['‘]", "“", cleaned)
    cleaned = re.sub(r"['’]”", "”", cleaned)
    cleaned = re.sub(r"(?m)^(\s*)”\s*(?=“[A-Za-zÀ-ÿ])", r"\1", cleaned)
    cleaned = re.sub(r"([”\"])\1+", r"\1", cleaned)
    cleaned = re.sub(r"([.!?…][”\"])\*(?=\s|$)", r"\1", cleaned)
    cleaned = re.sub(r"([?!…])”\.", r"\1”", cleaned)
    cleaned = re.sub(r"—,\s*", "—", cleaned)
    cleaned = re.sub(r"(?<=\S)—(?=\S)", " — ", cleaned)
    cleaned = re.sub(r"—(?=[A-Za-zÀ-ÿ])", "— ", cleaned)
    cleaned = re.sub(r"\s+([.!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = _normalize_quote_spacing(cleaned)
    cleaned = _normalize_line_initial_dashes(cleaned)

    cleaned = re.sub(r"###\s*TEXTO_TRADUZ[A-Z_]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"###\s*TEXTO_REFINADO_[A-Z_]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return _separate_narrative_paragraphs(cleaned)
