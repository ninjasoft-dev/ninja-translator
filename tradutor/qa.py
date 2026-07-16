from __future__ import annotations

import re

ELLIPSIS_IN_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]\.\.\.[A-Za-zÀ-ÿ]|[A-Za-zÀ-ÿ]…[A-Za-zÀ-ÿ]")
LOWERCASE_START_RE = re.compile(r"^[a-zà-ÿ].*")
TRUNCATED_ELLIPSIS_RE = re.compile(r"(?:^|\s)[A-Za-zÀ-ÿ]{1,4}(?:\.{3}|…)\s*$")
MALFORMED_QUOTE_BOUNDARY_RE = re.compile(r"”[ \t]*“(?=[A-Za-zÀ-ÿ])")


def _has_suspicious_repetition(text: str, min_repeats: int = 3) -> bool:
    """Sinais fortes; precisa de 2 ou mais."""
    signals = 0
    if re.search(r"(.{120,}?)(?:\s+\1){1,}", text, flags=re.DOTALL):
        signals += 1
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) >= 40:
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        if unique_ratio < 0.25:
            signals += 1
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    counts = {}
    for s in sentences:
        counts[s] = counts.get(s, 0) + 1
    if any(c >= min_repeats for c in counts.values()):
        signals += 1
    return signals >= 2


def _has_meta_noise(text: str) -> bool:
    """Verifica se a saída contém metatexto produzido pelo modelo."""
    lower = text.lower()
    markers = [
        "as an ai",
        "<think>",
        "</think>",
        "sou um modelo de linguagem",
        "analysis:",
    ]
    return any(m in lower for m in markers)


def count_quotes(text: str) -> int:
    """Conta aspas."""
    return len([ch for ch in text if ch in {'"', "“", "”", "‟", "❝", "❞"}])


def count_quote_lines(text: str) -> int:
    """Conta as linhas que contêm aspas de diálogo."""
    return sum(1 for ln in text.splitlines() if ln.strip().startswith(('"', "“", "”")))


def has_malformed_quote_boundary(text: str) -> bool:
    """Detecta fechamento seguido de abertura colados antes de uma fala.

    O padrão ``”“Fala`` é uma regressão de formatação: as aspas ficam
    globalmente balanceadas, mas a fala começa com um fechamento espúrio.
    """
    return bool(MALFORMED_QUOTE_BOUNDARY_RE.search(text or ""))


def _count_curly_quotes(text: str) -> tuple[int, int]:
    """Conta separadamente as aspas curvas de abertura e fechamento."""
    return text.count("“"), text.count("”")


def _has_internal_missing_open_quote(text: str) -> bool:
    """Detecta uma linha de fala fechada sem abertura no mesmo chunk.

    PDFs podem perder a abertura de uma fala e, em outro ponto do chunk,
    preservar uma abertura sem fechamento por causa da fronteira seguinte.
    A contagem global continua balanceada, mas a estrutura não.
    """
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or "“" in stripped or "”" not in stripped:
            continue
        content = stripped.replace("”", "").strip()
        if len(content) >= 8:
            return True
    return False


def _allows_single_source_quote_repair(input_text: str, output_text: str) -> bool:
    """Aceita um único par extra que repara defeito interno verificável da fonte."""
    source_open, source_close = _count_curly_quotes(input_text)
    output_open, output_close = _count_curly_quotes(output_text)
    return (
        source_open > 0
        and source_open == source_close
        and output_open == output_close
        and (output_open + output_close) == (source_open + source_close + 2)
        and _has_internal_missing_open_quote(input_text)
    )


def _allows_single_missing_open_repair(input_text: str, output_text: str) -> bool:
    """Aceita uma abertura adicionada para fechar uma fala quebrada pela extração."""
    source_open, source_close = _count_curly_quotes(input_text)
    output_open, output_close = _count_curly_quotes(output_text)
    return (
        source_close == source_open + 1
        and output_open == source_open + 1
        and output_close == source_close
        and output_open == output_close
        and _has_internal_missing_open_quote(input_text)
    )


def has_curly_quote_balance_regression(input_text: str, output_text: str) -> bool:
    """Verifica se a saída alterou o estado de aspas de um chunk.

    Um chunk pode iniciar ou terminar no meio de uma fala. Nessa situacao ele
    nao precisa ser balanceado isoladamente: a tradução deve preservar o mesmo
    delta entre aberturas e fechamentos presente no original.
    """
    input_curly = _count_curly_quotes(input_text)
    output_curly = _count_curly_quotes(output_text)
    input_has_curly = any(input_curly)
    output_has_curly = any(output_curly)

    if input_has_curly and output_has_curly:
        if (input_curly[0] - input_curly[1]) != (output_curly[0] - output_curly[1]):
            return not _allows_single_missing_open_repair(input_text, output_text)
        return False
    if input_has_curly:
        output_straight = output_text.count('"')
        if output_straight:
            # Alguns modelos preservam a estrutura da fala, mas trocam aspas
            # curvas por retas. Nessa situação só a paridade é comparável.
            return (sum(input_curly) % 2) != (output_straight % 2)
        return (input_curly[0] - input_curly[1]) != 0
    if output_has_curly:
        input_straight_parity = input_text.count('"') % 2
        output_curly_parity = (output_curly[0] + output_curly[1]) % 2
        return input_straight_parity != output_curly_parity
    return False


def has_curly_quote_count_regression(input_text: str, output_text: str) -> bool:
    """Evita que o modelo invente pares extras de aspas em um chunk PT-BR.

    Um trecho pode atravessar a fronteira de uma fala e ter contagem ímpar;
    por isso a diferença abertura-fechamento continua sendo validada pela
    função acima. Ainda assim, quando o original já usa aspas curvas, o
    refinador não deve acrescentar uma nova fala entre elas.
    """
    input_total = sum(_count_curly_quotes(input_text))
    output_total = sum(_count_curly_quotes(output_text))
    if input_total > 0 and output_total > input_total:
        return not (
            _allows_single_source_quote_repair(input_text, output_text)
            or _allows_single_missing_open_repair(input_text, output_text)
        )
    return False


def _excess_repeated_short_lines(input_text: str, output_text: str) -> bool:
    """
    Detecta se a saída introduziu repetição excessiva de linhas curtas.

    Entrada e saída usam idiomas diferentes, portanto não é válido comparar o
    texto literal das linhas. Comparamos o pico de repetição: uma fala que já
    se repete no original (como uma personagem insistindo em uma ordem) pode
    continuar repetida na tradução, mas não pode crescer muito além dele.
    """

    def _short_lines(text: str) -> list[str]:
        """Seleciona linhas curtas usadas na detecção de repetições."""
        lines = []
        for ln in text.splitlines():
            stripped = ln.strip()
            if not stripped:
                continue
            if len(stripped) <= 25:
                lines.append(stripped.lower())
        return lines

    from collections import Counter

    out_counts = Counter(_short_lines(output_text))
    if not out_counts:
        return False
    source_counts = Counter(_short_lines(input_text))
    source_peak = max(source_counts.values(), default=0)
    output_peak = max(out_counts.values(), default=0)
    return output_peak >= max(3, source_peak + 2)


def _paragraph_count(text: str) -> int:
    """Conta os parágrafos não vazios de um trecho."""
    return len([part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()])


def _meaningful_line_count(text: str) -> int:
    """Conta linhas com conteúdo relevante para comparar estruturas."""
    return sum(1 for line in (text or "").splitlines() if line.strip())


def needs_retry(
    input_text: str,
    output_text: str,
    *,
    input_quotes: int | None = None,
    output_quotes: int | None = None,
    input_quote_lines: int | None = None,
    output_quote_lines: int | None = None,
    contamination_detected: bool = False,
    sanitization_ratio: float = 1.0,
) -> tuple[bool, str]:
    """Indica se nova tentativa."""
    iq = input_quotes if input_quotes is not None else count_quotes(input_text)
    oq = output_quotes if output_quotes is not None else count_quotes(output_text)
    if has_malformed_quote_boundary(output_text):
        return True, "malformed_quote_boundary"
    if has_curly_quote_balance_regression(input_text, output_text):
        return True, "unbalanced_quotes"
    if has_curly_quote_count_regression(input_text, output_text):
        return True, "extra_curly_quotes"
    quote_repair_allowed = _allows_single_source_quote_repair(
        input_text, output_text
    ) or _allows_single_missing_open_repair(input_text, output_text)
    if (iq % 2 != oq % 2 or (iq and oq > iq + 4)) and not quote_repair_allowed:
        return True, "unbalanced_quotes_straight"
    if iq >= 4 and oq < iq - 2:
        return True, "omissao_dialogo_quotes"
    iql = input_quote_lines if input_quote_lines is not None else count_quote_lines(input_text)
    oql = output_quote_lines if output_quote_lines is not None else count_quote_lines(output_text)
    if iql >= 2 and oql < max(1, iql - 1):
        return True, "omissao_dialogo_linhas"
    if not output_text or not output_text.strip():
        return True, "output vazio"
    input_paragraphs = _paragraph_count(input_text)
    output_paragraphs = _paragraph_count(output_text)
    if (
        input_paragraphs >= 2
        and output_paragraphs < input_paragraphs
        and _meaningful_line_count(output_text) < input_paragraphs
    ):
        return True, f"omissao_paragrafos ({output_paragraphs}/{input_paragraphs})"
    if ELLIPSIS_IN_WORD_RE.search(output_text):
        return True, "ellipsis_in_word"
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", output_text) if p.strip()]
    for p in paragraphs:
        if (
            len(paragraphs) > 1
            and len(p) <= 30
            and LOWERCASE_START_RE.match(p)
            and not p.startswith(('"', "“", "”", "-", "—"))
        ):
            return True, "lowercased_fragment"
    for p in paragraphs:
        if LOWERCASE_START_RE.match(p) and not p.startswith(('"', "“", "”", "-", "—")):
            return True, "lowercase_narration_start"
    if TRUNCATED_ELLIPSIS_RE.search(output_text.strip()):
        return True, "truncated_token_ellipsis"
    input_ellipsis = input_text.count("...") + input_text.count("…")
    output_ellipsis = output_text.count("...") + output_text.count("…")
    if input_ellipsis == 0 and output_ellipsis >= 2:
        ellipsis_ratio = output_ellipsis / max(len(output_text), 1)
        if len(output_text) < 500 or ellipsis_ratio > 0.01:
            return True, "ellipsis_suspect"
    if _excess_repeated_short_lines(input_text, output_text):
        return True, "extra_short_repetition"
    ratio = len(output_text.strip()) / max(len(input_text.strip()), 1)
    if ratio < 0.6:
        return True, "output truncado (ratio < 0.6)"
    if ratio > 1.8:
        return True, "output longo (ratio > 1.8)"
    if _has_suspicious_repetition(output_text):
        return True, "repeticao suspeita"
    if _has_meta_noise(output_text):
        return True, "meta noise detectado"
    if _excess_repeated_short_lines(input_text, output_text):
        return True, "extra_short_repetition"
    if contamination_detected and sanitization_ratio < 0.95:
        return True, "sanitizacao_agressiva"
    return False, ""
