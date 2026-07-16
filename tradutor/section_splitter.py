from __future__ import annotations

import re
from typing import Dict, List

SECTION_PATTERN = re.compile(
    r"""
    ^(?P<title>(?:
        prologue|pr[oó]logo|prolog|序章|前言|プロローグ|프롤로그|
        epilogue|ep[ií]logo|epilog|終章|终章|エピローグ|에필로그|
        afterword|posf[aá]cio|p[oó]s[- ]?escrito|あとがき|後書き|后记|後記|후기|
        (?:chapter|cap[ií]tulo|chapitre|kapitel|capitolo|glava|глава)\s+
            (?:\d+|[一二三四五六七八九十百零〇]+)(?:(?::|\s*[–—-])\s*[^\n]*)?|
        第\s*(?:\d+|[一二三四五六七八九十百零〇]+)\s*[章話话巻卷](?:\s*[:：–—-]?\s*[^\n]+)?|
        제\s*(?:\d+|[일이삼사오육칠팔구십백]+)\s*장(?:\s*[:：–—-]?\s*[^\n]+)?
    ))\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_toc_stub_body(body: str) -> bool:
    """Indica se sumário simulado corpo."""
    stripped = body.strip()
    if not stripped:
        return True
    if len(stripped) <= 10 and re.fullmatch(r"[\d\s.]+", stripped):
        return True
    return False


def split_into_sections(text: str) -> List[Dict]:
    """
    Divide texto bruto em seções por marcadores de capítulo.
    Retorna lista de dicts: {"title": str, "body": str, "start_idx": int, "end_idx": int}
    Se nenhum marcador for encontrado, retorna uma única seção "Full Text".
    """
    lines = text.splitlines()
    matches = []
    for idx, ln in enumerate(lines):
        if SECTION_PATTERN.match(ln.strip()):
            matches.append((idx, ln.strip()))

    if not matches:
        return [
            {
                "title": "Full Text",
                "body": text.strip(),
                "start_idx": 0,
                "end_idx": len(text),
            }
        ]

    sections: List[Dict] = []
    first_start = matches[0][0]
    if first_start > 0:
        pre_body = "\n".join(lines[:first_start]).strip()
        if pre_body:
            pre_end_idx = sum(len(line) + 1 for line in lines[:first_start])
            sections.append(
                {
                    "title": "Full Text",
                    "body": pre_body,
                    "start_idx": 0,
                    "end_idx": pre_end_idx,
                }
            )
    for i, (start_line, title) in enumerate(matches):
        end_line = matches[i + 1][0] if i + 1 < len(matches) else len(lines)
        body_lines = lines[start_line + 1 : end_line]
        body = "\n".join(body_lines).strip()
        if _is_toc_stub_body(body):
            # Possível entrada de sumário; ignora se o corpo é vazio/curto ou numérico.
            continue
        header = f"# {title}"
        full_body = f"{header}\n\n{body}".strip()
        start_idx = sum(len(line) + 1 for line in lines[:start_line])  # approx byte offset
        end_idx = sum(len(line) + 1 for line in lines[:end_line])
        sections.append(
            {
                "title": title,
                "body": full_body,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
        )
    return sections
