"""
Conversão simples de Markdown para PDF usando ReportLab (compatível com Windows).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from .config import AppConfig
from .utils import ensure_dir


def select_font_path(preferred: str | Path | None, fallbacks: Iterable[str]) -> Path | None:
    """
    Retorna o primeiro caminho de fonte existente dentre preferida e fallbacks.
    """
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(fallbacks or [])
    for cand in candidates:
        p = Path(cand)
        if p.is_file():
            return p
    return None


def _inline_markdown_to_html(text: str) -> str:
    """
    Converte itálico/negrito simples para tags HTML suportadas pelo Paragraph.
    """
    escaped = escape(text)
    # bold **text** ou __text__
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"__(.+?)__", r"<b>\1</b>", escaped)
    # italic *text* ou _text_
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", escaped)
    return escaped


def normalize_markdown_for_pdf(text: str) -> list[str]:
    """
    Normaliza quebras para PDF:
    - converte <br> / <br/> em quebras reais
    - separa parágrafos por linhas em branco
    - retorna lista de parágrafos limpos (preservando quebras internas)
    """
    if not text:
        return []
    normalized = text.replace("<br/>", "\n").replace("<br>", "\n")
    # normaliza quebras
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", normalized)
    parts: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        parts.append(block)
    return parts


def convert_markdown_to_pdf(
    md_path: Path,
    output_path: Path,
    cfg: AppConfig,
    logger: logging.Logger,
    title: str | None = None,
) -> None:
    """
    Gera PDF simples a partir de Markdown leve (parágrafos e itálico/negrito).
    """
    try:
        from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except Exception as exc:  # pragma: no cover - depende de reportlab instalado
        raise RuntimeError(
            "ReportLab não está instalado; instale reportlab para gerar PDFs."
        ) from exc

    if not md_path.exists():
        raise FileNotFoundError(md_path)

    font_path = select_font_path(cfg.pdf_font_file, cfg.pdf_font_fallbacks)
    if not font_path:
        raise RuntimeError(
            "Nenhuma fonte encontrada para o PDF. "
            "Defina pdf_font.file ou pdf_font_fallbacks para um caminho TTF/OTF válido."
        )
    font_name = font_path.stem
    try:
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
        logger.info("Fonte registrada para PDF: %s", font_path)
    except Exception as exc:  # pragma: no cover - depende do ambiente
        raise RuntimeError(f"Falha ao registrar fonte {font_path}: {exc}") from exc

    ensure_dir(output_path.parent)

    # prepara estilos
    leading = cfg.pdf_font_leading if cfg.pdf_font_leading else cfg.pdf_font_size * 1.3
    body_style = ParagraphStyle(
        name="Body",
        fontName=font_name,
        fontSize=cfg.pdf_font_size,
        leading=leading,
        alignment=TA_JUSTIFY,
        spaceAfter=cfg.pdf_font_size * 0.4,
    )
    heading_style = ParagraphStyle(
        name="Heading",
        fontName=font_name,
        fontSize=cfg.pdf_font_size + 4,
        leading=(cfg.pdf_font_size + 4) * 1.2,
        alignment=TA_LEFT,
        spaceAfter=max(cfg.pdf_font_size * 1.5, 18),
        spaceBefore=cfg.pdf_font_size * 0.5,
    )

    text = md_path.read_text(encoding="utf-8")
    paragraphs = normalize_markdown_for_pdf(text)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=cfg.pdf_margin,
        rightMargin=cfg.pdf_margin,
        topMargin=cfg.pdf_margin,
        bottomMargin=cfg.pdf_margin,
        title=title or md_path.stem,
        author=cfg.pdf_author or "",
    )
    if getattr(doc, "_info", None) is not None:  # pragma: no cover - metadata optional
        try:
            doc._info.language = cfg.pdf_language or "pt-BR"
        except Exception:
            pass

    story = []

    if title:
        story.append(Paragraph(_inline_markdown_to_html(title), heading_style))
        story.append(Spacer(1, cfg.pdf_font_size * 0.6))

    heading_keywords = (
        "prólogo",
        "prologo",
        "capítulo",
        "capitulo",
        "epílogo",
        "epilogo",
        "interlúdio",
        "interludio",
    )

    for para in paragraphs:
        if not para.strip():
            continue
        raw = para.strip()
        lower = raw.lower()
        is_heading = False
        for kw in heading_keywords:
            if lower == kw or (lower.startswith(kw) and len(raw) <= 60):
                is_heading = True
                break
        html = _inline_markdown_to_html(raw)
        if is_heading:
            story.append(Paragraph(html, heading_style))
            story.append(Spacer(1, cfg.pdf_font_size * 0.8))
        else:
            story.append(Paragraph(html, body_style))
            story.append(Spacer(1, cfg.pdf_font_size * 0.3))

    doc.build(story)
