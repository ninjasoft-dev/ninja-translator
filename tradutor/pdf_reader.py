"""
Leitura de PDFs com PyMuPDF (fitz) para extração de texto.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyMuPDF (fitz) não está instalado. Instale com `pip install PyMuPDF` ou `pip install -r requirements.txt`."
    ) from exc


def extract_pdf_text(pdf_path: Union[str, Path], logger: Optional[logging.Logger] = None) -> str:
    """
    Extrai texto de um PDF usando PyMuPDF (fitz).
    Retorna o texto concatenado de todas as páginas.
    """
    path = Path(pdf_path)
    doc = fitz.open(str(path))

    chunks: list[str] = []

    for page in doc:
        page_text = page.get_text("text") or ""
        page_text = page_text.replace("\r\n", "\n").replace("\r", "\n")
        page_text = page_text.strip()
        if page_text:
            chunks.append(page_text)

    doc.close()

    text = "\n\n".join(chunks).strip()

    if logger is not None:
        logger.debug("PDF %s extraído com %d caracteres (PyMuPDF)", path.name, len(text))

    return text
