from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from pypdf import PdfReader
from docx import Document as DocxDocument


@dataclass
class LoadedPage:
    text: str
    page_number: Optional[int] = None  # 1-based for PDFs


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s.strip()


def load_document(file_path: str) -> List[LoadedPage]:
    ext = os.path.splitext(file_path.lower())[1]

    if ext == ".pdf":
        return _load_pdf(file_path)
    if ext == ".docx":
        return _load_docx(file_path)
    if ext in (".txt", ".md"):
        return _load_text(file_path)

    raise ValueError(f"Unsupported file type: {ext}")


def _load_pdf(file_path: str) -> List[LoadedPage]:
    reader = PdfReader(file_path)
    pages: List[LoadedPage] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = _clean_text(text)
        if text:
            pages.append(LoadedPage(text=text, page_number=i))
        else:
            pages.append(LoadedPage(text="", page_number=i))
    return pages


def _load_docx(file_path: str) -> List[LoadedPage]:
    doc = DocxDocument(file_path)
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    full = _clean_text("\n".join(parts))
    return [LoadedPage(text=full, page_number=None)]


def _load_text(file_path: str) -> List[LoadedPage]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    txt = _clean_text(txt)
    return [LoadedPage(text=txt, page_number=None)]
