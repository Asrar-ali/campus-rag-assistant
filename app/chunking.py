from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    page_number: Optional[int]  # 1-based for PDFs, None otherwise
    chunk_index: int


def chunk_pages(
    pages: List[str],
    page_numbers: List[Optional[int]],
    chunk_chars: int,
    overlap_chars: int,
    max_chunks: int,
) -> List[Chunk]:
    if chunk_chars <= 200:
        raise ValueError("chunk_chars too small; use >= 200")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    out: List[Chunk] = []
    chunk_index = 0

    for text, pg in zip(pages, page_numbers):
        t = (text or "").strip()
        if not t:
            continue

        start = 0
        n = len(t)

        while start < n:
            end = min(start + chunk_chars, n)
            chunk_text = t[start:end].strip()
            if chunk_text:
                out.append(Chunk(text=chunk_text, page_number=pg, chunk_index=chunk_index))
                chunk_index += 1
                if len(out) >= max_chunks:
                    return out

            if end == n:
                break
            start = max(0, end - overlap_chars)

    return out
