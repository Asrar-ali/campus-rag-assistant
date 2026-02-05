from __future__ import annotations

from typing import Dict, List


def build_rag_prompt(question: str, contexts: List[Dict]) -> str:
    # contexts item keys: text, filename, page_number, chunk_id, score
    context_blocks = []
    for c in contexts:
        fname = c.get("filename", "unknown")
        pg = c.get("page_number")
        chunk_id = c.get("chunk_id", "unknown")
        header = f"Source: {fname}"
        if pg is not None:
            header += f" | page: {pg}"
        header += f" | chunk_id: {chunk_id}"
        context_blocks.append(header + "\n" + (c.get("text", "")[:4000]))

    joined_context = "\n\n---\n\n".join(context_blocks) if context_blocks else "NO_CONTEXT"

    prompt = f"""You are a careful assistant. Answer the user's question using only the provided context.

Rules:
- Use only the context below. Do not use outside knowledge.
- If the answer is not clearly present in the context, reply exactly: "I don't know based on the provided documents."
- Keep the answer short and direct.
- Add citations in square brackets at the end of each sentence or paragraph.
  - For PDFs use: [filename p.X]
  - For non-PDF use: [filename]
- Do not invent page numbers or sources.

Context:
{joined_context}

Question:
{question}

Answer:
"""
    return prompt
