from __future__ import annotations

import os
import streamlit as st

from app.settings import get_settings
from app.rag_pipeline import RAGPipeline, save_uploaded_file


st.set_page_config(page_title="Campus RAG Assistant", layout="wide")


def main() -> None:
    s = get_settings()
    rag = RAGPipeline(s)

    st.title("Campus RAG Assistant (Local RAG with citations)")

    with st.sidebar:
        st.write("Settings")
        st.write(f"Qdrant: {s.qdrant_url}")
        st.write(f"Ollama: {s.ollama_url}")
        st.write(f"Model: {s.ollama_model}")
        st.write(f"Collection: {s.collection_name}")
        st.write(f"Top K: {s.top_k}")
        st.write(f"Chunk chars: {s.chunk_chars}")
        st.write(f"Overlap chars: {s.overlap_chars}")

        st.divider()

        st.write("Upload documents")
        uploads = st.file_uploader(
            "Upload PDFs/DOCX/TXT/MD",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )

        if st.button("Save uploads to data/raw_docs"):
            if not uploads:
                st.warning("No files uploaded.")
            else:
                saved = 0
                for f in uploads:
                    save_uploaded_file(f.getvalue(), f.name, s.data_raw_dir)
                    saved += 1
                st.success(f"Saved {saved} file(s) to data/raw_docs/")

        st.divider()

        if st.button("Index all files in data/raw_docs"):
            files = sorted(os.listdir(s.data_raw_dir))
            if not files:
                st.warning("No files found in data/raw_docs/")
            else:
                progress = st.progress(0)
                indexed_count = 0
                skipped_count = 0
                for i, fn in enumerate(files, start=1):
                    fp = os.path.join(s.data_raw_dir, fn)
                    try:
                        res = rag.index_file(fp, original_name=fn)
                        if res.indexed:
                            indexed_count += 1
                        else:
                            skipped_count += 1
                    except Exception as e:
                        st.error(f"Failed to index {fn}: {e}")

                    progress.progress(i / len(files))

                st.success(f"Index complete. Indexed: {indexed_count}, skipped: {skipped_count}")

    st.subheader("Ask a question")
    question = st.text_input("Question", placeholder="e.g., What is the grading policy for Assignment 2?")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    col_a, col_b = st.columns([1, 1])

    if st.button("Ask"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            try:
                result = rag.query(question.strip())
                st.session_state.chat.append(
                    {"q": question.strip(), "a": result["answer"], "sources": result["sources"], "timing": result["timing"]}
                )
            except Exception as e:
                st.error(str(e))

    with col_a:
        st.write("Conversation")
        for item in reversed(st.session_state.chat[-10:]):
            st.write(f"Q: {item['q']}")
            st.write(item["a"])
            st.caption(
                f"timing: retrieval {item['timing']['retrieval_seconds']:.2f}s | generation {item['timing']['generation_seconds']:.2f}s"
            )
            st.divider()

    with col_b:
        st.write("Sources (latest answer)")
        if st.session_state.chat:
            latest = st.session_state.chat[-1]
            sources = latest.get("sources", [])
            if not sources:
                st.info("No sources returned.")
            else:
                for ssrc in sources:
                    fname = ssrc.get("filename", "unknown")
                    pg = ssrc.get("page_number")
                    score = ssrc.get("score", 0.0)
                    chunk_id = ssrc.get("chunk_id", "")
                    title = f"{fname}"
                    if pg is not None:
                        title += f" (p.{pg})"
                    title += f" | score {score:.4f} | {chunk_id}"
                    with st.expander(title):
                        st.write((ssrc.get("text") or "")[:4000])
        else:
            st.info("Ask a question to see sources.")


if __name__ == "__main__":
    main()
