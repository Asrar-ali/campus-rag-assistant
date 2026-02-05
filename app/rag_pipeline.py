from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.settings import Settings
from app.loaders import load_document
from app.chunking import chunk_pages
from app.embeddings import Embedder
from app.qdrant_store import QdrantVectorStore
from app.ollama_client import OllamaClient
from app.prompts import build_rag_prompt


def sanitize_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9._ -]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        return "uploaded_file"
    return name[:180]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_chunk_id(file_hash: str, chunk_index: int) -> str:
    raw = f"{file_hash}:{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    if not os.path.exists(manifest_path):
        return {"files": {}}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"files": {}}


def save_manifest(manifest_path: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


@dataclass
class IndexResult:
    indexed: bool
    filename: str
    file_hash: str
    chunk_count: int
    message: str


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.embedder = Embedder(model_name="BAAI/bge-m3")
        self.store = QdrantVectorStore(url=self.s.qdrant_url, collection_name=self.s.collection_name)
        self.ollama = OllamaClient(base_url=self.s.ollama_url)

    def index_file(self, file_path: str, original_name: Optional[str] = None) -> IndexResult:
        filename = sanitize_filename(original_name or os.path.basename(file_path))
        file_hash = sha256_file(file_path)

        manifest = load_manifest(self.s.manifest_path)
        files = manifest.setdefault("files", {})
        existing = files.get(filename)

        if existing and existing.get("file_hash") == file_hash:
            return IndexResult(
                indexed=False,
                filename=filename,
                file_hash=file_hash,
                chunk_count=int(existing.get("chunk_count", 0)),
                message="Skipped (already indexed, unchanged).",
            )

        loaded_pages = load_document(file_path)
        texts = [p.text for p in loaded_pages]
        page_numbers = [p.page_number for p in loaded_pages]

        chunks = chunk_pages(
            pages=texts,
            page_numbers=page_numbers,
            chunk_chars=self.s.chunk_chars,
            overlap_chars=self.s.overlap_chars,
            max_chunks=self.s.max_chunks_per_file,
        )

        if not chunks:
            files[filename] = {
                "file_hash": file_hash,
                "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "chunk_count": 0,
            }
            save_manifest(self.s.manifest_path, manifest)
            return IndexResult(True, filename, file_hash, 0, "Indexed (0 chunks, no text found).")

        # embeddings
        chunk_texts = [c.text for c in chunks]
        vectors = self.embedder.embed_texts(chunk_texts)
        vector_size = int(vectors.shape[1])

        # qdrant collection
        self.store.ensure_collection(vector_size=vector_size)

        points = []
        for c, vec in zip(chunks, vectors):
            cid = stable_chunk_id(file_hash=file_hash, chunk_index=c.chunk_index)
            payload = {
                "filename": filename,
                "file_hash": file_hash,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "chunk_id": cid,
                "text": c.text,
            }
            points.append((cid, vec, payload))

        self.store.upsert_points(points)

        files[filename] = {
            "file_hash": file_hash,
            "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_count": len(chunks),
        }
        save_manifest(self.s.manifest_path, manifest)

        return IndexResult(True, filename, file_hash, len(chunks), "Indexed successfully.")

    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        k = top_k or self.s.top_k
        t0 = time.time()

        qvec = self.embedder.embed_query(question)
        hits = self.store.search(qvec, limit=k)
        retrieval_s = time.time() - t0

        contexts = []
        for h in hits:
            payload = h.get("payload", {}) or {}
            contexts.append(
                {
                    "chunk_id": payload.get("chunk_id", h.get("id")),
                    "filename": payload.get("filename", "unknown"),
                    "page_number": payload.get("page_number"),
                    "text": payload.get("text", ""),
                    "score": h.get("score", 0.0),
                }
            )

        prompt = build_rag_prompt(question, contexts)

        t1 = time.time()
        answer = self.ollama.generate(model=self.s.ollama_model, prompt=prompt, timeout_s=90)
        generation_s = time.time() - t1

        return {
            "answer": answer,
            "sources": contexts,
            "timing": {
                "retrieval_seconds": retrieval_s,
                "generation_seconds": generation_s,
                "total_seconds": retrieval_s + generation_s,
            },
        }


def save_uploaded_file(upload_bytes: bytes, filename: str, raw_dir: str) -> str:
    safe = sanitize_filename(filename)
    path = os.path.join(raw_dir, safe)
    # prevent overwrite collisions
    if os.path.exists(path):
        base, ext = os.path.splitext(safe)
        i = 2
        while True:
            candidate = os.path.join(raw_dir, f"{base}_{i}{ext}")
            if not os.path.exists(candidate):
                path = candidate
                break
            i += 1

    with open(path, "wb") as f:
        f.write(upload_bytes)
    return path
