from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    ollama_url: str
    ollama_model: str
    collection_name: str
    top_k: int
    chunk_chars: int
    overlap_chars: int
    max_chunks_per_file: int

    data_raw_dir: str
    data_processed_dir: str
    manifest_path: str


def get_settings() -> Settings:
    load_dotenv()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw_docs")
    processed_dir = os.path.join(data_dir, "processed")
    manifest_path = os.path.join(processed_dir, "index_manifest.json")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    def getenv_int(key: str, default: int) -> int:
        val = os.getenv(key)
        if val is None or val.strip() == "":
            return default
        try:
            return int(val)
        except ValueError:
            return default

    return Settings(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "phi3"),
        collection_name=os.getenv("COLLECTION_NAME", "campus_rag"),
        top_k=getenv_int("TOP_K", 5),
        chunk_chars=getenv_int("CHUNK_CHARS", 3000),
        overlap_chars=getenv_int("OVERLAP_CHARS", 400),
        max_chunks_per_file=getenv_int("MAX_CHUNKS_PER_FILE", 1200),
        data_raw_dir=raw_dir,
        data_processed_dir=processed_dir,
        manifest_path=manifest_path,
    )