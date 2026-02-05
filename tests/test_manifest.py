import os
import tempfile
from app.rag_pipeline import load_manifest, save_manifest

def test_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "index_manifest.json")
        m = {"files": {"x.pdf": {"file_hash": "abc", "chunk_count": 3}}}
        save_manifest(path, m)
        m2 = load_manifest(path)
        assert m2["files"]["x.pdf"]["chunk_count"] == 3
