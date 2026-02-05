from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


class QdrantVectorStore:
    def __init__(self, url: str, collection_name: str) -> None:
        self.url = url
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)

    def ensure_collection(self, vector_size: int) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def upsert_points(self, points: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> None:
        # points: (id, vector, payload)
        qpoints = []
        for pid, vec, payload in points:
            qpoints.append(
                qmodels.PointStruct(
                    id=pid,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=qpoints)

    def search(self, query_vec: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec.tolist(),
            limit=limit,
            with_payload=True,
        )
        results: List[Dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            results.append(
                {
                    "id": str(h.id),
                    "score": float(h.score),
                    "payload": payload,
                }
            )
        return results

    def delete_by_file_hash(self, file_hash: str) -> None:
        # Optional: not used by default; helpful if you want reindex ability later
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="file_hash",
                            match=qmodels.MatchValue(value=file_hash),
                        )
                    ]
                )
            ),
        )
