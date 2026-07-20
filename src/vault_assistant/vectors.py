"""In-process vector search.

Embeddings live as float32 BLOBs in the chunks table; this index loads them
into a normalized numpy matrix and does brute-force cosine search. At Phase 1
scale (hundreds of documents, low tens of thousands of chunks) a single
matrix-vector product is single-digit milliseconds — faster than an ANN index
and with zero extra infrastructure. The matrix is cached and reloaded only
when the ingest generation counter changes.

(sqlite-vec was the spec's suggestion, but this Python build ships sqlite3
without loadable-extension support; this module is the swap point if that
changes.)
"""

from __future__ import annotations

from sqlite3 import Connection

import numpy as np

from .db import get_generation


class VectorIndex:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn
        self._generation: int | None = None
        self._ids: np.ndarray | None = None
        self._matrix: np.ndarray | None = None

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            self._ids = np.empty(0, dtype=np.int64)
            self._matrix = None
            return
        ids = np.array([r["id"] for r in rows], dtype=np.int64)
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._ids = ids
        self._matrix = matrix / norms

    def _ensure_fresh(self) -> None:
        gen = get_generation(self._conn)
        if gen != self._generation:
            self._load()
            self._generation = gen

    def search(self, query_vec: list[float], k: int = 8) -> list[tuple[int, float]]:
        """Return [(chunk_id, cosine_score)] for the k nearest chunks."""
        self._ensure_fresh()
        if self._matrix is None or len(self._ids) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        scores = self._matrix @ (q / norm)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self._ids[i]), float(scores[i])) for i in top]
