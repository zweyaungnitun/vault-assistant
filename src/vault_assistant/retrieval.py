"""Hybrid retrieval: vector similarity + FTS5 BM25, fused with reciprocal
rank fusion (RRF), deduped by chunk id."""

from __future__ import annotations

import re
from dataclasses import dataclass
from sqlite3 import Connection

from .ollama_client import QUERY_PREFIX, OllamaClient
from .vectors import VectorIndex

RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_id: int
    filename: str
    path: str
    chunk_idx: int
    text: str
    char_start: int
    char_end: int
    page: int | None
    score: float


def build_fts_query(question: str) -> str | None:
    """Quote each word token and OR them — a recall-oriented keyword pass that
    can't be broken by FTS5 operator syntax in user input."""
    tokens = re.findall(r"\w+", question.lower())
    tokens = [t for t in tokens if len(t) > 1]
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))


def keyword_search(conn: Connection, question: str, k: int = 8) -> list[int]:
    query = build_fts_query(question)
    if query is None:
        return []
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (query, k),
    ).fetchall()
    return [r["rowid"] for r in rows]


def rrf_merge(ranked_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def hybrid_search(
    conn: Connection,
    index: VectorIndex,
    client: OllamaClient,
    question: str,
    k_vec: int = 8,
    k_kw: int = 8,
    limit: int = 8,
    doc_ids: list[int] | None = None,
) -> list[RetrievedChunk]:
    query_vec = client.embed([QUERY_PREFIX + question])[0]
    vec_ids = [cid for cid, _ in index.search(query_vec, k=k_vec)]
    kw_ids = keyword_search(conn, question, k=k_kw)

    fused = rrf_merge([vec_ids, kw_ids])[:limit]
    if not fused:
        return []

    by_id = {cid: score for cid, score in fused}
    placeholders = ",".join("?" * len(by_id))
    
    # Build document filter if doc_ids provided
    doc_filter = ""
    params = list(by_id)
    if doc_ids:
        doc_placeholders = ",".join("?" * len(doc_ids))
        doc_filter = f" AND c.doc_id IN ({doc_placeholders})"
        params.extend(doc_ids)
    
    rows = conn.execute(
        f"""
        SELECT c.id, c.doc_id, c.chunk_idx, c.text, c.char_start, c.char_end, c.page,
               d.filename, d.path
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE c.id IN ({placeholders}){doc_filter}
        """,
        params,
    ).fetchall()
    chunks = [
        RetrievedChunk(
            chunk_id=r["id"],
            doc_id=r["doc_id"],
            filename=r["filename"],
            path=r["path"],
            chunk_idx=r["chunk_idx"],
            text=r["text"],
            char_start=r["char_start"],
            char_end=r["char_end"],
            page=r["page"],
            score=by_id[r["id"]],
        )
        for r in rows
    ]
    chunks.sort(key=lambda c: -c.score)
    return chunks
