"""Incremental document ingestion.

Scan folders (or explicit files) for supported types; extract, chunk, embed,
and store. A file is re-processed only when its content hash changes — an
mtime+size match skips it without reading, and an unchanged hash refreshes
stats without re-embedding. Unreadable files are recorded with status='error'
and logged, never silently dropped. Documents whose files disappeared from a
scanned folder are pruned.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from sqlite3 import Connection

import numpy as np

from .chunking import chunk_text
from .db import bump_generation, get_meta, set_meta
from .extractors import SUPPORTED_EXTENSIONS, ExtractionError, extract
from .permissions import resolve_access_level
from .providers import LLMClient

logger = logging.getLogger("vault.ingest")


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0
    blocked: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        parts = (
            f"{self.added} added, {self.updated} updated, {self.skipped} unchanged, "
            f"{self.removed} removed, {self.blocked} blocked (no_access), "
            f"{len(self.failed)} failed"
        )
        return parts


def iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if (
                    f.is_file()
                    and f.suffix.lower() in SUPPORTED_EXTENSIONS
                    and not any(part.startswith(".") for part in f.relative_to(p).parts)
                ):
                    files.append(f)
    return files


def _verify_embedding_meta(conn: Connection, client: LLMClient, dim: int) -> None:
    stored_model = get_meta(conn, "embedding_model")
    stored_dim = get_meta(conn, "embedding_dim")
    stored_provider = get_meta(conn, "embedding_provider") or ""
    current_provider = getattr(client, "embed_provider_name", "")
    if stored_model is None:
        set_meta(conn, "embedding_model", client.embed_model)
        set_meta(conn, "embedding_dim", str(dim))
        set_meta(conn, "embedding_provider", current_provider)
        return
    if stored_model != client.embed_model or int(stored_dim or 0) != dim or stored_provider != current_provider:
        raise RuntimeError(
            f"embedding model mismatch: database was built with {stored_provider or 'unknown provider'}/"
            f"{stored_model} (dim {stored_dim}), current config uses {current_provider}/{client.embed_model} "
            f"(dim {dim}). Re-ingest from scratch with a fresh database, or restore the original "
            "provider/model."
        )


def ingest_paths(
    conn: Connection,
    client: LLMClient,
    paths: list[Path],
    force: bool = False,
    prune: bool = True,
) -> IngestReport:
    report = IngestReport()
    paths = [p.expanduser().resolve() for p in paths]
    files = iter_files(paths)

    accessible_files = []
    for f in files:
        if resolve_access_level(conn, str(f)) == "no_access":
            report.blocked += 1
        else:
            accessible_files.append(f)
    files = accessible_files

    seen: set[str] = set()

    for f in files:
        seen.add(str(f))
        try:
            changed = _ingest_file(conn, client, f, force=force, report=report)
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop the scan
            logger.error("failed to ingest %s: %s", f, exc)
            report.failed.append((str(f), str(exc)))
            _record_failure(conn, f, str(exc))
        else:
            if not changed:
                report.skipped += 1

    if prune:
        scanned_dirs = [str(p) for p in paths if p.is_dir()]
        for row in conn.execute("SELECT id, path FROM documents").fetchall():
            under_scan = any(row["path"].startswith(d + "/") for d in scanned_dirs)
            if under_scan and row["path"] not in seen and not Path(row["path"]).exists():
                conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
                report.removed += 1
        conn.commit()

    if report.added or report.updated or report.removed:
        bump_generation(conn)
    logger.info("ingest: %s", report.summary())
    return report


def _record_failure(conn: Connection, path: Path, error: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    doc_id = _upsert_document(conn, path, content_hash="", status="error", error=error, now=now)
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.commit()


def _upsert_document(
    conn: Connection,
    path: Path,
    content_hash: str,
    status: str,
    error: str | None,
    now: str,
) -> int:
    stat = path.stat() if path.exists() else None
    mtime = stat.st_mtime if stat else 0.0
    size = stat.st_size if stat else 0
    conn.execute(
        """
        INSERT INTO documents(path, filename, content_hash, mtime, size, status, error, ingested_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            content_hash = excluded.content_hash,
            mtime = excluded.mtime,
            size = excluded.size,
            status = excluded.status,
            error = excluded.error,
            ingested_at = excluded.ingested_at
        """,
        (str(path), path.name, content_hash, mtime, size, status, error, now),
    )
    row = conn.execute("SELECT id FROM documents WHERE path = ?", (str(path),)).fetchone()
    return row["id"]


def _ingest_file(
    conn: Connection,
    client: LLMClient,
    path: Path,
    force: bool,
    report: IngestReport,
) -> bool:
    """Returns True if the file was (re-)ingested, False if skipped as unchanged."""
    stat = path.stat()
    row = conn.execute("SELECT * FROM documents WHERE path = ?", (str(path),)).fetchone()

    if row and not force and row["status"] == "ok":
        if row["mtime"] == stat.st_mtime and row["size"] == stat.st_size:
            return False

    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    now = datetime.now().isoformat(timespec="seconds")

    if row and not force and row["content_hash"] == content_hash and row["status"] == "ok":
        # Touched but unchanged: refresh stats, keep existing chunks/embeddings.
        conn.execute(
            "UPDATE documents SET mtime = ?, size = ? WHERE id = ?",
            (stat.st_mtime, stat.st_size, row["id"]),
        )
        conn.commit()
        return False

    try:
        doc = extract(path)
    except ExtractionError as exc:
        raise exc

    chunks = chunk_text(doc.text, doc.pages)
    if not chunks:
        raise ExtractionError(f"{path.name}: no content after chunking")

    vectors = client.embed([client.doc_prefix + c.text for c in chunks])
    dim = len(vectors[0])
    _verify_embedding_meta(conn, client, dim)

    doc_id = _upsert_document(conn, path, content_hash, status="ok", error=None, now=now)
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.executemany(
        "INSERT INTO chunks(doc_id, chunk_idx, text, char_start, char_end, page, embedding) "
        "VALUES(?, ?, ?, ?, ?, ?, ?)",
        [
            (
                doc_id,
                idx,
                c.text,
                c.char_start,
                c.char_end,
                c.page,
                np.asarray(vec, dtype=np.float32).tobytes(),
            )
            for idx, (c, vec) in enumerate(zip(chunks, vectors))
        ],
    )
    conn.commit()

    if row:
        report.updated += 1
        logger.info("re-ingested %s (%d chunks)", path.name, len(chunks))
    else:
        report.added += 1
        logger.info("ingested %s (%d chunks)", path.name, len(chunks))
    return True
