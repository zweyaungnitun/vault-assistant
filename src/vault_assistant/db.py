"""SQLite storage: documents, chunks (with embeddings as float32 BLOBs),
FTS5 keyword index, reminders, and metadata.

Chunks store char offsets and page numbers so Phase 3 inline citations can be
added without re-ingesting existing corpora. The FTS index is kept in sync
with the chunks table via triggers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    filename     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    mtime        REAL NOT NULL,
    size         INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'ok',
    error        TEXT,
    ingested_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_idx  INTEGER NOT NULL,
    text       TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    page       INTEGER,
    embedding  BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TABLE IF NOT EXISTS reminders (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    due_at      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    source_text TEXT
);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    if isinstance(db_path, Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    if get_meta(conn, "schema_version") is None:
        set_meta(conn, "schema_version", "1")
    return conn


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


def get_generation(conn: sqlite3.Connection) -> int:
    return int(get_meta(conn, "ingest_generation", "0"))


def bump_generation(conn: sqlite3.Connection) -> None:
    set_meta(conn, "ingest_generation", str(get_generation(conn) + 1))


def list_documents(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT d.id, d.path, d.filename, d.status, d.error, d.ingested_at, "
        "       COUNT(c.id) AS chunk_count "
        "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
        "GROUP BY d.id ORDER BY d.filename"
    ).fetchall()
    return [dict(r) for r in rows]
