"""Folder-level access control.

This is a single-user, offline, localhost-only tool — there are no accounts
or sessions. What "access control" means here is narrower and enforceable:
each watched folder carries an access level that gates what the assistant
itself may do with documents under it, resolved by longest-prefix match so a
subfolder can override its parent.

- ``no_access``: excluded from ingestion entirely; any previously ingested
  documents under the path are purged immediately when the level is set.
- ``readonly``: ingested and fully queryable (ask/summarize/actions/PII) but
  cannot be force re-ingested or deleted through the API.
- ``edit``: everything ``readonly`` allows, plus force re-ingest and delete.
  This is the default for any path with no explicit folder_permissions row,
  so documents ingested before this feature existed keep working unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from sqlite3 import Connection

LEVELS = ("readonly", "edit", "no_access")
DEFAULT_LEVEL = "edit"


@dataclass
class FolderPermission:
    id: int
    path: str
    access_level: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _row_to_permission(row) -> FolderPermission:
    return FolderPermission(
        id=row["id"],
        path=row["path"],
        access_level=row["access_level"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_folders(conn: Connection) -> list[FolderPermission]:
    rows = conn.execute(
        "SELECT id, path, access_level, created_at, updated_at "
        "FROM folder_permissions ORDER BY path"
    ).fetchall()
    return [_row_to_permission(r) for r in rows]


def get_folder(conn: Connection, folder_id: int) -> FolderPermission:
    row = conn.execute(
        "SELECT id, path, access_level, created_at, updated_at "
        "FROM folder_permissions WHERE id = ?",
        (folder_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"no folder permission with id {folder_id}")
    return _row_to_permission(row)


def set_folder(conn: Connection, path: str, access_level: str, now: str | None = None) -> FolderPermission:
    if access_level not in LEVELS:
        raise ValueError(f"unknown access_level {access_level!r}; expected one of {LEVELS}")
    norm = _normalize(path)
    now = now or datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO folder_permissions(path, access_level, created_at, updated_at)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            access_level = excluded.access_level,
            updated_at = excluded.updated_at
        """,
        (norm, access_level, now, now),
    )
    conn.commit()
    if access_level == "no_access":
        purge_path(conn, norm)
    row = conn.execute(
        "SELECT id, path, access_level, created_at, updated_at FROM folder_permissions WHERE path = ?",
        (norm,),
    ).fetchone()
    return _row_to_permission(row)


def remove_folder(conn: Connection, folder_id: int) -> None:
    row = conn.execute("SELECT id FROM folder_permissions WHERE id = ?", (folder_id,)).fetchone()
    if row is None:
        raise ValueError(f"no folder permission with id {folder_id}")
    conn.execute("DELETE FROM folder_permissions WHERE id = ?", (folder_id,))
    conn.commit()


def purge_path(conn: Connection, path: str) -> int:
    """Delete any ingested documents whose path is under `path`. Called when a
    folder is set to no_access so revoked access takes effect immediately,
    not just on the next scan."""
    norm = _normalize(path)
    prefix = norm.rstrip("/") + "/"
    rows = conn.execute(
        "SELECT id FROM documents WHERE path = ? OR path LIKE ?",
        (norm, prefix + "%"),
    ).fetchall()
    for r in rows:
        conn.execute("DELETE FROM documents WHERE id = ?", (r["id"],))
    conn.commit()
    return len(rows)


def resolve_access_level(conn: Connection, doc_path: str) -> str:
    """Longest matching folder_permissions path wins; no match defaults to
    DEFAULT_LEVEL so pre-existing documents/folders keep full access."""
    path = _normalize(doc_path)
    best_level = DEFAULT_LEVEL
    best_len = -1
    for row in conn.execute("SELECT path, access_level FROM folder_permissions").fetchall():
        folder = row["path"]
        if path == folder or path.startswith(folder.rstrip("/") + "/"):
            if len(folder) > best_len:
                best_len = len(folder)
                best_level = row["access_level"]
    return best_level
