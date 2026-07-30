import pytest

from vault_assistant import permissions


def test_default_level_with_no_folders_registered(tmp_path, conn):
    assert permissions.resolve_access_level(conn, str(tmp_path / "anything.txt")) == "edit"


def test_exact_and_nested_path_match(tmp_path, conn):
    folder = tmp_path / "docs"
    folder.mkdir()
    permissions.set_folder(conn, str(folder), "readonly")

    assert permissions.resolve_access_level(conn, str(folder)) == "readonly"
    assert permissions.resolve_access_level(conn, str(folder / "a.txt")) == "readonly"
    assert permissions.resolve_access_level(conn, str(folder / "sub" / "b.txt")) == "readonly"
    # a sibling folder is unaffected
    assert permissions.resolve_access_level(conn, str(tmp_path / "other" / "c.txt")) == "edit"


def test_longest_prefix_wins(tmp_path, conn):
    parent = tmp_path / "docs"
    child = parent / "private"
    child.mkdir(parents=True)
    permissions.set_folder(conn, str(parent), "edit")
    permissions.set_folder(conn, str(child), "no_access")

    assert permissions.resolve_access_level(conn, str(parent / "a.txt")) == "edit"
    assert permissions.resolve_access_level(conn, str(child / "secret.txt")) == "no_access"


def test_set_folder_rejects_unknown_level(tmp_path, conn):
    with pytest.raises(ValueError, match="unknown access_level"):
        permissions.set_folder(conn, str(tmp_path), "superuser")


def test_set_folder_is_idempotent_upsert(tmp_path, conn):
    folder = tmp_path / "docs"
    folder.mkdir()
    a = permissions.set_folder(conn, str(folder), "readonly")
    b = permissions.set_folder(conn, str(folder), "edit")

    assert a.id == b.id
    assert b.access_level == "edit"
    assert len(permissions.list_folders(conn)) == 1


def test_no_access_purges_existing_documents(tmp_path, conn):
    folder = tmp_path / "docs"
    folder.mkdir()
    now = "2026-07-14T10:00:00"
    conn.execute(
        "INSERT INTO documents(path, filename, content_hash, mtime, size, status, ingested_at) "
        "VALUES (?, 'a.txt', 'x', 0, 0, 'ok', ?)",
        (str(folder / "a.txt"), now),
    )
    conn.execute(
        "INSERT INTO documents(path, filename, content_hash, mtime, size, status, ingested_at) "
        "VALUES (?, 'b.txt', 'x', 0, 0, 'ok', ?)",
        (str(tmp_path / "other" / "b.txt"), now),
    )
    conn.commit()

    permissions.set_folder(conn, str(folder), "no_access")

    remaining = [r["filename"] for r in conn.execute("SELECT filename FROM documents").fetchall()]
    assert remaining == ["b.txt"]


def test_remove_folder_missing_id_raises(conn):
    with pytest.raises(ValueError, match="no folder permission"):
        permissions.remove_folder(conn, 999)


def test_get_folder_missing_id_raises(conn):
    with pytest.raises(ValueError, match="no folder permission"):
        permissions.get_folder(conn, 999)
