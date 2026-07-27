import os

from vault_assistant import db
from vault_assistant.ingest import ingest_paths


def _write(path, text, mtime=None):
    path.write_text(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_add_skip_update_remove(tmp_path, conn, client):
    folder = tmp_path / "docs"
    folder.mkdir()
    _write(folder / "a.txt", "Alpha notes about apples and orchards. " * 20, mtime=1000)
    _write(folder / "b.md", "# Beta\n\nBravo notes about bananas and shipping. " * 20, mtime=1000)

    r = ingest_paths(conn, client, [folder])
    assert (r.added, r.updated, r.skipped, r.removed) == (2, 0, 0, 0)
    assert not r.failed
    gen1 = db.get_generation(conn)

    # unchanged rescan: everything skipped, generation stays put
    r2 = ingest_paths(conn, client, [folder])
    assert (r2.added, r2.updated, r2.skipped) == (0, 0, 2)
    assert db.get_generation(conn) == gen1

    # touched but identical content: stats refresh, no re-embed
    _write(folder / "a.txt", "Alpha notes about apples and orchards. " * 20, mtime=2000)
    r3 = ingest_paths(conn, client, [folder])
    assert (r3.added, r3.updated, r3.skipped) == (0, 0, 2)

    # content change: re-ingested
    _write(folder / "a.txt", "Rewritten alpha document about cider. " * 25, mtime=3000)
    r4 = ingest_paths(conn, client, [folder])
    assert r4.updated == 1
    assert db.get_generation(conn) > gen1

    # deletion: pruned on next scan
    (folder / "b.md").unlink()
    r5 = ingest_paths(conn, client, [folder])
    assert r5.removed == 1
    docs = db.list_documents(conn)
    assert [d["filename"] for d in docs] == ["a.txt"]


def test_corrupt_file_logged_not_fatal(tmp_path, conn, client):
    folder = tmp_path / "docs"
    folder.mkdir()
    _write(folder / "good.txt", "Perfectly fine text. " * 30)
    (folder / "bad.pdf").write_bytes(b"garbage, not a pdf")

    r = ingest_paths(conn, client, [folder])
    assert r.added == 1
    assert len(r.failed) == 1
    assert "bad.pdf" in r.failed[0][0]

    docs = {d["filename"]: d for d in db.list_documents(conn)}
    assert docs["good.txt"]["status"] == "ok"
    assert docs["bad.pdf"]["status"] == "error"
    assert docs["bad.pdf"]["error"]
    assert docs["bad.pdf"]["chunk_count"] == 0


def test_hidden_files_skipped(tmp_path, conn, client):
    folder = tmp_path / "docs"
    (folder / ".hidden").mkdir(parents=True)
    _write(folder / ".hidden" / "secret.txt", "should not be ingested " * 10)
    _write(folder / "visible.txt", "should be ingested " * 10)

    r = ingest_paths(conn, client, [folder])
    assert r.added == 1
    assert db.list_documents(conn)[0]["filename"] == "visible.txt"


def test_chunks_and_fts_stay_in_sync(tmp_path, conn, client):
    folder = tmp_path / "docs"
    folder.mkdir()
    _write(folder / "a.txt", "Distinctive zanzibar content. " * 20)
    ingest_paths(conn, client, [folder])

    n_chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    n_fts = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'zanzibar'"
    ).fetchone()["n"]
    assert n_chunks == n_fts > 0

    (folder / "a.txt").unlink()
    ingest_paths(conn, client, [folder])
    assert conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 0
    assert (
        conn.execute("SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'zanzibar'")
        .fetchone()["n"]
        == 0
    )


def test_embedding_model_mismatch_detected(tmp_path, conn, client):
    folder = tmp_path / "docs"
    folder.mkdir()
    _write(folder / "a.txt", "Some content here. " * 20)
    ingest_paths(conn, client, [folder])

    client.embed_model = "different-model"
    _write(folder / "a.txt", "Changed content now. " * 20)
    r = ingest_paths(conn, client, [folder], force=True)
    assert len(r.failed) == 1
    assert "mismatch" in r.failed[0][1]
