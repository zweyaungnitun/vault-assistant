from vault_assistant.ingest import ingest_paths
from vault_assistant.retrieval import build_fts_query, hybrid_search, keyword_search, rrf_merge
from vault_assistant.vectors import VectorIndex


def test_build_fts_query_quotes_and_dedupes():
    q = build_fts_query('What is the "budget" for Q3? budget!')
    assert q == '"what" OR "is" OR "the" OR "budget" OR "for" OR "q3"'


def test_build_fts_query_empty():
    assert build_fts_query("???") is None
    assert build_fts_query("") is None


def test_rrf_merge_prefers_items_in_both_lists():
    merged = rrf_merge([[1, 2, 3], [3, 4, 5]])
    ids = [i for i, _ in merged]
    assert ids[0] == 3  # appears in both lists


def _seed(tmp_path, conn, client):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "fruit.txt").write_text(
        "Apples and pears grow in the orchard. The apple harvest happens in autumn. " * 10
    )
    (folder / "cars.txt").write_text(
        "The engine needs oil changes. Brake pads wear out over time on vehicles. " * 10
    )
    ingest_paths(conn, client, [folder])
    return folder


def test_keyword_search_finds_matching_chunk(tmp_path, conn, client):
    _seed(tmp_path, conn, client)
    ids = keyword_search(conn, "apple orchard harvest")
    assert ids
    row = conn.execute(
        "SELECT d.filename FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = ?",
        (ids[0],),
    ).fetchone()
    assert row["filename"] == "fruit.txt"


def test_hybrid_search_ranks_relevant_doc_first(tmp_path, conn, client):
    _seed(tmp_path, conn, client)
    index = VectorIndex(conn)
    results = hybrid_search(conn, index, client, "when is the apple harvest?")
    assert results
    assert results[0].filename == "fruit.txt"
    assert results[0].char_start >= 0 and results[0].char_end > results[0].char_start


def test_index_refreshes_after_reingest(tmp_path, conn, client):
    folder = _seed(tmp_path, conn, client)
    index = VectorIndex(conn)
    assert hybrid_search(conn, index, client, "apple harvest")
    # remove everything; index must notice the generation bump
    (folder / "fruit.txt").unlink()
    (folder / "cars.txt").unlink()
    ingest_paths(conn, client, [folder])
    assert hybrid_search(conn, index, client, "apple harvest") == []
