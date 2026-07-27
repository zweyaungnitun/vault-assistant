from vault_assistant.config import Config
from vault_assistant.ingest import ingest_paths
from vault_assistant.qa import NOT_FOUND, answer_question
from vault_assistant.vectors import VectorIndex


def _seed(tmp_path, conn, client):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "policy.txt").write_text(
        "Vacation policy: employees receive 25 days of paid leave per year. " * 8
    )
    ingest_paths(conn, client, [folder])


def test_answer_with_citations_maps_to_real_files(tmp_path, conn, client):
    _seed(tmp_path, conn, client)
    client.chat_response = "Employees get 25 days of paid leave [1]."
    cfg = Config(data_dir=tmp_path)
    result = answer_question(conn, VectorIndex(conn), client, "how many vacation days?", cfg)
    assert "25 days" in result.answer
    assert result.sources
    assert all(s.filename == "policy.txt" for s in result.sources)


def test_not_found_returns_no_sources(tmp_path, conn, client):
    _seed(tmp_path, conn, client)
    client.chat_response = NOT_FOUND
    cfg = Config(data_dir=tmp_path)
    result = answer_question(conn, VectorIndex(conn), client, "what is the moon made of?", cfg)
    assert result.answer == NOT_FOUND
    assert result.sources == []


def test_empty_corpus_short_circuits(tmp_path, conn, client):
    cfg = Config(data_dir=tmp_path)
    result = answer_question(conn, VectorIndex(conn), client, "anything?", cfg)
    assert result.answer == NOT_FOUND
    assert result.sources == []
    assert client.chat_calls == []  # no model call without context
