import re

import numpy as np
import pytest

from vault_assistant.chunking import chunk_text
from vault_assistant.summarize import DIRECT_TOKEN_LIMIT, document_text, summarize_text


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _insert_doc(conn, text: str) -> int:
    conn.execute(
        "INSERT INTO documents(path, filename, content_hash, mtime, size, status, ingested_at) "
        "VALUES('/x/doc.txt', 'doc.txt', 'h', 0, 0, 'ok', 'now')"
    )
    doc_id = conn.execute("SELECT id FROM documents WHERE path = '/x/doc.txt'").fetchone()["id"]
    chunks = chunk_text(text, target_tokens=100)
    conn.executemany(
        "INSERT INTO chunks(doc_id, chunk_idx, text, char_start, char_end, page, embedding) "
        "VALUES(?, ?, ?, ?, ?, NULL, ?)",
        [
            (doc_id, i, c.text, c.char_start, c.char_end, np.zeros(4, dtype=np.float32).tobytes())
            for i, c in enumerate(chunks)
        ],
    )
    conn.commit()
    return doc_id


def test_document_text_reconstructs_despite_overlap(conn):
    text = "\n\n".join(f"Paragraph {i} with unique marker XK{i:03d} in it. " * 3 for i in range(30))
    doc_id = _insert_doc(conn, text)
    rebuilt = document_text(conn, doc_id)
    assert _normalize(rebuilt) == _normalize(text)
    # overlap must not duplicate content
    assert rebuilt.count("XK015") == text.count("XK015")


def test_document_text_missing_doc_raises(conn):
    with pytest.raises(ValueError):
        document_text(conn, 999)


def test_short_text_single_model_call(client):
    client.chat_response = "A tidy summary."
    out = summarize_text(client, "Short text about a thing.", "paragraph")
    assert out == "A tidy summary."
    assert len(client.chat_calls) == 1


def test_long_text_uses_map_reduce(client):
    client.chat_response = "Partial or final summary."
    long_text = "\n\n".join("A paragraph with plenty of words in it. " * 10 for _ in range(400))
    assert len(long_text) > DIRECT_TOKEN_LIMIT * 4
    out = summarize_text(client, long_text, "bullets")
    assert out == "Partial or final summary."
    assert len(client.chat_calls) >= 3  # >=2 map calls + 1 reduce call


def test_unknown_mode_rejected(client):
    with pytest.raises(ValueError):
        summarize_text(client, "text", "haiku")
