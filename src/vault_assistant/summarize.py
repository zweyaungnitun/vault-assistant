"""Summarization: one-line, paragraph, or bullet modes for 1-3 documents.

Document text is reconstructed exactly from stored chunk offsets (chunks
overlap, so naive joining would duplicate text). Documents longer than the
context budget are map-summarized in pieces and reduced.
"""

from __future__ import annotations

from sqlite3 import Connection

from .chunking import approx_tokens, chunk_text
from .providers import LLMClient

MODES = ("one_line", "paragraph", "bullets")

# Keep map/reduce inputs comfortably inside the model's num_ctx (8192 default).
DIRECT_TOKEN_LIMIT = 5000

_MODE_INSTRUCTIONS = {
    "one_line": "Summarize in exactly one sentence of at most 30 words.",
    "paragraph": "Summarize in one paragraph of roughly 80-150 words.",
    "bullets": "Summarize as 3-7 short bullet points (markdown '-' bullets).",
}

SYSTEM_PROMPT = """You summarize documents faithfully. Use only facts present in the provided text — never add outside information, opinions, or invented details. Output only the summary itself with no preamble."""


def document_text(conn: Connection, doc_id: int) -> str:
    """Reconstruct document text from chunk spans, dropping overlap."""
    rows = conn.execute(
        "SELECT text, char_start, char_end FROM chunks WHERE doc_id = ? ORDER BY char_start",
        (doc_id,),
    ).fetchall()
    if not rows:
        raise ValueError(f"document {doc_id} has no stored content")
    parts: list[str] = []
    prev_end = 0
    for r in rows:
        start, end, text = r["char_start"], r["char_end"], r["text"]
        if end <= prev_end:
            continue
        fresh = text[max(0, prev_end - start):]
        if parts and start > prev_end:
            parts.append("\n\n")
        parts.append(fresh)
        prev_end = end
    return "".join(parts)


def summarize_text(client: LLMClient, text: str, mode: str = "paragraph") -> str:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    instruction = _MODE_INSTRUCTIONS[mode]

    if approx_tokens(text) <= DIRECT_TOKEN_LIMIT:
        return client.chat(SYSTEM_PROMPT, f"{instruction}\n\nText:\n\n{text}", temperature=0.3)

    # Map: summarize each oversized piece to a paragraph; reduce: final pass in
    # the requested mode over the intermediate summaries.
    pieces = chunk_text(text, target_tokens=DIRECT_TOKEN_LIMIT, overlap_ratio=0.02)
    partials = [
        client.chat(
            SYSTEM_PROMPT,
            f"{_MODE_INSTRUCTIONS['paragraph']}\n\nText (part {i + 1} of {len(pieces)}):\n\n{p.text}",
            temperature=0.3,
        )
        for i, p in enumerate(pieces)
    ]
    combined = "\n\n".join(partials)
    return client.chat(
        SYSTEM_PROMPT,
        f"The following are partial summaries of one document, in order. "
        f"{instruction}\n\nPartial summaries:\n\n{combined}",
        temperature=0.3,
    )


def summarize_documents(
    conn: Connection,
    client: LLMClient,
    doc_ids: list[int],
    mode: str = "paragraph",
) -> str:
    if not 1 <= len(doc_ids) <= 3:
        raise ValueError("summarization takes 1-3 documents")
    if len(doc_ids) == 1:
        return summarize_text(client, document_text(conn, doc_ids[0]), mode)

    named: list[str] = []
    for doc_id in doc_ids:
        row = conn.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            raise ValueError(f"no document with id {doc_id}")
        part = summarize_text(client, document_text(conn, doc_id), "paragraph")
        named.append(f"## {row['filename']}\n{part}")
    combined = "\n\n".join(named)
    return client.chat(
        SYSTEM_PROMPT,
        f"The following are summaries of {len(doc_ids)} related documents. Produce one combined summary. "
        f"{_MODE_INSTRUCTIONS[mode]}\n\n{combined}",
        temperature=0.3,
    )
