"""Document Q&A (RAG) with citations.

The model is constrained to answer only from retrieved excerpts and to say
"Not found in your documents" otherwise. Citations are built from the actual
retrieved chunks — the model references excerpts by [n] marker and we map
markers back to real files, so a fabricated citation cannot appear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from sqlite3 import Connection

from .chunking import approx_tokens
from .config import Config
from .observability import observe, update_current_span
from .ollama_client import OllamaClient
from .retrieval import RetrievedChunk, hybrid_search
from .vectors import VectorIndex

NOT_FOUND = "Not found in your documents"

SYSTEM_PROMPT = f"""You are a personal document assistant. Answer the user's question using ONLY the numbered context excerpts provided. Rules:
- If the answer is not present in the excerpts, reply with exactly: {NOT_FOUND}
- Never use outside knowledge or guess.
- Cite the excerpts you used with bracketed numbers, e.g. [1] or [2][3].
- Be concise and direct."""


@dataclass
class Source:
    filename: str
    path: str
    page: int | None
    chunk_idx: int


@dataclass
class QAResult:
    answer: str
    sources: list[Source]
    chunks: list[RetrievedChunk]


def context_block(i: int, chunk: RetrievedChunk) -> str:
    loc = f", page {chunk.page}" if chunk.page else f", section {chunk.chunk_idx + 1}"
    return f"[{i}] ({chunk.filename}{loc})\n{chunk.text}"


def pack_context(chunks: list[RetrievedChunk], budget: int) -> list[RetrievedChunk]:
    """Greedily keep chunks (in given order) while staying inside a token budget."""
    included: list[RetrievedChunk] = []
    for c in chunks:
        cost = approx_tokens(c.text) + 20
        if included and cost > budget:
            break
        included.append(c)
        budget -= cost
    return included


def resolve_citations(answer: str, included: list[RetrievedChunk]) -> list[Source]:
    """Map [n] markers in the answer back to the chunks actually included in the
    prompt — a fabricated or out-of-range citation is silently ignored, never
    trusted to reference something it doesn't."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(included)}
    used = [included[n - 1] for n in sorted(cited)] if cited else included
    seen: set[tuple[str, int | None]] = set()
    sources: list[Source] = []
    for c in used:
        key = (c.path, c.page)
        if key not in seen:
            seen.add(key)
            sources.append(Source(filename=c.filename, path=c.path, page=c.page, chunk_idx=c.chunk_idx))
    return sources


@observe(name="qa.answer_question", as_type="chain", capture_input=False, capture_output=False)
def answer_question(
    conn: Connection,
    index: VectorIndex,
    client: OllamaClient,
    question: str,
    cfg: Config,
    extra_context: str = "",
) -> QAResult:
    update_current_span(input={"question": question})
    chunks = hybrid_search(
        conn,
        index,
        client,
        question,
        k_vec=cfg.vector_top_k,
        k_kw=cfg.keyword_top_k,
        limit=max(cfg.vector_top_k, cfg.keyword_top_k),
    )
    if not chunks:
        update_current_span(output={"answer": NOT_FOUND, "sources": 0})
        return QAResult(answer=NOT_FOUND, sources=[], chunks=[])

    included = pack_context(chunks, cfg.context_token_budget)

    context = "\n\n".join(context_block(i + 1, c) for i, c in enumerate(included))
    prefix = f"{extra_context}\n\n" if extra_context else ""
    user = f"{prefix}Context excerpts:\n\n{context}\n\nQuestion: {question}"
    answer = client.chat(SYSTEM_PROMPT, user, temperature=0.2, trace_name="answer")

    if answer.startswith(NOT_FOUND):
        update_current_span(output={"answer": NOT_FOUND, "sources": 0})
        return QAResult(answer=NOT_FOUND, sources=[], chunks=included)

    sources = resolve_citations(answer, included)
    update_current_span(output={"answer": answer, "sources": len(sources)})
    return QAResult(answer=answer, sources=sources, chunks=included)
