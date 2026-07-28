"""Multi-step agentic Q&A: decomposes compound questions, plans retrieval per
sub-question, filters evidence before generation, and verifies the answer —
each step a small, focused model call instead of one monolithic prompt.

Short/simple questions skip straight to `qa.answer_question` (the "fast
path"): decomposition only pays for itself on genuinely multi-part questions,
and this keeps the common case at today's Phase 1 latency. Citation safety is
inherited directly from qa.py's audited helpers (`pack_context`,
`context_block`, `resolve_citations`) rather than reimplemented here, so a
fabricated or out-of-range [n] citation can never surface from either
pipeline.

`cache`, `memory`, and `knowledge_base` (see memory.py) are optional,
duck-typed extension points — passing none of them reproduces the plain
5-step pipeline with no persistence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from sqlite3 import Connection
from typing import Any

from . import db, qa
from .config import Config
from .ollama_client import OllamaClient, OllamaError
from .retrieval import RetrievedChunk, hybrid_search
from .vectors import VectorIndex

logger = logging.getLogger("vault.agents")

_ENUMERATION_PATTERN = re.compile(r"(?:^|\s)\d[.)]\s")


@dataclass
class SubQuery:
    question: str
    search_query: str
    key_terms: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    confidence: float
    is_complete: bool
    gaps: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgenticQAResult:
    answer: str
    sources: list[qa.Source]
    chunks: list[RetrievedChunk]
    sub_queries: list[SubQuery]
    evidence_confidence: float | None
    verification: VerificationResult | None
    pipeline: str  # "fast" | "full" | "cached"
    cache_hit: bool = False


DECOMPOSE_SYSTEM_PROMPT = """You break a user's question into independent sub-questions when it asks about more than one distinct topic. Rules:
- If the question is already about ONE topic, return exactly one sub-question equal to the original question.
- Otherwise split into at most {max_subqueries} sub-questions, one per distinct topic.
- For each sub-question, also give a keyword-rich search_query (3-8 words, no filler words) optimized for search, and up to 5 key_terms (single words or short phrases).
- Never answer the question — only decompose and reformulate it."""

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_queries": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "search_query": {"type": "string"},
                    "key_terms": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question", "search_query", "key_terms"],
            },
        }
    },
    "required": ["sub_queries"],
}

SYNTHESIZE_SYSTEM_PROMPT = """You score how relevant numbered excerpts are to the question(s) below. For EVERY excerpt index given, return one relevance score from 0.0 (unrelated) to 1.0 (directly answers a question). Base scores only on the excerpt text, not outside knowledge. Return a score for every index, in any order, with no extra commentary."""

SYNTHESIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "relevance": {"type": "number"},
                },
                "required": ["index", "relevance"],
            },
        }
    },
    "required": ["scores"],
}

VERIFY_SYSTEM_PROMPT = """You check a generated answer against the excerpts it was based on. Rules:
- confidence: 0.0-1.0, how well the answer is supported by the excerpts AND how completely it addresses every part of the question.
- is_complete: true only if every part of the question is addressed.
- gaps: short phrases naming anything the question asked for that the answer is missing; empty array if none.
Judge only what is given; do not rewrite the answer or add new information."""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence": {"type": "number"},
        "is_complete": {"type": "boolean"},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["confidence", "is_complete", "gaps"],
}


def _needs_decomposition(question: str, cfg: Config) -> bool:
    """Cheap heuristic gate, no model call: only pay for the full pipeline
    when a question actually looks multi-part."""
    words = question.split()
    if len(words) < cfg.agentic_min_words_for_decompose:
        return False
    if len(words) > cfg.agentic_max_words_simple:
        return True
    if question.count("?") > 1:
        return True
    if _ENUMERATION_PATTERN.search(question):
        return True
    lowered = f" {question.lower()} "
    return any(conj in lowered for conj in cfg.agentic_conjunctions)


def _decompose(client: OllamaClient, question: str, cfg: Config) -> list[SubQuery]:
    system = DECOMPOSE_SYSTEM_PROMPT.format(max_subqueries=cfg.agentic_max_subqueries)
    try:
        result = client.chat_json(system, f"Question: {question}", DECOMPOSE_SCHEMA)
    except OllamaError as exc:
        logger.warning("decomposition failed, using single implicit sub-query: %s", exc)
        return [SubQuery(question, question, [])]

    sub_queries: list[SubQuery] = []
    raw_items = result.get("sub_queries", []) if isinstance(result, dict) else []
    for raw in raw_items:
        q = (raw.get("question") or "").strip()
        sq = (raw.get("search_query") or "").strip()
        if not q or not sq:
            continue
        raw_terms = raw.get("key_terms")
        terms = [str(t).strip() for t in raw_terms if str(t).strip()] if isinstance(raw_terms, list) else []
        sub_queries.append(SubQuery(question=q, search_query=sq, key_terms=terms[:5]))
        if len(sub_queries) >= cfg.agentic_max_subqueries:
            break
    return sub_queries or [SubQuery(question, question, [])]


def _merge_candidates(
    hits_per_subquery: list[list[RetrievedChunk]], max_candidates: int
) -> list[RetrievedChunk]:
    best: dict[int, RetrievedChunk] = {}
    match_counts: dict[int, int] = {}
    for hits in hits_per_subquery:
        for chunk in hits:
            match_counts[chunk.chunk_id] = match_counts.get(chunk.chunk_id, 0) + 1
            existing = best.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                best[chunk.chunk_id] = chunk
    ordered = sorted(best.values(), key=lambda c: (-match_counts[c.chunk_id], -c.score))
    return ordered[:max_candidates]


def _kb_candidates(
    knowledge_base: Any, sub_queries: list[SubQuery], cfg: Config, existing_ids: set[int]
) -> list[RetrievedChunk]:
    """Best-effort: pull KB facts related to the sub-queries' key terms and
    wrap them as synthetic chunks so synthesis scores them identically to
    retrieved text. Never lets a KB lookup failure affect Q&A."""
    terms: list[str] = []
    seen_terms: set[str] = set()
    for sq in sub_queries:
        for t in sq.key_terms:
            key = t.lower()
            if key not in seen_terms:
                seen_terms.add(key)
                terms.append(t)

    facts_by_id: dict[int, Any] = {}
    for term in terms:
        try:
            related = knowledge_base.find_related(term, depth=1, limit=cfg.kb_find_related_limit)
        except Exception as exc:  # noqa: BLE001 — KB enrichment is optional, must never break Q&A
            logger.warning("knowledge base lookup failed for %r: %s", term, exc)
            continue
        for fact in related:
            facts_by_id[fact.id] = fact

    candidates: list[RetrievedChunk] = []
    for fact in facts_by_id.values():
        chunk_id = -fact.id  # negative id space: never collides with a real chunk id
        if chunk_id in existing_ids:
            continue
        text = f"{fact.subject} {fact.predicate} {fact.object}"
        candidates.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=-1,
                filename="knowledge base",
                path="",
                chunk_idx=0,
                text=text,
                char_start=0,
                char_end=len(text),
                page=None,
                score=fact.confidence,
            )
        )
    return candidates


def _retrieve_pool(
    conn: Connection,
    index: VectorIndex,
    client: OllamaClient,
    sub_queries: list[SubQuery],
    cfg: Config,
    knowledge_base: Any,
) -> list[RetrievedChunk]:
    hits_per_subquery = [
        hybrid_search(
            conn,
            index,
            client,
            sq.search_query,
            k_vec=cfg.vector_top_k,
            k_kw=cfg.keyword_top_k,
            limit=max(cfg.vector_top_k, cfg.keyword_top_k),
        )
        for sq in sub_queries
    ]
    pool = _merge_candidates(hits_per_subquery, cfg.agentic_max_candidates)
    if knowledge_base is not None:
        pool = pool + _kb_candidates(knowledge_base, sub_queries, cfg, {c.chunk_id for c in pool})
    return pool


def _synthesize(
    client: OllamaClient,
    question: str,
    sub_queries: list[SubQuery],
    pool: list[RetrievedChunk],
    cfg: Config,
) -> tuple[list[RetrievedChunk], dict[int, float]]:
    """One call scores ALL candidates at once — never one call per chunk."""
    excerpts = [
        f"[{i + 1}] ({c.filename}) {c.text[: cfg.agentic_excerpt_chars]}" for i, c in enumerate(pool)
    ]
    questions_block = "\n".join(f"- {sq.question}" for sq in sub_queries) if len(sub_queries) > 1 else question
    user = f"Question(s):\n{questions_block}\n\nExcerpts:\n" + "\n".join(excerpts)

    try:
        result = client.chat_json(SYNTHESIZE_SYSTEM_PROMPT, user, SYNTHESIZE_SCHEMA)
    except OllamaError as exc:
        logger.warning("evidence synthesis failed, using unranked retrieval order: %s", exc)
        return pool[: cfg.agentic_evidence_top_n], {}

    relevance: dict[int, float] = {}
    raw_scores = result.get("scores", []) if isinstance(result, dict) else []
    for raw in raw_scores:
        try:
            idx = int(raw.get("index"))
            score = float(raw.get("relevance"))
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(pool):
            relevance[pool[idx - 1].chunk_id] = max(0.0, min(1.0, score))

    filtered = [c for c in pool if relevance.get(c.chunk_id, 0.0) >= cfg.agentic_relevance_threshold]
    filtered.sort(key=lambda c: -relevance.get(c.chunk_id, 0.0))
    if not filtered:
        # threshold discarded everything: fall back to the top few by retrieval
        # score rather than starving the generator — its own NOT_FOUND rule is
        # the real backstop against a bad answer.
        filtered = pool[:3]
    return filtered[: cfg.agentic_evidence_top_n], relevance


def _verify(
    client: OllamaClient, question: str, answer: str, context: str, cfg: Config
) -> VerificationResult | None:
    if not cfg.agentic_verify:
        return None
    user = f"Question: {question}\n\nAnswer given: {answer}\n\nExcerpts used:\n{context}"
    try:
        result = client.chat_json(VERIFY_SYSTEM_PROMPT, user, VERIFY_SCHEMA)
    except OllamaError as exc:
        logger.warning("verification failed, returning answer unverified: %s", exc)
        return None

    try:
        confidence = float(result.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    is_complete = bool(result.get("is_complete", True))
    raw_gaps = result.get("gaps")
    gaps = [str(g).strip() for g in raw_gaps if str(g).strip()][:5] if isinstance(raw_gaps, list) else []
    return VerificationResult(confidence=confidence, is_complete=is_complete, gaps=gaps)


def _finalize(cache: Any, memory: Any, cfg: Config, generation: int, question: str, result: AgenticQAResult) -> None:
    if memory is not None:
        memory.add_user_message(question)
        memory.add_assistant_message(result.answer)
    if cache is not None:
        cache.set(
            question,
            result.answer,
            [asdict(s) for s in result.sources],
            ttl=cfg.cache_default_ttl,
            generation=generation,
        )


def answer_question_agentic(
    conn: Connection,
    index: VectorIndex,
    client: OllamaClient,
    question: str,
    cfg: Config,
    *,
    cache: Any = None,
    memory: Any = None,
    knowledge_base: Any = None,
) -> AgenticQAResult:
    generation = db.get_generation(conn)

    if cache is not None:
        hit = cache.get(question, generation=generation)
        if hit is not None:
            return AgenticQAResult(
                answer=hit.answer,
                sources=[qa.Source(**s) for s in hit.sources],
                chunks=[],
                sub_queries=[SubQuery(question, question, [])],
                evidence_confidence=None,
                verification=None,
                pipeline="cached",
                cache_hit=True,
            )

    memory_ctx = memory.get_context(question) if memory is not None else ""

    if not _needs_decomposition(question, cfg):
        legacy = qa.answer_question(conn, index, client, question, cfg, extra_context=memory_ctx)
        result = AgenticQAResult(
            answer=legacy.answer,
            sources=legacy.sources,
            chunks=legacy.chunks,
            sub_queries=[SubQuery(question=question, search_query=question, key_terms=[])],
            evidence_confidence=None,
            verification=None,
            pipeline="fast",
        )
        _finalize(cache, memory, cfg, generation, question, result)
        return result

    sub_queries = _decompose(client, question, cfg)
    pool = _retrieve_pool(conn, index, client, sub_queries, cfg, knowledge_base)
    if not pool:
        result = AgenticQAResult(qa.NOT_FOUND, [], [], sub_queries, None, None, "full")
        _finalize(cache, memory, cfg, generation, question, result)
        return result

    filtered, relevance = _synthesize(client, question, sub_queries, pool, cfg)
    included = qa.pack_context(filtered, cfg.context_token_budget)
    context = "\n\n".join(qa.context_block(i + 1, c) for i, c in enumerate(included))
    prefix = f"{memory_ctx}\n\n" if memory_ctx else ""
    user = f"{prefix}Context excerpts:\n\n{context}\n\nQuestion: {question}"
    answer = client.chat(qa.SYSTEM_PROMPT, user, temperature=0.2)

    if answer.startswith(qa.NOT_FOUND):
        result = AgenticQAResult(qa.NOT_FOUND, [], included, sub_queries, None, None, "full")
        _finalize(cache, memory, cfg, generation, question, result)
        return result

    sources = qa.resolve_citations(answer, included)
    scores = [relevance[c.chunk_id] for c in included if c.chunk_id in relevance]
    evidence_confidence = sum(scores) / len(scores) if scores else None
    verification = _verify(client, question, answer, context, cfg)

    result = AgenticQAResult(
        answer=answer,
        sources=sources,
        chunks=included,
        sub_queries=sub_queries,
        evidence_confidence=evidence_confidence,
        verification=verification,
        pipeline="full",
    )
    _finalize(cache, memory, cfg, generation, question, result)
    return result
