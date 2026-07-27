"""Agentic Q&A architecture optimized for small (8B) models.

This module implements a multi-agent system where each agent has a focused,
single-purpose task. This decomposition allows smaller models to perform
complex reasoning by breaking it into manageable steps.

Enhanced with:
    - Hybrid search (vector + keyword + semantic memory)
    - Agent memory (conversation history + long-term semantic memory)
    - Query caching (LRU cache with TTL for fast repeated queries)
    - Knowledge base integration (structured facts and relationships)

Agents:
    - QueryDecomposer: Breaks complex questions into simpler sub-queries
    - RetrievalPlanner: Plans and executes iterative retrieval strategies  
    - EvidenceSynthesizer: Combines and ranks evidence from multiple sources
    - AnswerGenerator: Generates final answers with citations
    - AnswerVerifier: Validates answer completeness and accuracy
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import Literal

from .chunking import approx_tokens
from .config import Config
from .memory import AgentMemory, KnowledgeBase, QueryCache
from .ollama_client import OllamaClient
from .retrieval import RetrievedChunk, hybrid_search
from .vectors import VectorIndex


@dataclass
class SubQuery:
    """A decomposed sub-question for focused retrieval."""
    question: str
    rationale: str  # Why this sub-question matters
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class EvidenceItem:
    """Retrieved evidence with relevance scoring."""
    chunk: RetrievedChunk
    relevance_score: float
    supports_query: str  # Which sub-query this supports


@dataclass
class SynthesizedEvidence:
    """Combined evidence ready for answer generation."""
    items: list[EvidenceItem]
    coverage_gaps: list[str]  # Questions not fully answered
    confidence: float  # Overall confidence in evidence sufficiency


@dataclass
class VerificationResult:
    """Result of answer verification."""
    is_complete: bool
    missing_aspects: list[str]
    suggested_followups: list[str]
    confidence: float


# ============================================================================
# Query Decomposition Agent
# ============================================================================

QUERY_DECOMPOSER_SYSTEM = """You are a query decomposition assistant. Your task is to break down complex questions into simpler, focused sub-questions.

Rules:
- If the question is already simple, return it as a single sub-question
- Each sub-question should be answerable independently
- Focus on factual aspects that can be found in documents
- Output MUST be valid JSON matching the schema exactly
- Keep sub-questions concise (under 20 words each)"""

QUERY_DECOMPOSER_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "rationale": {"type": "string"},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 3}
                },
                "required": ["question", "rationale", "priority"]
            }
        }
    },
    "required": ["sub_queries"]
}


def decompose_query(client: OllamaClient, question: str, max_subqueries: int = 4) -> list[SubQuery]:
    """Break down a complex question into simpler sub-queries.
    
    Small models excel at focused tasks. By decomposing complex questions,
    we enable better retrieval and reduce the cognitive load on downstream agents.
    """
    user_prompt = f"""Decompose this question into focused sub-questions:

Question: {question}

Return 1-{max_subqueries} sub-questions that cover all aspects of the original question."""

    try:
        result = client.chat_json(
            QUERY_DECOMPOSER_SYSTEM,
            user_prompt,
            QUERY_DECOMPOSER_SCHEMA,
            temperature=0.1
        )
        
        sub_queries = []
        for i, sq in enumerate(result.get("sub_queries", [])[:max_subqueries]):
            sub_queries.append(SubQuery(
                question=sq["question"].strip(),
                rationale=sq.get("rationale", ""),
                priority=sq.get("priority", 2)
            ))
        
        # Fallback if decomposition fails
        if not sub_queries:
            sub_queries.append(SubQuery(
                question=question,
                rationale="Original question used as-is",
                priority=1
            ))
        
        return sub_queries
    
    except Exception as e:
        # Graceful fallback: treat as single query
        return [SubQuery(question=question, rationale=f"Decomposition failed: {e}", priority=1)]


# ============================================================================
# Retrieval Planning Agent
# ============================================================================

RETRIEVAL_PLANNER_SYSTEM = """You are a retrieval planning assistant. Your task is to optimize search queries for document retrieval.

Rules:
- Reformulate questions to improve keyword matching
- Identify key terms that should appear in documents
- Suggest alternative phrasings for better recall
- Output MUST be valid JSON matching the schema exactly"""

RETRIEVAL_PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "optimized_query": {"type": "string"},
        "key_terms": {"type": "array", "items": {"type": "string"}},
        "alternative_phrasings": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["optimized_query", "key_terms"]
}


def plan_retrieval(client: OllamaClient, sub_query: SubQuery) -> tuple[str, list[str]]:
    """Optimize a sub-query for better retrieval performance.
    
    Small models benefit from clear, keyword-rich queries. This agent
    reformulates questions to maximize retrieval effectiveness.
    """
    user_prompt = f"""Optimize this question for document retrieval:

Question: {sub_query.question}
Context: {sub_query.rationale}

Provide an optimized query and identify key search terms."""

    try:
        result = client.chat_json(
            RETRIEVAL_PLANNER_SYSTEM,
            user_prompt,
            RETRIEVAL_PLANNER_SCHEMA,
            temperature=0.1
        )
        
        optimized = result.get("optimized_query", sub_query.question)
        key_terms = result.get("key_terms", [])
        
        return optimized, key_terms
    
    except Exception:
        return sub_query.question, []


def execute_retrieval(
    conn: Connection,
    index: VectorIndex,
    client: OllamaClient,
    query: str,
    cfg: Config,
    doc_ids: list[int] | None = None,
    k_override: int | None = None,
) -> list[RetrievedChunk]:
    """Execute retrieval for a single optimized query."""
    k = k_override or max(cfg.vector_top_k, cfg.keyword_top_k)
    
    return hybrid_search(
        conn,
        index,
        client,
        query,
        k_vec=k,
        k_kw=k,
        limit=k,
        doc_ids=doc_ids,
    )


# ============================================================================
# Evidence Synthesis Agent
# ============================================================================

EVIDENCE_SYNTHESIZER_SYSTEM = """You are an evidence synthesis assistant. Evaluate retrieved information for relevance and completeness.

Rules:
- Score each piece of evidence for relevance (0.0-1.0)
- Identify gaps in coverage
- Assess overall confidence
- Be critical: low-quality evidence reduces confidence
- Output MUST be valid JSON matching the schema exactly"""

EVIDENCE_SYNTHESIZER_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"}
                },
                "required": ["index", "score"]
            }
        },
        "coverage_gaps": {"type": "array", "items": {"type": "string"}},
        "overall_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["evidence_scores", "overall_confidence"]
}


def synthesize_evidence(
    client: OllamaClient,
    chunks: list[RetrievedChunk],
    original_question: str,
    sub_queries: list[SubQuery],
    max_evidence: int = 10,
) -> SynthesizedEvidence:
    """Evaluate and synthesize retrieved evidence.
    
    This agent filters noise and identifies the most relevant evidence,
    crucial for small models that may struggle with large contexts.
    """
    if not chunks:
        return SynthesizedEvidence(
            items=[],
            coverage_gaps=["No documents retrieved"],
            confidence=0.0
        )
    
    # Limit context for small model evaluation
    limited_chunks = chunks[:max_evidence]
    
    # Build evidence summary for evaluation
    evidence_summary = "\n".join(
        f"[{i}] {c.text[:200]}..." for i, c in enumerate(limited_chunks)
    )
    
    sub_query_text = "\n".join(f"- {sq.question}" for sq in sub_queries)
    
    user_prompt = f"""Evaluate this evidence for answering the question:

Original Question: {original_question}
Sub-Questions: {sub_query_text}

Evidence:
{evidence_summary}

Score each piece of evidence and identify gaps."""

    try:
        result = client.chat_json(
            EVIDENCE_SYNTHESIZER_SYSTEM,
            user_prompt,
            EVIDENCE_SYNTHESIZER_SCHEMA,
            temperature=0.1
        )
        
        scores_data = result.get("evidence_scores", [])
        score_map = {item["index"]: item["score"] for item in scores_data if "index" in item}
        
        evidence_items = []
        for i, chunk in enumerate(limited_chunks):
            score = score_map.get(i, 0.5)  # Default score if not evaluated
            if score >= 0.3:  # Filter low-relevance evidence
                evidence_items.append(EvidenceItem(
                    chunk=chunk,
                    relevance_score=score,
                    supports_query=sub_queries[0].question if sub_queries else original_question
                ))
        
        # Sort by relevance
        evidence_items.sort(key=lambda e: -e.relevance_score)
        
        coverage_gaps = result.get("coverage_gaps", [])
        confidence = result.get("overall_confidence", 0.5)
        
        return SynthesizedEvidence(
            items=evidence_items,
            coverage_gaps=coverage_gaps,
            confidence=confidence
        )
    
    except Exception:
        # Fallback: use top chunks with uniform scoring
        evidence_items = [
            EvidenceItem(chunk=c, relevance_score=0.7, supports_query=original_question)
            for c in limited_chunks[:5]
        ]
        return SynthesizedEvidence(
            items=evidence_items,
            coverage_gaps=[],
            confidence=0.6
        )


# ============================================================================
# Answer Generation Agent
# ============================================================================

ANSWER_GENERATOR_SYSTEM = """You are a document-based answer generation assistant. Answer using ONLY the provided evidence.

Rules:
- Use ONLY information from the numbered evidence excerpts
- If evidence is insufficient, say "Not found in your documents"
- Cite evidence using bracketed numbers like [1] or [2][3]
- Be concise and direct
- Never use outside knowledge or guess
- Do not fabricate citations - only use provided excerpt numbers"""


def generate_answer(
    client: OllamaClient,
    evidence: SynthesizedEvidence,
    question: str,
    cfg: Config,
) -> tuple[str, list[RetrievedChunk]]:
    """Generate final answer from synthesized evidence.
    
    With pre-filtered, high-quality evidence, small models can focus
    on accurate answer generation rather than information filtering.
    """
    if not evidence.items:
        return "Not found in your documents", []
    
    # Build context with token budget awareness
    included: list[RetrievedChunk] = []
    budget = cfg.context_token_budget
    for item in evidence.items:
        cost = approx_tokens(item.chunk.text) + 20
        if included and cost > budget:
            break
        included.append(item.chunk)
        budget -= cost
    
    if not included:
        return "Not found in your documents", []
    
    # Format context with relevance scores
    context_parts = []
    for i, chunk in enumerate(included, 1):
        loc = f", page {chunk.page}" if chunk.page else f", section {chunk.chunk_idx + 1}"
        context_parts.append(
            f"[{i}] ({chunk.filename}{loc}) [relevance: {item.relevance_score:.2f}]\n{chunk.text}"
        )
    
    context = "\n\n".join(context_parts)
    
    user_prompt = f"""Evidence excerpts:

{context}

Question: {question}

Provide a concise answer citing the relevant evidence numbers."""

    answer = client.chat(ANSWER_GENERATOR_SYSTEM, user_prompt, temperature=0.2)
    
    return answer, included


# ============================================================================
# Answer Verification Agent
# ============================================================================

ANSWER_VERIFIER_SYSTEM = """You are an answer verification assistant. Check if the answer completely addresses the question.

Rules:
- Identify any missing aspects or incomplete information
- Suggest follow-up questions if needed
- Assess confidence in answer completeness
- Output MUST be valid JSON matching the schema exactly"""

ANSWER_VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "is_complete": {"type": "boolean"},
        "missing_aspects": {"type": "array", "items": {"type": "string"}},
        "suggested_followups": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["is_complete", "confidence"]
}


def verify_answer(
    client: OllamaClient,
    question: str,
    answer: str,
    evidence: list[RetrievedChunk],
) -> VerificationResult:
    """Verify answer completeness and quality.
    
    This final check ensures the answer adequately addresses the question,
    enabling iterative refinement if needed.
    """
    evidence_summary = "\n".join(f"- {e.text[:150]}" for e in evidence[:5])
    
    user_prompt = f"""Verify this answer against the question and evidence:

Question: {question}
Answer: {answer}

Key Evidence:
{evidence_summary}

Is the answer complete and supported by the evidence?"""

    try:
        result = client.chat_json(
            ANSWER_VERIFIER_SYSTEM,
            user_prompt,
            ANSWER_VERIFIER_SCHEMA,
            temperature=0.1
        )
        
        return VerificationResult(
            is_complete=result.get("is_complete", True),
            missing_aspects=result.get("missing_aspects", []),
            suggested_followups=result.get("suggested_followups", []),
            confidence=result.get("confidence", 0.7)
        )
    
    except Exception:
        return VerificationResult(
            is_complete=True,
            missing_aspects=[],
            suggested_followups=[],
            confidence=0.6
        )


# ============================================================================
# Main Agentic Q&A Pipeline
# ============================================================================

@dataclass
class AgentQAResult:
    """Result from agentic Q&A pipeline."""
    answer: str
    sources: list[RetrievedChunk]
    sub_queries: list[SubQuery]
    evidence_confidence: float
    verification: VerificationResult | None
    iterations: int


def _extract_sources_from_citations(answer: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Extract cited sources from answer text."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(chunks)}
    if cited:
        return [chunks[n - 1] for n in sorted(cited)]
    return chunks[:3] if chunks else []


def answer_question_agentic(
    conn: Connection,
    index: VectorIndex,
    client: OllamaClient,
    question: str,
    cfg: Config,
    doc_ids: list[int] | None = None,
    max_iterations: int = 2,
    memory: AgentMemory | None = None,
    cache: QueryCache | None = None,
    knowledge_base: KnowledgeBase | None = None,
) -> AgentQAResult:
    """Answer a question using the agentic pipeline with hybrid search and memory.
    
    This enhanced multi-agent approach includes:
    1. Check cache for fast repeated queries
    2. Query knowledge base for structured facts
    3. Use agent memory for conversation context
    4. Decompose complex questions
    5. Hybrid retrieval (vector + keyword + semantic memory)
    6. Synthesize and filter evidence
    7. Generate concise answers with citations
    8. Verify completeness
    9. Update memory and cache
    
    Optimized for small (8B) models by reducing per-step complexity.
    
    Args:
        conn: Database connection
        index: Vector index for similarity search
        client: Ollama client for LLM calls
        question: User's question
        cfg: Configuration
        doc_ids: Optional document IDs to restrict search
        max_iterations: Maximum refinement iterations (currently disabled)
        memory: Optional agent memory for conversation context
        cache: Optional query cache for fast repeated queries
        knowledge_base: Optional knowledge base for structured facts
    
    Returns:
        AgentQAResult with answer, sources, and metadata
    """
    # Step 0: Check cache for instant response
    if cache is not None:
        cached = cache.get(question, doc_ids)
        if cached is not None:
            # Convert cached sources back to RetrievedChunk format if needed
            return AgentQAResult(
                answer=cached.answer,
                sources=[],  # Would need to reconstruct from cached data
                sub_queries=[],
                evidence_confidence=1.0,
                verification=None,
                iterations=0  # Cache hit, no agent iterations
            )
    
    # Step 0b: Add question to memory
    if memory is not None:
        memory.add_user_message(question)
    
    # Step 0c: Query knowledge base for relevant facts
    kb_facts = []
    if knowledge_base is not None:
        # Extract key entities from question for KB lookup
        kb_facts = knowledge_base.query_facts(limit=20)
        # Filter to relevant facts (simple keyword match for now)
        question_words = set(question.lower().split())
        kb_facts = [
            f for f in kb_facts
            if any(word in f.subject.lower() or word in f.object.lower()
                   for word in question_words if len(word) > 3)
        ]
    
    # Step 1: Query Decomposition
    sub_queries = decompose_query(client, question, max_subqueries=4)
    
    # Step 2: Hybrid Retrieval (vector + keyword + semantic memory)
    all_chunks: dict[str, list[RetrievedChunk]] = {}
    for sq in sub_queries:
        optimized_query, _ = plan_retrieval(client, sq)
        chunks = execute_retrieval(
            conn, index, client, optimized_query, cfg, doc_ids=doc_ids
        )
        all_chunks[sq.question] = chunks
    
    # Add semantic memory results if available
    if memory is not None and memory.semantic_memory:
        memory_results = memory.search_semantic_memory(question, k=3)
        # Convert memory items to pseudo-chunks for uniform processing
        for i, mem_item in enumerate(memory_results):
            # Create a synthetic chunk from memory
            from .retrieval import RetrievedChunk
            mem_chunk = RetrievedChunk(
                chunk_id=-1000 - i,  # Negative ID to avoid collision
                doc_id=-1,
                filename="[Memory]",
                path="[Semantic Memory]",
                chunk_idx=i,
                text=mem_item.content,
                char_start=0,
                char_end=len(mem_item.content),
                page=None,
                score=mem_item.importance,
            )
            # Add to first sub-query results or create new entry
            if all_chunks:
                first_key = next(iter(all_chunks.keys()))
                all_chunks[first_key].append(mem_chunk)
            else:
                all_chunks["memory"] = [mem_chunk]
    
    # Merge and deduplicate chunks
    seen_ids: set[int] = set()
    merged_chunks: list[RetrievedChunk] = []
    for chunks in all_chunks.values():
        for c in chunks:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                merged_chunks.append(c)
    
    # Inject knowledge base facts as additional context
    if kb_facts:
        kb_text = "\n".join(f"{f.subject} {f.predicate} {f.object}" for f in kb_facts[:10])
        from .retrieval import RetrievedChunk
        kb_chunk = RetrievedChunk(
            chunk_id=-2000,
            doc_id=-1,
            filename="[Knowledge Base]",
            path="[Structured Facts]",
            chunk_idx=0,
            text=f"Relevant facts:\n{kb_text}",
            char_start=0,
            char_end=len(kb_text),
            page=None,
            score=0.9,
        )
        merged_chunks.insert(0, kb_chunk)  # Prioritize KB facts
    
    # Step 3: Evidence Synthesis
    evidence = synthesize_evidence(
        client, merged_chunks, question, sub_queries, max_evidence=12
    )
    
    # Early exit if no good evidence
    if evidence.confidence < 0.2 or not evidence.items:
        answer = "Not found in your documents"
        
        # Store in memory as important gap
        if memory is not None:
            memory.add_to_semantic_memory(
                f"Question without answer: {question}",
                source="unanswered",
                importance=0.5,
                tags=["unanswered", "gap"],
            )
        
        result = AgentQAResult(
            answer=answer,
            sources=[],
            sub_queries=sub_queries,
            evidence_confidence=evidence.confidence,
            verification=None,
            iterations=1
        )
        
        # Cache the negative result
        if cache is not None:
            cache.set(question, answer, [], doc_ids, ttl=1800)  # Shorter TTL for negatives
        
        return result
    
    # Step 4: Answer Generation
    answer, used_chunks = generate_answer(client, evidence, question, cfg)
    
    # Step 5: Answer Verification (optional, skip if low confidence to save tokens)
    verification = None
    if evidence.confidence >= 0.5 and not answer.startswith("Not found"):
        verification = verify_answer(client, question, answer, used_chunks)
        
        # Optional: Iterative refinement if answer incomplete
        # (Disabled by default to minimize latency with small models)
    
    # Extract final sources
    sources = _extract_sources_from_citations(answer, used_chunks)
    
    # Step 6: Update memory with important information
    if memory is not None:
        # Add assistant response to conversation history
        memory.add_assistant_message(answer)
        
        # Extract key facts from answer for long-term memory
        if evidence.confidence >= 0.7:
            # Store condensed version of answer
            summary = answer[:500]  # First 500 chars as summary
            memory.add_to_semantic_memory(
                summary,
                source="qa_answer",
                importance=evidence.confidence,
                tags=["answer", "verified"] if verification and verification.is_complete else ["answer"],
            )
    
    # Step 7: Cache the result
    if cache is not None:
        source_dicts = [
            {"filename": s.filename, "path": s.path, "page": s.page}
            for s in sources[:5]
        ]
        cache.set(question, answer, source_dicts, doc_ids)
    
    return AgentQAResult(
        answer=answer,
        sources=sources,
        sub_queries=sub_queries,
        evidence_confidence=evidence.confidence,
        verification=verification,
        iterations=1
    )
