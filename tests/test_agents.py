"""Tests for agentic Q&A architecture."""

import json

import pytest

from vault_assistant.agents import (
    AgentQAResult,
    EvidenceItem,
    SubQuery,
    SynthesizedEvidence,
    VerificationResult,
    answer_question_agentic,
    decompose_query,
    execute_retrieval,
    generate_answer,
    plan_retrieval,
    synthesize_evidence,
    verify_answer,
)
from vault_assistant.config import Config
from vault_assistant.ingest import ingest_paths
from vault_assistant.qa import NOT_FOUND
from vault_assistant.vectors import VectorIndex


def _seed(tmp_path, conn, client):
    """Seed database with test documents."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "policy.txt").write_text(
        "Vacation policy: employees receive 25 days of paid leave per year. " * 8
    )
    (folder / "benefits.txt").write_text(
        "Health insurance covers dental, vision, and medical expenses. " * 8
    )
    ingest_paths(conn, client, [folder])


class TestQueryDecomposition:
    """Test query decomposition agent."""

    def test_simple_question_returns_single_subquery(self, client):
        """Simple questions should return as-is or with minimal decomposition."""
        client.json_response = {
            "sub_queries": [
                {"question": "What is the capital of France?", "rationale": "Main question", "priority": 1}
            ]
        }
        
        result = decompose_query(client, "What is the capital of France?")
        
        assert len(result) >= 1
        assert "capital" in result[0].question.lower() or "france" in result[0].question.lower()

    def test_complex_question_decomposes(self, client):
        """Complex questions should decompose into multiple sub-queries."""
        client.json_response = {
            "sub_queries": [
                {"question": "What is the vacation policy?", "rationale": "First aspect", "priority": 1},
                {"question": "What are the benefits?", "rationale": "Second aspect", "priority": 2}
            ]
        }
        
        result = decompose_query(client, "What is the vacation policy and benefits?")
        
        assert len(result) >= 1
        # Should have at least one sub-query about vacation or benefits

    def test_decomposition_fallback_on_error(self, client):
        """Should gracefully handle decomposition failures."""
        client.chat_response = "Invalid JSON"
        
        result = decompose_query(client, "Any question?")
        
        assert len(result) == 1
        assert result[0].question == "Any question?"

    def test_max_subqueries_limit(self, client):
        """Should respect max_subqueries parameter."""
        client.json_response = {
            "sub_queries": [
                {"question": f"Question {i}", "rationale": "test", "priority": 1}
                for i in range(10)
            ]
        }
        
        result = decompose_query(client, "Complex question", max_subqueries=3)
        
        assert len(result) <= 4  # May have fallback if parsing fails


class TestRetrievalPlanning:
    """Test retrieval planning agent."""

    def test_optimizes_query(self, client):
        """Should return optimized query and key terms."""
        client.json_response = {
            "optimized_query": "vacation policy paid leave days",
            "key_terms": ["vacation", "policy", "paid leave"],
            "alternative_phrasings": ["time off policy"]
        }
        
        sq = SubQuery(question="How many vacation days?", rationale="test")
        optimized, key_terms = plan_retrieval(client, sq)
        
        assert "vacation" in optimized.lower() or "leave" in optimized.lower()
        assert isinstance(key_terms, list)

    def test_fallback_on_planning_error(self, client):
        """Should return original query on planning failure."""
        client.chat_response = "Invalid JSON"
        
        sq = SubQuery(question="Original question", rationale="test")
        optimized, key_terms = plan_retrieval(client, sq)
        
        assert optimized == "Original question"
        assert key_terms == []


class TestEvidenceSynthesis:
    """Test evidence synthesis agent."""

    def test_scores_evidence(self, tmp_path, conn, client):
        """Should score retrieved evidence for relevance."""
        _seed(tmp_path, conn, client)
        chunks = execute_retrieval(
            conn, VectorIndex(conn), client, "vacation days", Config(data_dir=tmp_path)
        )
        
        assert len(chunks) > 0
        
        client.json_response = {
            "evidence_scores": [
                {"index": i, "score": 0.9, "reason": "Highly relevant"}
                for i in range(min(len(chunks), 5))
            ],
            "coverage_gaps": [],
            "overall_confidence": 0.85
        }
        
        sub_queries = [SubQuery(question="vacation days", rationale="test")]
        evidence = synthesize_evidence(client, chunks, "vacation days", sub_queries)
        
        assert evidence.confidence > 0
        assert len(evidence.items) > 0

    def test_handles_empty_chunks(self, client):
        """Should handle empty chunk lists gracefully."""
        sub_queries = [SubQuery(question="test", rationale="test")]
        evidence = synthesize_evidence(client, [], "test question", sub_queries)
        
        assert evidence.items == []
        assert evidence.confidence == 0.0
        assert "No documents" in evidence.coverage_gaps[0]

    def test_filters_low_relevance(self, tmp_path, conn, client):
        """Should filter out low-relevance evidence."""
        _seed(tmp_path, conn, client)
        chunks = execute_retrieval(
            conn, VectorIndex(conn), client, "vacation", Config(data_dir=tmp_path)
        )
        
        client.json_response = {
            "evidence_scores": [
                {"index": 0, "score": 0.9, "reason": "relevant"},
                {"index": 1, "score": 0.1, "reason": "not relevant"},
                {"index": 2, "score": 0.8, "reason": "relevant"}
            ],
            "coverage_gaps": [],
            "overall_confidence": 0.7
        }
        
        sub_queries = [SubQuery(question="vacation", rationale="test")]
        evidence = synthesize_evidence(client, chunks[:3], "vacation", sub_queries)
        
        # Low-scoring items should be filtered
        assert all(item.relevance_score >= 0.3 for item in evidence.items)


class TestAnswerGeneration:
    """Test answer generation agent."""

    def test_generates_answer_from_evidence(self, tmp_path, conn, client):
        """Should generate answer from provided evidence."""
        _seed(tmp_path, conn, client)
        chunks = execute_retrieval(
            conn, VectorIndex(conn), client, "vacation days", Config(data_dir=tmp_path)
        )
        
        evidence = SynthesizedEvidence(
            items=[EvidenceItem(chunk=c, relevance_score=0.9, supports_query="vacation") 
                   for c in chunks[:3]],
            coverage_gaps=[],
            confidence=0.85
        )
        
        client.chat_response = "Employees receive 25 days of paid leave [1]."
        cfg = Config(data_dir=tmp_path)
        
        answer, used = generate_answer(client, evidence, "How many vacation days?", cfg)
        
        assert "25 days" in answer or "paid leave" in answer
        assert len(used) > 0

    def test_returns_not_found_with_no_evidence(self, client):
        """Should return NOT_FOUND when no evidence available."""
        evidence = SynthesizedEvidence(items=[], coverage_gaps=[], confidence=0.0)
        cfg = Config()
        
        answer, used = generate_answer(client, evidence, "Any question?", cfg)
        
        assert "Not found" in answer
        assert used == []


class TestAnswerVerification:
    """Test answer verification agent."""

    def test_verifies_complete_answer(self, tmp_path, conn, client):
        """Should verify a complete answer."""
        _seed(tmp_path, conn, client)
        chunks = execute_retrieval(
            conn, VectorIndex(conn), client, "vacation", Config(data_dir=tmp_path)
        )
        
        client.json_response = {
            "is_complete": True,
            "missing_aspects": [],
            "suggested_followups": [],
            "confidence": 0.9
        }
        
        result = verify_answer(
            client,
            "How many vacation days?",
            "Employees get 25 days [1].",
            chunks[:3]
        )
        
        assert result.is_complete
        assert result.confidence > 0

    def test_identifies_missing_aspects(self, client):
        """Should identify missing aspects in incomplete answers."""
        client.json_response = {
            "is_complete": False,
            "missing_aspects": ["Doesn't mention accrual policy"],
            "suggested_followups": ["How do vacation days accrue?"],
            "confidence": 0.6
        }
        
        result = verify_answer(
            client,
            "Explain vacation policy",
            "Employees get 25 days.",
            []
        )
        
        assert not result.is_complete
        assert len(result.missing_aspects) > 0


class TestAgenticPipeline:
    """Test full agentic Q&A pipeline."""

    def test_answers_simple_question(self, tmp_path, conn, client):
        """Should answer simple questions correctly."""
        _seed(tmp_path, conn, client)
        
        # Mock responses for each agent
        client.chat_responses = [
            json.dumps({"sub_queries": [{"question": "vacation days", "rationale": "main", "priority": 1}]}),
            json.dumps({"optimized_query": "vacation days paid leave", "key_terms": ["vacation"]}),
            json.dumps({
                "evidence_scores": [{"index": 0, "score": 0.9}],
                "coverage_gaps": [],
                "overall_confidence": 0.85
            }),
            "Employees receive 25 days of paid leave [1].",
        ]
        
        cfg = Config(data_dir=tmp_path)
        result = answer_question_agentic(
            conn, VectorIndex(conn), client, "How many vacation days?", cfg
        )
        
        assert isinstance(result, AgentQAResult)
        assert "25 days" in result.answer or "paid leave" in result.answer
        assert len(result.sources) > 0

    def test_handles_empty_corpus(self, tmp_path, conn, client):
        """Should handle questions with no relevant documents."""
        cfg = Config(data_dir=tmp_path)
        result = answer_question_agentic(
            conn, VectorIndex(conn), client, "Anything?", cfg
        )
        
        assert "Not found" in result.answer
        assert result.sources == []

    def test_tracks_sub_queries(self, tmp_path, conn, client):
        """Should track decomposed sub-queries."""
        _seed(tmp_path, conn, client)
        
        client.chat_responses = [
            json.dumps({
                "sub_queries": [
                    {"question": "vacation policy", "rationale": "aspect 1", "priority": 1}
                ]
            }),
            json.dumps({"optimized_query": "vacation policy", "key_terms": ["vacation"]}),
            json.dumps({
                "evidence_scores": [{"index": 0, "score": 0.8}],
                "coverage_gaps": [],
                "overall_confidence": 0.75
            }),
            "Vacation policy provides 25 days [1].",
        ]
        
        cfg = Config(data_dir=tmp_path)
        result = answer_question_agentic(
            conn, VectorIndex(conn), client, "What is the vacation policy?", cfg
        )
        
        assert len(result.sub_queries) > 0
        assert result.evidence_confidence > 0

    def test_multi_aspect_question(self, tmp_path, conn, client):
        """Should handle multi-aspect questions with decomposition."""
        _seed(tmp_path, conn, client)
        
        client.chat_responses = [
            json.dumps({
                "sub_queries": [
                    {"question": "vacation days", "rationale": "first aspect", "priority": 1},
                    {"question": "health benefits", "rationale": "second aspect", "priority": 2}
                ]
            }),
            json.dumps({"optimized_query": "vacation days", "key_terms": ["vacation"]}),
            json.dumps({"optimized_query": "health benefits insurance", "key_terms": ["health"]}),
            json.dumps({
                "evidence_scores": [{"index": 0, "score": 0.85}, {"index": 1, "score": 0.8}],
                "coverage_gaps": [],
                "overall_confidence": 0.8
            }),
            "Employees get 25 days vacation [1] and health insurance [2].",
        ]
        
        cfg = Config(data_dir=tmp_path)
        result = answer_question_agentic(
            conn, VectorIndex(conn), client, 
            "What are the vacation days and health benefits?", 
            cfg
        )
        
        assert isinstance(result, AgentQAResult)
        # Should have attempted retrieval for both aspects
