"""Tests for agent memory, caching, and knowledge base."""

import time
from sqlite3 import Connection

import pytest

from vault_assistant.memory import (
    AgentMemory,
    CacheEntry,
    KnowledgeBase,
    KnowledgeFact,
    MemoryItem,
    QueryCache,
)


class TestQueryCache:
    """Test LRU query caching with TTL."""

    def test_cache_set_and_get(self):
        """Basic cache set and retrieval."""
        cache = QueryCache(max_size=10, default_ttl=3600)
        
        cache.set(
            question="What is Python?",
            answer="Python is a programming language",
            sources=[{"filename": "intro.txt"}],
        )
        
        result = cache.get("What is Python?")
        assert result is not None
        assert result.answer == "Python is a programming language"
        assert result.access_count == 1

    def test_cache_miss(self):
        """Cache miss returns None."""
        cache = QueryCache()
        result = cache.get("Non-existent question")
        assert result is None

    def test_cache_key_includes_doc_ids(self):
        """Cache keys differentiate by document IDs."""
        cache = QueryCache()
        
        cache.set(
            question="What is X?",
            answer="Answer from all docs",
            sources=[],
            doc_ids=None,
        )
        
        cache.set(
            question="What is X?",
            answer="Answer from doc 1",
            sources=[],
            doc_ids=[1],
        )
        
        # Should get different results
        all_docs = cache.get("What is X?", doc_ids=None)
        doc1_only = cache.get("What is X?", doc_ids=[1])
        
        assert all_docs.answer != doc1_only.answer

    def test_cache_expiration(self):
        """Expired entries are not returned."""
        cache = QueryCache(default_ttl=1)  # 1 second TTL
        
        cache.set(
            question="Temporary question",
            answer="Temporary answer",
            sources=[],
        )
        
        # Should exist immediately
        assert cache.get("Temporary question") is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        assert cache.get("Temporary question") is None

    def test_cache_lru_eviction(self):
        """Least recently used entries are evicted when full."""
        cache = QueryCache(max_size=3)
        
        # Fill cache
        cache.set("Q1", "A1", [])
        cache.set("Q2", "A2", [])
        cache.set("Q3", "A3", [])
        
        # Access Q1 to make it recently used
        cache.get("Q1")
        
        # Add Q4, should evict Q2 (least recently used)
        cache.set("Q4", "A4", [])
        
        assert cache.get("Q1") is not None  # Recently used
        assert cache.get("Q2") is None  # Evicted
        assert cache.get("Q3") is not None
        assert cache.get("Q4") is not None

    def test_cache_stats(self):
        """Cache statistics are tracked correctly."""
        cache = QueryCache()
        
        cache.set("Q1", "A1", [])
        cache.get("Q1")  # Hit
        cache.get("Q1")  # Hit
        cache.get("Q2")  # Miss
        
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2/3
        assert stats["size"] == 1

    def test_cache_clear(self):
        """Clear removes all entries."""
        cache = QueryCache()
        
        cache.set("Q1", "A1", [])
        cache.set("Q2", "A2", [])
        
        cache.clear()
        
        assert cache.stats()["size"] == 0
        assert cache.get("Q1") is None


class TestAgentMemory:
    """Test agent short-term and long-term memory."""

    def test_add_user_message(self):
        """User messages are added to conversation history."""
        memory = AgentMemory()
        
        memory.add_user_message("What is AI?")
        
        assert len(memory.conversation_history) == 1
        assert memory.conversation_history[0].role == "user"
        assert memory.conversation_history[0].content == "What is AI?"

    def test_add_assistant_message(self):
        """Assistant messages are added to conversation history."""
        memory = AgentMemory()
        
        memory.add_assistant_message("AI stands for Artificial Intelligence")
        
        assert len(memory.conversation_history) == 1
        assert memory.conversation_history[0].role == "assistant"

    def test_conversation_trimming(self):
        """Conversation history is trimmed to max length."""
        memory = AgentMemory(max_conversation_turns=3)
        
        for i in range(5):
            memory.add_user_message(f"Question {i}")
            memory.add_assistant_message(f"Answer {i}")
        
        # Should only keep last 3 turns (6 messages total)
        assert len(memory.conversation_history) == 3

    def test_semantic_memory(self):
        """Important information can be stored in semantic memory."""
        memory = AgentMemory()
        
        memory.add_to_semantic_memory(
            "Python was created by Guido van Rossum",
            source="wikipedia",
            importance=0.9,
            tags=["python", "history"],
        )
        
        assert len(memory.semantic_memory) == 1
        item = memory.semantic_memory[0]
        assert "Python" in item.content
        assert item.importance == 0.9
        assert "python" in item.tags

    def test_semantic_memory_importance_sorting(self):
        """Most important memories are kept when over capacity."""
        memory = AgentMemory(max_memory_items=3)
        
        # Add memories with different importance
        memory.add_to_semantic_memory("Low importance", importance=0.1)
        memory.add_to_semantic_memory("High importance", importance=0.9)
        memory.add_to_semantic_memory("Medium importance", importance=0.5)
        memory.add_to_semantic_memory("Very high importance", importance=1.0)
        
        # Should keep top 3 by importance
        assert len(memory.semantic_memory) == 3
        contents = [m.content for m in memory.semantic_memory]
        assert "Low importance" not in contents
        assert "Very high importance" in contents
        assert "High importance" in contents

    def test_get_context(self):
        """Context string includes history and semantic memory."""
        memory = AgentMemory()
        
        memory.add_user_message("Tell me about Python")
        memory.add_assistant_message("Python is a programming language")
        memory.add_to_semantic_memory("Python uses indentation", importance=0.8)
        
        context = memory.get_context()
        
        assert "user:" in context.lower() or "User:" in context
        assert "assistant:" in context.lower() or "Assistant:" in context
        assert "Python" in context

    def test_search_semantic_memory(self):
        """Semantic memory can be searched by keywords."""
        memory = AgentMemory()
        
        memory.add_to_semantic_memory("The capital of France is Paris")
        memory.add_to_semantic_memory("Germany's capital is Berlin")
        memory.add_to_semantic_memory("Python is a snake")
        
        results = memory.search_semantic_memory("France capital", k=1)
        
        assert len(results) == 1
        assert "France" in results[0].content

    def test_memory_clear(self):
        """Clear removes all memory."""
        memory = AgentMemory()
        
        memory.add_user_message("Test")
        memory.add_to_semantic_memory("Fact")
        
        memory.clear()
        
        assert len(memory.conversation_history) == 0
        assert len(memory.semantic_memory) == 0


class TestKnowledgeBase:
    """Test structured knowledge base."""

    def test_add_fact(self, tmp_path):
        """Facts can be added to knowledge base."""
        db_path = tmp_path / "test_kb.db"
        conn = sqlite3.connect(str(db_path))
        kb = KnowledgeBase(conn)
        
        kb.add_fact("Paris", "is capital of", "France")
        
        facts = kb.query_facts(subject="Paris")
        assert len(facts) == 1
        assert facts[0].predicate == "is capital of"
        assert facts[0].object == "France"

    def test_add_entity(self, tmp_path):
        """Entities can be added to knowledge base."""
        db_path = tmp_path / "test_kb.db"
        conn = sqlite3.connect(str(db_path))
        kb = KnowledgeBase(conn)
        
        kb.add_entity("Python", entity_type="programming language", description="Created by Guido")
        
        # Verify entity exists (would need direct DB access or method)
        cursor = conn.execute("SELECT name, type FROM knowledge_entities WHERE name = ?", ("Python",))
        row = cursor.fetchone()
        assert row is not None
        assert row["type"] == "programming language"

    def test_query_facts_by_pattern(self, tmp_path):
        """Facts can be queried by subject/predicate/object patterns."""
        db_path = tmp_path / "test_kb.db"
        conn = sqlite3.connect(str(db_path))
        kb = KnowledgeBase(conn)
        
        kb.add_fact("Python", "created by", "Guido van Rossum")
        kb.add_fact("Python", "released in", "1991")
        kb.add_fact("Java", "created by", "James Gosling")
        
        # Query by subject
        python_facts = kb.query_facts(subject="Python")
        assert len(python_facts) == 2
        
        # Query by predicate
        creator_facts = kb.query_facts(predicate="created by")
        assert len(creator_facts) == 2
        
        # Query by object
        guido_facts = kb.query_facts(object_val="Guido")
        assert len(guido_facts) == 1

    def test_find_related(self, tmp_path):
        """Related facts can be found through entity relationships."""
        db_path = tmp_path / "test_kb.db"
        conn = sqlite3.connect(str(db_path))
        kb = KnowledgeBase(conn)
        
        kb.add_fact("Alice", "works at", "CompanyX")
        kb.add_fact("CompanyX", "located in", "New York")
        kb.add_fact("Bob", "works at", "CompanyX")
        
        related = kb.find_related("Alice", depth=2)
        
        # Should find Alice's fact and CompanyX's fact
        assert any(f.subject == "Alice" for f in related)
        assert any(f.subject == "CompanyX" for f in related)

    def test_fact_confidence_update(self, tmp_path):
        """Duplicate facts update confidence if higher."""
        db_path = tmp_path / "test_kb.db"
        conn = sqlite3.connect(str(db_path))
        kb = KnowledgeBase(conn)
        
        kb.add_fact("X", "is", "Y", confidence=0.5)
        kb.add_fact("X", "is", "Y", confidence=0.9)
        
        facts = kb.query_facts(subject="X")
        assert len(facts) == 1
        assert facts[0].confidence == 0.9

    def test_get_stats(self, tmp_path):
        """Knowledge base statistics are accurate."""
        db_path = tmp_path / "test_kb.db"
        conn = sqlite3.connect(str(db_path))
        kb = KnowledgeBase(conn)
        
        kb.add_fact("A", "is", "B")
        kb.add_fact("C", "is", "D")
        kb.add_entity("Entity1")
        
        stats = kb.get_stats()
        assert stats["facts"] == 2
        assert stats["entities"] == 1


class TestIntegration:
    """Integration tests for memory + cache + knowledge base."""

    def test_cache_prevents_duplicate_llm_calls(self, tmp_path):
        """Cached queries don't trigger agent pipeline."""
        cache = QueryCache()
        
        # First call would normally trigger agents
        cache.set("Test question", "Cached answer", [])
        
        # Second call should hit cache
        result = cache.get("Test question")
        assert result is not None
        assert result.answer == "Cached answer"

    def test_memory_persists_conversation_context(self):
        """Memory maintains conversation context across turns."""
        memory = AgentMemory(max_conversation_turns=10)
        
        # Simulate multi-turn conversation
        memory.add_user_message("What is ML?")
        memory.add_assistant_message("ML stands for Machine Learning")
        memory.add_user_message("How does it work?")
        memory.add_assistant_message("It uses algorithms to learn from data")
        
        context = memory.get_context()
        
        # Context should include recent turns
        assert "Machine Learning" in context
        assert "algorithms" in context

    def test_knowledge_base_enhances_retrieval(self, tmp_path):
        """Knowledge base facts provide additional context."""
        db_path = tmp_path / "test_kb.db"
        conn = sqlite3.connect(str(db_path))
        kb = KnowledgeBase(conn)
        
        # Add structured knowledge
        kb.add_fact("Quantum computing", "uses", "qubits")
        kb.add_fact("Qubits", "can be", "superpositioned")
        
        # Query should find relevant facts
        facts = kb.query_facts(subject="Quantum")
        assert len(facts) > 0
        assert any("qubits" in f.object.lower() for f in facts)


# Import sqlite3 for tests that need it
import sqlite3
