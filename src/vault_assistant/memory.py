"""Agent memory and caching system for fast, accurate Q&A.

This module provides:
- Short-term conversation memory (context window management)
- Long-term semantic memory (vector-based recall)
- Query result caching (LRU cache with TTL)
- Knowledge base integration (structured facts and relationships)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sqlite3 import Connection
from typing import Any

from .config import Config
from .ollama_client import OllamaClient
from .vectors import VectorIndex


@dataclass
class CacheEntry:
    """Cached query result with metadata."""
    question_hash: str
    answer: str
    sources: list[dict]
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    ttl_seconds: int = 3600  # 1 hour default
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class MemoryItem:
    """A unit of remembered information."""
    content: str
    embedding: list[float] | None = None
    source: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    importance: float = 1.0  # For prioritization
    tags: list[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """A single turn in conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    question_embedding: list[float] | None = None


@dataclass
class KnowledgeFact:
    """Structured knowledge fact."""
    subject: str
    predicate: str
    object: str
    source: str | None = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


class QueryCache:
    """LRU cache for query results with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, question: str, doc_ids: list[int] | None = None) -> str:
        """Create a deterministic cache key."""
        key_data = f"{question}::{sorted(doc_ids) if doc_ids else 'all'}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def get(self, question: str, doc_ids: list[int] | None = None) -> CacheEntry | None:
        """Retrieve cached result if available and not expired."""
        key = self._make_key(question, doc_ids)
        if key in self.cache:
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.touch()
            self.hits += 1
            return entry
        self.misses += 1
        return None
    
    def set(
        self,
        question: str,
        answer: str,
        sources: list[dict],
        doc_ids: list[int] | None = None,
        ttl: int | None = None,
    ) -> None:
        """Cache a query result."""
        key = self._make_key(question, doc_ids)
        
        # Evict oldest if at capacity
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        now = time.time()
        entry = CacheEntry(
            question_hash=key,
            answer=answer,
            sources=sources,
            created_at=now,
            last_accessed=now,
            ttl_seconds=ttl or self.default_ttl,
        )
        self.cache[key] = entry
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
    
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "expired_count": sum(1 for e in self.cache.values() if e.is_expired()),
        }


class AgentMemory:
    """Manages short-term and long-term memory for the agent."""
    
    def __init__(
        self,
        max_conversation_turns: int = 10,
        max_memory_items: int = 100,
    ):
        self.conversation_history: list[ConversationTurn] = []
        self.semantic_memory: list[MemoryItem] = []
        self.max_conversation_turns = max_conversation_turns
        self.max_memory_items = max_memory_items
    
    def add_user_message(self, content: str, embedding: list[float] | None = None) -> None:
        """Add a user message to conversation history."""
        turn = ConversationTurn(role="user", content=content, question_embedding=embedding)
        self.conversation_history.append(turn)
        self._trim_conversation()
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant response to conversation history."""
        turn = ConversationTurn(role="assistant", content=content)
        self.conversation_history.append(turn)
        self._trim_conversation()
    
    def _trim_conversation(self) -> None:
        """Keep only the most recent conversation turns."""
        if len(self.conversation_history) > self.max_conversation_turns:
            self.conversation_history = self.conversation_history[-self.max_conversation_turns:]
    
    def add_to_semantic_memory(
        self,
        content: str,
        source: str | None = None,
        importance: float = 1.0,
        tags: list[str] | None = None,
    ) -> None:
        """Add important information to long-term memory."""
        item = MemoryItem(
            content=content,
            source=source,
            importance=importance,
            tags=tags or [],
        )
        self.semantic_memory.append(item)
        
        # Trim oldest memories if over capacity
        if len(self.semantic_memory) > self.max_memory_items:
            # Keep most important memories
            self.semantic_memory.sort(key=lambda m: -m.importance)
            self.semantic_memory = self.semantic_memory[:self.max_memory_items]
    
    def get_context(self, include_history: bool = True, include_memory: bool = True) -> str:
        """Build context string from memory for the LLM."""
        parts = []
        
        if include_history and self.conversation_history:
            history_text = "\n".join(
                f"{turn.role}: {turn.content}"
                for turn in self.conversation_history[-5:]  # Last 5 turns
            )
            parts.append(f"Recent conversation:\n{history_text}")
        
        if include_memory and self.semantic_memory:
            # Get top memories by importance
            top_memories = sorted(self.semantic_memory, key=lambda m: -m.importance)[:10]
            memory_text = "\n".join(f"- {m.content}" for m in top_memories)
            parts.append(f"Important information:\n{memory_text}")
        
        return "\n\n".join(parts)
    
    def search_semantic_memory(
        self,
        query: str,
        index: VectorIndex | None = None,
        client: OllamaClient | None = None,
        k: int = 5,
    ) -> list[MemoryItem]:
        """Search semantic memory by similarity."""
        if not self.semantic_memory:
            return []
        
        if index is not None and client is not None and self.semantic_memory[0].embedding is not None:
            # Use vector search
            query_emb = client.embed([query])[0]
            # Would need to build temporary index - for now use simple text match
            pass
        
        # Fallback: keyword matching
        query_words = set(query.lower().split())
        scored = []
        for item in self.semantic_memory:
            words = set(item.content.lower().split())
            overlap = len(query_words & words)
            if overlap > 0:
                scored.append((overlap * item.importance, item))
        
        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored[:k]]
    
    def clear(self) -> None:
        """Clear all memory."""
        self.conversation_history.clear()
        self.semantic_memory.clear()


class KnowledgeBase:
    """Structured knowledge base for facts and relationships."""
    
    def __init__(self, conn: Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Create knowledge base tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_facts (
                id INTEGER PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject, predicate, object)
            );
            
            CREATE INDEX IF NOT EXISTS idx_knowledge_subject ON knowledge_facts(subject);
            CREATE INDEX IF NOT EXISTS idx_knowledge_object ON knowledge_facts(object);
            
            CREATE TABLE IF NOT EXISTS knowledge_entities (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                type TEXT,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()
    
    def add_fact(
        self,
        subject: str,
        predicate: str,
        object_val: str,
        source: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Add a fact to the knowledge base."""
        self.conn.execute(
            """
            INSERT INTO knowledge_facts (subject, predicate, object, source, confidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(subject, predicate, object) DO UPDATE SET
                confidence = MAX(excluded.confidence, knowledge_facts.confidence),
                source = COALESCE(excluded.source, knowledge_facts.source)
            """,
            (subject, predicate, object_val, source, confidence),
        )
        self.conn.commit()
    
    def add_entity(self, name: str, entity_type: str | None = None, description: str | None = None) -> None:
        """Add an entity to the knowledge base."""
        self.conn.execute(
            """
            INSERT INTO knowledge_entities (name, type, description)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                type = COALESCE(excluded.type, knowledge_entities.type),
                description = COALESCE(excluded.description, knowledge_entities.description)
            """,
            (name, entity_type, description),
        )
        self.conn.commit()
    
    def query_facts(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object_val: str | None = None,
        limit: int = 50,
    ) -> list[KnowledgeFact]:
        """Query facts by pattern."""
        conditions = []
        params = []
        
        if subject:
            conditions.append("subject LIKE ?")
            params.append(f"%{subject}%")
        if predicate:
            conditions.append("predicate LIKE ?")
            params.append(f"%{predicate}%")
        if object_val:
            conditions.append("object LIKE ?")
            params.append(f"%{object_val}%")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        rows = self.conn.execute(
            f"""
            SELECT subject, predicate, object, source, confidence, created_at
            FROM knowledge_facts
            WHERE {where_clause}
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        
        return [
            KnowledgeFact(
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
                source=r["source"],
                confidence=r["confidence"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]
    
    def find_related(self, entity: str, depth: int = 1) -> list[KnowledgeFact]:
        """Find facts related to an entity."""
        facts = []
        
        # Direct relationships
        direct = self.query_facts(subject=entity)
        direct.extend(self.query_facts(object_val=entity))
        facts.extend(direct)
        
        if depth > 1:
            # Expand to related entities
            related_entities = set()
            for f in direct:
                if f.subject != entity:
                    related_entities.add(f.subject)
                if f.object != entity:
                    related_entities.add(f.object)
            
            for rel_entity in related_entities:
                related = self.query_facts(subject=rel_entity)
                related.extend(self.query_facts(object_val=rel_entity))
                facts.extend(related)
        
        # Deduplicate
        seen = set()
        unique = []
        for f in facts:
            key = (f.subject, f.predicate, f.object)
            if key not in seen:
                seen.add(key)
                unique.append(f)
        
        return unique
    
    def extract_facts_from_text(
        self,
        text: str,
        client: OllamaClient,
        source: str | None = None,
    ) -> list[KnowledgeFact]:
        """Extract structured facts from text using LLM."""
        system_prompt = """You are a knowledge extraction assistant. Extract factual triplets (subject, predicate, object) from the given text.

Rules:
- Only extract clear, factual statements
- Use concise, normalized forms for subjects and objects
- Predicates should be simple verbs or relationships (e.g., "is", "works at", "located in")
- Output MUST be valid JSON matching the schema exactly"""
        
        schema = {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["subject", "predicate", "object"]
                    }
                }
            },
            "required": ["facts"]
        }
        
        user_prompt = f"Extract facts from this text:\n\n{text}"
        
        try:
            result = client.chat_json(system_prompt, user_prompt, schema, temperature=0.1)
            facts = []
            for f in result.get("facts", [])[:10]:  # Limit to 10 facts
                fact = KnowledgeFact(
                    subject=f["subject"].strip(),
                    predicate=f["predicate"].strip(),
                    object=f["object"].strip(),
                    source=source,
                    confidence=f.get("confidence", 1.0),
                )
                facts.append(fact)
                
                # Add to database
                self.add_fact(fact.subject, fact.predicate, fact.object, source, fact.confidence)
            
            return facts
        except Exception:
            return []
    
    def get_stats(self) -> dict[str, int]:
        """Get knowledge base statistics."""
        facts_count = self.conn.execute(
            "SELECT COUNT(*) as count FROM knowledge_facts"
        ).fetchone()["count"]
        entities_count = self.conn.execute(
            "SELECT COUNT(*) as count FROM knowledge_entities"
        ).fetchone()["count"]
        
        return {
            "facts": facts_count,
            "entities": entities_count,
        }
