"""Query cache, agent memory, and knowledge base — the persistence layer the
agentic Q&A pipeline (agents.py) optionally reads/writes through its
``cache``/``memory``/``knowledge_base`` parameters. This module knows nothing
about agents.py or qa.py: the dependency arrow points one way only.

QueryCache is in-process/in-memory only (a cache, not a store — losing it on
restart is correct; entries are additionally invalidated the moment the
corpus changes, via the same ingest-generation counter vectors.py already
uses). AgentMemory and KnowledgeBase are persisted in the same SQLite
database as everything else (see db.py's schema), following this codebase's
existing conn-first, dataclass-with-to_dict conventions (reminders.py).
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from sqlite3 import Connection

from .chunking import approx_tokens
from .providers import LLMClient

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1}


# ---------------------------------------------------------------------------
# Query cache
# ---------------------------------------------------------------------------


@dataclass
class CachedResult:
    answer: str
    sources: list[dict]
    created_at: float
    expires_at: float
    generation: int | None
    hits: int = 0


class QueryCache:
    """In-process LRU + TTL cache, keyed by exact question text."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._data: OrderedDict[str, CachedResult] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, question: str, generation: int | None = None) -> CachedResult | None:
        with self._lock:
            entry = self._data.get(question)
            now = time.time()
            stale = entry is not None and (
                entry.expires_at < now or (generation is not None and entry.generation != generation)
            )
            if entry is None or stale:
                if stale:
                    del self._data[question]
                self._misses += 1
                return None
            entry.hits += 1
            self._data.move_to_end(question)
            self._hits += 1
            return entry

    def set(
        self,
        question: str,
        answer: str,
        sources: list[dict],
        ttl: float | None = None,
        generation: int | None = None,
    ) -> None:
        now = time.time()
        entry = CachedResult(
            answer=answer,
            sources=sources,
            created_at=now,
            expires_at=now + (ttl if ttl is not None else self.default_ttl),
            generation=generation,
        )
        with self._lock:
            self._data[question] = entry
            self._data.move_to_end(question)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._data),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0


# ---------------------------------------------------------------------------
# Agent memory: short-term conversation + long-term semantic memory
# ---------------------------------------------------------------------------


@dataclass
class ConversationTurn:
    id: int
    role: str
    content: str
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(timespec="seconds"),
        }


@dataclass
class MemoryItem:
    id: int
    text: str
    source: str | None
    importance: float
    tags: list[str]
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "importance": self.importance,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(timespec="seconds"),
        }


def _row_to_turn(row) -> ConversationTurn:
    return ConversationTurn(
        id=row["id"], role=row["role"], content=row["content"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _add_turn(conn: Connection, role: str, content: str, max_turns: int, now: datetime | None) -> ConversationTurn:
    now = now or datetime.now()
    cur = conn.execute(
        "INSERT INTO conversation_turns(role, content, created_at) VALUES(?, ?, ?)",
        (role, content, now.isoformat(timespec="seconds")),
    )
    conn.execute(
        "DELETE FROM conversation_turns WHERE id NOT IN "
        "(SELECT id FROM conversation_turns ORDER BY id DESC LIMIT ?)",
        (max_turns,),
    )
    conn.commit()
    return ConversationTurn(cur.lastrowid, role, content, now)


def add_user_message(conn: Connection, content: str, max_turns: int = 10, now: datetime | None = None) -> ConversationTurn:
    return _add_turn(conn, "user", content, max_turns, now)


def add_assistant_message(conn: Connection, content: str, max_turns: int = 10, now: datetime | None = None) -> ConversationTurn:
    return _add_turn(conn, "assistant", content, max_turns, now)


def get_recent_turns(conn: Connection, limit: int = 10) -> list[ConversationTurn]:
    rows = conn.execute("SELECT * FROM conversation_turns ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_turn(r) for r in reversed(rows)]


def clear_conversation(conn: Connection) -> None:
    conn.execute("DELETE FROM conversation_turns")
    conn.commit()


def _row_to_memory_item(row) -> MemoryItem:
    tags = [t for t in (row["tags"] or "").split(",") if t]
    return MemoryItem(
        id=row["id"], text=row["text"], source=row["source"],
        importance=row["importance"], tags=tags,
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def add_to_semantic_memory(
    conn: Connection,
    text: str,
    source: str | None = None,
    importance: float = 0.5,
    tags: list[str] | None = None,
    max_items: int = 100,
    now: datetime | None = None,
) -> MemoryItem:
    now = now or datetime.now()
    importance = max(0.0, min(1.0, importance))
    tags = [t.strip() for t in (tags or []) if t.strip()]
    cur = conn.execute(
        "INSERT INTO semantic_memory(text, source, importance, tags, created_at) VALUES(?, ?, ?, ?, ?)",
        (text, source, importance, ",".join(tags), now.isoformat(timespec="seconds")),
    )
    conn.execute(
        "DELETE FROM semantic_memory WHERE id NOT IN "
        "(SELECT id FROM semantic_memory ORDER BY importance DESC, created_at DESC LIMIT ?)",
        (max_items,),
    )
    conn.commit()
    return MemoryItem(cur.lastrowid, text, source, importance, tags, now)


def search_semantic_memory(conn: Connection, query: str, limit: int = 5) -> list[MemoryItem]:
    """Keyword-overlap search over the (small, capped) semantic memory table;
    falls back to importance-only ranking when nothing overlaps, rather than
    returning nothing. No embedding call — the table is bounded to max_items,
    small enough that a linear scan needs no index."""
    rows = conn.execute("SELECT * FROM semantic_memory").fetchall()
    if not rows:
        return []
    query_tokens = _tokenize(query)
    scored = []
    for r in rows:
        item = _row_to_memory_item(r)
        overlap = len(query_tokens & _tokenize(item.text)) if query_tokens else 0
        scored.append((overlap, item.importance, item))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [item for _, _, item in scored[:limit]]


def clear_semantic_memory(conn: Connection) -> None:
    conn.execute("DELETE FROM semantic_memory")
    conn.commit()


def memory_stats(conn: Connection) -> dict:
    turns = conn.execute("SELECT COUNT(*) AS n FROM conversation_turns").fetchone()["n"]
    items = conn.execute("SELECT COUNT(*) AS n FROM semantic_memory").fetchone()["n"]
    return {"conversation_turns": turns, "semantic_memory_items": items}


class AgentMemory:
    def __init__(
        self,
        conn: Connection,
        max_turns: int = 10,
        max_items: int = 100,
        context_budget: int = 400,
        top_k_facts: int = 5,
    ) -> None:
        self.conn = conn
        self.max_turns = max_turns
        self.max_items = max_items
        self.context_budget = context_budget
        self.top_k_facts = top_k_facts

    def add_user_message(self, content: str) -> ConversationTurn:
        return add_user_message(self.conn, content, self.max_turns)

    def add_assistant_message(self, content: str) -> ConversationTurn:
        return add_assistant_message(self.conn, content, self.max_turns)

    def add_to_semantic_memory(
        self, text: str, source: str | None = None, importance: float = 0.5, tags: list[str] | None = None
    ) -> MemoryItem:
        return add_to_semantic_memory(self.conn, text, source, importance, tags, self.max_items)

    def get_context(self, current_question: str | None = None) -> str:
        """Bounded context block: recent turns (newest always kept, even if it
        alone exceeds its share of the budget) plus top-matching semantic
        memory, sized against context_budget via the same char/4 token
        estimate qa.py uses for retrieval evidence."""
        turns = get_recent_turns(self.conn, self.max_turns)
        reserve = int(self.context_budget * 0.3)
        turn_budget = self.context_budget - reserve

        kept: list[str] = []
        spent = 0
        for turn in reversed(turns):
            line = f"{turn.role}: {turn.content}"
            cost = approx_tokens(line)
            if kept and spent + cost > turn_budget:
                break
            kept.append(line)
            spent += cost
        kept.reverse()

        fact_lines: list[str] = []
        remaining = self.context_budget - spent
        if remaining > 0 and self.top_k_facts > 0:
            query = current_question or (turns[-1].content if turns else "")
            for item in search_semantic_memory(self.conn, query, limit=self.top_k_facts):
                line = f"known: {item.text}"
                cost = approx_tokens(line)
                if fact_lines and cost > remaining:
                    break
                fact_lines.append(line)
                remaining -= cost

        sections = []
        if kept:
            sections.append("Recent conversation:\n" + "\n".join(kept))
        if fact_lines:
            sections.append("Remembered facts:\n" + "\n".join(fact_lines))
        return "\n\n".join(sections)

    def clear(self) -> None:
        clear_conversation(self.conn)
        clear_semantic_memory(self.conn)

    def stats(self) -> dict:
        return memory_stats(self.conn)


# ---------------------------------------------------------------------------
# Knowledge base: subject-predicate-object facts + entities
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    id: int
    name: str
    entity_type: str | None
    description: str | None
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "created_at": self.created_at.isoformat(timespec="seconds"),
        }


@dataclass
class Fact:
    id: int
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str | None
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(timespec="seconds"),
        }


FACT_SCHEMA = {
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
                    "confidence": {"type": "number"},
                },
                "required": ["subject", "predicate", "object", "confidence"],
            },
        }
    },
    "required": ["facts"],
}

FACT_SYSTEM_PROMPT = """You extract factual subject-predicate-object triples from text. Rules:
- subject/predicate/object: short noun phrases; predicate is a short relation phrase (e.g. "created by").
- confidence: 0.0-1.0, your certainty this is explicitly, unambiguously stated in the text.
- Only extract facts explicitly stated. Never infer, guess, or add outside knowledge.
- If there are no clear facts, return an empty facts array."""


def _row_to_fact(row) -> Fact:
    return Fact(
        id=row["id"], subject=row["subject"], predicate=row["predicate"], object=row["object"],
        confidence=row["confidence"], source=row["source"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def add_fact(
    conn: Connection,
    subject: str,
    predicate: str,
    object: str,  # noqa: A002 — matches the domain term (subject/predicate/object triple)
    confidence: float = 1.0,
    source: str | None = None,
    now: datetime | None = None,
) -> Fact:
    now = now or datetime.now()
    confidence = max(0.0, min(1.0, confidence))
    conn.execute(
        "INSERT INTO kb_facts(subject, predicate, object, confidence, source, created_at) "
        "VALUES(?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(subject, predicate, object) DO UPDATE SET "
        "confidence = MAX(confidence, excluded.confidence), "
        "source = COALESCE(excluded.source, kb_facts.source)",
        (subject, predicate, object, confidence, source, now.isoformat(timespec="seconds")),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM kb_facts WHERE subject = ? AND predicate = ? AND object = ?",
        (subject, predicate, object),
    ).fetchone()
    return _row_to_fact(row)


def query_facts(
    conn: Connection,
    subject: str | None = None,
    predicate: str | None = None,
    object: str | None = None,  # noqa: A002
    min_confidence: float | None = None,
) -> list[Fact]:
    clauses: list[str] = []
    params: list = []
    for col, val in (("subject", subject), ("predicate", predicate), ("object", object)):
        if val is not None:
            clauses.append(f"{col} = ?")
            params.append(val)
    if min_confidence is not None:
        clauses.append("confidence >= ?")
        params.append(min_confidence)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(f"SELECT * FROM kb_facts {where} ORDER BY confidence DESC", params).fetchall()
    return [_row_to_fact(r) for r in rows]


def _row_to_entity(row) -> Entity:
    return Entity(
        id=row["id"], name=row["name"], entity_type=row["entity_type"],
        description=row["description"], created_at=datetime.fromisoformat(row["created_at"]),
    )


def add_entity(
    conn: Connection,
    name: str,
    entity_type: str | None = None,
    description: str | None = None,
    now: datetime | None = None,
) -> Entity:
    now = now or datetime.now()
    conn.execute(
        "INSERT INTO kb_entities(name, entity_type, description, created_at, updated_at) "
        "VALUES(?, ?, ?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET "
        "entity_type = COALESCE(excluded.entity_type, kb_entities.entity_type), "
        "description = COALESCE(excluded.description, kb_entities.description), "
        "updated_at = excluded.updated_at",
        (name, entity_type, description, now.isoformat(timespec="seconds"), now.isoformat(timespec="seconds")),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM kb_entities WHERE name = ?", (name,)).fetchone()
    return _row_to_entity(row)


def find_related(conn: Connection, entity: str, depth: int = 2, limit: int = 50) -> list[Fact]:
    """Bounded, cycle-safe BFS over the fact graph out to `depth` hops."""
    visited = {entity.lower()}
    frontier = {entity}
    facts_by_id: dict[int, Fact] = {}
    for _ in range(max(depth, 0)):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for node in frontier:
            rows = conn.execute("SELECT * FROM kb_facts WHERE subject = ? OR object = ?", (node, node)).fetchall()
            for r in rows:
                fact = _row_to_fact(r)
                facts_by_id[fact.id] = fact
                other = fact.object if fact.subject.lower() == node.lower() else fact.subject
                if other.lower() not in visited:
                    visited.add(other.lower())
                    next_frontier.add(other)
        frontier = next_frontier
    return sorted(facts_by_id.values(), key=lambda f: -f.confidence)[:limit]


def extract_facts_from_text(conn: Connection, client: LLMClient, text: str, source: str | None = None) -> list[Fact]:
    result = client.chat_json(FACT_SYSTEM_PROMPT, f"Extract facts from:\n\n{text}", FACT_SCHEMA)
    out: list[Fact] = []
    for raw in result.get("facts", []):
        subject = (raw.get("subject") or "").strip()
        predicate = (raw.get("predicate") or "").strip()
        obj = (raw.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue
        try:
            confidence = float(raw.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))
        out.append(add_fact(conn, subject, predicate, obj, confidence=confidence, source=source))
    return out


def clear_facts(conn: Connection) -> None:
    conn.execute("DELETE FROM kb_facts")
    conn.commit()


def clear_entities(conn: Connection) -> None:
    conn.execute("DELETE FROM kb_entities")
    conn.commit()


def kb_stats(conn: Connection) -> dict:
    facts = conn.execute("SELECT COUNT(*) AS n, AVG(confidence) AS avg FROM kb_facts").fetchone()
    entities = conn.execute("SELECT COUNT(*) AS n FROM kb_entities").fetchone()["n"]
    return {
        "facts": facts["n"],
        "entities": entities,
        "avg_confidence": facts["avg"] if facts["avg"] is not None else 0.0,
    }


class KnowledgeBase:
    def __init__(self, conn: Connection, find_related_limit: int = 50) -> None:
        self.conn = conn
        self.find_related_limit = find_related_limit

    def add_fact(self, subject: str, predicate: str, object: str, confidence: float = 1.0, source: str | None = None) -> Fact:  # noqa: A002
        return add_fact(self.conn, subject, predicate, object, confidence, source)

    def add_entity(self, name: str, entity_type: str | None = None, description: str | None = None) -> Entity:
        return add_entity(self.conn, name, entity_type, description)

    def query_facts(
        self, subject: str | None = None, predicate: str | None = None,
        object: str | None = None, min_confidence: float | None = None,  # noqa: A002
    ) -> list[Fact]:
        return query_facts(self.conn, subject, predicate, object, min_confidence)

    def find_related(self, entity: str, depth: int = 2, limit: int | None = None) -> list[Fact]:
        return find_related(self.conn, entity, depth, limit if limit is not None else self.find_related_limit)

    def extract_facts_from_text(self, client: LLMClient, text: str, source: str | None = None) -> list[Fact]:
        return extract_facts_from_text(self.conn, client, text, source)

    def clear(self) -> None:
        clear_facts(self.conn)
        clear_entities(self.conn)

    def stats(self) -> dict:
        return kb_stats(self.conn)
