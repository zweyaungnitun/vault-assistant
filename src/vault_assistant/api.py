"""Local FastAPI server: JSON API plus the single-page web UI.

Binds to localhost only — nothing here is meant to be reachable off-device.
A background thread fires macOS notifications for due reminders while the
server runs.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import db, reminders, summarize
from .actions import extract_actions
from .agents import answer_question_agentic, AgentQAResult
from .config import Config, load_config, setup_logging
from .ingest import ingest_paths
from .memory import AgentMemory, KnowledgeBase, QueryCache
from .ollama_client import OllamaClient
from .pii import scan as pii_scan
from .qa import answer_question
from .vectors import VectorIndex

logger = logging.getLogger("vault.api")

WEB_DIR = Path(__file__).parent / "web"


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_ids: list[int] | None = None


class IngestRequest(BaseModel):
    paths: list[str]
    force: bool = False


class SummarizeRequest(BaseModel):
    doc_ids: list[int] = Field(min_length=1, max_length=3)
    mode: str = "paragraph"


class ActionsRequest(BaseModel):
    text: str | None = None
    doc_id: int | None = None


class PIIRequest(BaseModel):
    text: str | None = None
    doc_id: int | None = None
    use_model: bool = True


class ReminderRequest(BaseModel):
    text: str = Field(min_length=1)


def _notifier_loop(app: FastAPI) -> None:
    while True:
        try:
            for r in reminders.due_reminders(app.state.conn):
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        f'display notification {_as_script_string(r.title)} with title "Vault reminder"',
                    ],
                    capture_output=True,
                    timeout=10,
                )
                reminders.set_status(app.state.conn, r.id, "notified")
                logger.info("notified reminder %d: %s", r.id, r.title)
        except Exception as exc:  # noqa: BLE001 — notifier must never die
            logger.warning("reminder notifier error: %s", exc)
        time.sleep(30)


def _as_script_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _doc_or_text(app: FastAPI, text: str | None, doc_id: int | None) -> str:
    if text:
        return text
    if doc_id is not None:
        try:
            return summarize.document_text(app.state.conn, doc_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
    raise HTTPException(400, "provide either text or doc_id")


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    setup_logging(cfg)
    app = FastAPI(title="Vault Assistant", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.cfg = cfg
    app.state.conn = db.connect(cfg.db_path)
    app.state.client = OllamaClient.from_config(cfg)
    app.state.index = VectorIndex(app.state.conn)
    app.state.ingest_lock = threading.Lock()
    
    # Initialize agentic components
    app.state.cache = QueryCache(max_size=1000, default_ttl=3600)
    app.state.memory = AgentMemory()
    app.state.knowledge_base = KnowledgeBase(app.state.conn)

    threading.Thread(target=_notifier_loop, args=(app,), daemon=True).start()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "index_enhanced.html").read_text()

    @app.get("/api/status")
    def status() -> dict:
        conn = app.state.conn
        client = app.state.client
        up = client.is_up()
        return {
            "ollama_up": up,
            "missing_models": client.missing_models() if up else [],
            "documents": conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"],
            "chunks": conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"],
            "errors": conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE status = 'error'"
            ).fetchone()["n"],
        }

    @app.get("/api/documents")
    def documents() -> list[dict]:
        return db.list_documents(app.state.conn)

    @app.post("/api/ingest")
    def ingest(req: IngestRequest) -> dict:
        paths = [Path(p) for p in req.paths]
        missing = [str(p) for p in paths if not p.expanduser().exists()]
        if missing:
            raise HTTPException(400, f"paths do not exist: {missing}")
        with app.state.ingest_lock:
            report = ingest_paths(app.state.conn, app.state.client, paths, force=req.force)
        return {
            "added": report.added,
            "updated": report.updated,
            "skipped": report.skipped,
            "removed": report.removed,
            "failed": [{"path": p, "error": e} for p, e in report.failed],
        }

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        result = answer_question(
            app.state.conn, app.state.index, app.state.client, req.question, app.state.cfg,
            doc_ids=req.doc_ids
        )
        return {
            "answer": result.answer,
            "sources": [
                {"filename": s.filename, "path": s.path, "page": s.page} for s in result.sources
            ],
        }

    @app.post("/api/ask-enhanced")
    def ask_enhanced(req: AskEnhancedRequest) -> dict:
        """Enhanced Q&A with agentic architecture, memory, cache, and knowledge base."""
        cache = app.state.cache if req.use_cache else None
        memory = app.state.memory if req.use_memory else None
        kb = app.state.knowledge_base if req.use_knowledge else None
        
        result: AgentQAResult = answer_question_agentic(
            app.state.conn,
            app.state.index,
            app.state.client,
            req.question,
            app.state.cfg,
            doc_ids=req.doc_ids,
            cache=cache,
            memory=memory,
            knowledge_base=kb,
        )
        
        return {
            "answer": result.answer,
            "sources": [
                {"filename": s.filename, "path": s.path, "page": s.page} for s in result.sources
            ],
            "cache_hit": result.iterations == 0,
            "iterations": result.iterations,
            "confidence": result.evidence_confidence,
        }

    @app.get("/api/memory/stats")
    def memory_stats() -> dict:
        """Get memory and cache statistics."""
        cache = app.state.cache
        memory = app.state.memory
        
        return {
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "cache_size": len(cache.cache),
            "memory_items": len(memory.semantic_memory) if memory.semantic_memory else 0,
            "conversation_turns": len(memory.conversation_history) if memory.conversation_history else 0,
        }

    @app.post("/api/cache/clear")
    def clear_cache() -> dict:
        """Clear the query cache."""
        app.state.cache.cache.clear()
        return {"ok": True}

    @app.post("/api/memory/clear")
    def clear_memory() -> dict:
        """Clear agent memory."""
        app.state.memory.conversation_history.clear()
        if app.state.memory.semantic_memory:
            app.state.memory.semantic_memory.clear()
        return {"ok": True}

    @app.get("/api/knowledge")
    def get_knowledge() -> list[dict]:
        """Get all knowledge base facts."""
        facts = app.state.knowledge_base.query_facts(limit=100)
        return [{"subject": f.subject, "predicate": f.predicate, "object": f.object} for f in facts]

    @app.post("/api/knowledge")
    def add_knowledge(req: KnowledgeFactRequest) -> dict:
        """Add a new fact to the knowledge base."""
        app.state.knowledge_base.add_fact(req.subject, req.predicate, req.object)
        return {"ok": True}

    @app.post("/api/knowledge/clear")
    def clear_knowledge() -> dict:
        """Clear all knowledge base facts."""
        app.state.knowledge_base.clear()
        return {"ok": True}

    @app.post("/api/settings")
    def update_settings(req: SettingsRequest) -> dict:
        """Update application settings."""
        cfg = app.state.cfg
        
        if req.gen_model is not None:
            cfg.gen_model = req.gen_model
        if req.embed_model is not None:
            cfg.embed_model = req.embed_model
        if req.ollama_url is not None:
            cfg.ollama_url = req.ollama_url
        if req.vector_top_k is not None:
            cfg.vector_top_k = req.vector_top_k
        if req.keyword_top_k is not None:
            cfg.keyword_top_k = req.keyword_top_k
        if req.context_token_budget is not None:
            cfg.context_token_budget = req.context_token_budget
        if req.cache_ttl is not None:
            app.state.cache.default_ttl = req.cache_ttl
        if req.cache_max_size is not None:
            # Recreate cache with new size
            old_cache = app.state.cache
            app.state.cache = QueryCache(max_size=req.cache_max_size, default_ttl=old_cache.default_ttl)
        
        # Update client if needed
        if req.ollama_url or req.gen_model or req.embed_model:
            app.state.client = OllamaClient.from_config(cfg)
        
        return {"ok": True}

    @app.post("/api/summarize")
    def summarize_docs(req: SummarizeRequest) -> dict:
        try:
            text = summarize.summarize_documents(
                app.state.conn, app.state.client, req.doc_ids, req.mode
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"summary": text}

    @app.post("/api/actions")
    def actions(req: ActionsRequest) -> dict:
        text = _doc_or_text(app, req.text, req.doc_id)
        items = extract_actions(app.state.client, text)
        return {"items": [i.to_dict() for i in items]}

    @app.post("/api/pii")
    def pii(req: PIIRequest) -> dict:
        text = _doc_or_text(app, req.text, req.doc_id)
        spans = pii_scan(text, client=app.state.client, use_model=req.use_model)
        return {"text": text, "spans": [s.to_dict() for s in spans]}

    @app.get("/api/reminders")
    def get_reminders(include_done: bool = False) -> list[dict]:
        return [r.to_dict() for r in reminders.list_reminders(app.state.conn, include_done)]

    @app.post("/api/reminders")
    def post_reminder(req: ReminderRequest) -> dict:
        try:
            r = reminders.create_from_text(app.state.conn, req.text, client=app.state.client)
        except reminders.ReminderParseError as exc:
            raise HTTPException(400, str(exc)) from exc
        return r.to_dict()

    @app.post("/api/reminders/{reminder_id}/done")
    def done_reminder(reminder_id: int) -> dict:
        try:
            reminders.set_status(app.state.conn, reminder_id, "done")
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    return app


def run(cfg: Config | None = None, host: str = "127.0.0.1", port: int = 8756) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
