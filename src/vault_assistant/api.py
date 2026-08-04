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

from . import db, permissions, reminders, summarize
from .actions import extract_actions
from .config import Config, load_config, setup_logging
from .ingest import ingest_paths
from .observability import init_observability
from .pii import scan as pii_scan
from .providers import ProviderError, UnavailableClient, build_client
from .qa import answer_question
from .vectors import VectorIndex

logger = logging.getLogger("vault.api")

WEB_DIR = Path(__file__).parent / "web"


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


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


class FolderCreateRequest(BaseModel):
    path: str = Field(min_length=1)
    access_level: str = "edit"


class FolderUpdateRequest(BaseModel):
    access_level: str


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
    init_observability(cfg)
    app = FastAPI(title="Vault Assistant", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.cfg = cfg
    app.state.conn = db.connect(cfg.db_path)
    try:
        app.state.client = build_client(cfg)
    except ProviderError as exc:
        logger.warning("LLM provider unavailable at startup: %s", exc)
        app.state.client = UnavailableClient(str(exc))
    app.state.index = VectorIndex(app.state.conn)
    app.state.ingest_lock = threading.Lock()

    threading.Thread(target=_notifier_loop, args=(app,), daemon=True).start()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "index.html").read_text()

    @app.get("/api/status")
    def status() -> dict:
        conn = app.state.conn
        client = app.state.client
        cfg = app.state.cfg
        up = client.is_up()
        return {
            "backend_up": up,
            "provider": cfg.provider,
            "embed_provider": cfg.embed_provider or cfg.provider,
            "gen_model": cfg.gen_model,
            "embed_model": cfg.embed_model,
            "missing_models": client.missing_models() if up else [],
            "documents": conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"],
            "chunks": conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"],
            "errors": conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE status = 'error'"
            ).fetchone()["n"],
        }

    @app.get("/api/models")
    def models() -> dict:
        client = app.state.client
        try:
            available = client.list_models()
        except Exception as exc:  # noqa: BLE001 — surface backend errors as data, not a 500
            raise HTTPException(502, f"could not list models: {exc}") from exc
        return {"provider": app.state.cfg.provider, "models": available}

    @app.get("/api/documents")
    def documents() -> list[dict]:
        conn = app.state.conn
        docs = db.list_documents(conn)
        for d in docs:
            d["access_level"] = permissions.resolve_access_level(conn, d["path"])
        return docs

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: int) -> dict:
        conn = app.state.conn
        row = conn.execute("SELECT path FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"no document {doc_id}")
        level = permissions.resolve_access_level(conn, row["path"])
        if level != "edit":
            raise HTTPException(403, f"document is {level}; deleting requires edit access")
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        db.bump_generation(conn)
        return {"ok": True}

    @app.get("/api/folders")
    def list_folders() -> list[dict]:
        conn = app.state.conn
        result = []
        for folder in permissions.list_folders(conn):
            prefix = folder.path.rstrip("/") + "/"
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE path = ? OR path LIKE ?",
                (folder.path, prefix + "%"),
            ).fetchone()["n"]
            result.append({**folder.to_dict(), "document_count": n})
        return result

    @app.post("/api/folders")
    def create_folder(req: FolderCreateRequest) -> dict:
        path = Path(req.path).expanduser()
        if not path.exists() or not path.is_dir():
            raise HTTPException(400, f"not a directory: {req.path}")
        try:
            folder = permissions.set_folder(app.state.conn, str(path), req.access_level)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return folder.to_dict()

    @app.put("/api/folders/{folder_id}")
    def update_folder(folder_id: int, req: FolderUpdateRequest) -> dict:
        conn = app.state.conn
        try:
            existing = permissions.get_folder(conn, folder_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        try:
            folder = permissions.set_folder(conn, existing.path, req.access_level)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return folder.to_dict()

    @app.delete("/api/folders/{folder_id}")
    def delete_folder(folder_id: int) -> dict:
        try:
            permissions.remove_folder(app.state.conn, folder_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.post("/api/ingest")
    def ingest(req: IngestRequest) -> dict:
        paths = [Path(p) for p in req.paths]
        missing = [str(p) for p in paths if not p.expanduser().exists()]
        if missing:
            raise HTTPException(400, f"paths do not exist: {missing}")
        if req.force:
            locked = [
                str(p) for p in paths
                if permissions.resolve_access_level(app.state.conn, str(p)) != "edit"
            ]
            if locked:
                raise HTTPException(403, f"force re-ingest requires edit access: {locked}")
        with app.state.ingest_lock:
            report = ingest_paths(app.state.conn, app.state.client, paths, force=req.force)
        return {
            "added": report.added,
            "updated": report.updated,
            "skipped": report.skipped,
            "removed": report.removed,
            "blocked": report.blocked,
            "failed": [{"path": p, "error": e} for p, e in report.failed],
        }

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        result = answer_question(
            app.state.conn, app.state.index, app.state.client, req.question, app.state.cfg
        )
        return {
            "answer": result.answer,
            "sources": [
                {"filename": s.filename, "path": s.path, "page": s.page} for s in result.sources
            ],
        }

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
