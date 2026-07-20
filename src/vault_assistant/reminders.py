"""Local reminders from natural language — the Phase 1 single tool call.

Date resolution is deterministic: dateparser finds and parses the time
expression ("next Friday", "in 2 hours"), which is far more reliable than
asking an 8B model to compute dates. The model is only used as a fallback to
separate title from time phrase when the regex patterns don't match, and even
then the phrase itself is still resolved by dateparser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time
from sqlite3 import Connection

import dateparser
from dateparser.search import search_dates

from .ollama_client import OllamaClient, OllamaError

DEFAULT_HOUR = 9  # date-only phrases ("tomorrow", "next Friday") land at 09:00

_PARSER_SETTINGS = {
    "PREFER_DATES_FROM": "future",
    "RETURN_AS_TIMEZONE_AWARE": False,
}

_LEAD_IN = re.compile(r"^\s*(?:please\s+)?remind\s+me\s+(?:to\s+|about\s+|that\s+)?", re.IGNORECASE)
_TRAILING_FILLER = re.compile(r"[\s,;.!-]+$")
_EXPLICIT_TIME = re.compile(r"\d{1,2}[:.]\d{2}|\d{1,2}\s*(?:am|pm)\b|\bnoon\b|\bmidnight\b|\bhour|\bminute", re.IGNORECASE)

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "when_phrase": {"type": ["string", "null"]},
    },
    "required": ["title", "when_phrase"],
}

EXTRACT_SYSTEM_PROMPT = """You split a reminder request into the thing to be reminded about and the time expression. Return:
- title: what to remind about, without the time expression and without lead-ins like "remind me to"
- when_phrase: the time expression EXACTLY as written in the input (e.g. "next Friday at 3pm", "in 2 hours"), or null if there is none
Do not compute or reformat dates — copy the phrase verbatim."""


class ReminderParseError(Exception):
    pass


@dataclass
class Reminder:
    id: int
    title: str
    due_at: datetime
    created_at: datetime
    status: str
    source_text: str | None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "due_at": self.due_at.isoformat(timespec="minutes"),
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "status": self.status,
            "source_text": self.source_text,
        }


def _normalize_due(parsed: datetime, phrase: str, now: datetime) -> datetime:
    # dateparser resolves date-only phrases to the current clock time (relative
    # phrases) or midnight (absolute dates); move those to a sane default hour.
    date_only = (parsed.hour, parsed.minute) in (((now.hour, now.minute)), (0, 0))
    if date_only and not _EXPLICIT_TIME.search(phrase):
        return parsed.replace(hour=DEFAULT_HOUR, minute=0, second=0, microsecond=0)
    return parsed.replace(second=0, microsecond=0)


def parse_reminder(
    text: str,
    now: datetime | None = None,
    client: OllamaClient | None = None,
) -> tuple[str, datetime]:
    """Return (title, due_at). Deterministic path first; LLM fallback only to
    split title from time phrase in unusual phrasings."""
    now = now or datetime.now()
    settings = dict(_PARSER_SETTINGS, RELATIVE_BASE=now)
    body = _LEAD_IN.sub("", text).strip()

    found = search_dates(body, languages=["en"], settings=settings)
    if found:
        # Use the last date expression (titles like "pay March invoice" can
        # contain earlier date-like words; the time expression trails in
        # normal phrasing).
        phrase, parsed = found[-1]
        title = body.replace(phrase, "").strip()
        # strip connective words left dangling where the time phrase was removed
        title = re.sub(r"\b(?:on|at|by|in|next|this|the)\s*$", "", title, flags=re.IGNORECASE).strip()
        title = _TRAILING_FILLER.sub("", title)
        if title:
            return title, _normalize_due(parsed, phrase, now)

    if client is not None:
        try:
            result = client.chat_json(EXTRACT_SYSTEM_PROMPT, f"Reminder request: {text}", EXTRACT_SCHEMA)
        except OllamaError:
            result = {}
        title = (result.get("title") or "").strip() or body
        phrase = result.get("when_phrase")
        if phrase:
            parsed = dateparser.parse(phrase, languages=["en"], settings=settings)
            if parsed:
                return title, _normalize_due(parsed, phrase, now)

    raise ReminderParseError(
        f"could not find a date/time in {text!r} — try e.g. 'remind me to X tomorrow at 3pm'"
    )


def create_reminder(
    conn: Connection,
    title: str,
    due_at: datetime,
    source_text: str | None = None,
    now: datetime | None = None,
) -> Reminder:
    now = now or datetime.now()
    cur = conn.execute(
        "INSERT INTO reminders(title, due_at, created_at, status, source_text) "
        "VALUES(?, ?, ?, 'pending', ?)",
        (title, due_at.isoformat(timespec="minutes"), now.isoformat(timespec="seconds"), source_text),
    )
    conn.commit()
    return Reminder(cur.lastrowid, title, due_at, now, "pending", source_text)


def create_from_text(
    conn: Connection,
    text: str,
    client: OllamaClient | None = None,
    now: datetime | None = None,
) -> Reminder:
    title, due_at = parse_reminder(text, now=now, client=client)
    return create_reminder(conn, title, due_at, source_text=text, now=now)


def _row_to_reminder(row) -> Reminder:
    return Reminder(
        id=row["id"],
        title=row["title"],
        due_at=datetime.fromisoformat(row["due_at"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        status=row["status"],
        source_text=row["source_text"],
    )


def list_reminders(conn: Connection, include_done: bool = False) -> list[Reminder]:
    query = "SELECT * FROM reminders"
    if not include_done:
        query += " WHERE status != 'done'"
    query += " ORDER BY due_at"
    return [_row_to_reminder(r) for r in conn.execute(query).fetchall()]


def due_reminders(conn: Connection, now: datetime | None = None) -> list[Reminder]:
    now = now or datetime.now()
    rows = conn.execute(
        "SELECT * FROM reminders WHERE status = 'pending' AND due_at <= ? ORDER BY due_at",
        (now.isoformat(timespec="minutes"),),
    ).fetchall()
    return [_row_to_reminder(r) for r in rows]


def set_status(conn: Connection, reminder_id: int, status: str) -> None:
    if status not in ("pending", "notified", "done"):
        raise ValueError(f"invalid status {status!r}")
    cur = conn.execute("UPDATE reminders SET status = ? WHERE id = ?", (status, reminder_id))
    if cur.rowcount == 0:
        raise ValueError(f"no reminder with id {reminder_id}")
    conn.commit()
