from datetime import datetime, timedelta

import pytest

from vault_assistant.reminders import (
    ReminderParseError,
    create_from_text,
    create_reminder,
    due_reminders,
    list_reminders,
    parse_reminder,
    set_status,
)

# Tuesday morning
NOW = datetime(2026, 7, 14, 10, 0)


def test_tomorrow_defaults_to_morning():
    title, due = parse_reminder("remind me to send the report tomorrow", now=NOW)
    assert title == "send the report"
    assert due == datetime(2026, 7, 15, 9, 0)


def test_explicit_time_kept():
    title, due = parse_reminder("remind me to call Sam on Friday at 3pm", now=NOW)
    assert "call Sam" in title
    assert due.weekday() == 4  # Friday
    assert due > NOW
    assert (due.hour, due.minute) == (15, 0)


def test_relative_hours():
    title, due = parse_reminder("remind me to stretch in 2 hours", now=NOW)
    assert title == "stretch"
    assert due == NOW + timedelta(hours=2)


def test_absolute_date():
    title, due = parse_reminder("remind me about the dentist on 2026-08-01", now=NOW)
    assert "dentist" in title
    assert (due.year, due.month, due.day) == (2026, 8, 1)
    assert due.hour == 9


def test_next_week_phrasing():
    title, due = parse_reminder("remind me to water the plants next Friday", now=NOW)
    assert due.weekday() == 4
    assert due > NOW
    assert title == "water the plants"  # no dangling "next" when only "Friday" matches


def test_no_date_raises_without_llm():
    with pytest.raises(ReminderParseError):
        parse_reminder("remind me to breathe", now=NOW, client=None)


def test_llm_failure_degrades_to_parse_error(client):
    from vault_assistant.ollama_client import OllamaError

    def boom(*a, **kw):
        raise OllamaError("model not available")

    client.chat_json = boom
    with pytest.raises(ReminderParseError):
        parse_reminder("remind me to breathe", now=NOW, client=client)


def test_llm_fallback_used_for_odd_phrasing(client):
    client.json_response = {"title": "renew passport", "when_phrase": "in 3 days"}
    title, due = parse_reminder("passport renewal needs doing pretty soon honestly", now=NOW, client=client)
    assert title == "renew passport"
    assert due.date() == (NOW + timedelta(days=3)).date()


def test_store_roundtrip(conn):
    r = create_reminder(conn, "test task", NOW + timedelta(days=1), now=NOW)
    items = list_reminders(conn)
    assert [i.id for i in items] == [r.id]
    assert items[0].title == "test task"

    assert due_reminders(conn, now=NOW) == []
    assert [d.id for d in due_reminders(conn, now=NOW + timedelta(days=2))] == [r.id]

    set_status(conn, r.id, "done")
    assert list_reminders(conn) == []
    assert len(list_reminders(conn, include_done=True)) == 1


def test_create_from_text_persists_source(conn):
    r = create_from_text(conn, "remind me to file taxes tomorrow", now=NOW)
    assert r.source_text == "remind me to file taxes tomorrow"
    assert r.due_at == datetime(2026, 7, 15, 9, 0)


def test_invalid_status_rejected(conn):
    r = create_reminder(conn, "x", NOW, now=NOW)
    with pytest.raises(ValueError):
        set_status(conn, r.id, "snoozed")
    with pytest.raises(ValueError):
        set_status(conn, 999, "done")
