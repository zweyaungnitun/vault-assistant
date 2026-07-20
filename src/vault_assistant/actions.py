"""Action item extraction with schema-constrained JSON output.

Only explicit action items are extracted; owner and due date must be stated in
the text or they stay null. Each item carries a source snippet quoted from the
input for verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .ollama_client import OllamaClient

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                    "due_date": {"type": ["string", "null"]},
                    "source_snippet": {"type": "string"},
                },
                "required": ["task", "owner", "due_date", "source_snippet"],
            },
        }
    },
    "required": ["items"],
}

SYSTEM_PROMPT = """You extract action items from meeting notes, emails, and documents. Rules:
- Extract only EXPLICIT action items: things someone stated must/should/will be done.
- owner: the person responsible, only if the text names one; otherwise null. Never guess.
- due_date: only if the text states a date or deadline, copied as written; otherwise null. Never invent one.
- source_snippet: a short verbatim quote from the text containing the action item.
- If there are no action items, return an empty items array."""


@dataclass
class ActionItem:
    task: str
    owner: str | None
    due_date: str | None
    source_snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


def extract_actions(client: OllamaClient, text: str) -> list[ActionItem]:
    result = client.chat_json(SYSTEM_PROMPT, f"Extract action items from:\n\n{text}", SCHEMA)
    items = []
    for raw in result.get("items", []):
        task = (raw.get("task") or "").strip()
        if not task:
            continue
        items.append(
            ActionItem(
                task=task,
                owner=(raw.get("owner") or None),
                due_date=(raw.get("due_date") or None),
                source_snippet=(raw.get("source_snippet") or "").strip(),
            )
        )
    return items
