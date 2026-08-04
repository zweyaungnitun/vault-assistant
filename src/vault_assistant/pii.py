"""PII flagging: regex pass for structured identifiers plus a model-assisted
pass for context-dependent PII (names, addresses).

Recall is prioritized over precision per spec — false positives are acceptable,
missed PII is not. Nothing is auto-redacted; output is annotated spans for the
user to review. The model returns quoted strings which are located in the text
programmatically (LLMs are unreliable at reporting character offsets).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

from .providers import LLMClient, ProviderError

logger = logging.getLogger("vault.pii")

_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # Phone: international or US-style, 7+ digits with common separators.
    ("phone", re.compile(r"(?<![\w./-])(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{2,4}[ .-]\d{2,4}(?:[ .-]\d{2,6}){1,3}(?![\w-])|(?<![\w./-])\+\d{7,15}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Card/account-like: 13-19 digits, optionally grouped. Recall over precision.
    ("card_or_account", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("account_number", re.compile(r"\b\d{8,12}\b")),
    ("ip_address", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
]

MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["name", "address", "email", "phone", "id_number", "other"],
                    },
                },
                "required": ["text", "kind"],
            },
        }
    },
    "required": ["items"],
}

MODEL_SYSTEM_PROMPT = """You detect personally identifiable information (PII) in text. Find every instance of: person names, physical addresses, email addresses, phone numbers, and ID/account numbers. Err on the side of flagging — missing PII is worse than a false positive. For each, return the EXACT text as it appears in the input (verbatim substring) and its kind. Return an empty items array if there is none."""


@dataclass
class PIISpan:
    start: int
    end: int
    text: str
    kind: str
    source: str  # "regex" | "model"

    def to_dict(self) -> dict:
        return asdict(self)


def regex_scan(text: str) -> list[PIISpan]:
    spans = [
        PIISpan(m.start(), m.end(), m.group(), kind, "regex")
        for kind, pattern in _REGEX_PATTERNS
        for m in pattern.finditer(text)
    ]
    return _merge(spans)


def locate_all(text: str, needle: str) -> list[tuple[int, int]]:
    needle = needle.strip()
    if not needle:
        return []
    out = []
    pos = 0
    while (idx := text.find(needle, pos)) != -1:
        out.append((idx, idx + len(needle)))
        pos = idx + 1
    return out


def model_scan(client: LLMClient, text: str) -> list[PIISpan]:
    result = client.chat_json(MODEL_SYSTEM_PROMPT, f"Find all PII in:\n\n{text}", MODEL_SCHEMA)
    spans: list[PIISpan] = []
    for item in result.get("items", []):
        for start, end in locate_all(text, item.get("text", "")):
            spans.append(PIISpan(start, end, text[start:end], item.get("kind", "other"), "model"))
    return spans


def _merge(spans: list[PIISpan]) -> list[PIISpan]:
    """Sort and drop spans fully contained in another span."""
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    out: list[PIISpan] = []
    for s in spans:
        if out and s.start >= out[-1].start and s.end <= out[-1].end:
            continue
        out.append(s)
    return out


def scan(text: str, client: LLMClient | None = None, use_model: bool = True) -> list[PIISpan]:
    spans = regex_scan(text)
    if use_model and client is not None:
        try:
            spans = _merge(spans + model_scan(client, text))
        except ProviderError as exc:
            # Recall-first: regex results are still returned, but the caller
            # should know the deeper pass failed.
            logger.warning("model-assisted PII pass failed, regex-only results: %s", exc)
    return spans
