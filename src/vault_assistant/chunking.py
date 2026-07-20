"""Chunking with exact char offsets.

Targets ~400 tokens per chunk with ~15% overlap (spec: 300-500, 15%). Token
counts are approximated as chars/4 — we deliberately avoid shipping a
tokenizer; the chunk-size target only needs to be roughly right. Splits on
paragraph boundaries, falling back to sentences and then fixed windows for
pathological unbroken text. Every chunk records the exact ``[char_start,
char_end)`` span of the source text (chunk.text == text[start:end]), which is
what makes Phase 3 inline citations and exact document reconstruction possible.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

CHARS_PER_TOKEN = 4

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)\s*")


@dataclass
class Chunk:
    text: str
    char_start: int
    char_end: int
    page: int | None = None


def approx_tokens(s: str) -> int:
    return max(1, len(s) // CHARS_PER_TOKEN)


def page_for_offset(pages: list[tuple[int, int, int]], offset: int) -> int | None:
    """Return the page number whose [start, end) range contains offset."""
    if not pages:
        return None
    starts = [p[1] for p in pages]
    idx = bisect_right(starts, offset) - 1
    if idx < 0:
        return pages[0][0]
    return pages[idx][0]


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _PARAGRAPH_BREAK.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _split_long(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split an over-long paragraph into sentence spans, hard-windowing any
    sentence that is itself longer than max_chars."""
    out: list[tuple[int, int]] = []
    for m in _SENTENCE.finditer(text, start, end):
        s, e = m.start(), m.end()
        if e - s <= max_chars:
            out.append((s, e))
        else:
            out.extend((p, min(p + max_chars, e)) for p in range(s, e, max_chars))
    if not out:  # no sentence structure at all
        out = [(p, min(p + max_chars, end)) for p in range(start, end, max_chars)]
    return out


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def chunk_text(
    text: str,
    pages: list[tuple[int, int, int]] | None = None,
    target_tokens: int = 400,
    overlap_ratio: float = 0.15,
) -> list[Chunk]:
    pages = pages or []
    target_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = int(target_chars * overlap_ratio)

    # Sentence-granularity segments (paragraph split, then sentences) so the
    # overlap window can step back mid-paragraph.
    segments: list[tuple[int, int]] = []
    for s, e in _paragraph_spans(text):
        segments.extend(_split_long(text, s, e, target_chars))
    if not segments:
        return []

    chunks: list[Chunk] = []
    i = 0
    n = len(segments)
    while i < n:
        j = i
        while j + 1 < n and segments[j + 1][1] - segments[i][0] <= target_chars:
            j += 1
        start, end = _trim(text, segments[i][0], segments[j][1])
        if end > start:
            chunks.append(
                Chunk(
                    text=text[start:end],
                    char_start=start,
                    char_end=end,
                    page=page_for_offset(pages, start),
                )
            )
        if j + 1 >= n:
            break
        # Start the next chunk far enough back to give ~overlap_chars of
        # overlap, but always at least one segment past the current start.
        k = j + 1
        while k - 1 > i and segments[j][1] - segments[k - 1][0] <= overlap_chars:
            k -= 1
        i = k
    return chunks
