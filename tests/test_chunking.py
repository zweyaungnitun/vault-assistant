from vault_assistant.chunking import Chunk, approx_tokens, chunk_text, page_for_offset


def test_empty_and_whitespace():
    assert chunk_text("") == []
    assert chunk_text("   \n\n   ") == []


def test_short_text_single_chunk():
    text = "One short paragraph about nothing much."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)


def test_offsets_are_exact():
    text = "\n\n".join(f"Paragraph {i}. " + "Words fill the space here. " * 8 for i in range(40))
    chunks = chunk_text(text, target_tokens=100)
    assert len(chunks) > 3
    for c in chunks:
        assert text[c.char_start:c.char_end] == c.text


def test_chunks_overlap_and_progress():
    text = "\n\n".join("Sentence padding for paragraph %d. " % i * 3 for i in range(60))
    chunks = chunk_text(text, target_tokens=100, overlap_ratio=0.15)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.char_start > prev.char_start  # always advances
    # overlap exists somewhere (paragraphs are small enough to step back)
    assert any(n.char_start < p.char_end for p, n in zip(chunks, chunks[1:]))


def test_full_coverage_no_content_lost():
    text = "\n\n".join(f"Unique marker UM{i:03d} inside paragraph." for i in range(80))
    chunks = chunk_text(text, target_tokens=80)
    joined = " ".join(c.text for c in chunks)
    for i in range(80):
        assert f"UM{i:03d}" in joined


def test_pathological_unbroken_text():
    text = "x" * 10_000
    chunks = chunk_text(text, target_tokens=100)
    assert chunks
    for c in chunks:
        assert len(c.text) <= 100 * 4
        assert text[c.char_start:c.char_end] == c.text
    assert chunks[-1].char_end == len(text)


def test_chunk_sizes_within_spec_band():
    text = "\n\n".join("A sentence with some words in it. " * 4 for _ in range(50))
    chunks = chunk_text(text, target_tokens=400)
    # all but the tail chunk should be reasonably close to target
    for c in chunks[:-1]:
        assert 100 <= approx_tokens(c.text) <= 500


def test_page_mapping():
    pages = [(1, 0, 100), (2, 102, 200), (3, 202, 300)]
    assert page_for_offset(pages, 0) == 1
    assert page_for_offset(pages, 99) == 1
    assert page_for_offset(pages, 150) == 2
    assert page_for_offset(pages, 250) == 3
    assert page_for_offset([], 50) is None


def test_page_assigned_to_chunks():
    page1 = "First page text. " * 30
    page2 = "Second page text. " * 30
    text = page1 + "\n\n" + page2
    pages = [(1, 0, len(page1)), (2, len(page1) + 2, len(text))]
    chunks = chunk_text(text, pages, target_tokens=100)
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
