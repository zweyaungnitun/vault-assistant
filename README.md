# Vault Assistant

Fully offline personal document assistant (Phase 1). Answers questions about
your own documents with citations, summarizes them, extracts action items,
flags PII, and creates local reminders — with zero data leaving the device.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with:
  - `ollama pull qwen3:8b` (generation, ~5 GB)
  - `ollama pull nomic-embed-text` (embeddings, ~270 MB)

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Usage

```sh
vault status                              # check Ollama, models, index counts
vault ingest ~/Documents/notes            # index a folder (.md .txt .pdf .docx)
vault watch ~/Documents/notes             # poll for changes and re-index
vault ask "what did the March invoice say about payment terms?"
vault summarize report.pdf --mode bullets # or --doc 3 for ingested docs
vault actions --file meeting-notes.md     # extract action items as a checklist
vault pii --file contract.pdf             # flag PII (add --no-model for regex-only)
vault remind "send the report next Friday at 3pm"
vault reminders                           # list; --done ID to complete
vault serve                               # web UI at http://127.0.0.1:8756
```

All state lives in `~/.vault-assistant/` (override with `VAULT_DATA_DIR`).
Optional config in `~/.vault-assistant/config.toml`:

```toml
gen_model = "qwen3:8b"
embed_model = "nomic-embed-text"
watch_folders = ["~/Documents/notes"]
```

## Architecture

```
folders → extract (pypdf/python-docx) → chunk (~400 tok, 15% overlap,
exact char offsets + page numbers) → embed (Ollama) → SQLite
                                                        ├── chunks + float32 BLOB embeddings
                                                        ├── FTS5 keyword index (BM25)
                                                        └── reminders

query → embed → vector top-k (numpy cosine)  ┐
      → FTS5 BM25 top-k                      ┴→ RRF merge → context → qwen3
                                                            (answer only from
                                                             context, cited)
```

Design notes:

- **Vector search is brute-force numpy** over embeddings stored as BLOBs in
  SQLite, not sqlite-vec: this machine's Python ships sqlite3 without
  loadable-extension support, and at Phase 1 scale (≤ ~25k chunks) a single
  matrix-vector product is faster than an ANN index anyway. Swap point:
  `vectors.py`.
- **Chunks store exact char offsets and PDF page numbers** so Phase 3 inline
  citations won't require re-ingesting corpora, and documents can be
  reconstructed exactly for summarization despite chunk overlap.
- **Citations cannot be fabricated**: source lists are built from the actually
  retrieved chunks; the model only picks which ones it used.
- **Reminder dates are parsed deterministically** (dateparser), not computed
  by the model. The model is only a fallback for splitting title from time
  phrase in unusual phrasings.
- **Generation runs with thinking disabled** (Qwen3 hybrid model) to keep
  latency inside targets.
- Ingestion failures (corrupt/encrypted/scanned-image files) are recorded with
  status + error and logged, never silently dropped. OCR is out of scope for
  Phase 1.

## Development

```sh
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```
