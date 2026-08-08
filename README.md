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
vault status                              # check the backend, models, index counts
vault models                              # list models available from the configured provider
vault ingest ~/Documents/notes            # index a folder (.md .txt .pdf .docx)
vault watch ~/Documents/notes             # poll for changes and re-index
vault ask "what did the March invoice say about payment terms?"
vault ask "summarize payment terms and list all late fees" --verbose  # shows sub-queries
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

## LLM providers (optional, defaults to local Ollama)

Everything above works fully offline against a local Ollama by default. To
use a different provider for chat/generation (and optionally a different one
for embeddings), set `provider` (and, if needed, `embed_provider`) in
`~/.vault-assistant/config.toml`:

```toml
provider = "anthropic"       # ollama (default) | openai | anthropic | gemini | vllm | openai_compatible
embed_provider = "ollama"    # required alongside provider="anthropic" — Claude has no embeddings API
gen_model = "claude-opus-5"
embed_model = "nomic-embed-text"
```

API keys are **never** read from `config.toml` — same rule as the Langfuse
keys below, to keep secrets out of a file that may get backed up. Export the
key that matches your `provider` before running `vault ...`:

| `provider` | env var | install extra | notes |
|---|---|---|---|
| `ollama` (default) | — | none | local, no key needed |
| `openai` | `OPENAI_API_KEY` | none (plain HTTP) | |
| `anthropic` | `ANTHROPIC_API_KEY` | `.[anthropic]` | chat only — set `embed_provider` to something else |
| `gemini` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `.[gemini]` | |
| `vllm` | `VLLM_API_KEY` (optional, defaults to `"EMPTY"`) | none (plain HTTP) | set `api_base_url` (default `http://localhost:8000/v1`) and `gen_model` to what your server serves |
| `openai_compatible` | `LLM_API_KEY` | none (plain HTTP) | any other provider speaking the OpenAI Chat Completions API (Together, Groq, Fireworks, Mistral, DeepSeek, ...); requires `api_base_url` |

```sh
.venv/bin/pip install -e ".[anthropic]"    # only needed for provider="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."
vault models                               # see what's available before picking gen_model
vault status                               # confirms the backend is reachable
```

`vault models` (optionally `--provider X` to preview a different provider
without editing config.toml) lists what the configured backend reports, so
you can pick a `gen_model`/`embed_model` without guessing a name. Switching
`embed_model`/`embed_provider` after documents are already ingested requires
re-ingesting from a fresh database — chunks embedded by different
models/providers aren't compatible with each other, and `vault ingest` will
refuse to mix them.

## Observability (optional, off by default)

Tracing of the Q&A pipeline (retrieval, decompose/synthesize/verify steps,
every Ollama call with token usage) via [Langfuse](https://langfuse.com),
self-hosted so trace data never leaves the device — this stays consistent
with the no-telemetry-by-default design above. Disabled unless you opt in.

1. Install the extra and run Langfuse locally via its own docker-compose:

   ```sh
   .venv/bin/pip install -e ".[observability]"
   git clone https://github.com/langfuse/langfuse.git /tmp/langfuse
   cd /tmp/langfuse && docker compose up -d   # http://localhost:3000 after ~2-3 min
   ```

2. In the Langfuse UI, create a project and an API key pair, then export them
   (never put these in `config.toml`):

   ```sh
   export LANGFUSE_PUBLIC_KEY="pk-lf-..."
   export LANGFUSE_SECRET_KEY="sk-lf-..."
   ```

3. Enable tracing in `~/.vault-assistant/config.toml`:

   ```toml
   langfuse_enabled = true
   langfuse_host = "http://localhost:3000"   # default; change only for a non-default self-host port
   ```

Run `vault ask ...` or `vault serve` as usual — each question becomes a trace
named `agentic_qa` or `qa.answer_question` depending on which pipeline
handled it (see below), with nested spans for decomposition, retrieval,
evidence scoring, generation, and verification, each carrying the model name,
prompt, output, and token usage. Leaving `langfuse_enabled` unset (or
`false`, the default) or the package uninstalled costs nothing — the tracing
calls silently no-op.

Note: `vault ask` / `/api/ask` use the agentic pipeline by default (`agentic_qa = true` in
config). Pass `--no-agentic` (CLI) or `"agentic": false` (API) to force the single-pass fast
path. The `/api/ask` response now includes `pipeline`, `sub_queries`, `evidence_confidence`,
and `verification` fields alongside the existing `answer` and `sources`.

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
