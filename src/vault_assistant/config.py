"""Configuration and logging.

All data lives under ``data_dir`` (default ``~/.vault-assistant``): the SQLite
database, logs, and an optional ``config.toml`` overriding any field below.
No telemetry. Network access is limited to whatever ``provider``/
``embed_provider`` you configure (see providers.py) — the default,
``"ollama"``, keeps everything local.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path(os.environ.get("VAULT_DATA_DIR", "~/.vault-assistant")).expanduser()

logger = logging.getLogger("vault")


@dataclass
class Config:
    data_dir: Path = DEFAULT_DATA_DIR
    ollama_url: str = "http://localhost:11434"
    gen_model: str = "qwen3:8b"
    embed_model: str = "nomic-embed-text"

    # LLM provider selection (providers.py). "ollama" (default) keeps
    # everything local; other values require the matching API key in the
    # environment (never in this file — see providers.py's module docstring)
    # and, for anthropic/gemini, an optional dependency extra. embed_provider
    # defaults to provider when empty; it must differ when provider is
    # "anthropic" (Claude has no embeddings API).
    provider: str = "ollama"
    embed_provider: str = ""
    api_base_url: str = ""

    watch_folders: list[Path] = field(default_factory=list)
    chunk_target_tokens: int = 400
    chunk_overlap_ratio: float = 0.15
    vector_top_k: int = 8
    keyword_top_k: int = 8
    context_token_budget: int = 3000
    num_ctx: int = 8192
    request_timeout: float = 300.0

    # Agentic Q&A pipeline (agents.py)
    agentic_qa: bool = True
    agentic_min_words_for_decompose: int = 8
    agentic_max_words_simple: int = 30
    agentic_conjunctions: list[str] = field(
        default_factory=lambda: [" and ", " as well as ", " along with ", " & ", " plus "]
    )
    agentic_max_subqueries: int = 4
    agentic_max_candidates: int = 20
    agentic_excerpt_chars: int = 300
    agentic_relevance_threshold: float = 0.3
    agentic_evidence_top_n: int = 12
    agentic_verify: bool = True

    # Query cache / agent memory / knowledge base (memory.py)
    cache_max_size: int = 1000
    cache_default_ttl: float = 3600.0
    memory_max_turns: int = 10
    memory_max_items: int = 100
    memory_context_budget: int = 400
    memory_top_k_facts: int = 5
    kb_find_related_limit: int = 50

    # Optional Langfuse tracing (observability.py). Off by default to preserve
    # the no-telemetry default above; enabling it still requires
    # LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY in the environment.
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vault.db"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "vault.log"


def load_config(config_file: Path | None = None) -> Config:
    cfg = Config()
    path = config_file or cfg.data_dir / "config.toml"
    if path.exists():
        raw = tomllib.loads(path.read_text())
        for key in (
            "ollama_url",
            "gen_model",
            "embed_model",
            "provider",
            "embed_provider",
            "api_base_url",
            "chunk_target_tokens",
            "chunk_overlap_ratio",
            "vector_top_k",
            "keyword_top_k",
            "context_token_budget",
            "num_ctx",
            "request_timeout",
            "agentic_qa",
            "agentic_min_words_for_decompose",
            "agentic_max_words_simple",
            "agentic_conjunctions",
            "agentic_max_subqueries",
            "agentic_max_candidates",
            "agentic_excerpt_chars",
            "agentic_relevance_threshold",
            "agentic_evidence_top_n",
            "agentic_verify",
            "cache_max_size",
            "cache_default_ttl",
            "memory_max_turns",
            "memory_max_items",
            "memory_context_budget",
            "memory_top_k_facts",
            "kb_find_related_limit",
            "langfuse_enabled",
            "langfuse_host",
        ):
            if key in raw:
                setattr(cfg, key, raw[key])
        if "data_dir" in raw:
            cfg.data_dir = Path(raw["data_dir"]).expanduser()
        if "watch_folders" in raw:
            cfg.watch_folders = [Path(p).expanduser() for p in raw["watch_folders"]]
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def setup_logging(cfg: Config, verbose: bool = False) -> None:
    root = logging.getLogger("vault")
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(cfg.log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if verbose else logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(fh)
    root.addHandler(sh)
