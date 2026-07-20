"""Configuration and logging.

All data lives under ``data_dir`` (default ``~/.vault-assistant``): the SQLite
database, logs, and an optional ``config.toml`` overriding any field below.
No telemetry, no network access other than the local Ollama endpoint.
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
    watch_folders: list[Path] = field(default_factory=list)
    chunk_target_tokens: int = 400
    chunk_overlap_ratio: float = 0.15
    vector_top_k: int = 8
    keyword_top_k: int = 8
    context_token_budget: int = 3000
    num_ctx: int = 8192
    request_timeout: float = 300.0

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
            "chunk_target_tokens",
            "chunk_overlap_ratio",
            "vector_top_k",
            "keyword_top_k",
            "context_token_budget",
            "num_ctx",
            "request_timeout",
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
