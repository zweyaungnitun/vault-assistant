"""Optional Langfuse tracing for the Ollama calls and agentic Q&A pipeline.

Off by default (``Config.langfuse_enabled = False``), matching the no-telemetry
default documented in config.py. Turning it on requires:

1. ``pip install -e ".[observability]"``
2. a Langfuse instance to send traces to — self-hosted (e.g. its own
   docker-compose, default ``http://localhost:3000``) keeps trace data
   on-device; Langfuse Cloud does not.
3. ``langfuse_enabled = true`` in config.toml plus ``LANGFUSE_PUBLIC_KEY`` /
   ``LANGFUSE_SECRET_KEY`` in the environment (keys are deliberately not read
   from config.toml, to keep secrets out of a file that may get backed up).

``observe``, ``update_current_generation`` and ``update_current_span`` are
re-exported/wrapped here so the rest of the codebase never imports
``langfuse`` directly: if the package isn't installed, tracing decorators
silently become no-ops instead of turning into a hard dependency.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from .config import Config

logger = logging.getLogger("vault.observability")

try:
    from langfuse import Langfuse, get_client, observe

    LANGFUSE_INSTALLED = True
except ImportError:
    LANGFUSE_INSTALLED = False

    def observe(*args: Any, **kwargs: Any) -> Callable:  # type: ignore[misc]
        """Stand-in for langfuse.observe when the package isn't installed:
        supports both bare `@observe` and `@observe(...)` forms as a no-op."""

        def decorator(func: Callable) -> Callable:
            return func

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

    def get_client() -> Any:  # type: ignore[misc]
        return None


def init_observability(cfg: Config) -> None:
    """Create the process-wide Langfuse client. Call once at startup, before
    any @observe-decorated function runs, so later get_client() lookups
    (used internally by @observe, and by update_current_* below) resolve to
    this instance rather than lazily creating a default one."""
    if not LANGFUSE_INSTALLED:
        if cfg.langfuse_enabled:
            logger.warning(
                "langfuse_enabled=true but the langfuse package is not installed; "
                "install with: pip install -e '.[observability]'"
            )
        return

    if not cfg.langfuse_enabled:
        Langfuse(tracing_enabled=False, public_key="disabled", secret_key="disabled")
        return

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        logger.warning(
            "langfuse_enabled=true but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY "
            "are not set in the environment; tracing disabled"
        )
        Langfuse(tracing_enabled=False, public_key="disabled", secret_key="disabled")
        return

    Langfuse(public_key=public_key, secret_key=secret_key, host=cfg.langfuse_host)
    logger.info("langfuse tracing enabled -> %s", cfg.langfuse_host)


def update_current_generation(**kwargs: Any) -> None:
    """Best-effort update of the active generation/embedding observation;
    a silent no-op if tracing is disabled or langfuse isn't installed."""
    client = get_client()
    if client is not None:
        client.update_current_generation(**kwargs)


def update_current_span(**kwargs: Any) -> None:
    """Best-effort update of the active span/agent/chain/retriever/tool/
    evaluator observation; a silent no-op if tracing is disabled or langfuse
    isn't installed."""
    client = get_client()
    if client is not None:
        client.update_current_span(**kwargs)
