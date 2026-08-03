"""Thin client for the local Ollama REST API (localhost only).

Generation runs Qwen3 with thinking disabled to keep latency inside Phase 1
targets. Structured extraction uses Ollama's ``format`` (JSON schema) support.
nomic-embed-text expects task prefixes on inputs; ingest and retrieval attach
DOC_PREFIX / QUERY_PREFIX respectively.
"""

from __future__ import annotations

import json
import logging

import httpx

from .observability import observe, update_current_generation

logger = logging.getLogger("vault.ollama")

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

EMBED_BATCH_SIZE = 32


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        gen_model: str = "qwen3:8b",
        embed_model: str = "nomic-embed-text",
        timeout: float = 300.0,
        num_ctx: int = 8192,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.gen_model = gen_model
        self.embed_model = embed_model
        self.num_ctx = num_ctx
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._supports_think = True

    @classmethod
    def from_config(cls, cfg) -> "OllamaClient":
        return cls(
            base_url=cfg.ollama_url,
            gen_model=cfg.gen_model,
            embed_model=cfg.embed_model,
            timeout=cfg.request_timeout,
            num_ctx=cfg.num_ctx,
        )

    def is_up(self) -> bool:
        try:
            return self._http.get("/api/version", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        resp = self._http.get("/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def missing_models(self) -> list[str]:
        have = self.list_models()

        def present(name: str) -> bool:
            return any(h == name or h.split(":")[0] == name for h in have)

        return [m for m in (self.gen_model, self.embed_model) if not present(m)]

    @observe(name="ollama.embed", as_type="embedding", capture_input=False, capture_output=False)
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts (callers attach DOC_PREFIX/QUERY_PREFIX as appropriate)."""
        update_current_generation(model=self.embed_model, input={"count": len(texts)})
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            resp = self._http.post("/api/embed", json={"model": self.embed_model, "input": batch})
            if resp.status_code != 200:
                raise OllamaError(f"embed failed ({resp.status_code}): {resp.text[:200]}")
            embeddings = resp.json().get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise OllamaError("embed returned wrong number of vectors")
            out.extend(embeddings)
        update_current_generation(output={"count": len(out)})
        return out

    @observe(name="ollama.chat", as_type="generation", capture_input=False, capture_output=False)
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        format: dict | None = None,
        num_ctx: int | None = None,
        trace_name: str | None = None,
    ) -> str:
        gen_kwargs: dict = {
            "model": self.gen_model,
            "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "model_parameters": {"temperature": temperature, "num_ctx": num_ctx or self.num_ctx},
        }
        if trace_name:
            gen_kwargs["name"] = f"ollama.chat:{trace_name}"
        update_current_generation(**gen_kwargs)

        payload: dict = {
            "model": self.gen_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx or self.num_ctx,
            },
        }
        if format is not None:
            payload["format"] = format
        if self._supports_think:
            payload["think"] = False
        resp = self._http.post("/api/chat", json=payload)
        if resp.status_code == 400 and self._supports_think and "think" in resp.text.lower():
            # Model doesn't accept the think flag; drop it for this process.
            self._supports_think = False
            payload.pop("think", None)
            resp = self._http.post("/api/chat", json=payload)
        if resp.status_code != 200:
            raise OllamaError(f"chat failed ({resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        content = data.get("message", {}).get("content", "").strip()
        update_current_generation(
            output=content,
            usage_details={"input": data.get("prompt_eval_count", 0), "output": data.get("eval_count", 0)},
        )
        return content

    def chat_json(
        self, system: str, user: str, schema: dict, temperature: float = 0.1, trace_name: str | None = None
    ):
        """Chat with a JSON-schema-constrained response; retries once on bad JSON."""
        for attempt in (1, 2):
            raw = self.chat(system, user, temperature=temperature, format=schema, trace_name=trace_name)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("structured output was not valid JSON (attempt %d)", attempt)
        raise OllamaError("model failed to produce valid JSON output")

    def close(self) -> None:
        self._http.close()
