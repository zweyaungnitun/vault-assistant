"""Pluggable LLM/embedding provider clients.

Every provider client below implements the same duck-typed surface consumed
throughout the codebase (qa.py, agents.py, summarize.py, actions.py, pii.py,
memory.py, reminders.py, ingest.py, retrieval.py): ``chat``, ``chat_json``,
``embed``, ``is_up``, ``missing_models``, ``list_models``, plus
``gen_model``/``embed_model``/``doc_prefix``/``query_prefix`` attributes —
see the ``LLMClient`` protocol. ``build_client(cfg)`` is the single place
that picks and constructs the right client(s) from ``Config.provider`` /
``Config.embed_provider``, and is the only thing cli.py/api.py call.

Supported values for ``Config.provider`` / ``Config.embed_provider``:

- ``ollama``            local Ollama server (default; see ollama_client.py)
- ``openai``            OpenAI's hosted API
- ``anthropic``         Claude, via the official ``anthropic`` SDK (chat only
                         — Claude has no embeddings endpoint)
- ``gemini``             Google's Gemini API, via the official ``google-genai``
                         SDK
- ``vllm``              a self-hosted vLLM server (OpenAI-compatible API)
- ``openai_compatible``  any other provider speaking the OpenAI Chat
                         Completions / Embeddings REST shape (Together, Groq,
                         Fireworks, Mistral, DeepSeek, ...); requires
                         ``api_base_url`` in config.toml

API keys are read from the environment, never from config.toml — the same
convention as LANGFUSE_PUBLIC_KEY/SECRET_KEY in observability.py — so a
backed-up config file never leaks a secret:

- OPENAI_API_KEY        (provider=openai)
- ANTHROPIC_API_KEY      (provider=anthropic)
- GEMINI_API_KEY or GOOGLE_API_KEY (provider=gemini)
- VLLM_API_KEY           (provider=vllm; optional, defaults to "EMPTY" — most
                         local vLLM servers don't check it)
- LLM_API_KEY            (provider=openai_compatible)

Since chat and embeddings can come from different providers (most notably:
Claude has no embeddings API), ``Config.embed_provider`` may name a different
provider than ``Config.provider``; ``build_client`` composes the two clients
behind ``CompositeClient`` so the rest of the codebase keeps passing around a
single ``client`` object.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol, runtime_checkable

import httpx

from .observability import observe, update_current_generation

logger = logging.getLogger("vault.providers")

PROVIDERS = ("ollama", "openai", "anthropic", "gemini", "vllm", "openai_compatible")

# The Config dataclass defaults these to Ollama's models; if a non-Ollama
# provider is selected and the user hasn't overridden gen_model/embed_model,
# substitute a sensible default for that provider instead of sending Ollama
# model names to a different API.
_OLLAMA_DEFAULT_GEN = "qwen3:8b"
_OLLAMA_DEFAULT_EMBED = "nomic-embed-text"

_DEFAULT_GEN_MODEL = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-5",
    "gemini": "gemini-2.5-flash",
}
_DEFAULT_EMBED_MODEL = {
    "openai": "text-embedding-3-small",
    "gemini": "text-embedding-004",
}


class ProviderError(Exception):
    """Raised for any provider-client failure (chat/embed call, missing API
    key, missing optional dependency, unsupported operation). ``OllamaError``
    (ollama_client.py) subclasses this, so ``except ProviderError`` catches
    failures uniformly across every provider."""


@runtime_checkable
class LLMClient(Protocol):
    """Structural interface every provider client implements. Not meant to be
    subclassed — just documents the duck-typed contract callers rely on."""

    gen_model: str
    embed_model: str
    doc_prefix: str
    query_prefix: str

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        format: dict | None = None,
        num_ctx: int | None = None,
        trace_name: str | None = None,
    ) -> str: ...

    def chat_json(
        self, system: str, user: str, schema: dict, temperature: float = 0.1, trace_name: str | None = None
    ) -> Any: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def is_up(self) -> bool: ...

    def missing_models(self) -> list[str]: ...

    def list_models(self) -> list[str]: ...


class OpenAICompatibleClient:
    """Chat + embeddings over any server exposing the OpenAI REST API shape:
    OpenAI itself, vLLM's built-in OpenAI-compatible server, or a third-party
    "OpenAI-compatible" provider (Together, Groq, Fireworks, Mistral,
    DeepSeek, ...). Plain httpx, no SDK dependency — same thin-wrapper style
    as OllamaClient. Structured output uses JSON object mode plus the schema
    embedded in the prompt (broadly supported, unlike strict json_schema mode
    which not every compatible server implements), with one retry on bad
    JSON — mirrors OllamaClient.chat_json.
    """

    embed_provider_name: str

    def __init__(
        self,
        base_url: str,
        api_key: str,
        gen_model: str,
        embed_model: str,
        timeout: float = 300.0,
        label: str = "openai",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.gen_model = gen_model
        self.embed_model = embed_model
        self.label = label
        self.embed_provider_name = label
        self.doc_prefix = ""
        self.query_prefix = ""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

    def is_up(self) -> bool:
        try:
            return self._http.get("/models", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        resp = self._http.get("/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def missing_models(self) -> list[str]:
        # Remote/self-hosted servers aren't "pulled" locally like Ollama's —
        # nothing here to check ahead of time; a bad model name surfaces as a
        # normal API error on first use.
        return []

    @observe(name="llm.embed", as_type="embedding", capture_input=False, capture_output=False)
    def embed(self, texts: list[str]) -> list[list[float]]:
        update_current_generation(model=self.embed_model, input={"count": len(texts)})
        resp = self._http.post("/embeddings", json={"model": self.embed_model, "input": texts})
        if resp.status_code != 200:
            raise ProviderError(f"{self.label} embed failed ({resp.status_code}): {resp.text[:200]}")
        data = sorted(resp.json().get("data", []), key=lambda d: d.get("index", 0))
        out = [d["embedding"] for d in data]
        if len(out) != len(texts):
            raise ProviderError(f"{self.label} embed returned wrong number of vectors")
        update_current_generation(output={"count": len(out)})
        return out

    @observe(name="llm.chat", as_type="generation", capture_input=False, capture_output=False)
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
            "model_parameters": {"temperature": temperature},
        }
        if trace_name:
            gen_kwargs["name"] = f"{self.label}.chat:{trace_name}"
        update_current_generation(**gen_kwargs)

        payload: dict = {
            "model": self.gen_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if format is not None:
            payload["response_format"] = {"type": "json_object"}
        resp = self._http.post("/chat/completions", json=payload)
        if resp.status_code != 200:
            raise ProviderError(f"{self.label} chat failed ({resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        update_current_generation(
            output=content,
            usage_details={"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0)},
        )
        return content

    def chat_json(
        self, system: str, user: str, schema: dict, temperature: float = 0.1, trace_name: str | None = None
    ) -> Any:
        schema_hint = f"{user}\n\nRespond with JSON only, matching this schema:\n{json.dumps(schema)}"
        for attempt in (1, 2):
            raw = self.chat(system, schema_hint, temperature=temperature, format=schema, trace_name=trace_name)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("%s: structured output was not valid JSON (attempt %d)", self.label, attempt)
        raise ProviderError(f"{self.label} model failed to produce valid JSON output")

    def close(self) -> None:
        self._http.close()


class AnthropicClient:
    """Chat via the official ``anthropic`` SDK. Claude has no embeddings
    endpoint — set ``Config.embed_provider`` to another provider (e.g.
    ``"ollama"``) whenever ``Config.provider`` is ``"anthropic"``;
    ``build_client`` enforces this at startup rather than failing on first
    use. Sampling params (temperature/top_p/top_k) are omitted entirely: they
    are rejected on current Claude models, so they are not sent at all rather
    than passed through and 400ing."""

    embed_model = ""
    embed_provider_name = "anthropic"
    doc_prefix = ""
    query_prefix = ""

    def __init__(self, api_key: str, gen_model: str = "claude-opus-5", timeout: float = 300.0) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "the 'anthropic' package is required for provider='anthropic'; "
                "install with: pip install -e '.[anthropic]'"
            ) from exc
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set in the environment")
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self.gen_model = gen_model

    def is_up(self) -> bool:
        try:
            next(iter(self._client.models.list()), None)
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        return [m.id for m in self._client.models.list()]

    def missing_models(self) -> list[str]:
        return []

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ProviderError(
            "Claude has no embeddings API; set embed_provider in config.toml "
            "to another provider (e.g. 'ollama') for document/query embeddings"
        )

    @observe(name="anthropic.chat", as_type="generation", capture_input=False, capture_output=False)
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        format: dict | None = None,
        num_ctx: int | None = None,
        trace_name: str | None = None,
    ) -> str:
        update_current_generation(model=self.gen_model, input={"system_chars": len(system), "user_chars": len(user)})
        kwargs: dict = {
            "model": self.gen_model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if format is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": format}}
        response = self._client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            raise ProviderError("Claude declined the request (safety refusal)")
        content = "".join(b.text for b in response.content if b.type == "text").strip()
        update_current_generation(
            output=content,
            usage_details={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
        )
        return content

    def chat_json(
        self, system: str, user: str, schema: dict, temperature: float = 0.1, trace_name: str | None = None
    ) -> Any:
        raw = self.chat(system, user, temperature=temperature, format=schema, trace_name=trace_name)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError("Claude failed to produce valid JSON output") from exc

    def close(self) -> None:
        pass


class GeminiClient:
    """Chat + embeddings via Google's official ``google-genai`` SDK."""

    embed_provider_name = "gemini"
    doc_prefix = ""
    query_prefix = ""

    def __init__(
        self,
        api_key: str,
        gen_model: str = "gemini-2.5-flash",
        embed_model: str = "text-embedding-004",
        timeout: float = 300.0,
    ) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise ProviderError(
                "the 'google-genai' package is required for provider='gemini'; "
                "install with: pip install -e '.[gemini]'"
            ) from exc
        if not api_key:
            raise ProviderError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in the environment")
        self._genai_types = genai_types
        self._client = genai.Client(api_key=api_key)
        self.gen_model = gen_model
        self.embed_model = embed_model

    def is_up(self) -> bool:
        try:
            next(iter(self._client.models.list()), None)
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        return [m.name for m in self._client.models.list()]

    def missing_models(self) -> list[str]:
        return []

    @observe(name="gemini.embed", as_type="embedding", capture_input=False, capture_output=False)
    def embed(self, texts: list[str]) -> list[list[float]]:
        update_current_generation(model=self.embed_model, input={"count": len(texts)})
        result = self._client.models.embed_content(model=self.embed_model, contents=texts)
        out = [e.values for e in result.embeddings]
        update_current_generation(output={"count": len(out)})
        return out

    @observe(name="gemini.chat", as_type="generation", capture_input=False, capture_output=False)
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        format: dict | None = None,
        num_ctx: int | None = None,
        trace_name: str | None = None,
    ) -> str:
        update_current_generation(model=self.gen_model, input={"system_chars": len(system), "user_chars": len(user)})
        config_kwargs: dict = {"system_instruction": system, "temperature": temperature}
        if format is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = format
        response = self._client.models.generate_content(
            model=self.gen_model,
            contents=user,
            config=self._genai_types.GenerateContentConfig(**config_kwargs),
        )
        content = (response.text or "").strip()
        update_current_generation(output=content)
        return content

    def chat_json(
        self, system: str, user: str, schema: dict, temperature: float = 0.1, trace_name: str | None = None
    ) -> Any:
        raw = self.chat(system, user, temperature=temperature, format=schema, trace_name=trace_name)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError("Gemini failed to produce valid JSON output") from exc

    def close(self) -> None:
        pass


class UnavailableClient:
    """Stand-in for callers that need *a* client object but don't strictly
    need the backend (``vault status``, ``vault docs``, ``vault folders``,
    the FastAPI app at startup). Used when ``build_client`` raises
    ``ProviderError`` (missing API key, missing optional SDK, unknown
    provider) and the caller doesn't want that to be fatal — mirrors how
    ``OllamaClient`` never failed at construction time, so read-only commands
    kept working even with Ollama down. Reports itself as down; any attempt
    to actually use it (chat/embed/list_models) raises the original error."""

    gen_model = ""
    embed_model = ""
    doc_prefix = ""
    query_prefix = ""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def is_up(self) -> bool:
        return False

    def missing_models(self) -> list[str]:
        return []

    def list_models(self) -> list[str]:
        raise ProviderError(self._reason)

    def chat(self, *args: Any, **kwargs: Any) -> str:
        raise ProviderError(self._reason)

    def chat_json(self, *args: Any, **kwargs: Any) -> Any:
        raise ProviderError(self._reason)

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ProviderError(self._reason)

    def close(self) -> None:
        pass


class CompositeClient:
    """Delegates chat to one client and embeddings to another. Needed when
    the chat provider has no embeddings API (Claude) or the user simply
    prefers a different embedding provider (``Config.embed_provider``)."""

    def __init__(self, chat_client: Any, embed_client: Any) -> None:
        self._chat = chat_client
        self._embed = embed_client
        self.gen_model = chat_client.gen_model
        self.embed_model = embed_client.embed_model
        self.doc_prefix = embed_client.doc_prefix
        self.query_prefix = embed_client.query_prefix

    def chat(self, *args: Any, **kwargs: Any) -> str:
        return self._chat.chat(*args, **kwargs)

    def chat_json(self, *args: Any, **kwargs: Any) -> Any:
        return self._chat.chat_json(*args, **kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed.embed(texts)

    def is_up(self) -> bool:
        return self._chat.is_up() and self._embed.is_up()

    def missing_models(self) -> list[str]:
        seen: dict[str, None] = {}
        for m in self._chat.missing_models() + self._embed.missing_models():
            seen[m] = None
        return list(seen)

    def list_models(self) -> list[str]:
        return self._chat.list_models()

    def close(self) -> None:
        for c in (self._chat, self._embed):
            close = getattr(c, "close", None)
            if callable(close):
                close()


def _require_key(env_var: str, provider: str) -> str:
    val = os.environ.get(env_var)
    if not val:
        raise ProviderError(f"{env_var} is not set in the environment (required for provider='{provider}')")
    return val


def _resolved_gen_model(cfg: Any, provider: str) -> str:
    if cfg.gen_model != _OLLAMA_DEFAULT_GEN:
        return cfg.gen_model
    return _DEFAULT_GEN_MODEL.get(provider, cfg.gen_model)


def _resolved_embed_model(cfg: Any, provider: str) -> str:
    if cfg.embed_model != _OLLAMA_DEFAULT_EMBED:
        return cfg.embed_model
    return _DEFAULT_EMBED_MODEL.get(provider, cfg.embed_model)


def _build_single(provider: str, cfg: Any) -> Any:
    if provider == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient.from_config(cfg)

    if provider == "openai":
        return OpenAICompatibleClient(
            base_url=cfg.api_base_url or "https://api.openai.com/v1",
            api_key=_require_key("OPENAI_API_KEY", "openai"),
            gen_model=_resolved_gen_model(cfg, "openai"),
            embed_model=_resolved_embed_model(cfg, "openai"),
            timeout=cfg.request_timeout,
            label="openai",
        )

    if provider == "vllm":
        gen_model = _resolved_gen_model(cfg, "vllm")
        if not gen_model or gen_model == _OLLAMA_DEFAULT_GEN:
            raise ProviderError("set gen_model in config.toml to the model your vLLM server is serving")
        return OpenAICompatibleClient(
            base_url=cfg.api_base_url or "http://localhost:8000/v1",
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            gen_model=gen_model,
            embed_model=_resolved_embed_model(cfg, "vllm"),
            timeout=cfg.request_timeout,
            label="vllm",
        )

    if provider == "openai_compatible":
        if not cfg.api_base_url:
            raise ProviderError("set api_base_url in config.toml for provider='openai_compatible'")
        gen_model = _resolved_gen_model(cfg, "openai_compatible")
        if not gen_model or gen_model == _OLLAMA_DEFAULT_GEN:
            raise ProviderError("set gen_model in config.toml for provider='openai_compatible'")
        return OpenAICompatibleClient(
            base_url=cfg.api_base_url,
            api_key=os.environ.get("LLM_API_KEY", ""),
            gen_model=gen_model,
            embed_model=_resolved_embed_model(cfg, "openai_compatible"),
            timeout=cfg.request_timeout,
            label="openai_compatible",
        )

    if provider == "anthropic":
        return AnthropicClient(
            api_key=_require_key("ANTHROPIC_API_KEY", "anthropic"),
            gen_model=_resolved_gen_model(cfg, "anthropic"),
            timeout=cfg.request_timeout,
        )

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        if not api_key:
            raise ProviderError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in the environment")
        return GeminiClient(
            api_key=api_key,
            gen_model=_resolved_gen_model(cfg, "gemini"),
            embed_model=_resolved_embed_model(cfg, "gemini"),
            timeout=cfg.request_timeout,
        )

    raise ProviderError(f"unknown provider '{provider}'; choose one of: {', '.join(PROVIDERS)}")


def build_client(cfg: Any) -> Any:
    """Construct the LLM client(s) for cfg.provider (+ cfg.embed_provider, if
    it names a different provider than cfg.provider). Fails fast — before
    any network call — on an unknown provider, a missing API key, a missing
    optional SDK, or embed_provider left pointed at Claude (which has no
    embeddings API)."""
    if cfg.provider not in PROVIDERS:
        raise ProviderError(f"unknown provider '{cfg.provider}'; choose one of: {', '.join(PROVIDERS)}")

    embed_provider = cfg.embed_provider or cfg.provider
    if embed_provider not in PROVIDERS:
        raise ProviderError(f"unknown embed_provider '{embed_provider}'; choose one of: {', '.join(PROVIDERS)}")
    if embed_provider == "anthropic":
        raise ProviderError(
            "Claude has no embeddings API — set embed_provider in config.toml to another "
            "provider (e.g. 'ollama') when provider='anthropic'"
        )

    chat_client = _build_single(cfg.provider, cfg)
    if embed_provider == cfg.provider:
        return chat_client
    embed_client = _build_single(embed_provider, cfg)
    return CompositeClient(chat_client, embed_client)
