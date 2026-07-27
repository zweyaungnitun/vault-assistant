from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from vault_assistant import db


class FakeClient:
    """Deterministic stand-in for OllamaClient: bag-of-words hash embeddings
    (so similar texts really are similar) and canned chat responses."""

    gen_model = "fake-gen"
    embed_model = "fake-embed"

    def __init__(self, chat_response: str = "ok", json_response: dict | None = None):
        self.chat_response = chat_response
        self.json_response = json_response if json_response is not None else {"items": []}
        self.chat_calls: list[tuple[str, str]] = []
        self.chat_responses: list = []
        self.response_index = 0

    @staticmethod
    def _vec(text: str, dim: int = 32) -> list[float]:
        v = np.zeros(dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def chat(self, system: str, user: str, **kw) -> str:
        self.chat_calls.append((system, user))
        if self.chat_responses and self.response_index < len(self.chat_responses):
            response = self.chat_responses[self.response_index]
            self.response_index += 1
            return response
        return self.chat_response

    def chat_json(self, system: str, user: str, schema: dict, **kw):
        self.chat_calls.append((system, user))
        if self.chat_responses and self.response_index < len(self.chat_responses):
            response = self.chat_responses[self.response_index]
            self.response_index += 1
            import json
            return json.loads(response)
        return self.json_response

    def is_up(self) -> bool:
        return True

    def missing_models(self) -> list[str]:
        return []


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    yield c
    c.close()


@pytest.fixture
def client():
    return FakeClient()
