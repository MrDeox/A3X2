"""Tests for the OpenRouter LLM client fallback behaviour."""

from __future__ import annotations

from typing import Any

import pytest

from a3x.llm import OpenRouterLLMClient


class _DummyResponse:
    status_code = 429
    text = "Too Many Requests"

    def json(self) -> dict[str, Any]:  # pragma: no cover - not used in fallback branch
        return {}


class _DummyHttpxClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _DummyHttpxClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None

    def post(self, *args: Any, **kwargs: Any) -> _DummyResponse:
        return _DummyResponse()


def test_openrouter_fallback_handles_missing_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the Ollama fallback reports a controlled error instead of AttributeError."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("a3x.llm.httpx.Client", _DummyHttpxClient)

    client = OpenRouterLLMClient(model="test-model")

    with pytest.raises(RuntimeError) as exc:
        client._send_request([{"role": "user", "content": "ping"}])

    message = str(exc.value)
    assert "fallback Ollama indisponível" in message
    assert "AttributeError" not in message
