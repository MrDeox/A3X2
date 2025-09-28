import json
from typing import Any

import httpx
import pytest

from a3x.actions import ActionType, AgentState
from a3x.llm import OpenRouterLLMClient


class DummyResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class DummyClient:
    def __init__(self, payload: dict[str, Any], *, fail_first: bool = False) -> None:
        self.payload = payload
        self.last_request: dict[str, Any] | None = None
        self.fail_first = fail_first
        self.calls = 0

    def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> DummyResponse:  # type: ignore[override]
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise httpx.TimeoutException("timeout")
        self.last_request = {"url": url, "headers": headers, "json": json}
        return DummyResponse(200, self.payload)

    def __enter__(self) -> "DummyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None


@pytest.fixture
def agent_state() -> AgentState:
    return AgentState(goal="Teste", history_snapshot="", iteration=1, max_iterations=5)


def test_openrouter_client_parses_action(monkeypatch, agent_state) -> None:
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "type": "run_command",
                            "command": ["ls", "-la"],
                        }
                    )
                }
            }
        ]
    }

    dummy_client = DummyClient(response_payload)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr("httpx.Client", lambda timeout: dummy_client)

    client = OpenRouterLLMClient(
        model="x-ai/grok-4-fast:free", base_url="https://example.com/api/v1"
    )
    client.start("Teste")

    action = client.propose_action(agent_state)

    assert action.type is ActionType.RUN_COMMAND
    assert action.command == ["ls", "-la"]
    assert dummy_client.last_request is not None
    assert dummy_client.last_request["url"].endswith("/chat/completions")


def test_openrouter_client_requires_json(monkeypatch, agent_state) -> None:
    response_payload = {"choices": [{"message": {"content": "not-json"}}]}

    dummy_client = DummyClient(response_payload)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr("httpx.Client", lambda timeout: dummy_client)

    client = OpenRouterLLMClient(model="x-ai/grok-4-fast:free")
    client.start("Goal")

    with pytest.raises(RuntimeError):
        client.propose_action(agent_state)


def test_openrouter_client_retries(
    monkeypatch, agent_state, monkeypatch_time_sleep
) -> None:
    response_payload = {
        "choices": [
            {"message": {"content": json.dumps({"type": "message", "text": "ok"})}}
        ]
    }

    dummy_client = DummyClient(response_payload, fail_first=True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr("httpx.Client", lambda timeout: dummy_client)

    client = OpenRouterLLMClient(model="x-ai/grok-4-fast:free", max_retries=2)
    client.start("Goal")

    action = client.propose_action(agent_state)

    assert action.type is ActionType.MESSAGE
    assert dummy_client.calls == 2
    metrics = client.get_last_metrics()
    assert metrics["llm_retries"] == 1.0


@pytest.fixture
def monkeypatch_time_sleep(monkeypatch):
    calls = []

    def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr("a3x.llm.time.sleep", fake_sleep)
    return calls
