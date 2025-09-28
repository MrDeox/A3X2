"""Testes para o módulo LLM, focando no fallback para Ollama."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from a3x.llm import OpenRouterLLMClient
from a3x.actions import AgentState, ActionType


@pytest.fixture
def mock_ollama_client():
    client = Mock()
    response = {
        "message": {
            "role": "assistant",
            "content": '{"type": "message", "text": "Test response from Ollama"}'
        }
    }
    client.chat.return_value = response
    return client


def test_ollama_fallback_on_rate_limit(mock_ollama_client):
    # Mock httpx.Client to simulate 429 response
    with patch("httpx.Client") as mock_client_class:
        # Mock the Client instance as MagicMock for context manager support
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock the context manager
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.__exit__.return_value = None
        
        # Mock the response from post
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_response.json.return_value = {}  # Not used in fallback
        mock_client_instance.post.return_value = mock_response

        # Mock ollama Client
        with patch("a3x.llm.Client", return_value=mock_ollama_client):
            client = OpenRouterLLMClient(
                model="x-ai/grok-4-fast:free",
                api_key_env="OPENROUTER_API_KEY",
            )
            client.start("Test goal")

            # Create state
            state = AgentState(
                goal="Test goal",
                history_snapshot="Test history",
                iteration=1,
                max_iterations=5,
                seed_context="Test context",
            )

            # Call propose_action to trigger the request and fallback
            action = client.propose_action(state)

            # Assert fallback was used (action from Ollama)
            assert action.type == ActionType.MESSAGE
            assert action.text == "Test response from Ollama"

            # Verify Ollama was called
            mock_ollama_client.chat.assert_called_once()

            # Check metrics for fallback usage
            metrics = client.get_last_metrics()
            assert metrics["llm_status_code"] == 429.0
            assert metrics["llm_fallback_used"] == 1.0
