"""Auto-generated unit tests for a3x/memory/insights.py.
AUTO-GENERATED. Edit via InsightsTestGenerator."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import hashlib

from a3x.actions import AgentState
from a3x.memory.insights import StatefulRetriever, Insight
from a3x.memory.store import MemoryEntry, SemanticMemory


@pytest.fixture
def mock_store():
    store = MagicMock(spec=SemanticMemory)
    mock_entry = MemoryEntry(
        id="test_id",
        created_at=datetime.now(timezone.utc).isoformat(),
        title="Test Insight",
        content="Test content with recent failure in actions.",
        tags=["test", "failure"],
        metadata={"goal": "test_goal", "snapshot_hash": "abc123"},
        embedding=[0.1] * 384  # Mock embedding
    )
    store.query.return_value = [(mock_entry, 0.8), (mock_entry, 0.6)]  # One >0.7
    store.entries = [mock_entry]
    return store


@pytest.fixture
def mock_state():
    return AgentState(
        goal="Test goal",
        history_snapshot="Test history with failures.",
        iteration=1,
        max_iterations=5,
        seed_context="Test context"
    )


def test_stateful_retriever_retrieve_session_context(mock_store, mock_state):
    retriever = StatefulRetriever()
    with patch.object(retriever, 'store', mock_store):
        insights = retriever.retrieve_session_context(mock_state)
    
    assert len(insights) == 1  # Filtered to >0.7
    assert isinstance(insights[0], Insight)
    assert insights[0].similarity > 0.7
    mock_store.query.assert_called_once()

def test_stateful_retriever_derivation_detection(mock_store, mock_state):
    retriever = StatefulRetriever()
    mock_state.history_snapshot = "Changed history"
    with patch.object(retriever, 'store', mock_store):
        insights = retriever.retrieve_session_context(mock_state)
    
    if insights:
        assert insights[0].metadata.get("derivation_flagged") is True  # Hash differs
        assert "snapshot_hash" in insights[0].metadata

def test_stateful_retriever_update_snapshot_hash():
    retriever = StatefulRetriever()
    snapshot = "Test snapshot"
    retriever.update_snapshot_hash(snapshot)
    assert retriever.last_snapshot_hash == hashlib.md5(snapshot.encode()).hexdigest()

def test_insight_from_entry():
    mock_entry = MemoryEntry(
        id="id", created_at="2023-01-01T00:00:00Z",
        title="Title", content="Content", tags=["tag"],
        metadata={"key": "value"},
        embedding=[0.1]*384
    )
    insight = Insight.from_entry(mock_entry, 0.85)
    assert insight.title == "Title"
    assert insight.similarity == 0.85
    assert insight.created_at == "2023-01-01T00:00:00Z"
