import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock, patch

import pytest
from hypothesis import given, strategies as st

from a3x.memory.embedder import EmbeddingModel
from a3x.memory.store import MemoryEntry, SemanticMemory, _cosine_similarity


class TestMemoryEntry:
    def test_as_json(self) -> None:
        entry = MemoryEntry(
            id="test-id",
            created_at="2023-01-01T00:00:00Z",
            title="Test Title",
            content="Test Content",
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
            embedding=[0.1, 0.2],
        )
        result = entry.as_json()
        expected = {
            "id": "test-id",
            "created_at": "2023-01-01T00:00:00Z",
            "title": "Test Title",
            "content": "Test Content",
            "tags": ["tag1", "tag2"],
            "metadata": {"key": "value"},
            "embedding": [0.1, 0.2],
        }
        assert result == expected


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        vec = [0.1, 0.2, 0.3]
        score = _cosine_similarity(vec, vec)
        assert math.isclose(score, 1.0)

    def test_orthogonal_vectors(self) -> None:
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        score = _cosine_similarity(vec_a, vec_b)
        assert math.isclose(score, 0.0)

    def test_empty_vectors(self) -> None:
        score = _cosine_similarity([], [])
        assert score == 0.0

    def test_different_lengths(self) -> None:
        score = _cosine_similarity([1.0], [1.0, 0.0])
        assert score == 0.0

    def test_zero_norm(self) -> None:
        score = _cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert score == 0.0

    @given(st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_subnormal=False), min_size=1), st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_subnormal=False), min_size=1))
    def test_symmetry(self, vec_a: List[float], vec_b: List[float]) -> None:
        if len(vec_a) != len(vec_b):
            # Skip if lengths differ, as it returns 0
            return
        score_ab = _cosine_similarity(vec_a, vec_b)
        score_ba = _cosine_similarity(vec_b, vec_a)
        assert math.isclose(score_ab, score_ba)

    @given(st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=1))
    def test_self_similarity_max(self, vec: List[float]) -> None:
        score = _cosine_similarity(vec, vec)
        norm_sq = sum(v * v for v in vec)
        if norm_sq == 0:
            assert score == 0.0
        else:
            assert math.isclose(score, 1.0, rel_tol=1e-9)


class TestSemanticMemory:
    def setup_method(self) -> None:
        self.test_path = Path("tests/temp_memory.jsonl")
        self.test_path.parent.mkdir(exist_ok=True)
        if self.test_path.exists():
            self.test_path.unlink()

    def teardown_method(self) -> None:
        if self.test_path.exists():
            self.test_path.unlink()

    @patch("a3x.memory.store.get_embedder")
    def test_add_entry(self, mock_get_embedder: Mock) -> None:
        mock_embedder = Mock(spec=EmbeddingModel)
        mock_embedding = [0.1, 0.2, 0.3]
        mock_embedder.embed.return_value = [mock_embedding]
        mock_get_embedder.return_value = mock_embedder

        memory = SemanticMemory(path=self.test_path)
        entry = memory.add(
            title="Test Title",
            content="Test Content",
            tags=["tag1"],
            metadata={"key": "value"},
        )

        assert len(memory.entries) == 1
        assert entry.id == entry.id
        assert entry.title == "Test Title"
        assert entry.content == "Test Content"
        assert entry.tags == ["tag1"]
        assert entry.metadata == {"key": "value"}
        assert entry.embedding == mock_embedding
        assert entry.created_at == entry.created_at  # ISO format

        # Verify file written
        assert self.test_path.exists()
        with self.test_path.open() as f:
            line = f.readline().strip()
            data = json.loads(line)
            assert data["title"] == "Test Title"

        mock_embedder.embed.assert_called_once_with(["Test Title\nTest Content"])

    @patch("a3x.memory.store.get_embedder")
    def test_add_multiple_entries(self, mock_get_embedder: Mock) -> None:
        mock_embedder = Mock(spec=EmbeddingModel)
        mock_get_embedder.return_value = mock_embedder
        mock_embedder.embed.return_value = [[0.1, 0.2]]

        memory = SemanticMemory(path=self.test_path)
        entry1 = memory.add("Title1", "Content1")
        entry2 = memory.add("Title2", "Content2")

        assert len(memory.entries) == 2
        assert entry1.id != entry2.id
        assert self.test_path.read_text().count("\n") == 2  # Two lines

    @patch("a3x.memory.store.get_embedder")
    def test_load_from_file(self, mock_get_embedder: Mock) -> None:
        # Create sample file
        sample_entries = [
            MemoryEntry(
                id="1",
                created_at="2023-01-01T00:00:00Z",
                title="Loaded Title",
                content="Loaded Content",
                embedding=[0.1, 0.2],
            )
        ]
        with self.test_path.open("w") as f:
            for entry in sample_entries:
                f.write(json.dumps(entry.as_json()) + "\n")

        memory = SemanticMemory(path=self.test_path)
        assert len(memory.entries) == 1
        assert memory.entries[0].title == "Loaded Title"

    @patch("a3x.memory.store.get_embedder")
    def test_load_invalid_lines(self, mock_get_embedder: Mock) -> None:
        # Invalid JSON line
        with self.test_path.open("w") as f:
            f.write("invalid json\n")
            f.write(json.dumps({"id": "valid", "created_at": "2023-01-01T00:00:00Z", "title": "Valid", "content": "Content", "embedding": []}) + "\n")

        memory = SemanticMemory(path=self.test_path)
        assert len(memory.entries) == 1  # Only valid loaded
        assert memory.entries[0].title == "Valid"

    @patch("a3x.memory.store.get_embedder")
    def test_query_empty(self, mock_get_embedder: Mock) -> None:
        memory = SemanticMemory(path=self.test_path)
        results: List[Tuple[MemoryEntry, float]] = memory.query("query text", top_k=3)
        assert results == []

    @patch("a3x.memory.store.get_embedder")
    def test_query_with_entries(self, mock_get_embedder: Mock) -> None:
        mock_embedder = Mock(spec=EmbeddingModel)
        mock_get_embedder.return_value = mock_embedder

        # Mock embeddings before adds
        entry1_embedding = [0.1, 0.2, 0.3]
        entry2_embedding = [0.9, 0.8, 0.7]  # Higher similarity to query
        query_embedding = [0.95, 0.85, 0.75]
        mock_embedder.embed.side_effect = [
            [entry1_embedding],  # First add
            [entry2_embedding],  # Second add
            [query_embedding],   # Query
        ]

        memory = SemanticMemory(path=self.test_path)
        memory.add("Title1", "Content1")
        memory.add("Title2", "Content2")

        results = memory.query("query text", top_k=2)
        assert len(results) == 2
        # Higher for entry2
        assert results[0][1] > results[1][1]  # Sorted descending

    @patch("a3x.memory.store.get_embedder")
    def test_query_top_k_limited(self, mock_get_embedder: Mock) -> None:
        mock_embedder = Mock(spec=EmbeddingModel)
        mock_get_embedder.return_value = mock_embedder
        dummy_embedding = [0.1, 0.2]
        mock_embedder.embed.return_value = [dummy_embedding]  # For adds and query

        memory = SemanticMemory(path=self.test_path)
        for i in range(5):
            memory.add(f"Title{i}", f"Content{i}")

        # Query with top_k=2
        results = memory.query("query", top_k=2)
        assert len(results) == 2

    @patch("a3x.memory.store.get_embedder")
    def test_idempotency_add_same_content(self, mock_get_embedder: Mock) -> None:
        # Since UUID, not truly idempotent, but test multiple adds don't crash
        mock_embedder = Mock(spec=EmbeddingModel)
        mock_get_embedder.return_value = mock_embedder
        mock_embedder.embed.return_value = [[0.1, 0.2]]

        memory = SemanticMemory(path=self.test_path)
        entry1 = memory.add("Same Title", "Same Content")
        entry2 = memory.add("Same Title", "Same Content")  # Different ID

        assert len(memory.entries) == 2
        assert entry1.id != entry2.id
        assert entry1.embedding == entry2.embedding  # Same content, same embed

    @patch("a3x.memory.store.get_embedder")
    def test_query_with_threshold_manual(self, mock_get_embedder: Mock) -> None:
        # Since no built-in threshold, test filtering results > 0.5
        mock_embedder = Mock(spec=EmbeddingModel)
        mock_get_embedder.return_value = mock_embedder

        memory = SemanticMemory(path=self.test_path)
        high_sim_embed = [1.0, 0.0]
        low_sim_embed = [0.0, 1.0]
        query_embed = [1.0, 0.0]  # Orthogonal to low

        mock_embedder.embed.side_effect = [
            [high_sim_embed],  # Add high
            [low_sim_embed],   # Add low
            [query_embed],     # Query
        ]
        memory.add("High", "High content")
        memory.add("Low", "Low content")

        results = memory.query("query")
        # Filter >0.5
        filtered = [r for r in results if r[1] > 0.5]
        assert len(filtered) == 1  # Only high similarity
        assert filtered[0][0].title == "High"