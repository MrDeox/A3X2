from unittest.mock import Mock, patch

import numpy as np

from a3x.memory.embedder import EmbeddingModel, get_embedder, set_embedder


class TestEmbeddingModel:
    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_happy_path(self, mock_st: Mock) -> None:
        mock_model = Mock()
        mock_st.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        model = EmbeddingModel()
        texts = ["sample text 1", "sample text 2"]
        result: list[list[float]] = model.embed(texts)

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        mock_model.encode.assert_called_once_with(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_empty_texts(self, mock_st: Mock) -> None:
        mock_model = Mock()
        mock_st.return_value = mock_model
        mock_model.encode.return_value = np.array([])

        model = EmbeddingModel()
        result: list[list[float]] = model.embed([])

        assert result == []
        mock_model.encode.assert_called_once_with(
            [],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_single_text(self, mock_st: Mock) -> None:
        mock_model = Mock()
        mock_st.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2]])

        model = EmbeddingModel()
        result: list[list[float]] = model.embed(["single text"])

        assert len(result) == 1
        assert result[0] == [0.1, 0.2]
        mock_model.encode.assert_called_once_with(
            ["single text"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class TestEmbedderFunctions:
    def test_get_embedder_creates_instance(self) -> None:
        # Clear any existing instance
        set_embedder(None)

        embedder = get_embedder()
        assert isinstance(embedder, EmbeddingModel)

        # Second call should return the same instance
        embedder2 = get_embedder()
        assert embedder is embedder2

    def test_set_embedder_overrides(self) -> None:
        # Clear
        set_embedder(None)

        mock_embedder = Mock(spec=EmbeddingModel)
        set_embedder(mock_embedder)

        result = get_embedder()
        assert result is mock_embedder

    @patch("sentence_transformers.SentenceTransformer")
    def test_get_embedder_with_mock(self, mock_st: Mock) -> None:
        mock_model = Mock()
        mock_st.return_value = mock_model

        set_embedder(None)
        embedder = get_embedder()

        # Verify it was created with mock
        mock_st.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
