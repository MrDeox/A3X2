"""Local embedding helper using sentence-transformers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable, List

_MODEL_LOCK = threading.Lock()
_MODEL_INSTANCE: "EmbeddingModel | None" = None


@dataclass
class EmbeddingModel:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except (
            ImportError
        ) as exc:  # pragma: no cover - executed only when dependency missing
            raise RuntimeError(
                "Pacote 'sentence-transformers' não encontrado. Instale com 'pip install sentence-transformers'."
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


def get_embedder() -> EmbeddingModel:
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE
    with _MODEL_LOCK:
        if _MODEL_INSTANCE is None:
            _MODEL_INSTANCE = EmbeddingModel()
    return _MODEL_INSTANCE


def set_embedder(embedder: EmbeddingModel | None) -> None:
    """Override the global embedder (useful for tests)."""

    global _MODEL_INSTANCE
    with _MODEL_LOCK:
        _MODEL_INSTANCE = embedder
