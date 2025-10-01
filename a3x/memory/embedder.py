"""Local embedding helper using sentence-transformers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import sys
import threading
from dataclasses import dataclass
from typing import Iterable, List

try:
    import sentence_transformers  # type: ignore
except ImportError:
    sentence_transformers = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None
except ImportError:  # pragma: no cover - executed when dependency missing
    import types

    stub_module = types.ModuleType("sentence_transformers")

    class _MissingSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "Pacote 'sentence-transformers' não encontrado. Instale com 'pip install sentence-transformers'."
            )

        def encode(self, *args, **kwargs):  # pragma: no cover - stub only
            raise RuntimeError(
                "Pacote 'sentence-transformers' não encontrado. Instale com 'pip install sentence-transformers'."
            )

    stub_module.SentenceTransformer = _MissingSentenceTransformer  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = stub_module

_MODEL_LOCK = threading.Lock()
_MODEL_INSTANCE: "EmbeddingModel | None" = None


@dataclass
class EmbeddingModel:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    use_tfidf: bool = False

    def __post_init__(self) -> None:
        logger = logging.getLogger(__name__)
        if sentence_transformers is None:
            self.use_tfidf = True
            if TfidfVectorizer is None:
                raise RuntimeError("Neither sentence-transformers nor scikit-learn available.")
            self._model = TfidfVectorizer()
            logger.info("sentence-transformers failed, using TF-IDF")
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            self.use_tfidf = True
            if TfidfVectorizer is None:
                raise RuntimeError(f"sentence-transformers failed: {exc}. scikit-learn also unavailable.") from exc
            self._model = TfidfVectorizer()
            logger.warning(f"sentence-transformers failed ({exc}), using TF-IDF")

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        logger = logging.getLogger(__name__)
        texts_list = list(texts)
        if not texts_list:
            return []
        if self.use_tfidf:
            hash_obj = hashlib.sha256(' '.join(texts_list).encode('utf-8')).hexdigest()
            cache_dir = Path("a3x/state")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = cache_dir / f"tfidf_cache_{hash_obj}.json"
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        return json.load(f)
                except Exception:
                    logger.warning("Cache load failed, recomputing TF-IDF")
            try:
                self._model.fit(texts_list)
                vectors = self._model.transform(texts_list).toarray().tolist()
                with open(cache_path, 'w') as f:
                    json.dump(vectors, f)
                return vectors
            except Exception as e:
                logger.error(f"TF-IDF computation failed: {e}")
                # Fallback to zero vectors or simple
                return [[0.0] * 100 for _ in texts_list]  # Dummy
        else:
            vectors = self._model.encode(
                texts_list,
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
