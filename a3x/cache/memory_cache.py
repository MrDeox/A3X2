"""Memory operation caching system for embeddings and semantic operations."""

import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import wraps

from .core import BaseCache, CacheStrategy, TTLCache, LRUCache, cache_decorator


@dataclass
class EmbeddingCacheKey:
    """Cache key for embedding requests."""
    text_hash: str
    model_name: str
    method: str  # "sentence_transformer", "tfidf", etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing."""
        return {
            "text_hash": self.text_hash,
            "model_name": self.model_name,
            "method": self.method
        }


@dataclass
class MemoryQueryCacheKey:
    """Cache key for memory query requests."""
    query_hash: str
    top_k: int
    filter_tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing."""
        return {
            "query_hash": self.query_hash,
            "top_k": self.top_k,
            "filter_tags": sorted(filter_tags) if self.filter_tags else None
        }


@dataclass
class SimilarityCacheKey:
    """Cache key for similarity calculations."""
    vec_a_hash: str
    vec_b_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing."""
        return {
            "vec_a_hash": self.vec_a_hash,
            "vec_b_hash": self.vec_b_hash
        }


@dataclass
class EmbeddingCacheEntry:
    """Embedding cache entry."""
    embedding: List[float]
    model_name: str
    method: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class MemoryQueryCacheEntry:
    """Memory query cache entry."""
    results: List[Tuple[Any, float]]  # List of (entry, score) tuples
    query_text: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class SimilarityCacheEntry:
    """Similarity calculation cache entry."""
    similarity: float
    vec_a_length: int
    vec_b_length: int
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


class EmbeddingCache:
    """Specialized cache for text embeddings."""

    def __init__(self, max_size: int = 2000, ttl: float = 7200,  # 2 hours for embeddings
                 strategy: CacheStrategy = CacheStrategy.LRU):
        if strategy == CacheStrategy.TTL:
            self.cache = TTLCache(max_size=max_size, default_ttl=ttl)
        else:
            self.cache = LRUCache(max_size=max_size, strategy=strategy)

        self.stats: Dict[str, int] = {"hits": 0, "misses": 0}

    def _generate_text_hash(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _create_embedding_key(self, text: str, model_name: str, method: str) -> EmbeddingCacheKey:
        """Create cache key for embedding."""
        return EmbeddingCacheKey(
            text_hash=self._generate_text_hash(text),
            model_name=model_name,
            method=method
        )

    def _cache_key_to_string(self, key: EmbeddingCacheKey) -> str:
        """Convert embedding cache key to string."""
        return hashlib.sha256(json.dumps(key.to_dict(), sort_keys=True).encode()).hexdigest()

    def get_embedding(self, text: str, model_name: str, method: str) -> Optional[List[float]]:
        """Get cached embedding."""
        cache_key = self._create_embedding_key(text, model_name, method)
        cache_key_str = self._cache_key_to_string(cache_key)

        cached_result = self.cache.get(cache_key_str)
        if cached_result:
            entry = EmbeddingCacheEntry(**cached_result)
            self.stats["hits"] += 1
            return entry.embedding

        self.stats["misses"] += 1
        return None

    def put_embedding(self, text: str, embedding: List[float], model_name: str, method: str) -> None:
        """Cache embedding result."""
        cache_key = self._create_embedding_key(text, model_name, method)
        cache_key_str = self._cache_key_to_string(cache_key)

        entry = EmbeddingCacheEntry(
            embedding=embedding,
            model_name=model_name,
            method=method
        )

        self.cache.put(cache_key_str, entry.__dict__)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0}

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        return {
            "embedding_cache_size": len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
            "total_requests": total_requests,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": self.stats["hits"] / total_requests if total_requests > 0 else 0.0,
            "cache_metrics": self.cache.metrics.__dict__
        }


class MemoryQueryCache:
    """Specialized cache for memory query results."""

    def __init__(self, max_size: int = 1000, ttl: float = 1800,  # 30 minutes for queries
                 strategy: CacheStrategy = CacheStrategy.LRU):
        if strategy == CacheStrategy.TTL:
            self.cache = TTLCache(max_size=max_size, default_ttl=ttl)
        else:
            self.cache = LRUCache(max_size=max_size, strategy=strategy)

        self.stats: Dict[str, int] = {"hits": 0, "misses": 0}

    def _generate_query_hash(self, query_text: str) -> str:
        """Generate hash for query text."""
        return hashlib.sha256(query_text.encode()).hexdigest()

    def _create_query_key(self, query_text: str, top_k: int, filter_tags: Optional[List[str]]) -> MemoryQueryCacheKey:
        """Create cache key for memory query."""
        return MemoryQueryCacheKey(
            query_hash=self._generate_query_hash(query_text),
            top_k=top_k,
            filter_tags=filter_tags
        )

    def _cache_key_to_string(self, key: MemoryQueryCacheKey) -> str:
        """Convert query cache key to string."""
        return hashlib.sha256(json.dumps(key.to_dict(), sort_keys=True).encode()).hexdigest()

    def get_query_results(self, query_text: str, top_k: int, filter_tags: Optional[List[str]]) -> Optional[List[Tuple[Any, float]]]:
        """Get cached query results."""
        cache_key = self._create_query_key(query_text, top_k, filter_tags)
        cache_key_str = self._cache_key_to_string(cache_key)

        cached_result = self.cache.get(cache_key_str)
        if cached_result:
            entry = MemoryQueryCacheEntry(**cached_result)
            self.stats["hits"] += 1
            return entry.results

        self.stats["misses"] += 1
        return None

    def put_query_results(self, query_text: str, results: List[Tuple[Any, float]], top_k: int, filter_tags: Optional[List[str]]) -> None:
        """Cache query results."""
        cache_key = self._create_query_key(query_text, top_k, filter_tags)
        cache_key_str = self._cache_key_to_string(cache_key)

        entry = MemoryQueryCacheEntry(
            results=results,
            query_text=query_text
        )

        self.cache.put(cache_key_str, entry.__dict__)

    def clear(self) -> None:
        """Clear all cached query results."""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0}

    def get_stats(self) -> Dict[str, Any]:
        """Get query cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        return {
            "query_cache_size": len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
            "total_requests": total_requests,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": self.stats["hits"] / total_requests if total_requests > 0 else 0.0,
            "cache_metrics": self.cache.metrics.__dict__
        }


class SimilarityCache:
    """Specialized cache for vector similarity calculations."""

    def __init__(self, max_size: int = 5000, ttl: float = 3600,  # 1 hour for similarities
                 strategy: CacheStrategy = CacheStrategy.LRU):
        if strategy == CacheStrategy.TTL:
            self.cache = TTLCache(max_size=max_size, default_ttl=ttl)
        else:
            self.cache = LRUCache(max_size=max_size, strategy=strategy)

        self.stats: Dict[str, int] = {"hits": 0, "misses": 0}

    def _generate_vector_hash(self, vec: List[float]) -> str:
        """Generate hash for vector content."""
        # Convert vector to string representation for hashing
        vec_str = json.dumps(vec, sort_keys=True)
        return hashlib.sha256(vec_str.encode()).hexdigest()

    def _create_similarity_key(self, vec_a: List[float], vec_b: List[float]) -> SimilarityCacheKey:
        """Create cache key for similarity calculation."""
        # Ensure consistent ordering (smaller hash first)
        hash_a = self._generate_vector_hash(vec_a)
        hash_b = self._generate_vector_hash(vec_b)

        if hash_a < hash_b:
            return SimilarityCacheKey(vec_a_hash=hash_a, vec_b_hash=hash_b)
        else:
            return SimilarityCacheKey(vec_a_hash=hash_b, vec_b_hash=hash_a)

    def _cache_key_to_string(self, key: SimilarityCacheKey) -> str:
        """Convert similarity cache key to string."""
        return hashlib.sha256(json.dumps(key.to_dict(), sort_keys=True).encode()).hexdigest()

    def get_similarity(self, vec_a: List[float], vec_b: List[float]) -> Optional[float]:
        """Get cached similarity score."""
        cache_key = self._create_similarity_key(vec_a, vec_b)
        cache_key_str = self._cache_key_to_string(cache_key)

        cached_result = self.cache.get(cache_key_str)
        if cached_result:
            entry = SimilarityCacheEntry(**cached_result)
            self.stats["hits"] += 1
            return entry.similarity

        self.stats["misses"] += 1
        return None

    def put_similarity(self, vec_a: List[float], vec_b: List[float], similarity: float) -> None:
        """Cache similarity calculation."""
        cache_key = self._create_similarity_key(vec_a, vec_b)
        cache_key_str = self._cache_key_to_string(cache_key)

        entry = SimilarityCacheEntry(
            similarity=similarity,
            vec_a_length=len(vec_a),
            vec_b_length=len(vec_b)
        )

        self.cache.put(cache_key_str, entry.__dict__)

    def clear(self) -> None:
        """Clear all cached similarities."""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0}

    def get_stats(self) -> Dict[str, Any]:
        """Get similarity cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        return {
            "similarity_cache_size": len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
            "total_requests": total_requests,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": self.stats["hits"] / total_requests if total_requests > 0 else 0.0,
            "cache_metrics": self.cache.metrics.__dict__
        }


class MemoryOperationCache:
    """Combined cache for all memory operations."""

    def __init__(self, embedding_cache_size: int = 2000, query_cache_size: int = 1000,
                 similarity_cache_size: int = 5000, ttl: float = 3600,
                 strategy: CacheStrategy = CacheStrategy.LRU):
        self.embedding_cache = EmbeddingCache(
            max_size=embedding_cache_size,
            ttl=ttl,
            strategy=strategy
        )
        self.query_cache = MemoryQueryCache(
            max_size=query_cache_size,
            ttl=ttl // 2,  # Queries expire faster
            strategy=strategy
        )
        self.similarity_cache = SimilarityCache(
            max_size=similarity_cache_size,
            ttl=ttl,
            strategy=strategy
        )

    def get_cached_embedding(self, text: str, model_name: str, method: str) -> Optional[List[float]]:
        """Get cached embedding or None if not found."""
        return self.embedding_cache.get_embedding(text, model_name, method)

    def cache_embedding(self, text: str, embedding: List[float], model_name: str, method: str) -> None:
        """Cache embedding result."""
        self.embedding_cache.put_embedding(text, embedding, model_name, method)

    def get_cached_query(self, query_text: str, top_k: int, filter_tags: Optional[List[str]]) -> Optional[List[Tuple[Any, float]]]:
        """Get cached query results or None if not found."""
        return self.query_cache.get_query_results(query_text, top_k, filter_tags)

    def cache_query(self, query_text: str, results: List[Tuple[Any, float]], top_k: int, filter_tags: Optional[List[str]]) -> None:
        """Cache query results."""
        self.query_cache.put_query_results(query_text, results, top_k, filter_tags)

    def get_cached_similarity(self, vec_a: List[float], vec_b: List[float]) -> Optional[float]:
        """Get cached similarity or None if not found."""
        return self.similarity_cache.get_similarity(vec_a, vec_b)

    def cache_similarity(self, vec_a: List[float], vec_b: List[float], similarity: float) -> None:
        """Cache similarity calculation."""
        self.similarity_cache.put_similarity(vec_a, vec_b, similarity)

    def clear_all(self) -> None:
        """Clear all memory caches."""
        self.embedding_cache.clear()
        self.query_cache.clear()
        self.similarity_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics for all memory caches."""
        return {
            "embedding_cache": self.embedding_cache.get_stats(),
            "query_cache": self.query_cache.get_stats(),
            "similarity_cache": self.similarity_cache.get_stats(),
            "total_caches": 3
        }


def cached_cosine_similarity(cache_name: str = "cosine_similarity", ttl: Optional[float] = None,
                           strategy: CacheStrategy = CacheStrategy.LRU):
    """Decorator for caching cosine similarity calculations."""
    def decorator(func):
        @wraps(func)
        def wrapper(vec_a, vec_b, *args, **kwargs):
            # Get cache manager from function context or create default
            cache_manager = getattr(func, '_memory_cache', None)
            if cache_manager is None:
                cache_manager = MemoryOperationCache()
                func._memory_cache = cache_manager

            # Try cache first
            cached = cache_manager.get_cached_similarity(vec_a, vec_b)
            if cached is not None:
                return cached

            # Execute calculation
            result = func(vec_a, vec_b, *args, **kwargs)

            # Cache the result
            cache_manager.cache_similarity(vec_a, vec_b, result)

            return result

        return wrapper
    return decorator


class MemoryCacheManager:
    """Manager for multiple memory operation caches."""

    def __init__(self):
        self.caches: Dict[str, MemoryOperationCache] = {}
        self.global_stats: Dict[str, Any] = {}

    def get_cache(self, name: str, **kwargs) -> MemoryOperationCache:
        """Get or create memory operation cache with specific configuration."""
        if name not in self.caches:
            self.caches[name] = MemoryOperationCache(**kwargs)
        return self.caches[name]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all memory caches."""
        return {
            "caches": {
                name: cache.get_stats()
                for name, cache in self.caches.items()
            },
            "total_caches": len(self.caches)
        }

    def clear_all(self) -> None:
        """Clear all memory operation caches."""
        for cache in self.caches.values():
            cache.clear_all()


# Global instance for easy access
memory_cache_manager = MemoryCacheManager()