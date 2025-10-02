"""Core caching infrastructure with multiple strategies and performance monitoring."""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union, Generic, TypeVar
from collections import OrderedDict

import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

T = TypeVar('T')


class CacheStrategy(Enum):
    """Available caching strategies."""
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # Time to live in seconds
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def access(self) -> None:
        """Update access metadata."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class PerformanceMetrics:
    """Performance metrics for cache operations."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    hit_rate: float = 0.0

    def update_metrics(self, hit: bool, response_time: float) -> None:
        """Update performance metrics."""
        self.total_requests += 1
        if hit:
            self.hits += 1
        else:
            self.misses += 1

        # Update hit rate
        self.hit_rate = self.hits / self.total_requests if self.total_requests > 0 else 0.0

        # Update average response time (simple moving average)
        if self.total_requests == 1:
            self.avg_response_time = response_time
        else:
            # Exponential moving average for recent performance
            alpha = 0.1
            self.avg_response_time = (alpha * response_time +
                                    (1 - alpha) * self.avg_response_time)

        # Update memory usage
        process = psutil.Process()
        self.memory_usage_mb = process.memory_info().rss / 1024 / 1024


class BaseCache(ABC, Generic[T]):
    """Abstract base class for cache implementations."""

    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self.metrics = PerformanceMetrics()
        self._lock = threading.RLock()

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        pass

    @abstractmethod
    def put(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        pass

    @abstractmethod
    def evict(self) -> None:
        """Evict entries based on strategy."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(json.dumps(value).encode('utf-8'))
        except (TypeError, ValueError):
            return 1024  # Default size for non-serializable objects

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps((args, sorted(kwargs.items())), sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()


class LRUCache(BaseCache[T]):
    """Least Recently Used cache implementation."""

    def __init__(self, max_size: int = 1000, **kwargs):
        super().__init__(max_size, CacheStrategy.LRU, **kwargs)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Optional[T]:
        """Get value from LRU cache."""
        start_time = time.time()

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired():
                    del self._cache[key]
                    self.metrics.evictions += 1
                    hit = False
                else:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    entry.access()
                    hit = True
                    result = entry.value
            else:
                hit = False
                result = None

        response_time = time.time() - start_time
        self.metrics.update_metrics(hit, response_time)

        return result

    def put(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Put value in LRU cache."""
        with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                self.evict()

            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                size_bytes=self._calculate_size(value)
            )

            self._cache[key] = entry

    def evict(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            # Remove oldest entry (first in OrderedDict)
            key, _ = self._cache.popitem(last=False)
            self.metrics.evictions += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()


class TTLCache(BaseCache[T]):
    """Time To Live cache implementation."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600, **kwargs):
        super().__init__(max_size, CacheStrategy.TTL, **kwargs)
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[T]:
        """Get value from TTL cache."""
        start_time = time.time()

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired():
                    del self._cache[key]
                    self.metrics.evictions += 1
                    hit = False
                    result = None
                else:
                    entry.access()
                    hit = True
                    result = entry.value
            else:
                hit = False
                result = None

        response_time = time.time() - start_time
        self.metrics.update_metrics(hit, response_time)

        return result

    def put(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Put value in TTL cache."""
        with self._lock:
            # Clean expired entries periodically
            self._cleanup_expired()

            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                self.evict()

            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl,
                size_bytes=self._calculate_size(value)
            )

            self._cache[key] = entry

    def evict(self) -> None:
        """Evict oldest entry."""
        if self._cache:
            # Find oldest non-expired entry
            oldest_key = None
            oldest_time = float('inf')

            for key, entry in self._cache.items():
                if not entry.is_expired() and entry.created_at < oldest_time:
                    oldest_time = entry.created_at
                    oldest_key = key

            if oldest_key:
                del self._cache[oldest_key]
                self.metrics.evictions += 1

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]
        self.metrics.evictions += len(expired_keys)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()


class CacheManager:
    """Central cache manager with multiple cache instances and performance monitoring."""

    def __init__(self, cache_dir: str = ".cache", default_strategy: CacheStrategy = CacheStrategy.LRU):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_strategy = default_strategy
        self.caches: Dict[str, BaseCache] = {}
        self.metrics_lock = threading.Lock()

        # Performance monitoring
        self.global_metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def get_cache(self, name: str, cache_type: Optional[type] = None,
                  strategy: Optional[CacheStrategy] = None,
                  max_size: int = 1000) -> BaseCache:
        """Get or create a cache instance."""
        if name not in self.caches:
            if cache_type is None:
                if strategy == CacheStrategy.TTL:
                    cache_type = TTLCache
                else:
                    cache_type = LRUCache

            self.caches[name] = cache_type(max_size=max_size, strategy=strategy or self.default_strategy)

        return self.caches[name]

    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get global performance metrics."""
        with self.metrics_lock:
            total_requests = sum(cache.metrics.total_requests for cache in self.caches.values())
            total_hits = sum(cache.metrics.hits for cache in self.caches.values())

            return {
                "global_hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
                "total_requests": total_requests,
                "total_hits": total_hits,
                "cache_count": len(self.caches),
                "cache_metrics": {
                    name: {
                        "hit_rate": cache.metrics.hit_rate,
                        "hits": cache.metrics.hits,
                        "misses": cache.metrics.misses,
                        "evictions": cache.metrics.evictions,
                        "avg_response_time": cache.metrics.avg_response_time,
                        "memory_usage_mb": cache.metrics.memory_usage_mb
                    }
                    for name, cache in self.caches.items()
                }
            }

    def async_get(self, cache_name: str, key: str) -> asyncio.Future:
        """Async cache get operation."""
        return self.executor.submit(self.caches[cache_name].get, key)

    def async_put(self, cache_name: str, key: str, value: T, ttl: Optional[float] = None) -> asyncio.Future:
        """Async cache put operation."""
        return self.executor.submit(self.caches[cache_name].put, key, value, ttl)


def cache_decorator(cache_name: str, ttl: Optional[float] = None,
                   strategy: CacheStrategy = CacheStrategy.LRU):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager from function context or create default
            cache_manager = getattr(func, '_cache_manager', None)
            if cache_manager is None:
                cache_manager = CacheManager()
                func._cache_manager = cache_manager

            cache = cache_manager.get_cache(cache_name, strategy=strategy)
            key = cache._generate_key(func.__name__, args, kwargs)

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(key, result, ttl)
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get cache manager from function context or create default
            cache_manager = getattr(func, '_cache_manager', None)
            if cache_manager is None:
                cache_manager = CacheManager()
                func._cache_manager = cache_manager

            cache = cache_manager.get_cache(cache_name, strategy=strategy)
            key = cache._generate_key(func.__name__, args, kwargs)

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.put(key, result, ttl)
            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator