"""LLM response caching system to avoid redundant API calls."""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .core import BaseCache, CacheStrategy, TTLCache, LRUCache, cache_decorator


@dataclass
class LLMCacheKey:
    """Cache key for LLM requests."""
    model: str
    messages_hash: str
    temperature: float
    max_tokens: Optional[int] = None
    system_prompt_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing."""
        return {
            "model": self.model,
            "messages_hash": self.messages_hash,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt_hash": self.system_prompt_hash
        }


@dataclass
class LLMCacheEntry:
    """LLM cache entry with response data."""
    response: Dict[str, Any]
    usage: Dict[str, int]
    model: str
    finish_reason: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

    def get_content(self) -> str:
        """Extract content from response."""
        choices = self.response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
        return ""


class LLMCache:
    """Specialized cache for LLM responses with smart key generation."""

    def __init__(self, max_size: int = 1000, ttl: float = 3600,
                 strategy: CacheStrategy = CacheStrategy.LRU):
        if strategy == CacheStrategy.TTL:
            self.cache = TTLCache(max_size=max_size, default_ttl=ttl)
        else:
            self.cache = LRUCache(max_size=max_size, strategy=strategy)

        self.key_stats: Dict[str, int] = {}
        self.model_stats: Dict[str, Dict[str, int]] = {}

    def _generate_messages_hash(self, messages: List[Dict[str, str]]) -> str:
        """Generate hash for messages content."""
        # Normalize messages for consistent hashing
        normalized = []
        for msg in messages:
            normalized.append({
                "role": msg.get("role", ""),
                "content": msg.get("content", "")
            })

        content = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _generate_system_prompt_hash(self, system_prompt: str) -> str:
        """Generate hash for system prompt."""
        return hashlib.sha256(system_prompt.encode()).hexdigest()

    def _create_cache_key(self, model: str, messages: List[Dict[str, str]],
                          temperature: float = 0.1,
                          max_tokens: Optional[int] = None) -> LLMCacheKey:
        """Create cache key from request parameters."""
        messages_hash = self._generate_messages_hash(messages)

        # Extract system prompt for separate tracking
        system_prompt = ""
        system_prompt_hash = None
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                system_prompt_hash = self._generate_system_prompt_hash(system_prompt)
                break

        return LLMCacheKey(
            model=model,
            messages_hash=messages_hash,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt_hash=system_prompt_hash
        )

    def _cache_key_to_string(self, key: LLMCacheKey) -> str:
        """Convert cache key to string for caching."""
        return hashlib.sha256(json.dumps(key.to_dict(), sort_keys=True).encode()).hexdigest()

    def get(self, model: str, messages: List[Dict[str, str]],
            temperature: float = 0.1, max_tokens: Optional[int] = None) -> Optional[LLMCacheEntry]:
        """Get cached LLM response."""
        cache_key = self._create_cache_key(model, messages, temperature, max_tokens)
        cache_key_str = self._cache_key_to_string(cache_key)

        cached_result = self.cache.get(cache_key_str)
        if cached_result:
            # Update statistics
            self.key_stats[cache_key_str] = self.key_stats.get(cache_key_str, 0) + 1

            if model not in self.model_stats:
                self.model_stats[model] = {"hits": 0, "misses": 0}
            self.model_stats[model]["hits"] += 1

            # Convert back to LLMCacheEntry
            return LLMCacheEntry(**cached_result)

        # Track miss
        if model not in self.model_stats:
            self.model_stats[model] = {"hits": 0, "misses": 0}
        self.model_stats[model]["misses"] += 1

        return None

    def put(self, model: str, messages: List[Dict[str, str]],
            response: Dict[str, Any], usage: Dict[str, int],
            finish_reason: str, temperature: float = 0.1,
            max_tokens: Optional[int] = None) -> None:
        """Cache LLM response."""
        cache_key = self._create_cache_key(model, messages, temperature, max_tokens)
        cache_key_str = self._cache_key_to_string(cache_key)

        entry = LLMCacheEntry(
            response=response,
            usage=usage,
            model=model,
            finish_reason=finish_reason
        )

        self.cache.put(cache_key_str, entry.__dict__)

        # Update statistics
        if model not in self.model_stats:
            self.model_stats[model] = {"hits": 0, "misses": 0}

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.key_stats.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = sum(stats["hits"] + stats["misses"]
                           for stats in self.model_stats.values())
        total_hits = sum(stats["hits"] for stats in self.model_stats.values())

        return {
            "cache_size": len(self.key_stats),
            "model_stats": self.model_stats,
            "total_requests": total_requests,
            "total_hits": total_hits,
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
            "cache_metrics": self.cache.metrics.__dict__
        }

    def invalidate_model(self, model: str) -> int:
        """Invalidate all cache entries for a specific model."""
        # This is a simplified implementation - in practice, we'd need
        # to scan all entries and remove those matching the model
        # For now, we'll just clear everything
        count = len(self.key_stats)
        self.clear()
        return count


def cached_llm_call(cache_name: str = "llm_responses", ttl: Optional[float] = None,
                   strategy: CacheStrategy = CacheStrategy.LRU):
    """Decorator for caching LLM API calls."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract parameters for cache key generation
            model = kwargs.get('model', args[0] if args else 'unknown')
            messages = kwargs.get('messages', args[1] if len(args) > 1 else [])

            # Get or create LLM cache
            if not hasattr(func, '_llm_cache'):
                func._llm_cache = LLMCache(strategy=strategy)

            llm_cache = func._llm_cache

            # Try cache first
            cached = llm_cache.get(model, messages)
            if cached:
                return cached.get_content(), cached.usage, cached.finish_reason

            # Execute LLM call
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            if isinstance(result, tuple) and len(result) >= 2:
                content, usage = result[0], result[1]
                finish_reason = result[2] if len(result) > 2 else "completed"
            else:
                content = str(result)
                usage = {"prompt_tokens": 0, "completion_tokens": 0}
                finish_reason = "completed"

            # Format response for caching
            response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content
                    }
                }]
            }

            # Cache the result
            llm_cache.put(model, messages, response, usage, finish_reason)

            return content, usage, finish_reason

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous calls, use the same logic but without async
            model = kwargs.get('model', args[0] if args else 'unknown')
            messages = kwargs.get('messages', args[1] if len(args) > 1 else [])

            if not hasattr(func, '_llm_cache'):
                func._llm_cache = LLMCache(strategy=strategy)

            llm_cache = func._llm_cache

            cached = llm_cache.get(model, messages)
            if cached:
                return cached.get_content(), cached.usage, cached.finish_reason

            result = func(*args, **kwargs)

            if isinstance(result, tuple) and len(result) >= 2:
                content, usage = result[0], result[1]
                finish_reason = result[2] if len(result) > 2 else "completed"
            else:
                content = str(result)
                usage = {"prompt_tokens": 0, "completion_tokens": 0}
                finish_reason = "completed"

            response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content
                    }
                }]
            }

            llm_cache.put(model, messages, response, usage, finish_reason)

            return content, usage, finish_reason

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class LLMCacheManager:
    """Manager for multiple LLM caches with different configurations."""

    def __init__(self):
        self.caches: Dict[str, LLMCache] = {}
        self.global_stats: Dict[str, Any] = {}

    def get_cache(self, name: str, **kwargs) -> LLMCache:
        """Get or create LLM cache with specific configuration."""
        if name not in self.caches:
            self.caches[name] = LLMCache(**kwargs)
        return self.caches[name]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            "caches": {
                name: cache.get_stats()
                for name, cache in self.caches.items()
            },
            "total_caches": len(self.caches)
        }

    def clear_all(self) -> None:
        """Clear all LLM caches."""
        for cache in self.caches.values():
            cache.clear()


# Global instance for easy access
llm_cache_manager = LLMCacheManager()