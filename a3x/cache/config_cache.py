"""Configuration caching system for expensive config operations."""

import hashlib
import json
import time
from typing import Any, Dict, Optional, TypeVar, Type, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .core import BaseCache, CacheStrategy, TTLCache, LRUCache, cache_decorator

T = TypeVar('T')


@dataclass
class ConfigCacheKey:
    """Cache key for configuration operations."""
    config_path: str
    config_type: str  # "yaml", "json", "toml", etc.
    operation: str   # "load", "parse", "validate", etc.
    parameters_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing."""
        return {
            "config_path": self.config_path,
            "config_type": self.config_type,
            "operation": self.operation,
            "parameters_hash": self.parameters_hash
        }


@dataclass
class ConfigCacheEntry:
    """Configuration cache entry."""
    config_data: Dict[str, Any]
    config_path: str
    config_type: str
    last_modified: float
    file_size: int
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


class ConfigCache:
    """Specialized cache for configuration file operations."""

    def __init__(self, max_size: int = 500, ttl: float = 300,  # 5 minutes for configs
                 strategy: CacheStrategy = CacheStrategy.LRU):
        if strategy == CacheStrategy.TTL:
            self.cache = TTLCache(max_size=max_size, default_ttl=ttl)
        else:
            self.cache = LRUCache(max_size=max_size, strategy=strategy)

        self.stats: Dict[str, int] = {"hits": 0, "misses": 0}
        self.file_stats: Dict[str, Dict[str, Any]] = {}

    def _generate_path_hash(self, config_path: str) -> str:
        """Generate hash for config file path."""
        return hashlib.sha256(config_path.encode()).hexdigest()

    def _generate_parameters_hash(self, **parameters) -> str:
        """Generate hash for operation parameters."""
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()

    def _get_file_metadata(self, config_path: str) -> Tuple[float, int]:
        """Get file modification time and size."""
        try:
            path = Path(config_path)
            if not path.exists():
                return 0.0, 0

            stat = path.stat()
            return stat.st_mtime, stat.st_size
        except (OSError, ValueError):
            return 0.0, 0

    def _create_cache_key(self, config_path: str, config_type: str, operation: str, **parameters) -> ConfigCacheKey:
        """Create cache key for configuration operation."""
        return ConfigCacheKey(
            config_path=config_path,
            config_type=config_type,
            operation=operation,
            parameters_hash=self._generate_parameters_hash(**parameters) if parameters else None
        )

    def _cache_key_to_string(self, key: ConfigCacheKey) -> str:
        """Convert config cache key to string."""
        return hashlib.sha256(json.dumps(key.to_dict(), sort_keys=True).encode()).hexdigest()

    def get_config(self, config_path: str, config_type: str, operation: str = "load", **parameters) -> Optional[Dict[str, Any]]:
        """Get cached configuration data."""
        cache_key = self._create_cache_key(config_path, config_type, operation, **parameters)
        cache_key_str = self._cache_key_to_string(cache_key)

        cached_result = self.cache.get(cache_key_str)
        if cached_result:
            entry = ConfigCacheEntry(**cached_result)

            # Check if file has been modified since caching
            current_mtime, current_size = self._get_file_metadata(config_path)
            if (current_mtime == entry.last_modified and
                current_size == entry.file_size):
                self.stats["hits"] += 1
                return entry.config_data

        self.stats["misses"] += 1
        return None

    def put_config(self, config_path: str, config_data: Dict[str, Any], config_type: str, operation: str = "load", **parameters) -> None:
        """Cache configuration data."""
        cache_key = self._create_cache_key(config_path, config_type, operation, **parameters)
        cache_key_str = self._cache_key_to_string(cache_key)

        # Get file metadata for cache invalidation
        current_mtime, current_size = self._get_file_metadata(config_path)

        entry = ConfigCacheEntry(
            config_data=config_data,
            config_path=config_path,
            config_type=config_type,
            last_modified=current_mtime,
            file_size=current_size
        )

        self.cache.put(cache_key_str, entry.__dict__)

        # Update file statistics
        if config_path not in self.file_stats:
            self.file_stats[config_path] = {"loads": 0, "size": current_size}
        self.file_stats[config_path]["loads"] += 1

    def invalidate_path(self, config_path: str) -> int:
        """Invalidate all cache entries for a specific config path."""
        # Simplified implementation - in practice would scan and remove matching entries
        count = sum(1 for path in self.file_stats.keys() if path == config_path)
        if config_path in self.file_stats:
            del self.file_stats[config_path]
        return count

    def clear(self) -> None:
        """Clear all cached configurations."""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0}
        self.file_stats.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get configuration cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        return {
            "config_cache_size": len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
            "total_requests": total_requests,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": self.stats["hits"] / total_requests if total_requests > 0 else 0.0,
            "cached_files": len(self.file_stats),
            "file_stats": self.file_stats,
            "cache_metrics": self.cache.metrics.__dict__
        }


def cached_config_load(cache_name: str = "config_load", ttl: Optional[float] = None,
                      strategy: CacheStrategy = CacheStrategy.LRU):
    """Decorator for caching configuration file loading."""
    def decorator(func):
        @wraps(func)
        def wrapper(config_path: str, *args, **kwargs):
            # Get cache manager from function context or create default
            cache_manager = getattr(func, '_config_cache', None)
            if cache_manager is None:
                cache_manager = ConfigCache(strategy=strategy)
                func._config_cache = cache_manager

            config_type = kwargs.get('config_type', 'yaml')
            operation = kwargs.get('operation', 'load')

            # Try cache first
            cached = cache_manager.get_config(config_path, config_type, operation)
            if cached is not None:
                return cached

            # Execute config loading
            result = func(config_path, *args, **kwargs)

            # Cache the result
            cache_manager.put_config(config_path, result, config_type, operation)

            return result

        return wrapper
    return decorator


class ConfigCacheManager:
    """Manager for multiple configuration caches."""

    def __init__(self):
        self.caches: Dict[str, ConfigCache] = {}
        self.global_stats: Dict[str, Any] = {}

    def get_cache(self, name: str, **kwargs) -> ConfigCache:
        """Get or create configuration cache with specific configuration."""
        if name not in self.caches:
            self.caches[name] = ConfigCache(**kwargs)
        return self.caches[name]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all configuration caches."""
        return {
            "caches": {
                name: cache.get_stats()
                for name, cache in self.caches.items()
            },
            "total_caches": len(self.caches)
        }

    def clear_all(self) -> None:
        """Clear all configuration caches."""
        for cache in self.caches.values():
            cache.clear()


# Global instance for easy access
config_cache_manager = ConfigCacheManager()