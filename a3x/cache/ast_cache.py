"""AST parsing and analysis result caching system."""

import ast
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .core import BaseCache, CacheStrategy, TTLCache, LRUCache, cache_decorator


@dataclass
class ASTCacheKey:
    """Cache key for AST parsing requests."""
    code_hash: str
    analysis_type: str  # "parse", "complexity", "validation", "full_analysis"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing."""
        return {
            "code_hash": self.code_hash,
            "analysis_type": self.analysis_type
        }


@dataclass
class ASTCacheEntry:
    """AST cache entry with parsed tree and analysis results."""
    tree: ast.AST
    analysis_results: Dict[str, Any]
    syntax_valid: bool
    complexity_score: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

    def get_tree_dict(self) -> Dict[str, Any]:
        """Convert AST tree to dictionary for serialization."""
        return ast.dump(self.tree, indent=2)


class ASTCache:
    """Specialized cache for AST parsing and analysis results."""

    def __init__(self, max_size: int = 500, ttl: float = 1800,
                 strategy: CacheStrategy = CacheStrategy.LRU):
        if strategy == CacheStrategy.TTL:
            self.cache = TTLCache(max_size=max_size, default_ttl=ttl)
        else:
            self.cache = LRUCache(max_size=max_size, strategy=strategy)

        self.parse_stats: Dict[str, int] = {}
        self.analysis_stats: Dict[str, Dict[str, int]] = {}

    def _generate_code_hash(self, code: str) -> str:
        """Generate hash for code content."""
        # Normalize code for consistent hashing
        normalized = code.strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _create_cache_key(self, code: str, analysis_type: str) -> ASTCacheKey:
        """Create cache key from code and analysis type."""
        return ASTCacheKey(
            code_hash=self._generate_code_hash(code),
            analysis_type=analysis_type
        )

    def _cache_key_to_string(self, key: ASTCacheKey) -> str:
        """Convert cache key to string for caching."""
        return hashlib.sha256(json.dumps(key.to_dict(), sort_keys=True).encode()).hexdigest()

    def _parse_code(self, code: str) -> Tuple[ast.AST, bool]:
        """Parse code and return AST tree and validity status."""
        try:
            tree = ast.parse(code)
            return tree, True
        except SyntaxError:
            # Return a dummy tree for invalid syntax to maintain consistency
            return ast.parse("pass"), False

    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate cyclomatic complexity and other metrics."""
        complexity_stats = {
            "cyclomatic_complexity": 1,  # Base complexity
            "function_count": 0,
            "class_count": 0,
            "line_count": 0,
            "node_count": 0
        }

        for node in ast.walk(tree):
            complexity_stats["node_count"] += 1

            if isinstance(node, ast.FunctionDef):
                complexity_stats["function_count"] += 1
                # Add complexity for each branch in the function
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)):
                        complexity_stats["cyclomatic_complexity"] += 1
                    elif isinstance(child, (ast.And, ast.Or)):
                        complexity_stats["cyclomatic_complexity"] += 1

            elif isinstance(node, ast.ClassDef):
                complexity_stats["class_count"] += 1

        # Estimate line count (rough approximation)
        try:
            complexity_stats["line_count"] = len(code.split('\n'))
        except:
            complexity_stats["line_count"] = 0

        return complexity_stats

    def get_parse_result(self, code: str) -> Tuple[ast.AST, bool]:
        """Get cached AST parsing result."""
        cache_key = self._create_cache_key(code, "parse")
        cache_key_str = self._cache_key_to_string(cache_key)

        cached_result = self.cache.get(cache_key_str)
        if cached_result:
            entry = ASTCacheEntry(**cached_result)
            self.parse_stats["parse_hits"] = self.parse_stats.get("parse_hits", 0) + 1
            return entry.tree, entry.syntax_valid

        # Cache miss - parse the code
        tree, syntax_valid = self._parse_code(code)

        # Cache the result
        entry = ASTCacheEntry(
            tree=tree,
            analysis_results={},
            syntax_valid=syntax_valid
        )

        self.cache.put(cache_key_str, entry.__dict__)
        self.parse_stats["parse_misses"] = self.parse_stats.get("parse_misses", 0) + 1

        return tree, syntax_valid

    def get_complexity_analysis(self, code: str) -> Dict[str, Any]:
        """Get cached complexity analysis."""
        cache_key = self._create_cache_key(code, "complexity")
        cache_key_str = self._cache_key_to_string(cache_key)

        cached_result = self.cache.get(cache_key_str)
        if cached_result:
            entry = ASTCacheEntry(**cached_result)
            self.analysis_stats["complexity_hits"] = self.analysis_stats.get("complexity_hits", 0) + 1
            return entry.analysis_results.get("complexity", {})

        # Cache miss - perform analysis
        tree, syntax_valid = self._parse_code(code)

        if not syntax_valid:
            return {"error": "Invalid syntax", "cyclomatic_complexity": 0}

        complexity_results = self._calculate_complexity(tree)

        # Cache the result
        entry = ASTCacheEntry(
            tree=tree,
            analysis_results={"complexity": complexity_results},
            syntax_valid=syntax_valid,
            complexity_score=complexity_results.get("cyclomatic_complexity")
        )

        self.cache.put(cache_key_str, entry.__dict__)
        self.analysis_stats["complexity_misses"] = self.analysis_stats.get("complexity_misses", 0) + 1

        return complexity_results

    def get_full_analysis(self, code: str) -> Dict[str, Any]:
        """Get cached full analysis (parse + complexity + validation)."""
        cache_key = self._create_cache_key(code, "full_analysis")
        cache_key_str = self._cache_key_to_string(cache_key)

        cached_result = self.cache.get(cache_key_str)
        if cached_result:
            entry = ASTCacheEntry(**cached_result)
            self.analysis_stats["full_hits"] = self.analysis_stats.get("full_hits", 0) + 1
            return {
                "syntax_valid": entry.syntax_valid,
                "complexity": entry.analysis_results.get("complexity", {}),
                "tree": entry.get_tree_dict()
            }

        # Cache miss - perform full analysis
        tree, syntax_valid = self._parse_code(code)

        complexity_results = {}
        if syntax_valid:
            complexity_results = self._calculate_complexity(tree)

        # Cache the result
        entry = ASTCacheEntry(
            tree=tree,
            analysis_results={"complexity": complexity_results},
            syntax_valid=syntax_valid,
            complexity_score=complexity_results.get("cyclomatic_complexity")
        )

        self.cache.put(cache_key_str, entry.__dict__)
        self.analysis_stats["full_misses"] = self.analysis_stats.get("full_misses", 0) + 1

        return {
            "syntax_valid": syntax_valid,
            "complexity": complexity_results,
            "tree": entry.get_tree_dict()
        }

    def validate_syntax(self, code: str) -> bool:
        """Get cached syntax validation result."""
        cache_key = self._create_cache_key(code, "validation")
        cache_key_str = self._cache_key_to_string(cache_key)

        cached_result = self.cache.get(cache_key_str)
        if cached_result:
            entry = ASTCacheEntry(**cached_result)
            self.analysis_stats["validation_hits"] = self.analysis_stats.get("validation_hits", 0) + 1
            return entry.syntax_valid

        # Cache miss - validate syntax
        tree, syntax_valid = self._parse_code(code)

        # Cache the result
        entry = ASTCacheEntry(
            tree=tree,
            analysis_results={},
            syntax_valid=syntax_valid
        )

        self.cache.put(cache_key_str, entry.__dict__)
        self.analysis_stats["validation_misses"] = self.analysis_stats.get("validation_misses", 0) + 1

        return syntax_valid

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.parse_stats.clear()
        self.analysis_stats.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_parse_requests = self.parse_stats.get("parse_hits", 0) + self.parse_stats.get("parse_misses", 0)
        total_analysis_requests = (self.analysis_stats.get("complexity_hits", 0) +
                                 self.analysis_stats.get("complexity_misses", 0) +
                                 self.analysis_stats.get("full_hits", 0) +
                                 self.analysis_stats.get("full_misses", 0) +
                                 self.analysis_stats.get("validation_hits", 0) +
                                 self.analysis_stats.get("validation_misses", 0))

        return {
            "parse_stats": self.parse_stats,
            "analysis_stats": self.analysis_stats,
            "total_parse_requests": total_parse_requests,
            "total_analysis_requests": total_analysis_requests,
            "parse_hit_rate": (self.parse_stats.get("parse_hits", 0) / total_parse_requests
                             if total_parse_requests > 0 else 0.0),
            "analysis_hit_rate": (self.analysis_stats.get("complexity_hits", 0) / total_analysis_requests
                                if total_analysis_requests > 0 else 0.0),
            "cache_metrics": self.cache.metrics.__dict__
        }

    def invalidate_code_pattern(self, pattern_hash: str) -> int:
        """Invalidate cache entries for code matching a pattern hash."""
        # Simplified implementation - in practice would scan and remove matching entries
        count = len(self.parse_stats) + len(self.analysis_stats)
        self.clear()
        return count


def cached_ast_parse(cache_name: str = "ast_parse", ttl: Optional[float] = None,
                    strategy: CacheStrategy = CacheStrategy.LRU):
    """Decorator for caching AST parsing operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract code from arguments
            code = kwargs.get('code', args[0] if args else '')

            if not hasattr(func, '_ast_cache'):
                func._ast_cache = ASTCache(strategy=strategy)

            ast_cache = func._ast_cache

            # For parse operations, return tree and validity
            if func.__name__ in ['parse', 'ast_parse']:
                return ast_cache.get_parse_result(code)

            # For complexity analysis
            elif func.__name__ in ['analyze_complexity', 'get_complexity']:
                return ast_cache.get_complexity_analysis(code)

            # For validation
            elif func.__name__ in ['validate', 'validate_syntax']:
                return ast_cache.validate_syntax(code)

            # For full analysis
            elif func.__name__ in ['analyze', 'full_analysis']:
                return ast_cache.get_full_analysis(code)

            # Fallback to original function
            return func(*args, **kwargs)

        return wrapper
    return decorator


class ASTCacheManager:
    """Manager for multiple AST caches with different configurations."""

    def __init__(self):
        self.caches: Dict[str, ASTCache] = {}
        self.global_stats: Dict[str, Any] = {}

    def get_cache(self, name: str, **kwargs) -> ASTCache:
        """Get or create AST cache with specific configuration."""
        if name not in self.caches:
            self.caches[name] = ASTCache(**kwargs)
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
        """Clear all AST caches."""
        for cache in self.caches.values():
            cache.clear()


# Global instance for easy access
ast_cache_manager = ASTCacheManager()