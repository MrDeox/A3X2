"""Centralized constants module for the A3X agent.

This module contains all magic numbers and hardcoded values extracted from
the core files to improve maintainability and support reliable evolution cycles.

Constants are organized into logical categories for easy maintenance and
documentation. All constants include type hints and descriptive docstrings.
"""

from __future__ import annotations

from typing import Final

# ============================================================================
# Performance & Limits
# ============================================================================

# Execution and resource limits
DEFAULT_COMMAND_TIMEOUT: Final[int] = 30
"""Default timeout for command execution in seconds."""

MEMORY_LIMIT_MB: Final[int] = 512
"""Memory limit for command execution in megabytes."""

MAX_LINTER_WORKERS: Final[int] = 4
"""Maximum number of parallel linter processes."""

SUBPROCESS_TIMEOUT: Final[int] = 10
"""Timeout for subprocess operations like linters."""

# Complexity and size limits
MAX_DIFF_LINES: Final[int] = 50
"""Maximum lines allowed in a single diff for impact analysis."""

MAX_FUNCTION_COMPLEXITY: Final[int] = 50
"""Maximum cyclomatic complexity threshold for rollback triggers."""

MAX_DIFF_COMPLEXITY_SCORE: Final[int] = 200
"""Maximum complexity score for diff impact analysis."""

# Resource monitoring
PERFORMANCE_DEGRADATION_THRESHOLD: Final[float] = 0.25
"""Threshold for performance degradation detection (25% drop)."""

# ============================================================================
# Thresholds & Scoring
# ============================================================================

# Success rate thresholds
HIGH_SUCCESS_THRESHOLD: Final[float] = 0.85
"""High success rate threshold for recursion depth adjustment."""

LOW_SUCCESS_THRESHOLD: Final[float] = 0.6
"""Low success rate threshold for recursion depth adjustment."""

HIGH_SUCCESS_RATE: Final[float] = 0.7
"""General high success rate threshold."""

LOW_SUCCESS_RATE: Final[float] = 0.5
"""General low success rate threshold."""

PATCH_HIGH_SUCCESS_RATE: Final[float] = 0.7
"""High success rate threshold for patch operations."""

PATCH_LOW_SUCCESS_RATE: Final[float] = 0.5
"""Low success rate threshold for patch operations."""

TEST_FAILURE_THRESHOLD: Final[float] = 0.3
"""Test failure rate threshold for triggering learning."""

HIGH_STABLE_SUCCESS_THRESHOLD: Final[float] = 0.8
"""Threshold for stabilizing recursion depth at higher levels."""

FAILURE_RATE_THRESHOLD: Final[float] = 0.7
"""Failure rate threshold for triggering rollback."""

TEST_FAILURE_RATE_THRESHOLD: Final[float] = 0.3
"""Test failure rate threshold for rollback triggers."""

# Fitness calculation weights
FITNESS_ACTIONS_WEIGHT: Final[float] = 0.4
"""Weight for actions success rate in fitness calculation."""

FITNESS_PATCH_WEIGHT: Final[float] = 0.3
"""Weight for patch success rate in fitness calculation."""

FITNESS_TESTS_WEIGHT: Final[float] = 0.2
"""Weight for test success rate in fitness calculation."""

FITNESS_RECURSION_WEIGHT: Final[float] = 0.1
"""Weight for recursion depth efficiency in fitness calculation."""

# Bias and adjustment values
BACKLOG_HIGH_DELTA_WEIGHT: Final[float] = 1.5
"""Weight multiplier for high-delta backlog items."""

DEFAULT_BACKLOG_MEMORY_USAGE_WEIGHT: Final[float] = 1.0
"""Default weight for memory usage in backlog prioritization."""

NEW_SKILL_BIAS: Final[float] = 2.0
"""Bias multiplier for newly discovered skills."""

# ============================================================================
# Memory Management
# ============================================================================

# Memory retention and cleanup
MEMORY_TTL_DAYS: Final[int] = 7
"""Time-to-live for memory entries in days."""

# Memory query and processing
MEMORY_TOP_K_DEFAULT: Final[int] = 5
"""Default number of top-k results for memory queries."""

MEMORY_TOP_K_MAX: Final[int] = 10
"""Maximum number of top-k results for memory queries."""

# Memory content limits
MEMORY_SNIPPET_MAX_LENGTH: Final[int] = 400
"""Maximum length for memory content snippets."""

OBSERVATION_EXCERPT_MAX_LENGTH: Final[int] = 2000
"""Maximum length for observation excerpts."""

# Similarity thresholds
DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.7
"""Default similarity threshold for memory matching."""

# ============================================================================
# Planning & Strategy
# ============================================================================

# Recursion and depth limits
MIN_RECURSION_DEPTH: Final[int] = 3
"""Minimum allowed recursion depth."""

MAX_RECURSION_DEPTH: Final[int] = 10
"""Maximum allowed recursion depth."""

DEFAULT_RECURSION_DEPTH: Final[int] = 3
"""Default recursion depth."""

STABLE_RECURSION_DEPTH: Final[int] = 5
"""Stable recursion depth for high success rates."""

DEFAULT_MAX_SUB_DEPTH: Final[int] = 3
"""Default maximum sub-task depth."""

MAX_RECURSION_FOR_FITNESS: Final[int] = 10
"""Maximum recursion depth considered for fitness calculation."""

# Mission and planning limits
DEFAULT_MISSION_MAX_STEPS: Final[int] = 6
"""Default maximum steps for mission execution."""

# LLM interaction limits
LLM_PROPOSAL_ATTEMPTS: Final[int] = 3
"""Number of attempts for LLM proposal generation."""

BIAS_PROPOSAL_CANDIDATES: Final[int] = 3
"""Number of candidates for biased action proposal."""

# ============================================================================
# Evolution & Learning
# ============================================================================

# Learning rate adjustments
RECURSION_DEPTH_ADJUSTMENT_STEP: Final[float] = 0.5
"""Step size for recursion depth adjustments."""

PATCH_BIAS_ADJUSTMENT_STEP: Final[float] = 0.1
"""Step size for patch action bias adjustments."""

MEMORY_WEIGHT_ADJUSTMENT_STEP: Final[float] = 0.1
"""Step size for memory weight adjustments."""

COMMAND_BIAS_ADJUSTMENT_STEP: Final[float] = 0.2
"""Step size for command action bias adjustments."""

# History and tracking
MAX_FITNESS_HISTORY_ENTRIES: Final[int] = 50
"""Maximum number of fitness history entries to retain."""

BASE_ITERATIONS: Final[int] = 10
"""Base number of iterations for agent execution."""

# ============================================================================
# File Paths & Directories
# ============================================================================

# Cache and state directories
DEFAULT_CACHE_DIR: Final[str] = "a3x/state"
"""Default directory for caching embeddings and other state."""

DEFAULT_LOGS_DIR: Final[str] = "a3x/logs"
"""Default directory for log files."""

DEFAULT_MEMORY_FILE: Final[str] = "seed/memory/memory.jsonl"
"""Default path for semantic memory storage."""

DEFAULT_PLANS_DIR: Final[str] = "seed/memory/plans"
"""Default directory for plan storage."""

# Archive and backup settings
DAYS_BEFORE_ARCHIVE: Final[int] = 7
"""Days before archiving old files."""

# ============================================================================
# Text Processing & Formatting
# ============================================================================

# Content truncation
MEMORY_ELLIPSIS_LENGTH: Final[int] = 397
"""Length for memory content truncation before ellipsis."""

OBSERVATION_ELLIPSIS_LENGTH: Final[int] = 1997
"""Length for observation truncation before ellipsis."""

# ============================================================================
# LLM & Embedding Configuration
# ============================================================================

# Embedding model settings
DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
"""Default embedding model name."""

DEFAULT_EMBEDDING_DEVICE: Final[str] = "cpu"
"""Default device for embedding computation."""

DEFAULT_EMBEDDING_DIMENSION: Final[int] = 100
"""Default embedding dimension for fallback vectors."""

# ============================================================================
# Risk Assessment
# ============================================================================

# Risk scoring thresholds
HIGH_RISK_THRESHOLD: Final[float] = 0.8
"""Threshold for high-risk classification."""

MEDIUM_RISK_THRESHOLD: Final[float] = 0.5
"""Threshold for medium-risk classification."""

# Linting violation limits
MAX_LINT_VIOLATIONS: Final[int] = 5
"""Maximum lint violations before high-risk classification."""

# ============================================================================
# Validation Constants
# ============================================================================

# These constants are used for validation and should not be changed
# without careful consideration of their impact on system behavior.

# Exponential backoff base for depth limits
MAX_DEPTH_BACKOFF_BASE: Final[int] = 2
"""Base for exponential backoff calculation in recursion depth limits."""

# Maximum depth before applying backoff
DEPTH_BACKOFF_THRESHOLD: Final[int] = 10
"""Depth threshold before applying exponential backoff."""

# Maximum backoff seconds
MAX_BACKOFF_SECONDS: Final[int] = 60
"""Maximum backoff time in seconds."""

# ============================================================================
# Default Configuration Dictionaries
# ============================================================================

# These are default configurations that can be overridden by user settings

DEFAULT_HINTS_ACTION_BIASES: Final[dict[str, float]] = {}
"""Default action biases for hint system."""

DEFAULT_HINTS_BACKLOG_WEIGHTS: Final[dict[str, float]] = {
    "high_delta": BACKLOG_HIGH_DELTA_WEIGHT,
    "memory_usage": DEFAULT_BACKLOG_MEMORY_USAGE_WEIGHT,
}
"""Default backlog weights for hint system."""

DEFAULT_HINTS_MAX_SUB_DEPTH: Final[int] = DEFAULT_MAX_SUB_DEPTH
"""Default maximum sub-depth for hints system."""

DEFAULT_HINTS_RECURSION_DEPTH: Final[int] = DEFAULT_RECURSION_DEPTH
"""Default recursion depth for hints system."""

_CAPABILITY_DEFAULT_CONFIG: Final[dict[str, str]] = {
    "core.diffing": "patch",
    "core.testing": "tests",
    "horiz.python": "manual",
    "horiz.docs": "manual",
}
"""Default configuration mapping for capabilities."""
