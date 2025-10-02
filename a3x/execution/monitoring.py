"""Performance monitoring for A3X execution system.

This module contains performance monitoring functionality including
execution metrics, timing, and performance tracking that was previously
embedded in the monolithic ActionExecutor class.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Generator, Optional

from ..actions import AgentAction, Observation
from ..cache import CacheManager, llm_cache_manager, ast_cache_manager, memory_cache_manager, config_cache_manager

class PerformanceMonitor:
    """Performance monitor for tracking execution metrics.

    This class handles performance monitoring, metrics collection,
    and execution timing for the execution system.
    """

    def __init__(self, orchestrator) -> None:
        """Initialize the performance monitor.

        Args:
            orchestrator (ExecutionOrchestrator): The main execution orchestrator.
        """
        self.orchestrator = orchestrator

        # Initialize cache manager for performance tracking
        self.cache_manager = CacheManager()

        # Performance metrics storage
        self.metrics = {
            "execution_times": [],
            "action_counts": {},
            "error_counts": {},
            "average_execution_time": 0.0,
            "total_executions": 0,
            "cache_statistics": {},
            "memory_usage": [],
        }

        # Performance thresholds
        self.performance_thresholds = {
            "max_execution_time": 30.0,  # seconds
            "warning_threshold": 10.0,   # seconds
            "low_cache_hit_rate": 0.3,   # 30% hit rate threshold
        }

    def get_status(self) -> Dict[str, Any]:
        """Get status information about the performance monitor.

        Returns:
            Dict[str, Any]: Status information including metrics and thresholds.
        """
        # Update cache statistics
        self._update_cache_statistics()

        return {
            "metrics": self.metrics,
            "thresholds": self.performance_thresholds,
            "performance_alerts": self._check_performance_alerts(),
            "cache_status": self._get_cache_status(),
        }

    @contextmanager
    def monitor_execution(self, action: AgentAction) -> Generator[None, None, None]:
        """Context manager for monitoring action execution.

        Args:
            action (AgentAction): The action being monitored.
        """
        start_time = time.perf_counter()
        action_type = action.type.name if action.type else "unknown"

        try:
            # Record execution start
            self._record_execution_start(action_type)

            yield

        finally:
            # Record execution completion
            execution_time = time.perf_counter() - start_time
            self._record_execution_complete(action_type, execution_time)

    def record_observation_metrics(self, action: AgentAction, observation: Observation) -> None:
        """Record metrics from an action execution observation.

        Args:
            action (AgentAction): The action that was executed.
            observation (Observation): The result observation.
        """
        action_type = action.type.name if action.type else "unknown"

        # Record timing if available
        if observation.duration is not None:
            self._record_execution_complete(action_type, observation.duration)

        # Record success/failure
        if observation.success:
            self._record_success(action_type)
        else:
            self._record_error(action_type, observation.error or "Unknown error")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.

        Returns:
            Dict[str, Any]: Performance metrics summary.
        """
        return {
            "summary": {
                "total_executions": self.metrics["total_executions"],
                "average_execution_time": self.metrics["average_execution_time"],
                "success_rate": self._calculate_success_rate(),
            },
            "by_action_type": self.metrics["action_counts"],
            "recent_performance": self._get_recent_performance(),
            "alerts": self._check_performance_alerts(),
        }

    def _record_execution_start(self, action_type: str) -> None:
        """Record the start of an action execution."""
        self.metrics["action_counts"][action_type] = self.metrics["action_counts"].get(action_type, 0) + 1
        self.metrics["total_executions"] += 1

    def _record_execution_end(self, action_type: str, execution_time: float) -> None:
        """Record the end of an action execution."""
        self._record_execution_complete(action_type, execution_time)

    def _record_execution_complete(self, action_type: str, execution_time: float) -> None:
        """Record the completion of an action execution."""
        # Store execution time
        self.metrics["execution_times"].append(execution_time)

        # Keep only recent executions for memory efficiency
        if len(self.metrics["execution_times"]) > 1000:
            self.metrics["execution_times"] = self.metrics["execution_times"][-500:]

        # Update average (only if we have execution times)
        if self.metrics["execution_times"]:
            self.metrics["average_execution_time"] = sum(self.metrics["execution_times"]) / len(self.metrics["execution_times"])

        # Check for performance issues
        if execution_time > self.performance_thresholds["warning_threshold"]:
            logging.warning(f"Slow execution detected: {action_type} took {execution_time:.2f}s")

        if execution_time > self.performance_thresholds["max_execution_time"]:
            logging.error(f"Very slow execution: {action_type} took {execution_time:.2f}s")

    def _record_success(self, action_type: str) -> None:
        """Record a successful action execution."""
        # Could track success rates per action type
        pass

    def _record_error(self, action_type: str, error: str) -> None:
        """Record an error in action execution."""
        self.metrics["error_counts"][action_type] = self.metrics["error_counts"].get(action_type, 0) + 1

        # Log error for monitoring
        logging.error(f"Execution error in {action_type}: {error}")

    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        total_actions = sum(self.metrics["action_counts"].values())
        total_errors = sum(self.metrics["error_counts"].values())

        if total_actions == 0:
            return 1.0

        return (total_actions - total_errors) / total_actions

    def _get_recent_performance(self) -> Dict[str, float]:
        """Get recent performance metrics."""
        recent_times = self.metrics["execution_times"][-100:]  # Last 100 executions

        if not recent_times:
            return {"recent_average": 0.0, "recent_min": 0.0, "recent_max": 0.0}

        return {
            "recent_average": sum(recent_times) / len(recent_times),
            "recent_min": min(recent_times),
            "recent_max": max(recent_times),
        }

    def _check_performance_alerts(self) -> List[str]:
        """Check for performance alerts that need attention."""
        alerts = []

        # Check average execution time
        if self.metrics["average_execution_time"] > self.performance_thresholds["warning_threshold"]:
            alerts.append(f"Average execution time high: {self.metrics['average_execution_time']:.2f}s")

        # Check error rate
        success_rate = self._calculate_success_rate()
        if success_rate < 0.8:  # Less than 80% success rate
            alerts.append(f"Low success rate: {success_rate:.1%}")

        # Check for slow recent executions
        recent_perf = self._get_recent_performance()
        if recent_perf["recent_max"] > self.performance_thresholds["max_execution_time"]:
            alerts.append(f"Recent slow execution: {recent_perf['recent_max']:.2f}s")

        return alerts

    def _update_cache_statistics(self) -> None:
        """Update cache statistics for performance monitoring."""
        try:
            # Get global cache metrics
            global_cache_metrics = self.cache_manager.get_metrics()

            # Get individual cache manager statistics
            llm_stats = llm_cache_manager.get_stats()
            ast_stats = ast_cache_manager.get_stats()
            memory_stats = memory_cache_manager.get_stats()
            config_stats = config_cache_manager.get_stats()

            # Combine all cache statistics
            self.metrics["cache_statistics"] = {
                "global_cache": global_cache_metrics,
                "llm_cache": llm_stats,
                "ast_cache": ast_stats,
                "memory_cache": memory_stats,
                "config_cache": config_stats,
                "timestamp": time.time()
            }

        except Exception as e:
            logging.warning(f"Failed to update cache statistics: {e}")

    def _get_cache_status(self) -> Dict[str, Any]:
        """Get comprehensive cache status."""
        cache_stats = self.metrics.get("cache_statistics", {})

        if not cache_stats:
            return {"status": "no_cache_data"}

        # Calculate overall cache performance
        total_requests = 0
        total_hits = 0

        for cache_type, stats in cache_stats.items():
            if cache_type != "timestamp" and isinstance(stats, dict):
                if "total_requests" in stats:
                    total_requests += stats["total_requests"]
                    total_hits += stats["total_hits"]

        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

        return {
            "overall_hit_rate": overall_hit_rate,
            "total_requests": total_requests,
            "total_hits": total_hits,
            "cache_count": len([k for k in cache_stats.keys() if k != "timestamp"]),
            "performance_rating": self._rate_cache_performance(overall_hit_rate),
            "details": cache_stats
        }

    def _rate_cache_performance(self, hit_rate: float) -> str:
        """Rate cache performance based on hit rate."""
        if hit_rate >= 0.8:
            return "excellent"
        elif hit_rate >= 0.6:
            return "good"
        elif hit_rate >= 0.4:
            return "fair"
        else:
            return "poor"

    def get_cache_performance_report(self) -> Dict[str, Any]:
        """Get detailed cache performance report."""
        cache_stats = self.metrics.get("cache_statistics", {})

        if not cache_stats:
            return {"error": "No cache statistics available"}

        report = {
            "summary": self._get_cache_status(),
            "cache_details": {},
            "recommendations": []
        }

        # Analyze each cache type
        for cache_type, stats in cache_stats.items():
            if cache_type != "timestamp" and isinstance(stats, dict):
                if "hit_rate" in stats:
                    report["cache_details"][cache_type] = {
                        "hit_rate": stats["hit_rate"],
                        "total_requests": stats.get("total_requests", 0),
                        "hits": stats.get("hits", 0),
                        "misses": stats.get("misses", 0),
                        "performance": self._rate_cache_performance(stats["hit_rate"])
                    }

        # Generate recommendations
        report["recommendations"] = self._generate_cache_recommendations(report)

        return report

    def _generate_cache_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate cache optimization recommendations."""
        recommendations = []
        summary = report.get("summary", {})

        if summary.get("performance_rating") == "poor":
            recommendations.append("Consider increasing cache sizes or adjusting TTL values")

        if summary.get("overall_hit_rate", 0) < 0.5:
            recommendations.append("Low cache hit rate detected - review cache key generation")

        # Check individual cache performance
        for cache_name, details in report.get("cache_details", {}).items():
            if details.get("performance") in ["poor", "fair"]:
                recommendations.append(f"Improve {cache_name} performance (current hit rate: {details.get('hit_rate', 0):.2%})")

        return recommendations

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.metrics = {
            "execution_times": [],
            "action_counts": {},
            "error_counts": {},
            "average_execution_time": 0.0,
            "total_executions": 0,
        }

    def get_action_performance_report(self, action_type: str) -> Dict[str, Any]:
        """Get performance report for a specific action type.

        Args:
            action_type (str): The action type to report on.

        Returns:
            Dict[str, Any]: Performance report for the action type.
        """
        execution_count = self.metrics["action_counts"].get(action_type, 0)
        error_count = self.metrics["error_counts"].get(action_type, 0)

        # Calculate times for this action type (approximation)
        relevant_times = [
            t for t in self.metrics["execution_times"]
            if self._is_execution_for_action_type(t, action_type)
        ]

        if not relevant_times:
            return {
                "action_type": action_type,
                "execution_count": execution_count,
                "error_count": error_count,
                "success_rate": 1.0 if execution_count == 0 else (execution_count - error_count) / execution_count,
                "average_time": 0.0,
            }

        return {
            "action_type": action_type,
            "execution_count": execution_count,
            "error_count": error_count,
            "success_rate": (execution_count - error_count) / execution_count,
            "average_time": sum(relevant_times) / len(relevant_times),
            "min_time": min(relevant_times),
            "max_time": max(relevant_times),
        }

    def _is_execution_for_action_type(self, execution_time: float, action_type: str) -> bool:
        """Determine if an execution time corresponds to a specific action type.

        This is a simplified implementation. In a real system, we'd track
        execution times per action type more precisely.
        """
        # Simple heuristic: assume recent executions are distributed across action types
        # In practice, we'd want to track this more precisely
        return True

    def export_metrics(self) -> Dict[str, Any]:
        """Export all current metrics for external monitoring.

        Returns:
            Dict[str, Any]: All current performance metrics.
        """
        return {
            "timestamp": time.time(),
            "metrics": self.metrics.copy(),
            "thresholds": self.performance_thresholds.copy(),
            "alerts": self._check_performance_alerts(),
        }

    def set_performance_threshold(self, threshold_name: str, value: float) -> None:
        """Set a performance threshold value.

        Args:
            threshold_name (str): Name of the threshold to set.
            value (float): New threshold value.
        """
        if threshold_name in self.performance_thresholds:
            self.performance_thresholds[threshold_name] = value
            logging.info(f"Performance threshold {threshold_name} set to {value}")
        else:
            logging.warning(f"Unknown performance threshold: {threshold_name}")

    def get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trends over time.

        Returns:
            Dict[str, List[float]]: Performance trends for different metrics.
        """
        # Group execution times into windows for trend analysis
        times = self.metrics["execution_times"]
        if not times:
            return {"execution_times": [], "windows": []}

        # Create windows of 50 executions each
        window_size = 50
        windows = []
        window_times = []

        for i in range(0, len(times), window_size):
            window = times[i:i + window_size]
            if window:
                windows.append(i // window_size)
                window_times.append(sum(window) / len(window))

        return {
            "execution_times": times,
            "windows": windows,
            "window_averages": window_times,
        }