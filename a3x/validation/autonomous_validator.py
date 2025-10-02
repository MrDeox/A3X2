"""
Comprehensive autonomous system validation environment for A3X SeedAI.

This module provides empirical validation of autonomous behavior and evolution,
including live monitoring, behavior tracking, evolution detection, performance
benchmarking, and health monitoring capabilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks

from ..agent import AgentOrchestrator, AgentResult
from ..autonomous_goal_generator import AutonomousGoal
from ..config import AgentConfig


@dataclass
class ValidationMetrics:
    """Comprehensive metrics for autonomous system validation."""

    # Timing metrics
    goal_generation_interval: float = 0.0
    average_execution_time: float = 0.0
    system_response_time: float = 0.0

    # Performance metrics
    goal_success_rate: float = 0.0
    iteration_efficiency: float = 0.0
    memory_utilization: float = 0.0
    cache_hit_rate: float = 0.0

    # Evolution metrics
    behavior_diversity: float = 0.0
    adaptation_rate: float = 0.0
    complexity_progression: float = 0.0

    # Stability metrics
    failure_rate: float = 0.0
    recovery_rate: float = 0.0
    system_stability: float = 0.0

    # Learning metrics
    knowledge_accumulation: float = 0.0
    skill_acquisition_rate: float = 0.0
    pattern_recognition: float = 0.0


@dataclass
class AutonomousBehaviorSnapshot:
    """Snapshot of autonomous system behavior at a specific point in time."""

    timestamp: datetime
    session_id: str

    # Goal generation patterns
    active_goals: List[AutonomousGoal] = field(default_factory=list)
    goal_generation_rate: float = 0.0
    goal_complexity_distribution: Dict[str, int] = field(default_factory=dict)

    # System state
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_threads: int = 0

    # Performance indicators
    response_times: List[float] = field(default_factory=list)
    success_rates: List[float] = field(default_factory=list)
    error_patterns: List[str] = field(default_factory=list)

    # Evolution indicators
    behavioral_changes: List[str] = field(default_factory=list)
    adaptation_events: List[str] = field(default_factory=list)
    learning_progression: List[str] = field(default_factory=list)


@dataclass
class EvolutionMarker:
    """Marker indicating potential evolutionary development."""

    timestamp: datetime
    marker_type: str  # 'behavioral_shift', 'capability_emergence', 'pattern_novelty'
    confidence: float
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    impact_score: float = 0.0


class AutonomousValidator:
    """
    Comprehensive validation system for autonomous A3X behavior.

    Provides real-time monitoring, evolution detection, performance benchmarking,
    and health monitoring for autonomous systems.
    """

    def __init__(self, config: AgentConfig, agent: AgentOrchestrator):
        self.config = config
        self.agent = agent
        self.session_id = str(uuid.uuid4())

        # Setup logging and storage
        self._setup_validation_environment()

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        # Data collection
        self.behavior_snapshots: deque = deque(maxlen=1000)
        self.evolution_markers: List[EvolutionMarker] = []
        self.performance_history: deque = deque(maxlen=500)

        # Analysis state
        self.baseline_metrics: Optional[ValidationMetrics] = None
        self.evolution_thresholds = self._initialize_evolution_thresholds()

        # Real-time tracking
        self.current_session_start = datetime.now()
        self.last_goal_generation = None
        self.behavioral_patterns: Dict[str, List[Any]] = defaultdict(list)

        # Alerting system
        self.alert_callbacks: List[callable] = []
        self.health_check_interval = 30  # seconds

    def _setup_validation_environment(self) -> None:
        """Setup validation environment directories and logging."""
        # Create validation directories
        self.validation_root = self.config.workspace_root / "a3x" / "validation"
        self.validation_root.mkdir(parents=True, exist_ok=True)

        self.logs_dir = self.validation_root / "logs"
        self.data_dir = self.validation_root / "data"
        self.reports_dir = self.validation_root / "reports"
        self.visualizations_dir = self.validation_root / "visualizations"

        for directory in [self.logs_dir, self.data_dir, self.reports_dir, self.visualizations_dir]:
            directory.mkdir(exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(f"autonomous_validator_{self.session_id}")
        self.logger.setLevel(logging.INFO)

        log_file = self.logs_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _initialize_evolution_thresholds(self) -> Dict[str, float]:
        """Initialize thresholds for evolution detection."""
        return {
            'behavioral_shift_threshold': 0.3,
            'capability_emergence_threshold': 0.6,
            'pattern_novelty_threshold': 0.4,
            'adaptation_rate_threshold': 0.25,
            'stability_threshold': 0.7,
            'learning_rate_threshold': 0.2,
            'diversity_threshold': 0.35,
        }

    def start_monitoring(self, monitoring_interval: float = 5.0) -> None:
        """
        Start real-time monitoring of autonomous behavior.

        Args:
            monitoring_interval: Interval between monitoring snapshots in seconds
        """
        if self.is_monitoring:
            self.logger.warning("Monitoring already active")
            return

        self.is_monitoring = True
        self._stop_monitoring.clear()
        self.current_session_start = datetime.now()

        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(monitoring_interval,),
            daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info(f"Started autonomous monitoring (interval: {monitoring_interval}s)")

    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop monitoring and generate comprehensive validation report.

        Returns:
            Validation report with all collected metrics and analysis
        """
        if not self.is_monitoring:
            self.logger.warning("Monitoring not active")
            return {}

        self.is_monitoring = False
        self._stop_monitoring.set()

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)

        # Generate final report
        report = self._generate_validation_report()
        self._save_validation_report(report)

        self.logger.info("Stopped autonomous monitoring and generated validation report")
        return report

    def _monitoring_loop(self, interval: float) -> None:
        """Main monitoring loop running in background thread."""
        while not self._stop_monitoring.is_set():
            try:
                # Capture behavior snapshot
                snapshot = self._capture_behavior_snapshot()

                # Analyze for evolution markers
                evolution_markers = self._detect_evolution_markers(snapshot)

                # Update performance tracking
                self._update_performance_tracking(snapshot)

                # Check system health
                self._check_system_health(snapshot)

                # Store snapshot
                self.behavior_snapshots.append(snapshot)
                self.evolution_markers.extend(evolution_markers)

                # Small delay to prevent overwhelming the system
                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)

    def _capture_behavior_snapshot(self) -> AutonomousBehaviorSnapshot:
        """Capture current state of autonomous system."""
        snapshot = AutonomousBehaviorSnapshot(
            timestamp=datetime.now(),
            session_id=self.session_id
        )

        # Get current autonomous goals
        try:
            status = self.agent.get_autonomous_mode_status()
            # This would need to be implemented in the agent
            snapshot.active_goals = getattr(self.agent, '_active_autonomous_goals', [])
            snapshot.goal_generation_rate = self._calculate_goal_generation_rate()
        except Exception as e:
            self.logger.warning(f"Could not capture goal information: {e}")

        # Get system resource usage
        snapshot.memory_usage = self._get_memory_usage()
        snapshot.cpu_usage = self._get_cpu_usage()
        snapshot.active_threads = threading.active_count()

        # Get performance indicators
        snapshot.response_times = self._get_recent_response_times()
        snapshot.success_rates = self._get_recent_success_rates()
        snapshot.error_patterns = self._get_recent_error_patterns()

        # Track behavioral patterns
        self._update_behavioral_patterns(snapshot)

        return snapshot

    def _calculate_goal_generation_rate(self) -> float:
        """Calculate rate of autonomous goal generation."""
        if not hasattr(self.agent, '_autonomous_goal_generator'):
            return 0.0

        # Calculate based on generation history
        # This would integrate with the agent's goal generation tracking
        current_time = time.time()
        if self.last_goal_generation:
            time_diff = current_time - self.last_goal_generation
            if time_diff > 0:
                return 1.0 / time_diff  # goals per second

        return 0.0

    def _get_memory_usage(self) -> float:
        """Get current memory usage as percentage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            return memory_percent
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage as percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _get_recent_response_times(self) -> List[float]:
        """Get recent response times from performance history."""
        if len(self.performance_history) < 2:
            return []

        recent_entries = list(self.performance_history)[-10:]
        return [entry.get('response_time', 0.0) for entry in recent_entries]

    def _get_recent_success_rates(self) -> List[float]:
        """Get recent success rates from performance history."""
        if len(self.performance_history) < 2:
            return []

        recent_entries = list(self.performance_history)[-10:]
        return [entry.get('success_rate', 0.0) for entry in recent_entries]

    def _get_recent_error_patterns(self) -> List[str]:
        """Get recent error patterns from performance history."""
        if len(self.performance_history) < 2:
            return []

        recent_entries = list(self.performance_history)[-10:]
        error_patterns = []
        for entry in recent_entries:
            errors = entry.get('errors', [])
            error_patterns.extend([str(error) for error in errors])

        return error_patterns[-20:]  # Last 20 errors

    def _update_behavioral_patterns(self, snapshot: AutonomousBehaviorSnapshot) -> None:
        """Update tracking of behavioral patterns over time."""
        # Track goal complexity patterns
        if snapshot.active_goals:
            complexities = [goal.estimated_impact for goal in snapshot.active_goals]
            self.behavioral_patterns['goal_complexity'].extend(complexities)

        # Track response time patterns
        if snapshot.response_times:
            self.behavioral_patterns['response_times'].extend(snapshot.response_times)

        # Track success rate patterns
        if snapshot.success_rates:
            self.behavioral_patterns['success_rates'].extend(snapshot.success_rates)

    def _detect_evolution_markers(self, snapshot: AutonomousBehaviorSnapshot) -> List[EvolutionMarker]:
        """Detect potential evolutionary developments in behavior."""
        markers = []

        # Detect behavioral shifts
        behavioral_shift = self._detect_behavioral_shift(snapshot)
        if behavioral_shift:
            markers.append(behavioral_shift)

        # Detect capability emergence
        capability_emergence = self._detect_capability_emergence(snapshot)
        if capability_emergence:
            markers.append(capability_emergence)

        # Detect pattern novelty
        pattern_novelty = self._detect_pattern_novelty(snapshot)
        if pattern_novelty:
            markers.append(pattern_novelty)

        # Detect adaptation events
        adaptation = self._detect_adaptation(snapshot)
        if adaptation:
            markers.append(adaptation)

        return markers

    def _detect_behavioral_shift(self, snapshot: AutonomousBehaviorSnapshot) -> Optional[EvolutionMarker]:
        """Detect significant shifts in behavioral patterns."""
        if len(self.behavioral_patterns['success_rates']) < 20:
            return None

        recent_rates = self.behavioral_patterns['success_rates'][-10:]
        older_rates = self.behavioral_patterns['success_rates'][-20:-10]

        if len(recent_rates) < 10 or len(older_rates) < 10:
            return None

        recent_mean = statistics.mean(recent_rates)
        older_mean = statistics.mean(older_rates)

        if older_mean == 0:
            return None

        change_ratio = abs(recent_mean - older_mean) / older_mean

        if change_ratio > self.evolution_thresholds['behavioral_shift_threshold']:
            confidence = min(change_ratio / self.evolution_thresholds['behavioral_shift_threshold'], 1.0)

            return EvolutionMarker(
                timestamp=snapshot.timestamp,
                marker_type='behavioral_shift',
                confidence=confidence,
                description=f"Significant behavioral shift detected: success rate changed by {change_ratio:.2%}",
                evidence={
                    'recent_success_rate': recent_mean,
                    'previous_success_rate': older_mean,
                    'change_ratio': change_ratio
                },
                impact_score=change_ratio
            )

        return None

    def _detect_capability_emergence(self, snapshot: AutonomousBehaviorSnapshot) -> Optional[EvolutionMarker]:
        """Detect emergence of new capabilities."""
        # This would analyze the agent's capabilities over time
        # For now, we'll use a simplified approach based on goal complexity

        if not snapshot.active_goals:
            return None

        complexities = [goal.estimated_impact for goal in snapshot.active_goals]
        avg_complexity = statistics.mean(complexities)

        # Check if complexity has increased significantly
        if len(self.behavioral_patterns['goal_complexity']) >= 20:
            recent_complexities = self.behavioral_patterns['goal_complexity'][-10:]
            older_complexities = self.behavioral_patterns['goal_complexity'][-20:-10]

            recent_avg = statistics.mean(recent_complexities)
            older_avg = statistics.mean(older_complexities)

            if older_avg > 0:
                complexity_increase = (recent_avg - older_avg) / older_avg

                if complexity_increase > self.evolution_thresholds['capability_emergence_threshold']:
                    confidence = min(complexity_increase / self.evolution_thresholds['capability_emergence_threshold'], 1.0)

                    return EvolutionMarker(
                        timestamp=snapshot.timestamp,
                        marker_type='capability_emergence',
                        confidence=confidence,
                        description=f"Potential capability emergence: goal complexity increased by {complexity_increase:.2%}",
                        evidence={
                            'current_complexity': avg_complexity,
                            'previous_complexity': older_avg,
                            'complexity_increase': complexity_increase
                        },
                        impact_score=complexity_increase
                    )

        return None

    def _detect_pattern_novelty(self, snapshot: AutonomousBehaviorSnapshot) -> Optional[EvolutionMarker]:
        """Detect novel patterns in behavior."""
        # Analyze response time patterns for novelty
        if len(self.behavioral_patterns['response_times']) < 20:
            return None

        recent_times = self.behavioral_patterns['response_times'][-15:]
        older_times = self.behavioral_patterns['response_times'][-30:-15]

        if len(recent_times) < 15 or len(older_times) < 15:
            return None

        # Calculate statistical measures
        recent_std = statistics.stdev(recent_times) if len(recent_times) > 1 else 0
        older_std = statistics.stdev(older_times) if len(older_times) > 1 else 0

        # Detect if variability pattern has changed significantly
        if older_std > 0:
            variability_change = abs(recent_std - older_std) / older_std

            if variability_change > self.evolution_thresholds['pattern_novelty_threshold']:
                confidence = min(variability_change / self.evolution_thresholds['pattern_novelty_threshold'], 1.0)

                return EvolutionMarker(
                    timestamp=snapshot.timestamp,
                    marker_type='pattern_novelty',
                    confidence=confidence,
                    description=f"Novel pattern detected: response time variability changed by {variability_change:.2%}",
                    evidence={
                        'recent_variability': recent_std,
                        'previous_variability': older_std,
                        'variability_change': variability_change
                    },
                    impact_score=variability_change
                )

        return None

    def _detect_adaptation(self, snapshot: AutonomousBehaviorSnapshot) -> Optional[EvolutionMarker]:
        """Detect adaptation events based on environmental response."""
        # This would analyze how the system adapts to different conditions
        # For now, we'll use error recovery patterns

        if len(self.behavioral_patterns['success_rates']) < 15:
            return None

        # Look for recovery patterns after failures
        recent_rates = self.behavioral_patterns['success_rates'][-10:]

        # Check for V-shaped recovery pattern (decline followed by improvement)
        if len(recent_rates) >= 5:
            first_half = recent_rates[:5]
            second_half = recent_rates[-5:]

            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            if first_avg < 0.5 and second_avg > 0.7:  # Decline followed by recovery
                improvement = (second_avg - first_avg) / (1 - first_avg) if first_avg < 1 else 0

                if improvement > self.evolution_thresholds['adaptation_rate_threshold']:
                    confidence = min(improvement / self.evolution_thresholds['adaptation_rate_threshold'], 1.0)

                    return EvolutionMarker(
                        timestamp=snapshot.timestamp,
                        marker_type='adaptation_event',
                        confidence=confidence,
                        description=f"Adaptation detected: system recovered with {improvement:.2%} improvement",
                        evidence={
                            'pre_adaptation_performance': first_avg,
                            'post_adaptation_performance': second_avg,
                            'improvement_ratio': improvement
                        },
                        impact_score=improvement
                    )

        return None

    def _update_performance_tracking(self, snapshot: AutonomousBehaviorSnapshot) -> None:
        """Update performance tracking with current snapshot."""
        performance_entry = {
            'timestamp': snapshot.timestamp.isoformat(),
            'memory_usage': snapshot.memory_usage,
            'cpu_usage': snapshot.cpu_usage,
            'response_time': statistics.mean(snapshot.response_times) if snapshot.response_times else 0,
            'success_rate': statistics.mean(snapshot.success_rates) if snapshot.success_rates else 0,
            'active_goals': len(snapshot.active_goals),
            'goal_generation_rate': snapshot.goal_generation_rate,
        }

        self.performance_history.append(performance_entry)

    def _check_system_health(self, snapshot: AutonomousBehaviorSnapshot) -> None:
        """Check system health and trigger alerts if necessary."""
        health_issues = []

        # Check memory usage
        if snapshot.memory_usage > 85:
            health_issues.append(f"High memory usage: {snapshot.memory_usage:.1f}%")

        # Check CPU usage
        if snapshot.cpu_usage > 80:
            health_issues.append(f"High CPU usage: {snapshot.cpu_usage:.1f}%")

        # Check for error patterns
        if snapshot.error_patterns:
            error_count = len(snapshot.error_patterns)
            if error_count > 10:
                health_issues.append(f"High error rate: {error_count} errors in recent history")

        # Check success rate
        if snapshot.success_rates:
            avg_success = statistics.mean(snapshot.success_rates)
            if avg_success < 0.3:
                health_issues.append(f"Low success rate: {avg_success:.2%}")

        # Trigger alerts if health issues detected
        if health_issues:
            self._trigger_health_alerts(health_issues, snapshot)

    def _trigger_health_alerts(self, issues: List[str], snapshot: AutonomousBehaviorSnapshot) -> None:
        """Trigger health alerts to registered callbacks."""
        alert_message = {
            'timestamp': snapshot.timestamp.isoformat(),
            'session_id': self.session_id,
            'issues': issues,
            'system_state': {
                'memory_usage': snapshot.memory_usage,
                'cpu_usage': snapshot.cpu_usage,
                'active_goals': len(snapshot.active_goals),
                'success_rate': statistics.mean(snapshot.success_rates) if snapshot.success_rates else 0,
            }
        }

        # Log the alert
        self.logger.warning(f"Health alert triggered: {issues}")

        # Notify all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_message)
            except Exception as e:
                self.logger.error(f"Error in health alert callback: {e}")

    def add_alert_callback(self, callback: callable) -> None:
        """Add callback function for health alerts."""
        self.alert_callbacks.append(callback)

    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        if not self.behavior_snapshots:
            return {'error': 'No monitoring data available'}

        # Calculate comprehensive metrics
        metrics = self._calculate_validation_metrics()

        # Generate evolution analysis
        evolution_analysis = self._analyze_evolution()

        # Generate performance analysis
        performance_analysis = self._analyze_performance()

        # Generate stability analysis
        stability_analysis = self._analyze_stability()

        # Generate visualizations
        visualizations = self._generate_visualizations()

        report = {
            'session_id': self.session_id,
            'monitoring_period': {
                'start_time': self.current_session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - self.current_session_start).total_seconds()
            },
            'metrics': metrics.__dict__,
            'evolution_analysis': evolution_analysis,
            'performance_analysis': performance_analysis,
            'stability_analysis': stability_analysis,
            'visualizations': visualizations,
            'evolution_markers': [
                {
                    'timestamp': marker.timestamp.isoformat(),
                    'type': marker.marker_type,
                    'confidence': marker.confidence,
                    'description': marker.description,
                    'impact_score': marker.impact_score
                }
                for marker in self.evolution_markers
            ],
            'summary': self._generate_executive_summary(metrics, evolution_analysis)
        }

        return report

    def _calculate_validation_metrics(self) -> ValidationMetrics:
        """Calculate comprehensive validation metrics."""
        metrics = ValidationMetrics()

        if not self.behavior_snapshots:
            return metrics

        snapshots = list(self.behavior_snapshots)
        performance_data = list(self.performance_history)

        # Calculate timing metrics
        if len(snapshots) >= 2:
            time_diffs = [
                (snapshots[i].timestamp - snapshots[i-1].timestamp).total_seconds()
                for i in range(1, len(snapshots))
            ]
            metrics.goal_generation_interval = statistics.mean(time_diffs) if time_diffs else 0.0

        # Calculate performance metrics
        if performance_data:
            execution_times = [entry.get('response_time', 0) for entry in performance_data]
            metrics.average_execution_time = statistics.mean(execution_times) if execution_times else 0.0

            success_rates = [entry.get('success_rate', 0) for entry in performance_data]
            metrics.goal_success_rate = statistics.mean(success_rates) if success_rates else 0.0

            memory_usage = [entry.get('memory_usage', 0) for entry in performance_data]
            metrics.memory_utilization = statistics.mean(memory_usage) if memory_usage else 0.0

        # Calculate evolution metrics
        metrics.behavior_diversity = self._calculate_behavior_diversity()
        metrics.adaptation_rate = self._calculate_adaptation_rate()
        metrics.complexity_progression = self._calculate_complexity_progression()

        # Calculate stability metrics
        metrics.failure_rate = 1.0 - metrics.goal_success_rate
        metrics.recovery_rate = self._calculate_recovery_rate()
        metrics.system_stability = self._calculate_system_stability()

        # Calculate learning metrics
        metrics.knowledge_accumulation = self._calculate_knowledge_accumulation()
        metrics.skill_acquisition_rate = self._calculate_skill_acquisition_rate()
        metrics.pattern_recognition = self._calculate_pattern_recognition()

        return metrics

    def _calculate_behavior_diversity(self) -> float:
        """Calculate behavioral diversity score."""
        if len(self.behavioral_patterns['success_rates']) < 10:
            return 0.0

        recent_patterns = self.behavioral_patterns['success_rates'][-20:]

        if len(recent_patterns) > 1:
            # Calculate coefficient of variation as diversity measure
            mean_rate = statistics.mean(recent_patterns)
            if mean_rate > 0:
                std_rate = statistics.stdev(recent_patterns)
                return std_rate / mean_rate

        return 0.0

    def _calculate_adaptation_rate(self) -> float:
        """Calculate rate of adaptation."""
        if len(self.evolution_markers) < 2:
            return 0.0

        adaptation_markers = [
            marker for marker in self.evolution_markers
            if marker.marker_type in ['adaptation_event', 'behavioral_shift']
        ]

        if not adaptation_markers:
            return 0.0

        # Calculate adaptation frequency
        session_duration = (datetime.now() - self.current_session_start).total_seconds()
        if session_duration > 0:
            return len(adaptation_markers) / (session_duration / 3600)  # adaptations per hour

        return 0.0

    def _calculate_complexity_progression(self) -> float:
        """Calculate progression in goal complexity."""
        if len(self.behavioral_patterns['goal_complexity']) < 20:
            return 0.0

        recent_complexities = self.behavioral_patterns['goal_complexity'][-10:]
        older_complexities = self.behavioral_patterns['goal_complexity'][-20:-10]

        recent_avg = statistics.mean(recent_complexities) if recent_complexities else 0
        older_avg = statistics.mean(older_complexities) if older_complexities else 0

        if older_avg > 0:
            return (recent_avg - older_avg) / older_avg

        return 0.0

    def _calculate_recovery_rate(self) -> float:
        """Calculate system recovery rate after failures."""
        if len(self.performance_history) < 10:
            return 0.0

        # Look for recovery patterns in performance data
        performance_data = list(self.performance_history)

        recovery_events = 0
        total_failures = 0

        for i in range(1, len(performance_data)):
            current = performance_data[i].get('success_rate', 0)
            previous = performance_data[i-1].get('success_rate', 0)

            if previous < 0.5 and current > 0.7:  # Recovery from failure
                recovery_events += 1

            if previous < 0.5:
                total_failures += 1

        if total_failures > 0:
            return recovery_events / total_failures

        return 0.0

    def _calculate_system_stability(self) -> float:
        """Calculate overall system stability."""
        if not self.performance_history:
            return 0.0

        performance_data = list(self.performance_history)

        # Use coefficient of variation of success rates as stability measure
        success_rates = [entry.get('success_rate', 0) for entry in performance_data]

        if len(success_rates) > 1:
            mean_rate = statistics.mean(success_rates)
            if mean_rate > 0:
                std_rate = statistics.stdev(success_rates)
                stability = 1.0 - (std_rate / mean_rate)  # Lower variation = higher stability
                return max(0.0, min(1.0, stability))

        return 0.0

    def _calculate_knowledge_accumulation(self) -> float:
        """Calculate rate of knowledge accumulation."""
        # This would integrate with the agent's memory system
        # For now, use a simplified proxy based on unique patterns observed

        total_patterns = sum(len(patterns) for patterns in self.behavioral_patterns.values())
        session_duration = (datetime.now() - self.current_session_start).total_seconds()

        if session_duration > 0:
            return total_patterns / (session_duration / 60)  # patterns per minute

        return 0.0

    def _calculate_skill_acquisition_rate(self) -> float:
        """Calculate rate of skill acquisition."""
        # This would integrate with the agent's skills registry
        # For now, use complexity progression as a proxy

        return self._calculate_complexity_progression()

    def _calculate_pattern_recognition(self) -> float:
        """Calculate pattern recognition capability."""
        # Measure how well the system recognizes and adapts to patterns
        if len(self.evolution_markers) == 0:
            return 0.0

        # Use adaptation rate as proxy for pattern recognition
        return min(1.0, self._calculate_adaptation_rate())

    def _analyze_evolution(self) -> Dict[str, Any]:
        """Analyze evolutionary developments."""
        if not self.evolution_markers:
            return {'evolution_detected': False, 'confidence': 0.0}

        # Group markers by type
        marker_types = defaultdict(list)
        for marker in self.evolution_markers:
            marker_types[marker.marker_type].append(marker)

        # Calculate overall evolution confidence
        total_confidence = sum(marker.confidence for marker in self.evolution_markers)
        avg_confidence = total_confidence / len(self.evolution_markers)

        # Calculate evolution rate
        session_duration = (datetime.now() - self.current_session_start).total_seconds()
        evolution_rate = len(self.evolution_markers) / (session_duration / 3600) if session_duration > 0 else 0

        # Determine if significant evolution has occurred
        evolution_threshold = 0.5
        evolution_detected = avg_confidence > evolution_threshold and len(self.evolution_markers) >= 3

        return {
            'evolution_detected': evolution_detected,
            'confidence': avg_confidence,
            'evolution_rate_per_hour': evolution_rate,
            'total_markers': len(self.evolution_markers),
            'marker_distribution': dict(marker_types),
            'significant_markers': [
                {
                    'type': marker.marker_type,
                    'confidence': marker.confidence,
                    'description': marker.description,
                    'timestamp': marker.timestamp.isoformat()
                }
                for marker in self.evolution_markers
                if marker.confidence > 0.7
            ]
        }

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance trends."""
        if not self.performance_history:
            return {'error': 'No performance data available'}

        performance_data = list(self.performance_history)

        # Extract metrics over time
        timestamps = [entry['timestamp'] for entry in performance_data]
        success_rates = [entry.get('success_rate', 0) for entry in performance_data]
        response_times = [entry.get('response_time', 0) for entry in performance_data]
        memory_usage = [entry.get('memory_usage', 0) for entry in performance_data]

        # Calculate trends
        success_trend = self._calculate_trend(success_rates)
        performance_trend = self._calculate_trend(response_times, invert=True)  # Lower is better
        memory_trend = self._calculate_trend(memory_usage)

        # Calculate efficiency metrics
        avg_success_rate = statistics.mean(success_rates) if success_rates else 0
        avg_response_time = statistics.mean(response_times) if response_times else 0
        avg_memory_usage = statistics.mean(memory_usage) if memory_usage else 0

        return {
            'success_rate_trend': success_trend,
            'performance_trend': performance_trend,
            'memory_trend': memory_trend,
            'average_success_rate': avg_success_rate,
            'average_response_time': avg_response_time,
            'average_memory_usage': avg_memory_usage,
            'data_points': len(performance_data),
            'time_range': {
                'start': timestamps[0] if timestamps else None,
                'end': timestamps[-1] if timestamps else None
            }
        }

    def _calculate_trend(self, values: List[float], invert: bool = False) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 3:
            return 'insufficient_data'

        # Simple linear regression slope
        x = list(range(len(values)))
        slope, _, r_value, _, _ = stats.linregress(x, values)

        if invert:
            slope = -slope

        if slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'stable'

    def _analyze_stability(self) -> Dict[str, Any]:
        """Analyze system stability."""
        if not self.performance_history:
            return {'error': 'No stability data available'}

        performance_data = list(self.performance_history)

        # Analyze stability metrics
        success_rates = [entry.get('success_rate', 0) for entry in performance_data]

        if len(success_rates) > 1:
            stability_score = 1.0 - (statistics.stdev(success_rates) / statistics.mean(success_rates))
            stability_score = max(0.0, min(1.0, stability_score))
        else:
            stability_score = 0.0

        # Detect stability patterns
        stability_events = 0
        for i in range(1, len(success_rates)):
            if abs(success_rates[i] - success_rates[i-1]) > 0.3:  # Significant change
                stability_events += 1

        stability_rate = stability_events / len(success_rates) if success_rates else 0

        return {
            'stability_score': stability_score,
            'stability_events': stability_events,
            'stability_rate': stability_rate,
            'assessment': 'stable' if stability_score > 0.7 else 'unstable' if stability_score < 0.3 else 'moderate'
        }

    def _generate_visualizations(self) -> Dict[str, str]:
        """Generate visualization files and return their paths."""
        visualizations = {}

        try:
            # Success rate over time
            success_plot = self._plot_success_rate_trend()
            visualizations['success_rate_trend'] = str(success_plot)

            # Evolution markers timeline
            evolution_plot = self._plot_evolution_timeline()
            visualizations['evolution_timeline'] = str(evolution_plot)

            # Performance heatmap
            performance_heatmap = self._plot_performance_heatmap()
            visualizations['performance_heatmap'] = str(performance_heatmap)

            # Behavior patterns
            patterns_plot = self._plot_behavior_patterns()
            visualizations['behavior_patterns'] = str(patterns_plot)

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            visualizations['error'] = str(e)

        return visualizations

    def _plot_success_rate_trend(self) -> Path:
        """Plot success rate trend over time."""
        if not self.performance_history:
            return Path()

        performance_data = list(self.performance_history)

        timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in performance_data]
        success_rates = [entry.get('success_rate', 0) for entry in performance_data]

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, success_rates, marker='o', linestyle='-', linewidth=2, markersize=4)
        plt.title('Success Rate Trend Over Time')
        plt.xlabel('Time')
        plt.ylabel('Success Rate')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        # Add trend line
        if len(success_rates) > 2:
            z = np.polyfit(range(len(success_rates)), success_rates, 1)
            p = np.poly1d(z)
            plt.plot(timestamps, p(range(len(success_rates))), "r--", alpha=0.8, label='Trend')

        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = self.visualizations_dir / f'success_rate_trend_{self.session_id}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path

    def _plot_evolution_timeline(self) -> Path:
        """Plot evolution markers on a timeline."""
        if not self.evolution_markers:
            return Path()

        # Group markers by type
        marker_types = defaultdict(list)
        for marker in self.evolution_markers:
            marker_types[marker.marker_type].append(marker)

        plt.figure(figsize=(14, 8))

        colors = {'behavioral_shift': 'blue', 'capability_emergence': 'red',
                 'pattern_novelty': 'green', 'adaptation_event': 'orange'}

        for marker_type, markers in marker_types.items():
            timestamps = [marker.timestamp for marker in markers]
            confidences = [marker.confidence for marker in markers]

            plt.scatter(timestamps, confidences, c=colors.get(marker_type, 'gray'),
                       label=marker_type.replace('_', ' ').title(), s=100, alpha=0.7)

        plt.title('Evolution Markers Timeline')
        plt.xlabel('Time')
        plt.ylabel('Confidence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = self.visualizations_dir / f'evolution_timeline_{self.session_id}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path

    def _plot_performance_heatmap(self) -> Path:
        """Create a heatmap of performance metrics."""
        if len(self.performance_history) < 10:
            return Path()

        performance_data = list(self.performance_history)

        # Create performance matrix
        metrics = ['success_rate', 'response_time', 'memory_usage']
        data_matrix = []

        for entry in performance_data:
            row = [
                entry.get('success_rate', 0),
                entry.get('response_time', 0),
                entry.get('memory_usage', 0)
            ]
            data_matrix.append(row)

        plt.figure(figsize=(10, 8))
        sns.heatmap(data_matrix, annot=True, fmt='.2f', cmap='viridis',
                   xticklabels=metrics, yticklabels=False)
        plt.title('Performance Metrics Heatmap')
        plt.tight_layout()

        plot_path = self.visualizations_dir / f'performance_heatmap_{self.session_id}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path

    def _plot_behavior_patterns(self) -> Path:
        """Plot behavioral patterns over time."""
        plt.figure(figsize=(12, 8))

        # Plot different behavioral patterns
        patterns_to_plot = {
            'Success Rates': self.behavioral_patterns['success_rates'],
            'Response Times': self.behavioral_patterns['response_times'],
            'Goal Complexity': self.behavioral_patterns['goal_complexity']
        }

        for i, (pattern_name, pattern_data) in enumerate(patterns_to_plot.items()):
            if pattern_data:
                plt.subplot(3, 1, i+1)
                plt.plot(pattern_data, linewidth=2, alpha=0.8)
                plt.title(f'{pattern_name} Over Time')
                plt.grid(True, alpha=0.3)
                plt.xlabel('Time Steps')
                plt.ylabel(pattern_name)

        plt.tight_layout()

        plot_path = self.visualizations_dir / f'behavior_patterns_{self.session_id}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path

    def _generate_executive_summary(self, metrics: ValidationMetrics, evolution_analysis: Dict[str, Any]) -> str:
        """Generate executive summary of validation results."""
        summary_parts = []

        # Overall assessment
        if evolution_analysis.get('evolution_detected', False):
            confidence = evolution_analysis.get('confidence', 0)
            summary_parts.append(f"EVOLUTION DETECTED with {confidence:.2%} confidence")
        else:
            summary_parts.append("NO SIGNIFICANT EVOLUTION DETECTED")

        # Performance summary
        summary_parts.append(f"Success rate: {metrics.goal_success_rate:.2%}")
        summary_parts.append(f"System stability: {metrics.system_stability:.2%}")

        # Key metrics
        if metrics.adaptation_rate > 0.1:
            summary_parts.append(f"Adaptation rate: {metrics.adaptation_rate:.2f} events/hour")

        if metrics.behavior_diversity > 0.3:
            summary_parts.append(f"High behavioral diversity: {metrics.behavior_diversity:.2%}")

        return " | ".join(summary_parts)

    def _save_validation_report(self, report: Dict[str, Any]) -> None:
        """Save validation report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f'validation_report_{timestamp}.json'

        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Validation report saved to {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")

    def get_current_metrics(self) -> ValidationMetrics:
        """Get current validation metrics."""
        return self._calculate_validation_metrics()

    def get_evolution_markers(self) -> List[EvolutionMarker]:
        """Get all detected evolution markers."""
        return self.evolution_markers.copy()

    def export_data(self, format: str = 'json') -> str:
        """Export all collected validation data."""
        data = {
            'session_id': self.session_id,
            'snapshots': [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'memory_usage': snapshot.memory_usage,
                    'cpu_usage': snapshot.cpu_usage,
                    'response_times': snapshot.response_times,
                    'success_rates': snapshot.success_rates,
                    'goal_generation_rate': snapshot.goal_generation_rate,
                    'active_goals_count': len(snapshot.active_goals)
                }
                for snapshot in self.behavior_snapshots
            ],
            'evolution_markers': [
                {
                    'timestamp': marker.timestamp.isoformat(),
                    'type': marker.marker_type,
                    'confidence': marker.confidence,
                    'description': marker.description,
                    'impact_score': marker.impact_score
                }
                for marker in self.evolution_markers
            ],
            'performance_history': list(self.performance_history)
        }

        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)

        return str(data)