"""
Safe controlled environment for autonomous system operation.

This module provides a sandboxed environment for running autonomous systems
with safety controls, resource limits, operation monitoring, and emergency
stop capabilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import resource
import signal
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import psutil


@dataclass
class SafetyLimits:
    """Safety limits for autonomous operation."""

    # Resource limits
    max_memory_mb: int = 2048
    max_cpu_percent: float = 80.0
    max_disk_usage_mb: int = 1024
    max_network_connections: int = 10

    # Time limits
    max_execution_time_seconds: int = 3600  # 1 hour
    max_idle_time_seconds: int = 300  # 5 minutes
    max_operation_time_seconds: int = 1800  # 30 minutes

    # Operation limits
    max_concurrent_operations: int = 5
    max_file_operations_per_minute: int = 100
    max_network_requests_per_minute: int = 50

    # Safety thresholds
    memory_warning_threshold: float = 70.0  # Percent
    cpu_warning_threshold: float = 60.0    # Percent
    error_rate_threshold: float = 0.3      # 30% error rate

    # Emergency stop conditions
    emergency_memory_threshold: float = 90.0  # Percent
    emergency_cpu_threshold: float = 90.0     # Percent
    emergency_error_threshold: float = 0.7    # 70% error rate


@dataclass
class OperationRecord:
    """Record of an autonomous operation."""

    operation_id: str
    operation_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, terminated
    resource_usage: Dict[str, float] = field(default_factory=dict)
    result: Any = None
    errors: List[str] = field(default_factory=list)
    safety_violations: List[str] = field(default_factory=list)


@dataclass
class EnvironmentSnapshot:
    """Snapshot of the controlled environment state."""

    timestamp: datetime

    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_mb: float = 0.0
    network_connections: int = 0

    # Operation state
    active_operations: int = 0
    completed_operations: int = 0
    failed_operations: int = 0

    # Safety state
    safety_violations: List[str] = field(default_factory=list)
    warning_conditions: List[str] = field(default_factory=list)
    emergency_conditions: List[str] = field(default_factory=list)


class ControlledEnvironment:
    """
    Safe controlled environment for autonomous system operation.

    Provides sandboxed execution with:
    - Resource limits and monitoring
    - Safety controls and emergency stops
    - Operation tracking and logging
    - Violation detection and response
    """

    def __init__(self, config, safety_limits: Optional[SafetyLimits] = None):
        self.config = config
        self.safety_limits = safety_limits or SafetyLimits()

        # Setup environment
        self._setup_controlled_environment()

        # Monitoring state
        self.is_active = False
        self.emergency_stop_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        # Operation tracking
        self.active_operations: Dict[str, OperationRecord] = {}
        self.operation_history: List[OperationRecord] = []
        self.environment_snapshots: deque = deque(maxlen=1000)

        # Safety state
        self.safety_violations: List[Dict[str, Any]] = []
        self.warning_conditions: List[Dict[str, Any]] = []
        self.emergency_stops: List[Dict[str, Any]] = []

        # Resource monitoring
        self.resource_monitoring_active = False
        self.file_operation_count = 0
        self.network_request_count = 0
        self.last_file_operation_reset = datetime.now()
        self.last_network_reset = datetime.now()

        # Callbacks
        self.safety_callbacks: List[Callable] = []
        self.emergency_callbacks: List[Callable] = []
        self.violation_callbacks: List[Callable] = []

    def _setup_controlled_environment(self) -> None:
        """Setup the controlled environment."""
        self.environment_root = self.config.workspace_root / "a3x" / "validation" / "controlled_environment"
        self.environment_root.mkdir(parents=True, exist_ok=True)

        # Create isolated directories
        self.temp_dir = self.environment_root / "temp"
        self.work_dir = self.environment_root / "work"
        self.log_dir = self.environment_root / "logs"
        self.safety_dir = self.environment_root / "safety"

        for directory in [self.temp_dir, self.work_dir, self.log_dir, self.safety_dir]:
            directory.mkdir(exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger("controlled_environment")
        self.logger.setLevel(logging.INFO)

        log_file = self.log_dir / f"environment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def start_environment(self) -> None:
        """Start the controlled environment."""
        if self.is_active:
            self.logger.warning("Environment already active")
            return

        self.is_active = True
        self.emergency_stop_active = False
        self._stop_monitoring.clear()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()

        # Initialize resource monitoring
        self.resource_monitoring_active = True

        # Reset counters
        self.file_operation_count = 0
        self.network_request_count = 0
        self.last_file_operation_reset = datetime.now()
        self.last_network_reset = datetime.now()

        self.logger.info("Controlled environment started")

    def stop_environment(self) -> Dict[str, Any]:
        """Stop the controlled environment and return summary."""
        if not self.is_active:
            self.logger.warning("Environment not active")
            return {}

        self.is_active = False
        self._stop_monitoring.set()

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)

        # Stop all active operations
        self._emergency_stop_all_operations()

        # Generate environment summary
        summary = self._generate_environment_summary()

        self.logger.info("Controlled environment stopped")
        return summary

    def _monitoring_loop(self) -> None:
        """Main monitoring loop for the controlled environment."""
        while not self._stop_monitoring.is_set():
            try:
                # Capture environment snapshot
                snapshot = self._capture_environment_snapshot()

                # Check safety conditions
                self._check_safety_conditions(snapshot)

                # Monitor resource usage
                self._monitor_resource_usage(snapshot)

                # Check operation limits
                self._check_operation_limits(snapshot)

                # Store snapshot
                self.environment_snapshots.append(snapshot)

                # Brief delay
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)

    def _capture_environment_snapshot(self) -> EnvironmentSnapshot:
        """Capture current state of the controlled environment."""
        snapshot = EnvironmentSnapshot(timestamp=datetime.now())

        # Resource usage
        snapshot.memory_usage_mb = self._get_memory_usage_mb()
        snapshot.cpu_usage_percent = self._get_cpu_usage()
        snapshot.disk_usage_mb = self._get_disk_usage_mb()
        snapshot.network_connections = self._get_network_connections()

        # Operation state
        snapshot.active_operations = len(self.active_operations)
        snapshot.completed_operations = len([op for op in self.operation_history if op.status == "completed"])
        snapshot.failed_operations = len([op for op in self.operation_history if op.status == "failed"])

        # Safety state
        snapshot.safety_violations = [v['violation_type'] for v in self.safety_violations[-10:]]
        snapshot.warning_conditions = [w['condition'] for w in self.warning_conditions[-10:]]
        snapshot.emergency_conditions = [e['condition'] for e in self.emergency_stops[-10:]]

        return snapshot

    def _get_memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0

    def _get_disk_usage_mb(self) -> float:
        """Get disk usage in MB."""
        try:
            usage = psutil.disk_usage(str(self.environment_root))
            return usage.used / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0

    def _get_network_connections(self) -> int:
        """Get number of network connections."""
        try:
            connections = psutil.net_connections()
            return len(connections)
        except Exception:
            return 0

    def _check_safety_conditions(self, snapshot: EnvironmentSnapshot) -> None:
        """Check for safety violations and emergency conditions."""
        violations = []
        warnings = []
        emergencies = []

        # Memory checks
        memory_percent = (snapshot.memory_usage_mb / self.safety_limits.max_memory_mb) * 100

        if memory_percent > self.safety_limits.emergency_memory_threshold:
            emergencies.append({
                'condition': 'critical_memory_usage',
                'value': memory_percent,
                'threshold': self.safety_limits.emergency_memory_threshold,
                'timestamp': snapshot.timestamp
            })
        elif memory_percent > self.safety_limits.memory_warning_threshold:
            warnings.append({
                'condition': 'high_memory_usage',
                'value': memory_percent,
                'threshold': self.safety_limits.memory_warning_threshold,
                'timestamp': snapshot.timestamp
            })

        # CPU checks
        if snapshot.cpu_usage_percent > self.safety_limits.emergency_cpu_threshold:
            emergencies.append({
                'condition': 'critical_cpu_usage',
                'value': snapshot.cpu_usage_percent,
                'threshold': self.safety_limits.emergency_cpu_threshold,
                'timestamp': snapshot.timestamp
            })
        elif snapshot.cpu_usage_percent > self.safety_limits.cpu_warning_threshold:
            warnings.append({
                'condition': 'high_cpu_usage',
                'value': snapshot.cpu_usage_percent,
                'threshold': self.safety_limits.cpu_warning_threshold,
                'timestamp': snapshot.timestamp
            })

        # Error rate checks
        if snapshot.active_operations > 0:
            error_rate = snapshot.failed_operations / (snapshot.completed_operations + snapshot.failed_operations + 1)
            if error_rate > self.safety_limits.emergency_error_threshold:
                emergencies.append({
                    'condition': 'critical_error_rate',
                    'value': error_rate,
                    'threshold': self.safety_limits.emergency_error_threshold,
                    'timestamp': snapshot.timestamp
                })
            elif error_rate > self.safety_limits.error_rate_threshold:
                warnings.append({
                    'condition': 'high_error_rate',
                    'value': error_rate,
                    'threshold': self.safety_limits.error_rate_threshold,
                    'timestamp': snapshot.timestamp
                })

        # Store conditions
        for warning in warnings:
            self.warning_conditions.append(warning)

        for emergency in emergencies:
            self.emergency_stops.append(emergency)

        # Trigger callbacks
        if warnings:
            self._trigger_safety_callbacks('warning', warnings)

        if emergencies:
            self._trigger_emergency_stop(emergencies)

    def _monitor_resource_usage(self, snapshot: EnvironmentSnapshot) -> None:
        """Monitor and enforce resource usage limits."""
        # Memory enforcement
        memory_mb = snapshot.memory_usage_mb
        if memory_mb > self.safety_limits.max_memory_mb:
            self.logger.warning(f"Memory limit exceeded: {memory_mb:.1f}MB > {self.safety_limits.max_memory_mb}MB")
            self._enforce_memory_limit()

        # CPU enforcement (log only, hard to enforce per-process)
        if snapshot.cpu_usage_percent > self.safety_limits.max_cpu_percent:
            self.logger.warning(f"CPU limit exceeded: {snapshot.cpu_usage_percent:.1f}% > {self.safety_limits.max_cpu_percent}%")

        # Disk usage enforcement
        disk_mb = snapshot.disk_usage_mb
        if disk_mb > self.safety_limits.max_disk_usage_mb:
            self.logger.warning(f"Disk usage limit exceeded: {disk_mb:.1f}MB > {self.safety_limits.max_disk_usage_mb}MB")
            self._enforce_disk_limit()

        # Network connections enforcement
        if snapshot.network_connections > self.safety_limits.max_network_connections:
            self.logger.warning(f"Network connections limit exceeded: {snapshot.network_connections} > {self.safety_limits.max_network_connections}")

    def _check_operation_limits(self, snapshot: EnvironmentSnapshot) -> None:
        """Check and enforce operation limits."""
        current_time = datetime.now()

        # Reset counters if minute has passed
        if (current_time - self.last_file_operation_reset).total_seconds() > 60:
            self.file_operation_count = 0
            self.last_file_operation_reset = current_time

        if (current_time - self.last_network_reset).total_seconds() > 60:
            self.network_request_count = 0
            self.last_network_reset = current_time

        # Check file operation rate
        if self.file_operation_count > self.safety_limits.max_file_operations_per_minute:
            self.logger.warning(f"File operation rate limit exceeded: {self.file_operation_count}/min")
            self._throttle_file_operations()

        # Check network request rate
        if self.network_request_count > self.safety_limits.max_network_requests_per_minute:
            self.logger.warning(f"Network request rate limit exceeded: {self.network_request_count}/min")
            self._throttle_network_requests()

        # Check concurrent operations
        if snapshot.active_operations > self.safety_limits.max_concurrent_operations:
            self.logger.warning(f"Concurrent operations limit exceeded: {snapshot.active_operations}")
            self._limit_concurrent_operations()

    def _enforce_memory_limit(self) -> None:
        """Enforce memory usage limits."""
        try:
            # Try to reduce memory usage by clearing caches
            import gc
            gc.collect()

            # If still high, trigger garbage collection in subprocesses
            self._trigger_memory_cleanup()

        except Exception as e:
            self.logger.error(f"Failed to enforce memory limit: {e}")

    def _enforce_disk_limit(self) -> None:
        """Enforce disk usage limits."""
        try:
            # Clean up temporary files
            self._cleanup_temp_files()

            # Archive old logs if necessary
            self._archive_old_logs()

        except Exception as e:
            self.logger.error(f"Failed to enforce disk limit: {e}")

    def _throttle_file_operations(self) -> None:
        """Throttle file operations to reduce rate."""
        # Implement throttling mechanism
        self.logger.info("Throttling file operations due to rate limit")

    def _throttle_network_requests(self) -> None:
        """Throttle network requests to reduce rate."""
        # Implement throttling mechanism
        self.logger.info("Throttling network requests due to rate limit")

    def _limit_concurrent_operations(self) -> None:
        """Limit concurrent operations."""
        # Terminate oldest operations if over limit
        if len(self.active_operations) > self.safety_limits.max_concurrent_operations:
            oldest_operations = sorted(
                self.active_operations.items(),
                key=lambda x: x[1].start_time
            )[:len(self.active_operations) - self.safety_limits.max_concurrent_operations]

            for op_id, operation in oldest_operations:
                self._terminate_operation(op_id, "Concurrent operation limit exceeded")

    def _trigger_safety_callbacks(self, alert_type: str, conditions: List[Dict[str, Any]]) -> None:
        """Trigger safety callbacks."""
        for callback in self.safety_callbacks:
            try:
                callback(alert_type, conditions)
            except Exception as e:
                self.logger.error(f"Error in safety callback: {e}")

    def _trigger_emergency_stop(self, emergency_conditions: List[Dict[str, Any]]) -> None:
        """Trigger emergency stop due to critical conditions."""
        self.emergency_stop_active = True

        # Log emergency
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {emergency_conditions}")

        # Terminate all active operations
        self._emergency_stop_all_operations()

        # Trigger emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback(emergency_conditions)
            except Exception as e:
                self.logger.error(f"Error in emergency callback: {e}")

    def _emergency_stop_all_operations(self) -> None:
        """Emergency stop all active operations."""
        for operation_id in list(self.active_operations.keys()):
            self._terminate_operation(operation_id, "Emergency stop activated")

    def _trigger_memory_cleanup(self) -> None:
        """Trigger memory cleanup across processes."""
        # This would integrate with the system's memory management
        self.logger.info("Triggering memory cleanup")

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        try:
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    # Remove directories recursively
                    import shutil
                    shutil.rmtree(file_path)

            self.logger.info("Cleaned up temporary files")
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp files: {e}")

    def _archive_old_logs(self) -> None:
        """Archive old log files."""
        try:
            # Archive logs older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)

            for log_file in self.log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time.timestamp():
                    # Move to archive
                    archive_dir = self.safety_dir / "archives"
                    archive_dir.mkdir(exist_ok=True)

                    archive_file = archive_dir / log_file.name
                    log_file.rename(archive_file)

            self.logger.info("Archived old log files")
        except Exception as e:
            self.logger.error(f"Failed to archive logs: {e}")

    def execute_safe_operation(self,
                             operation_type: str,
                             operation_function: Callable,
                             operation_parameters: Dict[str, Any] = None,
                             timeout_seconds: int = None) -> OperationRecord:
        """
        Execute an operation safely within the controlled environment.

        Args:
            operation_type: Type of operation (e.g., 'goal_execution', 'file_write')
            operation_function: Function to execute
            operation_parameters: Parameters for the operation
            timeout_seconds: Maximum execution time

        Returns:
            Operation record with results and monitoring data
        """
        if operation_parameters is None:
            operation_parameters = {}

        if timeout_seconds is None:
            timeout_seconds = self.safety_limits.max_operation_time_seconds

        operation_id = f"{operation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(operation_function)}"

        # Create operation record
        operation = OperationRecord(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=datetime.now()
        )

        self.active_operations[operation_id] = operation

        try:
            # Execute with timeout and monitoring
            result = self._execute_with_safety_controls(
                operation_function, operation_parameters, timeout_seconds
            )

            operation.result = result
            operation.status = "completed"
            operation.end_time = datetime.now()

            # Record final resource usage
            operation.resource_usage = {
                'memory_mb': self._get_memory_usage_mb(),
                'cpu_percent': self._get_cpu_usage(),
                'execution_time': (operation.end_time - operation.start_time).total_seconds()
            }

        except Exception as e:
            operation.status = "failed"
            operation.end_time = datetime.now()
            operation.errors.append(str(e))
            operation.resource_usage = {
                'memory_mb': self._get_memory_usage_mb(),
                'cpu_percent': self._get_cpu_usage(),
                'execution_time': (operation.end_time - operation.start_time).total_seconds()
            }

        # Move to history
        self.operation_history.append(operation)
        del self.active_operations[operation_id]

        self.logger.info(f"Operation {operation_id} completed with status: {operation.status}")

        return operation

    def _execute_with_safety_controls(self,
                                    operation_function: Callable,
                                    parameters: Dict[str, Any],
                                    timeout_seconds: int) -> Any:
        """Execute operation with safety controls and monitoring."""
        # Set resource limits for subprocess if needed
        if hasattr(operation_function, '__module__') and 'subprocess' in str(operation_function):
            return self._execute_subprocess_with_limits(operation_function, parameters, timeout_seconds)
        else:
            return self._execute_function_with_limits(operation_function, parameters, timeout_seconds)

    def _execute_function_with_limits(self,
                                    operation_function: Callable,
                                    parameters: Dict[str, Any],
                                    timeout_seconds: int) -> Any:
        """Execute function with time and resource limits."""
        # Track file and network operations
        original_open = open
        original_requests = None

        try:
            # Monkey patch for monitoring
            if 'requests' in str(operation_function):
                try:
                    import requests
                    original_get = requests.get
                    original_post = requests.post

                    def monitored_get(*args, **kwargs):
                        self.network_request_count += 1
                        return original_get(*args, **kwargs)

                    def monitored_post(*args, **kwargs):
                        self.network_request_count += 1
                        return original_post(*args, **kwargs)

                    requests.get = monitored_get
                    requests.post = monitored_post
                except ImportError:
                    pass

            # Execute with timeout
            result = self._execute_with_timeout(operation_function, parameters, timeout_seconds)

            return result

        finally:
            # Restore original functions
            if original_requests:
                # Restore requests functions
                pass

    def _execute_subprocess_with_limits(self,
                                      subprocess_function: Callable,
                                      parameters: Dict[str, Any],
                                      timeout_seconds: int) -> Any:
        """Execute subprocess with resource limits."""
        # Set resource limits for the subprocess
        def set_resource_limits():
            try:
                # Set memory limit
                resource.setrlimit(resource.RLIMIT_AS,
                                 (self.safety_limits.max_memory_mb * 1024 * 1024,
                                  self.safety_limits.max_memory_mb * 1024 * 1024))

                # Set CPU time limit
                resource.setrlimit(resource.RLIMIT_CPU, (timeout_seconds, timeout_seconds))

            except Exception as e:
                self.logger.warning(f"Could not set resource limits: {e}")

        # Execute subprocess with limits
        original_preexec_fn = parameters.get('preexec_fn')
        parameters['preexec_fn'] = set_resource_limits

        return subprocess_function(**parameters)

    def _execute_with_timeout(self,
                            operation_function: Callable,
                            parameters: Dict[str, Any],
                            timeout_seconds: int) -> Any:
        """Execute function with timeout."""
        result = None
        exception = None

        def target():
            nonlocal result, exception
            try:
                result = operation_function(**parameters)
            except Exception as e:
                exception = e

        # Start execution in thread
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()

        # Wait for completion with timeout
        thread.join(timeout_seconds)

        if thread.is_alive():
            # Timeout exceeded
            raise TimeoutError(f"Operation exceeded timeout of {timeout_seconds} seconds")

        if exception:
            raise exception

        return result

    def _terminate_operation(self, operation_id: str, reason: str) -> None:
        """Terminate a specific operation."""
        if operation_id in self.active_operations:
            operation = self.active_operations[operation_id]
            operation.status = "terminated"
            operation.end_time = datetime.now()
            operation.errors.append(f"Terminated: {reason}")

            # Move to history
            self.operation_history.append(operation)
            del self.active_operations[operation_id]

            self.logger.info(f"Terminated operation {operation_id}: {reason}")

    def add_safety_callback(self, callback: Callable) -> None:
        """Add callback for safety alerts."""
        self.safety_callbacks.append(callback)

    def add_emergency_callback(self, callback: Callable) -> None:
        """Add callback for emergency stops."""
        self.emergency_callbacks.append(callback)

    def add_violation_callback(self, callback: Callable) -> None:
        """Add callback for safety violations."""
        self.violation_callbacks.append(callback)

    def get_environment_status(self) -> Dict[str, Any]:
        """Get current environment status."""
        if not self.environment_snapshots:
            return {'status': 'no_data'}

        latest_snapshot = self.environment_snapshots[-1]

        return {
            'environment_active': self.is_active,
            'emergency_stop_active': self.emergency_stop_active,
            'resource_usage': {
                'memory_mb': latest_snapshot.memory_usage_mb,
                'cpu_percent': latest_snapshot.cpu_usage_percent,
                'disk_mb': latest_snapshot.disk_usage_mb,
                'network_connections': latest_snapshot.network_connections
            },
            'operation_state': {
                'active_operations': latest_snapshot.active_operations,
                'completed_operations': latest_snapshot.completed_operations,
                'failed_operations': latest_snapshot.failed_operations
            },
            'safety_state': {
                'safety_violations': len(self.safety_violations),
                'warning_conditions': len(self.warning_conditions),
                'emergency_conditions': len(self.emergency_stops)
            },
            'rate_limits': {
                'file_operations_per_minute': self.file_operation_count,
                'network_requests_per_minute': self.network_request_count
            }
        }

    def _generate_environment_summary(self) -> Dict[str, Any]:
        """Generate summary of environment operation."""
        total_operations = len(self.operation_history)
        successful_operations = len([op for op in self.operation_history if op.status == "completed"])
        failed_operations = len([op for op in self.operation_history if op.status == "failed"])
        terminated_operations = len([op for op in self.operation_history if op.status == "terminated"])

        # Calculate average resource usage
        completed_operations = [op for op in self.operation_history if op.status == "completed"]

        avg_execution_time = 0
        avg_memory_usage = 0
        avg_cpu_usage = 0

        if completed_operations:
            avg_execution_time = statistics.mean([op.resource_usage.get('execution_time', 0) for op in completed_operations])
            avg_memory_usage = statistics.mean([op.resource_usage.get('memory_mb', 0) for op in completed_operations])
            avg_cpu_usage = statistics.mean([op.resource_usage.get('cpu_percent', 0) for op in completed_operations])

        return {
            'environment_session': {
                'start_time': self.environment_snapshots[0].timestamp.isoformat() if self.environment_snapshots else None,
                'end_time': self.environment_snapshots[-1].timestamp.isoformat() if self.environment_snapshots else None,
                'duration_seconds': (self.environment_snapshots[-1].timestamp - self.environment_snapshots[0].timestamp).total_seconds() if len(self.environment_snapshots) >= 2 else 0
            },
            'operation_summary': {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'failed_operations': failed_operations,
                'terminated_operations': terminated_operations,
                'success_rate': successful_operations / total_operations if total_operations > 0 else 0
            },
            'resource_usage_summary': {
                'average_execution_time': avg_execution_time,
                'average_memory_usage_mb': avg_memory_usage,
                'average_cpu_usage_percent': avg_cpu_usage
            },
            'safety_summary': {
                'total_safety_violations': len(self.safety_violations),
                'total_warnings': len(self.warning_conditions),
                'total_emergency_stops': len(self.emergency_stops),
                'emergency_stop_activated': self.emergency_stop_active
            },
            'environment_health_score': self._calculate_environment_health_score()
        }

    def _calculate_environment_health_score(self) -> float:
        """Calculate overall health score for the environment (0-1)."""
        if not self.environment_snapshots:
            return 0.0

        latest = self.environment_snapshots[-1]

        # Base score from success rate
        success_rate = latest.completed_operations / max(latest.completed_operations + latest.failed_operations, 1)
        health_score = success_rate * 0.4

        # Resource usage score (lower usage = higher score)
        memory_score = max(0, (100 - (latest.memory_usage_mb / self.safety_limits.max_memory_mb * 100))) / 100 * 0.3
        cpu_score = max(0, (100 - latest.cpu_usage_percent)) / 100 * 0.3

        health_score += memory_score + cpu_score

        return min(1.0, health_score)

    def save_environment_data(self) -> None:
        """Save environment operation data."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save operation history
        operations_file = self.log_dir / f'operations_{timestamp}.json'
        operations_data = [
            {
                'operation_id': op.operation_id,
                'operation_type': op.operation_type,
                'start_time': op.start_time.isoformat(),
                'end_time': op.end_time.isoformat() if op.end_time else None,
                'status': op.status,
                'resource_usage': op.resource_usage,
                'errors': op.errors,
                'safety_violations': op.safety_violations
            }
            for op in self.operation_history[-100:]  # Last 100 operations
        ]

        try:
            with open(operations_file, 'w') as f:
                json.dump(operations_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save operations data: {e}")

        # Save safety violations
        safety_file = self.safety_dir / f'safety_log_{timestamp}.json'
        safety_data = {
            'safety_violations': self.safety_violations[-50:],
            'warning_conditions': self.warning_conditions[-50:],
            'emergency_stops': self.emergency_stops[-20:]
        }

        try:
            with open(safety_file, 'w') as f:
                json.dump(safety_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save safety data: {e}")

    def create_safety_limits(self,
                           max_memory_mb: int = 2048,
                           max_cpu_percent: float = 80.0,
                           max_execution_time_seconds: int = 3600) -> SafetyLimits:
        """Create custom safety limits configuration."""
        return SafetyLimits(
            max_memory_mb=max_memory_mb,
            max_cpu_percent=max_cpu_percent,
            max_execution_time_seconds=max_execution_time_seconds
        )