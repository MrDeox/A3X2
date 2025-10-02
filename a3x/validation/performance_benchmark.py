"""
Autonomous performance benchmarking system for A3X SeedAI.

This module provides comprehensive performance benchmarking capabilities
for autonomous systems, including efficiency measurement, scalability
testing, resource utilization analysis, and performance regression detection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark test."""

    benchmark_id: str
    test_name: str
    timestamp: datetime
    duration: float
    success: bool

    # Performance metrics
    execution_time: float = 0.0
    memory_usage_peak: float = 0.0
    cpu_usage_peak: float = 0.0
    throughput: float = 0.0

    # Quality metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Efficiency metrics
    resource_efficiency: float = 0.0
    time_efficiency: float = 0.0
    cost_efficiency: float = 0.0

    # Scalability metrics
    scalability_score: float = 0.0
    concurrent_operations: int = 0

    # Detailed results
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class PerformanceBaseline:
    """Baseline performance metrics for comparison."""

    baseline_id: str
    test_name: str
    established_date: datetime

    # Baseline metrics
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    acceptable_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Statistical properties
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    standard_deviations: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalabilityTest:
    """Configuration for scalability testing."""

    test_id: str
    test_name: str
    parameter_name: str  # 'load', 'complexity', 'concurrency'
    parameter_values: List[Any] = field(default_factory=list)
    repetitions: int = 5
    timeout_per_test: float = 300.0  # 5 minutes


class AutonomousPerformanceBenchmark:
    """
    Comprehensive performance benchmarking for autonomous systems.

    Provides systematic performance testing including:
    - Efficiency benchmarking
    - Scalability testing
    - Resource utilization analysis
    - Performance regression detection
    - Comparative analysis
    """

    def __init__(self, config, baseline_window: int = 50, regression_threshold: float = 0.2):
        self.config = config
        self.baseline_window = baseline_window
        self.regression_threshold = regression_threshold

        # Setup benchmarking environment
        self._setup_benchmarking_environment()

        # Benchmark storage
        self.benchmark_results: List[BenchmarkResult] = []
        self.performance_baselines: Dict[str, PerformanceBaseline] = {}
        self.scalability_tests: Dict[str, ScalabilityTest] = {}

        # Performance tracking
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.regression_alerts: List[Dict[str, Any]] = []

        # Real-time monitoring
        self.benchmark_in_progress: Optional[str] = None
        self.current_baseline_tests: Dict[str, List[BenchmarkResult]] = defaultdict(list)

    def _setup_benchmarking_environment(self) -> None:
        """Setup benchmarking environment and storage."""
        self.benchmark_root = self.config.workspace_root / "a3x" / "validation" / "performance_benchmarks"
        self.benchmark_root.mkdir(parents=True, exist_ok=True)

        self.results_dir = self.benchmark_root / "results"
        self.baselines_dir = self.benchmark_root / "baselines"
        self.reports_dir = self.benchmark_root / "reports"
        self.scalability_dir = self.benchmark_root / "scalability"

        for directory in [self.results_dir, self.baselines_dir, self.reports_dir, self.scalability_dir]:
            directory.mkdir(exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger("performance_benchmark")
        self.logger.setLevel(logging.INFO)

    def run_autonomous_benchmark(self,
                               test_name: str,
                               test_function: callable,
                               test_parameters: Dict[str, Any] = None,
                               repetitions: int = 3) -> List[BenchmarkResult]:
        """
        Run autonomous performance benchmark test.

        Args:
            test_name: Name of the benchmark test
            test_function: Function to execute for benchmarking
            test_parameters: Parameters for the test function
            repetitions: Number of times to repeat the test

        Returns:
            List of benchmark results
        """
        if test_parameters is None:
            test_parameters = {}

        self.benchmark_in_progress = test_name
        results = []

        try:
            for rep in range(repetitions):
                self.logger.info(f"Running benchmark '{test_name}' repetition {rep + 1}/{repetitions}")

                # Run single benchmark
                result = self._run_single_benchmark(test_name, test_function, test_parameters, rep)

                results.append(result)

                # Store result
                self.benchmark_results.append(result)
                self.performance_history[test_name].append(result.execution_time)

                # Update baseline tracking
                self._update_baseline_tracking(test_name, result)

                # Brief delay between repetitions
                if rep < repetitions - 1:
                    time.sleep(1.0)

        except Exception as e:
            self.logger.error(f"Error running benchmark '{test_name}': {e}")
            error_result = BenchmarkResult(
                benchmark_id=f"{test_name}_{datetime.now().isoformat()}_error",
                test_name=test_name,
                timestamp=datetime.now(),
                duration=0.0,
                success=False,
                errors=[str(e)]
            )
            results.append(error_result)

        finally:
            self.benchmark_in_progress = None

        # Analyze results
        self._analyze_benchmark_results(test_name, results)

        return results

    def _run_single_benchmark(self,
                            test_name: str,
                            test_function: callable,
                            test_parameters: Dict[str, Any],
                            repetition: int) -> BenchmarkResult:
        """Run a single benchmark test with detailed monitoring."""
        benchmark_id = f"{test_name}_{datetime.now().isoformat()}_{repetition}"

        # Monitor resource usage before test
        initial_memory = self._get_memory_usage()
        initial_cpu = self._get_cpu_usage()

        # Record start time
        start_time = time.perf_counter()
        test_start_time = datetime.now()

        try:
            # Run the test function
            test_result = test_function(**test_parameters)

            # Record end time
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # Monitor resource usage during/after test
            peak_memory = self._monitor_memory_during_test()
            peak_cpu = self._monitor_cpu_during_test()

            # Calculate throughput if applicable
            throughput = self._calculate_throughput(test_result, execution_time)

            # Calculate quality metrics if applicable
            quality_metrics = self._calculate_quality_metrics(test_result)

            # Calculate efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(
                execution_time, peak_memory, peak_cpu, test_result
            )

            # Create result object
            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                test_name=test_name,
                timestamp=test_start_time,
                duration=execution_time,
                success=True,
                execution_time=execution_time,
                memory_usage_peak=peak_memory,
                cpu_usage_peak=peak_cpu,
                throughput=throughput,
                accuracy=quality_metrics.get('accuracy', 0.0),
                precision=quality_metrics.get('precision', 0.0),
                recall=quality_metrics.get('recall', 0.0),
                f1_score=quality_metrics.get('f1_score', 0.0),
                resource_efficiency=efficiency_metrics.get('resource_efficiency', 0.0),
                time_efficiency=efficiency_metrics.get('time_efficiency', 0.0),
                cost_efficiency=efficiency_metrics.get('cost_efficiency', 0.0),
                metrics={
                    'test_result': test_result,
                    'initial_memory': initial_memory,
                    'initial_cpu': initial_cpu,
                    'quality_metrics': quality_metrics,
                    'efficiency_metrics': efficiency_metrics
                }
            )

        except Exception as e:
            # Record failed test
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                test_name=test_name,
                timestamp=test_start_time,
                duration=execution_time,
                success=False,
                execution_time=execution_time,
                memory_usage_peak=self._get_memory_usage(),
                cpu_usage_peak=self._get_cpu_usage(),
                errors=[str(e)]
            )

        return result

    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_percent()
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _monitor_memory_during_test(self) -> float:
        """Monitor memory usage during test execution."""
        # Simplified monitoring - in practice, would track peak during execution
        return self._get_memory_usage()

    def _monitor_cpu_during_test(self) -> float:
        """Monitor CPU usage during test execution."""
        # Simplified monitoring - in practice, would track peak during execution
        return self._get_cpu_usage()

    def _calculate_throughput(self, test_result: Any, execution_time: float) -> float:
        """Calculate throughput based on test result."""
        # Default implementation - override for specific test types
        if isinstance(test_result, dict) and 'operations_count' in test_result:
            operations = test_result['operations_count']
            if execution_time > 0:
                return operations / execution_time

        if isinstance(test_result, list):
            operations = len(test_result)
            if execution_time > 0:
                return operations / execution_time

        return 0.0

    def _calculate_quality_metrics(self, test_result: Any) -> Dict[str, float]:
        """Calculate quality metrics from test result."""
        metrics = {}

        # Default quality metrics - override for specific test types
        if isinstance(test_result, dict):
            if 'accuracy' in test_result:
                metrics['accuracy'] = test_result['accuracy']
            if 'precision' in test_result:
                metrics['precision'] = test_result['precision']
            if 'recall' in test_result:
                metrics['recall'] = test_result['recall']
            if 'f1_score' in test_result:
                metrics['f1_score'] = test_result['f1_score']

        # Calculate derived metrics
        if 'precision' in metrics and 'recall' in metrics:
            p = metrics['precision']
            r = metrics['recall']
            if p + r > 0:
                metrics['f1_score'] = 2 * p * r / (p + r)

        return metrics

    def _calculate_efficiency_metrics(self,
                                   execution_time: float,
                                   memory_usage: float,
                                   cpu_usage: float,
                                   test_result: Any) -> Dict[str, float]:
        """Calculate efficiency metrics."""
        metrics = {}

        # Time efficiency (inverse of execution time, normalized)
        if execution_time > 0:
            metrics['time_efficiency'] = min(1.0 / execution_time, 10.0)  # Cap at 10 ops/sec

        # Resource efficiency (inverse of resource usage)
        resource_cost = (memory_usage * 0.6) + (cpu_usage * 0.4)  # Weighted resource cost
        if resource_cost > 0:
            metrics['resource_efficiency'] = min(100.0 / resource_cost, 1.0)

        # Cost efficiency (combination of time and resource)
        time_eff = metrics.get('time_efficiency', 0)
        resource_eff = metrics.get('resource_efficiency', 0)
        metrics['cost_efficiency'] = (time_eff + resource_eff) / 2

        return metrics

    def _update_baseline_tracking(self, test_name: str, result: BenchmarkResult) -> None:
        """Update baseline tracking with new result."""
        # Add to current baseline tests
        self.current_baseline_tests[test_name].append(result)

        # Check if we have enough data to establish/update baseline
        if len(self.current_baseline_tests[test_name]) >= self.baseline_window:
            self._establish_or_update_baseline(test_name)

    def _establish_or_update_baseline(self, test_name: str) -> None:
        """Establish or update performance baseline for a test."""
        results = self.current_baseline_tests[test_name]

        if not results:
            return

        # Calculate baseline metrics
        baseline_metrics = {}

        # Execution time baseline
        execution_times = [r.execution_time for r in results if r.success]
        if execution_times:
            baseline_metrics['execution_time_mean'] = statistics.mean(execution_times)
            baseline_metrics['execution_time_std'] = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            baseline_metrics['execution_time_median'] = statistics.median(execution_times)

        # Memory usage baseline
        memory_usage = [r.memory_usage_peak for r in results if r.success]
        if memory_usage:
            baseline_metrics['memory_usage_mean'] = statistics.mean(memory_usage)
            baseline_metrics['memory_usage_std'] = statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0

        # CPU usage baseline
        cpu_usage = [r.cpu_usage_peak for r in results if r.success]
        if cpu_usage:
            baseline_metrics['cpu_usage_mean'] = statistics.mean(cpu_usage)
            baseline_metrics['cpu_usage_std'] = statistics.stdev(cpu_usage) if len(cpu_usage) > 1 else 0

        # Success rate baseline
        success_count = sum(1 for r in results if r.success)
        baseline_metrics['success_rate'] = success_count / len(results)

        # Calculate acceptable ranges (±2 standard deviations)
        acceptable_ranges = {}
        for metric, value in baseline_metrics.items():
            if metric.endswith('_mean') and f"{metric[:-5]}_std" in baseline_metrics:
                std = baseline_metrics[f"{metric[:-5]}_std"]
                acceptable_ranges[metric[:-5]] = (value - 2*std, value + 2*std)

        # Calculate confidence intervals (95%)
        confidence_intervals = {}
        for metric, value in baseline_metrics.items():
            if metric.endswith('_mean') and f"{metric[:-5]}_std" in baseline_metrics:
                std = baseline_metrics[f"{metric[:-5]}_std"]
                n = len([r for r in results if r.success])
                if n > 1:
                    std_error = std / math.sqrt(n)
                    confidence_intervals[metric[:-5]] = (
                        value - 1.96 * std_error,
                        value + 1.96 * std_error
                    )

        # Create or update baseline
        baseline_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        baseline = PerformanceBaseline(
            baseline_id=baseline_id,
            test_name=test_name,
            established_date=datetime.now(),
            baseline_metrics=baseline_metrics,
            acceptable_ranges=acceptable_ranges,
            confidence_intervals=confidence_intervals,
            standard_deviations={k.replace('_std', ''): v for k, v in baseline_metrics.items() if k.endswith('_std')}
        )

        self.performance_baselines[test_name] = baseline

        # Clear current tests for next baseline period
        self.current_baseline_tests[test_name] = []

        self.logger.info(f"Updated baseline for '{test_name}': {len(results)} samples")

    def _analyze_benchmark_results(self, test_name: str, results: List[BenchmarkResult]) -> None:
        """Analyze benchmark results for insights and regression detection."""
        if not results:
            return

        # Check for performance regression
        self._check_performance_regression(test_name, results)

        # Calculate performance statistics
        successful_results = [r for r in results if r.success]

        if successful_results:
            # Performance trends
            execution_times = [r.execution_time for r in successful_results]
            memory_usage = [r.memory_usage_peak for r in successful_results]
            cpu_usage = [r.cpu_usage_peak for r in successful_results]

            # Statistical analysis
            performance_stats = {
                'execution_time': {
                    'mean': statistics.mean(execution_times),
                    'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    'min': min(execution_times),
                    'max': max(execution_times)
                },
                'memory_usage': {
                    'mean': statistics.mean(memory_usage),
                    'std': statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
                    'max': max(memory_usage)
                },
                'cpu_usage': {
                    'mean': statistics.mean(cpu_usage),
                    'std': statistics.stdev(cpu_usage) if len(cpu_usage) > 1 else 0,
                    'max': max(cpu_usage)
                }
            }

            self.logger.info(f"Benchmark '{test_name}' analysis: {len(successful_results)}/{len(results)} successful")
            self.logger.info(f"Performance stats: {performance_stats}")

    def _check_performance_regression(self, test_name: str, results: List[BenchmarkResult]) -> None:
        """Check for performance regression against baseline."""
        if test_name not in self.performance_baselines:
            return

        baseline = self.performance_baselines[test_name]

        # Check each result against baseline
        for result in results:
            if not result.success:
                continue

            regression_detected = False
            regression_details = []

            # Check execution time regression
            if 'execution_time' in baseline.acceptable_ranges:
                time_range = baseline.acceptable_ranges['execution_time']
                if not (time_range[0] <= result.execution_time <= time_range[1]):
                    regression_detected = True
                    regression_ratio = result.execution_time / baseline.baseline_metrics.get('execution_time_mean', 1)
                    regression_details.append(f"Execution time: {regression_ratio:.2f}x baseline")

            # Check memory usage regression
            if 'memory_usage' in baseline.acceptable_ranges:
                memory_range = baseline.acceptable_ranges['memory_usage']
                if not (memory_range[0] <= result.memory_usage_peak <= memory_range[1]):
                    regression_detected = True
                    regression_ratio = result.memory_usage_peak / baseline.baseline_metrics.get('memory_usage_mean', 1)
                    regression_details.append(f"Memory usage: {regression_ratio:.2f}x baseline")

            if regression_detected:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'test_name': test_name,
                    'benchmark_id': result.benchmark_id,
                    'regression_type': 'performance',
                    'details': regression_details,
                    'severity': 'high' if len(regression_details) > 1 else 'medium'
                }

                self.regression_alerts.append(alert)
                self.logger.warning(f"Performance regression detected in '{test_name}': {regression_details}")

    def run_scalability_test(self, scalability_test: ScalabilityTest, test_function: callable) -> Dict[str, Any]:
        """
        Run scalability test across different parameter values.

        Args:
            scalability_test: Configuration for scalability testing
            test_function: Function to test scalability

        Returns:
            Scalability analysis results
        """
        self.logger.info(f"Running scalability test '{scalability_test.test_name}'")

        results = {}

        for param_value in scalability_test.parameter_values:
            self.logger.info(f"Testing {scalability_test.parameter_name} = {param_value}")

            # Run multiple repetitions for each parameter value
            test_results = []

            for rep in range(scalability_test.repetitions):
                # Prepare test parameters
                test_params = {scalability_test.parameter_name: param_value}

                # Run benchmark
                benchmark_results = self.run_autonomous_benchmark(
                    f"{scalability_test.test_name}_{param_value}_{rep}",
                    test_function,
                    test_params,
                    repetitions=1
                )

                if benchmark_results:
                    test_results.append(benchmark_results[0])

            # Analyze results for this parameter value
            results[param_value] = self._analyze_scalability_point(test_results, param_value)

        # Generate scalability analysis
        scalability_analysis = self._generate_scalability_analysis(results, scalability_test)

        # Save scalability results
        self._save_scalability_results(scalability_test.test_id, results, scalability_analysis)

        return scalability_analysis

    def _analyze_scalability_point(self, results: List[BenchmarkResult], parameter_value: Any) -> Dict[str, Any]:
        """Analyze results for a specific scalability test point."""
        if not results:
            return {'error': 'No results'}

        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {'error': 'No successful results'}

        # Calculate statistics
        execution_times = [r.execution_time for r in successful_results]
        memory_usage = [r.memory_usage_peak for r in successful_results]
        throughput_values = [r.throughput for r in successful_results]

        return {
            'parameter_value': parameter_value,
            'repetitions': len(results),
            'successful_runs': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'execution_time': {
                'mean': statistics.mean(execution_times),
                'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'min': min(execution_times),
                'max': max(execution_times)
            },
            'memory_usage': {
                'mean': statistics.mean(memory_usage),
                'max': max(memory_usage)
            },
            'throughput': {
                'mean': statistics.mean(throughput_values) if throughput_values else 0,
                'trend': self._calculate_trend(throughput_values)
            }
        }

    def _generate_scalability_analysis(self, results: Dict[Any, Dict[str, Any]], scalability_test: ScalabilityTest) -> Dict[str, Any]:
        """Generate comprehensive scalability analysis."""
        if not results:
            return {'error': 'No scalability data'}

        # Extract parameter values and performance metrics
        param_values = list(results.keys())
        execution_times = [results[pv]['execution_time']['mean'] for pv in param_values]
        throughput_values = [results[pv]['throughput']['mean'] for pv in param_values]
        memory_usage = [results[pv]['memory_usage']['mean'] for pv in param_values]

        # Calculate scalability metrics
        scalability_metrics = {}

        # Linear scalability check
        if len(param_values) >= 3 and len(execution_times) >= 3:
            # Check if execution time scales linearly with parameter
            time_trend = self._calculate_linear_trend(param_values, execution_times)
            scalability_metrics['time_scalability'] = time_trend

            # Check if throughput scales linearly (inverse of time)
            if all(t > 0 for t in execution_times):
                throughput_efficiency = [pv / et for pv, et in zip(param_values, execution_times)]
                throughput_trend = self._calculate_linear_trend(param_values, throughput_efficiency)
                scalability_metrics['throughput_scalability'] = throughput_trend

        # Memory scalability
        memory_trend = self._calculate_linear_trend(param_values, memory_usage)
        scalability_metrics['memory_scalability'] = memory_trend

        # Overall scalability assessment
        scalability_score = self._calculate_scalability_score(scalability_metrics)
        scalability_metrics['overall_scalability_score'] = scalability_score

        # Identify scalability bottlenecks
        bottlenecks = self._identify_scalability_bottlenecks(results, scalability_metrics)

        return {
            'test_id': scalability_test.test_id,
            'test_name': scalability_test.test_name,
            'parameter_range': {
                'min': min(param_values),
                'max': max(param_values),
                'values_tested': len(param_values)
            },
            'scalability_metrics': scalability_metrics,
            'bottlenecks': bottlenecks,
            'recommendations': self._generate_scalability_recommendations(scalability_metrics, bottlenecks),
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _calculate_linear_trend(self, x_values: List[float], y_values: List[float]) -> str:
        """Calculate if relationship is linear and direction."""
        if len(x_values) < 3 or len(y_values) < 3:
            return 'insufficient_data'

        # Simple linear regression slope
        x = np.array(x_values)
        y = np.array(y_values)

        slope, _, r_value, _, _ = stats.linregress(x, y)

        # Assess linearity and direction
        r_squared = r_value ** 2

        if r_squared < 0.7:  # Poor linear fit
            return 'non_linear'
        elif slope > 0.1:
            return 'positive_linear'
        elif slope < -0.1:
            return 'negative_linear'
        else:
            return 'stable'

    def _calculate_scalability_score(self, scalability_metrics: Dict[str, str]) -> float:
        """Calculate overall scalability score (0-1)."""
        scores = []

        # Time scalability score
        time_scalability = scalability_metrics.get('time_scalability', 'stable')
        if time_scalability == 'stable':
            scores.append(1.0)
        elif time_scalability == 'positive_linear':
            scores.append(0.8)
        elif time_scalability == 'negative_linear':
            scores.append(0.3)
        else:
            scores.append(0.5)

        # Throughput scalability score
        throughput_scalability = scalability_metrics.get('throughput_scalability', 'stable')
        if throughput_scalability == 'positive_linear':
            scores.append(1.0)
        elif throughput_scalability == 'stable':
            scores.append(0.8)
        elif throughput_scalability == 'negative_linear':
            scores.append(0.3)
        else:
            scores.append(0.5)

        # Memory scalability score
        memory_scalability = scalability_metrics.get('memory_scalability', 'stable')
        if memory_scalability == 'stable':
            scores.append(1.0)
        elif memory_scalability == 'positive_linear':
            scores.append(0.7)  # Memory growth is acceptable but not ideal
        elif memory_scalability == 'negative_linear':
            scores.append(0.9)  # Memory reduction is good
        else:
            scores.append(0.5)

        return statistics.mean(scores) if scores else 0.0

    def _identify_scalability_bottlenecks(self, results: Dict[Any, Dict[str, Any]], scalability_metrics: Dict[str, str]) -> List[str]:
        """Identify scalability bottlenecks."""
        bottlenecks = []

        # Check for time bottlenecks
        time_scalability = scalability_metrics.get('time_scalability', 'stable')
        if time_scalability in ['non_linear', 'positive_linear']:
            bottlenecks.append('execution_time')

        # Check for memory bottlenecks
        memory_scalability = scalability_metrics.get('memory_scalability', 'stable')
        if memory_scalability == 'positive_linear':
            bottlenecks.append('memory_usage')

        # Check for throughput bottlenecks
        throughput_scalability = scalability_metrics.get('throughput_scalability', 'stable')
        if throughput_scalability == 'negative_linear':
            bottlenecks.append('throughput_decline')

        return bottlenecks

    def _generate_scalability_recommendations(self, scalability_metrics: Dict[str, str], bottlenecks: List[str]) -> List[str]:
        """Generate recommendations for scalability improvements."""
        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck == 'execution_time':
                recommendations.append("Consider optimizing algorithms for better time complexity")
                recommendations.append("Implement caching for frequently accessed data")
                recommendations.append("Review and optimize I/O operations")

            elif bottleneck == 'memory_usage':
                recommendations.append("Implement memory pooling for frequently allocated objects")
                recommendations.append("Review data structures for memory efficiency")
                recommendations.append("Consider lazy loading for large datasets")

            elif bottleneck == 'throughput_decline':
                recommendations.append("Investigate lock contention in concurrent operations")
                recommendations.append("Consider parallel processing where applicable")
                recommendations.append("Review resource allocation strategies")

        if not bottlenecks:
            recommendations.append("System scales well - continue monitoring for changes")

        return recommendations

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series."""
        if len(values) < 3:
            return 'insufficient_data'

        # Simple linear regression
        x = list(range(len(values)))
        slope, _, _, _, _ = stats.linregress(x, values)

        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'

    def _save_scalability_results(self, test_id: str, results: Dict[Any, Dict[str, Any]], analysis: Dict[str, Any]) -> None:
        """Save scalability test results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.scalability_dir / f'scalability_{test_id}_{timestamp}.json'

        data = {
            'test_id': test_id,
            'results': results,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(results_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save scalability results: {e}")

    def create_scalability_test(self,
                              test_id: str,
                              test_name: str,
                              parameter_name: str,
                              parameter_values: List[Any],
                              repetitions: int = 5) -> ScalabilityTest:
        """Create a scalability test configuration."""
        test = ScalabilityTest(
            test_id=test_id,
            test_name=test_name,
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            repetitions=repetitions
        )

        self.scalability_tests[test_id] = test
        return test

    def generate_performance_report(self, test_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Filter results if test_name specified
        if test_name:
            relevant_results = [r for r in self.benchmark_results if r.test_name == test_name]
        else:
            relevant_results = self.benchmark_results

        if not relevant_results:
            return {'error': 'No benchmark results available'}

        # Performance summary
        performance_summary = self._generate_performance_summary(relevant_results)

        # Baseline comparison
        baseline_comparison = self._generate_baseline_comparison(relevant_results)

        # Regression analysis
        regression_analysis = self._generate_regression_analysis(relevant_results)

        # Trend analysis
        trend_analysis = self._generate_trend_analysis(relevant_results)

        report = {
            'report_timestamp': datetime.now().isoformat(),
            'test_coverage': test_name or 'all_tests',
            'total_benchmarks': len(relevant_results),
            'performance_summary': performance_summary,
            'baseline_comparison': baseline_comparison,
            'regression_analysis': regression_analysis,
            'trend_analysis': trend_analysis,
            'recommendations': self._generate_performance_recommendations(
                performance_summary, regression_analysis, trend_analysis
            )
        }

        return report

    def _generate_performance_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {'error': 'No successful results'}

        # Execution time analysis
        execution_times = [r.execution_time for r in successful_results]
        execution_summary = {
            'mean': statistics.mean(execution_times),
            'median': statistics.median(execution_times),
            'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'min': min(execution_times),
            'max': max(execution_times),
            'p95': np.percentile(execution_times, 95),
            'p99': np.percentile(execution_times, 99)
        }

        # Resource usage analysis
        memory_usage = [r.memory_usage_peak for r in successful_results]
        cpu_usage = [r.cpu_usage_peak for r in successful_results]

        resource_summary = {
            'memory': {
                'mean': statistics.mean(memory_usage),
                'max': max(memory_usage),
                'std': statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0
            },
            'cpu': {
                'mean': statistics.mean(cpu_usage),
                'max': max(cpu_usage),
                'std': statistics.stdev(cpu_usage) if len(cpu_usage) > 1 else 0
            }
        }

        # Success rate
        success_rate = len(successful_results) / len(results)

        # Quality metrics
        quality_metrics = {
            'accuracy': statistics.mean([r.accuracy for r in successful_results if r.accuracy > 0]),
            'precision': statistics.mean([r.precision for r in successful_results if r.precision > 0]),
            'recall': statistics.mean([r.recall for r in successful_results if r.recall > 0]),
            'f1_score': statistics.mean([r.f1_score for r in successful_results if r.f1_score > 0])
        }

        return {
            'execution_time': execution_summary,
            'resource_usage': resource_summary,
            'success_rate': success_rate,
            'quality_metrics': quality_metrics,
            'sample_size': len(successful_results)
        }

    def _generate_baseline_comparison(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compare current performance against baselines."""
        comparison = {}

        # Group results by test name
        test_groups = defaultdict(list)
        for result in results:
            test_groups[result.test_name].append(result)

        for test_name, test_results in test_groups.items():
            if test_name in self.performance_baselines:
                baseline = self.performance_baselines[test_name]
                test_comparison = self._compare_test_to_baseline(test_results, baseline)
                comparison[test_name] = test_comparison
            else:
                comparison[test_name] = {'baseline_status': 'no_baseline'}

        return comparison

    def _compare_test_to_baseline(self, results: List[BenchmarkResult], baseline: PerformanceBaseline) -> Dict[str, Any]:
        """Compare test results to baseline."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {'comparison_status': 'no_successful_results'}

        # Execution time comparison
        execution_times = [r.execution_time for r in successful_results]
        current_mean_time = statistics.mean(execution_times)
        baseline_time = baseline.baseline_metrics.get('execution_time_mean', 0)

        time_comparison = 'baseline_unavailable'
        if baseline_time > 0:
            time_ratio = current_mean_time / baseline_time
            if time_ratio > 1.2:
                time_comparison = 'slower'
            elif time_ratio < 0.8:
                time_comparison = 'faster'
            else:
                time_comparison = 'comparable'

        # Memory usage comparison
        memory_usage = [r.memory_usage_peak for r in successful_results]
        current_memory = statistics.mean(memory_usage)
        baseline_memory = baseline.baseline_metrics.get('memory_usage_mean', 0)

        memory_comparison = 'baseline_unavailable'
        if baseline_memory > 0:
            memory_ratio = current_memory / baseline_memory
            if memory_ratio > 1.2:
                memory_comparison = 'higher_usage'
            elif memory_ratio < 0.8:
                memory_comparison = 'lower_usage'
            else:
                memory_comparison = 'comparable'

        return {
            'time_comparison': time_comparison,
            'time_ratio': time_ratio if baseline_time > 0 else 0,
            'memory_comparison': memory_comparison,
            'memory_ratio': memory_ratio if baseline_memory > 0 else 0,
            'baseline_date': baseline.established_date.isoformat(),
            'current_performance_level': 'above_baseline' if time_comparison == 'faster' else 'below_baseline' if time_comparison == 'slower' else 'at_baseline'
        }

    def _generate_regression_analysis(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate regression analysis."""
        return {
            'regression_alerts': len(self.regression_alerts),
            'recent_alerts': self.regression_alerts[-5:],  # Last 5 alerts
            'regression_rate': len(self.regression_alerts) / max(len(results), 1),
            'affected_tests': list(set(alert['test_name'] for alert in self.regression_alerts[-10:]))
        }

    def _generate_trend_analysis(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate trend analysis."""
        # Group results by test and analyze trends over time
        test_trends = {}

        test_groups = defaultdict(list)
        for result in results:
            test_groups[result.test_name].append(result)

        for test_name, test_results in test_groups.items():
            # Sort by timestamp
            sorted_results = sorted(test_results, key=lambda r: r.timestamp)

            if len(sorted_results) >= 5:
                # Analyze execution time trend
                execution_times = [r.execution_time for r in sorted_results if r.success]

                if len(execution_times) >= 5:
                    time_trend = self._calculate_trend(execution_times)
                    test_trends[test_name] = {
                        'execution_time_trend': time_trend,
                        'sample_size': len(execution_times),
                        'time_range': {
                            'start': sorted_results[0].timestamp.isoformat(),
                            'end': sorted_results[-1].timestamp.isoformat()
                        }
                    }

        return test_trends

    def _generate_performance_recommendations(self,
                                           performance_summary: Dict[str, Any],
                                           regression_analysis: Dict[str, Any],
                                           trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        # Check success rate
        success_rate = performance_summary.get('success_rate', 1.0)
        if success_rate < 0.8:
            recommendations.append("Low success rate detected - investigate error patterns and improve reliability")

        # Check execution time
        exec_summary = performance_summary.get('execution_time', {})
        if exec_summary:
            mean_time = exec_summary.get('mean', 0)
            std_time = exec_summary.get('std', 0)

            if mean_time > 10:  # More than 10 seconds average
                recommendations.append("High execution times detected - consider performance optimization")

            if std_time > mean_time * 0.5:  # High variability
                recommendations.append("High performance variability - investigate sources of inconsistency")

        # Check resource usage
        resource_summary = performance_summary.get('resource_usage', {})
        if resource_summary:
            memory_max = resource_summary.get('memory', {}).get('max', 0)
            if memory_max > 80:
                recommendations.append("High memory usage detected - consider memory optimization strategies")

        # Check regression alerts
        if regression_analysis.get('regression_alerts', 0) > 0:
            recommendations.append("Performance regressions detected - prioritize regression fixes")

        # Check trends
        for test_name, trend in trend_analysis.items():
            if trend.get('execution_time_trend') == 'increasing':
                recommendations.append(f"Execution time increasing for '{test_name}' - investigate performance degradation")

        if not recommendations:
            recommendations.append("Performance appears stable - continue monitoring")

        return recommendations

    def save_benchmark_results(self) -> None:
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'benchmark_results_{timestamp}.json'

        # Convert results to serializable format
        serializable_results = [
            {
                'benchmark_id': result.benchmark_id,
                'test_name': result.test_name,
                'timestamp': result.timestamp.isoformat(),
                'duration': result.duration,
                'success': result.success,
                'execution_time': result.execution_time,
                'memory_usage_peak': result.memory_usage_peak,
                'cpu_usage_peak': result.cpu_usage_peak,
                'throughput': result.throughput,
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'resource_efficiency': result.resource_efficiency,
                'time_efficiency': result.time_efficiency,
                'cost_efficiency': result.cost_efficiency,
                'scalability_score': result.scalability_score,
                'concurrent_operations': result.concurrent_operations,
                'errors': result.errors
            }
            for result in self.benchmark_results[-100:]  # Last 100 results
        ]

        try:
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save benchmark results: {e}")

    def get_performance_status(self) -> Dict[str, Any]:
        """Get current performance status summary."""
        return {
            'total_benchmarks': len(self.benchmark_results),
            'active_baselines': len(self.performance_baselines),
            'regression_alerts': len(self.regression_alerts),
            'current_benchmark': self.benchmark_in_progress,
            'last_benchmark_time': self.benchmark_results[-1].timestamp.isoformat() if self.benchmark_results else None,
            'performance_trend': self._calculate_overall_performance_trend()
        }

    def _calculate_overall_performance_trend(self) -> str:
        """Calculate overall performance trend across all tests."""
        if len(self.benchmark_results) < 10:
            return 'insufficient_data'

        recent_results = self.benchmark_results[-20:]

        # Group by test and analyze each
        test_trends = []
        test_groups = defaultdict(list)

        for result in recent_results:
            test_groups[result.test_name].append(result)

        for test_name, results in test_groups.items():
            if len(results) >= 3:
                execution_times = [r.execution_time for r in results if r.success]
                if len(execution_times) >= 3:
                    trend = self._calculate_trend(execution_times)
                    test_trends.append(trend)

        if not test_trends:
            return 'insufficient_data'

        # Majority vote for overall trend
        trend_counts = defaultdict(int)
        for trend in test_trends:
            trend_counts[trend] += 1

        return max(trend_counts, key=trend_counts.get)