#!/usr/bin/env python3
"""
Comprehensive Autonomous System Validation and Demonstration Script

This script provides a complete demonstration of the SeedAI autonomous system's
validation capabilities, including real-time monitoring, evolution detection,
performance benchmarking, and comprehensive reporting.

Usage:
    python validate_autonomous_system.py [options]

Options:
    --config FILE           Configuration file path (default: configs/sample.yaml)
    --duration MINUTES      Duration to run validation (default: 30)
    --monitoring-interval S  Monitoring interval in seconds (default: 5)
    --output-dir DIR        Output directory for reports and visualizations
    --scenario NAME         Validation scenario to run (basic, evolution, performance, safety)
    --visualize             Enable real-time visualization
    --report-only           Generate report only (no live monitoring)
    --verbose               Enable verbose logging
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Optional imports for enhanced visualization
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Import A3X components
from a3x.agent import AgentOrchestrator
from a3x.autonomous_goal_generator import AutonomousGoal
from a3x.config import AgentConfig
from a3x.validation import ValidationEnvironment

# Setup console for beautiful output (with fallback)
if HAS_RICH:
    console = Console()
else:
    class SimpleConsole:
        def print(self, text, **kwargs):
            print(text)
        def input(self, text=""):
            return input(text)
    console = SimpleConsole()


class AutonomousSystemValidator:
    """
    Comprehensive validator and demonstrator for autonomous SeedAI system.

    This class orchestrates the entire validation process, providing real-time
    monitoring, visualization, and comprehensive reporting of autonomous behavior.
    """

    def __init__(self,
                 config_path: str = "configs/sample.yaml",
                 duration_minutes: int = 30,
                 monitoring_interval: float = 5.0,
                 output_dir: str = None,
                 scenario: str = "comprehensive",
                 enable_visualization: bool = True,
                 verbose: bool = False):

        self.config_path = config_path
        self.duration_minutes = duration_minutes
        self.monitoring_interval = monitoring_interval
        self.output_dir = Path(output_dir) if output_dir else Path("validation_reports")
        self.scenario = scenario
        self.enable_visualization = enable_visualization
        self.verbose = verbose

        # Runtime state
        self.is_running = False
        self.start_time = None
        self.end_time = None
        self.current_cycle = 0

        # Components
        self.config = None
        self.agent = None
        self.validation_env = None

        # Data collection
        self.monitoring_data = []
        self.evolution_events = []
        self.performance_metrics = []
        self.behavior_snapshots = []

        # Visualization state
        self.live_display = None
        self.visualization_thread = None

        # Setup logging
        self._setup_logging()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AutonomousValidator")

    def initialize_system(self) -> bool:
        """Initialize the autonomous system and validation environment."""
        try:
            console.print("[bold blue]Initializing Autonomous System...[/bold blue]")

            # Load configuration
            self.config = AgentConfig.load_from_yaml(self.config_path)
            console.print(f"[green]✓[/green] Loaded configuration from {self.config_path}")

            # Initialize agent orchestrator
            self.agent = AgentOrchestrator(self.config)
            console.print("[green]✓[/green] Initialized Agent Orchestrator")

            # Initialize validation environment
            self.validation_env = ValidationEnvironment(self.config, self.agent)
            console.print("[green]✓[/green] Initialized Validation Environment")

            # Setup scenario-specific configurations
            self._configure_scenario()

            return True

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize system: {e}")
            self.logger.error(f"Initialization failed: {e}")
            return False

    def _configure_scenario(self):
        """Configure validation environment for specific scenario."""
        if self.scenario == "basic":
            # Basic autonomous operation validation
            self.duration_minutes = 15
            console.print("[yellow]⚙[/yellow] Configured for basic validation scenario")

        elif self.scenario == "evolution":
            # Evolution tracking scenario
            self.duration_minutes = 45
            console.print("[yellow]⚙[/yellow] Configured for evolution tracking scenario")

        elif self.scenario == "performance":
            # Performance benchmarking scenario
            self.duration_minutes = 20
            console.print("[yellow]⚙[/yellow] Configured for performance benchmarking scenario")

        elif self.scenario == "safety":
            # Safety validation scenario
            self.duration_minutes = 25
            console.print("[yellow]⚙[/yellow] Configured for safety validation scenario")

        else:  # comprehensive
            self.duration_minutes = 30
            console.print("[yellow]⚙[/yellow] Configured for comprehensive validation scenario")

    def start_validation(self):
        """Start the comprehensive validation process."""
        if not self.initialize_system():
            return False

        console.print(f"\n[bold green]🚀 Starting Autonomous System Validation[/bold green]")
        console.print(f"Duration: {self.duration_minutes} minutes")
        console.print(f"Monitoring Interval: {self.monitoring_interval}s")
        console.print(f"Scenario: {self.scenario}")
        console.print(f"Output Directory: {self.output_dir}\n")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            self.is_running = True
            self.start_time = datetime.now()

            # Start comprehensive monitoring
            self.validation_env.start_comprehensive_monitoring(self.monitoring_interval)

            # Start visualization if enabled
            if self.enable_visualization:
                self._start_visualization()

            # Main validation loop
            self._validation_loop()

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow] Validation interrupted by user")
        except Exception as e:
            console.print(f"\n[red]✗[/red] Validation failed: {e}")
            self.logger.error(f"Validation error: {e}")
        finally:
            self._cleanup()

    def _validation_loop(self):
        """Main validation loop."""
        end_time = self.start_time + timedelta(minutes=self.duration_minutes)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            task = progress.add_task(
                f"Running {self.scenario} validation...",
                total=self.duration_minutes * 60
            )

            while datetime.now() < end_time and self.is_running:
                try:
                    # Collect current system state
                    current_data = self._collect_current_data()

                    # Update monitoring data
                    self.monitoring_data.append(current_data)

                    # Check for evolution events
                    evolution_events = self._detect_evolution_events(current_data)
                    self.evolution_events.extend(evolution_events)

                    # Update progress
                    elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
                    progress.update(task, completed=int(elapsed_seconds))

                    # Scenario-specific operations
                    self._run_scenario_operations(current_data)

                    # Brief pause between cycles
                    time.sleep(self.monitoring_interval)

                    self.current_cycle += 1

                except Exception as e:
                    self.logger.error(f"Error in validation cycle: {e}")
                    time.sleep(self.monitoring_interval)

            progress.update(task, completed=self.duration_minutes * 60)

    def _collect_current_data(self) -> Dict[str, Any]:
        """Collect current system state and metrics."""
        current_time = datetime.now()

        data = {
            'timestamp': current_time.isoformat(),
            'cycle': self.current_cycle,
            'elapsed_time': (current_time - self.start_time).total_seconds()
        }

        try:
            # Get validation environment status
            validation_status = self.validation_env.get_validation_status()
            data['validation_status'] = validation_status

            # Get current metrics from autonomous validator
            if hasattr(self.validation_env, 'autonomous_validator') and self.validation_env.autonomous_validator:
                current_metrics = self.validation_env.autonomous_validator.get_current_metrics()
                data['current_metrics'] = {
                    'goal_success_rate': current_metrics.goal_success_rate,
                    'system_stability': current_metrics.system_stability,
                    'adaptation_rate': current_metrics.adaptation_rate,
                    'behavior_diversity': current_metrics.behavior_diversity,
                    'memory_utilization': current_metrics.memory_utilization
                }

                # Get evolution markers
                evolution_markers = self.validation_env.autonomous_validator.get_evolution_markers()
                data['evolution_markers'] = len(evolution_markers)

            # Get behavior insights
            behavior_insights = self.validation_env.behavior_tracker.get_behavior_insights()
            data['behavior_insights'] = behavior_insights

            # Get environment status
            env_status = self.validation_env.controlled_environment.get_environment_status()
            data['environment_status'] = env_status

        except Exception as e:
            self.logger.error(f"Error collecting system data: {e}")
            data['error'] = str(e)

        return data

    def _detect_evolution_events(self, current_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect evolution events from current data."""
        events = []

        try:
            # Check for evolution markers
            if 'evolution_markers' in current_data and current_data['evolution_markers'] > 0:
                events.append({
                    'type': 'evolution_markers_detected',
                    'timestamp': current_data['timestamp'],
                    'count': current_data['evolution_markers'],
                    'description': f"Evolution markers detected: {current_data['evolution_markers']}"
                })

            # Check for behavioral shifts
            if 'behavior_insights' in current_data:
                insights = current_data['behavior_insights']

                if 'evolution_indicators' in insights and insights['evolution_indicators']:
                    for indicator in insights['evolution_indicators']:
                        events.append({
                            'type': 'behavioral_evolution',
                            'timestamp': current_data['timestamp'],
                            'indicator': indicator,
                            'description': f"Behavioral evolution indicator: {indicator}"
                        })

            # Check for adaptation rate increases
            if 'current_metrics' in current_data:
                metrics = current_data['current_metrics']
                if metrics.get('adaptation_rate', 0) > 0.5:  # High adaptation rate
                    events.append({
                        'type': 'high_adaptation',
                        'timestamp': current_data['timestamp'],
                        'rate': metrics['adaptation_rate'],
                        'description': f"High adaptation rate detected: {metrics['adaptation_rate']:.3f}"
                    })

        except Exception as e:
            self.logger.error(f"Error detecting evolution events: {e}")

        return events

    def _run_scenario_operations(self, current_data: Dict[str, Any]):
        """Run scenario-specific operations."""
        if self.scenario == "evolution":
            self._run_evolution_scenario(current_data)
        elif self.scenario == "performance":
            self._run_performance_scenario(current_data)
        elif self.scenario == "safety":
            self._run_safety_scenario(current_data)
        else:  # basic or comprehensive
            self._run_basic_scenario(current_data)

    def _run_basic_scenario(self, current_data: Dict[str, Any]):
        """Run basic autonomous operation validation."""
        # Monitor goal generation patterns
        if 'current_metrics' in current_data:
            metrics = current_data['current_metrics']

            # Log significant events
            if metrics.get('goal_success_rate', 0) > 0.8:
                self.logger.info(f"High success rate: {metrics['goal_success_rate']:.2%}")

            if metrics.get('system_stability', 0) < 0.5:
                self.logger.warning(f"Low system stability: {metrics['system_stability']:.2%}")

    def _run_evolution_scenario(self, current_data: Dict[str, Any]):
        """Run evolution tracking scenario."""
        # Focus on evolution detection and tracking
        evolution_analysis = self.validation_env.analyze_behavior_evolution(current_data)

        if evolution_analysis.get('evolution_detected', False):
            self.logger.info(f"Evolution detected: {evolution_analysis}")

            # Log evolution details
            evolution_score = evolution_analysis.get('evolution_score', 0)
            if evolution_score > 0.3:
                self.logger.info(f"Significant evolution score: {evolution_score:.3f}")

    def _run_performance_scenario(self, current_data: Dict[str, Any]):
        """Run performance benchmarking scenario."""
        # Run performance benchmarks periodically
        if self.current_cycle % 10 == 0:  # Every 50 seconds
            try:
                benchmark_results = self.validation_env.run_autonomous_benchmark(
                    test_name=f"cycle_{self.current_cycle}",
                    test_function=self._sample_performance_test,
                    repetitions=3
                )

                # Log benchmark results
                for result in benchmark_results:
                    self.logger.info(f"Benchmark result: {result}")

            except Exception as e:
                self.logger.error(f"Performance benchmark failed: {e}")

    def _run_safety_scenario(self, current_data: Dict[str, Any]):
        """Run safety validation scenario."""
        # Monitor safety constraints and environment health
        if 'environment_status' in current_data:
            env_status = current_data['environment_status']

            health_score = env_status.get('environment_health_score', 1.0)
            if health_score < 0.7:
                self.logger.warning(f"Low environment health: {health_score:.2%}")

            warnings = env_status.get('safety_summary', {}).get('total_warnings', 0)
            if warnings > 0:
                self.logger.warning(f"Safety warnings: {warnings}")

    def _sample_performance_test(self, **kwargs) -> Dict[str, Any]:
        """Sample performance test function."""
        # Simulate a performance test
        start_time = time.time()

        # Simulate some work
        result = sum(i * i for i in range(1000))

        execution_time = time.time() - start_time

        return {
            'result': result,
            'execution_time': execution_time,
            'test_type': 'cpu_intensive'
        }

    def _start_visualization(self):
        """Start real-time visualization."""
        if not self.enable_visualization or not HAS_RICH:
            return

        # Create initial layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="metrics", size=10),
            Layout(name="evolution", size=8),
            Layout(name="progress", size=3)
        )

        # Start live display
        self.live_display = Live(layout, refresh_per_second=2, screen=True)
        self.live_display.start()

        # Start visualization update thread
        self.visualization_thread = threading.Thread(
            target=self._visualization_loop,
            daemon=True
        )
        self.visualization_thread.start()

    def _visualization_loop(self):
        """Update visualization in real-time."""
        while self.is_running and self.live_display:
            try:
                self._update_visualization()
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                self.logger.error(f"Visualization error: {e}")
                break

    def _update_visualization(self):
        """Update the visualization display."""
        if not self.live_display or not self.monitoring_data:
            return

        # Header panel
        elapsed = datetime.now() - self.start_time
        header_text = f"[bold blue]Autonomous System Validation - Cycle {self.current_cycle}[/bold blue]\n"
        header_text += f"Elapsed: {elapsed.total_seconds():.0f}s | Scenario: {self.scenario}"

        # Metrics panel
        if self.monitoring_data:
            latest_data = self.monitoring_data[-1]
            metrics_table = self._create_metrics_table(latest_data)

            # Evolution panel
            evolution_info = self._create_evolution_info()

            # Progress panel
            progress_text = self._create_progress_info()

            # Update layout
            self.live_display.update(
                Layout(
                    Panel(header_text, title="Status"),
                    Layout(metrics_table, name="metrics"),
                    Layout(evolution_info, name="evolution"),
                    Layout(progress_text, name="progress")
                )
            )

    def _create_metrics_table(self, data: Dict[str, Any]) -> Panel:
        """Create metrics display table."""
        table = Table(title="Current Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if 'current_metrics' in data:
            metrics = data['current_metrics']
            table.add_row("Success Rate", f"{metrics.get('goal_success_rate', 0):.1%}")
            table.add_row("Stability", f"{metrics.get('system_stability', 0):.1%}")
            table.add_row("Adaptation Rate", f"{metrics.get('adaptation_rate', 0):.3f}")
            table.add_row("Memory Usage", f"{metrics.get('memory_utilization', 0):.1%}")

        return Panel(table)

    def _create_evolution_info(self) -> Panel:
        """Create evolution information display."""
        evolution_text = "[bold yellow]Evolution Status[/bold yellow]\n\n"

        if self.evolution_events:
            recent_events = self.evolution_events[-3:]  # Last 3 events
            for event in recent_events:
                evolution_text += f"• {event['description']}\n"

            evolution_text += f"\nTotal Events: {len(self.evolution_events)}"
        else:
            evolution_text += "No evolution events detected yet..."

        return Panel(evolution_text)

    def _create_progress_info(self) -> Panel:
        """Create progress information display."""
        elapsed = datetime.now() - self.start_time
        total_seconds = self.duration_minutes * 60
        progress_pct = min(100, (elapsed.total_seconds() / total_seconds) * 100)

        progress_text = f"[green]Progress: {progress_pct:.1f}%[/green]\n"
        progress_text += f"Est. completion: {datetime.now() + timedelta(seconds=total_seconds - elapsed.total_seconds()):%H:%M:%S}"

        return Panel(progress_text)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        console.print(f"\n[yellow]⚠[/yellow] Received signal {signum}, shutting down...")
        self.is_running = False

    def _cleanup(self):
        """Cleanup resources and generate final reports."""
        console.print("\n[bold blue]Cleaning up and generating reports...[/bold blue]")

        try:
            # Stop monitoring
            if hasattr(self.validation_env, 'stop_comprehensive_monitoring'):
                final_report = self.validation_env.stop_comprehensive_monitoring()
            else:
                final_report = {}

            # Stop visualization
            if self.live_display:
                self.live_display.stop()

            if self.visualization_thread and self.visualization_thread.is_alive():
                self.visualization_thread.join(timeout=5)

            # Generate comprehensive validation report
            self._generate_comprehensive_report(final_report)

            # Save all collected data
            self._save_monitoring_data()

            self.end_time = datetime.now()

            # Show final summary
            self._show_final_summary()

        except Exception as e:
            console.print(f"[red]✗[/red] Error during cleanup: {e}")
            self.logger.error(f"Cleanup error: {e}")

    def _generate_comprehensive_report(self, final_report: Dict[str, Any]):
        """Generate comprehensive validation report."""
        console.print("[yellow]📊[/yellow] Generating comprehensive validation report...")

        # Create report structure
        report = {
            'validation_summary': {
                'scenario': self.scenario,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_minutes': self.duration_minutes,
                'total_cycles': self.current_cycle,
                'evolution_events_detected': len(self.evolution_events),
                'monitoring_interval_seconds': self.monitoring_interval
            },
            'system_metrics': self._summarize_system_metrics(),
            'evolution_analysis': self._summarize_evolution_analysis(),
            'behavior_analysis': self._summarize_behavior_analysis(),
            'performance_analysis': self._summarize_performance_analysis(),
            'safety_analysis': self._summarize_safety_analysis(),
            'component_reports': final_report,
            'recommendations': self._generate_recommendations(),
            'generated_at': datetime.now().isoformat()
        }

        # Save report
        report_file = self.output_dir / f'autonomous_validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            console.print(f"[green]✓[/green] Report saved to {report_file}")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to save report: {e}")

        # Generate visualizations
        if self.enable_visualization:
            self._generate_validation_visualizations()

        return report

    def _summarize_system_metrics(self) -> Dict[str, Any]:
        """Summarize system metrics over the validation period."""
        if not self.monitoring_data:
            return {}

        # Extract metrics from all cycles
        success_rates = []
        stability_scores = []
        adaptation_rates = []
        memory_usage = []

        for data in self.monitoring_data:
            if 'current_metrics' in data:
                metrics = data['current_metrics']
                success_rates.append(metrics.get('goal_success_rate', 0))
                stability_scores.append(metrics.get('system_stability', 0))
                adaptation_rates.append(metrics.get('adaptation_rate', 0))
                memory_usage.append(metrics.get('memory_utilization', 0))

        return {
            'average_success_rate': np.mean(success_rates) if success_rates else 0,
            'average_stability': np.mean(stability_scores) if stability_scores else 0,
            'average_adaptation_rate': np.mean(adaptation_rates) if adaptation_rates else 0,
            'average_memory_usage': np.mean(memory_usage) if memory_usage else 0,
            'success_rate_trend': self._calculate_trend(success_rates),
            'stability_trend': self._calculate_trend(stability_scores),
            'total_data_points': len(self.monitoring_data)
        }

    def _summarize_evolution_analysis(self) -> Dict[str, Any]:
        """Summarize evolution analysis."""
        if not self.evolution_events:
            return {'evolution_detected': False, 'total_events': 0}

        # Categorize evolution events
        event_types = {}
        for event in self.evolution_events:
            event_type = event.get('type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1

        return {
            'evolution_detected': len(self.evolution_events) > 0,
            'total_events': len(self.evolution_events),
            'event_types': event_types,
            'evolution_rate_per_hour': len(self.evolution_events) / (self.duration_minutes / 60),
            'significant_events': [
                event for event in self.evolution_events
                if event.get('rate', 0) > 0.5 or event.get('confidence', 0) > 0.7
            ]
        }

    def _summarize_behavior_analysis(self) -> Dict[str, Any]:
        """Summarize behavior analysis."""
        try:
            behavior_report = self.validation_env.behavior_tracker.generate_behavior_report()
            return behavior_report
        except Exception as e:
            self.logger.error(f"Error generating behavior summary: {e}")
            return {}

    def _summarize_performance_analysis(self) -> Dict[str, Any]:
        """Summarize performance analysis."""
        try:
            perf_report = self.validation_env.performance_benchmark.generate_performance_report()
            return perf_report
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {}

    def _summarize_safety_analysis(self) -> Dict[str, Any]:
        """Summarize safety analysis."""
        try:
            env_status = self.validation_env.controlled_environment.get_environment_status()
            return env_status
        except Exception as e:
            self.logger.error(f"Error generating safety summary: {e}")
            return {}

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Analyze results and generate recommendations
        if self.monitoring_data:
            avg_success = np.mean([
                data.get('current_metrics', {}).get('goal_success_rate', 0)
                for data in self.monitoring_data
                if 'current_metrics' in data
            ])

            if avg_success < 0.5:
                recommendations.append("Low success rate detected - review goal complexity and agent capabilities")
            elif avg_success > 0.9:
                recommendations.append("Excellent success rate - consider increasing goal complexity")

        if len(self.evolution_events) == 0:
            recommendations.append("No evolution detected - consider adjusting evolution detection thresholds")
        elif len(self.evolution_events) > 10:
            recommendations.append("High evolution activity - ensure safety monitoring is adequate")

        # Default recommendation
        if not recommendations:
            recommendations.append("System operating within normal parameters - continue monitoring")

        return recommendations

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 3:
            return 'insufficient_data'

        # Simple linear regression slope
        x = list(range(len(values)))
        if len(x) > 1 and len(values) > 1:
            slope = np.polyfit(x, values, 1)[0]

            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'declining'
            else:
                return 'stable'

        return 'stable'

    def _generate_validation_visualizations(self):
        """Generate validation visualizations."""
        console.print("[yellow]📈[/yellow] Generating visualizations...")

        try:
            # Success rate trend
            self._plot_success_rate_trend()

            # Evolution timeline
            self._plot_evolution_timeline()

            # Performance heatmap
            self._plot_performance_heatmap()

            # Behavior patterns
            self._plot_behavior_patterns()

            console.print(f"[green]✓[/green] Visualizations saved to {self.output_dir}")

        except Exception as e:
            console.print(f"[red]✗[/red] Error generating visualizations: {e}")
            self.logger.error(f"Visualization error: {e}")

    def _plot_success_rate_trend(self):
        """Plot success rate trend over time."""
        if not self.monitoring_data:
            return

        timestamps = []
        success_rates = []

        for data in self.monitoring_data:
            if 'current_metrics' in data:
                timestamps.append(datetime.fromisoformat(data['timestamp']))
                success_rates.append(data['current_metrics'].get('goal_success_rate', 0))

        if not timestamps:
            return

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, success_rates, marker='o', linestyle='-', linewidth=2, markersize=4)
        plt.title(f'Success Rate Trend - {self.scenario.title()} Scenario')
        plt.xlabel('Time')
        plt.ylabel('Success Rate')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = self.output_dir / f'success_rate_trend_{self.scenario}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_evolution_timeline(self):
        """Plot evolution events timeline."""
        if not self.evolution_events:
            return

        plt.figure(figsize=(14, 8))

        # Group events by type
        event_types = {}
        for event in self.evolution_events:
            event_type = event.get('type', 'unknown')
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)

        colors = plt.cm.tab10(np.linspace(0, 1, len(event_types)))

        for i, (event_type, events) in enumerate(event_types.items()):
            timestamps = [datetime.fromisoformat(event['timestamp']) for event in events]
            # Use event rate/confidence as y-value, defaulting to 0.5 if not available
            y_values = [event.get('rate', event.get('confidence', 0.5)) for event in events]

            plt.scatter(timestamps, y_values, c=[colors[i]], label=event_type.replace('_', ' ').title(),
                       s=100, alpha=0.7)

        plt.title(f'Evolution Events Timeline - {self.scenario.title()} Scenario')
        plt.xlabel('Time')
        plt.ylabel('Event Significance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = self.output_dir / f'evolution_timeline_{self.scenario}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_heatmap(self):
        """Create performance metrics heatmap."""
        if not self.monitoring_data:
            return

        # Extract metrics for heatmap
        metrics_data = []
        for data in self.monitoring_data[-50:]:  # Last 50 data points
            if 'current_metrics' in data:
                metrics = data['current_metrics']
                row = [
                    metrics.get('goal_success_rate', 0),
                    metrics.get('system_stability', 0),
                    metrics.get('adaptation_rate', 0),
                    metrics.get('memory_utilization', 0)
                ]
                metrics_data.append(row)

        if not metrics_data:
            return

        plt.figure(figsize=(10, 8))
        if HAS_SEABORN:
            sns.heatmap(metrics_data, annot=True, fmt='.2f', cmap='viridis',
                       xticklabels=['Success', 'Stability', 'Adaptation', 'Memory'],
                       yticklabels=False)
        else:
            # Fallback to matplotlib imshow
            plt.imshow(metrics_data, cmap='viridis', aspect='auto')
            plt.colorbar(label='Performance')
            plt.xticks(range(len(['Success', 'Stability', 'Adaptation', 'Memory'])),
                      ['Success', 'Stability', 'Adaptation', 'Memory'])
            # Add text annotations
            for i in range(len(metrics_data)):
                for j in range(len(metrics_data[i])):
                    plt.text(j, i, f'{metrics_data[i][j]:.2f}',
                           ha='center', va='center', color='white')

        plt.title(f'Performance Metrics Heatmap - {self.scenario.title()} Scenario')
        plt.tight_layout()

        plot_path = self.output_dir / f'performance_heatmap_{self.scenario}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_behavior_patterns(self):
        """Plot behavioral patterns over time."""
        if not self.monitoring_data:
            return

        # Extract behavioral data
        success_rates = []
        adaptation_rates = []
        timestamps = []

        for data in self.monitoring_data:
            if 'current_metrics' in data:
                timestamps.append(datetime.fromisoformat(data['timestamp']))
                success_rates.append(data['current_metrics'].get('goal_success_rate', 0))
                adaptation_rates.append(data['current_metrics'].get('adaptation_rate', 0))

        if not timestamps:
            return

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(timestamps, success_rates, 'b-', linewidth=2, alpha=0.8)
        plt.title('Success Rate Pattern')
        plt.grid(True, alpha=0.3)
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)

        plt.subplot(2, 1, 2)
        plt.plot(timestamps, adaptation_rates, 'r-', linewidth=2, alpha=0.8)
        plt.title('Adaptation Rate Pattern')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time')
        plt.ylabel('Adaptation Rate')
        plt.xticks(rotation=45)

        plt.tight_layout()

        plot_path = self.output_dir / f'behavior_patterns_{self.scenario}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_monitoring_data(self):
        """Save all collected monitoring data."""
        try:
            # Save monitoring data
            data_file = self.output_dir / f'monitoring_data_{self.scenario}.json'
            with open(data_file, 'w') as f:
                json.dump({
                    'monitoring_data': self.monitoring_data,
                    'evolution_events': self.evolution_events,
                    'validation_config': {
                        'scenario': self.scenario,
                        'duration_minutes': self.duration_minutes,
                        'monitoring_interval': self.monitoring_interval
                    }
                }, f, indent=2, default=str)

            # Save evolution events separately
            if self.evolution_events:
                evolution_file = self.output_dir / f'evolution_events_{self.scenario}.json'
                with open(evolution_file, 'w') as f:
                    json.dump(self.evolution_events, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving monitoring data: {e}")

    def _show_final_summary(self):
        """Show final validation summary."""
        console.print("\n[bold green]🎉 Autonomous System Validation Complete![/bold green]")

        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            console.print(f"Duration: {duration.total_seconds():.0f} seconds")
            console.print(f"Total Cycles: {self.current_cycle}")

        # Show key metrics
        if self.monitoring_data:
            final_metrics = self._summarize_system_metrics()
            console.print("\n[bold blue]Final Metrics:[/bold blue]")
            console.print(f"• Average Success Rate: {final_metrics.get('average_success_rate', 0):.1%}")
            console.print(f"• Average Stability: {final_metrics.get('average_stability', 0):.1%}")
            console.print(f"• Evolution Events: {len(self.evolution_events)}")

        # Show recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            console.print("\n[bold yellow]Recommendations:[/bold yellow]")
            for rec in recommendations:
                console.print(f"• {rec}")

        console.print(f"\n[green]✓[/green] Reports and visualizations saved to: {self.output_dir}")


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Autonomous System Validation and Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/sample.yaml',
        help='Configuration file path'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Duration to run validation in minutes'
    )

    parser.add_argument(
        '--monitoring-interval',
        type=float,
        default=5.0,
        help='Monitoring interval in seconds'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for reports and visualizations'
    )

    parser.add_argument(
        '--scenario',
        type=str,
        choices=['basic', 'evolution', 'performance', 'safety', 'comprehensive'],
        default='comprehensive',
        help='Validation scenario to run'
    )

    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable real-time visualization'
    )

    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate report only (no live monitoring)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Create validator instance
    validator = AutonomousSystemValidator(
        config_path=args.config,
        duration_minutes=args.duration,
        monitoring_interval=args.monitoring_interval,
        output_dir=args.output_dir,
        scenario=args.scenario,
        enable_visualization=not args.no_visualize,
        verbose=args.verbose
    )

    # Run validation
    try:
        validator.start_validation()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/yellow] Validation terminated by user")
    except Exception as e:
        console.print(f"\n[red]✗[/red] Validation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())