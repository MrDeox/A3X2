"""
Advanced behavior tracking and analysis for autonomous systems.

This module provides sophisticated tracking and analysis of autonomous behavior
patterns, including statistical analysis, pattern recognition, and behavioral
modeling capabilities.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .autonomous_validator import AutonomousBehaviorSnapshot, ValidationMetrics


@dataclass
class BehaviorPattern:
    """Represents a detected behavioral pattern."""

    pattern_id: str
    pattern_type: str  # 'temporal', 'frequency', 'state_transition', 'goal_oriented'
    confidence: float
    description: str
    start_time: datetime
    end_time: datetime
    characteristics: Dict[str, Any] = field(default_factory=dict)
    frequency: int = 1
    evolution_score: float = 0.0


@dataclass
class BehaviorCluster:
    """Represents a cluster of similar behaviors."""

    cluster_id: str
    centroid: List[float]
    behaviors: List[BehaviorPattern]
    cluster_size: int
    stability_score: float
    dominant_patterns: List[str] = field(default_factory=list)


@dataclass
class BehaviorSequence:
    """Represents a sequence of behaviors over time."""

    sequence_id: str
    behaviors: List[BehaviorPattern]
    start_time: datetime
    end_time: datetime
    transition_matrix: List[List[float]] = field(default_factory=list)
    complexity_score: float = 0.0
    predictability_score: float = 0.0


class AdvancedBehaviorTracker:
    """
    Advanced tracking and analysis of autonomous behavior patterns.

    Provides sophisticated analysis including pattern recognition,
    behavioral clustering, sequence analysis, and evolution tracking.
    """

    def __init__(self, config, max_patterns: int = 1000, max_sequences: int = 500):
        self.config = config
        self.max_patterns = max_patterns
        self.max_sequences = max_sequences

        # Setup storage
        self._setup_tracking_environment()

        # Pattern storage
        self.detected_patterns: List[BehaviorPattern] = []
        self.behavior_sequences: List[BehaviorSequence] = []
        self.behavior_clusters: List[BehaviorCluster] = []

        # Analysis state
        self.pattern_history: Dict[str, List[BehaviorPattern]] = defaultdict(list)
        self.sequence_history: Dict[str, List[BehaviorSequence]] = defaultdict(list)

        # Statistical tracking
        self.behavioral_statistics = defaultdict(list)
        self.transition_counts = defaultdict(lambda: defaultdict(int))

        # Real-time analysis
        self.current_sequence: Optional[BehaviorSequence] = None
        self.pattern_embeddings: Dict[str, List[float]] = {}

    def _setup_tracking_environment(self) -> None:
        """Setup tracking environment and storage."""
        self.tracking_root = self.config.workspace_root / "a3x" / "validation" / "behavior_tracking"
        self.tracking_root.mkdir(parents=True, exist_ok=True)

        self.patterns_dir = self.tracking_root / "patterns"
        self.sequences_dir = self.tracking_root / "sequences"
        self.analysis_dir = self.tracking_root / "analysis"

        for directory in [self.patterns_dir, self.sequences_dir, self.analysis_dir]:
            directory.mkdir(exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger("behavior_tracker")
        self.logger.setLevel(logging.INFO)

    def track_behavior_snapshot(self, snapshot: AutonomousBehaviorSnapshot) -> None:
        """Track and analyze behavior from snapshot."""
        # Extract behavioral features from snapshot
        features = self._extract_behavioral_features(snapshot)

        # Update pattern tracking
        self._update_pattern_tracking(snapshot, features)

        # Update sequence tracking
        self._update_sequence_tracking(snapshot, features)

        # Update statistical tracking
        self._update_statistical_tracking(snapshot, features)

        # Perform real-time analysis
        self._perform_real_time_analysis(snapshot, features)

    def _extract_behavioral_features(self, snapshot: AutonomousBehaviorSnapshot) -> Dict[str, float]:
        """Extract behavioral features from snapshot for analysis."""
        features = {}

        # Goal-related features
        features['goal_count'] = len(snapshot.active_goals)
        features['goal_generation_rate'] = snapshot.goal_generation_rate

        if snapshot.active_goals:
            complexities = [goal.estimated_impact for goal in snapshot.active_goals]
            features['avg_goal_complexity'] = statistics.mean(complexities)
            features['goal_complexity_variance'] = statistics.variance(complexities) if len(complexities) > 1 else 0

        # Performance features
        if snapshot.response_times:
            features['avg_response_time'] = statistics.mean(snapshot.response_times)
            features['response_time_variance'] = statistics.variance(snapshot.response_times) if len(snapshot.response_times) > 1 else 0

        if snapshot.success_rates:
            features['avg_success_rate'] = statistics.mean(snapshot.success_rates)
            features['success_rate_variance'] = statistics.variance(snapshot.success_rates) if len(snapshot.success_rates) > 1 else 0

        # System state features
        features['memory_usage'] = snapshot.memory_usage
        features['cpu_usage'] = snapshot.cpu_usage
        features['active_threads'] = snapshot.active_threads

        # Error features
        features['error_count'] = len(snapshot.error_patterns)

        return features

    def _update_pattern_tracking(self, snapshot: AutonomousBehaviorSnapshot, features: Dict[str, float]) -> None:
        """Update pattern tracking with new snapshot."""
        # Detect temporal patterns
        temporal_patterns = self._detect_temporal_patterns(snapshot, features)

        # Detect frequency patterns
        frequency_patterns = self._detect_frequency_patterns(snapshot, features)

        # Detect state transition patterns
        transition_patterns = self._detect_state_transition_patterns(snapshot, features)

        # Combine all detected patterns
        all_patterns = temporal_patterns + frequency_patterns + transition_patterns

        for pattern in all_patterns:
            self.detected_patterns.append(pattern)
            self.pattern_history[pattern.pattern_type].append(pattern)

        # Limit stored patterns
        if len(self.detected_patterns) > self.max_patterns:
            self.detected_patterns = self.detected_patterns[-self.max_patterns:]

    def _detect_temporal_patterns(self, snapshot: AutonomousBehaviorSnapshot, features: Dict[str, float]) -> List[BehaviorPattern]:
        """Detect temporal behavioral patterns."""
        patterns = []

        # Analyze response time patterns
        if snapshot.response_times:
            response_times = snapshot.response_times

            # Detect consistent timing patterns
            if len(response_times) >= 5:
                mean_time = statistics.mean(response_times)
                std_time = statistics.stdev(response_times) if len(response_times) > 1 else 0

                # Low variance suggests consistent timing pattern
                if std_time / mean_time < 0.2 and mean_time > 0:
                    pattern = BehaviorPattern(
                        pattern_id=f"temp_consistent_{snapshot.timestamp.isoformat()}",
                        pattern_type="temporal",
                        confidence=1.0 - (std_time / mean_time),
                        description=f"Consistent response timing pattern (mean: {mean_time:.3f}s, std: {std_time:.3f}s)",
                        start_time=snapshot.timestamp,
                        end_time=snapshot.timestamp,
                        characteristics={
                            'mean_response_time': mean_time,
                            'response_time_std': std_time,
                            'coefficient_of_variation': std_time / mean_time if mean_time > 0 else 0
                        }
                    )
                    patterns.append(pattern)

        return patterns

    def _detect_frequency_patterns(self, snapshot: AutonomousBehaviorSnapshot, features: Dict[str, float]) -> List[BehaviorPattern]:
        """Detect frequency-based behavioral patterns."""
        patterns = []

        # Analyze goal generation frequency
        goal_rate = features.get('goal_generation_rate', 0)
        if goal_rate > 0:
            # Detect periodic goal generation
            if hasattr(self, '_last_goal_rate'):
                rate_change = abs(goal_rate - self._last_goal_rate) / max(self._last_goal_rate, 0.001)

                if rate_change < 0.1:  # Stable frequency
                    pattern = BehaviorPattern(
                        pattern_id=f"freq_stable_{snapshot.timestamp.isoformat()}",
                        pattern_type="frequency",
                        confidence=1.0 - rate_change,
                        description=f"Stable goal generation frequency ({goal_rate:.3f} goals/sec)",
                        start_time=snapshot.timestamp,
                        end_time=snapshot.timestamp,
                        characteristics={
                            'goal_generation_rate': goal_rate,
                            'rate_stability': 1.0 - rate_change
                        }
                    )
                    patterns.append(pattern)

            self._last_goal_rate = goal_rate

        return patterns

    def _detect_state_transition_patterns(self, snapshot: AutonomousBehaviorSnapshot, features: Dict[str, float]) -> List[BehaviorPattern]:
        """Detect state transition patterns."""
        patterns = []

        # Analyze system state transitions
        current_state = self._get_system_state_vector(features)

        if hasattr(self, '_last_state'):
            # Calculate state transition characteristics
            state_change = np.linalg.norm(np.array(current_state) - np.array(self._last_state))
            state_change_rate = state_change / max(np.linalg.norm(np.array(self._last_state)), 0.001)

            # Detect stable state patterns
            if state_change_rate < 0.1:
                pattern = BehaviorPattern(
                    pattern_id=f"state_stable_{snapshot.timestamp.isoformat()}",
                    pattern_type="state_transition",
                    confidence=1.0 - state_change_rate,
                    description=f"Stable system state pattern (change rate: {state_change_rate:.3f})",
                    start_time=snapshot.timestamp,
                    end_time=snapshot.timestamp,
                    characteristics={
                        'state_change_rate': state_change_rate,
                        'state_vector': current_state
                    }
                )
                patterns.append(pattern)

        self._last_state = current_state

        return patterns

    def _get_system_state_vector(self, features: Dict[str, float]) -> List[float]:
        """Convert features to state vector for analysis."""
        # Normalize key features into a state vector
        state_features = [
            features.get('memory_usage', 0) / 100,  # Normalize to 0-1
            features.get('cpu_usage', 0) / 100,     # Normalize to 0-1
            features.get('avg_success_rate', 0),     # Already 0-1
            features.get('goal_count', 0) / 10,      # Normalize assuming max 10 goals
            features.get('error_count', 0) / 20      # Normalize assuming max 20 errors
        ]
        return state_features

    def _update_sequence_tracking(self, snapshot: AutonomousBehaviorSnapshot, features: Dict[str, float]) -> None:
        """Update sequence tracking with new snapshot."""
        # Create or update current sequence
        if self.current_sequence is None:
            self.current_sequence = BehaviorSequence(
                sequence_id=f"seq_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}",
                behaviors=[],
                start_time=snapshot.timestamp,
                end_time=snapshot.timestamp
            )

        # Add current snapshot as a sequence point
        sequence_point = self._create_sequence_point(snapshot, features)
        self.current_sequence.behaviors.append(sequence_point)

        # Check if sequence should be completed
        if self._should_complete_sequence(snapshot, features):
            self._complete_current_sequence()

    def _create_sequence_point(self, snapshot: AutonomousBehaviorSnapshot, features: Dict[str, float]) -> BehaviorPattern:
        """Create a sequence point from snapshot."""
        return BehaviorPattern(
            pattern_id=f"point_{snapshot.timestamp.isoformat()}",
            pattern_type="sequence_point",
            confidence=1.0,
            description=f"Sequence point at {snapshot.timestamp.isoformat()}",
            start_time=snapshot.timestamp,
            end_time=snapshot.timestamp,
            characteristics=features.copy()
        )

    def _should_complete_sequence(self, snapshot: AutonomousBehaviorSnapshot, features: Dict[str, float]) -> bool:
        """Determine if current sequence should be completed."""
        if self.current_sequence is None:
            return False

        # Complete sequence if it's been running too long
        duration = (snapshot.timestamp - self.current_sequence.start_time).total_seconds()
        if duration > 300:  # 5 minutes
            return True

        # Complete sequence if significant behavioral shift detected
        if len(self.current_sequence.behaviors) > 1:
            current_behavior = features.get('avg_success_rate', 0.5)
            previous_behavior = self.current_sequence.behaviors[-2].characteristics.get('avg_success_rate', 0.5)

            if abs(current_behavior - previous_behavior) > 0.3:  # Significant change
                return True

        return False

    def _complete_current_sequence(self) -> None:
        """Complete current sequence and store it."""
        if self.current_sequence and len(self.current_sequence.behaviors) > 1:
            self.current_sequence.end_time = self.current_sequence.behaviors[-1].start_time

            # Calculate sequence characteristics
            self._calculate_sequence_characteristics(self.current_sequence)

            # Store sequence
            self.behavior_sequences.append(self.current_sequence)
            self.sequence_history[self.current_sequence.sequence_id].append(self.current_sequence)

            # Limit stored sequences
            if len(self.behavior_sequences) > self.max_sequences:
                self.behavior_sequences = self.behavior_sequences[-self.max_sequences:]

        # Start new sequence
        self.current_sequence = None

    def _calculate_sequence_characteristics(self, sequence: BehaviorSequence) -> None:
        """Calculate characteristics of a behavior sequence."""
        if len(sequence.behaviors) < 2:
            return

        # Extract feature vectors from behaviors
        feature_matrix = []
        for behavior in sequence.behaviors:
            features = behavior.characteristics
            feature_vector = [
                features.get('avg_success_rate', 0.5),
                features.get('avg_response_time', 0),
                features.get('memory_usage', 0),
                features.get('goal_count', 0),
                features.get('error_count', 0)
            ]
            feature_matrix.append(feature_vector)

        feature_matrix = np.array(feature_matrix)

        # Calculate complexity score
        if feature_matrix.shape[0] > 1:
            # Use variance as complexity measure
            variances = np.var(feature_matrix, axis=0)
            sequence.complexity_score = np.mean(variances)

        # Calculate predictability score
        if feature_matrix.shape[0] > 2:
            # Use autocorrelation as predictability measure
            try:
                autocorr = np.corrcoef(feature_matrix[:-1].T, feature_matrix[1:].T)
                sequence.predictability_score = np.mean(np.abs(autocorr))
            except:
                sequence.predictability_score = 0.0

    def _update_statistical_tracking(self, snapshot: AutonomousBehaviorSnapshot, features: Dict[str, float]) -> None:
        """Update statistical tracking of behavioral features."""
        # Track each feature over time
        for feature_name, feature_value in features.items():
            self.behavioral_statistics[feature_name].append(feature_value)

            # Limit history size
            max_history = 1000
            if len(self.behavioral_statistics[feature_name]) > max_history:
                self.behavioral_statistics[feature_name] = self.behavioral_statistics[feature_name][-max_history:]

    def _perform_real_time_analysis(self, snapshot: AutonomousBehaviorSnapshot, features: Dict[str, float]) -> None:
        """Perform real-time analysis of behavior patterns."""
        # Update transition matrix for state transitions
        current_state = tuple(round(f, 2) for f in self._get_system_state_vector(features))

        if hasattr(self, '_last_state_tuple'):
            self.transition_counts[self._last_state_tuple][current_state] += 1

        self._last_state_tuple = current_state

        # Detect emerging patterns in real-time
        emerging_patterns = self._detect_emerging_patterns(features)
        if emerging_patterns:
            for pattern in emerging_patterns:
                self.logger.info(f"Emerging pattern detected: {pattern.description}")

    def _detect_emerging_patterns(self, features: Dict[str, float]) -> List[BehaviorPattern]:
        """Detect emerging behavioral patterns in real-time."""
        patterns = []

        # Check for sudden changes in key metrics
        for feature_name in ['avg_success_rate', 'avg_response_time', 'memory_usage']:
            if feature_name in features:
                current_value = features[feature_name]

                # Get recent history for this feature
                history = self.behavioral_statistics.get(feature_name, [])
                if len(history) >= 10:
                    recent_values = history[-5:]
                    older_values = history[-10:-5]

                    if len(recent_values) >= 5 and len(older_values) >= 5:
                        recent_mean = statistics.mean(recent_values)
                        older_mean = statistics.mean(older_values)

                        if older_mean > 0:
                            change_ratio = abs(recent_mean - older_mean) / older_mean

                            # Significant sudden change
                            if change_ratio > 0.5:
                                pattern = BehaviorPattern(
                                    pattern_id=f"emerging_{feature_name}_{snapshot.timestamp.isoformat()}",
                                    pattern_type="emerging",
                                    confidence=min(change_ratio, 1.0),
                                    description=f"Sudden change in {feature_name}: {change_ratio:.2%} increase",
                                    start_time=snapshot.timestamp,
                                    end_time=snapshot.timestamp,
                                    characteristics={
                                        'feature_name': feature_name,
                                        'change_ratio': change_ratio,
                                        'current_value': current_value,
                                        'previous_mean': older_mean
                                    }
                                )
                                patterns.append(pattern)

        return patterns

    def analyze_behavioral_clusters(self) -> List[BehaviorCluster]:
        """Analyze and cluster similar behavioral patterns."""
        if len(self.detected_patterns) < 10:
            return []

        # Extract feature vectors for clustering
        feature_vectors = []
        pattern_indices = []

        for i, pattern in enumerate(self.detected_patterns[-200:]):  # Use recent patterns
            features = pattern.characteristics
            if features:
                # Create feature vector from pattern characteristics
                feature_vector = [
                    features.get('mean_response_time', 0),
                    features.get('coefficient_of_variation', 0),
                    features.get('goal_generation_rate', 0),
                    features.get('state_change_rate', 0),
                    len(features)  # Number of characteristics as complexity measure
                ]
                feature_vectors.append(feature_vector)
                pattern_indices.append(i)

        if len(feature_vectors) < 5:
            return []

        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_vectors)

        # Determine optimal number of clusters
        max_clusters = min(8, len(feature_vectors) // 3)
        if max_clusters < 2:
            return []

        best_n_clusters = self._find_optimal_clusters(normalized_features, max_clusters)

        if best_n_clusters < 2:
            return []

        # Perform clustering
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized_features)

        # Create clusters
        clusters = []
        for cluster_id in range(best_n_clusters):
            cluster_indices = [pattern_indices[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_patterns = [self.detected_patterns[i] for i in cluster_indices]

            if cluster_patterns:
                cluster = BehaviorCluster(
                    cluster_id=f"cluster_{cluster_id}",
                    centroid=kmeans.cluster_centers_[cluster_id].tolist(),
                    behaviors=cluster_patterns,
                    cluster_size=len(cluster_patterns),
                    stability_score=self._calculate_cluster_stability(cluster_patterns)
                )
                clusters.append(cluster)

        self.behavior_clusters = clusters
        return clusters

    def _find_optimal_clusters(self, features: np.ndarray, max_clusters: int) -> int:
        """Find optimal number of clusters using silhouette score."""
        best_score = -1
        best_n = 2

        for n_clusters in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)

                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
                    score = silhouette_score(features, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n = n_clusters
            except:
                continue

        return best_n

    def _calculate_cluster_stability(self, patterns: List[BehaviorPattern]) -> float:
        """Calculate stability score for a cluster of patterns."""
        if len(patterns) < 2:
            return 0.0

        # Calculate average confidence as stability measure
        confidences = [pattern.confidence for pattern in patterns]
        return statistics.mean(confidences)

    def analyze_behavioral_evolution(self) -> Dict[str, Any]:
        """Analyze evolution of behavioral patterns over time."""
        if len(self.detected_patterns) < 20:
            return {'evolution_detected': False}

        # Analyze pattern evolution
        recent_patterns = self.detected_patterns[-50:]
        older_patterns = self.detected_patterns[-100:-50] if len(self.detected_patterns) >= 100 else self.detected_patterns[:50]

        # Compare pattern characteristics
        evolution_metrics = {}

        # Pattern type distribution evolution
        recent_types = defaultdict(int)
        older_types = defaultdict(int)

        for pattern in recent_patterns:
            recent_types[pattern.pattern_type] += 1

        for pattern in older_patterns:
            older_types[pattern.pattern_type] += 1

        evolution_metrics['pattern_type_evolution'] = {
            'recent_distribution': dict(recent_types),
            'older_distribution': dict(older_types)
        }

        # Confidence evolution
        recent_confidences = [p.confidence for p in recent_patterns]
        older_confidences = [p.confidence for p in older_patterns]

        evolution_metrics['confidence_evolution'] = {
            'recent_avg_confidence': statistics.mean(recent_confidences) if recent_confidences else 0,
            'older_avg_confidence': statistics.mean(older_confidences) if older_confidences else 0,
            'confidence_improvement': (
                (statistics.mean(recent_confidences) - statistics.mean(older_confidences))
                if recent_confidences and older_confidences else 0
            )
        }

        # Complexity evolution
        recent_complexities = [len(p.characteristics) for p in recent_patterns]
        older_complexities = [len(p.characteristics) for p in older_patterns]

        evolution_metrics['complexity_evolution'] = {
            'recent_avg_complexity': statistics.mean(recent_complexities) if recent_complexities else 0,
            'older_avg_complexity': statistics.mean(older_complexities) if older_complexities else 0,
            'complexity_increase': (
                (statistics.mean(recent_complexities) - statistics.mean(older_complexities))
                if recent_complexities and older_complexities else 0
            )
        }

        # Determine if evolution is significant
        confidence_improvement = evolution_metrics['confidence_evolution']['confidence_improvement']
        complexity_increase = evolution_metrics['complexity_evolution']['complexity_increase']

        evolution_detected = confidence_improvement > 0.1 or complexity_increase > 0.2

        return {
            'evolution_detected': evolution_detected,
            'evolution_metrics': evolution_metrics,
            'evolution_score': max(confidence_improvement, complexity_increase / 2),
            'significant_changes': [
                key for key, value in [
                    ('confidence_improvement', confidence_improvement),
                    ('complexity_increase', complexity_increase)
                ] if value > 0.1
            ]
        }

    def generate_behavior_report(self) -> Dict[str, Any]:
        """Generate comprehensive behavior analysis report."""
        # Analyze clusters
        clusters = self.analyze_behavioral_clusters()

        # Analyze evolution
        evolution_analysis = self.analyze_behavioral_evolution()

        # Generate statistical summary
        statistical_summary = self._generate_statistical_summary()

        # Generate pattern summary
        pattern_summary = self._generate_pattern_summary()

        report = {
            'tracking_summary': {
                'total_patterns_tracked': len(self.detected_patterns),
                'total_sequences_tracked': len(self.behavior_sequences),
                'total_clusters_identified': len(clusters),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'clusters': [
                {
                    'cluster_id': cluster.cluster_id,
                    'size': cluster.cluster_size,
                    'stability_score': cluster.stability_score,
                    'centroid': cluster.centroid,
                    'dominant_pattern_types': self._get_dominant_pattern_types(cluster.behaviors)
                }
                for cluster in clusters
            ],
            'evolution_analysis': evolution_analysis,
            'statistical_summary': statistical_summary,
            'pattern_summary': pattern_summary,
            'recent_activity': {
                'patterns_last_hour': len([p for p in self.detected_patterns[-60:] if p.start_time > datetime.now() - timedelta(hours=1)]),
                'sequences_last_hour': len([s for s in self.behavior_sequences[-20:] if s.start_time > datetime.now() - timedelta(hours=1)]),
                'active_clusters': len([c for c in clusters if c.stability_score > 0.7])
            }
        }

        return report

    def _generate_statistical_summary(self) -> Dict[str, Any]:
        """Generate statistical summary of behavioral features."""
        summary = {}

        for feature_name, values in self.behavioral_statistics.items():
            if len(values) >= 5:
                summary[feature_name] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'trend': self._calculate_feature_trend(values)
                }

        return summary

    def _calculate_feature_trend(self, values: List[float]) -> str:
        """Calculate trend for a feature series."""
        if len(values) < 5:
            return 'insufficient_data'

        recent = values[-3:]
        older = values[:3]

        if len(recent) >= 3 and len(older) >= 3:
            recent_mean = statistics.mean(recent)
            older_mean = statistics.mean(older)

            if older_mean > 0:
                change_ratio = (recent_mean - older_mean) / older_mean

                if change_ratio > 0.1:
                    return 'increasing'
                elif change_ratio < -0.1:
                    return 'decreasing'
                else:
                    return 'stable'

        return 'stable'

    def _generate_pattern_summary(self) -> Dict[str, Any]:
        """Generate summary of detected patterns."""
        if not self.detected_patterns:
            return {}

        # Group patterns by type
        pattern_types = defaultdict(list)
        for pattern in self.detected_patterns:
            pattern_types[pattern.pattern_type].append(pattern)

        summary = {}
        for pattern_type, patterns in pattern_types.items():
            if patterns:
                confidences = [p.confidence for p in patterns]
                summary[pattern_type] = {
                    'count': len(patterns),
                    'avg_confidence': statistics.mean(confidences),
                    'recent_frequency': len([p for p in patterns[-10:] if (datetime.now() - p.start_time).total_seconds() < 300]),
                    'evolution_score': statistics.mean([p.evolution_score for p in patterns]) if patterns else 0
                }

        return summary

    def _get_dominant_pattern_types(self, patterns: List[BehaviorPattern]) -> List[str]:
        """Get dominant pattern types in a cluster."""
        if not patterns:
            return []

        type_counts = defaultdict(int)
        for pattern in patterns:
            type_counts[pattern.pattern_type] += 1

        # Return top 3 most common types
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        return [pattern_type for pattern_type, count in sorted_types[:3]]

    def save_behavior_data(self) -> None:
        """Save behavior tracking data to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save patterns
        patterns_file = self.patterns_dir / f'patterns_{timestamp}.json'
        patterns_data = [
            {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type,
                'confidence': pattern.confidence,
                'description': pattern.description,
                'start_time': pattern.start_time.isoformat(),
                'end_time': pattern.end_time.isoformat(),
                'characteristics': pattern.characteristics,
                'frequency': pattern.frequency,
                'evolution_score': pattern.evolution_score
            }
            for pattern in self.detected_patterns[-100:]  # Save last 100 patterns
        ]

        try:
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save patterns data: {e}")

        # Save sequences
        sequences_file = self.sequences_dir / f'sequences_{timestamp}.json'
        sequences_data = [
            {
                'sequence_id': sequence.sequence_id,
                'start_time': sequence.start_time.isoformat(),
                'end_time': sequence.end_time.isoformat(),
                'complexity_score': sequence.complexity_score,
                'predictability_score': sequence.predictability_score,
                'behavior_count': len(sequence.behaviors)
            }
            for sequence in self.behavior_sequences[-50:]  # Save last 50 sequences
        ]

        try:
            with open(sequences_file, 'w') as f:
                json.dump(sequences_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save sequences data: {e}")

    def get_behavior_insights(self) -> Dict[str, Any]:
        """Get insights about current behavioral patterns."""
        insights = {
            'current_patterns': len(self.detected_patterns),
            'active_sequences': 1 if self.current_sequence else 0,
            'dominant_behavior_types': self._get_dominant_behavior_types(),
            'behavioral_stability': self._calculate_overall_behavioral_stability(),
            'evolution_indicators': self._get_evolution_indicators()
        }

        return insights

    def _get_dominant_behavior_types(self) -> List[str]:
        """Get currently dominant behavior types."""
        if not self.detected_patterns:
            return []

        recent_patterns = self.detected_patterns[-20:]

        type_counts = defaultdict(int)
        for pattern in recent_patterns:
            type_counts[pattern.pattern_type] += pattern.confidence  # Weight by confidence

        # Return top types by weighted count
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        return [pattern_type for pattern_type, _ in sorted_types[:3]]

    def _calculate_overall_behavioral_stability(self) -> float:
        """Calculate overall behavioral stability."""
        if not self.detected_patterns:
            return 0.0

        recent_patterns = self.detected_patterns[-50:]

        if not recent_patterns:
            return 0.0

        # Use average confidence as stability measure
        confidences = [p.confidence for p in recent_patterns]
        return statistics.mean(confidences)

    def _get_evolution_indicators(self) -> List[str]:
        """Get indicators of behavioral evolution."""
        indicators = []

        if len(self.detected_patterns) >= 20:
            recent_patterns = self.detected_patterns[-10:]
            older_patterns = self.detected_patterns[-20:-10]

            recent_confidence = statistics.mean([p.confidence for p in recent_patterns])
            older_confidence = statistics.mean([p.confidence for p in older_patterns])

            if recent_confidence > older_confidence + 0.1:
                indicators.append('increasing_confidence')

            # Check for new pattern types
            recent_types = set(p.pattern_type for p in recent_patterns)
            older_types = set(p.pattern_type for p in older_patterns)

            new_types = recent_types - older_types
            if new_types:
                indicators.append(f'new_pattern_types_{len(new_types)}')

        return indicators