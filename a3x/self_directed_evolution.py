"""Self-directed evolution capabilities for goal-based autonomous growth in SeedAI."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

from .actions import AgentAction, ActionType, Observation
from .config import AgentConfig
from .capabilities import CapabilityRegistry, Capability
from .capability_metrics import compute_capability_metrics
from .meta_capabilities import MetaCapabilityEngine, SkillProposal
from .transfer_learning import TransferLearningEngine, DomainPattern, CrossDomainMapping
from .autoeval import AutoEvaluator, EvaluationSeed


@dataclass
class SelfAssessment:
    """Represents a self-assessment of the agent's current capabilities."""
    
    id: str
    timestamp: str
    overall_score: float
    capability_scores: Dict[str, float]
    performance_metrics: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    recommendations: List[str]


@dataclass
class StrategicGap:
    """Represents a strategic gap identified in the agent's capabilities."""
    
    id: str
    capability_domain: str
    gap_type: str  # critical, important, nice_to_have
    description: str
    impact_score: float  # 0.0 to 1.0
    urgency_score: float  # 0.0 to 1.0
    priority_score: float  # 0.0 to 1.0 (calculated)
    related_capabilities: List[str]
    suggested_approaches: List[str]
    estimated_effort: float
    created_at: str


@dataclass
class EvolutionGoal:
    """Represents a self-directed evolution goal."""
    
    id: str
    name: str
    description: str
    target_capabilities: List[str]
    success_criteria: List[str]
    timeline: str  # short_term, medium_term, long_term
    priority: str  # low, medium, high, critical
    estimated_duration: float  # in days
    resources_required: List[str]
    dependencies: List[str]
    progress_indicators: List[str]
    completion_criteria: List[str]
    created_at: str
    status: str = "pending"  # pending, in_progress, completed, failed


@dataclass
class EvolutionPlan:
    """Represents a comprehensive evolution plan."""
    
    id: str
    name: str
    description: str
    goals: List[EvolutionGoal]
    timeline_overview: str
    resource_allocation: Dict[str, float]
    risk_assessment: str
    success_metrics: List[str]
    checkpoints: List[str]
    created_at: str
    status: str = "active"  # active, completed, abandoned


class SelfDirectedEvolutionEngine:
    """Engine for self-directed goal-based autonomous growth."""
    
    def __init__(self, config: AgentConfig, auto_evaluator: AutoEvaluator, 
                 meta_engine: MetaCapabilityEngine, transfer_engine: TransferLearningEngine) -> None:
        self.config = config
        self.auto_evaluator = auto_evaluator
        self.meta_engine = meta_engine
        self.transfer_engine = transfer_engine
        self.workspace_root = Path(config.workspace.root).resolve()
        self.evolution_path = self.workspace_root / "seed" / "evolution"
        self.evolution_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing evolution artifacts
        self.assessments_path = self.evolution_path / "assessments"
        self.assessments_path.mkdir(parents=True, exist_ok=True)
        
        self.gaps_path = self.evolution_path / "gaps"
        self.gaps_path.mkdir(parents=True, exist_ok=True)
        
        self.goals_path = self.evolution_path / "goals"
        self.goals_path.mkdir(parents=True, exist_ok=True)
        
        self.plans_path = self.evolution_path / "plans"
        self.plans_path.mkdir(parents=True, exist_ok=True)
    
    def conduct_self_assessment(self) -> SelfAssessment:
        """Conduct a comprehensive self-assessment of current capabilities."""
        # Load current metrics and performance data
        metrics_history = self.auto_evaluator._read_metrics_history()
        current_metrics = {}
        
        # Get latest values for each metric
        for metric_name, values in metrics_history.items():
            if values:
                current_metrics[metric_name] = values[-1]
        
        # Calculate overall capability scores
        capability_scores = self._calculate_capability_scores(metrics_history)
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(capability_scores)
        
        # Analyze performance trends
        performance_metrics = self._analyze_performance_trends(metrics_history)
        
        # Identify opportunities and threats
        opportunities, threats = self._identify_opportunities_threats(metrics_history, capability_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(strengths, weaknesses, opportunities, threats)
        
        # Calculate overall score (weighted average of key metrics)
        overall_score = self._calculate_overall_score(capability_scores, performance_metrics)
        
        assessment = SelfAssessment(
            id=f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_score=overall_score,
            capability_scores=capability_scores,
            performance_metrics=performance_metrics,
            strengths=strengths,
            weaknesses=weaknesses,
            opportunities=opportunities,
            threats=threats,
            recommendations=recommendations
        )
        
        # Save assessment
        self._save_self_assessment(assessment)
        
        return assessment
    
    def _calculate_capability_scores(self, metrics_history: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate scores for each capability based on metrics history."""
        capability_scores = {}
        
        # Group metrics by capability
        capability_metrics = defaultdict(list)
        for metric_name, values in metrics_history.items():
            if "." in metric_name:
                capability, metric = metric_name.rsplit(".", 1)
                if values:
                    capability_metrics[capability].append(values[-1])  # Latest value
        
        # Calculate average score for each capability
        for capability, metrics in capability_metrics.items():
            if metrics:
                # Simple average for now - could be weighted later
                capability_scores[capability] = sum(metrics) / len(metrics)
            else:
                capability_scores[capability] = 0.0
        
        return capability_scores
    
    def _identify_strengths_weaknesses(self, capability_scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses based on capability scores."""
        strengths = []
        weaknesses = []
        
        for capability, score in capability_scores.items():
            if score >= 0.8:  # High performance
                strengths.append(capability)
            elif score <= 0.4:  # Low performance
                weaknesses.append(capability)
        
        return strengths, weaknesses
    
    def _analyze_performance_trends(self, metrics_history: Dict[str, List[float]]) -> Dict[str, float]:
        """Analyze performance trends from metrics history."""
        trends = {}
        
        for metric_name, values in metrics_history.items():
            if len(values) >= 2:
                # Calculate trend as difference between last two values
                trend = values[-1] - values[-2]
                trends[metric_name] = trend
            else:
                trends[metric_name] = 0.0
        
        return trends
    
    def _identify_opportunities_threats(self, metrics_history: Dict[str, List[float]], 
                                      capability_scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Identify opportunities and threats based on trends and scores."""
        opportunities = []
        threats = []
        
        # Look for improving trends
        for capability, score in capability_scores.items():
            # Check if capability is improving
            metric_name = f"{capability}.success_rate"
            if metric_name in metrics_history:
                values = metrics_history[metric_name]
                if len(values) >= 3:
                    # Check if last few values show improvement
                    recent_trend = sum(values[-3:]) / 3 - sum(values[-6:-3]) / 3 if len(values) >= 6 else 0
                    if recent_trend > 0.1:  # Significant improvement
                        opportunities.append(f"{capability}_improving")
                    elif recent_trend < -0.1:  # Significant decline
                        threats.append(f"{capability}_declining")
        
        return opportunities, threats
    
    def _generate_recommendations(self, strengths: List[str], weaknesses: List[str], 
                                opportunities: List[str], threats: List[str]) -> List[str]:
        """Generate recommendations based on SWOT analysis."""
        recommendations = []
        
        # Recommendations for weaknesses
        for weakness in weaknesses[:3]:  # Focus on top weaknesses
            recommendations.append(f"Prioritize improvement of {weakness}")
        
        # Recommendations for opportunities
        for opportunity in opportunities[:2]:
            recommendations.append(f"Leverage opportunity in {opportunity}")
        
        # Recommendations for threats
        for threat in threats[:2]:
            recommendations.append(f"Mitigate threat from {threat}")
        
        # Recommendations for strengths (leverage them)
        for strength in strengths[:2]:
            recommendations.append(f"Leverage strength in {strength} for cross-domain applications")
        
        # General recommendations
        if len(weaknesses) > len(strengths):
            recommendations.append("Focus on capability gap reduction")
        else:
            recommendations.append("Continue expanding current capabilities")
        
        return recommendations
    
    def _calculate_overall_score(self, capability_scores: Dict[str, float], 
                                performance_metrics: Dict[str, float]) -> float:
        """Calculate overall performance score."""
        if not capability_scores:
            return 0.0
        
        # Weighted average - higher weight to success rates
        total_weight = 0.0
        weighted_sum = 0.0
        
        for capability, score in capability_scores.items():
            # Look for success rate metric
            success_metric = f"{capability}.success_rate"
            if success_metric in performance_metrics:
                weight = performance_metrics[success_metric]
            else:
                weight = 1.0
            
            total_weight += weight
            weighted_sum += score * weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return sum(capability_scores.values()) / len(capability_scores)
    
    def identify_strategic_gaps(self, assessment: SelfAssessment) -> List[StrategicGap]:
        """Identify strategic gaps based on self-assessment."""
        gaps = []
        
        # Critical gaps - capabilities with very low scores
        for capability, score in assessment.capability_scores.items():
            if score <= 0.3:  # Very low performance
                gap = StrategicGap(
                    id=f"gap_critical_{capability}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    capability_domain=capability,
                    gap_type="critical",
                    description=f"Critical capability gap in {capability} (score: {score:.2f})",
                    impact_score=0.9,  # High impact
                    urgency_score=0.9,  # High urgency
                    priority_score=0.9,  # High priority
                    related_capabilities=[capability],
                    suggested_approaches=[f"Develop {capability} capability from scratch"],
                    estimated_effort=8.0,  # High effort
                    created_at=datetime.now(timezone.utc).isoformat()
                )
                gaps.append(gap)
        
        # Important gaps - capabilities with moderate scores
        for capability, score in assessment.capability_scores.items():
            if 0.3 < score <= 0.6:  # Moderate performance
                gap = StrategicGap(
                    id=f"gap_important_{capability}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    capability_domain=capability,
                    gap_type="important",
                    description=f"Important capability gap in {capability} (score: {score:.2f})",
                    impact_score=0.7,  # Medium-high impact
                    urgency_score=0.6,  # Medium urgency
                    priority_score=0.65,  # Medium-high priority
                    related_capabilities=[capability],
                    suggested_approaches=[f"Enhance existing {capability} capability"],
                    estimated_effort=5.0,  # Medium effort
                    created_at=datetime.now(timezone.utc).isoformat()
                )
                gaps.append(gap)
        
        # Nice-to-have gaps - emerging opportunities
        for opportunity in assessment.opportunities:
            if "improving" in opportunity:
                capability = opportunity.replace("_improving", "")
                gap = StrategicGap(
                    id=f"gap_nice_to_have_{capability}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    capability_domain=capability,
                    gap_type="nice_to_have",
                    description=f"Opportunity to enhance {capability} further",
                    impact_score=0.5,  # Medium impact
                    urgency_score=0.4,  # Low-medium urgency
                    priority_score=0.45,  # Medium priority
                    related_capabilities=[capability],
                    suggested_approaches=[f"Optimize {capability} for advanced use cases"],
                    estimated_effort=3.0,  # Low-medium effort
                    created_at=datetime.now(timezone.utc).isoformat()
                )
                gaps.append(gap)
        
        # Save gaps
        for gap in gaps:
            self._save_strategic_gap(gap)
        
        return gaps
    
    def prioritize_evolution_goals(self, gaps: List[StrategicGap]) -> List[EvolutionGoal]:
        """Prioritize evolution goals based on strategic gaps."""
        goals = []
        
        # Sort gaps by priority score
        sorted_gaps = sorted(gaps, key=lambda x: x.priority_score, reverse=True)
        
        # Create goals for top gaps
        for gap in sorted_gaps[:5]:  # Focus on top 5 gaps
            goal = self._create_evolution_goal_from_gap(gap)
            if goal:
                goals.append(goal)
        
        # Create cross-cutting goals
        cross_cutting_goals = self._create_cross_cutting_goals()
        goals.extend(cross_cutting_goals)
        
        # Save goals
        for goal in goals:
            self._save_evolution_goal(goal)
        
        return goals
    
    def _create_evolution_goal_from_gap(self, gap: StrategicGap) -> Optional[EvolutionGoal]:
        """Create an evolution goal from a strategic gap."""
        # Determine timeline based on gap type and priority
        if gap.gap_type == "critical":
            timeline = "short_term"
            priority = "critical"
            estimated_duration = 7.0  # 1 week
        elif gap.gap_type == "important":
            timeline = "medium_term"
            priority = "high"
            estimated_duration = 30.0  # 1 month
        else:  # nice_to_have
            timeline = "long_term"
            priority = "medium"
            estimated_duration = 90.0  # 3 months
        
        goal = EvolutionGoal(
            id=f"goal_{gap.capability_domain.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=f"Enhance {gap.capability_domain} Capability",
            description=f"Address strategic gap in {gap.capability_domain}: {gap.description}",
            target_capabilities=gap.related_capabilities,
            success_criteria=[
                f"Achieve {gap.capability_domain} performance score > 0.8",
                f"Reduce failure rate in {gap.capability_domain} by 50%",
                f"Increase success rate in {gap.capability_domain} to > 0.9"
            ],
            timeline=timeline,
            priority=priority,
            estimated_duration=estimated_duration,
            resources_required=["development_time", "computational_resources"],
            dependencies=[],
            progress_indicators=[
                f"{gap.capability_domain}.success_rate",
                f"{gap.capability_domain}.failure_rate",
                f"{gap.capability_domain}.execution_time_avg"
            ],
            completion_criteria=[
                f"{gap.capability_domain}.success_rate > 0.8 for 5 consecutive evaluations",
                f"No critical failures in {gap.capability_domain} for 10 consecutive operations"
            ],
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        return goal
    
    def _create_cross_cutting_goals(self) -> List[EvolutionGoal]:
        """Create cross-cutting evolution goals."""
        goals = []
        
        # Cross-domain integration goal
        integration_goal = EvolutionGoal(
            id=f"goal_cross_domain_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Cross-Domain Capability Integration",
            description="Integrate capabilities across different domains for enhanced synergy",
            target_capabilities=["horiz.integration"],
            success_criteria=[
                "Demonstrate successful transfer of techniques between at least 3 domain pairs",
                "Achieve 20% performance improvement in target domains through cross-domain learning"
            ],
            timeline="medium_term",
            priority="high",
            estimated_duration=60.0,
            resources_required=["development_time", "data_sets"],
            dependencies=[],
            progress_indicators=[
                "transfer.successful_transfers",
                "cross_domain.performance_improvement"
            ],
            completion_criteria=[
                "At least 3 successful cross-domain transfers documented",
                "Positive performance impact measured in target domains"
            ],
            created_at=datetime.now(timezone.utc).isoformat()
        )
        goals.append(integration_goal)
        
        # Self-improvement goal
        self_improvement_goal = EvolutionGoal(
            id=f"goal_self_improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Self-Directed Evolution Enhancement",
            description="Improve the agent's ability to self-direct its own evolution",
            target_capabilities=["meta.evolution"],
            success_criteria=[
                "Reduce time to identify strategic gaps by 30%",
                "Increase accuracy of gap prioritization to > 85%",
                "Achieve fully autonomous goal generation and execution"
            ],
            timeline="long_term",
            priority="medium",
            estimated_duration=180.0,
            resources_required=["development_time", "research_resources"],
            dependencies=["meta.capabilities"],
            progress_indicators=[
                "evolution.gap_identification_time",
                "evolution.prioritization_accuracy",
                "evolution.autonomy_level"
            ],
            completion_criteria=[
                "Fully automated evolution cycle demonstrated",
                "Self-generated goals achieve > 80% success rate"
            ],
            created_at=datetime.now(timezone.utc).isoformat()
        )
        goals.append(self_improvement_goal)
        
        return goals
    
    def create_evolution_plan(self, goals: List[EvolutionGoal]) -> EvolutionPlan:
        """Create a comprehensive evolution plan from prioritized goals."""
        # Sort goals by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_goals = sorted(goals, key=lambda x: priority_order.get(x.priority, 999))
        
        # Create timeline overview
        timeline_overview = self._generate_timeline_overview(sorted_goals)
        
        # Allocate resources
        resource_allocation = self._allocate_resources(sorted_goals)
        
        # Assess risks
        risk_assessment = self._assess_risks(sorted_goals)
        
        # Define success metrics
        success_metrics = self._define_success_metrics(sorted_goals)
        
        # Define checkpoints
        checkpoints = self._define_checkpoints(sorted_goals)
        
        plan = EvolutionPlan(
            id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Self-Directed Evolution Plan",
            description="Comprehensive plan for autonomous capability enhancement",
            goals=sorted_goals,
            timeline_overview=timeline_overview,
            resource_allocation=resource_allocation,
            risk_assessment=risk_assessment,
            success_metrics=success_metrics,
            checkpoints=checkpoints,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        # Save plan
        self._save_evolution_plan(plan)
        
        return plan
    
    def _generate_timeline_overview(self, goals: List[EvolutionGoal]) -> str:
        """Generate timeline overview for the evolution plan."""
        timeline_segments = []
        
        # Group goals by timeline
        short_term_goals = [g for g in goals if g.timeline == "short_term"]
        medium_term_goals = [g for g in goals if g.timeline == "medium_term"]
        long_term_goals = [g for g in goals if g.timeline == "long_term"]
        
        if short_term_goals:
            timeline_segments.append(f"Short-term (0-30 days): {len(short_term_goals)} goals")
        if medium_term_goals:
            timeline_segments.append(f"Medium-term (1-6 months): {len(medium_term_goals)} goals")
        if long_term_goals:
            timeline_segments.append(f"Long-term (6+ months): {len(long_term_goals)} goals")
        
        return "; ".join(timeline_segments) if timeline_segments else "Flexible timeline based on progress"
    
    def _allocate_resources(self, goals: List[EvolutionGoal]) -> Dict[str, float]:
        """Allocate resources across evolution goals."""
        allocation = defaultdict(float)
        
        # Simple proportional allocation based on estimated effort
        total_effort = sum(goal.estimated_duration for goal in goals)
        if total_effort > 0:
            for goal in goals:
                proportion = goal.estimated_duration / total_effort
                for resource in goal.resources_required:
                    allocation[resource] += proportion * 100  # Percentage
        
        return dict(allocation)
    
    def _assess_risks(self, goals: List[EvolutionGoal]) -> str:
        """Assess risks associated with the evolution plan."""
        high_priority_goals = [g for g in goals if g.priority in ["critical", "high"]]
        total_goals = len(goals)
        
        if not total_goals:
            return "No goals defined - minimal risk"
        
        high_priority_ratio = len(high_priority_goals) / total_goals
        
        if high_priority_ratio > 0.6:
            return "High risk: Many critical/high priority goals may strain resources"
        elif high_priority_ratio > 0.3:
            return "Medium risk: Balanced mix of priorities with manageable resource demands"
        else:
            return "Low risk: Mostly medium/low priority goals with gradual progression"
    
    def _define_success_metrics(self, goals: List[EvolutionGoal]) -> List[str]:
        """Define success metrics for the overall evolution plan."""
        metrics = []
        
        # Aggregate metrics from individual goals
        for goal in goals:
            metrics.extend(goal.progress_indicators)
        
        # Add overall plan metrics
        metrics.extend([
            "evolution.plan_completion_rate",
            "evolution.resource_utilization_efficiency",
            "evolution.goal_achievement_rate"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_metrics = []
        for metric in metrics:
            if metric not in seen:
                seen.add(metric)
                unique_metrics.append(metric)
        
        return unique_metrics
    
    def _define_checkpoints(self, goals: List[EvolutionGoal]) -> List[str]:
        """Define checkpoints for monitoring evolution progress."""
        checkpoints = []
        
        # Add periodic review checkpoints
        checkpoints.extend([
            "Monthly capability assessment and gap analysis",
            "Quarterly progress review and plan adjustment",
            "Biannual strategic alignment verification"
        ])
        
        # Add goal-specific checkpoints
        for goal in goals:
            if goal.timeline == "short_term":
                checkpoints.append(f"Weekly review of {goal.name}")
            elif goal.timeline == "medium_term":
                checkpoints.append(f"Biweekly review of {goal.name}")
            else:  # long_term
                checkpoints.append(f"Monthly review of {goal.name}")
        
        return checkpoints
    
    def execute_evolution_plan(self, plan: EvolutionPlan) -> None:
        """Execute the evolution plan by generating seeds for implementation."""
        # For each goal in the plan, generate implementation seeds
        for goal in plan.goals:
            self._generate_seeds_for_goal(goal)
    
    def _generate_seeds_for_goal(self, goal: EvolutionGoal) -> None:
        """Generate implementation seeds for a specific evolution goal."""
        # Create skill proposals for each target capability
        for capability in goal.target_capabilities:
            proposal = SkillProposal(
                id=f"proposal_{capability.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"Enhancement for {capability}",
                description=f"Capability enhancement to address: {goal.description}",
                implementation_plan=f"Implement improvements for {capability} based on evolution goal",
                required_dependencies=[capability],
                estimated_effort=goal.estimated_duration / len(goal.target_capabilities),
                priority=goal.priority,
                rationale=f"Strategic evolution goal: {goal.name}",
                target_domain=capability.split('.')[0] if '.' in capability else "core",
                created_at=datetime.now(timezone.utc).isoformat()
            )
            
            # Create evaluation seed for implementation
            seed = self.meta_engine.create_skill_seed(proposal)
            
            # Add seed to backlog (this would normally be handled by the auto-evaluator)
            # For now, we'll just save it as an example
            self._save_evaluation_seed(seed)
    
    def _save_evaluation_seed(self, seed: EvaluationSeed) -> None:
        """Save an evaluation seed to the backlog."""
        # In a real implementation, this would add the seed to the evaluation backlog
        # For now, we'll just acknowledge it
        pass
    
    def _save_self_assessment(self, assessment: SelfAssessment) -> None:
        """Save a self-assessment to disk."""
        assessment_file = self.assessments_path / f"{assessment.id}.json"
        with assessment_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(assessment), f, ensure_ascii=False, indent=2)
    
    def _save_strategic_gap(self, gap: StrategicGap) -> None:
        """Save a strategic gap to disk."""
        gap_file = self.gaps_path / f"{gap.id}.json"
        with gap_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(gap), f, ensure_ascii=False, indent=2)
    
    def _save_evolution_goal(self, goal: EvolutionGoal) -> None:
        """Save an evolution goal to disk."""
        goal_file = self.goals_path / f"{goal.id}.json"
        with goal_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(goal), f, ensure_ascii=False, indent=2)
    
    def _save_evolution_plan(self, plan: EvolutionPlan) -> None:
        """Save an evolution plan to disk."""
        plan_file = self.plans_path / f"{plan.id}.json"
        with plan_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(plan), f, ensure_ascii=False, indent=2)


# Integration with existing system
def integrate_self_directed_evolution(config: AgentConfig, auto_evaluator: AutoEvaluator, 
                                    meta_engine: MetaCapabilityEngine, 
                                    transfer_engine: TransferLearningEngine) -> EvolutionPlan:
    """Integrate self-directed evolution capabilities into the existing system."""
    # Create self-directed evolution engine
    evolution_engine = SelfDirectedEvolutionEngine(config, auto_evaluator, meta_engine, transfer_engine)
    
    # Conduct self-assessment
    assessment = evolution_engine.conduct_self_assessment()
    
    # Identify strategic gaps
    gaps = evolution_engine.identify_strategic_gaps(assessment)
    
    # Prioritize evolution goals
    goals = evolution_engine.prioritize_evolution_goals(gaps)
    
    # Create evolution plan
    plan = evolution_engine.create_evolution_plan(goals)
    
    # Execute the plan
    evolution_engine.execute_evolution_plan(plan)
    
    return plan


__all__ = [
    "SelfDirectedEvolutionEngine",
    "SelfAssessment",
    "StrategicGap",
    "EvolutionGoal",
    "EvolutionPlan",
    "integrate_self_directed_evolution",
]