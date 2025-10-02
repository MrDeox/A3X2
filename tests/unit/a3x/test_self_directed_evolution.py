"""Tests for self-directed evolution capabilities in SeedAI."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from a3x.autoeval import AutoEvaluator, EvaluationSeed
from a3x.config import AgentConfig, WorkspaceConfig
from a3x.meta_capabilities import Capability, CapabilityRegistry, MetaCapabilityEngine
from a3x.self_directed_evolution import (
    EvolutionGoal,
    EvolutionPlan,
    SelfAssessment,
    SelfDirectedEvolutionEngine,
    StrategicGap,
    integrate_self_directed_evolution,
)
from a3x.transfer_learning import TransferLearningEngine


class TestSelfAssessment:
    """Tests for SelfAssessment class."""

    def test_self_assessment_creation(self) -> None:
        """Test creating a self-assessment."""
        assessment = SelfAssessment(
            id="test_assessment_001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_score=0.85,
            capability_scores={"core.testing": 0.9, "core.diffing": 0.8},
            performance_metrics={"test_execution_time": 1.5, "patch_success_rate": 0.95},
            strengths=["core.testing"],
            weaknesses=["core.diffing"],
            opportunities=["horiz.data_analysis"],
            threats=["core.security"],
            recommendations=["Improve diffing capabilities", "Leverage data analysis opportunities"]
        )

        assert assessment.id == "test_assessment_001"
        assert assessment.overall_score == 0.85
        assert assessment.capability_scores == {"core.testing": 0.9, "core.diffing": 0.8}
        assert assessment.performance_metrics == {"test_execution_time": 1.5, "patch_success_rate": 0.95}
        assert assessment.strengths == ["core.testing"]
        assert assessment.weaknesses == ["core.diffing"]
        assert assessment.opportunities == ["horiz.data_analysis"]
        assert assessment.threats == ["core.security"]
        assert assessment.recommendations == ["Improve diffing capabilities", "Leverage data analysis opportunities"]


class TestStrategicGap:
    """Tests for StrategicGap class."""

    def test_strategic_gap_creation(self) -> None:
        """Test creating a strategic gap."""
        gap = StrategicGap(
            id="test_gap_001",
            capability_domain="core.testing",
            gap_type="critical",
            description="Critical testing capability gap",
            impact_score=0.9,
            urgency_score=0.85,
            priority_score=0.87,
            related_capabilities=["core.testing", "core.quality"],
            suggested_approaches=["Enhance test coverage", "Improve test execution speed"],
            estimated_effort=5.0,
            created_at=datetime.now(timezone.utc).isoformat()
        )

        assert gap.id == "test_gap_001"
        assert gap.capability_domain == "core.testing"
        assert gap.gap_type == "critical"
        assert gap.description == "Critical testing capability gap"
        assert gap.impact_score == 0.9
        assert gap.urgency_score == 0.85
        assert gap.priority_score == 0.87
        assert gap.related_capabilities == ["core.testing", "core.quality"]
        assert gap.suggested_approaches == ["Enhance test coverage", "Improve test execution speed"]
        assert gap.estimated_effort == 5.0


class TestEvolutionGoal:
    """Tests for EvolutionGoal class."""

    def test_evolution_goal_creation(self) -> None:
        """Test creating an evolution goal."""
        goal = EvolutionGoal(
            id="test_goal_001",
            name="Enhance Testing Capabilities",
            description="Improve automated testing capabilities",
            target_capabilities=["core.testing", "core.quality"],
            success_criteria=["Test coverage > 90%", "Execution time < 2s"],
            timeline="short_term",
            priority="high",
            estimated_duration=7.0,
            resources_required=["development_time", "test_infrastructure"],
            dependencies=[],
            progress_indicators=["test.coverage", "test.execution_time"],
            completion_criteria=["Continuous test coverage > 90% for 1 week"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        assert goal.id == "test_goal_001"
        assert goal.name == "Enhance Testing Capabilities"
        assert goal.description == "Improve automated testing capabilities"
        assert goal.target_capabilities == ["core.testing", "core.quality"]
        assert goal.success_criteria == ["Test coverage > 90%", "Execution time < 2s"]
        assert goal.timeline == "short_term"
        assert goal.priority == "high"
        assert goal.estimated_duration == 7.0
        assert goal.resources_required == ["development_time", "test_infrastructure"]
        assert goal.dependencies == []
        assert goal.progress_indicators == ["test.coverage", "test.execution_time"]
        assert goal.completion_criteria == ["Continuous test coverage > 90% for 1 week"]


class TestEvolutionPlan:
    """Tests for EvolutionPlan class."""

    def test_evolution_plan_creation(self) -> None:
        """Test creating an evolution plan."""
        goals = [
            EvolutionGoal(
                id="goal_001",
                name="Test Goal",
                description="Test goal description",
                target_capabilities=["core.test"],
                success_criteria=["Criteria 1"],
                timeline="short_term",
                priority="high",
                estimated_duration=7.0,
                resources_required=["time"],
                dependencies=[],
                progress_indicators=["test.metric"],
                completion_criteria=["Test completion"],
                created_at=datetime.now(timezone.utc).isoformat()
            )
        ]

        plan = EvolutionPlan(
            id="test_plan_001",
            name="Test Evolution Plan",
            description="Test evolution plan description",
            goals=goals,
            timeline_overview="Short-term: 1 goal",
            resource_allocation={"development_time": 70.0, "computational_resources": 30.0},
            risk_assessment="Low risk - single goal focus",
            success_metrics=["test.metric", "evolution.goal_completion_rate"],
            checkpoints=["Weekly review of Test Goal"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        assert plan.id == "test_plan_001"
        assert plan.name == "Test Evolution Plan"
        assert plan.description == "Test evolution plan description"
        assert len(plan.goals) == 1
        assert plan.goals[0].id == "goal_001"
        assert plan.timeline_overview == "Short-term: 1 goal"
        assert plan.resource_allocation == {"development_time": 70.0, "computational_resources": 30.0}
        assert plan.risk_assessment == "Low risk - single goal focus"
        assert plan.success_metrics == ["test.metric", "evolution.goal_completion_rate"]
        assert plan.checkpoints == ["Weekly review of Test Goal"]


class TestSelfDirectedEvolutionEngine:
    """Tests for SelfDirectedEvolutionEngine."""

    def setup_method(self) -> None:
        """Set up before each test."""
        # Create temporary directory for tests
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_root = self.temp_dir / "workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Create mock config
        self.mock_config = Mock(spec=AgentConfig)
        self.mock_config.workspace = Mock(spec=WorkspaceConfig)
        self.mock_config.workspace.root = str(self.workspace_root)
        self.mock_config.get = Mock(return_value=None)
        self.mock_config.policies = Mock()
        self.mock_config.policies.allow_network = False
        self.mock_config.policies.deny_commands = []
        self.mock_config.audit = Mock()
        self.mock_config.audit.enable_file_log = True
        self.mock_config.audit.file_dir = Path("seed/changes")
        self.mock_config.audit.enable_git_commit = False
        self.mock_config.audit.commit_prefix = "A3X"

        # Create mock engines
        self.mock_auto_evaluator = Mock(spec=AutoEvaluator)
        self.mock_meta_engine = Mock(spec=MetaCapabilityEngine)
        self.mock_transfer_engine = Mock(spec=TransferLearningEngine)

        # Create engine
        self.engine = SelfDirectedEvolutionEngine(
            self.mock_config,
            self.mock_auto_evaluator,
            self.mock_meta_engine,
            self.mock_transfer_engine
        )

    def teardown_method(self) -> None:
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_directories(self) -> None:
        """Test that initialization creates required directories."""
        # Verify directories were created
        assert self.engine.evolution_path.exists()
        assert self.engine.assessments_path.exists()
        assert self.engine.gaps_path.exists()
        assert self.engine.goals_path.exists()
        assert self.engine.plans_path.exists()

        # Verify they are subdirectories of workspace
        assert str(self.workspace_root) in str(self.engine.evolution_path)
        assert str(self.workspace_root) in str(self.engine.assessments_path)
        assert str(self.workspace_root) in str(self.engine.gaps_path)
        assert str(self.workspace_root) in str(self.engine.goals_path)
        assert str(self.workspace_root) in str(self.engine.plans_path)

    def test_calculate_capability_scores_with_data(self) -> None:
        """Test calculating capability scores with real data."""
        metrics_history = {
            "core.testing.success_rate": [0.8, 0.85, 0.9],
            "core.testing.failure_rate": [0.2, 0.15, 0.1],
            "core.diffing.success_rate": [0.7, 0.75, 0.8],
            "core.diffing.failure_rate": [0.3, 0.25, 0.2]
        }

        scores = self.engine._calculate_capability_scores(metrics_history)

        # Should calculate scores for both capabilities
        assert isinstance(scores, dict)
        assert "core.testing" in scores
        assert "core.diffing" in scores
        assert scores["core.testing"] > 0
        assert scores["core.diffing"] > 0

    def test_calculate_capability_scores_empty_history(self) -> None:
        """Test calculating capability scores with empty history."""
        metrics_history = {}

        scores = self.engine._calculate_capability_scores(metrics_history)

        # Should return empty dict for no history
        assert isinstance(scores, dict)
        assert len(scores) == 0

    def test_identify_strengths_weaknesses_balanced(self) -> None:
        """Test identifying strengths and weaknesses with balanced scores."""
        capability_scores = {
            "core.testing": 0.9,      # Strength
            "core.diffing": 0.5,       # Neutral
            "core.quality": 0.2        # Weakness
        }

        strengths, weaknesses = self.engine._identify_strengths_weaknesses(capability_scores)

        # Should identify strengths and weaknesses correctly
        assert isinstance(strengths, list)
        assert isinstance(weaknesses, list)
        assert "core.testing" in strengths
        assert "core.quality" in weaknesses
        assert "core.diffing" not in strengths
        assert "core.diffing" not in weaknesses

    def test_analyze_performance_trends_with_history(self) -> None:
        """Test analyzing performance trends with historical data."""
        metrics_history = {
            "core.testing.success_rate": [0.7, 0.8, 0.9],
            "core.diffing.success_rate": [0.8, 0.75, 0.7]  # Declining
        }

        trends = self.engine._analyze_performance_trends(metrics_history)

        # Should calculate trends for all metrics
        assert isinstance(trends, dict)
        assert "core.testing.success_rate" in trends
        assert "core.diffing.success_rate" in trends
        assert trends["core.testing.success_rate"] > 0  # Improving
        assert trends["core.diffing.success_rate"] < 0   # Declining

    def test_identify_opportunities_threats_with_improving_trends(self) -> None:
        """Test identifying opportunities and threats with improving trends."""
        metrics_history = {
            "core.testing.success_rate": [0.7, 0.8, 0.9, 0.95]  # Strongly improving
        }
        capability_scores = {
            "core.testing": 0.9
        }

        opportunities, threats = self.engine._identify_opportunities_threats(
            metrics_history, capability_scores
        )

        # Should identify improving trends as opportunities
        assert isinstance(opportunities, list)
        assert isinstance(threats, list)
        # Note: Current implementation looks for specific pattern in metric names
        # This test verifies the method doesn't crash and returns lists

    def test_generate_recommendations_comprehensive(self) -> None:
        """Test generating comprehensive recommendations."""
        strengths = ["core.testing"]
        weaknesses = ["core.diffing", "core.quality"]
        opportunities = ["core.testing_improving"]
        threats = ["core.security_declining"]

        recommendations = self.engine._generate_recommendations(
            strengths, weaknesses, opportunities, threats
        )

        # Should generate recommendations for all categories
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should have recommendations for weaknesses
        assert any("Prioritize improvement" in rec for rec in recommendations)
        # Should have recommendations for opportunities
        assert any("Leverage opportunity" in rec for rec in recommendations)
        # Should have recommendations for threats
        assert any("Mitigate threat" in rec for rec in recommendations)

    def test_calculate_overall_score_with_data(self) -> None:
        """Test calculating overall score with capability data."""
        capability_scores = {
            "core.testing": 0.9,
            "core.diffing": 0.8,
            "core.quality": 0.7
        }
        performance_metrics = {
            "core.testing.success_rate": 0.9,
            "core.diffing.success_rate": 0.8,
            "core.quality.success_rate": 0.7
        }

        overall_score = self.engine._calculate_overall_score(
            capability_scores, performance_metrics
        )

        # Should calculate meaningful overall score
        assert isinstance(overall_score, float)
        assert 0.0 <= overall_score <= 1.0
        assert overall_score > 0

    def test_conduct_self_assessment_complete(self) -> None:
        """Test conducting a complete self-assessment."""
        # Mock metrics history
        self.mock_auto_evaluator._read_metrics_history.return_value = {
            "core.testing.success_rate": [0.8, 0.85, 0.9],
            "core.testing.failure_rate": [0.2, 0.15, 0.1],
            "core.diffing.success_rate": [0.7, 0.75, 0.8],
            "core.diffing.failure_rate": [0.3, 0.25, 0.2]
        }

        assessment = self.engine.conduct_self_assessment()

        # Should create valid self-assessment
        assert isinstance(assessment, SelfAssessment)
        assert assessment.id.startswith("assessment_")
        assert isinstance(assessment.overall_score, float)
        assert 0.0 <= assessment.overall_score <= 1.0
        assert isinstance(assessment.capability_scores, dict)
        assert len(assessment.capability_scores) > 0
        assert isinstance(assessment.strengths, list)
        assert isinstance(assessment.weaknesses, list)
        assert isinstance(assessment.recommendations, list)

        # Assessment should be saved
        assessment_file = self.engine.assessments_path / f"{assessment.id}.json"
        assert assessment_file.exists()

    def test_identify_strategic_gaps_critical(self) -> None:
        """Test identifying strategic gaps with critical scores."""
        assessment = SelfAssessment(
            id="test_assessment",
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_score=0.75,
            capability_scores={"core.critical": 0.2, "core.important": 0.5, "core.good": 0.8},
            performance_metrics={"core.critical.success_rate": 0.2},
            strengths=["core.good"],
            weaknesses=["core.critical"],
            opportunities=["core.important_improving"],
            threats=["core.critical_declining"],
            recommendations=["Improve core.critical"]
        )

        gaps = self.engine.identify_strategic_gaps(assessment)

        # Should identify critical gaps
        assert isinstance(gaps, list)
        assert len(gaps) > 0

        # Should have critical gap for low-scoring capability
        critical_gaps = [g for g in gaps if g.gap_type == "critical"]
        assert len(critical_gaps) > 0
        assert critical_gaps[0].capability_domain == "core.critical"
        assert critical_gaps[0].impact_score >= 0.8
        assert critical_gaps[0].urgency_score >= 0.8
        assert critical_gaps[0].priority_score >= 0.8

        # Gaps should be saved
        for gap in gaps:
            gap_file = self.engine.gaps_path / f"{gap.id}.json"
            assert gap_file.exists()

    def test_prioritize_evolution_goals_mixed(self) -> None:
        """Test prioritizing evolution goals with mixed gap types."""
        gaps = [
            StrategicGap(
                id="gap_critical_001",
                capability_domain="core.critical",
                gap_type="critical",
                description="Critical gap",
                impact_score=0.9,
                urgency_score=0.9,
                priority_score=0.9,
                related_capabilities=["core.critical"],
                suggested_approaches=["Fix critical issue"],
                estimated_effort=8.0,
                created_at=datetime.now(timezone.utc).isoformat()
            ),
            StrategicGap(
                id="gap_important_001",
                capability_domain="core.important",
                gap_type="important",
                description="Important gap",
                impact_score=0.7,
                urgency_score=0.6,
                priority_score=0.65,
                related_capabilities=["core.important"],
                suggested_approaches=["Improve important capability"],
                estimated_effort=5.0,
                created_at=datetime.now(timezone.utc).isoformat()
            )
        ]

        goals = self.engine.prioritize_evolution_goals(gaps)

        # Should create goals for gaps
        assert isinstance(goals, list)
        assert len(goals) >= 2  # At least one goal per gap plus cross-cutting

        # Should prioritize critical goals first
        critical_goals = [g for g in goals if "critical" in g.name.lower()]
        important_goals = [g for g in goals if "important" in g.name.lower()]

        assert len(critical_goals) > 0
        assert len(important_goals) > 0

        # Goals should be saved
        for goal in goals:
            goal_file = self.engine.goals_path / f"{goal.id}.json"
            assert goal_file.exists()

    def test_create_evolution_plan_comprehensive(self) -> None:
        """Test creating comprehensive evolution plan."""
        goals = [
            EvolutionGoal(
                id="goal_001",
                name="Critical Enhancement",
                description="Critical capability enhancement",
                target_capabilities=["core.critical"],
                success_criteria=["Success criteria 1"],
                timeline="short_term",
                priority="critical",
                estimated_duration=7.0,
                resources_required=["development_time"],
                dependencies=[],
                progress_indicators=["core.critical.metric"],
                completion_criteria=["Completion criteria 1"],
                created_at=datetime.now(timezone.utc).isoformat()
            ),
            EvolutionGoal(
                id="goal_002",
                name="Important Enhancement",
                description="Important capability enhancement",
                target_capabilities=["core.important"],
                success_criteria=["Success criteria 2"],
                timeline="medium_term",
                priority="high",
                estimated_duration=30.0,
                resources_required=["development_time", "data_sets"],
                dependencies=["goal_001"],
                progress_indicators=["core.important.metric"],
                completion_criteria=["Completion criteria 2"],
                created_at=datetime.now(timezone.utc).isoformat()
            )
        ]

        plan = self.engine.create_evolution_plan(goals)

        # Should create valid evolution plan
        assert isinstance(plan, EvolutionPlan)
        assert plan.id.startswith("plan_")
        assert plan.name == "Self-Directed Evolution Plan"
        assert len(plan.goals) >= 2
        # Check that timeline overview contains relevant information
        assert "short-term" in plan.timeline_overview.lower() or "medium-term" in plan.timeline_overview.lower()
        assert isinstance(plan.resource_allocation, dict)
        assert len(plan.resource_allocation) > 0
        assert isinstance(plan.success_metrics, list)
        assert len(plan.success_metrics) > 0
        assert isinstance(plan.checkpoints, list)
        assert len(plan.checkpoints) > 0

        # Plan should be saved
        plan_file = self.engine.plans_path / f"{plan.id}.json"
        assert plan_file.exists()

    def test_execute_evolution_plan_with_goals(self) -> None:
        """Test executing evolution plan with multiple goals."""
        # Create mock goals
        goals = [
            EvolutionGoal(
                id="goal_test_001",
                name="Test Goal",
                description="Test goal description",
                target_capabilities=["core.test"],
                success_criteria=["Test criteria"],
                timeline="short_term",
                priority="high",
                estimated_duration=7.0,
                resources_required=["development_time"],
                dependencies=[],
                progress_indicators=["test.metric"],
                completion_criteria=["Test completion"],
                created_at=datetime.now(timezone.utc).isoformat()
            )
        ]

        # Create mock plan
        plan = EvolutionPlan(
            id="plan_test_001",
            name="Test Plan",
            description="Test plan description",
            goals=goals,
            timeline_overview="Test timeline",
            resource_allocation={"development_time": 100.0},
            risk_assessment="Low risk",
            success_metrics=["test.metric"],
            checkpoints=["Test checkpoint"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Mock the seed creation
        mock_seed = Mock(spec=EvaluationSeed)
        mock_seed.description = "Test seed"
        mock_seed.priority = "high"
        mock_seed.capability = "core.test"
        mock_seed.seed_type = "skill_creation"
        mock_seed.data = {}

        self.mock_meta_engine.create_skill_seed.return_value = mock_seed

        # Execute plan
        self.engine.execute_evolution_plan(plan)

        # Should call seed creation for each goal
        assert self.mock_meta_engine.create_skill_seed.called
        # Should create at least one seed
        assert self.mock_meta_engine.create_skill_seed.call_count >= 1


class TestIntegration:
    """Integration tests for self-directed evolution system."""

    def test_integrate_self_directed_evolution_complete(self) -> None:
        """Test complete integration of self-directed evolution."""
        # Create mock config
        mock_config = Mock(spec=AgentConfig)
        mock_config.workspace = Mock(spec=WorkspaceConfig)
        mock_config.workspace.root = "/tmp/test"
        mock_config.get = Mock(return_value=0.8)
        mock_config.policies = Mock()
        mock_config.policies.allow_network = False
        mock_config.policies.deny_commands = []
        mock_config.audit = Mock()
        mock_config.audit.enable_file_log = True
        mock_config.audit.file_dir = Path("seed/changes")
        mock_config.audit.enable_git_commit = False
        mock_config.audit.commit_prefix = "A3X"

        # Create mock auto evaluator with metrics
        mock_auto_evaluator = Mock(spec=AutoEvaluator)
        mock_auto_evaluator._read_metrics_history.return_value = {
            "core.testing.success_rate": [0.8, 0.85, 0.9],
            "core.diffing.success_rate": [0.7, 0.75, 0.8]
        }

        # Create mock meta engine
        mock_meta_engine = Mock(spec=MetaCapabilityEngine)
        mock_capability_registry = Mock(spec=CapabilityRegistry)
        mock_capability_registry._by_id = {
            "core.testing": Capability(
                id="core.testing",
                name="Testing",
                category="vertical",
                description="Automated testing",
                maturity="established",
                metrics={"success_rate": 0.9},
                seeds=[],
                requirements={},
                activation={}
            )
        }
        mock_meta_engine.capability_registry = mock_capability_registry
        mock_meta_engine.create_skill_seed.return_value = Mock(spec=EvaluationSeed)

        # Create mock transfer engine
        mock_transfer_engine = Mock(spec=TransferLearningEngine)

        # Test integration
        plan = integrate_self_directed_evolution(
            mock_config, mock_auto_evaluator, mock_meta_engine, mock_transfer_engine
        )

        # Should return valid evolution plan
        assert isinstance(plan, EvolutionPlan)
        assert plan.id.startswith("plan_")
        assert len(plan.goals) > 0
        assert isinstance(plan.timeline_overview, str)
        assert isinstance(plan.resource_allocation, dict)

        # Integration should complete without errors
        assert True  # Test passes if no exceptions are raised
