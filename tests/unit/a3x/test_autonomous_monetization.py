"""Tests for autonomous monetization capabilities in SeedAI."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from a3x.autoeval import AutoEvaluator
from a3x.autonomous_monetization import (
    AutonomousMonetizationEngine,
    AutonomousRevenueStream,
    CommercialSolution,
    MarketOpportunity,
    integrate_autonomous_monetization,
)
from a3x.config import AgentConfig, WorkspaceConfig
from a3x.meta_capabilities import MetaCapabilityEngine
from a3x.self_directed_evolution import SelfDirectedEvolutionEngine
from a3x.transfer_learning import TransferLearningEngine


class TestMarketOpportunity:
    """Tests for MarketOpportunity class."""

    def test_market_opportunity_creation(self) -> None:
        """Test creating a market opportunity."""
        opportunity = MarketOpportunity(
            id="opp_test_001",
            name="Test Opportunity",
            description="Test market opportunity description",
            domain="technology.software_development",
            market_size="large",
            competition_level="medium",
            entry_barrier="medium",
            revenue_potential="high",
            alignment_score=0.85,
            viability_score=0.75,
            estimated_roi=2.5,
            required_investments=["development_time", "computational_resources"],
            competitive_advantages=["Unique capability", "Autonomous operation"],
            risks=["High competition", "Market adoption uncertainty"],
            timelines={
                "development": "3-6 months",
                "market_entry": "6-12 months",
                "revenue_generation": "12-18 months"
            },
            success_metrics=["customer_acquisition_rate", "monthly_recurring_revenue"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        assert opportunity.id == "opp_test_001"
        assert opportunity.name == "Test Opportunity"
        assert opportunity.description == "Test market opportunity description"
        assert opportunity.domain == "technology.software_development"
        assert opportunity.market_size == "large"
        assert opportunity.competition_level == "medium"
        assert opportunity.entry_barrier == "medium"
        assert opportunity.revenue_potential == "high"
        assert opportunity.alignment_score == 0.85
        assert opportunity.viability_score == 0.75
        assert opportunity.estimated_roi == 2.5
        assert opportunity.required_investments == ["development_time", "computational_resources"]
        assert opportunity.competitive_advantages == ["Unique capability", "Autonomous operation"]
        assert opportunity.risks == ["High competition", "Market adoption uncertainty"]
        assert "development" in opportunity.timelines
        assert "customer_acquisition_rate" in opportunity.success_metrics


class TestCommercialSolution:
    """Tests for CommercialSolution class."""

    def test_commercial_solution_creation(self) -> None:
        """Test creating a commercial solution."""
        solution = CommercialSolution(
            id="sol_test_001",
            name="Test Commercial Solution",
            description="Test commercial solution description",
            target_market="technology.software_development",
            value_proposition="Value proposition for developers",
            target_customers=["Enterprises", "SMEs"],
            pricing_model="subscription",
            revenue_streams=["Monthly subscriptions", "Enterprise licensing"],
            implementation_plan="Implementation plan details",
            required_capabilities=["core.diffing", "core.testing"],
            estimated_development_time=90.0,
            estimated_cost=50000.0,
            expected_revenue={"year_1": 100000.0, "year_2": 250000.0, "year_3": 500000.0},
            success_probability=0.85,
            dependencies=[],
            milestones=["Market research", "Development", "Launch"],
            kpis=["Customer acquisition", "Revenue growth"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        assert solution.id == "sol_test_001"
        assert solution.name == "Test Commercial Solution"
        assert solution.description == "Test commercial solution description"
        assert solution.target_market == "technology.software_development"
        assert solution.value_proposition == "Value proposition for developers"
        assert solution.target_customers == ["Enterprises", "SMEs"]
        assert solution.pricing_model == "subscription"
        assert solution.revenue_streams == ["Monthly subscriptions", "Enterprise licensing"]
        assert solution.estimated_development_time == 90.0
        assert solution.estimated_cost == 50000.0
        assert "year_1" in solution.expected_revenue
        assert solution.success_probability == 0.85


class TestAutonomousRevenueStream:
    """Tests for AutonomousRevenueStream class."""

    def test_autonomous_revenue_stream_creation(self) -> None:
        """Test creating an autonomous revenue stream."""
        stream = AutonomousRevenueStream(
            id="revenue_test_001",
            name="Test Revenue Stream",
            description="Test revenue stream description",
            stream_type="subscription",
            customer_segment="Enterprises",
            pricing_structure="monthly_subscription",
            automation_level="full",
            monthly_recurring_revenue=10000.0,
            annual_recurring_revenue=120000.0,
            customer_acquisition_cost=500.0,
            customer_lifetime_value=5000.0,
            churn_rate=0.05,
            scalability_factor=0.95,
            maintenance_requirements=["monitoring", "updates"],
            revenue_growth_projection={"month_1": 10000.0, "month_2": 11500.0},
            operational_costs={"cloud": 500.0, "support": 200.0},
            profit_margin=0.85,
            risk_assessment="Low risk with stable enterprise market",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        assert stream.id == "revenue_test_001"
        assert stream.name == "Test Revenue Stream"
        assert stream.description == "Test revenue stream description"
        assert stream.stream_type == "subscription"
        assert stream.customer_segment == "Enterprises"
        assert stream.pricing_structure == "monthly_subscription"
        assert stream.automation_level == "full"
        assert stream.monthly_recurring_revenue == 10000.0
        assert stream.annual_recurring_revenue == 120000.0
        assert stream.customer_acquisition_cost == 500.0
        assert stream.customer_lifetime_value == 5000.0
        assert stream.churn_rate == 0.05
        assert stream.scalability_factor == 0.95
        assert stream.profit_margin == 0.85
        assert "monitoring" in stream.maintenance_requirements
        assert "month_1" in stream.revenue_growth_projection
        assert "cloud" in stream.operational_costs
        assert stream.risk_assessment == "Low risk with stable enterprise market"


class TestAutonomousMonetizationEngine:
    """Tests for AutonomousMonetizationEngine."""

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
        self.mock_evolution_engine = Mock(spec=SelfDirectedEvolutionEngine)

        # Create engine
        self.engine = AutonomousMonetizationEngine(
            self.mock_config,
            self.mock_auto_evaluator,
            self.mock_meta_engine,
            self.mock_transfer_engine,
            self.mock_evolution_engine
        )

    def teardown_method(self) -> None:
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_directories(self) -> None:
        """Test that initialization creates required directories."""
        # Verify directories were created
        assert self.engine.monetization_path.exists()
        assert self.engine.opportunities_path.exists()
        assert self.engine.solutions_path.exists()
        assert self.engine.revenue_path.exists()
        assert self.engine.analytics_path.exists()

        # Verify they are subdirectories of workspace
        assert str(self.workspace_root) in str(self.engine.monetization_path)
        assert str(self.workspace_root) in str(self.engine.opportunities_path)
        assert str(self.workspace_root) in str(self.engine.solutions_path)
        assert str(self.workspace_root) in str(self.engine.revenue_path)
        assert str(self.workspace_root) in str(self.engine.analytics_path)

    def test_analyze_current_capabilities_with_data(self) -> None:
        """Test analyzing current capabilities with real data."""
        # Mock metrics history
        self.mock_auto_evaluator._read_metrics_history.return_value = {
            "core.testing.success_rate": [0.8, 0.85, 0.9],
            "core.diffing.success_rate": [0.7, 0.75, 0.8],
            "core.testing.failure_rate": [0.2, 0.15, 0.1],
            "core.diffing.failure_rate": [0.3, 0.25, 0.2]
        }

        capabilities = self.engine._analyze_current_capabilities()

        # Should calculate scores for capabilities
        assert isinstance(capabilities, dict)
        assert "core.testing" in capabilities
        assert "core.diffing" in capabilities
        assert capabilities["core.testing"] > 0
        assert capabilities["core.diffing"] > 0

    def test_analyze_current_capabilities_empty_history(self) -> None:
        """Test analyzing current capabilities with empty history."""
        # Mock empty metrics history
        self.mock_auto_evaluator._read_metrics_history.return_value = {}

        capabilities = self.engine._analyze_current_capabilities()

        # Should return empty dict for no history
        assert isinstance(capabilities, dict)
        assert len(capabilities) == 0

    def test_analyze_market_trends_comprehensive(self) -> None:
        """Test analyzing market trends comprehensively."""
        trends = self.engine._analyze_market_trends()

        # Should return market trends structure
        assert isinstance(trends, dict)
        assert "technology" in trends
        assert "software_development" in trends["technology"]
        assert "data_science" in trends["technology"]
        assert "cybersecurity" in trends["technology"]
        assert "healthcare" in trends
        assert "finance" in trends

        # Should have pain points and market characteristics
        sw_dev_trends = trends["technology"]["software_development"]
        assert "pain_points" in sw_dev_trends
        assert "market_size" in sw_dev_trends
        assert "competition_level" in sw_dev_trends
        assert "entry_barrier" in sw_dev_trends

        assert len(sw_dev_trends["pain_points"]) > 0
        assert sw_dev_trends["market_size"] in ["small", "medium", "large", "massive"]
        assert sw_dev_trends["competition_level"] in ["low", "medium", "high", "intense"]
        assert sw_dev_trends["entry_barrier"] in ["low", "medium", "high", "massive"]

    def test_match_capabilities_to_markets_with_matches(self) -> None:
        """Test matching capabilities to markets with valid matches."""
        capabilities = {
            "core.diffing": 0.9,
            "core.testing": 0.85,
            "horiz.python": 0.8
        }

        market_trends = {
            "technology": {
                "software_development": {
                    "pain_points": ["Slow development cycles", "High bug rates"],
                    "market_size": "massive",
                    "competition_level": "high",
                    "entry_barrier": "medium"
                }
            }
        }

        matches = self.engine._match_capabilities_to_markets(capabilities, market_trends)

        # Should find matches for technology domains
        assert isinstance(matches, list)
        assert len(matches) > 0

        # Matches should have required attributes
        for match in matches:
            assert "capability" in match
            assert "capability_score" in match
            assert "domain" in match
            assert "subdomain" in match
            assert "market_info" in match
            assert "match_score" in match
            assert match["match_score"] >= 0.6  # Only strong matches

    def test_calculate_match_score_various_scenarios(self) -> None:
        """Test calculating match scores for various scenarios."""
        # High capability, favorable market
        score1 = self.engine._calculate_match_score(0.9, {"market_size": "large", "competition_level": "low"})
        assert isinstance(score1, float)
        assert 0.0 <= score1 <= 1.0
        assert score1 > 0.7  # Should be high

        # Low capability, challenging market
        score2 = self.engine._calculate_match_score(0.2, {"market_size": "small", "competition_level": "intense"})
        assert isinstance(score2, float)
        assert 0.0 <= score2 <= 1.0
        assert score2 < 0.5  # Should be low

        # Medium capability, balanced market
        score3 = self.engine._calculate_match_score(0.5, {"market_size": "medium", "competition_level": "medium"})
        assert isinstance(score3, float)
        assert 0.0 <= score3 <= 1.0
        assert 0.3 <= score3 <= 0.7  # Should be moderate

    def test_create_opportunity_from_match_valid(self) -> None:
        """Test creating opportunity from valid match."""
        match = {
            "capability": "core.testing",
            "capability_score": 0.85,
            "domain": "technology",
            "subdomain": "software_development",
            "market_info": {
                "pain_points": ["Slow development cycles", "High bug rates"],
                "market_size": "massive",
                "competition_level": "high",
                "entry_barrier": "medium"
            },
            "match_score": 0.75
        }

        opportunity = self.engine._create_opportunity_from_match(match)

        # Should create valid opportunity
        assert isinstance(opportunity, MarketOpportunity)
        assert opportunity.id.startswith("opp_")
        assert "Testing" in opportunity.name
        assert "Software Development" in opportunity.name
        assert opportunity.domain == "technology.software_development"
        assert opportunity.market_size == "massive"
        assert opportunity.competition_level == "high"
        assert opportunity.entry_barrier == "medium"
        assert opportunity.alignment_score == 0.75  # match score (not capability score)
        assert "development_time" in opportunity.required_investments
        assert len(opportunity.competitive_advantages) > 0
        assert len(opportunity.risks) > 0

    def test_estimate_revenue_potential_with_parameters(self) -> None:
        """Test estimating revenue potential with various parameters."""
        # Massive market, low competition
        potential1 = self.engine._estimate_revenue_potential("massive", "low")
        assert potential1 in ["massive", "high", "medium", "low"]
        assert potential1 in ["massive", "high"]  # Should be high

        # Small market, high competition
        potential2 = self.engine._estimate_revenue_potential("small", "high")
        assert potential2 in ["massive", "high", "medium", "low"]
        assert potential2 in ["low", "medium"]  # Should be low/medium

        # Medium market, medium competition
        potential3 = self.engine._estimate_revenue_potential("medium", "medium")
        assert potential3 in ["massive", "high", "medium", "low"]
        assert potential3 in ["low", "medium"]  # Should be low/medium

    def test_estimate_viability_with_factors(self) -> None:
        """Test estimating viability with different factors."""
        # High alignment, low barriers
        viability1 = self.engine._estimate_viability(0.9, "low")
        assert isinstance(viability1, float)
        assert 0.0 <= viability1 <= 1.0
        assert viability1 > 0.8  # Should be high

        # Low alignment, high barriers
        viability2 = self.engine._estimate_viability(0.2, "high")
        assert isinstance(viability2, float)
        assert 0.0 <= viability2 <= 1.0
        assert viability2 < 0.5  # Should be low

        # Medium alignment, medium barriers
        viability3 = self.engine._estimate_viability(0.5, "medium")
        assert isinstance(viability3, float)
        assert 0.0 <= viability3 <= 1.0
        assert 0.3 <= viability3 <= 0.7  # Should be moderate

    def test_estimate_roi_with_factors(self) -> None:
        """Test estimating ROI with different factors."""
        # Large market, low competition
        roi1 = self.engine._estimate_roi(0.8, "large", "low")
        assert isinstance(roi1, float)
        assert roi1 >= 0.0
        assert roi1 <= 10.0  # Capped at 10x
        assert roi1 > 2.0  # Should be decent ROI

        # Small market, high competition
        roi2 = self.engine._estimate_roi(0.3, "small", "high")
        assert isinstance(roi2, float)
        assert roi2 >= 0.0
        assert roi2 <= 10.0  # Capped at 10x
        assert roi2 < 3.0  # Should be lower ROI

        # Medium market, medium competition
        roi3 = self.engine._estimate_roi(0.6, "medium", "medium")
        assert isinstance(roi3, float)
        assert roi3 >= 0.0
        assert roi3 <= 10.0  # Capped at 10x
        assert roi3 >= 0.5  # Should be moderate ROI

    def test_determine_target_customers_with_domains(self) -> None:
        """Test determining target customers for different domains."""
        # Technology domain
        customers1 = self.engine._determine_target_customers("technology", "software_development")
        assert isinstance(customers1, list)
        assert len(customers1) > 0
        assert "Enterprises" in customers1
        assert "SMEs" in customers1

        # Healthcare domain
        customers2 = self.engine._determine_target_customers("healthcare", "medical_data_analysis")
        assert isinstance(customers2, list)
        assert len(customers2) > 0
        assert "Hospitals" in customers2

        # Unknown domain
        customers3 = self.engine._determine_target_customers("unknown", "unknown")
        assert isinstance(customers3, list)
        assert len(customers3) > 0
        assert "Enterprises" in customers3
        assert "SMEs" in customers3

    def test_choose_pricing_model_with_opportunities(self) -> None:
        """Test choosing pricing model for different opportunities."""
        # Technology domain
        model1 = self.engine._choose_pricing_model(MagicMock(domain="technology.software_development"))
        assert model1 in ["subscription", "transactional", "service", "freemium"]
        assert model1 == "subscription"  # Should default to subscription for tech

        # Healthcare domain
        model2 = self.engine._choose_pricing_model(MagicMock(domain="healthcare.medical_data_analysis"))
        assert model2 in ["subscription", "transactional", "service", "freemium"]
        assert model2 == "service"  # Should be service-based for healthcare

        # Finance domain
        model3 = self.engine._choose_pricing_model(MagicMock(domain="finance.algorithmic_trading"))
        assert model3 in ["subscription", "transactional", "service", "freemium"]
        assert model3 == "transactional"  # Should be transactional for finance

    def test_define_revenue_streams_with_models(self) -> None:
        """Test defining revenue streams for different models."""
        # Subscription model
        streams1 = self.engine._define_revenue_streams("subscription")
        assert isinstance(streams1, list)
        assert len(streams1) > 0
        assert "Monthly subscriptions" in streams1
        assert "Annual subscriptions" in streams1

        # Transactional model
        streams2 = self.engine._define_revenue_streams("transactional")
        assert isinstance(streams2, list)
        assert len(streams2) > 0
        assert "Pay-per-use fees" in streams2
        assert "Volume discounts" in streams2

        # Service model
        streams3 = self.engine._define_revenue_streams("service")
        assert isinstance(streams3, list)
        assert len(streams3) > 0
        assert "Consulting services" in streams3
        assert "Custom development" in streams3

    def test_estimate_development_requirements_with_opportunities(self) -> None:
        """Test estimating development requirements for opportunities."""
        # Small market, low competition
        opportunity1 = MagicMock(market_size="small", competition_level="low")
        time1, cost1 = self.engine._estimate_development_requirements(opportunity1)
        assert isinstance(time1, float)
        assert isinstance(cost1, float)
        assert time1 > 0
        assert cost1 > 0
        # Should be relatively low for easy markets

        # Massive market, intense competition
        opportunity2 = MagicMock(market_size="massive", competition_level="intense")
        time2, cost2 = self.engine._estimate_development_requirements(opportunity2)
        assert isinstance(time2, float)
        assert isinstance(cost2, float)
        assert time2 > 0
        assert cost2 > 0
        # Should be higher for challenging markets

    def test_project_revenue_with_opportunities(self) -> None:
        """Test projecting revenue for opportunities."""
        # Massive market, low competition
        opportunity1 = MagicMock(market_size="massive", competition_level="low")
        revenue1 = self.engine._project_revenue(opportunity1)
        assert isinstance(revenue1, dict)
        assert "year_1" in revenue1
        assert "year_2" in revenue1
        assert "year_3" in revenue1
        assert revenue1["year_1"] > 0
        assert revenue1["year_2"] > revenue1["year_1"]  # Growth
        assert revenue1["year_3"] > revenue1["year_2"]  # More growth

        # Small market, high competition
        opportunity2 = MagicMock(market_size="small", competition_level="high")
        revenue2 = self.engine._project_revenue(opportunity2)
        assert isinstance(revenue2, dict)
        assert "year_1" in revenue2
        assert "year_2" in revenue2
        assert "year_3" in revenue2
        assert revenue2["year_1"] > 0
        # Growth should be slower for harder markets

    def test_calculate_success_probability_with_opportunities(self) -> None:
        """Test calculating success probability for opportunities."""
        # High alignment, viable market
        opportunity1 = MagicMock(
            alignment_score=0.9,
            viability_score=0.85,
            market_size="large",
            competition_level="medium"
        )
        prob1 = self.engine._calculate_success_probability(opportunity1)
        assert isinstance(prob1, float)
        assert 0.1 <= prob1 <= 0.95  # Between 10% and 95%
        assert prob1 > 0.7  # Should be high for good opportunities

        # Low alignment, difficult market
        opportunity2 = MagicMock(
            alignment_score=0.2,
            viability_score=0.15,
            market_size="small",
            competition_level="intense"
        )
        prob2 = self.engine._calculate_success_probability(opportunity2)
        assert isinstance(prob2, float)
        assert 0.1 <= prob2 <= 0.95  # Between 10% and 95%
        assert prob2 < 0.5  # Should be lower for poor opportunities

        # Medium alignment, balanced market
        opportunity3 = MagicMock(
            alignment_score=0.6,
            viability_score=0.55,
            market_size="medium",
            competition_level="medium"
        )
        prob3 = self.engine._calculate_success_probability(opportunity3)
        assert isinstance(prob3, float)
        assert 0.1 <= prob3 <= 0.95  # Between 10% and 95%
        assert 0.3 <= prob3 <= 0.7  # Should be moderate

    def test_determine_stream_type_with_revenue_types(self) -> None:
        """Test determining stream type for different revenue types."""
        # Subscription revenues
        stream_type1 = self.engine._determine_stream_type("Monthly subscriptions")
        assert stream_type1 == "subscription"

        stream_type2 = self.engine._determine_stream_type("Premium subscriptions")
        assert stream_type2 == "subscription"

        # Licensing revenues
        stream_type3 = self.engine._determine_stream_type("Enterprise licensing")
        assert stream_type3 == "licensing"

        # Transactional revenues
        stream_type4 = self.engine._determine_stream_type("Pay-per-use fees")
        assert stream_type4 == "transactional"

        # Service revenues
        stream_type5 = self.engine._determine_stream_type("Consulting services")
        assert stream_type5 == "service"

        # Unknown revenue type defaults to subscription
        stream_type6 = self.engine._determine_stream_type("Unknown revenue")
        assert stream_type6 == "subscription"

    def test_estimate_initial_mrr_with_solutions(self) -> None:
        """Test estimating initial MRR for solutions."""
        # Enterprise-focused solution
        solution1 = MagicMock(
            expected_revenue={"year_1": 120000.0},
            target_customers=["Enterprises"]
        )
        mrr1 = self.engine._estimate_initial_mrr(solution1, "Monthly subscriptions")
        assert isinstance(mrr1, float)
        assert mrr1 > 0
        # Should be higher for enterprise solutions

        # Developer-focused solution
        solution2 = MagicMock(
            expected_revenue={"year_1": 60000.0},
            target_customers=["Independent Developers"]
        )
        mrr2 = self.engine._estimate_initial_mrr(solution2, "Monthly subscriptions")
        assert isinstance(mrr2, float)
        assert mrr2 > 0
        # Should be lower for individual users

        # Multiple customer segments
        solution3 = MagicMock(
            expected_revenue={"year_1": 240000.0},
            target_customers=["Enterprises", "SMEs"]
        )
        mrr3 = self.engine._estimate_initial_mrr(solution3, "Monthly subscriptions")
        assert isinstance(mrr3, float)
        assert mrr3 > 0
        # Should be balanced for mixed users

    def test_estimate_customer_acquisition_cost_with_segments(self) -> None:
        """Test estimating CAC for different customer segments."""
        # Enterprise customers
        solution1 = MagicMock(target_customers=["Enterprises"])
        cac1 = self.engine._estimate_customer_acquisition_cost(solution1, "Monthly subscriptions")
        assert isinstance(cac1, float)
        assert cac1 > 1000  # Should be high for enterprises

        # Individual developers
        solution2 = MagicMock(target_customers=["Independent Developers"])
        cac2 = self.engine._estimate_customer_acquisition_cost(solution2, "Monthly subscriptions")
        assert isinstance(cac2, float)
        assert cac2 < 500  # Should be low for individuals

        # Mixed segments
        solution3 = MagicMock(target_customers=["Enterprises", "SMEs"])
        cac3 = self.engine._estimate_customer_acquisition_cost(solution3, "Monthly subscriptions")
        assert isinstance(cac3, float)
        assert cac3 > 1000  # Should favor highest segment

    def test_estimate_customer_lifetime_value_with_streams(self) -> None:
        """Test estimating CLV for different revenue streams."""
        # High-value solution
        solution1 = MagicMock()
        solution1.target_customers = ["Enterprises"]
        solution1.expected_revenue = {"year_1": 120000.0}
        clv1 = self.engine._estimate_customer_lifetime_value(solution1, "Enterprise licensing")
        assert isinstance(clv1, float)
        assert clv1 > 0
        # Should be substantial for enterprise solutions

        # Lower-value solution
        clv2 = self.engine._estimate_customer_lifetime_value(solution1, "Freemium tier")
        assert isinstance(clv2, float)
        assert clv2 > 0
        # Should be lower for freemium

    def test_estimate_churn_rate_with_types(self) -> None:
        """Test estimating churn rate for different stream types."""
        # Subscription churn
        churn1 = self.engine._estimate_churn_rate("subscription")
        assert isinstance(churn1, float)
        assert 0.0 <= churn1 <= 1.0
        assert 0.02 <= churn1 <= 0.10  # Reasonable churn for subscriptions

        # Licensing churn (should be very low)
        churn2 = self.engine._estimate_churn_rate("licensing")
        assert isinstance(churn2, float)
        assert 0.0 <= churn2 <= 1.0
        assert churn2 <= 0.05  # Very low churn for licensing

        # Transactional churn (should be higher)
        churn3 = self.engine._estimate_churn_rate("transactional")
        assert isinstance(churn3, float)
        assert 0.0 <= churn3 <= 1.0
        assert churn3 >= 0.05  # Higher churn for transactional

        # Service churn (should be moderate)
        churn4 = self.engine._estimate_churn_rate("service")
        assert isinstance(churn4, float)
        assert 0.0 <= churn4 <= 1.0
        assert 0.05 <= churn4 <= 0.15  # Moderate churn for services

    def test_estimate_scalability_factor_with_types(self) -> None:
        """Test estimating scalability factor for different stream types."""
        # Subscription scalability (should be very high)
        scale1 = self.engine._estimate_scalability_factor("subscription")
        assert isinstance(scale1, float)
        assert 0.0 <= scale1 <= 1.0
        assert scale1 > 0.9  # Very scalable for subscriptions

        # Licensing scalability (should be high)
        scale2 = self.engine._estimate_scalability_factor("licensing")
        assert isinstance(scale2, float)
        assert 0.0 <= scale2 <= 1.0
        assert scale2 > 0.8  # Very scalable for licensing

        # Service scalability (should be moderate)
        scale3 = self.engine._estimate_scalability_factor("service")
        assert isinstance(scale3, float)
        assert 0.0 <= scale3 <= 1.0
        assert 0.5 <= scale3 <= 1.0  # Moderate to high scalability for services

        # Transactional scalability (should be good)
        scale4 = self.engine._estimate_scalability_factor("transactional")
        assert isinstance(scale4, float)
        assert 0.0 <= scale4 <= 1.0
        assert scale4 > 0.7  # Good scalability for transactional

    def test_estimate_profit_margin_with_types(self) -> None:
        """Test estimating profit margin for different stream types."""
        # Licensing margins (should be very high)
        margin1 = self.engine._estimate_profit_margin("licensing")
        assert isinstance(margin1, float)
        assert 0.0 <= margin1 <= 1.0
        assert margin1 > 0.8  # Very high margins for licensing

        # Subscription margins (should be high)
        margin2 = self.engine._estimate_profit_margin("subscription")
        assert isinstance(margin2, float)
        assert 0.0 <= margin2 <= 1.0
        assert margin2 > 0.7  # High margins for subscriptions

        # Service margins (should be moderate)
        margin3 = self.engine._estimate_profit_margin("service")
        assert isinstance(margin3, float)
        assert 0.0 <= margin3 <= 1.0
        assert 0.5 <= margin3 <= 1.0  # Moderate to high margins for services

        # Transactional margins (should be moderate)
        margin4 = self.engine._estimate_profit_margin("transactional")
        assert isinstance(margin4, float)
        assert 0.0 <= margin4 <= 1.0
        assert 0.6 <= margin4 <= 1.0  # Good margins for transactional

    def test_project_revenue_growth_with_types(self) -> None:
        """Test projecting revenue growth for different stream types."""
        # Subscription growth (should be steady)
        growth1 = self.engine._project_revenue_growth("subscription")
        assert isinstance(growth1, dict)
        assert len(growth1) == 12  # 12 months
        for month in range(1, 13):
            assert f"month_{month}" in growth1

        # Transactional growth (should be faster)
        growth2 = self.engine._project_revenue_growth("transactional")
        assert isinstance(growth2, dict)
        assert len(growth2) == 12  # 12 months
        for month in range(1, 13):
            assert f"month_{month}" in growth2

        # Service growth (should be moderate)
        growth3 = self.engine._project_revenue_growth("service")
        assert isinstance(growth3, dict)
        assert len(growth3) == 12  # 12 months
        for month in range(1, 13):
            assert f"month_{month}" in growth3

    def test_estimate_operational_costs_with_types(self) -> None:
        """Test estimating operational costs for different revenue types."""
        # Subscription costs
        costs1 = self.engine._estimate_operational_costs("subscription")
        assert isinstance(costs1, dict)
        assert "cloud_infrastructure" in costs1
        assert "support_tools" in costs1
        assert "monitoring" in costs1
        assert "marketing" in costs1
        assert "legal_compliance" in costs1

        # Service costs
        costs2 = self.engine._estimate_operational_costs("service")
        assert isinstance(costs2, dict)
        assert "cloud_infrastructure" in costs2
        assert "support_tools" in costs2
        assert "monitoring" in costs2
        assert "marketing" in costs2
        assert "legal_compliance" in costs2

    def test_assess_revenue_risks_with_types(self) -> None:
        """Test assessing revenue risks for different stream types."""
        # Subscription risks
        risks1 = self.engine._assess_revenue_risks("subscription")
        assert isinstance(risks1, str)
        assert "churn" in risks1.lower() or "competition" in risks1.lower()

        # Licensing risks
        risks2 = self.engine._assess_revenue_risks("licensing")
        assert isinstance(risks2, str)
        assert "market" in risks2.lower() or "competition" in risks2.lower()

        # Transactional risks
        risks3 = self.engine._assess_revenue_risks("transactional")
        assert isinstance(risks3, str)
        assert "volume" in risks3.lower() or "price-sensitive" in risks3.lower() or "churn" in risks3.lower()

        # Service risks
        risks4 = self.engine._assess_revenue_risks("service")
        assert isinstance(risks4, str)
        assert "labor" in risks4.lower() or "scaling" in risks4.lower() or "market" in risks4.lower()


class TestIntegration:
    """Integration tests for autonomous monetization system."""

    def test_integrate_autonomous_monetization_complete(self) -> None:
        """Test complete integration of autonomous monetization."""
        # Create mock config
        mock_config = Mock(spec=AgentConfig)
        mock_config.workspace = Mock(spec=WorkspaceConfig)
        mock_config.workspace.root = "/tmp/test"
        mock_config.policies = Mock()
        mock_config.policies.allow_network = False
        mock_config.policies.deny_commands = []
        mock_config.audit = Mock()
        mock_config.audit.enable_file_log = True
        mock_config.audit.file_dir = Path("seed/changes")
        mock_config.audit.enable_git_commit = False
        mock_config.audit.commit_prefix = "A3X"

        # Create mock engines
        mock_auto_evaluator = Mock(spec=AutoEvaluator)
        mock_meta_engine = Mock(spec=MetaCapabilityEngine)
        mock_transfer_engine = Mock(spec=TransferLearningEngine)
        mock_evolution_engine = Mock(spec=SelfDirectedEvolutionEngine)

        # Mock the metrics history to return realistic data
        mock_auto_evaluator._read_metrics_history.return_value = {
            "core.testing.success_rate": [0.8, 0.85, 0.9],
            "core.diffing.success_rate": [0.7, 0.75, 0.8],
            "core.testing.failure_rate": [0.2, 0.15, 0.1],
            "core.diffing.failure_rate": [0.3, 0.25, 0.2]
        }

        # Test integration
        revenue_streams = integrate_autonomous_monetization(
            mock_config,
            mock_auto_evaluator,
            mock_meta_engine,
            mock_transfer_engine,
            mock_evolution_engine
        )

        # Should return list of revenue streams (may be empty depending on analysis)
        assert isinstance(revenue_streams, list)

        # Integration should complete without errors
        assert True  # Test passes if no exceptions are raised


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
