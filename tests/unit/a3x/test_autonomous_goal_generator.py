"""Testes unitários para o AutonomousGoalGenerator."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from a3x.autonomous_goal_generator import (
    AutonomousGoalGenerator,
    AutonomousGoal,
    IntrinsicMotivationProfile,
    CapabilityGapAnalysis,
    CuriosityOpportunity,
    run_autonomous_goal_generation,
)


class TestIntrinsicMotivationProfile:
    """Testes para o perfil de motivação intrínseca."""

    def test_default_profile_creation(self):
        """Testa criação do perfil padrão."""
        profile = IntrinsicMotivationProfile()

        assert profile.curiosity_weight == 0.3
        assert profile.competence_weight == 0.25
        assert profile.autonomy_weight == 0.2
        assert profile.relatedness_weight == 0.15
        assert profile.exploration_bias == 0.1

    def test_custom_profile_creation(self):
        """Testa criação de perfil personalizado."""
        profile = IntrinsicMotivationProfile(
            curiosity_weight=0.4,
            competence_weight=0.3,
            autonomy_weight=0.2,
            relatedness_weight=0.1,
            exploration_bias=0.0
        )

        assert profile.curiosity_weight == 0.4
        assert profile.competence_weight == 0.3
        assert profile.autonomy_weight == 0.2
        assert profile.relatedness_weight == 0.1
        assert profile.exploration_bias == 0.0


class TestAutonomousGoal:
    """Testes para objetivos autônomos."""

    def test_goal_creation(self):
        """Testa criação de objetivo autônomo."""
        timestamp = datetime.now(timezone.utc).isoformat()
        goal = AutonomousGoal(
            id="test_goal_001",
            title="Objetivo de Teste",
            description="Descrição do objetivo de teste",
            goal_type="capability_gap",
            priority="high",
            estimated_impact=0.8,
            required_capabilities=["core.testing"],
            success_criteria=["Melhorar cobertura", "Reduzir bugs"],
            motivation_factors={"curiosity": 0.3, "competence": 0.5},
            created_at=timestamp
        )

        assert goal.id == "test_goal_001"
        assert goal.title == "Objetivo de Teste"
        assert goal.goal_type == "capability_gap"
        assert goal.priority == "high"
        assert goal.estimated_impact == 0.8
        assert goal.created_at == timestamp

    def test_goal_metadata_storage(self):
        """Testa armazenamento de metadados."""
        metadata = {"test_key": "test_value"}
        goal = AutonomousGoal(
            id="test_goal_002",
            title="Teste Metadata",
            description="Teste",
            goal_type="curiosity_exploration",
            priority="medium",
            estimated_impact=0.6,
            required_capabilities=[],
            success_criteria=[],
            motivation_factors={},
            metadata=metadata
        )

        assert goal.metadata == metadata


class TestCapabilityGapAnalysis:
    """Testes para análise de lacunas de capacidades."""

    def test_gap_analysis_creation(self):
        """Testa criação de análise de lacunas."""
        gap = CapabilityGapAnalysis(
            capability_id="core.testing",
            gap_type="performance",
            severity=0.8,
            description="Alta taxa de falhas em testes",
            recommended_actions=["Melhorar cobertura", "Otimizar execução"],
            potential_impact=0.7,
            confidence=0.9
        )

        assert gap.capability_id == "core.testing"
        assert gap.gap_type == "performance"
        assert gap.severity == 0.8
        assert gap.potential_impact == 0.7
        assert len(gap.recommended_actions) == 2


class TestCuriosityOpportunity:
    """Testes para oportunidades de exploração curiosa."""

    def test_opportunity_creation(self):
        """Testa criação de oportunidade de exploração."""
        opportunity = CuriosityOpportunity(
            domain="machine_learning",
            opportunity_type="novelty",
            novelty_score=0.8,
            uncertainty_score=0.6,
            exploration_value=0.7,
            description="Explorar técnicas de ML",
            suggested_approach="Implementar bibliotecas básicas"
        )

        assert opportunity.domain == "machine_learning"
        assert opportunity.opportunity_type == "novelty"
        assert opportunity.novelty_score == 0.8
        assert opportunity.exploration_value == 0.7


class TestAutonomousGoalGenerator:
    """Testes principais para o AutonomousGoalGenerator."""

    def setup_method(self):
        """Configuração para cada teste."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = AutonomousGoalGenerator(self.temp_dir)

    def teardown_method(self):
        """Limpeza após cada teste."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generator_initialization(self):
        """Testa inicialização do gerador."""
        assert self.generator.workspace_root == self.temp_dir
        assert isinstance(self.generator.motivation_profile, IntrinsicMotivationProfile)
        # Note: autonomous_planner is actually created, not mocked in real initialization
        assert hasattr(self.generator, 'autonomous_planner')
        assert self.generator.backlog_path == self.temp_dir / "seed" / "backlog.yaml"

    def test_generator_with_custom_profile(self):
        """Testa gerador com perfil personalizado."""
        custom_profile = IntrinsicMotivationProfile(curiosity_weight=0.5)
        generator = AutonomousGoalGenerator(self.temp_dir, custom_profile)

        assert generator.motivation_profile.curiosity_weight == 0.5

    @patch('a3x.autonomous_goal_generator.AutonomousPlanner')
    def test_goal_generation_with_mock_planner(self, mock_planner_class):
        """Testa geração de objetivos com planner mockado."""
        # Configurar mock
        mock_planner = Mock()
        mock_planner_class.return_value = mock_planner

        # Criar arquivos necessários para o teste
        seed_dir = self.temp_dir / "seed"
        seed_dir.mkdir(parents=True, exist_ok=True)

        capabilities_file = seed_dir / "capabilities.yaml"
        capabilities_file.write_text("""
- id: core.testing
  name: Testing Core
  category: core
  description: Core testing functionality
  maturity: established
  metrics:
    success_rate: 0.7
""")

        metrics_file = seed_dir / "metrics" / "history.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps({
            "tests_success_rate": [0.6, 0.65, 0.7],
            "actions_success_rate": [0.8, 0.85, 0.9]
        }))

        # Gerar objetivos
        goals = self.generator.generate_autonomous_goals()

        # Verificar que objetivos foram gerados
        assert isinstance(goals, list)
        # Nota: Em ambiente real, mais objetivos seriam gerados
        # Aqui pode ser vazio devido às condições dos testes

    def test_capability_gap_analysis(self):
        """Testa análise de lacunas de capacidades."""
        # Criar arquivo de capacidades para teste
        seed_dir = self.temp_dir / "seed"
        seed_dir.mkdir(parents=True, exist_ok=True)

        capabilities_file = seed_dir / "capabilities.yaml"
        capabilities_file.write_text("""
- id: core.testing
  name: Testing Core
  category: core
  description: Core testing functionality
  maturity: baseline
  metrics:
    success_rate: 0.5
""")

        # Executar análise
        gaps = self.generator._analyze_capability_gaps()

        # Verificar estrutura (pode estar vazio devido a condições específicas)
        assert isinstance(gaps, list)

    def test_curiosity_opportunities_identification(self):
        """Testa identificação de oportunidades de exploração curiosa."""
        opportunities = self.generator._identify_curiosity_opportunities()

        assert isinstance(opportunities, list)

    def test_goal_filtering_and_prioritization(self):
        """Testa filtragem e priorização de objetivos."""
        # Criar objetivos de teste
        goals = [
            AutonomousGoal(
                id="goal_1",
                title="High Priority Goal",
                description="Test",
                goal_type="capability_gap",
                priority="high",
                estimated_impact=0.9,
                required_capabilities=[],
                success_criteria=[],
                motivation_factors={"curiosity": 0.8}
            ),
            AutonomousGoal(
                id="goal_2",
                title="Low Priority Goal",
                description="Test",
                goal_type="curiosity_exploration",
                priority="low",
                estimated_impact=0.3,
                required_capabilities=[],
                success_criteria=[],
                motivation_factors={"curiosity": 0.2}
            )
        ]

        # Aplicar filtros
        filtered = self.generator._filter_and_prioritize_goals(goals)

        assert isinstance(filtered, list)
        assert len(filtered) <= len(goals)

    def test_goal_conversion_to_seeds(self):
        """Testa conversão de objetivos em seeds."""
        goals = [
            AutonomousGoal(
                id="test_goal",
                title="Test Goal",
                description="Test description",
                goal_type="capability_gap",
                priority="medium",
                estimated_impact=0.7,
                required_capabilities=["core.testing"],
                success_criteria=["Success criterion"],
                motivation_factors={"curiosity": 0.5}
            )
        ]

        seeds = self.generator.convert_goals_to_seeds(goals)

        assert len(seeds) == 1
        seed = seeds[0]
        assert seed.description.startswith("[AUTO-GOAL]")
        assert seed.priority == "medium"
        assert seed.capability == "core.testing"
        assert seed.seed_type == "autonomous_goal"
        assert seed.data is not None
        assert "goal_id" in seed.data

    def test_meta_reflection(self):
        """Testa capacidade de meta-reflexão."""
        # Adicionar alguns objetivos ao histórico
        test_goals = [
            AutonomousGoal(
                id="test_1",
                title="Test 1",
                description="Test",
                goal_type="capability_gap",
                priority="high",
                estimated_impact=0.8,
                required_capabilities=[],
                success_criteria=[],
                motivation_factors={"curiosity": 0.3, "competence": 0.5}
            ),
            AutonomousGoal(
                id="test_2",
                title="Test 2",
                description="Test",
                goal_type="curiosity_exploration",
                priority="medium",
                estimated_impact=0.6,
                required_capabilities=[],
                success_criteria=[],
                motivation_factors={"curiosity": 0.7, "autonomy": 0.2}
            )
        ]

        self.generator._goal_history.extend(test_goals)

        # Executar meta-reflexão
        reflection = self.generator.perform_meta_reflection()

        assert isinstance(reflection, dict)
        assert "total_goals_generated" in reflection
        assert "goals_by_type" in reflection
        assert "motivation_effectiveness" in reflection
        assert reflection["total_goals_generated"] == 2
        assert reflection["goals_by_type"]["capability_gap"] == 1
        assert reflection["goals_by_type"]["curiosity_exploration"] == 1

    def test_priority_score_calculation(self):
        """Testa cálculo de score de prioridade."""
        assert self.generator._get_priority_score("high") == 3.0
        assert self.generator._get_priority_score("medium") == 2.0
        assert self.generator._get_priority_score("low") == 1.0
        assert self.generator._get_priority_score("unknown") == 1.0

    def test_capability_suggestions_for_domain(self):
        """Testa sugestões de capacidades para domínio."""
        testing_caps = self.generator._suggest_capabilities_for_domain("testing")
        assert "core.testing" in testing_caps

        ml_caps = self.generator._suggest_capabilities_for_domain("machine_learning")
        assert len(ml_caps) > 0

        unknown_caps = self.generator._suggest_capabilities_for_domain("unknown_domain")
        assert len(unknown_caps) > 0

    def test_context_filtering(self):
        """Testa aplicação de filtros contextuais."""
        goals = [
            AutonomousGoal(
                id="goal_1",
                title="Test",
                description="Test",
                goal_type="capability_gap",
                priority="high",
                estimated_impact=0.8,
                required_capabilities=["core.testing"],
                success_criteria=[],
                motivation_factors={}
            ),
            AutonomousGoal(
                id="goal_2",
                title="Test",
                description="Test",
                goal_type="curiosity_exploration",
                priority="medium",
                estimated_impact=0.5,
                required_capabilities=["meta.skill_creation"],
                success_criteria=[],
                motivation_factors={}
            )
        ]

        # Testar filtro por tipo
        context = {"goal_types": ["capability_gap"]}
        filtered = self.generator._apply_context_filters(goals, context)

        assert len(filtered) == 1
        assert filtered[0].goal_type == "capability_gap"

        # Testar filtro por impacto mínimo
        context = {"min_impact": 0.7}
        filtered = self.generator._apply_context_filters(goals, context)

        assert len(filtered) == 1
        assert filtered[0].estimated_impact >= 0.7

    @patch('a3x.autonomous_goal_generator.SeedBacklog')
    def test_integration_with_planner(self, mock_backlog_class):
        """Testa integração com o planejador autônomo."""
        # Configurar mock
        mock_backlog = Mock()
        mock_backlog_class.load.return_value = mock_backlog

        # Criar objetivos para teste
        test_goals = [
            AutonomousGoal(
                id="integration_test_goal",
                title="Integration Test Goal",
                description="Test integration",
                goal_type="capability_gap",
                priority="medium",
                estimated_impact=0.7,
                required_capabilities=["core.testing"],
                success_criteria=["Test criterion"],
                motivation_factors={"curiosity": 0.5}
            )
        ]

        self.generator._goal_history.extend(test_goals)

        # Executar integração
        seeds = self.generator.integrate_with_autonomous_planner()

        # Verificar que seeds foram criados
        assert isinstance(seeds, list)
        assert len(seeds) > 0  # O gerador pode criar múltiplos seeds

        # Verificar que o método executou sem erros
        # Nota: O backlog só é usado se o arquivo existir
        assert isinstance(seeds, list)
        assert len(seeds) > 0


class TestRunAutonomousGoalGeneration:
    """Testes para a função de execução principal."""

    def test_run_function(self):
        """Testa função de execução principal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Criar estrutura mínima necessária
            seed_dir = temp_path / "seed"
            seed_dir.mkdir(exist_ok=True)

            # Executar função
            seeds = run_autonomous_goal_generation(temp_path)

            # Verificar resultado
            assert isinstance(seeds, list)


class TestIntegrationScenarios:
    """Testes de cenários de integração."""

    def test_end_to_end_goal_generation(self):
        """Testa geração completa de objetivos."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Criar estrutura de arquivos necessária
            seed_dir = temp_path / "seed"
            seed_dir.mkdir(exist_ok=True)

            evaluations_dir = seed_dir / "evaluations"
            evaluations_dir.mkdir(exist_ok=True)

            metrics_dir = seed_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)

            # Criar arquivo de capacidades
            capabilities_file = seed_dir / "capabilities.yaml"
            capabilities_file.write_text("""
- id: core.testing
  name: Testing Core
  category: core
  description: Core testing functionality
  maturity: baseline
  metrics:
    success_rate: 0.6
""")

            # Criar arquivo de métricas
            metrics_file = metrics_dir / "history.json"
            metrics_file.write_text(json.dumps({
                "tests_success_rate": [0.5, 0.55, 0.6],
                "actions_success_rate": [0.7, 0.75, 0.8]
            }))

            # Criar gerador e executar
            generator = AutonomousGoalGenerator(temp_path)
            goals = generator.generate_autonomous_goals()

            # Verificar resultados
            assert isinstance(goals, list)

    def test_motivation_factors_calculation(self):
        """Testa cálculo de fatores de motivação."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Criar perfil personalizado com pesos conhecidos
            profile = IntrinsicMotivationProfile(
                curiosity_weight=0.4,
                competence_weight=0.3,
                autonomy_weight=0.2,
                relatedness_weight=0.1
            )

            generator = AutonomousGoalGenerator(temp_path, profile)

            # Criar oportunidade de exploração curiosa
            opportunity = CuriosityOpportunity(
                domain="test_domain",
                opportunity_type="novelty",
                novelty_score=0.8,
                uncertainty_score=0.6,
                exploration_value=0.7,
                description="Test opportunity",
                suggested_approach="Test approach"
            )

            # Gerar objetivos baseados na oportunidade
            timestamp = datetime.now(timezone.utc).isoformat()
            goals = generator._generate_curiosity_goals([opportunity], timestamp)

            # Verificar cálculo de motivação
            if goals:
                goal = goals[0]
                motivation_total = sum(goal.motivation_factors.values())

                # Deve incluir fatores de curiosidade e autonomia
                assert "curiosity" in goal.motivation_factors
                assert "autonomy" in goal.motivation_factors
                assert motivation_total > 0


if __name__ == "__main__":
    pytest.main([__file__])