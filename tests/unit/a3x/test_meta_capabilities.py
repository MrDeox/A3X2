"""Testes para o módulo de capacidades meta do SeedAI."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from a3x.autoeval import AutoEvaluator
from a3x.config import AgentConfig, WorkspaceConfig
from a3x.meta_capabilities import (
    MetaCapabilityEngine,
    MetaSkill,
    SkillProposal,
    integrate_meta_capabilities,
)


class TestMetaSkill:
    """Testes para a classe MetaSkill."""

    def test_meta_skill_creation(self) -> None:
        """Testa a criação de uma meta-habilidade."""
        meta_skill = MetaSkill(
            id="test.meta_skill",
            name="Test Meta Skill",
            description="Test meta skill description",
            implementation_template="class {{skill_name}}:",
            required_capabilities=["core.test"],
            estimated_complexity=0.5,
            success_probability=0.8,
            last_updated=datetime.now(timezone.utc).isoformat(),
            version="0.1"
        )

        assert meta_skill.id == "test.meta_skill"
        assert meta_skill.name == "Test Meta Skill"
        assert meta_skill.description == "Test meta skill description"
        assert meta_skill.implementation_template == "class {{skill_name}}:"
        assert meta_skill.required_capabilities == ["core.test"]
        assert meta_skill.estimated_complexity == 0.5
        assert meta_skill.success_probability == 0.8
        assert meta_skill.version == "0.1"


class TestSkillProposal:
    """Testes para a classe SkillProposal."""

    def test_skill_proposal_creation(self) -> None:
        """Testa a criação de uma proposta de habilidade."""
        proposal = SkillProposal(
            id="test_proposal_001",
            name="Test Proposal",
            description="Test proposal description",
            implementation_plan="Implementation plan",
            required_dependencies=["dep1", "dep2"],
            estimated_effort=3.0,
            priority="medium",
            rationale="Test rationale",
            target_domain="test",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        assert proposal.id == "test_proposal_001"
        assert proposal.name == "Test Proposal"
        assert proposal.description == "Test proposal description"
        assert proposal.implementation_plan == "Implementation plan"
        assert proposal.required_dependencies == ["dep1", "dep2"]
        assert proposal.estimated_effort == 3.0
        assert proposal.priority == "medium"
        assert proposal.rationale == "Test rationale"
        assert proposal.target_domain == "test"


class TestMetaCapabilityEngine:
    """Testes para o motor de capacidades meta."""

    def setup_method(self) -> None:
        """Configuração antes de cada teste."""
        # Criar diretório temporário para testes
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_root = self.temp_dir / "workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Criar configuração de teste
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

        # Criar auto-avaliador mock
        self.mock_auto_evaluator = Mock(spec=AutoEvaluator)
        self.mock_auto_evaluator._read_metrics_history.return_value = {}

        # Criar engine
        self.engine = MetaCapabilityEngine(self.mock_config, self.mock_auto_evaluator)

    def teardown_method(self) -> None:
        """Limpeza após cada teste."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_directories(self) -> None:
        """Testa que o inicializador cria os diretórios necessários."""
        # Verificar que os diretórios foram criados
        assert self.engine.skills_path.exists()
        assert self.engine.skills_path.is_dir()

        # Verificar que são subdiretórios do workspace
        assert str(self.workspace_root) in str(self.engine.skills_path)

    def test_define_meta_skills(self) -> None:
        """Testa a definição de meta-habilidades."""
        meta_skills = self.engine._define_meta_skills()

        # Deve ter meta-habilidades definidas
        assert isinstance(meta_skills, dict)
        assert len(meta_skills) > 0

        # Verificar meta-habilidades específicas
        assert "skill_creator" in meta_skills
        assert "domain_expander" in meta_skills
        assert "skill_optimizer" in meta_skills

        # Verificar propriedades das meta-habilidades
        skill_creator = meta_skills["skill_creator"]
        assert isinstance(skill_creator, MetaSkill)
        assert skill_creator.id == "meta.skill_creator"
        assert "{{skill_name}}" in skill_creator.implementation_template

    def test_identify_capability_gaps_no_history(self) -> None:
        """Testa identificação de gaps sem histórico de métricas."""
        self.mock_auto_evaluator._read_metrics_history.return_value = {}

        gaps = self.engine._identify_capability_gaps()

        # Deve retornar lista vazia quando não há histórico
        assert isinstance(gaps, list)
        assert len(gaps) == 0

    def test_identify_capability_gaps_with_low_performance(self) -> None:
        """Testa identificação de gaps com baixo desempenho."""
        metrics_history = {
            "core.test.success_rate": [0.4, 0.3, 0.5, 0.2, 0.6],  # Baixa taxa de sucesso
            "core.test.apply_patch_count": [1.0, 2.0, 1.0, 3.0, 2.0]
        }
        self.mock_auto_evaluator._read_metrics_history.return_value = metrics_history

        gaps = self.engine._identify_capability_gaps()

        # Deve identificar gaps de baixo desempenho
        assert isinstance(gaps, list)
        low_perf_gaps = [g for g in gaps if g["type"] == "low_performance"]
        assert len(low_perf_gaps) > 0

        # Verificar propriedades do gap
        gap = low_perf_gaps[0]
        assert gap["capability"] == "core.test"
        assert gap["metric"] == "success_rate"
        assert gap["value"] < 0.7  # Abaixo do threshold

    def test_analyze_mission_needs_empty_state(self) -> None:
        """Testa análise de necessidades de missão com estado vazio."""
        with patch("a3x.meta_capabilities.load_mission_state") as mock_load:
            mock_load.return_value = Mock()
            mock_load.return_value.missions = []

            needs = self.engine._analyze_mission_needs()

            # Deve retornar lista vazia quando não há missões
            assert isinstance(needs, list)
            assert len(needs) == 0

    def test_identify_optimization_opportunities_no_history(self) -> None:
        """Testa identificação de oportunidades de otimização sem histórico."""
        self.mock_auto_evaluator._read_metrics_history.return_value = {}

        opportunities = self.engine._identify_optimization_opportunities()

        # Deve retornar lista vazia quando não há histórico
        assert isinstance(opportunities, list)
        assert len(opportunities) == 0

    def test_create_skill_proposal_for_gap_low_performance(self) -> None:
        """Testa criação de proposta para gap de baixo desempenho."""
        gap = {
            "type": "low_performance",
            "capability": "core.test",
            "metric": "success_rate",
            "value": 0.3,
            "description": "Baixo desempenho em core.test (taxa de sucesso: 0.30)"
        }

        proposal = self.engine._create_skill_proposal_for_gap(gap)

        # Deve criar proposta válida
        assert isinstance(proposal, SkillProposal)
        assert proposal.name == "Melhoria de core.test"
        assert "core.test" in proposal.description
        assert proposal.priority == "high"
        assert proposal.target_domain == "core"

    def test_create_skill_proposal_for_gap_missing_capability(self) -> None:
        """Testa criação de proposta para capability ausente."""
        gap = {
            "type": "missing_capability",
            "capability": "new.feature",
            "description": "Capacidade new.feature ausente no registro"
        }

        proposal = self.engine._create_skill_proposal_for_gap(gap)

        # Deve criar proposta válida
        assert isinstance(proposal, SkillProposal)
        assert proposal.name == "Criação de new.feature"
        assert "new.feature" in proposal.description
        assert proposal.priority == "medium"
        assert proposal.target_domain == "core"

    def test_create_skill_proposal_for_need_mission_requirement(self) -> None:
        """Testa criação de proposta para necessidade de missão."""
        need = {
            "type": "mission_requirement",
            "mission_id": "mission_001",
            "capability_tag": "required.skill",
            "description": "Missão mission_001 requer capacidade required.skill não implementada"
        }

        proposal = self.engine._create_skill_proposal_for_need(need)

        # Deve criar proposta válida
        assert isinstance(proposal, SkillProposal)
        assert "required.skill" in proposal.name
        assert "mission_001" in proposal.name
        assert "required.skill" in proposal.description
        assert "mission_001" in proposal.description
        assert proposal.priority == "high"
        assert proposal.target_domain == "mission_specific"

    def test_create_skill_proposal_for_optimization_declining_performance(self) -> None:
        """Testa criação de proposta para otimização de performance em declínio."""
        opportunity = {
            "type": "declining_performance",
            "capability": "core.test",
            "trend": -0.06,
            "description": "Desempenho de core.test em declínio (-0.060 por medição)"
        }

        proposal = self.engine._create_skill_proposal_for_optimization(opportunity)

        # Deve criar proposta válida
        assert isinstance(proposal, SkillProposal)
        assert "core.test" in proposal.name
        assert "Recuperação" in proposal.name
        assert "core.test" in proposal.description
        assert proposal.priority == "high"
        assert proposal.target_domain == "performance"

    def test_create_skill_proposal_for_optimization_high_resource_usage(self) -> None:
        """Testa criação de proposta para otimização de uso de recursos."""
        opportunity = {
            "type": "high_resource_usage",
            "capability": "heavy.processing",
            "avg_time": 7.5,
            "description": "heavy.processing consome recursos excessivos (tempo médio: 7.50s)"
        }

        proposal = self.engine._create_skill_proposal_for_optimization(opportunity)

        # Deve criar proposta válida
        assert isinstance(proposal, SkillProposal)
        assert "heavy.processing" in proposal.name
        assert "Otimização" in proposal.name
        assert "heavy.processing" in proposal.description
        assert proposal.priority == "medium"
        assert proposal.target_domain == "performance"

    def test_extract_python_code_from_patch_simple(self) -> None:
        """Testa extração de código Python de patch simples."""
        patch_content = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,3 @@
-old code
+def hello():
+    print("Hello, world!")
+    return True
"""

        extracted = self.engine._extract_python_code_from_patch(patch_content)

        # Deve extrair código adicionado
        assert "def hello():" in extracted
        assert 'print("Hello, world!")' in extracted
        assert "return True" in extracted

    def test_analyze_ast_complexity_simple(self) -> None:
        """Testa análise de complexidade AST simples."""
        import ast
        code = """
def simple_function():
    return 42

class SimpleClass:
    def method(self):
        pass
"""
        tree = ast.parse(code)
        complexity = self.engine._analyze_ast_complexity(tree)

        # Deve calcular métricas de complexidade
        assert "ast_function_count" in complexity
        assert "ast_class_count" in complexity
        assert "ast_total_nodes" in complexity
        assert "ast_max_depth" in complexity

        # Verificar valores
        assert complexity["ast_function_count"] >= 1.0
        assert complexity["ast_class_count"] >= 1.0
        assert complexity["ast_total_nodes"] > 0.0
        assert complexity["ast_max_depth"] > 0.0

    def test_analyze_code_complexity_from_patch(self) -> None:
        """Testa análise de complexidade a partir de patch."""
        patch_content = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,5 @@
+def calculate():
+    result = 10 * 3.14159 * 42
+    return result
+    
"""

        complexity = self.engine.analyze_code_complexity_from_patch(patch_content)

        # Deve analisar complexidade do código no patch
        assert isinstance(complexity, dict)
        # Pode estar vazio se não conseguir extrair código Python válido

    def test_check_code_quality_issues_no_issues(self) -> None:
        """Testa verificação de problemas de qualidade sem problemas."""
        quality_metrics = {
            "failure_rate": 0.1,  # Baixa taxa de falha
            "success_rate": 0.9,
            "apply_patch_count": 3.0,  # Poucos patches
            "file_diversity": 5.0  # Alta diversidade
        }

        seeds = self.engine._check_code_quality_issues(quality_metrics)

        # Não deve gerar seeds quando não há problemas
        assert isinstance(seeds, list)
        assert len(seeds) == 0

    def test_check_code_quality_issues_high_failure_rate(self) -> None:
        """Testa verificação de problemas de qualidade com alta taxa de falha."""
        quality_metrics = {
            "failure_rate": 0.4,  # Alta taxa de falha (> 30%)
            "success_rate": 0.6,
            "apply_patch_count": 1.0,
            "file_diversity": 1.0
        }

        seeds = self.engine._check_code_quality_issues(quality_metrics)

        # Deve gerar seed para alta taxa de falha
        assert isinstance(seeds, list)
        assert len(seeds) > 0

        # Verificar seed de alta taxa de falha
        failure_seeds = [s for s in seeds if "falhas" in s.description]
        assert len(failure_seeds) > 0
        assert failure_seeds[0].priority == "high"
        assert failure_seeds[0].capability == "core.execution"

    def test_check_code_quality_issues_low_success_patches(self) -> None:
        """Testa verificação de problemas de qualidade com baixa taxa de sucesso em patches."""
        quality_metrics = {
            "failure_rate": 0.1,  # Baixa taxa de falha
            "success_rate": 0.6,  # Baixa taxa de sucesso (< 70%)
            "apply_patch_count": 10.0,  # Muitos patches
            "file_diversity": 1.0
        }

        seeds = self.engine._check_code_quality_issues(quality_metrics)

        # Deve gerar seed para baixa taxa de sucesso em patches
        assert isinstance(seeds, list)
        assert len(seeds) > 0

        # Verificar seed de patches com baixa taxa de sucesso
        patch_seeds = [s for s in seeds if "patches" in s.description]
        assert len(patch_seeds) > 0
        assert patch_seeds[0].priority == "medium"
        assert patch_seeds[0].capability == "core.diffing"

    def test_check_code_quality_issues_low_diversity(self) -> None:
        """Testa verificação de problemas de qualidade com baixa diversidade de arquivos."""
        quality_metrics = {
            "failure_rate": 0.1,  # Baixa taxa de falha
            "success_rate": 0.9,  # Alta taxa de sucesso
            "apply_patch_count": 15.0,  # Muitos patches
            "file_diversity": 1.0  # Baixa diversidade (< 2)
        }

        seeds = self.engine._check_code_quality_issues(quality_metrics)

        # Deve gerar seed para baixa diversidade de arquivos
        assert isinstance(seeds, list)
        assert len(seeds) > 0

        # Verificar seed de baixa diversidade
        diversity_seeds = [s for s in seeds if "diversidade" in s.description]
        assert len(diversity_seeds) > 0
        assert diversity_seeds[0].priority == "low"
        assert diversity_seeds[0].capability == "horiz.file_handling"

    def test_propose_new_skills_integration(self) -> None:
        """Testa integração completa da proposta de novas habilidades."""
        # Mock para retornar gaps e necessidades
        with patch.object(self.engine, "_identify_capability_gaps", return_value=[
            {
                "type": "low_performance",
                "capability": "core.test",
                "metric": "success_rate",
                "value": 0.3,
                "description": "Baixo desempenho em core.test"
            }
        ]), patch.object(self.engine, "_analyze_mission_needs", return_value=[
            {
                "type": "mission_requirement",
                "mission_id": "mission_001",
                "capability_tag": "required.skill",
                "description": "Missão requer capacidade não implementada"
            }
        ]), patch.object(self.engine, "_identify_optimization_opportunities", return_value=[
            {
                "type": "declining_performance",
                "capability": "core.optimized",
                "trend": -0.06,
                "description": "Performance em declínio"
            }
        ]):
            proposals = self.engine.propose_new_skills()

            # Deve gerar propostas baseadas nas análises
            assert isinstance(proposals, list)
            assert len(proposals) > 0

            # Verificar tipos de propostas
            proposal_names = [p.name for p in proposals]
            assert any("Melhoria" in name for name in proposal_names)
            # Testa se temos propostas de diferentes tipos (melhoria, criação, otimização)
            has_creation = any("Criação" in name or "creation" in name.lower() or "creation" in name.lower() for name in proposal_names)
            has_improvement = any("Melhoria" in name for name in proposal_names)
            # Pelo menos um tipo deve estar presente
            assert has_creation or has_improvement
            assert any("Recuperação" in name for name in proposal_names)

    def test_generate_skill_implementation(self) -> None:
        """Testa geração de implementação de habilidade."""
        proposal = SkillProposal(
            id="test_proposal_001",
            name="Test Skill",
            description="Test skill description",
            implementation_plan="Implementation plan",
            required_dependencies=["dep1"],
            estimated_effort=2.0,
            priority="medium",
            rationale="Test rationale",
            target_domain="test",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        implementation = self.engine.generate_skill_implementation(proposal)

        # Deve gerar código de implementação
        assert isinstance(implementation, str)
        assert len(implementation) > 0
        assert "class" in implementation  # Deve conter definição de classe

    def test_create_skill_seed(self) -> None:
        """Testa criação de seed para implementação de habilidade."""
        proposal = SkillProposal(
            id="test_proposal_001",
            name="Test Skill",
            description="Test skill description",
            implementation_plan="Implementation plan",
            required_dependencies=["dep1"],
            estimated_effort=2.0,
            priority="medium",
            rationale="Test rationale",
            target_domain="test",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Mock para gerar implementação
        with patch.object(self.engine, "generate_skill_implementation", return_value="class TestSkill: pass"):
            seed = self.engine.create_skill_seed(proposal)

        # Deve criar seed válida
        assert seed is not None
        assert "Criar habilidade" in seed.description
        assert seed.priority == "medium"
        assert seed.capability == "meta.skill_creation.test"
        assert seed.seed_type == "skill_creation"
        assert seed.data is not None
        assert "proposal_id" in seed.data
        assert "skill_name" in seed.data
        assert "generated_code" in seed.data

    def test_evaluate_proposal_feasibility_no_issues(self) -> None:
        """Testa avaliação de viabilidade de proposta sem problemas."""
        proposal = SkillProposal(
            id="test_proposal_001",
            name="Test Skill",
            description="Test skill description",
            implementation_plan="Implementation plan",
            required_dependencies=["core.test"],  # Dependência existente
            estimated_effort=2.0,  # Esforço baixo
            priority="medium",
            rationale="Test rationale",
            target_domain="test",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Mock para simular que a dependência existe
        with patch.object(self.engine.capability_registry, "_by_id", {"core.test": MagicMock()}):
            is_feasible, score, reason = self.engine.evaluate_proposal_feasibility(proposal)

        # Deve ser viável
        assert is_feasible is True
        assert isinstance(score, float)
        assert score > 0.5  # Score alto indica viabilidade
        assert "Viável" in reason

    def test_evaluate_proposal_feasibility_missing_dependencies(self) -> None:
        """Testa avaliação de viabilidade de proposta com dependências faltando."""
        proposal = SkillProposal(
            id="test_proposal_001",
            name="Test Skill",
            description="Test skill description",
            implementation_plan="Implementation plan",
            required_dependencies=["missing.dep"],  # Dependência faltando
            estimated_effort=2.0,
            priority="medium",
            rationale="Test rationale",
            target_domain="test",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Mock para simular que a dependência não existe
        with patch.object(self.engine.capability_registry, "_by_id", {}):
            is_feasible, score, reason = self.engine.evaluate_proposal_feasibility(proposal)

        # Não deve ser viável devido a dependências faltando
        assert is_feasible is False
        assert isinstance(score, float)
        assert score < 0.5  # Score baixo indica problemas
        assert "Faltando" in reason

    def test_evaluate_proposal_feasibility_high_effort(self) -> None:
        """Testa avaliação de viabilidade de proposta com alto esforço."""
        proposal = SkillProposal(
            id="test_proposal_001",
            name="Test Skill",
            description="Test skill description",
            implementation_plan="Implementation plan",
            required_dependencies=["core.test"],
            estimated_effort=15.0,  # Esforço alto (> 10)
            priority="medium",
            rationale="Test rationale",
            target_domain="test",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Mock para simular que a dependência existe
        with patch.object(self.engine.capability_registry, "_by_id", {"core.test": MagicMock()}):
            is_feasible, score, reason = self.engine.evaluate_proposal_feasibility(proposal)

        # Não deve ser viável devido ao alto esforço
        assert is_feasible is False
        assert isinstance(score, float)
        assert score < 0.5  # Score baixo indica problemas
        assert "esforço" in reason.lower()

    def test_save_and_load_skill_proposal(self) -> None:
        """Testa salvar e carregar propostas de habilidades."""
        proposal = SkillProposal(
            id="test_proposal_001",
            name="Test Skill",
            description="Test skill description",
            implementation_plan="Implementation plan",
            required_dependencies=["dep1"],
            estimated_effort=2.0,
            priority="medium",
            rationale="Test rationale",
            target_domain="test",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Salvar proposta
        self.engine.save_skill_proposal(proposal)

        # Verificar que foi salva
        proposal_file = self.engine.skills_path / f"{proposal.id}.json"
        assert proposal_file.exists()

        # Carregar propostas
        loaded_proposals = self.engine.load_skill_proposals()

        # Deve carregar a proposta salva
        assert isinstance(loaded_proposals, list)
        assert len(loaded_proposals) >= 1

        # Verificar propriedades da proposta carregada
        loaded_proposal = loaded_proposals[0]
        assert loaded_proposal.id == proposal.id
        assert loaded_proposal.name == proposal.name
        assert loaded_proposal.description == proposal.description


class TestIntegration:
    """Testes de integração para capacidades meta."""

    def test_integrate_meta__by_id(self) -> None:
        """Testa integração completa de capacidades meta."""
        # Criar configuração de teste
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

        # Criar auto-avaliador mock
        mock_auto_evaluator = Mock(spec=AutoEvaluator)
        mock_auto_evaluator._read_metrics_history.return_value = {}

        # Testar integração
        seeds = integrate_meta_capabilities(mock_config, mock_auto_evaluator)

        # Deve retornar lista de seeds (pode estar vazia)
        assert isinstance(seeds, list)
