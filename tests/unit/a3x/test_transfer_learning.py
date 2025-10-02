"""Tests for transfer learning capabilities in SeedAI."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from a3x.config import AgentConfig, WorkspaceConfig
from a3x.meta_capabilities import Capability, CapabilityRegistry, MetaCapabilityEngine
from a3x.transfer_learning import (
    CrossDomainMapping,
    DomainPattern,
    TransferLearningEngine,
    TransferredSkill,
    integrate_transfer_learning,
)


class TestDomainPattern:
    """Tests for DomainPattern class."""

    def test_domain_pattern_creation(self) -> None:
        """Test creating a domain pattern."""
        pattern = DomainPattern(
            id="test_pattern_001",
            name="Test Pattern",
            description="Test pattern description",
            domain="test.domain",
            pattern_type="structural",
            confidence=0.8,
            examples=["Example 1", "Example 2"],
            related_capabilities=["cap1", "cap2"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        assert pattern.id == "test_pattern_001"
        assert pattern.name == "Test Pattern"
        assert pattern.description == "Test pattern description"
        assert pattern.domain == "test.domain"
        assert pattern.pattern_type == "structural"
        assert pattern.confidence == 0.8
        assert pattern.examples == ["Example 1", "Example 2"]
        assert pattern.related_capabilities == ["cap1", "cap2"]


class TestCrossDomainMapping:
    """Tests for CrossDomainMapping class."""

    def test_cross_domain_mapping_creation(self) -> None:
        """Test creating a cross-domain mapping."""
        mapping = CrossDomainMapping(
            id="test_mapping_001",
            source_domain="source.domain",
            target_domain="target.domain",
            source_pattern="source_pattern_001",
            target_pattern="target_pattern_001",
            similarity_score=0.75,
            adaptation_strategy="Adapt structure for new context",
            confidence=0.85,
            created_at=datetime.now(timezone.utc).isoformat()
        )

        assert mapping.id == "test_mapping_001"
        assert mapping.source_domain == "source.domain"
        assert mapping.target_domain == "target.domain"
        assert mapping.source_pattern == "source_pattern_001"
        assert mapping.target_pattern == "target_pattern_001"
        assert mapping.similarity_score == 0.75
        assert mapping.adaptation_strategy == "Adapt structure for new context"
        assert mapping.confidence == 0.85


class TestTransferredSkill:
    """Tests for TransferredSkill class."""

    def test_transferred_skill_creation(self) -> None:
        """Test creating a transferred skill."""
        skill = TransferredSkill(
            id="test_transferred_skill_001",
            name="Test Transferred Skill",
            description="Test transferred skill description",
            source_domains=["source.domain1", "source.domain2"],
            target_domain="target.domain",
            base_skill="base_skill_001",
            adaptations=["Adaptation 1", "Adaptation 2"],
            confidence=0.9,
            implementation_plan="Implementation plan details",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        assert skill.id == "test_transferred_skill_001"
        assert skill.name == "Test Transferred Skill"
        assert skill.description == "Test transferred skill description"
        assert skill.source_domains == ["source.domain1", "source.domain2"]
        assert skill.target_domain == "target.domain"
        assert skill.base_skill == "base_skill_001"
        assert skill.adaptations == ["Adaptation 1", "Adaptation 2"]
        assert skill.confidence == 0.9
        assert skill.implementation_plan == "Implementation plan details"


class TestTransferLearningEngine:
    """Tests for TransferLearningEngine."""

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

        # Create mock meta engine
        self.mock_meta_engine = Mock(spec=MetaCapabilityEngine)
        self.mock_meta_engine.capability_registry = Mock(spec=CapabilityRegistry)
        self.mock_meta_engine.capability_registry._by_id = {}

        # Create engine
        self.engine = TransferLearningEngine(self.mock_config, self.mock_meta_engine)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_directories(self) -> None:
        """Test that initialization creates required directories."""
        # Verify directories were created
        assert self.engine.transfer_path.exists()
        assert self.engine.patterns_path.exists()
        assert self.engine.mappings_path.exists()
        assert self.engine.transferred_skills_path.exists()

        # Verify they are subdirectories of workspace
        assert str(self.workspace_root) in str(self.engine.transfer_path)
        assert str(self.workspace_root) in str(self.engine.patterns_path)
        assert str(self.workspace_root) in str(self.engine.mappings_path)
        assert str(self.workspace_root) in str(self.engine.transferred_skills_path)

    def test_identify_structural_patterns_with_config_capabilities(self) -> None:
        """Test identifying structural patterns with config-related capabilities."""
        # Create mock capabilities
        capabilities = [
            Capability(id="core.config.loader", name="Config Loader", category="vertical",
                      description="Loads configuration files", maturity="established",
                      metrics={"success_rate": 0.9}, seeds=[], requirements={}, activation={}),
            Capability(id="core.config.parser", name="Config Parser", category="vertical",
                      description="Parses configuration data", maturity="baseline",
                      metrics={"success_rate": 0.8}, seeds=[], requirements={}, activation={})
        ]

        patterns = self.engine._identify_structural_patterns("core.config", capabilities)

        # Should identify config-driven behavior pattern
        assert isinstance(patterns, list)
        assert len(patterns) > 0

        config_pattern = None
        for pattern in patterns:
            if "Configuração" in pattern.name:
                config_pattern = pattern
                break

        assert config_pattern is not None
        assert config_pattern.name == "Comportamento Dirigido por Configuração"
        assert config_pattern.domain == "core.config"
        assert config_pattern.pattern_type == "structural"
        assert config_pattern.confidence >= 0.8
        assert "config" in config_pattern.description.lower()

    def test_identify_structural_patterns_with_pipeline_capabilities(self) -> None:
        """Test identifying structural patterns with pipeline-related capabilities."""
        # Create mock capabilities
        capabilities = [
            Capability(id="data.pipeline.processor", name="Data Processor", category="vertical",
                      description="Processes data through pipeline stages", maturity="established",
                      metrics={"success_rate": 0.95}, seeds=[], requirements={}, activation={}),
            Capability(id="data.pipeline.transformer", name="Data Transformer", category="vertical",
                      description="Transforms data in pipeline", maturity="baseline",
                      metrics={"success_rate": 0.85}, seeds=[], requirements={}, activation={})
        ]

        patterns = self.engine._identify_structural_patterns("data.pipeline", capabilities)

        # Should identify pipeline processing pattern
        assert isinstance(patterns, list)
        assert len(patterns) > 0

        pipeline_pattern = None
        for pattern in patterns:
            if "Pipeline" in pattern.name:
                pipeline_pattern = pattern
                break

        assert pipeline_pattern is not None
        assert pipeline_pattern.name == "Processamento em Pipeline"
        assert pipeline_pattern.domain == "data.pipeline"
        assert pipeline_pattern.pattern_type == "structural"
        assert pipeline_pattern.confidence >= 0.8
        # Check that description contains relevant terms (adjusted for Portuguese)
        assert "pipeline" in pattern.description.lower() or "sequência" in pattern.description.lower() or "etapas" in pattern.description.lower()

    def test_identify_behavioral_patterns_with_event_capabilities(self) -> None:
        """Test identifying behavioral patterns with event-related capabilities."""
        # Create mock capabilities
        capabilities = [
            Capability(id="core.event.listener", name="Event Listener", category="vertical",
                      description="Listens for system events", maturity="established",
                      metrics={"success_rate": 0.9}, seeds=[], requirements={}, activation={}),
            Capability(id="core.event.trigger", name="Event Trigger", category="vertical",
                      description="Triggers events in system", maturity="baseline",
                      metrics={"success_rate": 0.8}, seeds=[], requirements={}, activation={})
        ]

        patterns = self.engine._identify_behavioral_patterns("core.event", capabilities)

        # Should identify event-driven architecture pattern
        assert isinstance(patterns, list)
        assert len(patterns) > 0

        event_pattern = None
        for pattern in patterns:
            if "Eventos" in pattern.name:
                event_pattern = pattern
                break

        assert event_pattern is not None
        assert event_pattern.name == "Arquitetura Dirigida por Eventos"
        assert event_pattern.domain == "core.event"
        assert event_pattern.pattern_type == "behavioral"
        assert event_pattern.confidence >= 0.7
        assert "event" in pattern.description.lower()

    def test_identify_behavioral_patterns_with_error_capabilities(self) -> None:
        """Test identifying behavioral patterns with error-related capabilities."""
        # Create mock capabilities
        capabilities = [
            Capability(id="core.error.handler", name="Error Handler", category="vertical",
                      description="Handles system errors gracefully", maturity="established",
                      metrics={"success_rate": 0.95}, seeds=[], requirements={}, activation={}),
            Capability(id="core.error.recovery", name="Error Recovery", category="vertical",
                      description="Recovers from system errors", maturity="baseline",
                      metrics={"success_rate": 0.85}, seeds=[], requirements={}, activation={})
        ]

        patterns = self.engine._identify_behavioral_patterns("core.error", capabilities)

        # Should identify error handling pattern
        assert isinstance(patterns, list)
        assert len(patterns) > 0

        error_pattern = None
        for pattern in patterns:
            if "Erros" in pattern.name:
                error_pattern = pattern
                break

        assert error_pattern is not None
        assert error_pattern.name == "Tratamento e Recuperação de Erros"
        assert error_pattern.domain == "core.error"
        assert error_pattern.pattern_type == "behavioral"
        assert error_pattern.confidence >= 0.8
        # Check that description contains relevant terms (adjusted for Portuguese)
        assert "error" in pattern.description.lower() or "erro" in pattern.description.lower() or "falhas" in pattern.description.lower() or "recupera" in pattern.description.lower()

    def test_identify_data_processing_patterns_with_validation_capabilities(self) -> None:
        """Test identifying data processing patterns with validation-related capabilities."""
        # Create mock capabilities
        capabilities = [
            Capability(id="data.validation.validator", name="Data Validator", category="vertical",
                      description="Validates input data", maturity="established",
                      metrics={"success_rate": 0.95}, seeds=[], requirements={}, activation={}),
            Capability(id="data.validation.cleaner", name="Data Cleaner", category="vertical",
                      description="Cleans invalid data", maturity="baseline",
                      metrics={"success_rate": 0.9}, seeds=[], requirements={}, activation={})
        ]

        patterns = self.engine._identify_data_processing_patterns("data.validation", capabilities)

        # Should identify data validation pattern
        assert isinstance(patterns, list)
        assert len(patterns) > 0

        validation_pattern = None
        for pattern in patterns:
            if "Validação" in pattern.name:
                validation_pattern = pattern
                break

        assert validation_pattern is not None
        assert validation_pattern.name == "Validação e Limpeza de Dados"
        assert validation_pattern.domain == "data.validation"
        assert validation_pattern.pattern_type == "data_processing"
        assert validation_pattern.confidence >= 0.8
        # Check that description contains relevant terms (adjusted for Portuguese)
        assert "validation" in pattern.description.lower() or "validação" in pattern.description.lower() or "verificados" in pattern.description.lower() or "sanitizados" in pattern.description.lower()

    def test_identify_data_processing_patterns_with_batch_capabilities(self) -> None:
        """Test identifying data processing patterns with batch-related capabilities."""
        # Create mock capabilities
        capabilities = [
            Capability(id="data.batch.processor", name="Batch Processor", category="vertical",
                      description="Processes data in batches", maturity="established",
                      metrics={"success_rate": 0.9}, seeds=[], requirements={}, activation={}),
            Capability(id="data.batch.scheduler", name="Batch Scheduler", category="vertical",
                      description="Schedules batch operations", maturity="baseline",
                      metrics={"success_rate": 0.85}, seeds=[], requirements={}, activation={})
        ]

        patterns = self.engine._identify_data_processing_patterns("data.batch", capabilities)

        # Should identify batch processing pattern
        assert isinstance(patterns, list)
        assert len(patterns) > 0

        batch_pattern = None
        for pattern in patterns:
            if "Lotes" in pattern.name:
                batch_pattern = pattern
                break

        assert batch_pattern is not None
        assert batch_pattern.name == "Processamento em Lotes"
        assert batch_pattern.domain == "data.batch"
        assert batch_pattern.pattern_type == "data_processing"
        assert batch_pattern.confidence >= 0.7
        # Check that description contains relevant terms (adjusted for Portuguese)
        assert "batch" in pattern.description.lower() or "lote" in pattern.description.lower() or "grupos" in pattern.description.lower() or "eficiência" in pattern.description.lower()

    def test_calculate_pattern_similarity_same_type(self) -> None:
        """Test calculating similarity between patterns of the same type."""
        pattern1 = DomainPattern(
            id="pattern1",
            name="Pattern 1",
            description="Configuration driven behavior pattern with validation",
            domain="domain1",
            pattern_type="structural",
            confidence=0.9,
            examples=["Config loading", "Validation"],
            related_capabilities=["cap1"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        pattern2 = DomainPattern(
            id="pattern2",
            name="Pattern 2",
            description="Structural configuration pattern with validation checks",
            domain="domain2",
            pattern_type="structural",
            confidence=0.85,
            examples=["Config parsing", "Validation"],
            related_capabilities=["cap2"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        similarity = self.engine._calculate_pattern_similarity(pattern1, pattern2)

        # Should have high similarity due to same pattern type and overlapping descriptions
        assert isinstance(similarity, float)
        assert similarity >= 0.6  # Should be considered similar (or equal to threshold)
        assert similarity <= 1.0  # Should not exceed maximum

    def test_calculate_pattern_similarity_different_types(self) -> None:
        """Test calculating similarity between patterns of different types."""
        pattern1 = DomainPattern(
            id="pattern1",
            name="Pattern 1",
            description="Configuration driven behavior",
            domain="domain1",
            pattern_type="structural",
            confidence=0.9,
            examples=["Config loading"],
            related_capabilities=["cap1"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        pattern2 = DomainPattern(
            id="pattern2",
            name="Pattern 2",
            description="Event-driven architecture pattern",
            domain="domain2",
            pattern_type="behavioral",
            confidence=0.8,
            examples=["Event handling"],
            related_capabilities=["cap2"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        similarity = self.engine._calculate_pattern_similarity(pattern1, pattern2)

        # Should have lower similarity due to different pattern types
        assert isinstance(similarity, float)
        assert similarity >= 0.0
        assert similarity < 0.7  # Should not be considered very similar

    def test_generate_adaptation_strategy_known_combinations(self) -> None:
        """Test generating adaptation strategies for known pattern combinations."""
        # Test structural to structural
        pattern1 = DomainPattern(
            id="pattern1", name="Pattern 1", description="Config pattern",
            domain="domain1", pattern_type="structural", confidence=0.9,
            examples=["Config"], related_capabilities=["cap1"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        pattern2 = DomainPattern(
            id="pattern2", name="Pattern 2", description="Data pattern",
            domain="domain2", pattern_type="data_processing", confidence=0.8,
            examples=["Data"], related_capabilities=["cap2"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        strategy = self.engine._generate_adaptation_strategy(pattern1, pattern2)

        # Should generate appropriate strategy
        assert isinstance(strategy, str)
        assert len(strategy) > 0
        # Check that strategy contains relevant terms (adjusted for Portuguese)
        assert "Adaptação" in strategy or "adapt" in strategy.lower() or "Transformar" in strategy or "transform" in strategy.lower()

    def test_create_skill_from_mapping(self) -> None:
        """Test creating a transferred skill from a mapping."""
        # Create test patterns
        source_pattern = DomainPattern(
            id="source_pattern_001",
            name="Source Pattern",
            description="Source pattern description",
            domain="source.domain",
            pattern_type="structural",
            confidence=0.9,
            examples=["Example"],
            related_capabilities=["cap1"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        target_pattern = DomainPattern(
            id="target_pattern_001",
            name="Target Pattern",
            description="Target pattern description",
            domain="target.domain",
            pattern_type="data_processing",
            confidence=0.85,
            examples=["Example"],
            related_capabilities=["cap2"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Save patterns to disk for loading
        self.engine._save_domain_pattern(source_pattern)
        self.engine._save_domain_pattern(target_pattern)

        # Create mapping
        mapping = CrossDomainMapping(
            id="test_mapping_001",
            source_domain="source.domain",
            target_domain="target.domain",
            source_pattern="source_pattern_001",
            target_pattern="target_pattern_001",
            similarity_score=0.75,
            adaptation_strategy="Transform configuration to data processing pipeline",
            confidence=0.85,
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Create transferred skill
        skill = self.engine._create_skill_from_mapping(mapping)

        # Should create valid transferred skill
        assert skill is not None
        assert isinstance(skill, TransferredSkill)
        assert "Transferência" in skill.name
        assert skill.source_domains == ["source.domain"]
        assert skill.target_domain == "target.domain"
        assert skill.confidence == 0.85
        assert "TRANSFERÊNCIA" in skill.implementation_plan
        assert "Etapas de Implementação" in skill.implementation_plan

    def test_save_and_load_domain_pattern(self) -> None:
        """Test saving and loading domain patterns."""
        pattern = DomainPattern(
            id="test_save_pattern_001",
            name="Save Test Pattern",
            description="Test pattern for saving/loading",
            domain="test.save",
            pattern_type="structural",
            confidence=0.9,
            examples=["Save example"],
            related_capabilities=["save.cap"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Save pattern
        self.engine._save_domain_pattern(pattern)

        # Verify file was created
        pattern_file = self.engine.patterns_path / f"{pattern.id}.json"
        assert pattern_file.exists()

        # Load pattern
        loaded_pattern = self.engine._load_domain_pattern(pattern.id)

        # Should load identical pattern
        assert loaded_pattern is not None
        assert loaded_pattern.id == pattern.id
        assert loaded_pattern.name == pattern.name
        assert loaded_pattern.description == pattern.description
        assert loaded_pattern.domain == pattern.domain
        assert loaded_pattern.pattern_type == pattern.pattern_type
        assert loaded_pattern.confidence == pattern.confidence


class TestIntegration:
    """Integration tests for transfer learning system."""

    def test_integrate_transfer_learning(self) -> None:
        """Test integrating transfer learning with existing system."""
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

        # Create mock meta engine with capabilities
        mock_meta_engine = Mock(spec=MetaCapabilityEngine)
        mock_capability_registry = Mock(spec=CapabilityRegistry)

        # Create test capabilities from different domains
        test_capabilities = {
            "core.config": Capability(
                id="core.config.loader",
                name="Config Loader",
                category="vertical",
                description="Loads configuration files",
                maturity="established",
                metrics={"success_rate": 0.9},
                seeds=[],
                requirements={},
                activation={}
            ),
            "data.validation": Capability(
                id="data.validation.validator",
                name="Data Validator",
                category="vertical",
                description="Validates input data",
                maturity="baseline",
                metrics={"success_rate": 0.85},
                seeds=[],
                requirements={},
                activation={}
            )
        }

        mock_capability_registry._by_id = test_capabilities
        mock_meta_engine.capability_registry = mock_capability_registry

        # Test integration
        transferred_skills = integrate_transfer_learning(mock_config, mock_meta_engine)

        # Should return list of transferred skills (may be empty depending on mappings)
        assert isinstance(transferred_skills, list)

        # Integration should complete without errors
        assert True  # Test passes if no exceptions are raised
