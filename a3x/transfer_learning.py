"""Transfer learning capabilities for cross-domain knowledge application in SeedAI."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .capabilities import Capability
from .config import AgentConfig
from .meta_capabilities import MetaCapabilityEngine


@dataclass
class DomainPattern:
    """Represents a pattern identified in a domain."""

    id: str
    name: str
    description: str
    domain: str
    pattern_type: str  # structural, behavioral, data_processing, etc.
    confidence: float
    examples: list[str]
    related_capabilities: list[str]
    created_at: str


@dataclass
class CrossDomainMapping:
    """Represents a mapping between patterns from different domains."""

    id: str
    source_domain: str
    target_domain: str
    source_pattern: str
    target_pattern: str
    similarity_score: float
    adaptation_strategy: str
    confidence: float
    created_at: str


@dataclass
class TransferredSkill:
    """Represents a skill created through knowledge transfer."""

    id: str
    name: str
    description: str
    source_domains: list[str]
    target_domain: str
    base_skill: str
    adaptations: list[str]
    confidence: float
    implementation_plan: str
    created_at: str


class TransferLearningEngine:
    """Engine for cross-domain knowledge transfer and skill adaptation."""

    def __init__(self, config: AgentConfig, meta_engine: MetaCapabilityEngine) -> None:
        self.config = config
        self.meta_engine = meta_engine
        self.workspace_root = Path(config.workspace.root).resolve()
        self.transfer_path = self.workspace_root / "seed" / "transfer_learning"
        self.transfer_path.mkdir(parents=True, exist_ok=True)

        # Load existing patterns and mappings
        self.patterns_path = self.transfer_path / "patterns"
        self.patterns_path.mkdir(parents=True, exist_ok=True)

        self.mappings_path = self.transfer_path / "mappings"
        self.mappings_path.mkdir(parents=True, exist_ok=True)

        self.transferred_skills_path = self.transfer_path / "transferred_skills"
        self.transferred_skills_path.mkdir(parents=True, exist_ok=True)

    def identify_domain_patterns(self, domain: str, capabilities: list[Capability]) -> list[DomainPattern]:
        """Identify patterns in a specific domain based on capabilities."""
        patterns = []

        # Structural patterns (based on code structure)
        structural_patterns = self._identify_structural_patterns(domain, capabilities)
        patterns.extend(structural_patterns)

        # Behavioral patterns (based on execution behavior)
        behavioral_patterns = self._identify_behavioral_patterns(domain, capabilities)
        patterns.extend(behavioral_patterns)

        # Data processing patterns (based on data handling)
        data_patterns = self._identify_data_processing_patterns(domain, capabilities)
        patterns.extend(data_patterns)

        # Save patterns to disk
        for pattern in patterns:
            self._save_domain_pattern(pattern)

        return patterns

    def _identify_structural_patterns(self, domain: str, capabilities: list[Capability]) -> list[DomainPattern]:
        """Identify structural patterns in a domain."""
        patterns = []

        # Pattern: Configuration-driven behavior
        if any("config" in cap.id.lower() for cap in capabilities):
            patterns.append(DomainPattern(
                id=f"pattern_{domain.replace('.', '_')}_config_driven_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="Comportamento Dirigido por Configuração",
                description="Padrão onde o comportamento é controlado por arquivos de configuração",
                domain=domain,
                pattern_type="structural",
                confidence=0.9,
                examples=["Leitura de YAML/JSON", "Injeção de dependências", "Factory patterns"],
                related_capabilities=[cap.id for cap in capabilities if "config" in cap.id.lower()],
                created_at=datetime.now(timezone.utc).isoformat()
            ))

        # Pattern: Pipeline processing
        if any("pipeline" in cap.id.lower() or "process" in cap.id.lower() for cap in capabilities):
            patterns.append(DomainPattern(
                id=f"pattern_{domain.replace('.', '_')}_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="Processamento em Pipeline",
                description="Padrão onde dados fluem através de uma sequência de etapas de processamento",
                domain=domain,
                pattern_type="structural",
                confidence=0.85,
                examples=["ETL pipelines", "Data processing chains", "Workflow engines"],
                related_capabilities=[cap.id for cap in capabilities if "pipeline" in cap.id.lower() or "process" in cap.id.lower()],
                created_at=datetime.now(timezone.utc).isoformat()
            ))

        return patterns

    def _identify_behavioral_patterns(self, domain: str, capabilities: list[Capability]) -> list[DomainPattern]:
        """Identify behavioral patterns in a domain."""
        patterns = []

        # Pattern: Event-driven architecture
        if any("event" in cap.id.lower() or "trigger" in cap.id.lower() for cap in capabilities):
            patterns.append(DomainPattern(
                id=f"pattern_{domain.replace('.', '_')}_event_driven_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="Arquitetura Dirigida por Eventos",
                description="Padrão onde o fluxo é controlado por eventos e callbacks",
                domain=domain,
                pattern_type="behavioral",
                confidence=0.8,
                examples=["Event listeners", "Callback chains", "Pub/Sub patterns"],
                related_capabilities=[cap.id for cap in capabilities if "event" in cap.id.lower() or "trigger" in cap.id.lower()],
                created_at=datetime.now(timezone.utc).isoformat()
            ))

        # Pattern: Error handling and recovery
        if any("error" in cap.id.lower() or "recover" in cap.id.lower() for cap in capabilities):
            patterns.append(DomainPattern(
                id=f"pattern_{domain.replace('.', '_')}_error_handling_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="Tratamento e Recuperação de Erros",
                description="Padrão onde falhas são tratadas com mecanismos de recuperação",
                domain=domain,
                pattern_type="behavioral",
                confidence=0.85,
                examples=["Try/catch blocks", "Retry mechanisms", "Fallback strategies"],
                related_capabilities=[cap.id for cap in capabilities if "error" in cap.id.lower() or "recover" in cap.id.lower()],
                created_at=datetime.now(timezone.utc).isoformat()
            ))

        return patterns

    def _identify_data_processing_patterns(self, domain: str, capabilities: list[Capability]) -> list[DomainPattern]:
        """Identify data processing patterns in a domain."""
        patterns = []

        # Pattern: Data validation and cleaning
        if any("validate" in cap.id.lower() or "clean" in cap.id.lower() for cap in capabilities):
            patterns.append(DomainPattern(
                id=f"pattern_{domain.replace('.', '_')}_data_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="Validação e Limpeza de Dados",
                description="Padrão onde dados são verificados e sanitizados antes do processamento",
                domain=domain,
                pattern_type="data_processing",
                confidence=0.9,
                examples=["Input validation", "Data sanitization", "Schema validation"],
                related_capabilities=[cap.id for cap in capabilities if "validate" in cap.id.lower() or "clean" in cap.id.lower()],
                created_at=datetime.now(timezone.utc).isoformat()
            ))

        # Pattern: Batch processing
        if any("batch" in cap.id.lower() for cap in capabilities):
            patterns.append(DomainPattern(
                id=f"pattern_{domain.replace('.', '_')}_batch_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="Processamento em Lotes",
                description="Padrão onde dados são processados em grupos para eficiência",
                domain=domain,
                pattern_type="data_processing",
                confidence=0.75,
                examples=["Batch jobs", "Bulk operations", "Chunked processing"],
                related_capabilities=[cap.id for cap in capabilities if "batch" in cap.id.lower()],
                created_at=datetime.now(timezone.utc).isoformat()
            ))

        return patterns

    def find_cross_domain_mappings(self, source_patterns: list[DomainPattern],
                                   target_patterns: list[DomainPattern]) -> list[CrossDomainMapping]:
        """Find mappings between patterns from different domains."""
        mappings = []

        for source_pattern in source_patterns:
            for target_pattern in target_patterns:
                # Calculate similarity score based on pattern type and description
                similarity = self._calculate_pattern_similarity(source_pattern, target_pattern)

                if similarity > 0.6:  # Only consider strong similarities
                    mapping = CrossDomainMapping(
                        id=f"mapping_{source_pattern.domain.replace('.', '_')}_{target_pattern.domain.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        source_domain=source_pattern.domain,
                        target_domain=target_pattern.domain,
                        source_pattern=source_pattern.id,
                        target_pattern=target_pattern.id,
                        similarity_score=similarity,
                        adaptation_strategy=self._generate_adaptation_strategy(source_pattern, target_pattern),
                        confidence=min(similarity + 0.1, 1.0),  # Boost confidence slightly
                        created_at=datetime.now(timezone.utc).isoformat()
                    )
                    mappings.append(mapping)

        # Save mappings to disk
        for mapping in mappings:
            self._save_cross_domain_mapping(mapping)

        return mappings

    def _calculate_pattern_similarity(self, pattern1: DomainPattern, pattern2: DomainPattern) -> float:
        """Calculate similarity between two patterns."""
        # Simple similarity calculation based on pattern type and description overlap
        if pattern1.pattern_type == pattern2.pattern_type:
            # Same pattern type gives higher similarity
            base_similarity = 0.7
        else:
            # Different pattern types have lower base similarity
            base_similarity = 0.3

        # Check for common words in descriptions
        desc1_words = set(pattern1.description.lower().split())
        desc2_words = set(pattern2.description.lower().split())

        common_words = len(desc1_words.intersection(desc2_words))
        total_words = len(desc1_words.union(desc2_words))

        if total_words > 0:
            description_similarity = common_words / total_words
        else:
            description_similarity = 0.0

        # Combine base similarity with description similarity
        combined_similarity = (base_similarity + description_similarity) / 2

        return min(combined_similarity, 1.0)

    def _generate_adaptation_strategy(self, source_pattern: DomainPattern, target_pattern: DomainPattern) -> str:
        """Generate adaptation strategy for transferring knowledge between patterns."""
        strategies = {
            ("structural", "structural"): "Adaptar a estrutura de configuração para o novo domínio",
            ("behavioral", "behavioral"): "Transferir mecanismos de tratamento de eventos/erros com adaptações contextuais",
            ("data_processing", "data_processing"): "Aplicar técnicas de validação/processamento com modificação de formatos",
            ("structural", "data_processing"): "Transformar estruturas de configuração em pipelines de processamento",
            ("data_processing", "structural"): "Converter pipelines de processamento em estruturas configuráveis",
        }

        key = (source_pattern.pattern_type, target_pattern.pattern_type)
        return strategies.get(key, "Adaptação genérica baseada em similaridade de padrões")

    def create_transferred_skills(self, mappings: list[CrossDomainMapping]) -> list[TransferredSkill]:
        """Create new skills by transferring knowledge between domains."""
        transferred_skills = []

        for mapping in mappings:
            # Create a transferred skill based on the mapping
            skill = self._create_skill_from_mapping(mapping)
            if skill:
                transferred_skills.append(skill)
                self._save_transferred_skill(skill)

        return transferred_skills

    def _create_skill_from_mapping(self, mapping: CrossDomainMapping) -> TransferredSkill | None:
        """Create a transferred skill from a cross-domain mapping."""
        # Get pattern details
        source_pattern = self._load_domain_pattern(mapping.source_pattern)
        target_pattern = self._load_domain_pattern(mapping.target_pattern)

        if not source_pattern or not target_pattern:
            return None

        # Create skill name combining both domains
        skill_name = f"Transferência de {source_pattern.name} para {target_pattern.domain}"

        # Generate implementation plan based on adaptation strategy
        implementation_plan = f"""
        TRANSFERÊNCIA DE CONHECIMENTO AUTOMÁTICA
        
        Domínio Fonte: {mapping.source_domain}
        Domínio Alvo: {mapping.target_domain}
        Padrão Fonte: {source_pattern.name}
        Padrão Alvo: {target_pattern.name}
        Estratégia de Adaptação: {mapping.adaptation_strategy}
        
        Etapas de Implementação:
        1. Analisar padrões do domínio fonte ({source_pattern.domain})
        2. Identificar pontos de aplicação no domínio alvo ({target_pattern.domain})
        3. Adaptar mecanismos conforme estratégia: {mapping.adaptation_strategy}
        4. Validar transferência com casos de teste do domínio alvo
        5. Refinar implementação baseada em feedback
        """

        skill = TransferredSkill(
            id=f"transferred_skill_{mapping.source_domain.replace('.', '_')}_{mapping.target_domain.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=skill_name,
            description=f"Habilidade criada pela transferência de conhecimento de {mapping.source_domain} para {mapping.target_domain}",
            source_domains=[mapping.source_domain],
            target_domain=mapping.target_domain,
            base_skill=source_pattern.id,
            adaptations=[mapping.adaptation_strategy],
            confidence=mapping.confidence,
            implementation_plan=implementation_plan.strip(),
            created_at=datetime.now(timezone.utc).isoformat()
        )

        return skill

    def _save_domain_pattern(self, pattern: DomainPattern) -> None:
        """Save a domain pattern to disk."""
        pattern_file = self.patterns_path / f"{pattern.id}.json"
        with pattern_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(pattern), f, ensure_ascii=False, indent=2)

    def _save_cross_domain_mapping(self, mapping: CrossDomainMapping) -> None:
        """Save a cross-domain mapping to disk."""
        mapping_file = self.mappings_path / f"{mapping.id}.json"
        with mapping_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(mapping), f, ensure_ascii=False, indent=2)

    def _save_transferred_skill(self, skill: TransferredSkill) -> None:
        """Save a transferred skill to disk."""
        skill_file = self.transferred_skills_path / f"{skill.id}.json"
        with skill_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(skill), f, ensure_ascii=False, indent=2)

    def _load_domain_pattern(self, pattern_id: str) -> DomainPattern | None:
        """Load a domain pattern from disk."""
        pattern_file = self.patterns_path / f"{pattern_id}.json"
        if pattern_file.exists():
            try:
                with pattern_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    return DomainPattern(**data)
            except Exception:
                pass
        return None


# Integration with existing system
def integrate_transfer_learning(config: AgentConfig, meta_engine: MetaCapabilityEngine) -> list[TransferredSkill]:
    """Integrate transfer learning capabilities into the existing system."""
    # Create transfer learning engine
    transfer_engine = TransferLearningEngine(config, meta_engine)

    # Get capability registry
    capability_registry = meta_engine.capability_registry

    # Group capabilities by domain
    domains = defaultdict(list)
    for capability in capability_registry._by_id.values():
        if "." in capability.id:
            domain = capability.id.split(".")[0]
            domains[domain].append(capability)

    # Identify patterns in each domain
    domain_patterns = {}
    for domain, capabilities in domains.items():
        patterns = transfer_engine.identify_domain_patterns(domain, capabilities)
        domain_patterns[domain] = patterns

    # Find cross-domain mappings
    all_mappings = []
    domain_list = list(domain_patterns.keys())

    for i in range(len(domain_list)):
        for j in range(i + 1, len(domain_list)):
            source_domain = domain_list[i]
            target_domain = domain_list[j]

            mappings = transfer_engine.find_cross_domain_mappings(
                domain_patterns[source_domain],
                domain_patterns[target_domain]
            )
            all_mappings.extend(mappings)

    # Create transferred skills
    transferred_skills = transfer_engine.create_transferred_skills(all_mappings)

    return transferred_skills


__all__ = [
    "TransferLearningEngine",
    "DomainPattern",
    "CrossDomainMapping",
    "TransferredSkill",
    "integrate_transfer_learning",
]
