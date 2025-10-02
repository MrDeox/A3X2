"""Sistema de geração autônoma de objetivos com motivação intrínseca.

Este módulo implementa o AutonomousGoalGenerator que gera objetivos autônomos
baseados em análise de capacidades, exploração curiosa e auto-otimização.
Integra-se com o sistema de capacidades, seeds e planejamento autônomo existente.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .capabilities import CapabilityRegistry
from .autonomous_planner import AutonomousPlanner, EvolutionInsight
from .seeds import SeedBacklog, Seed
from .autoeval import AutoEvaluator, EvaluationSeed


@dataclass
class IntrinsicMotivationProfile:
    """Perfil de motivação intrínseca para guiar geração de objetivos."""

    curiosity_weight: float = 0.3
    competence_weight: float = 0.25
    autonomy_weight: float = 0.2
    relatedness_weight: float = 0.15
    exploration_bias: float = 0.1

    # Parâmetros específicos para exploração curiosa
    novelty_threshold: float = 0.6
    uncertainty_tolerance: float = 0.4
    domain_diversity_factor: float = 0.3


@dataclass
class AutonomousGoal:
    """Objetivo autônomo gerado pelo sistema de motivação intrínseca."""

    id: str
    title: str
    description: str
    goal_type: str  # "capability_gap", "curiosity_exploration", "self_optimization", "domain_expansion"
    priority: str
    estimated_impact: float
    required_capabilities: list[str]
    success_criteria: list[str]
    motivation_factors: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CapabilityGapAnalysis:
    """Análise de lacunas de capacidades identificadas."""

    capability_id: str
    gap_type: str  # "performance", "coverage", "integration", "evolution"
    severity: float  # 0.0 to 1.0
    description: str
    recommended_actions: list[str]
    potential_impact: float
    confidence: float


@dataclass
class CuriosityOpportunity:
    """Oportunidade de exploração curiosa identificada."""

    domain: str
    opportunity_type: str  # "novelty", "uncertainty", "diversity", "integration"
    novelty_score: float
    uncertainty_score: float
    exploration_value: float
    description: str
    suggested_approach: str


class AutonomousGoalGenerator:
    """Gerador autônomo de objetivos com sistema de motivação intrínseca.

    Esta classe implementa um sistema avançado de geração de objetivos que:
    - Analisa lacunas de capacidades e gera objetivos para preenchê-las
    - Implementa exploração curiosa baseada em novelty e uncertainty
    - Gera objetivos de auto-otimização baseados em análise de performance
    - Identifica oportunidades de expansão de domínio
    - Integra-se com o planejador autônomo existente
    """

    def __init__(
        self,
        workspace_root: Path,
        motivation_profile: IntrinsicMotivationProfile | None = None,
        random_seed: int | None = None
    ) -> None:
        """Inicializa o gerador de objetivos autônomos.

        Args:
            workspace_root: Diretório raiz do workspace
            motivation_profile: Perfil de motivação intrínseca personalizado
            random_seed: Semente para reprodutibilidade (None para aleatório)
        """
        self.workspace_root = workspace_root
        self.motivation_profile = motivation_profile or IntrinsicMotivationProfile()
        self.autonomous_planner = AutonomousPlanner(workspace_root)
        self.backlog_path = workspace_root / "seed" / "backlog.yaml"
        self.capabilities_path = workspace_root / "seed" / "capabilities.yaml"
        self.evaluations_path = workspace_root / "seed" / "evaluations" / "run_evaluations.jsonl"
        self.metrics_path = workspace_root / "seed" / "metrics" / "history.json"

        # Estado interno para rastreamento
        self._goal_history: list[AutonomousGoal] = []
        self._exploration_memory: dict[str, float] = {}
        self._capability_baselines: dict[str, float] = {}

        # Configurar reprodutibilidade
        if random_seed is not None:
            random.seed(random_seed)

        # Carregar baselines históricos
        self._load_capability_baselines()

    def generate_autonomous_goals(self, context: dict[str, Any] | None = None) -> list[AutonomousGoal]:
        """Gera uma lista de objetivos autônomos baseada em análise completa.

        Args:
            context: Contexto adicional para guiar geração (opcional)

        Returns:
            Lista de objetivos autônomos gerados
        """
        goals = []
        timestamp = datetime.now(timezone.utc).isoformat()

        print("🧠 INICIANDO GERAÇÃO AUTÔNOMA DE OBJETIVOS")
        print("=" * 60)

        # 1. Análise de lacunas de capacidades
        print("🔍 1. Analisando lacunas de capacidades...")
        capability_gaps = self._analyze_capability_gaps()
        gap_goals = self._generate_capability_gap_goals(capability_gaps, timestamp)
        goals.extend(gap_goals)
        print(f"   - {len(gap_goals)} objetivos de lacunas identificados")

        # 2. Oportunidades de exploração curiosa
        print("🎯 2. Identificando oportunidades de exploração curiosa...")
        curiosity_opportunities = self._identify_curiosity_opportunities()
        curiosity_goals = self._generate_curiosity_goals(curiosity_opportunities, timestamp)
        goals.extend(curiosity_goals)
        print(f"   - {len(curiosity_goals)} objetivos de exploração identificados")

        # 3. Objetivos de auto-otimização
        print("⚡ 3. Gerando objetivos de auto-otimização...")
        optimization_goals = self._generate_self_optimization_goals(timestamp)
        goals.extend(optimization_goals)
        print(f"   - {len(optimization_goals)} objetivos de otimização identificados")

        # 4. Oportunidades de expansão de domínio
        print("🌍 4. Identificando oportunidades de expansão de domínio...")
        domain_goals = self._generate_domain_expansion_goals(timestamp)
        goals.extend(domain_goals)
        print(f"   - {len(domain_goals)} objetivos de expansão identificados")

        # 5. Aplicar filtros e priorização
        print("🎛️  5. Aplicando filtros e priorização...")
        filtered_goals = self._filter_and_prioritize_goals(goals, context)
        print(f"   - {len(filtered_goals)} objetivos selecionados após filtros")

        # 6. Registrar histórico
        self._goal_history.extend(filtered_goals)

        print(f"\n✅ GERAÇÃO DE OBJETIVOS CONCLUÍDA: {len(filtered_goals)} objetivos gerados")
        print("=" * 60)

        return filtered_goals

    def _analyze_capability_gaps(self) -> list[CapabilityGapAnalysis]:
        """Analisa lacunas em capacidades baseado em métricas e uso."""
        gaps = []

        # Carregar dados de capacidades
        if not self.capabilities_path.exists():
            return gaps

        try:
            registry = CapabilityRegistry.from_yaml(self.capabilities_path)
        except Exception:
            return gaps

        # Carregar métricas históricas
        metrics_history = {}
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, encoding="utf-8") as f:
                    metrics_history = json.load(f)
            except Exception:
                pass

        # Carregar avaliações para análise de uso
        capability_usage = {}
        if self.evaluations_path.exists():
            try:
                with open(self.evaluations_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            evaluation = json.loads(line.strip())
                            capabilities = evaluation.get("capabilities", [])
                            for cap in capabilities:
                                capability_usage[cap] = capability_usage.get(cap, 0) + 1
                        except Exception:
                            continue
            except Exception:
                pass

        # Analisar cada capacidade
        for capability in registry.list():
            cap_id = capability.id

            # 1. Análise de performance baseada em métricas
            performance_gaps = self._analyze_performance_gaps(capability, metrics_history)

            # 2. Análise de cobertura baseada em uso
            usage_gaps = self._analyze_usage_gaps(capability, capability_usage)

            # 3. Análise de integração baseada em dependências
            integration_gaps = self._analyze_integration_gaps(capability, registry)

            # 4. Análise de evolução baseada em baselines
            evolution_gaps = self._analyze_evolution_gaps(capability, metrics_history)

            gaps.extend(performance_gaps)
            gaps.extend(usage_gaps)
            gaps.extend(integration_gaps)
            gaps.extend(evolution_gaps)

        # Ordenar por severidade
        gaps.sort(key=lambda x: x.severity, reverse=True)

        return gaps[:20]  # Limitar a top 20 gaps

    def _analyze_performance_gaps(
        self,
        capability: Any,
        metrics_history: dict[str, Any]
    ) -> list[CapabilityGapAnalysis]:
        """Analisa lacunas de performance para uma capacidade."""
        gaps = []
        cap_id = capability.id

        # Verificar métricas específicas por categoria
        if cap_id.startswith("core.diffing"):
            success_rate = self._get_latest_metric(metrics_history, "apply_patch_success_rate")
            if success_rate is not None and success_rate < 0.8:
                gaps.append(CapabilityGapAnalysis(
                    capability_id=cap_id,
                    gap_type="performance",
                    severity=1.0 - success_rate,
                    description=f"Taxa de sucesso baixa em diffing: {success_rate:.2f}",
                    recommended_actions=[
                        "Melhorar algoritmos de detecção de diferenças",
                        "Implementar validação prévia de patches",
                        "Adicionar mecanismos de recuperação de erro"
                    ],
                    potential_impact=0.8,
                    confidence=0.9
                ))

        elif cap_id.startswith("core.testing"):
            test_success = self._get_latest_metric(metrics_history, "tests_success_rate")
            if test_success is not None and test_success < 0.85:
                gaps.append(CapabilityGapAnalysis(
                    capability_id=cap_id,
                    gap_type="performance",
                    severity=1.0 - test_success,
                    description=f"Taxa de sucesso baixa em testes: {test_success:.2f}",
                    recommended_actions=[
                        "Expandir cobertura de testes",
                        "Melhorar geração de casos de teste",
                        "Otimizar execução de testes"
                    ],
                    potential_impact=0.7,
                    confidence=0.85
                ))

        return gaps

    def _analyze_usage_gaps(
        self,
        capability: Any,
        capability_usage: dict[str, int]
    ) -> list[CapabilityGapAnalysis]:
        """Analisa lacunas de uso para uma capacidade."""
        gaps = []
        cap_id = capability.id

        usage_count = capability_usage.get(cap_id, 0)

        # Capacidades pouco utilizadas podem indicar problemas de integração
        if usage_count < 3 and capability.maturity in ["established", "advanced"]:
            gaps.append(CapabilityGapAnalysis(
                capability_id=cap_id,
                gap_type="coverage",
                severity=0.6,
                description=f"Capacidade subutilizada: usada apenas {usage_count} vezes",
                recommended_actions=[
                    "Investigar barreiras de uso",
                    "Melhorar documentação e exemplos",
                    "Integrar melhor com outros componentes"
                ],
                potential_impact=0.5,
                confidence=0.7
            ))

        return gaps

    def _analyze_integration_gaps(
        self,
        capability: Any,
        registry: CapabilityRegistry
    ) -> list[CapabilityGapAnalysis]:
        """Analisa lacunas de integração para uma capacidade."""
        gaps = []
        cap_id = capability.id

        # Verificar se há capacidades relacionadas que podem se beneficiar de integração
        related_capabilities = [
            c for c in registry.list()
            if c.category == capability.category and c.id != cap_id
        ]

        if related_capabilities and not self._has_integration_evidence(cap_id, related_capabilities):
            gaps.append(CapabilityGapAnalysis(
                capability_id=cap_id,
                gap_type="integration",
                severity=0.4,
                description=f"Possível falta de integração com capacidades relacionadas",
                recommended_actions=[
                    "Explorar oportunidades de integração",
                    "Identificar APIs compartilhadas",
                    "Criar padrões de comunicação comuns"
                ],
                potential_impact=0.6,
                confidence=0.5
            ))

        return gaps

    def _analyze_evolution_gaps(
        self,
        capability: Any,
        metrics_history: dict[str, Any]
    ) -> list[CapabilityGapAnalysis]:
        """Analisa lacunas de evolução para uma capacidade."""
        gaps = []
        cap_id = capability.id

        # Comparar com baselines históricos
        baseline = self._capability_baselines.get(cap_id, {})
        current_metrics = {}

        # Extrair métricas atuais relevantes
        for metric_name in baseline.keys():
            current_metrics[metric_name] = self._get_latest_metric(metrics_history, metric_name)

        # Verificar regressões
        for metric_name, baseline_value in baseline.items():
            current_value = current_metrics.get(metric_name)
            if current_value is not None and current_value < baseline_value * 0.9:  # 10% piora
                gaps.append(CapabilityGapAnalysis(
                    capability_id=cap_id,
                    gap_type="evolution",
                    severity=(baseline_value - current_value) / baseline_value,
                    description=f"Regressão detectada em {metric_name}: {current_value:.3f} < {baseline_value:.3f}",
                    recommended_actions=[
                        "Investigar causa da regressão",
                        "Reverter mudanças recentes se necessário",
                        "Implementar medidas preventivas"
                    ],
                    potential_impact=0.7,
                    confidence=0.8
                ))

        return gaps

    def _generate_capability_gap_goals(
        self,
        gaps: list[CapabilityGapAnalysis],
        timestamp: str
    ) -> list[AutonomousGoal]:
        """Gera objetivos baseados em lacunas de capacidades."""
        goals = []

        for gap in gaps:
            # Calcular motivação baseada na severidade e impacto
            motivation_factors = {
                "competence": gap.severity * self.motivation_profile.competence_weight,
                "autonomy": gap.potential_impact * self.motivation_profile.autonomy_weight,
                "relatedness": 0.3 * self.motivation_profile.relatedness_weight
            }

            # Ajustar prioridade baseada na severidade
            priority = "high" if gap.severity > 0.7 else "medium" if gap.severity > 0.4 else "low"

            goal = AutonomousGoal(
                id=f"gap_{gap.capability_id}_{gap.gap_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title=f"Resolver Lacuna: {gap.capability_id} ({gap.gap_type})",
                description=gap.description,
                goal_type="capability_gap",
                priority=priority,
                estimated_impact=gap.potential_impact,
                required_capabilities=[gap.capability_id],
                success_criteria=[
                    f"Melhorar {gap.gap_type} em pelo menos 20%",
                    "Implementar ações recomendadas",
                    "Validar melhoria com métricas"
                ],
                motivation_factors=motivation_factors,
                metadata={
                    "gap_analysis": {
                        "type": gap.gap_type,
                        "severity": gap.severity,
                        "confidence": gap.confidence,
                        "recommended_actions": gap.recommended_actions
                    }
                },
                created_at=timestamp
            )

            goals.append(goal)

        return goals

    def _identify_curiosity_opportunities(self) -> list[CuriosityOpportunity]:
        """Identifica oportunidades de exploração curiosa."""
        opportunities = []

        # 1. Oportunidades baseadas em novelty (novos domínios/técnicas)
        novelty_ops = self._identify_novelty_opportunities()
        opportunities.extend(novelty_ops)

        # 2. Oportunidades baseadas em uncertainty (áreas pouco exploradas)
        uncertainty_ops = self._identify_uncertainty_opportunities()
        opportunities.extend(uncertainty_ops)

        # 3. Oportunidades baseadas em diversidade (novas perspectivas)
        diversity_ops = self._identify_diversity_opportunities()
        opportunities.extend(diversity_ops)

        # 4. Oportunidades de integração (conectar áreas existentes)
        integration_ops = self._identify_integration_opportunities()
        opportunities.extend(integration_ops)

        # Ordenar por exploration_value
        opportunities.sort(key=lambda x: x.exploration_value, reverse=True)

        return opportunities[:15]  # Limitar a top 15 oportunidades

    def _identify_novelty_opportunities(self) -> list[CuriosityOpportunity]:
        """Identifica oportunidades baseadas em novelty."""
        opportunities = []

        # Domínios emergentes baseados em análise de código existente
        code_analysis = self._analyze_codebase_for_novelty()

        for domain, novelty_score in code_analysis.items():
            if novelty_score > self.motivation_profile.novelty_threshold:
                opportunities.append(CuriosityOpportunity(
                    domain=domain,
                    opportunity_type="novelty",
                    novelty_score=novelty_score,
                    uncertainty_score=0.6,  # Domínios novos têm alta incerteza
                    exploration_value=novelty_score * 0.8 + 0.6 * 0.2,
                    description=f"Explorar domínio emergente: {domain}",
                    suggested_approach=f"Pesquisar e implementar funcionalidades básicas em {domain}"
                ))

        return opportunities

    def _identify_uncertainty_opportunities(self) -> list[CuriosityOpportunity]:
        """Identifica oportunidades baseadas em incerteza."""
        opportunities = []

        # Áreas com baixa cobertura de testes indicam incerteza
        test_coverage = self._analyze_test_coverage_uncertainty()

        for area, uncertainty_score in test_coverage.items():
            if uncertainty_score > self.motivation_profile.uncertainty_tolerance:
                opportunities.append(CuriosityOpportunity(
                    domain=area,
                    opportunity_type="uncertainty",
                    novelty_score=0.4,
                    uncertainty_score=uncertainty_score,
                    exploration_value=uncertainty_score * 0.7 + 0.4 * 0.3,
                    description=f"Explorar área com baixa cobertura: {area}",
                    suggested_approach=f"Implementar testes e validações para {area}"
                ))

        return opportunities

    def _identify_diversity_opportunities(self) -> list[CuriosityOpportunity]:
        """Identifica oportunidades baseadas em diversidade."""
        opportunities = []

        # Analisar diversidade de abordagens em áreas existentes
        approach_diversity = self._analyze_approach_diversity()

        for domain, diversity_score in approach_diversity.items():
            exploration_value = diversity_score * self.motivation_profile.domain_diversity_factor
            if exploration_value > 0.3:
                opportunities.append(CuriosityOpportunity(
                    domain=domain,
                    opportunity_type="diversity",
                    novelty_score=diversity_score,
                    uncertainty_score=0.3,
                    exploration_value=exploration_value,
                    description=f"Diversificar abordagens em {domain}",
                    suggested_approach=f"Explorar abordagens alternativas para {domain}"
                ))

        return opportunities

    def _identify_integration_opportunities(self) -> list[CuriosityOpportunity]:
        """Identifica oportunidades de integração entre áreas."""
        opportunities = []

        # Procurar por padrões de integração não explorados
        integration_patterns = self._analyze_integration_patterns()

        for pattern in integration_patterns:
            opportunities.append(CuriosityOpportunity(
                domain=pattern["domain"],
                opportunity_type="integration",
                novelty_score=pattern["novelty"],
                uncertainty_score=pattern["uncertainty"],
                exploration_value=pattern["integration_value"],
                description=f"Integrar {pattern['source']} com {pattern['target']}",
                suggested_approach=pattern["suggested_approach"]
            ))

        return opportunities

    def _generate_curiosity_goals(
        self,
        opportunities: list[CuriosityOpportunity],
        timestamp: str
    ) -> list[AutonomousGoal]:
        """Gera objetivos baseados em oportunidades de exploração curiosa."""
        goals = []

        for opp in opportunities:
            # Calcular motivação baseada na exploração curiosa
            motivation_factors = {
                "curiosity": opp.exploration_value * self.motivation_profile.curiosity_weight,
                "autonomy": opp.novelty_score * self.motivation_profile.autonomy_weight,
                "relatedness": 0.2 * self.motivation_profile.relatedness_weight
            }

            # Ajustar prioridade baseada no valor de exploração
            priority = "high" if opp.exploration_value > 0.7 else "medium" if opp.exploration_value > 0.4 else "low"

            goal = AutonomousGoal(
                id=f"curiosity_{opp.opportunity_type}_{opp.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title=f"Exploração Curiosa: {opp.domain} ({opp.opportunity_type})",
                description=opp.description,
                goal_type="curiosity_exploration",
                priority=priority,
                estimated_impact=opp.exploration_value,
                required_capabilities=self._suggest_capabilities_for_domain(opp.domain),
                success_criteria=[
                    f"Implementar exploração inicial de {opp.domain}",
                    "Documentar descobertas e incertezas",
                    "Identificar próximos passos de exploração"
                ],
                motivation_factors=motivation_factors,
                metadata={
                    "curiosity_analysis": {
                        "opportunity_type": opp.opportunity_type,
                        "novelty_score": opp.novelty_score,
                        "uncertainty_score": opp.uncertainty_score,
                        "exploration_value": opp.exploration_value,
                        "suggested_approach": opp.suggested_approach
                    }
                },
                created_at=timestamp
            )

            goals.append(goal)

        return goals

    def _generate_self_optimization_goals(self, timestamp: str) -> list[AutonomousGoal]:
        """Gera objetivos de auto-otimização baseada em análise de performance."""
        goals = []

        # 1. Otimização baseada em métricas de performance
        performance_goals = self._generate_performance_optimization_goals(timestamp)
        goals.extend(performance_goals)

        # 2. Otimização baseada em análise de eficiência
        efficiency_goals = self._generate_efficiency_optimization_goals(timestamp)
        goals.extend(efficiency_goals)

        # 3. Otimização baseada em análise de recursos
        resource_goals = self._generate_resource_optimization_goals(timestamp)
        goals.extend(resource_goals)

        return goals

    def _generate_performance_optimization_goals(self, timestamp: str) -> list[AutonomousGoal]:
        """Gera objetivos de otimização de performance."""
        goals = []

        # Analisar gargalos de performance
        performance_bottlenecks = self._analyze_performance_bottlenecks()

        for bottleneck in performance_bottlenecks:
            motivation_factors = {
                "competence": 0.8 * self.motivation_profile.competence_weight,
                "autonomy": 0.6 * self.motivation_profile.autonomy_weight
            }

            goal = AutonomousGoal(
                id=f"perf_opt_{bottleneck['component']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title=f"Otimizar Performance: {bottleneck['component']}",
                description=bottleneck['description'],
                goal_type="self_optimization",
                priority="high",
                estimated_impact=bottleneck['impact'],
                required_capabilities=bottleneck['related_capabilities'],
                success_criteria=[
                    f"Melhorar {bottleneck['metric']} em pelo menos 30%",
                    "Manter funcionalidade existente",
                    "Validar melhoria com benchmarks"
                ],
                motivation_factors=motivation_factors,
                metadata={
                    "optimization_type": "performance",
                    "bottleneck_analysis": bottleneck
                },
                created_at=timestamp
            )

            goals.append(goal)

        return goals

    def _generate_efficiency_optimization_goals(self, timestamp: str) -> list[AutonomousGoal]:
        """Gera objetivos de otimização de eficiência."""
        goals = []

        # Analisar oportunidades de melhoria de eficiência
        efficiency_opportunities = self._analyze_efficiency_opportunities()

        for opportunity in efficiency_opportunities:
            motivation_factors = {
                "competence": 0.7 * self.motivation_profile.competence_weight,
                "autonomy": 0.5 * self.motivation_profile.autonomy_weight
            }

            goal = AutonomousGoal(
                id=f"eff_opt_{opportunity['area']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title=f"Otimizar Eficiência: {opportunity['area']}",
                description=opportunity['description'],
                goal_type="self_optimization",
                priority="medium",
                estimated_impact=opportunity['potential_improvement'],
                required_capabilities=opportunity['related_capabilities'],
                success_criteria=[
                    f"Reduzir {opportunity['metric']} em pelo menos 20%",
                    "Manter qualidade de resultados",
                    "Documentar mudanças implementadas"
                ],
                motivation_factors=motivation_factors,
                metadata={
                    "optimization_type": "efficiency",
                    "efficiency_analysis": opportunity
                },
                created_at=timestamp
            )

            goals.append(goal)

        return goals

    def _generate_resource_optimization_goals(self, timestamp: str) -> list[AutonomousGoal]:
        """Gera objetivos de otimização de recursos."""
        goals = []

        # Analisar uso de recursos
        resource_usage = self._analyze_resource_usage()

        for resource_issue in resource_usage:
            motivation_factors = {
                "competence": 0.6 * self.motivation_profile.competence_weight,
                "autonomy": 0.4 * self.motivation_profile.autonomy_weight
            }

            goal = AutonomousGoal(
                id=f"res_opt_{resource_issue['resource']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title=f"Otimizar Uso de Recursos: {resource_issue['resource']}",
                description=resource_issue['description'],
                goal_type="self_optimization",
                priority="medium",
                estimated_impact=resource_issue['optimization_potential'],
                required_capabilities=resource_issue['related_capabilities'],
                success_criteria=[
                    f"Otimizar uso de {resource_issue['resource']}",
                    "Manter performance adequada",
                    "Monitorar impacto das mudanças"
                ],
                motivation_factors=motivation_factors,
                metadata={
                    "optimization_type": "resource",
                    "resource_analysis": resource_issue
                },
                created_at=timestamp
            )

            goals.append(goal)

        return goals

    def _generate_domain_expansion_goals(self, timestamp: str) -> list[AutonomousGoal]:
        """Gera objetivos de expansão de domínio."""
        goals = []

        # Identificar domínios adjacentes para expansão
        expansion_opportunities = self._identify_domain_expansion_opportunities()

        for opportunity in expansion_opportunities:
            motivation_factors = {
                "curiosity": 0.7 * self.motivation_profile.curiosity_weight,
                "competence": 0.5 * self.motivation_profile.competence_weight,
                "autonomy": 0.6 * self.motivation_profile.autonomy_weight
            }

            goal = AutonomousGoal(
                id=f"domain_exp_{opportunity['domain']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title=f"Expandir para Domínio: {opportunity['domain']}",
                description=opportunity['description'],
                goal_type="domain_expansion",
                priority="high",
                estimated_impact=opportunity['strategic_value'],
                required_capabilities=opportunity['required_capabilities'],
                success_criteria=[
                    f"Implementar funcionalidades básicas em {opportunity['domain']}",
                    "Criar integração com sistemas existentes",
                    "Estabelecer baselines de performance"
                ],
                motivation_factors=motivation_factors,
                metadata={
                    "expansion_analysis": {
                        "domain": opportunity['domain'],
                        "rationale": opportunity['rationale'],
                        "strategic_value": opportunity['strategic_value'],
                        "implementation_approach": opportunity['implementation_approach']
                    }
                },
                created_at=timestamp
            )

            goals.append(goal)

        return goals

    def _filter_and_prioritize_goals(
        self,
        goals: list[AutonomousGoal],
        context: dict[str, Any] | None = None
    ) -> list[AutonomousGoal]:
        """Aplica filtros e priorização aos objetivos gerados."""
        filtered_goals = []

        # 1. Remover duplicatas
        seen = set()
        for goal in goals:
            goal_key = f"{goal.goal_type}:{goal.title}"
            if goal_key not in seen:
                seen.add(goal_key)
                filtered_goals.append(goal)

        # 2. Aplicar filtros contextuais
        if context:
            filtered_goals = self._apply_context_filters(filtered_goals, context)

        # 3. Calcular score final de motivação
        for goal in filtered_goals:
            total_motivation = sum(goal.motivation_factors.values())
            goal.motivation_factors["_total"] = total_motivation

        # 4. Ordenar por score de motivação e impacto
        filtered_goals.sort(
            key=lambda g: (
                sum(g.motivation_factors.values()),
                g.estimated_impact,
                self._get_priority_score(g.priority)
            ),
            reverse=True
        )

        # 5. Limitar número de objetivos (evitar sobrecarga)
        max_goals = context.get("max_goals", 10) if context else 10
        return filtered_goals[:max_goals]

    def convert_goals_to_seeds(self, goals: list[AutonomousGoal]) -> list[EvaluationSeed]:
        """Converte objetivos autônomos em seeds executáveis."""
        seeds = []

        for goal in goals:
            seed = EvaluationSeed(
                description=f"[AUTO-GOAL] {goal.title} - {goal.description}",
                priority=goal.priority,
                capability=",".join(goal.required_capabilities) if goal.required_capabilities else "meta.autonomous_goal",
                seed_type="autonomous_goal",
                data={
                    "goal_id": goal.id,
                    "goal_type": goal.goal_type,
                    "estimated_impact": str(goal.estimated_impact),
                    "motivation_factors": json.dumps(goal.motivation_factors),
                    "success_criteria": "; ".join(goal.success_criteria),
                    "metadata": json.dumps(goal.metadata)
                }
            )
            seeds.append(seed)

        return seeds

    def generate_goals_from_templates(
        self,
        template_config_path: str = "configs/seed_autonomous_goals.yaml",
        context: dict[str, Any] | None = None
    ) -> list[Seed]:
        """Gera objetivos autônomos usando templates de configuração."""
        seeds = []

        try:
            import yaml
            with open(template_config_path, 'r', encoding='utf-8') as f:
                template_config = yaml.safe_load(f)
        except Exception:
            return seeds

        # Verificar se geração de objetivos autônomos está habilitada
        if not template_config.get("autonomous_goals", {}).get("enabled", True):
            return seeds

        # Gerar objetivos baseados em análise de lacunas (sempre habilitado)
        gap_goals = self._generate_goals_from_gap_analysis(template_config, context)
        seeds.extend(gap_goals)

        # Gerar objetivos de exploração curiosa se habilitado
        if template_config.get("curiosity_exploration", {}).get("enabled", True):
            curiosity_goals = self._generate_goals_from_curiosity_templates(template_config, context)
            seeds.extend(curiosity_goals)

        # Gerar objetivos de meta-reflexão se habilitado
        if template_config.get("meta_reflection", {}).get("enabled", True):
            reflection_goals = self._generate_goals_from_reflection_templates(template_config, context)
            seeds.extend(reflection_goals)

        # Gerar objetivos de auto-otimização se habilitado
        if template_config.get("self_optimization", {}).get("enabled", True):
            optimization_goals = self._generate_goals_from_optimization_templates(template_config, context)
            seeds.extend(optimization_goals)

        # Gerar objetivos de expansão de domínio se habilitado
        if template_config.get("domain_expansion", {}).get("enabled", True):
            expansion_goals = self._generate_goals_from_expansion_templates(template_config, context)
            seeds.extend(expansion_goals)

        # Aplicar limites de geração
        max_goals = template_config.get("integration", {}).get("max_goals_per_cycle", 10)
        if len(seeds) > max_goals:
            # Ordenar por prioridade e impacto antes de limitar
            seeds.sort(key=lambda s: (self._get_seed_priority_score(s), s.metadata.get("estimated_impact", "0.5")), reverse=True)
            seeds = seeds[:max_goals]

        return seeds

    def _generate_goals_from_gap_analysis(
        self,
        template_config: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> list[Seed]:
        """Gera objetivos baseados em análise de lacunas usando templates."""
        seeds = []

        # Usar o sistema existente de análise de lacunas
        gaps = self._analyze_capability_gaps()

        gap_templates = template_config.get("capability_gap", {}).get("templates", [])
        template_by_type = {t["id"]: t for t in gap_templates}

        for gap in gaps[:5]:  # Limitar a top 5 gaps
            template = template_by_type.get("performance_gap_closure") or template_by_type.get("coverage_gap_address")

            if template:
                # Criar AutonomousGoal simulado para usar com o template
                goal = self._create_goal_from_gap_and_template(gap, template)
                seed = self.create_autonomous_goal_seed(goal, template_config.get("capability_gap", {}))
                seeds.append(seed)

        return seeds

    def _generate_goals_from_curiosity_templates(
        self,
        template_config: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> list[Seed]:
        """Gera objetivos de exploração curiosa usando templates."""
        seeds = []

        curiosity_config = template_config.get("curiosity_exploration", {})
        templates = curiosity_config.get("templates", [])
        max_goals = curiosity_config.get("max_per_cycle", 3)

        # Usar oportunidades de exploração curiosa existentes
        opportunities = self._identify_curiosity_opportunities()

        for opp in opportunities[:max_goals]:
            # Encontrar template apropriado
            template = None
            if opp.opportunity_type == "novelty":
                template = next((t for t in templates if t["id"] == "novelty_domain_exploration"), None)
            elif opp.opportunity_type == "uncertainty":
                template = next((t for t in templates if t["id"] == "uncertainty_reduction"), None)
            elif opp.opportunity_type == "diversity":
                template = next((t for t in templates if t["id"] == "diversity_expansion"), None)

            if template:
                goal = self._create_goal_from_opportunity_and_template(opp, template)
                seed = self.create_autonomous_goal_seed(goal, curiosity_config)
                seeds.append(seed)

        return seeds

    def _generate_goals_from_reflection_templates(
        self,
        template_config: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> list[Seed]:
        """Gera objetivos de meta-reflexão usando templates."""
        seeds = []

        reflection_config = template_config.get("meta_reflection", {})
        templates = reflection_config.get("templates", [])
        max_goals = reflection_config.get("max_per_cycle", 2)

        # Verificar se é hora de fazer reflexão (baseado em histórico)
        if len(self._goal_history) > 10 and len(self._goal_history) % 10 == 0:
            for template_dict in templates[:max_goals]:
                goal = self._create_goal_from_reflection_template(template_dict)
                seed = self.create_autonomous_goal_seed(goal, reflection_config)
                seeds.append(seed)

        return seeds

    def _generate_goals_from_optimization_templates(
        self,
        template_config: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> list[Seed]:
        """Gera objetivos de auto-otimização usando templates."""
        seeds = []

        optimization_config = template_config.get("self_optimization", {})
        templates = optimization_config.get("templates", [])
        max_goals = optimization_config.get("max_per_cycle", 4)

        # Usar análise de performance existente
        bottlenecks = self._analyze_performance_bottlenecks()

        for bottleneck in bottlenecks[:max_goals]:
            template = next((t for t in templates if t["id"] == "performance_bottleneck_resolution"), None)
            if template:
                goal = self._create_goal_from_bottleneck_and_template(bottleneck, template)
                seed = self.create_autonomous_goal_seed(goal, optimization_config)
                seeds.append(seed)

        return seeds

    def _generate_goals_from_expansion_templates(
        self,
        template_config: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> list[Seed]:
        """Gera objetivos de expansão de domínio usando templates."""
        seeds = []

        expansion_config = template_config.get("domain_expansion", {})
        templates = expansion_config.get("templates", [])
        max_goals = expansion_config.get("max_per_cycle", 2)

        # Usar oportunidades de expansão existentes
        expansion_opportunities = self._identify_domain_expansion_opportunities()

        for opportunity in expansion_opportunities[:max_goals]:
            template = templates[0] if templates else None  # Usar primeiro template disponível
            if template:
                goal = self._create_goal_from_expansion_and_template(opportunity, template)
                seed = self.create_autonomous_goal_seed(goal, expansion_config)
                seeds.append(seed)

        return seeds

    def _create_goal_from_gap_and_template(self, gap: CapabilityGapAnalysis, template: dict[str, Any]) -> AutonomousGoal:
        """Cria AutonomousGoal a partir de CapabilityGapAnalysis e template."""
        return AutonomousGoal(
            id=f"gap_{gap.capability_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=template.get("title_template", f"Resolver Lacuna: {gap.capability_id}"),
            description=template.get("description_template", gap.description),
            goal_type=template.get("goal_type", "capability_gap"),
            priority="high" if gap.severity > 0.7 else "medium",
            estimated_impact=gap.potential_impact,
            required_capabilities=[gap.capability_id],
            success_criteria=template.get("success_criteria", gap.recommended_actions),
            motivation_factors={
                "competence": gap.severity * 0.3,
                "autonomy": gap.potential_impact * 0.2
            },
            metadata={"gap_analysis": {
                "type": gap.gap_type,
                "severity": gap.severity,
                "confidence": gap.confidence
            }}
        )

    def _create_goal_from_opportunity_and_template(self, opp: CuriosityOpportunity, template: dict[str, Any]) -> AutonomousGoal:
        """Cria AutonomousGoal a partir de CuriosityOpportunity e template."""
        return AutonomousGoal(
            id=f"curiosity_{opp.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=template.get("title_template", f"Explorar: {opp.domain}"),
            description=template.get("description_template", opp.description),
            goal_type=template.get("goal_type", "curiosity_exploration"),
            priority="high" if opp.exploration_value > 0.7 else "medium",
            estimated_impact=opp.exploration_value,
            required_capabilities=self._suggest_capabilities_for_domain(opp.domain),
            success_criteria=template.get("success_criteria", [opp.suggested_approach]),
            motivation_factors={
                "curiosity": opp.exploration_value * 0.3,
                "autonomy": opp.novelty_score * 0.2
            },
            metadata={"curiosity_analysis": {
                "opportunity_type": opp.opportunity_type,
                "novelty_score": opp.novelty_score,
                "uncertainty_score": opp.uncertainty_score,
                "exploration_value": opp.exploration_value
            }}
        )

    def _create_goal_from_reflection_template(self, template: dict[str, Any]) -> AutonomousGoal:
        """Cria AutonomousGoal a partir de template de reflexão."""
        return AutonomousGoal(
            id=f"reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=template.get("title_template", "Meta-reflexão do sistema"),
            description=template.get("description_template", "Análise reflexiva do sistema"),
            goal_type=template.get("goal_type", "meta_reflection"),
            priority="high",
            estimated_impact=0.8,
            required_capabilities=["meta.self_evaluation"],
            success_criteria=template.get("success_criteria", ["Gerar relatório de análise"]),
            motivation_factors={
                "competence": 0.3,
                "autonomy": 0.2,
                "relatedness": 0.2
            },
            metadata={"reflection_type": "system_meta_reflection"}
        )

    def _create_goal_from_bottleneck_and_template(self, bottleneck: dict[str, Any], template: dict[str, Any]) -> AutonomousGoal:
        """Cria AutonomousGoal a partir de bottleneck e template."""
        return AutonomousGoal(
            id=f"opt_{bottleneck['component']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=template.get("title_template", f"Otimizar: {bottleneck['component']}"),
            description=template.get("description_template", bottleneck['description']),
            goal_type=template.get("goal_type", "self_optimization"),
            priority="high",
            estimated_impact=bottleneck.get('impact', 0.7),
            required_capabilities=bottleneck.get('related_capabilities', []),
            success_criteria=template.get("success_criteria", [f"Melhorar {bottleneck.get('metric', 'performance')}"]),
            motivation_factors={
                "competence": 0.25,
                "autonomy": 0.2
            },
            metadata={"optimization_type": "performance", "bottleneck": bottleneck}
        )

    def _create_goal_from_expansion_and_template(self, opportunity: dict[str, Any], template: dict[str, Any]) -> AutonomousGoal:
        """Cria AutonomousGoal a partir de oportunidade de expansão e template."""
        return AutonomousGoal(
            id=f"expansion_{opportunity['domain']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=template.get("title_template", f"Expandir: {opportunity['domain']}"),
            description=template.get("description_template", opportunity['description']),
            goal_type=template.get("goal_type", "domain_expansion"),
            priority="high",
            estimated_impact=opportunity.get('strategic_value', 0.7),
            required_capabilities=opportunity.get('required_capabilities', []),
            success_criteria=template.get("success_criteria", ["Implementar funcionalidades básicas"]),
            motivation_factors={
                "curiosity": 0.25,
                "competence": 0.2,
                "autonomy": 0.2
            },
            metadata={"expansion_analysis": opportunity}
        )

    def _get_seed_priority_score(self, seed: Seed) -> float:
        """Calcula score de prioridade para ordenação de seeds."""
        priority_scores = {"high": 3.0, "medium": 2.0, "low": 1.0}
        base_score = priority_scores.get(seed.priority, 1.0)

        # Bonus para objetivos autônomos baseados em impacto estimado
        if seed.is_autonomous_goal:
            estimated_impact = float(seed.metadata.get("estimated_impact", "0.5"))
            base_score += estimated_impact

        return base_score

    def integrate_with_autonomous_planner(self, context: dict[str, Any] | None = None) -> list[EvaluationSeed]:
        """Integra objetivos gerados com o planejador autônomo existente."""
        # Gerar objetivos autônomos
        goals = self.generate_autonomous_goals(context)

        # Converter em seeds
        seeds = self.convert_goals_to_seeds(goals)

        # Adicionar ao backlog usando o sistema existente
        if self.backlog_path.exists():
            backlog = SeedBacklog.load(self.backlog_path)
            for seed in seeds:
                # Converter EvaluationSeed para Seed com metadados aprimorados
                goal_id_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
                seed_id = f"autonomous_{seed.seed_type}_{goal_id_suffix}"

                # Extrair metadados específicos de objetivos autônomos
                autonomous_metadata = {}
                if seed.data:
                    autonomous_metadata.update({
                        "description": seed.description,
                        "created_by": "autonomous_goal_generator",
                        "goal_id": seed.data.get("goal_id", ""),
                        "goal_type": seed.data.get("goal_type", "unknown"),
                        "estimated_impact": seed.data.get("estimated_impact", "0.5"),
                        "motivation_factors": seed.data.get("motivation_factors", "{}"),
                        "success_criteria": seed.data.get("success_criteria", ""),
                        "metadata": seed.data.get("metadata", "{}"),
                        "created_at": datetime.now(timezone.utc).isoformat()
                    })

                backlog_seed = Seed(
                    id=seed_id,
                    goal=seed.description,
                    priority=seed.priority,
                    type=seed.seed_type,
                    config="configs/seed_manual.yaml",
                    metadata=autonomous_metadata,
                    max_steps=15  # Objetivos autônomos podem precisar de mais passos
                )
                backlog.add_seed(backlog_seed)

        return seeds

    def create_autonomous_goal_seed(
        self,
        goal: AutonomousGoal,
        template_config: dict[str, Any] | None = None
    ) -> Seed:
        """Cria uma Seed a partir de um AutonomousGoal com configuração de template."""
        # Usar configuração de template se fornecida, caso contrário usar padrão
        config_file = "configs/seed_manual.yaml"
        max_steps = 15

        if template_config:
            config_file = template_config.get("config_template", config_file)
            max_steps = template_config.get("max_iterations", max_steps)

        return Seed(
            id=f"autonomous_{goal.goal_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            goal=f"[AUTO-GOAL] {goal.title} - {goal.description}",
            priority=goal.priority,
            type=goal.goal_type,
            config=config_file,
            max_steps=max_steps,
            metadata={
                "goal_id": goal.id,
                "goal_type": goal.goal_type,
                "estimated_impact": str(goal.estimated_impact),
                "motivation_factors": json.dumps(goal.motivation_factors),
                "success_criteria": "; ".join(goal.success_criteria),
                "metadata": json.dumps(goal.metadata),
                "created_by": "autonomous_goal_generator",
                "created_at": goal.created_at
            }
        )

    def perform_meta_reflection(self) -> dict[str, Any]:
        """Realiza meta-reflexão sobre o processo de geração de objetivos."""
        reflection = {
            "total_goals_generated": len(self._goal_history),
            "goals_by_type": {},
            "motivation_effectiveness": {},
            "areas_needing_improvement": [],
            "emerging_patterns": [],
            "recommendations": []
        }

        # Analisar distribuição por tipo
        for goal in self._goal_history:
            goal_type = goal.goal_type
            reflection["goals_by_type"][goal_type] = reflection["goals_by_type"].get(goal_type, 0) + 1

        # Analisar efetividade de motivação
        if self._goal_history:
            avg_motivation = sum(
                sum(goal.motivation_factors.values())
                for goal in self._goal_history
            ) / len(self._goal_history)

            reflection["motivation_effectiveness"]["average_total_motivation"] = avg_motivation

            # Identificar tipos mais/menos motivadores
            type_motivation = {}
            for goal in self._goal_history:
                goal_type = goal.goal_type
                if goal_type not in type_motivation:
                    type_motivation[goal_type] = []
                type_motivation[goal_type].append(sum(goal.motivation_factors.values()))

            for goal_type, motivations in type_motivation.items():
                reflection["motivation_effectiveness"][f"{goal_type}_avg_motivation"] = sum(motivations) / len(motivations)

        # Identificar áreas necessitando melhoria
        if reflection["goals_by_type"].get("capability_gap", 0) > reflection["goals_by_type"].get("curiosity_exploration", 0) * 2:
            reflection["areas_needing_improvement"].append(
                "Exploração curiosa pode estar sendo subvalorizada em relação a correções de lacunas"
            )

        # Gerar recomendações
        if avg_motivation < 0.5:
            reflection["recommendations"].append(
                "Ajustar perfil de motivação para aumentar engajamento com objetivos gerados"
            )

        return reflection

    # Métodos auxiliares para análise

    def _get_latest_metric(self, metrics_history: dict[str, Any], metric_name: str) -> float | None:
        """Obtém o valor mais recente de uma métrica."""
        if metric_name not in metrics_history:
            return None

        values = metrics_history[metric_name]
        if not values or not isinstance(values, list):
            return None

        try:
            return float(values[-1])
        except (ValueError, TypeError):
            return None

    def _load_capability_baselines(self) -> None:
        """Carrega baselines históricos de capacidades."""
        if not self.metrics_path.exists():
            return

        try:
            with open(self.metrics_path, encoding="utf-8") as f:
                history = json.load(f)

            # Usar médias dos últimos 10 valores como baseline
            for metric_name, values in history.items():
                if isinstance(values, list) and len(values) >= 5:
                    recent_values = values[-10:] if len(values) >= 10 else values
                    self._capability_baselines[metric_name] = sum(recent_values) / len(recent_values)

        except Exception:
            pass

    def _has_integration_evidence(self, cap_id: str, related_capabilities: list[Any]) -> bool:
        """Verifica se há evidência de integração entre capacidades."""
        # Por simplicidade, assume que há integração se múltiplas capacidades
        # do mesmo módulo são usadas frequentemente juntas
        return len(related_capabilities) > 0

    def _analyze_codebase_for_novelty(self) -> dict[str, float]:
        """Analisa codebase para identificar domínios emergentes."""
        novelty_scores = {}

        try:
            # Analisar arquivos Python para identificar padrões
            py_files = list(self.workspace_root.rglob("*.py"))
            if not py_files:
                return novelty_scores

            # Por simplicidade, usar análise básica de nomes de arquivos
            domains_found = set()

            for file_path in py_files:
                filename = file_path.stem.lower()

                # Identificar domínios baseados em nomes
                if any(word in filename for word in ["test", "spec"]):
                    domains_found.add("testing")
                elif any(word in filename for word in ["memory", "cache", "store"]):
                    domains_found.add("memory_management")
                elif any(word in filename for word in ["config", "settings"]):
                    domains_found.add("configuration")
                elif any(word in filename for word in ["metrics", "analytics"]):
                    domains_found.add("monitoring")
                elif any(word in filename for word in ["seed", "goal"]):
                    domains_found.add("autonomous_systems")

            # Calcular novelty baseado em recência e uso
            for domain in domains_found:
                # Domínios menos estabelecidos têm maior novelty
                base_novelty = {
                    "testing": 0.3,
                    "memory_management": 0.4,
                    "configuration": 0.2,
                    "monitoring": 0.5,
                    "autonomous_systems": 0.8
                }.get(domain, 0.5)

                novelty_scores[domain] = base_novelty

        except Exception:
            pass

        return novelty_scores

    def _analyze_test_coverage_uncertainty(self) -> dict[str, float]:
        """Analisa cobertura de testes para identificar incerteza."""
        uncertainty_scores = {}

        try:
            test_files = list(self.workspace_root.rglob("*test*.py"))
            source_files = list(self.workspace_root.rglob("*.py"))

            # Remover test files dos source files
            source_files = [f for f in source_files if "test" not in f.name.lower()]

            if source_files:
                # Calcular cobertura básica baseada em existência de testes
                for source_file in source_files[:20]:  # Limitar análise
                    source_name = source_file.stem

                    # Verificar se há teste correspondente
                    has_test = any(source_name in test_file.name for test_file in test_files)

                    if not has_test:
                        uncertainty_scores[source_name] = 0.8
                    else:
                        uncertainty_scores[source_name] = 0.3

        except Exception:
            pass

        return uncertainty_scores

    def _analyze_approach_diversity(self) -> dict[str, float]:
        """Analisa diversidade de abordagens em domínios existentes."""
        diversity_scores = {}

        # Para cada domínio identificado, calcular diversidade baseada em técnicas usadas
        # Esta é uma implementação simplificada
        domains = ["execution", "planning", "memory", "testing"]

        for domain in domains:
            # Baseado em análise de arquivos relacionados ao domínio
            related_files = list(self.workspace_root.rglob(f"*{domain}*.py"))

            # Mais arquivos = potencialmente mais diversidade
            file_count = len(related_files)
            diversity_score = min(file_count / 5.0, 1.0)  # Normalizar para 0-1

            diversity_scores[domain] = diversity_score

        return diversity_scores

    def _analyze_integration_patterns(self) -> list[dict[str, Any]]:
        """Analisa padrões de integração potenciais."""
        patterns = []

        # Identificar possíveis integrações entre módulos
        modules = ["execution", "planning", "memory", "testing", "capabilities"]

        for i, source_module in enumerate(modules):
            for target_module in modules[i+1:]:
                # Calcular potencial de integração
                integration_value = self._calculate_integration_value(source_module, target_module)

                if integration_value > 0.4:
                    patterns.append({
                        "domain": f"{source_module}_{target_module}_integration",
                        "source": source_module,
                        "target": target_module,
                        "novelty": 0.6,
                        "uncertainty": 0.5,
                        "integration_value": integration_value,
                        "suggested_approach": f"Criar APIs de comunicação entre {source_module} e {target_module}"
                    })

        return patterns

    def _calculate_integration_value(self, module1: str, module2: str) -> float:
        """Calcula valor potencial de integração entre dois módulos."""
        # Lógica simplificada baseada em complementaridade
        complementarity_map = {
            ("execution", "planning"): 0.8,
            ("memory", "planning"): 0.7,
            ("testing", "execution"): 0.9,
            ("capabilities", "planning"): 0.6,
        }

        return complementarity_map.get((module1, module2), 0.5)

    def _suggest_capabilities_for_domain(self, domain: str) -> list[str]:
        """Sugere capacidades necessárias para um domínio."""
        domain_capability_map = {
            "testing": ["core.testing", "meta.test_generation"],
            "memory": ["meta.memory_management", "core.caching"],
            "planning": ["meta.planning", "core.optimization"],
            "execution": ["core.execution", "meta.self_modification"],
            "monitoring": ["meta.monitoring", "core.metrics"],
            "configuration": ["meta.configuration", "core.validation"]
        }

        return domain_capability_map.get(domain, ["meta.skill_creation"])

    def _analyze_performance_bottlenecks(self) -> list[dict[str, Any]]:
        """Analisa gargalos de performance no sistema."""
        bottlenecks = []

        # Análise baseada em métricas conhecidas
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, encoding="utf-8") as f:
                    metrics = json.load(f)

                # Identificar métricas com valores preocupantes
                if "llm_latency_last" in metrics:
                    latency_values = metrics["llm_latency_last"]
                    if latency_values and float(latency_values[-1]) > 10.0:  # > 10s latência
                        bottlenecks.append({
                            "component": "llm_interface",
                            "description": f"Alta latência no LLM: {latency_values[-1]:.2f}s",
                            "metric": "llm_latency_last",
                            "impact": 0.8,
                            "related_capabilities": ["core.llm", "meta.optimization"]
                        })

            except Exception:
                pass

        return bottlenecks

    def _analyze_efficiency_opportunities(self) -> list[dict[str, Any]]:
        """Analisa oportunidades de melhoria de eficiência."""
        opportunities = []

        # Oportunidades baseadas em padrões comuns
        common_opportunities = [
            {
                "area": "memory_usage",
                "description": "Otimizar uso de memória em operações intensivas",
                "metric": "memory_efficiency",
                "potential_improvement": 0.3,
                "related_capabilities": ["meta.memory_management", "core.caching"]
            },
            {
                "area": "execution_loops",
                "description": "Otimizar loops de execução para reduzir iterações desnecessárias",
                "metric": "iteration_efficiency",
                "potential_improvement": 0.4,
                "related_capabilities": ["core.execution", "meta.optimization"]
            }
        ]

        opportunities.extend(common_opportunities)
        return opportunities

    def _analyze_resource_usage(self) -> list[dict[str, Any]]:
        """Analisa uso de recursos do sistema."""
        resource_issues = []

        # Análise básica baseada em padrões de uso
        basic_issues = [
            {
                "resource": "computation",
                "description": "Otimizar uso computacional em operações repetitivas",
                "optimization_potential": 0.3,
                "related_capabilities": ["core.optimization", "meta.self_modification"]
            }
        ]

        resource_issues.extend(basic_issues)
        return resource_issues

    def _identify_domain_expansion_opportunities(self) -> list[dict[str, Any]]:
        """Identifica oportunidades de expansão para novos domínios."""
        opportunities = []

        # Domínios estratégicos para expansão
        strategic_domains = [
            {
                "domain": "machine_learning",
                "description": "Expandir capacidades para análise de dados e machine learning",
                "rationale": "Complementar capacidades existentes de análise e otimização",
                "strategic_value": 0.9,
                "required_capabilities": ["horiz.data_science", "meta.skill_creation"],
                "implementation_approach": "Implementar bibliotecas básicas de ML e integração com sistema existente"
            },
            {
                "domain": "web_automation",
                "description": "Desenvolver capacidades de automação web",
                "rationale": "Expandir alcance para interação com sistemas externos",
                "strategic_value": 0.7,
                "required_capabilities": ["horiz.web", "core.execution"],
                "implementation_approach": "Criar módulos básicos de scraping e automação"
            }
        ]

        opportunities.extend(strategic_domains)
        return opportunities

    def _apply_context_filters(self, goals: list[AutonomousGoal], context: dict[str, Any]) -> list[AutonomousGoal]:
        """Aplica filtros contextuais aos objetivos."""
        filtered = goals

        # Filtrar por tipo de objetivo se especificado
        if "goal_types" in context:
            allowed_types = context["goal_types"]
            filtered = [g for g in filtered if g.goal_type in allowed_types]

        # Filtrar por impacto mínimo se especificado
        if "min_impact" in context:
            min_impact = context["min_impact"]
            filtered = [g for g in filtered if g.estimated_impact >= min_impact]

        # Filtrar por capacidades disponíveis se especificado
        if "available_capabilities" in context:
            available_caps = set(context["available_capabilities"])
            filtered = [
                g for g in filtered
                if not g.required_capabilities or any(cap in available_caps for cap in g.required_capabilities)
            ]

        return filtered

    def _get_priority_score(self, priority: str) -> float:
        """Converte prioridade em score numérico."""
        priority_scores = {
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0
        }
        return priority_scores.get(priority, 1.0)


def run_autonomous_goal_generation(
    workspace_root: Path,
    context: dict[str, Any] | None = None
) -> list[EvaluationSeed]:
    """Executa geração autônoma de objetivos e retorna seeds."""
    generator = AutonomousGoalGenerator(workspace_root)
    seeds = generator.integrate_with_autonomous_planner()
    return seeds


__all__ = [
    "AutonomousGoalGenerator",
    "AutonomousGoal",
    "IntrinsicMotivationProfile",
    "CapabilityGapAnalysis",
    "CuriosityOpportunity",
    "run_autonomous_goal_generation"
]