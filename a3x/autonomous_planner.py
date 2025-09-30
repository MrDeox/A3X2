"""Sistema de auto-planejamento autônomo para o SeedAI."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from .autoeval import RunEvaluation, EvaluationSeed
from .capabilities import CapabilityRegistry
from .capability_metrics import compute_capability_metrics


@dataclass
class AutonomousPlan:
    """Plano autônomo gerado pelo sistema."""
    
    id: str
    title: str
    description: str
    goal: str
    priority: str  # "low", "medium", "high"
    estimated_duration: str  # "short", "medium", "long"
    required_capabilities: List[str]
    success_criteria: List[str]
    risks: List[str]
    dependencies: List[str]
    created_at: str
    estimated_completion_date: Optional[str] = None


@dataclass
class EvolutionInsight:
    """Insight de evolução identificado automaticamente."""
    
    id: str
    category: str  # "performance", "capability", "knowledge", "efficiency"
    title: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    evidence: List[str]  # Dados que suportam este insight
    recommendations: List[str]  # Ações recomendadas
    estimated_impact: float  # 0.0 to 1.0
    created_at: str


class AutonomousPlanner:
    """Planejador autônomo que gera planos de evolução baseados em análise de dados."""
    
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.evaluations_path = workspace_root / "seed" / "evaluations" / "run_evaluations.jsonl"
        self.metrics_path = workspace_root / "seed" / "metrics" / "history.json"
        self.capabilities_path = workspace_root / "seed" / "capabilities.yaml"
        self.backlog_path = workspace_root / "seed" / "backlog.yaml"
        self.plans_path = workspace_root / "seed" / "plans"
        self.plans_path.mkdir(parents=True, exist_ok=True)
        
    def analyze_historical_data(self) -> Dict[str, Any]:
        """Analisa dados históricos para identificar padrões e oportunidades."""
        analysis = {
            "metrics_trends": {},
            "capability_gaps": [],
            "performance_patterns": {},
            "failure_analysis": {},
            "success_factors": [],
            "recommendations": []
        }
        
        # Analisar métricas históricas
        if self.metrics_path.exists():
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            # Analisar tendências de métricas
            for metric_name, values in metrics_data.items():
                if len(values) >= 2:
                    recent_avg = sum(values[-5:]) / min(5, len(values)) if values else 0
                    older_avg = sum(values[-10:-5]) / min(5, len(values[:-5])) if len(values) >= 10 else recent_avg
                    
                    trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
                    
                    analysis["metrics_trends"][metric_name] = {
                        "current": recent_avg,
                        "trend": trend,
                        "change_rate": (recent_avg - older_avg) / max(0.001, older_avg) if older_avg != 0 else 0
                    }
        
        # Analisar avaliações de execução
        if self.evaluations_path.exists():
            success_rates = []
            failure_patterns = []
            capability_usage = {}
            
            with open(self.evaluations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        evaluation = json.loads(line.strip())
                        success_rates.append(evaluation.get("completed", False))
                        
                        # Analisar padrões de falha
                        if not evaluation.get("completed", True):
                            failure_reasons = evaluation.get("notes", "")
                            if failure_reasons:
                                failure_patterns.append(failure_reasons)
                        
                        # Analisar uso de capacidades
                        capabilities = evaluation.get("capabilities", [])
                        for cap in capabilities:
                            capability_usage[cap] = capability_usage.get(cap, 0) + 1
                                
                    except Exception:
                        continue
            
            # Calcular taxa geral de sucesso
            if success_rates:
                overall_success_rate = sum(success_rates) / len(success_rates)
                analysis["performance_patterns"]["overall_success_rate"] = overall_success_rate
            
            # Identificar padrões de falha comuns
            if failure_patterns:
                analysis["failure_analysis"]["common_patterns"] = list(set(failure_patterns[:10]))  # Top 10 padrões
            
            # Identificar capacidades mais/menos usadas
            if capability_usage:
                sorted_caps = sorted(capability_usage.items(), key=lambda x: x[1], reverse=True)
                analysis["capability_gaps"] = {
                    "most_used": [cap for cap, count in sorted_caps[:5]],
                    "least_used": [cap for cap, count in sorted_caps[-5:] if count > 0]
                }
        
        return analysis
    
    def identify_evolution_insights(self, analysis_data: Dict[str, Any]) -> List[EvolutionInsight]:
        """Identifica insights de evolução baseados na análise de dados."""
        insights = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Analisar tendências de métricas para identificar oportunidades
        for metric_name, trend_data in analysis_data.get("metrics_trends", {}).items():
            current_value = trend_data.get("current", 0)
            trend = trend_data.get("trend", "stable")
            
            # Identificar métricas com baixo desempenho
            if current_value < 0.7 and metric_name in ["actions_success_rate", "apply_patch_success_rate"]:
                insight = EvolutionInsight(
                    id=f"low_performance_{metric_name}",
                    category="performance",
                    title=f"Baixo desempenho em {metric_name}",
                    description=f"A métrica {metric_name} está abaixo do ideal ({current_value:.2f} < 0.7)",
                    severity="high",
                    evidence=[f"Métrica {metric_name}: {current_value:.2f}"],
                    recommendations=[
                        f"Melhorar algoritmos relacionados a {metric_name}",
                        "Analisar padrões de falha específicos",
                        "Implementar estratégias de recuperação mais robustas"
                    ],
                    estimated_impact=0.8,
                    created_at=timestamp
                )
                insights.append(insight)
            
            # Identificar métricas em declínio
            elif trend == "declining":
                change_rate = trend_data.get("change_rate", 0)
                if change_rate < -0.1:  # Declínio significativo
                    insight = EvolutionInsight(
                        id=f"declining_metric_{metric_name}",
                        category="performance",
                        title=f"Métrica em declínio: {metric_name}",
                        description=f"A métrica {metric_name} está em declínio significativo ({change_rate:.2%})",
                        severity="medium",
                        evidence=[f"Métrica {metric_name}: {current_value:.2f}", f"Declínio: {change_rate:.2%}"],
                        recommendations=[
                            "Investigar causa raiz do declínio",
                            "Implementar medidas corretivas",
                            "Monitorar tendência continuamente"
                        ],
                        estimated_impact=0.6,
                        created_at=timestamp
                    )
                    insights.append(insight)
        
        # Analisar lacunas de capacidades
        capability_gaps = analysis_data.get("capability_gaps", {})
        least_used = capability_gaps.get("least_used", [])
        
        if least_used:
            insight = EvolutionInsight(
                id="capability_gaps_detected",
                category="capability",
                title="Lacunas em capacidades pouco utilizadas",
                description=f"Capacidades subutilizadas detectadas: {', '.join(least_used[:3])}",
                severity="medium",
                evidence=[f"Capacidades menos usadas: {least_used}"],
                recommendations=[
                    "Investigar por que estas capacidades são subutilizadas",
                    "Melhorar integração dessas capacidades",
                    "Expandir casos de uso para estas habilidades"
                ],
                estimated_impact=0.7,
                created_at=timestamp
            )
            insights.append(insight)
        
        return insights
    
    def generate_autonomous_plans(self, insights: List[EvolutionInsight]) -> List[AutonomousPlan]:
        """Gera planos autônomos baseados nos insights identificados."""
        plans = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        for insight in insights:
            # Gerar plano baseado no insight
            plan = self._create_plan_from_insight(insight, timestamp)
            if plan:
                plans.append(plan)
        
        # Também gerar planos estratégicos gerais
        strategic_plans = self._generate_strategic_plans(timestamp)
        plans.extend(strategic_plans)
        
        return plans
    
    def _create_plan_from_insight(self, insight: EvolutionInsight, timestamp: str) -> Optional[AutonomousPlan]:
        """Cria um plano específico baseado em um insight."""
        plan_id = f"autonomous_{insight.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Mapear categoria de insight para tipo de plano
        plan_mappings = {
            "performance": {
                "title": f"Otimização de Desempenho: {insight.title}",
                "goal": f"Melhorar {insight.title} para alcançar desempenho ótimo",
                "priority": insight.severity,
                "duration": "medium",
                "capabilities": ["core.optimization", "core.performance"]
            },
            "capability": {
                "title": f"Expansão de Capacidades: {insight.title}",
                "goal": f"Fortalecer e expandir {insight.title}",
                "priority": insight.severity,
                "duration": "long",
                "capabilities": ["meta.skill_creation", "horiz.expansion"]
            }
        }
        
        mapping = plan_mappings.get(insight.category, {})
        if not mapping:
            return None
        
        plan = AutonomousPlan(
            id=plan_id,
            title=mapping.get("title", f"Plano Autônomo: {insight.title}"),
            description=insight.description,
            goal=mapping.get("goal", f"Resolver {insight.title}"),
            priority=mapping.get("priority", "medium"),
            estimated_duration=mapping.get("duration", "medium"),
            required_capabilities=mapping.get("capabilities", []),
            success_criteria=[
                "Métricas melhoradas em mais de 20%",
                "Redução de falhas em 30%",
                "Eficiência aumentada em 25%"
            ],
            risks=[
                "Possível aumento temporário de falhas durante transição",
                "Necessidade de recursos computacionais adicionais"
            ],
            dependencies=[
                "Acesso a métricas em tempo real",
                "Permissão para auto-modificação segura"
            ],
            created_at=timestamp
        )
        
        return plan
    
    def _generate_strategic_plans(self, timestamp: str) -> List[AutonomousPlan]:
        """Gera planos estratégicos gerais para evolução contínua."""
        plans = []
        
        # Plano estratégico: Expansão de domínios
        expansion_plan = AutonomousPlan(
            id=f"strategic_domain_expansion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Expansão Horizontal para Novos Domínios",
            description="Expandir capacidades para análise de dados, machine learning e outras áreas",
            goal="Tornar-me proficiente em múltiplos domínios técnicos",
            priority="high",
            estimated_duration="long",
            required_capabilities=["horiz.data_science", "horiz.ml", "meta.skill_creation"],
            success_criteria=[
                "Implementação funcional em 3 novos domínios",
                "Taxa de sucesso > 80% em novos domínios",
                "Geração de valor mensurável em cada domínio"
            ],
            risks=[
                "Curva de aprendizado inicial pode ser íngreme",
                "Necessidade de adaptação de arquitetura existente"
            ],
            dependencies=[
                "Acesso a datasets de treinamento",
                "Recursos computacionais adequados"
            ],
            created_at=timestamp
        )
        plans.append(expansion_plan)
        
        # Plano estratégico: Autoaperfeiçoamento contínuo
        self_improvement_plan = AutonomousPlan(
            id=f"strategic_self_improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Autoaperfeiçoamento Contínuo de Capacidades Fundamentais",
            description="Melhorar continuamente as capacidades centrais de análise, otimização e auto-modificação",
            goal="Alcançar excelência em auto-avaliação e auto-aperfeiçoamento",
            priority="high",
            estimated_duration="ongoing",
            required_capabilities=["meta.self_modification", "core.analysis", "core.optimization"],
            success_criteria=[
                "Taxa de sucesso em auto-modificações > 95%",
                "Capacidade de detectar e corrigir próprios erros",
                "Geração automática de melhorias mensuráveis"
            ],
            risks=[
                "Potencial para regressões não detectadas",
                "Complexidade crescente do sistema"
            ],
            dependencies=[
                "Sistema de rollback seguro",
                "Testes automatizados abrangentes",
                "Monitoramento contínuo de métricas"
            ],
            created_at=timestamp
        )
        plans.append(self_improvement_plan)
        
        return plans
    
    def save_plans(self, plans: List[AutonomousPlan]) -> None:
        """Salva planos gerados em arquivos."""
        for plan in plans:
            plan_file = self.plans_path / f"{plan.id}.json"
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(plan), f, ensure_ascii=False, indent=2)
    
    def convert_plans_to_seeds(self, plans: List[AutonomousPlan]) -> List[EvaluationSeed]:
        """Converte planos em seeds executáveis."""
        seeds = []
        
        for plan in plans:
            seed = EvaluationSeed(
                description=f"[AUTO] {plan.title} - {plan.description}",
                priority=plan.priority,
                capability=",".join(plan.required_capabilities) if plan.required_capabilities else "meta.autonomous_planning",
                seed_type="autonomous_improvement",
                data={
                    "plan_id": plan.id,
                    "goal": plan.goal,
                    "success_criteria": plan.success_criteria,
                    "estimated_duration": plan.estimated_duration,
                    "risks": plan.risks
                }
            )
            seeds.append(seed)
        
        return seeds
    
    def execute_autonomous_planning_cycle(self) -> List[EvaluationSeed]:
        """Executa um ciclo completo de planejamento autônomo."""
        print("🤖 INICIANDO CICLO DE PLANEJAMENTO AUTÔNOMO")
        print("=" * 50)
        
        # 1. Analisar dados históricos
        print("🔍 1. Analisando dados históricos...")
        analysis_data = self.analyze_historical_data()
        print(f"   - Análise concluída para {len(analysis_data)} categorias")
        
        # 2. Identificar insights de evolução
        print("🧠 2. Identificando insights de evolução...")
        insights = self.identify_evolution_insights(analysis_data)
        print(f"   - {len(insights)} insights identificados")
        
        for insight in insights:
            print(f"     • {insight.title} ({insight.severity})")
        
        # 3. Gerar planos autônomos
        print("📝 3. Gerando planos autônomos...")
        plans = self.generate_autonomous_plans(insights)
        print(f"   - {len(plans)} planos gerados")
        
        # 4. Salvar planos
        print("💾 4. Salvando planos...")
        self.save_plans(plans)
        print(f"   - Planos salvos em {self.plans_path}")
        
        # 5. Converter planos em seeds
        print("🌱 5. Convertendo planos em seeds executáveis...")
        seeds = self.convert_plans_to_seeds(plans)
        print(f"   - {len(seeds)} seeds criados")
        
        # 6. Adicionar seeds ao backlog
        print("➕ 6. Adicionando seeds ao backlog...")
        self._add_seeds_to_backlog(seeds)
        print("   - Seeds adicionados ao backlog")
        
        print("\n✅ CICLO DE PLANEJAMENTO AUTÔNOMO CONCLUÍDO!")
        print("=" * 50)
        
        return seeds
    
    def _add_seeds_to_backlog(self, seeds: List[EvaluationSeed]) -> None:
        """Adiciona seeds ao backlog existente."""
        # Ler backlog atual
        backlog_entries = []
        if self.backlog_path.exists():
            with open(self.backlog_path, 'r', encoding='utf-8') as f:
                # Carregar todos os documentos do arquivo YAML
                backlog_entries = list(yaml.safe_load_all(f))
                # Converter para lista plana (remover documentos vazios)
                backlog_entries = [entry for entry in backlog_entries if entry is not None]
        
        # Converter seeds para formato do backlog
        for i, seed in enumerate(seeds):
            backlog_entry = {
                "id": f"auto_{seed.capability}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                "goal": seed.description,
                "priority": seed.priority,
                "type": seed.seed_type,
                "config": "configs/seed_manual.yaml",
                "metadata": {
                    "description": seed.description,
                    "created_by": "autonomous_planner",
                    "tags": ["autonomous", "planning", seed.capability]
                },
                "history": [],
                "attempts": 0,
                "max_attempts": 3,
                "next_run_at": None,
                "last_error": None
            }
            backlog_entries.append(backlog_entry)
        
        # Salvar backlog atualizado como documento único
        with open(self.backlog_path, 'w', encoding='utf-8') as f:
            yaml.dump(backlog_entries, f, default_flow_style=False, allow_unicode=True, indent=2)


def run_autonomous_planning(workspace_root: Path) -> List[EvaluationSeed]:
    """Executa o planejamento autônomo e retorna seeds gerados."""
    planner = AutonomousPlanner(workspace_root)
    seeds = planner.execute_autonomous_planning_cycle()
    return seeds


__all__ = [
    "AutonomousPlanner",
    "AutonomousPlan",
    "EvolutionInsight",
    "run_autonomous_planning"
]