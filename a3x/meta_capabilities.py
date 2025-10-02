"""Meta-capabilities for autonomous skill creation in SeedAI."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import AutoEvaluator and EvaluationSeed locally to avoid circular import
from .autoeval import AutoEvaluator, EvaluationSeed
from .capabilities import CapabilityRegistry
from .config import AgentConfig
from .planning.mission_state import MissionState
from .planning.storage import load_mission_state
from .seeds import Seed, SeedBacklog


@dataclass
class MetaSkill:
    """Represents a meta-skill that can create other skills."""

    id: str
    name: str
    description: str
    implementation_template: str
    required_capabilities: list[str]
    estimated_complexity: float
    success_probability: float
    last_updated: str
    version: str = "0.1"


@dataclass
class SkillProposal:
    """Represents a proposed new skill to be created."""

    id: str
    name: str
    description: str
    implementation_plan: str
    required_dependencies: list[str]
    estimated_effort: float
    priority: str  # low, medium, high
    rationale: str
    target_domain: str
    created_at: str
    blueprint_path: str | None = None


class MetaCapabilityEngine:
    """Engine for autonomous skill creation and meta-capability development."""

    def __init__(self, config: AgentConfig, auto_evaluator: AutoEvaluator) -> None:
        self.config = config
        self.auto_evaluator = auto_evaluator
        self.workspace_root = Path(config.workspace.root).resolve()
        self.capabilities_path = self.workspace_root / "seed" / "capabilities.yaml"
        self.missions_path = self.workspace_root / "seed" / "missions.yaml"
        self.skill_records_path = self.workspace_root / "seed" / "skills"
        self.skill_records_path.mkdir(parents=True, exist_ok=True)
        # Maintain backwards compatibility for existing tests/utilities
        self.skills_path = self.skill_records_path
        self.skill_repo_path = self.workspace_root / "a3x" / "skills"
        self.skill_repo_path.mkdir(parents=True, exist_ok=True)
        self.skill_tests_path = (
            self.workspace_root / "tests" / "unit" / "a3x" / "skills"
        )
        self.skill_tests_path.mkdir(parents=True, exist_ok=True)
        self.backlog_path = self.workspace_root / "seed" / "backlog.yaml"

        # Load existing capabilities
        self.capability_registry = self._load_capability_registry()

        # Define meta-skills for autonomous skill creation
        self.meta_skills = self._define_meta_skills()

    @staticmethod
    def _slugify(value: str, *, separator: str = "_") -> str:
        slug = re.sub(r"[^a-z0-9]+", separator, value.lower()).strip(separator)
        return slug or "nova_skill"

    def _load_capability_registry(self) -> CapabilityRegistry:
        """Load the existing capability registry."""
        try:
            if self.capabilities_path.exists():
                return CapabilityRegistry.from_yaml(self.capabilities_path)
            else:
                # Create a default registry if none exists
                return CapabilityRegistry(capabilities={}, raw_entries={})
        except Exception:
            # Return empty registry on error
            return CapabilityRegistry(capabilities={}, raw_entries={})

    def _define_meta_skills(self) -> dict[str, MetaSkill]:
        """Define meta-skills for autonomous skill creation."""
        meta_skills = {
            "skill_creator": MetaSkill(
                id="meta.skill_creator",
                name="Criador de Habilidades",
                description="Capacidade de criar novas habilidades autonomamente com base em necessidades identificadas",
                implementation_template="""
class {{skill_name}}:
    \"\"\"{{skill_description}}\"\"\"
    
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        # Initialize skill-specific components
        
    def execute(self, action: AgentAction) -> Observation:
        \"\"\"Execute the skill.\"\"\"
        try:
            # Implementation goes here
            pass
        except Exception as e:
            return Observation(success=False, error=str(e))
        
        return Observation(success=True, output="Skill executed successfully")
""",
                required_capabilities=["core.diffing", "core.testing", "horiz.python"],
                estimated_complexity=0.7,
                success_probability=0.8,
                last_updated=datetime.now(timezone.utc).isoformat(),
                version="0.1"
            ),
            "domain_expander": MetaSkill(
                id="meta.domain_expander",
                name="Expansor de Domínios",
                description="Capacidade de expandir para novos domínios com base em análise de necessidades",
                implementation_template="""
class {{domain_skill_name}}:
    \"\"\"{{domain_description}}\"\"\"
    
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        # Initialize domain-specific components
        
    def analyze(self, data: Any) -> Dict[str, Any]:
        \"\"\"Analyze data in the domain.\"\"\"
        # Analysis implementation
        pass
        
    def suggest_improvements(self) -> List[str]:
        \"\"\"Suggest improvements for the domain.\"\"\"
        # Improvement suggestions
        pass
""",
                required_capabilities=["core.diffing", "horiz.python", "core.testing"],
                estimated_complexity=0.8,
                success_probability=0.7,
                last_updated=datetime.now(timezone.utc).isoformat(),
                version="0.1"
            ),
            "skill_optimizer": MetaSkill(
                id="meta.skill_optimizer",
                name="Otimizador de Habilidades",
                description="Capacidade de otimizar habilidades existentes para melhor performance",
                implementation_template="""
class {{optimized_skill_name}}:
    \"\"\"{{optimization_description}}\"\"\"
    
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        # Optimized initialization
        
    def execute_optimized(self, action: AgentAction) -> Observation:
        \"\"\"Execute optimized version of the skill.\"\"\"
        # Optimized implementation
        pass
""",
                required_capabilities=["core.diffing", "core.testing", "horiz.python"],
                estimated_complexity=0.6,
                success_probability=0.9,
                last_updated=datetime.now(timezone.utc).isoformat(),
                version="0.1"
            )
        }

        return meta_skills

    def propose_new_skills(self) -> list[SkillProposal]:
        """Propose new skills based on current capabilities and needs."""
        proposals = []

        # Analyze current capabilities and identify gaps
        capability_gaps = self._identify_capability_gaps()

        # Analyze mission requirements and identify needed skills
        mission_needs = self._analyze_mission_needs()

        # Analyze performance metrics and identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities()

        # Generate proposals based on analysis
        for gap in capability_gaps:
            proposal = self._create_skill_proposal_for_gap(gap)
            if proposal:
                proposals.append(proposal)

        for need in mission_needs:
            proposal = self._create_skill_proposal_for_need(need)
            if proposal:
                proposals.append(proposal)

        for opportunity in optimization_opportunities:
            proposal = self._create_skill_proposal_for_optimization(opportunity)
            if proposal:
                proposals.append(proposal)

        return proposals

    def _identify_capability_gaps(self) -> list[dict[str, Any]]:
        """Identify gaps in current capabilities."""
        gaps = []

        # Load metrics history to analyze capability performance
        try:
            metrics_history = self.auto_evaluator._read_metrics_history()
        except Exception:
            metrics_history = {}

        # Check for capabilities with consistently low performance
        capability_metrics = {}
        for metric_name, values in metrics_history.items():
            if "." in metric_name:
                capability, metric = metric_name.rsplit(".", 1)
                if capability not in capability_metrics:
                    capability_metrics[capability] = {}
                capability_metrics[capability][metric] = values[-1] if values else 0.0

        # Identify capabilities below threshold
        for capability, metrics in capability_metrics.items():
            # Check success rate
            if "success_rate" in metrics and metrics["success_rate"] < 0.7:
                gaps.append({
                    "type": "low_performance",
                    "capability": capability,
                    "metric": "success_rate",
                    "value": metrics["success_rate"],
                    "description": f"Baixo desempenho em {capability} (taxa de sucesso: {metrics['success_rate']:.2f})"
                })

            # Check if capability is missing entirely
            if capability not in self.capability_registry._by_id:
                gaps.append({
                    "type": "missing_capability",
                    "capability": capability,
                    "description": f"Capacidade {capability} ausente no registro"
                })

        return gaps

    def _analyze_mission_needs(self) -> list[dict[str, Any]]:
        """Analyze mission requirements to identify needed skills."""
        needs = []

        # Load mission state
        try:
            mission_state = load_mission_state(self.missions_path)
        except Exception:
            mission_state = MissionState()

        # Analyze missions for unmet requirements
        for mission in mission_state.missions:
            # Check if mission has unmet capabilities
            for tag in mission.capability_tags:
                if tag not in self.capability_registry._by_id:
                    needs.append({
                        "type": "mission_requirement",
                        "mission_id": mission.id,
                        "capability_tag": tag,
                        "description": f"Missão {mission.id} requer capacidade {tag} não implementada"
                    })

            # Check mission telemetry for performance issues
            if mission.telemetry:
                # Look for metrics indicating need for new capabilities
                for metric_name, summary in mission.telemetry.metric_summaries.items():
                    if summary.current and summary.current < 0.5:  # Below 50% threshold
                        needs.append({
                            "type": "performance_improvement",
                            "mission_id": mission.id,
                            "metric": metric_name,
                            "value": summary.current,
                            "description": f"Missão {mission.id} tem baixo desempenho em {metric_name} ({summary.current:.2f})"
                        })

        return needs

    def _identify_optimization_opportunities(self) -> list[dict[str, Any]]:
        """Identify opportunities to optimize existing capabilities."""
        opportunities = []

        # Load metrics history
        try:
            metrics_history = self.auto_evaluator._read_metrics_history()
        except Exception:
            metrics_history = {}

        # Look for capabilities with declining performance or high resource usage
        capability_metrics = {}
        for metric_name, values in metrics_history.items():
            if "." in metric_name:
                capability, metric = metric_name.rsplit(".", 1)
                if capability not in capability_metrics:
                    capability_metrics[capability] = {}
                capability_metrics[capability][metric] = values[-5:] if len(values) >= 5 else values  # Last 5 values

        # Analyze trends
        for capability, metrics in capability_metrics.items():
            # Check for declining performance trend
            if "success_rate" in metrics:
                recent_values = metrics["success_rate"]
                if len(recent_values) >= 3:
                    # Calculate trend
                    trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                    if trend < -0.05:  # Declining more than 5% per measurement
                        opportunities.append({
                            "type": "declining_performance",
                            "capability": capability,
                            "trend": trend,
                            "description": f"Desempenho de {capability} em declínio ({trend:.3f} por medição)"
                        })

            # Check for high resource usage
            if "execution_time_avg" in metrics:
                recent_times = metrics["execution_time_avg"]
                if recent_times and max(recent_times) > 5.0:  # More than 5 seconds average
                    opportunities.append({
                        "type": "high_resource_usage",
                        "capability": capability,
                        "avg_time": max(recent_times),
                        "description": f"{capability} consome recursos excessivos (tempo médio: {max(recent_times):.2f}s)"
                    })

        return opportunities

    def _create_skill_proposal_for_gap(self, gap: dict[str, Any]) -> SkillProposal | None:
        """Create a skill proposal to address a capability gap."""
        gap_type = gap.get("type", "")
        capability = gap.get("capability", "")

        if gap_type == "low_performance":
            return SkillProposal(
                id=f"skill_{capability.replace('.', '_')}_improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"Melhoria de {capability}",
                description=f"Habilidade para melhorar o desempenho de {capability}",
                implementation_plan=f"Implementar otimizações específicas para a capacidade {capability} com base na análise de métricas",
                required_dependencies=[capability],
                estimated_effort=3.0,  # Medium effort
                priority="high",
                rationale=gap.get("description", ""),
                target_domain="core",
                created_at=datetime.now(timezone.utc).isoformat()
            )
        elif gap_type == "missing_capability":
            return SkillProposal(
                id=f"skill_{capability.replace('.', '_')}_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"Criação de {capability}",
                description=f"Habilidade para implementar a capacidade {capability}",
                implementation_plan=f"Implementar a capacidade {capability} com base nas necessidades identificadas",
                required_dependencies=[],
                estimated_effort=5.0,  # High effort
                priority="medium",
                rationale=gap.get("description", ""),
                target_domain="core",
                created_at=datetime.now(timezone.utc).isoformat()
            )

        return None

    def _create_skill_proposal_for_need(self, need: dict[str, Any]) -> SkillProposal | None:
        """Create a skill proposal to address a mission need."""
        need_type = need.get("type", "")
        mission_id = need.get("mission_id", "")
        capability_tag = need.get("capability_tag", "") or need.get("metric", "")

        if need_type == "mission_requirement":
            return SkillProposal(
                id=f"skill_{capability_tag.replace('.', '_')}_for_mission_{mission_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"{capability_tag} para Missão {mission_id}",
                description=f"Habilidade {capability_tag} necessária para a missão {mission_id}",
                implementation_plan=f"Implementar a capacidade {capability_tag} para atender aos requisitos da missão {mission_id}",
                required_dependencies=[capability_tag],
                estimated_effort=4.0,  # Medium-high effort
                priority="high",
                rationale=need.get("description", ""),
                target_domain="mission_specific",
                created_at=datetime.now(timezone.utc).isoformat()
            )
        elif need_type == "performance_improvement":
            metric = need.get("metric", "")
            return SkillProposal(
                id=f"skill_{capability_tag.replace('.', '_')}_perf_improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"Otimização de {metric} para {mission_id}",
                description=f"Otimização de desempenho para {metric} na missão {mission_id}",
                implementation_plan=f"Otimizar {metric} na missão {mission_id} para melhorar o desempenho",
                required_dependencies=[capability_tag],
                estimated_effort=2.0,  # Low-medium effort
                priority="medium",
                rationale=need.get("description", ""),
                target_domain="performance",
                created_at=datetime.now(timezone.utc).isoformat()
            )

        return None

    def _create_skill_proposal_for_optimization(self, opportunity: dict[str, Any]) -> SkillProposal | None:
        """Create a skill proposal to address an optimization opportunity."""
        opp_type = opportunity.get("type", "")
        capability = opportunity.get("capability", "")

        if opp_type == "declining_performance":
            trend = opportunity.get("trend", 0)
            return SkillProposal(
                id=f"skill_{capability.replace('.', '_')}_perf_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"Recuperação de Performance de {capability}",
                description=f"Habilidade para recuperar performance de {capability} (tendência: {trend:.3f})",
                implementation_plan=f"Implementar correções para reverter a tendência de declínio de performance em {capability}",
                required_dependencies=[capability],
                estimated_effort=3.0,  # Medium effort
                priority="high",
                rationale=opportunity.get("description", ""),
                target_domain="performance",
                created_at=datetime.now(timezone.utc).isoformat()
            )
        elif opp_type == "high_resource_usage":
            avg_time = opportunity.get("avg_time", 0)
            return SkillProposal(
                id=f"skill_{capability.replace('.', '_')}_resource_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"Otimização de Recursos de {capability}",
                description=f"Habilidade para otimizar uso de recursos em {capability} (tempo médio: {avg_time:.2f}s)",
                implementation_plan=f"Implementar otimizações para reduzir consumo de recursos em {capability}",
                required_dependencies=[capability],
                estimated_effort=4.0,  # Medium-high effort
                priority="medium",
                rationale=opportunity.get("description", ""),
                target_domain="performance",
                created_at=datetime.now(timezone.utc).isoformat()
            )

        return None

    def generate_skill_implementation(self, proposal: SkillProposal) -> str:
        """Generate implementation code for a proposed skill."""
        # Select appropriate meta-skill template based on proposal type
        template = ""
        if "otimização" in proposal.name.lower() or "otimização" in proposal.description.lower():
            template = self.meta_skills["skill_optimizer"].implementation_template
        elif "missão" in proposal.name.lower() or "missão" in proposal.description.lower():
            template = self.meta_skills["domain_expander"].implementation_template
        else:
            template = self.meta_skills["skill_creator"].implementation_template

        # Fill template with proposal details
        implementation = template.replace("{{skill_name}}", proposal.name.replace(" ", ""))
        implementation = implementation.replace("{{skill_description}}", proposal.description)
        implementation = implementation.replace("{{domain_skill_name}}", proposal.name.replace(" ", ""))
        implementation = implementation.replace("{{domain_description}}", proposal.description)
        implementation = implementation.replace("{{optimized_skill_name}}", proposal.name.replace(" ", ""))
        implementation = implementation.replace("{{optimization_description}}", proposal.description)

        return implementation

    def create_skill_seed(self, proposal: SkillProposal) -> EvaluationSeed:
        """Create an evaluation seed for implementing the proposed skill."""
        implementation_code = self.generate_skill_implementation(proposal)

        skill_slug = self._slugify(proposal.name)
        skill_file = self.skill_repo_path / f"{skill_slug}.py"
        test_file = self.skill_tests_path / f"test_{skill_slug}.py"
        proposal_record = self.skill_records_path / f"{proposal.id}.json"
        blueprint_path = self.skill_records_path / f"{proposal.id}.py"
        blueprint_path.write_text(implementation_code, encoding="utf-8")

        proposal.blueprint_path = str(blueprint_path.relative_to(self.workspace_root))

        description = (
            f"Criar habilidade '{proposal.name}' salvando o código em "
            f"{skill_file.relative_to(self.workspace_root)} e criando testes em "
            f"{test_file.relative_to(self.workspace_root)}. Utilize o blueprint "
            f"registrado em {proposal.blueprint_path} e atualize o registro "
            f"{proposal_record.relative_to(self.workspace_root)}."
        )

        seed = EvaluationSeed(
            description=description,
            priority=proposal.priority,
            capability=f"meta.skill_creation.{proposal.target_domain}",
            seed_type="skill_creation",
            data={
                "proposal_id": proposal.id,
                "skill_name": proposal.name,
                "implementation_plan": proposal.implementation_plan,
                "estimated_effort": str(proposal.estimated_effort),
                "rationale": proposal.rationale,
                "target_domain": proposal.target_domain,
                "generated_code": implementation_code,
                "skill_slug": skill_slug,
                "skill_module": str(skill_file.relative_to(self.workspace_root)),
                "skill_test_module": str(test_file.relative_to(self.workspace_root)),
                "proposal_record": str(proposal_record.relative_to(self.workspace_root)),
                "blueprint_file": proposal.blueprint_path,
            }
        )

        return seed

    def evaluate_proposal_feasibility(self, proposal: SkillProposal) -> tuple[bool, float, str]:
        """Evaluate the feasibility of implementing a skill proposal."""
        # Check if required dependencies are available
        missing_deps = []
        for dep in proposal.required_dependencies:
            if dep not in self.capability_registry._by_id:
                missing_deps.append(dep)

        if missing_deps:
            return False, 0.1, f"Faltando dependências: {', '.join(missing_deps)}"

        # Check estimated effort vs available resources
        if proposal.estimated_effort > 10.0:  # Too complex
            return False, 0.2, f"Esforço estimado muito alto: {proposal.estimated_effort}"

        # Check priority vs current workload
        if proposal.priority == "low" and len(self._get_pending_seeds()) > 5:
            return False, 0.3, "Prioridade baixa e backlog cheio"

        # Everything looks good
        feasibility_score = 1.0 - (len(missing_deps) * 0.2)  # Reduce score for each missing dep
        feasibility_score = max(0.1, feasibility_score)  # Minimum score

        return True, feasibility_score, "Viável para implementação"

    def _get_pending_seeds(self) -> list[Seed]:
        """Return pending seeds currently registered in the backlog."""
        try:
            backlog = SeedBacklog.load(self.backlog_path)
        except Exception:
            return []
        try:
            return backlog.list_pending()
        except Exception:
            return []

    def save_skill_proposal(self, proposal: SkillProposal) -> None:
        """Save a skill proposal to disk."""
        proposal_file = self.skill_records_path / f"{proposal.id}.json"
        with proposal_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(proposal), f, ensure_ascii=False, indent=2)

    def load_skill_proposals(self) -> list[SkillProposal]:
        """Load saved skill proposals from disk."""
        proposals = []
        for proposal_file in self.skill_records_path.glob("*.json"):
            try:
                with proposal_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    proposal = SkillProposal(**data)
                    proposals.append(proposal)
            except Exception:
                # Skip invalid files
                continue

        return proposals

    def _analyze_ast_complexity(self, tree: ast.AST) -> dict[str, float]:
        """Analyze AST for complexity metrics."""
        stats = {
            "function_count": 0,
            "class_count": 0,
            "total_nodes": 0,
            "max_depth": 0,
        }

        def count_nodes(node, depth=0):
            stats["total_nodes"] += 1
            stats["max_depth"] = max(stats["max_depth"], depth)

            if isinstance(node, ast.FunctionDef):
                stats["function_count"] += 1
            elif isinstance(node, ast.ClassDef):
                stats["class_count"] += 1

            for child in ast.iter_child_nodes(node):
                count_nodes(child, depth + 1)

        for node in ast.iter_child_nodes(tree):
            count_nodes(node)

        return {
            "ast_function_count": float(stats["function_count"]),
            "ast_class_count": float(stats["class_count"]),
            "ast_total_nodes": float(stats["total_nodes"]),
            "ast_max_depth": float(stats["max_depth"]),
        }

    def _extract_python_code_from_patch(self, diff: str) -> str:
        """Extract actual Python code from patch content."""
        lines = diff.split("\n")
        python_code = []

        in_diff = False
        for line in lines:
            if line.startswith("+++ ") and line.endswith(".py"):
                in_diff = True
                continue
            elif line.startswith("--- ") or line.startswith("@@ "):
                continue

            if in_diff and (line.startswith("+") or line.startswith(" ")):
                # Lines being added or context lines
                code_line = line[1:]  # Remove the prefix (+ or space)
                python_code.append(code_line)

        return "\n".join(python_code)

    def analyze_code_complexity_from_patch(self, patch_content: str) -> dict[str, float]:
        """Analyze code complexity from a patch/diff content."""
        complexity_metrics = {}

        # Extract Python code from the patch
        python_code = self._extract_python_code_from_patch(patch_content)

        if python_code:
            # Analyze the Python code for complexity
            try:
                tree = ast.parse(python_code)
                complexity_info = self._analyze_ast_complexity(tree)
                complexity_metrics.update(complexity_info)
            except SyntaxError:
                # If parsing fails, skip complexity analysis for this patch
                pass

        return complexity_metrics

    def _check_code_quality_issues(self, quality_metrics: dict[str, float]) -> list[EvaluationSeed]:
        """Generate seeds based on code quality issues."""
        seeds = []

        # Check if there are too many failures (indicating poor implementation quality)
        if quality_metrics.get("failure_rate", 0) > 0.3:  # More than 30% failure rate
            seeds.append(
                EvaluationSeed(
                    description="Reduzir taxa de falhas durante execução (alta taxa de falhas detectada).",
                    priority="high",
                    capability="core.execution",
                    seed_type="quality",
                    data={"metric": "failure_rate", "value": str(quality_metrics.get("failure_rate", 0))}
                )
            )

        # Check if too many patches are being applied without proper success
        if (quality_metrics.get("apply_patch_count", 0) > 5 and
            quality_metrics.get("success_rate", 1) < 0.7):
            seeds.append(
                EvaluationSeed(
                    description="Melhorar qualidade das alterações de código aplicadas (muitos patches com baixa taxa de sucesso).",
                    priority="medium",
                    capability="core.diffing",
                    seed_type="quality",
                    data={"metric": "apply_patch_success_rate", "value": str(quality_metrics.get("success_rate", 1))}
                )
            )

        # Check if the system is not diversifying file types enough (might indicate lack of features)
        if quality_metrics.get("file_diversity", 0) < 2 and quality_metrics.get("apply_patch_count", 0) > 10:
            seeds.append(
                EvaluationSeed(
                    description="Expandir diversidade de tipos de arquivos manipulados (sistema focado em poucos tipos de arquivos).",
                    priority="low",
                    capability="horiz.file_handling",
                    seed_type="quality",
                    data={"metric": "file_diversity", "value": str(quality_metrics.get("file_diversity", 0))}
                )
            )

        return seeds


# Integration with existing system
def integrate_meta_capabilities(config: AgentConfig, auto_evaluator: AutoEvaluator) -> None:
    """Integrate meta-capabilities into the existing system."""
    # Create meta-capability engine
    meta_engine = MetaCapabilityEngine(config, auto_evaluator)

    # Propose new skills based on current state
    proposals = meta_engine.propose_new_skills()

    # Evaluate and prioritize proposals
    prioritized_proposals = []
    for proposal in proposals:
        is_feasible, score, reason = meta_engine.evaluate_proposal_feasibility(proposal)
        if is_feasible:
            proposal.estimated_effort *= score  # Adjust effort based on feasibility
            prioritized_proposals.append((score, proposal))

    # Sort by feasibility score (highest first)
    prioritized_proposals.sort(reverse=True)

    # Create seeds for top proposals
    top_proposals = [proposal for _, proposal in prioritized_proposals[:3]]  # Top 3 proposals
    seeds = []
    for proposal in top_proposals:
        seed = meta_engine.create_skill_seed(proposal)
        seeds.append(seed)
        meta_engine.save_skill_proposal(proposal)  # Save for future reference

    # Add seeds to backlog (this would normally be done by the auto-evaluator)
    # For now, just return the seeds
    return seeds


__all__ = [
    "MetaCapabilityEngine",
    "MetaSkill",
    "SkillProposal",
    "integrate_meta_capabilities",
]
