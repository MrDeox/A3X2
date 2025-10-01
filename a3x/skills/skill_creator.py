"""Ferramentas para materializar habilidades a partir de propostas."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from textwrap import dedent
from typing import Tuple

from ..meta_capabilities import SkillProposal


class SkillCreator:
    """Cria implementações reais de skills com base em propostas aprovadas."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.skills_dir = workspace_root / "a3x" / "skills"
        self.tests_dir = workspace_root / "tests" / "unit" / "a3x" / "skills"
        self.registry_dir = workspace_root / "seed" / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def create_skill_from_proposal(self, proposal: SkillProposal) -> Tuple[bool, str]:
        """Materializa uma skill e seus testes a partir de uma proposta."""
        slug = self._slugify(proposal.name)
        skill_path = self.skills_dir / f"{slug}.py"
        test_path = self.tests_dir / f"test_{slug}.py"

        try:
            implementation = self._resolve_implementation(proposal)
            skill_path.write_text(implementation, encoding="utf-8")

            test_content = self._generate_skill_test(proposal, slug)
            test_path.write_text(test_content, encoding="utf-8")

            self._register_skill(proposal, slug, skill_path, test_path)
            message = (
                f"Skill '{proposal.name}' criada em "
                f"{skill_path.relative_to(self.workspace_root)}"
                f" com testes em {test_path.relative_to(self.workspace_root)}"
            )
            return True, message
        except Exception as exc:  # pragma: no cover - protegido em testes específicos
            return False, f"Falha ao criar skill '{proposal.name}': {exc}"

    def _resolve_implementation(self, proposal: SkillProposal) -> str:
        if proposal.blueprint_path:
            blueprint_path = Path(proposal.blueprint_path)
            if not blueprint_path.is_absolute():
                blueprint_path = self.workspace_root / blueprint_path
            if not blueprint_path.exists():
                raise FileNotFoundError(
                    f"Blueprint não encontrada em {blueprint_path}"
                )
            return blueprint_path.read_text(encoding="utf-8")
        return self._generate_skill_implementation(proposal)

    def _generate_skill_implementation(self, proposal: SkillProposal) -> str:
        class_name = proposal.name.replace(" ", "").replace("-", "")
        description = proposal.description or "Implementação de skill gerada automaticamente."
        escaped_description = description.replace("\"", "\\\"")
        skill_label = proposal.name.replace("\"", "\\\"")
        return dedent(
            f'''"""{escaped_description}"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..actions import AgentAction, Observation
from ..config import AgentConfig


@dataclass
class {class_name}:
    """{description}"""

    config: AgentConfig
    capabilities: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Inicializa componentes específicos da skill."""
        # TODO: Personalizar inicialização com base no plano da proposta

    def execute(self, action: AgentAction) -> Observation:
        """Executa a skill com a ação fornecida."""
        try:
            result = self._perform_action(action)
            return Observation(success=True, output=result)
        except Exception as exc:  # pragma: no cover - comportamento defensivo
            return Observation(success=False, error=str(exc))

    def _perform_action(self, action: AgentAction) -> str:
        """Implementa a lógica específica da skill."""
        return f"Executed {skill_label} with action: {{action.text or 'no text'}}"

    def analyze(self, data: Dict[str, object]) -> Dict[str, object]:
        """Realiza análises auxiliares previstas na proposta."""
        return {"analysis_result": "placeholder"}

    def suggest_improvements(self) -> List[str]:
        """Sugere evoluções para a skill."""
        return ["Implement core functionality", "Add more comprehensive tests"]
'''
        )

    def _generate_skill_test(self, proposal: SkillProposal, slug: str) -> str:
        class_name = proposal.name.replace(" ", "").replace("-", "")
        skill_label = proposal.name.replace("\"", "\\\"")
        return dedent(
            f'''"""Testes automáticos para a skill {proposal.name}."""

from unittest.mock import Mock
from pathlib import Path

from a3x.skills.{slug} import {class_name}
from a3x.actions import AgentAction, ActionType, Observation
from a3x.config import AgentConfig


class Test{class_name}:
    """Cenários básicos de validação da skill."""

    def setup_method(self) -> None:
        self.mock_config = Mock(spec=AgentConfig)
        self.mock_config.workspace = Mock()
        self.mock_config.workspace.root = Path("/tmp/test")
        self.mock_config.policies = Mock()
        self.mock_config.policies.allow_network = False
        self.mock_config.policies.deny_commands = []
        self.mock_config.audit = Mock()
        self.mock_config.audit.enable_file_log = True
        self.mock_config.audit.file_dir = Path("seed/changes")
        self.mock_config.audit.enable_git_commit = False
        self.mock_config.audit.commit_prefix = "A3X"
        self.skill = {class_name}(config=self.mock_config)

    def test_skill_initialization(self) -> None:
        assert isinstance(self.skill, {class_name})
        assert hasattr(self.skill, "config")

    def test_skill_execute_basic_action(self) -> None:
        action = AgentAction(type=ActionType.MESSAGE, text="Test action")
        observation = self.skill.execute(action)

        assert isinstance(observation, Observation)
        assert observation.success is True
        assert "Executed {skill_label}" in observation.output

    def test_skill_analyze_data(self) -> None:
        result = self.skill.analyze({"key": "value"})

        assert isinstance(result, dict)
        assert "analysis_result" in result

    def test_skill_suggest_improvements(self) -> None:
        suggestions = self.skill.suggest_improvements()

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
'''
        )

    def _register_skill(
        self,
        proposal: SkillProposal,
        slug: str,
        skill_path: Path,
        test_path: Path,
    ) -> None:
        record_path = self.registry_dir / f"{proposal.id}.json"
        if record_path.exists():
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                record = {}
        else:
            record = asdict(proposal)

        record.update(
            {
                "id": proposal.id,
                "name": proposal.name,
                "slug": slug,
                "description": proposal.description,
                "skill_path": str(skill_path.relative_to(self.workspace_root)),
                "test_path": str(test_path.relative_to(self.workspace_root)),
                "blueprint_path": proposal.blueprint_path,
                "status": "implemented",
            }
        )

        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _slugify(value: str) -> str:
        normalized = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
        normalized = normalized.strip("_")
        return normalized or "nova_skill"
