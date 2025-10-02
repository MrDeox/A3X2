"""Skill creator module for materializing skills from proposals."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from textwrap import dedent

from ..meta_capabilities import SkillProposal


class SkillCreator:
    """Creates actual skill implementations from proposals."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.skills_dir = workspace_root / "a3x" / "skills"
        self.tests_dir = workspace_root / "tests" / "unit" / "a3x" / "skills"
        self.registry_dir = workspace_root / "seed" / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def create_skill_from_proposal(self, proposal: SkillProposal) -> tuple[bool, str]:
        """Create an actual skill implementation from a proposal."""
        slug = self._slugify(proposal.name)
        skill_filename = f"{slug}.py"
        skill_path = self.skills_dir / skill_filename
        test_filename = f"test_{slug}.py"
        test_path = self.tests_dir / test_filename

        try:
            implementation = self._resolve_implementation(proposal)
            skill_path.write_text(implementation, encoding="utf-8")

            test_implementation = self._generate_skill_test(proposal, slug)
            test_path.write_text(test_implementation, encoding="utf-8")

            self._register_skill(proposal, slug, skill_path, test_path)
            return True, f"Skill '{proposal.name}' created successfully at {skill_path}"
        except Exception as exc:
            return False, f"Failed to create skill '{proposal.name}': {exc}"

    def _resolve_implementation(self, proposal: SkillProposal) -> str:
        if proposal.blueprint_path:
            blueprint_path = Path(proposal.blueprint_path)
            if not blueprint_path.is_absolute():
                blueprint_path = self.workspace_root / blueprint_path
            if not blueprint_path.exists():
                raise FileNotFoundError(f"Blueprint not found at {blueprint_path}")
            return blueprint_path.read_text(encoding="utf-8")
        return self._generate_skill_implementation(proposal)

    def _generate_skill_implementation(self, proposal: SkillProposal) -> str:
        """Generate the actual implementation code for a skill."""
        class_name = proposal.name.replace(" ", "").replace("-", "")
        description = proposal.description or "Generated skill implementation."
        return dedent(
            f'''
            """{description}"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

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
        """Initialize skill-specific components."""
        pass  # Customize based on proposal

    def execute(self, action: AgentAction) -> Observation:
        """Execute the skill with the provided action."""
        try:
            result = self._perform_action(action)
            return Observation(success=True, output=result)
        except Exception as exc:
            return Observation(success=False, error=str(exc))

    def _perform_action(self, action: AgentAction) -> str:
        """Perform the specific logic for this skill."""
        return f"Executed {proposal.name} with action: {{action.text or 'no text'}}"

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis as per proposal."""
        return {{"analysis_result": "placeholder"}}

    def suggest_improvements(self) -> List[str]:
        """Suggest improvements for this skill."""
        return ["Implement core functionality", "Add more comprehensive tests"]
'''
        )

    def _generate_skill_test(self, proposal: SkillProposal, slug: str) -> str:
        """Generate a test file for the skill."""
        class_name = proposal.name.replace(" ", "").replace("-", "")
        return dedent(
            f'''
            """Tests for {proposal.name} skill."""

from unittest.mock import Mock
from pathlib import Path

from a3x.skills.{slug} import {class_name}
from a3x.actions import AgentAction, ActionType, Observation
from a3x.config import AgentConfig


class Test{class_name}:
    """Basic validation scenarios for the skill."""

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
        assert "Executed" in observation.output

    def test_skill_analyze_data(self) -> None:
        result = self.skill.analyze({{"key": "value"}})

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
        record = asdict(proposal)
        record.update(
            {
                "slug": slug,
                "skill_path": str(skill_path.relative_to(self.workspace_root)),
                "test_path": str(test_path.relative_to(self.workspace_root)),
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
        return normalized or "skill"
