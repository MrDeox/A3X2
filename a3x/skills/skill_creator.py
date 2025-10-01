from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Tuple

from a3x.meta_capabilities import SkillProposal
from a3x.utils import slugify


class SkillCreator:
    """Creates actual skill implementations from proposals."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.skills_dir = workspace_root / "a3x" / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir = (
            workspace_root / "tests" / "unit" / "a3x" / "skills"
        )

    def create_skill_from_proposal(self, proposal: SkillProposal) -> Tuple[bool, str]:
        """Create an actual skill implementation from a proposal."""
        try:
            implementation = self._generate_skill_implementation(proposal)

            skill_slug = self._slugify(proposal.name)
            skill_slug, skill_path, test_path = self._resolve_unique_paths(skill_slug)
            skill_filename = f"{skill_slug}.py"
            skill_path.write_text(implementation, encoding="utf-8")

            test_path.parent.mkdir(parents=True, exist_ok=True)

            test_implementation = self._generate_skill_test(proposal, skill_filename)
            test_path.write_text(test_implementation, encoding="utf-8")

            return True, f"Skill '{proposal.name}' created successfully at {skill_path}"
        except Exception as exc:  # pragma: no cover - defensive branch
            return False, f"Failed to create skill '{proposal.name}': {exc}"

    @staticmethod
    def _slugify(value: str) -> str:
        """Generate a filesystem-friendly slug for skill names."""

        return slugify(value)

    def _resolve_unique_paths(self, base_slug: str) -> Tuple[str, Path, Path]:
        """Ensure skill and test paths do not overwrite existing files."""

        candidate_slug = base_slug
        counter = 1

        while True:
            skill_path = self.skills_dir / f"{candidate_slug}.py"
            test_path = self.tests_dir / f"test_{candidate_slug}.py"

            if not skill_path.exists() and not test_path.exists():
                return candidate_slug, skill_path, test_path

            counter += 1
            candidate_slug = f"{base_slug}_{counter}"

    def _generate_skill_implementation(self, proposal: SkillProposal) -> str:
        """Generate the actual implementation code for a skill."""
        class_name = proposal.name.replace(" ", "").replace("-", "")
        implementation = dedent(
            f'''
            """{proposal.description}"""

            from __future__ import annotations

            from dataclasses import dataclass, field
            from typing import Any, Dict, List

            from ..actions import AgentAction, Observation
            from ..config import AgentConfig


            @dataclass
            class {class_name}:
                """{proposal.description}"""

                config: AgentConfig
                # Add skill-specific fields based on proposal
                capabilities: List[str] = field(default_factory=list)

                def __post_init__(self) -> None:
                    """Initialize skill-specific components."""
                    self._initialize_components()

                def _initialize_components(self) -> None:
                    """Initialize skill-specific components."""
                    # Implementation goes here based on proposal requirements
                    pass

                def execute(self, action: AgentAction) -> Observation:
                    """Execute the skill."""
                    try:
                        # Implementation based on proposal
                        result = self._perform_action(action)
                        return Observation(success=True, output=result)
                    except Exception as exc:  # pragma: no cover - placeholder logic
                        return Observation(success=False, error=str(exc))

                def _perform_action(self, action: AgentAction) -> str:
                    """Perform the specific action for this skill."""
                    # Actual implementation would go here
                    # For now, return a placeholder result
                    return (
                        f"Executed {proposal.name} with action: "
                        f"{{action.text or 'no text'}}"
                    )

                def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
                    """Analyze data related to this skill."""
                    # Analysis implementation based on proposal
                    return {{"analysis_result": "placeholder"}}

                def suggest_improvements(self) -> List[str]:
                    """Suggest improvements for this skill."""
                    # Improvement suggestions based on proposal
                    return ["Implement core functionality", "Add more comprehensive tests"]
            '''
        ).strip("\n")
        return f"{implementation}\n"

    def _generate_skill_test(self, proposal: SkillProposal, skill_filename: str) -> str:
        """Generate a test file for the skill."""
        skill_class_name = proposal.name.replace(" ", "").replace("-", "")
        test_implementation = dedent(
            f'''
            """Tests for {proposal.name} skill."""

            import pytest
            from unittest.mock import Mock, patch
            from pathlib import Path

            from a3x.skills.{skill_filename[:-3]} import {skill_class_name}
            from a3x.actions import AgentAction, ActionType, Observation
            from a3x.config import AgentConfig


            class Test{skill_class_name}:
                """Tests for {proposal.name}."""

                def setup_method(self) -> None:
                    """Setup before each test."""
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
                    self.skill = {skill_class_name}(config=self.mock_config)

                def test_skill_initialization(self) -> None:
                    """Test {proposal.name} initialization."""
                    assert isinstance(self.skill, {skill_class_name})
                    assert hasattr(self.skill, "config")

                def test_skill_execute_basic_action(self) -> None:
                    """Test {proposal.name} execution with basic action."""
                    action = AgentAction(type=ActionType.MESSAGE, text="Test action")
                    observation = self.skill.execute(action)

                    assert isinstance(observation, Observation)
                    assert observation.success is True
                    assert "Executed {proposal.name}" in observation.output

                def test_skill_analyze_data(self) -> None:
                    """Test {proposal.name} data analysis."""
                    test_data = {{"key": "value"}}
                    result = self.skill.analyze(test_data)

                    assert isinstance(result, dict)
                    assert "analysis_result" in result

                def test_skill_suggest_improvements(self) -> None:
                    """Test {proposal.name} improvement suggestions."""
                    suggestions = self.skill.suggest_improvements()

                    assert isinstance(suggestions, list)
                    assert len(suggestions) > 0
            '''
        ).strip("\n")
        return f"{test_implementation}\n"
