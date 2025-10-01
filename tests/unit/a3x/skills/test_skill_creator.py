"""Testes para a rotina de criação de skills a partir de propostas."""

from __future__ import annotations

import json
from __future__ import annotations

from pathlib import Path

from a3x.meta_capabilities import SkillProposal
from a3x.skills.skill_creator import SkillCreator


def _base_proposal(**overrides) -> SkillProposal:
    data = {
        "id": "skill_proposal_test",
        "name": "Test Skill",
        "description": "Skill para testes automatizados",
        "implementation_plan": "Executar plano de teste",
        "required_dependencies": ["core.testing"],
        "estimated_effort": 1.5,
        "priority": "high",
        "rationale": "Cobrir lacuna de testes",
        "target_domain": "testing",
        "created_at": "2024-10-10T00:00:00Z",
        "blueprint_path": None,
    }
    data.update(overrides)
    return SkillProposal(**data)


def test_skill_creator_uses_blueprint_when_available(tmp_path: Path) -> None:
    blueprint = tmp_path / "seed" / "skills" / "skill_proposal_test.py"
    blueprint.parent.mkdir(parents=True, exist_ok=True)
    blueprint_content = "class Generated:\n    pass\n"
    blueprint.write_text(blueprint_content, encoding="utf-8")

    creator = SkillCreator(tmp_path)
    proposal = _base_proposal(blueprint_path=str(blueprint.relative_to(tmp_path)))

    success, message = creator.create_skill_from_proposal(proposal)

    assert success is True
    assert "Skill 'Test Skill'" in message

    skill_path = tmp_path / "a3x" / "skills" / "test_skill.py"
    test_path = tmp_path / "tests" / "unit" / "a3x" / "skills" / "test_test_skill.py"
    registry_path = tmp_path / "seed" / "skills" / f"{proposal.id}.json"

    assert skill_path.exists()
    assert skill_path.read_text(encoding="utf-8") == blueprint_content
    assert test_path.exists()
    assert registry_path.exists()

    registry_data = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry_data["slug"] == "test_skill"
    assert registry_data["skill_path"] == "a3x/skills/test_skill.py"
    assert registry_data["status"] == "implemented"


def test_skill_creator_generates_template_without_blueprint(tmp_path: Path) -> None:
    creator = SkillCreator(tmp_path)
    proposal = _base_proposal(blueprint_path=None)

    success, message = creator.create_skill_from_proposal(proposal)

    assert success is True
    assert "Skill 'Test Skill'" in message

    skill_path = tmp_path / "a3x" / "skills" / "test_skill.py"
    test_path = tmp_path / "tests" / "unit" / "a3x" / "skills" / "test_test_skill.py"
    registry_path = tmp_path / "seed" / "skills" / f"{proposal.id}.json"

    assert skill_path.exists()
    assert "class TestSkill" in skill_path.read_text(encoding="utf-8")
    assert test_path.exists()
    assert registry_path.exists()
def _build_fake_proposal() -> SkillProposal:
    return SkillProposal(
        id="skill.fake",
        name="Fake Skill",
        description="Uma skill fake apenas para testes.",
        implementation_plan="Implementação mínima",
        required_dependencies=["pytest"],
        estimated_effort=1.0,
        priority="low",
        rationale="Cobrir casos de teste",
        target_domain="testing",
        created_at="2024-01-01T00:00:00Z",
    )


def test_skill_creator_generates_files(tmp_path: Path) -> None:
    creator = SkillCreator(tmp_path)
    proposal = _build_fake_proposal()

    success, message = creator.create_skill_from_proposal(proposal)

    assert success, message

    skill_slug = proposal.name.lower().replace(" ", "_").replace("-", "_")
    skill_path = tmp_path / "a3x" / "skills" / f"{skill_slug}.py"
    test_path = (
        tmp_path
        / "tests"
        / "unit"
        / "a3x"
        / "skills"
        / f"test_{skill_slug}.py"
    )

    assert skill_path.exists(), f"Skill file not created: {message}"
    assert test_path.exists(), "Test file not created"

    assert "Fake Skill" in skill_path.read_text(encoding="utf-8")
    assert "Tests for Fake Skill" in test_path.read_text(encoding="utf-8")
