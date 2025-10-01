from __future__ import annotations

from pathlib import Path

from a3x.meta_capabilities import SkillProposal
from a3x.skills.skill_creator import SkillCreator


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

    skill_slug = SkillCreator._slugify(proposal.name)
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
