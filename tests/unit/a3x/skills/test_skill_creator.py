from __future__ import annotations

import sys
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _ensure_package(package: str, package_dir: Path) -> None:
    if package in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        package,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )

    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError(f"Unable to load package '{package}' from {package_dir}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)


_ensure_package("a3x", ROOT / "a3x")

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

    generated_skill = skill_path.read_text(encoding="utf-8")
    generated_test = test_path.read_text(encoding="utf-8")

    assert "Fake Skill" in generated_skill
    assert "class FakeSkill" in generated_skill
    assert "Tests for Fake Skill" in generated_test
    assert skill_path.name in message


def test_skill_creator_avoids_overwriting_existing_slug(tmp_path: Path) -> None:
    creator = SkillCreator(tmp_path)

    original_proposal = _build_fake_proposal()
    duplicate_proposal = replace(
        original_proposal,
        id="skill.fake.duplicate",
        name="Fake Skill!!!",
        description="Skill duplicada para validar sufixo único.",
    )

    first_success, first_message = creator.create_skill_from_proposal(original_proposal)
    assert first_success, first_message

    second_success, second_message = creator.create_skill_from_proposal(duplicate_proposal)
    assert second_success, second_message

    base_slug = SkillCreator._slugify(original_proposal.name)
    original_skill_path = tmp_path / "a3x" / "skills" / f"{base_slug}.py"
    duplicated_skill_path = tmp_path / "a3x" / "skills" / f"{base_slug}_2.py"

    assert original_skill_path.exists()
    assert duplicated_skill_path.exists()

    assert "Uma skill fake apenas para testes." in original_skill_path.read_text(encoding="utf-8")
    assert "Skill duplicada para validar sufixo único." in duplicated_skill_path.read_text(
        encoding="utf-8"
    )

    duplicated_test_path = (
        tmp_path
        / "tests"
        / "unit"
        / "a3x"
        / "skills"
        / f"test_{base_slug}_2.py"
    )

    assert duplicated_test_path.exists()
    assert "Tests for Fake Skill!!!" in duplicated_test_path.read_text(encoding="utf-8")


def test_skill_creator_handles_preexisting_skill_file(tmp_path: Path) -> None:
    creator = SkillCreator(tmp_path)
    proposal = _build_fake_proposal()

    base_slug = SkillCreator._slugify(proposal.name)
    existing_skill = tmp_path / "a3x" / "skills" / f"{base_slug}.py"
    existing_skill.parent.mkdir(parents=True, exist_ok=True)
    existing_skill.write_text("# existing skill implementation", encoding="utf-8")

    success, message = creator.create_skill_from_proposal(proposal)

    assert success, message

    new_skill_path = tmp_path / "a3x" / "skills" / f"{base_slug}_2.py"
    new_test_path = (
        tmp_path
        / "tests"
        / "unit"
        / "a3x"
        / "skills"
        / f"test_{base_slug}_2.py"
    )

    assert not message.endswith(f"{existing_skill}")
    assert new_skill_path.exists()
    assert new_test_path.exists()


def test_skill_creator_handles_preexisting_test_file(tmp_path: Path) -> None:
    creator = SkillCreator(tmp_path)
    proposal = _build_fake_proposal()

    base_slug = SkillCreator._slugify(proposal.name)
    existing_test = (
        tmp_path
        / "tests"
        / "unit"
        / "a3x"
        / "skills"
        / f"test_{base_slug}.py"
    )
    existing_test.parent.mkdir(parents=True, exist_ok=True)
    existing_test.write_text("# existing test placeholder", encoding="utf-8")

    success, message = creator.create_skill_from_proposal(proposal)

    assert success, message

    new_skill_path = tmp_path / "a3x" / "skills" / f"{base_slug}_2.py"
    new_test_path = (
        tmp_path
        / "tests"
        / "unit"
        / "a3x"
        / "skills"
        / f"test_{base_slug}_2.py"
    )

    assert new_skill_path.exists()
    assert new_test_path.exists()


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    (
        ("Fake Skill", "fake_skill"),
        ("Skill com Acentuação", "skill_com_acentua_o"),
        ("   --Complex*Skill--   ", "complex_skill"),
        ("", "nova_skill"),
    ),
)
def test_slugify_handles_various_inputs(raw_name: str, expected: str) -> None:
    assert SkillCreator._slugify(raw_name) == expected
