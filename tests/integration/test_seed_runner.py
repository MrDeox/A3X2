"""Integration facade tests for the SeedRunner orchestration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

from a3x.seed_runner import SeedRunner, SeedRunResult
from a3x.seeds import Seed


def _build_seed(**overrides) -> Seed:
    defaults = {
        "id": "seed-1",
        "goal": "Implementar endpoint /health",
        "priority": "high",
        "status": "pending",
        "type": "generic",
    }
    defaults.update(overrides)
    return Seed(**defaults)


def test_seed_runner_success_flow(tmp_path: Path) -> None:
    backlog_mock = Mock()
    backlog_mock.next_seed.return_value = _build_seed()

    runner = SeedRunner.__new__(SeedRunner)
    runner.backlog = backlog_mock

    orchestrator_result = SimpleNamespace(completed=True, errors=[])

    with patch("a3x.seed_runner.load_config") as mock_load_config, patch(
        "a3x.seed_runner.build_llm_client"
    ) as mock_build_llm, patch("a3x.seed_runner.AgentOrchestrator") as mock_orchestrator:
        mock_load_config.return_value = SimpleNamespace(
            limits=SimpleNamespace(max_iterations=5),
            llm=Mock(),
        )
        mock_build_llm.return_value = Mock()
        mock_instance = mock_orchestrator.return_value
        mock_instance.run.return_value = orchestrator_result

        result = SeedRunner.run_next(
            runner,
            default_config=tmp_path / "config.yaml",
        )

    assert isinstance(result, SeedRunResult)
    assert result.completed is True
    backlog_mock.mark_in_progress.assert_called_once()
    backlog_mock.mark_completed.assert_called_once()


def test_seed_runner_failure_marks_seed(tmp_path: Path) -> None:
    failing_seed = _build_seed(id="seed-2")
    backlog_mock = Mock()
    backlog_mock.next_seed.return_value = failing_seed

    runner = SeedRunner.__new__(SeedRunner)
    runner.backlog = backlog_mock

    orchestrator_result = SimpleNamespace(completed=False, errors=["timeout"])

    with patch("a3x.seed_runner.load_config") as mock_load_config, patch(
        "a3x.seed_runner.build_llm_client"
    ) as mock_build_llm, patch("a3x.seed_runner.AgentOrchestrator") as mock_orchestrator:
        mock_load_config.return_value = SimpleNamespace(
            limits=SimpleNamespace(max_iterations=5),
            llm=Mock(),
        )
        mock_build_llm.return_value = Mock()
        mock_orchestrator.return_value.run.return_value = orchestrator_result

        result = SeedRunner.run_next(
            runner,
            default_config=tmp_path / "config.yaml",
        )

    assert isinstance(result, SeedRunResult)
    assert result.completed is False
    backlog_mock.mark_failed.assert_called_once()
    backlog_mock.mark_completed.assert_not_called()


def test_seed_runner_skill_creation_success(tmp_path: Path) -> None:
    proposal_dir = tmp_path / "seed" / "skills"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = proposal_dir / "proposal.json"
    proposal_path.write_text(
        json.dumps(
            {
                "id": "skill_proposal_001",
                "name": "Test Skill",
                "description": "Habilidade gerada via seed",
                "implementation_plan": "Detalhes da implementação",
                "required_dependencies": ["core.diffing"],
                "estimated_effort": 3.0,
                "priority": "high",
                "rationale": "Cobrir lacuna",
                "target_domain": "testing",
                "created_at": "2024-10-10T00:00:00Z",
                "blueprint_path": "seed/skills/skill_proposal_001.py",
            }
        ),
        encoding="utf-8",
    )

    seed = _build_seed(
        id="skill-seed",
        type="skill_creation",
        metadata={
            "proposal_record": str(proposal_path.relative_to(tmp_path)),
        },
    )

    backlog_mock = Mock()
    backlog_mock.next_seed.return_value = seed

    runner = SeedRunner.__new__(SeedRunner)
    runner.backlog = backlog_mock

    orchestrator_result = SimpleNamespace(completed=True, errors=[])

    with patch("a3x.seed_runner.load_config") as mock_load_config, patch(
        "a3x.seed_runner.build_llm_client"
    ) as mock_build_llm, patch("a3x.seed_runner.AgentOrchestrator") as mock_orchestrator, patch.object(
        Path, "cwd", return_value=tmp_path
    ), patch("a3x.seed_runner.SkillCreator") as mock_skill_creator:
        mock_load_config.return_value = SimpleNamespace(
            limits=SimpleNamespace(max_iterations=5),
            llm=Mock(),
        )
        mock_build_llm.return_value = Mock()
        mock_instance = mock_orchestrator.return_value
        mock_instance.run.return_value = orchestrator_result
        creator_instance = mock_skill_creator.return_value
        creator_instance.create_skill_from_proposal.return_value = (True, "ok")

        result = SeedRunner.run_next(
            runner,
            default_config=tmp_path / "config.yaml",
        )

    creator_instance.create_skill_from_proposal.assert_called_once()
    proposal_arg = creator_instance.create_skill_from_proposal.call_args[0][0]
    assert proposal_arg.id == "skill_proposal_001"
    assert isinstance(result, SeedRunResult)
    assert result.completed is True
    backlog_mock.mark_completed.assert_called_once()


def test_seed_runner_skill_creation_failure_marks_backlog(tmp_path: Path) -> None:
    proposal_dir = tmp_path / "seed" / "skills"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = proposal_dir / "proposal.json"
    proposal_path.write_text(
        json.dumps(
            {
                "id": "skill_proposal_002",
                "name": "Skill Failure",
                "description": "Proposta que falha",
                "implementation_plan": "Plano",
                "required_dependencies": ["core.testing"],
                "estimated_effort": 2.0,
                "priority": "medium",
                "rationale": "Teste",
                "target_domain": "analysis",
                "created_at": "2024-10-11T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    seed = _build_seed(
        id="skill-seed-fail",
        type="skill_creation",
        metadata={
            "proposal_record": str(proposal_path.relative_to(tmp_path)),
        },
    )

    backlog_mock = Mock()
    backlog_mock.next_seed.return_value = seed

    runner = SeedRunner.__new__(SeedRunner)
    runner.backlog = backlog_mock

    orchestrator_result = SimpleNamespace(completed=True, errors=[])

    with patch("a3x.seed_runner.load_config") as mock_load_config, patch(
        "a3x.seed_runner.build_llm_client"
    ) as mock_build_llm, patch("a3x.seed_runner.AgentOrchestrator") as mock_orchestrator, patch.object(
        Path, "cwd", return_value=tmp_path
    ), patch("a3x.seed_runner.SkillCreator") as mock_skill_creator:
        mock_load_config.return_value = SimpleNamespace(
            limits=SimpleNamespace(max_iterations=5),
            llm=Mock(),
        )
        mock_build_llm.return_value = Mock()
        mock_instance = mock_orchestrator.return_value
        mock_instance.run.return_value = orchestrator_result
        creator_instance = mock_skill_creator.return_value
        creator_instance.create_skill_from_proposal.return_value = (False, "erro")

        result = SeedRunner.run_next(
            runner,
            default_config=tmp_path / "config.yaml",
        )

    assert isinstance(result, SeedRunResult)
    assert result.completed is False
    backlog_mock.mark_failed.assert_called_once()
    backlog_mock.mark_completed.assert_not_called()
