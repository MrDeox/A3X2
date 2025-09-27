from a3x.actions import ActionType, AgentAction, Observation
from a3x.agent import AgentOrchestrator, AgentResult
from a3x.history import AgentHistory


def _make_apply_patch_action(path: str) -> AgentAction:
    diff = f"--- a/{path}\n+++ b/{path}\n"
    return AgentAction(type=ActionType.APPLY_PATCH, diff=diff)


def test_analyze_history_infers_capabilities() -> None:
    history = AgentHistory()
    history.append(_make_apply_patch_action("foo.py"), Observation(success=True))
    history.append(
        AgentAction(type=ActionType.RUN_COMMAND, command=["pytest"]),
        Observation(success=True, return_code=0),
    )
    history.append(
        AgentAction(type=ActionType.WRITE_FILE, path="README.md", content="..."),
        Observation(success=True),
    )

    result = AgentResult(
        completed=True,
        iterations=3,
        failures=0,
        history=history,
        errors=[],
    )

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    metrics, capabilities = orchestrator._analyze_history(result)  # type: ignore[arg-type]

    assert metrics["actions_total"] == 3
    assert metrics["unique_commands"] == 1
    assert "core.diffing" in capabilities
    assert "core.testing" in capabilities
    assert "horiz.python" in capabilities
    assert "horiz.docs" in capabilities
