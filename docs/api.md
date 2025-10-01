# A3X API Reference

This document provides a reference for the core modules of the A3X autonomous agent system. It focuses on key classes and methods for orchestration, execution, planning, and evaluation. The API is designed for extensibility, with type hints and dataclasses for clarity.

The system is built in Python 3.10+ and uses YAML configs for customization. For full source, see `a3x/` directory. Examples assume an imported agent instance (e.g., `agent = AgentOrchestrator(config, llm_client)`).

## Core Concepts

- **AgentOrchestrator**: Main entry point for running agent loops, handling goals, subtasks, and metrics.
- **ActionExecutor**: Executes actions like file operations, patches, and commands with safety checks.
- **HierarchicalPlanner**: Manages goal decomposition into missions/tasks, with persistence and evaluation.
- **AutoEvaluator**: Tracks metrics, generates seeds for improvement, and computes fitness.

See [AGENTS.md](AGENTS.md) for architecture overview and [configs/README.md](configs/README.md) for configuration.

## a3x.agent - Agent Orchestration

### AgentOrchestrator

The central class for agent execution. Initializes with config and LLM client, manages recursion, memory, and planning.

#### `__init__(self, config: AgentConfig, llm_client: BaseLLMClient, auto_evaluator: AutoEvaluator | None = None, depth: int = 0) -> None`

Initializes the orchestrator.

**Args:**
- `config` (AgentConfig): Configuration for workspace, limits, policies, etc.
- `llm_client` (BaseLLMClient): LLM interface (e.g., OpenRouterLLMClient).
- `auto_evaluator` (AutoEvaluator, optional): Evaluator for metrics/seeds.
- `depth` (int, default 0): Recursion depth for sub-agents.

**Returns:**
- None

**Example:**
```python
from a3x.config import AgentConfig
from a3x.llm import OpenRouterLLMClient

config = AgentConfig.from_yaml("configs/sample.yaml")
llm = OpenRouterLLMClient(model="x-ai/grok-4-fast:free")
agent = AgentOrchestrator(config, llm)
```

#### `run(self, goal: str) -> AgentResult`

Executes the main loop to achieve the goal, handling decomposition, actions, and evaluation.

**Args:**
- `goal` (str): Objective (e.g., "Implement a new feature").

**Returns:**
- AgentResult: Summary with `completed`, `iterations`, `failures`, `history`, `errors`, `memories_reused`.

**Example:**
```python
result = agent.run("Add logging to executor")
if result.completed:
    print("Success!")
else:
    print(f"Failed after {result.iterations} iterations")
```

#### `_handle_subtasks(self, subtasks: List[str], history: AgentHistory, goal: str) -> AgentResult`

Handles subtask execution via PlanComposer for complex goals.

**Args:**
- `subtasks` (List[str]): Decomposed goals.
- `history` (AgentHistory): Execution log.
- `goal` (str): Parent goal.

**Returns:**
- AgentResult: Aggregated subtask result.

**Example:**
```python
subtasks = ["Analyze", "Implement", "Test"]
result = agent._handle_subtasks(subtasks, history, "Build app")
```

#### `_adjust_recursion_depth(self) -> None`

Dynamically adjusts recursion depth based on success rates (3-10).

**Returns:**
- None

#### `_decompose_goal(self, goal: str) -> List[str]`

Decomposes goal into subtasks and saves to `seed/subgoals.json`.

**Args:**
- `goal` (str): Goal to decompose.

**Returns:**
- List[str]: Subtasks.

**Example:**
```python
subtasks = agent._decompose_goal("Create web app")
# ["Design UI", "Backend", "Tests"]
```

#### `_gather_memory_lessons(self, goal: str) -> str`

Retrieves and formats relevant memories for the goal.

**Args:**
- `goal` (str): Query goal.

**Returns:**
- str: Formatted lessons or empty.

#### `_record_auto_evaluation(self, goal: str, result: AgentResult, started_at: float) -> Dict[str, float]`

Records metrics, updates hints, generates seeds, computes fitness.

**Args:**
- `goal` (str): Evaluated goal.
- `result` (AgentResult): Run result.
- `started_at` (float): Start timestamp.

**Returns:**
- Dict[str, float]: Metrics snapshot.

#### `_calculate_fitness(self, metrics: Dict[str, float]) -> float`

Computes weighted fitness score (0-1).

**Args:**
- `metrics` (Dict[str, float]): Metrics.

**Returns:**
- float: Fitness score.

**Example:**
```python
fitness = agent._calculate_fitness({"actions_success_rate": 0.9})
# 0.85
```

### AgentResult (dataclass)

**Fields:**
- `completed` (bool): Goal achieved.
- `iterations` (int): Loop count.
- `failures` (int): Failure count.
- `history` (AgentHistory): Events log.
- `errors` (List[str]): Errors.
- `memories_reused` (int, default 0): Memory hits.

## a3x.executor - Action Execution

### ActionExecutor

Executes actions with sandboxing, risk checks, and logging.

#### `__init__(self, config: AgentConfig) -> None`

Sets up executor with patch manager and logger.

**Args:**
- `config` (AgentConfig): Agent config.

**Returns:**
- None

#### `execute(self, action: AgentAction) -> Observation`

Dispatches to handlers (e.g., read_file, apply_patch).

**Args:**
- `action` (AgentAction): Action to execute.

**Returns:**
- Observation: Result with success, output, error, etc.

**Example:**
```python
action = AgentAction(type=ActionType.READ_FILE, path="config.yaml")
obs = executor.execute(action)
print(obs.output)
```

#### `_handle_self_modify(self, action: AgentAction) -> Observation`

Applies self-patches with analysis, tests, and conditional commit.

**Args:**
- `action` (AgentAction): Self-modify with diff.

**Returns:**
- Observation: Result, including commit status.

**Example:**
```python
action = AgentAction(type=ActionType.SELF_MODIFY, diff=patch)
obs = executor._handle_self_modify(action)
```

#### `_run_risk_checks(self, patch_content: str) -> Dict[str, str]`

Lints patch in temp dir with ruff/black.

**Args:**
- `patch_content` (str): Diff.

**Returns:**
- Dict[str, str]: Risks ('high'/'medium').

#### `_analyze_impact_before_apply(self, action: AgentAction) -> tuple[bool, str]`

Static analysis for self-modify safety.

**Args:**
- `action` (AgentAction): Action with diff.

**Returns:**
- tuple[bool, str]: (safe, message).

### Observation (dataclass)

**Fields:**
- `success` (bool): Execution success.
- `output` (str): Result/output.
- `error` (str, optional): Error message.
- `return_code` (int, optional): Command code.
- `duration` (float): Execution time.
- `type` (str): Action type.

## a3x.planning.hierarchical_planner - Planning

### HierarchicalPlanner

Manages goal -> mission -> task hierarchy with persistence.

#### `__init__(self, storage_dir: Path | str = Path("seed/memory/plans"), thresholds: Optional[PlannerThresholds] = None) -> None`

Initializes planner with storage and thresholds.

**Args:**
- `storage_dir` (Path | str): Plans directory.
- `thresholds` (PlannerThresholds, optional): Metric thresholds.

**Returns:**
- None

#### `ensure_plan(self, state: AgentState, missions: Optional[MissionState], objectives: Sequence[str], metrics_history: Dict[str, List[float]]) -> GoalPlan`

Builds or loads plan, replans if degraded.

**Args:**
- `state` (AgentState): Current state.
- `missions` (MissionState, optional): Backlog missions.
- `objectives` (Sequence[str]): Goals.
- `metrics_history` (Dict[str, List[float]]): Metrics.

**Returns:**
- GoalPlan: Hierarchical plan.

**Example:**
```python
plan = planner.ensure_plan(state, missions, ["Improve code"], history)
```

#### `force_replan(self, state: AgentState, missions: Optional[MissionState], objectives: Sequence[str], metrics_history: Dict[str, List[float]]) -> GoalPlan`

Forces new plan rebuild.

**Args:**
- Same as `ensure_plan`.

**Returns:**
- GoalPlan: New plan.

#### `record_action_result(self, action: AgentAction, observation: Observation, timestamp: float) -> PlanEvaluation`

Updates plan with action outcome, checks for replan.

**Args:**
- `action` (AgentAction): Executed action.
- `observation` (Observation): Result.
- `timestamp` (float): Time.

**Returns:**
- PlanEvaluation: With `needs_replan`, `alerts`.

### GoalPlan (dataclass)

**Fields:**
- `goal` (str): Root goal.
- `missions` (List[MissionPlan]): Missions.
- `plan_id` (str): ID.
- `metrics_snapshot` (Dict[str, float]): Metrics.
- `current_mission` (str, optional): Active mission ID.
- `current_task` (str, optional): Active task ID.
- `events` (List[PlanEvent]): Log.

### MissionPlan (dataclass)

**Fields:**
- `id` (str): ID.
- `description` (str): Description.
- `priority` (str): "high"/"medium"/"low".
- `tasks` (List[TaskPlan]): Tasks.

### TaskPlan (dataclass)

**Fields:**
- `id` (str): ID.
- `description` (str): Description.
- `expected_outcome` (str): Expected result.
- `metrics_target` (Dict[str, float]): Targets.
- `status` (str): "pending"/"in_progress"/"completed"/"blocked".
- `dependencies` (List[str]): Task IDs.
- `evidence` (List[PlanEvidence]): Proofs.
- `last_observation` (str, optional): Last output.
- `blocked_reason` (str, optional): Block reason.

### PlanEvaluation (dataclass)

**Fields:**
- `needs_replan` (bool): Replan needed.
- `alerts` (List[str]): Warnings.

## Additional Modules

- **a3x.config (AgentConfig)**: Loads YAML configs. Use `from_yaml(path)` to instantiate.
- **a3x.llm (BaseLLMClient)**: Abstract LLM interface; implement for providers.
- **a3x.autoeval (AutoEvaluator)**: Metrics tracking and seed generation.

For implementation details, refer to source code. Extend via inheritance (e.g., custom planners). Validate with `pytest tests/unit/a3x/`.

Generated: 2025-10-01. Update as API evolves.