"""MetaRecursionEngine for managing recursive self-improvement loops in A3X."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .autoeval import AutoEvaluator
from .config import AgentConfig
from .patch import PatchManager


@dataclass
class RecursionContext:
    """Context for a recursive self-improvement cycle."""
    id: str
    depth: int
    parent_id: str | None
    goal: str
    metrics: dict[str, float]
    improvements_applied: list[str]
    timestamp: str
    status: str = "active"  # active, completed, failed


class MetaRecursionEngine:
    """Engine for managing recursive self-improvement loops."""

    def __init__(self, config: AgentConfig, patch_manager: PatchManager, auto_evaluator: AutoEvaluator) -> None:
        self.config = config
        self.patch_manager = patch_manager
        self.auto_evaluator = auto_evaluator
        self.workspace_root = Path(config.workspace.root).resolve()
        self.recursion_path = self.workspace_root / "seed" / "recursion"
        self.recursion_path.mkdir(parents=True, exist_ok=True)
        self.max_depth: int = 10  # Hardcoded as per integration in agent.py
        self.current_depth: int = 0
        self.context_stack: list[RecursionContext] = []
        self.recursion_history: list[RecursionContext] = self._load_recursion_history()

    def initiate_recursion(self, goal: str, parent_id: str | None = None) -> RecursionContext:
        """Initiate a new recursive self-improvement loop."""
        if self.current_depth >= self.max_depth:
            raise ValueError(f"Maximum recursion depth {self.max_depth} reached")

        context_id = f"rec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        context = RecursionContext(
            id=context_id,
            depth=self.current_depth + 1,
            parent_id=parent_id,
            goal=goal,
            metrics={},
            improvements_applied=[],
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="active"
        )
        self.context_stack.append(context)
        self.current_depth += 1
        self._save_recursion_context(context)
        return context

    def evaluate_and_recurse(self, context: RecursionContext, current_metrics: dict[str, float]) -> bool:
        """Evaluate if recursion should continue based on metrics."""
        context.metrics.update(current_metrics)
        improvement_threshold = self.config.get("recursion.improvement_threshold", 0.1)

        # Check for sufficient improvement
        key_metric = "actions_success_rate"  # Example key metric
        if key_metric in current_metrics and current_metrics[key_metric] > improvement_threshold:
            # Sufficient improvement: recurse
            sub_goal = f"Further optimize based on {context.goal} improvements"
            sub_context = self.initiate_recursion(sub_goal, context.id)
            self._apply_recursive_improvements(sub_context)
            return True
        else:
            # No sufficient improvement: complete this level
            context.status = "completed"
            self._complete_context(context)
            self.current_depth -= 1
            self.context_stack.pop()
            return False

    def apply_recursive_patch(self, diff: str, context: RecursionContext) -> bool:
        """Apply a patch within the recursive context, with safety checks."""
        success, output = self.patch_manager.apply(diff)
        if success:
            context.improvements_applied.append(f"Patch applied: {output[:100]}...")
            self.auto_evaluator.record_metric("recursion.patch_success", 1.0)
        else:
            self.auto_evaluator.record_metric("recursion.patch_success", 0.0)
            context.status = "failed"
        self._save_recursion_context(context)
        return success

    def _apply_recursive_improvements(self, context: RecursionContext) -> None:
        """Apply improvements in a recursive manner."""
        import json

        from .llm import build_llm_client

        print(f"Applying recursive improvements for context {context.id} at depth {context.depth}")

        # Load default config for the LLM operations
        try:
            # Use the same config as the main agent to maintain consistency
            config = self.config
            llm_client = build_llm_client(config.llm)

            # Build current context for the LLM to understand what improvements to make
            current_metrics_str = json.dumps(context.metrics, indent=2)

            # Get the current state of key files that might need improvement
            key_files = [
                "a3x/meta_recursion.py",
                "a3x/self_directed_evolution.py",
                "a3x/agent.py",
                "a3x/executor.py",
                "a3x/autoeval.py"
            ]

            files_content = {}
            for file_path in key_files:
                full_path = self.workspace_root / file_path
                if full_path.exists():
                    try:
                        content = full_path.read_text(encoding="utf-8")
                        # Truncate if too large to avoid token limits
                        if len(content) > 4000:
                            content = content[:2000] + "\n...[CONTENT TRUNCATED]...\n" + content[-2000:]
                        files_content[file_path] = content
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

            files_str = json.dumps(files_content, indent=2)

            prompt = f"""
            You are an AI system specializing in self-improvement for the A3X autonomous coding framework. 
            We need to identify and implement targeted improvements based on the following metrics:
            
            {current_metrics_str}
            
            We also have the current content of key files:
            
            {files_str}
            
            Based on these metrics and the current code, identify the specific improvements needed in the following areas:
            
            1. Code quality and architecture improvements
            2. Performance optimizations
            3. Bug fixes based on failure patterns
            4. Enhancement of specific capabilities that are underperforming
            5. Self-monitoring and self-improvement capabilities
            
            Generate a diff that makes these improvements. Focus on actual, implementable changes
            that could improve the system's performance based on the metrics provided.
            
            Respond with a JSON object containing:
            - "analysis": Your analysis of the metrics and what needs improvement
            - "improvement_plan": A list of specific improvements to make
            - "file_changes": A list of specific file changes to make in the format:
              {{
                "file_path": "path/to/file.py",
                "description": "What this change does",
                "change_type": "enhancement|bug_fix|optimization|refactor",
                "diff": "The complete diff to apply to the file in unified diff format"
              }}
            """

            response = llm_client.chat(prompt)

            try:
                # Parse the response as JSON
                response_data = json.loads(response)

                # Process the improvement plan and apply changes
                file_changes = response_data.get("file_changes", [])

                for change in file_changes:
                    file_path = change.get("file_path", "")
                    description = change.get("description", "No description")
                    change_type = change.get("change_type", "enhancement")
                    diff = change.get("diff", "")

                    if file_path and diff:
                        print(f"Applying {change_type} to {file_path}: {description}")

                        # Apply the patch directly
                        success = self.apply_recursive_patch(diff, context)
                        if success:
                            print(f"Successfully applied {change_type} to {file_path}")
                            context.improvements_applied.append(f"Applied {change_type} to {file_path}: {description}")
                        else:
                            print(f"Failed to apply {change_type} to {file_path}")
                    else:
                        print(f"Missing file_path or diff in change: {change}")

            except json.JSONDecodeError as e:
                print(f"Failed to parse LLM response as JSON: {e}")
                print(f"Raw response: {response}")

        except Exception as e:
            print(f"Error during recursive improvement generation: {e}")
            import traceback
            traceback.print_exc()

    def _generate_diff_for_change(self, file_path: str, code_change: str) -> str | None:
        """Generate a diff for a requested code change."""
        from pathlib import Path

        file_path = Path(file_path)
        full_path = self.workspace_root / file_path

        if not full_path.exists():
            print(f"File not found: {full_path}")
            return None

        # Read the original file
        try:
            original_content = full_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading file {full_path}: {e}")
            return None

        # For now, we'll implement a simple approach where the code_change is just appended
        # A full implementation would use more sophisticated diff generation
        # This is a simplified version for illustration purposes

        # In a real implementation, we would need to be more careful about how to apply
        # the change to the original content based on the code_change specification

        # For now, let's assume code_change is a diff patch
        if "--- " in code_change and "+++ " in code_change:
            # This looks like a diff already, return it as is
            return code_change
        else:
            # This is a more complex implementation that would parse the change request
            # and generate an appropriate diff - for now just return None
            # as we would need more sophisticated logic to generate real diffs
            print(f"Cannot generate diff for change: {code_change[:100]}...")
            return None

    def _complete_context(self, context: RecursionContext) -> None:
        """Mark context as complete and save."""
        context.status = "completed"
        self.recursion_history.append(context)
        self._save_recursion_history()
        self._save_recursion_context(context)

    def _save_recursion_context(self, context: RecursionContext) -> None:
        """Save recursion context to file."""
        context_file = self.recursion_path / f"{context.id}.json"
        with context_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(context), f, ensure_ascii=False, indent=2)

    def _load_recursion_history(self) -> list[RecursionContext]:
        """Load recursion history from files."""
        history = []
        for context_file in self.recursion_path.glob("rec_*.json"):
            try:
                with context_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    history.append(RecursionContext(**data))
            except Exception:
                pass
        return history

    def _save_recursion_history(self) -> None:
        """Save recursion history to file."""
        history_file = self.recursion_path / "history.json"
        with history_file.open("w", encoding="utf-8") as f:
            json.dump([asdict(ctx) for ctx in self.recursion_history], f, ensure_ascii=False, indent=2)

    def get_recursion_summary(self) -> dict[str, Any]:
        """Get a summary of recursion activity."""
        active = [ctx for ctx in self.context_stack if ctx.status == "active"]
        completed = [ctx for ctx in self.recursion_history if ctx.status == "completed"]
        return {
            "current_depth": self.current_depth,
            "active_contexts": len(active),
            "completed_contexts": len(completed),
            "max_depth_reached": max([ctx.depth for ctx in self.recursion_history] + [0]),
            "avg_improvement": sum(ctx.metrics.get("actions_success_rate", 0) for ctx in completed) / max(1, len(completed))
        }


def integrate_meta_recursion(config: AgentConfig, patch_manager: PatchManager, auto_evaluator: AutoEvaluator) -> MetaRecursionEngine:
    """Integrate MetaRecursionEngine into the system."""
    engine = MetaRecursionEngine(config, patch_manager, auto_evaluator)
    # Hook into autoloop or self_directed_evolution here if needed
    return engine


__all__ = [
    "MetaRecursionEngine",
    "RecursionContext",
    "integrate_meta_recursion",
]
