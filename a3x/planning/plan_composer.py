"""PlanComposer module for goal decomposition and subtask execution."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List

import yaml

from a3x.llm import OpenRouterLLMClient


class PlanComposer:
    def __init__(self, timeout: int = 300, max_retries: int = 3) -> None:
        self.llm = OpenRouterLLMClient(model="x-ai/grok-4-fast:free")
        self.timeout = timeout
        self.max_retries = max_retries
        self.config_path = "configs/sample.yaml"  # Default config for sub-agent spawns

    def decompose_goal(self, goal: str) -> List[str]:
        """Decompose a goal into 3-5 atomic subtasks using LLM."""
        prompt = f"Decompose '{goal}' into 3-5 atomic subtasks. Output as a YAML list of strings."
        for attempt in range(self.max_retries):
            try:
                response = self.llm.chat(prompt)
                subtasks = yaml.safe_load(response)
                if isinstance(subtasks, list):
                    return [str(subtask).strip() for subtask in subtasks if subtask.strip()]
                else:
                    raise ValueError("LLM response is not a valid YAML list")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Fallback: simple decomposition
                    return [f"Step 1: Plan {goal}", f"Step 2: Implement {goal}", f"Step 3: Test {goal}"]
                time.sleep(2 ** attempt)  # Exponential backoff
        return []

    def _spawn_sub_agent(self, sub_goal: str, parent_agent: object = None) -> str:
        """Spawn a sub-agent either via direct method call or CLI subprocess."""
        # If parent agent is provided, use its run_sub_agent method directly (preferred)
        if parent_agent and hasattr(parent_agent, 'run_sub_agent'):
            try:
                from a3x.agent import AgentResult  # Import here to avoid circular import
                result: AgentResult = parent_agent.run_sub_agent(sub_goal)
                return f"Subtask '{sub_goal}' completed. Success: {result.completed}, Iterations: {result.iterations}, Failures: {result.failures}, Errors: {result.errors}"
            except Exception as e:
                return f"Error running sub-agent: {str(e)}"
        
        # Fallback to CLI subprocess
        output_file = tempfile.NamedTemporaryFile(mode='w+', suffix=f'_sub_agent_{uuid.uuid4()}.out', delete=False)
        output_path = Path(output_file.name)
        output_file.close()

        cmd = [
            'python', '-m', 'a3x.cli',
            '--goal', sub_goal,
            '--config', self.config_path
        ]

        for attempt in range(self.max_retries):
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.timeout
                )
                stdout, _ = proc.communicate(timeout=self.timeout)
                with open(output_path, 'w') as f:
                    f.write(stdout)
                if proc.returncode == 0:
                    with open(output_path, 'r') as f:
                        return f.read()
                else:
                    raise RuntimeError(f"Sub-agent failed with code {proc.returncode}")
            except subprocess.TimeoutExpired:
                proc.kill()
                if attempt == self.max_retries - 1:
                    raise
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
            finally:
                if output_path.exists():
                    output_path.unlink()
                time.sleep(2 ** attempt)
        return ""

    def execute_plan(self, subtasks: List[str], parent_agent: object = None) -> Dict[str, str]:
        """Execute subtasks sequentially, merge results."""
        results: Dict[str, str] = {}
        
        # Sequential execution only for now to avoid circular import issues
        for subtask in subtasks:
            try:
                output = self._spawn_sub_agent(subtask, parent_agent=parent_agent)
                results[subtask] = output
            except Exception as e:
                results[subtask] = f"Error: {str(e)}"

        # Merge results with LLM summarization
        if len(results) > 1:
            outputs_str = "\n\n".join([f"Subtask: {k}\nOutput: {v}" for k, v in results.items()])
            merge_prompt = f"Merge these subtask outputs into a cohesive summary: {outputs_str}"
            try:
                summary = self.llm.chat(merge_prompt)
                # Add summary as a special key
                results["merged_summary"] = summary
            except Exception:
                # Fallback: concatenate
                results["merged_summary"] = "\n\n".join(results.values())

        # Simple vote_on_best: pick longest output or LLM-ranked (here, longest)
        if len(results) > 1 and "merged_summary" not in results:
            best_subtask = max(results.keys(), key=lambda k: len(results[k]))
            results["best_output"] = results[best_subtask]

        return results