from a3x.agent import AgentOrchestrator
from a3x.config import load_config
from a3x.llm import OpenRouterLLMClient

config = load_config("configs/sample.yaml")
llm = OpenRouterLLMClient(model="x-ai/grok-4-fast:free")
agent = AgentOrchestrator(config, llm)
# Debug information (commented out to avoid print statements in test code)
# agent.hints: {agent.hints}
# agent.recursion_depth: {agent.recursion_depth}
# agent.action_biases: {agent.action_biases}
# agent.backlog_weights: {agent.backlog_weights}
# agent.hints_path: {agent.hints_path}
# log_path: {agent.config.workspace_root / "a3x/logs/hints.log"}
