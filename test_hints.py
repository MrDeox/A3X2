from a3x.agent import AgentOrchestrator
from a3x.config import load_config
from a3x.llm import OpenRouterLLMClient

config = load_config("configs/sample.yaml")
llm = OpenRouterLLMClient(model="x-ai/grok-4-fast:free")
agent = AgentOrchestrator(config, llm)
print("Hints loaded:", agent.hints)
print("Recursion depth:", agent.recursion_depth)
print("Action biases:", agent.action_biases)
print("Backlog weights:", agent.backlog_weights)
print("Hints path:", agent.hints_path)
print("Log path:", agent.config.workspace_root / "a3x/logs/hints.log")