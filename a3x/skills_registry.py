"""Skills registry for dynamic skill loading and management."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Type, Any

from .actions import AgentAction, Observation
from .agent import AgentOrchestrator
from .config import AgentConfig


class BaseSkill:
    """Base class for all dynamic skills."""
    
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
    
    def execute(self, action: AgentAction) -> Observation:
        raise NotImplementedError


class SkillsRegistry:
    """Registry for managing dynamically loaded skills."""
    
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.skills_dir = workspace_root / "a3x" / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Dictionary mapping skill names to skill classes
        self._skills: Dict[str, Type[BaseSkill]] = {}
        
        # Dictionary mapping skill names to module paths
        self._skill_modules: Dict[str, str] = {}
        
        # Load all available skills at startup
        self.load_all_skills()
    
    def load_all_skills(self) -> None:
        """Load all skills from the skills directory."""
        print(f"Loading skills from: {self.skills_dir}")
        
        for skill_file in self.skills_dir.glob("*.py"):
            if skill_file.name.startswith("__"):
                continue
                
            # Extract skill name from filename
            skill_name = skill_file.stem
            
            # Import the module
            module_path = str(skill_file.resolve())
            spec = importlib.util.spec_from_file_location(skill_name, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[skill_name] = module
                try:
                    spec.loader.exec_module(module)
                    print(f"Successfully loaded module: {skill_name}")
                    
                    # Find the skill class in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, BaseSkill) and 
                            attr != BaseSkill):
                            
                            print(f"Found skill class: {attr_name} in {skill_name}")
                            self._skills[attr_name] = attr
                            self._skill_modules[attr_name] = module_path
                            break
                except Exception as e:
                    print(f"Error loading module {skill_name}: {e}")
    
    def register_skill(self, skill_name: str, skill_class: Type[BaseSkill]) -> None:
        """Register a skill programmatically."""
        self._skills[skill_name] = skill_class
    
    def get_skill(self, skill_name: str) -> Type[BaseSkill]:
        """Get a skill class by name."""
        if skill_name not in self._skills:
            raise KeyError(f"Skill '{skill_name}' not found in registry")
        return self._skills[skill_name]
    
    def get_skill_instance(self, skill_name: str, config: AgentConfig) -> BaseSkill:
        """Get an instance of a skill."""
        skill_class = self.get_skill(skill_name)
        return skill_class(config)
    
    def list_skills(self) -> List[str]:
        """List all available skills."""
        return list(self._skills.keys())
    
    def load_new_skill(self, skill_path: str) -> bool:
        """Load a skill from a specific path."""
        try:
            path = Path(skill_path)
            if not path.exists():
                return False
                
            skill_name = path.stem
            spec = importlib.util.spec_from_file_location(skill_name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[skill_name] = module
                spec.loader.exec_module(module)
                
                # Find and register the skill class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BaseSkill) and 
                        attr != BaseSkill):
                        
                        self._skills[attr_name] = attr
                        self._skill_modules[attr_name] = str(path)
                        return True
        except Exception as e:
            print(f"Error loading new skill from {skill_path}: {e}")
        
        return False


# Global skills registry instance
_skills_registry: SkillsRegistry | None = None


def get_skills_registry(workspace_root: Path) -> SkillsRegistry:
    """Get the global skills registry instance."""
    global _skills_registry
    if _skills_registry is None:
        _skills_registry = SkillsRegistry(workspace_root)
    return _skills_registry


def register_skill_with_agent(agent: AgentOrchestrator, skill_name: str) -> bool:
    """Register a new skill with the agent's system."""
    try:
        # Get the registry
        registry = get_skills_registry(agent.config.workspace_root)
        
        # Try to get the skill from the registry
        skill_class = registry.get_skill(skill_name)
        
        # Create a hint to increase usage of this skill
        lesson = {
            "action_biases": {skill_name: 1.5}  # Increase bias for this skill
        }
        agent.update_hints(lesson)
        
        return True
    except KeyError:
        print(f"Skill {skill_name} not found in registry")
        return False