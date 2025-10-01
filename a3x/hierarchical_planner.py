"""Enhanced hierarchical planner for long-horizon objectives."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .planning.mission_state import (
    Mission,
    MissionMilestone,
    MissionState,
    MilestoneStatus,
    MissionStatus
)
from .planning.storage import load_mission_state, save_mission_state
from .autoeval import AutoEvaluator


class HierarchicalPlanner:
    """Manages long-horizon planning with persistent objectives and auto-generation of subgoals."""
    
    def __init__(self, workspace_root: Path, auto_evaluator: AutoEvaluator) -> None:
        self.workspace_root = workspace_root
        self.auto_evaluator = auto_evaluator
        self.missions_path = workspace_root / "seed" / "missions.yaml"
        self.objectives_path = workspace_root / "seed" / "objectives.json"
        
        # Load existing missions
        self.mission_state = load_mission_state(self.missions_path)
        
        # Load/initialize objectives
        self.objectives = self._load_objectives()
    
    def _load_objectives(self) -> Dict[str, dict]:
        """Load persistent objectives from file."""
        objectives_path = self.workspace_root / "seed" / "objectives.json"
        if objectives_path.exists():
            try:
                with open(objectives_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_objectives(self) -> None:
        """Save persistent objectives to file."""
        objectives_path = self.workspace_root / "seed" / "objectives.json"
        objectives_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(objectives_path, 'w', encoding='utf-8') as f:
                json.dump(self.objectives, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving objectives: {e}")
    
    def update_objective_progress(self, objective_id: str, progress: float) -> None:
        """Update progress for a specific objective."""
        if objective_id in self.objectives:
            self.objectives[objective_id]['progress'] = progress
            self.objectives[objective_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
            self._save_objectives()
    
    def add_objective(self, objective_id: str, description: str, target_metric: str, target_value: float) -> None:
        """Add a new persistent objective."""
        self.objectives[objective_id] = {
            'id': objective_id,
            'description': description,
            'target_metric': target_metric,
            'target_value': target_value,
            'current_value': 0.0,
            'progress': 0.0,
            'status': 'active',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        self._save_objectives()
    
    def generate_subgoals(self) -> List[Dict[str, str]]:
        """Generate new subgoals based on persistent objectives."""
        subgoals = []
        
        for obj_id, objective in self.objectives.items():
            if objective.get('status') == 'active':
                # Check if we need new subgoals for this objective
                current_progress = objective.get('progress', 0.0)
                if current_progress < 1.0:  # Not completed yet
                    # Generate a subgoal based on the objective
                    subgoal = {
                        'id': f"subgoal_{obj_id}_{len(subgoals) + 1}",
                        'description': f"Progress toward: {objective['description']}",
                        'objective_id': obj_id,
                        'target_completion': min(1.0, current_progress + 0.2),  # Increment by 20%
                    }
                    subgoals.append(subgoal)
        
        return subgoals
    
    def create_mission_from_objective(self, objective_id: str) -> Optional[Mission]:
        """Create a mission from a persistent objective."""
        if objective_id not in self.objectives:
            return None
            
        objective = self.objectives[objective_id]
        
        # Create a new mission based on the objective
        mission_id = f"mission_{objective_id}"
        
        # Check if mission already exists
        existing_mission = self.mission_state.find_mission(mission_id)
        if existing_mission:
            return None  # Already exists
        
        # Create new mission
        mission = Mission(
            id=mission_id,
            vision=objective['description'],
            status='active',
            priority='medium',
            success_criteria=[f"Achieve {objective['target_value']} for {objective['target_metric']}"],
            target_metrics={},
            capability_tags=['core.planning', 'horiz.objective_driven'],
            milestones=[],
        )
        
        # Add initial milestone
        initial_milestone = MissionMilestone(
            id=f"{mission_id}_milestone_1",
            goal=f"Begin work on {objective['description']}",
            status='planned',
            capability_tags=['core.planning'],
        )
        mission.add_milestone(initial_milestone)
        
        # Add the mission to the state
        self.mission_state.add_mission(mission)
        
        # Save updated mission state
        save_mission_state(self.mission_state, self.missions_path)
        
        return mission
    
    def roll_forward_objectives(self) -> List[Dict[str, str]]:
        """Roll forward persistent objectives to the next cycle."""
        # Load metrics history to understand current progress
        try:
            metrics_history = self.auto_evaluator._read_metrics_history()
        except Exception:
            metrics_history = {}
        
        # Update objective values based on latest metrics
        for obj_id, objective in self.objectives.items():
            target_metric = objective.get('target_metric')
            if target_metric and target_metric in metrics_history:
                metric_values = metrics_history[target_metric]
                if metric_values:
                    latest_value = metric_values[-1]  # Get the most recent value
                    objective['current_value'] = latest_value
                    # Calculate progress (0 to 1)
                    target_value = objective.get('target_value', 1.0)
                    progress = min(1.0, max(0.0, latest_value / target_value if target_value != 0 else 0))
                    objective['progress'] = progress
        
        # Save updated objectives
        self._save_objectives()
        
        # Generate new subgoals based on objectives
        return self.generate_subgoals()
    
    def track_mission_progress(self) -> Dict[str, float]:
        """Track overall mission progress for planning decisions."""
        progress_metrics = {}
        
        for mission in self.mission_state.missions:
            completed_milestones = sum(1 for m in mission.milestones if m.status == 'completed')
            total_milestones = len(mission.milestones)
            
            if total_milestones > 0:
                progress = completed_milestones / total_milestones
                progress_metrics[mission.id] = progress
            else:
                progress_metrics[mission.id] = 0.0
        
        return progress_metrics