import pytest
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

from a3x.planning.mission_state import (
    MetricSnapshot,
    Mission,
    MissionMilestone,
    MissionState,
    MissionTelemetry,
)
from a3x.planning.storage import load_mission_state, save_mission_state, sync_mission_state


class TestLoadMissionState:
    def setup_method(self) -> None:
        self.test_path = Path("tests/temp_mission_state.yaml")
        self.test_path.parent.mkdir(exist_ok=True)
        if self.test_path.exists():
            self.test_path.unlink()

    def teardown_method(self) -> None:
        if self.test_path.exists():
            self.test_path.unlink()
        self.test_path.parent.rmdir()

    def test_load_missing_file(self) -> None:
        state = load_mission_state(self.test_path)
        assert isinstance(state, MissionState)
        assert len(state.missions) == 0
        assert state.generated_at == datetime.now(timezone.utc).isoformat()
        assert state.version == "0.1"

    @patch("pathlib.Path.read_text")
    @patch("yaml.safe_load")
    def test_load_valid_yaml(self, mock_yaml: Mock, mock_read: Mock) -> None:
        sample_yaml = {
            "missions": [
                {
                    "id": "test-mission",
                    "vision": "Test Vision",
                    "status": "active",
                    "priority": "high",
                    "success_criteria": ["crit1"],
                    "target_metrics": {"metric1": {"current": 0.5, "target": 1.0}},
                    "capability_tags": ["planning"],
                    "milestones": [
                        {
                            "id": "mil1",
                            "goal": "Milestone Goal",
                            "status": "planned",
                            "capability_tags": ["exec"],
                            "dependencies": [],
                            "metrics": {},
                        }
                    ],
                    "telemetry": {
                        "metric_summaries": {},
                        "reflections": ["ref1"],
                        "last_updated": "2023-01-01T00:00:00Z",
                    },
                    "created_at": "2023-01-01T00:00:00Z",
                    "updated_at": "2023-01-01T00:00:00Z",
                }
            ],
            "generated_at": "2023-01-01T00:00:00Z",
            "version": "0.2",
        }
        mock_yaml.return_value = sample_yaml
        mock_read.return_value = "yaml content"

        state = load_mission_state(self.test_path)

        assert len(state.missions) == 1
        mission = state.missions[0]
        assert mission.id == "test-mission"
        assert mission.vision == "Test Vision"
        assert len(mission.milestones) == 1
        milestone = mission.milestones[0]
        assert milestone.id == "mil1"
        assert milestone.goal == "Milestone Goal"
        assert state.generated_at == "2023-01-01T00:00:00Z"
        assert state.version == "0.2"

    @patch("pathlib.Path.read_text")
    @patch("yaml.safe_load")
    def test_load_invalid_yaml(self, mock_yaml: Mock, mock_read: Mock) -> None:
        mock_yaml.return_value = {"missions": "invalid dict"}  # Not list of dicts
        mock_read.return_value = "invalid content"

        state = load_mission_state(self.test_path)

        assert len(state.missions) == 0  # Should filter invalid
        assert state.version == "0.1"

    @patch("pathlib.Path.read_text")
    @patch("yaml.safe_load")
    def test_load_malformed_mission(self, mock_yaml: Mock, mock_read: Mock) -> None:
        # Invalid date
        invalid_mission = {
            "id": "invalid",
            "created_at": "invalid-date",  # Will cause _mission_from_dict to return None
        }
        mock_yaml.return_value = {"missions": [invalid_mission]}
        mock_read.return_value = "content"

        state = load_mission_state(self.test_path)

        assert len(state.missions) == 0  # Filtered out

    def test_load_empty_yaml(self) -> None:
        with self.test_path.open("w") as f:
            f.write("")

        state = load_mission_state(self.test_path)
        assert len(state.missions) == 0


class TestSaveMissionState:
    def setup_method(self) -> None:
        self.test_path = Path("tests/temp_mission_state.yaml")
        self.test_path.parent.mkdir(exist_ok=True)
        if self.test_path.exists():
            self.test_path.unlink()

    def teardown_method(self) -> None:
        if self.test_path.exists():
            self.test_path.unlink()
        self.test_path.parent.rmdir()

    def test_save_and_round_trip(self) -> None:
        # Create sample state
        metric = MetricSnapshot(current=0.5, target=1.0, best=None, samples=10)
        milestone = MissionMilestone(
            id="mil1",
            goal="Test Milestone",
            status="planned",
            capability_tags=["test"],
            dependencies=[],
            eta=None,
            backlog_seed_id=None,
            metrics={"metric": metric},
            notes=None,
        )
        telemetry = MissionTelemetry(
            metric_summaries={"sum": metric},
            reflections=["test ref"],
            discovered_gaps=[],
            tools_required=[],
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        mission = Mission(
            id="test-mission",
            vision="Test Vision",
            status="active",
            priority="high",
            success_criteria=["crit"],
            target_metrics={"target": metric},
            capability_tags=["planning"],
            milestones=[milestone],
            telemetry=telemetry,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        state = MissionState(
            missions=[mission],
            generated_at=datetime.now(timezone.utc).isoformat(),
            version="0.2",
        )

        # Save
        save_mission_state(state, self.test_path)

        # Load back
        loaded_state = load_mission_state(self.test_path)

        # Verify fidelity
        assert len(loaded_state.missions) == 1
        loaded_mission = loaded_state.missions[0]
        assert loaded_mission.id == "test-mission"
        assert loaded_mission.vision == "Test Vision"
        assert len(loaded_mission.milestones) == 1
        assert loaded_mission.milestones[0].goal == "Test Milestone"
        assert loaded_state.version == "0.2"

    @patch("pathlib.Path.write_text")
    def test_save_permission_denied(self, mock_write: Mock) -> None:
        mock_write.side_effect = PermissionError("Access denied")

        state = MissionState()
        with pytest.raises(PermissionError):
            save_mission_state(state, self.test_path)

    @patch("pathlib.Path.parent.mkdir")
    @patch("pathlib.Path.write_text")
    def test_save_creates_directory(self, mock_write: Mock, mock_mkdir: Mock) -> None:
        save_mission_state(MissionState(), self.test_path)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestSyncMissionState:
    def test_sync_add_new_mission(self) -> None:
        original_state = MissionState()
        new_mission = Mission(
            id="new-mission",
            vision="New Vision",
            status="draft",
            priority="low",
            success_criteria=[],
            target_metrics={},
            capability_tags=[],
            milestones=[],
            telemetry=MissionTelemetry(),
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        synced_state = sync_mission_state(original_state, [new_mission])

        assert len(synced_state.missions) == 1
        assert synced_state.missions[0].id == "new-mission"
        assert synced_state.generated_at != original_state.generated_at  # Updated

    def test_sync_update_existing(self) -> None:
        original_mission = Mission(
            id="existing",
            vision="Old Vision",
            status="draft",
            priority="low",
            success_criteria=[],
            target_metrics={},
            capability_tags=[],
            milestones=[],
            telemetry=MissionTelemetry(),
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        original_state = MissionState(missions=[original_mission])

        updated_mission = Mock(spec=Mission)
        updated_mission.id = "existing"
        updated_mission.vision = "Updated Vision"
        updated_mission.status = "active"

        synced_state = sync_mission_state(original_state, [updated_mission])

        assert len(synced_state.missions) == 1
        assert synced_state.missions[0].vision == "Updated Vision"
        assert synced_state.missions[0].status == "active"

    def test_sync_multiple_missions(self) -> None:
        state = MissionState()
        missions = [
            Mission(id="m1", vision="V1", status="s1", priority="p1", success_criteria=[], target_metrics={}, capability_tags=[], milestones=[], telemetry=MissionTelemetry(), created_at="2023-01-01T00:00:00Z", updated_at="2023-01-01T00:00:00Z"),
            Mission(id="m2", vision="V2", status="s2", priority="p2", success_criteria=[], target_metrics={}, capability_tags=[], milestones=[], telemetry=MissionTelemetry(), created_at="2023-01-01T00:00:00Z", updated_at="2023-01-01T00:00:00Z"),
        ]

        synced = sync_mission_state(state, missions)

        assert len(synced.missions) == 2
        assert {m.id for m in synced.missions} == {"m1", "m2"}

    def test_sync_no_changes(self) -> None:
        original_state = MissionState(
            missions=[
                Mission(id="m1", vision="V1", status="s1", priority="p1", success_criteria=[], target_metrics={}, capability_tags=[], milestones=[], telemetry=MissionTelemetry(), created_at="2023-01-01T00:00:00Z", updated_at="2023-01-01T00:00:00Z")
            ]
        )

        synced = sync_mission_state(original_state, original_state.missions)

        assert synced.missions == original_state.missions
        assert synced.generated_at != original_state.generated_at  # Still updates timestamp