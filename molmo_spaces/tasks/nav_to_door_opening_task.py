import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.task_configs import DoorOpeningTaskConfig
from molmo_spaces.env.env import BaseMujocoEnv
from molmo_spaces.tasks.opening_tasks import DoorOpeningTask

log = logging.getLogger(__name__)


class NavToDoorOpeningTaskConfig(DoorOpeningTaskConfig):
    """Task metadata used only by the nav-to-door-opening extension."""

    target_grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "none"
    target_door_visibility_fraction: float | None = None
    target_handle_visibility_fraction: float | None = None
    visible_competing_door_names: list[str] = []


@dataclass
class DoorHandleNavTarget:
    """Navigation target wrapper centered on the active door handle."""

    name: str
    position: np.ndarray


class NavToDoorOpeningTask(DoorOpeningTask):
    """Long-horizon task: navigate to a door handle, then open the door."""

    def __init__(self, env: BaseMujocoEnv, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(env, exp_config)
        self.nav_objs = self._get_nav_objects()

    def _get_nav_objects(self) -> list[list[DoorHandleNavTarget]]:
        nav_target = DoorHandleNavTarget(
            name=f"{self.door_object.name}:handle",
            position=self.get_door_handle_position(),
        )
        return [[nav_target] for _ in self._env.mj_datas]

    def get_nav_object_priority(self, batch_index: int) -> list[DoorHandleNavTarget]:
        robot_base_pos = self._env.robots[batch_index].robot_view.base.pose[:3, 3]
        return sorted(
            self.nav_objs[batch_index],
            key=lambda obj: np.linalg.norm(obj.position[:2] - robot_base_pos[:2]),
        )

    def get_nearest_nav_object(self, batch_index: int) -> DoorHandleNavTarget | None:
        priority = self.get_nav_object_priority(batch_index)
        return priority[0] if priority else None

    def refresh_door_opening_side(self) -> None:
        """Recompute handle side and push/pull mode after navigation handoff."""

        self._use_other_side_handle = self.check_if_use_flip_side_handle()
        self._is_pushing_door = self.check_if_pushing_door()

    def get_task_description(self) -> str:
        if self.config.task_config.target_grounding_mode == "visible_unique":
            return f"Navigate to the visible door and {super().get_task_description().lower()}"
        if self.config.task_config.target_grounding_mode == "point_prompt":
            return f"Navigate to the pointed door and {super().get_task_description().lower()}"
        return f"Navigate to the door and {super().get_task_description().lower()}"

    def get_obs_scene(self) -> dict:
        obs_scene = super().get_obs_scene()
        task_config = self.config.task_config
        obs_scene.update(
            {
                "target_door_name": self.door_object.name,
                "target_grounding_mode": task_config.target_grounding_mode,
                "target_door_visibility_fraction": (
                    task_config.target_door_visibility_fraction
                ),
                "target_handle_visibility_fraction": (
                    task_config.target_handle_visibility_fraction
                ),
                "visible_competing_door_names": (
                    task_config.visible_competing_door_names
                ),
            }
        )
        return obs_scene
