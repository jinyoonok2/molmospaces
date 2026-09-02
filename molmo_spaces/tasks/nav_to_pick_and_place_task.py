import logging

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import BaseMujocoEnv
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask

log = logging.getLogger(__name__)


class NavToPickAndPlaceTask(PickAndPlaceTask):
    """Long-horizon task: navigate to the pickup object, then pick and place it."""

    def __init__(self, env: BaseMujocoEnv, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(env, exp_config)
        self.nav_objs = self._get_nav_objects()

    def _get_nav_objects(self) -> list[list[MlSpacesObject]]:
        task_config = self.config.task_config
        nav_objs_per_batch = []
        for data in self._env.mj_datas:
            nav_objs_per_batch.append(
                [MlSpacesObject(data=data, object_name=task_config.pickup_obj_name)]
            )
        return nav_objs_per_batch

    def get_nav_object_priority(self, batch_index: int) -> list[MlSpacesObject]:
        robot_base_pos = self._env.robots[batch_index].robot_view.base.pose[:3, 3]
        return sorted(
            self.nav_objs[batch_index],
            key=lambda obj: np.linalg.norm(obj.position[:2] - robot_base_pos[:2]),
        )

    def get_nearest_nav_object(self, batch_index: int) -> MlSpacesObject | None:
        priority = self.get_nav_object_priority(batch_index)
        return priority[0] if priority else None

    def get_task_description(self) -> str:
        pickup_name = self.config.task_config.referral_expressions["pickup_name"]
        place_name = self.config.task_config.referral_expressions["place_name"]
        return (
            f"Navigate to the {pickup_name}, pick it up, and place it in or on "
            f"the {place_name}"
        )
