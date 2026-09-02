import logging

import numpy as np

from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.nav_to_pick_and_place_task import NavToPickAndPlaceTask
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
from molmo_spaces.tasks.task_sampler_errors import RobotPlacementError
from molmo_spaces.utils.pose import pose_mat_to_7d

log = logging.getLogger(__name__)


class NavToPickAndPlaceTaskSampler(PickAndPlaceTaskSampler):
    """Sample pick-and-place tasks with a far robot start for navigation."""

    def _sample_task(self, env: CPUMujocoEnv) -> NavToPickAndPlaceTask:
        self._configure_pick_and_place(env)
        return NavToPickAndPlaceTask(env, self.config)

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        """Place the robot far from the pickup object for nav-to-manipulation rollouts."""
        task_cfg = self.config.task_config
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()

        if not isinstance(pickup_obj, MlSpacesObject):
            raise ValueError(f"Invalid pickup object type: {type(pickup_obj)}")

        robot_view = env.current_robot.robot_view
        target_pos = pickup_obj.position
        initial_robot_z = (
            target_pos[2]
            + self.config.task_sampler_config.robot_object_z_offset
            + np.random.uniform(
                self.config.task_sampler_config.robot_object_z_offset_random_min,
                self.config.task_sampler_config.robot_object_z_offset_random_max,
            )
        )

        sampling_radius_range = self.config.task_sampler_config.base_pose_sampling_radius_range
        log.info(
            "[NAV-TO-PNP] Placing robot %.2f-%.2fm from pickup object '%s'",
            sampling_radius_range[0],
            sampling_radius_range[1],
            pickup_obj.name,
        )

        if self._datagen_profiler is not None:
            self._datagen_profiler.start("nav_to_pnp_place_robot_far")

        robot_placed = env.place_robot_near(
            robot_view=robot_view,
            target=pickup_obj,
            max_tries=self.config.task_sampler_config.max_robot_placement_attempts,
            sampling_radius_range=sampling_radius_range,
            robot_safety_radius=self.config.task_sampler_config.robot_safety_radius,
            preserve_z=initial_robot_z,
            face_target=False,
            check_camera_visibility=self.config.task_sampler_config.check_robot_placement_visibility,
            visibility_resolver=self.get_visibility_resolver(env),
            excluded_positions=self.used_robot_positions[pickup_obj.name],
            save_visibility_frames_dir=self.config.output_dir,
        )

        if self._datagen_profiler is not None:
            self._datagen_profiler.end("nav_to_pnp_place_robot_far")

        if not robot_placed:
            log.info("[NAV-TO-PNP] Failed to place robot far from '%s'", pickup_obj.name)
            raise RobotPlacementError(f"Failed to place robot near object: {pickup_obj.name}")

        self.used_robot_positions[pickup_obj.name].append(robot_view.base.pose[:3, 3])
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
        pickup_obj_goal_pose[2] += 0.05
        task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()

        final_pos = robot_view.base.pose[:3, 3]
        distance_to_obj = np.linalg.norm(final_pos[:2] - target_pos[:2])
        log.info(
            "[NAV-TO-PNP] Robot start=(%.3f, %.3f, %.3f), object=(%.3f, %.3f, %.3f), distance=%.3fm",
            final_pos[0],
            final_pos[1],
            final_pos[2],
            target_pos[0],
            target_pos[1],
            target_pos[2],
            distance_to_obj,
        )
