import logging
from typing import Literal

import numpy as np
from scipy.ndimage import label as label_connected_components

from molmo_spaces.configs.task_sampler_configs import DoorOpeningTaskSamplerConfig
from molmo_spaces.env.data_views import Door
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.nav_to_door_opening_task import DoorHandleNavTarget
from molmo_spaces.tasks.opening_task_samplers import DoorOpeningTaskSampler
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask, RobotPlacementError
from molmo_spaces.tasks.util_samplers.navgoal_sampler import NavGoalSampler

log = logging.getLogger(__name__)


class NavToDoorOpeningTaskSamplerConfig(DoorOpeningTaskSamplerConfig):
    """Sampler controls used only by the nav-to-door-opening extension."""

    target_grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "none"
    target_visibility_camera: str = "head_camera"
    target_door_min_visibility_fraction: float = 0.0001
    target_handle_min_visibility_fraction: float = 0.00001
    competing_door_min_visibility_fraction: float = 0.0001


class NavToDoorOpeningTaskSampler(DoorOpeningTaskSampler):
    """Sample door-opening tasks with a far robot start for navigation."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._connectivity_labels: np.ndarray | None = None

    def init_scene(self, env) -> None:
        super().init_scene(env)
        self._connectivity_labels = None

    def reset(self) -> None:
        super().reset()
        self._connectivity_labels = None

    @property
    def connectivity_labels(self) -> np.ndarray:
        """Label the same four-connected, downscaled free-space grid used by A*."""

        if self._connectivity_labels is None:
            if self._cached_thormap is None:
                raise RuntimeError("Occupancy map is unavailable for connectivity validation")

            grid = np.asarray(self._cached_thormap.occupancy, dtype=bool)
            downscale = (
                self.config.policy_config.nav_policy_config.planner_config.downscale_factor
            )
            padded = np.zeros(
                (
                    grid.shape[0] + (downscale - grid.shape[0] % downscale),
                    grid.shape[1] + (downscale - grid.shape[1] % downscale),
                ),
                dtype=bool,
            )
            padded[: grid.shape[0], : grid.shape[1]] = grid
            downscaled_grid = (
                padded.reshape(
                    padded.shape[0] // downscale,
                    downscale,
                    padded.shape[1] // downscale,
                    downscale,
                )
                .min(axis=1)
                .min(axis=-1)
            )
            four_connected = np.array(
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                ],
                dtype=np.uint8,
            )
            self._connectivity_labels, num_components = label_connected_components(
                downscaled_grid, structure=four_connected
            )
            log.info(
                "[NAV-TO-DOOR CONNECTIVITY] Found %d navigable components",
                num_components,
            )

        return self._connectivity_labels

    def _component_at_position(self, position: np.ndarray) -> int | None:
        """Resolve a world position to a nearby A*-navigable component."""

        downscale = self.config.policy_config.nav_policy_config.planner_config.downscale_factor
        max_search = (
            self.config.policy_config.nav_policy_config.planner_config.max_start_goal_distance
        )
        discrete = np.floor(
            self._cached_thormap.pos_m_to_px(np.asarray(position)) / downscale
        ).astype(np.int32)
        labels = self.connectivity_labels

        def component_at(row: int, col: int) -> int | None:
            if row < 0 or col < 0 or row >= labels.shape[0] or col >= labels.shape[1]:
                return None
            component = int(labels[row, col])
            return component if component > 0 else None

        component = component_at(int(discrete[0]), int(discrete[1]))
        if component is not None:
            return component

        for radius in range(1, max_search + 1):
            for row_offset in range(-radius, radius + 1):
                for col_offset in range(-radius, radius + 1):
                    if abs(row_offset) != radius and abs(col_offset) != radius:
                        continue
                    component = component_at(
                        int(discrete[0] + row_offset),
                        int(discrete[1] + col_offset),
                    )
                    if component is not None:
                        return component

        return None

    def _navigation_goal_position(self, door_object: Door) -> np.ndarray | None:
        """Reproduce the runtime navigation goal sampled around the door handle."""

        handle_position = door_object.get_handle_pose()[:3]
        nav_target = DoorHandleNavTarget(
            name=f"{door_object.name}:handle",
            position=handle_position,
        )
        goal_sampler = NavGoalSampler(self._cached_thormap)
        goal_sampler.set_target(nav_target)
        sampled_goal = goal_sampler.sample()
        if sampled_goal is None:
            return None
        return np.asarray(sampled_goal[0])

    def _positions_are_connected(
        self,
        robot_position: np.ndarray,
        goal_position: np.ndarray,
    ) -> bool:
        start_component = self._component_at_position(robot_position)
        goal_component = self._component_at_position(goal_position)
        connected = start_component is not None and start_component == goal_component
        log.info(
            "[NAV-TO-DOOR CONNECTIVITY] start_component=%s goal_component=%s connected=%s",
            start_component,
            goal_component,
            connected,
        )
        return connected

    def _evaluate_target_grounding(
        self,
        env: CPUMujocoEnv,
        door_object: Door,
        door_body_names: list[str],
    ) -> tuple[bool, float, float, list[str]]:
        """Check whether the selected door is visible and unambiguous."""

        sampler_config = self.config.task_sampler_config
        camera_name = sampler_config.target_visibility_camera
        if camera_name not in env.camera_manager.registry:
            self.setup_cameras(env, deterministic_only=True)
        env.camera_manager.registry.update_all_cameras(env)

        try:
            segmentation = env.render_segmentation_frame(camera_name)
        except Exception as exc:
            log.warning(
                "[NAV-TO-DOOR GROUNDING] Failed to render '%s': %s",
                camera_name,
                exc,
            )
            return False, 0.0, 0.0, []

        door_visibility: dict[str, float] = {}
        for door_name in door_body_names:
            try:
                door_visibility[door_name] = env.segmentation_fraction(
                    segmentation, door_name
                )
            except (KeyError, ValueError):
                door_visibility[door_name] = 0.0

        target_visibility = door_visibility.get(door_object.name, 0.0)
        handle_visibility = 0.0
        for handle_index in range(door_object.num_handles):
            try:
                handle_visibility = max(
                    handle_visibility,
                    env.segmentation_fraction(
                        segmentation, door_object.handle_name(handle_index)
                    ),
                )
            except (KeyError, ValueError):
                continue

        competing_doors = sorted(
            door_name
            for door_name, visibility in door_visibility.items()
            if door_name != door_object.name
            and visibility >= sampler_config.competing_door_min_visibility_fraction
        )
        target_visible = (
            target_visibility >= sampler_config.target_door_min_visibility_fraction
            and (
                sampler_config.target_grounding_mode != "point_prompt"
                or handle_visibility
                >= sampler_config.target_handle_min_visibility_fraction
            )
        )
        unambiguous = (
            sampler_config.target_grounding_mode != "visible_unique"
            or len(competing_doors) == 0
        )
        accepted = target_visible and unambiguous
        log.info(
            "[NAV-TO-DOOR GROUNDING] camera=%s target_visibility=%.6f "
            "handle_visibility=%.6f competing_doors=%d accepted=%s",
            camera_name,
            target_visibility,
            handle_visibility,
            len(competing_doors),
            accepted,
        )
        return accepted, target_visibility, handle_visibility, competing_doors

    def _record_target_grounding(
        self,
        target_visibility: float,
        handle_visibility: float,
        competing_doors: list[str],
    ) -> None:
        task_config = self.config.task_config
        task_config.target_grounding_mode = (
            self.config.task_sampler_config.target_grounding_mode
        )
        task_config.target_door_visibility_fraction = target_visibility
        task_config.target_handle_visibility_fraction = handle_visibility
        task_config.visible_competing_door_names = competing_doors

    def _sample_door_and_place_robot(self, env: CPUMujocoEnv, door_body_names: list[str]):
        if len(door_body_names) == 0:
            raise HouseInvalidForTask("No doors found in the scene")

        task_config = self.config.task_config
        task_config.target_grounding_mode = "none"
        task_config.target_door_visibility_fraction = None
        task_config.target_handle_visibility_fraction = None
        task_config.visible_competing_door_names = []

        np.random.shuffle(door_body_names)
        sampling_radius_range = self.config.task_sampler_config.base_pose_sampling_radius_range
        max_attempts = self.config.task_sampler_config.max_robot_placement_attempts
        for chosen_door_name in door_body_names:
            door_object = Door(chosen_door_name, env.mj_datas[0])
            door_object.set_joint_position(door_object.get_hinge_joint_index(), 0.0)
            goal_position = self._navigation_goal_position(door_object)
            if goal_position is None:
                log.info(
                    "[NAV-TO-DOOR CONNECTIVITY] No navigation goal for door '%s'; "
                    "trying next door",
                    door_object.name,
                )
                continue

            log.info(
                "[NAV-TO-DOOR] Placing robot %.2f-%.2fm from door '%s'",
                sampling_radius_range[0],
                sampling_radius_range[1],
                door_object.name,
            )
            excluded_positions = list(self.used_robot_positions[door_object.name])
            for placement_attempt in range(1, max_attempts + 1):
                robot_placed = env.place_robot_near(
                    robot_view=env.current_robot.robot_view,
                    target=door_object,
                    max_tries=1,
                    sampling_radius_range=sampling_radius_range,
                    robot_safety_radius=self.config.task_sampler_config.robot_safety_radius,
                    face_target=self.config.task_sampler_config.target_grounding_mode
                    != "none",
                    check_camera_visibility=self.config.task_sampler_config.check_robot_placement_visibility,
                    visibility_resolver=self.get_visibility_resolver(env),
                    excluded_positions=excluded_positions,
                )
                if not robot_placed:
                    continue

                robot_pos = env.current_robot.robot_view.base.pose[:3, 3]
                if not self._positions_are_connected(robot_pos, goal_position):
                    excluded_positions.append(robot_pos.copy())
                    log.info(
                        "[NAV-TO-DOOR CONNECTIVITY] Rejected disconnected robot pose "
                        "on attempt %d/%d",
                        placement_attempt,
                        max_attempts,
                    )
                    continue

                grounding_mode = self.config.task_sampler_config.target_grounding_mode
                if grounding_mode in {"visible_unique", "point_prompt"}:
                    (
                        grounding_accepted,
                        target_visibility,
                        handle_visibility,
                        competing_doors,
                    ) = self._evaluate_target_grounding(
                        env,
                        door_object,
                        door_body_names,
                    )
                    if not grounding_accepted:
                        excluded_positions.append(robot_pos.copy())
                        log.info(
                            "[NAV-TO-DOOR GROUNDING] Rejected robot pose on attempt %d/%d",
                            placement_attempt,
                            max_attempts,
                        )
                        continue
                    self._record_target_grounding(
                        target_visibility,
                        handle_visibility,
                        competing_doors,
                    )
                handle_pos = door_object.get_handle_pose()[:3]
                log.info(
                    "[NAV-TO-DOOR] Robot start=(%.3f, %.3f, %.3f), handle=(%.3f, %.3f, %.3f), distance=%.3fm",
                    robot_pos[0],
                    robot_pos[1],
                    robot_pos[2],
                    handle_pos[0],
                    handle_pos[1],
                    handle_pos[2],
                    float(np.linalg.norm(robot_pos[:2] - handle_pos[:2])),
                )
                self.used_robot_positions[door_object.name].append(robot_pos)
                return door_object

            log.info(
                "[NAV-TO-DOOR] Failed to place a connected robot pose near '%s'. "
                "Trying next door...",
                door_object.name,
            )

        raise RobotPlacementError("Was not able to place robot far from any door in the scene.")
