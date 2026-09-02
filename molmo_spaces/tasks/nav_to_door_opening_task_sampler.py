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

    target_grounding_mode: Literal[
        "none", "visible_unique", "point_prompt", "room_door_id"
    ] = "none"
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

    def _nearby_room_ids(
        self,
        position: np.ndarray,
        max_distance_m: float = 1.0,
    ) -> list[str]:
        """Return room IDs nearest a world position, ordered by map distance."""

        room_map = self._cached_thormap.room_map
        room_names = self._cached_thormap.room_ids_to_name
        if room_map is None or room_names is None:
            return []

        pixel = self._cached_thormap.pos_m_to_px(np.asarray(position)).astype(int)
        radius = max(1, int(round(max_distance_m * self._cached_thormap.px_per_m)))
        row_min = max(0, int(pixel[0] - radius))
        row_max = min(room_map.shape[0], int(pixel[0] + radius + 1))
        col_min = max(0, int(pixel[1] - radius))
        col_max = min(room_map.shape[1], int(pixel[1] + radius + 1))
        local_map = room_map[row_min:row_max, col_min:col_max]
        local_center = np.array([pixel[0] - row_min, pixel[1] - col_min])

        rooms_by_distance: list[tuple[float, str]] = []
        for room_label in np.unique(local_map):
            room_label = int(room_label)
            if room_label == 0 or room_label not in room_names:
                continue
            coordinates = np.argwhere(local_map == room_label)
            min_distance = float(
                np.linalg.norm(coordinates - local_center[None, :], axis=1).min()
            )
            if min_distance <= radius:
                rooms_by_distance.append(
                    (min_distance, str(room_names[room_label]))
                )

        rooms_by_distance.sort(key=lambda item: (item[0], item[1]))
        return [room_name for _, room_name in rooms_by_distance]

    def _record_room_door_id_grounding(
        self,
        door_object: Door,
        door_body_names: list[str],
        robot_position: np.ndarray,
    ) -> None:
        """Record a stable scene-local door ID and actionable map location."""

        task_config = self.config.task_config
        sorted_door_names = sorted(door_body_names)
        door_index = sorted_door_names.index(door_object.name)
        handle_position = door_object.get_handle_pose()[:3]
        initial_rooms = self._nearby_room_ids(robot_position)
        adjacent_rooms = self._nearby_room_ids(handle_position)

        task_config.target_grounding_mode = "room_door_id"
        task_config.initial_room_id = initial_rooms[0] if initial_rooms else "unknown_room"
        task_config.target_door_id = (
            f"house_{self.current_house_index}/door_{door_index}"
        )
        task_config.target_door_position_xy = [
            float(handle_position[0]),
            float(handle_position[1]),
        ]
        task_config.target_adjacent_room_ids = adjacent_rooms[:2]
        log.info(
            "[NAV-TO-DOOR ID GROUNDING] initial_room=%s target_door=%s "
            "target_xy=(%.3f, %.3f) adjacent_rooms=%s",
            task_config.initial_room_id,
            task_config.target_door_id,
            handle_position[0],
            handle_position[1],
            task_config.target_adjacent_room_ids,
        )

    def _sample_door_and_place_robot(self, env: CPUMujocoEnv, door_body_names: list[str]):
        if len(door_body_names) == 0:
            raise HouseInvalidForTask("No doors found in the scene")

        task_config = self.config.task_config
        task_config.target_grounding_mode = "none"
        task_config.target_door_visibility_fraction = None
        task_config.target_handle_visibility_fraction = None
        task_config.visible_competing_door_names = []
        task_config.initial_room_id = None
        task_config.target_door_id = None
        task_config.target_door_position_xy = None
        task_config.target_adjacent_room_ids = []

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
                    in {"visible_unique", "point_prompt"},
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
                elif grounding_mode == "room_door_id":
                    self._record_room_door_id_grounding(
                        door_object,
                        door_body_names,
                        robot_pos,
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
