import logging
from enum import Enum
from typing import Any

import numpy as np
from scipy.interpolate import splev, splprep

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import PlannerPolicy
from molmo_spaces.policy.solvers.navigation.astar_planner_policy import (
    AStarSmoothPlannerPolicy,
)
from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)


class NavDoorAStarSmoothPlannerPolicy(AStarSmoothPlannerPolicy):
    """Door-opening navigation variant that guarantees valid spline inputs."""

    def _ensure_spline_waypoints(self, world_waypoints: np.ndarray) -> np.ndarray:
        if len(world_waypoints) >= 4:
            return world_waypoints

        if len(world_waypoints) == 0:
            start_xy = self.robot_view.base.pose[:2, 3]
            target_xy = self.target_object.position[:2]
            world_waypoints = np.stack([start_xy, target_xy])
        elif len(world_waypoints) == 1:
            start_xy = self.robot_view.base.pose[:2, 3]
            goal_xy = world_waypoints[0]
            if np.linalg.norm(goal_xy - start_xy) < 1e-3:
                target_xy = self.target_object.position[:2]
                direction = goal_xy - target_xy
                norm = np.linalg.norm(direction)
                if norm < 1e-3:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = direction / norm
                start_xy = goal_xy - direction * 0.1
            world_waypoints = np.stack([start_xy, goal_xy])

        deltas = np.diff(world_waypoints, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]
        if total_length < 1e-6:
            offsets = np.linspace(-0.05, 0.05, 4)
            return world_waypoints[-1][None, :] + np.stack(
                [offsets, np.zeros_like(offsets)], axis=1
            )

        sample_lengths = np.linspace(0.0, total_length, 4)
        x = np.interp(sample_lengths, cumulative_lengths, world_waypoints[:, 0])
        y = np.interp(sample_lengths, cumulative_lengths, world_waypoints[:, 1])
        log.info(
            "[NAV-DOOR A* SMOOTH] Upsampled %d path waypoints to 4 spline waypoints.",
            len(world_waypoints),
        )
        return np.stack([x, y], axis=1)

    def build_policy_plan(self, world_waypoints):
        world_waypoints = self.stop_plan(world_waypoints)
        world_waypoints = self._ensure_spline_waypoints(world_waypoints)

        plan_length = sum(
            np.linalg.norm(world_waypoints[it] - world_waypoints[it - 1])
            for it in range(1, len(world_waypoints))
        )
        num_points = max(
            2,
            int(np.ceil(plan_length / self.config.policy_config.path_max_inter_waypoint_dist)),
        )

        tck, u = splprep(world_waypoints.transpose(), s=1e-5)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)

        dx_du, dy_du = splev(u_new, tck, der=1)
        thetas = np.arctan2(dy_du, dx_du)

        combined_waypoints = []
        start_theta = float(
            np.asarray(self.robot_view.get_noop_ctrl_dict(["base"])["base"][2]).reshape(-1)[0]
        )
        for theta in self.max_angle_waypoints(
            np.stack([start_theta, float(thetas[0])])[:, None]
        ):
            combined_waypoints.append(np.concatenate((world_waypoints[0], theta)))

        for cur_x, cur_y, cur_theta in zip(x_new, y_new, thetas):
            combined_waypoints.append(np.stack((cur_x, cur_y, cur_theta)))

        final_pos = world_waypoints[-1]
        target_pos = self.target_object.position[:2]
        final_theta = np.arctan2(target_pos[1] - final_pos[1], target_pos[0] - final_pos[0])
        for theta in self.max_angle_waypoints(np.stack([thetas[-1], final_theta])[:, None]):
            combined_waypoints.append(np.concatenate((final_pos, theta)))

        return np.array(combined_waypoints)


class NavThenDoorOpeningPhase(str, Enum):
    NAVIGATE = "navigate"
    FINAL_ALIGN = "final_align"
    DOOR_OPENING = "door_opening"
    DONE = "done"


class NavThenDoorOpeningPolicy(PlannerPolicy):
    """Sequential planner that delegates navigation, then door opening."""

    @staticmethod
    def _format_phase(phase: Any) -> str:
        value = getattr(phase, "name", phase)
        value = getattr(value, "value", value)
        value = str(value)
        if "." in value:
            value = value.rsplit(".", 1)[-1]
        return value.lower()

    @staticmethod
    def add_auxiliary_objects(config: MlSpacesExpConfig, spec) -> None:
        nav_config = config.model_copy(
            deep=True, update={"policy_config": config.policy_config.nav_policy_config}
        )
        config.policy_config.nav_policy_config.policy_cls.add_auxiliary_objects(nav_config, spec)

        opening_config = config.model_copy(
            deep=True,
            update={
                "policy_config": config.policy_config.opening_policy_config,
                "task_type": "door_open",
            },
        )
        config.policy_config.opening_policy_config.policy_cls.add_auxiliary_objects(
            opening_config, spec
        )

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)
        self.current_phase = NavThenDoorOpeningPhase.NAVIGATE
        self.navigation_succeeded: bool | None = None
        self.final_align_steps = 0

        nav_config = config.model_copy(
            deep=True, update={"policy_config": config.policy_config.nav_policy_config}
        )
        self.nav_policy = nav_config.policy_config.policy_cls(nav_config, task)
        self._opening_policy = None

    @property
    def opening_policy(self):
        if self._opening_policy is None:
            opening_config = self.config.model_copy(
                deep=True,
                update={
                    "policy_config": self.config.policy_config.opening_policy_config,
                    "task_type": "door_open",
                },
            )
            self._opening_policy = opening_config.policy_config.policy_cls(opening_config, self.task)
        return self._opening_policy

    @property
    def is_done(self) -> bool:
        return self.current_phase == NavThenDoorOpeningPhase.DONE

    def planners(self) -> dict[str, Any]:
        planners = {"navigation": self.nav_policy.planners()}
        if self._opening_policy is not None:
            planners["door_opening"] = self.opening_policy.planners
        return planners

    def reset(self):
        self.current_phase = NavThenDoorOpeningPhase.NAVIGATE
        self.navigation_succeeded = None
        self.final_align_steps = 0
        self.nav_policy.reset()
        self._opening_policy = None

    def get_phase(self) -> str:
        if self.current_phase == NavThenDoorOpeningPhase.DOOR_OPENING:
            if self._opening_policy is None:
                return f"{self.current_phase.value}:delegate"
            return f"{self.current_phase.value}:{self._format_phase(self.opening_policy.get_phase())}"
        return self.current_phase.value

    def get_all_phases(self) -> dict[str, int]:
        phases = {phase.value: i for i, phase in enumerate(NavThenDoorOpeningPhase)}
        if self._opening_policy is not None:
            for name, value in self.opening_policy.get_all_phases().items():
                phases[f"door_opening:{self._format_phase(name)}"] = len(phases) + int(value)
        else:
            phases["door_opening:delegate"] = len(phases)
        return phases

    def get_info(self) -> dict:
        info = super().get_info()
        info["phase"] = self.get_phase()
        info["navigation_done"] = self.current_phase != NavThenDoorOpeningPhase.NAVIGATE
        info["navigation_succeeded"] = self.navigation_succeeded
        info["handoff_distance_to_handle"] = self._distance_to_handle()
        return info

    def _handle_position(self) -> np.ndarray | None:
        if not hasattr(self.task, "get_door_handle_position"):
            return None
        return self.task.get_door_handle_position()

    def _base_xy_yaw(self) -> tuple[np.ndarray, float]:
        base_pose = self.task.env.current_robot.robot_view.base.pose
        xy = base_pose[:2, 3].copy()
        yaw = float(np.arctan2(base_pose[1, 0], base_pose[0, 0]))
        return xy, yaw

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _distance_to_handle(self) -> float | None:
        handle_pos = self._handle_position()
        if handle_pos is None:
            return None

        robot_xy, _ = self._base_xy_yaw()
        return float(np.linalg.norm(robot_xy - handle_pos[:2]))

    def _handoff_metrics(self) -> tuple[float | None, float | None]:
        handle_pos = self._handle_position()
        if handle_pos is None:
            return None, None

        robot_xy, robot_yaw = self._base_xy_yaw()
        distance = float(np.linalg.norm(robot_xy - handle_pos[:2]))
        target_yaw = float(np.arctan2(handle_pos[1] - robot_xy[1], handle_pos[0] - robot_xy[0]))
        yaw_error = abs(self._normalize_angle(robot_yaw - target_yaw))
        return distance, yaw_error

    def _done_action(self, **metadata: Any) -> dict[str, Any]:
        return {
            **self.task.env.current_robot.robot_view.get_noop_ctrl_dict(),
            "done": True,
            **metadata,
        }

    def _can_start_door_opening(self) -> bool:
        handoff_distance, yaw_error = self._handoff_metrics()
        max_handoff_distance = self.config.policy_config.handoff_max_distance_to_handle_m
        max_yaw_error = self.config.policy_config.final_align_yaw_threshold_rad
        return (
            handoff_distance is None
            or (
                handoff_distance <= max_handoff_distance
                and (yaw_error is None or yaw_error <= max_yaw_error)
            )
        )

    def _target_base_pose_for_handle(self) -> np.ndarray | None:
        handle_pos = self._handle_position()
        if handle_pos is None:
            return None

        robot_xy, _ = self._base_xy_yaw()
        handle_xy = handle_pos[:2]
        away_from_handle = robot_xy - handle_xy
        norm = float(np.linalg.norm(away_from_handle))
        if norm < 1e-6:
            away_from_handle = np.array([-1.0, 0.0])
            norm = 1.0
        away_from_handle = away_from_handle / norm

        target_distance = self.config.policy_config.final_align_target_distance_to_handle_m
        target_xy = handle_xy + away_from_handle * target_distance
        step = self.config.policy_config.final_align_step_size_m
        delta = target_xy - robot_xy
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > step:
            target_xy = robot_xy + delta / delta_norm * step

        target_yaw = float(np.arctan2(handle_xy[1] - target_xy[1], handle_xy[0] - target_xy[0]))
        return np.array([target_xy[0], target_xy[1], target_yaw])

    def _start_door_opening(self, handoff_distance: float | None):
        if hasattr(self.task, "refresh_door_opening_side"):
            self.task.refresh_door_opening_side()

        log.info(
            "Navigation handoff aligned; switching to door-opening policy "
            "(handle distance %.3fm).",
            -1.0 if handoff_distance is None else handoff_distance,
        )
        self.current_phase = NavThenDoorOpeningPhase.DOOR_OPENING
        return self.task.env.current_robot.robot_view.get_noop_ctrl_dict()

    def _switch_to_door_opening(self, handoff_distance: float | None):
        max_handoff_distance = self.config.policy_config.handoff_max_distance_to_handle_m
        _, yaw_error = self._handoff_metrics()
        yaw_threshold = self.config.policy_config.final_align_yaw_threshold_rad
        needs_alignment = (
            self.config.policy_config.final_align_enabled
            and handoff_distance is not None
            and (
                handoff_distance > max_handoff_distance
                or (yaw_error is not None and yaw_error > yaw_threshold)
            )
        )
        if needs_alignment:
            log.info(
                "Navigation complete; entering final_align before door opening "
                "(handle distance %.3fm, yaw error %.3frad).",
                handoff_distance,
                -1.0 if yaw_error is None else yaw_error,
            )
            self.current_phase = NavThenDoorOpeningPhase.FINAL_ALIGN
            self.final_align_steps = 0
            return self.task.env.current_robot.robot_view.get_noop_ctrl_dict()

        if handoff_distance is not None and (
            handoff_distance > max_handoff_distance
            or (yaw_error is not None and yaw_error > yaw_threshold)
        ):
            log.warning(
                "Navigation reached its goal but door handle handoff is not aligned "
                "(distance %.3fm / %.3fm, yaw error %.3frad / %.3frad); "
                "ending rollout before opening.",
                handoff_distance,
                max_handoff_distance,
                -1.0 if yaw_error is None else yaw_error,
                yaw_threshold,
            )
            self.current_phase = NavThenDoorOpeningPhase.DONE
            return self._done_action(
                navigation_success=self.navigation_succeeded,
                handoff_success=False,
                handoff_distance_to_handle=handoff_distance,
                handoff_yaw_error=-1.0 if yaw_error is None else yaw_error,
            )

        return self._start_door_opening(handoff_distance)

    def _final_align_action(self):
        handoff_distance, yaw_error = self._handoff_metrics()
        if self._can_start_door_opening():
            return self._start_door_opening(handoff_distance)

        if self.final_align_steps >= self.config.policy_config.final_align_max_steps:
            log.warning(
                "Final door-opening handoff alignment timed out "
                "(distance %.3fm, yaw error %.3frad).",
                -1.0 if handoff_distance is None else handoff_distance,
                -1.0 if yaw_error is None else yaw_error,
            )
            self.current_phase = NavThenDoorOpeningPhase.DONE
            return self._done_action(
                navigation_success=self.navigation_succeeded,
                handoff_success=False,
                handoff_distance_to_handle=-1.0
                if handoff_distance is None
                else handoff_distance,
                handoff_yaw_error=-1.0 if yaw_error is None else yaw_error,
            )

        target_pose = self._target_base_pose_for_handle()
        if target_pose is None:
            return self._start_door_opening(handoff_distance)

        self.final_align_steps += 1
        return {"done": False, "base": target_pose}

    def get_action(self, observation):
        if self.current_phase == NavThenDoorOpeningPhase.NAVIGATE:
            nav_action = self.nav_policy.get_action(observation)
            if nav_action.get("done", False):
                self.navigation_succeeded = bool(nav_action.get("navigation_success", True))
                if not self.navigation_succeeded:
                    handoff_distance = self._distance_to_handle()
                    max_failure_handoff_distance = (
                        self.config.policy_config.handoff_after_nav_failure_max_distance_to_handle_m
                    )
                    if (
                        handoff_distance is not None
                        and handoff_distance <= max_failure_handoff_distance
                    ):
                        log.warning(
                            "Navigation reported failure %.3fm from the handle; attempting "
                            "door opening because it is within fallback handoff distance %.3fm.",
                            handoff_distance,
                            max_failure_handoff_distance,
                        )
                        return self._switch_to_door_opening(handoff_distance)

                    log.warning(
                        "Navigation failed; ending nav-to-door-opening rollout without "
                        "switching to opening."
                    )
                    self.current_phase = NavThenDoorOpeningPhase.DONE
                    return self._done_action(
                        navigation_success=False,
                        handoff_distance_to_handle=-1.0
                        if handoff_distance is None
                        else handoff_distance,
                    )

                return self._switch_to_door_opening(self._distance_to_handle())
            return nav_action

        if self.current_phase == NavThenDoorOpeningPhase.FINAL_ALIGN:
            return self._final_align_action()

        if self.current_phase == NavThenDoorOpeningPhase.DOOR_OPENING:
            try:
                opening_action = self.opening_policy.get_action(observation)
            except ValueError as exc:
                if "Max planning reattempts reached" not in str(exc):
                    raise

                log.warning(
                    "Door-opening planner failed after max reattempts; ending rollout cleanly."
                )
                self.current_phase = NavThenDoorOpeningPhase.DONE
                return self._done_action(
                    navigation_success=self.navigation_succeeded,
                    handoff_success=True,
                    opening_success=False,
                    opening_error_code=1,
                )

            if opening_action.get("done", False):
                self.current_phase = NavThenDoorOpeningPhase.DONE
                return self._done_action()
            return opening_action

        return self._done_action()
