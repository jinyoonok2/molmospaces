import logging
from enum import Enum
from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import PlannerPolicy
from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)


class NavThenPickAndPlacePhase(str, Enum):
    NAVIGATE = "navigate"
    FINAL_APPROACH = "final_approach"
    PICK_AND_PLACE = "pick_and_place"
    DONE = "done"


class NavThenPickAndPlacePolicy(PlannerPolicy):
    """Sequential planner that delegates navigation, then pick-and-place."""

    @staticmethod
    def _default_target_poses() -> dict[str, np.ndarray]:
        dummy_pose = np.eye(4)
        return {
            "grasp": dummy_pose,
            "pregrasp": dummy_pose,
            "lift": dummy_pose,
            "preplace": dummy_pose,
            "place": dummy_pose,
            "postplace": dummy_pose,
        }

    @staticmethod
    def _format_phase(phase: Any) -> str:
        value = getattr(phase, "value", phase)
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

        manip_config = config.model_copy(
            deep=True,
            update={
                "policy_config": config.policy_config.manip_policy_config,
                "task_type": "pick_and_place",
            },
        )
        config.policy_config.manip_policy_config.policy_cls.add_auxiliary_objects(
            manip_config, spec
        )

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)
        self.current_phase = NavThenPickAndPlacePhase.NAVIGATE
        self.target_poses = self._default_target_poses()
        self.navigation_succeeded: bool | None = None
        self.final_approach_steps = 0
        self.final_approach_best_distance = float("inf")
        self.final_approach_stall_steps = 0

        nav_config = config.model_copy(
            deep=True, update={"policy_config": config.policy_config.nav_policy_config}
        )
        self.nav_policy = nav_config.policy_config.policy_cls(nav_config, task)
        self._manip_policy = None

    @property
    def manip_policy(self):
        if self._manip_policy is None:
            manip_config = self.config.model_copy(
                deep=True,
                update={
                    "policy_config": self.config.policy_config.manip_policy_config,
                    "task_type": "pick_and_place",
                },
            )
            self._manip_policy = manip_config.policy_config.policy_cls(manip_config, self.task)
            self.target_poses.update(getattr(self._manip_policy, "target_poses", {}))
        return self._manip_policy

    @property
    def is_done(self) -> bool:
        return self.current_phase == NavThenPickAndPlacePhase.DONE

    def planners(self) -> dict[str, Any]:
        planners = {"navigation": self.nav_policy.planners()}
        if self._manip_policy is not None:
            planners["pick_and_place"] = self._manip_policy.planners()
        return planners

    def reset(self):
        self.current_phase = NavThenPickAndPlacePhase.NAVIGATE
        self.target_poses = self._default_target_poses()
        self.navigation_succeeded = None
        self.final_approach_steps = 0
        self.final_approach_best_distance = float("inf")
        self.final_approach_stall_steps = 0
        self.nav_policy.reset()
        self._manip_policy = None

    def get_phase(self) -> str:
        if self.current_phase == NavThenPickAndPlacePhase.PICK_AND_PLACE:
            if self._manip_policy is None:
                return f"{self.current_phase.value}:delegate"
            return f"{self.current_phase.value}:{self._format_phase(self.manip_policy.get_phase())}"
        return self.current_phase.value

    def get_all_phases(self) -> dict[str, int]:
        phases = {phase.value: i for i, phase in enumerate(NavThenPickAndPlacePhase)}
        if self._manip_policy is not None:
            for name, value in self._manip_policy.get_all_phases().items():
                phases[f"pick_and_place:{name}"] = len(phases) + int(value)
        else:
            # The delegate is lazy so avoid constructing cuRobo planners just to expose phase names.
            phases["pick_and_place:delegate"] = len(phases)
        return phases

    def get_info(self) -> dict:
        info = super().get_info()
        info["phase"] = self.get_phase()
        info["navigation_done"] = self.current_phase != NavThenPickAndPlacePhase.NAVIGATE
        info["navigation_succeeded"] = self.navigation_succeeded
        return info

    def _distance_to_pickup(self) -> float | None:
        pickup_obj = self._pickup_object()
        if pickup_obj is None:
            return None

        robot_pos = self.task.env.current_robot.robot_view.base.pose[:3, 3]
        return float(np.linalg.norm(robot_pos[:2] - pickup_obj.position[:2]))

    def _pickup_object(self):
        if not hasattr(self.task, "get_nearest_nav_object"):
            return None

        batch_index = self.task.env.current_batch_index
        return self.task.get_nearest_nav_object(batch_index)

    def _done_action(self, **metadata: Any) -> dict[str, Any]:
        return {
            **self.task.env.current_robot.robot_view.get_noop_ctrl_dict(),
            "done": True,
            **metadata,
        }

    def _handoff_or_done_action(self, handoff_distance: float | None):
        max_handoff_distance = self.config.policy_config.handoff_max_distance_to_pickup_m
        if handoff_distance is None or handoff_distance <= max_handoff_distance:
            log.info(
                "Navigation complete; switching to pick-and-place policy "
                "(pickup distance %.3fm).",
                -1.0 if handoff_distance is None else handoff_distance,
            )
            self.current_phase = NavThenPickAndPlacePhase.PICK_AND_PLACE
            return self.task.env.current_robot.robot_view.get_noop_ctrl_dict()

        if self.config.policy_config.final_approach_enabled:
            log.info(
                "Navigation complete but pickup distance %.3fm is above %.3fm; "
                "starting final approach.",
                handoff_distance,
                max_handoff_distance,
            )
            self.current_phase = NavThenPickAndPlacePhase.FINAL_APPROACH
            self.final_approach_steps = 0
            self.final_approach_best_distance = float("inf")
            self.final_approach_stall_steps = 0
            return self._final_approach_action()

        log.warning(
            "Navigation reached its goal but pickup handoff is too far "
            "(%.3fm > %.3fm); ending rollout before manipulation.",
            handoff_distance,
            max_handoff_distance,
        )
        self.current_phase = NavThenPickAndPlacePhase.DONE
        return self._done_action(
            navigation_success=True,
            handoff_success=False,
            handoff_distance_to_pickup=handoff_distance,
        )

    def _final_approach_action(self) -> dict[str, Any]:
        pickup_obj = self._pickup_object()
        if pickup_obj is None:
            log.warning("Final approach has no pickup object; ending rollout.")
            self.current_phase = NavThenPickAndPlacePhase.DONE
            return self._done_action(navigation_success=True, handoff_success=False)

        robot_pose = self.task.env.current_robot.robot_view.base.pose
        robot_xy = robot_pose[:2, 3]
        pickup_xy = pickup_obj.position[:2]
        delta = pickup_xy - robot_xy
        distance = float(np.linalg.norm(delta))
        max_handoff_distance = self.config.policy_config.handoff_max_distance_to_pickup_m
        fallback_handoff_distance = max(
            max_handoff_distance,
            self.config.policy_config.final_approach_fallback_handoff_max_distance_m,
        )

        if distance <= max_handoff_distance:
            log.info(
                "Final approach complete; switching to pick-and-place policy "
                "(pickup distance %.3fm).",
                distance,
            )
            self.current_phase = NavThenPickAndPlacePhase.PICK_AND_PLACE
            return self.task.env.current_robot.robot_view.get_noop_ctrl_dict()

        min_progress = self.config.policy_config.final_approach_min_progress_m
        if distance < self.final_approach_best_distance - min_progress:
            self.final_approach_best_distance = distance
            self.final_approach_stall_steps = 0
        else:
            self.final_approach_stall_steps += 1

        if (
            self.final_approach_stall_steps
            >= self.config.policy_config.final_approach_stall_window_steps
            and distance <= fallback_handoff_distance
        ):
            log.warning(
                "Final approach stalled at %.3fm from pickup; switching to "
                "pick-and-place fallback (fallback threshold %.3fm).",
                distance,
                fallback_handoff_distance,
            )
            self.current_phase = NavThenPickAndPlacePhase.PICK_AND_PLACE
            return self.task.env.current_robot.robot_view.get_noop_ctrl_dict()

        if self.final_approach_steps >= self.config.policy_config.final_approach_max_steps:
            if distance <= fallback_handoff_distance:
                log.warning(
                    "Final approach timed out at %.3fm from pickup; switching to "
                    "pick-and-place fallback (fallback threshold %.3fm).",
                    distance,
                    fallback_handoff_distance,
                )
                self.current_phase = NavThenPickAndPlacePhase.PICK_AND_PLACE
                return self.task.env.current_robot.robot_view.get_noop_ctrl_dict()

            log.warning(
                "Final approach timed out after %d steps at %.3fm from pickup "
                "(threshold %.3fm).",
                self.final_approach_steps,
                distance,
                max_handoff_distance,
            )
            self.current_phase = NavThenPickAndPlacePhase.DONE
            return self._done_action(
                navigation_success=True,
                handoff_success=False,
                handoff_distance_to_pickup=distance,
            )

        unit_to_pickup = delta / max(distance, 1e-6)
        target_distance = self.config.policy_config.final_approach_distance_to_pickup_m
        remaining_approach = max(distance - target_distance, 0.0)
        step_size = self.config.policy_config.final_approach_step_size_m
        step_distance = min(step_size, remaining_approach)
        target_xy = robot_xy + unit_to_pickup * step_distance
        target_yaw = float(np.arctan2(delta[1], delta[0]))
        self.final_approach_steps += 1

        log.debug(
            "Final approach step %d: distance %.3fm, commanding %.3fm toward target",
            self.final_approach_steps,
            distance,
            step_distance,
        )
        return {"done": False, "base": np.array([target_xy[0], target_xy[1], target_yaw])}

    def get_action(self, observation):
        if self.current_phase == NavThenPickAndPlacePhase.NAVIGATE:
            nav_action = self.nav_policy.get_action(observation)
            if nav_action.get("done", False):
                self.navigation_succeeded = bool(nav_action.get("navigation_success", True))
                if not self.navigation_succeeded:
                    handoff_distance = self._distance_to_pickup()
                    max_failure_approach_distance = (
                        self.config.policy_config.final_approach_after_nav_failure_max_distance_m
                    )
                    if (
                        self.config.policy_config.final_approach_enabled
                        and handoff_distance is not None
                        and handoff_distance <= max_failure_approach_distance
                    ):
                        log.warning(
                            "Navigation reported failure %.3fm from pickup; attempting "
                            "final approach before giving up.",
                            handoff_distance,
                        )
                        self.current_phase = NavThenPickAndPlacePhase.FINAL_APPROACH
                        self.final_approach_steps = 0
                        self.final_approach_best_distance = float("inf")
                        self.final_approach_stall_steps = 0
                        return self._final_approach_action()

                    log.warning(
                        "Navigation failed; ending nav-to-pick-and-place rollout without "
                        "switching to manipulation."
                    )
                    self.current_phase = NavThenPickAndPlacePhase.DONE
                    return self._done_action(navigation_success=False)

                handoff_distance = self._distance_to_pickup()
                return self._handoff_or_done_action(handoff_distance)
            return nav_action

        if self.current_phase == NavThenPickAndPlacePhase.FINAL_APPROACH:
            return self._final_approach_action()

        if self.current_phase == NavThenPickAndPlacePhase.PICK_AND_PLACE:
            try:
                manip_action = self.manip_policy.get_action(observation)
            except ValueError as exc:
                if "Max planning reattempts reached" not in str(exc):
                    raise

                log.warning(
                    "Pick-and-place planner failed after max reattempts; ending "
                    "rollout cleanly so the navigation and handoff attempt can be saved."
                )
                self.current_phase = NavThenPickAndPlacePhase.DONE
                return self._done_action(
                    navigation_success=self.navigation_succeeded,
                    handoff_success=True,
                    manipulation_success=False,
                    manipulation_error_code=1,
                )
            self.target_poses.update(getattr(self.manip_policy, "target_poses", {}))
            if manip_action.get("done", False):
                self.current_phase = NavThenPickAndPlacePhase.DONE
                return self._done_action()
            return manip_action

        return self._done_action()
