from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import RBY1GoProD455CameraSystem
from molmo_spaces.configs.policy_configs import (
    DoorOpeningPolicyConfig,
    NavThenDoorOpeningPolicyConfig,
)
from molmo_spaces.configs.robot_configs import RBY1MConfig
from molmo_spaces.configs.task_configs import DoorOpeningTaskConfig
from molmo_spaces.configs.task_sampler_configs import (
    DoorOpeningTaskSamplerConfig,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import (
    ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR,
    ASSETS_DIR,
    get_robot_paths,
)
from molmo_spaces.tasks.opening_task_samplers import (
    DoorOpeningTaskSampler,
)
from molmo_spaces.tasks.nav_to_door_opening_task import (
    NavToDoorOpeningTask,
    NavToDoorOpeningTaskConfig,
)
from molmo_spaces.tasks.nav_to_door_opening_task_sampler import (
    NavToDoorOpeningTaskSampler,
    NavToDoorOpeningTaskSamplerConfig,
)
from molmo_spaces.tasks.opening_tasks import DoorOpeningTask
from molmo_spaces.policy.solvers.nav_then_door_opening_policy import (
    NavDoorAStarSmoothPlannerPolicy,
)
from molmo_spaces.utils.profiler_utils import Profiler


@register_config("DoorOpeningDataGenConfig")
class DoorOpeningDataGenConfig(MlSpacesExpConfig):
    """
    All-ProcTHOR variant for RBY1 door opening dataset generation.

    Iterates through multiple houses from the ProcTHOR dataset for large-scale data generation.
    This is the main config - use DoorOpeningDebugConfig for single-scene testing.
    """

    num_envs: int = 1  # Number of environments to run in each thread
    task_type: str = "door_open"
    use_passive_viewer: bool = False  # Launch passive viewer for rendering
    viewer_cam_dict: dict = {
        "camera": "robot_0/camera_follower"
    }  # Dictionary containing viewer camera parameters
    policy_dt_ms: float = 100.0  # Default policy time step
    ctrl_dt_ms: float = 20.0  # Default control time step
    sim_dt_ms: float = 4.0  # Default simulation time step
    task_horizon: int = 400  # Maximum number of steps per episode

    # --- Data generation settings ---
    num_workers: int = 1  # Number of parallel worker processes for data generation
    profile: bool = False  # Whether to profile the data generation pipeline
    profiler: Profiler | None = None  # Profiler()
    output_dir: Path = (
        ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR / "experiment_output"
    )  # Directory to save generated data
    use_wandb: bool = False  # Whether to use Weights & Biases logging
    wandb_name: str | None = None  # Weights & Biases run name
    wandb_project: str = "molmo-spaces-data-generation"  # Weights & Biases project name

    # --- ProcTHOR dataset configuration ---
    scene_dataset: str = "procthor-10k"  # Name of the scene dataset to load
    data_split: str = "train"  # Data split to use
    # Robot configuration (imported from robot_configs.py)
    robot_config: RBY1MConfig = RBY1MConfig()

    # Camera configuration (imported from camera_configs.py)
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()

    # Task sampler configuration (imported from task_sampler_configs.py)
    task_sampler_config: DoorOpeningTaskSamplerConfig = DoorOpeningTaskSamplerConfig(
        task_sampler_class=DoorOpeningTaskSampler
    )

    # Task configuration (imported from task_configs.py)
    task_config: DoorOpeningTaskConfig = DoorOpeningTaskConfig(task_cls=DoorOpeningTask)

    # Policy configuration (imported from policy_configs.py)
    # Will be initialized in model_post_init
    policy_config: DoorOpeningPolicyConfig | None = None

    def _init_policy_config(self) -> DoorOpeningPolicyConfig:
        """Initialize policy config with dynamically computed planner configs"""
        # Import GPU-requiring modules only when actually creating policy (requires GPU)
        from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig
        from molmo_spaces.policy.solvers.opening_solver import DoorOpeningPlannerPolicy

        # Setup curobo planner configs with current ctrl_dt_ms
        rby1m_path = get_robot_paths().get("rby1m")
        assert rby1m_path is not None, "RBY1M robot path not found"

        left_curobo_planner_config = CuroboPlannerConfig(
            curobo_robot_config_path=str(
                rby1m_path / "curobo_config" / "rby1m_left_arm_holobase.yml"
            ),
            urdf_path=str(rby1m_path / "curobo_config" / "urdf" / "model_holobase.urdf"),
            asset_root_path=str(rby1m_path / "curobo_config" / "urdf" / "meshes"),
            usd_robot_root=str(rby1m_path / "curobo_config"),
            collision_spheres_path=str(rby1m_path / "curobo_config" / "rby1m_holobase_spheres.yml"),
            interpolation_dt=self.ctrl_dt_ms / 1000.0,  # 1x control dt
        )
        right_curobo_planner_config = CuroboPlannerConfig(
            curobo_robot_config_path=str(
                rby1m_path / "curobo_config" / "rby1m_right_arm_holobase.yml"
            ),
            urdf_path=str(rby1m_path / "curobo_config" / "urdf" / "model_holobase.urdf"),
            asset_root_path=str(rby1m_path / "curobo_config" / "urdf" / "meshes"),
            usd_robot_root=str(rby1m_path / "curobo_config"),
            collision_spheres_path=str(rby1m_path / "curobo_config" / "rby1m_holobase_spheres.yml"),
            interpolation_dt=self.ctrl_dt_ms / 1000.0,  # 1x control dt
        )

        return DoorOpeningPolicyConfig(
            policy_cls=DoorOpeningPlannerPolicy,
            left_curobo_planner_config=left_curobo_planner_config,
            right_curobo_planner_config=right_curobo_planner_config,
        )

    def model_post_init(self, __context) -> None:
        """Initialize policy config after Pydantic model initialization"""
        super().model_post_init(__context)
        # Set up policy config with dynamically computed planner configs
        # Skip if no GPU available (e.g., when launching jobs from manager)
        try:
            self.policy_config = self._init_policy_config()
        except RuntimeError as e:
            # Check if this is a CUDA/GPU-related error
            error_msg = str(e)
            if "NVIDIA" in error_msg or "CUDA" in error_msg or "GPU" in error_msg:
                # No GPU available - this is expected on manager nodes that just coordinate jobs
                # Policy config will be initialized later on worker nodes that have GPUs
                print(
                    f"Warning: Skipping policy config initialization due to missing GPU: {error_msg}"
                )
                self.policy_config = None
            else:
                raise

        # Auto-create profiler instance if profiling is enabled
        if self.profile and self.profiler is None:
            self.profiler = Profiler()

    @property
    def tag(self) -> str:
        return "rby1_door_opening_all_procthor"


@register_config("DoorOpeningDebugConfig")
class DoorOpeningDebugConfig(DoorOpeningDataGenConfig):
    """
    Debug config for door opening dataset generation.
    """

    num_workers: int = 1
    use_passive_viewer: bool = True
    filter_for_successful_trajectories: bool = False
    seed: int | None = 83067780
    policy_dt_ms: float = 100.0
    output_dir: Path = (
        ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR / "experiment_output" / "door_opening_debug"
    )
    task_horizon: int = 1000

    task_sampler_config: DoorOpeningTaskSamplerConfig = DoorOpeningTaskSamplerConfig(
        task_sampler_class=DoorOpeningTaskSampler,
        samples_per_house=1,
        house_inds=[22],
    )

    def tag(self) -> str:
        return "rby1_door_opening_debug"


@register_config("DoorOpeningNoViewerDebugConfig")
class DoorOpeningNoViewerDebugConfig(DoorOpeningDebugConfig):
    """Door opening debug config without the passive MuJoCo viewer."""

    use_passive_viewer: bool = False

    def tag(self) -> str:
        return "rby1_door_opening_no_viewer_debug"


@register_config("RBY1NavDoorOpeningDataGenConfig")
class RBY1NavDoorOpeningDataGenConfig(DoorOpeningDataGenConfig):
    """Prototype long-horizon RBY1 navigation followed by door opening."""

    task_type: str = "nav_to_door_opening"
    grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "none"
    task_horizon: int = 1200
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "rby1_nav_door_opening_v1"
    )
    policy_config: NavThenDoorOpeningPolicyConfig | None = None
    task_sampler_config: NavToDoorOpeningTaskSamplerConfig = (
        NavToDoorOpeningTaskSamplerConfig(
            task_sampler_class=NavToDoorOpeningTaskSampler
        )
    )
    task_config: NavToDoorOpeningTaskConfig = NavToDoorOpeningTaskConfig(
        task_cls=NavToDoorOpeningTask
    )

    def _init_policy_config(self) -> NavThenDoorOpeningPolicyConfig:
        opening_policy_config = DoorOpeningDataGenConfig._init_policy_config(self)
        opening_policy_config.max_steps_per_waypoint = 30
        nav_policy_config = NavThenDoorOpeningPolicyConfig().nav_policy_config
        nav_policy_config.path_interpolation_density = 0
        nav_policy_config.path_max_inter_waypoint_dist = 0.5
        nav_policy_config.intermediate_waypoint_xy_threshold_m = 0.25
        nav_policy_config.path_min_dist_to_target_center = 0.75
        nav_policy_config.plan_max_retries = 3
        nav_policy_config.plan_fail_after_waypoint_steps = 20
        nav_policy_config.planner_config.agent_radius = 0.5
        nav_policy_config.policy_cls = NavDoorAStarSmoothPlannerPolicy

        return NavThenDoorOpeningPolicyConfig(
            opening_policy_config=opening_policy_config,
            nav_policy_config=nav_policy_config,
        )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        grounding_mode = os.environ.get(
            "RBY1_NAV_DOOR_GROUNDING_MODE", self.grounding_mode
        )
        if grounding_mode not in {"none", "visible_unique", "point_prompt"}:
            raise ValueError(f"Invalid nav-door grounding mode: {grounding_mode}")
        self.grounding_mode = grounding_mode
        self.task_config.task_cls = NavToDoorOpeningTask
        self.task_sampler_config.task_sampler_class = NavToDoorOpeningTaskSampler
        self.task_sampler_config.target_grounding_mode = grounding_mode
        self.task_sampler_config.robot_safety_radius = 0.35
        self.task_sampler_config.base_pose_sampling_radius_range = (4.0, 12.0)
        self.task_sampler_config.max_robot_placement_attempts = 25

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_datagen"


@register_config("RBY1NavDoorOpeningDebugConfig")
class RBY1NavDoorOpeningDebugConfig(RBY1NavDoorOpeningDataGenConfig):
    """One-episode debug config for nav-to-door-opening validation."""

    num_workers: int = 1
    use_passive_viewer: bool = False
    filter_for_successful_trajectories: bool = False
    seed: int | None = 83067780
    policy_dt_ms: float = 100.0
    task_horizon: int = 1200
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "rby1_nav_door_opening_debug"
    )
    task_sampler_config: NavToDoorOpeningTaskSamplerConfig = (
        NavToDoorOpeningTaskSamplerConfig(
        task_sampler_class=NavToDoorOpeningTaskSampler,
        samples_per_house=1,
        house_inds=[22],
        base_pose_sampling_radius_range=(1.5, 3.0),
        robot_safety_radius=0.35,
        max_robot_placement_attempts=25,
        )
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (1.5, 3.0)
        self.task_sampler_config.robot_safety_radius = 0.35
        self.task_sampler_config.max_robot_placement_attempts = 25

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_debug"


@register_config("RBY1NavDoorOpeningSuccessSearchConfig")
class RBY1NavDoorOpeningSuccessSearchConfig(RBY1NavDoorOpeningDataGenConfig):
    """Small sweep that searches for successful nav-to-door-opening rollouts."""

    num_workers: int = 1
    use_passive_viewer: bool = False
    filter_for_successful_trajectories: bool = True
    seed: int | None = None
    policy_dt_ms: float = 100.0
    task_horizon: int = 1200
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_success_search"
    )
    task_sampler_config: NavToDoorOpeningTaskSamplerConfig = (
        NavToDoorOpeningTaskSamplerConfig(
        task_sampler_class=NavToDoorOpeningTaskSampler,
        samples_per_house=1,
        house_inds=[0, 1, 2, 3, 4, 5, 6, 7, 22],
        base_pose_sampling_radius_range=(0.8, 2.0),
        robot_safety_radius=0.35,
        max_robot_placement_attempts=35,
        max_total_attempts_multiplier=4,
        check_robot_placement_visibility=False,
        )
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (0.8, 2.0)
        self.task_sampler_config.robot_safety_radius = 0.35
        self.task_sampler_config.max_robot_placement_attempts = 35
        self.task_sampler_config.max_total_attempts_multiplier = 4
        self.task_sampler_config.check_robot_placement_visibility = False

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_success_search"


@register_config("RBY1NavDoorOpeningFarStandoffSearchConfig")
class RBY1NavDoorOpeningFarStandoffSearchConfig(RBY1NavDoorOpeningSuccessSearchConfig):
    """Success search variant that hands off farther from the door handle."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_far_standoff_search"
    )

    def _init_policy_config(self) -> NavThenDoorOpeningPolicyConfig:
        policy_config = super()._init_policy_config()
        policy_config.handoff_max_distance_to_handle_m = 1.45
        policy_config.handoff_after_nav_failure_max_distance_to_handle_m = 1.6
        policy_config.nav_policy_config.path_min_dist_to_target_center = 1.25
        return policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (1.2, 3.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_far_standoff_search"


@register_config("RBY1NavDoorOpeningLongDistanceSearchConfig")
class RBY1NavDoorOpeningLongDistanceSearchConfig(RBY1NavDoorOpeningSuccessSearchConfig):
    """Long-distance success search for nav-to-door-opening rollouts."""

    task_horizon: int = 1600
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_long_distance_search"
    )

    def _init_policy_config(self) -> NavThenDoorOpeningPolicyConfig:
        policy_config = super()._init_policy_config()
        policy_config.handoff_max_distance_to_handle_m = 0.95
        policy_config.handoff_after_nav_failure_max_distance_to_handle_m = 1.15
        policy_config.final_align_target_distance_to_handle_m = 0.85
        policy_config.final_align_max_steps = 60
        policy_config.nav_policy_config.path_min_dist_to_target_center = 0.95
        policy_config.nav_policy_config.plan_max_retries = 5
        policy_config.nav_policy_config.plan_fail_after_waypoint_steps = 30
        return policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (4.0, 12.0)
        self.task_sampler_config.max_total_attempts_multiplier = 3
        self.task_sampler_config.house_inds = [1, 4, 7, 22]

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_long_distance_search"


@register_config("RBY1NavDoorOpeningHandoffSmokeConfig")
class RBY1NavDoorOpeningHandoffSmokeConfig(RBY1NavDoorOpeningDataGenConfig):
    """Small end-to-end smoke test for navigation handoff and final alignment."""

    num_workers: int = 1
    use_passive_viewer: bool = False
    filter_for_successful_trajectories: bool = False
    seed: int | None = 83067780
    policy_dt_ms: float = 100.0
    task_horizon: int = 1200
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_handoff_smoke"
    )
    task_sampler_config: NavToDoorOpeningTaskSamplerConfig = (
        NavToDoorOpeningTaskSamplerConfig(
        task_sampler_class=NavToDoorOpeningTaskSampler,
        samples_per_house=1,
        house_inds=[22],
        base_pose_sampling_radius_range=(1.5, 3.0),
        robot_safety_radius=0.35,
        max_robot_placement_attempts=25,
        check_robot_placement_visibility=False,
        )
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (1.5, 3.0)
        self.task_sampler_config.house_inds = [22]
        self.task_sampler_config.max_total_attempts_multiplier = 1
        self.task_sampler_config.check_robot_placement_visibility = False

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_handoff_smoke"


@register_config("RBY1NavDoorOpeningVisibleGroundingSmokeConfig")
class RBY1NavDoorOpeningVisibleGroundingSmokeConfig(
    RBY1NavDoorOpeningHandoffSmokeConfig
):
    """Stage-1 smoke test with a unique visible target door."""

    grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "visible_unique"
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_visible_grounding_smoke"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        sampler_config = self.task_sampler_config
        sampler_config.target_visibility_camera = "head_camera"
        sampler_config.target_door_min_visibility_fraction = 0.0001
        sampler_config.target_handle_min_visibility_fraction = 0.00001
        sampler_config.competing_door_min_visibility_fraction = 0.0001
        sampler_config.max_robot_placement_attempts = 50

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_visible_grounding_smoke"


@register_config("RBY1NavDoorOpeningPointPromptGroundingSmokeConfig")
class RBY1NavDoorOpeningPointPromptGroundingSmokeConfig(
    RBY1NavDoorOpeningVisibleGroundingSmokeConfig
):
    """Stage-2 smoke test with a visible target-handle point prompt."""

    grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "point_prompt"
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_point_prompt_grounding_smoke"
    )

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_point_prompt_grounding_smoke"


class RBY1NavDoorOpeningDistanceSweepBaseConfig(RBY1NavDoorOpeningDataGenConfig):
    """Base config for small distance-specific nav-to-door-opening sweeps."""

    num_workers: int = 1
    use_passive_viewer: bool = False
    filter_for_successful_trajectories: bool = True
    seed: int | None = None
    policy_dt_ms: float = 100.0
    task_horizon: int = 1400
    task_sampler_config: NavToDoorOpeningTaskSamplerConfig = (
        NavToDoorOpeningTaskSamplerConfig(
        task_sampler_class=NavToDoorOpeningTaskSampler,
        samples_per_house=1,
        house_inds=[1, 4, 7, 22],
        base_pose_sampling_radius_range=(0.8, 2.0),
        robot_safety_radius=0.35,
        max_robot_placement_attempts=35,
        max_total_attempts_multiplier=3,
        check_robot_placement_visibility=False,
        )
    )

    def _init_policy_config(self) -> NavThenDoorOpeningPolicyConfig:
        policy_config = super()._init_policy_config()
        policy_config.handoff_max_distance_to_handle_m = 0.95
        policy_config.handoff_after_nav_failure_max_distance_to_handle_m = 1.15
        policy_config.final_align_target_distance_to_handle_m = 0.85
        policy_config.final_align_max_steps = 60
        policy_config.nav_policy_config.path_min_dist_to_target_center = 0.95
        policy_config.nav_policy_config.plan_max_retries = 5
        policy_config.nav_policy_config.plan_fail_after_waypoint_steps = 30
        return policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.house_inds = [1, 4, 7, 22]
        self.task_sampler_config.max_robot_placement_attempts = 35
        self.task_sampler_config.max_total_attempts_multiplier = 3
        self.task_sampler_config.check_robot_placement_visibility = False


@register_config("RBY1NavDoorOpeningShortDistanceSweepConfig")
class RBY1NavDoorOpeningShortDistanceSweepConfig(RBY1NavDoorOpeningDistanceSweepBaseConfig):
    """Distance sweep with 0.8-2.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_short"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (0.8, 2.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_short"


@register_config("RBY1NavDoorOpeningMediumDistanceSweepConfig")
class RBY1NavDoorOpeningMediumDistanceSweepConfig(RBY1NavDoorOpeningDistanceSweepBaseConfig):
    """Distance sweep with 2.0-4.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_medium"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (2.0, 4.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_medium"


@register_config("RBY1NavDoorOpeningLongDistanceSweepConfig")
class RBY1NavDoorOpeningLongDistanceSweepConfig(RBY1NavDoorOpeningDistanceSweepBaseConfig):
    """Distance sweep with 4.0-8.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_long"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (4.0, 8.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_long"


class RBY1NavDoorOpeningBalancedSweepBaseConfig(RBY1NavDoorOpeningDistanceSweepBaseConfig):
    """Balanced handoff variant that prioritizes orientation over close standoff."""

    def _init_policy_config(self) -> NavThenDoorOpeningPolicyConfig:
        policy_config = super()._init_policy_config()
        policy_config.handoff_max_distance_to_handle_m = 1.25
        policy_config.handoff_after_nav_failure_max_distance_to_handle_m = 1.4
        policy_config.final_align_target_distance_to_handle_m = 1.1
        policy_config.final_align_yaw_threshold_rad = float(np.deg2rad(15))
        policy_config.nav_policy_config.path_min_dist_to_target_center = 1.1
        return policy_config


@register_config("RBY1NavDoorOpeningMediumDistanceBalancedSweepConfig")
class RBY1NavDoorOpeningMediumDistanceBalancedSweepConfig(
    RBY1NavDoorOpeningBalancedSweepBaseConfig
):
    """Balanced-handoff sweep with 2.0-4.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_medium_balanced"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (2.0, 4.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_medium_balanced"


@register_config("RBY1NavDoorOpeningLongDistanceBalancedSweepConfig")
class RBY1NavDoorOpeningLongDistanceBalancedSweepConfig(
    RBY1NavDoorOpeningBalancedSweepBaseConfig
):
    """Balanced-handoff sweep with 4.0-8.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_long_balanced"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (4.0, 8.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_long_balanced"


@register_config("RBY1NavDoorOpeningSimpleVisibleDataGenConfig")
class RBY1NavDoorOpeningSimpleVisibleDataGenConfig(
    RBY1NavDoorOpeningBalancedSweepBaseConfig
):
    """Focused dataset with one clearly visible door and 0.8-3.0m starts."""

    grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "visible_unique"
    task_horizon: int = 1200
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_simple_visible"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        sampler_config = self.task_sampler_config
        sampler_config.base_pose_sampling_radius_range = (0.8, 3.0)
        sampler_config.samples_per_house = 2
        sampler_config.max_total_attempts_multiplier = 4
        sampler_config.max_robot_placement_attempts = 50
        sampler_config.target_visibility_camera = "head_camera"
        sampler_config.target_door_min_visibility_fraction = 0.001
        sampler_config.competing_door_min_visibility_fraction = 0.001

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_simple_visible"


@register_config("RBY1NavDoorOpeningSimpleVisibleSmokeConfig")
class RBY1NavDoorOpeningSimpleVisibleSmokeConfig(
    RBY1NavDoorOpeningSimpleVisibleDataGenConfig
):
    """One-episode smoke test for the simplified visible-door task."""

    filter_for_successful_trajectories: bool = False
    seed: int | None = 83067780
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_simple_visible_smoke"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        sampler_config = self.task_sampler_config
        sampler_config.house_inds = [22]
        sampler_config.samples_per_house = 1
        sampler_config.max_total_attempts_multiplier = 1

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_simple_visible_smoke"


class RBY1NavDoorOpeningSimpleVisibleProductionBaseConfig(
    RBY1NavDoorOpeningSimpleVisibleDataGenConfig
):
    """Production settings for collecting successful simple visible-door trajectories."""

    filter_for_successful_trajectories: bool = True
    seed: int | None = None
    task_horizon: int = 1200

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        sampler_config = self.task_sampler_config
        sampler_config.samples_per_house = 3
        sampler_config.max_total_attempts_multiplier = 6
        sampler_config.max_robot_placement_attempts = 75


@register_config("RBY1NavDoorOpeningSimpleVisibleProductionAConfig")
class RBY1NavDoorOpeningSimpleVisibleProductionAConfig(
    RBY1NavDoorOpeningSimpleVisibleProductionBaseConfig
):
    """Production shard A covering ProcTHOR houses 0-10."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_simple_visible_production_a"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.house_inds = list(range(0, 11))

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_simple_visible_production_a"


@register_config("RBY1NavDoorOpeningSimpleVisibleProductionBConfig")
class RBY1NavDoorOpeningSimpleVisibleProductionBConfig(
    RBY1NavDoorOpeningSimpleVisibleProductionBaseConfig
):
    """Production shard B covering ProcTHOR houses 11-22."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_simple_visible_production_b"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.house_inds = list(range(11, 23))

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_simple_visible_production_b"


class RBY1NavDoorOpeningPointPromptSweepBaseConfig(
    RBY1NavDoorOpeningBalancedSweepBaseConfig
):
    """Balanced success-search sweep with visible target-handle point grounding."""

    grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "point_prompt"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        sampler_config = self.task_sampler_config
        sampler_config.target_visibility_camera = "head_camera"
        sampler_config.target_door_min_visibility_fraction = 0.0001
        sampler_config.target_handle_min_visibility_fraction = 0.00001
        sampler_config.max_robot_placement_attempts = 50


@register_config("RBY1NavDoorOpeningMediumDistancePointPromptSweepConfig")
class RBY1NavDoorOpeningMediumDistancePointPromptSweepConfig(
    RBY1NavDoorOpeningPointPromptSweepBaseConfig
):
    """Point-prompt success sweep with 2.0-4.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_point_prompt_medium"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (2.0, 4.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_point_prompt_medium"


@register_config("RBY1NavDoorOpeningLongDistancePointPromptSweepConfig")
class RBY1NavDoorOpeningLongDistancePointPromptSweepConfig(
    RBY1NavDoorOpeningPointPromptSweepBaseConfig
):
    """Point-prompt success sweep with 4.0-8.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_point_prompt_long"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (4.0, 8.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_point_prompt_long"


@register_config("RBY1NavDoorOpeningVeryLongDistanceSweepConfig")
class RBY1NavDoorOpeningVeryLongDistanceSweepConfig(RBY1NavDoorOpeningDistanceSweepBaseConfig):
    """Distance sweep with 8.0-12.0m starts."""

    task_horizon: int = 1800
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_very_long"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (8.0, 12.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_very_long"
