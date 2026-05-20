import argparse
import logging

from molmo_spaces.configs.policy_configs_baselines import TeleopPolicyConfig
from molmo_spaces.configs.robot_configs import FloatingRUMRobotConfig

from run_pipeline import MyRolloutRunner, get_output_dir, setup_config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast keyboard teleop sandbox.")
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot", type=str, default="rum")
    parser.add_argument("--task_type", type=str, default="pick")
    parser.add_argument("--scene_dataset", type=str, default="ithor")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--house_inds", type=int, default=1)
    parser.add_argument("--target_types", type=str, default=None)
    parser.add_argument("--samples_per_house", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--task_horizon", type=int, default=1000)
    parser.add_argument("--policy_dt_ms", type=float, default=40.0)
    parser.add_argument("--step_size", type=float, default=0.02)
    parser.add_argument("--rot_step", type=float, default=0.06)
    parser.add_argument("--img_width", type=int, default=480)
    parser.add_argument("--img_height", type=int, default=360)
    parser.add_argument("--run_name_prefix", type=str, default="fast_teleop")

    # Compatibility fields expected by setup_config/get_output_dir.
    parser.set_defaults(
        eval=None,
        config=None,
        policy="teleop",
        single_step=False,
        filter_for_successful_trajectories=False,
        randomize_lighting=False,
        randomize_textures=False,
        randomize_dynamics=False,
        randomize_scene=False,
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    exp_config = setup_config(args)
    exp_config.num_workers = 1
    exp_config.use_passive_viewer = args.viewer
    exp_config.policy_dt_ms = args.policy_dt_ms
    exp_config.task_horizon = args.task_horizon

    if args.robot == "rum":
        exp_config.robot_config = FloatingRUMRobotConfig()
        exp_config.robot_config.init_qpos_noise_range = None

    if exp_config.camera_config is not None:
        exp_config.camera_config.img_resolution = (args.img_width, args.img_height)

    exp_config.policy_config = TeleopPolicyConfig(
        step_size=args.step_size,
        rot_step=args.rot_step,
        pos_sensitivity=args.step_size,
        rot_sensitivity=args.rot_step,
    )
    policy = exp_config.policy_config.policy_cls(exp_config)

    exp_config.output_dir = get_output_dir(args, exp_config)
    exp_config.save_config()

    log.info(
        "Starting fast teleop: robot=%s, resolution=%sx%s, step_size=%.4f, rot_step=%.4f",
        args.robot,
        args.img_width,
        args.img_height,
        args.step_size,
        args.rot_step,
    )
    MyRolloutRunner(exp_config).run(preloaded_policy=policy)


if __name__ == "__main__":
    main()
