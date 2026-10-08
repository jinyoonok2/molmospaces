# MolmoSpaces fork changes and branch history

This fork preserves the RB-Y1 compatibility fixes needed by our MolmoBot stack. The retained project branch is `rby1-compatible`. Its simulation code is the final state used for the navigation-to-door dataset, training, and evaluation; custom long-horizon behavior now lives in the parent project's `lrl_molmospaces_extensions/` package.

The comparison baseline is the original MolmoBot release, `cd23becebcf72dd93a4aa5872a60802d5eff03ef`, preserved on this fork's `main`. This is a pinned upstream release, not the latest AllenAI main. This record covers changes through `fabf2af3839ca59ac35913271379fb97cd60914e`, before adding this documentation.

## Branch ancestry

```text
main at cd23bec
  f5168f3  Simulation bring-up helpers
  30ccf7e  Terminate-on-success evaluation override
  abcc194  Planner evaluation robustness
  9b5f4e1  RB-Y1 benchmark and contact-scoring fixes
           Former rby1-custom tip; long-horizon branch created here
  dd7d55e  Long-horizon generation added inside MolmoSpaces
  7278de7  Visible-door generation simplified
  fabf2af  Core restored after custom behavior moved to parent extensions
           Retained rby1-compatible implementation
```

Removing the redundant `rby1-custom` branch name preserves its commits: `9b5f4e1` is an ancestor of the retained branch. No history rewrite or code merge is required.

## Changes retained from the original release

Paths below are relative to this repository. The final implementation changes 15 files relative to `cd23bec`, with 377 inserted and 11 deleted lines, excluding the change record and its README link.

| Files | Retained change and purpose |
|---|---|
| `molmo_spaces/robots/robot_views/rby1_view.py` | Use integer MuJoCo body IDs for the gripper end effectors and holonomic base, rather than body-view objects. |
| `molmo_spaces/tasks/pick_task.py`, `molmo_spaces/tasks/pick_and_place_task.py` | Map the base body through MuJoCo's `body_rootid` before comparing contacts. The robot's nested base can differ from its kinematic-tree root; Pick and release scoring must use the actual root. |
| `molmo_spaces/evaluation/benchmark_schema.py` | Accept legacy single-joint start/goal positions stored as `[value]` by unwrapping them to scalars. |
| `molmo_spaces/tasks/json_eval_task_sampler.py` | Normalize legacy `mujoco_thor.tasks.*` class paths when importing and inferring task types. Add planner auxiliary objects, such as grasp-collision helpers, when constructing benchmark scenes. |
| `molmo_spaces/evaluation/eval_main.py` | Expose optional `terminate_upon_success` and `save_partial_trajectories_on_exception` overrides through `run_evaluation`. |
| `molmo_spaces/configs/abstract_exp_config.py`, `molmo_spaces/data_generation/pipeline.py` | Add an opt-in setting to retain collected observations from rollouts that abort with exceptions, recorded as unsuccessful trajectories. The default remains disabled. |
| `molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py` | Let RB-Y1 Pick/PnP launchers select local cuRobo with `RBY1_CUROBO_SERVER_URLS=local`, or supply comma-separated server URLs. An unset variable preserves existing configuration behavior. |
| `molmo_spaces/policy/solvers/object_manipulation/curobo_pick_and_place_planner_policy.py` | Import pickup-type constants from the shared constants module instead of the editor module. |
| `molmo_spaces/env/env.py` | Forward `MUJOCO_EGL_DEVICE_ID` to the classic renderer, defaulting to device 0, and log the renderer choice. |
| `molmo_spaces/utils/synset_utils.py` | Recognize existing extracted or zipped NLTK WordNet corpora before attempting downloads. |
| `molmo_spaces/data_generation/config/door_opening_configs.py` | Add a door-opening debug configuration with the passive viewer disabled. |
| `scripts/benchmarks/prepare_benchmark_assets.py` | Add a helper for installing scene, object, and grasp assets for selected JSON benchmark episodes. |
| `scripts/datagen/run_fast_teleop.py` | Add a configurable keyboard-teleoperation sandbox. |

The specific RB-Y1 benchmark/contact fix is [9b5f4e1](https://github.com/jinyoonok2/molmospaces/commit/9b5f4e1005b81bd685299dbb19a8c8c5a3e9d2e4). The rendering, asset, and planner changes are supporting bring-up work. Making the framework runnable and correcting its scoring does not by itself reproduce the published model's performance.

## Changes from custom to the final long-horizon state

Three commits followed the former `rby1-custom` tip:

| Commit | Historical change |
|---|---|
| [dd7d55e](https://github.com/jinyoonok2/molmospaces/commit/dd7d55e4c9f15e1df38c8d25fa99118ba33e5641) | Added navigation-then-door-opening and navigation-then-pick/place policies, tasks, samplers, configuration, and supporting navigation changes inside MolmoSpaces. |
| [7278de7](https://github.com/jinyoonok2/molmospaces/commit/7278de77b92e697070bba0fb644b88b3666e2afe) | Simplified visible-door generation and its task/sampler logic. |
| [fabf2af](https://github.com/jinyoonok2/molmospaces/commit/fabf2af3839ca59ac35913271379fb97cd60914e) | Removed the custom long-horizon modules from the simulation repository and restored modified core files after relocating the implementation into the parent project's extension package. |

After the last refactor, `git diff 9b5f4e1 fabf2af` is empty. Both commits have tree ID `893e22e9ef4836dd41b0cd635fa27cc1c6f69fb4`. Thus they have identical files despite different commit histories. The long-horizon branch retains the compatibility fixes and the full development history; its active long-horizon implementation requires the parent extension package.

## Where the project extensions live

In the parent `LRL_project` repository, `lrl_molmospaces_extensions/` contains composite tasks and samplers, navigation/manipulation policies, per-door rollout accounting, the unified baseline, point-prompt utilities, grasp-quality checks, dataset splitting/preparation, opening-speed-based rerendering, and learned-policy evaluation configuration. Training uses the separate MolmoBot submodule and the parent's launchers.

The final dataset, checkpoint, and comparison videos are project artifacts, rather than changes to the MolmoSpaces simulation code. Their paths and usage are maintained in the parent project's `docs/PROJECT_ENVIRONMENT.md` and extension README.

## Inspecting the record

These commands use immutable commits, so they continue to work after old branch names are removed:

```bash
# Retained implementation changes from the pinned original release.
git diff cd23bec fabf2af

# Compatibility and bring-up changes before long-horizon development.
git log --reverse --oneline cd23bec..9b5f4e1

# Development and relocation of the custom long-horizon implementation.
git log --reverse --oneline 9b5f4e1..fabf2af

# Empty output confirms identical final source trees.
git diff 9b5f4e1 fabf2af

# Documentation added during consolidation is visible separately.
git diff fabf2af HEAD
```

MolmoBot's corresponding compatibility branch is `rby1-compatible`, with implementation pinned at `9c2ebfa`. See [its change and reproduction record](https://github.com/jinyoonok2/MolmoBot/blob/rby1-compatible/docs/FORK_CHANGES.md). Branch names belong to their individual repositories; the parent project records submodule commit IDs independently.

The retained branch was renamed from `rby1-long-horizon-vla` to `rby1-compatible` on 2026-10-08 to match the MolmoBot fork. This rename preserves the final implementation and all historical commits.

## Reproduce the compatible source pair

Use both forks together in a fresh directory. The immutable commits below reproduce the implementation used by this project; later commits on `rby1-compatible` add change records and README links. Existing compatible checkouts already contain the fixes.

```bash
mkdir rby1-reproduction
cd rby1-reproduction

git clone --branch rby1-compatible https://github.com/jinyoonok2/MolmoBot.git
git clone --branch rby1-compatible https://github.com/jinyoonok2/molmospaces.git

git -C MolmoBot checkout --detach 9c2ebfa5cafcf13f9664e70703e25de38e1d5fcb
git -C molmospaces checkout --detach fabf2af3839ca59ac35913271379fb97cd60914e
```

Activate a separate Python 3.11 environment. The observed working environment used Python 3.11.15, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, MuJoCo 3.4.0, transformers 4.57.1, and pydantic 2.13.4. This is a version snapshot, not a complete dependency lock or a validated clean installation. Select CUDA packages appropriate to the host before installing the projects.

For package wiring, run from the directory containing both clones:

```bash
python -m pip install -e './molmospaces[mujoco]'
python -m pip install -e './MolmoBot/MolmoBot[eval,train]'
python -m pip install -e './molmospaces[mujoco]'
python -m pip check
```

MolmoBot's `eval` extra declares the original upstream MolmoSpaces commit `cd23bec` as a Git dependency. Installing it can replace the patched package; the final editable MolmoSpaces install restores the compatible fork. Diagnose dependency errors from `pip check` before running inference. Planner-based generation additionally requires cuRobo: the project's installed source was `87e857d46fa5398f268c7f31d26566351be8671d`, while MolmoSpaces' optional `curobo` extra declares `417c995647fcb173a2bc094d1284b2a4f4b000ad`. Preserve that distinction when rebuilding the planner environment.

Verify package locations without loading model checkpoints:

```bash
export RBY1_SOURCE_ROOT="$PWD"
export PYTHONPATH="$RBY1_SOURCE_ROOT/MolmoBot/MolmoBot:$RBY1_SOURCE_ROOT/molmospaces:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu
export RBY1_MODEL_ACTION_SEED=42
python -c 'import sys, olmo, molmo_spaces; print(sys.executable); print(olmo.__file__); print(molmo_spaces.__file__)'
```

`olmo` must resolve inside the MolmoBot clone and `molmo_spaces` inside the MolmoSpaces clone. Seed 42 controls model action sampling, not all physics or rendering variation. Use a GPU allocation for rendering/inference and make the released checkpoint, scene assets, benchmark JSON, object assets, and grasp resources available before evaluation.

## Verify a benchmark episode

The historical RB-Y1 benchmark-loading setup used `molmospaces-bench-v2/20260327`. Verify the actual resource directories before using it: benchmark versions and assets are separate from Git source revisions. Required task directories are:

| Task | Directory within the benchmark release |
|---|---|
| Pick | `procthor-objaverse/rby1_benchmarks/pick_benchmark` |
| Pick and place | `procthor-objaverse/rby1_benchmarks/pnp_benchmark` |
| Opening | `ithor/rby1_benchmarks/opening_benchmark` |
| Door opening | `procthor-10k/rby1_benchmarks/door_opening_benchmark` |

The following is a single door-opening benchmark example, run from the directory containing both clones. Replace the paths with existing checkpoint, benchmark, resource-cache, asset, and output locations. It evaluates the released policy on a standard benchmark; it does not evaluate the parent's custom navigation-to-door task.

```bash
export RBY1_CHECKPOINT_DIR=/absolute/path/to/MolmoBot-RBY1Multitask
export RBY1_BENCHMARK_DIR=/absolute/path/to/door_opening_benchmark
export RBY1_EVAL_OUTPUT=/absolute/path/to/new/evaluation-output
export MLSPACES_CACHE_DIR=/absolute/path/to/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/absolute/path/to/molmospaces/assets

python MolmoBot/MolmoBot/launch_scripts/run_eval.py \
  --checkpoint_path "$RBY1_CHECKPOINT_DIR" \
  --benchmark_path "$RBY1_BENCHMARK_DIR" \
  --eval_config_cls olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorEvalConfig \
  --idx 0 --task_horizon 400 --terminate_upon_success \
  --output_dir "$RBY1_EVAL_OUTPUT"
```

To preview required assets for that episode before installing them, run:

```bash
python molmospaces/scripts/benchmarks/prepare_benchmark_assets.py \
  --benchmark_dir "$RBY1_BENCHMARK_DIR" --idx 0 --dry_run
```

Remove `--dry_run` to prepare missing assets in the configured storage location. The helper loads benchmark metadata; a successful preview does not verify model inference.

Check episode success, recorded H5 trajectories, and videos. A clean process exit verifies execution, not task success or reproduction of published success rates. Training and evaluation on our navigation-to-door dataset additionally use the parent `LRL_project` extensions, configs, and launchers. See [the project setup guide](https://github.com/jinyoonok2/LRL_project/blob/rby1-long-horizon-vla-training/docs/SETUP.md) for storage, asset configuration, Slurm, and the separate extension boundary.

## Rebuild the retained patch on the original baseline

From the directory containing both clones, use an unused worktree directory. Applying the net diff reproduces the final retained simulation changes while excluding long-horizon edits that were subsequently relocated to the parent extensions.

```bash
git -C molmospaces diff --binary \
  cd23becebcf72dd93a4aa5872a60802d5eff03ef \
  fabf2af3839ca59ac35913271379fb97cd60914e > molmospaces-rby1.patch

git -C molmospaces worktree add --detach ../molmospaces-rebuilt \
  cd23becebcf72dd93a4aa5872a60802d5eff03ef
git -C molmospaces-rebuilt apply --check ../molmospaces-rby1.patch
git -C molmospaces-rebuilt apply ../molmospaces-rby1.patch
git -C molmospaces-rebuilt add -A
git -C molmospaces-rebuilt diff --cached --exit-code \
  fabf2af3839ca59ac35913271379fb97cd60914e
```

The last command must produce no diff and exit with status 0. It verifies the reconstructed tracked files against the final implementation. The worktree has the original commit plus staged patches; historical commit IDs are preserved only when using the fork directly.
