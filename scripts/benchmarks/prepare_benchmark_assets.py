"""Install scene/object/grasp assets for selected JSON benchmark episodes.

Example:
    python scripts/benchmarks/prepare_benchmark_assets.py \
        --benchmark_dir ~/.cache/molmo-spaces-resources/benchmarks/.../pick_benchmark \
        --idx 0 1 2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec, load_all_episodes
from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MolmoSpaces assets for selected benchmark episodes."
    )
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        required=True,
        help="Benchmark directory containing benchmark.json.",
    )
    parser.add_argument(
        "--idx",
        type=int,
        nargs="+",
        required=True,
        help="One or more flat benchmark episode indices to prepare.",
    )
    parser.add_argument(
        "--variant",
        default="ceiling",
        choices=("base", "ceiling"),
        help="Scene XML variant to install. Eval currently loads the ceiling variant by default.",
    )
    parser.add_argument(
        "--grasp_source",
        default="droid_objaverse",
        help="Grasp source to install for Objaverse scene objects.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print selected episodes and scene XMLs without installing packages.",
    )
    return parser.parse_args()


def _scene_path_for_episode(episode: EpisodeSpec, variant: str) -> Path:
    scene_map = get_scenes(episode.scene_dataset, episode.data_split)
    split_map = scene_map[episode.data_split]

    if episode.house_index not in split_map:
        raise KeyError(
            f"House {episode.house_index} not found in "
            f"{episode.scene_dataset}/{episode.data_split} scene index."
        )

    scene_entry = split_map[episode.house_index]
    if isinstance(scene_entry, dict):
        scene_path = scene_entry.get(variant)
        if scene_path is None:
            raise KeyError(
                f"Variant {variant!r} not available for house {episode.house_index}. "
                f"Available variants: {sorted(k for k, v in scene_entry.items() if v is not None)}"
            )
    else:
        scene_path = scene_entry

    if scene_path is None:
        raise FileNotFoundError(
            f"No scene XML for house {episode.house_index} in "
            f"{episode.scene_dataset}/{episode.data_split}."
        )

    return Path(scene_path)


def _selected_episodes(episodes: list[EpisodeSpec], indices: list[int]) -> list[tuple[int, EpisodeSpec]]:
    selected = []
    for idx in indices:
        if idx < 0 or idx >= len(episodes):
            raise IndexError(
                f"Episode index {idx} is out of range. "
                f"Benchmark has {len(episodes)} episodes (0-{len(episodes) - 1})."
            )
        selected.append((idx, episodes[idx]))
    return selected


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()

    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    episodes = load_all_episodes(benchmark_dir)
    selected = _selected_episodes(episodes, args.idx)

    log.info("Loaded %d episodes from %s", len(episodes), benchmark_dir)
    log.info("Preparing %d selected episodes: %s", len(selected), args.idx)

    seen_scene_paths: set[Path] = set()
    for idx, episode in selected:
        scene_path = _scene_path_for_episode(episode, args.variant)
        log.info(
            "idx=%d house=%d dataset=%s split=%s scene=%s",
            idx,
            episode.house_index,
            episode.scene_dataset,
            episode.data_split,
            scene_path,
        )

        if scene_path in seen_scene_paths:
            log.info("Scene already handled in this run, skipping duplicate: %s", scene_path)
            continue
        seen_scene_paths.add(scene_path)

        if args.dry_run:
            continue

        installed = install_scene_with_objects_and_grasps_from_path(
            scene_path,
            grasp_sources=(args.grasp_source,),
            exclude_thor=True,
        )
        log.info("Installed or verified packages for %s: %s", scene_path, installed)

    if args.dry_run:
        log.info("Dry run complete. No packages were installed.")
    else:
        log.info("Asset preparation complete.")


if __name__ == "__main__":
    main()
