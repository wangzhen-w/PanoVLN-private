#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import ijson
except Exception:  # pragma: no cover
    ijson = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_RAW_ANNOTATIONS = (
    "/workspace/code_dir/a_property/dataset/general_VLN_data/ScaleVLN_total/annotations/"
    "R2R_scalevln_ft_aug_enc.json"
)
DEFAULT_EXISTING_SUBSET = (
    "/workspace/code_dir/a_property/dataset/janusvln_data/datasets/scalevln/"
    "scalevln_subset_150k.json.gz"
)
DEFAULT_CONNECTIVITY_DIR = (
    "/workspace/code_dir/a_property/dataset/general_VLN_data/ScaleVLN_total/connectivity"
)
DEFAULT_CONNECTIVITY_MP3D_DIR = (
    "/workspace/code_dir/a_property/dataset/general_VLN_data/ScaleVLN_total/connectivity_mp3d"
)
DEFAULT_OUTPUT_ROOT = "/workspace/code_dir/a_property/dataset/general_VLN_data/ScaleVLN_CE"
DEFAULT_SCENES_DIR = "/workspace/code_dir/a_property/dataset/janusvln_data/scene_datasets"
DEFAULT_CONFIG_PATH = (
    "/workspace/code_dir/VLN/config/vln_scalevln.yaml"
)
DEFAULT_REPO_ROOT = "/workspace/code_dir/VLN"
DEFAULT_DATASET_FILENAME = "scalevln_subset_150k.json.gz"
DEFAULT_DATASET_JSONL_FILENAME = "scalevln_subset_150k.jsonl"
DEFAULT_GT_FILENAME = "scalevln_subset_150k_gt.json.gz"
DEFAULT_GT_JSONL_FILENAME = "scalevln_subset_150k_gt.jsonl"
DEFAULT_RAW_TOTAL = 2891134

INSTRUCTION_VOCAB = {
    "word_list": [],
    "word2idx_dict": {},
    "stoi": {},
    "itos": [],
    "num_vocab": 0,
    "UNK_INDEX": 1,
    "PAD_INDEX": 0,
}

STOP_ACTION = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert discrete ScaleVLN paths to VLN-CE episodes and expert GT."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-subsets",
        help="Stream raw ScaleVLN annotations and write VLN-CE episode subsets.",
    )
    add_common_build_args(build_parser)

    gt_parser = subparsers.add_parser(
        "generate-gt",
        help="Generate VLN-CE GT actions/locations for one or more subsets.",
    )
    add_gt_args(gt_parser)

    full_parser = subparsers.add_parser(
        "full",
        help="Run subset building first, then generate GT for the requested subsets.",
    )
    add_common_build_args(full_parser)
    add_gt_args(
        full_parser,
        include_output_root=False,
        include_subset_prefix=False,
        include_goal_radius=False,
    )

    return parser.parse_args()


def add_common_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--raw-annotations",
        type=str,
        default=DEFAULT_RAW_ANNOTATIONS,
        help="Path to ScaleVLN_total raw annotation json.",
    )
    parser.add_argument(
        "--existing-subset",
        type=str,
        default=DEFAULT_EXISTING_SUBSET,
        help="Existing ScaleVLN_150k json.gz used for de-duplication.",
    )
    parser.add_argument(
        "--connectivity-dir",
        type=str,
        default=DEFAULT_CONNECTIVITY_DIR,
        help="Connectivity directory for HM3D-style scans.",
    )
    parser.add_argument(
        "--connectivity-mp3d-dir",
        type=str,
        default=DEFAULT_CONNECTIVITY_MP3D_DIR,
        help="Connectivity directory for MP3D-style scans.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root used to store all generated ScaleVLN_CE subsets.",
    )
    parser.add_argument(
        "--num-subsets",
        type=int,
        default=10,
        help="Number of 150k subsets to create.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=150000,
        help="Number of episodes per subset.",
    )
    parser.add_argument(
        "--goal-radius",
        type=float,
        default=0.3,
        help="Goal radius written into episodes and used for GT generation.",
    )
    parser.add_argument(
        "--subset-prefix",
        type=str,
        default="subset",
        help="Subset directory prefix under the output root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing subset directories/files.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50000,
        help="Print progress every N raw annotations while building subsets.",
    )
    parser.add_argument(
        "--raw-total",
        type=int,
        default=DEFAULT_RAW_TOTAL,
        help="Expected total raw annotations for the tqdm progress bar.",
    )


def add_gt_args(
    parser: argparse.ArgumentParser,
    *,
    include_output_root: bool = True,
    include_subset_prefix: bool = True,
    include_goal_radius: bool = True,
) -> None:
    if include_output_root:
        parser.add_argument(
            "--output-root",
            type=str,
            default=DEFAULT_OUTPUT_ROOT,
            help="Output root containing the generated subset directories.",
        )
    if include_subset_prefix:
        parser.add_argument(
            "--subset-prefix",
            type=str,
            default="subset",
            help="Subset directory prefix under the output root.",
        )
    parser.add_argument(
        "--subset-indices",
        type=int,
        nargs="*",
        default=None,
        help="Subset indices to process. Default: all discovered subsets.",
    )
    parser.add_argument(
        "--dataset-filename",
        type=str,
        default=DEFAULT_DATASET_FILENAME,
        help="Episode dataset filename inside each subset directory.",
    )
    parser.add_argument(
        "--gt-filename",
        type=str,
        default=DEFAULT_GT_FILENAME,
        help="Final GT json.gz filename inside each subset directory.",
    )
    parser.add_argument(
        "--gt-jsonl-filename",
        type=str,
        default=DEFAULT_GT_JSONL_FILENAME,
        help="Intermediate resumable GT jsonl filename inside each subset directory.",
    )
    parser.add_argument(
        "--scenes-dir",
        type=str,
        default=DEFAULT_SCENES_DIR,
        help="Scene dataset root joined with episode.scene_id.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Habitat config used to instantiate the environment.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=DEFAULT_REPO_ROOT,
        help="Repo root that contains habitat_extensions.",
    )
    if include_goal_radius:
        parser.add_argument(
            "--goal-radius",
            type=float,
            default=0.3,
            help="ShortestPathFollower success radius.",
        )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=500,
        help="Habitat max_episode_steps.",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
        help="GPU device id passed to habitat_sim.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the intermediate GT jsonl file if it already exists.",
    )
    parser.add_argument(
        "--keep-jsonl",
        action="store_true",
        help="Keep the intermediate GT jsonl after finalizing json.gz.",
    )
    parser.add_argument(
        "--skip-finalize",
        action="store_true",
        help="Only append to jsonl, do not finalize to json.gz.",
    )
    parser.add_argument(
        "--sort-by-scene",
        action="store_true",
        help="Process episodes grouped by scene to reduce scene reloads.",
    )
    parser.add_argument(
        "--minimal-observations",
        action="store_true",
        help="Disable rendering sensors for faster GT generation.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Shard rank used together with --world-size.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of shards used together with --rank.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Optional episode start index inside the selected subset(s).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional limit on processed episodes per subset.",
    )
    parser.add_argument(
        "--gt-log-every",
        type=int,
        default=200,
        help="Print GT progress every N completed episodes.",
    )


def iter_raw_annotations(path: str) -> Iterator[Dict]:
    if ijson is None:
        raise RuntimeError(
            "The ScaleVLN converter needs the optional `ijson` package to "
            "stream the raw annotation JSON. Install ijson before running "
            "`build-subsets` or `full`."
        )
    with open(path, "r", encoding="utf-8") as handle:
        yield from ijson.items(handle, "item")


def normalize_raw_path_id(path_id: str) -> str:
    return path_id[len("scalevln_") :] if path_id.startswith("scalevln_") else path_id


def load_existing_path_ids(existing_subset_path: str) -> Set[str]:
    with gzip.open(existing_subset_path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    episodes = data["episodes"] if isinstance(data, dict) else data
    return {
        normalize_raw_path_id(str(episode["trajectory_id"]))
        for episode in episodes
    }


def scan_kind(scan: str) -> str:
    return "hm3d" if "-" in scan else "mp3d"


def pose_to_position(pose: Sequence[float]) -> List[float]:
    return [float(pose[3]), float(pose[11]), float(-pose[7])]


def heading_to_quaternion(heading: float) -> List[float]:
    return [0.0, math.sin(heading / 2.0), 0.0, math.cos(heading / 2.0)]


def build_scene_id(scan: str) -> str:
    if scan_kind(scan) == "hm3d":
        scene_name = scan.split("-", 1)[1]
        return f"hm3d/{scan}/{scene_name}.basis.glb"
    return f"mp3d/{scan}/{scan}.glb"


class ConnectivityLookup:
    def __init__(self, connectivity_dir: str, connectivity_mp3d_dir: str):
        self.connectivity_dir = connectivity_dir
        self.connectivity_mp3d_dir = connectivity_mp3d_dir

    @lru_cache(maxsize=512)
    def viewpoint_positions(self, scan: str) -> Dict[str, List[float]]:
        if scan_kind(scan) == "mp3d":
            candidates = [
                os.path.join(
                    self.connectivity_mp3d_dir, f"{scan}_connectivity.json"
                ),
                os.path.join(self.connectivity_dir, f"{scan}_connectivity.json"),
            ]
        else:
            candidates = [
                os.path.join(self.connectivity_dir, f"{scan}_connectivity.json"),
                os.path.join(
                    self.connectivity_mp3d_dir, f"{scan}_connectivity.json"
                ),
            ]

        path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
        if path is None:
            raise FileNotFoundError(
                f"Missing connectivity file for scan={scan}. Candidates={candidates}"
            )

        with open(path, "r", encoding="utf-8") as handle:
            nodes = json.load(handle)

        positions = {}
        for node in nodes:
            if not node.get("included", False):
                continue
            positions[str(node["image_id"])] = pose_to_position(node["pose"])
        return positions


def raw_item_to_episode(
    item: Dict,
    episode_id: int,
    goal_radius: float,
    connectivity_lookup: ConnectivityLookup,
) -> Dict:
    scan = str(item["scan"])
    path_id = normalize_raw_path_id(str(item["path_id"]))
    discrete_path = [str(viewpoint_id) for viewpoint_id in item["path"]]
    if len(discrete_path) < 2:
        raise ValueError(f"Path too short for {path_id}: {discrete_path}")

    positions_by_viewpoint = connectivity_lookup.viewpoint_positions(scan)
    reference_path = [positions_by_viewpoint[viewpoint_id] for viewpoint_id in discrete_path]

    instruction_list = item.get("instructions") or []
    if not instruction_list:
        raise ValueError(f"Missing instructions for {path_id}")

    return {
        "episode_id": int(episode_id),
        "trajectory_id": f"scalevln_{path_id}",
        "scene_id": build_scene_id(scan),
        "start_position": reference_path[0],
        "start_rotation": heading_to_quaternion(float(item["heading"])),
        "info": {"geodesic_distance": None},
        "goals": [
            {
                "position": reference_path[-1],
                "radius": float(goal_radius),
            }
        ],
        "instruction": {
            "instruction_text": str(instruction_list[0]),
            "instruction_tokens": None,
        },
        "reference_path": reference_path,
    }


def subset_dir(output_root: str, subset_prefix: str, subset_index: int, width: int) -> str:
    return os.path.join(output_root, f"{subset_prefix}_{subset_index:0{width}d}")


def subset_jsonl_path(subset_directory: str) -> str:
    return os.path.join(subset_directory, DEFAULT_DATASET_JSONL_FILENAME)


def atomic_write_json_gz(path: str, payload: Dict) -> None:
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(tmp_path, path)


def make_tqdm(*, total: Optional[int], desc: str, unit: str):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=True)


def update_tqdm(progress, n: int = 1, **postfix) -> None:
    if progress is None:
        return
    if n:
        progress.update(n)
    if postfix:
        progress.set_postfix(postfix, refresh=False)


def close_tqdm(progress) -> None:
    if progress is not None:
        progress.close()


def append_jsonl(path: str, row: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def finalize_dataset_jsonl(jsonl_path: str, output_gz_path: str) -> None:
    tmp_path = f"{output_gz_path}.tmp"
    with open(jsonl_path, "r", encoding="utf-8") as src, gzip.open(
        tmp_path, "wt", encoding="utf-8"
    ) as dst:
        dst.write('{"episodes":[')
        first = True
        for line in src:
            if not line.strip():
                continue
            if not first:
                dst.write(",")
            dst.write(line.strip())
            first = False
        dst.write('],"instruction_vocab":')
        dst.write(json.dumps(INSTRUCTION_VOCAB, ensure_ascii=False))
        dst.write("}")
    os.replace(tmp_path, output_gz_path)


def stable_scene_offset(scene_id: str, num_subsets: int) -> int:
    return int(hashlib.md5(scene_id.encode("utf-8")).hexdigest(), 16) % num_subsets


def choose_target_subset(
    scene_id: str,
    subset_counts: Sequence[int],
    subset_size: int,
    scene_seen_counts: Dict[str, int],
    num_subsets: int,
) -> Optional[int]:
    offset = stable_scene_offset(scene_id, num_subsets)
    seen = scene_seen_counts.get(scene_id, 0)
    for step in range(num_subsets):
        subset_index = (offset + seen + step) % num_subsets
        if subset_counts[subset_index] < subset_size:
            return subset_index
    return None


def write_subset_manifest(
    subset_directory: str,
    subset_index: int,
    goal_radius: float,
    raw_count_seen: int,
    num_episodes: int,
    first_episode_id: Optional[int],
    last_episode_id: Optional[int],
    first_trajectory_id: Optional[str],
    last_trajectory_id: Optional[str],
    unique_scenes: int,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
) -> None:
    os.makedirs(subset_directory, exist_ok=True)
    manifest_path = os.path.join(subset_directory, "manifest.json")
    manifest = {
        "subset_index": subset_index,
        "num_episodes": num_episodes,
        "unique_scenes": unique_scenes,
        "goal_radius": goal_radius,
        "first_episode_id": first_episode_id,
        "last_episode_id": last_episode_id,
        "first_trajectory_id": first_trajectory_id,
        "last_trajectory_id": last_trajectory_id,
        "raw_annotations_seen": raw_count_seen,
        "dataset_filename": dataset_filename,
        "gt_filename": DEFAULT_GT_FILENAME,
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def build_subsets(args: argparse.Namespace) -> List[str]:
    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)

    excluded_path_ids = load_existing_path_ids(args.existing_subset)
    connectivity_lookup = ConnectivityLookup(
        args.connectivity_dir, args.connectivity_mp3d_dir
    )

    subset_paths: List[str] = []
    width = max(2, len(str(args.num_subsets - 1)))
    target_total = args.num_subsets * args.subset_size
    selected_total = 0
    raw_seen = 0
    subset_counts = [0 for _ in range(args.num_subsets)]
    scene_seen_counts: Dict[str, int] = {}
    subset_scene_sets = [set() for _ in range(args.num_subsets)]
    subset_first_episode_id: List[Optional[int]] = [None for _ in range(args.num_subsets)]
    subset_first_trajectory_id: List[Optional[str]] = [None for _ in range(args.num_subsets)]
    subset_last_episode_id: List[Optional[int]] = [None for _ in range(args.num_subsets)]
    subset_last_trajectory_id: List[Optional[str]] = [None for _ in range(args.num_subsets)]
    subset_jsonl_paths: List[str] = []

    for subset_index in range(args.num_subsets):
        subset_directory = subset_dir(output_root, args.subset_prefix, subset_index, width)
        dataset_path = os.path.join(subset_directory, DEFAULT_DATASET_FILENAME)
        if os.path.exists(dataset_path) and not args.overwrite:
            raise FileExistsError(
                f"{dataset_path} already exists. Use --overwrite to rebuild subsets."
            )
        if os.path.isdir(subset_directory) and args.overwrite:
            shutil.rmtree(subset_directory)
        os.makedirs(subset_directory, exist_ok=True)
        subset_jsonl_paths.append(subset_jsonl_path(subset_directory))

    start_time = time.time()
    progress = make_tqdm(
        total=args.raw_total if args.raw_total and args.raw_total > 0 else None,
        desc="Build subsets",
        unit="raw",
    )
    try:
        for raw_seen, item in enumerate(iter_raw_annotations(args.raw_annotations), start=1):
            update_tqdm(
                progress,
                1,
                selected=f"{selected_total}/{target_total}",
                subset_full=f"{sum(count == args.subset_size for count in subset_counts)}/{args.num_subsets}",
            )
            raw_path_id = normalize_raw_path_id(str(item["path_id"]))
            if raw_path_id in excluded_path_ids:
                continue

            scene_id = build_scene_id(str(item["scan"]))
            subset_index = choose_target_subset(
                scene_id=scene_id,
                subset_counts=subset_counts,
                subset_size=args.subset_size,
                scene_seen_counts=scene_seen_counts,
                num_subsets=args.num_subsets,
            )
            if subset_index is None:
                break

            try:
                episode = raw_item_to_episode(
                    item=item,
                    episode_id=subset_counts[subset_index],
                    goal_radius=args.goal_radius,
                    connectivity_lookup=connectivity_lookup,
                )
            except Exception as exc:
                print(
                    f"[build-subsets] skip raw_index={raw_seen} path_id={raw_path_id} "
                    f"because {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            append_jsonl(subset_jsonl_paths[subset_index], episode)
            excluded_path_ids.add(raw_path_id)
            scene_seen_counts[scene_id] = scene_seen_counts.get(scene_id, 0) + 1
            subset_scene_sets[subset_index].add(scene_id)
            if subset_first_episode_id[subset_index] is None:
                subset_first_episode_id[subset_index] = int(episode["episode_id"])
                subset_first_trajectory_id[subset_index] = str(episode["trajectory_id"])
            subset_last_episode_id[subset_index] = int(episode["episode_id"])
            subset_last_trajectory_id[subset_index] = str(episode["trajectory_id"])
            subset_counts[subset_index] += 1
            selected_total += 1
            update_tqdm(
                progress,
                0,
                selected=f"{selected_total}/{target_total}",
                subset_full=f"{sum(count == args.subset_size for count in subset_counts)}/{args.num_subsets}",
            )

            if subset_counts[subset_index] == args.subset_size:
                subset_directory = subset_dir(
                    output_root, args.subset_prefix, subset_index, width
                )
                dataset_path = os.path.join(subset_directory, DEFAULT_DATASET_FILENAME)
                finalize_dataset_jsonl(subset_jsonl_paths[subset_index], dataset_path)
                write_subset_manifest(
                    subset_directory=subset_directory,
                    subset_index=subset_index,
                    goal_radius=args.goal_radius,
                    raw_count_seen=raw_seen,
                    num_episodes=subset_counts[subset_index],
                    first_episode_id=subset_first_episode_id[subset_index],
                    last_episode_id=subset_last_episode_id[subset_index],
                    first_trajectory_id=subset_first_trajectory_id[subset_index],
                    last_trajectory_id=subset_last_trajectory_id[subset_index],
                    unique_scenes=len(subset_scene_sets[subset_index]),
                )
                subset_paths.append(subset_directory)
                os.remove(subset_jsonl_paths[subset_index])
                elapsed = time.time() - start_time
                print(
                    f"[build-subsets] wrote subset {subset_index} with "
                    f"{subset_counts[subset_index]} episodes and {len(subset_scene_sets[subset_index])} scenes "
                    f"after scanning {raw_seen} "
                    f"raw annotations in {elapsed:.1f}s",
                    flush=True,
                )
                update_tqdm(
                    progress,
                    0,
                    selected=f"{selected_total}/{target_total}",
                    subset_full=f"{sum(count == args.subset_size for count in subset_counts)}/{args.num_subsets}",
                )
                if selected_total >= target_total:
                    break

            if raw_seen % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"[build-subsets] raw_seen={raw_seen} selected={selected_total}/{target_total} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        close_tqdm(progress)

    for subset_index in range(args.num_subsets):
        if subset_counts[subset_index] > 0 and os.path.exists(subset_jsonl_paths[subset_index]):
            try:
                os.remove(subset_jsonl_paths[subset_index])
            except FileNotFoundError:
                pass

    completed_subsets = sum(count == args.subset_size for count in subset_counts)
    if completed_subsets < args.num_subsets:
        raise RuntimeError(
            f"Only built {completed_subsets} subsets. "
            f"Expected {args.num_subsets}. raw_seen={raw_seen} selected={selected_total}"
        )

    return sorted(subset_paths)


def import_habitat_runtime(repo_root: str):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import habitat  # noqa: WPS433
    from habitat.config import read_write  # noqa: WPS433
    from habitat.config.default import get_config  # noqa: WPS433
    from habitat.tasks.nav.shortest_path_follower import (  # noqa: WPS433
        ShortestPathFollower,
    )

    import habitat_extensions.task  # noqa: F401,WPS433
    import habitat_extensions.measures  # noqa: F401,WPS433

    return habitat, get_config, read_write, ShortestPathFollower


def load_completed_ids(jsonl_path: str) -> Set[str]:
    if not os.path.exists(jsonl_path):
        return set()
    completed_ids: Set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed_ids.add(str(row["episode_id"]))
    return completed_ids


def positions_equal(a: Sequence[float], b: Sequence[float], tol: float = 1e-5) -> bool:
    return (
        abs(float(a[0]) - float(b[0])) <= tol
        and abs(float(a[1]) - float(b[1])) <= tol
        and abs(float(a[2]) - float(b[2])) <= tol
    )


def generate_episode_gt(
    env,
    episode,
    follower_cls,
    goal_radius: float,
) -> Dict:
    env.current_episode = episode
    env.reset()
    follower = follower_cls(
        env.sim,
        goal_radius=goal_radius,
        return_one_hot=False,
        stop_on_error=True,
    )

    actions: List[int] = []
    locations: List[List[float]] = [env.sim.get_agent_state().position.tolist()]
    reference_path = [list(point) for point in episode.reference_path]
    next_waypoint_id = 1 if len(reference_path) > 1 else 0

    while not env.episode_over:
        if next_waypoint_id >= len(reference_path):
            break

        target_position = reference_path[next_waypoint_id]
        next_action = follower.get_next_action(target_position)

        while next_action == STOP_ACTION:
            next_waypoint_id += 1
            if next_waypoint_id >= len(reference_path):
                next_action = STOP_ACTION
                break
            target_position = reference_path[next_waypoint_id]
            next_action = follower.get_next_action(target_position)

        if next_action == STOP_ACTION:
            break

        env.step(next_action)
        actions.append(int(next_action))
        current_position = env.sim.get_agent_state().position.tolist()
        if not positions_equal(current_position, locations[-1]):
            locations.append(current_position)

    return {
        "locations": locations,
        "actions": actions,
        "forward_steps": len(locations) - 1,
    }


def finalize_gt_jsonl(jsonl_path: str, output_gz_path: str) -> None:
    tmp_path = f"{output_gz_path}.tmp"
    with open(jsonl_path, "r", encoding="utf-8") as src, gzip.open(
        tmp_path, "wt", encoding="utf-8"
    ) as dst:
        dst.write("{")
        first = True
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if not first:
                dst.write(",")
            dst.write(json.dumps(str(row["episode_id"]), ensure_ascii=False))
            dst.write(":")
            dst.write(json.dumps(row["data"], ensure_ascii=False))
            first = False
        dst.write("}")
    os.replace(tmp_path, output_gz_path)


def discover_subset_directories(
    output_root: str,
    subset_prefix: str,
    subset_indices: Optional[Sequence[int]],
) -> List[str]:
    if subset_indices:
        return [
            os.path.join(output_root, f"{subset_prefix}_{index:02d}")
            for index in subset_indices
        ]

    return sorted(
        str(path)
        for path in Path(output_root).glob(f"{subset_prefix}_*")
        if path.is_dir()
    )


def episode_id_of(episode) -> str:
    if isinstance(episode, dict):
        return str(episode["episode_id"])
    return str(episode.episode_id)


def slice_and_shard_episodes(
    episodes: List,
    start_index: int,
    max_episodes: Optional[int],
    rank: int,
    world_size: int,
) -> List:
    sliced = episodes[start_index:]
    if max_episodes is not None:
        sliced = sliced[:max_episodes]
    return sliced[rank::world_size]


def group_episode_dicts_by_scene(episodes: List[Dict]) -> List[Tuple[str, List[Dict]]]:
    grouped: Dict[str, List[Dict]] = {}
    for episode in episodes:
        scene_id = str(episode["scene_id"])
        grouped.setdefault(scene_id, []).append(episode)
    return list(grouped.items())


def build_scene_dataset_path(
    subset_directory: str,
    scene_id: str,
) -> str:
    scene_hash = hashlib.md5(scene_id.encode("utf-8")).hexdigest()[:10]
    tmp_dir = os.path.join(subset_directory, ".scene_batches")
    os.makedirs(tmp_dir, exist_ok=True)
    return os.path.join(tmp_dir, f"{scene_hash}.json.gz")


def write_scene_dataset(scene_dataset_path: str, episodes: List[Dict], instruction_vocab: Dict) -> None:
    payload = {
        "episodes": episodes,
        "instruction_vocab": instruction_vocab,
    }
    atomic_write_json_gz(scene_dataset_path, payload)


def cleanup_subset_temp_files(subset_directory: str) -> None:
    scene_batch_dir = os.path.join(subset_directory, ".scene_batches")
    if os.path.isdir(scene_batch_dir):
        shutil.rmtree(scene_batch_dir, ignore_errors=True)
    dataset_jsonl = os.path.join(subset_directory, DEFAULT_DATASET_JSONL_FILENAME)
    if os.path.exists(dataset_jsonl):
        try:
            os.remove(dataset_jsonl)
        except FileNotFoundError:
            pass
    for path in Path(subset_directory).glob("*.tmp"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def generate_gt_for_subset(
    subset_directory: str,
    args: argparse.Namespace,
    habitat_runtime: Optional[Tuple] = None,
) -> None:
    habitat_runtime = habitat_runtime or import_habitat_runtime(args.repo_root)
    habitat, get_config, read_write, shortest_path_follower = habitat_runtime

    dataset_path = os.path.join(subset_directory, args.dataset_filename)
    gt_output_path = os.path.join(subset_directory, args.gt_filename)
    gt_jsonl_path = os.path.join(subset_directory, args.gt_jsonl_filename)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset file: {dataset_path}")
    if os.path.exists(gt_output_path) and not args.resume and not args.skip_finalize:
        raise FileExistsError(
            f"{gt_output_path} already exists. Use --resume or remove it first."
        )

    cleanup_subset_temp_files(subset_directory)
    completed_ids = load_completed_ids(gt_jsonl_path) if args.resume else set()
    append_mode = "a" if args.resume and os.path.exists(gt_jsonl_path) else "w"

    with gzip.open(dataset_path, "rt", encoding="utf-8") as handle:
        dataset_payload = json.load(handle)
    all_episode_dicts = list(dataset_payload["episodes"])
    selected_episode_dicts = slice_and_shard_episodes(
        episodes=all_episode_dicts,
        start_index=args.start_index,
        max_episodes=args.max_episodes,
        rank=args.rank,
        world_size=args.world_size,
    )
    if args.sort_by_scene:
        selected_episode_dicts = sorted(
            selected_episode_dicts,
            key=lambda episode: (episode["scene_id"], episode["trajectory_id"]),
        )
    scene_batches = group_episode_dicts_by_scene(selected_episode_dicts)

    os.makedirs(subset_directory, exist_ok=True)
    started_at = time.time()
    processed = 0
    skipped = 0
    progress = make_tqdm(
        total=len(selected_episode_dicts),
        desc=f"GT {os.path.basename(subset_directory)}",
        unit="ep",
    )

    try:
        with open(gt_jsonl_path, append_mode, encoding="utf-8") as handle:
            for scene_id, scene_episode_dicts in scene_batches:
                pending_episode_dicts = [
                    episode
                    for episode in scene_episode_dicts
                    if episode_id_of(episode) not in completed_ids
                ]
                scene_skipped = len(scene_episode_dicts) - len(pending_episode_dicts)
                skipped += scene_skipped
                update_tqdm(
                    progress,
                    scene_skipped,
                    scene=Path(scene_id).stem,
                    done=f"{processed + skipped}/{len(selected_episode_dicts)}",
                )
                if not pending_episode_dicts:
                    continue

                scene_dataset_path = build_scene_dataset_path(
                    subset_directory=subset_directory,
                    scene_id=scene_id,
                )
                write_scene_dataset(
                    scene_dataset_path=scene_dataset_path,
                    episodes=scene_episode_dicts,
                    instruction_vocab=dataset_payload["instruction_vocab"],
                )

                config = get_config(args.config_path)
                with read_write(config):
                    config.habitat.dataset.data_path = scene_dataset_path
                    config.habitat.dataset.scenes_dir = args.scenes_dir
                    config.habitat.environment.iterator_options.shuffle = False
                    config.habitat.environment.max_episode_steps = args.max_episode_steps
                    config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu_device_id
                    if args.minimal_observations:
                        config.habitat.simulator.agents.main_agent.sim_sensors = {}
                        config.habitat.task.lab_sensors = {}

                env = habitat.Env(config=config)
                try:
                    for episode in env.episodes:
                        episode_id = str(episode.episode_id)
                        if episode_id in completed_ids:
                            continue

                        record = generate_episode_gt(
                            env=env,
                            episode=episode,
                            follower_cls=shortest_path_follower,
                            goal_radius=args.goal_radius,
                        )
                        handle.write(
                            json.dumps(
                                {
                                    "episode_id": episode_id,
                                    "trajectory_id": str(episode.trajectory_id),
                                    "data": record,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        processed += 1
                        update_tqdm(
                            progress,
                            1,
                            scene=Path(scene_id).stem,
                            done=f"{processed + skipped}/{len(selected_episode_dicts)}",
                        )

                        if processed % args.gt_log_every == 0:
                            elapsed = time.time() - started_at
                            print(
                                f"[generate-gt] subset={os.path.basename(subset_directory)} "
                                f"scene={scene_id} processed={processed} skipped={skipped} "
                                f"elapsed={elapsed:.1f}s",
                                flush=True,
                            )
                finally:
                    env.close()
                    if os.path.exists(scene_dataset_path):
                        os.remove(scene_dataset_path)

        if not args.skip_finalize:
            finalize_gt_jsonl(gt_jsonl_path, gt_output_path)
            if not args.keep_jsonl:
                os.remove(gt_jsonl_path)
            elapsed = time.time() - started_at
            print(
                f"[generate-gt] finalized {gt_output_path} "
                f"processed={processed} skipped={skipped} elapsed={elapsed:.1f}s",
                flush=True,
            )
    finally:
        close_tqdm(progress)
        cleanup_subset_temp_files(subset_directory)


def run_build_subsets_if_needed(args: argparse.Namespace) -> List[str]:
    subset_directories = discover_subset_directories(
        output_root=args.output_root,
        subset_prefix=args.subset_prefix,
        subset_indices=list(range(args.num_subsets)),
    )
    if subset_directories and not args.overwrite:
        missing = [
            directory
            for directory in subset_directories
            if not os.path.exists(os.path.join(directory, DEFAULT_DATASET_FILENAME))
        ]
        if missing:
            raise FileNotFoundError(
                "Some subset directories exist but dataset files are missing: "
                + ", ".join(missing)
            )
        return subset_directories
    return build_subsets(args)


def main() -> None:
    args = parse_args()

    if args.command == "build-subsets":
        build_subsets(args)
        return

    if args.command == "generate-gt":
        subset_directories = discover_subset_directories(
            output_root=args.output_root,
            subset_prefix=args.subset_prefix,
            subset_indices=args.subset_indices,
        )
        if not subset_directories:
            raise FileNotFoundError(
                f"No subset directories found under {args.output_root} "
                f"with prefix {args.subset_prefix}"
            )
        runtime = import_habitat_runtime(args.repo_root)
        for subset_directory in subset_directories:
            generate_gt_for_subset(
                subset_directory=subset_directory,
                args=args,
                habitat_runtime=runtime,
            )
        return

    if args.command == "full":
        run_build_subsets_if_needed(args)
        subset_directories = discover_subset_directories(
            output_root=args.output_root,
            subset_prefix=args.subset_prefix,
            subset_indices=args.subset_indices,
        )
        runtime = import_habitat_runtime(args.repo_root)
        for subset_directory in subset_directories:
            generate_gt_for_subset(
                subset_directory=subset_directory,
                args=args,
                habitat_runtime=runtime,
            )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
