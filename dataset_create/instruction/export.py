"""Strict Habitat R2R/ScaleVLN shape; instruction audit data stays in sidecars."""

from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath
import shutil

import numpy as np

from dataset_create.trajectory.io_utils import atomic_json_dump, atomic_json_gz_dump


EMPTY_VOCAB = {"word_list": [], "word2idx_dict": {}, "stoi": {}, "itos": [],
               "num_vocab": 0, "UNK_INDEX": 1, "PAD_INDEX": 0}
EPISODE_KEYS = {"episode_id", "trajectory_id", "scene_id", "start_position", "start_rotation",
                "info", "goals", "instruction", "reference_path"}


def make_r2r_episode(episode, episode_index, states, instruction_text, geodesic_distance,
                     scene_path, scene_root, goal_radius=.25):
    reference = []
    for state in states:
        position = list(map(float, state["position"]))
        if not reference or np.linalg.norm(np.asarray(position) - reference[-1]) > 1e-5:
            reference.append(position)
    # Scene IDs are relative to the real HM3D root used by Habitat's scenes_dir.
    # The input's logical 'hm3d/' prefix is not a directory beneath that root.
    result = {
        "episode_id": int(episode_index), "trajectory_id": episode["trajectory_id"],
        "scene_id": Path(scene_path).resolve().relative_to(Path(scene_root).resolve()).as_posix(),
        "start_position": copy.deepcopy(episode["start_position"]),
        "start_rotation": copy.deepcopy(episode["start_rotation_xyzw"]),
        "info": {"geodesic_distance": float(geodesic_distance)},
        "goals": [{"position": copy.deepcopy(states[-1]["position"]), "radius": float(goal_radius)}],
        "instruction": {"instruction_text": instruction_text, "instruction_tokens": None},
        "reference_path": reference,
    }
    validate_r2r({"episodes": [result], "instruction_vocab": EMPTY_VOCAB}, scene_root)
    return result


def validate_r2r(dataset, scene_root=None):
    if set(dataset) != {"episodes", "instruction_vocab"}:
        raise ValueError("R2R root must contain only episodes and instruction_vocab")
    if dataset["instruction_vocab"] != EMPTY_VOCAB:
        raise ValueError("Vocabulary must match the untokenized ScaleVLN reference")
    identifiers, trajectories = set(), set()
    for episode in dataset["episodes"]:
        if set(episode) != EPISODE_KEYS:
            raise ValueError("Unexpected or missing R2R episode fields")
        if episode["episode_id"] in identifiers or episode["trajectory_id"] in trajectories:
            raise ValueError("Duplicate R2R episode or trajectory ID")
        identifiers.add(episode["episode_id"])
        trajectories.add(episode["trajectory_id"])
        scene = PurePosixPath(episode["scene_id"])
        if scene.is_absolute() or ".." in scene.parts:
            raise ValueError("R2R scene_id must be portable and relative")
        if scene_root is not None and not (Path(scene_root) / episode["scene_id"]).is_file():
            raise ValueError(f"Exported scene does not resolve: {episode['scene_id']}")
        for position in [episode["start_position"], *episode["reference_path"], *[g["position"] for g in episode["goals"]]]:
            if np.asarray(position).shape != (3,) or not np.isfinite(position).all():
                raise ValueError("Invalid R2R position")
        rotation = np.asarray(episode["start_rotation"])
        if rotation.shape != (4,) or not np.isfinite(rotation).all() or abs(np.linalg.norm(rotation)-1) > 1e-4:
            raise ValueError("Invalid xyzw start_rotation")
        if not episode["reference_path"] or not np.allclose(episode["reference_path"][0], episode["start_position"], atol=.001):
            raise ValueError("Reference path does not start at start_position")
        if len(episode["goals"]) != 1 or not np.allclose(episode["goals"][0]["position"], episode["reference_path"][-1], atol=.001):
            raise ValueError("Goal/reference_path must end at the real STOP position")
        if set(episode["goals"][0]) != {"position", "radius"} or episode["goals"][0]["radius"] <= 0:
            raise ValueError("Invalid R2R goal")
        instruction = episode["instruction"]
        if set(instruction) != {"instruction_text", "instruction_tokens"} or instruction["instruction_tokens"] is not None:
            raise ValueError("Instruction does not match ScaleVLN schema")
        if not isinstance(instruction["instruction_text"], str) or not instruction["instruction_text"].strip():
            raise ValueError("Empty R2R instruction")
        if set(episode["info"]) != {"geodesic_distance"} or not np.isfinite(episode["info"]["geodesic_distance"]) or episode["info"]["geodesic_distance"] < 0:
            raise ValueError("Invalid geodesic distance")


def write_r2r(episodes, path, scene_root=None):
    dataset = {"episodes": sorted(episodes, key=lambda e: e["episode_id"]),
               "instruction_vocab": copy.deepcopy(EMPTY_VOCAB)}
    validate_r2r(dataset, scene_root)
    path = Path(path)
    if path.name.endswith(".json.gz"):
        gzip_path, json_path = path, path.with_suffix("")
    elif path.suffix == ".json":
        json_path, gzip_path = path, path.with_suffix(".json.gz")
    else:
        raise ValueError("R2R output must end in .json or .json.gz")
    # Keep interrupted serialization files together so a resumed export can
    # replace the pair and clean all leftovers under the run's output lock.
    staging = json_path.parent / f".{json_path.stem}.export"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        atomic_json_dump(dataset, staging / json_path.name, indent=None)
        atomic_json_gz_dump(dataset, staging / gzip_path.name)
        (staging / json_path.name).replace(json_path)
        (staging / gzip_path.name).replace(gzip_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return len(episodes)
