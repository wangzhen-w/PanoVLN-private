#!/usr/bin/env python3
"""Generate stride-6, 18-action R2R/RxR training JSONL data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, MutableMapping, Sequence

from tqdm import tqdm


ACTION_HORIZON = 18
BODY_STRIDE = 6
EXECUTION_HORIZON = 6
DEFAULT_SEED = 42
STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
TURN_ACTIONS = frozenset({LEFT, RIGHT})
ACTION_NAMES = {
    STOP: "stop",
    FORWARD: "forward",
    LEFT: "left",
    RIGHT: "right",
}
DATASET_ORDER = ("r2r", "rxr")
FRAME_PATTERN = re.compile(r"frame_(\d+)\.png")

EXPECTED_SOURCES = {
    "r2r": {
        "episodes": 10_692,
        "starts": 660_055,
    },
    "rxr": {
        "episodes": 18_063,
        "starts": 1_898_944,
    },
}
EXPECTED_ROWS = {
    ("r2r",): 240_806,
    ("rxr",): 570_528,
    ("r2r", "rxr"): 811_334,
}


@dataclass(frozen=True)
class Episode:
    dataset: str
    episode_id: str
    instruction: str
    actions: tuple[int, ...]
    image_id: str


@dataclass(frozen=True)
class ActionBlock:
    """A half-open maximal same-action run: [start, end)."""

    start: int
    end: int
    action: int

    @property
    def length(self) -> int:
        return self.end - self.start


def stable_choice(options: Sequence[int], *parts: object) -> int:
    ordered = tuple(sorted(int(option) for option in options))
    if not ordered:
        raise ValueError("stable_choice requires at least one option")
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return ordered[value % len(ordered)]


def maximal_blocks(
    actions: Sequence[int], accepted_actions: frozenset[int]
) -> Iterable[ActionBlock]:
    stop_index = len(actions) - 1
    index = 0
    while index < stop_index:
        action = int(actions[index])
        if action not in accepted_actions:
            index += 1
            continue
        end = index + 1
        while end < stop_index and int(actions[end]) == action:
            end += 1
        yield ActionBlock(index, end, action)
        index = end


def validate_actions(actions: Sequence[int], context: str) -> None:
    if len(actions) < ACTION_HORIZON + 1:
        raise ValueError(
            f"{context}: expected at least {ACTION_HORIZON + 1} actions, "
            f"got {len(actions)}"
        )
    invalid = sorted(set(int(value) for value in actions) - set(ACTION_NAMES))
    if invalid:
        raise ValueError(f"{context}: invalid action IDs {invalid}")
    if int(actions[-1]) != STOP or STOP in actions[:-1]:
        raise ValueError(f"{context}: actions must contain exactly one final stop")


def action_target(actions: Sequence[int], start: int) -> tuple[int, ...]:
    target = tuple(int(value) for value in actions[start : start + ACTION_HORIZON])
    if len(target) < ACTION_HORIZON:
        if not target or target[-1] != STOP:
            raise ValueError(
                "Only a terminal target ending in stop may be padded: "
                f"start={start}, target={target}"
            )
        target += (STOP,) * (ACTION_HORIZON - len(target))
    return target


def add_reason(
    selected: MutableMapping[int, set[str]],
    actions: Sequence[int],
    start: int,
    reason: str,
) -> None:
    if not 0 <= start < len(actions):
        raise ValueError(f"Invalid start {start} for {len(actions)} actions")
    selected.setdefault(start, set()).add(reason)


def centered_forward_options(block: ActionBlock) -> tuple[int, ...]:
    if block.action != FORWARD or block.length < ACTION_HORIZON:
        return ()
    candidates = tuple(range(block.start, block.end - ACTION_HORIZON + 1))
    best = min(
        abs((start - block.start) - (block.end - start - ACTION_HORIZON))
        for start in candidates
    )
    return tuple(
        start
        for start in candidates
        if abs((start - block.start) - (block.end - start - ACTION_HORIZON))
        == best
    )


def select_starts(
    dataset: str,
    episode_id: str,
    actions: Sequence[int],
    seed: int,
) -> tuple[dict[int, set[str]], Counter]:
    """Apply the stride-6 H=18 sampling strategy."""

    validate_actions(actions, f"{dataset}:{episode_id}")
    selected: dict[int, set[str]] = {}
    audit = Counter()

    for start in range(0, len(actions), BODY_STRIDE):
        add_reason(selected, actions, start, "stride6")
        audit["stride6"] += 1

    for block in maximal_blocks(actions, TURN_ACTIONS):
        audit["turn_blocks"] += 1
        if block.length >= 2:
            add_reason(selected, actions, block.start, "multi_turn_onset")
            audit["multi_turn_onset"] += 1

    for block in maximal_blocks(actions, frozenset({FORWARD})):
        options = centered_forward_options(block)
        if not options:
            continue
        center = stable_choice(
            options,
            seed,
            dataset,
            episode_id,
            block.start,
            block.end - 1,
            "forward_center_ge18",
        )
        add_reason(selected, actions, center, "forward_center")
        audit["forward_center"] += 1

    for position in range(1, EXECUTION_HORIZON + 1):
        start = len(actions) - position
        add_reason(selected, actions, start, "terminal_executed_dense")
        audit["terminal_executed_dense"] += 1

    for first_position in range(
        EXECUTION_HORIZON + 1,
        ACTION_HORIZON + 1,
        2,
    ):
        position = stable_choice(
            (first_position, first_position + 1),
            seed,
            dataset,
            episode_id,
            "stop_future_pair",
            first_position,
        )
        start = len(actions) - position
        add_reason(selected, actions, start, "terminal_future_pair")
        audit["terminal_future_pair"] += 1

    return dict(sorted(selected.items())), audit


def normalize_dataset_names(dataset_names: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(name).lower() for name in dataset_names)
    if not requested:
        raise ValueError("At least one dataset must be selected")
    duplicates = sorted(
        name for name, count in Counter(requested).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate datasets: {duplicates}")
    unsupported = sorted(set(requested) - set(DATASET_ORDER))
    if unsupported:
        raise ValueError(
            f"Unsupported datasets {unsupported}; choose from {list(DATASET_ORDER)}"
        )
    return tuple(name for name in DATASET_ORDER if name in requested)


def load_episodes(
    input_root: Path,
    dataset_names: Sequence[str] = DATASET_ORDER,
) -> list[Episode]:
    normalized_names = normalize_dataset_names(dataset_names)
    episodes: list[Episode] = []
    seen: set[tuple[str, str]] = set()
    for dataset in normalized_names:
        path = input_root / "sub_dataset" / f"{dataset}.jsonl"
        raw = path.read_bytes()
        expected = EXPECTED_SOURCES[dataset]

        starts = 0
        count = 0
        for line_number, line in enumerate(raw.splitlines(), 1):
            row = json.loads(line)
            episode_id = str(row["episode_id"])
            key = (dataset, episode_id)
            if key in seen:
                raise ValueError(f"Duplicate episode at {path}:{line_number}: {key}")
            seen.add(key)

            actions = tuple(int(value) for value in row["actions"])
            validate_actions(actions, f"{path}:{line_number}")
            instruction = row.get("instruction")
            if not isinstance(instruction, str) or not instruction:
                raise ValueError(f"Missing instruction at {path}:{line_number}")
            image_id = str(row.get("trajectory_id", row["episode_id"]))
            episodes.append(
                Episode(dataset, episode_id, instruction, actions, image_id)
            )
            starts += len(actions)
            count += 1

        if count != expected["episodes"] or starts != expected["starts"]:
            raise ValueError(
                f"Source count mismatch for {dataset}: episodes={count}, "
                f"starts={starts}, expected={expected}"
            )
    return episodes


def load_image_paths(input_root: Path, episode: Episode) -> list[str]:
    directory = input_root / "images" / episode.dataset / episode.image_id
    if not directory.is_dir():
        raise ValueError(f"Missing image directory: {directory}")
    indexed: dict[int, str] = {}
    for name in os.listdir(directory):
        match = FRAME_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"Unexpected frame filename in {directory}: {name}")
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"Duplicate frame index {index} in {directory}")
        indexed[index] = name
    if sorted(indexed) != list(range(len(episode.actions))):
        raise ValueError(
            f"Frame/action mismatch in {directory}: "
            f"frames={len(indexed)}, actions={len(episode.actions)}"
        )
    return [
        f"images/{episode.dataset}/{episode.image_id}/{indexed[index]}"
        for index in range(len(episode.actions))
    ]


def create_temp_path(output_path: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def build_training_jsonl(
    input_root: Path,
    output_path: Path,
    seed: int,
    overwrite: bool,
    dataset_names: Sequence[str] = DATASET_ORDER,
) -> dict[str, object]:
    if seed != DEFAULT_SEED:
        raise ValueError(f"This dataset is fixed to seed={DEFAULT_SEED}")
    normalized_names = normalize_dataset_names(dataset_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite"
        )

    episodes = load_episodes(input_root, normalized_names)
    temporary = create_temp_path(output_path)
    dataset_counts = Counter()
    audit = Counter()
    row_count = 0

    try:
        with temporary.open("wb") as output:
            for episode in tqdm(
                episodes,
                desc="stride6_h18",
                dynamic_ncols=True,
            ):
                selected, episode_audit = select_starts(
                    episode.dataset,
                    episode.episode_id,
                    episode.actions,
                    seed,
                )
                audit.update(episode_audit)
                images = load_image_paths(input_root, episode)
                history = [ACTION_NAMES[action] for action in episode.actions]

                for start in selected:
                    target = action_target(episode.actions, start)

                    real_action_count = (
                        target.index(STOP) + 1 if STOP in target else ACTION_HORIZON
                    )
                    training_row = {
                        "instruction": episode.instruction,
                        "action_sequence": [ACTION_NAMES[action] for action in target],
                        "images": images[: start + 1],
                        "episode_id": episode.episode_id,
                        "dataset": episode.dataset,
                        "step_index": start,
                        "end_step": start + real_action_count - 1,
                        "real_action_count": real_action_count,
                        "history_actions": history[:start],
                    }
                    output.write(
                        (
                            json.dumps(
                                training_row,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
                    row_count += 1
                    dataset_counts[episode.dataset] += 1
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)

        expected_rows = EXPECTED_ROWS[normalized_names]
        if row_count != expected_rows:
            raise AssertionError(f"Row mismatch: {row_count} != {expected_rows}")

        os.replace(temporary, output_path)
        result = {
            "output_path": str(output_path),
            "datasets": list(normalized_names),
            "rows": row_count,
            "dataset_counts": dict(dataset_counts),
            "anchor_events": dict(audit),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return result
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stride-6 H18 R2R/RxR training JSONL data."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("/workspace/data2/dataset/PanoVLN"),
    )
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        choices=DATASET_ORDER,
        default=list(DATASET_ORDER),
        help="Dataset subsets to include (default: r2r rxr).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path(
            "/workspace/data2/dataset/ablation/18-action/"
            "train_r2r_rxr_h18_stop_1-6_stride1_7-18_stride2_seed42.jsonl"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_training_jsonl(
        input_root=args.input_root.resolve(),
        output_path=args.output_path.resolve(),
        seed=args.seed,
        overwrite=args.overwrite,
        dataset_names=args.dataset_name,
    )


if __name__ == "__main__":
    main()
