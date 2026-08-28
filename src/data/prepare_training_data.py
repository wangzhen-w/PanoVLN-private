#!/usr/bin/env python3
"""Generate the R2R+RxR 12-action maneuver-phase training JSONL."""

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


ACTION_HORIZON = 12
DEFAULT_SEED = 42
STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
TURN_ACTIONS = frozenset({LEFT, RIGHT})
ACTION_NAMES = {
    STOP: "stop",
    FORWARD: "forward",
    LEFT: "left",
    RIGHT: "right",
}
ACTION_CHARS = {
    STOP: "S",
    FORWARD: "F",
    LEFT: "L",
    RIGHT: "R",
}
DATASET_ORDER = ("r2r", "rxr")
STOP_BINS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
)
REASON_ORDER = {
    "episode_start": 0,
    "turn_onset": 1,
    "complete_centered": 2,
    "long_turn_entry": 3,
    "long_turn_exit": 4,
    "forward_center": 5,
}
FRAME_PATTERN = re.compile(r"frame_(\d+)\.png")

EXPECTED_SOURCES = {
    "r2r": {
        "sha256": "7338528cb9ffe55283c7c9faf817c75e02be3e24a51da35ef6dfd40c0a38aaaa",
        "bytes": 3_340_690,
        "episodes": 10_692,
        "starts": 660_055,
    },
    "rxr": {
        "sha256": "b04f348f2e5d9ca3d5e037e762f7ecd87adc63891568d0202df088cc8dc78891",
        "bytes": 14_259_993,
        "episodes": 18_057,
        "starts": 1_897_951,
    },
}
DELETED_RXR_EPISODES = frozenset(
    {"18639", "18640", "18641", "19236", "19243", "19244"}
)
EXPECTED_ROWS = 788_847
EXPECTED_SELECTION_SHA256 = (
    "a0d3a213f2ae586673e79a3a971ab1b2ce799a33dc5e1ed61048fb71c6ef5eeb"
)
EXPECTED_TRAINING_SHA256 = (
    "859186155df5cf778076802c3ea82500f6c1dd7184bc4637b7a2a4841784a609"
)


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


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    for index, (left, right) in enumerate(zip(actions, actions[1:])):
        if left in TURN_ACTIONS and right in TURN_ACTIONS and left != right:
            raise ValueError(
                f"{context}: immediate L/R reversal at action boundary {index}"
            )
    turn_lengths = [
        block.length for block in maximal_blocks(actions, TURN_ACTIONS)
    ]
    if turn_lengths and max(turn_lengths) > ACTION_HORIZON:
        raise ValueError(
            f"{context}: turn block length {max(turn_lengths)} exceeds "
            f"{ACTION_HORIZON}"
        )


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


def centered_turn_options(
    actions: Sequence[int], block: ActionBlock
) -> tuple[int, ...]:
    if block.length > ACTION_HORIZON - 2:
        return ()
    stop_index = len(actions) - 1
    if (
        block.start == 0
        or block.end >= stop_index
        or actions[block.start - 1] != FORWARD
        or actions[block.end] != FORWARD
    ):
        return ()

    low = max(0, block.end - (ACTION_HORIZON - 1))
    high = block.start - 1
    candidates: list[tuple[int, int]] = []
    for start in range(low, high + 1):
        before = block.start - start
        after = start + ACTION_HORIZON - block.end
        if before >= 1 and after >= 1:
            candidates.append((start, abs(before - after)))
    if not candidates:
        return ()
    best = min(score for _, score in candidates)
    return tuple(start for start, score in candidates if score == best)


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


def reason_sort_key(reason: str) -> tuple[int, str]:
    if reason.startswith("stop_bin_"):
        return (6, reason)
    return (REASON_ORDER[reason], reason)


def select_starts(
    dataset: str,
    episode_id: str,
    actions: Sequence[int],
    seed: int,
) -> tuple[dict[int, set[str]], Counter]:
    """Apply only the final H=12 maneuver-phase sampling strategy."""

    validate_actions(actions, f"{dataset}:{episode_id}")
    selected: dict[int, set[str]] = {}
    audit = Counter()

    add_reason(selected, actions, 0, "episode_start")
    audit["episode_start"] += 1

    for block in maximal_blocks(actions, TURN_ACTIONS):
        audit["turn_blocks"] += 1
        add_reason(selected, actions, block.start, "turn_onset")
        audit["turn_onset"] += 1

        if block.length <= 10:
            options = centered_turn_options(actions, block)
            if not options:
                continue
            center = stable_choice(
                options,
                seed,
                dataset,
                episode_id,
                block.start,
                block.end - 1,
                "center_tie",
            )
            keep_center = block.length > 1
            if block.length == 1:
                approach = actions[center:block.start]
                keep_center = bool(approach) and all(
                    action == FORWARD for action in approach
                )
                audit[
                    "length1_direct_center"
                    if keep_center
                    else "length1_center_skipped"
                ] += 1
            if keep_center:
                add_reason(selected, actions, center, "complete_centered")
                audit["complete_centered"] += 1
        else:
            entry = block.start - 1
            if entry >= 0 and actions[entry] == FORWARD:
                add_reason(selected, actions, entry, "long_turn_entry")
                audit["long_turn_entry"] += 1
            if block.length == 12:
                exit_start = block.end - 11
                if block.end < len(actions) - 1 and actions[block.end] == FORWARD:
                    add_reason(selected, actions, exit_start, "long_turn_exit")
                    audit["long_turn_exit"] += 1

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
            "forward_center_ge12",
        )
        add_reason(selected, actions, center, "forward_center")
        audit["forward_center"] += 1

    for bin_index, positions in enumerate(STOP_BINS):
        starts = tuple(len(actions) - position for position in positions)
        existing = tuple(start for start in starts if start in selected)
        if existing:
            audit["stop_bin_precovered"] += 1
            audit["stop_bin_multiple"] += len(existing) > 1
            continue
        position = stable_choice(
            positions,
            seed,
            dataset,
            episode_id,
            "stop_pair6",
            bin_index,
        )
        start = len(actions) - position
        add_reason(
            selected,
            actions,
            start,
            f"stop_bin_{bin_index + 1}_pos_{position}",
        )
        audit["stop_supplement"] += 1

    return dict(sorted(selected.items())), audit


def load_episodes(input_root: Path) -> list[Episode]:
    episodes: list[Episode] = []
    seen: set[tuple[str, str]] = set()
    for dataset in DATASET_ORDER:
        path = input_root / "sub_dataset" / f"{dataset}.jsonl"
        raw = path.read_bytes()
        expected = EXPECTED_SOURCES[dataset]
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(
                f"Source fingerprint mismatch for {path}: "
                f"bytes={len(raw)} sha256={digest}"
            )

        starts = 0
        count = 0
        for line_number, line in enumerate(raw.splitlines(), 1):
            row = json.loads(line)
            episode_id = str(row["episode_id"])
            key = (dataset, episode_id)
            if key in seen:
                raise ValueError(f"Duplicate episode at {path}:{line_number}: {key}")
            seen.add(key)
            if dataset == "rxr" and episode_id in DELETED_RXR_EPISODES:
                raise ValueError(f"Deleted reversal episode remains: {key}")

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
) -> dict[str, object]:
    if seed != DEFAULT_SEED:
        raise ValueError(f"This dataset is fixed to seed={DEFAULT_SEED}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite"
        )

    episodes = load_episodes(input_root)
    temporary = create_temp_path(output_path)
    selection_hash = hashlib.sha256()
    dataset_counts = Counter()
    audit = Counter()
    row_count = 0

    try:
        with temporary.open("wb") as output:
            for episode in tqdm(
                episodes,
                desc="maneuver_h12",
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

                for start, raw_reasons in selected.items():
                    reasons = sorted(raw_reasons, key=reason_sort_key)
                    target = action_target(episode.actions, start)
                    target_chars = "".join(ACTION_CHARS[action] for action in target)
                    selection_row = {
                        "action_sequence": target_chars,
                        "dataset": episode.dataset,
                        "episode_id": episode.episode_id,
                        "reasons": reasons,
                        "step_index": start,
                    }
                    selection_hash.update(
                        (canonical_json(selection_row) + "\n").encode("utf-8")
                    )

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

        selection_sha = selection_hash.hexdigest()
        if row_count != EXPECTED_ROWS:
            raise AssertionError(f"Row mismatch: {row_count} != {EXPECTED_ROWS}")
        if selection_sha != EXPECTED_SELECTION_SHA256:
            raise AssertionError(
                f"Selection mismatch: {selection_sha} != "
                f"{EXPECTED_SELECTION_SHA256}"
            )
        training_sha = sha256_file(temporary)
        if training_sha != EXPECTED_TRAINING_SHA256:
            raise AssertionError(
                f"Training JSONL mismatch: {training_sha} != "
                f"{EXPECTED_TRAINING_SHA256}"
            )

        os.replace(temporary, output_path)
        result = {
            "output_path": str(output_path),
            "rows": row_count,
            "dataset_counts": dict(dataset_counts),
            "sha256": training_sha,
            "selection_sha256": selection_sha,
            "anchor_events": dict(audit),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return result
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the final R2R+RxR maneuver-phase H12 JSONL."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("/workspace/data2/dataset/PanoVLN"),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path(
            "/workspace/data2/dataset/ablation/12-action/"
            "train_r2r_rxr_maneuver_h12_seed42.jsonl"
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
    )


if __name__ == "__main__":
    main()
