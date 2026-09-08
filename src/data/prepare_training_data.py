#!/usr/bin/env python3
"""Generate H18 training JSONL from stride-6 R2R/RxR and DAgger decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
STATIC_DATASETS = ("r2r", "rxr")
DATASET_ORDER = STATIC_DATASETS + ("dagger",)
FRAME_PATTERN = re.compile(r"frame_(\d+)\.(?:png|jpe?g)", re.IGNORECASE)

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
class OracleChunk:
    step_index: int
    actions: tuple[int, ...]


@dataclass(frozen=True)
class Episode:
    dataset: str
    episode_id: str
    instruction: str
    actions: tuple[int, ...]
    image_id: str
    oracle_chunks: tuple[OracleChunk, ...] = ()


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


def validate_trajectory_actions(
    actions: Sequence[int],
    context: str,
    minimum_length: int = 1,
) -> None:
    if len(actions) < minimum_length:
        raise ValueError(
            f"{context}: expected at least {minimum_length} actions, "
            f"got {len(actions)}"
        )
    invalid = sorted(set(int(value) for value in actions) - set(ACTION_NAMES))
    if invalid:
        raise ValueError(f"{context}: invalid action IDs {invalid}")
    if int(actions[-1]) != STOP or STOP in actions[:-1]:
        raise ValueError(f"{context}: actions must contain exactly one final stop")


def validate_actions(actions: Sequence[int], context: str) -> None:
    validate_trajectory_actions(
        actions,
        context,
        minimum_length=ACTION_HORIZON + 1,
    )


def validate_dagger_execution_policy(policy: dict) -> None:
    """Validate the variable-horizon collector's persisted configuration."""
    if not isinstance(policy, dict) or policy.get("actions_per_replan") != "uncertainty":
        raise ValueError("DAgger execution_policy must use uncertainty")
    budget = policy.get("uncertainty_budget")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or budget <= 0
    ):
        raise ValueError("DAgger uncertainty_budget must be finite and positive")
    action_range = policy.get("replan_action_range")
    if (
        not isinstance(action_range, (list, tuple))
        or len(action_range) != 2
        or any(isinstance(k, bool) or not isinstance(k, int) for k in action_range)
        or not 1 <= action_range[0] <= action_range[1] <= ACTION_HORIZON
    ):
        raise ValueError("DAgger replan_action_range must satisfy 1 <= MIN <= MAX <= 18")
    stop_window = policy.get("stop_oracle_max_actions")
    if (
        isinstance(stop_window, bool)
        or not isinstance(stop_window, int)
        or not action_range[1] <= stop_window <= ACTION_HORIZON
    ):
        raise ValueError("DAgger stop_oracle_max_actions must be between MAX and 18")


def parse_dagger_oracle_chunks(
    row: dict,
    actions: Sequence[int],
    context: str,
) -> tuple[OracleChunk, ...]:
    if (
        isinstance(row.get("action_horizon"), bool)
        or not isinstance(row.get("action_horizon"), int)
        or row.get("action_horizon") != ACTION_HORIZON
    ):
        raise ValueError(
            f"{context}: DAgger action_horizon must be {ACTION_HORIZON}, "
            f"got {row.get('action_horizon')!r}"
        )
    variable_horizon = "execution_policy" in row
    if variable_horizon:
        validate_dagger_execution_policy(row["execution_policy"])
    elif (
        isinstance(row.get("execute_horizon"), bool)
        or not isinstance(row.get("execute_horizon"), int)
        or row.get("execute_horizon") != EXECUTION_HORIZON
    ):
        raise ValueError(
            f"{context}: DAgger execute_horizon must be {EXECUTION_HORIZON}, "
            f"got {row.get('execute_horizon')!r}"
        )

    raw_chunks = row.get("oracle_chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError(f"{context}: DAgger episode has no oracle_chunks")

    chunks: list[OracleChunk] = []
    previous_step = -1
    previous_chunk = None
    non_stop_action_count = len(actions) - 1
    for chunk_index, raw_chunk in enumerate(raw_chunks):
        if not isinstance(raw_chunk, dict):
            raise ValueError(
                f"{context}: oracle chunk {chunk_index} must be an object"
            )
        step_index = raw_chunk.get("step_index")
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or step_index <= previous_step
            or step_index > non_stop_action_count
        ):
            raise ValueError(
                f"{context}: invalid oracle chunk step_index={step_index!r}"
            )

        try:
            oracle_actions = tuple(
                int(action) for action in raw_chunk.get("oracle_actions", [])
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context}: invalid oracle actions at step {step_index}"
            ) from exc
        if len(oracle_actions) != ACTION_HORIZON:
            raise ValueError(
                f"{context}: oracle chunk at step {step_index} must contain "
                f"{ACTION_HORIZON} actions, got {len(oracle_actions)}"
            )
        invalid = sorted(set(oracle_actions) - set(ACTION_NAMES))
        if invalid:
            raise ValueError(
                f"{context}: oracle chunk at step {step_index} has invalid "
                f"action IDs {invalid}"
            )

        if STOP in oracle_actions:
            first_stop = oracle_actions.index(STOP)
            if any(action != STOP for action in oracle_actions[first_stop:]):
                raise ValueError(
                    f"{context}: oracle chunk has an action after STOP at "
                    f"step {step_index}"
                )
            if (
                not variable_horizon
                and chunk_index < len(raw_chunks) - 1
                and first_stop < EXECUTION_HORIZON
            ):
                raise ValueError(
                    f"{context}: non-final oracle chunk requests STOP inside "
                    f"its executable prefix at step {step_index}"
                )

        if variable_horizon:
            minimum, maximum = row["execution_policy"]["replan_action_range"]
            horizon = raw_chunk.get("execute_horizon")
            count = raw_chunk.get("executed_count")
            source = raw_chunk.get("executed_policy")
            terminal = source == "terminal_oracle"
            if source not in {
                "model", "oracle", "model_stop_oracle", "model_fallback_oracle",
                "terminal_oracle",
            }:
                raise ValueError(f"{context}: invalid executed_policy at step {step_index}")
            if (
                isinstance(horizon, bool)
                or not isinstance(horizon, int)
                or (horizon != 1 if terminal else not minimum <= horizon <= maximum)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or not 1 <= count <= horizon
                or (terminal and oracle_actions != (STOP,) * ACTION_HORIZON)
            ):
                raise ValueError(f"{context}: invalid execution length at step {step_index}")
            if previous_chunk is not None:
                if step_index - previous_step != previous_chunk["executed_count"]:
                    raise ValueError(f"{context}: executed_count does not match decision gap")
                if (
                    previous_chunk["executed_count"] < previous_chunk["execute_horizon"]
                    and not terminal
                ):
                    raise ValueError(f"{context}: only terminal decisions may interrupt a prefix")
            actual = tuple(actions[step_index:step_index + count])
            if len(actual) != count:
                raise ValueError(f"{context}: execution extends beyond trajectory")
            if source == "model":
                if STOP in actual:
                    raise ValueError(f"{context}: a model decision must not execute STOP")
            elif actual != oracle_actions[:count]:
                raise ValueError(f"{context}: executed expert actions do not match oracle labels")
            if chunk_index == len(raw_chunks) - 1 and step_index + count != len(actions):
                raise ValueError(f"{context}: final execution does not cover trajectory")
            previous_chunk = raw_chunk
        elif previous_step >= 0:
            step_gap = step_index - previous_step
            if step_gap != EXECUTION_HORIZON:
                is_early_terminal_decision = bool(
                    0 < step_gap < EXECUTION_HORIZON
                    and step_index == non_stop_action_count
                    and oracle_actions[0] == STOP
                )
                if not is_early_terminal_decision:
                    raise ValueError(
                        f"{context}: expected DAgger decision gap "
                        f"{EXECUTION_HORIZON}, got {step_gap} before step "
                        f"{step_index}"
                    )

        chunks.append(OracleChunk(step_index, oracle_actions))
        previous_step = step_index

    if chunks[0].step_index != 0:
        raise ValueError(f"{context}: first DAgger decision must be at step 0")
    final_actions = chunks[-1].actions
    if STOP not in final_actions:
        raise ValueError(f"{context}: final DAgger oracle chunk has no STOP")
    if chunks[-1].step_index + final_actions.index(STOP) != non_stop_action_count:
        raise ValueError(
            f"{context}: final oracle STOP does not align with the executed "
            "trajectory"
        )
    return tuple(chunks)


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
    """Apply the H18 strategy with body stride 6."""

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
    dataset_names: Sequence[str] = STATIC_DATASETS,
) -> list[Episode]:
    normalized_names = normalize_dataset_names(dataset_names)
    episodes: list[Episode] = []
    seen: set[tuple[str, str]] = set()
    for dataset in normalized_names:
        path = input_root / "sub_dataset" / f"{dataset}.jsonl"
        raw = path.read_bytes()
        expected = EXPECTED_SOURCES.get(dataset)

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
            context = f"{path}:{line_number}"
            if dataset == "dagger":
                validate_trajectory_actions(actions, context)
                oracle_chunks = parse_dagger_oracle_chunks(
                    row,
                    actions,
                    context,
                )
            else:
                validate_actions(actions, context)
                oracle_chunks = ()
            instruction = row.get("instruction")
            if not isinstance(instruction, str) or not instruction:
                raise ValueError(f"Missing instruction at {path}:{line_number}")
            raw_image_id = row.get("trajectory_id", row["episode_id"])
            if (
                isinstance(raw_image_id, bool)
                or raw_image_id is None
                or not str(raw_image_id).strip()
            ):
                raise ValueError(f"Missing trajectory image ID at {path}:{line_number}")
            image_id = str(raw_image_id)
            episodes.append(
                Episode(
                    dataset,
                    episode_id,
                    instruction,
                    actions,
                    image_id,
                    oracle_chunks,
                )
            )
            starts += len(actions)
            count += 1

        if expected is not None and (
            count != expected["episodes"] or starts != expected["starts"]
        ):
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
    dataset_names: Sequence[str] = STATIC_DATASETS,
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
                desc="prepare_h18",
                dynamic_ncols=True,
            ):
                images = load_image_paths(input_root, episode)
                history = [ACTION_NAMES[action] for action in episode.actions]

                if episode.dataset == "dagger":
                    targets = tuple(
                        (chunk.step_index, chunk.actions)
                        for chunk in episode.oracle_chunks
                    )
                    audit["dagger_oracle_decisions"] += len(targets)
                else:
                    selected, episode_audit = select_starts(
                        episode.dataset,
                        episode.episode_id,
                        episode.actions,
                        seed,
                    )
                    audit.update(episode_audit)
                    targets = tuple(
                        (start, action_target(episode.actions, start))
                        for start in selected
                    )

                for start, target in targets:

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

        static_names = tuple(
            dataset for dataset in normalized_names if dataset in STATIC_DATASETS
        )
        if static_names:
            expected_static_rows = EXPECTED_ROWS[static_names]
            actual_static_rows = sum(
                dataset_counts[dataset] for dataset in static_names
            )
            if actual_static_rows != expected_static_rows:
                raise AssertionError(
                    f"Static row mismatch: {actual_static_rows} != "
                    f"{expected_static_rows}"
                )

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
        description=(
            "Generate H18 training JSONL: stride-6 sampling for R2R/RxR "
            "and complete oracle-decision preservation for DAgger."
        )
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
        default=list(STATIC_DATASETS),
        help="Dataset subsets to include (default: r2r rxr; dagger is optional).",
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
