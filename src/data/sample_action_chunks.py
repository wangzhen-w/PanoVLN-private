import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm


DEFAULT_WEIGHTS = {
    "first_stop": 0.08,
    "second_stop": 0.08,
    "third_stop": 0.03,
    "fourth_stop": 0.03,
    "nonstop_first_forward": 0.33,
    "nonstop_first_left": 0.225,
    "nonstop_first_right": 0.225,
}
ACTION_WORDS = {"forward", "left", "right", "stop"}


def parse_weights(raw_weights: str) -> Dict[str, float]:
    if not raw_weights:
        return dict(DEFAULT_WEIGHTS)

    weights = {}
    for item in raw_weights.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight item {item!r}; expected bucket=value")
        bucket, value = item.split("=", 1)
        weights[bucket.strip()] = float(value)
    return weights


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        raise ValueError("At least one sampling bucket weight is required")
    for bucket, weight in weights.items():
        if weight < 0:
            raise ValueError(f"Bucket weight must be non-negative: {bucket}={weight}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Sampling bucket weights must sum to a positive value")
    return {bucket: weight / total for bucket, weight in weights.items()}


def action_sequence_from_item(item: Dict) -> List[str]:
    action_sequence = item.get("action_sequence")
    if not isinstance(action_sequence, list):
        raise ValueError("JSONL item field 'action_sequence' must be a list")

    actions = []
    for action in action_sequence:
        if not isinstance(action, str):
            raise ValueError(f"Action must be a string, got {action!r}")
        normalized = action.strip().lower()
        if normalized not in ACTION_WORDS:
            raise ValueError(f"Unknown action {action!r}; expected one of {sorted(ACTION_WORDS)}")
        actions.append(normalized)

    if not (1 <= len(actions) <= 4):
        raise ValueError(f"Expected 1 to 4 actions, got {len(actions)}")
    if "stop" in actions and actions[-1] != "stop":
        raise ValueError(f"Action sequence must end immediately after stop: {actions}")
    return actions


def assign_bucket(actions: List[str]) -> str:
    if actions[0] == "stop":
        return "first_stop"
    if len(actions) >= 2 and actions[1] == "stop":
        return "second_stop"
    if len(actions) >= 3 and actions[2] == "stop":
        return "third_stop"
    if len(actions) >= 4 and actions[3] == "stop":
        return "fourth_stop"
    return f"nonstop_first_{actions[0]}"


def allocate_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    raw_counts = {bucket: total * weight for bucket, weight in weights.items()}
    counts = {bucket: int(value) for bucket, value in raw_counts.items()}
    remaining = total - sum(counts.values())
    remainders = sorted(
        raw_counts,
        key=lambda bucket: (raw_counts[bucket] - counts[bucket], bucket),
        reverse=True,
    )
    for bucket in remainders[:remaining]:
        counts[bucket] += 1
    return counts


def max_without_replacement_samples(
    buckets: Dict[str, List[int]],
    weights: Dict[str, float],
) -> int:
    limits = []
    for bucket, weight in weights.items():
        if weight <= 0:
            continue
        bucket_size = len(buckets.get(bucket, []))
        if bucket_size == 0:
            raise ValueError(f"Bucket {bucket!r} is empty but has weight={weight}")
        limits.append(bucket_size / weight)

    if not limits:
        raise ValueError("At least one positive bucket weight is required")

    target = int(min(limits))
    while target > 0:
        counts = allocate_counts(target, weights)
        if all(count <= len(buckets.get(bucket, [])) for bucket, count in counts.items()):
            return target
        target -= 1
    raise ValueError("Could not allocate any samples without replacement")


def update_distribution_stats(stats: Dict, item: Dict) -> None:
    actions = action_sequence_from_item(item)
    dataset = str(item.get("dataset", "unknown"))
    stats["samples"] += 1
    stats["datasets"][dataset] += 1
    stats["lengths"][len(actions)] += 1
    if actions:
        stats["first_action"][actions[0]] += 1
    for position, action in enumerate(actions, start=1):
        stats["action_token"][action] += 1
        stats["position_action"][str(position)][action] += 1
        if action == "stop":
            stats["stop_position"][str(position)] += 1
    stats["top_sequences"][" ".join(actions)] += 1
    stats["bucket"][assign_bucket(actions)] += 1


def counter_to_dict(value):
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, defaultdict):
        return {key: counter_to_dict(child) for key, child in value.items()}
    if isinstance(value, dict):
        return {key: counter_to_dict(child) for key, child in value.items()}
    return value


def init_distribution_stats() -> Dict:
    return {
        "samples": 0,
        "datasets": Counter(),
        "lengths": Counter(),
        "first_action": Counter(),
        "action_token": Counter(),
        "position_action": defaultdict(Counter),
        "stop_position": Counter(),
        "top_sequences": Counter(),
        "bucket": Counter(),
    }


def index_input(input_jsonl: str) -> Tuple[Dict[str, List[int]], Dict]:
    buckets = defaultdict(list)
    stats = init_distribution_stats()

    with open(input_jsonl, "rb") as handle:
        progress = tqdm(desc="index", dynamic_ncols=True)
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            item = json.loads(line)
            actions = action_sequence_from_item(item)
            bucket = assign_bucket(actions)
            buckets[bucket].append(offset)
            update_distribution_stats(stats, item)
            progress.update(1)
        progress.close()

    return dict(buckets), stats


def sample_offsets(
    buckets: Dict[str, List[int]],
    target_counts: Dict[str, int],
    rng: random.Random,
    with_replacement: bool = False,
) -> List[Tuple[str, int]]:
    sampled = []
    for bucket, target_count in target_counts.items():
        offsets = buckets.get(bucket, [])
        if target_count <= 0:
            continue
        if not offsets:
            raise ValueError(f"Bucket {bucket!r} is empty but target_count={target_count}")
        if with_replacement:
            bucket_offsets = [rng.choice(offsets) for _ in range(target_count)]
        else:
            if target_count > len(offsets):
                raise ValueError(
                    f"Bucket {bucket!r} needs {target_count} samples, "
                    f"but only {len(offsets)} are available without replacement"
                )
            bucket_offsets = rng.sample(offsets, target_count)
        sampled.extend((bucket, offset) for offset in bucket_offsets)
    rng.shuffle(sampled)
    return sampled


def write_sampled_jsonl(
    input_jsonl: str,
    output_jsonl: str,
    sampled_offsets: List[Tuple[str, int]],
) -> Dict:
    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    stats = init_distribution_stats()
    with open(input_jsonl, "rb") as input_handle, open(output_jsonl, "wb") as output_handle:
        for _, offset in tqdm(sampled_offsets, desc="write", dynamic_ncols=True):
            input_handle.seek(offset)
            line = input_handle.readline()
            item = json.loads(line)
            output_handle.write(line)
            update_distribution_stats(stats, item)
    return stats


def add_percentages(stats: Dict) -> Dict:
    result = counter_to_dict(stats)
    sample_count = max(1, int(stats["samples"]))
    token_count = max(1, sum(stats["action_token"].values()))
    result["length_percent"] = {
        key: round(value / sample_count * 100, 4)
        for key, value in result["lengths"].items()
    }
    result["first_action_percent"] = {
        key: round(value / sample_count * 100, 4)
        for key, value in result["first_action"].items()
    }
    result["action_token_percent"] = {
        key: round(value / token_count * 100, 4)
        for key, value in result["action_token"].items()
    }
    result["bucket_percent"] = {
        key: round(value / sample_count * 100, 4)
        for key, value in result["bucket"].items()
    }
    result["top_sequences"] = dict(stats["top_sequences"].most_common(30))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a balanced VLN action-chunk JSONL from an existing 1-4 action JSONL."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--stats_json", default=None)
    parser.add_argument("--target_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with_replacement",
        action="store_true",
        help="Sample with replacement. By default this script writes a true subset with no duplicate rows.",
    )
    parser.add_argument(
        "--weights",
        default="",
        help=(
            "Comma-separated bucket weights. Default: "
            + ",".join(f"{key}={value}" for key, value in DEFAULT_WEIGHTS.items())
        ),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    weights = normalize_weights(parse_weights(args.weights))
    buckets, input_stats = index_input(args.input_jsonl)
    if args.target_samples is None:
        if args.with_replacement:
            target_samples = int(input_stats["samples"])
        else:
            target_samples = max_without_replacement_samples(buckets, weights)
    else:
        target_samples = args.target_samples
    target_counts = allocate_counts(target_samples, weights)
    if not args.with_replacement:
        max_target_samples = max_without_replacement_samples(buckets, weights)
        if target_samples > max_target_samples:
            raise ValueError(
                f"target_samples={target_samples} is not feasible without replacement; "
                f"max feasible value for these weights is {max_target_samples}"
            )

    sampled_offsets = sample_offsets(
        buckets,
        target_counts,
        rng,
        with_replacement=args.with_replacement,
    )
    output_stats = write_sampled_jsonl(args.input_jsonl, args.output_jsonl, sampled_offsets)

    stats = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "seed": args.seed,
        "with_replacement": args.with_replacement,
        "weights": weights,
        "target_samples": target_samples,
        "target_counts": target_counts,
        "input_bucket_sizes": {bucket: len(offsets) for bucket, offsets in buckets.items()},
        "input": add_percentages(input_stats),
        "output": add_percentages(output_stats),
    }

    stats_json = args.stats_json or args.output_jsonl + ".stats.json"
    stats_dir = os.path.dirname(stats_json)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)
    with open(stats_json, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print(f"input={args.input_jsonl}")
    print(f"output={args.output_jsonl}")
    print(f"stats={stats_json}")
    print(f"seed={args.seed}")
    print(f"with_replacement={args.with_replacement}")
    print(f"target_samples={target_samples}")
    print(f"target_counts={target_counts}")
    print(f"input_bucket_sizes={stats['input_bucket_sizes']}")
    print(f"output_first_action={stats['output']['first_action_percent']}")
    print(f"output_action_token={stats['output']['action_token_percent']}")
    print(f"output_bucket={stats['output']['bucket_percent']}")


if __name__ == "__main__":
    main()
