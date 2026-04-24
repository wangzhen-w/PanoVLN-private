import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional


ACTIONS = ("move_forward", "turn_left", "turn_right", "stop")


def extract_action(sample: Dict) -> Optional[str]:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, list):
            for item in reversed(content):
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text", "")).strip().lower()
                    if text in ACTIONS:
                        return text
        elif isinstance(content, str):
            text = content.strip().lower()
            if text in ACTIONS:
                return text
    return None


def parse_target_counts(raw: str) -> Dict[str, int]:
    target_counts = {action: 0 for action in ACTIONS}
    if not raw.strip():
        raise ValueError("--target-counts must not be empty")

    for part in raw.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(
                f"Invalid target-count item '{piece}'. Expected format action=value."
            )
        action, value = piece.split("=", 1)
        action = action.strip()
        if action not in ACTIONS:
            raise ValueError(f"Unknown action '{action}'. Valid actions: {', '.join(ACTIONS)}")
        target_counts[action] = int(value.strip())
        if target_counts[action] < 0:
            raise ValueError(f"Target count for '{action}' must be >= 0")

    return target_counts


def load_grouped_samples(input_path: str) -> Dict[str, List[str]]:
    grouped_samples = defaultdict(list)
    with open(input_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            sample = json.loads(line)
            action = extract_action(sample)
            if action is None:
                continue
            grouped_samples[action].append(line)
    return grouped_samples


def sample_to_target_count(
    items: List[str],
    target_count: int,
    rng: random.Random,
) -> List[str]:
    if target_count <= 0 or not items:
        return []
    if len(items) >= target_count:
        return rng.sample(items, target_count)

    sampled = list(items)
    sampled.extend(rng.choices(items, k=target_count - len(items)))
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target-counts", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    target_counts = parse_target_counts(args.target_counts)
    grouped_samples = load_grouped_samples(args.input)

    merged_samples: List[str] = []
    for action in ACTIONS:
        merged_samples.extend(
            sample_to_target_count(
                items=grouped_samples.get(action, []),
                target_count=target_counts[action],
                rng=rng,
            )
        )

    if args.shuffle:
        rng.shuffle(merged_samples)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as handle:
        for line in merged_samples:
            handle.write(line)
            handle.write("\n")

    written_counts = {action: target_counts[action] for action in ACTIONS}
    available_counts = {action: len(grouped_samples.get(action, [])) for action in ACTIONS}
    print(json.dumps(
        {
            "input": args.input,
            "output": args.output,
            "available_counts": available_counts,
            "written_counts": written_counts,
            "written_total": len(merged_samples),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
