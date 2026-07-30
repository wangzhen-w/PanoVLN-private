import argparse
import json
import math
from pathlib import Path

import numpy as np


TARGET_KEYS = ("success", "spl", "oracle_success", "distance_to_goal", "path_length", "ndtw")
METRIC_DIRECTIONS = {
    "success": "higher",
    "spl": "higher",
    "oracle_success": "higher",
    "distance_to_goal": "lower",
    # Unconditional path length is descriptive: a failed agent that stops
    # immediately can be shorter without being better.
    "path_length": "descriptive",
    "ndtw": "higher",
}
RESULT_FILENAME = "result.jsonl"
RESULT_SUMMARY_FILENAME = "result_summary.json"
PAIRED_COMPARISON_FILENAME = "paired_comparison.json"


def check_inf_nan(value):
    if math.isinf(value) or math.isnan(value):
        return 0.0
    return value


def iter_result_paths(output_path):
    paths = []
    merged_path = output_path / RESULT_FILENAME
    if merged_path.exists():
        paths.append(merged_path)
    paths.extend(sorted(output_path.glob("result_rank*.jsonl")))
    return paths


def reorder_row(row):
    ordered = {}
    for key in ("id", "scene_id"):
        if key in row:
            ordered[key] = row[key]
    for key in TARGET_KEYS:
        if key in row:
            ordered[key] = row[key]
    for key, value in row.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def load_rows(output_path):
    row_map = {}
    for path in iter_result_paths(output_path):
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episode_id = row.get("id", row.get("episode_id"))
                scene_id = row.get("scene_id")
                if episode_id is None or scene_id is None:
                    continue
                key = (str(scene_id), str(episode_id))
                row["id"] = str(episode_id)
                row["scene_id"] = str(scene_id)
                row_map[key] = reorder_row(row)
    return [row_map[key] for key in sorted(row_map.keys())]


def summarize_rows(rows):
    summary = {"num_episodes": len(rows)}
    for key in TARGET_KEYS:
        values = [
            check_inf_nan(float(row[key]))
            for row in rows
            if key in row
        ]
        summary[key] = float(sum(values) / len(values)) if values else 0.0
    return summary


def write_outputs(output_path, rows, summary):
    merged_path = output_path / RESULT_FILENAME
    with merged_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(reorder_row(row), ensure_ascii=False) + "\n")

    summary_path = output_path / RESULT_SUMMARY_FILENAME
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    for shard_path in sorted(output_path.glob("result_rank*.jsonl")):
        shard_path.unlink()


def _row_key(row):
    return str(row["scene_id"]), str(row["id"])


def paired_bootstrap_comparison(
    baseline_rows,
    candidate_rows,
    *,
    num_samples,
    seed,
):
    baseline_map = {_row_key(row): row for row in baseline_rows}
    candidate_map = {_row_key(row): row for row in candidate_rows}
    paired_keys = sorted(set(baseline_map) & set(candidate_map))
    if not paired_keys:
        raise ValueError("Baseline and candidate have no paired episodes")

    rng = np.random.default_rng(seed)
    metrics = {}
    for metric_name in TARGET_KEYS:
        pairs = [
            (
                check_inf_nan(float(baseline_map[key][metric_name])),
                check_inf_nan(float(candidate_map[key][metric_name])),
            )
            for key in paired_keys
            if metric_name in baseline_map[key]
            and metric_name in candidate_map[key]
        ]
        if not pairs:
            continue
        values = np.asarray(pairs, dtype=np.float64)
        raw_differences = values[:, 1] - values[:, 0]
        bootstrap_means = np.empty(int(num_samples), dtype=np.float64)
        chunk_size = 256
        for start in range(0, int(num_samples), chunk_size):
            stop = min(start + chunk_size, int(num_samples))
            indices = rng.integers(
                0,
                raw_differences.size,
                size=(stop - start, raw_differences.size),
            )
            bootstrap_means[start:stop] = raw_differences[indices].mean(
                axis=1
            )

        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        direction = METRIC_DIRECTIONS[metric_name]
        if direction == "higher":
            improvement_probability = float(
                (bootstrap_means > 0.0).mean()
            )
        elif direction == "lower":
            improvement_probability = float(
                (bootstrap_means < 0.0).mean()
            )
        else:
            improvement_probability = None
        metrics[metric_name] = {
            "num_paired_episodes": int(raw_differences.size),
            "baseline_mean": float(values[:, 0].mean()),
            "candidate_mean": float(values[:, 1].mean()),
            "candidate_minus_baseline": float(raw_differences.mean()),
            "paired_bootstrap_95_ci": [float(lower), float(upper)],
            "direction": direction,
            "bootstrap_probability_of_improvement": improvement_probability,
        }

    return {
        "num_common_episodes": len(paired_keys),
        "num_baseline_episodes": len(baseline_map),
        "num_candidate_episodes": len(candidate_map),
        "bootstrap_samples": int(num_samples),
        "seed": int(seed),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help="optional Pano-only result directory for paired bootstrap",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.path)
    rows = load_rows(output_path)
    baseline_rows = (
        load_rows(Path(args.baseline_path))
        if args.baseline_path is not None
        else None
    )
    summary = summarize_rows(rows)
    write_outputs(output_path, rows, summary)

    num_rows = len(rows)
    success_total = sum(int(check_inf_nan(float(row.get("success", 0.0)))) for row in rows)
    oracle_total = sum(int(check_inf_nan(float(row.get("oracle_success", 0.0)))) for row in rows)

    if num_rows == 0:
        print("No results found.")
        return

    print(f"Success rate: {success_total}/{num_rows} ({summary['success']:.3f})")
    print(f"Oracle success rate: {oracle_total}/{num_rows} ({summary['oracle_success']:.3f})")
    print(f"SPL: {summary['spl']:.3f}")
    print(f"Distance to goal: {summary['distance_to_goal']:.3f}")
    print(f"Path length: {summary['path_length']:.3f}")
    print(f"ndtw: {summary['ndtw']:.3f}")

    if baseline_rows is not None:
        if args.bootstrap_samples <= 0:
            raise ValueError("--bootstrap-samples must be positive")
        comparison = paired_bootstrap_comparison(
            baseline_rows,
            rows,
            num_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        comparison_path = output_path / PAIRED_COMPARISON_FILENAME
        comparison_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "Paired bootstrap: "
            f"{comparison['num_common_episodes']} common episodes, "
            f"saved to {comparison_path}"
        )


if __name__ == "__main__":
    main()
