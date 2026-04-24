import argparse
import json
import math
from pathlib import Path


TARGET_KEYS = ("success", "spl", "oracle_success", "distance_to_goal", "path_length", "ndtw")
RESULT_FILENAME = "result.jsonl"
RESULT_SUMMARY_FILENAME = "result_summary.json"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    output_path = Path(args.path)
    rows = load_rows(output_path)
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


if __name__ == "__main__":
    main()
