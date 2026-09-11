"""Combine compatible, disjoint trajectory collections into one training input."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path

import numpy as np

from dataset_create.trajectory.io_utils import atomic_json_dump, atomic_json_gz_dump, load_json, load_json_gz
from dataset_create.trajectory.schema import validate_dataset


def merge_collections(datasets, reports, name="trajectories"):
    if not datasets or len(datasets) != len(reports):
        raise ValueError("Each input dataset needs its collection statistics")
    metadata = None
    episodes, scenes = [], []
    for dataset, report in zip(datasets, reports):
        validate_dataset(dataset)
        current = {k: v for k, v in dataset["metadata"].items()
                   if k not in {"dataset_name", "split", "scene_count", "purpose"}}
        if metadata is not None and current != metadata:
            raise ValueError("Input collection metadata differs; replay and selection parameters must match")
        metadata = copy.deepcopy(current)
        if report["trajectories"] != len(dataset["episodes"]) or report["scene_count"] != len(report["scenes"]):
            raise ValueError("Input statistics do not match the dataset")
        counts = Counter(e["scene_id"] for e in dataset["episodes"])
        if any(counts.pop(s["scene_id"], 0) != s["trajectories"] for s in report["scenes"]) or counts:
            raise ValueError("Per-scene input counts do not match the dataset")
        episodes.extend(dataset["episodes"])
        scenes.extend(copy.deepcopy(report["scenes"]))
    if len({s["scene_id"] for s in scenes}) != len(scenes):
        raise ValueError("Input scene collections overlap")
    metadata.update(dataset_name=name, purpose="training", scene_count=len(scenes))
    result = {"schema_version": datasets[0]["schema_version"], "metadata": metadata,
              "episodes": sorted(episodes, key=lambda e: (e["scene_id"], e["trajectory_id"]))}
    validate_dataset(result)
    report = {"schema_version": result["schema_version"], "dataset": f"{name}.json.gz",
              "scene_count": len(scenes), "trajectories": len(episodes),
              "scenes": sorted(scenes, key=lambda s: s["scene_id"])}
    for key in ("regions", "connections", "decision_relations", "route_candidates", "route_families"):
        report[key] = sum(r[key] for r in reports)
    for key, metric in (("decision_event_histogram", "decision_event_count"), ("region_count_histogram", "region_count")):
        counts = Counter(int(e["metrics"][metric]) for e in episodes)
        report[key] = {str(k): counts[k] for k in sorted(counts)}
    report["diagnoses"] = dict(sorted(Counter(s["diagnosis"] for s in scenes).items()))
    lengths = np.asarray([e["metrics"]["length_m"] for e in episodes], dtype=float)
    report["length_m"] = {"minimum": float(lengths.min()) if len(lengths) else 0.,
                          "median": float(np.median(lengths)) if len(lengths) else 0.,
                          "p90": float(np.percentile(lengths, 90)) if len(lengths) else 0.,
                          "maximum": float(lengths.max()) if len(lengths) else 0.,
                          "mean": float(lengths.mean()) if len(lengths) else 0.}
    counts = sorted(s["trajectories"] for s in scenes)
    report["trajectories_per_scene"] = {"minimum": min(counts, default=0), "maximum": max(counts, default=0),
                                        "mean": len(episodes) / max(1, len(scenes))}
    for key, fraction in (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9)):
        report["trajectories_per_scene"][key] = counts[int((len(counts)-1)*fraction)] if counts else 0
    replays = [r.get("independent_replay", {}) for r in reports]
    if all(r.get("success_rate") == 1.0 and r.get("trajectories") == len(d["episodes"])
           for r, d in zip(replays, datasets)):
        report["independent_replay"] = {
            "trajectories": len(episodes), "success_rate": 1.0,
            "basis": "inherited from unchanged source episodes; not a new full replay",
            "maximum_errors": {k: max(r["maximum_errors"][k] for r in replays)
                               for k in replays[0]["maximum_errors"]},
        }
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--stats", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", default="trajectories")
    args = parser.parse_args(argv)
    if not args.name or Path(args.name).name != args.name:
        raise ValueError("name must be a filename component")
    dataset, report = merge_collections([load_json_gz(p) for p in args.datasets],
                                         [load_json(p) for p in args.stats], args.name)
    destination = args.output_root / f"{args.name}.json.gz"
    if destination.exists():
        if load_json_gz(destination) != dataset:
            raise ValueError(f"Refusing to replace a different existing dataset: {destination}")
    else:
        atomic_json_gz_dump(dataset, destination)
    atomic_json_dump(report, args.output_root / f"{args.name}_stats.json")
    print(f"Merged {len(dataset['episodes'])} trajectories from {report['scene_count']} scenes into {destination}")


if __name__ == "__main__":
    main()
