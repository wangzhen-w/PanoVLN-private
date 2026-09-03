"""Build matched action-sequence-length ablation datasets for R2R and RxR.

The manifest is sampled only from image trajectory identities and frame counts.
No instruction, action label, event class, or distance-to-tail information is
read until after the manifest has been fixed.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import random
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_INPUT_ROOT = Path("/workspace/code/a_property/dataset/PanoVLN")
DEFAULT_OUTPUT_DIR = Path("/workspace/data2/dataset/ablation/action_sequence")
DEFAULT_DATASETS = ("r2r", "rxr")
DEFAULT_ACTION_SEQUENCE_LENGTHS = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 24, 36, 50)
DEFAULT_SAMPLE_COUNT = 512_000
DEFAULT_SEED = 42
STOP_ACTION_ID = 0
ACTION_ID_TO_WORD = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}
FRAME_PATTERN = re.compile(r"^frame_(\d+)\.(?:png|jpe?g)$", re.IGNORECASE)
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_ENTRY_FIELDS = ("dataset", "episode_id", "start_index")


@dataclass(frozen=True)
class ImageTrajectory:
    dataset: str
    episode_id: str
    frame_names: tuple[str, ...]

    @property
    def start_count(self) -> int:
        # R2R/RxR extraction stores one observation for every action label,
        # including the terminal-stop observation.  Eligibility therefore
        # comes directly from the image count and never from action contents.
        return len(self.frame_names)

    def relative_history(self, start_index: int) -> list[str]:
        return [
            f"images/{self.dataset}/{self.episode_id}/{frame_name}"
            for frame_name in self.frame_names[: start_index + 1]
        ]


def _frame_sort_key(path: Path) -> int:
    match = FRAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unsupported frame filename: {path}")
    return int(match.group(1))


def _episode_sort_key(episode_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(episode_id))
    except ValueError:
        return (1, episode_id)


def scan_image_catalog(
    input_root: Path,
    datasets: Sequence[str],
) -> list[ImageTrajectory]:
    """Enumerate start positions using images only; annotations are untouched."""

    catalog: list[ImageTrajectory] = []
    for dataset in datasets:
        dataset_image_dir = input_root / "images" / dataset
        if not dataset_image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {dataset_image_dir}")
        episode_dirs = sorted(
            (path for path in dataset_image_dir.iterdir() if path.is_dir()),
            key=lambda path: _episode_sort_key(path.name),
        )
        if not episode_dirs:
            raise ValueError(f"No episode image directories found in {dataset_image_dir}")

        for episode_dir in episode_dirs:
            frame_paths = [
                path
                for path in episode_dir.iterdir()
                if path.is_file() and FRAME_PATTERN.fullmatch(path.name)
            ]
            frame_paths.sort(key=_frame_sort_key)
            if len(frame_paths) < 2:
                raise ValueError(
                    f"Trajectory must contain at least two frames: {episode_dir}"
                )
            frame_indices = [_frame_sort_key(path) for path in frame_paths]
            expected_indices = list(range(len(frame_paths)))
            if frame_indices != expected_indices:
                raise ValueError(
                    f"Non-contiguous frame indices in {episode_dir}: "
                    f"first={frame_indices[:3]}, last={frame_indices[-3:]}"
                )
            catalog.append(
                ImageTrajectory(
                    dataset=dataset,
                    episode_id=episode_dir.name,
                    frame_names=tuple(path.name for path in frame_paths),
                )
            )
    return catalog


def image_catalog_fingerprint(catalog: Sequence[ImageTrajectory]) -> str:
    digest = hashlib.sha256()
    for trajectory in catalog:
        digest.update(trajectory.dataset.encode("utf-8"))
        digest.update(b"\0")
        digest.update(trajectory.episode_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(trajectory.frame_names)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sample_manifest_entries(
    catalog: Sequence[ImageTrajectory],
    *,
    sample_count: int,
    seed: int,
) -> tuple[list[list[Any]], int]:
    """Uniformly sample unique image-derived start positions."""

    cumulative_ends: list[int] = []
    population_size = 0
    for trajectory in catalog:
        population_size += trajectory.start_count
        cumulative_ends.append(population_size)
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if sample_count > population_size:
        raise ValueError(
            "Cannot sample unique starts without replacement: "
            f"requested={sample_count}, population={population_size}"
        )

    sampled_population_indices = random.Random(seed).sample(
        range(population_size),
        sample_count,
    )
    entries: list[list[Any]] = []
    for population_index in sampled_population_indices:
        catalog_index = bisect.bisect_right(cumulative_ends, population_index)
        previous_end = cumulative_ends[catalog_index - 1] if catalog_index else 0
        trajectory = catalog[catalog_index]
        entries.append(
            [
                trajectory.dataset,
                trajectory.episode_id,
                population_index - previous_end,
            ]
        )
    return entries, population_size


def _atomic_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            handle.write("\n")
        os.replace(temporary_path, path)
        path.chmod(0o644)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def create_manifest(
    *,
    catalog: Sequence[ImageTrajectory],
    input_root: Path,
    datasets: Sequence[str],
    sample_count: int,
    seed: int,
    manifest_path: Path,
) -> dict[str, Any]:
    entries, population_size = sample_manifest_entries(
        catalog,
        sample_count=sample_count,
        seed=seed,
    )
    source_counts = Counter(entry[0] for entry in entries)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "entry_fields": list(MANIFEST_ENTRY_FIELDS),
        "sampling": "uniform_without_replacement_over_image_derived_start_indices",
        "manifest_generation_reads": [
            "dataset_name",
            "episode_image_directory_name",
            "frame_filename_and_count",
        ],
        "manifest_generation_excludes": [
            "actions",
            "instruction",
            "event_type",
            "distance_to_trajectory_tail",
        ],
        "seed": int(seed),
        "sample_count": int(sample_count),
        "population_size": int(population_size),
        "datasets": list(datasets),
        "source_counts": dict(sorted(source_counts.items())),
        "input_root": str(input_root.resolve()),
        "image_catalog_sha256": image_catalog_fingerprint(catalog),
        "entries": entries,
    }
    _atomic_json_dump(manifest_path, manifest)
    return manifest


def load_and_validate_manifest(
    *,
    manifest_path: Path,
    catalog: Sequence[ImageTrajectory],
    datasets: Sequence[str],
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_entries, expected_population_size = sample_manifest_entries(
        catalog,
        sample_count=sample_count,
        seed=seed,
    )
    expected_source_counts = Counter(entry[0] for entry in expected_entries)
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "entry_fields": list(MANIFEST_ENTRY_FIELDS),
        "sampling": "uniform_without_replacement_over_image_derived_start_indices",
        "seed": int(seed),
        "sample_count": int(sample_count),
        "population_size": expected_population_size,
        "datasets": list(datasets),
        "source_counts": dict(sorted(expected_source_counts.items())),
        "image_catalog_sha256": image_catalog_fingerprint(catalog),
    }
    for field_name, expected_value in expected.items():
        actual_value = manifest.get(field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"Existing manifest {field_name} mismatch: "
                f"expected={expected_value!r}, actual={actual_value!r}. "
                "Use --overwrite-manifest to regenerate it."
            )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != sample_count:
        raise ValueError(
            f"Existing manifest has invalid entries: expected {sample_count}"
        )
    if entries != expected_entries:
        first_mismatch = next(
            (
                index
                for index, (actual, expected_entry) in enumerate(
                    zip(entries, expected_entries)
                )
                if actual != expected_entry
            ),
            None,
        )
        raise ValueError(
            "Existing manifest entries do not match the deterministic "
            f"seed={seed} sample; first_mismatch={first_mismatch}. "
            "Use --overwrite-manifest to regenerate it."
        )
    return manifest


def load_annotations(
    input_root: Path,
    datasets: Sequence[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in datasets:
        annotation_path = input_root / "sub_dataset" / f"{dataset}.jsonl"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                episode_id = str(row.get("episode_id", ""))
                if not episode_id:
                    raise ValueError(
                        f"Missing episode_id in {annotation_path}:{line_number}"
                    )
                key = (dataset, episode_id)
                if key in annotations:
                    raise ValueError(f"Duplicate annotation identity: {key}")
                annotations[key] = row
    return annotations


def _validate_lengths(lengths: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(int(length) for length in lengths)
    if not normalized or any(length <= 0 for length in normalized):
        raise ValueError(f"All action sequence lengths must be positive: {normalized}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate action sequence lengths: {normalized}")
    return normalized


def _manifest_entry(entry: Any) -> tuple[str, str, int]:
    if not isinstance(entry, list) or len(entry) != len(MANIFEST_ENTRY_FIELDS):
        raise ValueError(f"Invalid manifest entry: {entry!r}")
    dataset, episode_id, start_index = entry
    if not isinstance(dataset, str) or not isinstance(episode_id, str):
        raise ValueError(f"Invalid manifest identity: {entry!r}")
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise ValueError(f"Invalid manifest start_index: {entry!r}")
    return dataset, episode_id, start_index


def _padded_action_target(
    actions: Sequence[int],
    *,
    start_index: int,
    action_sequence_length: int,
) -> tuple[list[str], int]:
    action_ids = [int(action_id) for action_id in actions[start_index:]]
    action_ids = action_ids[:action_sequence_length]
    if not action_ids:
        raise ValueError(f"No action label at start_index={start_index}")
    if len(action_ids) < action_sequence_length:
        if action_ids[-1] != STOP_ACTION_ID:
            raise ValueError(
                "Only a naturally terminal suffix may be STOP padded: "
                f"start={start_index}, suffix={action_ids}"
            )
        action_ids.extend(
            [STOP_ACTION_ID] * (action_sequence_length - len(action_ids))
        )
    invalid_ids = [action_id for action_id in action_ids if action_id not in ACTION_ID_TO_WORD]
    if invalid_ids:
        raise ValueError(f"Unsupported action ids: {invalid_ids}")

    try:
        first_stop_index = action_ids.index(STOP_ACTION_ID)
    except ValueError:
        real_action_count = action_sequence_length
    else:
        if any(action_id != STOP_ACTION_ID for action_id in action_ids[first_stop_index:]):
            raise ValueError(f"Executable action appears after stop: {action_ids}")
        real_action_count = first_stop_index + 1
    return [ACTION_ID_TO_WORD[action_id] for action_id in action_ids], real_action_count


def materialize_datasets(
    *,
    manifest: dict[str, Any],
    catalog: Sequence[ImageTrajectory],
    annotations: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    action_sequence_lengths: Sequence[int],
    overwrite: bool,
) -> dict[str, Any]:
    lengths = _validate_lengths(action_sequence_lengths)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        length: output_dir / f"train_r2r_rxr_action_sequence_length_{length}.jsonl"
        for length in lengths
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing ablation datasets: "
            f"{existing}. Pass --overwrite to replace all requested lengths."
        )

    catalog_by_key = {
        (trajectory.dataset, trajectory.episode_id): trajectory
        for trajectory in catalog
    }
    temporary_paths: dict[int, Path] = {}
    handles: dict[int, Any] = {}
    stop_padded_counts = Counter()
    source_counts = Counter()
    start_time = time.perf_counter()
    try:
        for length, output_path in output_paths.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_dir,
                text=True,
            )
            temporary_paths[length] = Path(temporary_name)
            handles[length] = os.fdopen(descriptor, "w", encoding="utf-8")

        entries = manifest["entries"]
        for row_index, raw_entry in enumerate(entries, start=1):
            dataset, episode_id, start_index = _manifest_entry(raw_entry)
            key = (dataset, episode_id)
            trajectory = catalog_by_key.get(key)
            annotation = annotations.get(key)
            if trajectory is None:
                raise ValueError(f"Manifest trajectory is missing from image catalog: {key}")
            if annotation is None:
                raise ValueError(f"Manifest trajectory is missing from annotations: {key}")
            if start_index < 0 or start_index >= trajectory.start_count:
                raise ValueError(
                    f"Manifest start_index is outside image trajectory: {raw_entry!r}"
                )

            actions = annotation.get("actions")
            instruction = annotation.get("instruction")
            if not isinstance(actions, list) or not actions:
                raise ValueError(f"Annotation has no action sequence: {key}")
            actions = [int(action_id) for action_id in actions]
            if actions[-1] != STOP_ACTION_ID or STOP_ACTION_ID in actions[:-1]:
                raise ValueError(f"Annotation must have one terminal stop: {key}")
            if len(actions) != trajectory.start_count:
                raise ValueError(
                    "Image/action trajectory length mismatch: "
                    f"{key}, image_starts={trajectory.start_count}, actions={len(actions)}"
                )
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Annotation has no instruction: {key}")

            image_history = trajectory.relative_history(start_index)
            history_actions = [
                ACTION_ID_TO_WORD[action_id]
                for action_id in actions[:start_index]
            ]
            source_counts[dataset] += 1
            remaining_action_count = len(actions) - start_index
            for length in lengths:
                action_sequence, real_action_count = _padded_action_target(
                    actions,
                    start_index=start_index,
                    action_sequence_length=length,
                )
                if remaining_action_count < length:
                    stop_padded_counts[length] += 1
                # Preserve the exact compact-field schema and field order used
                # by the regular PanoVLN training JSONL files.
                row = {
                    "instruction": instruction,
                    "action_sequence": action_sequence,
                    "images": image_history,
                    "episode_id": episode_id,
                    "dataset": dataset,
                    "step_index": start_index,
                    "end_step": start_index + real_action_count - 1,
                    "real_action_count": real_action_count,
                    "history_actions": history_actions,
                }
                handles[length].write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            if row_index % 25_000 == 0 or row_index == len(entries):
                elapsed = time.perf_counter() - start_time
                print(
                    f"materialized {row_index}/{len(entries)} matched starts "
                    f"in {elapsed:.1f}s",
                    flush=True,
                )

        for handle in handles.values():
            handle.close()
        handles.clear()
        for length, output_path in output_paths.items():
            os.replace(temporary_paths[length], output_path)
            output_path.chmod(0o644)
    except Exception:
        for handle in handles.values():
            handle.close()
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise

    return {
        "sample_count_per_length": len(manifest["entries"]),
        "source_counts": dict(sorted(source_counts.items())),
        "stop_padded_sample_counts": {
            str(length): stop_padded_counts[length] for length in lengths
        },
        "datasets": {
            str(length): str(output_paths[length]) for length in lengths
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reusable_dataset_summary(
    *,
    summary_path: Path,
    manifest_path: Path,
    output_paths: dict[int, Path],
    existing_lengths: Sequence[int],
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise FileNotFoundError(
            "Existing ablation datasets require dataset_summary.json for safe reuse: "
            f"{summary_path}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Existing ablation datasets require the original manifest: {manifest_path}"
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    expected_metadata = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "seed": int(seed),
        "sample_count_per_length": int(sample_count),
    }
    for field_name, expected_value in expected_metadata.items():
        if summary.get(field_name) != expected_value:
            raise ValueError(
                f"Cannot safely reuse existing datasets: {field_name} mismatch; "
                f"expected={expected_value!r}, actual={summary.get(field_name)!r}"
            )

    recorded_paths = summary.get("datasets", {})
    recorded_hashes = summary.get("dataset_sha256", {})
    for key, recorded_path in recorded_paths.items():
        dataset_path = Path(recorded_path).resolve()
        if not recorded_hashes.get(key):
            raise ValueError(
                f"Cannot safely reuse length={key}: missing recorded SHA256"
            )
        if not dataset_path.is_file() or dataset_path.stat().st_size <= 0:
            raise ValueError(
                f"Cannot safely reuse missing or empty dataset: {dataset_path}"
            )
    for length in existing_lengths:
        key = str(length)
        output_path = output_paths[length]
        recorded_path = recorded_paths.get(key)
        if recorded_path is None or Path(recorded_path).resolve() != output_path:
            raise ValueError(
                f"Cannot safely reuse length={length}: dataset_summary.json path mismatch"
            )
    return summary


def merge_dataset_summaries(
    existing_summary: dict[str, Any] | None,
    new_summary: dict[str, Any],
) -> dict[str, Any]:
    if existing_summary is None:
        return new_summary

    merged_summary = dict(new_summary)
    for field_name in (
        "stop_padded_sample_counts",
        "datasets",
        "dataset_sha256",
    ):
        merged_values = dict(existing_summary.get(field_name, {}))
        merged_values.update(new_summary.get(field_name, {}))
        merged_summary[field_name] = merged_values
    merged_summary["action_sequence_lengths"] = sorted(
        int(length) for length in merged_summary["datasets"]
    )
    return merged_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create matched R2R/RxR action-sequence-length ablation data."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_ACTION_SEQUENCE_LENGTHS),
    )
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    datasets = tuple(str(dataset) for dataset in args.datasets)
    lengths = _validate_lengths(args.lengths)
    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path is not None
        else output_dir / "manifest.json"
    )
    summary_path = output_dir / "dataset_summary.json"
    output_paths = {
        length: output_dir / f"train_r2r_rxr_action_sequence_length_{length}.jsonl"
        for length in lengths
    }
    existing_summary = None
    lengths_to_materialize = lengths
    if not args.manifest_only and not args.overwrite:
        existing_lengths = tuple(
            length for length, path in output_paths.items() if path.is_file()
        )
        missing_lengths = tuple(
            length for length, path in output_paths.items() if not path.is_file()
        )
        if existing_lengths or summary_path.is_file():
            if args.overwrite_manifest:
                raise ValueError(
                    "--overwrite-manifest cannot reuse existing datasets; "
                    "also pass --overwrite to regenerate the requested lengths"
                )
            existing_summary = load_reusable_dataset_summary(
                summary_path=summary_path,
                manifest_path=manifest_path,
                output_paths=output_paths,
                existing_lengths=existing_lengths,
                sample_count=args.sample_count,
                seed=args.seed,
            )
            if existing_lengths:
                print(
                    f"Reusing completed datasets for lengths={existing_lengths}",
                    flush=True,
                )
        if not missing_lengths:
            print("All requested action-sequence datasets already exist; nothing to do.")
            return
        lengths_to_materialize = missing_lengths
        print(
            f"Materializing only missing lengths={lengths_to_materialize}",
            flush=True,
        )

    print(
        "Scanning image-only start-index population "
        f"for datasets={datasets} under {input_root}",
        flush=True,
    )
    catalog = scan_image_catalog(input_root, datasets)
    print(
        f"Image catalog trajectories={len(catalog)} "
        f"start_population={sum(item.start_count for item in catalog)}",
        flush=True,
    )

    if manifest_path.exists() and not args.overwrite_manifest:
        manifest = load_and_validate_manifest(
            manifest_path=manifest_path,
            catalog=catalog,
            datasets=datasets,
            sample_count=args.sample_count,
            seed=args.seed,
        )
        print(f"Reusing validated manifest: {manifest_path}", flush=True)
    else:
        manifest = create_manifest(
            catalog=catalog,
            input_root=input_root,
            datasets=datasets,
            sample_count=args.sample_count,
            seed=args.seed,
            manifest_path=manifest_path,
        )
        print(f"Wrote action-independent manifest: {manifest_path}", flush=True)

    if args.manifest_only:
        return

    # Action/instruction annotations are deliberately loaded only after the
    # random manifest has been created or validated.
    annotations = load_annotations(input_root, datasets)
    summary = materialize_datasets(
        manifest=manifest,
        catalog=catalog,
        annotations=annotations,
        output_dir=output_dir,
        action_sequence_lengths=lengths_to_materialize,
        overwrite=args.overwrite,
    )
    summary.update(
        manifest=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        seed=args.seed,
        sampling=manifest["sampling"],
        action_sequence_lengths=list(lengths),
    )
    summary["dataset_sha256"] = {
        length: sha256_file(Path(path))
        for length, path in summary["datasets"].items()
    }
    summary = merge_dataset_summaries(existing_summary, summary)
    _atomic_json_dump(summary_path, summary)
    print(f"Wrote dataset summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
