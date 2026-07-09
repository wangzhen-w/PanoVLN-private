import argparse
import gzip
import json
import os
import tarfile
from collections import Counter
from typing import Dict, Iterable, Optional, Tuple


DEFAULT_DATASET_ROOT = "/workspace/data1/dataset/Panoworld"
DEFAULT_RAW_JSONL = "training_data_1m_caption_aug.jsonl"
DEFAULT_PROCESSED_DIR = "."
DEFAULT_TRAIN_OUTPUT = "train_outdoor.jsonl"
DEFAULT_INDEX_OUTPUT = "outdoor_image_index.jsonl"
DEFAULT_EXTRACT_DIR = "images"


def _resolve_path(path: str, root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _jsonl_rows(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _source_basename_from_released_basename(released_basename: str) -> str:
    basename = os.path.basename(released_basename)
    if "__" in basename:
        return basename.split("__", 1)[1]
    return basename


def _safe_extract_tar(tar: tarfile.TarFile, destination: str) -> int:
    destination = os.path.abspath(destination)
    extracted = 0
    for member in tar:
        target_path = os.path.abspath(os.path.join(destination, member.name))
        if not target_path.startswith(destination + os.sep):
            raise ValueError(f"Unsafe tar member path: {member.name}")
        if not member.isfile():
            continue
        if os.path.exists(target_path) and os.path.getsize(target_path) == member.size:
            continue
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        source = tar.extractfile(member)
        if source is None:
            continue
        with source, open(target_path, "wb") as output:
            output.write(source.read())
        extracted += 1
    return extracted


def extract_tar_images(
    dataset_root: str,
    *,
    extract_dir: str,
    force: bool = False,
) -> None:
    images_dir = os.path.join(dataset_root, "images")
    tar_paths = sorted(
        os.path.join(images_dir, name)
        for name in os.listdir(images_dir)
        if name.endswith(".tar")
    )
    if not tar_paths:
        raise FileNotFoundError(f"No tar shards found under {images_dir}")

    extract_dir = _resolve_path(extract_dir, dataset_root)
    marker_dir = os.path.join(extract_dir, ".markers")
    os.makedirs(marker_dir, exist_ok=True)

    for shard_index, tar_path in enumerate(tar_paths, start=1):
        marker_path = os.path.join(marker_dir, os.path.basename(tar_path) + ".done")
        if os.path.exists(marker_path) and not force:
            continue
        if force and os.path.exists(marker_path):
            os.remove(marker_path)
        with tarfile.open(tar_path, "r") as tar:
            extracted = _safe_extract_tar(tar, extract_dir)
        with open(marker_path, "w", encoding="utf-8") as handle:
            handle.write(f"{os.path.basename(tar_path)}\n")
        print(
            f"[prepare] extracted shard {shard_index}/{len(tar_paths)} "
            f"{os.path.basename(tar_path)} ({extracted} new/updated files)",
            flush=True,
        )


def build_tar_image_index(
    dataset_root: str,
    *,
    output_path: Optional[str] = None,
    extract_dir: str = DEFAULT_EXTRACT_DIR,
) -> Tuple[Dict[str, dict], Dict[str, dict], Counter]:
    images_dir = os.path.join(dataset_root, "images")
    extract_dir_abs = _resolve_path(extract_dir, dataset_root)
    tar_paths = sorted(
        os.path.join(images_dir, name)
        for name in os.listdir(images_dir)
        if name.endswith(".tar")
    )
    if not tar_paths:
        raise FileNotFoundError(f"No tar shards found under {images_dir}")

    released_index: Dict[str, dict] = {}
    source_index: Dict[str, dict] = {}
    source_counts = Counter()
    total_members = 0

    index_handle = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        index_handle = open(output_path, "w", encoding="utf-8")

    try:
        for tar_path in tar_paths:
            relative_tar = os.path.relpath(tar_path, dataset_root)
            with tarfile.open(tar_path, "r") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    member_name = member.name
                    released_basename = os.path.basename(member_name)
                    source_basename = _source_basename_from_released_basename(released_basename)
                    extracted_path = os.path.relpath(
                        os.path.join(extract_dir_abs, member_name),
                        dataset_root,
                    )
                    entry = {
                        "released_basename": released_basename,
                        "source_basename": source_basename,
                        "tar": relative_tar,
                        "member": member_name,
                        "path": extracted_path,
                    }
                    total_members += 1
                    released_index[released_basename] = entry
                    source_counts[source_basename] += 1
                    if source_basename not in source_index:
                        source_index[source_basename] = entry
                    if index_handle is not None:
                        index_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    finally:
        if index_handle is not None:
            index_handle.close()

    duplicate_sources = Counter(
        {key: count for key, count in source_counts.items() if count > 1}
    )
    if duplicate_sources:
        preview = ", ".join(key for key, _ in duplicate_sources.most_common(5))
        print(
            f"[prepare] note: {sum(duplicate_sources.values())} images share "
            f"{len(duplicate_sources)} non-unique source basenames; "
            f"examples: {preview}",
            flush=True,
        )

    print(
        f"[prepare] indexed {len(released_index)} released images from {len(tar_paths)} tar shards "
        f"({total_members} file members)",
        flush=True,
    )
    return released_index, source_index, source_counts


def load_tar_image_index(
    index_path: str,
    *,
    dataset_root: str,
    extract_dir: str = DEFAULT_EXTRACT_DIR,
) -> Tuple[Dict[str, dict], Dict[str, dict], Counter]:
    extract_dir_abs = _resolve_path(extract_dir, dataset_root)
    released_index = {}
    source_index = {}
    source_counts = Counter()
    for row in _jsonl_rows(index_path):
        released_basename = row.get("released_basename")
        if not released_basename:
            member = row.get("member")
            if member:
                released_basename = os.path.basename(member)
                row["released_basename"] = released_basename
        if not row.get("path") and row.get("member"):
            row["path"] = os.path.relpath(
                os.path.join(extract_dir_abs, row["member"]),
                dataset_root,
            )
        source_basename = row.get("source_basename") or (
            _source_basename_from_released_basename(released_basename)
            if released_basename else None
        )
        if released_basename:
            released_index[released_basename] = row
        if source_basename:
            source_counts[source_basename] += 1
            if source_basename not in source_index:
                source_index[source_basename] = row
    if not released_index:
        raise ValueError(f"Image index is empty: {index_path}")
    print(f"[prepare] loaded {len(released_index)} image index entries from {index_path}", flush=True)
    return released_index, source_index, source_counts


def load_metadata_release_map(dataset_root: str) -> Dict[str, str]:
    metadata_dir = os.path.join(dataset_root, "metadata")
    metadata_paths = sorted(
        os.path.join(metadata_dir, name)
        for name in os.listdir(metadata_dir)
        if name.startswith("outdoor-") and name.endswith(".jsonl.gz")
    )
    release_map: Dict[str, str] = {}
    for metadata_path in metadata_paths:
        with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                released_path = row.get("released_image_path")
                released_basename = os.path.basename(released_path) if released_path else None
                if not released_basename:
                    continue
                for key in ("image_path", "fixed_source_image_path"):
                    source_path = row.get(key)
                    if source_path:
                        release_map[source_path] = released_basename
    print(f"[prepare] loaded {len(release_map)} metadata image-path mappings", flush=True)
    return release_map


def _image_to_local_ref(
    image_path: str,
    *,
    released_index: Dict[str, dict],
    source_index: Dict[str, dict],
    source_counts: Counter,
    metadata_release_map: Dict[str, str],
) -> Optional[str]:
    released_basename = metadata_release_map.get(image_path)
    if released_basename is not None:
        entry = released_index.get(released_basename)
        if entry is not None:
            return entry["path"]
        return None

    source_basename = os.path.basename(image_path)
    if source_counts[source_basename] != 1:
        return None
    entry = source_index.get(source_basename)
    if entry is None:
        return None
    return entry["path"]


def convert_training_jsonl(
    *,
    dataset_root: str,
    raw_jsonl: str,
    output_jsonl: str,
    released_index: Dict[str, dict],
    source_index: Dict[str, dict],
    source_counts: Counter,
    metadata_release_map: Dict[str, str],
    max_raw_rows: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> dict:
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    stats = Counter()
    task_family = Counter()
    generation_mode = Counter()

    with open(output_jsonl, "w", encoding="utf-8") as output:
        for raw_index, sample in enumerate(_jsonl_rows(raw_jsonl)):
            if max_raw_rows is not None and raw_index >= max_raw_rows:
                break
            stats["raw_rows"] += 1

            images = sample.get("images")
            if not isinstance(images, list) or not images:
                stats["missing_images_field"] += 1
                continue

            local_refs = []
            missing = False
            for image_path in images:
                if not isinstance(image_path, str) or not image_path:
                    missing = True
                    break
                local_ref = _image_to_local_ref(
                    image_path,
                    released_index=released_index,
                    source_index=source_index,
                    source_counts=source_counts,
                    metadata_release_map=metadata_release_map,
                )
                if local_ref is None:
                    missing = True
                    break
                local_refs.append(local_ref)

            if missing:
                stats["missing_local_image"] += 1
                continue

            output_sample = dict(sample)
            output_sample["images"] = local_refs
            output_sample.pop("source_images", None)
            output.write(json.dumps(output_sample, ensure_ascii=False) + "\n")

            stats["kept_rows"] += 1
            task_family[output_sample.get("task_family", "unknown")] += 1
            generation_mode[output_sample.get("generation_mode", "unknown")] += 1

            if max_samples is not None and stats["kept_rows"] >= max_samples:
                break

    stats_dict = dict(stats)
    stats_dict["task_family_top"] = task_family.most_common(20)
    stats_dict["generation_mode_top"] = generation_mode.most_common(20)
    stats_dict["dataset_root"] = dataset_root
    stats_dict["raw_jsonl"] = raw_jsonl
    stats_dict["output_jsonl"] = output_jsonl
    return stats_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local outdoor PanoWorld SFT jsonl with ordinary image refs.",
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--raw-jsonl", default=DEFAULT_RAW_JSONL)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--train-output", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--index-output", default=DEFAULT_INDEX_OUTPUT)
    parser.add_argument("--extract-dir", default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--max-raw-rows", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = os.path.abspath(args.dataset_root)
    raw_jsonl = _resolve_path(args.raw_jsonl, dataset_root)
    processed_dir = _resolve_path(args.processed_dir, dataset_root)
    output_jsonl = _resolve_path(args.train_output, processed_dir)
    index_path = _resolve_path(args.index_output, processed_dir)

    if not os.path.exists(raw_jsonl):
        raise FileNotFoundError(f"Raw PanoWorld jsonl not found: {raw_jsonl}")

    if not args.skip_extract:
        extract_tar_images(
            dataset_root,
            extract_dir=args.extract_dir,
            force=args.force_extract,
        )

    if args.reuse_index and os.path.exists(index_path):
        released_index, source_index, source_counts = load_tar_image_index(
            index_path,
            dataset_root=dataset_root,
            extract_dir=args.extract_dir,
        )
    else:
        released_index, source_index, source_counts = build_tar_image_index(
            dataset_root,
            output_path=index_path,
            extract_dir=args.extract_dir,
        )

    metadata_release_map = load_metadata_release_map(dataset_root)

    stats = convert_training_jsonl(
        dataset_root=dataset_root,
        raw_jsonl=raw_jsonl,
        output_jsonl=output_jsonl,
        released_index=released_index,
        source_index=source_index,
        source_counts=source_counts,
        metadata_release_map=metadata_release_map,
        max_raw_rows=args.max_raw_rows,
        max_samples=args.max_samples,
    )

    stats_path = output_jsonl + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(f"[prepare] wrote {output_jsonl}", flush=True)
    print(f"[prepare] wrote {stats_path}", flush=True)


if __name__ == "__main__":
    main()
