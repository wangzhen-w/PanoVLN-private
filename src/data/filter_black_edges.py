import argparse
import json
import shutil
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


DARK_THRESHOLD = 12
RESIZE_SIZE = (240, 120)


def frame_index(path):
    return int(path.stem.split("_")[-1])


def dark_ratio(image):
    if image.width == 0 or image.height == 0:
        return 0.0

    if image.mode != "RGB":
        image = image.convert("RGB")
    dark_mask = (np.asarray(image) < DARK_THRESHOLD).all(axis=2)
    return float(dark_mask.mean())


def image_black_stats(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize(RESIZE_SIZE)

    dark_mask = (np.asarray(image) < DARK_THRESHOLD).all(axis=2)
    height, width = dark_mask.shape
    top_height = max(1, height // 5)
    side_width = max(1, width // 16)

    top_dark = float(dark_mask[:top_height, :].mean())
    bottom_dark = float(dark_mask[height - top_height :, :].mean())
    left_dark = float(dark_mask[:, :side_width].mean())
    right_dark = float(dark_mask[:, width - side_width :].mean())
    return {
        "dark": float(dark_mask.mean()),
        "top_dark": top_dark,
        "bottom_dark": bottom_dark,
        "border_dark": max(top_dark, bottom_dark, left_dark, right_dark),
    }


def score_episode(stats):
    avg_dark = sum(item["dark"] for item in stats) / len(stats)
    max_dark = max(item["dark"] for item in stats)
    avg_top = sum(item["top_dark"] for item in stats) / len(stats)
    max_top = max(item["top_dark"] for item in stats)
    max_bottom = max(item["bottom_dark"] for item in stats)
    max_border = max(item["border_dark"] for item in stats)
    dirty_frame_count = sum(
        1
        for item in stats
        if item["dark"] >= 0.20
        or item["top_dark"] >= 0.45
        or item["border_dark"] >= 0.80
    )
    severe_frame_count = sum(
        1
        for item in stats
        if item["dark"] >= 0.30
        or item["top_dark"] >= 0.70
        or item["border_dark"] >= 0.90
    )
    dirty_frame_ratio = dirty_frame_count / len(stats)
    severe_frame_ratio = severe_frame_count / len(stats)

    reasons = []
    if dirty_frame_ratio >= 0.30:
        reasons.append(f"dirty_frame_ratio={dirty_frame_ratio:.3f}")
    if severe_frame_ratio >= 0.20:
        reasons.append(f"severe_frame_ratio={severe_frame_ratio:.3f}")
    if avg_dark >= 0.12:
        reasons.append(f"avg_dark={avg_dark:.3f}")
    if avg_top >= 0.25:
        reasons.append(f"avg_top={avg_top:.3f}")
    if max_dark >= 0.30:
        reasons.append(f"max_dark={max_dark:.3f}")
    if max_top >= 0.70:
        reasons.append(f"max_top={max_top:.3f}")
    if max_bottom >= 0.70:
        reasons.append(f"max_bottom={max_bottom:.3f}")
    if max_border >= 0.80:
        reasons.append(f"max_border={max_border:.3f}")

    remove = (
        dirty_frame_ratio >= 0.30
        and (avg_dark >= 0.12 or avg_top >= 0.25 or max_border >= 0.80)
    ) or severe_frame_ratio >= 0.20
    score = round(dirty_frame_ratio * 100 + severe_frame_ratio * 100)

    return {
        "score": score,
        "remove": remove,
        "reasons": reasons,
        "avg_dark": round(avg_dark, 4),
        "max_dark": round(max_dark, 4),
        "avg_top": round(avg_top, 4),
        "max_top": round(max_top, 4),
        "max_border": round(max_border, 4),
        "dirty_frame_count": dirty_frame_count,
        "severe_frame_count": severe_frame_count,
        "dirty_frame_ratio": round(dirty_frame_ratio, 4),
        "severe_frame_ratio": round(severe_frame_ratio, 4),
    }


def scan_episode(episode_dir):
    frame_paths = sorted(episode_dir.glob("frame_*.jpg"), key=frame_index)
    if not frame_paths:
        return {
            "episode_id": episode_dir.name,
            "frame_count": 0,
            "remove": True,
            "score": 999,
            "reasons": ["no_frames"],
        }

    stats = [image_black_stats(path) for path in frame_paths]
    result = score_episode(stats)
    result.update(
        {
            "episode_id": episode_dir.name,
            "frame_count": len(frame_paths),
        }
    )
    return result


def scan_image_dirs(image_root, num_workers):
    episode_dirs = sorted(
        [path for path in image_root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    print(f"Scanning {len(episode_dirs)} image dirs under {image_root}")

    workers = min(num_workers, max(1, cpu_count()))
    chunksize = max(1, min(32, len(episode_dirs) // max(1, workers * 4)))
    scan_results = []
    filtered_count = 0
    started_at = time.perf_counter()
    with Pool(workers) as pool:
        with tqdm(
            total=len(episode_dirs),
            desc="Scanning episodes",
            unit="episode",
            dynamic_ncols=True,
        ) as progress:
            for result in pool.imap_unordered(
                scan_episode,
                episode_dirs,
                chunksize=chunksize,
            ):
                scan_results.append(result)
                if result["remove"]:
                    filtered_count += 1
                progress.set_postfix(filtered_episodes=filtered_count)
                progress.update(1)

    scan_seconds = time.perf_counter() - started_at
    total_frame_count = sum(result.get("frame_count", 0) for result in scan_results)
    return episode_dirs, scan_results, {
        "total_frame_count": total_frame_count,
        "scan_seconds": round(scan_seconds, 4),
        "episodes_per_second": round(len(episode_dirs) / scan_seconds, 2)
        if scan_seconds > 0
        else 0.0,
        "frames_per_second": round(total_frame_count / scan_seconds, 2)
        if scan_seconds > 0
        else 0.0,
        "num_workers": workers,
        "chunksize": chunksize,
    }


def top_bad_examples(bad_results):
    return sorted(
        bad_results,
        key=lambda item: (-item["score"], -item.get("max_dark", 0.0), int(item["episode_id"])),
    )[:30]


def inspect_image_root(image_root, num_workers):
    _, scan_results, scan_summary = scan_image_dirs(image_root, num_workers)
    bad_results = sorted(
        [result for result in scan_results if result["remove"]],
        key=lambda item: int(item["episode_id"]),
    )
    bad_ids = {result["episode_id"] for result in bad_results}

    summary = {
        "dry_run": True,
        "image_root": str(image_root),
        "scanned_image_dir_count": len(scan_results),
        "would_remove_image_dir_count": len(bad_ids),
        "would_remove_episode_ids": sorted(bad_ids, key=int),
        **scan_summary,
        "top_bad_examples": top_bad_examples(bad_results),
    }
    print(
        "Scan complete: "
        f"scanned={summary['scanned_image_dir_count']} "
        f"filtered={summary['would_remove_image_dir_count']}"
    )
    return summary


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def clean_dataset(output_root, dataset_name, num_workers, dry_run=False):
    annotation_path = output_root / "sub_dataset" / f"{dataset_name}.jsonl"
    image_root = output_root / "images" / dataset_name
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    if not image_root.exists():
        raise FileNotFoundError(image_root)

    episode_dirs, scan_results, scan_summary = scan_image_dirs(image_root, num_workers)

    bad_results = sorted(
        [result for result in scan_results if result["remove"]],
        key=lambda item: int(item["episode_id"]),
    )
    bad_ids = {result["episode_id"] for result in bad_results}

    rows = load_jsonl(annotation_path)
    kept_rows = [row for row in rows if str(row["episode_id"]) not in bad_ids]
    removed_rows = [row for row in rows if str(row["episode_id"]) in bad_ids]
    if not dry_run:
        write_jsonl(annotation_path, kept_rows)

    removed_image_dirs = []
    if not dry_run:
        for episode_id in sorted(bad_ids, key=int):
            episode_image_dir = image_root / episode_id
            if episode_image_dir.is_dir():
                shutil.rmtree(episode_image_dir)
                removed_image_dirs.append(str(episode_image_dir))

    summary = {
        "dataset_name": dataset_name,
        "dry_run": dry_run,
        "annotation_path": str(annotation_path),
        "image_root": str(image_root),
        "before_count": len(rows),
        "after_count": len(rows) if dry_run else len(kept_rows),
        "would_remove_count": len(removed_rows),
        "removed_count": 0 if dry_run else len(removed_rows),
        "scanned_image_dir_count": len(episode_dirs),
        "would_remove_image_dir_count": len(bad_ids),
        "removed_image_dir_count": len(removed_image_dirs),
        "would_remove_episode_ids": sorted(
            [str(row["episode_id"]) for row in removed_rows], key=int
        ),
        "removed_episode_ids": []
        if dry_run
        else sorted([str(row["episode_id"]) for row in removed_rows], key=int),
        **scan_summary,
        "top_bad_examples": top_bad_examples(bad_results),
    }

    print(
        f"{dataset_name}: "
        f"before={summary['before_count']} "
        f"after={summary['after_count']} "
        f"filtered={summary['would_remove_count']} "
        f"image_dirs_filtered={summary['would_remove_image_dir_count']}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        default="/workspace/code_dir/a_property/dataset/PanoVLN",
    )
    parser.add_argument("--dataset_name", nargs="+", default=["scalevln"])
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--image_root",
        help="Scan one image root directly without reading annotations or deleting files.",
    )
    args = parser.parse_args()

    if args.image_root:
        inspect_image_root(Path(args.image_root), args.num_workers)
        return

    output_root = Path(args.output_root)
    [
        clean_dataset(
            output_root=output_root,
            dataset_name=dataset_name,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )
        for dataset_name in args.dataset_name
    ]


if __name__ == "__main__":
    main()
