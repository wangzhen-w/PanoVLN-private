"""Display aggregate and per-worker progress for recovery collection."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm


class IncrementalLineCounter:
    """Count appended JSONL rows without rereading the complete file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.inode: Optional[int] = None
        self.offset = 0
        self.lines = 0

    def value(self) -> int:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return self.lines
        if self.inode != stat.st_ino or stat.st_size < self.offset:
            self.inode = stat.st_ino
            self.offset = 0
            self.lines = 0
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            while chunk := handle.read(1024 * 1024):
                self.lines += chunk.count(b"\n")
            self.offset = handle.tell()
        return self.lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress_root", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--refresh_seconds", type=float, default=2.0)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one machine-readable snapshot instead of live progress bars.",
    )
    return parser.parse_args()


def load_configuration(progress_root: Path) -> dict:
    configuration_path = progress_root / "configuration.json"
    with configuration_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_totals(configuration: dict) -> tuple[Optional[int], list[Optional[int]]]:
    num_workers = int(configuration["num_workers"])
    uncapped = (
        configuration.get("max_routes_to_try") is None
        and configuration.get("max_samples_per_dataset") is None
    )
    if not uncapped:
        return None, [None] * num_workers
    total = sum(int(value) for value in configuration["route_inventory_counts"].values())
    worker_loads = [
        int(value)
        for value in configuration["worker_assignment"]["uncapped_route_loads"]
    ]
    if len(worker_loads) != num_workers or sum(worker_loads) != total:
        raise ValueError("worker route totals disagree with the route inventory")
    return total, worker_loads


def snapshot(
    success_counters: list[IncrementalLineCounter],
    outcome_counters: list[IncrementalLineCounter],
    image_root: Path,
) -> dict:
    success = [counter.value() for counter in success_counters]
    completed = [counter.value() for counter in outcome_counters]
    if any(ok > done for ok, done in zip(success, completed, strict=True)):
        raise ValueError("a worker manifest has more successes than terminal outcomes")
    free_gib = shutil.disk_usage(image_root).free / (1024**3)
    return {
        "completed": completed,
        "success": success,
        "failed": [
            done - ok for ok, done in zip(success, completed, strict=True)
        ],
        "free_gib": free_gib,
    }


def main() -> None:
    args = parse_args()
    if args.refresh_seconds <= 0:
        raise ValueError("refresh_seconds must be positive")
    configuration = load_configuration(args.progress_root)
    num_workers = int(configuration["num_workers"])
    total, worker_totals = expected_totals(configuration)
    success_counters = [
        IncrementalLineCounter(args.progress_root / f"rank_{index:02d}.jsonl")
        for index in range(num_workers)
    ]
    outcome_counters = [
        IncrementalLineCounter(
            args.progress_root / f"rank_{index:02d}.routes.jsonl"
        )
        for index in range(num_workers)
    ]

    first = snapshot(success_counters, outcome_counters, args.image_root)
    if args.once:
        print(
            json.dumps(
                {
                    "total": total,
                    "worker_totals": worker_totals,
                    **first,
                },
                ensure_ascii=False,
            )
        )
        return

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    initial_completed = sum(first["completed"])
    total_bar = tqdm(
        total=total,
        initial=initial_completed,
        desc="all routes",
        unit="route",
        dynamic_ncols=True,
        position=0,
        leave=True,
        smoothing=0.2,
    )
    worker_bars = [
        tqdm(
            total=worker_totals[index],
            initial=first["completed"][index],
            desc=f"worker {index:02d}",
            unit="route",
            dynamic_ncols=True,
            position=index + 1,
            leave=True,
            smoothing=0.2,
        )
        for index in range(num_workers)
    ]

    latest = first
    try:
        while True:
            latest = snapshot(success_counters, outcome_counters, args.image_root)
            completed_total = sum(latest["completed"])
            success_total = sum(latest["success"])
            failed_total = sum(latest["failed"])
            total_bar.update(completed_total - total_bar.n)
            yield_percent = 100.0 * success_total / max(completed_total, 1)
            total_bar.set_postfix_str(
                f"success={success_total} failed={failed_total} "
                f"yield={yield_percent:.1f}% free={latest['free_gib']:.0f}GiB",
                refresh=True,
            )
            for index, bar in enumerate(worker_bars):
                bar.update(latest["completed"][index] - bar.n)
                bar.set_postfix_str(
                    f"ok={latest['success'][index]} fail={latest['failed'][index]}",
                    refresh=True,
                )
            if stop_requested or (total is not None and completed_total >= total):
                break
            time.sleep(args.refresh_seconds)
    finally:
        for bar in worker_bars:
            bar.close()
        total_bar.close()


if __name__ == "__main__":
    main()
