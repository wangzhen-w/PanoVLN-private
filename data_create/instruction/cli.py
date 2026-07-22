"""Command line interface for the PanoVLN instruction generator."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .io_utils import str2bool
from .runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--records-output-jsonl",
        default=None,
        help=(
            "Optional canonical per-episode generation record. Candidate journals "
            "are deduplicated by episode and atomically published here; a .gz "
            "suffix enables gzip compression."
        ),
    )
    parser.add_argument(
        "--summary-output-json",
        default=None,
        help="Optional persistent copy of the final run summary.",
    )
    parser.add_argument("--mode", choices=("generate", "rewrite"), default="generate")
    parser.add_argument("--instruction-profile", choices=("concise", "dense"), default="concise")

    parser.add_argument("--provider", default="qwen")
    parser.add_argument("--base-url", default="http://127.0.0.1:10420/v1")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--api-key", default="test")
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--stage-retries", type=int, default=2)
    parser.add_argument("--api-preflight", type=str2bool, default=True)
    parser.add_argument("--disable-thinking", type=str2bool, default=True)

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--evidence-workers",
        type=int,
        default=0,
        help=(
            "CPU processes that prebuild visual evidence before API generation. "
            "0 keeps the legacy all-in-one thread execution."
        ),
    )
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--episode-ids", default=None)
    parser.add_argument("--sample-mode", choices=("first", "random", "stratified"), default="first")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-waypoints", type=int, default=14)
    parser.add_argument("--route-evidence-mode", choices=("sheet", "segmented", "auto"), default="auto")
    parser.add_argument("--segmented-min-actions", type=int, default=80)
    parser.add_argument("--segment-max-waypoints", type=int, default=0, help="0 means reuse --max-waypoints")
    parser.add_argument("--segment-rows", type=int, default=5)
    parser.add_argument("--segment-overlap", type=int, default=1)
    parser.add_argument("--segment-fact-max-tokens", type=int, default=280)
    parser.add_argument("--start-window-frames", type=int, default=5)
    parser.add_argument("--endpoint-window-frames", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=384)
    parser.add_argument("--tile-height", type=int, default=288)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--use-action-heading", type=str2bool, default=False)
    parser.add_argument("--save-contact-sheets", action="store_true")
    parser.add_argument("--write-gallery", action="store_true")
    parser.add_argument("--gallery-path", default=None)
    parser.add_argument("--keep-raw-responses", type=str2bool, default=False)

    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=360)
    parser.add_argument("--planner-temperature", type=float, default=0.0)
    parser.add_argument("--planner-max-tokens", type=int, default=1000)
    parser.add_argument("--review-temperature", type=float, default=0.0)
    parser.add_argument("--review-max-tokens", type=int, default=700)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--candidate-temperature", type=float, default=0.4)

    # Backwards-compatible flags retained so existing scripts do not break.
    parser.add_argument("--fact-max-tokens", type=int, default=320)
    parser.add_argument("--start-fact-pass", type=str2bool, default=True)
    parser.add_argument("--endpoint-fact-pass", type=str2bool, default=True)
    parser.add_argument("--route-plan-pass", type=str2bool, default=True)
    parser.add_argument("--require-start-facts", type=str2bool, default=True)
    parser.add_argument("--require-endpoint-facts", type=str2bool, default=True)
    parser.add_argument("--require-route-plan", type=str2bool, default=True)
    parser.add_argument("--self-check", type=str2bool, default=True)
    parser.add_argument("--blind-grounding-audit", type=str2bool, default=True)
    parser.add_argument("--route-audit", type=str2bool, default=True)
    parser.add_argument("--spatial-audit", type=str2bool, default=True)
    parser.add_argument(
        "--drop-failed",
        type=str2bool,
        default=False,
        help=(
            "Treat failures that exhausted the quality/repair gates as terminal: "
            "skip them on resume and omit them from the clean JSONL. Runtime/API "
            "failures remain retryable."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        type=str2bool,
        default=False,
        help=(
            "Permit publishing while retryable runtime/incomplete episodes remain. "
            "Terminal quality drops do not require this option."
        ),
    )
    parser.add_argument("--save-failed-contact-sheets", type=str2bool, default=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.provider != "qwen":
        raise ValueError("Only the OpenAI-compatible Qwen provider is implemented")
    try:
        summary = run(args)
    except KeyboardInterrupt:
        # Completed candidate journals are already closed. Exit immediately
        # instead of waiting for active HTTP threads; the next run resumes by
        # episode_id from those journals.
        print("\n[instruction] interrupted; resumable progress has been preserved", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
