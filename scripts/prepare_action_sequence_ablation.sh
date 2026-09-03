#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="/opt/conda/bin/python"
INPUT_ROOT="/workspace/code/a_property/dataset/PanoVLN"
OUTPUT_DIR="/workspace/data2/dataset/ablation/action_sequence"

"$PYTHON_BIN" src/data/prepare_action_sequence_ablation.py \
    --input-root "$INPUT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --datasets r2r rxr \
    --lengths 1 2 4 6 8 10 12 14 16 18 24 36 50 \
    --sample-count 512000 \
    --seed 42 \
    "$@"
