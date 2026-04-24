#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"
echo "Switched to directory: $PROJECT_ROOT"

SAVE_PATH="eval_log/navida_r2r"

echo "$SAVE_PATH"
python src/eval/analyze_results.py \
    --path "$SAVE_PATH"
