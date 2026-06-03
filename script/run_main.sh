#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${1:-/mnt/data}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
python main.py --data-dir "$DATA_DIR" --results-dir "$SCRIPT_DIR/results"
