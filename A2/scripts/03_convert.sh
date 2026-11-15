#!/usr/bin/env bash
# Convert the raw demonstrations into a LeRobot dataset (encodes the videos; run once after recording).
# ./03_convert.sh --overwrite rebuilds it after you recorded more episodes.
set -euo pipefail
cd "$(dirname "$0")" && source ./config.sh
python convert_to_lerobot.py --raw "$RAW_ROOT" --out "$DATASET_ROOT" --repo-id "$DATASET_REPO_ID" "$@"
