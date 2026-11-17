#!/usr/bin/env bash
# Merge converted LeRobot datasets (in data/lerobot/) into a new one, without re-encoding videos.
# Usage: scripts/08_merge.sh <output name> <dataset name> <dataset name> ...
#   e.g. scripts/08_merge.sh so101_pick_place_all so101_pick_place so101_pick_place_b2
# The inputs stay as they are. All must have the same cameras, fps and joints.
set -euo pipefail
cd "$(dirname "$0")" && source ./config.sh
[[ $# -ge 3 ]] || { sed -n 3,4p "$0"; exit 1; }
OUT=$1; shift
LEROBOT="$A2_DIR/data/lerobot"
[[ -e "$LEROBOT/$OUT" ]] && { echo "$LEROBOT/$OUT already exists"; exit 1; }
IDS=$(printf "'local/%s', " "$@"); ROOTS=$(printf "'$LEROBOT/%s', " "$@")
lerobot-edit-dataset --new_repo_id "local/$OUT" --new_root "$LEROBOT/$OUT" \
  --operation.type merge --operation.repo_ids "[${IDS%, }]" --operation.roots "[${ROOTS%, }]"
python -c "import sys; from pathlib import Path; from dataset_summary import lerobot_summary; lerobot_summary(Path(sys.argv[1]))" "$LEROBOT/$OUT"
