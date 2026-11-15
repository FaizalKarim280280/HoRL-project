#!/usr/bin/env bash
# Live view of the wrist and scene cameras side by side (no robot needed). Use it to aim the scene camera.
# Keys: q / ESC quit, s save snapshots to snapshots/, g toggle a 3x3 grid.
set -euo pipefail
cd "$(dirname "$0")" && source ./config.sh
python camera_view.py "wrist=$WRIST_CAMERA" "scene=$SCENE_CAMERA" \
  --width $CAMERA_WIDTH --height $CAMERA_HEIGHT --fps $FPS "$@"
