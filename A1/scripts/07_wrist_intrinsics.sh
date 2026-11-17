#!/usr/bin/env bash
# Step 1: wrist-camera intrinsics with the ChArUco board (camera only; arm not controlled).
# Show the board from many angles/positions; views are auto-captured; then "Calibrate & save".
# If your board is not the generated 5x7 / 30 mm / DICT_4X4_50 one, pass e.g. --square-mm 29.6
cd "$(dirname "$0")" && source ./common.sh
python ../src/calibrate_wrist_intrinsics.py --camera-index $WRIST_CAMERA_INDEX $BOARD_ARGS "$@"
