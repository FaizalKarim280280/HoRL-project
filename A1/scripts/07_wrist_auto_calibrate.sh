#!/usr/bin/env bash
# Automatic wrist-camera calibration (intrinsics + hand-eye in one run). Board flat on the table.
# Set the scan pose by hand in viser (http://localhost:8080); the arm then collects all views itself.
cd "$(dirname "$0")" && source ./common.sh
python ../src/calibrate_wrist_auto.py --port $FOLLOWER_PORT --id $FOLLOWER_ID --camera-index $WRIST_CAMERA_INDEX $BOARD_ARGS "$@"
