#!/usr/bin/env bash
# Step 2: hand-eye calibration (camera pose on the wrist). Board flat on the table.
# Set the scan pose by hand in viser, then the arm samples ~20 poses automatically.
cd "$(dirname "$0")" && source ./common.sh
python ../src/calibrate_wrist_handeye.py --port $FOLLOWER_PORT --id $FOLLOWER_ID --camera-index $WRIST_CAMERA_INDEX "$@"
