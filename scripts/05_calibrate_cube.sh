#!/usr/bin/env bash
# Guided camera calibration for red-cube picking (viser at http://localhost:8080).
# Start with the arm in a pose that does NOT block the camera's view of the table (park pose).
# Camera defaults: ZED index 0, 1344x376 @ 15 fps, left image. Override, e.g. --camera-index 1
cd "$(dirname "$0")" && source ./common.sh
python ../src/calibrate_cube_grasp.py --port $FOLLOWER_PORT --id $FOLLOWER_ID "$@"
