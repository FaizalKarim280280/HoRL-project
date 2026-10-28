#!/usr/bin/env bash
# Detect the red cube and pick it (needs cube_calibration.json from 05_calibrate_cube.sh).
# Nothing moves until you click "Pick" in viser (http://localhost:8080).
cd "$(dirname "$0")" && source ./common.sh
python ../src/pick_red_cube.py --port $FOLLOWER_PORT --id $FOLLOWER_ID "$@"
