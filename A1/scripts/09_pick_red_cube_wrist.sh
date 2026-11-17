#!/usr/bin/env bash
# Step 3: scan for the red cube with the wrist camera and pick it ("Pick" button in viser).
cd "$(dirname "$0")" && source ./common.sh
python ../src/pick_red_cube_wrist.py --port $FOLLOWER_PORT --id $FOLLOWER_ID --camera-index $WRIST_CAMERA_INDEX --table-z $TABLE_Z --target-offset $TARGET_OFFSET --plate-offset $PLATE_OFFSET "$@"
