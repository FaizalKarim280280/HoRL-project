#!/usr/bin/env bash
# Keyboard end-effector teleop of the follower, visualized in viser (http://localhost:8080).
# Keys: W/S fwd/back, A/D left/right, R/F up/down, E/Q gripper open/close.
# Or drag / rotate the gizmo in viser (pitch + roll only). Ctrl+C to quit.
# Extra args are passed through, e.g. ./04_keyboard_ee_viser.sh --camera-index 1
cd "$(dirname "$0")" && source ./common.sh
python ../src/keyboard_ee_viser.py --port $FOLLOWER_PORT --id $FOLLOWER_ID "$@"
