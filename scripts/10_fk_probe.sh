#!/usr/bin/env bash
# Torque off + live gripper-tip position from the robot model. Touch the table: z should be ~0 mm.
cd "$(dirname "$0")" && source ./common.sh
python ../src/fk_probe.py --port $FOLLOWER_PORT --id $FOLLOWER_ID
