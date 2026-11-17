#!/usr/bin/env bash
# Teleoperate the follower with the leader (no recording).
cd "$(dirname "$0")" && source ./common.sh
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=$LEADER_ID
