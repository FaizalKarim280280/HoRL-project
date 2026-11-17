#!/usr/bin/env bash
# Record a small local dummy dataset (no cameras, no Hub upload).
# Keys: -> next, <- redo episode, ESC finish.
cd "$(dirname "$0")" && source ./common.sh
lerobot-record \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=$LEADER_ID \
  --dataset.repo_id=local/dummy_so101 \
  --dataset.single_task="task_9oct_01" \
  --dataset.num_episodes=3 \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false

