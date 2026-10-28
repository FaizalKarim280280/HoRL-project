#!/usr/bin/env bash
# Replay an episode on the follower. Usage: ./03_replay_episode.sh <stamped_repo_id> [episode]
# e.g. ./03_replay_episode.sh local/dummy_so101_20261009_120000 0
# Keep the workspace clear: the arm will move.
cd "$(dirname "$0")" && source ./common.sh
lerobot-replay \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --dataset.repo_id=${1:?repo_id required} --dataset.episode=${2:-0}
