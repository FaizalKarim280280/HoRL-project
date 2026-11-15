#!/usr/bin/env bash
# Replay one episode of the converted dataset on the follower (checks the recorded actions). Usage: ./05_replay_episode.sh [episode]
# Keep the workspace clear and put the cube back where it was: the arm will move.
set -euo pipefail
cd "$(dirname "$0")" && source ./config.sh
lerobot-replay \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --dataset.repo_id=$DATASET_REPO_ID --dataset.root="$DATASET_ROOT" --dataset.episode=${1:-0}
