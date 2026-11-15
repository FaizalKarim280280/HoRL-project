#!/usr/bin/env bash
# Play back one episode of the converted dataset (both videos + joint plots) in the rerun viewer.
# Usage: ./04_view_episode.sh [episode]
set -euo pipefail
cd "$(dirname "$0")" && source ./config.sh
lerobot-dataset-viz --repo-id $DATASET_REPO_ID --root "$DATASET_ROOT" --episode-index ${1:-0}
