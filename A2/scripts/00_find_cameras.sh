#!/usr/bin/env bash
# List the cameras with their OpenCV indices; put the scene camera's name in config.sh (SCENE_CAMERA).
# With --snapshot, also save one picture per camera to outputs/captured_images/ (opens every
# camera, including the MacBook's).
set -euo pipefail
cd "$(dirname "$0")" && source ./config.sh
python cameras.py list
if [ "${1:-}" = "--snapshot" ]; then
  lerobot-find-cameras opencv
fi
