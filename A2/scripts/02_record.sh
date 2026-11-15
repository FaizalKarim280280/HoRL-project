#!/usr/bin/env bash
# Record demonstrations with the leader arm. A window shows both cameras; keys work in that window:
#   SPACE  start a demonstration / stop and save it      d  discard the one being recorded
#   [ ]    previous / next initial-configuration ID      q/ESC  quit
# Every attempt is logged in data/raw/<DATASET_NAME>/demos.csv. Who is demonstrating and the starting
# configuration ID can be set:  OPERATOR=Faizal CONFIG=3 ./02_record.sh
# Saves raw JPEGs + joint data to data/raw/<DATASET_NAME>/ (instant; no video encoding). Running it
# again adds episodes. Then ./03_convert.sh. Optional: ./02_record.sh 10 stops after 10 saved episodes.
set -euo pipefail
cd "$(dirname "$0")" && source ./config.sh
WRIST=$(camera_index "$WRIST_CAMERA")
SCENE=$(camera_index "$SCENE_CAMERA")
python record_demos.py --root "$RAW_ROOT" \
  --follower-port $FOLLOWER_PORT --follower-id $FOLLOWER_ID \
  --leader-port $LEADER_PORT --leader-id $LEADER_ID \
  --wrist $WRIST --scene $SCENE --width $CAMERA_WIDTH --height $CAMERA_HEIGHT --fps $FPS \
  --task "$TASK" --jpeg-quality $JPEG_QUALITY --max-episodes ${1:-0} \
  --operator "${OPERATOR:-}" --config "${CONFIG:-0}"
