#!/usr/bin/env bash
# Run the evaluation trials (the arm moves). Usage, from A2/:
#   OPERATOR=Faizal eval/02_evaluate.sh            counted trials (schedule.csv -> eval_log.csv)
#   PILOT=1 eval/02_evaluate.sh                    practice trials (pilot_schedule.csv -> pilot_log.csv)
# Keys are listed in evaluate.py and README.md. Quit any time with q; rerunning continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")" && source ../scripts/config.sh
WRIST=$(camera_index "$WRIST_CAMERA")
SCENE=$(camera_index "$SCENE_CAMERA")
python evaluate.py --follower-port $FOLLOWER_PORT --follower-id $FOLLOWER_ID \
  --wrist $WRIST --scene $SCENE --width $CAMERA_WIDTH --height $CAMERA_HEIGHT \
  --device mps --operator "${OPERATOR:-}" ${PILOT:+--pilot}
