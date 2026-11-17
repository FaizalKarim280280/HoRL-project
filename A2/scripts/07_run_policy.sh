#!/usr/bin/env bash
# Run a trained policy on the follower (the arm moves on its own). Usage, from A2/:
#   scripts/07_run_policy.sh [policy]      policy = a pretrained_model folder or a Hub repo id
# A window shows both cameras; the keys work in that window:
#   SPACE  start the policy / pause it        r  return to the start pose
# Start pose = the median first pose of all demonstrations in data/raw (the arm glides there at launch).
#   q/ESC  quit (the follower goes limp: support it)
# Executes half of each predicted chunk before predicting again (chunk 30 -> 15 steps);
# N_ACTION_STEPS=10 overrides it.
# Smoother motion:
#   ENSEMBLE=0.01   temporal ensembling: predict a chunk every step and average all the chunks'
#                   predictions for the current step (ACT paper; ~25 ms per step on the Mac)
#   INTERP=2        send 2 interpolated commands per action (60 Hz to the motors)
# STEPS=5 (flow matching / diffusion) denoising steps per chunk; fewer = faster chunks on the Mac.
# MAX_STEP_DEG=0.75 (default) largest change of an ARM joint between consecutive commands (0 = off).
# MAX_STEP_GRIPPER=0 (default: no limit) the same for the gripper; limiting it delays grasps.
# DURATION=40 pauses automatically after 40 s.
# The cameras must be the ones used for the demonstrations, with the scene camera in the same place.
set -euo pipefail
POLICY=${1:-}
[[ -n "$POLICY" && -d "$POLICY" ]] && POLICY=$(cd "$POLICY" && pwd)   # relative to where you run it, before the cd below
cd "$(dirname "$0")" && source ./config.sh
POLICY=${POLICY:-$A2_DIR/outputs/train/act_c30_b128/020000/pretrained_model}
WRIST=$(camera_index "$WRIST_CAMERA")
SCENE=$(camera_index "$SCENE_CAMERA")
python run_policy.py --policy "$POLICY" \
  --follower-port $FOLLOWER_PORT --follower-id $FOLLOWER_ID \
  --wrist $WRIST --scene $SCENE --width $CAMERA_WIDTH --height $CAMERA_HEIGHT --fps $FPS \
  --task "$TASK" --device mps --n-action-steps ${N_ACTION_STEPS:-0} \
  ${ENSEMBLE:+--ensemble $ENSEMBLE} --interp ${INTERP:-1} --num-inference-steps ${STEPS:-0} --start-pose-from "$A2_DIR/data/raw" --max-step-deg ${MAX_STEP_DEG:-0.75} --max-step-gripper ${MAX_STEP_GRIPPER:-0} --duration ${DURATION:-0}
