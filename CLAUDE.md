# CLAUDE.md

SO-101 arm projects for a robot-learning course (HRL). Two assignments share this repo:

- **Assignment 1 (root):** classical wrist-camera pick and place: find a red cube, grasp it, drop it in a blue tray. Hand-eye calibrated camera, colour detection, IK. Done.
- **A2 (`A2/`):** learn the same task from teleoperated demonstrations with ACT. Assignment text: `A2/HRL Assignment 2.pdf` (Task 1: compare >= 2 policies/configs, >= 25 matched eval rollouts each, STEP `MirroredStepTest`, 90% Wilson CIs, log all demos and evals).

## Environment

- macOS, conda env `lerobot` (Python 3.12): `/Users/faizal/miniconda3/envs/lerobot/bin/python`. The base `python` on PATH lacks cv2/torch.
- `lerobot/` is the upstream LeRobot repo, installed editable. Treat it as a library: don't edit it.
- No `nbformat`/`nbconvert` in the env; notebooks are run in VS Code with the env's kernel.
- `torchcodec` video decoding is broken on this Mac: use `video_backend="pyav"` / `--dataset.video_backend=pyav` when reading LeRobot datasets locally.

## Hardware

- Follower and leader ports and ids: `scripts/common.sh` (single source of machine-specific settings).
- Cameras are chosen by **name**, never by index (the MacBook webcam must not open): wrist = `"1080p"` (Innomaker U20CAM-1080p), scene = `"720P"`. Resolved via `src/cube_pick_common.resolve_camera`.
- Scripts turn torque off on exit, so the follower goes limp: warn the user to support it.
- Never move the arm (or run anything that does) without the user asking.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | Assignment-1 launchers (`NN_name.sh`), all sourcing `common.sh` |
| `src/` | Assignment-1 Python: calibration, detection (`wrist_vision.py`, `plate_detection.py`), `pick_red_cube_wrist.py` (viser UI at :8080) |
| `calibration/` | Camera intrinsics, hand-eye result, colour thresholds |
| `report/` | Assignment-1 report and evaluation Q&A |
| `A2/scripts/` | A2 launchers `00_`-`06_` + `config.sh` (dataset name, cameras, task) and their Python |
| `A2/notebooks/inspect_demos.ipynb` | Demonstration quality checks |
| `A2/data/` | Recordings (git-ignored): `raw/<name>/` and `lerobot/<name>/` |

## A2 data pipeline

1. `A2/scripts/02_record.sh`: own fast recorder (`record_demos.py`), not `lerobot-record`. 30 Hz, OpenCV window (wrist above scene), SPACE start/stop, `d` discard, `q` quit. Writes JPEGs + `data.npz` per episode and logs every attempt in `demos.csv`.
   - `action` = **leader** joint positions (degrees, gripper 0-100); `state` = follower joint positions.
2. `A2/scripts/03_convert.sh`: raw -> LeRobot v3 dataset with the same features `lerobot-record` would produce (camera keys `wrist`, `scene`, 640x480).
3. Training happens on a separate GPU machine; dataset and policy move via the Hugging Face Hub (private repos) or rsync. Inference runs on the Mac (`--policy.device=mps`) with the same camera names, resolution and scene-camera pose as training.

Demonstrations end with: cube released in the tray, gripper lifted slightly, return to rest with the gripper closed.

## Conventions

- Every step is a numbered shell launcher that sources the shared config; Python goes next to it (`src/` or `A2/scripts/`). Run A2 scripts from `A2/`.
- Match existing style: short docstrings, comments explain *why*, argparse with defaults from the launcher.
- READMEs stay short: only the important steps, tables for keys/scripts.
- Explanations to the user: plain language, step by step; render maths properly when asked.
- Don't commit datasets, `outputs/` or snapshots. Commit and push only when asked.
