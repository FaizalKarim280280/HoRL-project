# SO-101 pick and place: classical and learned

An SO-101 arm picks up a red cube and places it in a blue tray, first with a classical pipeline (A1), then with
policies learned from teleoperated demonstrations (A2). Built on [LeRobot](lerobot/).

| Folder | Contents |
|---|---|
| [`A1/`](A1/README.md) | Assignment 1: classical wrist-camera pick and place (calibration, detection, IK) |
| [`A2/`](A2/README.md) | Assignment 2: demonstrations, ACT training and inference, Task 1 evaluation ([results](A2/results/task1/README.md)) |
| `common/` | Shared by both: `robot.sh` (arm ports, ids, wrist camera name), `camera_lookup.py` (cameras by name) |
| `lerobot/` | LeRobot repository, installed in editable mode |

## Installation

```bash
conda create -y -n lerobot python=3.12 && conda activate lerobot
git clone https://github.com/huggingface/lerobot.git lerobot   # skip if lerobot/ exists
pip install -e "./lerobot[feetech,core_scripts,kinematics]" viser yourdfpy trimesh jupyter matplotlib
pip install git+https://github.com/TRI-ML/sequentialized_barnard_tests   # STEP, for the A2 evaluation
```
If you move this folder, run `pip install -e ./lerobot --no-deps` again.

## Setup (once per arm)

1. **Find the ports and calibrate the motors.** The full guide is `lerobot/docs/source/so101.mdx`.
   ```bash
   lerobot-find-port
   lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT> --robot.id=so101_follower_arm
   ```
2. **Edit `common/robot.sh`:** the arm ports and ids, and `WRIST_CAMERA_INDEX`, the wrist camera's name
   (e.g. `1080p`), so the laptop webcam is never opened.
3. **macOS permissions:** grant Camera to your terminal or VS Code. The keyboard teleop also needs Accessibility
   and Input Monitoring.

## Safety

- Keep a hand near the arm and use Ctrl+C to stop. Scripts turn the torque off when they exit, so the arm goes limp.
- Scripts that move the arm on their own (calibration, policies, evaluation) do so only after a key press or click.
