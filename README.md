# SO-101 wrist-camera pick and place

An SO-101 arm finds a red cube with its wrist camera, picks it up and drops it in the middle of a blue tray. Built on [LeRobot](lerobot/). Each step runs from a shell launcher in `scripts/`, with a viser page at http://localhost:8080 for the live view and buttons.

| Folder | Contents |
|---|---|
| `lerobot/` | LeRobot repository, installed in editable mode |
| `scripts/` | Launchers, plus `common.sh` with all machine-specific settings |
| `src/` | Python code used by the launchers |
| `calibration/` | Camera calibration, colour settings, learned place correction |
| `assets/` | SO-101 URDF and meshes |
| `notebooks/` | ChArUco and colour detection on a single image |

## Installation

```bash
conda create -y -n lerobot python=3.12 && conda activate lerobot
git clone https://github.com/huggingface/lerobot.git lerobot   # skip if lerobot/ exists
pip install -e "./lerobot[feetech,core_scripts,kinematics]" viser yourdfpy trimesh jupyter matplotlib
```
If you move this folder, run `pip install -e ./lerobot --no-deps` again.

## Setup (once per arm)

1. **Find the ports and calibrate the motors.** The full guide is `lerobot/docs/source/so101.mdx`.
   ```bash
   lerobot-find-port
   lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT> --robot.id=so101_follower_arm
   ```
2. **Edit `scripts/common.sh`:**
   - The arm ports and ids.
   - `WRIST_CAMERA_INDEX`: the camera's name, e.g. `Innomaker`, so the laptop webcam is never opened.
   - `BOARD_ARGS`: your ChArUco board's layout, dictionary and measured sizes.
   - `TABLE_Z`: the table height relative to the robot base.
   - `TARGET_OFFSET` / `PLATE_OFFSET`: small x, y corrections in metres.
3. **macOS permissions:** grant Camera to your terminal or VS Code. The keyboard teleop also needs Accessibility and Input Monitoring.

## Calibrate the wrist camera

Tape the ChArUco board flat on the table, then run:
```bash
scripts/07_wrist_auto_calibrate.sh
```
1. In viser, tick **Release arm** and move the arm by hand so the camera looks down at the whole board from 20–25 cm. Keep every joint away from its limits.
2. Untick **Release arm** and click **Set scan pose**. The arm then collects 30+ views by itself.

When it finishes, check two numbers: the lens RMS should be below 1 px, and the board spread below 3 mm. Redo the calibration whenever the camera moves on the wrist.

## Pick and place

```bash
scripts/09_pick_red_cube_wrist.sh
```
Put the cube and the tray 10–28 cm from the base. The whole tray must be in view from the scan pose. Then click **Pick & place**. The left panel shows each stage:
1. Find the tray and the cube.
2. Take a close-up look at the cube, grasp it and lift it.
3. Carry it in a straight line above the tray and take a close-up look at the tray.
4. Lower the cube and drop it.
5. Check where it landed.

Each landing updates the learned correction (**Reset place correction** clears it). The 3D view shows the robot, the cube and tray models, and the planned paths (purple = joint moves, orange = straight-line moves).

**Tuning in viser**
- **Grasp height:** lower it if the fingers close above the cube; raise it if they touch the table.
- **Red / Blue detection:** use these sliders if the cube or tray isn't outlined, then click **Save detection settings**.
- **`TARGET_OFFSET`:** adjust it in `common.sh` if grasps consistently land beside the cube.

## Other launchers

| Script | Purpose |
|---|---|
| `00_release_arm.sh` | Torque off, so the arm can be moved by hand (support it first) |
| `01_teleoperate.sh`, `02_record_dummy_dataset.sh`, `03_replay_episode.sh` | LeRobot teleop, recording and replay with the leader arm |
| `04_keyboard_ee_viser.sh` | End-effector teleop with the keyboard or the viser gizmo |
| `07_wrist_intrinsics.sh`, `08_wrist_handeye.sh` | Manual, two-step alternative to the automatic calibration |
| `10_fk_probe.sh` | Prints the fingertip position: touch the table, and z should be about `TABLE_Z` |

## Safety

- Keep a hand near the arm and use Ctrl+C to stop. Scripts turn the torque off when they exit, so the arm goes limp.
- The pick script moves only when you click a button. The automatic calibration moves on its own, within limits around the scan pose.
