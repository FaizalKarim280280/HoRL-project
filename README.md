# SO-101 workspace

Pick up a red object with an SO-101 arm using only its wrist camera. Built on [LeRobot](lerobot/).

The arm looks at the table from a scan pose and finds the red object in the image. It works out where the object is on the table from the camera calibration, then grasps it top-down. Everything runs from the shell launchers in `scripts/`, with a viser web page (http://localhost:8080) for live views and buttons.

## Layout

| Folder | Contents |
|---|---|
| `lerobot/` | The LeRobot repository (git clone of `huggingface/lerobot`), installed in editable mode |
| `scripts/` | Launchers (`00_…sh` to `10_…sh`) and `common.sh` with all machine-specific settings |
| `src/` | Python code used by the launchers |
| `notebooks/` | `charuco_marker_detection.ipynb`: marker, board and red-object detection on one image |
| `assets/` | SO-101 URDF and meshes (`SO101/`), printable ChArUco boards |
| `calibration/` | Wrist-camera results (`wrist_intrinsics.json`, `wrist_calibration.json`, raw views and samples) |
| `data/` | Captured images |

---

## 1. What you need

**Hardware**
- **SO-101 follower arm** with its power supply and USB controller board. A **leader arm** is optional, used only for teleoperation and recording.
- **Wrist camera:** the standard SO-101 wrist camera (Innomaker U20CAM-1080p) on the gripper. Take off the lens cap.
- **Red object:** the default setup assumes a 25 × 25 mm footprint and 45 mm height.
- **Table:** a flat table, with the arm fixed so it can't slide.

**Software**
- **OS:** macOS or Linux. All commands here were used on macOS.
- **Python:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda), plus `git`.

---

## 2. Installation

Run everything from the workspace root, the folder containing this README.

```bash
# 1. Python environment
conda create -y -n lerobot python=3.12
conda activate lerobot

# 2. LeRobot (skip the clone if lerobot/ already exists)
git clone https://github.com/huggingface/lerobot.git lerobot
#    optional: the exact version this workspace was built against
#    git -C lerobot checkout e0d50211

# 3. LeRobot + motor SDK + robot tools + inverse kinematics, plus our extra packages
pip install -e "./lerobot[feetech,core_scripts,kinematics]" viser yourdfpy jupyter matplotlib

# 4. Optional: ffmpeg, only needed for recording/decoding video datasets
conda install -y -c conda-forge ffmpeg
```

Check it worked:
```bash
python -c "import lerobot, viser, placo, cv2; print('ok', cv2.__version__)"
lerobot-find-port --help
```

**Notes**
- **Editable install:** `lerobot` is installed in editable mode, so the env points at this `lerobot/` folder. If you move or rename the folder, run `pip install -e ./lerobot --no-deps` again, or `import lerobot` and the `lerobot-*` commands break.
- **Version pins:** `placo` (inverse kinematics) is pinned by lerobot's `kinematics` extra to versions that work with its compiled dependencies. Don't upgrade it on its own.
- **`uv`:** the lerobot repo also supports `uv`. These instructions use conda because the launchers call `python` from the active conda env.

---

## 3. First-time hardware setup

Do this once per arm. The full guide is in `lerobot/docs/source/so101.mdx`.

**3.1 Find the USB ports.** Run this once per arm, and unplug that arm when it asks:
```bash
lerobot-find-port
```
On macOS ports look like `/dev/tty.usbmodem…`; on Linux like `/dev/ttyACM0` (you may need `sudo chmod 666 /dev/ttyACM0`).

**3.2 Motor IDs.** Only needed for a freshly assembled arm:
```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=<FOLLOWER_PORT>
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>
```

**3.3 Calibrate the motors.** Centre every joint, press Enter, then sweep each joint through its full range. The `id` names the calibration, so use the same id everywhere afterwards:
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=so101_follower_arm
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>   --teleop.id=so101_leader_arm
```
Calibrate carefully. The robot model (URDF) assumes lerobot's calibration convention, so a sloppy "centre" pose shows up later as position errors when grasping.

**3.4 Find the wrist camera index:**
```bash
lerobot-find-cameras opencv      # saves a snapshot per camera to outputs/captured_images/
```
On a MacBook the built-in webcam is usually one index and the wrist camera another. Indices can swap when you replug devices. An all-black image means the lens cap is still on.

**3.5 Fill in `scripts/common.sh`:**

| Setting | Meaning |
|---|---|
| `FOLLOWER_PORT`, `FOLLOWER_ID` | follower arm port and calibration id (from 3.1 / 3.3) |
| `LEADER_PORT`, `LEADER_ID` | leader arm, only for teleop and recording |
| `WRIST_CAMERA_INDEX` | wrist camera index (from 3.4) |
| `BOARD_ARGS` | your printed ChArUco board (see 4.1) |
| `TABLE_Z` | table height relative to the robot base in metres, e.g. `-0.03` if the table surface is 3 cm below the base's bottom plate |
| `TARGET_OFFSET` | constant (x, y) correction in metres added to detected positions; start with `"0 0"` (see 6) |

**3.6 macOS permissions.** These are given to the app you run the commands from (Terminal, iTerm or VS Code), in *System Settings → Privacy & Security*:
- **Camera:** needed for any script that uses the wrist camera. macOS asks the first time.
- **Accessibility** and **Input Monitoring:** only for the keyboard teleop (`04`). Without them keys are silently ignored. Restart the app after granting.

**3.7 Sanity check:** teleoperate with the leader arm. The follower should mirror it:
```bash
scripts/01_teleoperate.sh
```

---

## 4. Calibrate the wrist camera (once, and again if the camera moves)

The pick script needs two calibrations:
1. **Intrinsics:** the camera's lens model (focal length, image centre, lens distortion).
2. **Hand-eye:** exactly where the camera sits on the gripper.

Together with the arm's joint angles, these turn an image pixel into a position on the table.

### 4.1 Print the board
- **Print it:** `assets/charuco_board_20mm.pdf` at **100% / actual size** (not "fit to page"). It's a 5×7-square board, DICT_4X4_50, 20 mm squares and 15 mm markers.
- **Measure it:** measure a square and a marker with a ruler; measure across several squares and divide for accuracy. Every distance the camera measures scales with this number.
- **Mount it:** tape it **completely flat** on the table, in front of the arm.
- **Set `BOARD_ARGS`** in `scripts/common.sh` to match:
  ```bash
  BOARD_ARGS="--squares-x 5 --squares-y 7 --dictionary DICT_4X4_50 --square-mm 20 --marker-mm 15"
  ```

A different board works if you set its values here, but keep its markers large in the image. Small or dense markers (for example 6×6 markers only 10–15 mm wide) are hard to decode with this 640×480 camera. Use `notebooks/charuco_marker_detection.ipynb` to check that a board is detected and to identify an unknown board's dictionary and layout.

### 4.2 Run the automatic calibration
```bash
scripts/07_wrist_auto_calibrate.sh
```
1. Open http://localhost:8080.
2. Tick **Release arm** and move the arm by hand to the **scan pose**: camera about 20–25 cm above the table, looking down at the **whole board**, and every joint at least 15° from its limit. Bend the elbow rather than the wrist. Then untick **Release arm** and click **Set scan pose**. The pick script also scans from this pose.
3. Stand clear. The arm steers the board into different parts of the image, rolls the wrist, and visits random nearby poses (30+ views, a few minutes).
4. It solves both calibrations and writes them to `calibration/`.

**Check the result** (viser status line / terminal):

| Value | Good |
|---|---|
| Lens RMS error | < 0.5 px (< 1 px usable) |
| Lens-model check | **passed** |
| Image centre `cx, cy` | near the image middle (about 320, 240 for 640×480) |
| Board spread (hand-eye) | < 3 mm |
| Board height in base frame | close to your `TABLE_Z` |

If the lens check fails or the image centre is far off, the board didn't reach the image edges. Re-run with a scan pose that leaves the arm more room to move. As alternatives, there's a manual step-by-step route: `scripts/07_wrist_intrinsics.sh` (you move the board around the camera yourself), then `scripts/08_wrist_handeye.sh`.

---

## 5. Pick the red object

```bash
scripts/09_pick_red_cube_wrist.sh --cube-size 0.045     # --cube-size = object HEIGHT in metres
```
1. The arm moves to the scan pose. Put the object on the table **8–32 cm from the base**.
2. In viser, the camera image should outline it in **green**, and the red ball in the 3D view should sit on it. If not, open **Red detection**, tune the sliders, and click **Save detection settings**.
3. Click **Pick**. The arm plans a grasp (straight down if possible, otherwise tilted up to 30°), hovers, re-detects close up if it can see the object, descends, closes, lifts, and reports **Holding** or **Missed**. **Put back** sets the object down.
4. Not in view? **Search wider** (or **Pick** with *Auto-search* ticked) looks again from higher up, and to the left and right.

The 3D view shows the **base** frame and the **eef** frame. The eef is the centre between the gripper jaws, the point that is moved onto the object.

---

## 6. Tuning

| Symptom | Fix |
|---|---|
| Fingers close **above** the object | lower the **Grasp height** slider in about 5 mm steps (it's the fingertip height in the base frame) |
| Fingers hit the table | raise **Grasp height**, and check `TABLE_Z` |
| Grasps consistently land beside the object | adjust `TARGET_OFFSET` in `common.sh`: base-frame metres, **x = forward** (away from the base), **y = robot's left** (seen from behind the robot) |
| Gripper opens when it should close | add `--gripper-open 0 --gripper-closed 50` (swap the values) |
| Object not outlined / other things outlined | tune **Red detection**: raise *Saturation min* / *Value min* to reject pale or dark things |
| A different object width | change `DEFAULT_TCP_OFFSET` in `src/cube_pick_common.py` to minus half the object's width |

---

## 7. Safety

- **Keep a hand near the arm** on first runs of anything that moves it. **Ctrl+C** stops a script.
- **Scripts leave the arm limp when they exit** (lerobot turns motor torque off on disconnect). Support the arm, or stop the script only when it's in a resting pose.
- **The interactive scripts only move when you click** (Pick, Put back, Continue…). The automatic calibration moves on its own after you set the scan pose, within limits: ±35° pan, ±20° shoulder/elbow, ±40° wrist flex, ±60° wrist roll around the scan pose, gripper at least 10 cm above the base plane.
- **Keyboard teleop (`04`) reads the keyboard globally**, so typing in other windows moves the target too. It starts with "Send commands to arm" **off**. Check the directions with the preview first.

---

## 8. Things worth knowing

- **5 joints, not 6.** The SO-101 arm can't reach every gripper orientation. With the gripper pointing straight down, the jaw direction is set by the base pan plus the wrist roll. The planner handles this, but some positions only work tilted.
- **Reach.** Pointing straight down, the gripper reaches about 8–32 cm from the base at table height, but only about 14–24 cm at 8 cm above the base. That's why hover heights adapt.
- **Wrist bend limit.** From high up, the wrist camera looks forward and down, not straight down. That limits how high the scan pose can be.
- **Single camera, flat table.** Positions come from intersecting the camera ray with a horizontal plane: the table height (`TABLE_Z`) plus the object height (`--cube-size`). Wrong heights shift the estimated position along the ray. Objects that aren't on the table, or are a different height, will be mislocated.
- **Calibration accuracy depends on the arm's motor calibration.** If the board height reported by hand-eye calibration disagrees with the real table height by centimetres, run `scripts/10_fk_probe.sh` (torque off, prints the fingertip position) and touch the table. z should read about `TABLE_Z`. If not, redo the motor calibration (3.3).
- **Harmless warnings:** `objc: Class AVF… is implemented in both …` (two packages bundle the same video library) and `Could not load libtorchcodec` (no ffmpeg; it falls back to PyAV) can be ignored.

## 9. Troubleshooting

| Problem | Try |
|---|---|
| Feetech timeout / no response from motors | check the red motor LEDs: all steady = wiring OK; one dark = reseat cables; blinking = overload or **wrong power supply** (5 V/7.4 V and 12 V arms aren't interchangeable). Check `FOLLOWER_PORT` |
| Arm is stiff and won't move by hand | a script left torque on: `scripts/00_release_arm.sh` (support the arm, it goes limp) or unplug its power |
| Camera image black | lens cap, wrong `WRIST_CAMERA_INDEX`, or missing camera permission |
| `failed to set fps/width` | the camera doesn't support that mode; check with `lerobot-find-cameras opencv` |
| Board never detected | board too small or blurry in the image, wrong `BOARD_ARGS`, glare; check with the notebook |
| "Too close to a joint limit" when setting the scan pose | move the joint it names away from its limit (usually wrist flex: bend the elbow more instead) |
| "No grasp found" | the object is too close or too far; the status shows its distance from the base |

More LeRobot help: `lerobot/AGENT_GUIDE.md` and `lerobot/docs/source/`.
