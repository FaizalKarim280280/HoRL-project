# Wrist-Camera Pick and Place with the SO-101

**Team:** *&lt;names&gt;*

## How it works

The arm picks up a red 25 mm cube and drops it in a dark-blue $10 \times 10$ cm tray. Its only sensor is the wrist camera ($640 \times 480$).

1. **Calibration.** A ChArUco board lies on the table, and the arm moves the camera around it by itself to record 30+ views; the board corners are found with `cv2.aruco.CharucoDetector.detectBoard`. OpenCV estimates the lens model (`cv2.calibrateCamera`), the board's pose in each view (`cv2.solvePnP` + `cv2.solvePnPRefineLM`), and from those the camera's pose on the gripper (`cv2.calibrateHandEye`). Any camera pose then follows from the joint angles: $T^{\text{base}}_{\text{cam}} = \mathrm{FK}(q)\, T^{\text{gripper}}_{\text{cam}}$, with FK from LeRobot's `RobotKinematics.forward_kinematics` (placo).
2. **Locating.** The cube is found by HSV colour thresholding (`cv2.inRange`, `cv2.findContours`, centre from `cv2.moments`, edge direction from `cv2.minAreaRect`). The ray through its centre pixel (`cv2.undistortPoints`) is intersected with the plane of its top face, which gives its (x, y) in the robot frame. The tray's outline (`cv2.convexHull`, `cv2.approxPolyDP`) is projected onto its top surface, and the centre comes from its real $\sim 90^\circ$ corners (all four, or two opposite ones). A tray touching the image edge is rejected.
3. **Grasping.** The arm has only 5 joints, so the planner tries a top-down grasp first, then tilts of up to $\pm 30^\circ$, each with the 4 jaw directions that line up with the cube's edges. It keeps the first combination inverse kinematics (`RobotKinematics.inverse_kinematics`) can reach. The arm hovers above the cube, re-detects it close up, descends and grasps. If the gripper closed completely, it missed.
4. **Placing.** The arm carries the cube in a straight, level line above the tray, re-measures the tray close up, and drops the cube from 1.5 cm above it. Then it checks, from the scan pose, how far from the tray centre the cube landed. Joint targets go to the arm with `SO101Follower.send_action`, and joints and camera frames come back from `get_observation`; the robot and planned paths are shown in `viser`.

![Setup: SO-101 arm, wrist camera, red cube, blue tray](setup.jpg)

## Conditions it needs

- A flat table at a known height, and even lighting, with no other red or dark-blue objects in view.
- Cube and tray 10–28 cm from the base, with the whole tray visible from the scan pose.
- A valid camera calibration (redone if the camera moves) and an accurate motor calibration.

## Performance

Over *&lt;N&gt;* tests the cube was grasped in *&lt;a/N&gt;* and landed in the tray in *&lt;b/N&gt;*, on average *&lt;d&gt;* mm from the tray centre. The camera calibration agreed with itself to within about 4 mm.

## What went wrong

- **Partial tray view:** the centre shifted towards the visible part, so the cube was dropped short of the tray. Now only full views of the tray are used.
- **Oscillation:** continuous camera correction above the tray made the arm oscillate. Now it takes one close-up measurement, then makes a straight move.
- **Calibration:** too few views near the image corners gave a bad lens model. Fixed with a simpler model and automatic corner coverage.
- **Limits:** colour detection is sensitive to lighting, and reach is small at height, so distant cubes need tilted grasps.

## Test results

Videos: *&lt;link&gt;*

Complexity: **E** = cube and tray close and centred; **M** = rotated cube or tray off to the side; **H** = near the reach limit, partly hidden or poor lighting.

| # | Complexity | Scenario | Result | Error (mm) | Failure reason |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |
| 9 | | | | | |
| 10 | | | | | |

## Team contributions and AI use

*&lt;Member 1&gt;*: *&lt;tasks&gt;*. *&lt;Member 2&gt;*: *&lt;tasks&gt;*. We used Claude Code (Anthropic) to write and debug most of the Python code (calibration, detection, grasp planning, motion) and to explain the geometry. We tested and tuned every step ourselves on the real arm.
