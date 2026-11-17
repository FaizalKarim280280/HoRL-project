#!/usr/bin/env python
"""Hand-eye calibration of the SO-101 wrist camera: finds T_gripper_cam, the camera's pose
in the gripper frame, so that T_base_cam = FK(q) * T_gripper_cam for any arm pose.

Needs wrist_intrinsics.json (calibrate_wrist_intrinsics.py) and the printed ChArUco board
taped flat on the table, in front of the robot.

1. Tick "Release arm", move the arm by hand to the SCAN pose: high above the table, camera
   looking down at the area where cubes will be, board in view. Untick, then click
   "Set scan pose". This pose is also what pick_red_cube_wrist.py scans from.
2. The arm then visits ~20 poses around the scan pose (varying pan, pitch and wrist roll).
   At each, the board's pose in the camera is measured and paired with the gripper pose (FK).
3. cv2.calibrateHandEye solves for T_gripper_cam. The board's base-frame pose is then
   recomputed from every sample: its spread is the real accuracy, and its height checks the
   "table and base in the same plane" assumption.
"""

import argparse
import threading
from pathlib import Path

import cv2
import numpy as np
from viser.transforms import SO3

from cube_pick_common import ALL_JOINTS, CameraSettings, DetectionSettings, Station
from wrist_vision import INTRINSICS_FILE, WRIST_CALIB_FILE, BoardDetector, BoardSpec, Intrinsics, WristCalibration, check_distortion

METHODS = {
    "Tsai": cv2.CALIB_HAND_EYE_TSAI,
    "Park": cv2.CALIB_HAND_EYE_PARK,
    "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}
# Joint offsets (deg) around the scan pose: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.
PERTURB_RANGE = np.array([15.0, 10.0, 10.0, 20.0, 45.0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--camera-index", default="0", help="Wrist camera index (run lerobot-find-cameras opencv)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--intrinsics", type=Path, default=INTRINSICS_FILE)
    p.add_argument("--square-mm", type=float, default=None, help="Override the board square size (measured)")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--min-height", type=float, default=0.10, help="Lowest allowed gripper height (m) while sampling")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ignore-intrinsics-check", action="store_true", help="Use intrinsics that fail the lens-model check")
    p.add_argument("--out", type=Path, default=WRIST_CALIB_FILE)
    p.add_argument("--viser-port", type=int, default=8080)
    return p.parse_args()


def sample_poses(station: Station, scan_q: np.ndarray, n: int, min_height: float, rng) -> list[np.ndarray]:
    """Random joint perturbations around the scan pose that stay high and within limits,
    ordered to keep each move short. Every third pose uses a large wrist roll for diversity."""
    poses = []
    tries = 0
    while len(poses) < n and tries < 50 * n:
        tries += 1
        offset = rng.uniform(-1, 1, 5) * PERTURB_RANGE
        if len(poses) % 3 == 0:
            offset[4] = np.sign(offset[4] or 1) * rng.uniform(25, 45)
        q = scan_q.copy()
        q[:5] += offset
        if not station.within_limits(q, 3) or station.kin.forward_kinematics(q)[2, 3] < min_height:
            continue
        poses.append(q)
    ordered, current = [], scan_q
    while poses:
        i = int(np.argmin([np.max(np.abs(p[:5] - current[:5])) for p in poses]))
        current = poses.pop(i)
        ordered.append(current)
    return ordered


def solve(samples: list[dict]) -> tuple[str, np.ndarray, float, float, np.ndarray]:
    """Runs every OpenCV hand-eye method; keeps the one whose board estimates agree best."""
    r_g2b = [s["t_base_gripper"][:3, :3] for s in samples]
    t_g2b = [s["t_base_gripper"][:3, 3] for s in samples]
    r_t2c = [s["t_cam_board"][:3, :3] for s in samples]
    t_t2c = [s["t_cam_board"][:3, 3] for s in samples]
    best = None
    for name, method in METHODS.items():
        r, t = cv2.calibrateHandEye(r_g2b, t_g2b, r_t2c, t_t2c, method=method)
        x = np.eye(4)
        x[:3, :3], x[:3, 3] = r, t.ravel()
        boards = [s["t_base_gripper"] @ x @ s["t_cam_board"] for s in samples]
        mean_pos = np.mean([b[:3, 3] for b in boards], axis=0)
        pos_err = float(np.sqrt(np.mean([np.sum((b[:3, 3] - mean_pos) ** 2) for b in boards]))) * 1000
        rot_err = float(np.degrees(np.sqrt(np.mean([np.sum(SO3.from_matrix(boards[0][:3, :3].T @ b[:3, :3]).log() ** 2) for b in boards]))))
        print(f"  {name:10s} board spread {pos_err:5.1f} mm, {rot_err:4.2f} deg")
        if best is None or pos_err < best[2]:
            mean_board = boards[0].copy()
            mean_board[:3, 3] = mean_pos
            best = (name, x, pos_err, rot_err, mean_board)
    return best


def wait_for_scan_pose(station: Station, detector: BoardDetector, min_corners: int = 10, limit_margin: float = 15.0) -> np.ndarray:
    """Lets the user move the arm by hand (torque off) to the scan pose and confirm it in viser."""
    gui = station.server.gui
    set_scan, torque_toggled = threading.Event(), threading.Event()
    with gui.add_folder("Scan pose"):
        release = gui.add_checkbox("Release arm (move by hand)", initial_value=False)
        gui.add_button("Set scan pose", color="green").on_click(lambda _: set_scan.set())
    release.on_update(lambda _: torque_toggled.set())  # torque is switched on the main thread
    released = False

    def sync_torque() -> None:
        nonlocal released
        if torque_toggled.is_set():
            torque_toggled.clear()
            if release.value and not released:
                station.release()
                released = True
            elif not release.value and released:
                station.hold()
                released = False

    station.set_status(
        "**Set the scan pose.** Tick *Release arm*, move the arm by hand so the camera looks down at the "
        "table with the whole board in view (high, ~20-25 cm), untick it, then click **Set scan pose**."
    )
    while True:
        station.wait_for(set_scan, on_tick=sync_torque)
        sync_torque()
        if released:
            station.set_status("Untick *Release arm* first, then click **Set scan pose**.")
            continue
        station.idle(0.5)
        found = detector.detect(station.frame)
        if found is None or len(found[1]) < min_corners:
            station.set_status(f"**Board not visible enough** from this pose (need {min_corners}+ corners). Adjust and retry.")
            continue
        # The arm must be able to move around the scan pose in every direction.
        near = [
            f"{name} {station.q[i]:.0f}° (limit {lo:.0f}..{hi:.0f})"
            for i, name in enumerate(ALL_JOINTS[:5])
            for lo, hi in [station.joint_limits_deg[name]]
            if station.q[i] < lo + limit_margin or station.q[i] > hi - limit_margin
        ]
        if near:
            station.set_status(
                f"**Too close to a joint limit:** {', '.join(near)}. The arm could not move around this pose. "
                "Choose a pose with every joint at least "
                f"{limit_margin:.0f}° from its limit (e.g. bend the elbow more instead of the wrist), then retry."
            )
            continue
        return station.q.copy()


def save_and_show(station: Station, out: Path, camera: CameraSettings, intr: Intrinsics, samples: list[dict], scan_q: np.ndarray) -> None:
    """Solves hand-eye, saves wrist_calibration.json and shows the board and camera in viser."""
    print("Hand-eye methods:")
    name, x, pos_err, rot_err, board = solve(samples)
    WristCalibration(
        camera=camera,
        intrinsics=intr,
        t_gripper_cam=x.tolist(),
        scan_q=scan_q.tolist(),
        method=name,
        consistency_mm=pos_err,
        consistency_deg=rot_err,
        board_z_m=float(board[2, 3]),
        num_samples=len(samples),
    ).save(out)
    k = intr.k_mat
    scene = station.server.scene
    t_base_cam = station.kin.forward_kinematics(station.q) @ x
    scene.add_frame("/board", axes_length=0.05, axes_radius=0.002, wxyz=SO3.from_matrix(board[:3, :3]).wxyz, position=board[:3, 3])
    scene.add_camera_frustum(
        "/camera", fov=2 * np.arctan(intr.height / 2 / k[1, 1]), aspect=intr.width / intr.height, scale=0.05,
        wxyz=SO3.from_matrix(t_base_cam[:3, :3]).wxyz, position=t_base_cam[:3, 3],
    )
    axis_angle = np.degrees(np.arccos(np.clip(x[:3, 2] @ np.array([0, 0, 1.0]), -1, 1)))
    station.set_status(
        f"**Saved {out.name}** ({name}, {len(samples)} samples). Board spread **{pos_err:.1f} mm**, "
        f"{rot_err:.2f} deg (under ~3 mm is good).  \n"
        f"Camera is {np.linalg.norm(x[:3, 3]) * 100:.1f} cm from the gripper frame, optical axis "
        f"{axis_angle:.0f} deg from the gripper axis. Board height in base frame: "
        f"**{board[2, 3] * 1000:.1f} mm** (should be ~0 if table and base share a plane).  \n"
        "Check the 3D view: the camera frustum should sit on the wrist. Ctrl+C to exit."
    )


def main() -> None:
    args = parse_args()
    intr = Intrinsics.load(args.intrinsics)
    spec = intr.board
    if args.square_mm is not None:  # measured size; the marker scales with it
        scale = args.square_mm / 1000 / spec.square_m
        spec = BoardSpec(spec.squares_x, spec.squares_y, spec.square_m * scale, spec.marker_m * scale, spec.dictionary)
    if (intr.width, intr.height) != (args.width, args.height):
        raise ValueError(f"Intrinsics are for {intr.width}x{intr.height}, camera set to {args.width}x{args.height}")
    k, dist = intr.k_mat, intr.dist_vec
    ok, worst = check_distortion(k, dist, intr.width, intr.height)
    if not ok and not args.ignore_intrinsics_check:
        raise SystemExit(
            f"{args.intrinsics.name} fails the lens-model check ({worst:.0f} px error at the image edges): the "
            "distortion model is invalid near the image corners. Re-run 07_wrist_intrinsics.sh with views covering "
            "all four image corners (or pass --ignore-intrinsics-check)."
        )
    detector = BoardDetector(spec)
    camera = CameraSettings(args.camera_index, args.width, args.height, args.fps)
    station = Station(args.port, args.id, camera, DetectionSettings(), viser_port=args.viser_port, detect_red=False)
    station.overlay = detector.overlay_for(station)
    station.connect()
    rng = np.random.default_rng(args.seed)
    try:
        scan_q = wait_for_scan_pose(station, detector)

        poses = sample_poses(station, scan_q, args.samples, args.min_height, rng)
        samples = []
        for i, q in enumerate([scan_q, *poses], 1):
            station.set_status(f"Sample {i}/{len(poses) + 1}: moving...")
            station.joint_move(q, speed_deg_s=30)
            station.idle(0.8)  # let vibrations and motion blur settle
            found = detector.detect(station.frame)
            result = detector.pose(*found, k, dist) if found is not None else None
            if result is None or result[1] > 2.0:
                print(f"  sample {i}: skipped ({'no board' if result is None else f'reprojection {result[1]:.1f} px'})")
                continue
            samples.append({"t_base_gripper": station.kin.forward_kinematics(station.q), "t_cam_board": result[0]})
            print(f"  sample {i}: ok ({len(found[1])} corners, reprojection {result[1]:.2f} px)")
        station.joint_move(scan_q, speed_deg_s=30)

        if len(samples) < 8:
            station.set_status(f"**Only {len(samples)} good samples** (need 8+). Keep the board in view and retry.")
            return
        intr = Intrinsics(intr.k, intr.dist, intr.width, intr.height, intr.rms_px, intr.num_images, spec)
        save_and_show(station, args.out, camera, intr, samples, scan_q)
        while True:
            station.idle(1.0)
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        station.disconnect()


if __name__ == "__main__":
    main()
