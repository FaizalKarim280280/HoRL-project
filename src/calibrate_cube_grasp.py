#!/usr/bin/env python
"""Guided camera-to-table calibration for picking a red cube with the SO-101.

The arm visits a grid of grasp poses (gripper pointing down, jaws open, at grasp height).
At each one you place the red cube between the jaws and click Continue in viser; the gripper
closes briefly to seat the cube, re-opens, and the arm returns to its park pose (the pose it
was in at start - choose one that keeps the arm out of the camera view). The camera then
detects the cube's pixel position (u, v), which is paired with the commanded (x, y).

A homography fitted to these pairs maps pixels straight to grasp targets. Because it is
fitted on the cube itself and on the robot's own commands, it also absorbs the cube height
(parallax), the gripper's offset and the arm's systematic errors at the table.

Output: cube_calibration.json, used by pick_red_cube.py.
"""

import argparse
import math
import threading
from pathlib import Path

import cv2
import numpy as np

from cube_pick_common import DEFAULT_CALIB, Calibration, CameraSettings, DetectionSettings, Station, topdown_pose


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--camera-index", default="0")
    p.add_argument("--camera-width", type=int, default=1344)
    p.add_argument("--camera-height", type=int, default=376)
    p.add_argument("--camera-fps", type=int, default=15)
    p.add_argument("--stereo-view", choices=["both", "left", "right"], default="left")
    p.add_argument("--radii", type=float, nargs="+", default=[0.16, 0.20, 0.235], help="Grid radii from the base (m)")
    p.add_argument("--angles", type=float, nargs="+", default=[-30, 0, 30], help="Grid angles (deg, + = robot's left)")
    p.add_argument("--grasp-z", type=float, default=0.012, help="Gripper frame height at grasp (m); tunable live")
    p.add_argument("--hover-z", type=float, default=0.08)
    p.add_argument("--gripper-open", type=float, default=50.0)
    p.add_argument("--gripper-closed", type=float, default=0.0)
    p.add_argument("--speed", type=float, default=0.04, help="Straight-line speed (m/s)")
    p.add_argument("--viser-port", type=int, default=8080)
    return p.parse_args()


def fit(points: list[dict]) -> tuple[np.ndarray, float, list[float]]:
    """Least-squares homography; returns it, the RMS fit error and leave-one-out errors (mm)."""
    uv = np.array([p["uv"] for p in points], dtype=np.float64)
    xy = np.array([p["xy"] for p in points], dtype=np.float64)
    h, _ = cv2.findHomography(uv, xy, 0)

    def apply(hm, pts):
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), hm).reshape(-1, 2)

    rms = float(np.sqrt(np.mean(np.sum((apply(h, uv) - xy) ** 2, axis=1)))) * 1000
    loo = []
    if len(points) >= 5:
        for i in range(len(points)):
            keep = np.arange(len(points)) != i
            h_i, _ = cv2.findHomography(uv[keep], xy[keep], 0)
            loo.append(float(np.linalg.norm(apply(h_i, uv[i : i + 1])[0] - xy[i])) * 1000)
    return h, rms, loo


def main() -> None:
    args = parse_args()
    camera = CameraSettings(args.camera_index, args.camera_width, args.camera_height, args.camera_fps, args.stereo_view)
    station = Station(args.port, args.id, camera, DetectionSettings(), viser_port=args.viser_port)
    gui = station.server.gui

    cont, skip = threading.Event(), threading.Event()
    with gui.add_folder("Calibration"):
        gui.add_button("Continue", color="green").on_click(lambda _: cont.set())
        gui.add_button("Skip this point").on_click(lambda _: skip.set())
        height_mm = gui.add_slider("Grasp height (mm)", 0, 40, 1, round(args.grasp_z * 1000))

    grid = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for r in args.radii for a in args.angles]
    points, holding = [], []

    station.connect()
    park_q = station.q.copy()
    park_q[5] = args.gripper_open
    try:
        station.set_status(
            "**Setup.** The arm's current pose is the park pose: it must not block the camera's view of "
            "the table. Remove the cube from the table. Expand *Red detection* to tune the colour "
            "thresholds (red cube should be outlined green). Click **Continue** to start."
        )
        station.wait_for(cont)
        station.set_gripper(args.gripper_open)

        for i, (x, y) in enumerate(grid, 1):
            yaw = math.atan2(y, x)  # jaws close radially
            grasp_z = height_mm.value / 1000
            q_hover = station.solve_topdown(x, y, args.hover_z, yaw)
            q_grasp = station.solve_topdown(x, y, grasp_z, yaw)
            if q_hover is None or q_grasp is None:
                station.set_status(f"Point {i}/{len(grid)} ({x:.3f}, {y:.3f}) is out of reach - skipping.")
                station.idle(1.0)
                continue

            station.show_target(np.array([x, y, grasp_z]))
            station.set_status(f"Point {i}/{len(grid)}: moving to ({x:.3f}, {y:.3f})...")
            q_hover[5] = args.gripper_open
            station.joint_move(q_hover)
            station.cart_move(topdown_pose(x, y, grasp_z, yaw), args.speed)

            def follow_height_slider():
                nonlocal grasp_z
                new_z = height_mm.value / 1000
                if abs(new_z - grasp_z) > 1e-4:
                    grasp_z = new_z
                    station.show_target(np.array([x, y, grasp_z]))
                    station.cart_move(topdown_pose(x, y, grasp_z, yaw), args.speed)

            station.set_status(
                f"**Point {i}/{len(grid)}.** Place the cube flat on the table between the open jaws, "
                "centred. Adjust *Grasp height* so the fingertips are just above the table. "
                "Then click **Continue** (or **Skip this point**)."
            )
            if station.wait_for(cont, skip, on_tick=follow_height_slider) is skip:
                station.cart_move(topdown_pose(x, y, args.hover_z, yaw), args.speed)
                station.joint_move(park_q)
                continue

            # Close to seat the cube against the fixed jaw (the same thing happens when picking).
            station.set_gripper(args.gripper_closed, 0.8)
            holding.append(float(station.q[5]))
            station.set_gripper(args.gripper_open)
            station.cart_move(topdown_pose(x, y, args.hover_z, yaw), args.speed)
            station.joint_move(park_q)
            station.show_target(None)
            station.idle(0.8)

            while True:
                det = station.average_detection()
                if det is not None:
                    break
                station.set_status(
                    "**Cube not detected.** Tune *Red detection*, then **Continue** to retry or **Skip this point**."
                )
                if station.wait_for(cont, skip) is skip:
                    break
            if det is None:
                continue
            points.append({"uv": [det.u, det.v], "xy": [x, y]})
            station.set_status(f"Point {i}: pixel ({det.u:.1f}, {det.v:.1f}) -> robot ({x:.3f}, {y:.3f}). Remove the cube.")
            station.idle(1.5)

        if len(points) < 4:
            station.set_status(f"**Only {len(points)} points collected; need at least 4.** Nothing saved.")
            return

        h, rms, loo = fit(points)
        cal = Calibration(
            camera=camera,
            detection=station.detection_settings(),
            homography=h.tolist(),
            grasp_z=height_mm.value / 1000,
            hover_z=args.hover_z,
            gripper_open=args.gripper_open,
            gripper_closed=args.gripper_closed,
            gripper_holding=float(np.median(holding)) if holding else None,
            park_q=park_q.tolist(),
            points=points,
            fit_error_mm=rms,
        )
        cal.save(args.out)
        loo_txt = f" · leave-one-out error: mean {np.mean(loo):.1f} mm, max {np.max(loo):.1f} mm" if loo else ""
        station.set_status(
            f"**Saved {args.out.name}** with {len(points)} points. Fit RMS {rms:.1f} mm{loo_txt}.  \n"
            "Leave-one-out error is the honest accuracy estimate; under ~5 mm is good."
        )
        station.idle(2.0)
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        station.disconnect()


if __name__ == "__main__":
    main()
