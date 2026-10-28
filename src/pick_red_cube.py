#!/usr/bin/env python
"""Detect the red cube with the camera and pick it up with the SO-101 (top-down grasp).

Needs cube_calibration.json from calibrate_cube_grasp.py. The viser page shows the camera
with the detection, and a red ball where the arm would grasp. Nothing moves until you click
**Pick**: the arm opens the gripper, hovers above the cube, descends, closes, lifts and checks
the gripper reading to tell whether it is holding the cube. **Put back** sets it down again
and returns to the park pose.
"""

import argparse
import math
import threading
from pathlib import Path

import numpy as np

from cube_pick_common import (
    DEFAULT_CALIB,
    Calibration,
    Station,
    choose_grasp_yaw,
    cube_yaw_in_robot,
    pixel_to_robot,
    topdown_pose,
    wrap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--speed", type=float, default=0.04, help="Straight-line speed (m/s)")
    p.add_argument("--viser-port", type=int, default=8080)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.calib.exists():
        raise FileNotFoundError(f"{args.calib} not found - run calibrate_cube_grasp.py first.")
    cal = Calibration.load(args.calib)
    h = np.array(cal.homography)
    park_q = np.array(cal.park_q)

    station = Station(args.port, args.id, cal.camera, cal.detection, viser_port=args.viser_port)
    gui = station.server.gui
    pick, put_back = threading.Event(), threading.Event()
    with gui.add_folder("Pick"):
        gui.add_button("Pick", color="green").on_click(lambda _: pick.set())
        put_back_btn = gui.add_button("Put back", disabled=True)
        put_back_btn.on_click(lambda _: put_back.set())

    def plan(det):
        """Grasp plan (x, y, yaw) for a detection, plus whether it is reachable."""
        x, y = pixel_to_robot(h, det.u, det.v)
        yaw = choose_grasp_yaw(x, y, cube_yaw_in_robot(h, det))
        reachable = (
            station.solve_topdown(x, y, cal.hover_z, yaw) is not None
            and station.solve_topdown(x, y, cal.grasp_z, yaw) is not None
        )
        return x, y, yaw, reachable

    last = {"uv": None, "plan": None}

    def preview():
        det = station.detection
        if det is None:
            station.show_target(None)
            station.status_md.content = "**No cube detected.** Place the red cube in the workspace (or tune *Red detection*)."
            return
        if last["uv"] is None or np.hypot(det.u - last["uv"][0], det.v - last["uv"][1]) > 1.0:
            last["uv"], last["plan"] = (det.u, det.v), plan(det)  # IK only when the cube moved
        x, y, yaw, reachable = last["plan"]
        station.show_target(np.array([x, y, cal.grasp_z]))
        station.status_md.content = (
            f"Cube at pixel ({det.u:.0f}, {det.v:.0f}) -> robot ({x:.3f}, {y:.3f}) m, jaw yaw {math.degrees(wrap(yaw)):.0f} deg  \n"
            + ("**Reachable** - click **Pick**." if reachable else "**Out of reach** for a top-down grasp - move the cube.")
        )

    station.connect()
    try:
        station.joint_move(park_q)
        while True:
            station.wait_for(pick, on_tick=preview)
            det = station.average_detection()
            if det is None:
                station.set_status("Cube not detected reliably - try again.")
                continue
            x, y, yaw, reachable = plan(det)
            if not reachable:
                station.set_status(f"({x:.3f}, {y:.3f}) is out of reach - move the cube.")
                continue
            q_hover = station.solve_topdown(x, y, cal.hover_z, yaw)
            q_hover[5] = cal.gripper_open
            station.show_target(np.array([x, y, cal.grasp_z]))

            station.set_status(f"Picking at ({x:.3f}, {y:.3f})...")
            station.set_gripper(cal.gripper_open)
            station.joint_move(q_hover)
            station.cart_move(topdown_pose(x, y, cal.grasp_z, yaw), args.speed)
            station.set_gripper(cal.gripper_closed, 0.8)
            station.cart_move(topdown_pose(x, y, cal.hover_z, yaw), args.speed)
            station.idle(0.3)

            reading = station.q[5]
            if cal.gripper_holding is not None:
                holding = abs(reading - cal.gripper_holding) < max(8.0, 0.5 * abs(cal.gripper_holding - cal.gripper_closed))
            else:
                holding = abs(reading - cal.gripper_closed) > 5.0
            if not holding:
                station.set_status(f"**Missed** (gripper reading {reading:.0f}). Returning to park.")
                station.set_gripper(cal.gripper_open)
                station.joint_move(park_q)
                continue

            station.set_status(f"**Holding the cube** (gripper reading {reading:.0f}). Click **Put back** to set it down.")
            put_back_btn.disabled = False
            station.wait_for(put_back)
            put_back_btn.disabled = True
            station.cart_move(topdown_pose(x, y, cal.grasp_z + 0.003, yaw), args.speed)
            station.set_gripper(cal.gripper_open)
            station.cart_move(topdown_pose(x, y, cal.hover_z, yaw), args.speed)
            station.joint_move(park_q)
            last["uv"] = None
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        station.disconnect()


if __name__ == "__main__":
    main()
