#!/usr/bin/env python
"""Find the red cube with the SO-101 wrist camera and pick it up (top-down grasp).

Needs wrist_calibration.json from calibrate_wrist_handeye.py. The arm goes to the scan pose
and looks for the cube. Each camera pixel is turned into a ray (intrinsics + FK + hand-eye)
and intersected with the plane of the cube's top face, giving its (x, y) on the table.
Nothing moves until you click **Pick**; the arm then hovers above the estimate, re-detects
the cube from close up to correct it, descends, grasps, lifts and checks the gripper reading.
"""

import argparse
import math
import threading
from pathlib import Path

import numpy as np

from cube_pick_common import ALL_JOINTS, Detection, Station, tilted_pose, wrap
from wrist_vision import WRIST_CALIB_FILE, WristCalibration, pixel_to_plane


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--calib", type=Path, default=WRIST_CALIB_FILE)
    p.add_argument("--cube-size", type=float, default=0.025, help="Cube edge length (m)")
    p.add_argument("--grasp-z", type=float, default=None, help="Gripper frame height at grasp in the base frame (m); default table + 12 mm; tunable live")
    p.add_argument(
        "--table-z", type=float, default=None,
        help="Table height in the base frame (m) for locating the cube; default: the board height measured by hand-eye",
    )
    p.add_argument("--hover-z", type=float, default=0.08, help="Height for the close-up re-detection (m); top-down reach ends ~8 cm up")
    p.add_argument(
        "--target-offset", type=float, nargs=2, default=[0.0, 0.0], metavar=("DX", "DY"),
        help="Constant correction (m) added to the detected object's x, y in the base frame",
    )
    p.add_argument("--gripper-open", type=float, default=50.0)
    p.add_argument("--gripper-closed", type=float, default=0.0)
    p.add_argument("--speed", type=float, default=0.04, help="Straight-line speed (m/s)")
    p.add_argument("--viser-port", type=int, default=8080)
    return p.parse_args()


def find_search_poses(station: Station, t_gripper_cam: np.ndarray, scan_q: np.ndarray, table_z: float) -> list[np.ndarray]:
    """Wider views for when the object is not visible from the scan pose: the highest camera
    position (shoulder/elbow/wrist offsets from the scan pose) that still looks down at the
    same area (within 4 cm), then the same view panned left and right."""
    limits = [station.joint_limits_deg[n] for n in ALL_JOINTS[:5]]

    def ok_limits(q):
        return all(lo + 5 <= q[i] <= hi - 5 for i, (lo, hi) in enumerate(limits))

    def view(q):
        cam = station.kin.forward_kinematics(q) @ t_gripper_cam
        axis = cam[:3, 2]  # optical axis
        if axis[2] > -math.cos(math.radians(50)):  # must look at least 40 deg below horizontal
            return None
        look = cam[:3, 3] + axis * (table_z - cam[2, 3]) / axis[2]
        return cam[2, 3], np.hypot(look[0], look[1])

    scan_view = view(scan_q)
    best, best_h = None, scan_view[0] if scan_view else 0.0
    look_r = scan_view[1] if scan_view else 0.22  # keep looking at the same area, just from higher
    for d_lift in range(-40, 1, 5):
        for d_elbow in range(-40, 41, 5):
            for d_flex in range(-40, 41, 5):
                q = scan_q.copy()
                q[1:4] += (d_lift, d_elbow, d_flex)
                if not ok_limits(q) or station.kin.forward_kinematics(q)[2, 3] < 0.10:
                    continue
                v = view(q)
                if v and abs(v[1] - look_r) <= 0.04 and v[0] > best_h + 0.02:
                    best, best_h = q, v[0]
    high = best if best is not None else scan_q
    poses = [high] if best is not None else []
    for d_pan in (35.0, -35.0):  # positive pan turns the arm toward -y
        q = high.copy()
        q[0] += d_pan
        if ok_limits(q):
            poses.append(q)
    return poses


def main() -> None:
    args = parse_args()
    if not args.calib.exists():
        raise FileNotFoundError(f"{args.calib} not found - run calibrate_wrist_handeye.py first.")
    cal = WristCalibration.load(args.calib)
    k, dist = cal.intrinsics.k_mat, cal.intrinsics.dist_vec
    t_gripper_cam = np.array(cal.t_gripper_cam)
    scan_q = np.array(cal.scan_q)
    scan_q[5] = args.gripper_open
    # The cube's top face lies on the table plane + cube size. The table height defaults to where
    # hand-eye calibration measured the board, which keeps the camera chain self-consistent.
    table_z = args.table_z if args.table_z is not None else (cal.board_z_m or 0.0)
    cube_top_z = table_z + args.cube_size
    if args.grasp_z is None:
        args.grasp_z = table_z + 0.012  # fingertips 12 mm above the table
    print(f"Table plane at z = {table_z * 1000:.1f} mm in the base frame (cube top at {cube_top_z * 1000:.1f} mm)")

    station = Station(args.port, args.id, cal.camera, cal.detection, viser_port=args.viser_port)
    gui = station.server.gui
    pick, put_back, rescan, search = threading.Event(), threading.Event(), threading.Event(), threading.Event()
    with gui.add_folder("Pick"):
        gui.add_button("Pick", color="green").on_click(lambda _: pick.set())
        gui.add_button("Search wider").on_click(lambda _: search.set())
        auto_search = gui.add_checkbox("Auto-search on Pick if not visible", initial_value=True)
        put_back_btn = gui.add_button("Put back", disabled=True)
        put_back_btn.on_click(lambda _: put_back.set())
        gui.add_button("Back to scan pose").on_click(lambda _: rescan.set())
        # Start high and lower in ~10 mm steps until the fingertips reach the cube: a too-high
        # grasp just misses, a too-low one would push the fingers into the table.
        height_mm = gui.add_slider("Grasp height (mm, base frame)", -80, 40, 1, round(args.grasp_z * 1000))
        save_btn = gui.add_button("Save detection settings")

    @save_btn.on_click
    def _(_) -> None:
        cal.detection = station.detection_settings()
        cal.save(args.calib)
        print(f"Saved detection settings to {args.calib}")

    def locate(det: Detection) -> tuple[float, float, float] | None:
        """Object (x, y, edge direction) in the base frame from a detection at the current arm pose."""
        t_base_cam = station.kin.forward_kinematics(station.q) @ t_gripper_cam
        centre = pixel_to_plane(det.u, det.v, k, dist, t_base_cam, cube_top_z)
        a = math.radians(det.angle_deg)
        edge = pixel_to_plane(det.u + 15 * math.cos(a), det.v + 15 * math.sin(a), k, dist, t_base_cam, cube_top_z)
        if centre is None or edge is None:
            return None
        cube_yaw = math.atan2(edge[1] - centre[1], edge[0] - centre[0])
        dx, dy = args.target_offset  # constant empirical correction of the camera estimate
        return float(centre[0]) + dx, float(centre[1]) + dy, cube_yaw

    # Hover heights to try, highest first, always >= 25 mm above the object's top.
    hover_options = sorted({h for h in (args.hover_z, 0.07, 0.06, 0.05, 0.04, 0.03) if h >= cube_top_z + 0.025}, reverse=True)
    if not hover_options:
        hover_options = [cube_top_z + 0.025]

    def plan_grasp(x: float, y: float, cube_yaw: float) -> dict | None:
        """First feasible grasp: straight down if possible, else tilted up to 30 deg; the jaws
        aligned with one of the four footprint edges (least wrist roll first); the highest
        hover height that the same orientation can reach."""
        radial = math.atan2(y, x)
        yaws = sorted({round(wrap(cube_yaw + k * math.pi / 2), 6) for k in range(4)}, key=lambda a: abs(wrap(a - radial)))
        grasp_z = height_mm.value / 1000
        for tilt in (0, 10, -10, 20, -20, 30, -30):
            for yaw in yaws:
                q_grasp = station.solve_pose(tilted_pose(x, y, grasp_z, yaw, tilt), yaw)
                if q_grasp is None:
                    continue
                for hover_z in hover_options:
                    q_hover = station.solve_pose(tilted_pose(x, y, hover_z, yaw, tilt), yaw)
                    if q_hover is not None:
                        return {"x": x, "y": y, "yaw": yaw, "tilt": tilt, "hover_z": hover_z, "grasp_z": grasp_z, "q_hover": q_hover}
        return None

    def describe(plan: dict) -> str:
        how = "straight down" if plan["tilt"] == 0 else f"tilted {abs(plan['tilt'])} deg {'inward' if plan['tilt'] > 0 else 'outward'}"
        return f"{how}, jaw yaw {math.degrees(plan['yaw']):.0f} deg, hover {plan['hover_z'] * 1000:.0f} mm"

    def pose(plan: dict, z: float) -> np.ndarray:
        return tilted_pose(plan["x"], plan["y"], z, plan["yaw"], plan["tilt"])

    cache = {"uv": None, "plan": None}

    def preview() -> None:
        det = station.detection
        if det is None:
            station.show_target(None)
            station.status_md.content = "**No cube detected.** Place the red cube in view (or tune *Red detection*)."
            return
        if cache["uv"] is None or np.hypot(det.u - cache["uv"][0], det.v - cache["uv"][1]) > 4.0:
            where = locate(det)  # re-plan only when the object moved (planning tries many poses)
            cache["uv"], cache["plan"] = (det.u, det.v), (where, plan_grasp(*where) if where else None)
        where, plan = cache["plan"]
        if where is None:
            station.status_md.content = "Object seen, but its ray does not hit the table - check the calibration."
            return
        x, y, _ = where
        station.show_target(np.array([x, y, cube_top_z]))
        station.status_md.content = (
            f"Object at robot ({x:.3f}, {y:.3f}) m, {np.hypot(x, y) * 100:.0f} cm from the base  \n"
            + (f"**Grasp found** ({describe(plan)}) - click **Pick**." if plan else
               "**No grasp found** even tilted up to 30 deg - move the object (about 10-28 cm from the base works best).")
        )

    def go_scan() -> None:
        station.joint_move(scan_q, speed_deg_s=40)
        station.idle(0.5)
        cache["uv"] = None

    search_poses = find_search_poses(station, t_gripper_cam, scan_q, table_z)
    print(f"{len(search_poses)} wider search views prepared")

    def search_wider() -> Detection | None:
        """Visits the wider views until the object is seen; stays there if found."""
        for i, q in enumerate(search_poses, 1):
            station.set_status(f"Searching wider: view {i}/{len(search_poses)}...")
            q = q.copy()
            q[5] = station.q_cmd[5]
            station.joint_move(q, speed_deg_s=40)
            det = station.average_detection()
            if det is not None:
                cache["uv"] = None
                station.set_status(f"**Found** the object from search view {i}. Click **Pick**, or **Back to scan pose**.")
                return det
        station.set_status("**Not found** from any search view. Move the object into the workspace and try again.")
        go_scan()
        return None

    station.connect()
    try:
        station.set_gripper(args.gripper_open)
        go_scan()
        while True:
            event = station.wait_for(pick, rescan, search, on_tick=preview)
            if event is rescan:
                go_scan()
                continue
            if event is search:
                search_wider()
                continue
            det = station.average_detection()
            if det is None and auto_search.value:
                det = search_wider()
            where = locate(det) if det is not None else None
            plan = plan_grasp(*where) if where is not None else None
            if plan is None:
                station.set_status("Object not detected reliably, or no grasp found - try again.")
                continue
            station.set_status(f"Scan estimate ({plan['x']:.3f}, {plan['y']:.3f}), {describe(plan)}. Moving above it...")
            q_hover = plan["q_hover"].copy()
            q_hover[5] = args.gripper_open
            station.joint_move(q_hover)

            # Close-up re-detection (twice): errors from the high scan pose shrink with distance.
            for _ in range(2):
                station.idle(0.4)
                det = station.average_detection()
                refined = locate(det) if det is not None else None
                if refined is None:
                    print("Object not visible from the hover pose (the camera may not see under the gripper); "
                          "using the scan estimate.")
                    break
                new_plan = plan_grasp(*refined) if np.hypot(refined[0] - plan["x"], refined[1] - plan["y"]) < 0.06 else None
                if new_plan is None:
                    print(f"Close-up estimate ({refined[0]:.3f}, {refined[1]:.3f}) unusable; keeping ({plan['x']:.3f}, {plan['y']:.3f}).")
                    break
                print(f"Refined ({plan['x']:.3f}, {plan['y']:.3f}) -> ({new_plan['x']:.3f}, {new_plan['y']:.3f})")
                plan = new_plan
                station.cart_move(pose(plan, plan["hover_z"]), args.speed)

            x, y, grasp_z = plan["x"], plan["y"], plan["grasp_z"]
            station.show_target(np.array([x, y, cube_top_z]))
            station.set_status(f"Grasping at ({x:.3f}, {y:.3f}), gripper height {grasp_z * 1000:.0f} mm, {describe(plan)}...")
            station.cart_move(pose(plan, grasp_z), args.speed)
            station.set_gripper(args.gripper_closed, 0.8)
            station.cart_move(pose(plan, plan["hover_z"]), args.speed)
            station.idle(0.3)

            reading = station.q[5]
            if abs(reading - args.gripper_closed) < 5.0:
                station.set_status(
                    f"**Missed** (gripper closed fully: {reading:.0f}) at height {grasp_z * 1000:.0f} mm. If the fingers "
                    "closed above the cube, lower *Grasp height* ~10 mm and Pick again. Returning to scan pose."
                )
                station.set_gripper(args.gripper_open)
                go_scan()
                continue
            station.set_status(f"**Holding the cube** (gripper reading {reading:.0f}). Click **Put back** to set it down.")
            put_back_btn.disabled = False
            station.wait_for(put_back)
            put_back_btn.disabled = True
            station.cart_move(pose(plan, grasp_z + 0.003), args.speed)
            station.set_gripper(args.gripper_open)
            station.cart_move(pose(plan, plan["hover_z"]), args.speed)
            go_scan()
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        station.disconnect()


if __name__ == "__main__":
    main()
