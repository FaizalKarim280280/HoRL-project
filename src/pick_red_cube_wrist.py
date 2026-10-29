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

import cv2
import numpy as np
import trimesh
from viser.transforms import SO3

from cube_pick_common import ALL_JOINTS, Detection, Station, tilted_pose, wrap
from plate_detection import BlueSettings, PlateDetection, detect_plate
from wrist_vision import WRIST_CALIB_FILE, WristCalibration, pixel_to_plane


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--calib", type=Path, default=WRIST_CALIB_FILE)
    p.add_argument(
        "--camera-index", default=None,
        help="Wrist camera: name (e.g. Innomaker) or index; default: the one stored in the calibration file",
    )
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
    p.add_argument("--plate-size", type=float, default=0.10, help="Blue plate edge length (m)")
    p.add_argument("--plate-height", type=float, default=0.01, help="Blue plate thickness (m)")
    p.add_argument("--drop-height", type=float, default=0.015, help="Cube bottom above the plate when it is dropped (m)")
    p.add_argument(
        "--plate-offset", type=float, nargs=2, default=[0.0, 0.0], metavar=("DX", "DY"),
        help="Constant correction (m) added to the detected plate centre (separate from --target-offset)",
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
    if args.camera_index is not None:
        cal.camera.index = args.camera_index
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
        gui.add_button("Pick & place", color="green").on_click(lambda _: pick.set())
        place_box = gui.add_checkbox("Place on blue plate", initial_value=True)
        gui.add_button("Search wider").on_click(lambda _: search.set())
        auto_search = gui.add_checkbox("Auto-search on Pick if not visible", initial_value=True)
        put_back_btn = gui.add_button("Put back", disabled=True)
        put_back_btn.on_click(lambda _: put_back.set())
        gui.add_button("Back to scan pose").on_click(lambda _: rescan.set())
        # Start high and lower in ~10 mm steps until the fingertips reach the cube: a too-high
        # grasp just misses, a too-low one would push the fingers into the table.
        height_mm = gui.add_slider("Grasp height (mm, base frame)", -80, 40, 1, round(args.grasp_z * 1000))
        save_btn = gui.add_button("Save detection settings")

    blue = BlueSettings.load()
    with gui.add_folder("Blue detection", expand_by_default=False):
        blue_sliders = {
            "h_min": gui.add_slider("Hue min", 80, 140, 1, blue.h_min),
            "h_max": gui.add_slider("Hue max", 90, 150, 1, blue.h_max),
            "s_min": gui.add_slider("Saturation min", 0, 255, 1, blue.s_min),
            "v_min": gui.add_slider("Value min", 0, 255, 1, blue.v_min),
            "min_area": gui.add_slider("Min area (px)", 100, 20000, 100, blue.min_area),
        }

    def blue_settings() -> BlueSettings:
        return BlueSettings(**{k: int(sl.value) for k, sl in blue_sliders.items()})

    @save_btn.on_click
    def _(_) -> None:
        cal.detection = station.detection_settings()
        cal.save(args.calib)
        blue_settings().save()
        print(f"Saved red detection settings to {args.calib} and blue to the calibration folder")

    plate_top_z = table_z + args.plate_height
    plate_offset = args.plate_offset
    # 3D models of the tray and the cube, built from their real dimensions and placed where the
    # camera estimates them (the cube follows the gripper while it is held).
    def box_mesh(sx: float, sy: float, sz: float, rgba: tuple) -> trimesh.Trimesh:
        mesh = trimesh.creation.box(extents=(sx, sy, sz))
        mesh.visual.face_colors = rgba
        return mesh

    plate_marker = station.server.scene.add_mesh_trimesh(
        "/plate", box_mesh(args.plate_size, args.plate_size, args.plate_height, (40, 90, 255, 255)), visible=False
    )
    cube_mesh = station.server.scene.add_mesh_trimesh(
        "/cube", box_mesh(args.cube_size, args.cube_size, args.cube_size, (220, 30, 35, 255)), visible=False
    )
    cube_state = {"held": False}

    def yaw_wxyz(yaw: float) -> tuple:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def show_cube(x: float, y: float, yaw: float, bottom_z: float) -> None:
        """Cube resting with its bottom at bottom_z (table or tray top)."""
        cube_mesh.position, cube_mesh.wxyz, cube_mesh.visible = (x, y, bottom_z + args.cube_size / 2), yaw_wxyz(yaw), True

    def follow_gripper() -> None:
        """Held cube: it sits between the jaws, its bottom `grip height` below the fingertips."""
        tcp = station.tcp_pose(station.q)
        grip_above_bottom = height_mm.value / 1000 - table_z
        # the TCP z axis points down through the fingers; the cube centre is above the fingertips
        centre = tcp[:3, 3] - tcp[:3, 2] * (args.cube_size / 2 - grip_above_bottom)
        cube_mesh.position, cube_mesh.wxyz, cube_mesh.visible = centre, SO3.from_matrix(tcp[:3, :3]).wxyz, True
    plate_seen: dict = {"det": None}

    def draw_plate(view):
        """Camera overlay: blue tint + outline + corners of the detected plate (and keeps the held
        cube's 3D model on the gripper)."""
        if cube_state["held"]:
            follow_gripper()
        if station.frame is None:
            return view
        det, mask = detect_plate(station.frame, blue_settings())
        plate_seen["det"] = det
        view[mask > 0] = (0.5 * view[mask > 0] + [0, 0, 127]).astype(np.uint8)
        if det is not None:
            cv2.polylines(view, [np.round(det.corners).astype(np.int32)], True, (60, 140, 255), 2)
            c = det.corners.mean(axis=0)
            cv2.drawMarker(view, (int(c[0]), int(c[1])), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        return view

    station.overlay = draw_plate

    def locate_plate(det: PlateDetection) -> tuple[float, float, float, float] | None:
        """Plate (x, y, yaw, measured edge length) in the base frame, at the current arm pose.

        Each image corner is projected onto the plate's top surface; the centre is the mean of
        the 4 table points, which is correct under perspective (unlike the image centroid).
        """
        t_base_cam = station.kin.forward_kinematics(station.q) @ t_gripper_cam
        pts = [pixel_to_plane(u, v, k, dist, t_base_cam, plate_top_z) for u, v in det.corners]
        if any(p_ is None for p_ in pts):
            return None
        pts = np.array(pts)[:, :2]
        edge = pts[1] - pts[0]
        side = float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))
        cx, cy = pts.mean(axis=0) + np.asarray(plate_offset)
        return float(cx), float(cy), math.atan2(edge[1], edge[0]), side

    def measure_plate(frames: int = 6) -> tuple[float, float, float, float] | None:
        """Median plate estimate over several frames at the current arm pose."""
        found = []
        for _ in range(frames):
            station.idle(station.dt)
            det, _ = detect_plate(station.frame, blue_settings()) if station.frame is not None else (None, None)
            loc = locate_plate(det) if det is not None else None
            if loc is not None:
                found.append(loc)
        if len(found) < frames // 2:
            return None
        x, y = float(np.median([f[0] for f in found])), float(np.median([f[1] for f in found]))
        yaw = math.atan2(np.mean([math.sin(4 * f[2]) for f in found]), np.mean([math.cos(4 * f[2]) for f in found])) / 4
        side = float(np.median([f[3] for f in found]))
        plate_marker.position, plate_marker.visible = (x, y, table_z + args.plate_height / 2), True
        plate_marker.wxyz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        print(f"Plate centre ({x:.3f}, {y:.3f}) m, yaw {math.degrees(yaw):.0f} deg, measured edge {side * 100:.1f} cm "
              f"(expected {args.plate_size * 100:.1f})")
        return x, y, yaw, side

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

    def plan_topdown(x: float, y: float, square_yaw: float, z: float, hovers: list[float]) -> dict | None:
        """First feasible gripper pose at (x, y, z): straight down if possible, else tilted up to
        30 deg; the jaws aligned with the square's edges (least wrist roll first); the highest
        hover height that the same orientation can reach."""
        radial = math.atan2(y, x)
        yaws = sorted({round(wrap(square_yaw + k * math.pi / 2), 6) for k in range(4)}, key=lambda a: abs(wrap(a - radial)))
        for tilt in (0, 10, -10, 20, -20, 30, -30):
            for yaw in yaws:
                if station.solve_pose(tilted_pose(x, y, z, yaw, tilt), yaw) is None:
                    continue
                for hover_z in hovers:
                    q_hover = station.solve_pose(tilted_pose(x, y, hover_z, yaw, tilt), yaw)
                    if q_hover is not None:
                        return {"x": x, "y": y, "yaw": yaw, "tilt": tilt, "hover_z": hover_z, "grasp_z": z, "q_hover": q_hover}
        return None

    def plan_grasp(x: float, y: float, cube_yaw: float) -> dict | None:
        return plan_topdown(x, y, cube_yaw, height_mm.value / 1000, hover_options)

    def plan_place(x: float, y: float, plate_yaw: float) -> dict | None:
        """Drop pose over the plate centre: the held cube's bottom `drop_height` above the plate,
        its edges aligned with the plate's; the approach (hover) pose is directly above it, at
        least 2 cm higher. The fingertips hold the cube at the same height above its bottom as
        when it was grasped."""
        grip_above_bottom = height_mm.value / 1000 - table_z
        z = plate_top_z + args.drop_height + grip_above_bottom
        min_hover = z + 0.02
        hovers = [h for h in (0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04) if h >= min_hover] or [min_hover]
        return plan_topdown(x, y, plate_yaw, z, hovers)

    def plan_carry(pick_plan: dict, place_plan: dict) -> float | None:
        """Highest carry height at which the gripper can travel in a straight, level line from
        above the cube to above the plate centre, pointing down the whole way (checked at 12
        points along the line). The held cube's bottom stays >= 2 cm above the plate."""
        grip_above_bottom = height_mm.value / 1000 - table_z
        min_z = plate_top_z + 0.02 + grip_above_bottom
        start_z = max(pick_plan["hover_z"], place_plan["hover_z"], min_z)
        for z in np.arange(start_z, min_z - 1e-9, -0.005):
            a, b = pose(pick_plan, z), pose(place_plan, z)
            rot = SO3.from_matrix(a[:3, :3].T @ b[:3, :3]).log()
            ok = True
            for s_ in np.linspace(0, 1, 12):
                t = np.eye(4)
                t[:3, 3] = a[:3, 3] + (b[:3, 3] - a[:3, 3]) * s_
                t[:3, :3] = a[:3, :3] @ SO3.exp(rot * s_).as_matrix()
                yaw = wrap(pick_plan["yaw"] + wrap(place_plan["yaw"] - pick_plan["yaw"]) * s_)
                if station.solve_pose(t, yaw) is None:
                    ok = False
                    break
            if ok:
                return float(z)
        return None

    def describe(plan: dict) -> str:
        how = "straight down" if plan["tilt"] == 0 else f"tilted {abs(plan['tilt'])} deg {'inward' if plan['tilt'] > 0 else 'outward'}"
        return f"{how}, jaw yaw {math.degrees(plan['yaw']):.0f} deg, hover {plan['hover_z'] * 1000:.0f} mm"

    def pose(plan: dict, z: float) -> np.ndarray:
        return tilted_pose(plan["x"], plan["y"], z, plan["yaw"], plan["tilt"])

    cache = {"uv": None, "plan": None}

    def preview() -> None:
        preview_cube()
        plate_line = ("Blue plate: **seen**" if plate_seen["det"] is not None else
                      "Blue plate: **not seen** (tune *Blue detection* if it is in view)") if place_box.value else ""
        station.status_md.content += "  \n" + plate_line

    def preview_cube() -> None:
        det = station.detection
        if det is None:
            station.show_target(None)
            if not cube_state["held"]:
                cube_mesh.visible = False
            station.status_md.content = "**No cube detected.** Place the red cube in view (or tune *Red detection*)."
            return
        if cache["uv"] is None or np.hypot(det.u - cache["uv"][0], det.v - cache["uv"][1]) > 4.0:
            where = locate(det)  # re-plan only when the object moved (planning tries many poses)
            cache["uv"], cache["plan"] = (det.u, det.v), (where, plan_grasp(*where) if where else None)
        where, plan = cache["plan"]
        if where is None:
            station.status_md.content = "Object seen, but its ray does not hit the table - check the calibration."
            return
        x, y, cube_yaw = where
        station.show_target(np.array([x, y, cube_top_z]))
        if not cube_state["held"]:
            show_cube(x, y, cube_yaw, table_z)
        station.status_md.content = (
            f"Cube at robot ({x:.3f}, {y:.3f}) m, {np.hypot(x, y) * 100:.0f} cm from the base  \n"
            + (f"**Grasp found** ({describe(plan)}) - click **Pick & place**." if plan else
               "**No grasp found** even tilted up to 30 deg - move the object (about 10-28 cm from the base works best).")
        )

    def go_scan() -> None:
        station.joint_move(scan_q, speed_deg_s=40)
        station.idle(0.5)
        cache["uv"] = None

    search_poses = find_search_poses(station, t_gripper_cam, scan_q, table_z)
    print(f"{len(search_poses)} wider search views prepared")

    def search_wider(look=None, what: str = "cube"):
        """Visits the wider views until `look()` finds the object; stays there if found."""
        look = look or station.average_detection
        for i, q in enumerate(search_poses, 1):
            station.set_status(f"Searching wider for the {what}: view {i}/{len(search_poses)}...")
            q = q.copy()
            q[5] = station.q_cmd[5]
            station.joint_move(q, speed_deg_s=40)
            found = look()
            if found is not None:
                cache["uv"] = None
                station.set_status(f"**Found** the {what} from search view {i}.")
                return found
        station.set_status(f"**No {what} found** from any search view. Move it into the workspace and try again.")
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
            plate = None
            if place_box.value:  # find the plate first, so we never pick up a cube we cannot place
                plate = measure_plate()
                if plate is None and auto_search.value:
                    plate = search_wider(measure_plate, "blue plate")
                    if plate is not None:
                        go_scan()  # look for the cube from the normal scan pose again
                if plate is None:
                    station.set_status("**No blue plate found.** Put it in view (or tune *Blue detection*), or untick *Place on blue plate*.")
                    continue
                if plan_place(plate[0], plate[1], plate[2]) is None:
                    station.set_status(f"Plate at ({plate[0]:.3f}, {plate[1]:.3f}) is out of reach for placing - move it closer.")
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
            cube_state["held"] = True
            station.cart_move(pose(plan, plan["hover_z"]), args.speed)
            station.idle(0.3)

            reading = station.q[5]
            if abs(reading - args.gripper_closed) < 5.0:
                station.set_status(
                    f"**Missed** (gripper closed fully: {reading:.0f}) at height {grasp_z * 1000:.0f} mm. If the fingers "
                    "closed above the cube, lower *Grasp height* ~10 mm and Pick again. Returning to scan pose."
                )
                cube_state["held"] = False
                show_cube(plan["x"], plan["y"], plan["yaw"], table_z)  # still where it was
                station.set_gripper(args.gripper_open)
                go_scan()
                continue
            if plate is not None:
                place = plan_place(plate[0], plate[1], plate[2])
                if place is None:
                    station.set_status("**Holding the cube, but the plate is out of reach.** Click **Put back**.")
                else:
                    # 1) lift straight up, 2) straight level line to above the plate centre,
                    # 3) straight down over the centre, 4) check position, 5) drop, 6) back up
                    carry_z = plan_carry(plan, place)
                    if carry_z is not None:
                        station.set_status(f"Lifting to {carry_z * 1000:.0f} mm, then moving above the plate centre "
                                           f"({place['x']:.3f}, {place['y']:.3f})...")
                        station.cart_move(pose(plan, carry_z), args.speed)
                        station.cart_move(pose(place, carry_z), args.speed)
                    else:
                        print("No straight level path to the plate is reachable pointing down; using a joint move instead.")
                        q_place_hover = place["q_hover"].copy()
                        q_place_hover[5] = station.q_cmd[5]  # keep the gripper closed on the cube
                        station.joint_move(q_place_hover)
                    station.idle(0.3)
                    station.set_status(f"Lowering to {args.drop_height * 100:.1f} cm above the plate and dropping...")
                    drop_pose = pose(place, place["grasp_z"])
                    station.cart_move(drop_pose, args.speed / 2)
                    reached = station.tcp_pose(station.q)[:3, 3]  # for information only: no corrections (they oscillate)
                    print(f"Drop: plate target ({place['x']:.3f}, {place['y']:.3f}), reached ({reached[0]:.3f}, {reached[1]:.3f}), "
                          f"off by {np.hypot(reached[0] - place['x'], reached[1] - place['y']) * 1000:.0f} mm")
                    station.set_gripper(args.gripper_open, 0.3)
                    cube_state["held"] = False
                    show_cube(reached[0], reached[1], place["yaw"], plate_top_z)  # now on the tray
                    station.cart_move(pose(place, carry_z or place["hover_z"]), args.speed)
                    go_scan()
                    station.set_status("**Placed** the cube on the blue plate.")
                    continue
            station.set_status(f"**Holding the cube** (gripper reading {reading:.0f}). Click **Put back** to set it down.")
            put_back_btn.disabled = False
            station.wait_for(put_back)
            put_back_btn.disabled = True
            station.cart_move(pose(plan, grasp_z + 0.003), args.speed)
            station.set_gripper(args.gripper_open)
            cube_state["held"] = False
            show_cube(plan["x"], plan["y"], plan["yaw"], table_z)
            station.cart_move(pose(plan, plan["hover_z"]), args.speed)
            go_scan()
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        station.disconnect()


if __name__ == "__main__":
    main()
