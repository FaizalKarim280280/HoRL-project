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

from cube_pick_common import Detection, Station, Stopped, tilted_pose, wrap
from plate_detection import BlueSettings, PlateDetection, detect_plate, plate_centre
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
    p.add_argument("--max-place-tilt", type=float, default=50.0,
                   help="Placing: how far (deg) the gripper may lean from straight down to reach a far tray (grasping stays within 30)")
    p.add_argument(
        "--plate-offset", type=float, nargs=2, default=[0.0, 0.0], metavar=("DX", "DY"),
        help="Constant correction (m) added to the detected plate centre (separate from --target-offset)",
    )
    p.add_argument("--gripper-open", type=float, default=50.0)
    p.add_argument("--gripper-closed", type=float, default=0.0)
    p.add_argument("--speed", type=float, default=0.04, help="Straight-line speed (m/s)")
    p.add_argument("--viser-port", type=int, default=8080)
    return p.parse_args()


PLACE_STAGES = [
    "Find tray", "Find cube", "Move above cube", "Close-up check of cube", "Descend & grasp", "Lift",
    "Carry to tray", "Close-up check of tray", "Lower & drop", "Return to scan pose", "Check landing",
]  # fmt: skip
HOLD_STAGES = ["Find cube", "Move above cube", "Close-up check of cube", "Descend & grasp", "Lift", "Holding (Put back)"]


class StageOutline:
    """Checklist of the current pick-and-place in a panel docked on the left of the viser page:
    done stages ticked with a short result, the current one highlighted, the rest still open."""

    def __init__(self, gui, on_end=None):
        panel = gui.add_panel()
        panel.dock_left()
        with panel.add_tab("Pick & place"):
            self.md = gui.add_markdown("Idle - click **Pick & place**.")
        self.on_end = on_end  # called when an attempt ends (clears the trajectories)
        self.stages: list[str] = []
        self.notes: dict[str, str] = {}
        self.current = -1
        self.footer = ""

    def begin(self, stages: list[str]) -> None:
        self.stages, self.notes, self.current, self.footer = list(stages), {}, -1, ""
        self._render()

    def step(self, name: str, note: str = "") -> None:
        self.current = self.stages.index(name)
        if note:
            self.notes[name] = note
        self._render()

    def note(self, name: str, text: str) -> None:
        self.notes[name] = text
        self._render()

    def finish(self, text: str) -> None:
        self.current, self.footer = len(self.stages), f"**Done:** {text}"
        self._render()
        if self.on_end:
            self.on_end()

    def fail(self, text: str) -> None:
        self.footer = f"**Stopped** at *{self.stages[max(self.current, 0)]}*: {text}" if self.stages else f"**Stopped:** {text}"
        self.current = -(self.current + 2)  # remember where it stopped (negative = failed)
        self._render()
        if self.on_end:
            self.on_end()

    def _render(self) -> None:
        failed_at = -self.current - 2 if self.current < -1 else None
        lines = []
        for i, name in enumerate(self.stages):
            note = f" - {self.notes[name]}" if name in self.notes else ""
            if failed_at is not None:
                mark = "✔" if i < failed_at else ("✖" if i == failed_at else "○")
                text = f"**{name}**" if i == failed_at else name
            else:
                mark = "✔" if i < self.current else ("▶" if i == self.current else "○")
                text = f"**{name}**" if i == self.current else name
            lines.append(f"{mark} {i + 1}. {text}{note}")
        self.md.content = "  \n".join(lines) + (f"\n\n{self.footer}" if self.footer else "")


def find_search_poses(station: Station, t_gripper_cam: np.ndarray, scan_q: np.ndarray, table_z: float,
                      pan_range: float = 60.0, pan_step: float = 20.0) -> tuple[np.ndarray, list[np.ndarray]]:
    """The high view: the highest camera position (shoulder/elbow/wrist offsets from the scan
    pose) that still looks down at the same area (within 4 cm), so more of the table is visible.
    Plus the sweep: that view panned from the robot's left to its right in pan_step steps up to
    +-pan_range (views outside the joint limits are skipped)."""
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
                if not station.within_limits(q, 5) or station.kin.forward_kinematics(q)[2, 3] < 0.10:
                    continue
                v = view(q)
                if v and abs(v[1] - look_r) <= 0.04 and v[0] > best_h + 0.02:
                    best, best_h = q, v[0]
    high = best if best is not None else scan_q
    poses = []
    for d_pan in np.arange(-pan_range, pan_range + 1e-9, pan_step):  # positive pan turns the arm toward -y (right)
        q = high.copy()
        q[0] += d_pan
        if station.within_limits(q, 5):
            poses.append(q)
    return high, poses


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
    outline = StageOutline(gui, on_end=station.clear_trajectories)
    pick, put_back, rescan, search, go_high, open_req = (threading.Event() for _ in range(6))
    with gui.add_folder("Pick"):
        gui.add_button("Pick & place", color="green").on_click(lambda _: pick.set())
        gui.add_button("Stop", color="red").on_click(lambda _: station.stop_requested.set())
        gui.add_button("Open gripper").on_click(lambda _: open_req.set())
        place_box = gui.add_checkbox("Place on blue plate", initial_value=True)
        gui.add_button("Sweep for cube & tray").on_click(lambda _: search.set())
        auto_search = gui.add_checkbox("Sweep on Pick if not visible", initial_value=True)
        put_back_btn = gui.add_button("Put back", disabled=True)
        put_back_btn.on_click(lambda _: put_back.set())
        gui.add_button("Back to scan pose").on_click(lambda _: rescan.set())
        gui.add_button("High view (see more)").on_click(lambda _: go_high.set())
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
    # Constant correction of the tray centre (base frame), live-tunable: the camera->table geometry
    # has the same kind of error that --target-offset corrects for the cube (largely the hand-eye
    # table height disagreeing with TABLE_Z), so the tray needs its own correction too.
    with gui.add_folder("Tray offset"):
        plate_dx = gui.add_slider("Tray x offset (mm, + = away from base)", -40, 40, 1, round(args.plate_offset[0] * 1000))
        plate_dy = gui.add_slider("Tray y offset (mm, + = robot's left)", -40, 40, 1, round(args.plate_offset[1] * 1000))

    @plate_dx.on_update
    @plate_dy.on_update
    def _(_) -> None:
        print(f'Tray offset now ({plate_dx.value}, {plate_dy.value}) mm - to keep it, set in common.sh: '
              f'PLATE_OFFSET="{plate_dx.value / 1000:g} {plate_dy.value / 1000:g}"')
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
        blob_size["tray"] = None
        det, mask = detect_plate(station.frame, blue_settings(), tray_ok)
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

        The outline's real ~90 deg corners are found on the plate's top surface (see
        plate_detection.plate_centre), so a corner hidden by the gripper or the cube does not
        shift the centre. Falls back to the mean of the 4 fitted image corners.
        """
        t_base_cam = station.kin.forward_kinematics(station.q) @ t_gripper_cam

        def to_table(u, v):
            return pixel_to_plane(u, v, k, dist, t_base_cam, plate_top_z)

        pts = [to_table(u, v) for u, v in det.corners]
        if any(p_ is None for p_ in pts):
            return None
        pts = np.array(pts)[:, :2]
        side = float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))
        found = plate_centre(det.contour.reshape(-1, 2), to_table, args.plate_size)
        if found is not None:
            centre, yaw, method = found
        else:
            edge = pts[1] - pts[0]
            centre, yaw, method = pts.mean(axis=0), math.atan2(edge[1], edge[0]), "fitted quad (fallback)"
        plate_seen["method"] = method
        cx, cy = np.asarray(centre) + np.array([plate_dx.value, plate_dy.value]) / 1000
        return float(cx), float(cy), float(yaw), side, method

    def measure_plate(frames: int = 6, full_only: bool = False) -> tuple[float, float, float, float] | None:
        """Median plate estimate over several frames at the current arm pose. With full_only, a
        frame counts only if the whole tray is in view (its outline does not touch the image edge)
        and its centre came from all 4 corners or a full diagonal (at most one corner hidden, e.g.
        by the gripper) - a partial view would pull the centre towards the visible part."""
        found = []
        for _ in range(frames):
            station.idle(station.dt)
            det, _ = detect_plate(station.frame, blue_settings(), tray_ok) if station.frame is not None else (None, None)
            if det is None or (full_only and det.touches_border):
                continue
            loc = locate_plate(det)
            if loc is not None and (not full_only or loc[4] in ("4 corners", "diagonal")):
                found.append(loc)
        if len(found) < frames // 2:
            return None
        x, y = float(np.median([f[0] for f in found])), float(np.median([f[1] for f in found]))
        yaw = math.atan2(np.mean([math.sin(4 * f[2]) for f in found]), np.mean([math.cos(4 * f[2]) for f in found])) / 4
        side = float(np.median([f[3] for f in found]))
        plate_marker.position, plate_marker.wxyz, plate_marker.visible = (x, y, table_z + args.plate_height / 2), yaw_wxyz(yaw), True
        print(f"Plate centre ({x:.3f}, {y:.3f}) m from {plate_seen.get('method', '?')}, yaw {math.degrees(yaw):.0f} deg, measured edge {side * 100:.1f} cm "
              f"(expected {args.plate_size * 100:.1f})")
        return x, y, yaw, side

    def footprint(contour: np.ndarray, plane_z: float) -> tuple[float, float] | None:
        """Long and short side (m) of a blob's outline projected onto the plane z = plane_z,
        at the current arm pose: its real size on the table, whatever the camera distance."""
        px = contour.reshape(-1, 2).astype(np.float64)
        px = px[np.linspace(0, len(px) - 1, min(len(px), 48)).astype(int)]
        t_base_cam = station.kin.forward_kinematics(station.q) @ t_gripper_cam
        n = cv2.undistortPoints(px.reshape(-1, 1, 2), k, dist).reshape(-1, 2)
        rays = np.column_stack([n, np.ones(len(n))]) @ t_base_cam[:3, :3].T
        if np.any(rays[:, 2] > -1e-6):
            return None  # some ray does not go down to the table
        pts = t_base_cam[:3, 3] + rays * ((plane_z - t_base_cam[2, 3]) / rays[:, 2])[:, None]
        (_, _), (w, h), _ = cv2.minAreaRect((pts[:, :2] * 1000).astype(np.float32))  # in mm for precision
        return max(w, h) / 1000, min(w, h) / 1000

    def cube_ok(contour: np.ndarray) -> bool:
        """The red blob is roughly the cube's size. Seen at a slant (e.g. from the high or sweep
        views), its sides make it look up to ~3x longer than the top face; this only rejects
        things clearly too small or too big."""
        fp = footprint(contour, cube_top_z)
        return fp is not None and fp[1] >= 0.5 * args.cube_size and fp[0] <= 3.0 * args.cube_size

    blob_size: dict = {"tray": None}  # measured size of the largest blue blob, for the status line

    def tray_ok(contour: np.ndarray) -> bool:
        """The blue blob is roughly the tray's size. Loose on purpose: the tray's outer walls,
        and any error in the table height, make it measure larger than 10 cm; this only rejects
        things clearly too small or too big. If it is cut by the image edge, only a minimum."""
        fp = footprint(contour, plate_top_z)
        if fp is None or station.frame is None:
            return False
        x, y, w, h = cv2.boundingRect(contour)
        height, width = station.frame.shape[:2]
        cut = x <= 3 or y <= 3 or x + w >= width - 3 or y + h >= height - 3
        size = args.plate_size
        ok = fp[0] <= 1.8 * size and fp[1] >= (0.25 if cut else 0.5) * size
        if blob_size["tray"] is None:  # the largest blob is checked first
            blob_size["tray"] = (fp, ok)
        return ok

    station.cube_accept = cube_ok

    def locate(det: Detection, top_z: float | None = None) -> tuple[float, float, float] | None:
        """Object (x, y, edge direction) in the base frame from a detection at the current arm pose,
        with the cube's top face at top_z (default: cube standing on the table)."""
        top_z = cube_top_z if top_z is None else top_z
        t_base_cam = station.kin.forward_kinematics(station.q) @ t_gripper_cam
        centre = pixel_to_plane(det.u, det.v, k, dist, t_base_cam, top_z)
        a = math.radians(det.angle_deg)
        edge = pixel_to_plane(det.u + 15 * math.cos(a), det.v + 15 * math.sin(a), k, dist, t_base_cam, top_z)
        if centre is None or edge is None:
            return None
        cube_yaw = math.atan2(edge[1] - centre[1], edge[0] - centre[0])
        dx, dy = args.target_offset  # constant empirical correction of the camera estimate
        return float(centre[0]) + dx, float(centre[1]) + dy, cube_yaw

    # Hover heights to try, highest first, always >= 25 mm above the object's top.
    hover_options = sorted({h for h in (args.hover_z, 0.07, 0.06, 0.05, 0.04, 0.03) if h >= cube_top_z + 0.025}, reverse=True)
    if not hover_options:
        hover_options = [cube_top_z + 0.025]

    def plan_topdown(x: float, y: float, square_yaw: float, z: float, hovers: list[float], prefer_yaw: float | None = None,
                     tilts=(0, 10, -10, 20, -20, 30, -30)) -> dict | None:
        """First feasible gripper pose at (x, y, z): straight down if possible, else tilted by the
        smallest of `tilts` that reaches; the jaws aligned with the square's edges (least wrist
        roll first); the highest hover height that the same orientation can reach."""
        radial = math.atan2(y, x)
        # try first the jaw yaw closest to `prefer_yaw` (the current one, for placing: least wrist
        # turning) or else to the arm's reach direction (least wrist roll)
        ref = radial if prefer_yaw is None else prefer_yaw
        yaws = sorted({round(wrap(square_yaw + k * math.pi / 2), 6) for k in range(4)}, key=lambda a: abs(wrap(a - ref)))
        for tilt in tilts:
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

    PLACE_TILTS = sorted({t for a in range(0, int(args.max_place_tilt) + 1, 5) for t in (a, -a)}, key=lambda t: (abs(t), t > 0))

    def plan_place(x: float, y: float, plate_yaw: float, prefer_yaw: float | None = None) -> dict | None:
        """Drop pose over the plate centre: the held cube's bottom `drop_height` above the plate,
        its edges aligned with the plate's; the approach (hover) pose is directly above it, at
        least 2 cm higher. The fingertips hold the cube at the same height above its bottom as
        when it was grasped."""
        grip_above_bottom = height_mm.value / 1000 - table_z
        z = plate_top_z + args.drop_height + grip_above_bottom
        min_hover = z + 0.02
        hovers = [h for h in (0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04) if h >= min_hover] or [min_hover]
        # A dropped cube does not need a top-down gripper: reaching the tray matters more. Lean
        # angles are tried smallest first in 5 deg steps up to --max-place-tilt (leaning outward
        # reaches further), so a near tray is still placed straight down.
        return plan_topdown(x, y, plate_yaw, z, hovers, prefer_yaw, PLACE_TILTS) or plan_position_first(x, y, z, prefer_yaw)

    def plan_position_first(x: float, y: float, z: float, prefer_yaw: float | None = None) -> dict | None:
        """Drop pose when no lean angle reaches: the TCP exactly at (x, y, z), orientation whatever
        the arm ends up with (seeded from straight down and from leaning 20/40/60 deg outward).
        The approach pose is 2-5 cm straight above, reached the same way."""
        yaw = math.atan2(y, x) if prefer_yaw is None else prefer_yaw
        seeds = []
        for tilt in (0, -20, -40, -60):
            target = tilted_pose(x, y, z, yaw, tilt)
            q = station._seed(target, yaw)
            for _ in range(20):
                q = station.kin.inverse_kinematics(q, target @ station.t_tcp_flange, orientation_weight=0.1)
            seeds.append(q)
        q_drop = station.solve_position(np.array([x, y, z]), seeds)
        if q_drop is None:
            return None
        reached = station.tcp_pose(q_drop)
        lean = math.degrees(math.acos(np.clip(-reached[2, 2], -1.0, 1.0)))
        for dz in (0.05, 0.04, 0.03, 0.02):
            q_hover = station.solve_position(np.array([x, y, z + dz]), [q_drop])
            if q_hover is not None:
                return {"x": x, "y": y, "yaw": yaw, "tilt": -lean, "hover_z": z + dz, "grasp_z": z,
                        "q_hover": q_hover, "rot": reached[:3, :3]}
        return None

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
        if plan.get("rot") is not None:
            how = f"position first, leaning {abs(plan['tilt']):.0f} deg"
        else:
            how = "straight down" if plan["tilt"] == 0 else f"tilted {abs(plan['tilt'])} deg {'inward' if plan['tilt'] > 0 else 'outward'}"
        return f"{how}, jaw yaw {math.degrees(plan['yaw']):.0f} deg, hover {plan['hover_z'] * 1000:.0f} mm"

    def pose(plan: dict, z: float) -> np.ndarray:
        if plan.get("rot") is not None:  # position-first plan: keeps the orientation IK found
            t = np.eye(4)
            t[:3, :3], t[:3, 3] = plan["rot"], (plan["x"], plan["y"], z)
            return t
        return tilted_pose(plan["x"], plan["y"], z, plan["yaw"], plan["tilt"])

    cache = {"uv": None, "plan": None}

    def preview() -> None:
        preview_cube()
        measured = blob_size["tray"]
        size_txt = f" - measures {measured[0][0] * 100:.1f} x {measured[0][1] * 100:.1f} cm" if measured else ""
        if not place_box.value:
            plate_line = ""
        elif plate_seen["det"] is not None:
            plate_line = f"Blue plate: **seen**{size_txt}"
        elif measured:
            plate_line = f"Blue region{size_txt}, **not counted as the tray** (expected about {args.plate_size * 100:.0f} cm)"
        else:
            plate_line = "Blue plate: **not seen** (tune *Blue detection* if it is in view)"
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
        station.joint_move(scan_q, speed_deg_s=30)
        station.idle(0.5)
        cache["uv"] = None

    high_q, sweep_poses = find_search_poses(station, t_gripper_cam, scan_q, table_z, pan_range=60.0, pan_step=15.0)
    print(f"Sweep: {len(sweep_poses)} views from left to right")
    # Direction the scan pose looks in (from the base): panning the base turns the view by the same angle.
    scan_cam = station.kin.forward_kinematics(scan_q) @ t_gripper_cam
    scan_look = scan_cam[:3, 3] + scan_cam[:3, 2] * (table_z - scan_cam[2, 3]) / scan_cam[2, 2]
    scan_look_angle = math.atan2(scan_look[1], scan_look[0])

    def best_cluster(found: list[tuple]) -> tuple | None:
        """The position seen most often (sightings within 3 cm of each other agree): median x, y."""
        if not found:
            return None
        xy = np.array([f[:2] for f in found])
        votes = [np.sum(np.hypot(*(xy - p).T) < 0.03) for p in xy]
        i = int(np.argmax(votes))
        near = np.hypot(*(xy - xy[i]).T) < 0.03
        return (float(np.median(xy[near, 0])), float(np.median(xy[near, 1])), *found[i][2:])

    def sweep(want_cube: bool, want_tray: bool) -> tuple[tuple | None, tuple | None]:
        """Pans from the robot's left to its right through the sweep views and records every
        sighting of the cube and/or tray (only blobs with their real size count). It stops early
        only once every wanted object has been seen during this sweep; otherwise it goes through
        all views. Returns the position seen most often for each, or None."""
        cubes, trays = [], []
        for i, q in enumerate(sweep_poses, 1):
            seen = ", ".join(f"{name} {'seen' if found else 'not seen yet'}" for name, want, found in
                             (("cube", want_cube, cubes), ("tray", want_tray, trays)) if want)
            station.set_status(f"Sweeping left to right ({i}/{len(sweep_poses)}): {seen}")
            q = q.copy()
            q[5] = station.q_cmd[5]
            station.joint_move(q, speed_deg_s=20)
            station.idle(0.5)  # let the camera catch up with the arm
            if want_cube:
                det = station.average_detection(10, min_frac=0.4)
                where = locate(det) if det is not None else None
                if where is not None:
                    cubes.append(where)
            if want_tray:
                tray = measure_plate(frames=4)
                if tray is not None:
                    trays.append(tray)
            print(f"Sweep view {i}/{len(sweep_poses)}: " + ", ".join(
                f"{name} {len(found)} sighting(s)" for name, want, found in (("cube", want_cube, cubes), ("tray", want_tray, trays)) if want))
            if (cubes or not want_cube) and (trays or not want_tray):
                print("Sweep: " + " and ".join(n for n, w in (("cube", want_cube), ("tray", want_tray)) if w) + " seen, stopping early.")
                break
        else:
            missing = [n for n, w, f in (("cube", want_cube, cubes), ("tray", want_tray, trays)) if w and not f]
            if missing:
                print(f"Sweep: went through all {len(sweep_poses)} views; never saw the {' or the '.join(missing)}.")
        cache["uv"] = None
        return best_cluster(cubes), best_cluster(trays)

    def look_at(x: float, y: float) -> bool:
        """Turns the scan pose towards (x, y) by panning the base, for a sharper look."""
        q = scan_q.copy()
        q[0] -= math.degrees(wrap(math.atan2(y, x) - scan_look_angle))  # positive pan turns the arm toward -y
        q[5] = station.q_cmd[5]
        if not station.within_limits(q, 5):
            return False
        station.joint_move(q, speed_deg_s=30)
        station.idle(0.4)
        cache["uv"] = None
        return True

    def retreat() -> None:
        """Back to the scan pose from wherever the arm was stopped: first straight up if the
        gripper is low (so it does not drag across the table, the cube or the tray)."""
        here = station.tcp_pose(station.q_cmd)
        if here[2, 3] < table_z + 0.08:
            up = here.copy()
            up[2, 3] = table_z + 0.08
            station.cart_move(up, args.speed)
        go_scan()

    station.connect()
    try:
        station.set_gripper(args.gripper_open)
        go_scan()
        while True:
            try:
                event = station.wait_for(pick, rescan, search, go_high, open_req, on_tick=preview)
                if event is open_req:
                    station.set_gripper(args.gripper_open)
                    cube_state["held"] = False
                    continue
                if event is rescan:
                    go_scan()
                    continue
                if event is go_high:  # stays there; Pick & place looks from here, Back to scan pose returns
                    q = high_q.copy()
                    q[5] = station.q_cmd[5]
                    station.joint_move(q, speed_deg_s=30)
                    cache["uv"] = None
                    continue
                if event is search:
                    cube_found, tray_found = sweep(True, place_box.value)
                    go_scan()
                    station.set_status("Sweep: " + "; ".join(
                        f"{name} {'at (%.3f, %.3f) m' % f[:2] if f else 'not found'}"
                        for name, f in (("cube", cube_found), ("tray", tray_found)) if name == "cube" or place_box.value))
                    continue
                station.clear_trajectories()  # draw this pick-and-place's moves on a clean slate
                outline.begin(PLACE_STAGES if place_box.value else HOLD_STAGES)
                plate, where = None, None
                if place_box.value:  # find the plate first, so we never pick up a cube we cannot place
                    outline.step("Find tray")
                    plate = measure_plate()
                det = station.average_detection()
                where = locate(det) if det is not None else None
                need_tray, need_cube = place_box.value and plate is None, where is None
                if (need_tray or need_cube) and auto_search.value:
                    cube_found, tray_found = sweep(True, place_box.value)  # always both: stops only once both are seen
                    if need_tray and tray_found is not None:
                        plate = (measure_plate() if look_at(*tray_found[:2]) else None) or tray_found
                    if need_cube and cube_found is not None:
                        det = station.average_detection() if look_at(*cube_found[:2]) else None
                        where = (locate(det) if det is not None else None) or cube_found
                    if where is None or not need_cube:
                        go_scan()
                if place_box.value:
                    if plate is None:
                        station.set_status(f"**No blue tray found** (looking for a {args.plate_size * 100:.0f} cm square). "
                                           "Put it in reach (or tune *Blue detection*), or untick *Place on blue plate*.")
                        outline.fail("no blue tray found")
                        continue
                    if plan_place(plate[0], plate[1], plate[2]) is None:
                        station.set_status(f"**Tray out of reach** at {np.hypot(plate[0], plate[1]) * 100:.0f} cm from the base "
                                           "- move it to 10-28 cm.")
                        outline.fail("tray out of reach")
                        continue
                    outline.note("Find tray", f"({plate[0]:.3f}, {plate[1]:.3f}) m")
                outline.step("Find cube")
                if where is None:
                    station.set_status(f"**No red cube found** (looking for a {args.cube_size * 1000:.0f} mm cube) from the scan pose"
                                       f"{' or in any of the ' + str(len(sweep_poses)) + ' sweep views' if auto_search.value else ''}. "
                                       "Put it in reach (or tune *Red detection*).")
                    outline.fail("no red cube found")
                    continue
                plan = plan_grasp(*where)
                if plan is None:
                    station.set_status(f"**Cube out of reach** at {np.hypot(where[0], where[1]) * 100:.0f} cm from the base "
                                       "(no grasp even tilted 30 deg) - move it to 10-28 cm.")
                    outline.fail("cube out of reach")
                    continue
                outline.note("Find cube", f"({plan['x']:.3f}, {plan['y']:.3f}) m")
                outline.step("Move above cube")
                station.set_status(f"Scan estimate ({plan['x']:.3f}, {plan['y']:.3f}), {describe(plan)}. Moving above it...")
                q_hover = plan["q_hover"].copy()
                q_hover[5] = args.gripper_open
                station.joint_move(q_hover)

                # Close-up re-detection (twice): errors from the high scan pose shrink with distance.
                outline.step("Close-up check of cube")
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
                outline.note("Close-up check of cube", f"({x:.3f}, {y:.3f}) m")
                outline.step("Descend & grasp", describe(plan))
                station.show_target(np.array([x, y, cube_top_z]))
                station.set_status(f"Grasping at ({x:.3f}, {y:.3f}), gripper height {grasp_z * 1000:.0f} mm, {describe(plan)}...")
                station.cart_move(pose(plan, grasp_z), args.speed)
                station.set_gripper(args.gripper_closed, 0.8)
                # The jaws stop on the cube; commanding "fully closed" would sit >5 units beyond where the
                # gripper is, and the safety limiter would clamp it to "5 tighter than now" every step.
                # Command that squeeze directly: same grip force, no constant clamping.
                hold = station.q_cmd.copy()
                hold[5] = max(args.gripper_closed, station.q[5] - 4.5) if args.gripper_closed < args.gripper_open \
                    else min(args.gripper_closed, station.q[5] + 4.5)
                station.send(hold)
                cube_state["held"] = True
                outline.step("Lift")
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
                    outline.fail("missed the cube (gripper closed on nothing)")
                    continue
                if plate is not None:
                    place = plan_place(plate[0], plate[1], plate[2])
                    if place is None:
                        station.set_status("**Holding the cube, but the plate is out of reach.** Click **Put back**.")
                    else:
                        # 1) lift straight up, 2) straight level line to above the tray, 3) re-measure the tray
                        # close up and move over its centre, 4) straight down, 5) drop, 6) back to the scan
                        # pose, 7) report where the cube landed
                        def place_at(tray):
                            # keep the cube's current orientation as far as possible (it repeats every 90 deg)
                            return plan_place(tray[0], tray[1], tray[2], prefer_yaw=plan["yaw"])

                        place = place_at(plate) or place
                        outline.step("Carry to tray")
                        carry_z = plan_carry(plan, place)
                        if carry_z is not None:
                            station.set_status(f"Lifting to {carry_z * 1000:.0f} mm, then moving above the tray "
                                               f"({place['x']:.3f}, {place['y']:.3f})...")
                            station.cart_move(pose(plan, carry_z), args.speed)
                            station.cart_move(pose(place, carry_z), args.speed)
                        else:
                            print("No straight level path to the tray is reachable pointing down; using a joint move instead.")
                            q_place_hover = place["q_hover"].copy()
                            q_place_hover[5] = station.q_cmd[5]  # keep the gripper closed on the cube
                            station.joint_move(q_place_hover)
                        above_z = carry_z or place["hover_z"]

                        # Above the tray the camera is close and usually sees all of it: re-measure, and if
                        # the whole tray is in view, that measurement replaces the earlier (farther, often
                        # partial) estimate. Repeat after moving, until the centre stops moving.
                        outline.step("Close-up check of tray")
                        updated = False
                        for _ in range(3):
                            station.idle(0.4)
                            close = measure_plate(frames=8, full_only=True)
                            if close is None:
                                print("Tray not fully visible from above it; keeping the "
                                      f"{'updated' if updated else 'scan'} estimate ({plate[0]:.3f}, {plate[1]:.3f}).")
                                break
                            shift = float(np.hypot(close[0] - plate[0], close[1] - plate[1]))
                            if not (0.8 <= close[3] / args.plate_size <= 1.2) or shift > 0.08:
                                print(f"Full view of the tray looks wrong (edge {close[3] * 100:.1f} cm, shift {shift * 1000:.0f} mm); "
                                      "keeping the previous estimate.")
                                break
                            new_place = place_at(close)
                            if new_place is None:
                                print("Updated tray centre is out of reach; keeping the previous estimate.")
                                break
                            print(f"Tray centre updated from a full view: ({plate[0]:.3f}, {plate[1]:.3f}) -> "
                                  f"({close[0]:.3f}, {close[1]:.3f}), shift {shift * 1000:.0f} mm")
                            plate, place, updated = close, new_place, True
                            if shift < 0.003:
                                break  # already above the centre
                            station.cart_move(pose(place, above_z), args.speed)

                        outline.note("Close-up check of tray", f"({plate[0]:.3f}, {plate[1]:.3f}) m - "
                                     + ("updated from a full view" if updated else "scan estimate (tray not fully visible above it)"))
                        outline.step("Lower & drop")
                        station.idle(0.3)
                        station.set_status(f"Lowering to {args.drop_height * 100:.1f} cm above the tray and dropping ({describe(place)})...")
                        drop_pose = pose(place, place["grasp_z"])
                        station.cart_move(drop_pose, args.speed / 2)
                        # The descent weights orientation softly, which can cost a few mm of position.
                        # Position is what matters here: put the TCP exactly on the drop point (from the
                        # arm model, no camera, so this is a single correction and cannot oscillate).
                        target_xyz = drop_pose[:3, 3]
                        off = float(np.linalg.norm(station.tcp_pose(station.q_cmd)[:3, 3] - target_xyz))
                        if off > 0.002:
                            q_fix = station.solve_position(target_xyz, [station.q_cmd])
                            if q_fix is not None:
                                print(f"Drop point corrected by {off * 1000:.1f} mm (position first, orientation free)")
                                station.joint_move(q_fix, speed_deg_s=10)
                        reached = station.tcp_pose(station.q)[:3, 3]  # measured: includes motor sag
                        print(f"Drop: target ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}), "
                              f"reached ({reached[0]:.3f}, {reached[1]:.3f}, {reached[2]:.3f}) - "
                              f"{np.linalg.norm(reached - target_xyz) * 1000:.0f} mm off")
                        station.set_gripper(args.gripper_open, 0.3)
                        cube_state["held"] = False
                        show_cube(reached[0], reached[1], place["yaw"], plate_top_z)  # now on the tray
                        station.cart_move(pose(place, above_z), args.speed)
                        outline.step("Return to scan pose")
                        go_scan()

                        # Where did it land? Tray and cube seen from the same (scan) view.
                        tray_now = measure_plate() or plate
                        det = station.average_detection()
                        # a cube in the tray stands on its floor, so its top face is plate-height higher
                        landed = locate(det, plate_top_z + args.cube_size) if det is not None else None
                        outline.step("Check landing")
                        if landed is None:
                            station.set_status("**Dropped** on the tray (could not see the cube afterwards to check).")
                            outline.finish("dropped on the tray (could not see the cube afterwards)")
                            continue
                        miss = np.array([landed[0] - tray_now[0], landed[1] - tray_now[1]])
                        c_, s_ = math.cos(tray_now[2]), math.sin(tray_now[2])
                        in_tray = np.array([c_ * miss[0] + s_ * miss[1], -s_ * miss[0] + c_ * miss[1]])  # tray axes
                        inside = bool(np.all(np.abs(in_tray) < args.plate_size / 2 - args.cube_size / 2))
                        if not inside:  # on the table after all: locate it on the table plane instead
                            landed = locate(det) or landed
                            miss = np.array([landed[0] - tray_now[0], landed[1] - tray_now[1]])
                        show_cube(landed[0], landed[1], landed[2], plate_top_z if inside else table_z)
                        print(f"Landed {np.linalg.norm(miss) * 1000:.0f} mm from the tray centre ({'inside' if inside else 'OUTSIDE'}), "
                              f"offset ({miss[0] * 1000:.0f}, {miss[1] * 1000:.0f}) mm")
                        outline.finish(f"landed {np.linalg.norm(miss) * 1000:.0f} mm from the tray centre, "
                                       f"{'inside' if inside else 'OUTSIDE'} the tray")
                        station.set_status(
                            f"**Placed**: the cube landed {np.linalg.norm(miss) * 1000:.0f} mm from the tray centre, "
                            f"**{'inside' if inside else 'OUTSIDE'}** the tray."
                        )
                        continue
                if place_box.value:
                    outline.fail("holding the cube, but the tray is out of reach - click Put back")
                else:
                    outline.step("Holding (Put back)")
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
                if not place_box.value:
                    outline.finish("put back")
            except Stopped:
                # Stop button: whatever was running is abandoned; go back to the scan pose. A held
                # cube stays in the gripper (it is not dropped mid-air) until Open gripper.
                put_back_btn.disabled = True
                if outline.stages and 0 <= outline.current < len(outline.stages):  # an attempt was running
                    outline.fail("stopped by the user")
                station.set_status("**Stopped.** Returning to the scan pose" + (
                    " - still holding the cube: take it out and click **Open gripper**." if cube_state["held"] else "."))
                station.show_target(None)
                while True:  # another Stop during the return just restarts it
                    station.stop_requested.clear()
                    try:
                        retreat()
                        break
                    except Stopped:
                        continue
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        station.disconnect()


if __name__ == "__main__":
    main()
