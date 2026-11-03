"""Shared code for the wrist-camera scripts: robot + camera + viser station, red-cube detection,
grasp poses, inverse kinematics and smooth motions for the SO-101 follower."""

import json
import math
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf
from viser.transforms import SO3

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

ROOT = Path(__file__).resolve().parent.parent  # workspace root: src/, scripts/, assets/, calibration/, notebooks/, data/, lerobot/
ASSETS_DIR = ROOT / "assets"
CALIB_DIR = ROOT / "calibration"
DEFAULT_URDF = ASSETS_DIR / "SO101" / "so101_new_calib.urdf"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINTS = [*ARM_JOINTS, "gripper"]
CAMERA_NAME = "cam"
# Tool centre point (TCP) in the gripper_frame_link frame. gripper_frame_link sits on the inner
# face of the FIXED jaw at its tip; the moving jaw opens towards -x. When the jaws close on an
# object, the moving jaw pushes it against the fixed jaw, so the object's centre ends up half its
# width along -x: for a 25 mm wide object, 12.5 mm. Same fingertip depth, so grasp heights are
# still fingertip heights.
DEFAULT_TCP_OFFSET = (-0.0125, 0.0, 0.0)
# Speeds the motors actually reach (measured: wrist ~42 deg/s, gripper ~40 units/s). Commanding
# faster just makes lerobot's max_relative_target limiter clamp every step (and warn).
MAX_ROT_SPEED_DEG_S = 40.0
GRIPPER_SPEED = 40.0  # gripper units (0..100) per second


# --- camera selection -------------------------------------------------------------------------


def list_cameras_macos() -> list[str]:
    """Camera names in OpenCV's index order on macOS.

    OpenCV's AVFoundation backend numbers cameras by sorting them on their system unique ID
    (so external USB cameras, "0x...", usually come before the built-in "6C70..." one).
    system_profiler gives the same names and unique IDs, so sorting them the same way
    reproduces OpenCV's indices.
    """
    out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True, check=True)
    cams = json.loads(out.stdout).get("SPCameraDataType", [])
    return [c["_name"] for c in sorted(cams, key=lambda c: c.get("spcamera_unique-id", ""))]


def resolve_camera(spec: str) -> int | str:
    """Camera setting -> OpenCV index_or_path.

    A number is used as-is, an existing file path (e.g. /dev/video0) too; anything else is a
    camera NAME (case-insensitive substring, e.g. "Innomaker"), looked up so the script always
    opens that camera even if macOS renumbers them - never silently a different one.
    """
    if spec.isdigit():
        return int(spec)
    if Path(spec).exists():
        return spec
    if platform.system() != "Darwin":
        raise ValueError(f"Camera name lookup is only implemented on macOS; use an index or /dev/video path, not {spec!r}")
    names = list_cameras_macos()
    matches = [i for i, name in enumerate(names) if spec.lower() in name.lower()]
    if len(matches) != 1:
        found = ", ".join(f"{i}: {n}" for i, n in enumerate(names)) or "none"
        raise RuntimeError(f"Camera {spec!r} matched {len(matches)} cameras (found {found}). Is it plugged in?")
    print(f"Camera {spec!r} -> OpenCV index {matches[0]} ({names[matches[0]]})")
    return matches[0]


# --- settings saved in the calibration file ------------------------------------------------


@dataclass
class CameraSettings:
    index: str = "0"
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class DetectionSettings:
    # Red wraps around the hue circle (OpenCV hue is 0..180), so two ranges are used.
    h_low_max: int = 10
    h_high_min: int = 170
    s_min: int = 120
    v_min: int = 70
    min_area: int = 80  # pixels


# --- detection --------------------------------------------------------------------------------


@dataclass
class Detection:
    u: float
    v: float
    angle_deg: float  # in-image angle of one cube edge (minAreaRect)
    area: float
    contour: np.ndarray


def red_mask(rgb: np.ndarray, s: DetectionSettings) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    low = cv2.inRange(hsv, (0, s.s_min, s.v_min), (s.h_low_max, 255, 255))
    high = cv2.inRange(hsv, (s.h_high_min, s.s_min, s.v_min), (180, 255, 255))
    mask = low | high
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_red_cube(rgb: np.ndarray, s: DetectionSettings, accept=None) -> tuple[Detection | None, np.ndarray]:
    """Returns the largest red blob that `accept(contour)` agrees is the cube (or None), and the mask."""
    mask = red_mask(rgb, s)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in sorted(contours, key=cv2.contourArea, reverse=True) if cv2.contourArea(c) >= s.min_area]
    contour = next((c for c in contours if accept is None or accept(c)), None)
    if contour is None:
        return None, mask
    area = cv2.contourArea(contour)
    m = cv2.moments(contour)
    (_, _), (_, _), angle = cv2.minAreaRect(contour)
    return Detection(m["m10"] / m["m00"], m["m01"] / m["m00"], angle, area, contour), mask


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def topdown_pose(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    """4x4 gripper_frame_link pose pointing straight down, jaws closing along `yaw`.

    In this URDF the gripper's z axis points out through the fingers and the jaws close
    along its x axis.
    """
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = -np.array([math.cos(yaw), math.sin(yaw), 0.0])
    t = np.eye(4)
    t[:3, :3] = np.stack([x_axis, np.cross(z_axis, x_axis), z_axis], axis=1)
    t[:3, 3] = [x, y, z]
    return t


def tilted_pose(x: float, y: float, z: float, yaw: float, tilt_deg: float) -> np.ndarray:
    """topdown_pose tilted by tilt_deg about the horizontal axis across the arm's reach direction:
    positive tips the approach back toward the base, negative leans it outward (reaches further)."""
    t = topdown_pose(x, y, z, yaw)
    radial = math.atan2(y, x)
    axis = np.array([-math.sin(radial), math.cos(radial), 0.0])
    t[:3, :3] = SO3.exp(axis * math.radians(tilt_deg)).as_matrix() @ t[:3, :3]
    return t


# --- robot + camera + viser -----------------------------------------------------------------


class Station:
    """Owns the follower, its camera, the kinematics and the viser page.

    Everything runs on the calling thread; `tick()` must be called regularly (all the motion
    and wait helpers do) to refresh the observation, detection and visualization.
    """

    def __init__(
        self,
        port: str,
        robot_id: str,
        camera: CameraSettings,
        detection: DetectionSettings,
        fps: int = 30,
        viser_port: int = 8080,
        max_step_deg: float = 5.0,
        urdf: Path = DEFAULT_URDF,
        detect_red: bool = True,
        tcp_offset: tuple[float, float, float] = DEFAULT_TCP_OFFSET,
    ):
        self.detect_red = detect_red
        self.cube_accept = None  # optional callable(contour) -> bool: is this red blob the cube?
        self.overlay = None  # optional callable(rgb view) -> rgb view, drawn on the camera image
        index = resolve_camera(camera.index)
        cams = {CAMERA_NAME: OpenCVCameraConfig(index_or_path=index, width=camera.width, height=camera.height, fps=camera.fps)}
        self.robot = SO101Follower(
            SO101FollowerConfig(port=port, id=robot_id, use_degrees=True, max_relative_target=max_step_deg, cameras=cams)
        )
        self.kin = RobotKinematics(str(urdf), "gripper_frame_link", joint_names=ARM_JOINTS)
        self.dt = 1.0 / fps
        # All IK targets and cartesian moves refer to the TCP (centre between the jaws), not the flange.
        self.t_flange_tcp = np.eye(4)
        self.t_flange_tcp[:3, 3] = tcp_offset
        self.t_tcp_flange = np.linalg.inv(self.t_flange_tcp)

        urdf_model = yourdfpy.URDF.load(str(urdf))
        self.joint_limits_deg = {
            name: np.rad2deg([j.limit.lower, j.limit.upper]) for name, j in urdf_model.joint_map.items() if name in ARM_JOINTS
        }
        self.server = viser.ViserServer(port=viser_port)
        self.server.scene.add_grid("/ground", width=1.0, height=1.0, cell_size=0.05)
        self.urdf_vis = ViserUrdf(self.server, urdf_model, root_node_name="/robot")
        self.urdf_joint_names = list(self.urdf_vis.get_actuated_joint_names())
        self.gripper_lo, self.gripper_hi = self.urdf_vis.get_actuated_joint_limits()["gripper"]
        self.target_marker = self.server.scene.add_icosphere("/target", radius=0.012, color=(255, 60, 60), visible=False)
        # Coordinate frames (x red, y green, z blue): the robot base and the end effector
        # (gripper_frame_link, the point all grasp targets refer to).
        scene = self.server.scene
        scene.add_frame("/base_frame", axes_length=0.10, axes_radius=0.004)
        scene.add_label("/base_frame/label", "base", position=(0.0, 0.0, 0.11))
        self.ee_frame = scene.add_frame("/ee_frame", axes_length=0.06, axes_radius=0.003)
        scene.add_label("/ee_frame/label", "eef", position=(0.0, 0.0, 0.07))

        self.status_md = self.server.gui.add_markdown("Connecting...")
        self.image = self.server.gui.add_image(np.zeros((8, 8, 3), np.uint8), label="Camera", format="jpeg")
        with self.server.gui.add_folder("Red detection", expand_by_default=False):
            self.sliders = {
                "h_low_max": self.server.gui.add_slider("Hue low max", 0, 40, 1, detection.h_low_max),
                "h_high_min": self.server.gui.add_slider("Hue high min", 140, 180, 1, detection.h_high_min),
                "s_min": self.server.gui.add_slider("Saturation min", 0, 255, 1, detection.s_min),
                "v_min": self.server.gui.add_slider("Value min", 0, 255, 1, detection.v_min),
                "min_area": self.server.gui.add_slider("Min area (px)", 10, 5000, 10, detection.min_area),
            }

        with self.server.gui.add_folder("Trajectories", expand_by_default=False):
            self.show_traj = self.server.gui.add_checkbox("Show trajectories", initial_value=True)
            self.server.gui.add_button("Clear trajectories").on_click(lambda _: self.clear_trajectories())
        self._traj_handles: list[list] = []  # one list of scene handles per drawn move
        self._traj_count = 0

        self.q = np.zeros(6)  # measured joints (deg, gripper 0..100)
        self.q_cmd = np.zeros(6)  # last commanded joints
        self.frame: np.ndarray | None = None
        self.detection: Detection | None = None
        self._last_image_t = 0.0

    # -- lifecycle ---------------------------------------------------------------------------

    def connect(self) -> None:
        self.robot.connect()
        self.tick()
        self.q_cmd = self.q.copy()
        print(f"viser running at http://localhost:{self.server.get_port()}")

    def disconnect(self) -> None:
        self.robot.disconnect()
        self.server.stop()

    def detection_settings(self) -> DetectionSettings:
        return DetectionSettings(**{k: int(s.value) for k, s in self.sliders.items()})

    def set_status(self, text: str) -> None:
        self.status_md.content = text
        print(text.replace("**", "").replace("  \n", " | "))

    def show_target(self, xyz: np.ndarray | None) -> None:
        self.target_marker.visible = xyz is not None
        if xyz is not None:
            self.target_marker.position = tuple(xyz)

    # -- observation + visualization ---------------------------------------------------------

    def tick(self) -> None:
        obs = self.robot.get_observation()
        self.q = np.array([obs[f"{n}.pos"] for n in ALL_JOINTS], dtype=float)
        frame = obs.get(CAMERA_NAME)
        mask = None
        if frame is not None:
            self.frame = np.ascontiguousarray(frame)
            if self.detect_red:
                self.detection, mask = detect_red_cube(self.frame, self.detection_settings(), self.cube_accept)
            else:
                self.detection, mask = None, None

        cfg = {n: np.deg2rad(self.q[i]) for i, n in enumerate(ARM_JOINTS)}
        cfg["gripper"] = self.gripper_lo + (self.q[5] / 100.0) * (self.gripper_hi - self.gripper_lo)  # approximate
        self.urdf_vis.update_cfg(np.array([cfg[n] for n in self.urdf_joint_names]))
        ee = self.tcp_pose(self.q)
        self.ee_frame.position, self.ee_frame.wxyz = ee[:3, 3], SO3.from_matrix(ee[:3, :3]).wxyz

        now = time.perf_counter()
        if self.frame is not None and now - self._last_image_t > 0.1:
            self._last_image_t = now
            view = self.frame.copy()
            if mask is not None:
                view[mask > 0] = (0.5 * view[mask > 0] + [0, 127, 0]).astype(np.uint8)
            if self.detection is not None:
                d = self.detection
                cv2.drawContours(view, [d.contour], -1, (0, 255, 0), 2)
                cv2.circle(view, (int(d.u), int(d.v)), 4, (255, 255, 0), -1)
            if self.overlay is not None:
                view = self.overlay(view)
            self.image.image = view

    def idle(self, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            t0 = time.perf_counter()
            self.tick()
            precise_sleep(max(self.dt - (time.perf_counter() - t0), 0.0))

    def wait_for(self, *events: threading.Event, on_tick=None) -> threading.Event:
        """Ticks until one of the events is set; clears and returns it."""
        for e in events:
            e.clear()
        while True:
            t0 = time.perf_counter()
            self.tick()
            if on_tick is not None:
                on_tick()
            for e in events:
                if e.is_set():
                    e.clear()
                    return e
            precise_sleep(max(self.dt - (time.perf_counter() - t0), 0.0))

    def average_detection(self, ticks: int = 10, min_frac: float = 0.6) -> Detection | None:
        """Median of several detections, or None if the cube was seen in fewer than min_frac of the frames."""
        found = []
        for _ in range(ticks):
            self.idle(self.dt)
            if self.detection is not None:
                found.append(self.detection)
        if len(found) < max(1, min_frac * ticks):
            return None
        best = found[-1]
        return Detection(
            float(np.median([d.u for d in found])),
            float(np.median([d.v for d in found])),
            best.angle_deg,
            float(np.median([d.area for d in found])),
            best.contour,
        )

    # -- kinematics --------------------------------------------------------------------------

    def tcp_pose(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics of the TCP (centre between the jaws)."""
        return self.kin.forward_kinematics(q) @ self.t_flange_tcp

    def solve_pose(self, target: np.ndarray, yaw: float, iters: int = 40) -> np.ndarray | None:
        """Joint solution (deg) for a TCP pose (4x4) pointing roughly down, or None if out of reach."""
        x, y = target[0, 3], target[1, 3]
        radial = math.atan2(y, x)
        # Seed: positive pan turns the arm toward -y in this URDF; roll ~ yaw offset from radial.
        q = np.array([-math.degrees(radial), 0.0, 20.0, 70.0, math.degrees(wrap(yaw - radial)), self.q_cmd[5]])
        flange_target = target @ self.t_tcp_flange
        for _ in range(iters):
            q = self.kin.inverse_kinematics(q, flange_target, orientation_weight=0.1)
        reached = self.tcp_pose(q)
        pos_err = np.linalg.norm(reached[:3, 3] - target[:3, 3])
        rot_err = np.rad2deg(np.linalg.norm(SO3.from_matrix(target[:3, :3].T @ reached[:3, :3]).log()))
        return q if pos_err < 0.005 and rot_err < 5.0 and self.within_limits(q) else None

    def within_limits(self, q: np.ndarray, margin_deg: float = 0.0) -> bool:
        """All arm joints at least margin_deg inside their URDF limits."""
        return all(lo + margin_deg <= q[i] <= hi - margin_deg for i, (lo, hi) in enumerate(self.joint_limits_deg[n] for n in ARM_JOINTS))

    # -- torque ------------------------------------------------------------------------------

    def release(self) -> None:
        """Torque off so the arm can be moved by hand."""
        self.robot.bus.disable_torque()

    def hold(self) -> None:
        """Torque back on at the current position (goal is set first, so the arm does not jump)."""
        self.tick()
        self.robot.bus.sync_write("Goal_Position", {n: float(v) for n, v in zip(ALL_JOINTS, self.q)})
        self.robot.bus.enable_torque()
        self.q_cmd = self.q.copy()

    # -- trajectory visualization ----------------------------------------------------------

    TRAJ_COLORS = {"joint": (170, 60, 220), "cartesian": (255, 140, 0)}
    MAX_TRAJECTORIES = 12

    def clear_trajectories(self) -> None:
        for handles in self._traj_handles:
            for h in handles:
                h.remove()
        self._traj_handles.clear()

    def draw_trajectory(self, poses: list[np.ndarray], kind: str) -> None:
        """Shows the TCP poses a move will command: its path as a line and ~12 small frames along
        it (position + orientation of the gripper)."""
        if not self.show_traj.value or len(poses) < 2:
            return
        poses = np.asarray(poses)
        pos = poses[:, :3, 3]
        scene, name = self.server.scene, f"/traj/{self._traj_count}"
        self._traj_count += 1
        segments = np.stack([pos[:-1], pos[1:]], axis=1)
        idx = np.unique(np.linspace(0, len(poses) - 1, min(12, len(poses))).astype(int))
        handles = [
            scene.add_line_segments(f"{name}/path", segments, self.TRAJ_COLORS[kind], thickness=3, thickness_units="screen"),
            scene.add_batched_axes(
                f"{name}/poses",
                np.array([SO3.from_matrix(r).wxyz for r in poses[idx, :3, :3]]),
                pos[idx],
                axes_length=0.012,
                axes_radius=0.001,
            ),
        ]
        self._traj_handles.append(handles)
        while len(self._traj_handles) > self.MAX_TRAJECTORIES:
            for h in self._traj_handles.pop(0):
                h.remove()

    # -- motion ------------------------------------------------------------------------------

    def send(self, q: np.ndarray) -> None:
        self.robot.send_action({f"{n}.pos": float(v) for n, v in zip(ALL_JOINTS, q)})
        self.q_cmd = np.array(q, dtype=float)

    def joint_move(self, q_goal: np.ndarray, speed_deg_s: float = 30.0) -> None:
        """Smooth joint-space move from the last command to q_goal (gripper included).

        Every joint goes from its start to its goal angle along the same ease-in/ease-out
        profile s(t) = 3t^2 - 2t^3, so all joints start and stop together; the duration is set by
        the joint that has to move furthest. The gripper's path in space is whatever these joint
        angles give (a curve, not a straight line)."""
        q_start = self.q_cmd.copy()
        span = np.max(np.abs(q_goal[:5] - q_start[:5]))
        steps = max(int(math.ceil(span / speed_deg_s / self.dt)), int(0.3 / self.dt))
        profile = [3 * (i / steps) ** 2 - 2 * (i / steps) ** 3 for i in range(steps + 1)]
        self.draw_trajectory([self.tcp_pose(q_start + (q_goal - q_start) * s) for s in profile], "joint")
        for i in range(1, steps + 1):
            t0 = time.perf_counter()
            self.send(q_start + (q_goal - q_start) * profile[i])
            self.tick()
            precise_sleep(max(self.dt - (time.perf_counter() - t0), 0.0))
        self._settle(q_goal)

    def cart_move(self, t_goal: np.ndarray, speed: float = 0.04) -> None:
        """Straight-line move of the TCP to t_goal, via IK.

        The TCP poses are planned first: position interpolated linearly from the current
        commanded pose to the goal, orientation along the shortest rotation between them
        (axis-angle, scaled by the same fraction), at constant speed (`speed` m/s, and at most
        MAX_ROT_SPEED_DEG_S for the rotation). Each control step then solves IK for the next pose,
        starting from the previous solution."""
        q_ik = self.q_cmd.copy()
        t_start = self.tcp_pose(q_ik)
        rot_delta = SO3.from_matrix(t_start[:3, :3].T @ t_goal[:3, :3]).log()
        dist = np.linalg.norm(t_goal[:3, 3] - t_start[:3, 3])
        steps = max(int(math.ceil(dist / speed / self.dt)), int(math.ceil(np.linalg.norm(rot_delta) / math.radians(MAX_ROT_SPEED_DEG_S) / self.dt)), 1)
        targets = []
        for i in range(steps + 1):
            s = i / steps
            t = np.eye(4)
            t[:3, 3] = t_start[:3, 3] + (t_goal[:3, 3] - t_start[:3, 3]) * s
            t[:3, :3] = t_start[:3, :3] @ SO3.exp(rot_delta * s).as_matrix()
            targets.append(t)
        self.draw_trajectory(targets, "cartesian")
        for i in range(1, steps + 1):
            t0 = time.perf_counter()
            t = targets[i]
            q_ik = self.kin.inverse_kinematics(q_ik, t @ self.t_tcp_flange, orientation_weight=0.1)
            self.send(q_ik)
            self.tick()
            precise_sleep(max(self.dt - (time.perf_counter() - t0), 0.0))
        for _ in range(10):
            q_ik = self.kin.inverse_kinematics(q_ik, t_goal @ self.t_tcp_flange, orientation_weight=0.1)
        self.send(q_ik)
        self._settle(q_ik)

    def set_gripper(self, value: float, seconds: float = 0.6) -> None:
        q_goal = self.q_cmd.copy()
        start = q_goal[5]
        # never faster than the gripper can actually move (else the safety limiter clamps every step)
        steps = max(int(seconds / self.dt), int(math.ceil(abs(value - start) / GRIPPER_SPEED / self.dt)), 1)
        for i in range(1, steps + 1):
            t0 = time.perf_counter()
            q_goal[5] = start + (value - start) * i / steps
            self.send(q_goal)
            self.tick()
            precise_sleep(max(self.dt - (time.perf_counter() - t0), 0.0))
        self.idle(0.4)

    def _settle(self, q_goal: np.ndarray, tol_deg: float = 2.0, timeout: float = 1.5) -> None:
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            self.idle(self.dt)
            if np.max(np.abs(self.q[:5] - q_goal[:5])) < tol_deg:
                return
