#!/usr/bin/env python
"""Keyboard / viser end-effector teleop for the SO-101 follower.

The keyboard, or dragging the gizmo in the viser page, moves a target gripper position in
the robot base frame. Inverse kinematics
(placo, via lerobot's RobotKinematics) turns it into joint targets. The viser page shows the
URDF at the measured joint angles, the target and actual end-effector frames, and the camera.

Keys (held down = continuous motion):
    W / S   forward / back   (+x / -x)
    A / D   left / right     (+y / -y, seen from behind the robot)
    R / F   up / down        (+z / -z)
    E / Q   open / close gripper

Or drag the gizmo in viser: the target follows it at the "EE speed" / "Rotation speed"
settings. The SO-101 has 5 arm joints, so only the pitch and roll parts of a gizmo rotation
are applied; yaw follows the base pan, and the gizmo snaps back to what was reached.

Motion is OFF at start: the target marker moves in viser but nothing is sent to the arm.
Tick "Send commands to arm" in the viser panel once the directions look right.
Stop with Ctrl+C.
"""

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from pynput import keyboard
from viser.extras import ViserUrdf
from viser.transforms import SO3

from cube_pick_common import ALL_JOINTS, ARM_JOINTS, DEFAULT_URDF, resolve_camera
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.keyboard_input import pynput_listener_is_trusted
from lerobot.utils.robot_utils import precise_sleep

CAMERA_NAME = "front"

# key -> (axis index, direction); axis 0..2 = x/y/z, axis 3 = gripper
KEY_MAP = {
    "w": (0, +1), "s": (0, -1),
    "a": (1, +1), "d": (1, -1),
    "r": (2, +1), "f": (2, -1),
    "e": (3, +1), "q": (3, -1),
}  # fmt: skip


class HeldKeys:
    """Tracks which letter keys are currently held, using a global pynput listener."""

    def __init__(self):
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)

    def start(self) -> None:
        self._listener.start()
        if not pynput_listener_is_trusted(self._listener):
            raise RuntimeError(
                "pynput cannot read the keyboard. Grant your terminal app Accessibility and "
                "Input Monitoring in System Settings > Privacy & Security, then restart it."
            )

    def stop(self) -> None:
        self._listener.stop()

    @staticmethod
    def _char(key) -> str | None:
        char = getattr(key, "char", None)
        return char.lower() if char else None

    def _on_press(self, key) -> None:
        if (c := self._char(key)) in KEY_MAP:
            with self._lock:
                self._held.add(c)

    def _on_release(self, key) -> None:
        if (c := self._char(key)) in KEY_MAP:
            with self._lock:
                self._held.discard(c)

    def command(self) -> np.ndarray:
        """Returns [dx, dy, dz, dgripper], each in {-1, 0, 1}."""
        cmd = np.zeros(4)
        with self._lock:
            for c in self._held:
                axis, sign = KEY_MAP[c]
                cmd[axis] += sign
        return np.clip(cmd, -1, 1)


def wxyz_from_matrix(rot: np.ndarray) -> np.ndarray:
    return SO3.from_matrix(rot).wxyz


def wrist_projector(kinematics: RobotKinematics, q: np.ndarray) -> np.ndarray:
    """3x3 projector onto the gripper-frame rotations the wrist can make at q (flex and roll).

    With the gripper position held, a 5-joint arm has exactly these two rotational freedoms;
    any other rotation (yaw) would need the base to pan and so would move the gripper.
    """
    rot0 = kinematics.forward_kinematics(q)[:3, :3]
    axes = []
    for joint in (ARM_JOINTS.index("wrist_flex"), ARM_JOINTS.index("wrist_roll")):
        dq = q.copy()
        dq[joint] += 1.0
        rot1 = kinematics.forward_kinematics(dq)[:3, :3]
        axes.append(SO3.from_matrix(rot0.T @ rot1).log() / np.deg2rad(1.0))
    a = np.stack(axes, axis=1)
    return a @ np.linalg.pinv(a)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="Follower serial port")
    p.add_argument("--id", required=True, help="Follower id (calibration key)")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--camera-index", default="0", help="OpenCV index or path; 'none' to disable")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--fps", type=int, default=30, help="Control loop rate")
    p.add_argument("--speed", type=float, default=0.05, help="End-effector speed in m/s")
    p.add_argument("--rot-speed", type=float, default=60.0, help="Gizmo rotation speed in deg/s")
    p.add_argument("--gripper-speed", type=float, default=60.0, help="Gripper speed in %%/s")
    p.add_argument("--max-step-deg", type=float, default=5.0, help="Max joint change per loop (safety)")
    p.add_argument("--workspace", type=float, default=0.15, help="Half-size (m) of the box around the start pose")
    p.add_argument("--min-z", type=float, default=0.02, help="Lowest target height (m) above the base origin")
    p.add_argument("--viser-port", type=int, default=8080)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.urdf.exists():
        raise FileNotFoundError(f"URDF not found: {args.urdf}")

    cameras = {}
    if args.camera_index.lower() != "none":
        index = resolve_camera(args.camera_index)
        cameras[CAMERA_NAME] = OpenCVCameraConfig(
            index_or_path=index, width=args.camera_width, height=args.camera_height, fps=args.camera_fps
        )

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.id,
            use_degrees=True,
            max_relative_target=args.max_step_deg,
            cameras=cameras,
        )
    )
    kinematics = RobotKinematics(str(args.urdf), "gripper_frame_link", joint_names=ARM_JOINTS)

    # --- viser scene -------------------------------------------------------------------
    server = viser.ViserServer(port=args.viser_port)
    server.scene.add_grid("/ground", width=1.0, height=1.0, cell_size=0.05)
    urdf_vis = ViserUrdf(server, yourdfpy.URDF.load(str(args.urdf)), root_node_name="/robot")
    urdf_joint_names = list(urdf_vis.get_actuated_joint_names())
    gripper_lo, gripper_hi = urdf_vis.get_actuated_joint_limits()["gripper"]
    ee_frame = server.scene.add_frame("/ee_actual", axes_length=0.05, axes_radius=0.003)
    target_frame = server.scene.add_frame(
        "/ee_target", axes_length=0.03, axes_radius=0.002, origin_radius=0.012, origin_color=(255, 80, 80)
    )

    with server.gui.add_folder("Control"):
        send_box = server.gui.add_checkbox("Send commands to arm", initial_value=False)
        reset_btn = server.gui.add_button("Reset target to current pose")
        speed_slider = server.gui.add_slider("EE speed (m/s)", min=0.01, max=0.15, step=0.01, initial_value=args.speed)
        rot_slider = server.gui.add_slider("Rotation speed (deg/s)", min=10, max=180, step=10, initial_value=args.rot_speed)
    server.gui.add_markdown(
        "**Keys:** W/S fwd/back · A/D left/right · R/F up/down · E/Q gripper open/close  \n"
        "Or drag the gizmo: the target follows at EE / rotation speed. Only pitch and roll rotations "
        "are possible (5-joint arm); out-of-reach drags snap back.  \n"
        "Red ball = target, axes = actual gripper. Ctrl+C in the terminal to quit."
    )
    status_md = server.gui.add_markdown("Connecting...")
    image_handle = (
        server.gui.add_image(np.zeros((8, 8, 3), dtype=np.uint8), label="Camera", format="jpeg")
        if cameras
        else None
    )

    reset_requested = threading.Event()
    reset_btn.on_click(lambda _: reset_requested.set())

    # Gizmo drags arrive on viser's threads; the control loop reads the latest goal.
    drag_lock = threading.Lock()
    drag = {"goal": None, "active": False}

    keys = HeldKeys()
    robot.connect()
    keys.start()
    print(f"viser running at http://localhost:{args.viser_port}")

    def read_joints(obs: dict) -> np.ndarray:
        return np.array([obs[f"{name}.pos"] for name in ALL_JOINTS], dtype=float)

    obs = robot.get_observation()
    q = read_joints(obs)
    target = kinematics.forward_kinematics(q)  # 4x4 target gripper pose
    start_pos = target[:3, 3].copy()
    q_ik = q.copy()  # last IK solution, reused as the next initial guess
    gripper_target = q[5]

    gizmo = server.scene.add_transform_controls(
        "/ee_gizmo", scale=0.1, depth_test=False, position=target[:3, 3], wxyz=wxyz_from_matrix(target[:3, :3])
    )

    @gizmo.on_update
    def _(event) -> None:
        with drag_lock:
            goal_pose = np.eye(4)
            goal_pose[:3, :3] = SO3(np.asarray(event.target.wxyz)).as_matrix()
            goal_pose[:3, 3] = event.target.position
            drag["goal"] = goal_pose
            drag["active"] = event.phase != "end"

    was_sending = False
    dt = 1.0 / args.fps
    last_ui = 0.0
    loop_hz = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            q = read_joints(obs)
            ee = kinematics.forward_kinematics(q)
            sending = send_box.value

            # Re-sync the target on request, and whenever sending is switched on, so the
            # arm never jumps to a target that drifted while it was not following.
            if reset_requested.is_set() or (sending and not was_sending):
                target = ee.copy()
                q_ik = q.copy()
                gripper_target = q[5]
                reset_requested.clear()
                with drag_lock:
                    drag["goal"] = None
            was_sending = sending

            cmd = keys.command()
            with drag_lock:
                if cmd[:3].any():  # the keyboard overrides a pending gizmo goal
                    drag["goal"] = None
                goal, dragging = drag["goal"], drag["active"]

            step = speed_slider.value * dt
            rot_step = np.deg2rad(rot_slider.value) * dt
            turn = np.zeros(3)  # rotation in the gripper frame, as a rotation vector
            if goal is not None:
                to_goal = goal[:3, 3] - target[:3, 3]
                dist = np.linalg.norm(to_goal)
                move = to_goal if dist <= step else to_goal * (step / dist)
                # Keep only the part of the requested rotation the wrist can make.
                turn = wrist_projector(kinematics, q_ik) @ SO3.from_matrix(target[:3, :3].T @ goal[:3, :3]).log()
                if (angle := np.linalg.norm(turn)) > rot_step:
                    turn *= rot_step / angle
            else:
                move = cmd[:3] * step
            # While sending, stop advancing the target if the arm has fallen 5 cm behind it
            # (e.g. it is blocked), so it does not lunge when it frees up.
            lagging = sending and np.linalg.norm(target[:3, 3] - ee[:3, 3]) > 0.05
            if lagging:
                move, turn = np.zeros(3), np.zeros(3)
            prev_target = target.copy()
            proposed = target.copy()
            proposed[:3, 3] += move
            proposed[:3, :3] = target[:3, :3] @ SO3.exp(turn).as_matrix()
            proposed[:3, 3] = np.clip(proposed[:3, 3], start_pos - args.workspace, start_pos + args.workspace)
            proposed[2, 3] = max(proposed[2, 3], args.min_z)
            gripper_target = float(np.clip(gripper_target + cmd[3] * args.gripper_speed * dt, 0.0, 100.0))

            # Accept the move only if IK can actually reach it; near the edge of the workspace
            # the solution otherwise jumps around and the arm oscillates.
            q_try = kinematics.inverse_kinematics(q_ik, proposed)  # orientation weight 0.01 (soft)
            reach_err = np.linalg.norm(kinematics.forward_kinematics(q_try)[:3, 3] - proposed[:3, 3])
            reachable = reach_err < 0.01
            if reachable:
                # Let the orientation follow the arm in the direction the wrist cannot control
                # (yaw from the base pan); otherwise sideways moves fight a fixed yaw and stall.
                reached = kinematics.forward_kinematics(q_try)[:3, :3]
                err = SO3.from_matrix(proposed[:3, :3].T @ reached).log()
                err -= wrist_projector(kinematics, q_try) @ err
                proposed[:3, :3] = proposed[:3, :3] @ SO3.exp(err).as_matrix()
                target, q_ik = proposed, q_try

            # Once released, drop the gizmo goal when the target reaches it or can get no
            # closer (outside the workspace box or out of reach); the gizmo snaps back.
            if goal is not None and not dragging and not lagging:
                moved = np.linalg.norm(target[:3, 3] - prev_target[:3, 3])
                turned = np.linalg.norm(SO3.from_matrix(prev_target[:3, :3].T @ target[:3, :3]).log())
                if moved < 1e-5 and turned < 1e-3:
                    with drag_lock:
                        if drag["goal"] is goal:
                            drag["goal"] = None
                            goal = None
            if goal is None and not dragging:
                if not np.allclose(gizmo.position, target[:3, 3]):
                    gizmo.position = target[:3, 3]
                if not np.allclose(SO3(np.asarray(gizmo.wxyz)).as_matrix(), target[:3, :3], atol=1e-4):
                    gizmo.wxyz = wxyz_from_matrix(target[:3, :3])

            if sending:
                q_goal = q_ik.copy()
                q_goal[5] = gripper_target
                robot.send_action({f"{name}.pos": float(v) for name, v in zip(ALL_JOINTS, q_goal)})

            # --- visualization ---------------------------------------------------------
            cfg = {name: np.deg2rad(q[i]) for i, name in enumerate(ARM_JOINTS)}
            cfg["gripper"] = gripper_lo + (q[5] / 100.0) * (gripper_hi - gripper_lo)  # approximate
            urdf_vis.update_cfg(np.array([cfg[name] for name in urdf_joint_names]))
            ee_frame.position, ee_frame.wxyz = ee[:3, 3], wxyz_from_matrix(ee[:3, :3])
            target_frame.position, target_frame.wxyz = target[:3, 3], wxyz_from_matrix(target[:3, :3])

            if t0 - last_ui > 0.1:  # GUI text + camera at ~10 Hz
                last_ui = t0
                if image_handle is not None and CAMERA_NAME in obs:
                    image_handle.image = obs[CAMERA_NAME]
                joints_txt = " · ".join(f"{n}: {v:.1f}" for n, v in zip(ALL_JOINTS, q))
                status_md.content = (
                    f"**{'SENDING' if sending else 'not sending (preview only)'}** · loop {loop_hz:.0f} Hz  \n"
                    f"EE (m): {np.round(ee[:3, 3], 3)} · target: {np.round(target[:3, 3], 3)}  \n"
                    f"gripper target: {gripper_target:.0f}%"
                    f"{'' if reachable else ' · **at reach limit**'}  \n{joints_txt}"
                )

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
            loop_hz = 0.9 * loop_hz + 0.1 / (time.perf_counter() - t0)
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        keys.stop()
        robot.disconnect()
        server.stop()


if __name__ == "__main__":
    main()
