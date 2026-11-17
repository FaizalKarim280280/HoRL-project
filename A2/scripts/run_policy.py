#!/usr/bin/env python
"""Run a trained policy on the follower, with a live camera window (no rerun).

Same inference steps as lerobot-rollout (pre/post processors from the checkpoint, select_action
every control tick), in our own loop so the cameras can be shown. One window shows the wrist and
scene cameras; the keys work in that window:
    SPACE   start the policy / pause it (the arm holds where it is)
    r       pause and glide back to the start pose, ready for the next trial
The start pose is the median first pose of the demonstrations (--start-pose-from), so every run starts
where the training episodes started; the arm glides there at launch.
    q, ESC  quit (the follower goes limp: support it)
The policy is reset (action queue / ensemble cleared) every time it is started.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DISPLAY_SCALE = 0.75


def demo_start_pose(raw_root: Path) -> np.ndarray:
    """Median of the follower's first recorded state over all raw demonstrations under raw_root."""
    starts = [np.load(f)["state"][0] for f in sorted(raw_root.glob("*/episode_*/data.npz"))]
    if not starts:
        raise SystemExit(f"No demonstrations under {raw_root} to take the start pose from")
    return np.median(starts, axis=0).astype(np.float64)


def step_limits(arm: float, gripper: float) -> np.ndarray:
    """Per-joint limit for limit_step: `arm` for the 5 arm joints, `gripper` for the gripper (0 = no limit).

    The gripper has its own limit: slowing it down makes it close long after the policy expects, so the
    policy can stall waiting for a grasp that hasn't happened yet.
    """
    return np.array([arm] * 5 + [gripper], dtype=np.float64)


def limit_step(target: np.ndarray, last_cmd: np.ndarray, max_step) -> tuple[np.ndarray, bool]:
    """Caps each joint's change from the previous command (degrees per control step; 0 = no limit).

    max_step: one value for all joints, or one per joint (step_limits).

    Limits the change between consecutive commands, not the distance to the measured position: the
    follower lags its commands by up to ~20 deg in normal motion (and the gripper sits ~10 above its
    command while squeezing the cube), but the commands themselves changed by at most 6.2 deg per step
    in the 147 demonstrations. A cap above that only acts on sudden jumps, which become fast ramps.
    """
    cap = np.broadcast_to(np.asarray(max_step, dtype=np.float64), target.shape)
    cap = np.where(cap > 0, cap, np.inf)  # 0 = no limit for that joint
    step = np.clip(target - last_cmd, -cap, cap)
    return last_cmd + step, bool(np.any(step != target - last_cmd))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="pretrained_model folder or Hub repo id")
    p.add_argument("--follower-port", required=True)
    p.add_argument("--follower-id", required=True)
    p.add_argument("--wrist", required=True, help="Wrist camera OpenCV index")
    p.add_argument("--scene", required=True, help="Scene camera OpenCV index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--task", default="Pick up the red cube and place it in the blue tray.")
    p.add_argument("--device", default="mps")
    p.add_argument("--n-action-steps", type=int, default=0, help="Steps executed per predicted chunk (0 = half the chunk)")
    p.add_argument("--ensemble", type=float, default=None,
                   help="Temporal ensembling coefficient (e.g. 0.01): new chunk every step, predictions averaged")
    p.add_argument("--num-inference-steps", type=int, default=0,
                   help="Flow matching / diffusion: denoising steps per chunk (0 = the checkpoint's value)")
    p.add_argument("--interp", type=int, default=1, help="Interpolated commands sent per action (2 = 60 Hz to the motors)")
    p.add_argument("--start-pose-from", type=Path, default=None,
                   help="Raw data folder (e.g. A2/data/raw): start pose = median first pose of its demos "
                        "(default: the pose at launch)")
    p.add_argument("--max-step-gripper", type=float, default=0,
                   help="Largest gripper change between consecutive commands (0 = no limit)")
    p.add_argument("--max-step-deg", type=float, default=5,
                   help="Largest change of any joint between consecutive commands, degrees (0 = no limit)")
    p.add_argument("--duration", type=float, default=0, help="Pause automatically after this many seconds (0 = never)")
    return p.parse_args()


def load_policy(args: argparse.Namespace):
    register_third_party_plugins()  # policies installed as lerobot_policy_* packages (e.g. flow_matching)
    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.device = args.device
    # Predicted chunk length: ACT calls it chunk_size, Diffusion Policy / flow matching horizon.
    chunk = getattr(cfg, "chunk_size", None) or cfg.horizon
    if args.ensemble is not None:
        if not hasattr(cfg, "temporal_ensemble_coeff"):
            raise SystemExit(f"Temporal ensembling is only available for ACT, not {cfg.type}.")
        cfg.temporal_ensemble_coeff, cfg.n_action_steps = args.ensemble, 1
    else:
        cfg.n_action_steps = args.n_action_steps or max(1, chunk // 2)
    if args.num_inference_steps and hasattr(cfg, "num_inference_steps"):  # ACT has none
        cfg.num_inference_steps = args.num_inference_steps
    cfg.__post_init__()  # re-validate (e.g. n_action_steps fits in the chunk)
    policy = get_policy_class(cfg.type).from_pretrained(args.policy, config=cfg)
    policy.to(args.device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.policy,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    cams = [k.removeprefix("observation.images.") for k in cfg.input_features if k.startswith("observation.images.")]
    mode = (f"temporal ensembling (coeff {cfg.temporal_ensemble_coeff})" if args.ensemble is not None
            else f"{cfg.type}: {cfg.n_action_steps} of {chunk} steps per chunk"
            + (f", {cfg.num_inference_steps} inference steps" if getattr(cfg, "num_inference_steps", None) else ""))
    return policy, pre, post, cams, mode


def main() -> None:
    args = parse_args()
    policy, pre, post, policy_cams, mode = load_policy(args)
    cams = {"wrist": int(args.wrist), "scene": int(args.scene)}
    if sorted(policy_cams) != sorted(cams):
        raise SystemExit(f"Policy expects cameras {policy_cams}, this script provides {list(cams)}")
    device = torch.device(args.device)
    print(f"Policy {args.policy}: {mode}")

    follower = SO101Follower(SO101FollowerConfig(
        port=args.follower_port, id=args.follower_id,
        cameras={k: OpenCVCameraConfig(index_or_path=i, width=args.width, height=args.height, fps=args.fps) for k, i in cams.items()},
    ))
    window = "A2 policy - SPACE start/pause, r return to start, q quit"
    dt = 1.0 / args.fps

    def send(q: np.ndarray) -> None:
        follower.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})

    follower.connect()
    try:
        obs = follower.get_observation()
        q0 = np.array([obs[f"{j}.pos"] for j in JOINTS])
        home = demo_start_pose(args.start_pose_from) if args.start_pose_from else q0.copy()
        last_cmd = q0.copy()
        print("Start pose:", {j: round(float(v), 1) for j, v in zip(JOINTS, home)})
        # Warm-up inference (the first call on the GPU is slow), so the arm doesn't stall at /start.
        with torch.inference_mode():
            frame = {"observation.state": q0.astype(np.float32), **{f"observation.images.{c}": obs[c] for c in cams}}
            post(policy.select_action(pre(prepare_observation_for_inference(frame, device, args.task, "so101_follower"))))

        state, steps, loop_hz, infer_ms, clipped = "paused", 0, 0.0, 0.0, 0
        glide: list[np.ndarray] = []
        if args.start_pose_from:  # glide to the demos' start pose before the first run
            glide = [q0 + s * (home - q0) for s in np.linspace(0, 1, int(2 * args.fps))[1:]]
            state = "returning"
        print("Keys in the camera window: SPACE start/pause, r return to start pose, q quit.")
        while True:
            t0 = time.perf_counter()
            obs = follower.get_observation()
            q = np.array([obs[f"{j}.pos"] for j in JOINTS])

            if state == "running":
                ti = time.perf_counter()
                frame = {"observation.state": q.astype(np.float32), **{f"observation.images.{c}": obs[c] for c in cams}}
                with torch.inference_mode():
                    action = post(policy.select_action(pre(prepare_observation_for_inference(frame, device, args.task, "so101_follower"))))
                target = action.squeeze(0).cpu().numpy().astype(np.float64)
                infer_ms = 0.9 * infer_ms + 0.1 * (time.perf_counter() - ti) * 1000
                target, was_clipped = limit_step(target, last_cmd, step_limits(args.max_step_deg, args.max_step_gripper))
                if was_clipped:
                    clipped += 1
                # Spread the move over --interp commands within this tick.
                for k in range(1, args.interp + 1):
                    send(last_cmd + (target - last_cmd) * k / args.interp)
                    if k < args.interp:
                        precise_sleep(dt / args.interp)
                last_cmd, steps = target, steps + 1
                if args.duration and steps * dt >= args.duration:
                    state = "paused"
                    print(f"Paused after {args.duration:.0f} s.")
            elif state == "returning":
                last_cmd = glide.pop(0)
                send(last_cmd)
                if not glide:
                    state = "paused"
                    print("At the start pose. Set up the next trial, then SPACE.")

            # Window: wrist above scene, status bar below
            frames = []
            for c in cams:
                f = cv2.resize(cv2.cvtColor(obs[c], cv2.COLOR_RGB2BGR), None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
                cv2.putText(f, c, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                frames.append(f)
            bar = np.zeros((58, frames[0].shape[1], 3), np.uint8)
            if state == "running":
                line1, colour = f"RUNNING  {steps * dt:5.1f} s   inference {infer_ms:4.1f} ms" + (f"   limited {clipped}" if clipped else ""), (0, 0, 255)
                line2 = "SPACE pause   r return to start"
                cv2.circle(frames[0], (frames[0].shape[1] - 18, 18), 9, (0, 0, 255), -1)
            elif state == "returning":
                line1, colour, line2 = "returning to start pose...", (0, 200, 255), ""
            else:
                line1, colour = f"paused  ({mode})", (0, 200, 0)
                line2 = "SPACE start   r return to start   q quit"
            cv2.putText(bar, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
            cv2.putText(bar, f"{line2}   {loop_hz:4.1f} Hz", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.imshow(window, np.vstack([*frames, bar]))
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and state == "paused":
                policy.reset()  # fresh action queue / ensemble for this trial
                state, steps, last_cmd, clipped = "running", 0, q, 0
                print("Policy running.")
            elif key == ord(" ") and state == "running":
                state = "paused"
                send(q)  # hold where the arm is now
                last_cmd = q
                print(f"Paused after {steps * dt:.1f} s." + (f" {clipped} commands limited to {args.max_step_deg:g} deg/step." if clipped else ""))
            elif key == ord("r") and state != "returning":
                glide = [q + s * (home - q) for s in np.linspace(0, 1, int(2 * args.fps))[1:]]
                state = "returning"
                print("Returning to the start pose...")
            elif key in (ord("q"), 27):
                break

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
            loop_hz = 0.9 * loop_hz + 0.1 / (time.perf_counter() - t0)
    finally:
        cv2.destroyAllWindows()
        follower.disconnect()  # torque off: support the follower arm


if __name__ == "__main__":
    main()
