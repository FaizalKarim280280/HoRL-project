#!/usr/bin/env python
"""Run the pre-registered evaluation: every trial of schedule.csv, blind, with videos and a log.

Both policies are loaded once; each trial sets the variant's executed actions per chunk and resets the
policy. The window shows the wrist and scene cameras but never which variant is running (blind labelling).
Keys (in the window):
  setup     the scene view shows the config's TARGET outlines (dashed: cube yellow, tray green) and the
            detected cube and tray (solid), with their errors; place them until the bar turns green.
            SPACE  start the trial (only when both are within tolerance; the measured poses are logged)
            g      start anyway (detection failing, e.g. glare): logged as a setup override
            o      overlay of the config's reference photo (saved at its first trial) on/off
            h      glide back to the home pose        q / ESC  quit (resume later from the next trial)
  running   s      success: cube in the tray, released, arm lifted -> completion time
            f      stop as a FAILURE (policy behaviour: stuck, cube knocked away, unsafe motion)
            x      abort as INVALID (not the policy's fault: hardware, someone touched the scene...);
                   the reason is typed in the terminal and the same trial is run again
            (60 s: timeout = failure)
  label     0-3    stages completed before failing: 0 none, 1 reached the cube, 2 grasped + lifted,
                   3 carried it above the tray. Then optional notes in the terminal.
Inputs from plan/ (eval_plan.yaml, configs, schedule, home pose). Outputs in --out (A2/results/task1):
logs/eval_log.csv (logs/pilot_log.csv with --pilot), rollouts/<rollout_id>/{wrist,scene}.mp4 + data.npz,
references/<config_id>.jpg. Rerunning continues with the first trial that
has no valid result.
"""

import argparse
import csv
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "task1"
sys.path.insert(0, str(HERE.parent / "scripts"))
from run_policy import limit_step, load_policy, step_limits  # noqa: E402  (same loading and inference path as 07_run_policy.sh)

from make_schedule import target_poses  # noqa: E402
from scene_detect import detect, draw, yaw_error  # noqa: E402

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMS = ["wrist", "scene"]
DISPLAY_SCALE = 0.75
STAGES = ["none", "reached cube", "grasped + lifted", "carried to tray", "placed + released (success)"]
FAIL_AT = ["reaching the cube", "grasping", "carrying", "placing / releasing"]
LOG_FIELDS = ["rollout_id", "trial", "block", "config_id", "variant", "policy", "checkpoint", "n_action_steps",
              "status", "success", "stages_completed", "progress", "end_reason", "failure_stage",
              "time_to_success_s", "duration_s", "start_time", "mean_hz", "slow_frames", "chunks", "limited_cmds",
              "cube_err_px", "cube_err_deg", "tray_err_px", "tray_err_deg", "setup_override", "camera_shift_px",
              "cube_meas", "tray_meas", "operator", "notes"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def append_log(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def chunk_limit(cfg) -> int:
    """Largest number of executable actions per chunk for this policy type."""
    if hasattr(cfg, "chunk_size"):
        return cfg.chunk_size
    return cfg.horizon - cfg.n_obs_steps + 1


class VideoWriter:
    """Writes the rollout's camera frames to mp4 in a background thread (the control loop never waits)."""

    def __init__(self, folder: Path, fps: int, size: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writers = {c: cv2.VideoWriter(str(folder / f"{c}.mp4"), fourcc, fps, size) for c in CAMS}
        self.queue: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while (item := self.queue.get()) is not None:
            for c, rgb in item.items():
                self.writers[c].write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def put(self, frames: dict) -> None:
        self.queue.put(frames)

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        for w in self.writers.values():
            w.release()


def setup_errors(scene_bgr: np.ndarray, targets: tuple) -> dict:
    """Detected cube and tray vs the config's targets: centre error (px) and rotation error (deg)."""
    det = detect(scene_bgr)
    out = {"det": det}
    for name, tgt in zip(("cube", "tray"), targets):
        d = det[name]
        out[name] = (None, None) if d is None else (float(np.hypot(d[0] - tgt[0], d[1] - tgt[1])), yaw_error(d[4], tgt[4]))
    return out


def camera_shift(scene_bgr: np.ndarray, reference_bgr: np.ndarray) -> float:
    """Image shift (px) of the scene's fixed structure (table edges, robot base) vs the demos' median frame."""
    def edges(img):
        return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 120).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(edges(reference_bgr), edges(scene_bgr),
                                     cv2.createHanningWindow(reference_bgr.shape[1::-1], cv2.CV_32F))
    return float(np.hypot(dx, dy))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=HERE / "plan" / "eval_plan.yaml")
    p.add_argument("--out", type=Path, default=RESULTS, help="Results folder: logs/, rollouts/, references/")
    p.add_argument("--pilot", action="store_true", help="Run the pilot schedule (practice, logged to pilot_log.csv)")
    p.add_argument("--follower-port", required=True)
    p.add_argument("--follower-id", required=True)
    p.add_argument("--wrist", required=True, help="Wrist camera OpenCV index")
    p.add_argument("--scene", required=True, help="Scene camera OpenCV index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--device", default="mps")
    p.add_argument("--operator", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pdir, out = args.plan.parent, args.out  # plan inputs, results outputs
    plan = yaml.safe_load(args.plan.read_text())
    fps, timeout = plan["fps"], plan["timeout_s"]
    tol_px, tol_deg = plan["configurations"]["tolerance_px"], plan["configurations"]["tolerance_deg"]
    scene_ref = cv2.imread(str(pdir / "scene_reference.jpg"))
    # per-step command limits (run_policy.limit_step), part of the plan: arm joints, gripper (0 = none)
    max_step = step_limits(plan.get("max_step_deg", 0), plan.get("max_step_gripper", 0))
    dt = 1.0 / fps
    schedule = read_csv(pdir / ("pilot_schedule.csv" if args.pilot else "schedule.csv"))
    if not schedule:
        raise SystemExit("No schedule: run make_schedule.py first.")
    configs = {c["config_id"]: c for c in read_csv(pdir / "configs.csv")}
    home = np.array(json.loads((pdir / "home_pose.json").read_text())["pose"])
    (out / "logs").mkdir(parents=True, exist_ok=True)
    log_file = out / "logs" / ("pilot_log.csv" if args.pilot else "eval_log.csv")
    rollouts, refs = out / "rollouts", out / "references"
    rollouts.mkdir(exist_ok=True)
    refs.mkdir(exist_ok=True)

    # Policies: loaded once each, with the largest executed-steps value any variant needs.
    policies = {}
    for name, spec in plan["policies"].items():
        path = (pdir / spec["checkpoint"]).resolve()
        if not path.is_dir():
            raise SystemExit(f"Checkpoint of policy '{name}' not found: {path}")
        need = max(v["n_action_steps"] for v in plan["variants"].values() if v["policy"] == name)
        pol, pre, post, cams, _ = load_policy(argparse.Namespace(
            policy=str(path), device=args.device, ensemble=None, n_action_steps=need, num_inference_steps=0))
        if sorted(cams) != sorted(CAMS):
            raise SystemExit(f"Policy '{name}' expects cameras {cams}, not {CAMS}")
        if need > chunk_limit(pol.config):
            raise SystemExit(f"Policy '{name}': {need} executed steps but the chunk allows {chunk_limit(pol.config)}")
        policies[name] = dict(policy=pol, pre=pre, post=post, path=path)
        print(f"Loaded policy '{name}' ({pol.config.type}) from {path}")
    device = torch.device(args.device)

    def infer(p: dict, obs: dict, q: np.ndarray) -> np.ndarray:
        frame = {"observation.state": q.astype(np.float32), **{f"observation.images.{c}": obs[c] for c in CAMS}}
        with torch.inference_mode():
            a = p["post"](p["policy"].select_action(p["pre"](prepare_observation_for_inference(
                frame, device, plan["task"], "so101_follower"))))
        return a.squeeze(0).cpu().numpy().astype(np.float64)

    def remaining() -> list[dict]:
        done = {r["trial"] for r in read_csv(log_file) if r["status"] in ("valid", "pilot")}
        return [t for t in schedule if t["trial"] not in done]

    def next_rollout_id() -> str:
        ids = [int(r["rollout_id"].lstrip("pr")) for r in read_csv(out / "logs" / "eval_log.csv") + read_csv(out / "logs" / "pilot_log.csv")]
        return f"{'p' if args.pilot else 'r'}{max(ids, default=0) + 1:04d}"

    follower = SO101Follower(SO101FollowerConfig(
        port=args.follower_port, id=args.follower_id,
        cameras={k: OpenCVCameraConfig(index_or_path=int(i), width=args.width, height=args.height, fps=fps)
                 for k, i in (("wrist", args.wrist), ("scene", args.scene))},
    ))
    window = "A2 evaluation"

    def send(q: np.ndarray) -> None:
        follower.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})

    def glide_to(q0: np.ndarray, q1: np.ndarray, seconds: float = 2.0) -> list[np.ndarray]:
        return [q0 + s * (q1 - q0) for s in np.linspace(0, 1, max(2, int(seconds * fps)))[1:]]

    follower.connect()
    try:
        obs = follower.get_observation()
        q = np.array([obs[f"{j}.pos"] for j in JOINTS])
        for p in policies.values():  # warm-up: the first inference on the GPU is slow
            p["policy"].reset()
            infer(p, obs, q)
        glide = glide_to(q, home)
        state, overlay, loop_hz, ref_img = "setup", False, 0.0, None
        trial = run = None
        todo = remaining()
        print(f"{len(schedule) - len(todo)} of {len(schedule)} trials done; logging to {log_file.name}")

        while True:
            t0 = time.perf_counter()
            obs = follower.get_observation()
            q = np.array([obs[f"{j}.pos"] for j in JOINTS])

            if state == "setup" and trial is None:
                if not todo:
                    print("All scheduled trials are done.")
                    break
                trial = todo[0]
                cfg = configs[trial["config_id"]]
                ref_path = refs / f"{trial['config_id']}.jpg"
                targets = target_poses(cfg)
                print(f"\nTrial {trial['trial']}/{len(schedule)} (block {trial['block']}): {trial['config_id']}  "
                      "- place the cube and tray on the dashed outlines")
                ref_img = cv2.imread(str(ref_path)) if ref_path.exists() else None
            if glide:
                send(glide.pop(0))

            key_now = None
            if state == "running":
                p = policies[run["variant"]["policy"]]
                target = infer(p, obs, q)
                if len(p["policy"]._action_queue if hasattr(p["policy"], "_action_queue") else p["policy"]._queues["action"]) \
                        == run["variant"]["n_action_steps"] - 1:
                    run["chunks"] += 1  # a new chunk was just predicted
                cmd, was_limited = limit_step(target, run["last_cmd"], max_step)
                run["limited"] += was_limited
                send(cmd)
                run["last_cmd"] = cmd
                elapsed = t0 - run["t_start"]
                run["rows"].append((q.tolist(), target.tolist(), cmd.tolist(), elapsed))
                run["video"].put({c: obs[c] for c in CAMS})
                if elapsed >= timeout:
                    key_now = "timeout"

            setup_ok = False
            if state == "setup" and trial:
                err = setup_errors(cv2.cvtColor(obs["scene"], cv2.COLOR_RGB2BGR), targets)
                setup_ok = all(e[0] is not None and e[0] <= tol_px and e[1] <= tol_deg for e in (err["cube"], err["tray"]))

            # Window: wrist above scene (scene with targets / reference overlay), status bar below. No variant shown.
            frames = []
            for c in CAMS:
                f = cv2.cvtColor(obs[c], cv2.COLOR_RGB2BGR)
                if c == "scene" and overlay and state == "setup" and ref_img is not None:
                    f = cv2.addWeighted(f, 0.5, ref_img, 0.5, 0)
                if c == "scene" and state == "setup" and trial:
                    for name, tgt, col in (("tray", targets[1], (0, 255, 0)), ("cube", targets[0], (0, 255, 255))):
                        draw(f, tgt, col, 2, dashed=True)
                        if err["det"][name]:
                            draw(f, err["det"][name], (255, 255, 255) if not setup_ok else (0, 255, 0), 1)
                f = cv2.resize(f, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
                cv2.putText(f, c, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                frames.append(f)
            bar = np.zeros((84, frames[0].shape[1], 3), np.uint8)
            cfg_txt = f"trial {trial['trial']}/{len(schedule)}  block {trial['block']}  {trial['config_id']}" if trial else ""
            if state == "running":
                lines = [(f"RUNNING  {elapsed:4.1f} / {timeout} s", (0, 0, 255)),
                         ("s success   f fail   x invalid", (220, 220, 220))]
                cv2.circle(frames[0], (frames[0].shape[1] - 18, 18), 9, (0, 0, 255), -1)
            elif state == "label":
                lines = [("How many stages completed?", (0, 200, 255)),
                         ("0 none  1 reached  2 grasped  3 carried to tray", (220, 220, 220))]
            else:
                fmt = lambda e: "not found" if e[0] is None else f"{e[0]:.0f}px {e[1]:.0f}deg"  # noqa: E731
                lines = [(f"cube {fmt(err['cube'])}   tray {fmt(err['tray'])}" + ("   READY" if setup_ok else "")
                          + ("  (moving home)" if glide else ""), (0, 220, 0) if setup_ok else (0, 140, 255)),
                         ("SPACE start   g start anyway   o photo   h home   q quit", (220, 220, 220))]
            for i, (txt, col) in enumerate([(cfg_txt, (255, 255, 255))] + lines):
                cv2.putText(bar, txt, (8, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            cv2.putText(bar, f"{loop_hz:4.1f} Hz", (bar.shape[1] - 70, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            cv2.imshow(window, np.vstack([*frames, bar]))
            key = cv2.waitKey(1) & 0xFF

            if state == "setup":
                if key == ord(" ") and not glide and not setup_ok:
                    print(f"Not started: cube / tray not within {tol_px} px and {tol_deg} deg of the targets (g = start anyway)")
                elif key in (ord(" "), ord("g")) and not glide:
                    if not ref_path.exists():
                        cv2.imwrite(str(ref_path), cv2.cvtColor(obs["scene"], cv2.COLOR_RGB2BGR))
                    variant = plan["variants"][trial["variant"]]
                    pol = policies[variant["policy"]]["policy"]
                    pol.config.n_action_steps = variant["n_action_steps"]
                    pol.reset()  # fresh action queue with this variant's length
                    rid = next_rollout_id()
                    folder = rollouts / rid
                    folder.mkdir()
                    scene_bgr = cv2.cvtColor(obs["scene"], cv2.COLOR_RGB2BGR)
                    meas = setup_errors(scene_bgr, targets)
                    run_setup = {"cube_err_px": meas["cube"][0], "cube_err_deg": meas["cube"][1],
                                 "tray_err_px": meas["tray"][0], "tray_err_deg": meas["tray"][1],
                                 "setup_override": int(key == ord("g") and not setup_ok),
                                 "camera_shift_px": camera_shift(scene_bgr, scene_ref),
                                 **{f"{k}_meas": "" if meas["det"][k] is None else " ".join(f"{v:.1f}" for v in meas["det"][k])
                                    for k in ("cube", "tray")}}
                    if run_setup["camera_shift_px"] > 8:  # measured with the arm at home, as in the demos' first frames
                        print(f"WARNING: scene camera shifted {run_setup['camera_shift_px']:.0f} px from the demos - check it")
                    run_setup = {k: (f"{v:.1f}" if isinstance(v, float) else ("" if v is None else v)) for k, v in run_setup.items()}
                    run = dict(id=rid, folder=folder, variant=variant, rows=[], chunks=0, slow=0, limited=0, last_cmd=q,
                               setup=run_setup,
                               t_start=time.perf_counter(), start_time=datetime.now().isoformat(timespec="seconds"),
                               video=VideoWriter(folder, fps, (args.width, args.height)))
                    state = "running"
                    print(f"Rollout {rid} running...")
                elif key == ord("o"):
                    overlay = not overlay
                elif key == ord("h") and not glide:
                    glide = glide_to(q, home)
                elif key in (ord("q"), 27):
                    break

            elif state == "running":
                end = key_now or {ord("s"): "success", ord("f"): "stopped", ord("x"): "aborted"}.get(key)
                if end:
                    send(q)  # hold where the arm is
                    run["end"], run["duration"] = end, time.perf_counter() - run["t_start"]
                    run["video"].close()
                    state = "label"
                    if end == "success":
                        run["stages"] = 4
                    elif end == "aborted":
                        run["stages"] = ""
                    else:
                        run["stages"] = None  # ask in the window
                    print(f"Ended: {end} after {run['duration']:.1f} s")

            if state == "label" and run.get("stages") is None and key in tuple(ord(str(i)) for i in range(4)):
                run["stages"] = int(chr(key))
            if state == "label" and run.get("stages") is not None:
                if run["end"] == "aborted":
                    reason = ""
                    while not reason:
                        reason = input("INVALID rollout - reason (required): ").strip()
                    notes = reason
                else:
                    notes = input("Notes (Enter to skip): ").strip()
                rows = run["rows"]
                st, ac, cm, ts = (np.array(x, dtype=np.float32) for x in zip(*rows)) if rows else (np.zeros((0, 6)),) * 4
                # action = the policy's output, command = what was sent after the per-step limit
                np.savez(run["folder"] / "data.npz", state=st, action=ac, command=cm, timestamp=ts)
                success = run["end"] == "success"
                stages = run["stages"]
                append_log(log_file, {
                    "rollout_id": run["id"], "trial": trial["trial"], "block": trial["block"],
                    "config_id": trial["config_id"], "variant": trial["variant"],
                    "policy": run["variant"]["policy"], "checkpoint": str(policies[run["variant"]["policy"]]["path"]),
                    "n_action_steps": run["variant"]["n_action_steps"],
                    "status": "invalid" if run["end"] == "aborted" else ("pilot" if args.pilot else "valid"),
                    "success": "" if run["end"] == "aborted" else int(success),
                    "stages_completed": stages, "progress": "" if stages == "" else stages / 4,
                    "end_reason": run["end"],
                    "failure_stage": "" if success or stages == "" else FAIL_AT[stages],
                    "time_to_success_s": f"{run['duration']:.2f}" if success else "",
                    "duration_s": f"{run['duration']:.2f}", "start_time": run["start_time"],
                    "mean_hz": f"{(len(ts) - 1) / (ts[-1] - ts[0]):.1f}" if len(ts) > 1 else "",
                    "slow_frames": int((np.diff(ts) > 1.5 * dt).sum()) if len(ts) > 1 else 0, "chunks": run["chunks"],
                    "limited_cmds": run["limited"],
                    **run["setup"],
                    "operator": args.operator, "notes": notes,
                })
                print(f"Logged {run['id']}: {STAGES[stages] if stages != '' else 'INVALID (trial will be repeated)'}")
                todo, trial, run, state = remaining(), None, None, "setup"
                glide = glide_to(q, home)

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
            loop_hz = 0.9 * loop_hz + 0.1 / (time.perf_counter() - t0)
    finally:
        if run and "video" in run:
            run["video"].close()
        cv2.destroyAllWindows()
        follower.disconnect()  # torque off: support the follower arm


if __name__ == "__main__":
    main()
