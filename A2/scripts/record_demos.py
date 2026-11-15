#!/usr/bin/env python
"""Fast demonstration recorder: leader-arm teleop, live camera window, raw images + joint data.

The follower copies the leader at --fps. One window shows the wrist and scene cameras; the keys
work in that window:
    SPACE   start recording a demonstration / stop and save it
    d       discard the demonstration being recorded
    [ ]     previous / next initial-configuration ID (tag the cube/tray setup you are demonstrating)
    q, ESC  quit (finishes writing everything first)

Each demonstration is saved as raw data, with no video encoding, so saving is instant:
    <root>/episode_007/wrist/000000.jpg ...   one JPEG per frame and camera (written in background threads)
    <root>/episode_007/scene/000000.jpg ...
    <root>/episode_007/data.npz               state (N, 6) follower joints, action (N, 6) leader joints,
                                              timestamp (N,) seconds from the episode start
    <root>/meta.json                          fps, joint names, cameras, task, image size
    <root>/demos.csv                          log of every attempt: episode, status (saved / discarded /
                                              too_short / quit), start time, duration, frames, measured
                                              rate, frames over budget, configuration ID, operator, notes
Convert to a LeRobot dataset for training with convert_to_lerobot.py.
"""

import argparse
import csv
import json
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DISPLAY_SCALE = 0.75  # window only; the saved images are full size
LOG_FIELDS = ["episode", "status", "start_time", "duration_s", "frames", "mean_hz", "slow_frames",
              "config_id", "operator", "notes"]


def log_attempt(path: Path, row: dict) -> None:
    """Appends one attempt to the demonstration log (CSV), writing the header on first use."""
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


class ImageWriter:
    """Writes JPEGs from a queue in background threads, so the control loop never waits on disk."""

    def __init__(self, threads: int, quality: int):
        self.queue: queue.Queue = queue.Queue()
        self.params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        self.workers = [threading.Thread(target=self._run, daemon=True) for _ in range(threads)]
        for w in self.workers:
            w.start()

    def _run(self) -> None:
        while (item := self.queue.get()) is not None:
            path, rgb = item
            cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.params)
            self.queue.task_done()
        self.queue.task_done()

    def put(self, path: Path, rgb: np.ndarray) -> None:
        self.queue.put((path, rgb))

    def wait(self) -> None:
        self.queue.join()

    def close(self) -> None:
        self.wait()
        for _ in self.workers:
            self.queue.put(None)
        for w in self.workers:
            w.join()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="Folder of the raw dataset (created or extended)")
    p.add_argument("--follower-port", required=True)
    p.add_argument("--follower-id", required=True)
    p.add_argument("--leader-port", required=True)
    p.add_argument("--leader-id", required=True)
    p.add_argument("--wrist", required=True, help="Wrist camera OpenCV index")
    p.add_argument("--scene", required=True, help="Scene camera OpenCV index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--task", default="Pick up the red cube and place it in the blue tray.")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--config", type=int, default=0, help="Initial-configuration ID to start with (change with [ ] in the window)")
    p.add_argument("--operator", default="", help="Who is demonstrating (logged)")
    p.add_argument("--max-episodes", type=int, default=0, help="Stop after this many saved in this session (0 = no limit)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    cams = {"wrist": int(args.wrist), "scene": int(args.scene)}
    meta = {"fps": args.fps, "joints": JOINTS, "cameras": list(cams), "image_size": [args.height, args.width],
            "task": args.task, "robot_type": "so101_follower"}
    meta_file = root / "meta.json"
    if meta_file.exists():  # adding to an existing dataset: everything must match
        old = json.loads(meta_file.read_text())
        diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{root} was recorded with different settings {diff}; use a new DATASET_NAME.")
    meta_file.write_text(json.dumps(meta, indent=2))
    next_ep = 1 + max((int(p.name.split("_")[1]) for p in root.glob("episode_*") if (p / "data.npz").exists()), default=-1)

    follower = SO101Follower(SO101FollowerConfig(
        port=args.follower_port, id=args.follower_id,
        cameras={k: OpenCVCameraConfig(index_or_path=i, width=args.width, height=args.height, fps=args.fps) for k, i in cams.items()},
    ))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
    writer = ImageWriter(threads=6, quality=args.jpeg_quality)
    window = "A2 recorder - SPACE start/stop, d discard, [ ] config, q quit"
    log_file = root / "demos.csv"
    config_id = args.config

    def log(status: str, rows: list, start_time: str, slow: int) -> None:
        ts = [r[2] for r in rows]
        dur = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        log_attempt(log_file, {
            "episode": next_ep, "status": status, "start_time": start_time, "duration_s": f"{dur:.2f}",
            "frames": len(rows), "mean_hz": f"{(len(ts) - 1) / dur:.2f}" if dur > 0 else "",
            "slow_frames": slow, "config_id": config_id, "operator": args.operator,
        })

    leader.connect()
    follower.connect()
    try:
        # Glide the follower to the leader's pose instead of jumping there.
        start = np.array([follower.get_observation()[f"{j}.pos"] for j in JOINTS])
        goal = np.array([leader.get_action()[f"{j}.pos"] for j in JOINTS])
        for s in np.linspace(0, 1, 2 * args.fps):
            follower.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, start + s * (goal - start))})
            precise_sleep(1 / args.fps)

        dt = 1.0 / args.fps
        recording, ep_dir, rows, t_start, start_time = False, None, [], 0.0, ""
        saved_this_session, loop_hz, slow = 0, 0.0, 0
        print(f"Recording to {root} - next episode {next_ep}. Keys in the camera window: SPACE start/stop, d discard, q quit.")
        while True:
            t0 = time.perf_counter()
            action = leader.get_action()
            follower.send_action(action)
            obs = follower.get_observation()

            if recording:
                i = len(rows)
                for cam in cams:
                    writer.put(ep_dir / cam / f"{i:06d}.jpg", obs[cam])
                rows.append(([obs[f"{j}.pos"] for j in JOINTS], [action[f"{j}.pos"] for j in JOINTS], t0 - t_start))

            # Window: wrist above scene (shown at 75% so it fits a laptop screen), status bar below
            frames = []
            for c in cams:
                f = cv2.resize(cv2.cvtColor(obs[c], cv2.COLOR_RGB2BGR), None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
                cv2.putText(f, c, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                frames.append(f)
            bar = np.zeros((58, frames[0].shape[1], 3), np.uint8)
            if recording:
                line1, colour = f"REC  ep {next_ep}  config {config_id}  {t0 - t_start:5.1f} s  {len(rows)} fr", (0, 0, 255)
                line2 = "SPACE save   d discard"
                cv2.circle(frames[0], (frames[0].shape[1] - 18, 18), 9, (0, 0, 255), -1)
            else:
                line1, colour = f"ready  ep {next_ep}  config {config_id}  ({saved_this_session} saved)", (0, 200, 0)
                line2 = "SPACE start   [ ] config   q quit"
            cv2.putText(bar, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
            cv2.putText(bar, f"{line2}   {loop_hz:4.1f} Hz", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            view = np.vstack([*frames, bar])
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and not recording:
                ep_dir = root / f"episode_{next_ep:03d}"
                shutil.rmtree(ep_dir, ignore_errors=True)  # leftovers of an unsaved attempt
                for cam in cams:
                    (ep_dir / cam).mkdir(parents=True)
                recording, rows, t_start, slow = True, [], time.perf_counter(), 0
                start_time = datetime.now().isoformat(timespec="seconds")
                print(f"Episode {next_ep} (config {config_id}): recording...")
            elif key == ord(" ") and recording:
                recording = False
                if len(rows) < args.fps:  # under a second: a double press, not a demonstration
                    writer.wait()
                    shutil.rmtree(ep_dir, ignore_errors=True)
                    print(f"Episode {next_ep}: too short ({len(rows)} frames), not saved.")
                    log("too_short", rows, start_time, slow)
                    continue
                state, act, ts = (np.array(x, dtype=np.float32) for x in zip(*rows))
                np.savez(ep_dir / "data.npz", state=state, action=act, timestamp=ts)
                log("saved", rows, start_time, slow)
                print(f"Episode {next_ep}: saved {len(rows)} frames ({len(rows) / args.fps:.1f} s)"
                      + (f", {slow} frames over the {dt * 1000:.0f} ms budget" if slow else ""))
                next_ep += 1
                saved_this_session += 1
                if args.max_episodes and saved_this_session >= args.max_episodes:
                    break
            elif key == ord("d") and recording:
                recording = False
                writer.wait()
                shutil.rmtree(ep_dir, ignore_errors=True)
                print(f"Episode {next_ep}: discarded.")
                log("discarded", rows, start_time, slow)
            elif key in (ord("["), ord("]")):
                config_id = max(0, config_id + (1 if key == ord("]") else -1))
            elif key in (ord("q"), 27):
                if recording:
                    print(f"Episode {next_ep}: not saved (quit while recording).")
                    log("quit", rows, start_time, slow)
                    writer.wait()
                    shutil.rmtree(ep_dir, ignore_errors=True)
                break

            elapsed = time.perf_counter() - t0
            if recording and elapsed > dt:
                slow += 1
            precise_sleep(max(dt - elapsed, 0.0))
            loop_hz = 0.9 * loop_hz + 0.1 / (time.perf_counter() - t0)
    finally:
        print("Finishing image writes...")
        writer.close()
        cv2.destroyAllWindows()
        follower.disconnect()  # torque off: support the follower arm
        leader.disconnect()
        total = sum(1 for p in root.glob("episode_*") if (p / "data.npz").exists())
        print(f"Done. {total} episodes in {root}.")


if __name__ == "__main__":
    main()
