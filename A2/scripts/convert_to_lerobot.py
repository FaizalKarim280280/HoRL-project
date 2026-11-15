#!/usr/bin/env python
"""Convert raw demonstrations (record_demos.py) into a LeRobot dataset for training.

    python convert_to_lerobot.py --raw A2/data/raw/so101_pick_place --out A2/data/lerobot/so101_pick_place

The dataset has the same features lerobot-record would produce for the SO-101 with a wrist and a
scene camera: observation.state, action (6 joints each), observation.images.wrist and
observation.images.scene (videos). Video encoding happens here, once, instead of while recording.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts


def dataset_features(cameras: list[str], height: int, width: int, fps: int) -> dict:
    """Exactly the features lerobot-record builds for this robot and these cameras."""
    robot = SO101Follower(SO101FollowerConfig(
        port="unused", cameras={c: OpenCVCameraConfig(index_or_path=0, width=width, height=height, fps=fps) for c in cameras},
    ))
    teleop_proc, _, obs_proc = make_default_processors()
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(pipeline=teleop_proc, initial_features=create_initial_features(action=robot.action_features), use_videos=True),
        aggregate_pipeline_dataset_features(pipeline=obs_proc, initial_features=create_initial_features(observation=robot.observation_features), use_videos=True),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo-id", default=None, help="Default: local/<out folder name>")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing converted dataset")
    args = p.parse_args()

    meta = json.loads((args.raw / "meta.json").read_text())
    episodes = sorted(d for d in args.raw.glob("episode_*") if (d / "data.npz").exists())
    if not episodes:
        raise SystemExit(f"No recorded episodes in {args.raw}")
    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} exists - pass --overwrite to rebuild it from the raw data.")
        shutil.rmtree(args.out)

    height, width = meta["image_size"]
    features = dataset_features(meta["cameras"], height, width, meta["fps"])
    ds = LeRobotDataset.create(
        args.repo_id or f"local/{args.out.name}", meta["fps"], root=args.out, robot_type=meta["robot_type"],
        features=features, use_videos=True, image_writer_threads=4 * len(meta["cameras"]),
    )
    joints = meta["joints"]
    t0 = time.perf_counter()
    for n, ep in enumerate(episodes, 1):
        data = np.load(ep / "data.npz")
        frames = len(data["state"])
        for i in range(frames):
            obs = {f"{j}.pos": float(v) for j, v in zip(joints, data["state"][i])}
            for cam in meta["cameras"]:
                img = cv2.imread(str(ep / cam / f"{i:06d}.jpg"))
                if img is None:
                    raise SystemExit(f"Missing image {ep / cam / f'{i:06d}.jpg'}")
                obs[cam] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            act = {f"{j}.pos": float(v) for j, v in zip(joints, data["action"][i])}
            ds.add_frame({**build_dataset_frame(ds.features, obs, prefix=OBS_STR),
                          **build_dataset_frame(ds.features, act, prefix=ACTION), "task": meta["task"]})
        ds.save_episode()
        print(f"[{n}/{len(episodes)}] {ep.name}: {frames} frames ({time.perf_counter() - t0:.0f} s so far)")
    ds.finalize()
    print(f"Done: {ds.num_episodes} episodes, {ds.num_frames} frames -> {args.out}")


if __name__ == "__main__":
    main()
