#!/usr/bin/env python
"""Summary of the raw recordings (record_demos.py) and of the converted LeRobot dataset.

    python dataset_summary.py A2/data/raw/so101_pick_place A2/data/lerobot/so101_pick_place
"""

import json
import sys
from pathlib import Path

import numpy as np


def raw_summary(root: Path) -> None:
    if not (root / "meta.json").exists():
        print(f"Raw: nothing recorded yet in {root}")
        return
    meta = json.loads((root / "meta.json").read_text())
    lengths = [len(np.load(ep / "data.npz")["state"]) for ep in sorted(root.glob("episode_*")) if (ep / "data.npz").exists()]
    if not lengths:
        print(f"Raw: no saved episodes in {root}")
        return
    fps = meta["fps"]
    print(f"Raw {root}")
    print(f"  {len(lengths)} episodes, {sum(lengths)} frames ({sum(lengths) / fps / 60:.1f} min) at {fps} fps, cameras {meta['cameras']}")
    print(f"  episode length: min {min(lengths) / fps:.1f} s, mean {np.mean(lengths) / fps:.1f} s, max {max(lengths) / fps:.1f} s")


def lerobot_summary(root: Path) -> None:
    if not (root / "meta" / "info.json").exists():
        print(f"LeRobot: not converted yet ({root}) - run ./03_convert.sh")
        return
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("local/" + root.name, root=root)
    print(f"LeRobot {root}")
    print(f"  {meta.total_episodes} episodes, {meta.total_frames} frames at {meta.fps} fps, robot {meta.robot_type}")
    print(f"  cameras: {', '.join(f'{k} {meta.shapes[k]}' for k in meta.camera_keys)}; "
          f"state {meta.shapes.get('observation.state')}, action {meta.shapes.get('action')}")


if __name__ == "__main__":
    raw_summary(Path(sys.argv[1]))
    if len(sys.argv) > 2:
        lerobot_summary(Path(sys.argv[2]))
