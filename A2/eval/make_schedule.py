#!/usr/bin/env python
"""Sample the initial configurations and the randomised trial schedule from eval_plan.yaml.

Writes, next to the plan (plan/):
    configs.csv          config_id + target cube and tray pose in the scene image (centre px, size px, rotation
                         deg), sampled inside the area where the demonstrations started (pilot configs: pilot01 ...)
    configs.jpg          every configuration's targets drawn on the scene background (for the report)
    scene_reference.jpg  median scene image of the demos' first frames (camera-position check at evaluation)
    schedule.csv         trial, block, config_id, variant: every config is one block in which each variant
                         runs once; block order and variant order within a block are random (fixed seed)
    pilot_schedule.csv   one practice trial per pilot config (never counted)
    home_pose.json       the follower pose every rollout starts from: median start pose of the demos
Refuses to overwrite an existing schedule (it is pre-registered): pass --force only before evaluation starts.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from scene_detect import box, detect, draw

HERE = Path(__file__).resolve().parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def home_pose(raw_dirs: list[Path]) -> dict:
    """Median of the follower's first recorded state over all demonstrations."""
    starts = [np.load(p)["state"][0] for d in raw_dirs for p in sorted(d.glob("episode_*/data.npz"))]
    if not starts:
        raise SystemExit(f"No demonstrations found in {raw_dirs}")
    return {"joints": JOINTS, "pose": [round(float(x), 2) for x in np.median(starts, axis=0)], "n_demos": len(starts)}


def demo_starts(raw_dirs: list[Path]) -> tuple[list[dict], np.ndarray]:
    """Cube and tray detected in the first scene frame of every demonstration, and the median image."""
    frames = [p for d in raw_dirs for p in sorted(d.glob("episode_*/scene/000000.jpg"))]
    imgs = [cv2.imread(str(f)) for f in frames]
    dets = [d for d in map(detect, imgs) if d["cube"] and d["tray"]]
    return dets, np.median(np.stack(imgs[::2]), axis=0).astype(np.uint8)


def size_model(objs: np.ndarray, shape: tuple):
    """Apparent edge length (px) of a square object as a linear function of its image position.

    Fitted on the demos' fully visible objects (not touching the image border, near-square outline), so a
    target drawn anywhere has the size the real object shows there (perspective: smaller further away).
    """
    H, W = shape[:2]
    keep = [o for o in objs if 0.8 < o[2] / o[3] < 1.25 and
            (box(*o)[:, 0].min() > 3) and (box(*o)[:, 1].min() > 3) and (box(*o)[:, 0].max() < W - 3) and (box(*o)[:, 1].max() < H - 3)]
    k = np.array(keep)
    A = np.c_[k[:, 0], k[:, 1], np.ones(len(k))]
    coef, *_ = np.linalg.lstsq(A, k[:, 2:4].mean(1), rcond=None)
    return lambda x, y: float(coef @ [x, y, 1.0])


def sample_configs(dets: list[dict], n: int, rng: np.random.Generator, c: dict, shape: tuple) -> list[dict]:
    """n random (cube, tray) target poses where the demos' cubes and trays started.

    Centres: uniform over the region within `near_demo_px` of a demo start position (per object), so
    targets stay where demonstrations actually were. Rotations: uniform. Sizes: from size_model. A sample
    is rejected if the cube is closer than min_gap_px to the tray or either object leaves the image.
    """
    H, W = shape[:2]
    starts = {k: np.array([d[k] for d in dets]) for k in ("cube", "tray")}
    size = {k: size_model(v, shape) for k, v in starts.items()}

    def point_near(k: str) -> np.ndarray:
        pts = starts[k][:, :2]
        lo, hi = pts.min(0), pts.max(0)
        while True:
            pt = rng.uniform(lo, hi)
            if np.linalg.norm(pts - pt, axis=1).min() <= c["near_demo_px"]:
                return pt

    out = []
    while len(out) < n:
        cp, tp = point_near("cube"), point_near("tray")
        cs, ts = size["cube"](*cp), size["tray"](*tp)
        cpose = (cp[0], cp[1], cs, cs, rng.uniform(0, 90))
        tpose = (tp[0], tp[1], ts, ts, rng.uniform(0, 90))
        corners = np.vstack([box(*cpose), box(*tpose)])
        if corners.min() < 5 or (corners[:, 0] > W - 5).any() or (corners[:, 1] > H - 5).any():
            continue
        gap = -cv2.pointPolygonTest(box(*tpose).astype(np.float32), (float(cp[0]), float(cp[1])), True)
        if gap < cs * 0.71 + c["min_gap_px"]:  # cube half-diagonal + gap
            continue
        out.append(dict(cube_x=round(cp[0], 1), cube_y=round(cp[1], 1), cube_size=round(cs, 1),
                        cube_yaw=round(cpose[4], 1), tray_x=round(tp[0], 1), tray_y=round(tp[1], 1),
                        tray_w=round(ts, 1), tray_h=round(ts, 1), tray_yaw=round(tpose[4], 1)))
    return out


def target_poses(cfg: dict) -> tuple[tuple, tuple]:
    f = {k: float(v) for k, v in cfg.items() if k.startswith(("cube_", "tray_"))}
    return ((f["cube_x"], f["cube_y"], f["cube_size"], f["cube_size"], f["cube_yaw"]),
            (f["tray_x"], f["tray_y"], f["tray_w"], f["tray_h"], f["tray_yaw"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=HERE / "plan" / "eval_plan.yaml")
    p.add_argument("--force", action="store_true", help="Overwrite an existing schedule (only before evaluating)")
    args = p.parse_args()
    plan = yaml.safe_load(args.plan.read_text())
    out = args.plan.parent
    if (out / "schedule.csv").exists() and not args.force:
        raise SystemExit(f"{out / 'schedule.csv'} exists (pre-registered). Use --force only if no trial was run.")

    c = plan["configurations"]
    n_cfg, n_pilot = plan["n_max"], plan.get("n_pilot_configs", 0)
    dets, background = demo_starts([(out / d).resolve() for d in c["sample_from"]])
    print(f"Cube and tray found in the first scene frame of {len(dets)} demonstrations")
    rng = np.random.default_rng(plan["seed"])
    picked = sample_configs(dets, n_cfg + n_pilot, rng, c, background.shape)
    configs = [dict(config_id=f"cfg{i + 1:02d}", pilot=0, **t) for i, t in enumerate(picked[:n_cfg])]
    configs += [dict(config_id=f"pilot{i + 1:02d}", pilot=1, **t) for i, t in enumerate(picked[n_cfg:])]
    write_csv(out / "configs.csv", configs)
    cv2.imwrite(str(out / "scene_reference.jpg"), background)
    tiles = []
    for cfg in configs:
        img = background.copy()
        cube_t, tray_t = target_poses(cfg)
        draw(img, tray_t, (0, 255, 0), 3)
        draw(img, cube_t, (0, 255, 255), 3)
        cv2.putText(img, cfg["config_id"], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        tiles.append(cv2.resize(img, (320, 240)))
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
    cv2.imwrite(str(out / "configs.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))

    variants = list(plan["variants"])
    schedule, trial = [], 0
    for block, k in enumerate(rng.permutation(n_cfg), 1):
        for v in rng.permutation(variants):
            trial += 1
            schedule.append(dict(trial=trial, block=block, config_id=configs[k]["config_id"], variant=v))
    write_csv(out / "schedule.csv", schedule)
    pilots = [c for c in configs if c["pilot"]]
    if pilots:
        write_csv(out / "pilot_schedule.csv", [dict(trial=i + 1, block=i + 1, config_id=c["config_id"],
                                                    variant=variants[i % len(variants)]) for i, c in enumerate(pilots)])

    pose = home_pose([(out / d).resolve() for d in plan["home_pose_from"]])
    (out / "home_pose.json").write_text(json.dumps(pose, indent=2))
    print(f"{n_cfg} configs + {n_pilot} pilot configs -> configs.csv; {len(schedule)} trials -> schedule.csv")
    print(f"Home pose (median start of {pose['n_demos']} demos): {dict(zip(JOINTS, pose['pose']))}")


if __name__ == "__main__":
    main()
