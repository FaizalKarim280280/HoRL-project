#!/usr/bin/env python
"""Automatic wrist-camera calibration: lens intrinsics AND hand-eye in one run, with the arm
moving the camera around the ChArUco board by itself.

1. Tick "Release arm", move the arm by hand to the scan pose (camera looking down at the
   table, whole board in view, ~20-25 cm up), untick it, click "Set scan pose".
2. Coverage phase: the arm steers the board into each image corner, each edge and the centre
   (and rolls the wrist at each), using image feedback - no calibration needed: it nudges
   the base pan and wrist flex, measures how the board moves in the image, and corrects.
3. Variety phase: random poses around the scan pose (pan, pitch, roll) for tilted views.
4. Solves the intrinsics (k3 fixed unless --full-model) and checks the lens model, then
   solves hand-eye from the same views. Writes wrist_intrinsics.json (+ views) and
   wrist_calibration.json - the same files as 07_wrist_intrinsics.sh + 08_wrist_handeye.sh.
"""

import argparse
from pathlib import Path

import numpy as np

from calibrate_wrist_handeye import sample_poses, save_and_show, wait_for_scan_pose
from calibrate_wrist_intrinsics import as_object_array, calibrate
from cube_pick_common import CameraSettings, DetectionSettings, Station
from wrist_vision import WRIST_CALIB_FILE, BoardDetector, BoardSpec, Intrinsics, check_distortion

PAN, FLEX, ROLL = 0, 3, 4
MAX_OFFSET = np.array([35.0, 20.0, 20.0, 40.0, 60.0])  # max deviation from the scan pose per arm joint (deg)
MAX_STEP = 6.0  # max joint change per servo step (deg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--camera-index", default="0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    BoardSpec.add_args(p)
    p.add_argument("--margin", type=float, default=0.18, help="How far into the corners to steer (fraction of image)")
    p.add_argument("--random-samples", type=int, default=12, help="Random poses to visit at least")
    p.add_argument("--target-views", type=int, default=30, help="Keep sampling random poses until this many views")
    p.add_argument("--min-height", type=float, default=0.10, help="Lowest allowed gripper height (m)")
    p.add_argument("--full-model", action="store_true", help="Also fit lens distortion k3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=WRIST_CALIB_FILE)
    p.add_argument("--viser-port", type=int, default=8080)
    return p.parse_args()


class AutoCapture:
    def __init__(self, station: Station, detector: BoardDetector, scan_q: np.ndarray, min_height: float):
        self.station, self.detector = station, detector
        self.scan_q = scan_q
        self.min_height = min_height
        self.captures: list[dict] = []
        self.jacobian: np.ndarray | None = None  # d(board centroid px) / d(pan, flex deg)

    # -- helpers -------------------------------------------------------------------------------

    def safe(self, q: np.ndarray) -> bool:
        if np.any(np.abs(q[:5] - self.scan_q[:5]) > MAX_OFFSET):
            return False
        return self.station.within_limits(q, 3) and self.station.kin.forward_kinematics(q)[2, 3] >= self.min_height

    def look(self, settle: float = 0.5):
        """Settles, then returns (corners, ids, centroid) or None."""
        self.station.idle(settle)
        found = self.detector.detect(self.station.frame)
        if found is None:
            return None
        return found[0], found[1], found[0].reshape(-1, 2).mean(axis=0)

    def move(self, q: np.ndarray) -> None:
        self.station.joint_move(q, speed_deg_s=30)

    def capture(self, label: str) -> bool:
        seen = self.look(0.6)
        if seen is None or len(seen[1]) < 8:
            print(f"  {label}: not captured ({'no board' if seen is None else f'{len(seen[1])} corners'})")
            return False
        corners, ids, _ = seen
        obj, img = self.detector.object_image_points(corners, ids)
        self.captures.append({"obj": obj, "img": img, "corners": corners, "ids": ids, "q": self.station.q.copy()})
        print(f"  {label}: captured ({len(ids)} corners) - {len(self.captures)} views")
        return True

    # -- coverage phase: steer the board to image targets --------------------------------------

    def estimate_jacobian(self) -> bool:
        base = self.look()
        if base is None:
            return False
        q0 = self.station.q_cmd.copy()
        cols = []
        for joint in (PAN, FLEX):
            for delta in (4.0, -4.0):
                q = q0.copy()
                q[joint] += delta
                if not self.safe(q):
                    continue
                self.move(q)
                seen = self.look()
                self.move(q0)
                if seen is not None:
                    cols.append((seen[2] - base[2]) / delta)
                    break
            else:
                return False
        self.jacobian = np.column_stack(cols)
        print(f"  image Jacobian (px/deg): pan {np.round(cols[0], 1)}, wrist flex {np.round(cols[1], 1)}")
        return abs(np.linalg.det(self.jacobian)) > 1.0

    def servo_to(self, target: np.ndarray, iters: int = 8, tol_px: float = 25.0) -> bool:
        """Moves pan + wrist flex until the board's centroid is near `target` (pixels)."""
        seen = self.look(0.3)
        for _ in range(iters):
            if seen is None:
                return False
            err = target - seen[2]
            if np.linalg.norm(err) < tol_px:
                return True
            step = np.clip(np.linalg.lstsq(self.jacobian, err, rcond=None)[0], -MAX_STEP, MAX_STEP)
            q = self.station.q_cmd.copy()
            q[[PAN, FLEX]] += step
            if not self.safe(q):
                return True  # the target needs the arm outside its safe range: capture as close as we got
            prev = seen
            self.move(q)
            seen = self.look(0.3)
            if seen is None:  # board left the view: back off half a step
                q[[PAN, FLEX]] -= step / 2
                self.move(q)
                seen = self.look(0.3)
                continue
            # Broyden update: refine the Jacobian from what this step actually did
            dc = seen[2] - prev[2]
            self.jacobian += np.outer(dc - self.jacobian @ step, step) / (step @ step)
        return seen is not None

    def coverage_phase(self, width: int, height: int, margin: float) -> None:
        m = margin
        targets = [
            (m, m), (0.5, m), (1 - m, m), (1 - m, 0.5), (1 - m, 1 - m),
            (0.5, 1 - m), (m, 1 - m), (m, 0.5), (0.5, 0.5),
        ]  # fmt: skip
        for i, (tx, ty) in enumerate(targets, 1):
            target = np.array([tx * width, ty * height])
            self.station.set_status(f"**Coverage {i}/{len(targets)}:** steering the board towards pixel ({target[0]:.0f}, {target[1]:.0f})...")
            if not self.servo_to(target):
                self.move(self.scan_q)  # lost it: restart from the scan pose for the next target
                continue
            self.capture(f"target {i}")
            here = self.station.q_cmd.copy()
            jacobian = self.jacobian.copy()  # rolling the wrist rotates it; restore it afterwards
            for roll in (30.0, -30.0):
                q = here.copy()
                q[ROLL] += roll
                if self.safe(q):
                    self.move(q)
                    self.servo_to(target, iters=3)
                    self.capture(f"target {i}, roll {roll:+.0f}")  # any view with the board helps
                    self.jacobian = jacobian.copy()
            self.move(here)

    # -- variety phase -------------------------------------------------------------------------

    def variety_phase(self, n: int, target_views: int, rng) -> None:
        """At least n random poses, and more until target_views views exist (at most 3x as many tries)."""
        i = 0
        while i < 3 * max(n, target_views):
            if i >= n and len(self.captures) >= target_views:
                break
            q = sample_poses(self.station, self.scan_q, 1, self.min_height, rng)
            if not q:
                break
            i += 1
            self.station.set_status(
                f"**Variety {i}:** random pose around the scan pose · views so far **{len(self.captures)}/{target_views}**"
            )
            self.move(q[0])
            self.capture(f"random {i}")


def main() -> None:
    args = parse_args()
    spec = BoardSpec.from_args(args)
    detector = BoardDetector(spec)
    camera = CameraSettings(args.camera_index, args.width, args.height, args.fps)
    station = Station(args.port, args.id, camera, DetectionSettings(), viser_port=args.viser_port, detect_red=False)
    station.overlay = detector.overlay_for(station)
    station.connect()
    rng = np.random.default_rng(args.seed)
    try:
        scan_q = wait_for_scan_pose(station, detector)
        auto = AutoCapture(station, detector, scan_q, args.min_height)
        auto.capture("scan pose")

        station.set_status("Measuring how the board moves in the image when the arm moves...")
        if auto.estimate_jacobian():
            auto.coverage_phase(args.width, args.height, args.margin)
        else:
            print("Could not measure the image Jacobian; skipping the coverage phase.")
        auto.move(scan_q)
        auto.variety_phase(args.random_samples, args.target_views, rng)
        auto.move(scan_q)

        caps = auto.captures
        if len(caps) < 10:
            station.set_status(f"**Only {len(caps)} views captured** (need 10+). Use a scan pose with the whole board in view.")
            return
        report = calibrate([c["obj"] for c in caps], [c["img"] for c in caps], args.width, args.height, spec, args.full_model)
        intr = Intrinsics.load()
        ok, _ = check_distortion(intr.k_mat, intr.dist_vec, intr.width, intr.height)
        if not ok:
            station.set_status(report + "  \nHand-eye skipped: fix the intrinsics first (more corner coverage).")
            while True:
                station.idle(1.0)

        np.savez(  # raw data, so the calibration can be analysed or re-solved offline
            args.out.with_name("wrist_calibration_samples.npz"),
            q=np.array([c["q"] for c in caps]),
            corners=as_object_array([c["corners"] for c in caps]),
            ids=as_object_array([c["ids"] for c in caps]),
        )
        samples = []
        for c in caps:
            result = detector.pose(c["corners"], c["ids"], intr.k_mat, intr.dist_vec)
            if result is not None and result[1] < 2.0:
                samples.append({"t_base_gripper": station.kin.forward_kinematics(c["q"]), "t_cam_board": result[0]})
        if len(samples) < 8:
            station.set_status(report + f"  \n**Only {len(samples)} usable hand-eye samples** (need 8+).")
            while True:
                station.idle(1.0)
        save_and_show(station, args.out, camera, intr, samples, scan_q)
        station.set_status(report + "  \n" + station.status_md.content)
        while True:
            station.idle(1.0)
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        station.disconnect()


if __name__ == "__main__":
    main()
