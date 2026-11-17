#!/usr/bin/env python
"""Live view of the cameras, one above the other in one window (to place the scene camera).

    python camera_view.py wrist=Innomaker scene=Logitech [--width 640 --height 480 --fps 30]

Each camera is given as label=NAME, where NAME is part of its name (or an OpenCV index). The window
shows the frames at the size they will be recorded, with the measured frame rate.
Keys: q / ESC quit, s save a snapshot of each camera to A2/snapshots/, g toggle a 3x3 grid.
"""

import argparse
import contextlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from camera_lookup import resolve_camera  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cameras", nargs="+", help="label=NAME pairs, e.g. wrist=Innomaker scene=Logitech")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    caps = {}
    for spec in args.cameras:
        label, _, name = spec.partition("=")
        if not name:
            print(f"Skipping {label}: no camera set (see A2/config.sh).")
            continue
        with contextlib.redirect_stdout(sys.stderr):
            index = resolve_camera(name)
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        if not cap.isOpened():
            sys.exit(f"Could not open the {label} camera ({name}, index {index}).")
        caps[label] = cap
        print(f"{label}: {name} (index {index}), "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    if not caps:
        sys.exit("No cameras to show.")

    snap_dir = Path(__file__).resolve().parents[1] / "snapshots"  # A2/snapshots
    grid, rates, last = False, {k: 0.0 for k in caps}, {k: time.perf_counter() for k in caps}
    print("Keys: q / ESC quit, s snapshot, g grid")
    try:
        while True:
            views = []
            for label, cap in caps.items():
                ok, frame = cap.read()
                now = time.perf_counter()
                if not ok:
                    frame = np.zeros((args.height, args.width, 3), np.uint8)
                    cv2.putText(frame, "no frame", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                else:
                    rates[label] = 0.9 * rates[label] + 0.1 / max(now - last[label], 1e-3)
                last[label] = now
                frame = cv2.resize(frame, (args.width, args.height))
                if grid:
                    for i in (1, 2):
                        cv2.line(frame, (i * args.width // 3, 0), (i * args.width // 3, args.height), (255, 255, 255), 1)
                        cv2.line(frame, (0, i * args.height // 3), (args.width, i * args.height // 3), (255, 255, 255), 1)
                cv2.putText(frame, f"{label}  {rates[label]:.0f} fps", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                views.append(frame)
            stacked = np.vstack(views)  # one above the other, shown at 75% so it fits a laptop screen
            cv2.imshow("A2 cameras (q to quit)", cv2.resize(stacked, None, fx=0.75, fy=0.75))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                grid = not grid
            if key == ord("s"):
                snap_dir.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                for label, view in zip(caps, views):
                    cv2.imwrite(str(snap_dir / f"{label}_{stamp}.png"), view)
                print(f"Saved snapshots to {snap_dir}")
    except KeyboardInterrupt:
        pass
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
