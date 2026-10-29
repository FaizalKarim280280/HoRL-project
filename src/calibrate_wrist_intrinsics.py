#!/usr/bin/env python
"""Calibrate the wrist camera's intrinsics (focal length, centre, lens distortion) with the
printed ChArUco board. Only the camera is used - the arm is not controlled.

Open the viser page, then show the board to the camera from many angles and distances:
tilt it up to ~45 deg, and put it near the image corners and edges as well as the centre
(distortion is strongest there). You can hold the board, or put it on the table and move the
arm by hand (with the motors unpowered or torque off). Sharp, still views are captured
automatically when they differ enough from earlier ones; aim for 25+, including views with
the board in each of the four image CORNERS (the status line shows which corners are still
missing). Then click "Calibrate & save". RMS reprojection error below ~0.5 px is good, below
1 px is usable, and the lens-model check must pass.

The captured views are saved to wrist_intrinsics_views.npz, so the calibration can be
re-solved later without the camera: --from-views (optionally with --full-model).
"""

import argparse
import threading
import time

import cv2
import numpy as np
import viser

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from cube_pick_common import resolve_camera
from wrist_vision import INTRINSICS_FILE, BoardDetector, BoardSpec, Intrinsics, check_distortion

VIEWS_FILE = INTRINSICS_FILE.with_name("wrist_intrinsics_views.npz")


def view_signature(img_pts: np.ndarray) -> np.ndarray:
    """Where the board is in the image and how big / tilted it looks."""
    centre = img_pts.mean(axis=0)
    spread = img_pts.std(axis=0)
    return np.array([centre[0], centre[1], spread[0], spread[1]])


def corner_coverage(points: list[np.ndarray], width: int, height: int) -> dict[str, int]:
    """Board corners captured inside each quarter-size corner region of the image."""
    pts = np.concatenate(points) if points else np.zeros((0, 2))
    left, right = pts[:, 0] < width / 4, pts[:, 0] > 3 * width / 4
    top, bottom = pts[:, 1] < height / 4, pts[:, 1] > 3 * height / 4
    return {
        "top-left": int(np.sum(left & top)),
        "top-right": int(np.sum(right & top)),
        "bottom-left": int(np.sum(left & bottom)),
        "bottom-right": int(np.sum(right & bottom)),
    }


def as_object_array(arrays: list[np.ndarray]) -> np.ndarray:
    """1-D object array of per-view arrays (np.array() would stack equal-sized views into 3-D)."""
    out = np.empty(len(arrays), dtype=object)
    out[:] = arrays
    return out


def calibrate(obj_list, img_list, width: int, height: int, spec: BoardSpec, full_model: bool) -> str:
    """Solves, saves wrist_intrinsics.json and the views, and returns a status report."""
    # k3 is fixed at 0 unless asked: with views that do not reach the image corners it overfits
    # and makes the model fold back on itself near the corners.
    flags = 0 if full_model else cv2.CALIB_FIX_K3
    rms, k, dist, _, _ = cv2.calibrateCamera(obj_list, img_list, (width, height), None, None, flags=flags)
    ok, worst = check_distortion(k, dist, width, height)
    model = "full" if full_model else "k3 fixed"
    if not ok:
        # Uneven coverage (e.g. one side of the image never saw the board) lets the image centre
        # and tangential distortion drift to fit the data, and the model fails near the empty
        # side. Small wrist cameras are well centred and untilted, so pin those down and re-solve.
        k_guess = np.array([[k[0, 0], 0, width / 2], [0, k[1, 1], height / 2], [0, 0, 1.0]])
        flags_c = flags | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS
        rms_c, k_c, dist_c, _, _ = cv2.calibrateCamera(obj_list, img_list, (width, height), k_guess, np.zeros(5), flags=flags_c)
        ok_c, worst_c = check_distortion(k_c, dist_c, width, height)
        if ok_c:
            rms, k, dist, ok, worst = rms_c, k_c, dist_c, ok_c, worst_c
            model += ", centre fixed, no tangential (fallback: uneven coverage)"
    Intrinsics(k.tolist(), dist.ravel().tolist(), width, height, float(rms), len(obj_list), spec).save()
    np.savez(VIEWS_FILE, obj=as_object_array(obj_list), img=as_object_array(img_list), size=(width, height))
    cov = corner_coverage(img_list, width, height)
    missing = [name for name, n in cov.items() if n < 10]
    return (
        f"**Saved {INTRINSICS_FILE.name}** from {len(obj_list)} views ({model} model): "
        f"RMS **{rms:.2f} px**, fx {k[0, 0]:.0f}, fy {k[1, 1]:.0f}, cx {k[0, 2]:.0f}, cy {k[1, 2]:.0f}.  \n"
        + ("RMS good. " if rms < 0.5 else "RMS usable. " if rms < 1.0 else "RMS high - recapture sharper views. ")
        + (f"Lens-model check **passed** ({worst:.2f} px)." if ok else
           f"Lens-model check **FAILED** ({worst:.0f} px at the image edges) - capture more views near the image corners.")
        + (f"  \nCorners with little coverage: {', '.join(missing)}." if missing else "")
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-index", default="0", help="Wrist camera index (run lerobot-find-cameras opencv)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--squares-x", type=int, default=5, help="Board squares across")
    p.add_argument("--squares-y", type=int, default=7, help="Board squares down")
    p.add_argument("--square-mm", type=float, default=30.0, help="Measured square size of the printed board")
    p.add_argument("--marker-mm", type=float, default=22.0, help="Measured marker size of the printed board")
    p.add_argument("--dictionary", default="DICT_4X4_50", help="ArUco dictionary of the board")
    p.add_argument("--full-model", action="store_true", help="Also fit k3 (only with good corner coverage)")
    p.add_argument("--from-views", action="store_true", help=f"Re-solve from {VIEWS_FILE.name}, no camera")
    p.add_argument("--viser-port", type=int, default=8080)
    args = p.parse_args()

    spec = BoardSpec(args.squares_x, args.squares_y, args.square_mm / 1000, args.marker_mm / 1000, args.dictionary)
    if args.from_views:
        views = np.load(VIEWS_FILE, allow_pickle=True)
        width, height = (int(v) for v in views["size"])
        obj_list = [np.asarray(v, dtype=np.float32) for v in views["obj"]]
        img_list = [np.asarray(v, dtype=np.float32) for v in views["img"]]
        print(calibrate(obj_list, img_list, width, height, spec, args.full_model).replace("**", ""))
        return
    detector = BoardDetector(spec)
    index = resolve_camera(args.camera_index)
    camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=index, width=args.width, height=args.height, fps=args.fps))

    server = viser.ViserServer(port=args.viser_port)
    status = server.gui.add_markdown("Starting camera...")
    image = server.gui.add_image(np.zeros((8, 8, 3), np.uint8), label="Wrist camera", format="jpeg")
    auto = server.gui.add_checkbox("Auto-capture", initial_value=True)
    capture_now, calibrate_now = threading.Event(), threading.Event()
    server.gui.add_button("Capture now").on_click(lambda _: capture_now.set())
    server.gui.add_button("Calibrate & save", color="green").on_click(lambda _: calibrate_now.set())

    obj_list, img_list, signatures = [], [], []
    coverage = np.zeros((args.height, args.width), np.uint8)
    prev_pts = None
    last_ui = 0.0
    camera.connect()
    print(f"viser running at http://localhost:{args.viser_port}")
    try:
        while True:
            frame = camera.async_read(1000)
            found = detector.detect(frame)
            view = frame.copy()
            view[coverage > 0] = (0.6 * view[coverage > 0] + [0, 0, 100]).astype(np.uint8)
            note = "board not found"
            if found is not None:
                corners, ids = found
                detector.draw(view, corners, ids)
                obj, img = detector.object_image_points(corners, ids)
                pts = corners.reshape(-1, 2)
                still = prev_pts is not None and len(prev_pts) == len(pts) and np.max(np.abs(prev_pts - pts)) < 1.5
                prev_pts = pts
                sig = view_signature(img)
                novel = all(
                    np.hypot(*(sig[:2] - s[:2])) > 40 or np.max(np.abs(sig[2:] - s[2:]) / (s[2:] + 1)) > 0.25
                    for s in signatures
                )
                note = f"{len(ids)} corners" + ("" if still else " (hold still)") + ("" if novel else " (similar view)")
                if len(ids) >= 8 and (capture_now.is_set() or (auto.value and still and novel)):
                    obj_list.append(obj.astype(np.float32))
                    img_list.append(img.astype(np.float32))
                    signatures.append(sig)
                    for u, v in img.astype(int):
                        cv2.circle(coverage, (u, v), 6, 255, -1)
                    note = "captured"
            else:
                prev_pts = None
            capture_now.clear()

            if calibrate_now.is_set():
                calibrate_now.clear()
                if len(obj_list) < 10:
                    status.content = f"**Need at least 10 views** (have {len(obj_list)})."
                else:
                    status.content = calibrate(obj_list, img_list, args.width, args.height, spec, args.full_model)
                    print(status.content.replace("**", ""))
                    continue

            now = time.perf_counter()
            if now - last_ui > 0.1:
                last_ui = now
                image.image = view
                if not status.content.startswith("**Saved"):
                    cov = corner_coverage(img_list, args.width, args.height)
                    todo = [name for name, n in cov.items() if n < 10]
                    status.content = (
                        f"Views: **{len(obj_list)}** (aim for 25+; blue = covered areas) · {note}  \n"
                        + (f"Still need the board in the **{', '.join(todo)}** corner(s)." if todo else "All four image corners covered.")
                    )
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        camera.disconnect()
        server.stop()


if __name__ == "__main__":
    main()
