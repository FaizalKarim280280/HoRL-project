"""Red cube and blue tray in the fixed scene camera: position, size and rotation in image pixels.

Plain HSV thresholds + the largest blob's minimum-area rectangle. Both objects are square from above, so
rotations are taken modulo 90 deg. Used to sample the evaluation configurations from the demonstrations'
start frames and to guide (and log) the setup of every evaluation rollout.
"""

import cv2
import numpy as np

RED = [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))]
BLUE = [((100, 120, 50), (130, 255, 255))]
MIN_AREA = {"cube": 150, "tray": 2000}


def detect(bgr: np.ndarray) -> dict:
    """{'cube': (cx, cy, w, h, yaw) or None, 'tray': ...}; yaw in [0, 90) degrees."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = {}
    for name, ranges in (("cube", RED), ("tray", BLUE)):
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = [c for c in cs if cv2.contourArea(c) > MIN_AREA[name]]
        if cs:
            (cx, cy), (w, h), a = cv2.minAreaRect(max(cs, key=cv2.contourArea))
            out[name] = (cx, cy, w, h, a % 90)
        else:
            out[name] = None
    return out


def yaw_error(a: float, b: float) -> float:
    """Smallest difference between two rotations of a square, in [0, 45] degrees."""
    d = abs(a - b) % 90
    return min(d, 90 - d)


def box(cx: float, cy: float, w: float, h: float, yaw: float) -> np.ndarray:
    return np.int32(cv2.boxPoints(((cx, cy), (w, h), yaw)))


def draw(img: np.ndarray, pose, colour, thickness: int = 2, dashed: bool = False) -> None:
    """Outline of a detected or target pose (cx, cy, w, h, yaw); dashed for targets."""
    pts = box(*pose)
    if not dashed:
        cv2.polylines(img, [pts], True, colour, thickness)
        return
    for i in range(4):
        p, q = pts[i].astype(float), pts[(i + 1) % 4].astype(float)
        n = max(2, int(np.linalg.norm(q - p) / 8))
        for k in range(0, n, 2):
            cv2.line(img, tuple(np.int32(p + (q - p) * k / n)), tuple(np.int32(p + (q - p) * (k + 1) / n)), colour, thickness)
