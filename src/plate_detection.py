"""Blue plate detection: colour threshold, outline and its 4 corners in the image."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from cube_pick_common import CALIB_DIR

PLATE_SETTINGS_FILE = CALIB_DIR / "blue_detection.json"


@dataclass
class BlueSettings:
    # OpenCV hue runs 0..180; blue is roughly 100..130.
    # Defaults measured on the dark-blue tray: floor H 111-118, S 163-190, V down to ~40 in shadow;
    # the dark table has S < 70, so saturation separates them (S >= 120 already leaks).
    h_min: int = 100
    h_max: int = 128
    s_min: int = 140
    v_min: int = 10
    min_area: int = 3000  # pixels; the tray is large, so this rejects small blue things

    def save(self, path: Path = PLATE_SETTINGS_FILE) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = PLATE_SETTINGS_FILE) -> "BlueSettings":
        return cls(**json.loads(path.read_text())) if path.exists() else cls()


@dataclass
class PlateDetection:
    corners: np.ndarray  # (4, 2) pixel corners of the plate outline
    contour: np.ndarray
    area: float


def plate_centre(outline_px: np.ndarray, to_table, size: float, angle_tol_deg: float = 15.0, length_tol: float = 0.15):
    """Centre (x, y), yaw and method of a square plate from its outline, robust to a hidden part.

    `to_table(u, v)` projects a pixel onto the plate's top surface (base-frame metres). The
    outline is simplified to a polygon whose vertices are projected onto the table. Real plate
    corners are vertices with a ~90 deg angle; points where something (e.g. the gripper) cuts
    the outline make other angles and are ignored. Then:
      - 4 corners: centre = their mean
      - 2 opposite corners (~ size * sqrt(2) apart): centre = midpoint of that diagonal
      - 2 adjacent corners (~ size apart): centre = that edge's midpoint, moved size/2 inward
    Returns None if no corner pattern fits.
    """
    hull = cv2.convexHull(outline_px.astype(np.float32))
    poly = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    pts = [to_table(u, v) for u, v in poly]
    if len(poly) < 3 or any(p is None for p in pts):
        return None
    pts = np.array(pts, dtype=float)[:, :2]
    n = len(pts)
    corners = []
    for i in range(n):
        a, b = pts[i - 1] - pts[i], pts[(i + 1) % n] - pts[i]
        angle = np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1)))
        if abs(angle - 90) <= angle_tol_deg:
            corners.append(pts[i])
    corners = np.array(corners)

    def near(d, target):
        return abs(d - target) <= length_tol * target

    if len(corners) >= 4:
        # use the 4 whose pairwise distances best match a square
        c = corners[:4] if len(corners) == 4 else corners[np.argsort(np.linalg.norm(corners - corners.mean(0), axis=1))[-4:]]
        centre, method = c.mean(axis=0), "4 corners"
        edge = c[1] - c[0]
        return centre, np.arctan2(edge[1], edge[0]), method
    pairs = [(np.linalg.norm(corners[i] - corners[j]), i, j) for i in range(len(corners)) for j in range(i + 1, len(corners))]
    diag = [p for p in pairs if near(p[0], size * np.sqrt(2))]
    if diag:
        _, i, j = min(diag, key=lambda p: abs(p[0] - size * np.sqrt(2)))
        d = corners[j] - corners[i]
        return (corners[i] + corners[j]) / 2, np.arctan2(d[1], d[0]) - np.pi / 4, "diagonal"
    sides = [p for p in pairs if near(p[0], size)]
    if sides:
        _, i, j = min(sides, key=lambda p: abs(p[0] - size))
        mid, d = (corners[i] + corners[j]) / 2, corners[j] - corners[i]
        normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        if normal @ (pts.mean(axis=0) - mid) < 0:  # point into the plate
            normal = -normal
        return mid + normal * size / 2, np.arctan2(d[1], d[0]), "one edge"
    return None


def blue_mask(rgb: np.ndarray, s: BlueSettings) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (s.h_min, s.s_min, s.v_min), (s.h_max, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_plate(rgb: np.ndarray, s: BlueSettings, accept=None) -> tuple[PlateDetection | None, np.ndarray]:
    """Largest blue region that `accept(contour)` agrees is the tray, and its 4 corners, or None;
    plus the mask.

    The corners come from simplifying the region's convex outline to a quadrilateral (the
    plate's perspective-distorted square). If that does not give exactly 4 points (e.g. the
    plate is partly out of view), the minimum-area rectangle around it is used instead.
    """
    mask = blue_mask(rgb, s)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in sorted(contours, key=cv2.contourArea, reverse=True) if cv2.contourArea(c) >= s.min_area]
    contour = next((c for c in contours if accept is None or accept(c)), None)
    if contour is None:
        return None, mask
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    quad = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(contour))
    return PlateDetection(quad.astype(np.float64), contour, area), mask
