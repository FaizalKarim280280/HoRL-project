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
    h_min: int = 95
    h_max: int = 130
    s_min: int = 100
    v_min: int = 50
    min_area: int = 800  # pixels; the plate is large, so this rejects small blue things

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


def blue_mask(rgb: np.ndarray, s: BlueSettings) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (s.h_min, s.s_min, s.v_min), (s.h_max, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_plate(rgb: np.ndarray, s: BlueSettings) -> tuple[PlateDetection | None, np.ndarray]:
    """Largest blue region and its 4 corners, or None; plus the mask.

    The corners come from simplifying the region's convex outline to a quadrilateral (the
    plate's perspective-distorted square). If that does not give exactly 4 points (e.g. the
    plate is partly out of view), the minimum-area rectangle around it is used instead.
    """
    mask = blue_mask(rgb, s)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < s.min_area:
        return None, mask
    hull = cv2.convexHull(contour)
    quad = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(contour))
    return PlateDetection(quad.astype(np.float64), contour, area), mask
