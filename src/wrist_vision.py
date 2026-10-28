"""Wrist-camera geometry for the SO-101: ChArUco board, camera intrinsics, hand-eye result,
and projecting image pixels onto the table plane."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cube_pick_common import CALIB_DIR, CameraSettings, DetectionSettings

INTRINSICS_FILE = CALIB_DIR / "wrist_intrinsics.json"
WRIST_CALIB_FILE = CALIB_DIR / "wrist_calibration.json"


# --- ChArUco board ------------------------------------------------------------------------------


@dataclass
class BoardSpec:
    squares_x: int = 5
    squares_y: int = 7
    square_m: float = 0.030  # measure the printed board and pass the real value!
    marker_m: float = 0.022
    dictionary: str = "DICT_4X4_50"

    def board(self) -> cv2.aruco.CharucoBoard:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_m, self.marker_m, dictionary)


def marker_detector_params() -> cv2.aruco.DetectorParameters:
    """ArUco settings for small markers on a low-resolution wrist camera.

    With 6x6 markers only ~20 px wide, each bit covers ~2.5 px; reading more pixels per bit,
    with a smaller margin and more error correction, decodes far more of them.
    """
    p = cv2.aruco.DetectorParameters()
    p.perspectiveRemovePixelPerCell = 8
    p.perspectiveRemoveIgnoredMarginPerCell = 0.2
    p.errorCorrectionRate = 0.8
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 33, 4
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


class BoardDetector:
    def __init__(self, spec: BoardSpec, upscale: int = 2):
        self.spec = spec
        self.board = spec.board()
        self.upscale = upscale  # detect on an enlarged image: small markers decode much more reliably
        self.detector = cv2.aruco.CharucoDetector(self.board, cv2.aruco.CharucoParameters(), marker_detector_params())

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """ChArUco corners (N,1,2) and ids (N,1), or None if fewer than 4 corners were found."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.upscale > 1:
            gray = cv2.resize(gray, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_CUBIC)
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < 4:
            return None
        # cv2.resize aligns pixel centres, so map back with that offset (0.25 px at 2x)
        return (corners - (self.upscale - 1) / 2) / self.upscale, ids

    def object_image_points(self, corners, ids) -> tuple[np.ndarray, np.ndarray]:
        obj, img = self.board.matchImagePoints(corners, ids)
        return obj.reshape(-1, 3), img.reshape(-1, 2)

    def pose(self, corners, ids, k: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, float] | None:
        """Board pose in the camera frame (4x4) and RMS reprojection error (px)."""
        obj, img = self.object_image_points(corners, ids)
        if len(obj) < 6:
            return None
        ok, rvec, tvec = cv2.solvePnP(obj, img, k, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, dist, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        return to_matrix(rvec, tvec), err

    @staticmethod
    def draw(view: np.ndarray, corners, ids) -> np.ndarray:
        cv2.aruco.drawDetectedCornersCharuco(view, corners, ids, (0, 255, 0))
        return view


def to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3))[0]
    t[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return t


# --- calibration files ----------------------------------------------------------------------------


@dataclass
class Intrinsics:
    k: list
    dist: list
    width: int
    height: int
    rms_px: float
    num_images: int
    board: BoardSpec = field(default_factory=BoardSpec)

    def save(self, path: Path = INTRINSICS_FILE) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = INTRINSICS_FILE) -> "Intrinsics":
        raw = json.loads(path.read_text())
        raw["board"] = BoardSpec(**raw["board"])
        return cls(**raw)

    @property
    def k_mat(self) -> np.ndarray:
        return np.array(self.k, dtype=float)

    @property
    def dist_vec(self) -> np.ndarray:
        return np.array(self.dist, dtype=float)


@dataclass
class WristCalibration:
    camera: CameraSettings
    intrinsics: Intrinsics
    t_gripper_cam: list  # 4x4 camera pose in the gripper_frame_link frame
    scan_q: list  # joint pose (deg) used to scan the table
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    method: str = ""
    consistency_mm: float | None = None
    consistency_deg: float | None = None
    board_z_m: float | None = None  # board height in the base frame (table-plane check)
    num_samples: int = 0

    def save(self, path: Path = WRIST_CALIB_FILE) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = WRIST_CALIB_FILE) -> "WristCalibration":
        raw = json.loads(path.read_text())
        raw["camera"] = CameraSettings(**raw["camera"])
        raw["detection"] = DetectionSettings(**raw["detection"])
        intr = raw["intrinsics"]
        intr["board"] = BoardSpec(**intr["board"])
        raw["intrinsics"] = Intrinsics(**intr)
        return cls(**raw)


def check_distortion(k: np.ndarray, dist: np.ndarray, width: int, height: int) -> tuple[bool, float]:
    """Is the lens model usable over the whole image? Undistorts the image corners and edge
    midpoints and distorts them back; an overfitted model (typically a large k3 fitted from
    views that never reached the corners) fails this by many pixels. Returns (ok, worst px)."""
    w, h = width - 1, height - 1
    pts = np.array([[0, 0], [w, 0], [0, h], [w, h], [w / 2, 0], [w / 2, h], [0, h / 2], [w, h / 2]], dtype=np.float64)
    norm = cv2.undistortPoints(pts.reshape(-1, 1, 2), k, dist).reshape(-1, 2)
    back, _ = cv2.projectPoints(np.column_stack([norm, np.ones(len(norm))]), np.zeros(3), np.zeros(3), k, dist)
    worst = float(np.max(np.linalg.norm(back.reshape(-1, 2) - pts, axis=1)))
    return worst < 0.5, worst


# --- pixels -> table --------------------------------------------------------------------------------


def pixel_to_plane(
    u: float, v: float, k: np.ndarray, dist: np.ndarray, t_base_cam: np.ndarray, plane_z: float
) -> np.ndarray | None:
    """Base-frame point where the camera ray through pixel (u, v) meets the plane z = plane_z."""
    n = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), k, dist).reshape(2)
    ray = t_base_cam[:3, :3] @ np.array([n[0], n[1], 1.0])
    origin = t_base_cam[:3, 3]
    if abs(ray[2]) < 1e-6:
        return None
    s = (plane_z - origin[2]) / ray[2]
    return origin + s * ray if s > 0 else None
