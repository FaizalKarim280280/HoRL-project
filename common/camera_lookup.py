"""Camera names <-> OpenCV indices, shared by A1 and A2.

Cameras are always chosen by NAME (e.g. "1080p", "720P") so a script never opens the wrong one (such as the
MacBook webcam) when macOS renumbers them.
"""

import json
import platform
import subprocess
from pathlib import Path


def list_cameras_macos() -> list[str]:
    """Camera names in OpenCV's index order on macOS.

    OpenCV's AVFoundation backend numbers cameras by sorting them on their system unique ID
    (so external USB cameras, "0x...", usually come before the built-in "6C70..." one).
    system_profiler gives the same names and unique IDs, so sorting them the same way
    reproduces OpenCV's indices.
    """
    out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True, check=True)
    cams = json.loads(out.stdout).get("SPCameraDataType", [])
    return [c["_name"] for c in sorted(cams, key=lambda c: c.get("spcamera_unique-id", ""))]


def resolve_camera(spec: str) -> int | str:
    """Camera setting -> OpenCV index_or_path.

    A number is used as-is, an existing file path (e.g. /dev/video0) too; anything else is a
    camera NAME (case-insensitive substring, e.g. "Innomaker"), looked up so the script always
    opens that camera even if macOS renumbers them - never silently a different one.
    """
    if spec.isdigit():
        return int(spec)
    if Path(spec).exists():
        return spec
    if platform.system() != "Darwin":
        raise ValueError(f"Camera name lookup is only implemented on macOS; use an index or /dev/video path, not {spec!r}")
    names = list_cameras_macos()
    matches = [i for i, name in enumerate(names) if spec.lower() in name.lower()]
    if len(matches) != 1:
        found = ", ".join(f"{i}: {n}" for i, n in enumerate(names)) or "none"
        raise RuntimeError(f"Camera {spec!r} matched {len(matches)} cameras (found {found}). Is it plugged in?")
    print(f"Camera {spec!r} -> OpenCV index {matches[0]} ({names[matches[0]]})")
    return matches[0]
