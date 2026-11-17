#!/usr/bin/env python
"""Camera names <-> OpenCV indices (macOS), using the shared lookup in common/camera_lookup.py.

    python cameras.py list          # every camera with its OpenCV index
    python cameras.py index NAME    # the index of the one camera whose name contains NAME
"""

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from camera_lookup import list_cameras_macos, resolve_camera  # noqa: E402


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "list":
        for i, name in enumerate(list_cameras_macos()):
            print(f"{i}: {name}")
    elif len(sys.argv) == 3 and sys.argv[1] == "index":
        if not sys.argv[2]:
            sys.exit("No camera set - put its name in A2/config.sh (see ./00_find_cameras.sh).")
        with contextlib.redirect_stdout(sys.stderr):  # keep stdout for the index only
            index = resolve_camera(sys.argv[2])
        print(index)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
