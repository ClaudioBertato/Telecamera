"""roi_exposure.py — sweep ROI / exposure / gain combinations.

Verifies that parameter writes are honoured by the firmware and that the
resulting image stats look reasonable.

Usage::

    python roi_exposure.py
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

# Running this as `python examples/<script>.py` puts examples/ on sys.path, not
# the package root, so make the package importable however it is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import u3v_cam


def main() -> int:
    if not u3v_cam.list_cameras():
        print("No camera found.", file=sys.stderr)
        return 1

    rois: List[Tuple[int, int, int, int]] = [
        (1456, 1088, 0,   0),     # full frame
        ( 800,  600, 328, 244),   # centered crop
        ( 320,  240, 568, 424),
    ]
    exposures = [500, 2000, 10000]   # microseconds
    gains     = [0, 100, 300]

    with u3v_cam.Camera() as cam:
        cam.pixel_format = u3v_cam.PFNC_MONO8

        for w, h, ox, oy in rois:
            cam.set_roi(w, h, ox, oy)
            for exp in exposures:
                for g in gains:
                    cam.exposure_us = exp
                    cam.gain = g
                    cam.start()
                    frame = cam.read_frame()
                    cam.stop()
                    print(f"ROI={w}x{h} @({ox},{oy})  "
                          f"exp={cam.exposure_us}us  gain={cam.gain}  "
                          f"shape={frame.shape}  "
                          f"mean={frame.mean():.1f}  "
                          f"min={frame.min()}  max={frame.max()}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
