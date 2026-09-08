"""basic_capture.py — Python counterpart of examples/basic_capture.c.

Discover → open → configure → capture 5 frames → save the first as PGM.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Running this as `python examples/<script>.py` puts examples/ on sys.path, not
# the package root, so make the package importable however it is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import u3v_cam


def save_pgm(path: str, frame: np.ndarray) -> None:
    if frame.ndim != 2 or frame.dtype != np.uint8:
        raise ValueError("PGM requires Mono8 frame")
    h, w = frame.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(frame.tobytes())


def main() -> int:
    print("=== U3V Camera SDK - IMX296 Basic Capture (Python) ===\n")

    cams = u3v_cam.list_cameras()
    print(f"Found {len(cams)} U3V camera(s)")
    if not cams:
        print("No camera found. Check USB cable, driver, and power.")
        return 1
    for c in cams:
        print(f"  [{c['index']}] VID={c['vendor_id']:04X} PID={c['product_id']:04X} "
              f"{c['manufacturer']} {c['model']}  S/N={c['serial']}")

    with u3v_cam.Camera(index=0) as cam:
        info = cam.info
        print(f"\nConnected: {info['manufacturer']} {info['model']} (S/N {info['serial']})")
        print(f"Sensor: {info['sensor_width']} x {info['sensor_height']}")

        cam.width = info["width_max"]
        cam.height = info["height_max"]
        cam.pixel_format = u3v_cam.PFNC_MONO8
        cam.exposure_us = 10000
        cam.gain = 0

        print(f"\nConfig: {cam.width}x{cam.height}  exposure={cam.exposure_us} us  "
              f"payload={cam.payload_size} bytes")

        print("\nStarting acquisition...")
        cam.start()

        for i in range(5):
            frame = cam.read_frame()
            print(f"Frame {i}: {frame.shape}  dtype={frame.dtype}  "
                  f"block_id={cam.last_frame_block_id}  "
                  f"ts={cam.last_frame_timestamp_ns}  "
                  f"status={cam.last_frame_status}")
            if i == 0:
                save_pgm("capture_0.pgm", frame)
                print("  -> Saved to capture_0.pgm")

        print("\nStopping...")
        cam.stop()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
