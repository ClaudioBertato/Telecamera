"""color_capture.py — capture a demosaiced color image from a Bayer sensor.

Discover -> open -> select a Bayer format -> enable the ISP color pipeline ->
capture 5 frames -> save the first as a PPM.

With the color pipeline enabled, ``read_frame()`` returns an ``(H, W, 3)``
uint8 RGB image — the same demosaiced output the Qt viewer shows — instead of
the raw single-channel Bayer mosaic. On a mono sensor the pipeline passes
frames through unchanged, so this example still runs (saving a PGM).
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Running this as `python examples/<script>.py` puts examples/ on sys.path, not
# the package root, so make the package importable however it is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import u3v_cam


def save_ppm(path: str, frame: np.ndarray) -> None:
    """Save an (H, W, 3) RGB8 frame as a binary PPM."""
    h, w, _ = frame.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(frame).tobytes())


def save_pgm(path: str, frame: np.ndarray) -> None:
    """Save an (H, W) Mono8 frame as a binary PGM."""
    h, w = frame.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(frame).tobytes())


def main() -> int:
    print("=== U3V Camera SDK - Color Capture (Python) ===\n")

    cams = u3v_cam.list_cameras()
    print(f"Found {len(cams)} U3V camera(s)")
    if not cams:
        print("No camera found. Check USB cable, driver, and power.")
        return 1

    with u3v_cam.Camera(index=0) as cam:
        info = cam.info
        print(f"Connected: {info['manufacturer']} {info['model']} (S/N {info['serial']})")

        cam.set_roi(info["width_max"], info["height_max"])

        # Select an 8-bit Bayer format for color sensors. The ISP pipeline
        # auto-detects the Bayer phase from the frame and demosaics to RGB8.
        try:
            cam.pixel_format = u3v_cam.PFNC_BAYERRG8
        except u3v_cam.U3VError:
            print("Camera did not accept BayerRG8; using its current pixel format.")

        cam.exposure_us = 10000
        cam.gain = 0

        # Turn on the color pipeline. Requires the ISP plugins that ship in
        # the package (u3v_cam/_libs/<platform>/plugins/).
        try:
            n = cam.enable_color()
            print(f"Color pipeline enabled ({n} ISP plugin(s) loaded).")
        except FileNotFoundError as e:
            print(f"Color pipeline unavailable: {e}")
            print("Falling back to raw capture (single-channel Bayer mosaic).")

        print(f"\nConfig: {cam.width}x{cam.height}  exposure={cam.exposure_us} us  "
              f"color={'on' if cam.color_enabled else 'off'}")

        print("\nStarting acquisition...")
        cam.start()

        # One-shot auto white balance. It only applies with the colour pipeline
        # on; the gain stage is off until enabled, which auto_white_balance()
        # does for you. AWB is computed from a processed frame, so grab a few
        # frames to let exposure settle first, then arm it and read one more —
        # the computed red/blue gains apply from that frame on.
        #   Fixed gains instead of auto:  cam.set_white_balance(1.6, 1.0, 1.9)
        if cam.color_enabled:
            for _ in range(5):
                cam.read_frame()
            cam.auto_white_balance()
            cam.read_frame()
            print(f"Auto white balance applied: gains={cam.get_white_balance()}")

        for i in range(5):
            frame = cam.read_frame()
            print(f"Frame {i}: shape={frame.shape}  dtype={frame.dtype}  "
                  f"block_id={cam.last_frame_block_id}  status={cam.last_frame_status}")
            if i == 0:
                if frame.ndim == 3:
                    save_ppm("color_0.ppm", frame)
                    print("  -> Saved RGB image to color_0.ppm")
                elif frame.ndim == 2 and frame.dtype == np.uint8:
                    save_pgm("color_0.pgm", frame)
                    print("  -> Saved mono/raw frame to color_0.pgm")
                else:
                    print(f"  -> Skipped save for unsupported frame {frame.shape} {frame.dtype}")

        print("\nStopping...")
        cam.stop()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
