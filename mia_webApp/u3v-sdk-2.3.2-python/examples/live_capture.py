"""live_capture.py — continuous capture with FPS / dropped-frame statistics.

Counterpart to the customer's ``imx296_control.live`` script.  Runs in headless
mode by default (no GUI); pass ``--show`` to open an OpenCV preview window.

Usage::

    python live_capture.py --width 1456 --height 1088 --fps 60 --headless
    python live_capture.py --show --exposure 5000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

# Running this as `python examples/<script>.py` puts examples/ on sys.path, not
# the package root, so make the package importable however it is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import u3v_cam


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live capture from a USB3 Vision IMX296 camera.")
    p.add_argument("--index", type=int, default=0, help="Camera index from discovery (default: 0)")
    p.add_argument("--width", type=int, default=None, help="ROI width (default: sensor max)")
    p.add_argument("--height", type=int, default=None, help="ROI height (default: sensor max)")
    p.add_argument("--offset-x", type=int, default=0)
    p.add_argument("--offset-y", type=int, default=0)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--exposure", dest="exposure_us", type=int, default=1000,
                   help="Exposure time in microseconds (default: 1000)")
    p.add_argument("--gain", type=int, default=0)
    p.add_argument("--frames", type=int, default=0,
                   help="Stop after N frames (0 = run until Ctrl+C)")
    p.add_argument("--report-interval", type=float, default=1.0,
                   help="Print stats every N seconds (default: 1.0)")
    p.add_argument("--show", action="store_true", help="Open an OpenCV preview window")
    p.add_argument("--headless", action="store_true",
                   help="(default) headless mode, no GUI")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cams = u3v_cam.list_cameras()
    if not cams:
        print("No camera found.", file=sys.stderr)
        return 1
    print(f"Camera: {cams[args.index]}")

    cv2 = None
    if args.show:
        try:
            import cv2 as _cv2  # noqa
            cv2 = _cv2
        except ImportError:
            print("[WARN] --show requested but opencv-python not installed; "
                  "running headless.", file=sys.stderr)

    with u3v_cam.Camera(index=args.index, num_buffers=4, timeout_ms=2000) as cam:
        info = cam.info
        w = args.width  or info["width_max"]
        h = args.height or info["height_max"]
        cam.set_roi(w, h, args.offset_x, args.offset_y)
        cam.pixel_format = u3v_cam.PFNC_MONO8

        cam.start()

        cam.exposure_us = args.exposure_us
        cam.gain = args.gain
        try:
            cam.frame_rate = args.fps
        except u3v_cam.U3VError:
            pass  # not all firmwares accept set_frame_rate

        print("Readback:")
        print(f"  width:        {cam.width}")
        print(f"  height:       {cam.height}")
        print(f"  offset_x:     {cam.offset_x}")
        print(f"  offset_y:     {cam.offset_y}")
        print(f"  frame_rate:   {cam.frame_rate}")
        print(f"  exposure:     {cam.exposure_us}")
        print(f"  gain:         {cam.gain}")
        print(f"  payload_size: {cam.payload_size}")

        good = 0
        dropped = 0
        last_block: Optional[int] = None
        t0 = time.monotonic()
        t_last = t0

        try:
            while True:
                try:
                    frame = cam.read_frame(copy=False)
                except u3v_cam.U3VError as e:
                    print(f"[grab] {e}", file=sys.stderr)
                    dropped += 1
                    continue

                bid = cam.last_frame_block_id
                if last_block is not None and bid > last_block + 1:
                    dropped += bid - last_block - 1
                last_block = bid

                if cam.last_frame_status != 0:
                    dropped += 1
                else:
                    good += 1

                if cv2 is not None:
                    cv2.imshow("u3v_cam", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                now = time.monotonic()
                if now - t_last >= args.report_interval:
                    elapsed = now - t0
                    fps_avg = good / elapsed if elapsed > 0 else 0.0
                    total = good + dropped
                    drop_pct = (dropped / total * 100.0) if total else 0.0
                    if frame.size:
                        mean = float(frame.mean())
                        mn   = int(frame.min())
                        mx   = int(frame.max())
                    else:
                        mean = mn = mx = 0
                    print(f"{cam.width}x{cam.height} | good={good} | "
                          f"{fps_avg:.1f} FPS | dropped={dropped} ({drop_pct:.1f}%) | "
                          f"exp={cam.exposure_us}us gain={cam.gain} "
                          f"mean={mean:.1f} min={mn} max={mx}",
                          flush=True)
                    t_last = now

                if args.frames and good >= args.frames:
                    break
        except KeyboardInterrupt:
            print("\n[interrupt] stopping...")
        finally:
            if cv2 is not None:
                cv2.destroyAllWindows()

        print(f"\nTotal: good={good}  dropped={dropped}  "
              f"elapsed={time.monotonic() - t0:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
