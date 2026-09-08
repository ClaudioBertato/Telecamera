"""trigger_mode.py — software-trigger demo.

Configures the camera for software trigger, fires N triggers,
grabs one frame per trigger, prints inter-trigger timing.

Usage::

    python trigger_mode.py --count 10 --exposure 5000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Running this as `python examples/<script>.py` puts examples/ on sys.path, not
# the package root, so make the package importable however it is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import u3v_cam


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--count", type=int, default=10, help="Number of triggers to send")
    p.add_argument("--exposure", dest="exposure_us", type=int, default=5000)
    p.add_argument("--gain", type=int, default=0)
    p.add_argument("--interval-ms", type=float, default=100.0,
                   help="Wait between triggers (default: 100 ms)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not u3v_cam.list_cameras():
        print("No camera found.", file=sys.stderr)
        return 1

    with u3v_cam.Camera(index=args.index) as cam:
        cam.exposure_us = args.exposure_us
        cam.gain = args.gain

        # Trigger mode must be configured BEFORE start().
        cam.configure_trigger(on=True, source="software",
                              activation="rising", selector="frame_start")
        cam.start()

        print(f"Sending {args.count} software triggers...")
        for i in range(args.count):
            t_send = time.monotonic()
            cam.software_trigger()
            try:
                frame = cam.read_frame()
            except u3v_cam.U3VError as e:
                print(f"  [{i}] grab failed: {e}", file=sys.stderr)
                continue
            t_grab = time.monotonic()
            print(f"  [{i}] {frame.shape} dtype={frame.dtype}  "
                  f"latency={(t_grab - t_send) * 1000:.2f} ms  "
                  f"block_id={cam.last_frame_block_id}")
            time.sleep(args.interval_ms / 1000.0)

        cam.stop()
        # Restore continuous mode for next session
        cam.configure_trigger(on=False)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
