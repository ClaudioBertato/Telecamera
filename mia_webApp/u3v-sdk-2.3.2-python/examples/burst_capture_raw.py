"""burst_capture_raw.py — capture a burst of raw frames into memory.

Acquire N triggered (or free-run) frames as raw sensor data straight into a
preallocated array, then write the burst to disk together with the settings
used. Nothing is converted while the camera is running, so the host does the
least possible work during acquisition — which is what keeps a multi-camera
system inside its bandwidth budget.

Colour is applied afterwards by ``offline_to_color.py``, which reads the same
settings file written here.

Usage::

    # 200 frames, external trigger on line1, 1024x600 colour sensor
    python burst_capture_raw.py --count 200 --roi-w 1024 --roi-h 600

    # apply a prepared settings file, save into ./run1
    python burst_capture_raw.py --profile camera_profile.json --outdir run1

    # free-run instead of waiting for a trigger
    python burst_capture_raw.py --trigger off --count 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Running this as `python examples/<script>.py` puts examples/ on sys.path, not
# the package root, so make the package importable however it is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import u3v_cam

# Raw sensor formats this example captures. Colour sensors deliver a Bayer
# mosaic; the name is recorded so the offline step knows how to convert it.
PIXEL_FORMATS = {
    "bayerrg8": u3v_cam.PFNC_BAYERRG8,
    "bayergr8": u3v_cam.PFNC_BAYERGR8,
    "bayergb8": u3v_cam.PFNC_BAYERGB8,
    "bayerbg8": u3v_cam.PFNC_BAYERBG8,
    "mono8":    u3v_cam.PFNC_MONO8,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", type=int, default=0,
                   help="Camera index (default: 0). Prefer --serial.")
    p.add_argument("--serial", default=None,
                   help="Select the camera by serial number instead of index.")
    p.add_argument("--count", type=int, default=200,
                   help="Frames to capture (default: 200)")
    p.add_argument("--format", default="bayerrg8", choices=sorted(PIXEL_FORMATS),
                   help="Raw pixel format (default: bayerrg8)")
    p.add_argument("--roi-w", type=int, default=None, help="ROI width")
    p.add_argument("--roi-h", type=int, default=None, help="ROI height")
    p.add_argument("--exposure", dest="exposure_us", type=int, default=None,
                   help="Exposure in microseconds")
    p.add_argument("--gain", type=int, default=None, help="Sensor gain")
    p.add_argument("--trigger", default="line1",
                   choices=["line0", "line1", "line2", "line3", "software", "off"],
                   help="Trigger source, or 'off' for free-run (default: line1)")
    p.add_argument("--activation", default="rising", choices=["rising", "falling"])
    p.add_argument("--rate", type=float, default=60.0,
                   help="Software-trigger rate in Hz (default: 60)")
    p.add_argument("--profile", default=None,
                   help="Settings file to apply before capturing")
    p.add_argument("--buffers", type=int, default=64,
                   help="Stream buffers (default: 64). Raise for long bursts.")
    p.add_argument("--timeout-ms", type=int, default=2000,
                   help="Per-frame receive timeout (default: 2000)")
    p.add_argument("--warmup", type=int, default=5,
                   help="Frames to discard before the burst (default: 5)")
    p.add_argument("--outdir", default=".", help="Where to write the burst")
    p.add_argument("--no-save", action="store_true",
                   help="Capture and report only; write nothing")
    return p.parse_args()


def pick_camera(args: argparse.Namespace) -> int:
    """Resolve the camera index. Serial numbers are stable across replugs;
    indices are not, so prefer --serial when more than one camera is present."""
    cameras = u3v_cam.list_cameras()
    if not cameras:
        raise SystemExit("No camera found.")
    if args.serial is None:
        return args.index
    for cam in cameras:
        if cam["serial"] == args.serial:
            return cam["index"]
    found = ", ".join(c["serial"] for c in cameras)
    raise SystemExit(f"Serial {args.serial!r} not found. Visible: {found}")


def configure(cam: u3v_cam.Camera, args: argparse.Namespace) -> None:
    cam.pixel_format = PIXEL_FORMATS[args.format]
    if args.profile:
        skipped = cam.apply_profile(u3v_cam.load_profile(args.profile))
        if skipped:
            print(f"  note: settings not applied here: {', '.join(skipped)}")
    if args.roi_w and args.roi_h:
        cam.set_roi(args.roi_w, args.roi_h)
    if args.exposure_us is not None:
        cam.exposure_us = args.exposure_us
    if args.gain is not None:
        cam.gain = args.gain

    # Trigger must be configured before start().
    if args.trigger == "off":
        cam.configure_trigger(on=False)
    else:
        cam.configure_trigger(on=True, source=args.trigger,
                              activation=args.activation,
                              selector="frame_start")


def main() -> int:
    args = parse_args()
    index = pick_camera(args)

    with u3v_cam.Camera(index=index, num_buffers=args.buffers,
                        timeout_ms=args.timeout_ms) as cam:
        info = cam.info
        print(f"Camera {info['serial']}  {info['model']}")
        configure(cam, args)

        width, height = cam.width, cam.height
        software = args.trigger == "software"
        period = 1.0 / args.rate if args.rate > 0 else 0.0

        print(f"  {width}x{height}  {args.format}  "
              f"exposure={cam.exposure_us} us  gain={cam.gain}")
        print(f"  trigger: {args.trigger}")

        # One allocation for the whole burst. read_frame_into() writes each
        # frame directly into its slice, so no per-frame array is created.
        frames = np.empty((args.count, height, width), dtype=np.uint8)
        print(f"  buffer: {frames.nbytes / 1e6:.0f} MB for {args.count} frames")

        cam.start()

        if args.trigger != "off":
            print(f"  waiting for triggers at {args.rate:g} Hz "
                  f"(start your pulse source now if it is not running)")

        for _ in range(args.warmup):
            try:
                if software:
                    cam.software_trigger()
                cam.read_frame(copy=False)
            except u3v_cam.U3VError:
                pass

        received = incomplete = timeouts = 0
        lost = 0
        previous_id = None
        t0 = time.monotonic()

        while received < args.count:
            if software:
                cam.software_trigger()
            try:
                cam.read_frame_into(frames[received])
            except u3v_cam.U3VIncompleteFrame:
                incomplete += 1
                continue
            except u3v_cam.U3VError:
                timeouts += 1
                if timeouts > args.count:
                    print("  giving up: no frames are arriving", file=sys.stderr)
                    break
                continue

            block_id = cam.last_frame_block_id
            if previous_id is not None:
                gap = block_id - previous_id
                if gap > 1:
                    lost += gap - 1
            previous_id = block_id
            received += 1

            if software and period:
                time.sleep(period)

        elapsed = time.monotonic() - t0
        cam.stop()

        frame_bytes = width * height
        rate = received / elapsed if elapsed > 0 else 0.0
        print(f"\n  captured {received}/{args.count} frames in {elapsed:.2f} s")
        print(f"  {rate:.1f} fps   {rate * frame_bytes / 1e6:.1f} MB/s")
        print(f"  dropped by the camera link: {lost}")
        print(f"  incomplete: {incomplete}   timeouts: {timeouts}")

        if args.no_save or received == 0:
            return 0 if received else 2

        os.makedirs(args.outdir, exist_ok=True)
        stem = f"burst_{info['serial']}"
        raw_path = os.path.join(args.outdir, stem + ".npy")
        meta_path = os.path.join(args.outdir, stem + ".json")

        np.save(raw_path, frames[:received])
        metadata = {
            "serial":       info["serial"],
            "model":        info["model"],
            "width":        width,
            "height":       height,
            "frames":       received,
            "pixel_format": args.format,
            "exposure_us":  cam.exposure_us,
            "gain":         cam.gain,
            "trigger":      args.trigger,
            "profile":      cam.get_profile(),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
            fh.write("\n")

        print(f"\n  wrote {raw_path}  ({os.path.getsize(raw_path) / 1e6:.0f} MB)")
        print(f"  wrote {meta_path}")
        print(f"\n  convert to colour with:\n"
              f"    python offline_to_color.py --input {raw_path}")

    return 0 if lost == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
