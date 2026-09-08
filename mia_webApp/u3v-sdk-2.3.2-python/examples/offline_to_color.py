"""offline_to_color.py — turn a captured raw burst into finished images.

Reads a burst written by ``burst_capture_raw.py`` and converts every frame
using the SDK's ISP: demosaic, white balance, colour correction, gamma and
level adjustment. The image adjustments come from a settings file, so every
camera in a system produces the same colour from the same raw data, and a
burst can be re-converted with different settings without capturing it again.

The conversion uses the same ISP as live colour capture, so the result matches
what the camera would have delivered with colour enabled.

A mono burst is handled by the same command: there is no colour to
reconstruct, so the frames are written out as single-channel images.

Usage::

    # convert a burst using the settings recorded when it was captured
    python offline_to_color.py --input burst_0A590D30.npy

    # override the image adjustments
    python offline_to_color.py --input burst_0A590D30.npy \\
                               --profile camera_profile.json

    # first 10 frames only, as PNG
    python offline_to_color.py --input burst_0A590D30.npy --limit 10 --png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Running this as `python examples/<script>.py` puts examples/ on sys.path, not
# the package root, so make the package importable however it is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import u3v_cam

PIXEL_FORMATS = {
    "bayerrg8": u3v_cam.PFNC_BAYERRG8,
    "bayergr8": u3v_cam.PFNC_BAYERGR8,
    "bayergb8": u3v_cam.PFNC_BAYERGB8,
    "bayerbg8": u3v_cam.PFNC_BAYERBG8,
    "mono8":    u3v_cam.PFNC_MONO8,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True,
                   help="Burst file (.npy) written by burst_capture_raw.py")
    p.add_argument("--profile", default=None,
                   help="Settings file with the image adjustments to apply. "
                        "Defaults to the settings recorded with the burst.")
    p.add_argument("--format", default=None, choices=sorted(PIXEL_FORMATS),
                   help="Raw pixel format. Defaults to the recorded one.")
    p.add_argument("--outdir", default=None,
                   help="Output folder (default: alongside the input)")
    p.add_argument("--limit", type=int, default=0,
                   help="Convert only the first N frames (0 = all)")
    p.add_argument("--png", action="store_true",
                   help="Write PNG instead of Netpbm "
                        "(needs opencv-python or Pillow)")
    p.add_argument("--save-profile", default=None,
                   help="Write the settings actually used to this file")
    return p.parse_args()


def load_metadata(raw_path: str) -> dict:
    meta_path = os.path.splitext(raw_path)[0] + ".json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)


def write_netpbm(path: str, image: np.ndarray) -> None:
    """Netpbm needs no third-party library and every image tool reads it.

    P6 for a three-channel image, P5 for the single-channel one a mono burst
    produces.
    """
    height, width = image.shape[:2]
    magic = "P5" if image.ndim == 2 else "P6"
    with open(path, "wb") as fh:
        fh.write(f"{magic}\n{width} {height}\n255\n".encode("ascii"))
        fh.write(np.ascontiguousarray(image).tobytes())


def make_png_writer():
    try:
        import cv2

        def write(path: str, image: np.ndarray) -> None:
            # OpenCV expects BGR; a mono image has no channel order to swap.
            cv2.imwrite(path, image if image.ndim == 2 else image[:, :, ::-1])
        return write
    except ImportError:
        pass
    try:
        from PIL import Image

        def write(path: str, image: np.ndarray) -> None:
            Image.fromarray(image).save(path)
        return write
    except ImportError:
        raise SystemExit("--png needs opencv-python or Pillow installed.")


def main() -> int:
    args = parse_args()

    frames = np.load(args.input)
    if frames.ndim != 3:
        raise SystemExit(f"Expected a (frames, height, width) burst, "
                         f"got shape {frames.shape}")
    metadata = load_metadata(args.input)

    format_name = args.format or metadata.get("pixel_format", "bayerrg8")
    if format_name not in PIXEL_FORMATS:
        raise SystemExit(f"Unknown pixel format {format_name!r}")
    pixel_format = PIXEL_FORMATS[format_name]

    if args.profile:
        settings = u3v_cam.load_profile(args.profile)
    else:
        settings = metadata.get("profile", {})
    adjustments = settings.get("isp", settings) or {}

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    count = len(frames) if args.limit <= 0 else min(args.limit, len(frames))
    write = make_png_writer() if args.png else write_netpbm

    print(f"{args.input}: {len(frames)} frames  "
          f"{frames.shape[2]}x{frames.shape[1]}  {format_name}")
    if metadata.get("serial"):
        print(f"  camera {metadata['serial']}")

    with u3v_cam.Processor() as isp:
        if adjustments:
            skipped = isp.apply_isp_profile(adjustments)
            print(f"  applied {len(adjustments) - len(skipped)} image settings"
                  + (f", skipped {', '.join(skipped)}" if skipped else ""))
        else:
            print("  no image settings supplied; using defaults")

        extension = None
        for i in range(count):
            image = isp.process(frames[i], pixel_format, copy=False)
            if extension is None:
                # A mono burst has no demosaic stage to pass through, so the
                # ISP hands it back two-dimensional. Netpbm spells that P5/.pgm
                # rather than P6/.ppm, so the extension follows the result
                # rather than being assumed up front.
                mono = image.ndim == 2
                extension = ".png" if args.png else (".pgm" if mono else ".ppm")
                print(f"  output {image.shape[1]}x{image.shape[0]} "
                      f"{'Mono' if mono else 'RGB'}")
            path = os.path.join(outdir, f"{stem}_{i:04d}{extension}")
            write(path, image)

        used = isp.get_isp_profile()

    print(f"  wrote {count} images to {outdir}")

    if args.save_profile:
        u3v_cam.save_profile(args.save_profile, {"isp": used})
        print(f"  wrote settings to {args.save_profile}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
