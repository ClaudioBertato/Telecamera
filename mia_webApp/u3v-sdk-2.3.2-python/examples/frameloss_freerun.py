#!/usr/bin/env python3
"""Free-run frame-loss test, one or more cameras streaming together.

What it does
    - Opens the selected cameras (all of them by default).
    - Puts each camera in free-run: continuous acquisition, no trigger.
    - Streams every camera at the same time for --seconds seconds.
    - Every frame carries a sequence number, so a gap in the sequence is a
      dropped frame. Reports frames captured, frames lost, loss %, achieved
      frame rate and throughput, per camera.

Two ways to run it
    --color 1   (default) each frame is converted to RGB while the camera is
                running. This is the heavier of the two paths, so a pass here
                is a conservative result.
    --color 0   raw data only, no conversion. Use this when you plan to
                capture raw and convert afterwards, which is the recommended
                workflow for many cameras at full resolution.

    --format selects what the camera sends: mono8 for mono sensors, one of the
    bayer**8 formats for colour sensors. Colour conversion applies to the Bayer
    formats; mono is measured as raw data.

Requirements: numpy.

Examples
    python frameloss_freerun.py                          # all cameras, 1024x600, 20 s
    python frameloss_freerun.py --roi-w 1456 --roi-h 1088
    python frameloss_freerun.py --seconds 60 --color 0
    python frameloss_freerun.py --format mono8
    python frameloss_freerun.py --serials A1B2C3,D4E5F6

Exit code: 0 = PASS, 2 = NO-DATA, 3 = LOSS.
"""
import argparse
import os
import sys
import threading
import time
from collections import Counter

import numpy as np

# Running this as `python examples/<script>.py` puts examples/ on sys.path, not
# the package root, so make the package importable however it is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import u3v_cam

PIXEL_FORMATS = {
    "mono8":    u3v_cam.PFNC_MONO8,
    "bayerrg8": u3v_cam.PFNC_BAYERRG8,
    "bayergr8": u3v_cam.PFNC_BAYERGR8,
    "bayergb8": u3v_cam.PFNC_BAYERGB8,
    "bayerbg8": u3v_cam.PFNC_BAYERBG8,
}

# Colour conversion applies to Bayer input; mono streams as raw data.
COLOR_CAPABLE = {"bayerrg8", "bayergr8", "bayergb8", "bayerbg8"}


def select_cameras(args):
    """Return the [(index, serial)] to test.

    Enumeration indices shift with plug order, so --serials is the reliable
    way to name a fixed set of cameras; --ncams is a convenience for "use
    whatever is connected".
    """
    cams = u3v_cam.list_cameras()
    print(f"cameras found = {len(cams)}")
    for c in cams:
        print("  ", c)

    if args.serials:
        wanted = [s.strip() for s in args.serials.split(",") if s.strip()]
        by_serial = {c["serial"]: c for c in cams}
        missing = [s for s in wanted if s not in by_serial]
        if missing:
            visible = ", ".join(c["serial"] for c in cams) or "(none)"
            raise SystemExit(f"serial(s) not found: {', '.join(missing)}. "
                             f"Visible: {visible}")
        return [(by_serial[s]["index"], s) for s in wanted]

    n = len(cams) if args.ncams <= 0 else min(args.ncams, len(cams))
    return [(c["index"], c["serial"]) for c in cams[:n]]


# Sampled frames are held in memory until the run ends, so cap how many.
MAX_SAVED_FRAMES = 200


def save_frame(path, frame):
    """Write one frame as Netpbm: P5 for the single-channel frame produced with
    --color 0 (mono or the raw Bayer mosaic), P6 for the RGB frame produced with
    --color 1. Needs no third-party library and every image tool reads it."""
    height, width = frame.shape[:2]
    magic = "P5" if frame.ndim == 2 else "P6"
    with open(path, "wb") as fh:
        fh.write(f"{magic}\n{width} {height}\n255\n".encode("ascii"))
        fh.write(np.ascontiguousarray(frame).tobytes())


def run_cam(slot, index, serial, a, results, barrier):
    rec = {"slot": slot, "serial": serial, "warmup_frames": 0, "reopened": 0}
    results[slot] = rec
    cam = None
    try:
        # Open, configure and warm up. A camera that has just been reconfigured
        # - a new ROI, or a switch out of trigger mode - can need one
        # acquisition cycle to settle and deliver nothing at all on the first
        # attempt. Re-open and try again rather than reporting an empty run.
        for attempt in range(a.reopen_retries + 1):
            cam = u3v_cam.Camera(index=index, num_buffers=a.buffers,
                                 timeout_ms=5000)
            info = cam.info
            rec["serial"] = info.get("serial", serial)
            rec["model"] = info.get("model")

            cam.set_roi(a.roi_w, a.roi_h, 0, 0)
            cam.pixel_format = PIXEL_FORMATS[a.format]
            cam.frame_rate = a.fps
            cam.exposure_us = a.exposure
            cam.gain = a.gain
            cam.set_roi(a.roi_w, a.roi_h, 0, 0)

            color_on = False
            if a.color:
                cam.enable_color()
                cam.auto_white_balance()
                color_on = True
            rec["color_on"] = color_on
            rec["payload"] = int(cam.payload_size)

            cam.configure_trigger(on=False)   # free-run, no trigger
            cam.start()

            # Discarding these also keeps the settling transient out of the
            # measurement, which would otherwise be counted as lost frames.
            warm = 0
            for _ in range(a.warmup):
                try:
                    cam.read_frame(copy=False)
                    warm += 1
                except Exception:
                    pass
            rec["warmup_frames"] = warm
            if warm > 0 or a.warmup == 0:
                break
            if attempt < a.reopen_retries:
                rec["reopened"] += 1
                try:
                    cam.stop()
                    cam.close()
                except Exception:
                    pass
                cam = None
                time.sleep(a.reopen_delay)
    except Exception as e:
        rec["error"] = f"open/configure failed: {e}"
        try:
            if cam is not None:
                cam.close()
        except Exception:
            pass
        barrier.wait()
        return

    # Wait for every camera to be configured and started, then all begin
    # measuring together (fair concurrent-bandwidth window).
    barrier.wait()

    good = gaps = lost = saved = 0
    pending = []
    last = None
    bytes_total = 0
    first_shape = None
    t0 = time.time()
    t_end = t0 + a.seconds
    while time.time() < t_end:
        try:
            frame = cam.read_frame(copy=False)
        except Exception as e:
            rec.setdefault("grab_errs", []).append(str(e)[:60])
            continue
        blk = cam.last_frame_block_id
        if first_shape is None:
            first_shape = getattr(frame, "shape", None)
        if last is not None:
            d = blk - last
            if d != 1:
                gaps += 1
                if d > 1:
                    lost += (d - 1)
        last = blk
        good += 1
        bytes_total += rec["payload"]

        # Keep a copy in memory and write the files after the run. Writing to
        # disk here would stall the grab loop: with --color 0 the frame is a
        # view of the streaming buffer, and holding the loop during file I/O
        # collapses the stream instead of merely slowing it.
        if a.save and good % a.save == 0 and len(pending) < MAX_SAVED_FRAMES:
            ext = ".pgm" if frame.ndim == 2 else ".ppm"
            pending.append((f"{serial}_{good:06d}{ext}", np.array(frame)))
    dt = time.time() - t0
    try:
        cam.stop()
        cam.close()
    except Exception:
        pass

    for name, image in pending:
        try:
            save_frame(os.path.join(a.save_dir, name), image)
            saved += 1
        except Exception as e:
            rec.setdefault("save_errs", []).append(str(e)[:60])

    total_expected = good + lost
    rec.update({
        "shape": first_shape,
        "good": good, "gaps": gaps, "lost": lost, "saved": saved,
        "loss_pct": (100.0 * lost / total_expected) if total_expected else 0.0,
        "fps": good / dt if dt else 0.0,
        # Decimal MB, matching the throughput figures quoted in the docs.
        "mbs": (bytes_total / dt / 1e6) if dt else 0.0,
        "elapsed": dt,
    })


def main():
    ap = argparse.ArgumentParser(description="Free-run frame-loss test")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="measurement time per run")
    ap.add_argument("--roi-w", type=int, default=1024)
    ap.add_argument("--roi-h", type=int, default=600)
    ap.add_argument("--exposure", type=int, default=10000,
                    help="exposure, microseconds")
    ap.add_argument("--gain", type=int, default=16)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--format", default="bayerrg8", choices=sorted(PIXEL_FORMATS),
                    help="pixel format (default: bayerrg8)")
    ap.add_argument("--color", type=int, default=1,
                    help="1 = convert to RGB while streaming, 0 = raw data only")
    ap.add_argument("--buffers", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5,
                    help="frames to discard after starting, before measuring "
                         "(default: 5)")
    ap.add_argument("--save", type=int, default=0, metavar="N",
                    help="save every Nth frame while measuring (0 = off). "
                         "--color 0 writes the raw frame as .pgm, --color 1 "
                         "writes RGB as .ppm")
    ap.add_argument("--save-dir", default="capture",
                    help="folder for --save output (default: capture)")
    ap.add_argument("--reopen-retries", type=int, default=1,
                    help="re-open and retry this many times if the warm-up "
                         "receives nothing at all (default: 1)")
    ap.add_argument("--reopen-delay", type=float, default=1.0,
                    help="seconds to wait before re-opening (default: 1.0)")
    ap.add_argument("--ncams", type=int, default=0,
                    help="0 = use all cameras found")
    ap.add_argument("--serials", default=None,
                    help="comma-separated serial numbers; overrides --ncams")
    a = ap.parse_args()

    if a.color and a.format not in COLOR_CAPABLE:
        print(f"NOTE: {a.format} is not a Bayer format, so no colour "
              f"conversion is applied; measuring the raw path instead.")
        a.color = 0

    selected = select_cameras(a)
    n = len(selected)
    if n == 0:
        print("NO CAMERA")
        return 2

    print(f"ROI={a.roi_w}x{a.roi_h} format={a.format} exposure={a.exposure}us "
          f"gain={a.gain} fps={a.fps} color={a.color} buffers={a.buffers} "
          f"seconds={a.seconds} cameras={n}")

    if a.save:
        os.makedirs(a.save_dir, exist_ok=True)
        print(f"saving every {a.save}th frame to {os.path.abspath(a.save_dir)} "
              f"(held in memory during the run, written afterwards, "
              f"at most {MAX_SAVED_FRAMES} per camera)")

    results = {}
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=run_cam,
                                args=(slot, index, serial, a, results, barrier))
               for slot, (index, serial) in enumerate(selected)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n================ PER-CAMERA RESULT ================")
    any_loss = any_empty = False
    agg_mbs = 0.0
    for slot in range(n):
        r = results.get(slot, {})
        if "error" in r:
            print(f"cam{slot} serial={r.get('serial')}: ERROR {r['error']}")
            any_empty = True
            continue
        good = r.get("good", 0)
        if good == 0:
            any_empty = True
        if r.get("lost", 0) > 0:
            any_loss = True
        agg_mbs += r.get("mbs", 0.0)
        print(f"cam{slot} serial={r.get('serial')} model={r.get('model')} "
              f"color_on={r.get('color_on')} shape={r.get('shape')}")
        print(f"      frames={good} lost={r.get('lost')} "
              f"loss={r.get('loss_pct', 0):.4f}%  "
              f"fps={r.get('fps', 0):.2f}  MB/s={r.get('mbs', 0):.2f}  "
              f"elapsed={r.get('elapsed', 0):.2f}s"
              + (f"  saved={r['saved']}" if r.get('saved') else "")
              + (f"  reopened={r['reopened']}" if r.get('reopened') else ""))
        if r.get("grab_errs"):
            print(f"      incomplete/timeout events: "
                  f"{dict(Counter(r['grab_errs']))}")
    print(f"  aggregate throughput: {agg_mbs:.2f} MB/s over {n} camera(s)")
    print("===================================================")
    if any_empty:
        print("VERDICT: NO-DATA (a camera captured 0 frames)")
        return 2
    if any_loss:
        print("VERDICT: LOSS")
        return 3
    print("VERDICT: PASS (0 loss on all cameras)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
