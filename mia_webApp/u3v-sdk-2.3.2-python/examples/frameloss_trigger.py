#!/usr/bin/env python3
"""Triggered frame-loss test, 1 to 8 cameras streaming together.

What it does
    - Opens the selected cameras (all of them by default, up to 8).
    - Puts each camera in trigger mode - either an external hardware trigger
      on a line input (default: line1, rising edge, frame start), or the
      built-in software trigger.
    - Warms up each camera (captures and discards a few frames) so the first,
      unrepresentative moments after arming are not measured.
    - Streams every camera at the same time for --seconds seconds.
    - Every frame carries a sequence number, so a gap in the sequence is a
      dropped frame. Reports frames captured, frames lost, loss %, achieved
      frame rate, throughput and missed triggers, per camera and per run.

Trigger source
    --source line1     external hardware trigger (default). You supply the
                       pulse train; use a hardware-timed source (PWM / timer /
                       FPGA / PLC) and keep the trigger rate below the frame-
                       rate ceiling for the ROI in use. Tell the script the
                       rate you are generating with --rate so it can report
                       how many triggers produced no frame.
    --source software  the script issues each trigger itself at --rate Hz.
                       Software triggering is a request/response round trip,
                       so the achieved rate is limited by round-trip latency
                       and is well below what the hardware trigger reaches.
                       Use it to verify the camera and USB link without
                       external wiring - NOT to validate frame-rate timing.

Two ways to run it
    --color 1   (default) each frame is converted to RGB while the camera is
                running. This is the heavier of the two paths, so a pass here
                is a conservative result.
    --color 0   raw data only, no conversion. Use this when you plan to
                capture raw and convert afterwards.

    --format selects what the camera sends: mono8 for mono sensors, one of the
    bayer**8 formats for colour sensors. Colour conversion applies to the Bayer
    formats; mono is measured as raw data.

Requirements: numpy.

Examples
    # 1 camera, external 60 Hz trigger on line1, 1024x600, 30 s
    python frameloss_trigger.py --ncams 1 --rate 60 --seconds 30

    # all cameras (up to 8), external 60 Hz trigger, full frame, 3 runs
    python frameloss_trigger.py --rate 60 --roi-w 1456 --roi-h 1088 --runs 3

    # no external trigger available - software trigger at 30 Hz
    python frameloss_trigger.py --source software --rate 30

    # named cameras only, results appended to a file you can send back
    python frameloss_trigger.py --serials A1B2C3,D4E5F6 --rate 60 --csv result.csv

Exit code: 0 = PASS, 2 = NO-DATA, 3 = LOSS.
"""
import argparse
import csv
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

MAX_CAMS = 8

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
        return [(by_serial[s]["index"], s) for s in wanted][:MAX_CAMS]

    n = len(cams) if args.ncams <= 0 else min(args.ncams, len(cams))
    return [(c["index"], c["serial"]) for c in cams[:min(n, MAX_CAMS)]]


# --------------------------------------------------------------------------
# per-camera worker
# --------------------------------------------------------------------------
def open_and_arm(index, a):
    """Open one camera, apply the configuration, arm the trigger and start.

    Returns the camera handle plus the settings worth reporting.
    """
    cam = u3v_cam.Camera(index=index, num_buffers=a.buffers,
                         timeout_ms=a.timeout_ms)
    try:
        info = cam.info
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

        cam.configure_trigger(on=True, source=a.source,
                              activation=a.activation, selector=a.selector)
        cam.start()
    except Exception:
        try:
            cam.close()
        except Exception:
            pass
        raise
    return cam, {
        "serial": info.get("serial"),
        "model": info.get("model"),
        "color_on": color_on,
        "payload": int(cam.payload_size),
        "trigger": f"{a.source}/{a.activation}/{a.selector}",
    }


def warm_up(cam, a, sw, period):
    """Capture and discard frames so the measurement starts from a steady state.

    Returns the number of frames actually discarded. Zero means nothing came
    through, which for an external trigger usually means no pulses are
    arriving - check that the pulse source is running and wired to the camera.
    """
    warm = 0
    deadline = time.time() + a.warmup_timeout
    while warm < a.warmup and time.time() < deadline:
        if sw:
            try:
                cam.software_trigger()
            except Exception:
                break
            if period:
                time.sleep(period)
        try:
            cam.read_frame(copy=False)
            warm += 1
        except Exception:
            pass
    return warm


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


def measure(cam, a, rec, sw, period):
    """Stream for --seconds and collect the statistics."""
    good = gaps = lost = timeouts = saved = 0
    pending = []
    last = None
    bytes_total = 0
    first_shape = None
    t0 = time.time()
    t_end = t0 + a.seconds
    next_due = t0
    while time.time() < t_end:
        if sw:
            # pace the software trigger, then grab the frame it produced
            now = time.time()
            if now < next_due:
                time.sleep(next_due - now)
            next_due += period
            try:
                cam.software_trigger()
            except Exception as e:
                rec.setdefault("trig_errs", []).append(str(e)[:60])
                continue
        try:
            frame = cam.read_frame(copy=False)
        except Exception as e:
            msg = str(e)
            if "Timeout" in msg:
                timeouts += 1
            rec.setdefault("grab_errs", []).append(msg[:60])
            continue
        blk = cam.last_frame_block_id
        if first_shape is None:
            first_shape = getattr(frame, "shape", None)
        if last is not None:
            d = blk - last
            if d != 1:
                gaps += 1
                if d > 1:
                    lost += (d - 1)      # d <= 0 = sequence restart, not a loss
        last = blk
        good += 1
        bytes_total += rec["payload"]

        # Keep a copy in memory and write the files after the run. Writing to
        # disk here would stall the grab loop: with --color 0 the frame is a
        # view of the streaming buffer, and holding the loop during file I/O
        # collapses the stream instead of merely slowing it.
        if a.save and good % a.save == 0 and len(pending) < MAX_SAVED_FRAMES:
            ext = ".pgm" if frame.ndim == 2 else ".ppm"
            pending.append(
                (f"{rec.get('serial', 'cam')}_{good:06d}{ext}", np.array(frame)))
    dt = time.time() - t0

    for name, image in pending:
        try:
            save_frame(os.path.join(a.save_dir, name), image)
            saved += 1
        except Exception as e:
            rec.setdefault("save_errs", []).append(str(e)[:60])

    total_expected = good + lost
    # How many triggers should have arrived in this window, if the trigger rate
    # is known. Only meaningful for an external source running at a fixed rate.
    expected_by_rate = int(a.rate * dt) if a.rate > 0 else 0
    rec.update({
        "shape": first_shape,
        "good": good, "gaps": gaps, "lost": lost, "timeouts": timeouts,
        "saved": saved,
        "loss_pct": (100.0 * lost / total_expected) if total_expected else 0.0,
        "fps": good / dt if dt else 0.0,
        # Decimal MB, matching the throughput figures quoted in the docs.
        "mbs": (bytes_total / dt / 1e6) if dt else 0.0,
        "elapsed": dt,
        "expected_by_rate": expected_by_rate,
        "shortfall": max(0, expected_by_rate - good) if expected_by_rate else 0,
    })


def run_cam(slot, index, serial, a, results, barrier):
    rec = {"slot": slot, "serial": serial, "warmup_frames": 0, "reopened": 0}
    results[slot] = rec
    cam = None
    waited = False

    def sync():
        # Release the other cameras exactly once, whatever happens here, so a
        # failure on one camera can never hang the whole test.
        nonlocal waited
        if not waited:
            waited = True
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass

    sw = (a.source == "software")
    period = (1.0 / a.rate) if (sw and a.rate > 0) else 0.0

    try:
        # Arm and warm up. If the warm-up saw nothing at all, re-open and try
        # again up to --reopen-retries times before giving up on this camera.
        for attempt in range(a.reopen_retries + 1):
            try:
                cam, cfg = open_and_arm(index, a)
            except Exception as e:
                rec["error"] = f"open/configure failed: {e}"
                return
            rec.update(cfg)
            rec["warmup_frames"] = warm_up(cam, a, sw, period)
            if rec["warmup_frames"] > 0 or a.warmup == 0:
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

        sync()                      # all cameras warmed up -> measure together
        measure(cam, a, rec, sw, period)
    except Exception as e:
        rec.setdefault("error", f"unexpected: {e}")
    finally:
        sync()
        if cam is not None:
            try:
                cam.stop()
            except Exception:
                pass
            try:
                cam.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# one measurement run over all cameras
# --------------------------------------------------------------------------
def one_run(a, selected, run_no):
    n = len(selected)
    results = {}
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=run_cam,
                                args=(slot, index, serial, a, results, barrier))
               for slot, (index, serial) in enumerate(selected)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n=============== RUN {run_no} - PER-CAMERA RESULT ===============")
    any_loss = any_empty = any_error = False
    agg_mbs = 0.0
    streaming = 0
    for slot in range(n):
        r = results.get(slot, {})
        if "error" in r:
            print(f"cam{slot} serial={r.get('serial')}: ERROR {r['error']}")
            any_error = True
            continue
        good = r.get("good", 0)
        if good == 0:
            any_empty = True
        else:
            streaming += 1
        if r.get("lost", 0) > 0:
            any_loss = True
        agg_mbs += r.get("mbs", 0.0)
        print(f"cam{slot} serial={r.get('serial')} model={r.get('model')} "
              f"color_on={r.get('color_on')} trigger={r.get('trigger')} "
              f"shape={r.get('shape')}")
        line = (f"      warmup={r.get('warmup_frames', 0)} frames={good} "
                f"lost={r.get('lost')} loss={r.get('loss_pct', 0):.4f}%  "
                f"rate={r.get('fps', 0):.2f} fps  MB/s={r.get('mbs', 0):.2f}  "
                f"timeouts={r.get('timeouts', 0)}  "
                f"elapsed={r.get('elapsed', 0):.2f}s")
        if r.get("saved"):
            line += f"  saved={r['saved']}"
        if r.get("reopened"):
            line += f"  reopened={r['reopened']}"
        print(line)
        if r.get("expected_by_rate"):
            print(f"      triggers expected at {a.rate:g} Hz = "
                  f"{r['expected_by_rate']}, frames received = {good}, "
                  f"shortfall = {r['shortfall']}")
            # More frames than triggers means the camera is not actually
            # gated by the trigger - it is free-running, so the numbers say
            # nothing about triggered operation.
            if good > 1.2 * r["expected_by_rate"]:
                print(f"      WARNING cam{slot}: received far more frames than "
                      f"triggers - the camera does not appear to be in trigger "
                      f"mode. Check the trigger wiring and that no other "
                      f"application reconfigured the camera.")
        if r.get("grab_errs"):
            print(f"      grab events: {dict(Counter(r['grab_errs']))}")
        if r.get("trig_errs"):
            print(f"      trigger events: {dict(Counter(r['trig_errs']))}")
    print(f"  aggregate throughput: {agg_mbs:.2f} MB/s over "
          f"{streaming} streaming camera(s) of {n}")
    print("==========================================================")
    return results, any_loss, any_empty, any_error


def write_csv(path, a, run_no, results, n):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run", "cam", "serial", "model", "source", "rate_hz",
                        "roi_w", "roi_h", "format", "exposure_us", "gain",
                        "color", "seconds", "warmup", "frames", "lost",
                        "loss_pct", "fps", "mbs", "timeouts", "shortfall",
                        "error"])
        for slot in range(n):
            r = results.get(slot, {})
            w.writerow([run_no, slot, r.get("serial", ""), r.get("model", ""),
                        a.source, a.rate, a.roi_w, a.roi_h, a.format,
                        a.exposure, a.gain,
                        a.color, a.seconds, r.get("warmup_frames", 0),
                        r.get("good", 0), r.get("lost", 0),
                        f"{r.get('loss_pct', 0):.4f}", f"{r.get('fps', 0):.2f}",
                        f"{r.get('mbs', 0):.2f}", r.get("timeouts", 0),
                        r.get("shortfall", 0), r.get("error", "")])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Triggered frame-loss test (1-8 cameras)")
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="measurement time per run")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat the measurement N times")
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
    ap.add_argument("--timeout-ms", type=int, default=1000,
                    help="per-frame receive timeout; keep it a few frame "
                         "periods, not seconds, or one missed trigger stalls "
                         "the measurement")
    ap.add_argument("--ncams", type=int, default=0,
                    help="0 = use all cameras found (max 8)")
    ap.add_argument("--serials", default=None,
                    help="comma-separated serial numbers; overrides --ncams")
    ap.add_argument("--source", default="line1",
                    choices=["line0", "line1", "line2", "line3", "software"],
                    help="trigger source (default line1 = external hardware)")
    ap.add_argument("--activation", default="rising", choices=["rising", "falling"])
    ap.add_argument("--selector", default="frame_start",
                    choices=["frame_start", "acq_start"])
    ap.add_argument("--rate", type=float, default=60.0,
                    help="trigger rate in Hz. With --source software the script "
                         "issues triggers at this rate; with a hardware source "
                         "it is the rate you are generating, used to report the "
                         "shortfall")
    ap.add_argument("--warmup", type=int, default=10,
                    help="frames to capture and discard before measuring (0 = none)")
    ap.add_argument("--warmup-timeout", type=float, default=10.0,
                    help="seconds to spend on warm-up at most")
    ap.add_argument("--reopen-retries", type=int, default=1,
                    help="re-open a camera this many times if warm-up gets nothing")
    ap.add_argument("--reopen-delay", type=float, default=1.0,
                    help="seconds to wait before re-opening")
    ap.add_argument("--csv", default=None, help="append results to this CSV file")
    ap.add_argument("--save", type=int, default=0, metavar="N",
                    help="save every Nth frame while measuring (0 = off). "
                         "--color 0 writes the raw frame as .pgm, --color 1 "
                         "writes RGB as .ppm")
    ap.add_argument("--save-dir", default="capture",
                    help="folder for --save output (default: capture)")
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
          f"timeout={a.timeout_ms}ms seconds={a.seconds} runs={a.runs} "
          f"cameras={n}")
    if a.save:
        os.makedirs(a.save_dir, exist_ok=True)
        print(f"SAVE: every {a.save}th frame to {os.path.abspath(a.save_dir)} "
              f"(held in memory during the run, written afterwards, "
              f"at most {MAX_SAVED_FRAMES} per camera)")
    if a.source == "software":
        print(f"TRIGGER: software, {a.rate:g} Hz issued by this script.")
        print("         Software triggering is a request/response round trip - "
              "the achieved")
        print("         rate will be below the requested rate. Use an external "
              "hardware")
        print("         trigger to validate frame-rate timing.")
    else:
        print(f"TRIGGER: external hardware on {a.source}, {a.activation} edge, "
              f"expected {a.rate:g} Hz.")
        print("         Make sure your pulse source is running before the "
              "warm-up starts.")

    worst_loss = worst_empty = worst_error = False
    for run_no in range(1, a.runs + 1):
        results, any_loss, any_empty, any_error = one_run(a, selected, run_no)
        worst_loss = worst_loss or any_loss
        worst_empty = worst_empty or any_empty
        worst_error = worst_error or any_error
        if a.csv:
            write_csv(a.csv, a, run_no, results, n)
        if run_no < a.runs:
            time.sleep(2.0)          # let the link settle between runs

        # Flag a software-trigger run that could not keep up, so the achieved
        # rate is not mistaken for a camera limitation.
        if a.source == "software":
            for slot in range(n):
                fps = results.get(slot, {}).get("fps", 0.0)
                if fps and a.rate and fps < 0.8 * a.rate:
                    print(f"  NOTE cam{slot}: achieved {fps:.1f} fps against a "
                          f"{a.rate:g} Hz software trigger - this is the "
                          f"round-trip limit of software triggering, not a "
                          f"camera or link limit.")

    print()
    if worst_error:
        print("VERDICT: NO-DATA (a camera could not be opened or configured - "
              "check the connection, the driver, and that no other application "
              "is holding the camera)")
        return 2
    if worst_empty:
        if a.source == "software":
            print("VERDICT: NO-DATA (a camera captured 0 frames)")
        else:
            print("VERDICT: NO-DATA (a camera captured 0 frames - check that "
                  "the trigger signal reaches every camera and that the pulse "
                  "source is running)")
        return 2
    if worst_loss:
        print("VERDICT: LOSS")
        return 3
    print("VERDICT: PASS (0 loss on all cameras)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
