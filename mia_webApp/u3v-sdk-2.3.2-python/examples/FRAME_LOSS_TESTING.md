# Frame-Loss Testing Guide

Two test tools ship with this package. They stream from one or more cameras at
a resolution and frame rate you choose — free-running or under your external
trigger — and report, per camera, how many frames arrived, how many were lost,
and at what data rate.

| Tool | Test |
|---|---|
| `frameloss_freerun.py` | The camera free-runs at its own frame rate. |
| `frameloss_trigger.py` | Every frame is produced by your trigger pulse. |

Both accept the same `--color`, `--format` and ROI options, so the same
configuration can be measured in both modes and compared.

Loss is counted from the image sequence number: any gap in the sequence is a
dropped frame, and any frame that arrives incomplete is counted as not usable.
No image is written to disk, so storage speed does not affect the result.

---

## 1. Requirements

- Python 3.8 or newer with `numpy` installed.
- One or more cameras on a USB3 port.
- **Windows:** the camera's USB3 Vision interface must be bound to the WinUSB
  driver. If another vendor's vision software has installed its own driver, this
  tool cannot open the camera.
- **Linux:** permission to access the device (a udev rule, or run as root).
- For trigger tests: a pulse source wired to the camera's trigger line, common
  to every camera under test.

Install dependencies and confirm the camera is visible first:

```
install_deps.bat          (Windows)     ./install_deps.sh   (Linux/macOS)
python examples/basic_capture.py
```

---

## 2. The two test conditions: `--color 0` and `--color 1`

This is the main comparison the tools are built for. **Run both.**

| | What the host does per frame | Use it to measure |
|---|---|---|
| `--color 0` | One memory copy. Nothing else. | The link on its own — the lightest possible host load. |
| `--color 1` | Converts the frame to RGB while the camera streams. | The heavier case, where image processing runs during acquisition. |

`--color 1` does **not** change how much data crosses the cable — the camera
sends the same raw frame either way. It changes only how much work the host
does while receiving. So:

- **Loss in both conditions** points at the link, the topology or the trigger.
- **Loss only with `--color 1`** points at host processing load during
  acquisition.

Expect the gap between the two conditions to be far wider under an external
trigger than in free-run. Free-running, the camera sets its own pace and the
buffer queue absorbs variation in how long the host takes per frame. Under a
trigger the pace is imposed from outside: if the host is still working on one
frame when the next arrives, there is no slack to absorb it. A configuration
that looks healthy free-running with `--color 1` can still lose frames heavily
once the same rate is trigger-driven, so **always confirm with the trigger
tool** rather than extrapolating from free-run.

If `--color 1` loses frames and `--color 0` does not, capturing raw and
converting to colour after acquisition removes the loss without changing
anything about the camera or the link. `burst_capture_raw.py` and
`offline_to_color.py` in this folder are a worked example of that split.

`--color 1` is the default. Add `--color 0` explicitly for the light case.

---

## 3. Pixel formats

`--format` accepts:

```
mono8      bayerrg8   bayergr8   bayergb8   bayerbg8
```

Use `mono8` on a monochrome camera and the Bayer order that matches your colour
camera. Colour conversion applies to the Bayer formats only; on `mono8` there is
no colour to reconstruct, so `--color` is reported as off and the raw path is
measured.

The camera delivers 8-bit formats. Colour is produced on the PC.

---

## 4. Test 1 — free-run

```
python examples/frameloss_freerun.py --roi-w 1456 --roi-h 1088 --seconds 30
python examples/frameloss_freerun.py --roi-w 1456 --roi-h 1088 --seconds 30 --color 0
```

Every camera found streams at the same time, and all of them begin measuring
together, so the reported aggregate is a genuine simultaneous load.

| Option | Default | Meaning |
|---|---|---|
| `--seconds` | 20 | Measurement time. |
| `--roi-w`, `--roi-h` | 1024 × 600 | Region of interest. |
| `--exposure` | 10000 | Exposure in microseconds. |
| `--gain` | 16 | Sensor gain. |
| `--fps` | 60 | Requested frame rate. |
| `--format` | `bayerrg8` | See section 3. |
| `--color` | 1 | See section 2. |
| `--buffers` | 64 | Stream buffers per camera. |
| `--warmup` | 5 | Frames discarded before measuring. |
| `--ncams` | 0 (all) | Number of cameras to use. |
| `--serials` | — | Comma-separated serials to use instead. |
| `--save` | 0 (off) | Save every Nth frame, written after the run. See section 8. |
| `--save-dir` | `capture` | Folder for `--save` output. |

---

## 5. Test 2 — external hardware trigger

```
python examples/frameloss_trigger.py --rate 56 --roi-w 1456 --roi-h 1088 ^
                                     --runs 3 --csv result.csv
```

Start your pulse source before starting the test.

| Option | Default | Meaning |
|---|---|---|
| `--source` | `line1` | Trigger input line. |
| `--activation` | `rising` | Trigger edge. |
| `--rate` | 60 | The rate your pulse source is running at, used to compute the expected frame count. |
| `--runs` | 1 | Repeat the whole measurement N times. |
| `--seconds` | 30 | Measurement time per run. |
| `--warmup` | 10 | Frames discarded before measuring. |
| `--csv` | — | Append every run to a CSV file. |
| `--timeout-ms` | 1000 | Per-frame receive timeout. |
| `--save` | 0 (off) | Save every Nth frame, written after the run. See section 8. |
| `--save-dir` | `capture` | Folder for `--save` output. |

All other options match the free-run tool.

If the trigger is not wired yet, `--source software` exercises the camera and
the link without it. Software triggering is a request/response round trip, so
its achieved rate is far below what a hardware trigger reaches — use it to
verify function, not timing.

### Trigger signal requirements

For the measurement to reflect the cameras rather than the pulse source:

- Distribute one common trigger to every camera under test.
- Keep the trigger rate below the frame rate the camera is configured for. A
  trigger faster than the camera's rate causes it to skip pulses; the tool
  correctly reports those as missing frames, but the cause is the trigger.
- Use a hardware-timed source — a hardware PWM, a timer peripheral, or an
  FPGA/PLC output — rather than a software GPIO toggle. This matters more as
  the ROI grows: at a large ROI the readout occupies nearly the whole frame
  period, leaving little margin for timing jitter. At a smaller ROI the margin
  is generous and jitter is absorbed.
- Provide adequate drive for all trigger inputs — a buffer or a per-camera
  opto-isolator — and a common ground between the pulse source and the cameras.
  A single unbuffered 3.3 V output driving eight trigger inputs produces
  degraded edges and intermittent missed triggers.
- Either edge may be used. If a run produces no frames at all, try the other
  edge with `--activation falling` and confirm the line with `--source`.

---

## 6. Reading the report

Each run ends with one row per camera and an aggregate line:

```
cam0 serial=0A512E1D model=U3V1456-60GC color_on=False shape=(1088, 1456)
      frames=601 lost=0 loss=0.0000%  fps=60.05  MB/s=95.13  elapsed=10.01s
  aggregate throughput: 95.13 MB/s over 1 camera(s)
VERDICT: PASS (0 loss on all cameras)
```

| Field | Meaning |
|---|---|
| `frames` | Complete frames received. |
| `lost` | Frames missing from the sequence, plus frames received incomplete. |
| `loss` | `lost` as a percentage of expected. |
| `fps` | Completed frames per second. |
| `MB/s` | Data rate for that camera (decimal MB). |
| `shape` | `(H, W)` raw, `(H, W, 3)` after colour conversion. |

| Verdict | Exit code | Meaning |
|---|---|---|
| `PASS` | 0 | No frames lost on any camera. |
| `NO-DATA` | 2 | A camera returned no frames, so there was nothing to measure. **This is not a pass.** |
| `LOSS` | 3 | Frames were lost; the rate is in `loss`. |

Always read `frames` before the verdict.

### `shortfall` under an external trigger

The trigger tool also reports how many pulses it expected, based on the `--rate`
you passed, and the difference from what arrived:

```
triggers expected at 59.88 Hz = 898, frames received = 886, shortfall = 12
```

`shortfall` and `lost` answer different questions, and a large `shortfall` with
a small `lost` is common:

- **`lost`** counts gaps in the image sequence — frames the camera started and
  did not deliver.
- **`shortfall`** counts pulses that never became a frame at all. A pulse the
  camera never acted on consumes no sequence number, so it cannot appear as a
  gap.

A `shortfall` therefore has two possible causes, and they are easy to tell
apart: **repeat the run at a much smaller ROI.**

- If the shortfall stays about the same, your pulse source is not running at the
  rate you told the tool. Set `--rate` to the rate it actually produces.
  Software-timed sources — a GPIO toggled from a script or a shell command —
  commonly run 1–2 % slow.
- If the shortfall shrinks with the ROI, the camera is missing pulses at the
  larger ROI. See the trigger signal requirements in section 5.

A verdict describes **one run**. Run each configuration several times — or use
`--runs 3` on the trigger tool — before drawing a conclusion.

---

## 7. Suggested test sequence

Work upward. Stop at the first configuration that loses frames and record it.

1. **One camera, free-run, `--color 0`** — establishes the link on its own.
2. **One camera, free-run, `--color 1`** — adds host processing.
3. **Repeat 1 and 2 alternately**, not one after the other in a block. Running
   all of one condition and then all of the other lets unrelated machine
   activity look like a difference between the conditions.
4. **Add cameras one at a time**, repeating 1–3.
5. **Switch to the trigger tool** at your intended trigger rate, with
   `--runs 3 --csv`.
6. **Reduce the ROI height** and repeat if any configuration loses frames.

### Bandwidth to budget for

Frame size × frame rate, at 8 bits per pixel:

| ROI | Per camera @ 60 fps | 8 cameras |
|---|---|---|
| 1456 × 1088 | 95.0 MB/s | 760 MB/s |
| 1456 × 544 | 47.5 MB/s | 380 MB/s |
| 1024 × 600 | 36.9 MB/s | 295 MB/s |
| 1456 × 400 | 34.9 MB/s | 280 MB/s |
| 1456 × 200 | 17.5 MB/s | 140 MB/s |

A single USB3 Gen1 controller sustains roughly 350–400 MB/s in practice.
Spreading cameras across more than one independent controller raises the
available headroom accordingly. These figures are for planning; the measurement
on your own host is the one that counts.

---

## 8. Building your own capture loop

The binding is plain Python over the shared library, so everything in this
folder is meant to be copied and adapted. `burst_capture_raw.py` is the
shortest complete example of a *trigger → receive → store in memory* loop.

Three rules keep an adapted loop running at full rate:

**Do only receiving inside the loop.** `read_frame(copy=False)` hands back a
read-only view that stays valid only until the next call, and the loop is the
only thing feeding the camera. Writing a file, encoding a JPEG, updating a
display or converting colour inside the loop stalls it, and at a large ROI the
stream does not merely slow down — it breaks up and takes several frames to
recover. Buffer what you need in memory and do the work after the run. This is
why `--save` collects frames during the run and writes them all afterwards.

**Preallocate, then use `read_frame_into()`.** Allocate one array for the whole
burst up front and let each frame be written straight into its slice. This
avoids allocating and freeing an array per frame, which is the other common
reason a loop cannot keep up.

**Sample rather than stream to disk.** At full resolution a colour camera
produces about 95 MB/s. Writing every frame ties the capture rate to your disk;
saving every Nth frame does not.

If you need colour, capture raw and convert afterwards — `offline_to_color.py`
shows the second half. The conversion uses the same image pipeline as live
colour capture, so the result is the same; only the timing of the work changes.

---

## 9. Troubleshooting

**No cameras found**
Check the USB3 connection and that no other application currently has the
camera open.

**Windows: the camera cannot be opened**
Its USB3 Vision interface is not bound to WinUSB, typically because another
vendor's vision software installed its own driver. Bind the interface to WinUSB
in Device Manager and replug the camera.

**`NO-DATA`, or a run with `frames=0`**
Under an external trigger: confirm the pulse source is running, that it reaches
the line selected with `--source`, and try the other edge. Confirm the trigger
rate does not exceed the camera's configured frame rate.
In free-run: repeat the run once — a camera needs one acquisition cycle to
settle after being reconfigured.

**Too little light, or no usable frames under trigger**
Pass an explicit exposure and a non-zero gain, for example
`--exposure 10000 --gain 16`.

**Loss appears only under trigger, not in free-run at the same data rate**
This points at the trigger signal rather than at bandwidth. See the trigger
signal requirements in section 5.

**Loss grows as cameras are added**
Total throughput depends on the host controller and hub layout. Spread the
cameras across more than one USB3 controller, or reduce the ROI height. On
Windows, *Device Manager → View → Devices by connection* shows whether all
cameras hang off the same controller.

**Occasional loss at a large ROI on an otherwise healthy link**
A large frame occupies nearly the whole frame period, so any brief interruption
on the host is likely to land inside a transfer. Things worth trying, in order:

- Set the Windows power plan to High Performance while testing.
- Check for background activity — indexing, backup, antivirus scans, updates.
- Close other applications using the same USB controller.
- Reduce the ROI height and confirm whether the loss scales with frame size.
