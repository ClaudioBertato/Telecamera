# u3v_cam — Python interface for the U3V Camera SDK

`u3v_cam` is a Python package that wraps the C SDK for USB3 Vision cameras
with the Sony IMX296 sensor.  It uses `ctypes` to call the same shared
library (`u3v_cam.dll` on Windows, `libu3v_cam.so` on Linux) as the C
example programs and the Qt viewer.

The standalone `u3v-sdk-<version>-python` package bundles the native libraries
for all five supported platforms (Windows x64, Linux x64/ARM64, macOS
arm64/x64) inside `u3v_cam/_libs/<platform>/`, together with the ISP color
plugins under `u3v_cam/_libs/<platform>/plugins/`. Extract, install Python
deps, and run — no build step required.

For what changed in this release, see `RELEASE_NOTES.md` in the SDK package.

---

## Two ways to use it

The same package supports both workflows. Pick the one that matches your
system; you can switch between them at any time, and both produce the same
colour result.

**1 — Raw capture, colour afterwards.** The camera delivers Bayer data, one
byte per pixel, straight into memory you allocated. Colour and all image
adjustments are applied after acquisition. This is the lightest possible host
load while the cameras are running, uses a third of the memory of storing
colour, and lets you re-process the same recording with different settings.
Recommended for many cameras, for full resolution, and for triggered burst
acquisition.

→ `read_frame_into()` during capture, `Processor` afterwards.
  See **Burst capture**. Examples: `burst_capture_raw.py`,
  `offline_to_color.py`.

**2 — Live colour.** The colour pipeline runs while the camera streams, and
each `read_frame()` returns a finished RGB image. Simplest to use, and the
right choice for preview, alignment and lower data rates.

→ `enable_color()`, then `read_frame()`. See **Color capture**.
  Examples: `color_capture.py`, `live_capture.py`, `viewer_pyqt.py`.

Exposure, gain, region of interest and trigger are camera settings and work
identically in both. Everything else — white balance, brightness, gamma,
contrast, colour correction — is applied on the PC, by the same image
processing in either workflow.

---

## Requirements

| Component | Version |
|---|---|
| Python  | 3.8 – 3.12 |
| numpy   | required |
| PyQt6 + pyqtgraph | for the live viewer (`pip install u3v_cam[viewer]`) |
| opencv-python | optional; enables `--show` preview in `live_capture.py` |

On Windows you also need:
- WinUSB driver installed for the camera (use Zadig — see
  `WINUSB_DRIVER_INSTALL.md` in the SDK package)
- Microsoft Visual C++ Runtime (already installed on Windows 10/11)

On Linux:
- `libusb-1.0` runtime (`apt install libusb-1.0-0`)
- udev rule for non-root access (`build-tools/pi-build/99-u3v.rules`,
  installed automatically by `install_deps.sh`)

---

## Install

### From the standalone customer package (recommended)

```text
u3v-sdk-<version>-python/
├── u3v_cam/
│   └── _libs/
│       ├── windows-x64/   u3v_cam.dll + libusb-1.0.dll + plugins/
│       ├── linux-x64/     libu3v_cam.so.2 + plugins/
│       ├── linux-arm64/   libu3v_cam.so.2 + plugins/
│       ├── macos-arm64/   libu3v_cam.2.dylib + plugins/
│       └── macos-x64/     libu3v_cam.2.dylib + plugins/
├── examples/
├── tests/
├── run_basic_capture.{bat,sh}
├── run_live_capture.{bat,sh}
├── run_viewer.{bat,sh}
└── install_deps.{bat,sh}
```

**Windows:**
```bat
unzip u3v-sdk-<version>-python.zip
cd u3v-sdk-<version>-python
install_deps.bat                      :: installs numpy + optional viewer/cv2
run_basic_capture.bat                 :: capture 5 frames, save first to .pgm
run_viewer.bat                        :: live PyQt6 viewer
```

**Linux (Ubuntu 22.04 / Pi 5 / Jetson Orin Nano):**
```bash
unzip u3v-sdk-<version>-python.zip
cd u3v-sdk-<version>-python
chmod +x *.sh
./install_deps.sh
./run_basic_capture.sh
./run_viewer.sh
```

### Override the library location

If you want to use a different `u3v_cam.dll` / `libu3v_cam.so` than the one
bundled in `_libs/`:

```bash
# Windows
set U3V_CAM_DLL=C:\path\to\u3v_cam.dll

# Linux / macOS
export U3V_CAM_LIB=/path/to/libu3v_cam.so.2
```

### As a regular Python package (development)

```bash
cd u3v-sdk-<version>-python
pip install -e .          # core
pip install -e .[viewer]  # core + PyQt6 viewer
pip install -e .[opencv]  # core + cv2 preview
```

---

## Quick start

```python
import u3v_cam

with u3v_cam.Camera() as cam:
    print(cam.info)               # {'manufacturer': ..., 'serial': ..., ...}
    cam.exposure_us = 5000        # 5 ms
    cam.gain = 0
    cam.start()
    for _ in range(60):
        frame = cam.read_frame()  # numpy.ndarray, shape (H, W) uint8 for Mono8
        # ... process frame
    cam.stop()
```

The first `read_frame()` auto-starts the stream if `start()` wasn't called
explicitly.  `with`-statement exit closes everything cleanly.

---

## Color capture

Color sensors output a Bayer mosaic. Enable the ISP pipeline with
`cam.enable_color()` and `read_frame()` returns a demosaiced `(H, W, 3)` uint8
RGB image — the same output the Qt viewer produces — instead of the raw
single-channel mosaic.

```python
import u3v_cam

with u3v_cam.Camera() as cam:
    cam.pixel_format = u3v_cam.PFNC_BAYERRG8   # 8-bit Bayer color
    cam.enable_color()                         # load ISP plugins, demosaic on
    cam.start()
    rgb = cam.read_frame()                     # numpy.ndarray (H, W, 3) RGB8
    cam.stop()
```

- `enable_color(plugin_dir=None)` loads the ISP plugins bundled under
  `u3v_cam/_libs/<platform>/plugins/` and attaches them to the stream. It
  returns the number of plugins loaded. Pass `plugin_dir=...` or set the
  `U3V_PLUGIN_DIR` environment variable to load plugins from another location.
- `disable_color()` reverts to raw single-channel capture.
- `color_enabled` reports whether the pipeline is active.
- The demosaic operates on 8-bit Bayer input (`PFNC_BAYERRG8` / `GR8` / `GB8`
  / `BG8`); the Bayer phase is auto-detected from each frame.
- Mono sensors are unaffected — the plugins pass non-Bayer frames through
  unchanged, so `enable_color()` is safe to call on any camera.

Full example: `examples/color_capture.py`.

### Image adjustments

With the colour pipeline on, these adjustments are available:

| Adjustment | Call |
|---|---|
| White balance, automatic | `auto_white_balance()` |
| White balance, fixed gains | `set_white_balance(red, green, blue)` |
| Per-channel brightness | `set_color_param("rgb.red_offset", -20)` |
| Gamma | `set_gamma(1.8)` |
| Contrast / level stretch | `set_levels(dark_point, light_point)` |
| Hue and saturation | `set_ccm(matrix)` — 3×3 colour correction matrix |
| Demosaic quality | `set_color_param("demosaic.algorithm", 1)` |

`list_color_params()` returns every adjustment the loaded plugins expose, with
its type, range and default, so you can discover them at runtime instead of
hard-coding names:

```python
for p in cam.list_color_params():
    print(p["name"], p["type_name"], p["min"], p["max"], p["default"])
```

Read any of them back with `get_color_param(name)`.

Exposure and gain are camera settings rather than image adjustments — set them
with `cam.exposure_us` and `cam.gain`.

---

## Burst capture

To capture a burst of frames with the least possible host work while the
camera is running, capture **raw** and apply colour afterwards:

```python
import numpy as np, u3v_cam

with u3v_cam.Camera(num_buffers=64) as cam:
    cam.pixel_format = u3v_cam.PFNC_BAYERRG8
    cam.configure_trigger(on=True, source="line1", activation="rising")
    cam.start()

    frames = np.empty((200, cam.height, cam.width), dtype=np.uint8)
    for i in range(200):
        cam.read_frame_into(frames[i])       # one copy, no allocation
    cam.stop()
```

Then convert when the acquisition is over:

```python
with u3v_cam.Processor() as isp:
    isp.set_white_balance(1.6, 1.0, 1.35)
    isp.set_gamma(1.8)
    rgb = [isp.process(f, u3v_cam.PFNC_BAYERRG8) for f in frames]
```

`Processor` runs the same ISP as live colour capture, so the result matches
what `enable_color()` would have produced for the same settings. Converting
afterwards also uses a third of the memory of storing RGB, and lets you
re-convert the same burst with different settings without capturing again.

Full examples: `examples/burst_capture_raw.py` and
`examples/offline_to_color.py`.

---

## Settings profiles

A profile is a plain JSON file holding camera settings and image adjustments.
Apply the same profile to every camera in a system and they all produce the
same colour from the same scene.

```python
import u3v_cam

profile = u3v_cam.load_profile("camera_profile.json")

with u3v_cam.Camera() as cam:
    cam.enable_color()
    cam.apply_profile(profile)        # camera settings + image adjustments
    ...

with u3v_cam.Processor() as isp:
    isp.apply_isp_profile(profile["isp"])   # same adjustments, offline
```

```json
{
  "camera": { "width": 1024, "height": 600, "exposure_us": 8000, "gain": 16 },
  "isp":    { "rgb.enabled": true, "rgb.red_gain": 1.6, "gamma.enabled": true,
              "gamma.value": 1.8 }
}
```

- `cam.get_profile()` snapshots the current settings into the same structure;
  `u3v_cam.save_profile(path, profile)` writes it out.
- Applying a profile is a handful of settings writes at open time — it does
  not affect acquisition.
- Unknown entries are skipped and returned to you rather than failing the
  whole profile; pass `strict=True` to raise instead.
- A sample file ships as `examples/camera_profile.json`.

---

## Measuring frame loss

Two ready-to-run tests report, per camera, how many frames arrived, how many
were lost, the achieved frame rate and the throughput. Use them to validate a
USB3 topology before building the rest of your application around it.

> **`examples/FRAME_LOSS_TESTING.md` is the full guide** — every option, how to
> read the report, a suggested test sequence, bandwidth planning figures and
> troubleshooting. Start there if you are qualifying a multi-camera setup.

```bash
# free-run, every camera connected, 20 s
python examples/frameloss_freerun.py

# external 60 Hz trigger on line1, full resolution, 3 consecutive runs
python examples/frameloss_trigger.py --rate 60 --roi-w 1456 --roi-h 1088 --runs 3

# no trigger wiring yet - software trigger instead
python examples/frameloss_trigger.py --source software --rate 30
```

- All selected cameras stream simultaneously, so the numbers reflect the real
  contention on your host controllers rather than one camera at a time.
- `--color 1` (the default) converts every frame to RGB while streaming;
  `--color 0` measures the raw path. A pass with `--color 1` is the
  conservative result.
- `--format` covers `mono8` and the four 8-bit Bayer orders, so the same test
  serves mono and colour sensors. Colour conversion applies to the Bayer
  formats; mono is measured as raw data.
- `--serials A1B2C3,D4E5F6` selects specific cameras. Serial numbers are
  stable; enumeration order is not.
- `--csv result.csv` appends every run to a file.
- Exit code is `0` for pass, `2` when a camera produced no data, `3` when
  frames were lost — so the tests drop straight into an automated check.

Software triggering is a request/response round trip, so its achieved rate is
well below what an external hardware trigger reaches. Use it to verify the
camera and the link, not to validate frame-rate timing.

---

## API at a glance

### Discovery

```python
u3v_cam.list_cameras()
# -> [{'index': 0, 'manufacturer': 'XYZ', 'model': '...', 'serial': '...'}]
```

### `Camera`

| Property | Type | Notes |
|---|---|---|
| `info` | dict | manufacturer, model, serial, firmware, sensor size |
| `width`, `height` | int | live ROI |
| `offset_x`, `offset_y` | int | live ROI offset |
| `pixel_format` | int | use `u3v_cam.PFNC_*` constants |
| `payload_size` | int | bytes per frame (read-only) |
| `exposure_us` | int | exposure in microseconds |
| `gain` | int | sensor gain, 0–480 on IMX296 |
| `frame_rate` | int | target fps |
| `trigger_mode` | bool | trigger enable |
| `temperature` | int | sensor temperature reading |
| `last_frame_status` | int | status of last grab (0 = OK) |
| `last_frame_block_id` | int | monotonic frame counter from the camera |
| `last_frame_timestamp_ns` | int | per-frame timestamp from the camera |
| `color_enabled` | bool | whether the ISP demosaic pipeline is active |

| Method | Purpose |
|---|---|
| `start()`, `stop()` | acquisition control |
| `read_frame(copy=True)` | grab one frame as `np.ndarray` |
| `read_frame_into(out)` | grab into memory you allocated; ideal for bursts |
| `enable_color(plugin_dir=None)` | load ISP plugins; color frames become `(H, W, 3)` RGB8 |
| `disable_color()` | revert to raw single-channel capture |
| `auto_white_balance()` | one-shot gray-world white balance (needs color pipeline) |
| `set_white_balance(red, green, blue)` | apply fixed per-channel RGB gains |
| `get_white_balance()` | read the current RGB gains |
| `set_gamma(value)` | enable gamma and set the exponent |
| `set_ccm(matrix)` | enable colour correction and load a 3×3 matrix |
| `set_levels(dark, light)` | enable the contrast / level stretch |
| `list_color_params()` | every image adjustment, with type and range |
| `set_color_param(name, value)` / `get_color_param(name)` | any adjustment by name |
| `get_profile()` / `apply_profile(profile)` | snapshot / restore all settings |
| `set_roi(w, h, ox=0, oy=0)` | bulk ROI update with safe ordering |
| `configure_trigger(on, source, activation, selector)` | full trigger config |
| `software_trigger()` | fire one software trigger |
| `set_strobe(duration, delay, pre_delay)` | strobe output config |
| `find_me_led(on=True)` | toggle find-me LED |
| `set_exposure_auto(enable)` | auto-exposure on/off |
| `flush()` | discard pending USB data |
| `read_register(addr)` / `write_register(addr, val)` | low-level register access (advanced) |
| `close()` | release resources (also via `with`) |

### Iteration

```python
for frame in cam:           # auto-start, infinite loop, Ctrl+C ends
    ...
```

---

## Examples

| File | Description |
|---|---|
| `basic_capture.py` | Discover, open, capture 5 frames, save first as PGM |
| `live_capture.py`  | Continuous capture with FPS / dropped statistics; `--show` enables OpenCV preview |
| `trigger_mode.py`  | Software trigger demo with per-trigger latency timing |
| `roi_exposure.py`  | Sweep ROI / exposure / gain combinations |
| `color_capture.py` | Enable the ISP pipeline and save a demosaiced RGB image as PPM |
| `burst_capture_raw.py` | Capture N triggered frames as raw data into memory and save the burst |
| `offline_to_color.py` | Convert a saved burst to colour images using a settings file |
| `camera_profile.json` | Sample settings profile for the two scripts above |
| `frameloss_freerun.py` | Free-run frame-loss and throughput test, all cameras at once |
| `frameloss_trigger.py` | Triggered frame-loss test, 1–8 cameras, hardware or software trigger |
| `FRAME_LOSS_TESTING.md` | Full guide to both frame-loss tests |
| `viewer_pyqt.py`   | Full live viewer with PyQt6 + pyqtgraph (sliders, ROI, trigger) |

Run any of them with `python examples/<file>.py`.

---

## Troubleshooting

**`OSError: Failed to locate u3v_cam shared library`**

The Python package didn't find the C library next to itself.  Either:
- copy `u3v_cam.dll` / `libu3v_cam.so.2` into the same folder as
  `u3v_cam/_loader.py`, or
- set `U3V_CAM_DLL` / `U3V_CAM_LIB` to the full path of the library.

**`No camera found`**

- Windows: confirm WinUSB is installed (Device Manager → camera shows
  "WinUSB Devices") — use Zadig if not.
- Linux: confirm udev rules are loaded (`ls /etc/udev/rules.d/` should
  contain `99-u3v.rules`) and replug the camera.

**Color frames look wrong (raw mosaic or tinted)**

- Call `cam.enable_color()` before `start()` so frames come back demosaiced
  as `(H, W, 3)` RGB8. Without it, `read_frame()` returns the raw Bayer
  mosaic.
- Confirm the ISP plugins are present under
  `u3v_cam/_libs/<platform>/plugins/` (or point `U3V_PLUGIN_DIR` at them).
- Use an 8-bit Bayer pixel format (e.g. `cam.pixel_format = u3v_cam.PFNC_BAYERRG8`).

---

## Performance notes

- At 1456×1088 8-bit, 60 fps is ~95 MB/s per camera. Multiply by the number of
  cameras on one USB3 host controller and keep the total comfortably under
  what that controller sustains.
- `read_frame(copy=True)` returns a writable array. `read_frame(copy=False)`
  returns a read-only one and saves a copy; both stay valid after the next
  grab, so either is safe to keep.
- `read_frame_into(out)` writes straight into memory you allocated. For a
  burst, allocate once as `(count, H, W)` and pass `frames[i]` — no per-frame
  allocation, and the frame is copied only once.
- Capturing raw and converting to colour afterwards costs the least during
  acquisition. See **Burst capture** above.
- The `Camera` class is **not thread-safe**.  If you need concurrent
  control + grab, do control on the same thread that calls
  `read_frame()`, or use a `threading.Lock`.
- A `Processor` is single-consumer too: give each worker thread its own.

---

## Support

- Email: **sales@inno-maker.com**
- Web: **https://www.inno-maker.com**

When reporting an issue, please include the SDK version so we can match it:

```python
import u3v_cam
print(u3v_cam.sdk_version())   # native library, e.g. "innomaker U3V Camera SDK 2.3.2"
print(u3v_cam.__version__)     # Python wrapper version
```
