"""USB3 Vision camera SDK — Python interface.

Quick start::

    import u3v_cam

    with u3v_cam.Camera() as cam:
        cam.exposure_us = 5000
        cam.gain = 0
        cam.start()
        for _ in range(60):
            frame = cam.read_frame()      # numpy.ndarray (h, w) uint8 for Mono8
            ...
        cam.stop()
"""
from __future__ import annotations

import atexit
import ctypes
import threading
from contextlib import suppress
from typing import Any, Dict, List, Optional

import numpy as np

from . import _raw as r
from ._loader import plugin_dir as _default_plugin_dir

__version__ = "2.3.2"


# --------------------------------------------------------------------------
# Exception type
# --------------------------------------------------------------------------
class U3VError(RuntimeError):
    """SDK-level error.  ``code`` is the underlying ``u3v_status_t``."""

    def __init__(self, code: int, where: str = ""):
        self.code = int(code)
        msg = r.status_str(self.code)
        if where:
            msg = f"{where}: {msg}"
        super().__init__(f"{msg} (code={self.code})")


class U3VIncompleteFrame(U3VError):
    """Raised by :meth:`Camera.read_frame` when the SDK returns a torn or short
    frame (``U3V_ERR_INCOMPLETE``): the image data is unreliable and should be
    dropped rather than used. Subclasses :class:`U3VError`, so existing
    ``except U3VError`` handlers still catch it. Iterating a camera
    (``for frame in cam``) skips these frames automatically."""


def _check(code: int, where: str = "") -> None:
    if code != r.U3V_OK:
        raise U3VError(code, where)


# --------------------------------------------------------------------------
# SDK singleton init / shutdown
# --------------------------------------------------------------------------
_sdk_lock = threading.Lock()
_sdk_inited = False


def _sdk_ensure_init() -> None:
    global _sdk_inited
    with _sdk_lock:
        if _sdk_inited:
            return
        _check(r.sdk_init(), "u3v_sdk_init")
        _sdk_inited = True
        atexit.register(_sdk_at_exit)


def _sdk_at_exit() -> None:
    global _sdk_inited
    with _sdk_lock:
        if _sdk_inited:
            with suppress(Exception):
                r.sdk_shutdown()
            _sdk_inited = False


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def sdk_version() -> str:
    """Native SDK version/identity string, e.g. 'innomaker U3V Camera SDK 2.3.0'.
    This is the compiled library's version; ``u3v_cam.__version__`` is the
    Python wrapper's."""
    return r.get_version().decode("utf-8", "replace")


def list_cameras(max_devices: int = 8) -> List[Dict[str, Any]]:
    """Return a list of visible USB3 Vision cameras."""
    _sdk_ensure_init()
    arr = (r.u3v_device_info_t * max_devices)()
    count = r.discover(arr, max_devices)
    out: List[Dict[str, Any]] = []
    for i in range(count):
        d = arr[i]
        out.append({
            "index":        d.device_index,
            "vendor_id":    d.vendor_id,
            "product_id":   d.product_id,
            "manufacturer": d.manufacturer.decode(errors="replace").strip("\x00"),
            "model":        d.model.decode(errors="replace").strip("\x00"),
            "serial":       d.serial.decode(errors="replace").strip("\x00"),
        })
    return out


# --------------------------------------------------------------------------
# Pixel format helper
# --------------------------------------------------------------------------
def _frame_layout(pixel_format: int, width: int, height: int):
    """Return ``(dtype, shape, element_count)`` for a frame of this format and
    size, or ``None`` when the format is not one we can shape."""
    bpp = r.PFNC_BPP.get(int(pixel_format))
    if not bpp or not width or not height:
        return None
    if bpp == 8:
        return np.uint8, (height, width), height * width
    if bpp == 16:
        return np.uint16, (height, width), height * width
    if bpp == 24:
        return np.uint8, (height, width, 3), height * width * 3
    return None


def _frame_to_array(buf: r.u3v_buffer_t) -> np.ndarray:
    """One-copy view of buffer.data into a numpy array shaped to the frame."""
    size = int(buf.received_size) or int(buf.size)
    if size == 0 or not buf.data:
        return np.empty(0, dtype=np.uint8)

    raw = ctypes.string_at(buf.data, size)
    pf = int(buf.pixel_format)
    w, h = int(buf.width), int(buf.height)
    bpp = r.PFNC_BPP.get(pf, 8)

    if bpp == 8:
        arr = np.frombuffer(raw, dtype=np.uint8)
        if h and w and arr.size >= h * w:
            return arr[:h * w].reshape(h, w)
        return arr
    if bpp == 16:
        arr = np.frombuffer(raw, dtype=np.uint16)
        if h and w and arr.size >= h * w:
            return arr[:h * w].reshape(h, w)
        return arr
    if bpp == 24:
        arr = np.frombuffer(raw, dtype=np.uint8)
        if h and w and arr.size >= h * w * 3:
            return arr[:h * w * 3].reshape(h, w, 3)
        return arr
    return np.frombuffer(raw, dtype=np.uint8)


# --------------------------------------------------------------------------
# ISP pipeline: shared parameter access
# --------------------------------------------------------------------------
# Every plugin parameter declares its own C type. Handing param_set a pointer
# to the wrong type does not fail — the plugin reinterprets the bytes, so a
# float parameter written as an integer lands on a garbage value. So the type
# always comes from the parameter's own descriptor, never from its name.
_PARAM_SCALAR_CTYPE = {
    r.U3V_PARAM_TYPE_BOOL:   ctypes.c_uint8,
    r.U3V_PARAM_TYPE_INT32:  ctypes.c_int32,
    r.U3V_PARAM_TYPE_UINT32: ctypes.c_uint32,
    r.U3V_PARAM_TYPE_ENUM:   ctypes.c_uint32,
    r.U3V_PARAM_TYPE_FLOAT:  ctypes.c_float,
}

_PARAM_TYPE_NAMES = {
    r.U3V_PARAM_TYPE_BOOL:      "bool",
    r.U3V_PARAM_TYPE_INT32:     "int",
    r.U3V_PARAM_TYPE_UINT32:    "uint",
    r.U3V_PARAM_TYPE_FLOAT:     "float",
    r.U3V_PARAM_TYPE_ENUM:      "enum",
    r.U3V_PARAM_TYPE_MATRIX3X3: "matrix3x3",
    r.U3V_PARAM_TYPE_BUTTON:    "button",
}


def _make_pipeline(plugin_dir: Optional[str]):
    """Create a pipeline and load the ISP plugins into it. Returns the handle."""
    directory = plugin_dir if plugin_dir is not None else _default_plugin_dir()
    if not directory:
        raise FileNotFoundError(
            "No ISP plugin directory found. Pass plugin_dir=..., set "
            "U3V_PLUGIN_DIR, or place the plugins under "
            "u3v_cam/_libs/<platform>/plugins/."
        )
    pipe = r.u3v_pipeline_p()
    _check(r.pipeline_create(ctypes.byref(pipe)), "pipeline_create")
    r.pipeline_load_dir(pipe, str(directory).encode())
    if int(r.pipeline_plugin_count(pipe)) == 0:
        r.pipeline_destroy(pipe)
        raise FileNotFoundError(
            f"No ISP plugins loaded from {directory!r}. Color demosaic is "
            "unavailable; check that the plugins/ folder shipped with this "
            "package."
        )
    return pipe


class _IspHost:
    """Typed access to the ISP plugin parameters of an owned pipeline.

    Mixed into both :class:`Camera` (live path) and :class:`Processor`
    (offline path) so one profile drives both and the colour comes out the
    same either way. Subclasses provide ``self._pipeline``.
    """

    _pipeline = None
    _param_cache: Optional[Dict[str, Dict[str, Any]]] = None

    # ---- descriptor table ----
    def _require_pipeline(self) -> Any:
        if self._pipeline is None:
            raise RuntimeError(
                "Color pipeline not enabled; call enable_color() first."
            )
        return self._pipeline

    def _invalidate_params(self) -> None:
        self._param_cache = None

    def _params(self) -> Dict[str, Dict[str, Any]]:
        """Map of parameter name -> descriptor, built once per pipeline."""
        if self._param_cache is not None:
            return self._param_cache
        pipe = self._require_pipeline()
        table: Dict[str, Dict[str, Any]] = {}
        for i in range(int(r.pipeline_plugin_count(pipe))):
            for j in range(int(r.pipeline_plugin_param_count(pipe, i))):
                ptr = r.pipeline_plugin_param_get(pipe, i, j)
                if not ptr:
                    continue
                p = ptr.contents
                if not p.name:
                    continue
                labels = []
                if p.type == r.U3V_PARAM_TYPE_ENUM and p.enum_labels:
                    labels = [p.enum_labels[k].decode("utf-8", "replace")
                              for k in range(int(p.enum_count))]
                name = p.name.decode("utf-8", "replace")
                table.setdefault(name, {
                    "name":     name,
                    "plugin":   i,
                    "type":     int(p.type),
                    "type_name": _PARAM_TYPE_NAMES.get(int(p.type), "?"),
                    "label":    p.label.decode("utf-8", "replace") if p.label else "",
                    "category": p.category.decode("utf-8", "replace") if p.category else "",
                    "unit":     p.unit.decode("utf-8", "replace") if p.unit else "",
                    "min":      float(p.min_value),
                    "max":      float(p.max_value),
                    "default":  float(p.default_value),
                    "enum_labels": labels,
                })
        self._param_cache = table
        return table

    def list_color_params(self) -> List[Dict[str, Any]]:
        """Every ISP parameter the loaded plugins expose, with type and range.

        Use this to discover what is adjustable instead of hard-coding names —
        it stays correct when plugins are added or updated.
        """
        return [dict(d) for d in self._params().values()]

    def _descriptor(self, name: str) -> Dict[str, Any]:
        try:
            return self._params()[name]
        except KeyError:
            raise U3VError(
                r.U3V_ERR_INVALID_PARAM,
                f"unknown ISP parameter {name!r}; see list_color_params()",
            ) from None

    # ---- typed set / get ----
    def set_color_param(self, name: str, value: Any = None) -> None:
        """Set one ISP parameter, converted to the type the plugin declares.

        Scalars take a number (``rgb.red_gain``, ``gamma.value``,
        ``histogram.dark_point``), booleans take anything truthy
        (``rgb.enabled``), a 3x3 matrix takes 9 numbers or a 3x3 nested
        sequence (``ccm.matrix``), and buttons take no value at all
        (``rgb.white_balance``, ``ccm.reset``). Requires the colour pipeline.
        """
        pipe = self._require_pipeline()
        d = self._descriptor(name)
        ptype, idx = d["type"], d["plugin"]

        if ptype == r.U3V_PARAM_TYPE_BUTTON:
            storage: Any = None
            vptr = None
        elif ptype == r.U3V_PARAM_TYPE_MATRIX3X3:
            flat = np.asarray(value, dtype=np.float32).reshape(-1)
            if flat.size != 9:
                raise ValueError(
                    f"{name} is a 3x3 matrix: expected 9 values, got {flat.size}"
                )
            storage = (ctypes.c_float * 9)(*[float(v) for v in flat])
            vptr = ctypes.cast(storage, ctypes.c_void_p)
        else:
            ctype = _PARAM_SCALAR_CTYPE.get(ptype)
            if ctype is None:
                raise U3VError(r.U3V_ERR_INVALID_PARAM,
                               f"{name}: unsupported parameter type {ptype}")
            if value is None:
                raise ValueError(f"{name} requires a value")
            if ptype == r.U3V_PARAM_TYPE_BOOL:
                storage = ctype(1 if value else 0)
            elif ptype == r.U3V_PARAM_TYPE_FLOAT:
                storage = ctype(float(value))
            else:
                storage = ctype(int(value))
            vptr = ctypes.cast(ctypes.pointer(storage), ctypes.c_void_p)

        _check(r.pipeline_plugin_param_set(pipe, idx, name.encode(), vptr),
               f"set_color_param({name})")

    def get_color_param(self, name: str) -> Any:
        """Read one ISP parameter back, typed. Buttons return ``None``."""
        pipe = self._require_pipeline()
        d = self._descriptor(name)
        ptype, idx = d["type"], d["plugin"]

        if ptype == r.U3V_PARAM_TYPE_BUTTON:
            return None
        if ptype == r.U3V_PARAM_TYPE_MATRIX3X3:
            storage: Any = (ctypes.c_float * 9)()
            vptr = ctypes.cast(storage, ctypes.c_void_p)
            _check(r.pipeline_plugin_param_get_value(pipe, idx, name.encode(), vptr),
                   f"get_color_param({name})")
            return [[float(storage[row * 3 + col]) for col in range(3)]
                    for row in range(3)]

        ctype = _PARAM_SCALAR_CTYPE.get(ptype)
        if ctype is None:
            raise U3VError(r.U3V_ERR_INVALID_PARAM,
                           f"{name}: unsupported parameter type {ptype}")
        storage = ctype()
        vptr = ctypes.cast(ctypes.pointer(storage), ctypes.c_void_p)
        _check(r.pipeline_plugin_param_get_value(pipe, idx, name.encode(), vptr),
               f"get_color_param({name})")
        if ptype == r.U3V_PARAM_TYPE_BOOL:
            return bool(storage.value)
        if ptype == r.U3V_PARAM_TYPE_FLOAT:
            return float(storage.value)
        return int(storage.value)

    # ---- named convenience wrappers ----
    def auto_white_balance(self) -> None:
        """Run a one-shot gray-world auto white balance.

        Enables the RGB stage and arms the white-balance button; the red/blue
        gains are computed from the NEXT processed frame and then applied to
        every following frame. Call while streaming on a representative,
        well-lit scene, then keep reading frames.
        """
        self.set_color_param("rgb.enabled", True)
        self.set_color_param("rgb.white_balance")

    def set_white_balance(self, red_gain: float, green_gain: float = 1.0,
                          blue_gain: float = 1.0) -> None:
        """Apply fixed per-channel gains (1.0 = unity) and enable the RGB stage.
        Use this to set a known white balance instead of the one-shot auto."""
        self.set_color_param("rgb.enabled", True)
        self.set_color_param("rgb.red_gain", red_gain)
        self.set_color_param("rgb.green_gain", green_gain)
        self.set_color_param("rgb.blue_gain", blue_gain)

    def get_white_balance(self) -> Dict[str, float]:
        """Return the current RGB gains as ``{'red':.., 'green':.., 'blue':..}``."""
        return {
            "red":   self.get_color_param("rgb.red_gain"),
            "green": self.get_color_param("rgb.green_gain"),
            "blue":  self.get_color_param("rgb.blue_gain"),
        }

    def set_gamma(self, value: float) -> None:
        """Enable the gamma stage and set its exponent (0.1 - 4.0)."""
        self.set_color_param("gamma.enabled", True)
        self.set_color_param("gamma.value", value)

    def set_ccm(self, matrix) -> None:
        """Enable the colour-correction stage and load a 3x3 matrix.
        Accepts 9 numbers row-major, or a 3x3 nested sequence."""
        self.set_color_param("ccm.enabled", True)
        self.set_color_param("ccm.matrix", matrix)

    def set_levels(self, dark_point: int, light_point: int) -> None:
        """Enable the histogram stage and set the black/white clip points.
        This is the contrast / brightness stretch."""
        self.set_color_param("histogram.enabled", True)
        self.set_color_param("histogram.dark_point", dark_point)
        self.set_color_param("histogram.light_point", light_point)

    # ---- profile ----
    def get_isp_profile(self) -> Dict[str, Any]:
        """Snapshot every readable ISP parameter as a plain dict.
        Buttons are stateless and are not included."""
        out: Dict[str, Any] = {}
        for name, d in sorted(self._params().items()):
            if d["type"] == r.U3V_PARAM_TYPE_BUTTON:
                continue
            with suppress(U3VError):
                out[name] = self.get_color_param(name)
        return out

    def apply_isp_profile(self, profile: Dict[str, Any],
                          strict: bool = False) -> List[str]:
        """Apply a dict of ISP parameters. Returns the names that were skipped.

        With ``strict=False`` (default) a name this build does not know is
        skipped and reported, so a profile written against a newer plugin set
        still applies as far as it can. ``strict=True`` raises instead.
        """
        known = self._params()
        skipped: List[str] = []
        # Enable flags first: some stages ignore values while disabled.
        ordered = sorted(profile.items(),
                         key=lambda kv: 0 if kv[0].endswith(".enabled") else 1)
        for name, value in ordered:
            if name not in known:
                if strict:
                    raise U3VError(r.U3V_ERR_INVALID_PARAM,
                                   f"unknown ISP parameter {name!r}")
                skipped.append(name)
                continue
            if known[name]["type"] == r.U3V_PARAM_TYPE_BUTTON:
                skipped.append(name)
                continue
            self.set_color_param(name, value)
        return skipped


# --------------------------------------------------------------------------
# Camera class
# --------------------------------------------------------------------------
class Camera(_IspHost):
    """High-level handle for a single USB3 Vision camera."""

    def __init__(self, index: int = 0,
                 num_buffers: int = 4,
                 timeout_ms: int = 5000,
                 transfer_size: int = 0):
        _sdk_ensure_init()

        self._handle = r.u3v_camera_p()
        _check(r.camera_open(ctypes.byref(self._handle), int(index)),
               "u3v_camera_open")

        self._stream: Optional[r.u3v_stream_p] = None
        self._buf: Optional[r.u3v_buffer_t] = None
        self._pipeline: Optional[r.u3v_pipeline_p] = None
        self._view: Optional[r.u3v_buffer_t] = None
        self._streaming = False
        self._acq_mode_touched = False
        self._stream_cfg = r.u3v_stream_config_t(
            num_buffers=int(num_buffers),
            timeout_ms=int(timeout_ms),
            transfer_size=int(transfer_size),
        )

    # ---- context manager ----
    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- info / device ----
    @property
    def info(self) -> Dict[str, Any]:
        ptr = r.camera_get_info(self._handle)
        if not ptr:
            raise U3VError(r.U3V_ERR_NOT_CONNECTED, "camera_get_info")
        i = ptr.contents
        return {
            "manufacturer":     i.manufacturer.decode(errors="replace").strip("\x00"),
            "model":            i.model.decode(errors="replace").strip("\x00"),
            "serial":           i.serial.decode(errors="replace").strip("\x00"),
            "firmware_version": i.firmware_version.decode(errors="replace").strip("\x00"),
            "sensor_width":     int(i.sensor_width),
            "sensor_height":    int(i.sensor_height),
            "width_max":        int(i.width_max),
            "height_max":       int(i.height_max),
        }

    # ---- ROI / format ----
    def _get_u32(self, fn) -> int:
        v = ctypes.c_uint32(0)
        _check(fn(self._handle, ctypes.byref(v)), fn.__name__)
        return int(v.value)

    @property
    def width(self) -> int:        return self._get_u32(r.camera_get_width)
    @width.setter
    def width(self, v: int) -> None: _check(r.camera_set_width(self._handle, int(v)), "set_width")

    @property
    def height(self) -> int:       return self._get_u32(r.camera_get_height)
    @height.setter
    def height(self, v: int) -> None: _check(r.camera_set_height(self._handle, int(v)), "set_height")

    @property
    def offset_x(self) -> int:     return self._get_u32(r.camera_get_offset_x)
    @offset_x.setter
    def offset_x(self, v: int) -> None: _check(r.camera_set_offset_x(self._handle, int(v)), "set_offset_x")

    @property
    def offset_y(self) -> int:     return self._get_u32(r.camera_get_offset_y)
    @offset_y.setter
    def offset_y(self, v: int) -> None: _check(r.camera_set_offset_y(self._handle, int(v)), "set_offset_y")

    @property
    def pixel_format(self) -> int: return self._get_u32(r.camera_get_pixel_format)
    @pixel_format.setter
    def pixel_format(self, v: int) -> None:
        # The SDK verifies the format by reading it back, so INVALID_PARAM here
        # means the camera declined it rather than that the argument was junk.
        code = r.camera_set_pixel_format(self._handle, int(v))
        if code == r.U3V_ERR_INVALID_PARAM:
            raise U3VError(code, f"set_pixel_format(0x{int(v):08X}): this camera "
                                 f"does not support that pixel format")
        _check(code, "set_pixel_format")

    @property
    def payload_size(self) -> int: return self._get_u32(r.camera_get_payload_size)

    def set_roi(self, width: int, height: int,
                offset_x: int = 0, offset_y: int = 0) -> None:
        """Set ROI in one go.  Order matters on some firmwares."""
        self.offset_x = 0
        self.offset_y = 0
        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y

    # ---- exposure / gain / framerate ----
    @property
    def exposure_us(self) -> int:  return self._get_u32(r.camera_get_exposure)
    @exposure_us.setter
    def exposure_us(self, v: int) -> None: _check(r.camera_set_exposure(self._handle, int(v)), "set_exposure")

    def set_exposure_auto(self, enable: bool) -> None:
        _check(r.camera_set_exposure_auto(self._handle, 1 if enable else 0),
               "set_exposure_auto")

    def get_exposure_auto(self) -> int:
        """Current ExposureAuto mode: 0=Off, 1=Once, 2=Continuous."""
        return self._get_u32(r.camera_get_exposure_auto)

    @property
    def gain(self) -> int:         return self._get_u32(r.camera_get_gain)
    @gain.setter
    def gain(self, v: int) -> None: _check(r.camera_set_gain(self._handle, int(v)), "set_gain")

    @property
    def frame_rate(self) -> int:   return self._get_u32(r.camera_get_frame_rate)
    @frame_rate.setter
    def frame_rate(self, v: int) -> None: _check(r.camera_set_frame_rate(self._handle, int(v)), "set_frame_rate")

    @property
    def frame_count(self) -> int:
        """Frames to acquire when acquisition mode is MultiFrame."""
        return self._get_u32(r.camera_get_frame_count)
    @frame_count.setter
    def frame_count(self, v: int) -> None:
        _check(r.camera_set_frame_count(self._handle, int(v)), "set_frame_count")

    def set_acq_mode(self, mode: int) -> None:
        """Set acquisition mode: ACQ_MODE_CONTINUOUS / SINGLE_FRAME / MULTI_FRAME."""
        _check(r.camera_set_acq_mode(self._handle, int(mode)), "set_acq_mode")
        self._acq_mode_touched = True

    # ---- trigger ----
    @property
    def trigger_mode(self) -> bool:
        return bool(self._get_u32(r.camera_get_trigger_mode))

    @trigger_mode.setter
    def trigger_mode(self, on: bool) -> None:
        _check(r.camera_set_trigger_mode(self._handle, 1 if on else 0), "set_trigger_mode")

    @property
    def trigger_source(self) -> int:
        return self._get_u32(r.camera_get_trigger_source)

    @trigger_source.setter
    def trigger_source(self, src: int) -> None:
        _check(r.camera_set_trigger_source(self._handle, int(src)), "set_trigger_source")

    @property
    def trigger_activation(self) -> int:
        """Trigger edge: 2=FallingEdge, 3=RisingEdge."""
        return self._get_u32(r.camera_get_trigger_activation)

    @trigger_activation.setter
    def trigger_activation(self, act: int) -> None:
        _check(r.camera_set_trigger_activation(self._handle, int(act)), "set_trigger_activation")

    def configure_trigger(self,
                          on: bool = True,
                          source: str = "software",
                          activation: str = "rising",
                          selector: str = "frame_start") -> None:
        """Convenience: configure trigger before ``start()``.

        ``source`` ∈ {"line0".."line3", "software"}
        ``activation`` ∈ {"rising", "falling"}
        ``selector`` ∈ {"frame_start", "acq_start"}
        """
        sel_map = {
            "frame_start": r.IMX296_TRIGGER_SEL_FRAME_START,
            "acq_start":   r.IMX296_TRIGGER_SEL_ACQ_START,
        }
        src_map = {
            "line0":    r.IMX296_TRIGGER_SRC_LINE0,
            "line1":    r.IMX296_TRIGGER_SRC_LINE1,
            "line2":    r.IMX296_TRIGGER_SRC_LINE2,
            "line3":    r.IMX296_TRIGGER_SRC_LINE3,
            "software": r.IMX296_TRIGGER_SRC_SOFTWARE,
        }
        act_map = {
            "rising":  r.IMX296_TRIGGER_ACT_RISING_EDGE,
            "falling": r.IMX296_TRIGGER_ACT_FALLING_EDGE,
        }
        if selector not in sel_map:    raise ValueError(f"selector: {selector}")
        if source   not in src_map:    raise ValueError(f"source: {source}")
        if activation not in act_map:  raise ValueError(f"activation: {activation}")

        _check(r.camera_set_trigger_selector(self._handle, sel_map[selector]),  "set_trigger_selector")
        _check(r.camera_set_trigger_mode(self._handle, 1 if on else 0),         "set_trigger_mode")
        _check(r.camera_set_trigger_source(self._handle, src_map[source]),      "set_trigger_source")
        _check(r.camera_set_trigger_activation(self._handle, act_map[activation]), "set_trigger_activation")

    def software_trigger(self) -> None:
        _check(r.camera_send_software_trigger(self._handle), "send_software_trigger")

    def set_line_debounce_us(self, us: int) -> None:
        _check(r.camera_set_line_debounce(self._handle, int(us)), "set_line_debounce")

    def get_line_debounce_us(self) -> int:
        return self._get_u32(r.camera_get_line_debounce)

    def set_strobe(self, duration_us: int, delay_us: int = 0,
                   pre_delay_us: int = 0) -> None:
        _check(r.camera_set_strobe(self._handle, int(duration_us),
                                   int(delay_us), int(pre_delay_us)),
               "set_strobe")

    def get_strobe(self) -> tuple:
        """Return current strobe timing as (duration_us, delay_us, pre_delay_us)."""
        duration = ctypes.c_uint32(0)
        delay = ctypes.c_uint32(0)
        pre_delay = ctypes.c_uint32(0)
        _check(r.camera_get_strobe(self._handle, ctypes.byref(duration),
                                   ctypes.byref(delay), ctypes.byref(pre_delay)),
               "get_strobe")
        return (int(duration.value), int(delay.value), int(pre_delay.value))

    # ---- misc ----
    @property
    def temperature(self) -> int:
        """Sensor temperature, raw register units."""
        return self._get_u32(r.camera_get_temperature)

    def find_me_led(self, on: bool = True) -> None:
        """Toggle the find-me indicator LED.  SDK call alternates state."""
        _check(r.camera_find_me(self._handle), "find_me")

    def reset(self) -> None:
        _check(r.camera_reset(self._handle), "camera_reset")

    def flush(self) -> None:
        r.camera_flush_stream(self._handle)

    # ---- raw register access (escape hatch) ----
    def read_register(self, address: int) -> int:
        v = ctypes.c_uint32(0)
        _check(r.camera_read_reg(self._handle, int(address), ctypes.byref(v)),
               f"read_reg(0x{address:08X})")
        return int(v.value)

    def write_register(self, address: int, value: int) -> None:
        _check(r.camera_write_reg(self._handle, int(address), int(value)),
               f"write_reg(0x{address:08X})")

    # ---- streaming ----
    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return

        # set acquisition mode to continuous unless the caller already touched it
        if not self._acq_mode_touched:
            with suppress(U3VError):
                r.camera_set_acq_mode(self._handle, r.IMX296_ACQ_MODE_CONTINUOUS)

        s = r.u3v_stream_p()
        _check(r.stream_create(ctypes.byref(s), self._handle,
                               ctypes.byref(self._stream_cfg)),
               "stream_create")
        self._stream = s

        # If a color pipeline was requested before the stream existed, attach
        # it now so read_frame() routes through the ISP (grab_view) path.
        if self._pipeline is not None:
            _check(r.stream_set_pipeline(self._stream, self._pipeline),
                   "stream_set_pipeline")

        size = self.payload_size or (self.width * self.height) or (1456 * 1088)
        buf = r.u3v_buffer_t()
        _check(r.buffer_alloc(ctypes.byref(buf), int(size)), "buffer_alloc")
        self._buf = buf

    def start(self) -> None:
        if self._streaming:
            return
        self._ensure_stream()
        _check(r.camera_start(self._handle), "camera_start")
        self._streaming = True

    def stop(self) -> None:
        if not self._streaming:
            return
        with suppress(Exception):
            r.camera_stop(self._handle)
        self._streaming = False

    # ---- color / ISP pipeline ----
    def enable_color(self, plugin_dir: Optional[str] = None) -> int:
        """Enable the ISP pipeline so color frames come back demosaiced.

        Loads the demosaic (+ optional white balance / gamma / CCM) plugins
        that ship next to the native library and attaches them to the stream —
        the same path the Qt viewer uses. After this, ``read_frame()`` returns
        an ``(H, W, 3)`` RGB8 array for Bayer color sensors instead of the raw
        single-channel mosaic.

        ``plugin_dir`` overrides the auto-detected plugin folder. Returns the
        number of plugins loaded. Raises ``FileNotFoundError`` if no plugins
        are found. Mono sensors are unaffected — the plugins pass non-Bayer
        frames through unchanged. Call before or after ``start()``.
        """
        if self._pipeline is not None:
            return int(r.pipeline_plugin_count(self._pipeline))

        pipe = _make_pipeline(plugin_dir)
        self._pipeline = pipe
        self._invalidate_params()
        self._view = r.u3v_buffer_t()
        if self._stream is not None:
            _check(r.stream_set_pipeline(self._stream, self._pipeline),
                   "stream_set_pipeline")
        return int(r.pipeline_plugin_count(pipe))

    def disable_color(self) -> None:
        """Detach and destroy the ISP pipeline; frames revert to raw grab."""
        if self._stream is not None and self._pipeline is not None:
            with suppress(Exception):
                r.stream_set_pipeline(self._stream, None)
        if self._pipeline is not None:
            with suppress(Exception):
                r.pipeline_destroy(self._pipeline)
            self._pipeline = None
        self._invalidate_params()
        self._view = None

    @property
    def color_enabled(self) -> bool:
        """True if the ISP demosaic pipeline is active."""
        return self._pipeline is not None

    # ---- colour parameters ----
    # enable_color() only demosaics; the white-balance / per-channel-gain stage
    # (``rgb.enabled``), gamma, CCM and the histogram stretch are all OFF by
    # default, so newly demosaiced frames carry no colour correction until
    # something turns them on. The typed setters/getters, the parameter
    # reflection and the profile helpers all live in _IspHost, shared with
    # Processor so live and offline colour stay identical.

    # ---- camera-side settings profile ----
    _PROFILE_CAMERA_KEYS = (
        "width", "height", "offset_x", "offset_y", "pixel_format",
        "exposure_us", "gain",
    )

    def get_camera_profile(self) -> Dict[str, Any]:
        """Snapshot the camera-side settings (ROI, format, exposure, gain)."""
        out: Dict[str, Any] = {}
        for key in self._PROFILE_CAMERA_KEYS:
            with suppress(U3VError):
                out[key] = getattr(self, key)
        return out

    def apply_camera_profile(self, profile: Dict[str, Any]) -> None:
        """Apply camera-side settings. Call before :meth:`start`.

        ROI goes through :meth:`set_roi` so the offsets are cleared before the
        new size is written; the remaining keys are applied individually.
        """
        prof = dict(profile)
        width, height = prof.pop("width", None), prof.pop("height", None)
        off_x, off_y = prof.pop("offset_x", 0), prof.pop("offset_y", 0)
        # Pixel format before ROI: payload size depends on both.
        if "pixel_format" in prof:
            self.pixel_format = prof.pop("pixel_format")
        if width is not None and height is not None:
            self.set_roi(int(width), int(height), int(off_x or 0), int(off_y or 0))
        for key in ("exposure_us", "gain"):
            if key in prof:
                setattr(self, key, prof.pop(key))
        if prof:
            raise ValueError(f"unknown camera profile keys: {sorted(prof)}")

    def get_profile(self) -> Dict[str, Any]:
        """Full settings snapshot: ``{'camera': {...}, 'isp': {...}}``.
        The ``isp`` section is present only when the colour pipeline is on."""
        prof: Dict[str, Any] = {"camera": self.get_camera_profile()}
        if self._pipeline is not None:
            prof["isp"] = self.get_isp_profile()
        return prof

    def apply_profile(self, profile: Dict[str, Any],
                      strict: bool = False) -> List[str]:
        """Apply a ``{'camera': ..., 'isp': ...}`` profile.

        The ISP section is applied only when the colour pipeline is enabled;
        otherwise it is reported as skipped rather than silently dropped.
        Returns the list of skipped parameter names.
        """
        skipped: List[str] = []
        if "camera" in profile:
            self.apply_camera_profile(profile["camera"])
        isp = profile.get("isp")
        if isp:
            if self._pipeline is None:
                if strict:
                    raise RuntimeError(
                        "profile has an 'isp' section but the colour pipeline "
                        "is not enabled; call enable_color() first."
                    )
                skipped.extend(f"isp.{k}" for k in isp)
            else:
                skipped.extend(self.apply_isp_profile(isp, strict=strict))
        return skipped

    def read_frame(self, copy: bool = True) -> np.ndarray:
        """Grab a single frame.

        Shape is ``(H, W)`` for Mono / raw Bayer, or ``(H, W, 3)`` RGB8 when
        the color pipeline is enabled (see :meth:`enable_color`).

        The returned array always owns memory that outlives the next grab, so
        it is safe to keep. ``copy=False`` skips one redundant copy and returns
        a read-only array; prefer it when accumulating a burst. To fill memory
        you allocated yourself with no extra copy at all, use
        :meth:`read_frame_into`.
        """
        buf = self._grab()
        arr = _frame_to_array(buf)
        return arr.copy() if copy else arr

    def _grab(self) -> r.u3v_buffer_t:
        """Block for the next frame and return the buffer holding it."""
        if not self._streaming:
            self.start()
        assert self._stream is not None
        if self._pipeline is not None:
            assert self._view is not None
            code = r.stream_grab_view(self._stream, ctypes.byref(self._view))
            buf, where = self._view, "stream_grab_view"
        else:
            assert self._buf is not None
            code = r.stream_grab(self._stream, ctypes.byref(self._buf))
            buf, where = self._buf, "stream_grab"
        if code == r.U3V_ERR_INCOMPLETE:
            raise U3VIncompleteFrame(code, where)
        _check(code, where)
        return buf

    def read_frame_into(self, out: np.ndarray) -> np.ndarray:
        """Grab one frame straight into memory you already own.

        ``out`` must be C-contiguous, writable and large enough for the frame.
        Pass a slice of a preallocated burst array — e.g. ``frames[i]`` of a
        ``(count, H, W)`` uint8 array — to capture a whole burst with no
        per-frame allocation and a single copy out of the SDK buffer.

        Returns a correctly shaped view onto the filled region of ``out``:
        ``(H, W)`` for Mono / raw Bayer, ``(H, W, 3)`` when the colour pipeline
        is on. On a torn frame or a mismatched ``out``, raises without having
        written anything.
        """
        if not isinstance(out, np.ndarray):
            raise TypeError("out must be a numpy array")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("out must be C-contiguous")
        if not out.flags["WRITEABLE"]:
            raise ValueError("out must be writable")

        buf = self._grab()
        size = int(buf.received_size) or int(buf.size)
        if size == 0 or not buf.data:
            raise U3VError(r.U3V_ERR_INCOMPLETE, "read_frame_into: empty frame")
        if out.nbytes < size:
            raise ValueError(f"out holds {out.nbytes} bytes, frame needs {size}")

        # Validate the destination fully before touching it, so a caller that
        # passed the wrong array still has its data intact after the raise.
        layout = _frame_layout(int(buf.pixel_format), int(buf.width),
                               int(buf.height))
        if layout is not None:
            dtype, shape, count = layout
            if out.dtype != dtype:
                raise ValueError(
                    f"out has dtype {out.dtype}, frame is {np.dtype(dtype)}"
                )
            if out.size < count:
                raise ValueError(
                    f"out holds {out.size} elements, frame needs {count}"
                )

        ctypes.memmove(out.ctypes.data, buf.data, size)

        if layout is None:
            return out.reshape(-1).view(np.uint8)[:size]
        return out.reshape(-1)[:count].reshape(shape)

    def __iter__(self):
        if not self._streaming:
            self.start()
        try:
            while True:
                try:
                    yield self.read_frame()
                except U3VIncompleteFrame:
                    continue   # drop torn/short frames, keep streaming
        except KeyboardInterrupt:
            return

    # ---- last-frame metadata ----
    @property
    def _last_buf(self) -> Optional[r.u3v_buffer_t]:
        """Buffer holding the most recent frame's metadata (view when the ISP
        pipeline is active, otherwise the raw grab buffer)."""
        return self._view if self._pipeline is not None else self._buf

    @property
    def last_frame_status(self) -> int:
        buf = self._last_buf
        return int(buf.status) if buf is not None else -1

    @property
    def last_frame_block_id(self) -> int:
        buf = self._last_buf
        return int(buf.block_id) if buf is not None else 0

    @property
    def last_frame_timestamp_ns(self) -> int:
        buf = self._last_buf
        return int(buf.timestamp) if buf is not None else 0

    # ---- shutdown ----
    def close(self) -> None:
        self.stop()
        # Detach + destroy the ISP pipeline before the stream it is bound to.
        self.disable_color()
        if self._buf is not None:
            with suppress(Exception):
                r.buffer_free(ctypes.byref(self._buf))
            self._buf = None
        if self._stream is not None:
            with suppress(Exception):
                r.stream_destroy(self._stream)
            self._stream = None
        if self._handle:
            with suppress(Exception):
                r.camera_close(self._handle)
            self._handle = r.u3v_camera_p()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


# --------------------------------------------------------------------------
# Offline ISP
# --------------------------------------------------------------------------
class Processor(_IspHost):
    """Run the ISP plugin chain over frames that are already in memory.

    This is the offline half of the colour path. Capture raw Bayer with the
    pipeline off — the cheapest thing the host can do during acquisition —
    then demosaic and correct afterwards through these same plugins, so the
    result matches what the live path would have produced for the same
    settings.

    A pipeline is single-consumer: give each worker thread its own
    ``Processor``.

    ::

        with u3v_cam.Processor() as isp:
            isp.apply_isp_profile(profile["isp"])
            for bayer in frames:
                rgb = isp.process(bayer, u3v_cam.PFNC_BAYERRG8)
    """

    def __init__(self, plugin_dir: Optional[str] = None):
        self._pipeline = _make_pipeline(plugin_dir)
        self._invalidate_params()

    def __enter__(self) -> "Processor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def plugin_count(self) -> int:
        return int(r.pipeline_plugin_count(self._require_pipeline()))

    def process(self, frame: np.ndarray, pixel_format: int,
                copy: bool = True) -> np.ndarray:
        """Run the chain over one frame and return the result.

        ``frame`` is the captured image — ``(H, W)`` for Bayer / Mono — and
        ``pixel_format`` is the PFNC constant it was captured as, e.g.
        ``u3v_cam.PFNC_BAYERRG8``. Returns ``(H, W, 3)`` RGB8 once a demosaic
        plugin is loaded.

        The returned array owns its memory. ``copy=False`` returns a read-only
        array instead, saving one copy; it stays valid after the next
        ``process`` call, so it is safe to keep either way.
        """
        pipe = self._require_pipeline()
        src = np.ascontiguousarray(frame)
        if src.ndim < 2:
            raise ValueError("frame must be at least 2-D (H, W)")
        height, width = int(src.shape[0]), int(src.shape[1])

        in_buf = r.u3v_buffer_t()
        in_buf.data = src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        in_buf.size = in_buf.received_size = src.nbytes
        in_buf.width = width
        in_buf.height = height
        in_buf.pixel_format = int(pixel_format)
        in_buf.timestamp = 0
        in_buf.block_id = 0
        in_buf.status = 0

        out_buf = r.u3v_buffer_t()
        _check(r.pipeline_process(pipe, ctypes.byref(in_buf),
                                  ctypes.byref(out_buf)),
               "pipeline_process")
        # `src` must outlive the call: with zero enabled plugins the pipeline
        # aliases the input rather than copying it.
        arr = _frame_to_array(out_buf)
        del src
        return arr.copy() if copy else arr

    def close(self) -> None:
        if self._pipeline is not None:
            with suppress(Exception):
                r.pipeline_destroy(self._pipeline)
            self._pipeline = None
        self._invalidate_params()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


# --------------------------------------------------------------------------
# Settings profiles
# --------------------------------------------------------------------------
def save_profile(path: str, profile: Dict[str, Any]) -> None:
    """Write a settings profile to a JSON file."""
    import json
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_profile(path: str) -> Dict[str, Any]:
    """Read a settings profile written by :func:`save_profile`."""
    import json
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


__all__ = [
    "Camera",
    "Processor",
    "U3VError",
    "U3VIncompleteFrame",
    "list_cameras", "sdk_version",
    "save_profile", "load_profile",
    "__version__",
    # acquisition mode constants
    "ACQ_MODE_CONTINUOUS", "ACQ_MODE_SINGLE_FRAME", "ACQ_MODE_MULTI_FRAME",
    # re-export commonly needed format constants for convenience
    "PFNC_MONO8", "PFNC_MONO10", "PFNC_MONO12", "PFNC_MONO16",
    "PFNC_BAYERGR8", "PFNC_BAYERRG8", "PFNC_BAYERGB8", "PFNC_BAYERBG8",
    "PFNC_BAYERGR10", "PFNC_BAYERRG10", "PFNC_BAYERGB10", "PFNC_BAYERBG10",
    "PFNC_BAYERGR12", "PFNC_BAYERRG12", "PFNC_BAYERGB12", "PFNC_BAYERBG12",
    "PFNC_RGB8",
]

# acquisition mode constants (GenICam semantics)
ACQ_MODE_CONTINUOUS   = r.IMX296_ACQ_MODE_CONTINUOUS
ACQ_MODE_SINGLE_FRAME = r.IMX296_ACQ_MODE_SINGLE
ACQ_MODE_MULTI_FRAME  = r.IMX296_ACQ_MODE_MULTI

# convenience re-exports
PFNC_MONO8     = r.PFNC_MONO8
PFNC_MONO10    = r.PFNC_MONO10
PFNC_MONO12    = r.PFNC_MONO12
PFNC_MONO16    = r.PFNC_MONO16
PFNC_BAYERGR8  = r.PFNC_BAYERGR8
PFNC_BAYERRG8  = r.PFNC_BAYERRG8
PFNC_BAYERGB8  = r.PFNC_BAYERGB8
PFNC_BAYERBG8  = r.PFNC_BAYERBG8
PFNC_BAYERGR10 = r.PFNC_BAYERGR10
PFNC_BAYERRG10 = r.PFNC_BAYERRG10
PFNC_BAYERGB10 = r.PFNC_BAYERGB10
PFNC_BAYERBG10 = r.PFNC_BAYERBG10
PFNC_BAYERGR12 = r.PFNC_BAYERGR12
PFNC_BAYERRG12 = r.PFNC_BAYERRG12
PFNC_BAYERGB12 = r.PFNC_BAYERGB12
PFNC_BAYERBG12 = r.PFNC_BAYERBG12
PFNC_RGB8      = r.PFNC_RGB8
