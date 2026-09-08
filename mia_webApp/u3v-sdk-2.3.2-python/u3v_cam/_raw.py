"""1:1 ctypes binding to the U3V Camera SDK C API.

Mirrors the public headers under ``include/u3v/`` of the C SDK distribution.
Each ``u3v_*`` function is bound with ``argtypes`` and ``restype`` so callers
get TypeError on mistakes instead of silent corruption.

This module is intentionally low-level. Use ``u3v_cam.Camera`` for normal work.
"""
from __future__ import annotations

import ctypes
from ctypes import (
    POINTER, Structure, byref, c_char, c_char_p, c_double, c_float, c_int,
    c_int32, c_uint, c_uint8, c_uint16, c_uint32, c_uint64, c_void_p,
)

from ._loader import load_library

_lib = load_library()


# --------------------------------------------------------------------------
# Status / error codes (u3v_status_t)
# --------------------------------------------------------------------------
U3V_OK                    =   0
U3V_ERR_INVALID_PARAM     =  -1
U3V_ERR_NO_DEVICE         =  -2
U3V_ERR_USB_OPEN          =  -3
U3V_ERR_USB_IO            =  -4
U3V_ERR_TIMEOUT           =  -5
U3V_ERR_PROTOCOL          =  -6
U3V_ERR_NO_MEMORY         =  -7
U3V_ERR_NOT_CONNECTED     =  -8
U3V_ERR_BUSY              =  -9
U3V_ERR_ABORTED           = -10
U3V_ERR_BUFFER_TOO_SMALL  = -11
U3V_ERR_STREAM            = -12
U3V_ERR_INCOMPLETE        = -13

_STATUS_NAMES = {
    U3V_OK: "OK",
    U3V_ERR_INVALID_PARAM: "Invalid parameter",
    U3V_ERR_NO_DEVICE: "No device found",
    U3V_ERR_USB_OPEN: "USB open failed",
    U3V_ERR_USB_IO: "USB I/O error",
    U3V_ERR_TIMEOUT: "Timeout",
    U3V_ERR_PROTOCOL: "Protocol error",
    U3V_ERR_NO_MEMORY: "Out of memory",
    U3V_ERR_NOT_CONNECTED: "Not connected",
    U3V_ERR_BUSY: "Device busy",
    U3V_ERR_ABORTED: "Aborted",
    U3V_ERR_BUFFER_TOO_SMALL: "Buffer too small",
    U3V_ERR_STREAM: "Stream error",
    U3V_ERR_INCOMPLETE: "Incomplete frame",
}


def status_str(code: int) -> str:
    return _STATUS_NAMES.get(code, f"Unknown error ({code})")


# --------------------------------------------------------------------------
# Pixel formats (PFNC subset shipped in u3v_types.h)
# --------------------------------------------------------------------------
PFNC_MONO8         = 0x01080001
PFNC_MONO10        = 0x01100003
PFNC_MONO10_PACKED = 0x010C0004
PFNC_MONO12        = 0x01100005
PFNC_MONO12_PACKED = 0x010C0006
PFNC_MONO16        = 0x01100007
PFNC_BAYERGR8      = 0x01080008
PFNC_BAYERRG8      = 0x01080009
PFNC_BAYERGB8      = 0x0108000A
PFNC_BAYERBG8      = 0x0108000B
PFNC_BAYERGR10     = 0x0110000C
PFNC_BAYERRG10     = 0x0110000D
PFNC_BAYERGB10     = 0x0110000E
PFNC_BAYERBG10     = 0x0110000F
PFNC_BAYERGR12     = 0x01100010
PFNC_BAYERRG12     = 0x01100011
PFNC_BAYERGB12     = 0x01100012
PFNC_BAYERBG12     = 0x01100013
PFNC_RGB8          = 0x02180014
PFNC_YUV422_8      = 0x02100032

# bits-per-pixel for the common formats; used by Camera to size numpy buffers.
# The 10/12-bit Bayer formats are unpacked 16-bit containers (PFNC size
# nibble 0x10 = 16), so they decode through the uint16 path just like Mono10/12.
PFNC_BPP = {
    PFNC_MONO8:    8,
    PFNC_MONO10:  16,
    PFNC_MONO12:  16,
    PFNC_MONO16:  16,
    PFNC_BAYERGR8:  8, PFNC_BAYERRG8:  8, PFNC_BAYERGB8:  8, PFNC_BAYERBG8:  8,
    PFNC_BAYERGR10: 16, PFNC_BAYERRG10: 16, PFNC_BAYERGB10: 16, PFNC_BAYERBG10: 16,
    PFNC_BAYERGR12: 16, PFNC_BAYERRG12: 16, PFNC_BAYERGB12: 16, PFNC_BAYERBG12: 16,
    PFNC_RGB8:    24,
    PFNC_YUV422_8: 16,
}


# --------------------------------------------------------------------------
# IMX296 enum values (u3v_imx296.h)
# --------------------------------------------------------------------------
IMX296_ACQ_MODE_CONTINUOUS    = 0
IMX296_ACQ_MODE_SINGLE        = 1
IMX296_ACQ_MODE_MULTI         = 2

IMX296_TRIGGER_SEL_ACQ_START   = 1
IMX296_TRIGGER_SEL_FRAME_START = 2

IMX296_TRIGGER_MODE_OFF = 0
IMX296_TRIGGER_MODE_ON  = 1

IMX296_TRIGGER_SRC_LINE0    = 0
IMX296_TRIGGER_SRC_LINE1    = 1
IMX296_TRIGGER_SRC_LINE2    = 2
IMX296_TRIGGER_SRC_LINE3    = 3
IMX296_TRIGGER_SRC_SOFTWARE = 4

IMX296_TRIGGER_ACT_FALLING_EDGE = 2
IMX296_TRIGGER_ACT_RISING_EDGE  = 3


# --------------------------------------------------------------------------
# Structs
# --------------------------------------------------------------------------
class u3v_device_info_t(Structure):
    _fields_ = [
        ("vendor_id",    c_uint16),
        ("product_id",   c_uint16),
        ("serial",       c_char * 64),
        ("model",        c_char * 64),
        ("manufacturer", c_char * 64),
        ("device_index", c_int),
    ]


class u3v_cam_info_t(Structure):
    _fields_ = [
        ("manufacturer",     c_char * 64),
        ("model",            c_char * 64),
        ("serial",           c_char * 64),
        ("firmware_version", c_char * 64),
        ("sensor_width",     c_uint32),
        ("sensor_height",    c_uint32),
        ("width_max",        c_uint32),
        ("height_max",       c_uint32),
    ]


class u3v_buffer_t(Structure):
    _fields_ = [
        ("data",          POINTER(c_uint8)),
        ("size",          c_uint32),
        ("received_size", c_uint32),
        ("width",         c_uint32),
        ("height",        c_uint32),
        ("pixel_format",  c_uint32),
        ("timestamp",     c_uint64),
        ("block_id",      c_uint64),
        ("status",        c_uint16),
    ]


class u3v_stream_config_t(Structure):
    _fields_ = [
        ("num_buffers",   c_uint32),
        ("timeout_ms",    c_uint32),
        ("transfer_size", c_uint32),
    ]


# --------------------------------------------------------------------------
# ISP plugin parameter reflection (u3v_plugin.h)
# --------------------------------------------------------------------------
# Every plugin parameter declares its own type. Callers must hand
# u3v_pipeline_plugin_param_set/get a pointer to storage of exactly that
# type — guessing from the parameter name silently corrupts the value.
U3V_PARAM_TYPE_BOOL      = 0   # uint8_t
U3V_PARAM_TYPE_INT32     = 1   # int32_t
U3V_PARAM_TYPE_UINT32    = 2   # uint32_t
U3V_PARAM_TYPE_FLOAT     = 3   # float
U3V_PARAM_TYPE_ENUM      = 4   # uint32_t with enumerated labels
U3V_PARAM_TYPE_MATRIX3X3 = 5   # float[9], row-major
U3V_PARAM_TYPE_BUTTON    = 6   # stateless action; value ignored / NULL


class u3v_param_t(Structure):
    _fields_ = [
        ("name",          c_char_p),
        ("label",         c_char_p),
        ("category",      c_char_p),
        ("unit",          c_char_p),
        ("type",          c_int),
        ("min_value",     c_double),
        ("max_value",     c_double),
        ("default_value", c_double),
        ("enum_labels",   POINTER(c_char_p)),
        ("enum_count",    c_uint32),
        ("flags",         c_uint32),
    ]


class u3v_plugin_info_t(Structure):
    _fields_ = [
        ("abi_version",   c_uint32),
        ("name",          c_char_p),
        ("vendor",        c_char_p),
        ("version",       c_char_p),
        ("order_hint",    c_uint32),
        ("accepts_pfnc",  c_uint32),
        ("produces_pfnc", c_uint32),
        ("reserved",      c_uint32 * 8),
    ]


# Opaque handles
u3v_camera_p   = c_void_p
u3v_stream_p   = c_void_p
u3v_pipeline_p = c_void_p


# --------------------------------------------------------------------------
# Function binding helpers
# --------------------------------------------------------------------------
def _bind(name: str, restype, *argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


# ---- SDK lifecycle / discovery (u3v_camera.h) -----------------------------
get_version   = _bind("u3v_get_version",   c_char_p)
sdk_init      = _bind("u3v_sdk_init",      c_int)
sdk_shutdown  = _bind("u3v_sdk_shutdown",  None)
discover      = _bind("u3v_discover",      c_int, POINTER(u3v_device_info_t), c_int)
camera_open   = _bind("u3v_camera_open",   c_int, POINTER(u3v_camera_p), c_int)
camera_close  = _bind("u3v_camera_close",  None,  u3v_camera_p)
camera_get_info = _bind("u3v_camera_get_info", POINTER(u3v_cam_info_t), u3v_camera_p)

# ---- Image format ---------------------------------------------------------
camera_get_width  = _bind("u3v_camera_get_width",  c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_width  = _bind("u3v_camera_set_width",  c_int, u3v_camera_p, c_uint32)
camera_get_height = _bind("u3v_camera_get_height", c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_height = _bind("u3v_camera_set_height", c_int, u3v_camera_p, c_uint32)
camera_get_offset_x = _bind("u3v_camera_get_offset_x", c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_offset_x = _bind("u3v_camera_set_offset_x", c_int, u3v_camera_p, c_uint32)
camera_get_offset_y = _bind("u3v_camera_get_offset_y", c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_offset_y = _bind("u3v_camera_set_offset_y", c_int, u3v_camera_p, c_uint32)
camera_get_pixel_format = _bind("u3v_camera_get_pixel_format", c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_pixel_format = _bind("u3v_camera_set_pixel_format", c_int, u3v_camera_p, c_uint32)
camera_get_payload_size = _bind("u3v_camera_get_payload_size", c_int, u3v_camera_p, POINTER(c_uint32))

# ---- Acquisition control --------------------------------------------------
camera_set_acq_mode    = _bind("u3v_camera_set_acq_mode",    c_int, u3v_camera_p, c_uint32)
camera_start           = _bind("u3v_camera_start",           c_int, u3v_camera_p)
camera_stop            = _bind("u3v_camera_stop",            c_int, u3v_camera_p)
camera_get_frame_rate  = _bind("u3v_camera_get_frame_rate",  c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_frame_rate  = _bind("u3v_camera_set_frame_rate",  c_int, u3v_camera_p, c_uint32)
camera_get_frame_count = _bind("u3v_camera_get_frame_count", c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_frame_count = _bind("u3v_camera_set_frame_count", c_int, u3v_camera_p, c_uint32)

# ---- Exposure / gain ------------------------------------------------------
camera_get_exposure       = _bind("u3v_camera_get_exposure",       c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_exposure       = _bind("u3v_camera_set_exposure",       c_int, u3v_camera_p, c_uint32)
camera_set_exposure_auto  = _bind("u3v_camera_set_exposure_auto",  c_int, u3v_camera_p, c_uint32)
camera_get_exposure_auto  = _bind("u3v_camera_get_exposure_auto",  c_int, u3v_camera_p, POINTER(c_uint32))
camera_get_gain           = _bind("u3v_camera_get_gain",           c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_gain           = _bind("u3v_camera_set_gain",           c_int, u3v_camera_p, c_uint32)

# ---- Trigger --------------------------------------------------------------
camera_set_trigger_selector   = _bind("u3v_camera_set_trigger_selector",   c_int, u3v_camera_p, c_uint32)
camera_get_trigger_mode       = _bind("u3v_camera_get_trigger_mode",       c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_trigger_mode       = _bind("u3v_camera_set_trigger_mode",       c_int, u3v_camera_p, c_uint32)
camera_get_trigger_source     = _bind("u3v_camera_get_trigger_source",     c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_trigger_source     = _bind("u3v_camera_set_trigger_source",     c_int, u3v_camera_p, c_uint32)
camera_set_trigger_activation = _bind("u3v_camera_set_trigger_activation", c_int, u3v_camera_p, c_uint32)
camera_get_trigger_activation = _bind("u3v_camera_get_trigger_activation", c_int, u3v_camera_p, POINTER(c_uint32))
camera_send_software_trigger  = _bind("u3v_camera_send_software_trigger",  c_int, u3v_camera_p)
camera_set_line_debounce      = _bind("u3v_camera_set_line_debounce",      c_int, u3v_camera_p, c_uint32)
camera_get_line_debounce      = _bind("u3v_camera_get_line_debounce",      c_int, u3v_camera_p, POINTER(c_uint32))
camera_set_strobe             = _bind("u3v_camera_set_strobe",             c_int, u3v_camera_p, c_uint32, c_uint32, c_uint32)
camera_get_strobe             = _bind("u3v_camera_get_strobe",             c_int, u3v_camera_p, POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32))

# ---- Misc -----------------------------------------------------------------
camera_get_temperature = _bind("u3v_camera_get_temperature", c_int, u3v_camera_p, POINTER(c_uint32))
camera_reset           = _bind("u3v_camera_reset",           c_int, u3v_camera_p)
camera_find_me         = _bind("u3v_camera_find_me",         c_int, u3v_camera_p)
camera_flush_stream    = _bind("u3v_camera_flush_stream",    None,  u3v_camera_p)

# ---- Raw register access --------------------------------------------------
camera_read_reg  = _bind("u3v_camera_read_reg",  c_int, u3v_camera_p, c_uint64, POINTER(c_uint32))
camera_write_reg = _bind("u3v_camera_write_reg", c_int, u3v_camera_p, c_uint64, c_uint32)

# ---- Stream engine (u3v_stream.h) -----------------------------------------
stream_create  = _bind("u3v_stream_create",  c_int, POINTER(u3v_stream_p), u3v_camera_p, POINTER(u3v_stream_config_t))
stream_destroy = _bind("u3v_stream_destroy", None,  u3v_stream_p)
buffer_alloc   = _bind("u3v_buffer_alloc",   c_int, POINTER(u3v_buffer_t), c_uint32)
buffer_free    = _bind("u3v_buffer_free",    None,  POINTER(u3v_buffer_t))
stream_grab    = _bind("u3v_stream_grab",    c_int, u3v_stream_p, POINTER(u3v_buffer_t))
buffer_save    = _bind("u3v_buffer_save",    c_int, POINTER(u3v_buffer_t), c_char_p)

# ---- ISP pipeline (u3v_pipeline.h + u3v_stream.h, 2.2.0+) ------------------
# Color sensors emit a Bayer mosaic; the pipeline runs demosaic (+ optional
# white balance / gamma / CCM) and returns an RGB8 view — the same path the
# Qt viewer uses. Without it, a color frame comes back as a raw single-channel
# mosaic that the caller must demosaic themselves.
pipeline_create       = _bind("u3v_pipeline_create",       c_int,    POINTER(u3v_pipeline_p))
pipeline_destroy      = _bind("u3v_pipeline_destroy",      None,     u3v_pipeline_p)
pipeline_load_dir     = _bind("u3v_pipeline_load_dir",     c_int,    u3v_pipeline_p, c_char_p)
pipeline_plugin_count = _bind("u3v_pipeline_plugin_count", c_uint32, u3v_pipeline_p)
stream_set_pipeline   = _bind("u3v_stream_set_pipeline",   c_int,    u3v_stream_p, u3v_pipeline_p)
stream_grab_view      = _bind("u3v_stream_grab_view",      c_int,    u3v_stream_p, POINTER(u3v_buffer_t))

# Run the loaded chain over an arbitrary buffer. This is what makes offline
# processing possible: capture raw Bayer now, demosaic + correct it later
# through the very same plugins the live path uses, so live and offline colour
# match exactly. `out` points at pipeline-owned scratch that stays valid only
# until the next process() call on the same pipeline.
pipeline_process      = _bind("u3v_pipeline_process",      c_int, u3v_pipeline_p, POINTER(u3v_buffer_t), POINTER(u3v_buffer_t))
pipeline_set_enabled  = _bind("u3v_pipeline_set_enabled",  c_int, u3v_pipeline_p, c_uint32, c_int)
pipeline_is_enabled   = _bind("u3v_pipeline_is_enabled",   c_int, u3v_pipeline_p, c_uint32)
pipeline_plugin_info  = _bind("u3v_pipeline_plugin_info",  POINTER(u3v_plugin_info_t), u3v_pipeline_p, c_uint32)

# Plugin parameter I/O (white balance, per-channel gain, gamma, CCM, ...). The
# `value` pointer is interpreted per the parameter's declared type — read it
# from the descriptor returned by pipeline_plugin_param_get(), never guess it
# from the parameter name. Buttons take a NULL value.
pipeline_plugin_param_count     = _bind("u3v_pipeline_plugin_param_count",     c_uint32, u3v_pipeline_p, c_uint32)
pipeline_plugin_param_get       = _bind("u3v_pipeline_plugin_param_get",       POINTER(u3v_param_t), u3v_pipeline_p, c_uint32, c_uint32)
pipeline_plugin_param_set       = _bind("u3v_pipeline_plugin_param_set",       c_int,    u3v_pipeline_p, c_uint32, c_char_p, c_void_p)
pipeline_plugin_param_get_value = _bind("u3v_pipeline_plugin_param_get_value", c_int,    u3v_pipeline_p, c_uint32, c_char_p, c_void_p)


__all__ = [
    # status helpers
    "status_str",
    # status constants
    "U3V_OK", "U3V_ERR_INVALID_PARAM", "U3V_ERR_NO_DEVICE", "U3V_ERR_USB_OPEN",
    "U3V_ERR_USB_IO", "U3V_ERR_TIMEOUT", "U3V_ERR_PROTOCOL", "U3V_ERR_NO_MEMORY",
    "U3V_ERR_NOT_CONNECTED", "U3V_ERR_BUSY", "U3V_ERR_ABORTED",
    "U3V_ERR_BUFFER_TOO_SMALL", "U3V_ERR_STREAM", "U3V_ERR_INCOMPLETE",
    # pixel formats
    "PFNC_MONO8", "PFNC_MONO10", "PFNC_MONO10_PACKED", "PFNC_MONO12",
    "PFNC_MONO12_PACKED", "PFNC_MONO16", "PFNC_BAYERGR8", "PFNC_BAYERRG8",
    "PFNC_BAYERGB8", "PFNC_BAYERBG8",
    "PFNC_BAYERGR10", "PFNC_BAYERRG10", "PFNC_BAYERGB10", "PFNC_BAYERBG10",
    "PFNC_BAYERGR12", "PFNC_BAYERRG12", "PFNC_BAYERGB12", "PFNC_BAYERBG12",
    "PFNC_RGB8", "PFNC_YUV422_8", "PFNC_BPP",
    # IMX296 enums
    "IMX296_ACQ_MODE_CONTINUOUS", "IMX296_ACQ_MODE_SINGLE", "IMX296_ACQ_MODE_MULTI",
    "IMX296_TRIGGER_SEL_ACQ_START", "IMX296_TRIGGER_SEL_FRAME_START",
    "IMX296_TRIGGER_MODE_OFF", "IMX296_TRIGGER_MODE_ON",
    "IMX296_TRIGGER_SRC_LINE0", "IMX296_TRIGGER_SRC_LINE1",
    "IMX296_TRIGGER_SRC_LINE2", "IMX296_TRIGGER_SRC_LINE3",
    "IMX296_TRIGGER_SRC_SOFTWARE",
    "IMX296_TRIGGER_ACT_FALLING_EDGE", "IMX296_TRIGGER_ACT_RISING_EDGE",
    # ISP plugin parameter types
    "U3V_PARAM_TYPE_BOOL", "U3V_PARAM_TYPE_INT32", "U3V_PARAM_TYPE_UINT32",
    "U3V_PARAM_TYPE_FLOAT", "U3V_PARAM_TYPE_ENUM", "U3V_PARAM_TYPE_MATRIX3X3",
    "U3V_PARAM_TYPE_BUTTON",
    # structs
    "u3v_device_info_t", "u3v_cam_info_t", "u3v_buffer_t", "u3v_stream_config_t",
    "u3v_param_t", "u3v_plugin_info_t",
    "u3v_camera_p", "u3v_stream_p", "u3v_pipeline_p",
    # bound functions
    "get_version", "sdk_init", "sdk_shutdown", "discover",
    "camera_open", "camera_close", "camera_get_info",
    "camera_get_width", "camera_set_width", "camera_get_height", "camera_set_height",
    "camera_get_offset_x", "camera_set_offset_x", "camera_get_offset_y", "camera_set_offset_y",
    "camera_get_pixel_format", "camera_set_pixel_format", "camera_get_payload_size",
    "camera_set_acq_mode", "camera_start", "camera_stop",
    "camera_get_frame_rate", "camera_set_frame_rate",
    "camera_get_frame_count", "camera_set_frame_count",
    "camera_get_exposure", "camera_set_exposure",
    "camera_set_exposure_auto", "camera_get_exposure_auto",
    "camera_get_gain", "camera_set_gain",
    "camera_set_trigger_selector", "camera_get_trigger_mode", "camera_set_trigger_mode",
    "camera_get_trigger_source", "camera_set_trigger_source",
    "camera_set_trigger_activation", "camera_get_trigger_activation",
    "camera_send_software_trigger",
    "camera_set_line_debounce", "camera_get_line_debounce",
    "camera_set_strobe", "camera_get_strobe",
    "camera_get_temperature", "camera_reset", "camera_find_me", "camera_flush_stream",
    "camera_read_reg", "camera_write_reg",
    "stream_create", "stream_destroy", "buffer_alloc", "buffer_free",
    "stream_grab", "buffer_save",
    "pipeline_create", "pipeline_destroy", "pipeline_load_dir",
    "pipeline_plugin_count", "stream_set_pipeline", "stream_grab_view",
    "pipeline_process", "pipeline_set_enabled", "pipeline_is_enabled",
    "pipeline_plugin_info",
    "pipeline_plugin_param_count", "pipeline_plugin_param_get",
    "pipeline_plugin_param_set", "pipeline_plugin_param_get_value",
]
