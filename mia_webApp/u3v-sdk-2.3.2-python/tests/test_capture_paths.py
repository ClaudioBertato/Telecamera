"""Burst-capture and settings-profile tests.

No camera required: the only device-dependent step is ``Camera._grab()``, so a
stub buffer exercises everything downstream of it, and a stub property set
exercises the settings profile.

Run with::

    cd python
    python -m pytest tests/   # or:  python tests/test_capture_paths.py
"""
from __future__ import annotations

import ctypes
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

import u3v_cam  # noqa: E402
from u3v_cam import _raw as r  # noqa: E402

H, W = 48, 64


class StubGrabCamera(u3v_cam.Camera):
    """Camera whose grab returns a buffer we control.

    ``Camera.__init__`` is deliberately not called: no device is opened.
    """

    def __init__(self, payload: np.ndarray, pixel_format: int,
                 width: int, height: int, short_by: int = 0):
        self._payload = np.ascontiguousarray(payload)
        self._pipeline = None
        self._buf = r.u3v_buffer_t()
        self._buf.data = self._payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._buf.size = self._payload.nbytes
        self._buf.received_size = self._payload.nbytes - short_by
        self._buf.width = width
        self._buf.height = height
        self._buf.pixel_format = pixel_format
        self._buf.block_id = 7
        self._buf.status = 0
        self.grabs = 0

    def _grab(self):
        self.grabs += 1
        return self._buf

    def close(self) -> None:
        pass


def ramp8(h: int = H, w: int = W) -> np.ndarray:
    return (np.arange(h * w, dtype=np.uint32) % 251).astype(np.uint8).reshape(h, w)


class ReadFrameIntoTests(unittest.TestCase):
    """read_frame_into() exists so a 200-frame burst needs one allocation and
    one copy per frame, instead of two copies and a fresh array each time."""

    def test_fills_a_slice_of_a_preallocated_burst(self) -> None:
        src = ramp8()
        cam = StubGrabCamera(src, u3v_cam.PFNC_BAYERRG8, W, H)
        burst = np.zeros((5, H, W), dtype=np.uint8)

        view = cam.read_frame_into(burst[2])

        self.assertEqual(view.shape, (H, W))
        self.assertTrue(np.array_equal(burst[2], src))
        self.assertEqual(burst[1].max(), 0)
        self.assertEqual(burst[3].max(), 0)

    def test_returns_a_view_not_a_copy(self) -> None:
        cam = StubGrabCamera(ramp8(), u3v_cam.PFNC_BAYERRG8, W, H)
        out = np.zeros((H, W), dtype=np.uint8)
        view = cam.read_frame_into(out)
        self.assertTrue(np.shares_memory(view, out))

    def test_whole_burst(self) -> None:
        src = ramp8()
        cam = StubGrabCamera(src, u3v_cam.PFNC_BAYERRG8, W, H)
        frames = np.empty((10, H, W), dtype=np.uint8)
        for i in range(10):
            cam.read_frame_into(frames[i])
        self.assertEqual(cam.grabs, 10)
        for i in range(10):
            self.assertTrue(np.array_equal(frames[i], src))

    def test_rgb_output(self) -> None:
        rgb = (np.arange(H * W * 3, dtype=np.uint32) % 253).astype(np.uint8).reshape(H, W, 3)
        cam = StubGrabCamera(rgb, u3v_cam.PFNC_RGB8, W, H)
        out = np.zeros((H, W, 3), dtype=np.uint8)
        view = cam.read_frame_into(out)
        self.assertEqual(view.shape, (H, W, 3))
        self.assertTrue(np.array_equal(out, rgb))

    def test_sixteen_bit_output(self) -> None:
        src = (np.arange(H * W, dtype=np.uint32) % 4000).astype(np.uint16).reshape(H, W)
        cam = StubGrabCamera(src, u3v_cam.PFNC_MONO12, W, H)
        out = np.zeros((H, W), dtype=np.uint16)
        view = cam.read_frame_into(out)
        self.assertEqual(view.shape, (H, W))
        self.assertTrue(np.array_equal(out, src))

    def test_matches_read_frame(self) -> None:
        cam = StubGrabCamera(ramp8(), u3v_cam.PFNC_BAYERRG8, W, H)
        expected = cam.read_frame()
        out = np.zeros((H, W), dtype=np.uint8)
        self.assertTrue(np.array_equal(cam.read_frame_into(out), expected))

    def test_short_frame_leaves_the_tail_alone(self) -> None:
        src = ramp8()
        cam = StubGrabCamera(src, u3v_cam.PFNC_BAYERRG8, W, H, short_by=W)
        dst = np.full((H, W), 200, dtype=np.uint8)
        cam.read_frame_into(dst)
        self.assertTrue(np.array_equal(dst[:-1], src[:-1]))
        self.assertTrue((dst[-1] == 200).all())


class ReadFrameIntoRejectionTests(unittest.TestCase):
    """A rejected destination must come back untouched, so a caller that
    passed the wrong array has not lost the data already in it."""

    def setUp(self) -> None:
        self.cam = StubGrabCamera(ramp8(), u3v_cam.PFNC_BAYERRG8, W, H)

    def test_too_small(self) -> None:
        small = np.full(16, 9, dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.cam.read_frame_into(small)
        self.assertTrue((small == 9).all())

    def test_wrong_dtype(self) -> None:
        wrong = np.full((H, W), 9, dtype=np.uint16)
        with self.assertRaises(ValueError):
            self.cam.read_frame_into(wrong)
        self.assertTrue((wrong == 9).all())

    def test_non_contiguous(self) -> None:
        with self.assertRaises(ValueError):
            self.cam.read_frame_into(np.zeros((H, W * 2), dtype=np.uint8)[:, ::2])

    def test_read_only(self) -> None:
        ro = np.zeros((H, W), dtype=np.uint8)
        ro.flags.writeable = False
        with self.assertRaises(ValueError):
            self.cam.read_frame_into(ro)

    def test_not_an_array(self) -> None:
        with self.assertRaises(TypeError):
            self.cam.read_frame_into([0] * (H * W))


class StubSettingsCamera(u3v_cam.Camera):
    """Camera that records setting writes instead of performing them."""

    FIELDS = ("width", "height", "offset_x", "offset_y", "pixel_format",
              "exposure_us", "gain")

    def __init__(self) -> None:
        self._pipeline = None
        self.log: list = []
        self.state = {"width": 1456, "height": 1088, "offset_x": 0,
                      "offset_y": 0, "pixel_format": u3v_cam.PFNC_MONO8,
                      "exposure_us": 10000, "gain": 0}

    def close(self) -> None:
        pass


def _stub_property(field: str) -> property:
    def getter(self):
        return self.state[field]

    def setter(self, value):
        self.log.append((field, int(value)))
        self.state[field] = int(value)

    return property(getter, setter)


for _field in StubSettingsCamera.FIELDS:
    setattr(StubSettingsCamera, _field, _stub_property(_field))


class CameraProfileTests(unittest.TestCase):
    def test_snapshot_captures_every_field(self) -> None:
        cam = StubSettingsCamera()
        cam.state.update(width=1024, height=600, exposure_us=8000, gain=16,
                         pixel_format=u3v_cam.PFNC_BAYERRG8)
        profile = cam.get_camera_profile()
        self.assertEqual(set(profile), set(StubSettingsCamera.FIELDS))
        self.assertEqual(profile["width"], 1024)
        self.assertEqual(profile["gain"], 16)

    def test_pixel_format_is_written_before_the_roi(self) -> None:
        """Payload size depends on both, so the format has to land first."""
        cam = StubSettingsCamera()
        cam.apply_camera_profile({
            "width": 1024, "height": 600, "offset_x": 208, "offset_y": 244,
            "pixel_format": u3v_cam.PFNC_BAYERRG8, "exposure_us": 8000, "gain": 16,
        })
        order = [field for field, _ in cam.log]
        self.assertLess(order.index("pixel_format"), order.index("width"))

    def test_offsets_are_cleared_before_the_new_size(self) -> None:
        cam = StubSettingsCamera()
        cam.state.update(offset_x=400, offset_y=400)
        cam.apply_camera_profile({"width": 1024, "height": 600,
                                  "offset_x": 208, "offset_y": 244})
        order = [field for field, _ in cam.log]
        self.assertEqual(cam.log[order.index("offset_x")], ("offset_x", 0))
        self.assertLess(order.index("offset_x"), order.index("width"))
        self.assertEqual(cam.state["offset_x"], 208)
        self.assertEqual(cam.state["offset_y"], 244)

    def test_round_trip_is_stable(self) -> None:
        source = StubSettingsCamera()
        source.state.update(width=1024, height=600, exposure_us=8000, gain=16,
                            pixel_format=u3v_cam.PFNC_BAYERRG8)
        profile = source.get_camera_profile()
        target = StubSettingsCamera()
        target.apply_camera_profile(profile)
        self.assertEqual(target.get_camera_profile(), profile)

    def test_partial_profile(self) -> None:
        cam = StubSettingsCamera()
        cam.apply_camera_profile({"exposure_us": 3000})
        self.assertEqual(cam.state["exposure_us"], 3000)
        self.assertEqual(cam.state["width"], 1456)

    def test_unknown_key_raises(self) -> None:
        cam = StubSettingsCamera()
        with self.assertRaises(ValueError) as ctx:
            cam.apply_camera_profile({"exposure_us": 1, "shutter_angle": 180})
        self.assertIn("shutter_angle", str(ctx.exception))

    def test_isp_section_without_a_pipeline_is_reported(self) -> None:
        """Silently dropping the colour half of a profile would ship the wrong
        colour, so it is reported instead."""
        cam = StubSettingsCamera()
        skipped = cam.apply_profile({"camera": {"gain": 8},
                                     "isp": {"gamma.value": 2.2,
                                             "rgb.enabled": True}})
        self.assertEqual(cam.state["gain"], 8)
        self.assertEqual(sorted(skipped), ["isp.gamma.value", "isp.rgb.enabled"])

    def test_strict_mode_raises_without_a_pipeline(self) -> None:
        cam = StubSettingsCamera()
        with self.assertRaises(RuntimeError):
            cam.apply_profile({"isp": {"gamma.value": 2.2}}, strict=True)


if __name__ == "__main__":
    unittest.main()
