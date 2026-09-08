"""Offline ISP and typed plugin-parameter tests.

No camera required — the whole ISP path is host-side. The plugins do have to
be findable (bundled ``_libs/<platform>/plugins`` or ``U3V_PLUGIN_DIR``);
tests skip themselves when they are not.

Run with::

    cd python
    python -m pytest tests/   # or:  python tests/test_isp_offline.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

import u3v_cam  # noqa: E402


def _plugins_available() -> bool:
    try:
        u3v_cam.Processor().close()
        return True
    except Exception:
        return False


HAVE_PLUGINS = _plugins_available()
skip_no_plugins = unittest.skipUnless(HAVE_PLUGINS, "ISP plugins not found")


def bayer_rg8(h: int = 64, w: int = 64) -> np.ndarray:
    """Synthetic BayerRG8 mosaic: R ramps up, B ramps down, G flat mid-grey."""
    img = np.zeros((h, w), dtype=np.uint8)
    _, xx = np.mgrid[0:h, 0:w]
    ramp = ((xx / max(w - 1, 1)) * 200 + 30).astype(np.uint8)
    img[0::2, 0::2] = ramp[0::2, 0::2]
    img[0::2, 1::2] = 128
    img[1::2, 0::2] = 128
    img[1::2, 1::2] = 255 - ramp[1::2, 1::2]
    return img


@skip_no_plugins
class ParameterReflectionTests(unittest.TestCase):
    """Parameter types must come from the plugin descriptors, never from a
    guess based on the parameter name — a float written as an int is not
    rejected by the plugin, it is reinterpreted into a garbage value."""

    def setUp(self) -> None:
        self.isp = u3v_cam.Processor()

    def tearDown(self) -> None:
        self.isp.close()

    def test_expected_parameters_are_exposed(self) -> None:
        names = {p["name"] for p in self.isp.list_color_params()}
        for expected in ("demosaic.algorithm", "rgb.enabled", "rgb.red_gain",
                         "rgb.red_offset", "rgb.white_balance", "gamma.value",
                         "ccm.matrix", "histogram.dark_point"):
            self.assertIn(expected, names)

    def test_declared_types(self) -> None:
        params = {p["name"]: p for p in self.isp.list_color_params()}
        for name, type_name in (
            ("rgb.red_gain", "float"),
            ("rgb.red_offset", "int"),
            ("rgb.enabled", "bool"),
            ("rgb.white_balance", "button"),
            ("gamma.value", "float"),
            ("ccm.matrix", "matrix3x3"),
            ("histogram.dark_point", "uint"),
            ("demosaic.algorithm", "enum"),
        ):
            self.assertEqual(params[name]["type_name"], type_name, name)

    def test_float_round_trip(self) -> None:
        self.isp.set_color_param("gamma.value", 2.2)
        self.assertAlmostEqual(self.isp.get_color_param("gamma.value"), 2.2, places=5)

    def test_white_balance_round_trip(self) -> None:
        self.isp.set_white_balance(1.75, 1.0, 1.25)
        wb = self.isp.get_white_balance()
        self.assertAlmostEqual(wb["red"], 1.75, places=5)
        self.assertAlmostEqual(wb["blue"], 1.25, places=5)

    def test_signed_int_round_trip(self) -> None:
        self.isp.set_color_param("rgb.red_offset", -40)
        self.assertEqual(self.isp.get_color_param("rgb.red_offset"), -40)

    def test_uint_bool_enum_round_trip(self) -> None:
        self.isp.set_color_param("histogram.dark_point", 12)
        self.assertEqual(self.isp.get_color_param("histogram.dark_point"), 12)
        self.isp.set_color_param("demosaic.algorithm", 1)
        self.assertEqual(self.isp.get_color_param("demosaic.algorithm"), 1)
        self.isp.set_color_param("rgb.enabled", True)
        self.assertIs(self.isp.get_color_param("rgb.enabled"), True)

    def test_matrix_round_trip(self) -> None:
        m = [[1.2, -0.1, -0.1], [-0.2, 1.4, -0.2], [0.0, -0.3, 1.3]]
        self.isp.set_ccm(m)
        self.assertTrue(np.allclose(np.array(self.isp.get_color_param("ccm.matrix")),
                                    np.array(m), atol=1e-5))

    def test_wrong_sized_matrix_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.isp.set_color_param("ccm.matrix", [1, 2, 3])

    def test_unknown_parameter_raises(self) -> None:
        with self.assertRaises(u3v_cam.U3VError):
            self.isp.set_color_param("nope.not_a_param", 1)
        with self.assertRaises(u3v_cam.U3VError):
            self.isp.get_color_param("nope.not_a_param")

    def test_buttons_need_no_value(self) -> None:
        self.isp.set_color_param("rgb.white_balance")
        self.assertIsNone(self.isp.get_color_param("rgb.white_balance"))


@skip_no_plugins
class OfflineProcessTests(unittest.TestCase):
    """Capture raw Bayer during acquisition, colour it afterwards through the
    same plugins the live path uses."""

    def test_demosaic_produces_rgb(self) -> None:
        src = bayer_rg8()
        with u3v_cam.Processor() as isp:
            rgb = isp.process(src, u3v_cam.PFNC_BAYERRG8)
        self.assertEqual(rgb.shape, (64, 64, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertLess(rgb[0, 0, 0], rgb[0, -1, 0])   # red follows the ramp

    def test_input_is_not_modified(self) -> None:
        src = bayer_rg8()
        before = src.copy()
        with u3v_cam.Processor() as isp:
            isp.process(src, u3v_cam.PFNC_BAYERRG8)
        self.assertTrue(np.array_equal(src, before))

    def test_mono_passes_through(self) -> None:
        mono = (np.arange(64 * 64, dtype=np.uint32) % 256).astype(np.uint8).reshape(64, 64)
        with u3v_cam.Processor() as isp:
            out = isp.process(mono, u3v_cam.PFNC_MONO8)
        self.assertEqual(out.shape, (64, 64))
        self.assertTrue(np.array_equal(out, mono))

    def test_gain_changes_only_its_own_channel(self) -> None:
        src = bayer_rg8()
        with u3v_cam.Processor() as a, u3v_cam.Processor() as b:
            base = a.process(src, u3v_cam.PFNC_BAYERRG8).astype(np.int32)
            b.set_white_balance(2.0, 1.0, 1.0)
            boosted = b.process(src, u3v_cam.PFNC_BAYERRG8).astype(np.int32)
        self.assertGreater(boosted[:, :, 0].mean(), base[:, :, 0].mean() + 1)
        self.assertLess(abs(boosted[:, :, 2].mean() - base[:, :, 2].mean()), 1.0)

    def test_gamma_moves_midtones_both_ways(self) -> None:
        src = bayer_rg8()
        with u3v_cam.Processor() as a, u3v_cam.Processor() as b, u3v_cam.Processor() as c:
            base = a.process(src, u3v_cam.PFNC_BAYERRG8).astype(np.int32).mean()
            b.set_gamma(2.2)
            bright = b.process(src, u3v_cam.PFNC_BAYERRG8).astype(np.int32).mean()
            c.set_gamma(0.5)
            dark = c.process(src, u3v_cam.PFNC_BAYERRG8).astype(np.int32).mean()
        self.assertGreater(bright, base + 5)
        self.assertLess(dark, base - 5)

    def test_frame_can_be_processed_twice(self) -> None:
        src = bayer_rg8()
        with u3v_cam.Processor() as isp:
            first = isp.process(src, u3v_cam.PFNC_BAYERRG8)
            second = isp.process(src, u3v_cam.PFNC_BAYERRG8)
        self.assertTrue(np.array_equal(first, second))


@skip_no_plugins
class ProfileTests(unittest.TestCase):
    """A profile is the contract that makes every camera, and every later
    re-run, produce the same colour from the same raw data."""

    def _configured(self) -> u3v_cam.Processor:
        isp = u3v_cam.Processor()
        isp.set_white_balance(1.6, 1.0, 1.35)
        isp.set_gamma(1.8)
        isp.set_levels(10, 240)
        isp.set_ccm([[1.2, -0.1, -0.1], [-0.2, 1.4, -0.2], [0.0, -0.3, 1.3]])
        return isp

    def test_profile_excludes_buttons(self) -> None:
        with self._configured() as isp:
            profile = isp.get_isp_profile()
        for name in profile:
            self.assertFalse(name.endswith(("white_balance", "reset",
                                            "auto_configure")), name)

    def test_same_profile_gives_identical_pixels(self) -> None:
        src = bayer_rg8()
        with self._configured() as ref:
            profile = ref.get_isp_profile()
            expected = ref.process(src, u3v_cam.PFNC_BAYERRG8)
        with u3v_cam.Processor() as clone:
            self.assertEqual(clone.apply_isp_profile(profile), [])
            got = clone.process(src, u3v_cam.PFNC_BAYERRG8)
        self.assertTrue(np.array_equal(got, expected))

    def test_profile_survives_json(self) -> None:
        with self._configured() as isp:
            profile = isp.get_isp_profile()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "profile.json")
            u3v_cam.save_profile(path, {"isp": profile})
            self.assertEqual(u3v_cam.load_profile(path)["isp"], profile)
            with open(path, encoding="utf-8") as fh:
                json.load(fh)   # must be plain, human-editable JSON

    def test_unknown_key_is_reported_not_fatal(self) -> None:
        with u3v_cam.Processor() as isp:
            skipped = isp.apply_isp_profile({"gamma.value": 2.0, "future.param": 1})
            self.assertEqual(skipped, ["future.param"])
            self.assertAlmostEqual(isp.get_color_param("gamma.value"), 2.0, places=5)

    def test_strict_mode_raises_on_unknown_key(self) -> None:
        with u3v_cam.Processor() as isp:
            with self.assertRaises(u3v_cam.U3VError):
                isp.apply_isp_profile({"future.param": 1}, strict=True)


class NoPipelineTests(unittest.TestCase):
    def test_processor_reports_missing_plugins_clearly(self) -> None:
        """A missing plugin folder must fail loudly, not silently pass frames
        through undemosaiced."""
        with self.assertRaises(FileNotFoundError):
            u3v_cam.Processor(plugin_dir=os.path.join(tempfile.gettempdir(),
                                                      "u3v-no-such-plugins"))


if __name__ == "__main__":
    unittest.main()
