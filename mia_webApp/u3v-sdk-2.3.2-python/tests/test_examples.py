"""Smoke tests for the shipped examples — no camera required.

``burst_capture_raw.py`` needs hardware, so only its syntax is checked here.
``offline_to_color.py`` is host-side end to end, so it runs for real against a
synthetic burst.

Run with::

    cd python
    python -m pytest tests/   # or:  python tests/test_examples.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES = os.path.join(PYTHON_DIR, "examples")
sys.path.insert(0, PYTHON_DIR)

import numpy as np  # noqa: E402

import u3v_cam  # noqa: E402


def _plugins_available() -> bool:
    try:
        u3v_cam.Processor().close()
        return True
    except Exception:
        return False


HAVE_PLUGINS = _plugins_available()


def bayer_burst(count: int = 4, h: int = 48, w: int = 64) -> np.ndarray:
    burst = np.zeros((count, h, w), dtype=np.uint8)
    _, xx = np.mgrid[0:h, 0:w]
    ramp = ((xx / max(w - 1, 1)) * 200 + 30).astype(np.uint8)
    for i in range(count):
        burst[i, 0::2, 0::2] = ramp[0::2, 0::2]
        burst[i, 0::2, 1::2] = 128
        burst[i, 1::2, 0::2] = 128
        burst[i, 1::2, 1::2] = 255 - ramp[1::2, 1::2]
    return burst


def run_example(name: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = PYTHON_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, os.path.join(EXAMPLES, name), *args],
                          capture_output=True, text=True, env=env)


class ExampleSyntaxTests(unittest.TestCase):
    def test_all_examples_compile(self) -> None:
        for name in sorted(f for f in os.listdir(EXAMPLES) if f.endswith(".py")):
            with self.subTest(example=name):
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile",
                     os.path.join(EXAMPLES, name)],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


class ExampleCliTests(unittest.TestCase):
    """``--help`` exercises the imports and the argument parser without
    touching a camera, which ``py_compile`` cannot do."""

    CLI_EXAMPLES = ("burst_capture_raw.py", "offline_to_color.py",
                    "frameloss_freerun.py", "frameloss_trigger.py")

    def test_help_runs(self) -> None:
        for name in self.CLI_EXAMPLES:
            with self.subTest(example=name):
                result = run_example(name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage", result.stdout.lower())


class SampleProfileTests(unittest.TestCase):
    PATH = os.path.join(EXAMPLES, "camera_profile.json")

    def test_is_valid_json_with_both_sections(self) -> None:
        profile = u3v_cam.load_profile(self.PATH)
        self.assertIn("camera", profile)
        self.assertIn("isp", profile)

    @unittest.skipUnless(HAVE_PLUGINS, "ISP plugins not found")
    def test_every_isp_key_is_recognised(self) -> None:
        """A shipped sample that silently skips half its settings would teach
        customers the wrong parameter names."""
        profile = u3v_cam.load_profile(self.PATH)
        with u3v_cam.Processor() as isp:
            self.assertEqual(isp.apply_isp_profile(profile["isp"]), [])
            self.assertAlmostEqual(isp.get_color_param("gamma.value"), 1.8, places=5)
            self.assertAlmostEqual(isp.get_color_param("rgb.red_gain"), 1.6, places=5)


@unittest.skipUnless(HAVE_PLUGINS, "ISP plugins not found")
class OfflineToColorTests(unittest.TestCase):
    """The offline converter is the second half of the raw-capture workflow,
    so it has to run end to end without a camera."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.count, self.h, self.w = 4, 48, 64
        self.burst = bayer_burst(self.count, self.h, self.w)
        self.raw = os.path.join(self.dir, "burst_TEST1234.npy")
        np.save(self.raw, self.burst)
        self.profile = u3v_cam.load_profile(
            os.path.join(EXAMPLES, "camera_profile.json"))
        with open(os.path.join(self.dir, "burst_TEST1234.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"serial": "TEST1234", "width": self.w, "height": self.h,
                       "frames": self.count, "pixel_format": "bayerrg8",
                       "profile": self.profile}, fh)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _convert(self, *extra: str) -> str:
        out = os.path.join(self.dir, "rgb" + str(len(extra)))
        result = run_example("offline_to_color.py",
                             "--input", self.raw, "--outdir", out, *extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        return out

    def test_writes_one_image_per_frame(self) -> None:
        out = self._convert()
        self.assertEqual(len(os.listdir(out)), self.count)

    def test_limit_is_respected(self) -> None:
        out = self._convert("--limit", "2")
        self.assertEqual(len(os.listdir(out)), 2)

    def test_output_is_valid_ppm(self) -> None:
        out = self._convert()
        path = os.path.join(out, sorted(os.listdir(out))[0])
        with open(path, "rb") as fh:
            self.assertTrue(fh.readline().startswith(b"P6"))
            fh.readline()
            fh.readline()
            data = fh.read()
        self.assertEqual(len(data), self.h * self.w * 3)

    def test_matches_the_live_colour_path(self) -> None:
        """Offline conversion must reproduce the live ISP exactly, otherwise
        capturing raw would change the colour."""
        out = self._convert()
        path = os.path.join(out, sorted(os.listdir(out))[0])
        with open(path, "rb") as fh:
            for _ in range(3):
                fh.readline()
            got = np.frombuffer(fh.read(), dtype=np.uint8).reshape(self.h, self.w, 3)

        with u3v_cam.Processor() as reference:
            reference.apply_isp_profile(self.profile["isp"])
            expected = reference.process(self.burst[0], u3v_cam.PFNC_BAYERRG8)

        self.assertTrue(np.array_equal(got, expected))


@unittest.skipUnless(HAVE_PLUGINS, "ISP plugins not found")
class OfflineMonoTests(unittest.TestCase):
    """A mono burst has no colour to reconstruct, so the ISP returns it
    two-dimensional. Writing that with an RGB header produces a file a third
    of the size the header claims, which most viewers open as garbage rather
    than reject."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.count, self.h, self.w = 3, 48, 64
        self.burst = (np.arange(self.count * self.h * self.w, dtype=np.uint32)
                      % 256).astype(np.uint8).reshape(self.count, self.h, self.w)
        self.raw = os.path.join(self.dir, "burst_MONO0001.npy")
        np.save(self.raw, self.burst)
        with open(os.path.join(self.dir, "burst_MONO0001.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"serial": "MONO0001", "width": self.w, "height": self.h,
                       "frames": self.count, "pixel_format": "mono8",
                       "profile": {}}, fh)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mono_burst_writes_valid_pgm(self) -> None:
        out = os.path.join(self.dir, "gray")
        result = run_example("offline_to_color.py",
                             "--input", self.raw, "--outdir", out)
        self.assertEqual(result.returncode, 0, result.stderr)

        names = sorted(os.listdir(out))
        self.assertEqual(len(names), self.count)
        self.assertTrue(all(n.endswith(".pgm") for n in names), names)

        with open(os.path.join(out, names[0]), "rb") as fh:
            self.assertTrue(fh.readline().startswith(b"P5"))
            self.assertEqual(fh.readline().split(), [str(self.w).encode(),
                                                     str(self.h).encode()])
            fh.readline()
            data = fh.read()
        self.assertEqual(len(data), self.h * self.w)
        self.assertTrue(np.array_equal(
            np.frombuffer(data, dtype=np.uint8).reshape(self.h, self.w),
            self.burst[0]))


if __name__ == "__main__":
    unittest.main()
