"""Smoke tests that don't require a connected camera.

Run with::

    cd python
    python -m pytest tests/   # or:  python tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys
import unittest


# Make the package importable when running directly from this dir.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class LibraryLoadTests(unittest.TestCase):
    def test_loader_finds_library(self) -> None:
        from u3v_cam._loader import load_library
        lib = load_library()
        self.assertIsNotNone(lib)

    def test_raw_binds_all_expected_symbols(self) -> None:
        from u3v_cam import _raw as r
        # spot-check a representative subset of the public C API
        for name in [
            "sdk_init", "sdk_shutdown", "discover",
            "camera_open", "camera_close", "camera_get_info",
            "camera_set_width", "camera_set_height",
            "camera_start", "camera_stop",
            "camera_set_exposure", "camera_set_gain",
            "camera_set_trigger_mode", "camera_send_software_trigger",
            "stream_create", "stream_destroy",
            "buffer_alloc", "buffer_free", "stream_grab", "buffer_save",
        ]:
            self.assertTrue(hasattr(r, name), f"missing binding: {name}")
            fn = getattr(r, name)
            self.assertTrue(callable(fn), f"{name} not callable")

    def test_status_str(self) -> None:
        from u3v_cam._raw import status_str, U3V_OK, U3V_ERR_TIMEOUT
        self.assertEqual(status_str(U3V_OK), "OK")
        self.assertEqual(status_str(U3V_ERR_TIMEOUT), "Timeout")
        self.assertIn("Unknown", status_str(-999))


class StructLayoutTests(unittest.TestCase):
    def test_buffer_struct_size(self) -> None:
        import ctypes
        from u3v_cam._raw import u3v_buffer_t
        # ptr(8) + 4 + 4 + 4 + 4 + 4 + 8 + 8 + 2 = 46 -> 48 with padding on 64-bit
        # Just ensure the struct is non-trivial and matches sane bounds.
        size = ctypes.sizeof(u3v_buffer_t)
        self.assertGreaterEqual(size, 40)
        self.assertLessEqual(size, 64)


if __name__ == "__main__":
    unittest.main()
