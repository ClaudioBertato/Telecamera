"""Locate and load the u3v_cam shared library on Windows / Linux."""
from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import List


_LOADED: ctypes.CDLL | None = None


def _platform_subdir() -> str:
    """Map (sys.platform, arch) onto a subdir inside ``u3v_cam/_libs/``."""
    arch = platform.machine().lower()
    if sys.platform == "win32":
        return "windows-x64"
    if sys.platform == "darwin":
        return "macos-arm64" if arch in ("arm64", "aarch64") else "macos-x64"
    if arch in ("aarch64", "arm64"):
        return "linux-arm64"
    return "linux-x64"


def _candidate_paths() -> List[Path]:
    """Search order for the shared library."""
    here = Path(__file__).resolve().parent
    pkg_root = here.parent          # package root (e.g. u3v-sdk-<ver>-python/)
    sdk_root = pkg_root.parent      # extracted SDK root, if the package sits in one

    if sys.platform == "win32":
        names = ["u3v_cam.dll"]
        sub = ["bin"]
        env_var = "U3V_CAM_DLL"
    elif sys.platform == "darwin":
        names = ["libu3v_cam.dylib", "libu3v_cam.2.dylib"]
        sub = ["lib"]
        env_var = "U3V_CAM_LIB"
    else:
        names = ["libu3v_cam.so.2", "libu3v_cam.so"]
        sub = ["lib"]
        env_var = "U3V_CAM_LIB"

    paths: List[Path] = []

    override = os.environ.get(env_var)
    if override:
        paths.append(Path(override))

    bundled = here / "_libs" / _platform_subdir()
    search_dirs = [
        bundled,
        here,
        pkg_root,
        sdk_root,
        *(sdk_root / s for s in sub),
        *(pkg_root / s for s in sub),
    ]
    for d in search_dirs:
        for n in names:
            paths.append(d / n)

    return paths


def plugin_dir() -> str | None:
    """Locate the ISP plugin directory that ships next to the native library.

    Mirrors :func:`_candidate_paths`: the bundled ``_libs/<platform>/plugins``
    is preferred, then the ``plugins`` folder beside the package or a native
    ``bin/``/``lib/`` layout. ``U3V_PLUGIN_DIR`` overrides everything.
    Returns the first existing directory, or ``None`` if no plugins are found.
    """
    here = Path(__file__).resolve().parent
    pkg_root = here.parent
    sdk_root = pkg_root.parent

    candidates: List[Path] = []
    override = os.environ.get("U3V_PLUGIN_DIR")
    if override:
        candidates.append(Path(override))
    candidates += [
        here / "_libs" / _platform_subdir() / "plugins",
        here / "plugins",
        pkg_root / "plugins",
        sdk_root / "bin" / "plugins",
        sdk_root / "lib" / "plugins",
    ]
    for d in candidates:
        if d.is_dir():
            return str(d)
    return None


def _preload_dependencies() -> None:
    """libusb-1.0.dll on Windows must be loadable from the same dir as u3v_cam.dll."""
    if sys.platform != "win32":
        return
    here = Path(__file__).resolve().parent
    candidates = [
        here / "_libs" / _platform_subdir(),
        here,
        here.parent,
        here.parent.parent / "bin",
    ]
    for d in candidates:
        cand = d / "libusb-1.0.dll"
        if cand.is_file():
            try:
                ctypes.WinDLL(str(cand))
            except OSError:
                pass
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(d))
                except OSError:
                    pass
            return


def load_library() -> ctypes.CDLL:
    global _LOADED
    if _LOADED is not None:
        return _LOADED

    _preload_dependencies()

    last_error: Exception | None = None
    for p in _candidate_paths():
        if not p.is_file():
            continue
        try:
            if sys.platform == "win32":
                _LOADED = ctypes.WinDLL(str(p))
            else:
                _LOADED = ctypes.CDLL(str(p))
            return _LOADED
        except OSError as e:
            last_error = e

    arch = platform.machine()
    searched = "\n  ".join(str(p) for p in _candidate_paths()[:6])
    raise OSError(
        f"Failed to locate u3v_cam shared library ({sys.platform}/{arch}).\n"
        f"Set U3V_CAM_{'DLL' if sys.platform == 'win32' else 'LIB'} env var, "
        f"or place the library next to one of:\n  {searched}\n"
        f"Last error: {last_error}"
    )
