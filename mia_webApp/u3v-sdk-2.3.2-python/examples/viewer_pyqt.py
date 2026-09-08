"""viewer_pyqt.py — PyQt6 + pyqtgraph live viewer.

Reference implementation, intended to be copied and modified.  Shows live
images, lets the user adjust exposure / gain / ROI / trigger / FPS, and
displays per-second FPS / dropped statistics.

Install dependencies::

    pip install u3v_cam pyqt6 pyqtgraph

Run::

    python viewer_pyqt.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# Running this as `python examples/<script>.py` puts examples/ on sys.path, not
# the package root, so make the package importable however it is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import u3v_cam


# --------------------------------------------------------------------------
# Background grab thread
# --------------------------------------------------------------------------
class GrabThread(QtCore.QThread):
    frame_ready  = QtCore.pyqtSignal(np.ndarray, int, int)   # frame, block_id, status
    error        = QtCore.pyqtSignal(str)

    def __init__(self, cam: u3v_cam.Camera) -> None:
        super().__init__()
        self._cam = cam
        self._running = False

    def run(self) -> None:
        self._running = True
        try:
            self._cam.start()
        except u3v_cam.U3VError as e:
            self.error.emit(str(e))
            return
        while self._running:
            try:
                frame = self._cam.read_frame(copy=True)
            except u3v_cam.U3VError as e:
                self.error.emit(str(e))
                continue
            self.frame_ready.emit(frame,
                                  self._cam.last_frame_block_id,
                                  self._cam.last_frame_status)
        try:
            self._cam.stop()
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        self.wait(2000)


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, cam: u3v_cam.Camera) -> None:
        super().__init__()
        self._cam = cam
        self._grabber: Optional[GrabThread] = None

        self._frame_count = 0
        self._drop_count = 0
        self._last_block: Optional[int] = None
        self._fps_window_t0 = time.monotonic()
        self._fps_window_n = 0
        self._fps_value = 0.0

        self._build_ui()
        self._populate_from_camera()

        self._fps_timer = QtCore.QTimer(self)
        self._fps_timer.timeout.connect(self._update_status)
        self._fps_timer.start(500)

    # ---- UI layout ----
    def _build_ui(self) -> None:
        info = self._cam.info
        self.setWindowTitle(f"u3v_cam viewer — {info['model']} (S/N {info['serial']})")
        self.resize(1200, 800)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)

        self._image_view = pg.ImageView()
        self._image_view.ui.histogram.hide()
        self._image_view.ui.roiBtn.hide()
        self._image_view.ui.menuBtn.hide()
        self._image_view.getView().invertY(True)
        layout.addWidget(self._image_view, stretch=1)

        # ---- side panel ----
        side = QtWidgets.QWidget()
        side.setFixedWidth(280)
        v = QtWidgets.QVBoxLayout(side)

        v.addWidget(self._make_camera_info_box())
        v.addWidget(self._make_acquisition_box())
        v.addWidget(self._make_exposure_gain_box())
        v.addWidget(self._make_roi_box())
        v.addWidget(self._make_trigger_box())
        v.addWidget(self._make_misc_box())
        v.addStretch(1)

        layout.addWidget(side)
        self.setCentralWidget(central)

        sb = self.statusBar()
        self._status_label = QtWidgets.QLabel("ready")
        sb.addPermanentWidget(self._status_label, 1)

    def _make_camera_info_box(self) -> QtWidgets.QGroupBox:
        info = self._cam.info
        box = QtWidgets.QGroupBox("Camera")
        f = QtWidgets.QFormLayout(box)
        f.addRow("Vendor:", QtWidgets.QLabel(info["manufacturer"]))
        f.addRow("Model:",  QtWidgets.QLabel(info["model"]))
        f.addRow("Serial:", QtWidgets.QLabel(info["serial"]))
        f.addRow("Sensor:", QtWidgets.QLabel(f"{info['sensor_width']} x {info['sensor_height']}"))
        return box

    def _make_acquisition_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Acquisition")
        h = QtWidgets.QHBoxLayout(box)
        self._start_btn = QtWidgets.QPushButton("Start")
        self._stop_btn  = QtWidgets.QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        h.addWidget(self._start_btn)
        h.addWidget(self._stop_btn)
        return box

    def _make_exposure_gain_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Exposure / Gain / FPS")
        f = QtWidgets.QFormLayout(box)
        self._exp_spin = QtWidgets.QSpinBox()
        self._exp_spin.setRange(10, 1_000_000)
        self._exp_spin.setSuffix(" us")
        self._exp_spin.editingFinished.connect(
            lambda: self._safe_set("exposure_us", self._exp_spin.value()))
        self._gain_spin = QtWidgets.QSpinBox()
        self._gain_spin.setRange(0, 480)
        self._gain_spin.editingFinished.connect(
            lambda: self._safe_set("gain", self._gain_spin.value()))
        self._fps_spin = QtWidgets.QSpinBox()
        self._fps_spin.setRange(1, 240)
        self._fps_spin.editingFinished.connect(
            lambda: self._safe_set("frame_rate", self._fps_spin.value()))
        f.addRow("Exposure:", self._exp_spin)
        f.addRow("Gain:",     self._gain_spin)
        f.addRow("FPS:",      self._fps_spin)
        return box

    def _make_roi_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("ROI")
        f = QtWidgets.QFormLayout(box)
        self._w_spin  = QtWidgets.QSpinBox(); self._w_spin.setRange(8, 4096)
        self._h_spin  = QtWidgets.QSpinBox(); self._h_spin.setRange(8, 4096)
        self._ox_spin = QtWidgets.QSpinBox(); self._ox_spin.setRange(0, 4096)
        self._oy_spin = QtWidgets.QSpinBox(); self._oy_spin.setRange(0, 4096)
        apply_btn = QtWidgets.QPushButton("Apply ROI")
        apply_btn.clicked.connect(self._on_apply_roi)
        f.addRow("Width:",    self._w_spin)
        f.addRow("Height:",   self._h_spin)
        f.addRow("Offset X:", self._ox_spin)
        f.addRow("Offset Y:", self._oy_spin)
        f.addRow(apply_btn)
        return box

    def _make_trigger_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Trigger")
        v = QtWidgets.QVBoxLayout(box)
        self._trig_check = QtWidgets.QCheckBox("Software trigger mode")
        self._trig_check.toggled.connect(self._on_toggle_trigger)
        self._sw_trig_btn = QtWidgets.QPushButton("Send software trigger")
        self._sw_trig_btn.setEnabled(False)
        self._sw_trig_btn.clicked.connect(self._on_send_software_trigger)
        v.addWidget(self._trig_check)
        v.addWidget(self._sw_trig_btn)
        return box

    def _make_misc_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Misc")
        v = QtWidgets.QVBoxLayout(box)
        find_btn = QtWidgets.QPushButton("Find-me LED")
        find_btn.clicked.connect(self._on_find_me)
        v.addWidget(find_btn)
        return box

    # ---- camera <-> UI sync ----
    def _populate_from_camera(self) -> None:
        try:
            self._exp_spin.setValue(self._cam.exposure_us)
            self._gain_spin.setValue(self._cam.gain)
            self._fps_spin.setValue(self._cam.frame_rate)
            self._w_spin.setValue(self._cam.width)
            self._h_spin.setValue(self._cam.height)
            self._ox_spin.setValue(self._cam.offset_x)
            self._oy_spin.setValue(self._cam.offset_y)
            self._trig_check.setChecked(self._cam.trigger_mode)
            self._sw_trig_btn.setEnabled(self._cam.trigger_mode)
        except u3v_cam.U3VError as e:
            self._status_label.setText(f"populate error: {e}")

    def _safe_set(self, attr: str, value) -> None:
        try:
            setattr(self._cam, attr, value)
        except u3v_cam.U3VError as e:
            self._status_label.setText(f"set {attr}: {e}")

    # ---- handlers ----
    def _on_start(self) -> None:
        if self._grabber is not None:
            return
        self._grabber = GrabThread(self._cam)
        self._grabber.frame_ready.connect(self._on_frame)
        self._grabber.error.connect(self._on_grab_error)
        self._grabber.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._frame_count = 0
        self._drop_count = 0
        self._last_block = None
        self._fps_window_t0 = time.monotonic()
        self._fps_window_n = 0

    def _on_stop(self) -> None:
        if self._grabber is None:
            return
        self._grabber.stop()
        self._grabber = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _on_apply_roi(self) -> None:
        was_running = self._grabber is not None
        if was_running:
            self._on_stop()
        try:
            self._cam.set_roi(self._w_spin.value(),  self._h_spin.value(),
                              self._ox_spin.value(), self._oy_spin.value())
            self._populate_from_camera()
        except u3v_cam.U3VError as e:
            self._status_label.setText(f"set_roi: {e}")
        if was_running:
            self._on_start()

    def _on_toggle_trigger(self, on: bool) -> None:
        was_running = self._grabber is not None
        if was_running:
            self._on_stop()
        try:
            if on:
                self._cam.configure_trigger(on=True, source="software")
            else:
                self._cam.configure_trigger(on=False)
        except u3v_cam.U3VError as e:
            self._status_label.setText(f"trigger: {e}")
        self._sw_trig_btn.setEnabled(on)
        if was_running:
            self._on_start()

    def _on_send_software_trigger(self) -> None:
        try:
            self._cam.software_trigger()
        except u3v_cam.U3VError as e:
            self._status_label.setText(f"sw_trigger: {e}")

    def _on_find_me(self) -> None:
        try:
            self._cam.find_me_led()
        except u3v_cam.U3VError as e:
            self._status_label.setText(f"find_me: {e}")

    @QtCore.pyqtSlot(np.ndarray, int, int)
    def _on_frame(self, frame: np.ndarray, block_id: int, status: int) -> None:
        if self._last_block is not None and block_id > self._last_block + 1:
            self._drop_count += block_id - self._last_block - 1
        self._last_block = block_id
        if status != 0:
            self._drop_count += 1
        else:
            self._frame_count += 1
            self._fps_window_n += 1

        # pyqtgraph wants (cols, rows) = (W, H) by default. Disable autoLevels
        # after first frame to avoid flicker.
        if frame.ndim == 2:
            self._image_view.setImage(frame.T, autoLevels=False, autoRange=False)
        else:
            self._image_view.setImage(np.transpose(frame, (1, 0, 2)),
                                      autoLevels=False, autoRange=False)

    @QtCore.pyqtSlot(str)
    def _on_grab_error(self, msg: str) -> None:
        self._status_label.setText(f"grab: {msg}")

    def _update_status(self) -> None:
        now = time.monotonic()
        dt = now - self._fps_window_t0
        if dt >= 0.5:
            self._fps_value = self._fps_window_n / dt
            self._fps_window_t0 = now
            self._fps_window_n = 0
        self._status_label.setText(
            f"frames={self._frame_count}  dropped={self._drop_count}  "
            f"FPS={self._fps_value:.1f}"
        )

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self._on_stop()
        super().closeEvent(e)


# --------------------------------------------------------------------------
def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    cams = u3v_cam.list_cameras()
    if not cams:
        QtWidgets.QMessageBox.critical(None, "u3v_cam viewer",
                                       "No camera found.")
        return 1

    cam = u3v_cam.Camera(index=0)
    try:
        # baseline configuration
        info = cam.info
        cam.set_roi(info["width_max"], info["height_max"], 0, 0)
        cam.pixel_format = u3v_cam.PFNC_MONO8
    except u3v_cam.U3VError:
        pass

    win = Viewer(cam)
    win.show()
    rc = app.exec()
    cam.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
