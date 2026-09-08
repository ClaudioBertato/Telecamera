#!/usr/bin/env bash
# Install Python dependencies. Run once after extracting the package.
set -e

PIP="${PIP:-python3 -m pip}"

echo "==> Installing core dependencies"
$PIP install --upgrade pip
$PIP install numpy

echo
echo "Optional packages:"
echo "  [1] PyQt6 + pyqtgraph  (for viewer_pyqt.py)"
echo "  [2] opencv-python      (for live_capture.py --show)"
echo "  [3] both"
echo "  [N] none"
read -r -p "Select (1/2/3/N): " choice
case "${choice,,}" in
    1) $PIP install PyQt6 pyqtgraph ;;
    2) $PIP install opencv-python ;;
    3) $PIP install PyQt6 pyqtgraph opencv-python ;;
    *) ;;
esac
echo
echo "Done."
