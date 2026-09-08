@echo off
rem Install Python dependencies. Run once after extracting the package.
setlocal
echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install numpy
echo.
echo Optional packages:
echo   [1] PyQt6 + pyqtgraph  (for viewer_pyqt.py)
echo   [2] opencv-python      (for live_capture.py --show)
echo   [3] both
echo   [N] none
set /p choice="Select (1/2/3/N): "
if /i "%choice%"=="1" python -m pip install PyQt6 pyqtgraph
if /i "%choice%"=="2" python -m pip install opencv-python
if /i "%choice%"=="3" python -m pip install PyQt6 pyqtgraph opencv-python
echo.
echo Done.
endlocal
