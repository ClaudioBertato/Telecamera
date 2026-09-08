@echo off
setlocal
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
python "%HERE%examples\viewer_pyqt.py" %*
endlocal
