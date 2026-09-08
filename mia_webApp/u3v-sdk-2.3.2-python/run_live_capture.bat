@echo off
setlocal
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
python "%HERE%examples\live_capture.py" %*
endlocal
