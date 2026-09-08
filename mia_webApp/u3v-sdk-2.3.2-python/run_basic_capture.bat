@echo off
rem Run the basic_capture example from this folder.
setlocal
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
python "%HERE%examples\basic_capture.py" %*
endlocal
