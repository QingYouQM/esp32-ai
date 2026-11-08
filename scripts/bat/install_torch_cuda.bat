@echo off
cd /d d:\CodingTrash\ESP32-AI-01
start "" /MIN cmd /c "uv sync > %TEMP%\torch_install.log 2>&1 & echo EXIT=%errorlevel% >> %TEMP%\torch_install.log"
echo launched uv sync
