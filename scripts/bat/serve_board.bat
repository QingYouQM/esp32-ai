@echo off
cd /d d:\CodingTrash\ESP32-AI-01
title PLE TinyLM - board backend (ESP32-S3)

echo ============================================
echo  ESP32-S3 is the model, this PC is the wire
echo  ~35 s per story, one request at a time
echo  open the printed URL on any device
echo ============================================
echo.

.venv\Scripts\python.exe scripts\board_openai.py --http-port 8001

echo.
echo server stopped
pause
