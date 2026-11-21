@echo off
cd /d d:\CodingTrash\ESP32-AI-01
title PLE TinyLM - OpenAI-compatible API

echo ============================================
echo  PLE TinyLM  local API + web page
echo  phone / tablet / other PC: open the URL
echo  printed below (same WiFi)
echo ============================================
echo.

.venv\Scripts\python.exe scripts\serve_openai.py --port 8000

echo.
echo server stopped
pause
