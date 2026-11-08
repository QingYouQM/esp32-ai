@echo off
ping -n 130 127.0.0.1 >nul
cd /d d:\CodingTrash\ESP32-AI-01
powershell -NoProfile -Command "Get-Content %TEMP%\train_zh4.log | Select-String -NotMatch 'Loading weights' | Select-Object -Last 3"
