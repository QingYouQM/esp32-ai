@echo off
cd /d d:\CodingTrash\ESP32-AI-01
taskkill /f /im python.exe >nul 2>nul
start "" /MIN cmd /c "uv run python -m research.zh.topics_from_wiki --n 12000 --max-chars 6000000 > %TEMP%\topics.log 2>&1"
echo relaunched topic mining
