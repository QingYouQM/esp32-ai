@echo off
cd /d d:\CodingTrash\ESP32-AI-01
taskkill /f /im python.exe >nul 2>nul
del /q data\zh\raw\synth_pilot.jsonl 2>nul
start "" /MIN cmd /c "uv run python -m research.zh.synth --stage write --topics data/zh/raw/topics_pilot.json --out data/zh/raw/synth_pilot.jsonl --batch-size 64 --max-new-tokens 240 > %TEMP%\synth_pilot.log 2>&1"
start "" /MIN cmd /c "uv run python -m research.zh.topics_from_wiki --n 12000 --max-chars 8000000 > %TEMP%\topics.log 2>&1"
echo relaunched pilot + topics
