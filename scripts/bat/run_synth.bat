@echo off
cd /d d:\CodingTrash\ESP32-AI-01
start "" /MIN cmd /c "uv run python -m research.zh.synth --stage topics --n-topics 600 > %TEMP%\synth_topics.log 2>&1"
echo launched topics
