@echo off
cd /d d:\CodingTrash\ESP32-AI-01
start "" /MIN cmd /c "uv run python -m research.zh.synth --stage write --batch-size 192 --variants 12 --max-new-tokens 240 --limit 20000 > %TEMP%\synth_full.log 2>&1"
echo launched synth to 20000
