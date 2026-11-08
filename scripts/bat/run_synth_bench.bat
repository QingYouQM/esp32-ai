@echo off
cd /d d:\CodingTrash\ESP32-AI-01
start "" /MIN cmd /c "uv run python -m research.zh.synth --stage write --batch-size 128 --limit 768 --max-new-tokens 240 > %TEMP%\synth_bench.log 2>&1"
echo launched bench
