@echo off
cd /d d:\CodingTrash\ESP32-AI-01
del /q data\zh\raw\synth_story_pilot.jsonl 2>nul
start "" /MIN cmd /c "uv run python -m research.zh.synth --stage write --mode story --topics data/zh/raw/story_topics.json --out data/zh/raw/synth_story_pilot.jsonl --batch-size 64 --limit 128 --variants 3 --max-new-tokens 240 --temperature 0.9 > %TEMP%\synth_story.log 2>&1"
echo launched story pilot
