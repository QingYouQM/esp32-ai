@echo off
cd /d d:\CodingTrash\ESP32-AI-01
start "" /MIN cmd /c "uv run python -m research.zh.synth --stage write --mode story --topics data/zh/raw/story_topics.json --out data/zh/raw/synth_story.jsonl --batch-size 96 --variants 6 --max-new-tokens 240 --temperature 0.9 > %TEMP%\synth_story_full.log 2>&1"
echo relaunched story synthesis batch 96
