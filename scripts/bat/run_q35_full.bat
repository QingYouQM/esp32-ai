@echo off
cd /d d:\CodingTrash\ESP32-AI-01
start "" /MIN cmd /c "uv run python -m research.zh.synth --stage write --mode story --model models\Qwen3.5-2B --topics data\zh\raw\story_topics_full.json --out data\zh\raw\synth_story_q35.jsonl --batch-size 64 --variants 4 --max-new-tokens 240 --temperature 0.7 --top-p 0.8 --top-k 20 --apply-chat-template > %TEMP%\synth_q35_full.log 2>&1"
echo launched q35 full re-distillation (batch 64, 11024 topics x 4 variants)
