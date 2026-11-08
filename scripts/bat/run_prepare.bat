@echo off
cd /d d:\CodingTrash\ESP32-AI-01
del /q data\zh\raw\corpus.txt 2>nul
rmdir /s /q data\zh\vocab-16384 2>nul
start "" /MIN cmd /c "uv run python -m research.zh.prepare_zh --vocab 16384 --wiki-chars 40000000 > %TEMP%\prep.log 2>&1"
echo launched v4 prepare (wiki restored to 40M chars)
