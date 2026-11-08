@echo off
cd /d d:\CodingTrash\ESP32-AI-01
start "" /MIN cmd /c "uv run python -m research.tinystories.train --arm ple --corpus zh --vocab 16384 --d-model 192 --n-layers 8 --n-heads 4 --ple-dim 128 --seq-len 512 --target-core 3000000 --batch-size 16 --steps 10000 --eval-every 500 --eval-iters 50 --tag zh-v4 --amp > %TEMP%\train_zh4.log 2>&1"
echo launched v4 training 10000 steps
