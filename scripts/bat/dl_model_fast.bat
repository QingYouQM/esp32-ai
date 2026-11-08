@echo off
cd /d d:\CodingTrash\ESP32-AI-01
taskkill /f /im curl.exe >nul 2>nul
timeout /t 2 >nul
if exist models\Qwen2.5-1.5B-Instruct\model.safetensors del /q models\Qwen2.5-1.5B-Instruct\model.safetensors
set U=https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors
start "" /MIN curl -L --max-time 3600 -r 0-771866785 -o models\Qwen2.5-1.5B-Instruct\p0.bin "%U%"
start "" /MIN curl -L --max-time 3600 -r 771866786-1543733571 -o models\Qwen2.5-1.5B-Instruct\p1.bin "%U%"
start "" /MIN curl -L --max-time 3600 -r 1543733572-2315600357 -o models\Qwen2.5-1.5B-Instruct\p2.bin "%U%"
start "" /MIN curl -L --max-time 3600 -r 2315600358-3087467143 -o models\Qwen2.5-1.5B-Instruct\p3.bin "%U%"
echo launched 4 ranges
