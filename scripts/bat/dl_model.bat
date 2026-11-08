@echo off
cd /d d:\CodingTrash\ESP32-AI-01
start "" /MIN curl -L --max-time 3600 -o models\Qwen2.5-1.5B-Instruct\model.safetensors https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors
echo launched
