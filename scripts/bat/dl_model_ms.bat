@echo off
cd /d d:\CodingTrash\ESP32-AI-01
del /q models\Qwen2.5-1.5B-Instruct\p0.bin models\Qwen2.5-1.5B-Instruct\p1.bin models\Qwen2.5-1.5B-Instruct\p2.bin models\Qwen2.5-1.5B-Instruct\p3.bin 2>nul
start "" /MIN curl -L --max-time 3600 -o models\Qwen2.5-1.5B-Instruct\model.safetensors "https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-1.5B-Instruct/repo?Revision=master&FilePath=model.safetensors"
echo launched modelscope download
