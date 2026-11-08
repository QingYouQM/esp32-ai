@echo off
cd /d d:\CodingTrash\ESP32-AI-01
if not exist data\zh\raw mkdir data\zh\raw
start "" /MIN curl -L --max-time 2400 -o data\zh\raw\wiki-00000.parquet https://hf-mirror.com/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00000-of-00006.parquet
start "" /MIN curl -L --max-time 2400 -o data\zh\raw\wiki-00001.parquet https://hf-mirror.com/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00001-of-00006.parquet
echo launched
