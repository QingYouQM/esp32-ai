@echo off
cd /d d:\CodingTrash\ESP32-AI-01
set GIT=C:\Program Files\Git\bin\git.exe
"%GIT%" status --short
"%GIT%" --no-pager log -1 --format=%%H%%n%%s
