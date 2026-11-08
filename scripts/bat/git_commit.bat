@echo off
cd /d d:\CodingTrash\ESP32-AI-01
set GIT=C:\Program Files\Git\bin\git.exe
"%GIT%" add research/zh/story_topics.py research/zh/synth.py research/zh/audit_story.py research/zh/prepare_zh.py scripts/zh_status.ps1
"%GIT%" commit -m "Add story-mode synthesis; fix prepare_zh missing the story corpus"

TinyStories-style narrative data: story_topics.py crosses 62 characters with
50 situations into 2,800 concrete seeds ("一只流浪猫比赛输了"), synth.py grows a
--mode story register, and 16,203 paragraphs were generated. audit_story.py is
the corpus check for that file.

v3 trained without the stories at all: build_corpus only read synth.jsonl, so
the 1500-char wiki cut left nothing but template-heavy prose, and the model
collapsed on out-of-distribution prompts. prepare_zh now ingests
synth_story.jsonl; zh_status.ps1 tracks both corpus files and whichever
synthesis log is newest.
"%GIT%" --no-pager log --oneline -3
