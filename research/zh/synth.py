"""Synthesise Chinese training paragraphs with a local teacher model.

The point of this file: a ~3M-core model cannot learn "fluent, coherent,
factually sane" Chinese from raw web text. It can learn it from a teacher that
already writes that way, exactly the trick TinyStories used. So the corpus is
generated, not crawled, and every paragraph is length-checked before it is kept.

Two stages:
  --stage topics    ask the teacher for a large pool of distinct topics
  --stage write     one paragraph per topic, batched, filtered, appended

Output is JSONL: {"topic": ..., "text": ...}. prepare_zh.py joins the texts
with <|endoftext|> and trains the BPE on them.
"""

import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
TOPICS = ROOT / "data" / "zh" / "raw" / "topics.jsonl"
OUT = ROOT / "data" / "zh" / "raw" / "synth.jsonl"

# Topic groups rotate so the pool is not 10k paragraphs about the same thing.
GROUPS = [
    "自然科学常识",
    "中国历史与文化",
    "世界历史",
    "日常生活与家居",
    "计算机与工程技术",
    "地理、城市与旅行",
    "人物与成长故事",
    "社会、经济与法律",
    "艺术、音乐与文学",
    "健康、医学与运动",
    "动物与植物",
    "食物与烹饪",
    "天气、环境与天文",
    "教育与学习方法",
    "心理与情绪",
]

TOPIC_PROMPT = (
    "请列出 {n} 个互不重复的中文写作主题词，主题范围限定为：{group}。"
    "每个主题词 2 到 8 个字，要具体、可写，不要抽象空泛的词。"
    "只输出主题词，每行一个，不要编号，不要解释，不要标点以外的任何内容。"
)

# One topic is reused many times, so the phrasing has to vary or the corpus
# becomes 100k paragraphs that all start the same way.
WRITE_PROMPTS = [
    "请围绕「{topic}」写一段 150 到 200 字的中文短文。"
    "要求：语句通顺流畅，前后语义连贯，逻辑清晰，内容准确，有具体细节；"
    "只输出正文，不要标题，不要分点列表，不要 Markdown 符号，不要写「好的」之类的开场白。",

    "请用说明性的语言介绍「{topic}」，一段 150 到 200 字。"
    "要讲清它是什么、有什么特点，语句通顺，事实准确，不要夸大；只输出正文。",

    "以「{topic}」为主题写一段 150 到 200 字的中文描写，"
    "要求内容连贯、用词准确、有具体画面或例子，不要空洞抒情；只输出正文。",

    "请解释「{topic}」的成因或原理，一段 150 到 200 字，"
    "要求因果清楚、逻辑严密、不编造事实；只输出正文。",

    "请写一段 150 到 200 字的中文文字，把「{topic}」和一个具体的生活场景结合起来，"
    "要求自然、通顺、有细节；只输出正文。",
]

CJK = re.compile(r"[\u4e00-\u9fff]")
NOISE = re.compile(r"^[\s]*(好的|以下是|当然|没问题|这是一段|短文[:：])?[^\u4e00-\u9fff]{0,12}[:：]?\s*")
TAIL_OK = ("。", "！", "？", "”", "」", ".", "!", "?")


def han_count(text):
    return len(CJK.findall(text))


def trim_to_han(text, min_han, max_han):
    """Keep whole sentences until the paragraph is in range.

    The teacher writes ~480 hanzi when asked for 200, so filtering long output
    threw away nearly everything. Cutting on a sentence boundary keeps the
    length honest and the paragraph intact.
    """
    out = ""
    seen = set()
    for sent in re.split(r"(?<=[。！？])", text):
        # A 1.5B teacher loops: it restarts the paragraph verbatim once it runs
        # out of things to say. Cut there instead of shipping the duplicate.
        if sent.strip() and sent.strip() in seen:
            break
        seen.add(sent.strip())
        out += sent
        if han_count(out) >= max_han:
            break
    n = han_count(out)
    return out if min_han <= n else ""


def clean(text):
    """Strip the teacher's chat habits and keep one clean paragraph."""
    t = text.strip()
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    t = t.replace("**", "").replace("##", "").replace("---", "")
    t = re.sub(r"^[#>\-\*\+]+\s*", "", t, flags=re.M)
    t = NOISE.sub("", t)
    # Cut everything after the paragraph ends: teachers love adding a footnote.
    end = max(t.rfind(c) for c in TAIL_OK)
    if end > 40:
        t = t[: end + 1]
    t = re.sub(r"\s+", "", t)          # Chinese has no meaningful spaces here
    t = re.sub(r"[ \t]*\n[ \t]*", "", t)
    return t.strip()


def load_model(path, device, dtype):
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # sdpa matters: the default attention path made a 1.5B teacher generate at
    # ~13 tok/s on a 4060, which puts a real corpus out of reach.
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, attn_implementation="sdpa")
    model.to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def generate_batch(tok, model, prompts, device, max_new_tokens, temperature, top_p):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=512).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def stage_topics(tok, model, device, args):
    seen = set()
    if os.path.exists(TOPICS):
        for line in open(TOPICS, encoding="utf-8"):
            seen.add(json.loads(line)["topic"])
        print(f"already have {len(seen)} topics")
    want = args.n_topics - len(seen)
    if want <= 0:
        return
    per_batch = 120
    batches = (want // per_batch) + 1
    f = open(TOPICS, "a", encoding="utf-8")
    written = 0
    t0 = time.time()
    for b in range(batches):
        group = GROUPS[b % len(GROUPS)]
        prompt = TOPIC_PROMPT.format(n=per_batch, group=group)
        text = generate_batch(tok, model, [prompt], device,
                              max_new_tokens=per_batch * 24,
                              temperature=1.0, top_p=0.98)[0]
        for line in text.splitlines():
            t = line.strip().strip("。-—*#0123456789.、,， ")
            if not t or len(t) > 12 or len(t) < 2:
                continue
            if not CJK.search(t):
                continue
            if t in seen:
                continue
            seen.add(t)
            f.write(json.dumps({"topic": t, "group": group}, ensure_ascii=False) + "\n")
            written += 1
        f.flush()
        print(f"  batch {b + 1}/{batches} group={group} topics={len(seen)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if len(seen) >= args.n_topics:
            break
    f.close()


def stage_write(tok, model, device, args):
    if not os.path.exists(args.topics):
        raise SystemExit(f"{args.topics} missing; mine topics first:\n"
                         f"  uv run python -m research.zh.topics_from_wiki")
    topics = [json.loads(l)["topic"] for l in open(args.topics, encoding="utf-8")]
    work = [(t, v) for v in range(args.variants) for t in topics]
    done = 0
    if os.path.exists(args.out):
        done = sum(1 for _ in open(args.out, encoding="utf-8"))
        print(f"resuming: {done} paragraphs already written")
    todo = work[done:]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to do")
        return
    f = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    kept = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i: i + args.batch_size]
        prompts = [WRITE_PROMPTS[v].format(topic=t) for t, v in batch]
        texts = generate_batch(tok, model, prompts, device, args.max_new_tokens,
                               args.temperature, args.top_p)
        for (topic, _v), raw in zip(batch, texts):
            text = clean(raw)
            text = trim_to_han(text, args.min_han, args.max_han)
            if not text:
                continue
            n = han_count(text)
            if i == 0 and topic == batch[0][0]:
                print(f"--- sample topic={topic}")
                print(f"--- raw ({han_count(raw)} han): {raw[:110]!r}")
                print(f"--- kept ({n} han): {text[:110]!r}")
            f.write(json.dumps({"topic": topic, "text": text, "han": n},
                               ensure_ascii=False) + "\n")
            kept += 1
        f.flush()
        rate = (i + len(batch)) / max(1e-9, time.time() - t0)
        print(f"  {i + len(batch)}/{len(todo)} kept={kept} "
              f"({rate:.1f} para/s, {time.time() - t0:.0f}s)", flush=True)
    f.close()
    print(f"done: {kept} paragraphs -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["topics", "write"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--topics", default=str(TOPICS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-topics", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    # 240 new tokens is ~360 hanzi from this teacher: enough to cut a 200-hanzi
    # paragraph out of it, and half the cost of asking for the full answer.
    ap.add_argument("--max-new-tokens", type=int, default=240)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-han", type=int, default=140)
    ap.add_argument("--max-han", type=int, default=205)
    ap.add_argument("--limit", type=int, default=0, help="pilot: only first N items")
    ap.add_argument("--variants", type=int, default=12,
                    help="prompt templates per topic; total = topics * variants")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(
            f"{args.model} not found. Download the teacher first:\n"
            f"  set HF_ENDPOINT=https://hf-mirror.com\n"
            f"  uv run hf download Qwen/Qwen2.5-1.5B-Instruct "
            f"--local-dir {args.model}")

    dtype = getattr(torch, args.dtype)
    tok, model = load_model(args.model, args.device, dtype)
    print(f"model on {args.device} ({args.dtype}), stage={args.stage}")
    if args.stage == "topics":
        stage_topics(tok, model, args.device, args)
    else:
        stage_write(tok, model, args.device, args)


if __name__ == "__main__":
    sys.exit(main())
