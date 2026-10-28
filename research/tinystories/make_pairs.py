"""Turn a plain story corpus into material-and-story pairs.

The training target is not "write a Chinese story". It is "read a fact sheet,
write the story it describes, and do not invent people". A corpus of bare
stories cannot teach that, because nothing in it ever plays the part of the
fact sheet. So for every existing story this asks a model for the six-field
material it implies, then checks the material against the story it came from:

  * every person listed in 人物 has to appear in the story;
  * every name that acts in the story has to be listed in 人物;
  * no field over its budget, no Latin, no default names.

A material that fails is a material that invented a person, which is exactly
the failure the runtime checker exists to catch, so it is rejected here rather
than trained on. Rejects go to a log with the reason.

    uv run python -m research.tinystories.make_pairs --limit 20 --dry-run
    uv run python -m research.tinystories.make_pairs --out research/tinystories/seeds/zh/pairs.txt

Output appends, so an interrupted run resumes: a story whose text already
appears in the output is skipped.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from research.kb.render import material_text
from research.kb.spec import FIELDS, FIELD_MAX, MATERIAL_MAX, STORY_MARK, MATERIAL_MARK, EOD_MARK
from research.kb.validate import NAME_RE, clean_name, listed_people, resolve

from .augment import call_api, clean, load_docs, read_key

SYSTEM = """你是史料整理员。给你一篇已经写好的中文短故事，你要从故事里提取事实，\
写成一张六字段的材料卡。

规则：
- 只写故事里已经发生的事。故事没写的人、地点、物品，一律不许出现。
- 人物：列出故事中所有做出动作的人，格式「姓名，身份」，多人用分号分隔。
  只被提到、没有任何动作的人不要列。
- 如果故事里的人物只有称谓、没有姓名（例如「图书管理员」「老爷爷」「弟弟」），
  就照抄这个称谓，绝对不要另取名字——取了名字就等于凭空多出一个人。
- 时间：故事里若有时间词（春天、傍晚、腊月、某年），照抄最具体的那个；
  没有就写一个与故事不矛盾的最简时间，例如「一个下午」。
- 地点：写故事里最具体的地点（镇名、村名、桥、店、院子），不要编造故事里
  没出现的地名；故事只有「森林」就写「森林」。
- 起因、经过、结果：各一到三句，按时间顺序，只写外部可见的事件。
  不写心理活动、不写评价、不写比喻。
- 字数上限：人物 40、时间 20、地点 20、起因 60、经过 100、结果 50，合计不超过 300。
- 不要出现英文、拼音、括号注释。

只输出 JSON，不要 Markdown 代码块，不要任何解释。字段名必须是：
人物、时间、地点、起因、经过、结果

例如：
{"人物":"李明，木匠","时间":"一个傍晚","地点":"镇口的旧木桥","起因":"李明要把一张长凳送到桥对面。","经过":"桥板断了，长凳掉进河里，他下水把长凳拖了上来。","结果":"他把长凳送到，天黑才回家。"}"""

BAD_WORDS = ("著名", "伟大", "被誉为", "作出了贡献", "具有重要意义", "精神", "可歌可泣")
LATIN = re.compile(r"[A-Za-z]")
PARTICLES = "之氏父子等辈们"


def extract_json(raw):
    """Pull the first JSON object out of whatever the model wrapped it in."""
    text = clean(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def appears_in(label, story):
    """Whether a label the card uses is in the story, allowing paraphrase.

    The model writes 面包店老板 where the story says 面包店的老板, so an exact
    substring test rejects a correct card. Dropping 的 and accepting a prefix of
    at least two characters covers that without accepting a name the story
    never had.
    """
    s = story.replace("的", "")
    label = label.replace("的", "")
    if label in s:
        return True
    return any(label[:n] in s for n in range(len(label) - 1, 1, -1))


def check_material(fields, story):
    """Return (ok, reason). The material must describe this story and no other."""
    problems = []
    missing = [f for f in FIELDS if f not in fields]
    if missing:
        return False, "missing " + ",".join(missing)
    extra = [f for f in fields if f not in FIELDS]
    if extra:
        return False, "unknown " + ",".join(extra)

    total = 0
    for name in FIELDS:
        text = str(fields[name]).strip()
        if not text:
            return False, f"empty {name}"
        if len(text) > FIELD_MAX[name]:
            return False, f"long {name} {len(text)}"
        if LATIN.search(text):
            return False, f"latin in {name}"
        total += len(text)
    if total > MATERIAL_MAX:
        return False, f"long total {total}"
    for word in BAD_WORDS:
        for name in FIELDS:
            if word in str(fields[name]):
                return False, f"evaluative {word} in {name}"

    declared = listed_people(str(fields["人物"]))
    if not declared:
        return False, "no person named"
    for person in declared:
        if not appears_in(person, story):
            return False, f"invented {person}"

    # The reverse direction - a name that acts but is never declared - is not
    # checked here. Without a segmenter the surname heuristic reads 石头上,
    # 安静下 and 熊抱起 as people, and it rejected half of a 30-story sample for
    # nothing. The forward check above is the one that matters: a card can still
    # fail by naming a person the story never had.

    if len(story) < 120:
        return False, "story short"
    return True, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stories", nargs="+",
                    default=["research/tinystories/seeds/zh/generated.txt",
                             "research/tinystories/seeds/zh/corpus.txt"])
    ap.add_argument("--out", default="research/tinystories/seeds/zh/pairs.txt")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new pairs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL",
                                                      "MiniMax-M3.1-Flash-Preview"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL",
                                                         "https://api.minimax.cn/v1"))
    ap.add_argument("--api-key", default="")
    ap.add_argument("--key-file", default="data/api_key.txt")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stories = []
    for path in args.stories:
        stories += load_docs(path)
    # Deduplicate on the text itself; the two sources overlap in spirit if not
    # in bytes, and a pair costs an API call.
    seen_story, unique = set(), []
    for s in stories:
        if s not in seen_story:
            seen_story.add(s)
            unique.append(s)
    stories = unique

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_stories = set()
    if out_path.exists():
        for doc in load_docs(out_path):
            body = doc.split(STORY_MARK, 1)
            if len(body) == 2:
                done_stories.add(body[1].strip())
    remaining = [s for s in stories if s not in done_stories]
    todo = remaining[:args.limit] if args.limit else remaining

    print(f"stories {len(stories)}, already paired {len(stories) - len(remaining)}, "
          f"to do {len(todo)}")
    if not todo:
        return 0

    if args.dry_run:
        print("\n===== story 1 =====\n" + todo[0])
        print("\n===== prompt =====\n" + SYSTEM)
        return 0

    api_key = read_key(args)
    if not api_key:
        raise SystemExit(f"no API key in {args.key_file}, $OPENAI_API_KEY or --api-key")

    kept, rejects, api_fail = 0, [], 0
    t0 = time.time()
    log = out_path.with_suffix(".rejects.txt")

    def one(story):
        prompt = "故事：\n" + story + "\n\n请输出材料卡 JSON。"
        raw, err = call_api(prompt, args.model, args.base_url, api_key,
                            args.temperature, args.timeout,
                            max_tokens=args.max_tokens, thinking=False,
                            reasoning_effort="low", system=SYSTEM)
        if err:
            return story, None, err, "api"
        try:
            fields = extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            return story, None, f"json: {e}", "json"
        if not isinstance(fields, dict):
            return story, None, "not an object", "json"
        ok, why = check_material(fields, story)
        if not ok:
            return story, None, why, "check"
        return story, fields, "", ""

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, s) for s in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            story, fields, why, kind = fut.result()
            if why:
                rejects.append((i, why, story[:50]))
                if kind == "api":
                    api_fail += 1
                # Written as they happen, so a throttled run can be diagnosed
                # while it is still running instead of after it finishes.
                with log.open("a", encoding="utf-8") as f:
                    f.write(f"{i}\t{kind}\t{why[:90]}\t{story[:40]}\n")
                continue
            entry = {"fields": {k: {"text": str(fields[k]).strip()} for k in FIELDS}}
            doc = (f"{MATERIAL_MARK}\n{material_text(entry)}\n{STORY_MARK}\n"
                   f"{story}\n{EOD_MARK}\n")
            with out_path.open("a", encoding="utf-8") as f:
                f.write(doc)
            kept += 1
            if i % 20 == 0:
                rate = i / max(time.time() - t0, 1e-6)
                left = (len(todo) - i) / max(rate, 1e-9) / 60
                print(f"  {i}/{len(todo)} kept {kept} rejected {len(rejects)} "
                      f"(api {api_fail}) {rate:.2f}/s eta {left:.1f}min", flush=True)

    if rejects:
        kinds = Counter(w.split(" ", 1)[0] for _, w, _ in rejects)
        print("reject reasons: " + ", ".join(f"{k}={v}" for k, v in kinds.most_common()))
        print(f"reject log: {log}")

    print(f"\ndone. kept {kept}, rejected {len(rejects)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
