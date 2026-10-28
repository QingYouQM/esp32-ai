"""Expand a seed corpus through an LLM API without collapsing into a mould.

The seed corpus is a recipe, not training data: at 11,961 tokens it is a
style template, and the target is roughly 300k. Nothing here writes stories -
it writes prompts and ships the results to the same checks as hand-written
text, because generated prose fails in exactly the same ways.

Two decisions carry the design.

The recipe is locked and only the axes move. Reading level, sentence length,
single-turn structure and length are fixed in every prompt; protagonist,
setting, turn and ending tone vary. Letting a model decide freely is how a
corpus turns into one mould - batch 1 of the seed was 'small animal, forest,
danger, warm ending' forty-three times, and asking for "children's stories"
is what produced it.

Combinations are assigned by index, not sampled. A shuffled round-robin over
each axis means every value appears the same number of times in the output, so
coverage is a property of the run rather than of luck, and re-running with
--resume picks up the same combinations in the same order.

Nothing trusts the model. Every document is checked against the corpus rules
and against the corpus so far, and rejects are logged with reasons rather than
silently dropped, because a rejection rate that climbs quietly is the leading
indicator of a corpus that has stopped being diverse.

    OPENAI_API_KEY=... OPENAI_BASE_URL=... \\
    uv run python -m research.tinystories.augment \\
        --axes research/tinystories/seeds/zh/axes.json \\
        --seed-corpus research/tinystories/seeds/zh/corpus.txt \\
        --out research/tinystories/seeds/zh/generated \\
        --count 1400 --workers 8
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
except ImportError:
    raise SystemExit("requests is required: pip install requests")

DOC_SEP = "<|endoftext|>"

# Corpus rules, matching check_corpus.py. The length band is the corpus band,
# not the 150-200 generation target: clipping to the target truncates plot.
MIN_CHARS, MAX_CHARS = 120, 300
LATIN = re.compile(r"[A-Za-z]")
REPLACEMENT = "�"

# Character 4-gram Jaccard above which a document counts as a remake.
#
# Measured, not guessed: across all 6,105 pairs of the 111 seed documents the
# highest overlap is 0.028 and the median is near 0.004. An initial 0.50 was
# three orders of magnitude off and would never have rejected anything - a
# de-duplicator that cannot fire is worse than none, because the corpus looks
# guarded. 0.15 sits ~5x above the seed's own worst pair, which catches a model
# re-serving a combination while leaving genuinely distinct prose alone.
NEAR_DUP = 0.15
NGRAM = 4

SYSTEM = """你是一位 children's short-story writer for Chinese primary school readers. \
严格遵守以下要求，只输出故事正文本身。

写作要求：
- 面向中国大陆小学三四年级学生（约 8-10 岁能独立读完）。
- 用短句。一句话尽量不超过 20 个字。偶尔可以长一点，但不要连续三句都长。
- 结构固定三段：开头一句交代情境；中间发生一次转折（只转折一次，不要起伏）；\
结尾一句收束。
- 只用具体的动作和看得见的东西。不要写"他感到很温暖"这种抽象描写，\
要写他做了什么。
- 全文不要出现英文、拼音、括号、编号或任何解释性文字。
- 不要加标题，不要加引号，不要写"故事"之类的开头。

特别注意：故事必须自然结束，不要在半句话处停下。"""


def load_docs(path, sep=DOC_SEP):
    text = Path(path).read_text(encoding="utf-8")
    return [d.strip() for d in text.split(sep) if d.strip()]


def grams(doc, n=NGRAM):
    s = re.sub(r"\s+", "", doc)
    return {s[i : i + n] for i in range(max(1, len(s) - n + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Combinator:
    """Deterministic axis assignment.

    Index i takes element (i * stride) % len(axis) from each axis. The stride
    has to be COPRIME with the axis length, or the walk only visits
    len/gcd(len, stride) of the values: a stride of 5 over 30 settings covers
    6 of them and silently never uses the other 24. Verified rather than
    assumed - see the coverage assertion below.
    """

    def __init__(self, axes, seed=0):
        rng = random.Random(seed)
        self.pools = {}
        for name, values in axes.items():
            if name.startswith("_") or not isinstance(values, list):
                continue
            pool = list(values)
            rng.shuffle(pool)
            self.pools[name] = pool
        self.strides = {}
        for i, name in enumerate(sorted(self.pools)):
            n = len(self.pools[name])
            step = 1 + 2 * (i % 3)                       # 1, 3 or 5 as a start
            while math.gcd(step, n) != 1:
                step += 1
            self.strides[name] = step

    def verify(self, n):
        """Walk the first n indices and assert full coverage on every axis.

        Cheap, and the alternative is discovering after 1,400 paid API calls
        that a third of the settings never appeared. Coverage is only
        meaningful once n reaches the longest axis; a short run is checked for
        unique combinations alone.
        """
        seen = {name: set() for name in self.pools}
        combos = set()
        for i in range(n):
            a = self.at(i)
            for k, v in a.items():
                seen[k].add(v)
            combos.add(tuple(sorted(a.items())))
        if len(combos) != n:
            raise SystemExit(
                f"only {len(combos)} unique combinations in {n} indices; the "
                f"strides {self.strides} are not separating the product")
        longest = max(len(p) for p in self.pools.values())
        if n < longest:
            return
        for k, pool in self.pools.items():
            if len(seen[k]) != len(pool):
                raise SystemExit(
                    f"axis {k!r} covers only {len(seen[k])}/{len(pool)} values in "
                    f"{n} indices (stride {self.strides[k]}, length "
                    f"{len(pool)}). Pick a stride coprime with the length.")

    def at(self, i):
        return {name: pool[(i * self.strides[name]) % len(pool)]
                for name, pool in self.pools.items()}

    def total(self):
        n = 1
        for pool in self.pools.values():
            n *= len(pool)
        return n


def build_prompt(axes_at, shots, rng):
    who = axes_at.get("protagonists", "一个小朋友")
    where = axes_at.get("settings", "一个小镇")
    turn = axes_at.get("turns", "遇到一件意外的事")
    ending = axes_at.get("endings", "平静的")

    body = (
        f"请写一个故事。\n\n"
        f"主角：{who}\n"
        f"地点：{where}\n"
        f"中间发生的转折：{turn}\n"
        f"结尾的基调：{ending}\n\n"
        f"篇幅 180 到 220 个汉字。\n"
    )
    if shots:
        body += "\n下面几篇是你要模仿的文风样例，注意只学文风，不学内容：\n\n"
        for s in shots:
            body += f"<样例>\n{s}\n</样例>\n\n"
    body += "现在写。要求：不要重复样例里的任何情节、角色名或句子。"
    return body


# Reasoning controls that not every model in the family accepts. The 400
# handler drops whichever of these the live model names in its error text.
OPTIONAL_PARAMS = ("thinking", "reasoning_effort")


def call_api(prompt, model, base_url, key, temperature, timeout,
             max_tokens=3000, thinking=False, reasoning_effort="low"):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not thinking:
        # MiniMax's reasoning controls differ per model, and neither field is
        # safe to send unconditionally:
        #
        #   M3                  honours thinking.type=disabled; ignores effort
        #   M3.1-Flash-Preview  ALWAYS thinks, 400s on disabled; honours effort
        #   M2.x                ignores thinking.type=disabled entirely
        #
        # reasoning_effort is the one that bites. M3.1-Flash-Preview - the only
        # model M Plan subscribers can reach - defaults it to "max" and rejects
        # "none", so omitting it means maximum reasoning whose tokens count
        # against max_tokens: a 250-token story arrives as an EMPTY content field
        # and looks exactly like a refusal. Writing one children's story needs no
        # reasoning at all, so "low" is requested and the budget left generous.
        payload["thinking"] = {"type": "disabled"}
        if reasoning_effort:
            # Sent only alongside the disable above, so --thinking means what
            # it says: either both controls cap the reasoning or neither does.
            # Sending effort on its own would cap the depth while leaving
            # thinking enabled, which is the opposite of what the flag claims.
            payload["reasoning_effort"] = reasoning_effort

    last = "no attempt"
    for attempt in range(5):
        try:
            r = requests.post(url, json=payload,
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"},
                              timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                # Throttled or a server fault: both are transient, so back off
                # rather than counting the attempt as a rejection.
                time.sleep(min(2 ** attempt, 20) + rng_offset(attempt))
                continue
            if r.status_code == 400:
                dropped = [k for k in OPTIONAL_PARAMS
                           if k in payload and k in r.text]
                if dropped:
                    # The live model rejected a reasoning control that its
                    # siblings handle differently. Drop just that key and retry:
                    # M3.1-Flash-Preview 400s on thinking disabled, older
                    # models 400 on reasoning_effort. One command therefore
                    # spans the whole family instead of failing on the model
                    # the plan actually grants.
                    for k in dropped:
                        payload.pop(k)
                    last = (f"http 400, dropped {', '.join(dropped)}: "
                            f"{r.text[:100]}")
                    continue
                return None, f"http 400: {r.text[:140]}"
            if r.status_code != 200:
                return None, f"http {r.status_code}: {r.text[:120]}"
            msg = r.json()["choices"][0]["message"]
            # M3.1-Flash-Preview returns thinking as reasoning_content, leaving
            # content clean; M2.x leaves a <think> block inside content, which
            # clean() strips. Either way content is what the corpus wants.
            if not msg.get("content") and msg.get("reasoning_content"):
                return None, ("empty content, model spent the budget reasoning "
                              f"({len(msg['reasoning_content'])} chars)")
            return msg.get("content") or "", None
        except (requests.RequestException, KeyError, ValueError, IndexError) as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 20) + rng_offset(attempt))
    return None, f"retries exhausted ({last})"


def rng_offset(attempt):
    import random as _r
    return _r.random() * 0.5


def clean(doc):
    """Strip anything a model adds around the story.

    A refusal to answer, a leading 'Sure, here's a story:', a markdown fence or
    a stray <|endoftext|> all pass a length check and poison a tokenizer, so the
    cleaning is aggressive and the result is re-measured.
    """
    d = doc.strip()
    # Reasoning models that cannot be switched off (MiniMax-M2.x: "thinking
    # cannot be disabled") return their chain of thought INSIDE content,
    # wrapped in <think> tags. Left in place it inflates the length past
    # MAX_CHARS and trips the Latin-character check, so without this one line
    # essentially every document is rejected.
    d = re.sub(r"<think>.*?</think>", "", d, flags=re.S | re.I)
    d = re.sub(r"</?think>", "", d, flags=re.I)
    d = re.sub(r"^```[a-z]*\s*|\s*```$", "", d, flags=re.S)
    d = re.sub(r"^\s*(Sure|Certainly|Here(\s+is|'s))\b.*?[:：]\s*", "", d,
               flags=re.I | re.S)
    d = d.replace(DOC_SEP, "")
    d = d.strip().strip("“”\"")
    # Remove a leading title line if the model emitted one.
    lines = [ln for ln in d.splitlines() if ln.strip()]
    if len(lines) > 1 and len(lines[0]) <= 20 and not lines[0].endswith(("。", "！", "？", "…")):
        lines = lines[1:]
    d = "\n".join(lines).strip()
    # Collapse to a single paragraph; the corpus is one paragraph per document.
    d = re.sub(r"\s*\n\s*", "", d)
    return d


def validate(doc):
    """Return (ok, reason). Mirrors check_corpus.py so generated and written
    text are held to one standard."""
    if REPLACEMENT in doc:
        return False, "lost-character"
    if LATIN.search(doc):
        return False, "latin"
    n = len(doc)
    if n < MIN_CHARS:
        return False, f"short {n}"
    if n > MAX_CHARS:
        return False, f"long {n}"
    if doc.count("“") != doc.count("”"):
        return False, "unbalanced-quotes"
    return True, ""


def read_key(args):
    """Resolve the API key: --api-key, then $OPENAI_API_KEY, then a file.

    The file route is the reason this is a function rather than a default=.
    An env var set in someone interactive shell is invisible to the next
    process, and pasting a key into a chat transcript leaves it there; a file
    under data/ is both readable here and already gitignored.
    """
    if args.api_key:
        return args.api_key.strip()
    env = os.environ.get("OPENAI_API_KEY", "").strip()
    if env:
        return env
    p = Path(args.key_file)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", required=True)
    ap.add_argument("--seed-corpus", required=True)
    ap.add_argument("--out", required=True, help="output corpus; appends and resumes")
    ap.add_argument("--count", type=int, required=True,
                    help="how many documents to have in the OUTPUT, total")
    ap.add_argument("--model",
                    default=os.environ.get("OPENAI_MODEL",
                                            "MiniMax-M3.1-Flash-Preview"),
                    help="M Plan grants MiniMax-M3.1-Flash-Preview. "
                         "Pay-as-you-go keys: MiniMax-M3, MiniMax-M2.7, "
                         "MiniMax-M2.5, MiniMax-M2.1")
    ap.add_argument("--base-url",
                    default=os.environ.get("OPENAI_BASE_URL",
                                           "https://api.minimax.cn/v1"),
                    help="China: https://api.minimax.cn/v1 ; "
                         "global: https://api.minimax.io/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--key-file", default="data/api_key.txt",
                    help="file to read the API key from; gitignored")
    ap.add_argument("--workers", type=int, default=8)
    # MiniMax accepts [0, 2] and recommends 1.0; outside that range is an error
    # rather than a clamp.
    ap.add_argument("--temperature", type=float, default=1.0)
    # Reasoning tokens count against this budget on every MiniMax model, and
    # M3.1-Flash-Preview defaults its thinking depth to max, so this has to be
    # far above the ~250 tokens an actual story needs.
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--thinking", action="store_true",
                    help="let the model reason at the server's default depth; off "
                         "by default because reasoning is not corpus material")
    ap.add_argument("--reasoning-effort",
                    default=os.environ.get("MINIMAX_REASONING_EFFORT", "low"),
                    choices=["low", "medium", "high", "xhigh", "max", "none"],
                    help="M3.1-Flash-Preview only, and ignored unless --thinking "
                         "is off. That model defaults to max server-side and "
                         "rejects 'none'.")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed for axis assignment; also the shuffle for few-shot")
    ap.add_argument("--near-dup", type=float, default=NEAR_DUP)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first few prompts and exit; no API calls")
    args = ap.parse_args()
    api_key = read_key(args)

    if not api_key and not args.dry_run:
        raise SystemExit(
            f"no API key. Put it in {args.key_file}, or set OPENAI_API_KEY, "
            f"or pass --api-key. (Checked in that order.)")

    axes = json.loads(Path(args.axes).read_text(encoding="utf-8"))
    comb = Combinator(axes, seed=args.seed)
    seeds = load_docs(args.seed_corpus)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    have = load_docs(out_path) if out_path.exists() else []
    start = len(have)
    # Everything already produced, so a resumed run does not re-emit documents
    # it already rejected as near-duplicates, and so novelty is judged against
    # the real accumulated corpus.
    seen = [grams(d) for d in have]
    rng = random.Random(args.seed + 1)

    # Refuse before the first API call rather than after 1,400 of them.
    comb.verify(args.count)

    print(f"axes: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(comb.pools.items())))
    print(f"combinations available: {comb.total():,}")
    print(f"seed corpus: {len(seeds)} docs   already generated: {len(have)}   "
          f"target: {args.count}")

    if args.dry_run:
        for i in range(start, min(start + 3, args.count)):
            shots = rng.sample(seeds, min(2, len(seeds)))
            print(f"\n===== index {i} =====")
            print(build_prompt(comb.at(i), shots, rng))
        return 0

    reject_log = out_path.with_suffix(".rejects.txt")
    kept, rejected, failed = [], [], 0
    todo = list(range(start, args.count))
    t0 = time.time()

    def one(i):
        shots = rng.sample(seeds, min(2, len(seeds)))
        prompt = build_prompt(comb.at(i), shots, rng)
        return i, call_api(prompt, args.model, args.base_url, api_key,
                           args.temperature, args.timeout,
                           max_tokens=args.max_tokens, thinking=args.thinking,
                           reasoning_effort=args.reasoning_effort)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, i): i for i in todo}
        done = 0
        for fut in as_completed(futures):
            i, (raw, err) = fut.result()
            done += 1
            if err:
                failed += 1
                rejected.append((i, err))
                continue
            doc = clean(raw)
            ok, reason = validate(doc)
            if not ok:
                rejected.append((i, f"{reason}: {doc[:60]}"))
                continue
            g = grams(doc)
            if any(jaccard(g, s) >= args.near_dup for s in seen):
                rejected.append((i, f"near-dup: {doc[:40]}"))
                continue
            seen.append(g)
            kept.append(doc)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(doc + "\n" + DOC_SEP + "\n")
            if done % 20 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done}/{len(todo)}  kept {len(kept)}  rejected {len(rejected)}  "
                      f"{rate:.2f}/s  eta {(len(todo) - done) / max(rate, 1e-9) / 60:.1f}min",
                      flush=True)

    if rejected:
        with reject_log.open("a", encoding="utf-8") as f:
            for i, why in rejected:
                f.write(f"{i}\t{why}\n")

    print(f"\ndone. kept {len(kept)}, rejected {len(rejected)} (api failures {failed})")
    if rejected:
        from collections import Counter
        kinds = Counter(w.split(":")[0].split()[0] for _, w in rejected)
        print("reject reasons: " + ", ".join(f"{k}={v}" for k, v in kinds.most_common()))
        print(f"full log: {reject_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
