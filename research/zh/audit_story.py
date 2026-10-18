"""Audit the story corpus specifically: it never got the health check that
the explanatory corpus and wiki got, and v3's gibberish output
("xixixixi...", "三字四字") smells like training data poison."""

import json
import re
import sys
from collections import Counter

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import ngram_repeat_ratio, trad_ratio

ROOT = Path(__file__).resolve().parents[2]
STORY = ROOT / "data" / "zh" / "raw" / "synth_story.jsonl"

LEAK = re.compile(r"Human:|Assistant:|请注意|我会尽力|【|答案[:：]|xixi|哈哈|嘻嘻|嘿嘿")
ASCII_RUN = re.compile(r"[A-Za-z]{6,}")
NON_CJK = re.compile(r"[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s]")


def main():
    n = bad_repeat = trad = leak = ascii_run = 0
    worst = []
    han = Counter()
    samples = []
    with open(STORY, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = d["text"]
            n += 1
            r = ngram_repeat_ratio(t)
            if r > 0.05:
                bad_repeat += 1
                worst.append((r, d["topic"], t[:60]))
            if trad_ratio(t) > 0.01:
                trad += 1
            if LEAK.search(t):
                leak += 1
                if len(samples) < 5:
                    samples.append((d["topic"], t[:80]))
            if ASCII_RUN.search(t):
                ascii_run += 1
            for ch in t:
                if "\u4e00" <= ch <= "\u9fff":
                    han[ch] += 1
    print(f"docs={n:,}")
    print(f"  repeat>5%   : {bad_repeat:,} ({100*bad_repeat/n:.1f}%)")
    print(f"  traditional : {trad:,} ({100*trad/n:.1f}%)")
    print(f"  leak marks  : {leak:,} ({100*leak/n:.1f}%)")
    print(f"  ascii runs  : {ascii_run:,} ({100*ascii_run/n:.1f}%)")
    print("  worst repeat:")
    for r, topic, head in sorted(worst, reverse=True)[:3]:
        print(f"    {r:5.1%} [{topic}] {head}")
    print("  leak samples:")
    for topic, head in samples:
        print(f"    [{topic}] {head}")
    print(f"  top hanzi: {' '.join(c for c, _ in han.most_common(15))}")


if __name__ == "__main__":
    main()
