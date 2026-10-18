"""Health check for the Chinese training corpus.

Three suspects explain a model that loops ("日本國史家" x30) and drifts:

1. Traditional characters: zh.wikipedia ships a traditional/simplified mix, and
   the teacher occasionally emits them too. A model trained on both writes both.
2. Repeated n-grams inside one paragraph: the teacher loops mid-paragraph and
   trim_to_han only cuts *exact sentence* repeats, not phrase-level ones
   ("埃及、埃及、埃及" survives).
3. Boilerplate sentences: if a handful of transition sentences dominate the
   corpus, the model learns them as universal glue.

Usage:
    uv run python -m research.zh.audit
"""

import json
import re
import sys
from collections import Counter

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "data" / "zh" / "raw" / "synth.jsonl"
WIKI = ROOT / "data" / "zh" / "raw" / "wiki_corpus.txt"

CJK = re.compile(r"[\u4e00-\u9fff]")

# A representative set of traditional-only forms; if the corpus were clean
# simplified these would be near zero.
TRAD = set("們會這學國時說對開關體點裡麼後來沒個為與歷機業經濟藝醫藥隨處義務員廳縣區車馬鳥語書長門問題務員驚態擊壓應變數觀點見證據樂園書館圖團員風飛飯館電話腦網絡際聯賽運動員歲質氣層雲霧燈燭營護衛證據齣劇務")


def ngram_repeat_ratio(text, n=6):
    """Share of n-grams that appear more than once in the paragraph."""
    grams = [text[i:i + n] for i in range(0, max(len(text) - n + 1, 1))]
    if len(grams) < 8:
        return 0.0
    c = Counter(grams)
    repeated = sum(v - 1 for v in c.values() if v > 1)
    return repeated / len(grams)


def trad_ratio(text):
    cjk = [ch for ch in text if CJK.match(ch)]
    if not cjk:
        return 0.0
    return sum(1 for ch in cjk if ch in TRAD) / len(cjk)


def audit_synth():
    n_docs = 0
    bad_repeat = 0
    trad_docs = 0
    worst = []
    sent_counter = Counter()
    with open(SYNTH, encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc["text"]
            n_docs += 1
            r = ngram_repeat_ratio(text)
            if r > 0.05:
                bad_repeat += 1
                worst.append((r, doc["topic"], text[:60]))
            if trad_ratio(text) > 0.01:
                trad_docs += 1
            for s in re.split(r"[。！？]", text):
                s = s.strip()
                if 8 <= len(s) <= 40:
                    sent_counter[s] += 1
    print(f"== synth.jsonl: {n_docs:,} paragraphs ==")
    print(f"  6-gram repeat >5% : {bad_repeat:,} ({100*bad_repeat/max(n_docs,1):.1f}%)")
    print(f"  traditional chars : {trad_docs:,} docs ({100*trad_docs/max(n_docs,1):.1f}%)")
    print("  worst repeat offenders:")
    for r, topic, head in sorted(worst, reverse=True)[:5]:
        print(f"    {r:5.1%} [{topic}] {head}")
    print("  top boilerplate sentences:")
    for s, c in sent_counter.most_common(8):
        print(f"    {c:6,}x  {s}")


def audit_wiki(max_lines=100000):
    n = 0
    trad_docs = 0
    trad_chars = 0
    total_cjk = 0
    bad_repeat = 0
    with open(WIKI, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            n += 1
            cjk = [ch for ch in text if CJK.match(ch)]
            total_cjk += len(cjk)
            t = sum(1 for ch in cjk if ch in TRAD)
            trad_chars += t
            if t > len(cjk) * 0.01:
                trad_docs += 1
            if ngram_repeat_ratio(text) > 0.05:
                bad_repeat += 1
            if n >= max_lines:
                break
    print(f"== wiki (first {n:,} paras) ==")
    print(f"  traditional char share: {100*trad_chars/max(total_cjk,1):.2f}%")
    print(f"  docs with >1% trad    : {trad_docs:,} ({100*trad_docs/max(n,1):.1f}%)")
    print(f"  6-gram repeat >5%     : {bad_repeat:,} ({100*bad_repeat/max(n,1):.1f}%)")


if __name__ == "__main__":
    audit_synth()
    print()
    audit_wiki()
