"""Audit a pairs corpus before it reaches the tokenizer.

Everything here answers one question: is this corpus the thing we meant to
build? A pair can pass per-document validation and still be worthless - if every
protagonist is 小兔子, if the story is a verbatim copy of its own material, or if
two thousand documents are rephrasings of one template.

    uv run python -m research.tinystories.audit_pairs [path] [--sample 3000]
"""

import argparse
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from research.kb.spec import EOD_MARK, FIELDS, MATERIAL_MARK, STORY_MARK

LATIN = re.compile(r"[A-Za-z]")
FIELD_LINE = re.compile(r"^(人物|时间|地点|起因|经过|结果)：(.*)$")


def load(path):
    text = Path(path).read_text(encoding="utf-8")
    return [d.strip() for d in text.split(EOD_MARK) if d.strip()]


def shingles(s, n=5):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="research/tinystories/seeds/zh/pairs.txt")
    ap.add_argument("--sample", type=int, default=400,
                    help="pairs to compare for near-duplication")
    args = ap.parse_args()

    docs = load(args.path)
    bad_structure, field_len, protag, first_person = [], Counter(), Counter(), 0
    mat_chars, story_chars, copy_ratio = [], [], []
    latin = replacement = unbalanced = missing = 0

    for i, doc in enumerate(docs):
        if doc.count(MATERIAL_MARK) != 1 or doc.count(STORY_MARK) != 1:
            bad_structure.append(i)
            continue
        head, story = doc.split(STORY_MARK)
        material = head.replace(MATERIAL_MARK, "").strip()
        story = story.strip()
        lines = [ln for ln in material.splitlines() if ln.strip()]
        labels = [FIELD_LINE.match(ln).group(1) for ln in lines if FIELD_LINE.match(ln)]
        if tuple(labels) != FIELDS:
            missing += 1
            continue
        values = {FIELD_LINE.match(ln).group(1): FIELD_LINE.match(ln).group(2) for ln in lines}
        for k, v in values.items():
            field_len[k] += len(v)
        mat_chars.append(len(material))
        story_chars.append(len(story))
        if LATIN.search(material + story):
            latin += 1
        if "\ufffd" in material + story:
            replacement += 1
        if material.count("“") != material.count("”") or story.count("“") != story.count("”"):
            unbalanced += 1
        head_name = re.split(r"[、，,；;]", values["人物"])[0].strip()
        protag[head_name] += 1
        if story[:40].count("我") or story.startswith("我"):
            first_person += 1
        ms = shingles(material, 6)
        ss = shingles(story, 6)
        if ss:
            copy_ratio.append(len(ms & ss) / len(ss))

    n = len(docs)
    print(f"documents {n:,}  characters {sum(mat_chars) + sum(story_chars):,}")
    print(f"structure: {len(bad_structure)} broken, {missing} with wrong fields")
    print(f"language : latin {latin}, U+FFFD {replacement}, unbalanced quotes {unbalanced}")
    print(f"material : median {statistics.median(mat_chars):.0f} chars "
          f"(min {min(mat_chars)}, max {max(mat_chars)})")
    print(f"story    : median {statistics.median(story_chars):.0f} chars "
          f"(min {min(story_chars)}, max {max(story_chars)})")
    print(f"copy     : median {statistics.median(copy_ratio):.3f} of story 6-grams "
          f"also appear in its material (max {max(copy_ratio):.3f})")
    print(f"first person openings: {first_person} ({first_person / n:.1%})")

    top = protag.most_common(8)
    print("protagonists: " + ", ".join(f"{k} {v / n:.1%}" for k, v in top))
    print(f"distinct protagonist labels: {len(protag):,}")

    # Near duplication: comparing every pair is O(n^2), so sample.
    rng = random.Random(0)
    idx = rng.sample(range(n), min(args.sample, n))
    grams = {i: shingles(docs[i].split(STORY_MARK)[-1]) for i in idx}
    worst, above = 0.0, 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            j = jaccard(grams[idx[a]], grams[idx[b]])
            worst = max(worst, j)
            if j >= 0.6:
                above += 1
    print(f"near duplicates: worst {worst:.3f} in {len(idx) * (len(idx) - 1) // 2:,} "
          f"sampled pairs, {above} above 0.60")

    # Register bias: a phrase reaching a large share of documents is a mould.
    share = Counter()
    for i in rng.sample(range(n), min(2000, n)):
        # The story only: the material carries fixed labels, so counting there
        # reports "人物：" as the commonest phrase in the corpus and says nothing.
        body = docs[i].split(STORY_MARK)[-1]
        for g in {body[j:j + 5] for j in range(len(body) - 4)}:
            share[g] += 1
    top_share = share.most_common(5)
    print("commonest 5-grams: " + ", ".join(f"{g} {c / min(2000, n):.1%}"
                                            for g, c in top_share))

    problems = []
    if bad_structure or missing:
        problems.append("structure")
    if latin or replacement or unbalanced:
        problems.append("language")
    if statistics.median(copy_ratio) > 0.25:
        problems.append("story copies its material")
    if top and top[0][1] / n > 0.15:
        problems.append("one protagonist dominates")
    if worst >= 0.6:
        problems.append("near duplicates")
    print("\nverdict: " + ("pass" if not problems else "review " + ", ".join(problems)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
