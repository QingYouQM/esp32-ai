"""Measure how repetitive a hand-written corpus is.

Diversity collapse is the failure mode of generated text: nothing is wrong
document by document, but the set converges. A corpus can pass every
per-document check in check_corpus.py and still be worthless, because a
tokenizer trained on 60 near-copies of the same story learns a phrasebook
instead of a language.

Five probes, each answering a different question:

  exact duplicates     - is the same document present twice
  near duplicates      - do two documents say the same thing differently
  shared sentences     - which sentences got reused across documents
  top n-grams          - which phrases carry the corpus
  type-token ratio     - how many distinct characters exist per token

The TTR caveat matters: it falls with corpus length no matter how varied
the text is, so it only means anything compared against another corpus of
similar size. The other probes are length-independent.

    uv run python -m research.tinystories.diversity <corpus>
"""

import argparse
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

SEP = "<|endoftext|>"
SENT = re.compile(r"[^。！？\n]+[。！？]?")


def sentences(doc):
    return [s.strip() for s in SENT.findall(doc) if s.strip()]


def ngrams(seq, n):
    return set(seq[i:i + n] for i in range(len(seq) - n + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus")
    ap.add_argument("--near", type=float, default=0.60,
                    help="Jaccard threshold for near-duplicate reports")
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--open", type=int, default=4,
                    help="characters of document prefix to compare")
    args = ap.parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split(SEP) if d.strip()]
    print(f"{len(docs)} documents, {sum(len(d) for d in docs):,} chars\n")

    # 1. exact duplicates
    dupes = [d for d, n in Counter(docs).items() if n > 1]
    print(f"exact duplicates      {len(dupes)}")
    for d in dupes:
        print(f"    x{docs.count(d)}  {d[:40]}...")

    # 2. near duplicates, char 3-gram Jaccard
    grams = [ngrams(d, 3) for d in docs]
    pairs = []
    for i, j in combinations(range(len(docs)), 2):
        inter = len(grams[i] & grams[j])
        if not inter:
            continue
        jac = inter / len(grams[i] | grams[j])
        if jac >= args.near:
            pairs.append((jac, i, j))
    pairs.sort(reverse=True)
    print(f"\nnear duplicates       {len(pairs)} pairs >= {args.near:.2f}")
    for jac, i, j in pairs[:10]:
        print(f"    {jac:.3f}  [{i}] {docs[i][:26]}")
        print(f"           [{j}] {docs[j][:26]}")

    # 3. sentences reused across documents
    seen = {}
    reused = []
    for idx, doc in enumerate(docs):
        for s in sentences(doc):
            if len(s) < 8:
                continue
            if s in seen and seen[s] != idx:
                reused.append((s, seen[s], idx))
            else:
                seen[s] = idx
    print(f"\nshared sentences      {len(reused)}")
    for s, a, b in reused[:10]:
        print(f"    [{a}/{b}]  {s}")

    # 4. phrases carrying the corpus
    all_grams = Counter()
    for d in docs:
        all_grams.update(d[i:i + args.ngram]
                         for i in range(len(d) - args.ngram + 1))
    total = sum(all_grams.values())
    print(f"\ntop {args.ngram}-grams")
    for g, n in all_grams.most_common(args.top):
        print(f"    {n:5d}  {100 * n / total:5.2f}%  {g}")

    # 5. vocabulary spread
    chars = Counter("".join(docs))
    toks = sum(len(d) for d in docs)
    print(f"\ndistinct characters   {len(chars)}")
    print(f"type-token ratio      {len(chars) / toks:.3f}")

    print("\ntop characters")
    for c, n in chars.most_common(args.top):
        print(f"    {n:5d}  {100 * n / toks:5.2f}%  {c}")

    # 6. structural patterns. These survive the probes above: two documents can
    # share no sentence and no n-gram and still both open the same way, because
    # the repetition is in the shape rather than the words. A register like
    # children's stories wants *some* of this - restricted vocabulary is the
    # point of TinyStories - but a template opener recurring in a third of the
    # corpus means the generator only learned one way to begin.
    print("\nstructural patterns")
    for label, pat in (("opens with yiyitian", "^有一天"),
                       ("opens with xiaoshi", "^小时"),
                       ("opens with zhichi", "^支持|^直到"),
                       ("mentions henjiu", "很久"),
                       ("mentions yihou/houlai", "以后|后来"),
                       ("has dialogue", "[“”]"),
                       ("mentions shuo", "说"),
                       ("mentions meiyou", "没有")):
        hits = sum(1 for d in docs if re.search(pat, d))
        print(f"    {label:24s} {hits:3d}/{len(docs)}  "
              f"{100 * hits / len(docs):5.1f}%")

    opens = Counter(d[:args.open] for d in docs)
    print(f"\nopening {args.open}-char prefixes")
    for o, n in opens.most_common(8):
        print(f"    {n:3d}  {o}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
