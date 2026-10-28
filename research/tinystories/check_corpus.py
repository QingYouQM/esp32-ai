"""Validate a corpus before it reaches the tokenizer.

A corpus written or translated by hand fails in ways a download cannot: a
Latin word slips through, a CJK character is lost to the editor on the way
out, a story drifts outside a sane length band. All of them poison a
tokenizer silently - the BPE trainer will happily merge around them.

Checks per document: length in characters, U+FFFD, Latin letters, and the
empty tail document a trailing separator leaves behind.

Length is a band, not the generation target. A model asked for 150-200
character stories trains happily on 190-275 character documents; forcing
corpus documents down to the generation length truncates plot rather than
improving anything.

    uv run python -m research.tinystories.check_corpus <corpus> \\
        --min-chars 150 --max-chars 300
"""

import argparse
import re
import sys
from pathlib import Path

SEP = "<|endoftext|>"
LATIN = re.compile(r"[A-Za-z]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus")
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-chars", type=int, default=300)
    args = ap.parse_args()

    path = Path(args.corpus)
    text = path.read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split(SEP)]

    bad = 0
    for i, doc in enumerate(docs):
        if not doc:
            continue
        n = len(doc)
        problems = []
        if not args.min_chars <= n <= args.max_chars:
            problems.append(f"length {n}")
        if "\ufffd" in doc:
            problems.append(f"U+FFFD x{doc.count(chr(0xfffd))}")
        latin = LATIN.findall(doc)
        if latin:
            problems.append("latin: " + " ".join(sorted(set(latin))))
        if problems:
            bad += 1
            print(f"[{i}] {'; '.join(problems)}")
            if "\ufffd" in doc:
                j = doc.index("\ufffd")
                print(f"      ...{doc[max(0, j - 12):j + 12]}...")

    kept = [d for d in docs if d]
    lens = [len(d) for d in kept]
    print(f"\ndocs {len(kept)}, chars {sum(lens):,}, "
          f"len min {min(lens)} / median "
          f"{sorted(lens)[len(lens) // 2]} / max {max(lens)}")
    print(f"bytes {path.stat().st_size:,}")
    if bad:
        print(f"\n{bad} document(s) need fixing")
        return 1
    print("\nclean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
