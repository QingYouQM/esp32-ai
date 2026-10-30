"""Re-check an existing pairs file and drop documents that no longer pass.

The per-document checks tighten as bugs are found - the story half was never
checked for Latin, lost characters, quote balance or copying, so a few percent
of a finished corpus can be contaminated. Re-filtering is cheaper than
regenerating, and it is the only way to apply a new rule to documents that were
written before it existed.

    uv run python -m research.tinystories.filter_pairs --dry-run
    uv run python -m research.tinystories.filter_pairs --apply
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from research.kb.spec import EOD_MARK, FIELDS, MATERIAL_MARK, STORY_MARK

from .make_pairs import check_material


def parse(doc):
    head, story = doc.split(STORY_MARK, 1)
    material = head.replace(MATERIAL_MARK, "").strip()
    fields = {}
    for line in material.splitlines():
        line = line.strip()
        if "：" in line:
            k, v = line.split("：", 1)
            if k in FIELDS:
                fields[k] = v.strip()
    return fields, story.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="research/tinystories/seeds/zh/pairs.txt")
    ap.add_argument("--apply", action="store_true", help="rewrite the file")
    args = ap.parse_args()

    path = Path(args.path)
    text = path.read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split(EOD_MARK) if d.strip()]

    keep, reasons = [], Counter()
    for doc in docs:
        try:
            fields, story = parse(doc)
        except ValueError:
            reasons["structure"] += 1
            continue
        ok, why = check_material(fields, story)
        if ok:
            keep.append(doc)
        else:
            reasons[why.split(" ")[0] + " " + why.split(" ")[1] if " " in why else why] += 1

    print(f"documents {len(docs):,} -> {len(keep):,} "
          f"({len(docs) - len(keep):,} dropped, {(len(docs) - len(keep)) / len(docs):.1%})")
    for why, n in reasons.most_common():
        print(f"  {n:5d}  {why}")

    if args.apply:
        out = "\n".join(d + "\n" + EOD_MARK for d in keep) + "\n"
        path.write_text(out, encoding="utf-8", newline="\n")
        print(f"rewrote {path}")
    else:
        print("dry run; pass --apply to rewrite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
