"""Open every story with its card's 起因 sentence.

Round one of the synthetic copy task proved the 3M core can copy: on 200 held-out
synthetic cards it scores 100% entity and 14% event with no forcing at all. On
the 24 real cards it scores 0% and 0%. The behaviour was learned as a match to
the twelve synthetic templates, not as "look at the card", and the real cards
match none of them.

The template is therefore removed and the prose is left alone. The 起因 field is
already one sentence about a concrete action that names the people involved, so
prepending it to the story makes the target copyable *inside real prose* -- the
model sees a real card, real narration, and a first sentence that it must lift
from the material. Nothing else about the corpus changes.

    uv run python -m research.tinystories.ground_stories --in data/combined.txt ^
        --out data/copy2/combined.txt
"""

import argparse
import re
import sys
from pathlib import Path

FIELD = re.compile(r"^(人物|时间|地点|起因|经过|结果)：(.*)$", re.M)
STORY = "<|story|>"


def split_doc(doc):
    """(material block, story) or None when this document is not a pair."""
    if STORY not in doc:
        return None
    material, story = doc.split(STORY, 1)
    fields = dict(FIELD.findall(material))
    if "起因" not in fields:
        return None
    return material, story, fields["起因"].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    text = Path(args.src).read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split("<|endoftext|>") if d.strip()]

    out, opened, skipped = [], 0, 0
    for doc in docs:
        parts = split_doc(doc)
        if parts is None:
            out.append(doc)
            skipped += 1
            continue
        material, story, cause = parts
        body = story.lstrip("\n")
        if body.startswith(cause):
            out.append(doc)          # already grounded; do not say it twice
            continue
        out.append(f"{material}{STORY}\n{cause}{body}")
        opened += 1

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    Path(args.dst).write_text("\n<|endoftext|>\n".join(out) + "\n<|endoftext|>\n",
                              encoding="utf-8")
    print(f"{len(docs)} docs: {opened} opened with their 起因, "
          f"{skipped} not pairs, -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
