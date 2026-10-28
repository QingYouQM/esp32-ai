"""Render a KB entry into the exact bytes the model sees.

The agent that writes entries never writes the card. It fills JSON, and this
module turns that into text, so field order, the full-width colon and the
trailing punctuation are produced once, by code, instead of being re-typed by
something that will eventually get it wrong.
"""

import argparse
import json
import sys
from pathlib import Path

from .spec import EOD_MARK, FIELDS, MATERIAL_MARK, STORY_MARK

ENDINGS = ("。", "！", "？", "…", "；", "：")


def material_text(entry):
    """The six labelled lines, and nothing else."""
    fields = entry["fields"]
    lines = []
    for name in FIELDS:
        text = fields[name]["text"].strip()
        if not text.endswith(ENDINGS):
            text += "。"
        lines.append(f"{name}：{text}")
    return "\n".join(lines)


def prompt_text(entry):
    """What the device feeds the model: material block, then an empty story."""
    return f"{MATERIAL_MARK}\n{material_text(entry)}\n{STORY_MARK}\n"


def pair_text(entry, story):
    """A training document: the same material block plus its story."""
    return f"{prompt_text(entry)}{story.strip()}\n{EOD_MARK}\n"


def main():
    ap = argparse.ArgumentParser(description="render a KB entry to card text")
    ap.add_argument("path", type=Path)
    ap.add_argument("--story", type=Path,
                    help="story file, to render a training pair instead")
    args = ap.parse_args()

    entry = json.loads(args.path.read_text(encoding="utf-8"))
    if args.story:
        sys.stdout.write(pair_text(entry, args.story.read_text(encoding="utf-8")))
    else:
        sys.stdout.write(prompt_text(entry))


if __name__ == "__main__":
    main()
