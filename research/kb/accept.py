"""Acceptance metrics for the shipped contract: a short story generator.

The entity/event numbers in coverage.py measure a promise this project does not
keep. Measured on 24 real cards, both are ~0% at 3M and at 10M, before and after
constrained decoding -- the model writes a story, it does not retell a card.
C decided to sell the thing that works (see docs/zh-story-prompt.md), so the
gate has to move with it.

What is measured here is whether a story is *usable*:

  stop      the model ended the document itself instead of running to the
            token limit. The corpus had no document boundary until 15396eb,
            so any checkpoint older than that scores 0 and must not ship.
  chars     median length in characters. Too short is a fragment, too long is
            a runaway.
  echo      the story started reciting the material template ("人物：...").
            The old bug that made every long decode collapse into a card.
  hybrid    a name from the card appears truncated ("陈嘉" for "陈嘉庚"). Not
            fatal, but the reader sees a person who does not exist.
  exact     full names that happen to land in the story. Diagnostic only: it
            is not a target, and a checkpoint that scores 0 is not broken.

Names are read from the card, never required of the story.

    uv run python -m research.kb.accept --run runs/x.pt --tokenizer data/.../tokenizer.json
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from model import Config, TinyLM  # noqa: E402

from research.kb.render import prompt_text  # noqa: E402
from research.kb.validate import listed_people  # noqa: E402

PUNCT = re.compile(r"[，。；：、！？“”‘’（）\s]")
# Field labels, as render.py writes them.
ECHO = re.compile(r"(人物|时间|地点|起因|经过|结果)：")


def story_stats(story, names, limit_hit=False):
    """Everything the gate needs to know about one decoded story."""
    chars = len(PUNCT.sub("", story))
    prefixes = []
    for n in names:
        if len(n) >= 3 and n not in story:
            # Longest proper prefix that does appear: 2 chars is the floor,
            # since a single shared character means nothing.
            hit = max((k for k in range(2, len(n)) if n[:k] in story), default=0)
            if hit:
                prefixes.append(n[:hit])
    words = PUNCT.sub("", story)
    grams = [words[i:i + 4] for i in range(len(words) - 3)]
    return {
        "chars": chars,
        "echo": bool(ECHO.search(story)),
        "hybrid": len(prefixes) > 0,
        "prefixes": prefixes,
        "exact": sum(1 for n in names if n in story) / max(1, len(names)),
        "repeat4": 0.0 if not grams else 1 - len(set(grams)) / len(grams),
        "limit_hit": limit_hit,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--entries", default="research/kb/entries")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.run, map_location=dev, weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)
    eod = tok.token_to_id("<|endoftext|>")

    rows = []
    for path in sorted(Path(args.entries).glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        ids = tok.encode(prompt_text(entry)).ids
        if len(ids) + args.tokens > ck["cfg"]["seq_len"]:
            continue
        torch.manual_seed(args.seed)
        out = model.generate(torch.tensor([ids], device=dev), args.tokens,
                             temperature=args.temperature, top_k=40,
                             repetition_penalty=args.repetition_penalty,
                             no_repeat_ngram=4)[0].tolist()[len(ids):]
        # Whether the model closed the document is the first thing to read, so
        # it is recorded before the EOD is stripped for scoring.
        stopped = eod is not None and eod in out
        if stopped:
            out = out[:out.index(eod)]
        story = tok.decode(out).strip()
        s = story_stats(story, listed_people(entry["fields"]["人物"]["text"]),
                        limit_hit=not stopped)
        s["title"] = entry["title"]
        s["text"] = story
        rows.append(s)

    if not rows:
        raise SystemExit("nothing measured")

    print(f"{'entry':22s} {'stop':>5s} {'chars':>6s} {'echo':>5s} {'hyb':>5s} {'rpt4':>6s}")
    print("-" * 52)
    for r in rows:
        print(f"{r['title'][:20]:22s} {'no' if r['limit_hit'] else 'yes':>5s} "
              f"{r['chars']:6d} {'YES' if r['echo'] else '.':>5s} "
              f"{'YES' if r['hybrid'] else '.':>5s} {r['repeat4']:6.0%}")

    n = len(rows)
    stop = sum(1 for r in rows if not r["limit_hit"]) / n
    print(f"\nstop {stop:.0%}  median chars {statistics.median(r['chars'] for r in rows):.0f}"
          f"  echo {sum(r['echo'] for r in rows)}/{n}"
          f"  hybrid {sum(r['hybrid'] for r in rows)}/{n}"
          f"  median repeat4 {statistics.median(r['repeat4'] for r in rows):.0%}"
          f"  exact names {statistics.median(r['exact'] for r in rows):.0%} (diagnostic)")

    if args.show:
        for r in rows:
            print(f"\n=== {r['title']} ({r['chars']} chars"
                  f"{', truncated' if r['limit_hit'] else ''}) ===")
            print(r["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
