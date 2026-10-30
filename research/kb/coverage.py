"""Measure whether a checkpoint actually uses the material it is given.

Loss on a held-out split says nothing about the only thing that matters here:
does the story contain the people and events of the card in front of it. Two
numbers, both cheap and both brutal:

  entity  share of the names declared in 人物 that appear in the story
  event   share of the character bigrams in 经过 and 结果 that appear in it

A model can score well on val loss and zero on both, which is exactly what a
3M-parameter model trained on a corpus without names or digits does.

    uv run python -m research.kb.coverage --run runs/x.pt --tokenizer data/.../tokenizer.json
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


def bigrams(text):
    s = PUNCT.sub("", text)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--entries", default="research/kb/entries")
    ap.add_argument("--tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
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
            print(f"skip {entry['id']}: prompt {len(ids)} tokens does not leave "
                  f"room for {args.tokens} more")
            continue
        x = torch.tensor([ids], device=dev)
        torch.manual_seed(0)
        out = model.generate(x, args.tokens, temperature=args.temperature,
                             top_k=40, repetition_penalty=args.repetition_penalty,
                             no_repeat_ngram=4)[0].tolist()[len(ids):]
        if eod in out:
            out = out[:out.index(eod)]
        story = tok.decode(out).strip()

        names = listed_people(entry["fields"]["人物"]["text"])
        entity = sum(1 for n in names if n in story) / max(1, len(names))
        events = bigrams(entry["fields"]["经过"]["text"] + entry["fields"]["结果"]["text"])
        event = len(events & bigrams(story)) / max(1, len(events))
        rows.append((entry["title"], entity, event, story))

    if not rows:
        raise SystemExit("nothing measured")
    print(f"{'entry':24s} entity  event")
    for title, e, v, _ in rows:
        print(f"{title[:22]:24s} {e:5.0%} {v:6.0%}")
    print(f"\nmedian entity {statistics.median(r[1] for r in rows):.0%}  "
          f"median event {statistics.median(r[2] for r in rows):.0%}  "
          f"({len(rows)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
