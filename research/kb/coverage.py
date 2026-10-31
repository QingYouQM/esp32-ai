"""Measure whether a checkpoint actually uses the material it is given.

Loss on a held-out split says nothing about the only thing that matters here:
does the story contain the people and events of the card in front of it. Two
numbers, both cheap and both brutal:

  entity  share of the names declared in 人物 that appear in the story
  event   share of the character bigrams in 经过 and 结果 that appear in it

A model can score well on val loss and zero on both, which is exactly what a
3M-parameter model trained on a corpus without names or digits does.

--force-names decodes every card a second time under a schedule that forces the
declared names into the story. The two columns then answer two different
questions: what the model reaches for on its own, and whether it can carry a
name it was handed. Entity coverage under the schedule is 100% by construction,
so read the prose in --show, not the number.

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

from research.kb.constrain import ForcedNames, boundary_ids  # noqa: E402
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
    ap.add_argument("--force-names", action="store_true",
                    help="also decode each card with the names forced into the text")
    ap.add_argument("--show", action="store_true",
                    help="print the story behind each row; with --force-names, both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.run, map_location=dev, weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)
    eod = tok.token_to_id("<|endoftext|>")
    sentence_ends = boundary_ids(tok)

    def decode(ids, policy=None):
        x = torch.tensor([ids], device=dev)
        # Same stream both ways, so the two columns differ by the constraint and
        # by nothing else.
        torch.manual_seed(args.seed)
        out = model.generate(x, args.tokens, temperature=args.temperature,
                             top_k=40, repetition_penalty=args.repetition_penalty,
                             no_repeat_ngram=4, logits_processor=policy)[0].tolist()
        out = out[len(ids):]
        if eod is not None and eod in out:
            out = out[:out.index(eod)]
        return tok.decode(out).strip()

    rows, starved = [], []
    for path in sorted(Path(args.entries).glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        ids = tok.encode(prompt_text(entry)).ids
        if len(ids) + args.tokens > ck["cfg"]["seq_len"]:
            print(f"skip {entry['id']}: prompt {len(ids)} tokens does not leave "
                  f"room for {args.tokens} more")
            continue
        names = listed_people(entry["fields"]["人物"]["text"])
        events = bigrams(entry["fields"]["经过"]["text"] + entry["fields"]["结果"]["text"])

        def score(story):
            entity = sum(1 for n in names if n in story) / max(1, len(names))
            return entity, len(events & bigrams(story)) / max(1, len(events))

        free = decode(ids)
        row = {"title": entry["title"], "free": score(free), "free_text": free}
        if args.force_names:
            policy = ForcedNames([tok.encode(n).ids for n in names],
                                 sentence_ends, args.tokens)
            forced = decode(ids, policy)
            if policy.fired < len(names):
                starved.append(f"{entry['title']}: {policy.fired}/{len(names)}")
            row["forced"] = score(forced)
            row["forced_text"] = forced
        rows.append(row)

    if not rows:
        raise SystemExit("nothing measured")

    if args.force_names:
        print(f"{'entry':22s} | {'free':>13s} | {'forced':>13s}")
        print(f"{'':22s} | {'ent':>5s} {'evt':>7s} | {'ent':>5s} {'evt':>7s}")
        print("-" * 58)
    else:
        print(f"{'entry':22s} {'ent':>5s} {'evt':>7s}")
        print("-" * 38)
    for r in rows:
        e, v = r["free"]
        if args.force_names:
            fe, fv = r["forced"]
            print(f"{r['title'][:20]:22s} | {e:5.0%} {v:7.0%} | {fe:5.0%} {fv:7.0%}")
        else:
            print(f"{r['title'][:20]:22s} {e:5.0%} {v:7.0%}")

    med = lambda k, i: statistics.median(r[k][i] for r in rows)
    line = f"median entity {med('free', 0):.0%}  median event {med('free', 1):.0%}"
    if args.force_names:
        line += (f"   |   forced entity {med('forced', 0):.0%}  "
                 f"forced event {med('forced', 1):.0%}")
    print(f"\n{line}  ({len(rows)} entries)")
    if starved:
        print(f"\nWARNING: {len(starved)} card(s) ran out of budget before every "
              f"name got a slot: {', '.join(starved)}")

    if args.show:
        for r in rows:
            print(f"\n=== {r['title']} (free {r['free'][0]:.0%}/{r['free'][1]:.0%}) ===")
            print(r["free_text"])
            if args.force_names:
                print(f"--- names forced ({r['forced'][0]:.0%}/{r['forced'][1]:.0%}) ---")
                print(r["forced_text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
