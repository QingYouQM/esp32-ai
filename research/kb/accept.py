"""Generate and automatically check material-grounded Chinese historical stories.

The automatic gate is deliberately conservative. Every entry must declare
literal acceptance.required_spans supported by its material. A candidate must
retain those spans and every declared person, stay within 150-200 Chinese
characters (punctuation excluded), avoid undeclared people and numbers, and
stop cleanly. Paraphrases that do not retain the selected spans are rejected;
that costs recall, but it makes the first host-side gate strict about copying.

These checks do NOT prove historical truth or semantic entailment. A reviewer
must still compare the selected story with the source, and the source entry
must be approved before release. In particular, bigram overlap is not a fact
check and is not used here.

    uv run python -m research.kb.accept --run runs/x.pt \
        --tokenizer data/.../tokenizer.json --attempts 4 --show
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

from research.kb.constrain import ForcedSpans, boundary_ids  # noqa: E402
from research.kb.render import prompt_text  # noqa: E402
from research.kb.spec import STORY_MAX, STORY_MIN  # noqa: E402
from research.kb.validate import (  # noqa: E402
    NAME_RE,
    clean_name,
    listed_people,
    resolve,
)

PUNCT = re.compile(r"[，。；：、！？“”‘’（）\s]")
ECHO = re.compile(r"(人物|时间|地点|起因|经过|结果)：")
NUMBER = re.compile(r"[0-9０-９]+")
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def compact(text):
    return PUNCT.sub("", text)


def story_stats(story, names, limit_hit=False):
    """Language and surface-form measurements for one decoded story."""
    chars = len(compact(story))
    prefixes = []
    for n in names:
        if len(n) >= 3 and n not in story:
            hit = max((k for k in range(2, len(n)) if n[:k] in story), default=0)
            if hit:
                prefixes.append(n[:hit])
    words = compact(story)
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


def material_fields(entry):
    return "\n".join(str(field.get("text", "")) for field in entry["fields"].values())


def story_errors(entry, story, stopped=True):
    """Return every automatic release-gate failure, including fact anchors."""
    names = listed_people(entry["fields"]["人物"]["text"])
    stats = story_stats(story, names, limit_hit=not stopped)
    errors = []
    if not stopped:
        errors.append("STOP")
    if not STORY_MIN <= stats["chars"] <= STORY_MAX:
        errors.append(f"LENGTH:{stats['chars']}")
    if stats["echo"]:
        errors.append("ECHO")
    if stats["hybrid"]:
        errors.append("TRUNCATED_NAME:" + ",".join(stats["prefixes"]))
    if stats["repeat4"] > 0.05:
        errors.append(f"REPEAT:{stats['repeat4']:.0%}")

    for name in names:
        if name not in story:
            errors.append(f"MISSING_PERSON:{name}")

    acceptance = entry.get("acceptance")
    required = (acceptance.get("required_spans") or []) if isinstance(acceptance, dict) else []
    if not required:
        errors.append("FACT_SPEC")
    folded_story = compact(story)
    for span in required:
        if compact(span) not in folded_story:
            errors.append(f"MISSING_FACT:{span}")

    material = material_fields(entry)
    material_numbers = {n.translate(FULLWIDTH_DIGITS) for n in NUMBER.findall(material)}
    for number in sorted(set(NUMBER.findall(story))):
        if number.translate(FULLWIDTH_DIGITS) not in material_numbers:
            errors.append(f"UNSUPPORTED_NUMBER:{number}")

    return errors


def story_warnings(entry, story):
    """Return noisy NER candidates for human review, never as hard failures.

    Chinese surname matching without word segmentation mistakes words such as
    陆续 and 查看 for people. Silently hiding the candidates would miss invented
    people; hard-failing them would reject ordinary prose. Keep them visible.
    """
    names = listed_people(entry["fields"]["人物"]["text"])
    material = material_fields(entry)
    unknown = set()
    for match in NAME_RE.finditer(story):
        candidate = resolve(clean_name(match), names)
        if candidate not in names and candidate not in material:
            unknown.add(candidate)
    return [f"UNDECLARED_PERSON:{name}" for name in sorted(unknown)]


def error_codes(errors):
    return ",".join(sorted({error.split(":", 1)[0] for error in errors})) or "PASS"


def fact_span_ids(tok, entry):
    names = listed_people(entry["fields"]["人物"]["text"])
    acceptance = entry.get("acceptance")
    required = (acceptance.get("required_spans") or []) if isinstance(acceptance, dict) else []
    return [tok.encode(span).ids for span in names + required]


def decode_candidate(model, tok, entry, args, device, attempt):
    ids = tok.encode(prompt_text(entry)).ids
    if len(ids) + args.tokens > model.cfg.seq_len:
        return {
            "attempt": attempt,
            "text": "",
            "stopped": False,
            "errors": ["PROMPT_BUDGET"],
            "warnings": [],
        }
    policy = None
    if args.constrain_facts:
        eod = tok.token_to_id("<|endoftext|>")
        policy = ForcedSpans(fact_span_ids(tok, entry), boundary_ids(tok),
                             args.tokens, eod_id=eod)
    torch.manual_seed(args.seed + attempt)
    out = model.generate(
        torch.tensor([ids], device=device),
        args.tokens,
        temperature=args.temperature,
        top_k=40,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram=4,
        logits_processor=policy,
    )[0].tolist()[len(ids):]
    eod = tok.token_to_id("<|endoftext|>")
    stopped = eod is not None and eod in out
    if stopped:
        out = out[:out.index(eod)]
    story = tok.decode(out).strip()
    errors = story_errors(entry, story, stopped)
    if policy is not None and not policy.complete:
        errors.append("FACT_BUDGET")
    return {
        "attempt": attempt,
        "text": story,
        "stopped": stopped,
        "errors": errors,
        "warnings": story_warnings(entry, story),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--entries", default="research/kb/entries")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--limit", type=int, help="evaluate only the first N entries (smoke tests)")
    ap.add_argument("--constrain-facts", action="store_true",
                    help="force declared people and required spans in narrative order")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--report", type=Path, help="write every candidate and error as JSON")
    args = ap.parse_args()
    if args.attempts < 1:
        ap.error("--attempts must be at least 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.run, map_location=device, weights_only=False)
    model = TinyLM(Config(**checkpoint["cfg"])).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)

    paths = sorted(Path(args.entries).glob("*.json"))
    if args.limit is not None:
        if args.limit < 1:
            ap.error("--limit must be at least 1")
        paths = paths[:args.limit]
    rows = []
    for path in paths:
        entry = json.loads(path.read_text(encoding="utf-8"))
        candidates = []
        for attempt in range(args.attempts):
            candidate = decode_candidate(model, tok, entry, args, device, attempt)
            candidates.append(candidate)
            if not candidate["errors"]:
                break
        selected = next((c for c in candidates if not c["errors"]), None)
        if selected is None:
            selected = min(candidates, key=lambda c: (len(c["errors"]), c["attempt"]))
        names = listed_people(entry["fields"]["人物"]["text"])
        stats = story_stats(selected["text"], names, limit_hit=not selected["stopped"])
        rows.append({
            "id": entry["id"],
            "title": entry["title"],
            "source_status": (entry.get("review") or {}).get("status", "draft"),
            "first_pass": not candidates[0]["errors"],
            "passed": not selected["errors"],
            "attempts": len(candidates),
            "chars": stats["chars"],
            "text": selected["text"],
            "errors": selected["errors"],
            "warnings": selected["warnings"],
            "candidates": candidates,
        })

    if not rows:
        raise SystemExit("nothing measured")

    print(f"{'entry':22s} {'gate':>5s} {'first':>6s} {'tries':>5s} {'chars':>6s}  errors")
    print("-" * 92)
    for row in rows:
        print(f"{row['title'][:20]:22s} "
              f"{'PASS' if row['passed'] else 'FAIL':>5s} "
              f"{'yes' if row['first_pass'] else 'no':>6s} {row['attempts']:5d} "
              f"{row['chars']:6d}  {error_codes(row['errors'])}")

    accepted = sum(row["passed"] for row in rows)
    first = sum(row["first_pass"] for row in rows)
    warned = sum(bool(row["warnings"]) for row in rows)
    print(f"\nautomatic gate {accepted}/{len(rows)}; first candidate {first}/{len(rows)}; "
          f"median chars {statistics.median(row['chars'] for row in rows):.0f}; "
          f"review warnings {warned}/{len(rows)}")
    print("Automatic checks are not historical review. Approved sources and human "
          "story review are still required before release.")

    if args.show:
        for row in rows:
            print(f"\n=== {row['title']} ({row['chars']} chars; "
                  f"{error_codes(row['errors'])}) ===")
            print(row["text"] or "[no story decoded]")
            if row["errors"]:
                print("errors: " + "; ".join(row["errors"]))
            if row["warnings"]:
                print("review warnings: " + "; ".join(row["warnings"]))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "run": args.run,
            "tokenizer": args.tokenizer,
            "attempts": args.attempts,
            "constrain_facts": args.constrain_facts,
            "temperature": args.temperature,
            "seed": args.seed,
            "accepted": accepted,
            "total": len(rows),
            "first_pass": first,
            "rows": rows,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"report: {args.report}")
    return 0 if accepted == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
