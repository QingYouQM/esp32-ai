"""E1 generalisation probe: copying must survive names it has never seen *as a
combination*.

make_copy_task's "100% entity on held-out cards" was a false positive. Its
held-out split kept documents apart but not surface forms: 32 surnames x 24
given characters means every evaluation name was covered character-by-character
so well that the model could score by prior, not by reading the card. The 24
real cards, whose names (陈嘉庚, 黄乃裳) live outside that space, scored 0%.

This generator keeps the twelve scenes and the verbatim turn/result quotes
(same copy supervision as round one) and changes only the name space:

  surnames   ~400, shared between train and evaluation
  given      400 single-token CJK characters drawn from the tokenizer, shared
             between train and evaluation
  split      the *full name string* of an eval-unseen card never appears in the
             training corpus, while every character in it appears often

Two evaluation sets are written as KB entries with acceptance.required_spans,
so research/kb/accept.py scores them exactly like the real cards:

  eval-unseen  names absent from training (character prior only)
  eval-seen    names that occur in training, on *other* cards

The difference between the two is what the name prior buys; the absolute
unseen score is the conditional-copy capability this experiment measures.

    uv run python -m research.tinystories.make_copy_gen \
        --count 8000 --tokenizer data/zh-ground/vocab-16384/tokenizer.json \
        --out data/copygen/pairs.txt --entries-dir data/copygen/entries
"""

import argparse
import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders

from research.kb.render import pair_text
from research.kb.spec import (
    COMPOUND_SURNAMES,
    FIELD_MAX,
    MATERIAL_MAX,
    STORY_MAX,
    STORY_MIN,
    SURNAMES,
)
from research.tinystories.make_copy_task import (
    BODIES,
    BEATS,
    CLOSERS,
    OPENERS,
    SCENES,
    SCENE_OBJS,
    SCENE_ROLES,
    make_place,
    make_time,
)

# Characters that cannot start a plausible given name, and characters that are
# surnames (a name like 王王 reads as a typo, and reusing surname characters
# lets the NER regex swallow them).
STOP_CHARS = set(
    "的了是你我他她它这那和就也都而及与之或被在从对给让向跟往上下中前后里外"
    "很太更最又再还只都就便即令使得把个些每各某此那哪吗呢吧啊呀哦嘛么"
    "一二三四五六七八九十百千万亿零两双几个许多少有无没是不"
    "说说看看听听走走跑跑来到去开开关关起落上下出入进来回过"
    "地得着过呢吧啊嘛哟咧啦嗡哼嗯嘘"
    "，。；：、！？“”‘’（）—…《》〈〉「」『』【】"
)
GIVEN_POOL = 400               # single-token characters the given may draw on
GIVEN_LENGTHS = (1, 2, 2, 2)   # real cards are almost all two-char givens


def single_char_cjk(tok):
    """Characters the tokenizer encodes as exactly one token."""
    bl = decoders.ByteLevel()
    out = []
    for t in tok.get_vocab():
        try:
            d = bl.decode([t])
        except Exception:
            continue
        if len(d) == 1 and "\u4e00" <= d <= "\u9fff":
            out.append(d)
    return out


def build_given_pool(tok, rng, surname_chars):
    cand = [c for c in single_char_cjk(tok)
            if c not in surname_chars and c not in STOP_CHARS]
    if len(cand) < GIVEN_POOL:
        raise SystemExit(f"tokenizer has only {len(cand)} usable characters; "
                         f"need {GIVEN_POOL}")
    return rng.sample(cand, GIVEN_POOL)


def make_name(rng, surnames, given_pool):
    """Return (full name, given characters), surname first."""
    surname = rng.choice(surnames)
    given = [rng.choice(given_pool)
             for _ in range(rng.choice(GIVEN_LENGTHS))]
    return surname + "".join(given), [surname] + given


def build(rng, name, tries=50):
    """A document from the round-one scene set, with the name supplied.

    The scene is resampled until the story lands in 150-200 chars: short
    opener/body/closer combinations exist and a truncated story teaches the
    model to stop mid-sentence, which is exactly what the corpus rules forbid.
    """
    for _ in range(tries):
        scene, fields, story, when = _build_once(rng, name)
        if STORY_MIN <= len(story) <= STORY_MAX:
            return scene, fields, story, when
    raise RuntimeError(f"no in-length story for {name!r}")


def _build_once(rng, name):
    scene = rng.choice(SCENES)
    role = rng.choice(SCENE_ROLES[scene["title"]])
    obj = rng.choice(SCENE_OBJS[scene["title"]])
    p1 = make_place(rng, ())
    p2 = make_place(rng, (p1,))
    when = make_time(rng)
    s = dict(name=name, role=role, obj=obj, p1=p1, p2=p2, time=when)

    turn = scene["turn"].format(**s)
    result = scene["result"].format(**s)
    cause = scene["cause"].format(**s)

    beat_a, beat_b = rng.sample(BEATS, 2)
    story = (rng.choice(OPENERS).format(**s)
             + rng.choice(BODIES).format(**s, cause=cause)
             + turn + "。"
             + beat_a + "。"
             + beat_b + "。"
             + result + "。"
             + rng.choice(CLOSERS).format(**s) + "。")

    fields = {
        "人物": f"{name}，{p1}的{role}。",
        "时间": f"{when}。",
        "地点": f"{p1}，还有{p2}。",
        "起因": f"{name}{cause}。",
        "经过": f"{name}走到{p1}，{turn}。",
        "结果": f"{result}。",
    }
    assert name in story, "name missing from the story"
    assert turn in story, "the turn is not quoted"
    assert result in story, "the result is not quoted"
    for k, v in fields.items():
        assert len(v) <= FIELD_MAX[k], f"{k} over {FIELD_MAX[k]}: {len(v)}"
    assert sum(len(v) for v in fields.values()) <= MATERIAL_MAX
    return scene, fields, story, when


def entry(idx, scene, fields, when, group, name):
    # The spans a story must retain verbatim: the 经过 turn and the 结果
    # phrase, both of which quote the material by construction. The turn span
    # is the part after the leading "name走到p1，" so it tests copying, not
    # prefix matching.
    return {
        "id": f"syn-gen-{group}-{idx:04d}",
        "title": f"{scene['title']}·{when}"[:24],
        "source": {"work": "synthetic E1 copy probe", "url": "",
                   "revision": None,
                   "quotes": ["generated by make_copy_gen; no external source"]},
        "fields": {k: {"text": v, "evidence": None, "inferred": True}
                   for k, v in fields.items()},
        "acceptance": {
            "required_spans": [fields["经过"].strip("。").split("，", 1)[-1],
                               fields["结果"].strip("。")],
        },
        "notes": f"{scene['axis']}; group={group}; name={name}",
        "review": {"status": "draft", "by": None, "at": None},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=8000)
    ap.add_argument("--tokenizer", required=True,
                    help="paired tokenizer whose single-token characters seed "
                         "the given-name pool")
    ap.add_argument("--out", required=True, help="training pairs file")
    ap.add_argument("--entries-dir", required=True,
                    help="where the seen/unseen evaluation cards are written")
    ap.add_argument("--eval-seen", type=int, default=100)
    ap.add_argument("--eval-unseen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = Tokenizer.from_file(args.tokenizer)
    surname_chars = set(SURNAMES)
    surnames = list(SURNAMES) + list(COMPOUND_SURNAMES)
    given_pool = build_given_pool(tok, rng, surname_chars)

    out, held = Path(args.out), Path(args.entries_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    (held / "eval-unseen").mkdir(parents=True, exist_ok=True)
    (held / "eval-seen").mkdir(parents=True, exist_ok=True)

    # ---- training documents -------------------------------------------------
    docs, train_names, char_freq = [], {}, {}
    for _ in range(args.count):
        name, chars = make_name(rng, surnames, given_pool)
        train_names[name] = train_names.get(name, 0) + 1
        for ch in chars:
            char_freq[ch] = char_freq.get(ch, 0) + 1
        scene, fields, story, when = build(rng, name)
        e = entry(0, scene, fields, when, "train", name)
        docs.append(pair_text(e, story))
    out.write_text("".join(docs), encoding="utf-8")

    # ---- evaluation cards ---------------------------------------------------
    def unseen_name():
        for _ in range(1000):
            name, _ = make_name(rng, surnames, given_pool)
            if name not in train_names:
                return name
        raise RuntimeError("name space exhausted")

    for i in range(args.eval_seen):
        name = rng.choice(list(train_names))
        scene, fields, story, when = build(rng, name)
        e = entry(i, scene, fields, when, "eval-seen", name)
        (held / "eval-seen" / f"{e['id']}.json").write_text(
            json.dumps(e, ensure_ascii=False, indent=2), encoding="utf-8")

    unseen_names = []
    for i in range(args.eval_unseen):
        name = unseen_name()
        unseen_names.append(name)
        scene, fields, story, when = build(rng, name)
        e = entry(i, scene, fields, when, "eval-unseen", name)
        (held / "eval-unseen" / f"{e['id']}.json").write_text(
            json.dumps(e, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- hygiene report -----------------------------------------------------
    # The claims this generator makes about its own split, checked here so the
    # numbers travel with the corpus instead of living in a chat log.
    corpus = "".join(docs)
    dupes = sum(1 for v in train_names.values() if v > 1)
    compounds = set(COMPOUND_SURNAMES)
    def given_chars(name):
        skip = 2 if name[:2] in compounds else 1
        return name[skip:]
    freqs = sorted(char_freq.get(ch, 0)
                   for n in unseen_names for ch in given_chars(n))
    print(f"{len(docs)} training pairs -> {out} "
          f"({out.stat().st_size / 1e6:.2f} MB)")
    print(f"distinct training names: {len(train_names)} "
          f"({dupes} repeated); surname pool {len(surnames)}, "
          f"given pool {len(given_pool)}")
    print(f"given-character coverage over training: min {min(freqs)} / median "
          f"{freqs[len(freqs) // 2]} occurrences per character "
          f"(surname characters excluded)")
    leak_seen = sum(json.loads(p.read_text(encoding="utf-8"))
                    ["notes"].split("name=")[-1] in corpus
                    for p in (held / "eval-seen").glob("*.json"))
    leak_unseen = sum(n in corpus for n in unseen_names)
    print(f"eval-seen: {args.eval_seen} cards, {leak_seen} names appear "
          f"verbatim in training (expected: all)")
    print(f"eval-unseen: {args.eval_unseen} cards, {leak_unseen} names appear "
          f"verbatim in training (must be 0)")
    assert leak_unseen == 0, "unseen split leaked"
    return 0


if __name__ == "__main__":
    sys.exit(main())
