"""Check a KB entry against docs/zh-material-spec.md.

Three things are being defended here, in rising order of importance:

  * the card is well formed, so the renderer's output is what training saw;
  * the material has no narration or hearsay in it, because the model would
    repeat either as fact;
  * every field's text is backed by a verbatim quote from the source. A field
    with no quote has to declare itself inferred and is reported as such, so a
    guess never enters the set disguised as a citation.

Run it on a file or a directory:

    uv run python -m research.kb.validate research/kb/examples
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .spec import (
    COMPOUND_SURNAMES,
    DEFAULT_NAMES,
    FIELD_MAX,
    FIELDS,
    MATERIAL_MAX,
    NARRATION_STRONG,
    NARRATION_WEAK,
    PLACE_TAILS,
    SURNAMES,
    TITLE_MAX,
    VERBS_AFTER_NAME,
)

NAME_RE = re.compile(
    "(?:" + "|".join(COMPOUND_SURNAMES) + "|[" + SURNAMES + "])[\u4e00-\u9fa5]{1,2}"
)
PLACE_RE = re.compile("[\u4e00-\u9fa5]{1,4}[" + PLACE_TAILS + "]")
LATIN_RE = re.compile("[A-Za-zＡ-Ｚａ-ｚ]")
DIGIT_RE = re.compile("[0-9０-９]")
HALFWIDTH_RE = re.compile(r"[,.;:!?()\[\]\"']")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
PARTICLES = "之氏父子等辈们"


@dataclass
class Problem:
    level: str
    code: str
    detail: str

    def __str__(self):
        return f"[{self.level}] {self.code}: {self.detail}"


def clean_name(match):
    name = match.group(0)
    while len(name) > 2 and name[-1] in PARTICLES:
        name = name[:-1]
    return name


def listed_people(text):
    """The names declared in 人物, one per comma-separated segment."""
    names = []
    for chunk in re.split(r"[；;]", text):
        head = re.split(r"[、，,]", chunk)[0].strip()
        m = NAME_RE.match(head)
        if m:
            names.append(clean_name(m))
    return names


def resolve(candidate, declared):
    """Map a raw match back to a declared name when the match swallowed a verb.

    The surname regex cannot see word boundaries, so "孙权送来" matches as one
    candidate. If a declared name is a prefix of it and what follows is a verb,
    the candidate is that person, not a new one.
    """
    while len(candidate) > 2 and candidate[-1] in PARTICLES:
        candidate = candidate[:-1]
    if candidate in declared:
        return candidate
    for name in declared:
        if (len(candidate) > len(name) and candidate.startswith(name)
                and candidate[len(name)] in VERBS_AFTER_NAME):
            return name
    if len(candidate) > 2 and candidate[-1] in VERBS_AFTER_NAME:
        candidate = candidate[:-1]
    return candidate


def normalise(text):
    return re.sub(r"\s+", "", text)


def check(entry, verbose=False):
    problems = []
    add = lambda level, code, detail: problems.append(Problem(level, code, detail))

    for key in ("id", "title", "source", "fields"):
        if key not in entry:
            add("error", "E1-SCHEMA", f"missing key {key!r}")
    if problems:
        return problems

    if not ID_RE.match(str(entry["id"])):
        add("error", "E1-SCHEMA", "id must be lower-case letters, digits and dashes")
    if len(str(entry["title"])) > TITLE_MAX:
        add("error", "E2-TITLE", f"title {len(entry['title'])} > {TITLE_MAX}")

    source = entry["source"]
    if not str(source.get("work", "")).strip():
        add("error", "H1-SOURCE", "source.work is empty")
    if not str(source.get("url", "")).strip():
        add("warn", "H1-SOURCE", "no source.url; the entry cannot be re-checked")
    quotes = source.get("quotes") or []
    if not quotes or not all(isinstance(q, str) and q.strip() for q in quotes):
        add("error", "H1-SOURCE", "source.quotes must hold at least one quote")

    fields = entry["fields"]
    missing = [f for f in FIELDS if f not in fields]
    extra = [f for f in fields if f not in FIELDS]
    for f in missing:
        add("error", "E3-FIELDS", f"missing field {f}")
    for f in extra:
        add("error", "E3-FIELDS", f"unknown field {f}")
    if missing:
        return problems

    total = 0
    for name in FIELDS:
        text = str(fields[name].get("text", "")).strip()
        total += len(text)
        if not text:
            add("error", "E3-FIELDS", f"{name} is empty")
            continue
        if len(text) > FIELD_MAX[name]:
            add("error", "M2-LENGTH", f"{name} {len(text)} > {FIELD_MAX[name]}")
        if LATIN_RE.search(text):
            add("error", "M5-CHARSET", f"{name} contains Latin letters")
        if "\ufffd" in text:
            add("error", "M5-CHARSET", f"{name} contains U+FFFD")
        if HALFWIDTH_RE.search(text):
            add("error", "M6-PUNCT", f"{name} uses half-width punctuation")
        for word in DEFAULT_NAMES:
            if word in text:
                add("error", "M8-DEFAULT", f"{name} uses the default name {word}")
        for word in NARRATION_STRONG:
            if word in text:
                add("error", "H2-NARRATION", f"{name} contains {word!r}")
        for word in NARRATION_WEAK:
            if word in text:
                add("warn", "H2-NARRATION", f"{name} contains {word!r}")
        if verbose and name != "时间" and DIGIT_RE.search(text):
            add("warn", "M4-DIGITS", f"{name} uses Arabic digits")

    if total > MATERIAL_MAX:
        add("error", "M2-LENGTH", f"material {total} > {MATERIAL_MAX}")

    for name in FIELDS:
        text = str(fields[name].get("text", ""))
        if text.count("“") != text.count("”"):
            add("error", "M6-PUNCT", f"{name} has unbalanced quotes")

    # Evidence: every field is quoted or explicitly declared inferred.
    haystack = "".join(normalise(q) for q in quotes)
    for name in FIELDS:
        spec = fields[name]
        evidence = normalise(str(spec.get("evidence") or ""))
        inferred = bool(spec.get("inferred"))
        if not evidence:
            if inferred:
                add("warn", "E4-INFERRED", f"{name} has no quote and is marked inferred")
            else:
                add("error", "E4-EVIDENCE", f"{name} has no quote and is not inferred")
        elif evidence not in haystack:
            add("error", "E4-EVIDENCE", f"{name}'s quote is not verbatim in source")

    # M3: the people the card declares must actually appear in the events.
    declared = listed_people(str(fields["人物"].get("text", "")))
    if not declared:
        add("error", "M3-ENTITY", "人物 names nobody")
    body = "".join(str(fields[f].get("text", "")) for f in ("起因", "经过", "结果"))
    for person in declared:
        if person not in body:
            add("error", "M3-ENTITY", f"{person} is declared but never acts")

    seen, acts = {}, {}
    for m in NAME_RE.finditer(body):
        name = resolve(clean_name(m), declared)
        seen[name] = seen.get(name, 0) + 1
        after = body[m.start() + len(name):m.start() + len(name) + 1]
        acts[name] = acts.get(name, False) or after in VERBS_AFTER_NAME
    # The reverse direction - a name used but never declared - cannot be read
    # off a text without a word segmenter: "石头" starts with a surname too.
    # It is a review aid, so it is reported only on request.
    if verbose:
        for name, count in seen.items():
            if name not in declared and (acts[name] or count >= 2):
                add("warn", "M3-ENTITY", f"{name!r} appears but is not declared in 人物")

    unknown_places = {
        clean_name(m) for m in PLACE_RE.finditer(body)
    } - {clean_name(m) for m in PLACE_RE.finditer(str(fields["地点"].get("text", "")))}
    for place in sorted(unknown_places):
        if body.count(place) >= 2:
            add("warn", "M3-ENTITY", f"place {place!r} is used but not in 地点")

    status = (entry.get("review") or {}).get("status", "draft")
    if status not in ("draft", "pending", "approved"):
        add("error", "E5-REVIEW", f"unknown review.status {status!r}")

    return problems


def iter_entries(paths):
    for path in paths:
        files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for file in files:
            if file.name == "schema.json":
                continue
            yield file, json.loads(file.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="validate KB entries")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also report names used but never declared (noisy)")
    args = ap.parse_args()

    errors = 0
    seen_ids = {}
    for file, entry in iter_entries(args.paths):
        problems = check(entry, verbose=args.verbose)
        entry_id = entry.get("id")
        if entry_id in seen_ids and seen_ids[entry_id] != file:
            problems.append(Problem("error", "E6-DUPLICATE",
                                    f"id {entry_id!r} also used by {seen_ids[entry_id]}"))
        seen_ids[entry_id] = file
        bad = sum(1 for p in problems if p.level == "error")
        errors += bad
        if problems:
            print(f"{file}: {bad} error(s), {len(problems) - bad} warning(s)")
            for problem in problems:
                print(f"  {problem}")

    if errors:
        print(f"\n{errors} error(s). Nothing here may reach the model.")
        return 1
    print("all entries pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
