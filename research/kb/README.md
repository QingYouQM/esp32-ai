# research/kb

The external resource set: one event per entry, six fields, every field either
quoted from the source or explicitly marked inferred.

The format is fixed by [docs/zh-material-spec.md](../../docs/zh-material-spec.md).
The training corpus and this set have to be byte-identical, because the model
learns one shape and is deployed on that same shape.

The card is a **seed for a story, not a source of facts**: measured on 24 real
cards the model reproduces neither the names nor the events, at 3M or at 10M.
The shipped contract (what a story must and must not do) is
[docs/zh-story-prompt.md](../../docs/zh-story-prompt.md); the release gate there
is `accept.py`, not entity coverage.

```
schema.json          what an entry may contain
render.py            entry -> card text (the only writer of the format)
validate.py          entry -> errors and warnings
examples/            one worked entry
PROMPT.md            the prompt that produces entries from sources
```

## Use

```bash
uv run python -m research.kb.validate research/kb/entries      # gate
uv run python -m research.kb.render research/kb/entries/x.json # card text
```

Validation is not advisory. An entry with an error has no quote behind one of
its fields, or a person who never acts, or two hundred characters of material
where three hundred are allowed. None of that shows up in the model's output as
an error - it shows up as a prompt that reads wrong and seeds a worse story.

## What is checked

| code | refuses |
|---|---|
| E1-SCHEMA | missing keys, malformed id |
| E2-TITLE | title too long |
| E3-FIELDS | missing or unknown field, empty text |
| E4-EVIDENCE | a field claims content its quote does not contain |
| E5-REVIEW | unknown review status |
| E6-DUPLICATE | the same id used twice |
| M2-LENGTH | field or material over budget |
| M3-ENTITY | a declared person who never acts |
| M4-DIGITS | Arabic digits outside 时间 (warning) |
| M5-CHARSET | Latin letters, U+FFFD |
| M6-PUNCT | half-width punctuation, unbalanced quotes |
| M8-DEFAULT | 小明/张三 and friends: the model must have no default name |
| H1-SOURCE | no source, or no verbatim quotes |
| H2-NARRATION | 据说/我认为 inside the material |

`--verbose` additionally reports names used but never declared. It stays off
because a text without word segmentation cannot tell 石头 from a surname.

## What is not checked

Entailment. evidence is verified to *appear* in the quote, not to *support* the
claim. That is the reviewer's job, and the reason `review.status` exists: an
entry is `draft` until somebody reads it against the source.

## Packing

```bash
uv run python -m research.kb.pack research/kb/entries --verify
```

`--verify` parses the blob back and compares every id, title and material
byte against the source, so a layout bug surfaces here rather than on the
board. Refuses to pack anything that fails validation.

    header  12 B   4s "KBRD", u16 version, u16 count, u16 stride, u16 reserved
    index   N x 12  u32 off, u16 id_len, u16 title_len, u16 mat_len, u16 flags
    blob    ...    id, title, material, concatenated, UTF-8, no terminators

flags bit 0 marks an entry with at least one inferred field. The device reads
record `i`, bounds-checks it, and copies the material into the prompt; it never
parses a field or a delimiter. 24 entries come to 9.3 KiB, so a 2,600-entry set
is about 1 MiB - the resource set is not what fills the flash.

## Not built yet

- batch statistics from the spec's M7/M9: near-duplicate detection across
  entries and phrase-coverage reporting
