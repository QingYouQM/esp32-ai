"""Turn Chinese Wikipedia parquet shards into clean ~180-hanzi paragraphs.

The synthetic corpus is what teaches style and coherence; public text is here
for breadth of vocabulary and subject matter. Wikipedia is markup-heavy, so the
cleaning below is not optional cosmetics: templates and ref tags left in place
teach the model to emit "[[链接]]" and "{{cite web}}" mid-sentence.
"""

import argparse
import re
import sys

import pyarrow.parquet as pq

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "zh" / "raw" / "wiki_corpus.txt"

CJK = re.compile(r"[\u4e00-\u9fff]")
HEADING = re.compile(r"^=+.*?=+\s*$", re.M)
TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.S)
HTML = re.compile(r"<[^>]{1,80}>")
LINK = re.compile(r"\[\[([^\]|]*\|)?([^\]]*)\]\]")
BOLD = re.compile(r"'{2,}")
FILE_LINE = re.compile(r"^\s*(File|Image|文件|图像)\s*:", re.I)
TABLE_LINE = re.compile(r"^\s*[\|\!\{]")
SENT_END = re.compile(r"[。！？!?]")


def clean_article(text):
    t = REF.sub("", text)
    t = TEMPLATE.sub("", t)
    t = HTML.sub("", t)
    t = LINK.sub(r"\2", t)
    t = BOLD.sub("", t)
    lines = []
    for line in t.splitlines():
        s = line.strip()
        if not s or HEADING.match(s) or FILE_LINE.match(s) or TABLE_LINE.match(s):
            continue
        if s.startswith(("*", "#", ";", ":")):
            continue
        lines.append(s)
    return "\n".join(lines)


def to_paragraphs(text, min_han, max_han):
    """Group sentences until the paragraph is long enough, then cut."""
    out = []
    buf = ""
    for sent in SENT_END.split(text):
        s = sent.strip()
        if not s:
            continue
        s += "。"
        buf += s
        if len(CJK.findall(buf)) >= max_han:
            # Emit whole sentences until the target, keep the remainder.
            emit, rest = split_at_han(buf, max_han)
            if len(CJK.findall(emit)) >= min_han and not SENT_END.search(emit[:-1][-1:] or ""):
                pass
            if len(CJK.findall(emit)) >= min_han:
                out.append(emit)
            buf = rest
    if len(CJK.findall(buf)) >= min_han:
        out.append(buf)
    return out


def split_at_han(buf, max_han):
    """Cut at the last sentence boundary that still fits in max_han hanzi."""
    best = 0
    count = 0
    for i, ch in enumerate(buf):
        if CJK.match(ch):
            count += 1
        if ch in "。！？!?" and count <= max_han:
            best = i + 1
        if count > max_han:
            break
    if best == 0:
        return buf, ""
    return buf[:best], buf[best:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="+")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-han", type=int, default=140)
    ap.add_argument("--max-han", type=int, default=220)
    ap.add_argument("--max-chars", type=int, default=60_000_000,
                    help="stop after this many characters of output")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    n_par = 0
    with open(out, "w", encoding="utf-8") as f:
        for shard in args.parquet:
            print(f"reading {shard}", flush=True)
            table = pq.read_table(shard, columns=["text"])
            for text in table.column("text").to_pylist():
                if not text:
                    continue
                body = clean_article(text)
                if len(CJK.findall(body)) < args.max_han:
                    continue
                for para in to_paragraphs(body, args.min_han, args.max_han):
                    para = re.sub(r"\s+", "", para)
                    if len(CJK.findall(para)) < args.min_han:
                        continue
                    if len(CJK.findall(para)) / max(1, len(para)) < 0.7:
                        continue  # mostly Latin/table junk
                    f.write(para + "\n")
                    n_par += 1
                    total += len(para)
                    if n_par % 20000 == 0:
                        print(f"  {n_par:,} paragraphs, {total / 1e6:.1f}M chars",
                              flush=True)
                    if total >= args.max_chars:
                        break
                if total >= args.max_chars:
                    break
            if total >= args.max_chars:
                break
    print(f"done: {n_par:,} paragraphs, {total / 1e6:.1f}M chars -> {out}")


if __name__ == "__main__":
    sys.exit(main())
