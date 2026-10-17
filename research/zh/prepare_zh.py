"""Build the Chinese corpus, train a 16,384 ByteLevel BPE, emit uint16 bins.

Same contract as research/tinystories/prepare.py (uint16 bins, one shared
tokenizer per vocabulary variant, <|endoftext|> as the document separator) so
the existing train.py / export.py paths work untouched. What differs:

  * the corpus is a mix of teacher-generated paragraphs and wiki paragraphs
  * the metric that matters is hanzi per token, not bytes per token: a tokenizer
    that needs 3 tokens per hanzi would put a 200-hanzi paragraph out of reach
    of a 512-token context.

Writes to data/zh/vocab-<V>/{tokenizer.json,train.bin,val.bin}.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "zh"
SYNTH = DATA / "raw" / "synth.jsonl"
WIKI = DATA / "raw" / "wiki_corpus.txt"
CORPUS = DATA / "raw" / "corpus.txt"
EOT = "<|endoftext|>"
DEFAULT_VOCAB = 16384
VAL_FRACTION = 0.005
# 24MB of Chinese is ~8M hanzi: far more than a merge table this size can use.
TOK_TRAIN_BYTES = 24 * 1024 * 1024
CJK = re.compile(r"[\u4e00-\u9fff]")


def variant_dir(vocab_size):
    return DATA / f"vocab-{vocab_size}"


def build_corpus(synth_chars, wiki_chars):
    """Join both sources into one document-separated text file."""
    if os.path.exists(CORPUS):
        print(f"already have {CORPUS}")
        return
    parts = []
    if synth_chars and os.path.exists(SYNTH):
        n = 0
        with open(SYNTH, encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"].strip()
                if not text:
                    continue
                parts.append(text)
                n += len(text)
                if n >= synth_chars:
                    break
        print(f"synth: {len(parts):,} paragraphs, {n / 1e6:.1f}M chars")
    if wiki_chars and os.path.exists(WIKI):
        n = 0
        k = 0
        with open(WIKI, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                parts.append(text)
                n += len(text)
                k += 1
                if n >= wiki_chars:
                    break
        print(f"wiki:  {k:,} paragraphs, {n / 1e6:.1f}M chars")
    if not parts:
        raise SystemExit(
            f"no corpus found. Generate one first:\n"
            f"  uv run python -m research.zh.synth --stage topics\n"
            f"  uv run python -m research.zh.synth --stage write\n"
            f"  uv run python -m research.zh.wiki_extract data/zh/raw/*.parquet")
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS, "w", encoding="utf-8") as f:
        f.write(f"\n{EOT}\n".join(parts))
        f.write(f"\n{EOT}\n")
    print(f"corpus: {len(parts):,} docs -> {CORPUS}")


def train_tokenizer(text, vocab_size):
    path = variant_dir(vocab_size) / "tokenizer.json"
    if os.path.exists(path):
        print(f"already have {path}")
        return Tokenizer.from_file(str(path))
    print(f"training BPE vocab={vocab_size} on {TOK_TRAIN_BYTES / 1e6:.0f}MB...")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator([text[:TOK_TRAIN_BYTES]], trainer=trainer)
    tok.save(str(path))
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    ap.add_argument("--synth-chars", type=int, default=120_000_000)
    ap.add_argument("--wiki-chars", type=int, default=40_000_000)
    args = ap.parse_args()
    if not 0 < args.vocab <= 65536:
        raise SystemExit("--vocab must be 1..65536; the bins are uint16")

    out = variant_dir(args.vocab)
    out.mkdir(parents=True, exist_ok=True)
    build_corpus(args.synth_chars, args.wiki_chars)

    with open(CORPUS, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tok = train_tokenizer(text, args.vocab)
    eot = tok.token_to_id(EOT)
    print(f"eot id = {eot}")

    print("encoding...")
    docs = text.split(EOT)
    ids = []
    han = 0
    for i in range(0, len(docs), 20000):
        batch = [d for d in docs[i: i + 20000] if d.strip()]
        for enc in tok.encode_batch(batch):
            ids.extend(enc.ids)
            ids.append(eot)
        print(f"  {i + len(batch)}/{len(docs)} docs, {len(ids) / 1e6:.1f}M tokens",
              flush=True)

    arr = np.array(ids, dtype=np.uint16)
    assert arr.max() < args.vocab, f"token id {arr.max()} >= vocab {args.vocab}"
    n_val = int(len(arr) * VAL_FRACTION)
    arr[:-n_val].tofile(out / "train.bin")
    arr[-n_val:].tofile(out / "val.bin")
    han = len(CJK.findall(text))
    print(f"train {len(arr) - n_val:,} tokens / val {n_val:,} tokens")
    print(f"compression: {len(text) / len(arr):.2f} bytes/token, "
          f"{han / len(arr):.2f} hanzi/token")
    if han / len(arr) < 1.2:
        print("WARNING: under 1.2 hanzi/token. A 200-hanzi paragraph will not "
              "fit a 512-token context; retrain with more tokenizer data.")


if __name__ == "__main__":
    sys.exit(main())
