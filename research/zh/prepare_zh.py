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
from collections import Counter

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from zhconv import convert as zh_convert

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
LATIN_RUN = re.compile(r"[A-Za-z]{12,}")

# The teacher occasionally fails to stay in character and leaks the prompt
# scaffolding ("Human:", "请注意使用换行", "我会尽力为您提供帮助", "【译文】").
# Trained on as-is, the model later prints "请问答题目" mid-story; every one of
# those artifacts in the first checkpoint traces back to lines like these.
LEAK = re.compile(
    r"Human:|Assistant:|请注意|我会尽力|【译文】|【?答案[】？:]|以下是|作为一个|"
    r"抱歉[，,]|无法提供|elinat|El\s?rey")
# Share of one paragraph allowed to be 6-grams seen earlier in the same
# paragraph. The teacher loops; trim_to_han only cuts exact sentence repeats,
# so phrase-level loops ("埃及、埃及、埃及") need their own gate.
REPEAT_LIMIT = 0.05


def ngram_repeat_ratio(text, n=6):
    grams = [text[i:i + n] for i in range(0, max(len(text) - n + 1, 1))]
    if len(grams) < 8:
        return 0.0
    c = Counter(grams)
    repeated = sum(v - 1 for v in c.values() if v > 1)
    return repeated / len(grams)


def clean_doc(text, stats):
    """Traditional->simplified, then drop the paragraphs worth dropping.

    zh.wikipedia is a traditional/simplified mix (6.3% of characters, two
    thirds of paragraphs touched) and the teacher leaks scaffolding now and
    then; both taught the first checkpoint to switch scripts mid-paragraph and
    to print its own prompt.
    """
    text = zh_convert(text, "zh-cn")
    if LEAK.search(text):
        stats["leak"] += 1
        return None
    if LATIN_RUN.search(text):
        stats["latin"] += 1
        return None
    if ngram_repeat_ratio(text) > REPEAT_LIMIT:
        stats["repeat"] += 1
        return None
    return text


def variant_dir(vocab_size):
    return DATA / f"vocab-{vocab_size}"


def build_corpus(synth_chars, wiki_chars):
    """Join both sources into one cleaned, document-separated text file."""
    if os.path.exists(CORPUS):
        print(f"already have {CORPUS}")
        return
    stats = Counter()
    parts = []
    if synth_chars and os.path.exists(SYNTH):
        n = 0
        kept = 0
        with open(SYNTH, encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"].strip()
                if not text:
                    continue
                text = clean_doc(text, stats)
                if not text:
                    continue
                parts.append(text)
                kept += 1
                n += len(text)
                if n >= synth_chars:
                    break
        print(f"synth: {kept:,} paragraphs, {n / 1e6:.1f}M chars")
    if wiki_chars and os.path.exists(WIKI):
        n = 0
        k = 0
        with open(WIKI, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                text = clean_doc(text, stats)
                if not text:
                    continue
                parts.append(text)
                k += 1
                n += len(text)
                if n >= wiki_chars:
                    break
        print(f"wiki:  {k:,} paragraphs, {n / 1e6:.1f}M chars")
    if not parts:
        raise SystemExit(
            f"no corpus found. Generate one first:\n"
            f"  uv run python -m research.zh.synth --stage topics\n"
            f"  uv run python -m research.zh.synth --stage write\n"
            f"  uv run python -m research.zh.wiki_extract data/zh/raw/*.parquet")
    total = stats["leak"] + stats["latin"] + stats["repeat"]
    print(f"dropped: {total:,} docs "
          f"(leak {stats['leak']:,}, latin-run {stats['latin']:,}, "
          f"repeat {stats['repeat']:,})")
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
