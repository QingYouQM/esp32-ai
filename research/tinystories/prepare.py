"""Prepare a corpus slice, train a small BPE, emit uint16 token bins.

The bins are uint16, so the vocabulary cannot exceed 65,536: train.py and
quantize_eval.py memmap them as uint16 and a wider dtype would be read as
garbage rather than fail.

The vocab is deliberately fixed across every ablation arm: cross-entropy is only
comparable between models that share a tokenizer.

Defaults reproduce the published TinyStories run exactly. Any other corpus comes
in through --tag/--corpus/--source-url/--doc-sep and never touches the download
path, so the English cohort stays reproducible byte for byte.

ByteLevel pretokenisation already covers UTF-8: it splits on raw bytes, so a
Chinese corpus needs no tokenizer changes here. The vocab is the thing that
matters - a Chinese short-story domain fits a much smaller vocabulary than the
English one, because it draws on far fewer characters.
"""

import argparse
import os
import sys

import numpy as np
import requests
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"

# Set from the command line before anything reads them.
TAG = "tinystories"
SOURCE_URL = TINYSTORIES_URL
CORPUS = None
DOC_SEP = "<|endoftext|>"
SLICE_BYTES = 300 * 1024 * 1024   # ~300MB of English stories is ~75M tokens
# Structural markers that must survive as one token each. A paired corpus puts
# <|material|> and <|story|> inside every document, and ByteLevel would shred
# them into six or eight merges that no longer line up with anything.
MARKERS = []
VAL_FRACTION = 0.005
VOCAB_SIZE = 32768   # what ships; every other size has to be asked for


def data_dir():
    return ROOT / "data" / TAG


def raw_path():
    return data_dir() / "raw" / "corpus.txt"


def variant_dir(vocab_size):
    # Each vocabulary variant keeps its tokenizer beside the bins it produced.
    return data_dir() / f"vocab-{vocab_size}"


def download():
    """Fetch a byte-exact prefix of the remote corpus.

    Exactly the requested size, not approximately: a truncated slice trains a
    different tokenizer and different bins, silently.
    """
    RAW = raw_path()
    if os.path.exists(RAW):
        have = os.path.getsize(RAW)
        if have == SLICE_BYTES:
            print(f"already have {RAW}")
            return
        print(f"{RAW} is {have:,} bytes, expected exactly {SLICE_BYTES:,}; "
              f"re-downloading")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading first {SLICE_BYTES / 1e6:.0f}MB from {SOURCE_URL}...")
    got = 0
    with requests.get(SOURCE_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(RAW, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                # Trim the last chunk: writing it whole overshoots the slice, and
                # a slice of a different length is a different corpus.
                if got + len(chunk) > SLICE_BYTES:
                    chunk = chunk[:SLICE_BYTES - got]
                f.write(chunk)
                got += len(chunk)
                if got >= SLICE_BYTES:
                    break
                if got % (25 << 20) < (1 << 20):
                    print(f"  {got / 1e6:.0f}MB", flush=True)
    final = os.path.getsize(RAW)
    if final != SLICE_BYTES:
        raise SystemExit(f"downloaded {final:,} bytes, expected exactly "
                         f"{SLICE_BYTES:,}; the source may be shorter than the "
                         f"requested slice")
    print(f"done, {got / 1e6:.0f}MB")


def load_text():
    """Return the corpus as one string, from a local file or the downloaded slice.

    A local corpus is used whole: it was not sliced, so there is no length to
    match and trimming it would silently change the corpus.
    """
    if CORPUS is not None:
        path = Path(CORPUS)
        if not path.exists():
            raise SystemExit(f"corpus not found: {path}")
        print(f"reading local corpus {path} "
              f"({path.stat().st_size / 1e6:.1f}MB)")
        return path.read_text(encoding="utf-8", errors="ignore")
    download()
    RAW = raw_path()
    with open(RAW, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def train_tokenizer(text):
    path = variant_dir(VOCAB_SIZE) / "tokenizer.json"
    if os.path.exists(path):
        print(f"already have {path}")
        return Tokenizer.from_file(str(path))
    print(f"training BPE vocab={VOCAB_SIZE}...")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[DOC_SEP] + MARKERS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    # 40MB is plenty to fit a merge table of the sizes used here; feeding the
    # full slice just burns time for the same merges.
    tok.train_from_iterator([text[: 40 * 1024 * 1024]], trainer=trainer)
    tok.save(str(path))
    return tok


def flatten(docs, eot):
    """Documents in order, each closed by exactly one endoftext.

    This has to be interleaved, not appended in a block at the end. Appending
    reads the same in code and is wrong in every way that shows up later: with
    no boundary between documents the model is trained to continue straight from
    one story into the next document's header, so it never learns to stop, and
    the only endoftext tokens it ever sees are one huge run at the very tail of
    the array. The symptom is a sampler that answers a story with the next
    card's "人物：..." instead of ending.
    """
    out = []
    for enc in docs:
        out.extend(enc)
        out.append(eot)
    return np.array(out, dtype=np.uint16)


def main():
    global VOCAB_SIZE, TAG, SOURCE_URL, CORPUS, DOC_SEP, SLICE_BYTES, MARKERS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    ap.add_argument("--seq-len", type=int, default=512,
                    help="sequence length the bins will be trained at; sizes "
                         "the floor under the validation split")
    ap.add_argument("--tag", default=TAG,
                    help="dataset directory under data/; keeps corpora apart")
    ap.add_argument("--corpus", default=None,
                    help="local UTF-8 corpus file; used whole, skips download")
    ap.add_argument("--source-url", default=SOURCE_URL,
                    help="remote corpus to slice when --corpus is absent")
    ap.add_argument("--doc-sep", default=DOC_SEP,
                    help="document separator; becomes a special token")
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION,
                    help="share of tokens held out, taken on document "
                         "boundaries. 0.005 is right for the 75M-token English "
                         "slice; a 1M-token corpus needs 0.02 to give the "
                         "evaluation more than a dozen windows")
    ap.add_argument("--markers", default="",
                    help="comma-separated structural markers to register as "
                         "single tokens, e.g. '<|material|>,<|story|>'; the "
                         "separator is registered automatically")
    ap.add_argument("--slice-bytes", type=int, default=SLICE_BYTES,
                    help="byte-exact prefix to fetch when --corpus is absent")
    args = ap.parse_args()

    if not 0 < args.vocab <= 65536:
        raise SystemExit(f"--vocab must be 1..65536; the bins are uint16 and "
                         f"every reader memmaps them as uint16")
    VOCAB_SIZE = args.vocab
    TAG = args.tag
    SOURCE_URL = args.source_url
    CORPUS = args.corpus
    DOC_SEP = args.doc_sep
    SLICE_BYTES = args.slice_bytes
    MARKERS = [m.strip() for m in args.markers.split(",") if m.strip()]
    if DOC_SEP in MARKERS:
        raise SystemExit("--markers should not repeat --doc-sep")

    if not DOC_SEP:
        raise SystemExit("--doc-sep must be non-empty: it is both the document "
                         "boundary and the endoftext token")
    if DOC_SEP in SOURCE_URL and args.corpus is None:
        print(f"note: --doc-sep {DOC_SEP!r} looks like part of a URL; pass "
              f"--corpus if the corpus is already local", flush=True)

    out = variant_dir(VOCAB_SIZE)
    out.mkdir(parents=True, exist_ok=True)

    text = load_text()

    # Drop the trailing partial document left by the byte-slice, or by however
    # the local corpus happens to end.
    cut = text.rfind(DOC_SEP)
    if cut < 0:
        raise SystemExit(f"corpus contains no {DOC_SEP!r}; every document "
                         f"boundary has to be marked for the split to work")
    text = text[: cut + len(DOC_SEP)]

    for marker in MARKERS:
        if marker not in text:
            raise SystemExit(f"corpus never contains the marker {marker!r}")

    tok = train_tokenizer(text)
    eot = tok.token_to_id(DOC_SEP)
    print(f"doc-sep id = {eot}")
    for marker in MARKERS:
        mid = tok.token_to_id(marker)
        if mid is None:
            raise SystemExit(f"{marker!r} did not become a special token")
        # One token, not a merge chain: that is the whole point of registering
        # it, and a silent split here would only show up as a model that learns
        # to spell '<|mat'.
        nl = len(tok.encode(marker, add_special_tokens=False).ids)
        if nl != 1:
            raise SystemExit(f"{marker!r} encodes to {nl} tokens, not 1")
        print(f"marker {marker} id = {mid}")

    print("encoding...")
    docs = text.split(DOC_SEP)
    # Encode per document and keep the boundaries. The split below needs them,
    # and flattening here discards the only record of where one document ends
    # and the next begins.
    enc_docs = []
    for i in range(0, len(docs), 20000):
        batch = [d for d in docs[i : i + 20000] if d.strip()]
        for enc in tok.encode_batch(batch):
            enc_docs.append(enc.ids)
        print(f"  {i + len(batch)}/{len(docs)} docs, "
              f"{sum(len(e) + 1 for e in enc_docs) / 1e6:.1f}M tokens", flush=True)

    # Split on document boundaries, not on a token offset.
    #
    # A contiguous slice cuts through documents. The model reads the first half
    # of a sentence in train and is scored on its second half in val, which
    # leaks rather than validates, and the leak is invisible because val loss
    # looks healthy. It also has no floor: at 0.5% a 12k-token corpus leaves 60
    # tokens for validation, too few for one seq_len+1 window, and Batcher dies
    # inside rng.integers on high <= 0. The 300MB English corpus hides both. A
    # small one hits them on the first run.
    min_window = args.seq_len + 1
    total = sum(len(e) + 1 for e in enc_docs)
    need = max(int(total * args.val_fraction), min_window)

    n_val_docs, val_len = 0, 0
    while n_val_docs < len(enc_docs) - 1:      # always leave >= 1 for training
        val_len += len(enc_docs[-1 - n_val_docs]) + 1
        n_val_docs += 1
        if val_len >= need:
            break

    train_docs = enc_docs[: len(enc_docs) - n_val_docs]
    val_docs = enc_docs[len(enc_docs) - n_val_docs :]
    train_len = sum(len(e) + 1 for e in train_docs)

    if val_len < min_window or train_len < min_window:
        raise SystemExit(
            f"cannot split {len(enc_docs)} documents into a train and a "
            f"validation set that each hold one --seq-len {args.seq_len} "
            f"window (train {train_len:,} tokens, val {val_len:,}). "
            f"Lower --seq-len or add documents."
        )

    arr = flatten(train_docs, eot)
    assert arr.max() < VOCAB_SIZE
    arr.tofile(out / "train.bin")
    val_arr = flatten(val_docs, eot)
    assert val_arr.max() < VOCAB_SIZE
    val_arr.tofile(out / "val.bin")
    print(f"train {len(arr):,} tokens / val {len(val_arr):,} tokens "
          f"({len(train_docs)}/{len(enc_docs)} documents, "
          f"floor {min_window} tokens)")
    # Two ratios, because len() on a str counts characters and for a CJK corpus
    # that is a third of its byte length. "chars/token" is the one that predicts
    # how many tokens a 150-200 character story costs; "bytes/token" is what the
    # ASCII original meant and is comparable with RESULTS.md. Both are over the
    # whole corpus, not just the training split.
    all_tokens = len(arr) + len(val_arr)
    print(f"compression: {len(text) / all_tokens:.2f} chars/token "
          f"({len(text.encode('utf-8')) / all_tokens:.2f} bytes/token)")


if __name__ == "__main__":
    sys.exit(main())
