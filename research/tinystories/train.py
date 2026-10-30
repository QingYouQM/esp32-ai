"""Train one ablation arm and report val loss at matched core-parameter budget."""

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, TinyLM, make_model

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# A vocabulary variant keeps its tokenizer and token bins together.
DATA = ROOT / "data" / "tinystories"
RUNS = str(ROOT / "runs")


def variant_dir(vocab_size):
    return DATA / f"vocab-{vocab_size}"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def story_mask(ids, material_id, story_id, eod_id):
    """1 where a token belongs to the story half of its document.

    With a paired corpus, the material is 40% of the text and predicting it is
    copying the prompt. Training on it spends capacity on transcription instead
    of on "read this, then write", which is the only behaviour we want.
    """
    kind = np.full(len(ids), -1, dtype=np.int8)
    kind[ids == story_id] = 1
    kind[ids == material_id] = 0
    kind[ids == eod_id] = 0
    last = np.where(kind >= 0, np.arange(len(ids)), 0)
    np.maximum.accumulate(last, out=last)
    filled = kind[last]
    filled[kind >= 0] = kind[kind >= 0]
    return (filled == 1).astype(np.uint8)


def masked_loss(logits, targets, weights):
    flat = F.cross_entropy(logits.view(-1, logits.size(-1)),
                           targets.reshape(-1), reduction="none")
    w = weights.reshape(-1).float()
    return (flat * w).sum() / w.sum().clamp(min=1.0)


class Batcher:
    def __init__(self, split, batch_size, seq_len, device, dataset, seed=0,
                 mask=None):
        # The bins are uint16. Reading a wider vocabulary through that dtype
        # yields plausible token ids rather than an error, so check before
        # opening.
        self.data = np.memmap(dataset / f"{split}.bin", dtype=np.uint16, mode="r")
        self.bs, self.sl, self.device = batch_size, seq_len, device
        # Batch order is part of the run. Without the seed here, torch.manual_seed
        # fixes initialisation only and two runs at the same --seed still see
        # different data order. Validation keeps a fixed stream so every arm is
        # scored on identical batches.
        self.rng = np.random.default_rng(1234 if split == "val" else seed)
        self.mask = mask

    def __call__(self):
        ix = self.rng.integers(0, len(self.data) - self.sl - 1, self.bs)
        x = np.stack([self.data[i : i + self.sl] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.sl] for i in ix]).astype(np.int64)
        xb = torch.from_numpy(x).to(self.device)
        yb = torch.from_numpy(y).to(self.device)
        if self.mask is None:
            return xb, yb
        # Position i is scored for predicting i+1, so the weight that belongs to
        # a target is the mask at i+1.
        w = np.stack([self.mask[i + 1 : i + 1 + self.sl] for i in ix])
        return xb, yb, torch.from_numpy(w.astype(np.float32)).to(self.device)


@torch.no_grad()
def evaluate(model, batcher, iters):
    model.eval()
    batcher.rng = np.random.default_rng(1234)  # same val batches for every arm
    losses = []
    for _ in range(iters):
        batch = batcher()
        if len(batch) == 3:
            x, y, w = batch
            logits, _ = model(x)
            losses.append(masked_loss(logits, y, w).item())
        else:
            losses.append(model(*batch)[1].item())
    model.train()
    return sum(losses) / len(losses)


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=["baseline", "ple", "ple_notable", "fatembed", "bigcore"],
    )
    ap.add_argument("--target-core", type=int, default=1_500_000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--ple-dim", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--fixed-ffn", type=int, default=None,
                    help="pin ffn_hidden and skip the core solver (table-scaling sweep)")
    ap.add_argument("--mask-material", action="store_true",
                    help="compute loss on the story half only; a paired corpus "
                         "otherwise spends a third of its gradient predicting "
                         "the prompt back")
    ap.add_argument("--keep-best", action="store_true",
                    help="also save the checkpoint at the best validation loss. "
                         "A small corpus overfits long before the step budget "
                         "runs out, and the final weights are then the worst "
                         "ones in the run")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout on each sublayer output before its residual add; "
                         "0.0 is the historical no-op")
    # Published experiments always pass --vocab; the default is for ad-hoc runs.
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="directory holding train.bin/val.bin/tokenizer.json. Defaults to "
             "data/tinystories/vocab-<vocab>, which is where the published arms "
             "live; a corpus prepared with --tag lands elsewhere and needs this.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # Before anything expensive: the tokenizer that produced these bins. Its
    # hash is what lets the exporter and sampler refuse a mismatched tokenizer
    # later, and a checkpoint written without it cannot be tied to one. Failing
    # here costs nothing; failing after a 30-minute run costs the run.
    if not 0 < args.vocab <= 65536:
        raise SystemExit(f"--vocab must be 1..65536; the token bins are uint16 "
                         f"and are memmapped as uint16")

    dataset = args.data_dir or variant_dir(args.vocab)
    tok_path = dataset / "tokenizer.json"
    if not os.path.exists(tok_path):
        raise SystemExit(
            f"{tok_path} missing. The bins this run trains on came from it, and "
            f"without its hash the checkpoint cannot be tied to a tokenizer. "
            f"Run: python -m research.tinystories.prepare --vocab {args.vocab}")
    tok_sha = hashlib.sha256(open(tok_path, "rb").read()).hexdigest()

    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(RUNS, exist_ok=True)

    base = Config(seq_len=args.seq_len, ple_dim=args.ple_dim, vocab_size=args.vocab,
                  d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                  dropout=args.dropout)
    model = make_model(args.arm, args.target_core, base, fixed_ffn=args.fixed_ffn).to(device)
    budget = model.param_budget()
    cfg = model.cfg

    # No weight decay on 1-D params (norms) or on lookup tables.
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "table" in n or "tok_emb" in n else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    mask_train = mask_val = None
    if args.mask_material:
        tok = Tokenizer.from_file(str(tok_path))
        mid = tok.token_to_id("<|material|>")
        sid = tok.token_to_id("<|story|>")
        eid = tok.token_to_id("<|endoftext|>")
        if None in (mid, sid, eid):
            raise SystemExit(
                "--mask-material needs a paired tokenizer: it has no "
                "<|material|>, <|story|> or <|endoftext|> token")
        mask_train = story_mask(
            np.fromfile(dataset / "train.bin", dtype=np.uint16), mid, sid, eid)
        mask_val = story_mask(
            np.fromfile(dataset / "val.bin", dtype=np.uint16), mid, sid, eid)
        share = mask_train.mean()
        print(f"loss mask: {share:.1%} of training tokens are story, "
              f"{1 - share:.1%} are material and no longer trained on",
              flush=True)
        if share < 0.2:
            raise SystemExit("mask keeps almost nothing; check the tokenizer")

    train_b = Batcher("train", args.batch_size, args.seq_len, device, dataset,
                      seed=args.seed)
    val_b = Batcher("val", args.batch_size, args.seq_len, device, dataset,
                    mask=mask_val)

    name = f"{args.arm}{'-' + args.tag if args.tag else ''}-s{args.seed}"
    history, best = [], float("inf")
    best_state, best_step = None, 0
    t0 = time.time()

    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = train_b()
        if len(batch) == 3:
            x, y, w = batch
            logits, _ = model(x)
            loss = masked_loss(logits, y, w)
        else:
            x, y = batch
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_b, args.eval_iters)
            if vl < best:
                best, best_step = vl, step
                if args.keep_best:
                    # Off the GPU and cloned: the live tensors keep changing.
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
            tok = (step + 1) * args.batch_size * args.seq_len
            history.append({"step": step, "tokens": tok, "train": loss.item(), "val": vl})
            print(
                f"{name} step {step:5d} | tok {tok / 1e6:6.1f}M | train {loss.item():.4f} "
                f"| val {vl:.4f} | ppl {math.exp(vl):7.2f} | {time.time() - t0:5.0f}s",
                flush=True,
            )

    result = {
        "arm": args.arm,
        "seed": args.seed,
        "tag": args.tag,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "training": {
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "warmup": args.warmup,
            "eval_every": args.eval_every,
            "eval_iters": args.eval_iters,
            "target_core": args.target_core,
            "fixed_ffn": args.fixed_ffn,
            "mask_material": args.mask_material,
            "seed": args.seed,
        },
        "tokenizer_sha256": tok_sha,
        "params": budget,
        "final_val": history[-1]["val"],
        "best_val": best,
        "final_ppl": math.exp(history[-1]["val"]),
        "tokens_seen": args.steps * args.batch_size * args.seq_len,
        "steps": args.steps,
        "wall_seconds": time.time() - t0,
        "history": history,
    }
    with open(os.path.join(RUNS, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    # Identity and schedule live only in the filename and the sidecar JSON
    # otherwise, so a checkpoint copied over another name, or trained on a
    # different schedule, would pass every content check.
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict(),
                "tokenizer_sha256": tok_sha,
                "seed": args.seed, "tag": args.tag, "name": name,
                "training": result["training"]},
               os.path.join(RUNS, f"{name}.pt"))
    if args.keep_best and best_state is not None:
        torch.save({"cfg": cfg.__dict__, "state": best_state,
                    "tokenizer_sha256": tok_sha,
                    "seed": args.seed, "tag": args.tag, "name": name + "-best",
                    "training": result["training"],
                    "best_val": best, "best_step": best_step},
                   os.path.join(RUNS, f"{name}-best.pt"))
        print(f"{name} best val {best:.4f} at step {best_step} "
              f"-> runs/{name}-best.pt")

    print(f"{name} DONE core={budget['core']:,} table={budget['table']:,} "
          f"val={result['final_val']:.4f} ppl={result['final_ppl']:.2f}")


if __name__ == "__main__":
    main()
