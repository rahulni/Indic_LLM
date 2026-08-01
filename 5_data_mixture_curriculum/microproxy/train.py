# -*- coding: utf-8 -*-
"""
train.py - the micro-proxy.

This is NOT a scaled-down version of the full screen in section 13. It tests the
two claims in the plan that are training-DYNAMICS claims rather than capability
claims, because those are the only ones an ~11M-parameter model can speak to:

  A0 vs A6  - does the 15% band crossfade actually suppress loss spikes and
              gradient-norm excursions at a mixture transition? The plan asserts
              this from second-hand V4 experience and has never tested it.

  A0 vs A2  - does keeping a minority lane always-on at a floor preserve it
              better than introducing it later, at MATCHED total token share?
              This is the retention argument the protected floor rests on.

What this cannot test, and does not claim: HumanEval, MMLU, GSM8K, BFCL, or
anything about the 75 tokens/param regime. An 11M model is at chance on all of
them. Reported metrics are per-lane held-out bits-per-byte and training
stability, nothing else.

    python train.py --arm A0 --steps 3000
"""

import argparse
import io
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(HERE, "runs")

VOCAB = 256

# Two-lane schedule with a deliberately large swing, so a transition is
# actually a transition. Shares are percentages per phase and must sum to 100.
PHASES = [("A", 0.40), ("B", 0.30), ("C", 0.20), ("D", 0.10)]

ARMS = {
    # crossfade on, indic held at a 10% floor from step 0
    "A0": dict(name="Baseline (crossfade + floor)", crossfade=0.15,
               indic=[10.0, 30.0, 60.0, 80.0]),
    # no floor: indic absent early, introduced late. Total indic share is
    # matched to A0 (33.0% vs 33.3%) so this isolates ORDERING, not budget.
    "A2": dict(name="No floor (indic introduced late)", crossfade=0.15,
               indic=[0.0, 37.0, 68.0, 86.0]),
    # identical to A0 except transitions step instead of ramping
    "A6": dict(name="Sharp transitions (no crossfade)", crossfade=0.0,
               indic=[10.0, 30.0, 60.0, 80.0]),
}


# ---------------------------------------------------------------------------
# model - a small decoder-only transformer
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab=VOCAB, d=384, h=6, layers=6, block=512):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.register_buffer("mask", torch.triu(
            torch.full((block, block), float("-inf")), diagonal=1), persistent=False)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        m = self.mask[:t, :t]
        for blk in self.blocks:
            x = blk(x, m)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class Lanes:
    def __init__(self, block, device):
        self.block, self.device = block, device
        self.train, self.held = {}, {}
        for lane in ("indic", "reasoning"):
            self.train[lane] = np.fromfile(
                os.path.join(DATA, f"{lane}_train.bin"), dtype=np.uint8)
            self.held[lane] = np.fromfile(
                os.path.join(DATA, f"{lane}_heldout.bin"), dtype=np.uint8)

    def batch(self, lane, bs, rng, split="train"):
        src = self.train[lane] if split == "train" else self.held[lane]
        ix = rng.integers(0, len(src) - self.block - 1, size=bs)
        x = np.stack([src[i:i + self.block] for i in ix]).astype(np.int64)
        y = np.stack([src[i + 1:i + 1 + self.block] for i in ix]).astype(np.int64)
        return (torch.from_numpy(x).to(self.device, non_blocking=True),
                torch.from_numpy(y).to(self.device, non_blocking=True))


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------

def phase_at(frac):
    acc = 0.0
    for i, (pid, wgt) in enumerate(PHASES):
        nxt = acc + wgt
        if frac < nxt or i == len(PHASES) - 1:
            return i, acc, nxt
        acc = nxt
    return len(PHASES) - 1, acc, 1.0


def indic_share(arm, frac):
    """Share of the batch drawn from the indic lane, with the crossfade applied.
    This is the micro-proxy's version of plan mixture_at()."""
    cfg = ARMS[arm]
    i, lo, hi = phase_at(frac)
    cur = cfg["indic"][i]
    ov = cfg["crossfade"]
    if ov > 0 and i + 1 < len(PHASES):
        span = hi - lo
        start = hi - span * ov
        if frac > start:
            t = min(1.0, (frac - start) / (span * ov))
            cur = (1 - t) * cur + t * cfg["indic"][i + 1]
    return cur / 100.0


def lr_at(frac, peak, warmup=0.02, anneal_start=0.90, floor_frac=0.10):
    if frac < warmup:
        return peak * frac / warmup
    if frac < anneal_start:
        t = (frac - warmup) / (anneal_start - warmup)
        return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * t)))
    return peak * floor_frac * (1 - (frac - anneal_start) / (1 - anneal_start))


# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, lanes, rng, bs, iters=20):
    model.eval()
    out = {}
    for lane in ("indic", "reasoning"):
        tot = 0.0
        for _ in range(iters):
            x, y = lanes.batch(lane, bs, rng, split="heldout")
            _, loss = model(x, y)
            tot += loss.item()
        # bits per byte - the natural unit for a byte-level model, and
        # comparable across lanes in a way raw nats are not.
        out[lane] = (tot / iters) / math.log(2)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-every", type=int, default=150)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RUNS, exist_ok=True)

    lanes = Lanes(args.block, device)
    model = GPT(block=args.block).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    print(f"arm {args.arm}: {ARMS[args.arm]['name']}")
    print(f"  {n_params/1e6:.2f}M params, {args.steps} steps x {args.batch} x "
          f"{args.block} = {args.steps*args.batch*args.block/1e6:.1f}M byte-tokens, "
          f"device={device}")

    log, evals = [], []
    t0 = time.time()
    for step in range(args.steps):
        frac = step / args.steps
        share = indic_share(args.arm, frac)
        lr = lr_at(frac, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        n_ind = int(round(share * args.batch))
        parts = []
        if n_ind:
            parts.append(lanes.batch("indic", n_ind, rng))
        if args.batch - n_ind:
            parts.append(lanes.batch("reasoning", args.batch - n_ind, rng))
        x = torch.cat([p[0] for p in parts])
        y = torch.cat([p[1] for p in parts])

        with torch.amp.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        scaler.step(opt)
        scaler.update()

        log.append(dict(step=step, frac=frac, loss=loss.item(), gnorm=gnorm,
                        indic_share=share, lr=lr,
                        phase=PHASES[phase_at(frac)[0]][0]))

        if step % args.eval_every == 0 or step == args.steps - 1:
            bpb = evaluate(model, lanes, rng, bs=8)
            evals.append(dict(step=step, frac=frac, **bpb))
            print(f"  step {step:5d} f={frac:.2f} indic={share*100:5.1f}% "
                  f"loss={loss.item():.4f} gn={gnorm:6.3f} "
                  f"| bpb indic={bpb['indic']:.4f} reasoning={bpb['reasoning']:.4f}")

    elapsed = time.time() - t0
    out = dict(arm=args.arm, name=ARMS[args.arm]["name"], params=n_params,
               steps=args.steps, batch=args.batch, block=args.block,
               tokens=args.steps * args.batch * args.block, seed=args.seed,
               device=device, elapsed_s=elapsed,
               crossfade=ARMS[args.arm]["crossfade"],
               indic_schedule=ARMS[args.arm]["indic"],
               log=log, evals=evals)
    path = os.path.join(RUNS, f"{args.arm}_seed{args.seed}.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"  done in {elapsed/60:.1f} min -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
