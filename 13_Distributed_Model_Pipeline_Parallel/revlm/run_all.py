"""The experiment matrix.

    python -m revlm.run_all --quick     a few minutes, writes assets/quick/
    python -m revlm.run_all             the submitted numbers, writes assets/

The matrix answers the assignment's three questions and two more that the assignment does
not ask but a reader will:

    A  baseline, activations stored          the fixed batch B*
    E  baseline + gradient checkpointing     the honest rival: recomputation without
                                             reversibility already saves most of it
    B  symplectic Euler                      assignment variant 1
    C  midpoint / leapfrog                   assignment variant 2
    G  implicit Euler (fixed point)          the obvious reading of "Euler", which is only
                                             conditionally reversible - trained anyway, so
                                             the cost of a wrong gradient is visible
    D  the winner of B/C at B_max            the assignment's "push the batch size"
    D2 the same run at the unscaled LR       so the batch-size confound is visible, not
                                             argued about
    F  RevNet coupling                       the "etc" variant, a short probe

Every run gets the same seed, the same sampler stream, the same schedule shape, the same
token budget, and no weight decay or dropout anywhere.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import math
import os
import sys

from . import data as D
from . import ladder as L
from .train import RunSpec, environment, save, train_one


def keep_awake():
    """Ask Windows not to sleep while a multi-hour matrix is running.

    Not a system setting - a process-local request that lapses the moment this process
    exits. Worth doing because an overnight sleep killed a run at step 1520 of 2034, and
    an eight-run matrix on a laptop is otherwise at the mercy of the lid.
    """
    if sys.platform != "win32":
        return False
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return True
    except Exception:
        return False


def pick_batches(out_dir, quick, log=print):
    """B* is the largest batch the *baseline* survives; B_max the largest a reversible one does."""
    path = os.path.join(out_dir, "ladder.json")
    if quick:
        return 8, 32, {"note": "quick mode uses fixed batches, no ladder"}
    if not os.path.exists(path):
        log("no ladder.json found - running the ladder first")
        L.main(["revlm.ladder"])
    lad = json.load(open(path))
    mb = lad["max_batch"]
    b_star = mb.get("store", 32)
    # The batch to "push to" is the one reached with a chunked cross entropy. Without it
    # the fp32 logits cap the batch long before the activations would: 112 against 320.
    ch = lad.get("chunked", {}).get("max_batch", {})
    b_max = max(ch.get("midpoint", 0), ch.get("euler", 0)) or         max(mb.get("midpoint", b_star), mb.get("euler", b_star))
    return b_star, b_max, lad


def build_matrix(b_star, b_max, tokens, quick):
    probe_tokens = max(1, tokens // 10)
    m = [
        RunSpec(name="A_baseline", mode="store", batch_size=b_star, tokens=tokens,
                note="activations stored, the ordinary pre-LN GPT", tags=["baseline"]),
        RunSpec(name="E_checkpoint", mode="checkpoint", batch_size=b_star, tokens=tokens,
                note="recomputation without reversibility - the honest rival", tags=["rival"]),
        RunSpec(name="B_euler", mode="euler", batch_size=b_star, tokens=tokens,
                note="symplectic Euler: velocity stream, exactly invertible",
                tags=["reversible", "variant"]),
        RunSpec(name="C_midpoint", mode="midpoint", batch_size=b_star, tokens=tokens,
                note="leapfrog: h_{l+1} = h_{l-1} + 2 G(h_l), exactly invertible",
                tags=["reversible", "variant"]),
        RunSpec(name="G_euler_implicit", mode="euler_implicit", batch_size=b_star,
                tokens=tokens, euler_iters=4,
                note="the obvious reading of Euler: invert by fixed point. Only "
                     "conditionally a contraction; trained to price the wrong gradient.",
                tags=["reversible", "variant", "negative"]),
        RunSpec(name="F_coupling", mode="coupling", batch_size=b_star, tokens=probe_tokens,
                note="RevNet additive coupling - a probe, not a full run", tags=["probe"]),
    ]
    return m


def add_max_batch_runs(matrix, results, b_star, b_max, tokens):
    """Pick the better of B and C on validation loss, then push its batch size."""
    scored = {r["spec"]["name"]: r["final_val_loss"] for r in results
              if r["spec"]["name"] in ("B_euler", "C_midpoint")}
    winner = min(scored, key=scored.get)
    mode = "euler" if winner == "B_euler" else "midpoint"
    scale = math.sqrt(b_max / b_star)
    return winner, [
        RunSpec(name="D_maxbatch", mode=mode, batch_size=b_max, tokens=tokens,
                lr=RunSpec.lr * scale, ce_chunks=4,
                note=f"{mode} pushed to the largest batch that fits, learning rate scaled "
                     f"by sqrt({b_max}/{b_star}) = {scale:.2f}", tags=["maxbatch"]),
        RunSpec(name="D2_maxbatch_same_lr", mode=mode, batch_size=b_max, tokens=tokens,
                ce_chunks=4,
                note="the same run at the unscaled learning rate, so the batch-size "
                     "confound can be read off rather than argued about",
                tags=["maxbatch", "control"]),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="a few minutes, into assets/quick/")
    ap.add_argument("--tokens", type=int, default=50_000_000)
    ap.add_argument("--only", type=str, default="", help="comma-separated run names")
    ap.add_argument("--resume", action="store_true",
                    help="keep runs already in results.json and only do the missing ones")
    args = ap.parse_args(argv)

    out_dir = os.path.join("assets", "quick") if args.quick else "assets"
    tokens = 2_000_000 if args.quick else args.tokens
    os.makedirs(out_dir, exist_ok=True)

    if keep_awake():
        print("asked Windows to stay awake for the duration")
    meta = D.prepare("data")
    b_star, b_max, lad = pick_batches("assets", args.quick)
    print(f"\nB* = {b_star} (largest the baseline survives), B_max = {b_max}")

    matrix = build_matrix(b_star, b_max, tokens, args.quick)
    if args.only:
        wanted = set(args.only.split(","))
        matrix = [s for s in matrix if s.name in wanted]

    results, stamp = [], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    done = set()
    prior = os.path.join(out_dir, "results.json")
    if args.resume and os.path.exists(prior):
        old = json.load(open(prior))
        results.extend(old["runs"])
        done = {r["spec"]["name"] for r in results}
        print(f"resuming: keeping {sorted(done)}")
    payload = {"run_stamp": stamp, "environment": environment(), "tokens_budget": tokens,
               "b_star": b_star, "b_max": b_max, "data": {k: v for k, v in meta.items()},
               "ladder": lad, "runs": results}

    for spec in matrix:
        if spec.name in done:
            continue
        results.append(train_one(spec, meta, out_dir))
        save(payload, out_dir)                      # stream: a crash keeps what finished

    if not args.only and len(results) >= 4:
        winner, extra = add_max_batch_runs(matrix, results, b_star, b_max, tokens)
        payload["winner"] = winner
        print(f"\nwinner on validation loss: {winner} -> pushing its batch to {b_max}")
        for spec in extra:
            results.append(train_one(spec, meta, out_dir))
            save(payload, out_dir)

    path = save(payload, out_dir)
    print(f"\nRUN STAMP {stamp}")
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main(sys.argv[1:])
