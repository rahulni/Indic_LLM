"""CLI trainer for Track B's two word-embedding arms.

    python model/train.py --arm holographic --profile fast
    python model/train.py --arm kronecker --profile default

A word-level decoder-only LM with character-composed *input* word
embeddings and a standard learned softmax *output* head over the closed
vocabulary (compositional-unbind generation is out of scope -- a documented
scoping decision, see ARCHITECTURE.md). The two arms differ only in how a
word's characters are composed into its input embedding; the trunk, the
sequence-position embedding, and the output head are identical and shared.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.device import count_params, device_report, get_device  # noqa: E402
from common.seeding import seed_everything  # noqa: E402
from common.transformer import TransformerTrunk, TrunkConfig  # noqa: E402
from track_b_holographic_binding.data.corpus import CONTEXT_LEN, load_corpus  # noqa: E402
from track_b_holographic_binding.model.embeddings import build_embedding  # noqa: E402
from track_b_holographic_binding.model.evaluate import (  # noqa: E402
    embed_word_holographic,
    embed_word_kronecker,
    make_truncation_pairs,
    perplexity,
    perplexity_by_length_bucket,
    truncation_information_loss,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

D_MODEL = 192
N_LAYER = 6
N_HEAD = 6
D_FF = 768
LENGTH_BUCKETS = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 20)]

PROFILES = {
    "fast": dict(steps=50, batch_size=16, eval_every=25, lr=3e-4),
    "default": dict(steps=2000, batch_size=128, eval_every=250, lr=3e-4),
}


class WordLevelLM(nn.Module):
    def __init__(self, word_embedding: nn.Module, d_model: int, context_len: int, vocab_size: int):
        super().__init__()
        self.word_embedding = word_embedding
        self.pos_embedding = nn.Embedding(context_len, d_model)
        self.trunk = TransformerTrunk(
            TrunkConfig(d_model=d_model, n_layer=N_LAYER, n_head=N_HEAD, d_ff=D_FF, max_seq_len=context_len)
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        B, T = word_ids.shape
        positions = torch.arange(T, device=word_ids.device).unsqueeze(0).expand(B, T)
        x = self.word_embedding(word_ids) + self.pos_embedding(positions)
        h = self.trunk(x)
        return self.head(h)


def run(arm: str, profile_name: str, seed: int, dress: bool = False) -> dict:
    seed_everything(seed)
    device = get_device()
    cfg = PROFILES[profile_name]

    corpus = load_corpus(seed=seed)
    id_to_word, char_to_id = corpus["id_to_word"], corpus["char_to_id"]
    vocab_size = corpus["vocab_size"]
    train_windows, val_windows = corpus["train_windows"], corpus["val_windows"]

    word_embedding, extra = build_embedding(
        arm, id_to_word, char_to_id, D_MODEL, seed, max_len=64, dress=dress
    )
    model = WordLevelLM(word_embedding, D_MODEL, CONTEXT_LEN, vocab_size).to(device)
    n_params = count_params(model)
    embed_params = sum(p.numel() for p in word_embedding.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    rng = np.random.default_rng(seed)
    loss_curve, val_ppl_curve = [], []
    t0 = time.time()

    for step in range(1, cfg["steps"] + 1):
        idx = rng.integers(0, len(train_windows), size=cfg["batch_size"])
        batch = train_windows[idx]
        input_ids = torch.from_numpy(batch[:, :-1]).to(device)
        target_ids = torch.from_numpy(batch[:, 1:]).to(device)

        model.train()
        logits = model(input_ids)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), target_ids.reshape(B * T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            loss_curve.append({"step": step, "loss": float(loss.item())})
            model.eval()
            val_ppl_curve.append({"step": step, "val_perplexity": perplexity(model, val_windows, device)})

    elapsed = time.time() - t0

    model.eval()
    final_val_ppl = perplexity(model, val_windows, device)
    ppl_by_length = perplexity_by_length_bucket(model, val_windows, id_to_word, device, LENGTH_BUCKETS)

    # controlled truncation-information-loss probe (independent of the
    # natural corpus's word-length distribution, per evaluate.py's docstring)
    probe_rng = np.random.default_rng(seed + 1)
    pairs = make_truncation_pairs(probe_rng, corpus["alphabet"])
    if arm == "kronecker":
        embed_fn = lambda w: embed_word_kronecker(w, char_to_id, word_embedding.projection)  # noqa: E731
    else:
        embed_fn = lambda w: embed_word_holographic(  # noqa: E731
            w, char_to_id, extra["roles"], extra["fillers"]
        )
    truncation_result = truncation_information_loss(pairs, embed_fn)

    result = {
        "arm": arm,
        "dress": dress,
        "profile": profile_name,
        "seed": seed,
        "d_model": D_MODEL,
        "n_layer": N_LAYER,
        "n_head": N_HEAD,
        "d_ff": D_FF,
        "vocab_size": vocab_size,
        "context_len": CONTEXT_LEN,
        "n_params_total": n_params,
        "n_params_embedding_learned": embed_params,
        "corpus_raw_bytes": corpus["raw_bytes"],
        "corpus_n_word_tokens": corpus["n_word_tokens"],
        "corpus_max_natural_word_length": corpus["max_natural_word_length"],
        "steps": cfg["steps"],
        "batch_size": cfg["batch_size"],
        "elapsed_seconds": elapsed,
        "device": device_report(),
        "loss_curve": loss_curve,
        "val_perplexity_curve": val_ppl_curve,
        "final_val_perplexity": final_val_ppl,
        "perplexity_by_length_bucket": ppl_by_length,
        "truncation_information_loss": truncation_result,
    }
    if arm == "kronecker":
        from track_b_holographic_binding.model.embeddings import KroneckerWordEmbedding

        assert isinstance(word_embedding, KroneckerWordEmbedding)
        result["raw_slot_width"] = word_embedding.raw_width()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_tag = "" if seed == 0 else f"_seed{seed}"
    (RESULTS_DIR / f"{arm}{'_dress' if dress else ''}_{profile_name}{seed_tag}.json").write_text(
        json.dumps(result, indent=2)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["kronecker", "holographic"], required=True)
    parser.add_argument("--profile", choices=list(PROFILES), default="fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dress",
        action="store_true",
        help="holographic arm only: add a thin learned linear layer on top of the fixed table, so the "
        "comparison against the Kronecker arm (which has a learned projection) is not confounded by "
        "adaptability. Reported as its own arm.",
    )
    args = parser.parse_args()

    if args.dress and args.arm != "holographic":
        parser.error("--dress only applies to --arm holographic")

    result = run(args.arm, args.profile, args.seed, args.dress)

    print(
        f"arm={result['arm']} profile={result['profile']} "
        f"params_total={result['n_params_total']:,} params_embedding_learned={result['n_params_embedding_learned']:,} "
        f"device={result['device']['device']} elapsed={result['elapsed_seconds']:.1f}s"
    )
    print(f"final val perplexity: {result['final_val_perplexity']:.2f}")
    print("perplexity by natural word-length bucket:")
    for bucket, stats in result["perplexity_by_length_bucket"].items():
        if stats is None:
            print(f"  {bucket}: (no examples)")
        else:
            print(f"  {bucket}: ppl={stats['perplexity']:.2f}  (n={stats['n_tokens']})")
    tloss = result["truncation_information_loss"]
    print(
        f"truncation probe (pairs sharing first 32 chars, differing after): "
        f"mean cosine similarity={tloss['mean_cosine_similarity']:.4f} over {tloss['n_pairs']} pairs "
        f"({'INDISTINGUISHABLE, as expected for a 32-slot cap' if tloss['mean_cosine_similarity'] > 0.999 else 'distinguishable'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
