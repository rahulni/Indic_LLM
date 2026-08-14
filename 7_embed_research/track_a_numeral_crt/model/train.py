"""CLI trainer for Track A's three embedding arms.

    python model/train.py --arm crt --task add --profile fast
    python model/train.py --arm abacus --task mul --profile default

Every run answers a single, honestly-labeled question: does this embedding
scheme help a small transformer learn digit arithmetic and generalize to
longer operands than it was trained on? This is NOT the exact-arithmetic
proof -- that is proofs/analytic_proof.py, which needs no trained model and
is run separately.
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
from track_a_numeral_crt.data.generate_arithmetic import VOCAB_SIZE, sample_batch  # noqa: E402
from track_a_numeral_crt.model.embeddings import build_embedding  # noqa: E402
from track_a_numeral_crt.model.evaluate import evaluate_length_generalization  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

D_MODEL = 192
N_LAYER = 6
N_HEAD = 6
D_FF = 768
MAX_SEQ_LEN = 48
TRAIN_LEN_RANGE = (1, 4)
TEST_LENGTHS = [1, 2, 3, 4, 5, 6, 7, 8]

PROFILES = {
    "fast": dict(
        steps=50, batch_size=16, eval_every=25, lr=3e-4, quick_eval_examples=5, final_eval_examples=5
    ),
    "default": dict(
        steps=3000, batch_size=256, eval_every=250, lr=3e-4, quick_eval_examples=20, final_eval_examples=50
    ),
}


class DigitTransformer(nn.Module):
    def __init__(self, embedding: nn.Module, d_model: int):
        super().__init__()
        self.embedding = embedding
        self.trunk = TransformerTrunk(
            TrunkConfig(d_model=d_model, n_layer=N_LAYER, n_head=N_HEAD, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN)
        )
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, input_ids: torch.Tensor, place_value_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids, place_value_ids)
        h = self.trunk(x)
        return self.head(h)


def compute_loss(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(B * T, V), target_ids.reshape(B * T), reduction="none").reshape(B, T)
    mask = loss_mask.float()
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def run(
    arm: str,
    task: str,
    profile_name: str,
    dress: bool,
    seed: int,
    random_offset_max: int = 0,
    eval_examples: int | None = None,
) -> dict:
    seed_everything(seed)
    device = get_device()
    cfg = PROFILES[profile_name]

    embedding = build_embedding(arm, D_MODEL, dress=dress)
    model = DigitTransformer(embedding, D_MODEL).to(device)
    n_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    rng = np.random.default_rng(seed)
    loss_curve = []
    quick_acc_curve = []
    t0 = time.time()

    for step in range(1, cfg["steps"] + 1):
        batch = sample_batch(
            rng, cfg["batch_size"], task, TRAIN_LEN_RANGE, MAX_SEQ_LEN, random_offset_max=random_offset_max
        )
        input_ids = torch.from_numpy(batch["input_ids"]).to(device)
        target_ids = torch.from_numpy(batch["target_ids"]).to(device)
        place_value_ids = torch.from_numpy(batch["place_value_ids"]).to(device)
        loss_mask = torch.from_numpy(batch["loss_mask"]).to(device)

        model.train()
        logits = model(input_ids, place_value_ids)
        loss = compute_loss(logits, target_ids, loss_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            loss_curve.append({"step": step, "loss": float(loss.item())})
            model.eval()
            quick = evaluate_length_generalization(
                model,
                task,
                test_lengths=[TRAIN_LEN_RANGE[1]],
                n_examples=cfg["quick_eval_examples"],
                device=device,
                seed=1000 + step,
            )
            quick_acc_curve.append({"step": step, "accuracy": quick[TRAIN_LEN_RANGE[1]]["accuracy"]})

    elapsed = time.time() - t0

    model.eval()
    n_final_eval = eval_examples if eval_examples is not None else cfg["final_eval_examples"]
    final_eval = evaluate_length_generalization(
        model, task, test_lengths=TEST_LENGTHS, n_examples=n_final_eval, device=device, seed=999
    )

    result = {
        "arm": arm,
        "task": task,
        "profile": profile_name,
        "dress": dress,
        "seed": seed,
        "random_offset_max": random_offset_max,
        "d_model": D_MODEL,
        "n_layer": N_LAYER,
        "n_head": N_HEAD,
        "d_ff": D_FF,
        "n_params": n_params,
        "train_len_range": list(TRAIN_LEN_RANGE),
        "steps": cfg["steps"],
        "batch_size": cfg["batch_size"],
        "elapsed_seconds": elapsed,
        "device": device_report(),
        "loss_curve": loss_curve,
        "quick_accuracy_curve": quick_acc_curve,
        "final_eval_by_length": {str(k): v for k, v in final_eval.items()},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{'_dress' if dress else ''}{'_offset' if random_offset_max else ''}"
    seed_tag = "" if seed == 0 else f"_seed{seed}"
    out_name = f"{arm}{suffix}_{task}_{profile_name}{seed_tag}.json"
    (RESULTS_DIR / out_name).write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["baseline", "abacus", "crt"], required=True)
    parser.add_argument("--task", choices=["add", "mul"], required=True)
    parser.add_argument("--profile", choices=list(PROFILES), default="fast")
    parser.add_argument(
        "--dress",
        action="store_true",
        help="CRT arm only: add a thin learned linear layer on top of the fixed code (ablation)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random-offset",
        type=int,
        default=0,
        metavar="K",
        help="McLeish et al. 2024's length-generalization trick: add a random offset in [0, K] to every "
        "digit's place-value index, per training example (evaluation always uses offset 0). "
        "K=24 keeps train-time indices inside MAX_PLACE_VALUE=32. Ignored by --arm baseline (NoPE).",
    )
    parser.add_argument(
        "--eval-examples", type=int, default=None, help="override the profile's final-eval example count"
    )
    args = parser.parse_args()

    if args.dress and args.arm != "crt":
        parser.error("--dress only applies to --arm crt")
    if args.random_offset and args.arm == "baseline":
        parser.error("--random-offset has no effect on --arm baseline, which uses no place-value signal")

    result = run(
        args.arm, args.task, args.profile, args.dress, args.seed, args.random_offset, args.eval_examples
    )

    arm_label = result["arm"]
    if result["dress"]:
        arm_label += "+dress"
    if result["random_offset_max"]:
        arm_label += f"+offset{result['random_offset_max']}"
    print(
        f"arm={arm_label} task={result['task']} profile={result['profile']} "
        f"params={result['n_params']:,} device={result['device']['device']} "
        f"elapsed={result['elapsed_seconds']:.1f}s"
    )
    print(f"final accuracy by test digit-length (train range was {result['train_len_range']}):")
    for length_str, stats in result["final_eval_by_length"].items():
        tag = "ID" if int(length_str) <= TRAIN_LEN_RANGE[1] else "OOD"
        print(f"  len={int(length_str):>2} [{tag}]  acc={stats['accuracy']:.2%}  (n={stats['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
