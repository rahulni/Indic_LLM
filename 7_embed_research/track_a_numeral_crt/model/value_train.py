"""The instructor's literal ask, tested in a transformer.

    python model/value_train.py --arm crt_value --task add --profile default

Everywhere else in Track A, numbers are tokenized digit-by-digit and the CRT
code is used only as a *positional* signal. Here each whole operand is a
SINGLE token whose embedding is the fixed 83-dimensional CRT **value** code of
the number itself -- the literal "embedding(9) knows it is 9" construction
that proofs/analytic_proof.py proves is exactly arithmetic-preserving. The
model must still emit the answer digit by digit.

    input:  [A] [op] [B] [=]        (A, B are one token each)
    output: answer digits, reversed, then EOS

Two arms, identical trunk and identical output head:

  crt_value  -- fixed, non-learned 83-dim CRT residue code of the operand's
                VALUE (from the same encode_onehot() the proof uses), plus a
                learned linear projection up to d_model. Zero learned
                parameters describe the number itself; the projection only
                reads the fixed code.
  learned    -- a plain nn.Embedding over the same operand vocabulary. Same
                parameter budget order, no arithmetic structure at all.

Both arms end in the SAME LayerNorm. That is not decoration: the CRT code is
a sparse binary vector (5 ones in 83 dims), so through a default-initialised
Linear its output scale sits far below nn.Embedding's, and a first run of this
experiment showed the crt_value arm's loss essentially flat (1.657 -> 1.653)
while the learned arm trained normally (1.568 -> 1.024). That gap was an
artifact of initialisation scale, not of the code's content. Normalising both
arms identically removes the artifact so the comparison is about the code.

Operands are capped at 3 digits so that both a+b and a*b stay inside
[0, N) = [0, 1,062,347) -- the range where the proof's guarantee actually
holds. Beyond it the code wraps, which would silently change the task.

NOT directly comparable to the digit-tokenized arms in model/train.py: the
tokenization is different, the sequence is much shorter, and the operand
vocabulary is bounded. This is a separate experiment answering a separate
question, and is reported as such.
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
from track_a_numeral_crt.data.generate_arithmetic import DIGIT_BASE, EOS, PAD, digits_reversed  # noqa: E402
from track_a_numeral_crt.proofs.analytic_proof import VALUE_PRIMES, encode_onehot  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

MAX_OPERAND = 1000  # exclusive: operands 0..999, so a*b <= 998,001 < N
VALUE_CODE_WIDTH = sum(VALUE_PRIMES)  # 83

# token layout: 0=PAD, 1=EOS, 2..11=digits, 12=PLUS, 13=STAR, 14=EQ (shared with
# generate_arithmetic.py so the digit ids mean the same thing), then operands
OP_PLUS, OP_STAR, OP_EQ = 12, 13, 14
OPERAND_BASE = 15  # operand v -> token OPERAND_BASE + v
VOCAB_SIZE = OPERAND_BASE + MAX_OPERAND

D_MODEL = 192
N_LAYER = 6
N_HEAD = 6
D_FF = 768
MAX_SEQ_LEN = 16

PROFILES = {
    "fast": dict(steps=50, batch_size=16, eval_every=25, lr=3e-4, eval_examples=5),
    "default": dict(steps=3000, batch_size=256, eval_every=250, lr=3e-4, eval_examples=200),
}


def value_code_table() -> torch.Tensor:
    """(MAX_OPERAND, 83) fixed CRT value codes, built by the SAME
    encode_onehot() the analytic proof verifies -- not a reimplementation."""
    n_arr = np.arange(MAX_OPERAND)
    mats = encode_onehot(n_arr, VALUE_PRIMES)
    return torch.tensor(np.concatenate(mats, axis=1), dtype=torch.float32)


class CRTValueEmbedding(nn.Module):
    """The CRT value code is a sparse binary vector (exactly one 1 per prime,
    so 5 ones in 83 dims). Pushed through a default-initialised nn.Linear its
    output std is far below what nn.Embedding produces, and the operand signal
    is simply drowned out by the symbol embeddings -- which would make this
    arm look like a failure of the *code* when it is really a failure of
    *scale*. The LayerNorm fixes that, and the learned control arm gets the
    identical LayerNorm so the comparison isolates the code's content rather
    than its initialisation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.symbol_embed = nn.Embedding(OPERAND_BASE, d_model)  # PAD/EOS/digits/ops
        self.register_buffer("code_table", value_code_table(), persistent=False)
        self.project = nn.Linear(VALUE_CODE_WIDTH, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        is_operand = input_ids >= OPERAND_BASE
        symbol_ids = torch.where(is_operand, torch.zeros_like(input_ids), input_ids)
        out = self.symbol_embed(symbol_ids)
        if is_operand.any():
            operand_values = (input_ids[is_operand] - OPERAND_BASE).clamp_(0, MAX_OPERAND - 1)
            out = out.clone()
            out[is_operand] = self.project(self.code_table[operand_values])
        return self.norm(out)


class LearnedValueEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.norm = nn.LayerNorm(d_model)  # identical treatment to the CRT arm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embed(input_ids))


class ValueTransformer(nn.Module):
    def __init__(self, embedding: nn.Module, d_model: int):
        super().__init__()
        self.embedding = embedding
        self.pos_embed = nn.Embedding(MAX_SEQ_LEN, d_model)
        self.trunk = TransformerTrunk(
            TrunkConfig(d_model=d_model, n_layer=N_LAYER, n_head=N_HEAD, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN)
        )
        self.head = nn.Linear(d_model, DIGIT_BASE + 10)  # only ever emits digits or EOS

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.embedding(input_ids) + self.pos_embed(pos)
        return self.head(self.trunk(x))


def encode_example(a: int, b: int, op: str) -> dict:
    answer = a + b if op == "+" else a * b
    tokens = [OPERAND_BASE + a, OP_PLUS if op == "+" else OP_STAR, OPERAND_BASE + b, OP_EQ]
    answer_start = len(tokens)
    tokens += [DIGIT_BASE + d for d in digits_reversed(answer)]
    tokens.append(EOS)
    return {"tokens": tokens, "answer_start": answer_start, "a": a, "b": b, "answer": answer}


def sample_batch(rng: np.random.Generator, batch_size: int, task: str) -> dict:
    op = "+" if task == "add" else "*"
    examples = [
        encode_example(int(rng.integers(0, MAX_OPERAND)), int(rng.integers(0, MAX_OPERAND)), op)
        for _ in range(batch_size)
    ]
    max_len = max(len(ex["tokens"]) for ex in examples)
    input_ids = np.full((batch_size, max_len), PAD, dtype=np.int64)
    loss_mask = np.zeros((batch_size, max_len), dtype=bool)
    for i, ex in enumerate(examples):
        T = len(ex["tokens"])
        input_ids[i, :T] = ex["tokens"]
        loss_mask[i, ex["answer_start"] - 1 : T - 1] = True
    target_ids = np.full_like(input_ids, PAD)
    target_ids[:, :-1] = input_ids[:, 1:]
    # The head only spans PAD/EOS/digits (DIGIT_BASE + 10 classes), but the raw
    # next-token targets also contain operand and operator ids far outside that
    # range. loss_mask excludes those positions from the loss, yet
    # cross_entropy still indexes every position -- so neutralise them here
    # rather than letting an out-of-range class id reach the kernel.
    target_ids[~loss_mask] = PAD
    return {"input_ids": input_ids, "target_ids": target_ids, "loss_mask": loss_mask}


@torch.no_grad()
def exact_match_accuracy(model, task: str, n_examples: int, device, seed: int) -> dict:
    op = "+" if task == "add" else "*"
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(n_examples):
        a = int(rng.integers(0, MAX_OPERAND))
        b = int(rng.integers(0, MAX_OPERAND))
        expected = a + b if op == "+" else a * b
        tokens = [OPERAND_BASE + a, OP_PLUS if op == "+" else OP_STAR, OPERAND_BASE + b, OP_EQ]
        digits: list[int] = []
        for _ in range(8):
            logits = model(torch.tensor([tokens], dtype=torch.long, device=device))
            nxt = int(torch.argmax(logits[0, -1]).item())
            if nxt == EOS:
                break
            if not (DIGIT_BASE <= nxt < DIGIT_BASE + 10):
                digits = []
                break
            digits.append(nxt - DIGIT_BASE)
            tokens.append(nxt)
        if digits and sum(d * 10**i for i, d in enumerate(digits)) == expected:
            correct += 1
    return {"accuracy": correct / n_examples, "n": n_examples}


def run(arm: str, task: str, profile_name: str, seed: int) -> dict:
    seed_everything(seed)
    device = get_device()
    cfg = PROFILES[profile_name]

    embedding = CRTValueEmbedding(D_MODEL) if arm == "crt_value" else LearnedValueEmbedding(D_MODEL)
    model = ValueTransformer(embedding, D_MODEL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    rng = np.random.default_rng(seed)
    loss_curve, acc_curve = [], []
    t0 = time.time()
    for step in range(1, cfg["steps"] + 1):
        batch = sample_batch(rng, cfg["batch_size"], task)
        input_ids = torch.from_numpy(batch["input_ids"]).to(device)
        target_ids = torch.from_numpy(batch["target_ids"]).to(device)
        loss_mask = torch.from_numpy(batch["loss_mask"]).to(device)

        model.train()
        logits = model(input_ids)
        B, T, V = logits.shape
        ce = F.cross_entropy(logits.reshape(B * T, V), target_ids.reshape(B * T), reduction="none").reshape(B, T)
        loss = (ce * loss_mask.float()).sum() / loss_mask.float().sum().clamp_min(1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            loss_curve.append({"step": step, "loss": float(loss.item())})
            model.eval()
            acc_curve.append(
                {"step": step, **exact_match_accuracy(model, task, min(cfg["eval_examples"], 25), device, 1000 + step)}
            )

    elapsed = time.time() - t0
    model.eval()
    final = exact_match_accuracy(model, task, cfg["eval_examples"], device, 999)

    result = {
        "experiment": "value_embedding",
        "arm": arm,
        "task": task,
        "profile": profile_name,
        "seed": seed,
        "max_operand": MAX_OPERAND,
        "value_code_width": VALUE_CODE_WIDTH,
        "value_primes": VALUE_PRIMES,
        "n_params": count_params(model),
        "n_params_embedding": count_params(model.embedding),
        "steps": cfg["steps"],
        "batch_size": cfg["batch_size"],
        "elapsed_seconds": elapsed,
        "device": device_report(),
        "loss_curve": loss_curve,
        "accuracy_curve": acc_curve,
        "final_accuracy": final,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"value_{arm}_{task}_{profile_name}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["crt_value", "learned"], required=True)
    parser.add_argument("--task", choices=["add", "mul"], required=True)
    parser.add_argument("--profile", choices=list(PROFILES), default="fast")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    r = run(args.arm, args.task, args.profile, args.seed)
    print(
        f"arm={r['arm']} task={r['task']} profile={r['profile']} params={r['n_params']:,} "
        f"(embedding {r['n_params_embedding']:,}) device={r['device']['device']} "
        f"elapsed={r['elapsed_seconds']:.1f}s"
    )
    print(
        f"final exact-match accuracy on operands in [0,{r['max_operand']}): "
        f"{r['final_accuracy']['accuracy']:.2%} (n={r['final_accuracy']['n']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
