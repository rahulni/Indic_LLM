"""Exact-match evaluation via greedy autoregressive generation.

Deliberately NOT the same thing as analytic_proof.py: that script proves the
CRT construction is exact by direct computation, with no model involved.
This module asks a different, second question -- does a *trained
transformer*, using one of the three embedding arms, actually produce the
right answer digits? -- and never conflates the two.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from track_a_numeral_crt.data.generate_arithmetic import (  # noqa: E402
    DIGIT_BASE,
    EOS,
    EQ,
    PLACE_VALUE_SENTINEL,
    PLUS,
    STAR,
    digits_reversed,
    random_operand,
)


@torch.no_grad()
def generate_answer(model, a: int, b: int, op: str, max_answer_digits: int, device) -> int | None:
    """Greedy-decodes the answer to `a op b`, one digit at a time, using the
    model's own place-value scheme for whichever digit it is about to emit
    (the k-th generated digit gets place value k -- identical to how
    training examples are labeled). Returns None if generation never emits
    EOS within the budget (a real failure mode, not silently papered over)."""
    a_digits = digits_reversed(a)
    b_digits = digits_reversed(b)
    tokens: list[int] = []
    place_values: list[int] = []
    for i, d in enumerate(a_digits):
        tokens.append(DIGIT_BASE + d)
        place_values.append(i)
    tokens.append(PLUS if op == "+" else STAR)
    place_values.append(PLACE_VALUE_SENTINEL)
    for i, d in enumerate(b_digits):
        tokens.append(DIGIT_BASE + d)
        place_values.append(i)
    tokens.append(EQ)
    place_values.append(PLACE_VALUE_SENTINEL)

    generated_digits: list[int] = []
    for _ in range(max_answer_digits + 1):  # +1 budget slot for EOS itself
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        pv_ids = torch.tensor([place_values], dtype=torch.long, device=device)
        logits = model(input_ids, pv_ids)
        next_token = int(torch.argmax(logits[0, -1]).item())
        if next_token == EOS:
            value = 0
            for i, d in enumerate(generated_digits):
                value += d * (10**i)
            return value
        if not (DIGIT_BASE <= next_token < DIGIT_BASE + 10):
            return None  # generated a structurally invalid token -- a real failure
        place_values.append(len(generated_digits))
        tokens.append(next_token)
        generated_digits.append(next_token - DIGIT_BASE)
    return None  # ran out of budget without EOS


def evaluate_length_generalization(
    model,
    task: str,
    test_lengths: list[int],
    n_examples: int,
    device,
    seed: int,
) -> dict:
    """For each test digit-length, samples n_examples random (a, b) pairs of
    exactly that length and reports exact-match accuracy. Returns a dict
    keyed by length -> {accuracy, n, max_answer_digits}."""
    op = "+" if task == "add" else "*"
    rng = np.random.default_rng(seed)
    results = {}
    for length in test_lengths:
        max_answer_digits = 2 * length + 1  # generous bound covering + and *
        correct = 0
        for _ in range(n_examples):
            a = random_operand(rng, length)
            b = random_operand(rng, length)
            expected = a + b if op == "+" else a * b
            predicted = generate_answer(model, a, b, op, max_answer_digits, device)
            if predicted == expected:
                correct += 1
        results[length] = {
            "accuracy": correct / n_examples,
            "n": n_examples,
            "max_answer_digits": max_answer_digits,
        }
    return results
