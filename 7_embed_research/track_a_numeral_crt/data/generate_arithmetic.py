"""Synthetic multi-digit addition / multiplication dataset generator.

Sequence layout (standard in this literature: reversed digits, ones-place
first, so the model reads/writes each answer digit in the same order carries
propagate):

    [a's digits, reversed] [OP] [b's digits, reversed] [=] [answer digits, reversed] [EOS] [PAD...]

Loss is computed only on the positions predicting the answer digits and the
final EOS -- not on the operand echo or the padding.

Each digit token also carries a `place_value_id`: its position within its
OWN number (units=0, tens=1, ...), restarting at 0 for a, b, and the answer
independently. This is what the Abacus-lite and CRT/Kronecker embedding arms
consume; the baseline arm ignores it entirely (NoPE -- see model/embeddings.py).
"""
from __future__ import annotations

import numpy as np

PAD, EOS = 0, 1
DIGIT_BASE = 2  # digit d -> token id DIGIT_BASE + d, for d in 0..9
PLUS, STAR, EQ = 12, 13, 14
VOCAB_SIZE = 15
PLACE_VALUE_SENTINEL = 0  # for non-digit tokens; harmless, never semantically used


def digits_reversed(n: int, width: int | None = None) -> list[int]:
    if n == 0:
        digits = [0]
    else:
        digits = []
        m = n
        while m > 0:
            digits.append(m % 10)
            m //= 10
    if width is not None:
        digits = digits + [0] * (width - len(digits))
    return digits  # ones place first


def random_operand(rng: np.random.Generator, length: int) -> int:
    if length <= 1:
        return int(rng.integers(0, 10))
    lo, hi = 10 ** (length - 1), 10**length - 1
    return int(rng.integers(lo, hi + 1))


def encode_example(a: int, b: int, op: str) -> dict:
    assert op in ("+", "*")
    answer = a + b if op == "+" else a * b

    a_digits = digits_reversed(a)
    b_digits = digits_reversed(b)
    ans_digits = digits_reversed(answer)

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

    answer_start = len(tokens)  # index of the first answer-digit token
    for i, d in enumerate(ans_digits):
        tokens.append(DIGIT_BASE + d)
        place_values.append(i)
    tokens.append(EOS)
    place_values.append(PLACE_VALUE_SENTINEL)

    return {
        "tokens": tokens,
        "place_values": place_values,
        "answer_start": answer_start,  # predicting tokens[answer_start:] is the task
        "a": a,
        "b": b,
        "answer": answer,
    }


def sample_batch(
    rng: np.random.Generator,
    batch_size: int,
    task: str,
    len_range: tuple[int, int],
    max_seq_len: int,
    random_offset_max: int = 0,
) -> dict:
    """Returns padded numpy arrays ready for teacher-forced training:
    input_ids[:, :-1] is fed to the model, target_ids[:, :-1] = input_ids[:, 1:]
    is what it should predict, and loss_mask marks which of those predicted
    positions are actual answer digits / the closing EOS (not operand echo,
    not padding).

    `random_offset_max > 0` enables McLeish et al. 2024's actual
    length-generalization mechanism: one random offset k in
    [0, random_offset_max] is drawn PER EXAMPLE and added to every place-value
    index in that example (the offset is shared across a, b, and the answer, so
    their digit alignment -- the thing the model must learn -- is untouched;
    only the absolute index changes). The model therefore never learns to rely
    on "units digit == place value 0", which is what breaks at test time when
    numbers are longer than anything seen in training. Evaluation always uses
    offset 0.
    """
    op = "+" if task == "add" else "*"
    examples = []
    for _ in range(batch_size):
        la = int(rng.integers(len_range[0], len_range[1] + 1))
        lb = int(rng.integers(len_range[0], len_range[1] + 1))
        a = random_operand(rng, la)
        b = random_operand(rng, lb)
        ex = encode_example(a, b, op)
        if random_offset_max > 0:
            offset = int(rng.integers(0, random_offset_max + 1))
            ex["place_values"] = [
                pv + offset if tok_is_digit else pv
                for pv, tok_is_digit in zip(
                    ex["place_values"], [DIGIT_BASE <= t < DIGIT_BASE + 10 for t in ex["tokens"]]
                )
            ]
        examples.append(ex)

    max_len = max(len(ex["tokens"]) for ex in examples)
    assert max_len <= max_seq_len, f"example of length {max_len} exceeds max_seq_len={max_seq_len}"
    # trim each batch to the length it actually needs (not the global upper
    # bound) -- most training examples are far shorter than max_seq_len, and
    # padding them all the way out would waste attention compute for nothing
    input_ids = np.full((batch_size, max_len), PAD, dtype=np.int64)
    place_value_ids = np.full((batch_size, max_len), PLACE_VALUE_SENTINEL, dtype=np.int64)
    loss_mask = np.zeros((batch_size, max_len), dtype=bool)

    for i, ex in enumerate(examples):
        T = len(ex["tokens"])
        input_ids[i, :T] = ex["tokens"]
        place_value_ids[i, :T] = ex["place_values"]
        # predicting position t means: does target at t (== input[t+1]) fall
        # in the answer-digit-or-EOS region? that region is
        # tokens[answer_start : T], produced by predicting from position
        # (answer_start - 1) through (T - 2).
        loss_mask[i, ex["answer_start"] - 1 : T - 1] = True

    target_ids = np.full_like(input_ids, PAD)
    target_ids[:, :-1] = input_ids[:, 1:]

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "place_value_ids": place_value_ids,
        "loss_mask": loss_mask,
        "examples": examples,  # for exact-match eval (has a, b, answer, answer_start)
    }
