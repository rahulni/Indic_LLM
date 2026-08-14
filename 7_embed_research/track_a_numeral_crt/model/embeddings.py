"""Three embedding arms, sharing the same digit/op identity table shape and
the same transformer trunk (common/transformer.py). The ONLY thing that
differs between arms is how -- or whether -- place-value position
information is added. That is the entire ablation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from track_a_numeral_crt.data.generate_arithmetic import VOCAB_SIZE  # noqa: E402
from track_a_numeral_crt.proofs.analytic_proof import POSITION_PRIMES, encode_onehot  # noqa: E402

CODE_WIDTH = sum(POSITION_PRIMES)  # 12 (primes [2,3,7])
MAX_PLACE_VALUE = 32  # digit place-value index only ever runs 0..31


def crt_position_code_table(max_place_value: int = MAX_PLACE_VALUE) -> torch.Tensor:
    """Fixed (max_place_value, CODE_WIDTH) table: row i is the concatenated
    one-hot CRT residue code of place-value i, using the same primes and the
    same encode_onehot() used by the standalone proof -- this is not a
    reimplementation, it is the literal proved construction, reused."""
    n_arr = np.arange(max_place_value)
    mats = encode_onehot(n_arr, POSITION_PRIMES)
    table = np.concatenate(mats, axis=1)
    return torch.tensor(table, dtype=torch.float32)


class BaselineEmbedding(nn.Module):
    """Arm 1 (NoPE): digit/op identity only, no positional signal at all.
    The causal attention mask is the model's only source of token-order
    information. A standard, citable baseline in the length-generalization
    literature (position embeddings are not always necessary or helpful)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, d_model)

    def forward(self, input_ids: torch.Tensor, place_value_ids: torch.Tensor) -> torch.Tensor:
        del place_value_ids
        return self.token_embed(input_ids)


class AbacusLiteEmbedding(nn.Module):
    """Arm 2: digit/op identity (token_dim) concatenated with a LEARNED
    embedding of the token's place-value index within its own operand
    (code_width dims). A lightweight reimplementation of McLeish et al.
    2024's core mechanism -- not their full looped/recurrent architecture."""

    def __init__(self, d_model: int, code_width: int = CODE_WIDTH, max_place_value: int = MAX_PLACE_VALUE):
        super().__init__()
        token_dim = d_model - code_width
        self.token_embed = nn.Embedding(VOCAB_SIZE, token_dim)
        self.place_embed = nn.Embedding(max_place_value, code_width)

    def forward(self, input_ids: torch.Tensor, place_value_ids: torch.Tensor) -> torch.Tensor:
        tok = self.token_embed(input_ids)
        pos = self.place_embed(place_value_ids)
        return torch.cat([tok, pos], dim=-1)


class CRTPositionalEmbedding(nn.Module):
    """Arm 3: digit/op identity (token_dim) concatenated with the FIXED,
    non-learned CRT/Kronecker residue code of the token's place-value index
    (code_width dims) -- appended dimensions, exactly as proven in
    analytic_proof.py. `dress=True` optionally adds a thin learned linear
    layer on top (the ablation named in the plan); this breaks the
    "zero learned parameters for position" property, so it is reported
    separately, never silently blended into the core arm-3 numbers."""

    def __init__(
        self,
        d_model: int,
        code_width: int = CODE_WIDTH,
        max_place_value: int = MAX_PLACE_VALUE,
        dress: bool = False,
    ):
        super().__init__()
        token_dim = d_model - code_width
        self.token_embed = nn.Embedding(VOCAB_SIZE, token_dim)
        self.register_buffer("code_table", crt_position_code_table(max_place_value), persistent=False)
        self.dress = nn.Linear(d_model, d_model) if dress else None

    def forward(self, input_ids: torch.Tensor, place_value_ids: torch.Tensor) -> torch.Tensor:
        tok = self.token_embed(input_ids)
        pos = self.code_table[place_value_ids]
        out = torch.cat([tok, pos], dim=-1)
        if self.dress is not None:
            out = self.dress(out)
        return out


def build_embedding(arm: str, d_model: int, dress: bool = False) -> nn.Module:
    if arm == "baseline":
        return BaselineEmbedding(d_model)
    if arm == "abacus":
        return AbacusLiteEmbedding(d_model)
    if arm == "crt":
        return CRTPositionalEmbedding(d_model, dress=dress)
    raise ValueError(f"unknown arm {arm!r}, choose from baseline, abacus, crt")
