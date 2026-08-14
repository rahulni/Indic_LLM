"""A minimal decoder-only Transformer trunk, shared by both tracks.

The trunk consumes already-embedded sequences, shape [B, T, d_model], rather
than token ids — the embedding scheme under test (baseline / Abacus-lite /
CRT-positional for Track A; Kronecker-slot / holographic for Track B) is
fully pluggable and lives outside this file, in each track's own
model/embeddings.py. This is what makes the arm comparisons a fair,
embedding-only ablation: every arm shares this exact trunk.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrunkConfig:
    d_model: int
    n_layer: int
    n_head: int
    d_ff: int
    max_seq_len: int
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrunkConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout
        # True = masked out (future position); inverted at use-site because
        # F.scaled_dot_product_attention's bool mask means "allowed to attend".
        self.register_buffer(
            "future_mask",
            torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        allowed = ~self.future_mask[:T, :T]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: TrunkConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: TrunkConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerTrunk(nn.Module):
    """Pre-norm decoder-only trunk.

    Input: [B, T, d_model] pre-computed embeddings.
    Output: [B, T, d_model] hidden states — the caller attaches its own
    output head (e.g. a digit classifier for Track A, a word softmax for
    Track B).
    """

    def __init__(self, cfg: TrunkConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)
