"""The model every run shares: one Block, one residual stream, four ways to back-propagate.

The residual stream is carried in **fp32** while `Block` computes in bf16. That is not
fussiness. A reversible backward pass reconstructs h_l from h_{l+1} by subtracting what it
just added, and the rounding error of each subtraction feeds into the next one, so the error
compounds with depth. bf16 keeps ~8 mantissa bits; fp32 keeps ~24. `nb01` measures exactly
what that buys - see `reconstruction_drift()` in reversible.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class GPTConfig:
    """21.05M parameters at the defaults.

    `vocab_size` is 8192 rather than GPT-2's 50257 on purpose. At 50257 the tied embedding
    table alone is 50257 x 384 = 19.3M, which would eat a "20M model" whole and leave no
    transformer inside it to study. An 8k BPE trained on the corpus keeps the embedding at
    3.1M (15%) and spends the other 17.9M on the ten blocks we actually care about.
    """

    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 10
    n_head: int = 6
    n_embd: int = 384
    mlp_ratio: int = 4
    gamma_init: float = 0.1  # LayerScale; also the Euler contraction knob, see below
    tie_embeddings: bool = True
    # Reversibility forbids both of these (transcript, 01:32). They are off in *every* run
    # so that the baseline is not quietly given an advantage the reversible runs cannot have.
    dropout: float = 0.0
    weight_decay: float = 0.0

    def params_total(self) -> int:
        emb = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        per_block = (4 * self.n_embd ** 2) + (2 * self.mlp_ratio * self.n_embd ** 2)
        per_block += 4 * self.n_embd + self.n_embd  # two LayerNorms + gamma
        head = 0 if self.tie_embeddings else self.vocab_size * self.n_embd
        return emb + self.n_layer * per_block + head + 2 * self.n_embd

    def dict(self):
        return asdict(self)


# --------------------------------------------------------------------------------------
# the block - this is `F_l`, WITHOUT its residual connection
# --------------------------------------------------------------------------------------
class Block(nn.Module):
    """G_l(h) = gamma_l * (attn(LN(h)) + mlp(LN(h + attn)))  -- the residual branch only.

    Every integrator adds this to the stream itself, in its own way. Keeping the residual
    *outside* the block is what lets STORE and EULER be the same function: there is exactly
    one definition of "what this layer adds", and the four stacks differ only in where they
    add it and how they get it back.
    """

    def __init__(self, cfg: GPTConfig, width: int | None = None):
        super().__init__()
        d = width if width is not None else cfg.n_embd
        assert d % cfg.n_head == 0, f"width {d} not divisible by n_head {cfg.n_head}"
        self.d, self.n_head = d, cfg.n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, cfg.mlp_ratio * d, bias=False)
        self.fc2 = nn.Linear(cfg.mlp_ratio * d, d, bias=False)
        # LayerScale. Two jobs at once:
        #   1. the usual one - a small init keeps deep residual stacks stable early on;
        #   2. it is the Euler contraction knob. The fixed-point inverse
        #      x <- h_{l+1} - G_l(x) converges only while  Lip(G_l) = gamma * Lip(F_l) < 1,
        #      and gamma is the only factor we get to set. `contraction_estimate()` in
        #      reversible.py measures whether it is still holding during training.
        self.gamma = nn.Parameter(torch.full((d,), cfg.gamma_init))
        self.use_amp = True

    def _attn(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        # flash SDPA: never materialises the T x T matrix, so attention costs no activation
        # memory to speak of. Worth saying out loud - it means the memory this study saves
        # is the *residual stream*, not the attention map.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

    def forward(self, x):
        amp = self.use_amp and x.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            h = self._attn(self.ln1(x))
            h = h + self.fc2(F.gelu(self.fc1(self.ln2(x + h))))
        # back up to the stream's dtype before the integrator adds it. `x.dtype`, not
        # `.float()`: the stream is fp32 in training but fp64 in the gradient-equivalence
        # tests, and hard-coding fp32 would quietly cap those tests at fp32 precision.
        return self.gamma * h.to(x.dtype)


class CoupledBlock(nn.Module):
    """A RevNet coupling layer: y1 = x1 + F(x2); y2 = x2 + G(y1).

    Exactly invertible with no step size and no fixed point, which makes it the strongest
    of the three on paper. The catch is in the shapes: F and G each see d/2 channels, so at
    equal parameter count it is a different - narrower - model, and that is what the probe
    run is there to price.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % 2 == 0
        self.half = cfg.n_embd // 2
        self.f = Block(cfg, width=self.half)
        self.g = Block(cfg, width=self.half)

    def forward(self, x):  # used by the STORE path only; the engine calls f/g directly
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return torch.cat([y1, y2], dim=-1)


# --------------------------------------------------------------------------------------
# the loss head
# --------------------------------------------------------------------------------------
def chunked_cross_entropy(h, targets, head_weight, n_chunks: int = 4):
    """Cross entropy without ever holding all the fp32 logits at once.

    Once the activations are gone, this is what is left standing. At B=128, T=512, V=8192
    the logits are 128*512*8192*4 = 2.1 GiB in fp32 - on its own, larger than everything
    reversibility just saved. Each chunk is wrapped in `checkpoint`, so its logits are
    recomputed in backward and only one chunk is ever live.
    """
    from torch.utils.checkpoint import checkpoint

    def upcast(t):
        # softmax in reduced precision costs real loss quality; fp64 streams stay fp64
        return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t

    flat_h = h.reshape(-1, h.size(-1))
    flat_t = targets.reshape(-1)
    n = flat_h.size(0)
    if n_chunks <= 1:
        return F.cross_entropy(upcast(F.linear(flat_h, head_weight)), flat_t)

    size = math.ceil(n / n_chunks)
    total = flat_h.new_zeros(())
    for i in range(0, n, size):
        hs, ts = flat_h[i : i + size], flat_t[i : i + size]

        def one(hs, ts, w):
            return F.cross_entropy(upcast(F.linear(hs, w)), ts, reduction="sum")

        total = total + checkpoint(one, hs, ts, head_weight, use_reentrant=False)
    return total / n


# --------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------
class GPT(nn.Module):
    """One GPT, four backward strategies.

    `mode` selects the stack: "store", "euler", "midpoint", "coupling", "checkpoint".
    "store" and "euler" produce *identical* forward values; the assertion is in the tests.
    """

    MODES = ("store", "checkpoint", "euler", "midpoint", "coupling", "euler_implicit")

    def __init__(self, cfg: GPTConfig, mode: str = "store", euler_iters: int = 4,
                 ce_chunks: int = 1):
        super().__init__()
        assert mode in self.MODES, f"mode must be one of {self.MODES}"
        self.cfg, self.mode, self.euler_iters, self.ce_chunks = cfg, mode, euler_iters, ce_chunks
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        block_cls = CoupledBlock if mode == "coupling" else Block
        self.blocks = nn.ModuleList([block_cls(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.wte.weight
        self.apply(self._init)
        # per-layer diagnostics the engines fill in during backward, read by train.py
        self.diag: dict[str, float] = {}

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def embed(self, idx):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} exceeds block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        # this defines the residual stream's dtype: whatever the parameters are (fp32 in
        # training, fp64 under test). The blocks compute in bf16 and come back up to it.
        return self.wte(idx) + self.wpe(pos)

    def forward(self, idx, targets=None):
        from . import reversible  # local import: reversible.py imports model.py for types

        h = self.embed(idx)
        h = reversible.run_stack(self, h)
        h = self.ln_f(h)
        if targets is None:
            return self.head(h)
        return chunked_cross_entropy(h, targets, self.head.weight, self.ce_chunks)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            crop = idx[:, -self.cfg.block_size :]
            logits = self(crop)[:, -1, :].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx

    def param_groups(self):
        """Every run uses weight_decay = 0.0 (see GPTConfig). Kept as a seam so the
        `weight_decay` side-probe can turn it on for one run and see what actually breaks."""
        return [{"params": [p for p in self.parameters() if p.requires_grad],
                 "weight_decay": self.cfg.weight_decay}]
