# -*- coding: utf-8 -*-
"""A small neural language model with hand-written backpropagation.

Stdlib only -- no numpy, no torch. The loss values in the learning ledger are
real cross-entropy from a real forward pass, and the gradient norms are real L2
norms over real accumulated gradients. Nothing here is simulated.

Architecture (a Bengio-style neural n-gram, not a transformer)::

    context   = the previous k tokens
    x         = concat(embed(t_{i-k}) ... embed(t_{i-1}))     k*d
    hidden    = tanh(W1 x + b1)                                h
    logits    = W2 hidden + b2                                 V
    loss_i    = -ln softmax(logits)[target_i]

**The attention mask is load-bearing, not decorative.** A context slot is zeroed
when ``segment_ids`` says that position belongs to a different document, or when
it is padding. So a token at the start of the second document in a packed
sequence genuinely cannot see the first document -- the information is not in
its input at all. Changing the mask changes the loss, which is what makes the
mask worth recording.

Why this model and not a transformer: a 256x256 attention matrix is ~2M
multiply-adds per sample per layer, which in CPython is three orders of
magnitude over the demo's time budget. The masked-context model keeps the
property that matters here -- the mask genuinely gates what the model can
condition on -- at about 1% of the cost, with no dependencies at all.

A real transformer lives in :mod:`tdes.model_torch` and is selected with
``--backend torch``. Both satisfy the :class:`tdes.lm.LanguageModel` protocol,
so the trainer and every ledger record are identical either way. Earlier
versions of this docstring claimed such a protocol already existed; it did not,
and the claim is what prompted building it.

Cost is dominated by the V x h output layer, which is why vocabulary size is the
main lever in the profile table.
"""
from __future__ import annotations

import math
from operator import mul

from .config import PAD_ID
from .determinism import DeterministicRNG
from .lm import SampleLoss


class NeuralLM:
    """Embedding -> tanh hidden -> softmax over the vocabulary."""

    backend_name = "stdlib"

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 context_window: int, seed: str = "model") -> None:
        self.V = vocab_size
        self.d = embed_dim
        self.h = hidden_dim
        self.k = context_window
        self.in_dim = context_window * embed_dim

        rng = DeterministicRNG(seed)
        # Scaled so the initial logits are near zero and step-0 loss lands at
        # ln(V), which is the sanity check the whole run is anchored on.
        se = 1.0 / math.sqrt(embed_dim)
        s1 = 1.0 / math.sqrt(self.in_dim)
        s2 = 1.0 / math.sqrt(hidden_dim)
        self.E = [[rng.gauss(0.0, se) for _ in range(self.d)] for _ in range(self.V)]
        self.W1 = [[rng.gauss(0.0, s1) for _ in range(self.in_dim)] for _ in range(self.h)]
        self.b1 = [0.0] * self.h
        self.W2 = [[rng.gauss(0.0, s2) for _ in range(self.h)] for _ in range(self.V)]
        self.b2 = [0.0] * self.V

        self._zero_grads()
        self.mE = None      # momentum buffers, allocated lazily on first step
        self.mW1 = self.mb1 = self.mW2 = self.mb2 = None

    # -- gradients ---------------------------------------------------------

    def _zero_grads(self) -> None:
        self.gE: dict[int, list[float]] = {}          # sparse: only touched rows
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.h)]
        self.gb1 = [0.0] * self.h
        self.gW2 = [[0.0] * self.h for _ in range(self.V)]
        self.gb2 = [0.0] * self.V
        self.n_accum = 0

    def zero_grad(self) -> None:
        self._zero_grads()

    # -- forward / backward ------------------------------------------------

    def build_context(self, tokens: list[int], segment_ids: list[int], i: int) -> list[int]:
        """The k tokens before position ``i``, with cross-document slots masked.

        Returns token ids, using ``-1`` for a masked slot. This is where the
        attention mask actually bites: a slot whose segment differs from
        position ``i``'s segment contributes nothing to the input vector.
        """
        seg_i = segment_ids[i]
        ctx = []
        for off in range(self.k, 0, -1):
            j = i - off
            if j < 0 or segment_ids[j] == 0 or segment_ids[j] != seg_i or tokens[j] == PAD_ID:
                ctx.append(-1)          # masked: contributes a zero vector
            else:
                ctx.append(tokens[j])
        return ctx

    def _embed(self, ctx: list[int]) -> list[float]:
        x: list[float] = []
        for t in ctx:
            if t < 0:
                x.extend([0.0] * self.d)
            else:
                x.extend(self.E[t])
        return x

    def forward_loss(self, ctx: list[int], target: int) -> tuple[float, dict]:
        """One position. Returns ``(loss, cache)``; cache feeds ``backward``."""
        x = self._embed(ctx)
        W1, b1 = self.W1, self.b1
        tanh = math.tanh
        hid = [tanh(sum(map(mul, W1[j], x)) + b1[j]) for j in range(self.h)]

        W2, b2 = self.W2, self.b2
        logits = [sum(map(mul, W2[o], hid)) + b2[o] for o in range(self.V)]

        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        Z = sum(exps)
        loss = -(logits[target] - m - math.log(Z))
        return loss, {"ctx": ctx, "x": x, "hid": hid, "exps": exps, "Z": Z,
                      "target": target}

    def backward(self, cache: dict) -> None:
        """Accumulate gradients for one position. Analytic, no autodiff."""
        x, hid, exps, Z = cache["x"], cache["hid"], cache["exps"], cache["Z"]
        target, ctx = cache["target"], cache["ctx"]
        invZ = 1.0 / Z
        gW2, gb2 = self.gW2, self.gb2
        W2 = self.W2
        dh = [0.0] * self.h

        for o in range(self.V):
            g = exps[o] * invZ
            if o == target:
                g -= 1.0
            if -1e-9 < g < 1e-9:
                continue
            row_g, row_w = gW2[o], W2[o]
            for j in range(self.h):
                row_g[j] += g * hid[j]
                dh[j] += g * row_w[j]
            gb2[o] += g

        gW1, gb1 = self.gW1, self.gb1
        dx = [0.0] * self.in_dim
        for j in range(self.h):
            gj = dh[j] * (1.0 - hid[j] * hid[j])      # tanh'
            if -1e-12 < gj < 1e-12:
                continue
            row_g, row_w = gW1[j], self.W1[j]
            for i in range(self.in_dim):
                row_g[i] += gj * x[i]
                dx[i] += gj * row_w[i]
            gb1[j] += gj

        # Embedding rows are sparse: only the context tokens were used.
        d = self.d
        for slot, t in enumerate(ctx):
            if t < 0:
                continue
            row = self.gE.get(t)
            if row is None:
                row = self.gE[t] = [0.0] * d
            base = slot * d
            for c in range(d):
                row[c] += dx[base + c]

        self.n_accum += 1

    # -- optimisation ------------------------------------------------------

    def grad_norm(self) -> float:
        """Global L2 norm over every accumulated gradient. Real, not estimated."""
        total = 0.0
        for row in self.gW1:
            total += sum(v * v for v in row)
        total += sum(v * v for v in self.gb1)
        for row in self.gW2:
            total += sum(v * v for v in row)
        total += sum(v * v for v in self.gb2)
        for row in self.gE.values():
            total += sum(v * v for v in row)
        return math.sqrt(total)

    def step(self, lr: float, momentum: float = 0.9,
             clip: float = 1.0) -> dict:
        """SGD with momentum and global-norm clipping.

        Momentum is real optimiser state and is saved in the checkpoint --
        resuming without it produces a visible loss spike while the moments are
        re-estimated, which is exactly the failure the checkpoint contract
        exists to prevent.
        """
        n = max(1, self.n_accum)
        gn = self.grad_norm() / n
        scale = 1.0 / n
        if clip and gn > clip:
            scale *= clip / gn

        if self.mW1 is None:
            self.mE = {}
            self.mW1 = [[0.0] * self.in_dim for _ in range(self.h)]
            self.mb1 = [0.0] * self.h
            self.mW2 = [[0.0] * self.h for _ in range(self.V)]
            self.mb2 = [0.0] * self.V

        for j in range(self.h):
            gr, mr, wr = self.gW1[j], self.mW1[j], self.W1[j]
            for i in range(self.in_dim):
                mr[i] = momentum * mr[i] + gr[i] * scale
                wr[i] -= lr * mr[i]
            self.mb1[j] = momentum * self.mb1[j] + self.gb1[j] * scale
            self.b1[j] -= lr * self.mb1[j]

        for o in range(self.V):
            gr = self.gW2[o]
            if not any(gr):
                continue
            mr, wr = self.mW2[o], self.W2[o]
            for j in range(self.h):
                mr[j] = momentum * mr[j] + gr[j] * scale
                wr[j] -= lr * mr[j]
        for o in range(self.V):
            if self.gb2[o]:
                self.mb2[o] = momentum * self.mb2[o] + self.gb2[o] * scale
                self.b2[o] -= lr * self.mb2[o]

        for t, gr in self.gE.items():
            mr = self.mE.get(t)
            if mr is None:
                mr = self.mE[t] = [0.0] * self.d
            wr = self.E[t]
            for c in range(self.d):
                mr[c] = momentum * mr[c] + gr[c] * scale
                wr[c] -= lr * mr[c]

        out = {"grad_norm": gn, "clipped": bool(clip and gn > clip),
               "positions": self.n_accum, "lr": lr}
        self._zero_grads()
        return out

    # -- gradient direction (for OPUS) -------------------------------------

    def grad_vector(self, dims: int = 256) -> list[float]:
        """A fixed-length projection of the current gradient.

        OPUS compares a candidate batch's gradient against the proxy direction.
        Comparing full gradients would mean carrying a vector the size of the
        model; projecting onto a fixed set of coordinates keeps the cosine
        meaningful while keeping the ledger small. The coordinates are chosen
        deterministically, so two runs project onto the same subspace.
        """
        v = [0.0] * dims
        idx = 0
        for row in self.gW2:
            for g in row:
                if g:
                    v[idx % dims] += g
                idx += 1
        for row in self.gW1:
            for g in row:
                if g:
                    v[idx % dims] += g
                idx += 1
        for t in sorted(self.gE):
            for g in self.gE[t]:
                if g:
                    v[idx % dims] += g
                idx += 1
        return v

    # -- the LanguageModel seam --------------------------------------------

    def loss_batch(self, samples: list[dict], *, backward: bool = True,
                   collect_tokens: bool = True) -> list[dict]:
        """Score each sample's loss-bearing positions. See :mod:`tdes.lm`.

        This model's natural unit is one position, so "batching" here is a loop.
        It lives on the model rather than in the trainer because the transformer
        backend's natural unit is a padded rectangle of sequences, and the
        trainer must not have to know which it is talking to.

        The iteration order -- samples in the order given, then positions
        ascending -- is part of the contract in :mod:`tdes.lm`: float addition
        is not associative, so reordering would change every loss in the
        learning ledger.
        """
        out: list[dict] = []
        for sample in samples:
            tokens = sample["tokens"]
            lm = sample["loss_mask"]
            seg = sample["segment_ids"]
            total = 0.0
            n = 0
            per_token: list[dict] = []

            for i in range(len(tokens) - 1):
                if not lm[i]:
                    continue
                target = tokens[i + 1]
                if target == PAD_ID:
                    continue
                ctx = self.build_context(tokens, seg, i + 1)
                loss, cache = self.forward_loss(ctx, target)
                if backward:
                    self.backward(cache)
                total += loss
                n += 1
                if collect_tokens:
                    per_token.append({"pos": i + 1, "token_id": target,
                                      "loss": loss})

            out.append(SampleLoss(sum_loss=total, n_tokens=n,
                                  mean_loss=(total / n) if n else 0.0,
                                  per_token=per_token))
        return out

    # -- checkpointing -----------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "config": {"V": self.V, "d": self.d, "h": self.h, "k": self.k},
            "E": self.E, "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
            "momentum": None if self.mW1 is None else {
                "E": {str(k): v for k, v in sorted(self.mE.items())},
                "W1": self.mW1, "b1": self.mb1, "W2": self.mW2, "b2": self.mb2,
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        c = sd["config"]
        if (c["V"], c["d"], c["h"], c["k"]) != (self.V, self.d, self.h, self.k):
            raise ValueError(f"checkpoint shape {c} does not match this model")
        self.E = [list(r) for r in sd["E"]]
        self.W1 = [list(r) for r in sd["W1"]]
        self.b1 = list(sd["b1"])
        self.W2 = [list(r) for r in sd["W2"]]
        self.b2 = list(sd["b2"])
        m = sd.get("momentum")
        if m:
            self.mE = {int(k): list(v) for k, v in m["E"].items()}
            self.mW1 = [list(r) for r in m["W1"]]
            self.mb1 = list(m["b1"])
            self.mW2 = [list(r) for r in m["W2"]]
            self.mb2 = list(m["b2"])
        else:
            self.mE = self.mW1 = self.mb1 = self.mW2 = self.mb2 = None
        self._zero_grads()

    @property
    def parameters_count(self) -> int:
        """Trainable scalars. Reported so the two backends can be compared on the
        same axis rather than one of them showing 'n/a'."""
        return (self.V * self.d                      # embeddings
                + self.h * self.in_dim + self.h      # W1, b1
                + self.V * self.h + self.V)          # W2, b2

    def initial_loss(self) -> float:
        """ln(V): the loss a model with no information must produce.

        Uniform probability 1/V over the vocabulary gives -ln(1/V) = ln(V). It
        depends only on vocabulary size -- not on the data, not on the
        architecture -- which makes it the single most useful sanity check in
        the run. A step-0 loss that is not close to this means the loss, the
        masking, or the label alignment is wrong.
        """
        return math.log(self.V)
