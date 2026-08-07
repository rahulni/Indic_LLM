# -*- coding: utf-8 -*-
"""The model boundary: what the training loop is allowed to assume.

`ARCHITECTURE.md` claimed for a while that the model "sits behind a
``LanguageModel`` protocol so a torch transformer can be dropped in without the
ledger noticing". That was not true -- there was no protocol, and
:mod:`tdes.trainer` drove a *per-position* API (``build_context`` ->
``forward_loss`` -> ``backward``, once per loss-bearing token). A transformer
computes a whole sequence in one forward, so that shape could not have accepted
one. This module is the seam that makes the claim true, and it is written down
here rather than implied.

**The one design decision that matters: batching belongs to the model.**

Batching is precisely what differs between the two implementations. The n-gram
model scores one position at a time because that is its natural unit; a
transformer scores a padded rectangle of sequences because that is its natural
unit. If the trainer owned the loop, every backend would have to pretend to be
the other one. So :meth:`LanguageModel.loss_batch` takes a *list* of samples and
returns one result per sample, and each implementation batches however it likes.

The return shape is deliberately unchanged from what ``score_sample`` returned
before the seam existed, so the trainer, the learning ledger and every record
written to disk are untouched by which backend produced them::

    {"sum_loss": float, "n_tokens": int, "mean_loss": float,
     "per_token": [{"pos": int, "token_id": int, "loss": float}, ...]}

Two rules an implementation must honour, because the ledgers are built on them:

1. **Order is part of the contract.** Gradients must accumulate in the order the
   samples were given, and ``per_token`` must be in increasing ``pos``. Float
   addition is not associative, so a backend that reorders silently changes
   every loss in the learning ledger.
2. **Only ``loss_mask`` positions count.** A position that is read as context --
   a prompt, a tool observation, padding -- contributes to no loss and to no
   gradient. That distinction is the difference between the model *seeing* a
   token and being trained to *produce* it.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class SampleLoss(dict):
    """The per-sample result. A ``dict`` subclass so existing code that indexes
    it keeps working; the class exists to give the shape a name."""

    __slots__ = ()


@runtime_checkable
class LanguageModel(Protocol):
    """What :mod:`tdes.trainer`, :mod:`tdes.opus` and :mod:`tdes.checkpoint` may
    assume about a model. Nothing else is permitted to touch model internals."""

    def loss_batch(self, samples: list[dict], *, backward: bool = True,
                   collect_tokens: bool = True) -> list[SampleLoss]:
        """Forward (and optionally backward) over each sample's scored positions.

        Returns one result per input sample, in input order. When ``backward`` is
        true, gradients accumulate into the model until :meth:`zero_grad`.
        """
        ...

    def zero_grad(self) -> None:
        """Discard accumulated gradients."""
        ...

    def step(self, lr: float, *, momentum: float = 0.9,
             clip: float = 1.0) -> dict:
        """Apply one optimiser step. Returns ``{"grad_norm", "clipped"}``."""
        ...

    def grad_vector(self, dims: int = 256) -> list[float]:
        """A fixed-length projection of the current gradient, for OPUS.

        Both backends must fold gradients into ``dims`` buckets the same way, or
        an ``opus_score`` from one backend would not be comparable with the
        other's.
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to restore the model, including optimiser moments."""
        ...

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        ...

    def initial_loss(self) -> float:
        """``ln(V)``: what an uninformed model must produce. The run's anchor."""
        ...


def build(cfg, vocab_size: int, *, total_steps: int = 0) -> Any:
    """Construct the backend the profile asks for.

    The only place either model is instantiated, so nothing else needs to know
    which exists. ``torch`` is imported lazily and only when asked for -- the
    stdlib path must keep working on a machine with no torch installed, which is
    the whole point of keeping two backends.
    """
    P = cfg.profile
    backend = getattr(P, "backend", "stdlib")
    if backend == "stdlib":
        from .model import NeuralLM
        return NeuralLM(vocab_size, P.embed_dim, P.hidden_dim,
                        P.context_window, seed=cfg.seed)
    if backend == "torch":
        try:
            from .model_torch import TorchLM, configure_determinism
        except ImportError as e:                        # pragma: no cover
            raise RuntimeError(
                "--backend torch needs PyTorch: pip install -r "
                "requirements-torch.txt (the stdlib backend needs nothing)"
            ) from e
        # Seeded from the run seed so the weights are a function of the run, not
        # of import order.
        configure_determinism(abs(hash_seed(cfg.seed)) % (2 ** 31))
        return TorchLM(vocab_size, d_model=P.d_model, n_layers=P.n_layers,
                       n_heads=P.n_heads, seq_len=max(P.seq_len_early,
                                                      P.seq_len_late),
                       dropout=P.dropout, device=P.device, amp=P.amp,
                       # Activations stay bounded by one microbatch even though
                       # the trainer hands over a whole step.
                       max_batch=P.microbatch,
                       seed=abs(hash_seed(cfg.seed)) % (2 ** 31))
    raise ValueError(f"unknown backend {backend!r}")


def hash_seed(seed: str) -> int:
    """A stable integer from the run seed.

    Not :func:`hash` -- that is randomised per process, which would make the
    initial weights differ between runs. :mod:`tdes.determinism` bans it for
    exactly this reason.
    """
    from .hashing import sha256_text
    return int(sha256_text(str(seed))[:8], 16)


def describe(model: Any) -> dict:
    """Backend identity for ``run_meta.json`` and the manifests.

    Recorded because a loss curve is only interpretable next to the thing that
    produced it, and because two backends must never be mistaken for one run.
    """
    d = {"backend": getattr(model, "backend_name", "unknown"),
         "class": type(model).__name__}
    # Note "parameters_count", not "parameters": on an nn.Module the latter is a
    # bound method, which would land an unserialisable object in the manifest.
    for k in ("V", "d", "h", "k", "n_layers", "n_heads", "d_model",
              "seq_len", "device", "amp_dtype", "parameters_count"):
        v = getattr(model, k, None)
        if v is not None:
            d[k] = v
    # Which accelerator produced the numbers, and with which library. A
    # throughput figure is meaningless without it, and "cuda" alone does not
    # distinguish a laptop 3070 from an H100.
    hw = getattr(model, "hardware_report", None)
    if callable(hw):
        d.update(hw())
    return d
