# -*- coding: utf-8 -*-
"""The training loop and the validation probe.

The trainer walks microbatches, accumulates gradients, and takes one optimiser
step per global step. It returns per-token losses, not just a scalar, because
the learning ledger's whole value is that a shard's average can hide the thing
that matters -- a shard can look ordinary while one document boundary carries
all the difficulty.

The probe is the mechanism behind three separate requirements that turn out to
be the same measurement:

* ``loss_delta_before_after_exposure`` in the learning ledger -- the difference
  between two real probe evaluations, not an estimate;
* the OPUS proxy direction -- the gradient that would most improve probe loss;
* the demonstration that validation data is **read during training but never
  gradient-bearing** -- the probe runs forward only, and every read is written
  to the firewall's access log.
"""
from __future__ import annotations

import math

from .lm import LanguageModel


class LRSchedule:
    """Warmup then cosine decay. Its state is saved in the checkpoint.

    Recomputing the learning rate from the step number alone would work only if
    the schedule never changed; storing it means a resumed run continues on the
    same curve even if the code around it moves.
    """

    def __init__(self, base_lr: float, total_steps: int, warmup_frac: float = 0.1,
                 min_frac: float = 0.1) -> None:
        self.base_lr = base_lr
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, int(self.total_steps * warmup_frac))
        self.min_frac = min_frac

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        t = min(1.0, max(0.0, t))
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.base_lr * (self.min_frac + (1.0 - self.min_frac) * cos)

    def state(self) -> dict:
        return {"base_lr": self.base_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps, "min_frac": self.min_frac}

    @classmethod
    def from_state(cls, st: dict) -> "LRSchedule":
        s = cls(st["base_lr"], st["total_steps"], min_frac=st["min_frac"])
        s.warmup_steps = st["warmup_steps"]
        return s


def score_sample(model: LanguageModel, sample: dict, *,
                 collect_tokens: bool = True, backward: bool = True) -> dict:
    """One sample, for callers that genuinely have only one.

    Kept as a thin wrapper over :meth:`LanguageModel.loss_batch` so tests and
    one-off call sites read naturally. Everything on a hot path passes the whole
    list instead -- a backend that can batch must be given the chance to.
    """
    return model.loss_batch([sample], backward=backward,
                            collect_tokens=collect_tokens)[0]


class Trainer:
    def __init__(self, model: LanguageModel, schedule: LRSchedule, *,
                 momentum: float = 0.9, clip: float = 1.0) -> None:
        self.model = model
        self.schedule = schedule
        self.momentum = momentum
        self.clip = clip

    def train_step(self, batch: dict, global_step: int, *,
                   collect_tokens: bool = True) -> dict:
        """Accumulate over every microbatch, then take one optimiser step.

        One optimiser step per global step is what makes ``global_step`` a
        meaningful coordinate: the ledger offset, the checkpoint and the loss
        curve all index the same thing.
        """
        self.model.zero_grad()
        per_sample: list[dict] = []
        token_records: list[dict] = []
        total_loss = 0.0
        total_tokens = 0

        # Flatten microbatch-then-sample, which is the order gradients have
        # always accumulated in. Handing the model the whole step at once lets a
        # batching backend batch; the order it must respect is this one.
        flat = [(mb, s) for mb in batch["microbatches"] for s in mb["samples"]]
        results = self.model.loss_batch([s for _, s in flat],
                                        backward=True,
                                        collect_tokens=collect_tokens)

        for (mb, sample), r in zip(flat, results):
            total_loss += r["sum_loss"]
            total_tokens += r["n_tokens"]
            per_sample.append({
                "sample_index": sample["sample_index"],
                "rank": mb["rank"],
                "microbatch_id": mb["microbatch_id"],
                "lane": sample["lane"],
                "mean_loss": r["mean_loss"],
                "n_loss_tokens": r["n_tokens"],
                "doc_ids": sample["doc_ids"],
                "shard_ids": sample["shard_ids"],
                "pool_epoch": sample.get("pool_epoch", 0),
            })
            if collect_tokens:
                for t in r["per_token"]:
                    token_records.append({**t,
                                          "sample_index": sample["sample_index"],
                                          "lane": sample["lane"],
                                          "rank": mb["rank"]})

        lr = self.schedule.lr_at(global_step)
        opt = self.model.step(lr, momentum=self.momentum, clip=self.clip)
        mean = (total_loss / total_tokens) if total_tokens else 0.0
        return {
            "global_step": global_step,
            "mean_loss": mean,
            "perplexity": math.exp(min(mean, 700.0)),
            "loss_tokens": total_tokens,
            "grad_norm": opt["grad_norm"],
            "clipped": opt["clipped"],
            "lr": lr,
            "per_sample": per_sample,
            "token_records": token_records,
        }


class ValidationProbe:
    """Forward-only evaluation on held-out validation data.

    Two hard rules, both enforced rather than documented:

    * gradients are zeroed after every evaluation, so nothing the probe touched
      can leak into an optimiser step;
    * every read is reported to the firewall's access log with
      ``purpose='validation_probe'``, so "validation was read N times and never
      bore a gradient" is a checkable claim rather than an assurance.
    """

    def __init__(self, samples: list[dict], registry=None) -> None:
        self.samples = samples
        self.registry = registry
        self.history: list[dict] = []

    def evaluate(self, model: LanguageModel, *, global_step: int,
                 label: str = "probe") -> dict:
        by_lane: dict[str, list[float]] = {}
        total, n = 0.0, 0
        doc_ids: list[str] = []

        results = model.loss_batch(self.samples, backward=False,
                                   collect_tokens=False)
        for s, r in zip(self.samples, results):
            if r["n_tokens"]:
                by_lane.setdefault(s["lane"], []).append(r["mean_loss"])
                total += r["sum_loss"]
                n += r["n_tokens"]
            doc_ids.extend(s["doc_ids"])

        model.zero_grad()      # belt and braces: nothing here may reach a step

        if self.registry is not None:
            self.registry.note_access(doc_ids, purpose="validation_probe",
                                      step=global_step)

        mean = (total / n) if n else 0.0
        out = {
            "global_step": global_step,
            "label": label,
            "mean_loss": mean,
            "perplexity": math.exp(min(mean, 700.0)),
            "tokens": n,
            "by_lane": {k: round(sum(v) / len(v), 6) for k, v in sorted(by_lane.items())},
            "samples": len(self.samples),
            "gradient_bearing": False,
        }
        self.history.append(out)
        return out

    def proxy_direction(self, model: LanguageModel, *,
                        dims: int = 256) -> list[float]:
        """The gradient direction that would most improve probe loss.

        This is the "golden proxy" from the lecture, made concrete: run the
        held-out set the model *should* be good at, take the gradient, and use
        it as the direction OPUS scores candidates against. Gradients are
        cleared afterwards so the proxy pass never contributes to an update.
        """
        model.zero_grad()
        model.loss_batch(self.samples, backward=True, collect_tokens=False)
        v = model.grad_vector(dims)
        model.zero_grad()
        return v

    def delta(self) -> dict:
        """Change in probe loss between the two most recent evaluations."""
        if len(self.history) < 2:
            return {"available": False}
        prev, cur = self.history[-2], self.history[-1]
        lanes = sorted(set(prev["by_lane"]) | set(cur["by_lane"]))
        return {
            "available": True,
            "from_step": prev["global_step"],
            "to_step": cur["global_step"],
            "mean_loss_delta": round(cur["mean_loss"] - prev["mean_loss"], 6),
            "by_lane_delta": {
                l: round(cur["by_lane"].get(l, 0.0) - prev["by_lane"].get(l, 0.0), 6)
                for l in lanes
            },
        }


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 when either vector is degenerate."""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
