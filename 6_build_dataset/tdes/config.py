# -*- coding: utf-8 -*-
"""Run configuration: profiles, lanes, stages, floors, cost constants.

Every number here is either measured on this machine, carried forward from an
earlier stage of this project with a citation, or quoted from the course lecture.
Nothing is invented. The provenance tag on each block says which.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS_DIR = os.path.join(ROOT, "corpus")
DEFAULT_OUT = os.path.join(ROOT, "submission_artifacts")

DATALOADER_VERSION = "tdes/1.0.0"


# ---------------------------------------------------------------------------
# Profiles
#
# MEASURED on this machine (Python 3.14.5, pure-Python NPLM fwd+bwd, optimised
# inner loop). Softmax dominates, so throughput roughly halves per doubling of
# vocab:
#
#     V=256  -> 689 tok/s      V=1024 -> 141 tok/s
#     V=512  -> 284 tok/s      V=2048 ->  79 tok/s
#
# The profiles are sized from those numbers, not from an estimate.
#
# Vocabulary floor: byte-mode BPE has 256 byte symbols plus an end-of-word
# marker, so with 7 specials no profile can request fewer than 264 tokens.
# MEASURED alternatives on this corpus: codepoint mode needs 320 base symbols,
# akshara mode needs 2,118 -- which is why byte mode is the default.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    name: str
    vocab_size: int
    embed_dim: int          # d
    hidden_dim: int         # h
    context_window: int     # k -- how many previous tokens the model conditions on
    seq_len_early: int      # phases A/B
    seq_len_late: int       # phases C/D  (the sequence ladder)
    ranks: int
    microbatch: int
    grad_accum: int
    base_steps: int
    fork_steps: int
    checkpoint_every: int
    crash_at_step: int
    replay_from: int
    replay_to: int
    fork_from_step: int
    opus_accept_ratio: float
    opus_prefix_tokens: int          # the Admin's "only send the first 512"
    prefetch_depth: int
    cache_capacity: int
    loader_workers: int
    docs_per_lane_cap: int
    expected_tok_per_s: float        # measured; used only for the time budget note
    fertility_sweep_sizes: tuple[int, ...] = ()

    # -- backend ----------------------------------------------------------
    # "torch" selects the transformer in tdes/model_torch.py and is what the
    # default profile uses. "stdlib" needs no dependencies at all and is the
    # fallback. The fields below are ignored by the stdlib backend, and
    # embed_dim/hidden_dim/context_window are ignored by the torch one.
    backend: str = "stdlib"
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    dropout: float = 0.0             # refused if nonzero; see model_torch.py
    device: str = "auto"
    amp: bool = True

    @property
    def samples_per_step(self) -> int:
        return self.ranks * self.microbatch * self.grad_accum

    def tokens_per_step(self, seq_len: int) -> int:
        return self.samples_per_step * seq_len


PROFILES: dict[str, Profile] = {
    # CI / unit tests. Must finish well under a minute.
    "fast": Profile(
        name="fast",
        vocab_size=320, embed_dim=8, hidden_dim=32, context_window=4,
        seq_len_early=64, seq_len_late=128,
        ranks=2, microbatch=1, grad_accum=2,
        base_steps=30, fork_steps=8, checkpoint_every=6,
        crash_at_step=21, replay_from=6, replay_to=12, fork_from_step=12,
        opus_accept_ratio=0.5, opus_prefix_tokens=16,
        prefetch_depth=2, cache_capacity=4, loader_workers=2,
        docs_per_lane_cap=120,
        expected_tok_per_s=560.0,
        fertility_sweep_sizes=(320, 1024),
    ),
    # The dependency-free path, and the control for the backend-equivalence
    # check: same data plane as `torch`, driven by the stdlib model. Used when
    # PyTorch is unavailable.
    "demo": Profile(
        name="demo",
        vocab_size=512, embed_dim=8, hidden_dim=32, context_window=4,
        seq_len_early=64, seq_len_late=128,
        ranks=2, microbatch=3, grad_accum=1,
        base_steps=100, fork_steps=20, checkpoint_every=10,
        crash_at_step=68, replay_from=20, replay_to=40, fork_from_step=40,
        opus_accept_ratio=0.4, opus_prefix_tokens=32,
        prefetch_depth=3, cache_capacity=8, loader_workers=3,
        docs_per_lane_cap=400,
        expected_tok_per_s=284.0,
        fertility_sweep_sizes=(512, 1024, 2048),
    ),
    # Longer run for a fuller learning curve. Not the submission path.
    #
    # MEASURED, not projected: construction 34s, then 28.1s per step at
    # 16 samples x 256 tokens. 300 base + 50 fork steps is therefore roughly
    # 2.7 hours. An earlier comment here guessed "~85 min", which was wrong by
    # a factor of two -- the figure below is from an actual step.
    "full": Profile(
        name="full",
        vocab_size=1024, embed_dim=16, hidden_dim=48, context_window=6,
        seq_len_early=128, seq_len_late=256,
        ranks=2, microbatch=4, grad_accum=2,
        base_steps=300, fork_steps=50, checkpoint_every=25,
        crash_at_step=205, replay_from=50, replay_to=100, fork_from_step=100,
        opus_accept_ratio=0.35, opus_prefix_tokens=64,
        prefetch_depth=4, cache_capacity=16, loader_workers=4,
        docs_per_lane_cap=2000,
        expected_tok_per_s=141.0,
        fertility_sweep_sizes=(512, 1024, 2048, 4096),
    ),
    # THE DEFAULT. A real decoder-only transformer on a GPU. It needs PyTorch
    # (requirements-torch.txt); `--profile demo` is the fallback that needs
    # nothing. Everything the assignment grades is identical between the two --
    # only the model differs, which is the point of the equivalence check.
    #
    # Its weight blobs are 124MB each and are gitignored; the checkpoint
    # envelopes, manifests and ledgers are committed, so the recovery evidence is
    # complete without them.
    #
    # Sized for an RTX 3070 Laptop (8GB, sm_86). Activations are bounded by
    # `microbatch`, not by samples_per_step, because TorchLM chunks internally.
    #
    # Vocabulary 8,192 rather than 512: a transformer makes a real vocabulary
    # nearly free, and Indic fertility at 512 is dominated by the vocabulary
    # being tiny rather than by the script. The stdlib profiles stay at 512
    # because MEASURED throughput there is 79 tok/s at V=2048 -- an 8k stdlib run
    # would take hours.
    #
    # The corpus is ~207k tokens at V=512 and fewer at 8k, so 240 steps x 4-8k
    # tokens demands several times the supply. That is deliberate: the 4.0 epoch
    # cap and the scarcity policies finally bind instead of merely being
    # reported. `scarcity.py` resolves this at planning time, so nothing starves.
    "torch": Profile(
        name="torch",
        vocab_size=8192, embed_dim=0, hidden_dim=0, context_window=0,
        seq_len_early=256, seq_len_late=512,
        ranks=2, microbatch=4, grad_accum=2,          # 16 samples/step
        base_steps=240, fork_steps=40, checkpoint_every=20,
        crash_at_step=158, replay_from=40, replay_to=80, fork_from_step=80,
        opus_accept_ratio=0.4, opus_prefix_tokens=64,
        prefetch_depth=4, cache_capacity=16, loader_workers=3,
        docs_per_lane_cap=2000,
        expected_tok_per_s=0.0,          # measured at run time; see performance.json
        fertility_sweep_sizes=(512, 2048, 8192),
        backend="torch",
        n_layers=6, n_heads=6, d_model=384,
        dropout=0.0, device="auto", amp=True,
    ),
}


# ---------------------------------------------------------------------------
# Special tokens. Ids are fixed and reserved before BPE training so a retrained
# tokenizer cannot silently move them.
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>",
                  "<tool_call>", "<tool_obs>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, SEP_ID, TOOL_CALL_ID, TOOL_OBS_ID, UNK_ID = range(7)
N_SPECIAL = len(SPECIAL_TOKENS)


# ---------------------------------------------------------------------------
# Curriculum. Carried from Session 5 (5_data_mixture_curriculum/mixture_results.json).
# Phase weights and protected floors are that file's real values; only the token
# budget is scaled down to demo size.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage:
    stage_id: str
    label: str
    weight: float                    # share of total budget
    sequence_length: int
    mixture: dict[str, float]        # lane -> share, sums to 1.0
    protected_floors: dict[str, float]
    warmup_frac: float               # crossfade band, as a fraction of stage length
    anneal_reserve_only: bool = False


# Lane shares per phase come from the `phases` column of Session 5's lane table
# (percentages per phase), renormalised to 1.0 over the lanes we actually hold
# data for. Lanes from Session 5 with no corpus here (civic, parallel) are
# folded into `web`; that substitution is recorded in the mixture report.
STAGES: list[Stage] = [
    Stage(
        stage_id="A", label="Foundation", weight=0.40, sequence_length=0,
        mixture={"web": 0.62, "code": 0.16, "indic": 0.08, "multiling": 0.05,
                 "reasoning": 0.05, "agentic": 0.02, "longctx": 0.02},
        protected_floors={"indic": 0.03, "longctx": 0.015, "reasoning": 0.01, "agentic": 0.01},
        warmup_frac=0.15,
    ),
    Stage(
        stage_id="B", label="Consolidation", weight=0.32, sequence_length=0,
        mixture={"web": 0.50, "code": 0.24, "indic": 0.08, "multiling": 0.04,
                 "reasoning": 0.08, "agentic": 0.03, "longctx": 0.03},
        protected_floors={"indic": 0.03, "longctx": 0.015, "reasoning": 0.01, "agentic": 0.01},
        warmup_frac=0.15,
    ),
    Stage(
        stage_id="C", label="Specialisation", weight=0.25, sequence_length=1,
        mixture={"web": 0.34, "code": 0.32, "indic": 0.09, "multiling": 0.02,
                 "reasoning": 0.13, "agentic": 0.05, "longctx": 0.05},
        protected_floors={"indic": 0.03, "longctx": 0.015, "reasoning": 0.01, "agentic": 0.01},
        warmup_frac=0.15,
    ),
    Stage(
        stage_id="D", label="Anneal", weight=0.03, sequence_length=1,
        mixture={"web": 0.16, "code": 0.22, "indic": 0.12, "multiling": 0.0,
                 "reasoning": 0.26, "agentic": 0.16, "longctx": 0.08},
        protected_floors={"indic": 0.03, "longctx": 0.015, "reasoning": 0.01, "agentic": 0.01},
        warmup_frac=0.10,
        anneal_reserve_only=True,
    ),
]

LANES = ["web", "code", "indic", "multiling", "reasoning", "agentic", "longctx"]

# Which packing policy each lane uses, and why. Lecture section 5.
LANE_PACKING = {
    "web":        "concat_chop",           # plain text tolerates mechanical cuts
    "multiling":  "concat_chop",
    "indic":      "greedy",                # keep whole docs where cheap; fertility already costs us
    "code":       "best_fit",              # never split a file that fits
    "reasoning":  "structure_preserving",  # a trace must not be merged with another
    "agentic":    "structure_preserving",
    "longctx":    "long_context",          # only docs >= seq_len, reserved for the late rung
}

# Fraction of tokens in a lane that bear loss. Session 5's `loss_bearing_frac`.
# agentic is low because tool observations are context, never target.
LANE_LOSS_BEARING = {
    "web": 1.0, "code": 1.0, "indic": 1.0, "multiling": 1.0,
    "reasoning": 0.85, "agentic": 0.30, "longctx": 1.0,
}

# Session 5, spec.py: EPOCH_CAP_DEFAULT = 4.0, from Muennighoff et al.,
# "Scaling Data-Constrained Language Models", NeurIPS 2023 (arXiv:2305.16264).
EPOCH_CAP = 4.0
EPOCH_DECAY_R_STAR = 15.387   # fitted R_D* from the same paper, eq. 6

# Session 5: unverified Indic may never substitute for the verified portion of
# the floor. Half the indic floor must come from tier=verified.
INDIC_VERIFIED_FLOOR_FRACTION = 0.5

# Fraction of the total budget held back and released only in the anneal stage.
ANNEAL_RESERVE_FRACTION = 0.03


# ---------------------------------------------------------------------------
# Learning-ledger thresholds. Quoted from the course lecture.
# ---------------------------------------------------------------------------

USEFULNESS_THRESHOLDS = {
    "broken_below": 0.5,        # ~0.3 => boilerplate / duplication / leak
    "already_learned_below": 1.2,  # the Admin's explicit cutoff
    "neutral_below": 2.0,       # ">= 2 and above, it is fine"
    # >= 2.0 is useful; ~3.4 was called a healthy mid-training shard
}


# ---------------------------------------------------------------------------
# Cost constants. From the course lecture: an AWS p4de.24xlarge (8x A100 80GB,
# 96 vCPU, ~1TB RAM) quoted at about Rs 2,700/hour.
# ---------------------------------------------------------------------------

COST = {
    "instance": "AWS p4de.24xlarge (8x A100 80GB)",
    "inr_per_hour": 2700.0,
    "gpus": 8,
    "source": "course lecture, 00:29-00:31",
}


@dataclass
class RunConfig:
    profile: Profile
    seed: str = "tdes-v1"
    run_id: str = "run-0001"
    out_dir: str = DEFAULT_OUT
    corpus_dir: str = CORPUS_DIR
    position_policy_early: str = "reset_per_document"
    position_policy_late: str = "continuous"
    attention_policy: str = "document_causal"
    extra: dict[str, Any] = field(default_factory=dict)

    def seq_len_for(self, stage: Stage) -> int:
        return (self.profile.seq_len_early if stage.sequence_length == 0
                else self.profile.seq_len_late)

    def position_policy_for(self, stage: Stage) -> str:
        return (self.position_policy_early if stage.sequence_length == 0
                else self.position_policy_late)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["profile"] = asdict(self.profile)
        return d


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    return PROFILES[name]
