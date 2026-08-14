"""Two word-embedding schemes, built from characters, sharing everything
downstream (same transformer trunk, same learned softmax output head over
the closed word vocabulary). The only difference is how a word's characters
are composed into a fixed-size vector -- that composition is precomputed
ONCE per vocabulary word into a lookup table, since both schemes are built
from fixed (non-learned) per-character pieces.

  (a) Kronecker/tensor-product baseline: literal 32-slot cap, one-hot per
      position concatenated (V1's actual mechanism). Words over 32
      characters are truncated -- information is genuinely lost, not just
      theoretically at risk. This raw (n_words, 32*|alphabet|) table is
      large, so a LEARNED linear projection down to d_model is required;
      that projection is where most of this arm's extra parameters live.

  (b) Holographic/circular-convolution: fixed dimension d_model regardless
      of word length, built exactly as in proofs/capacity_proof.py (reusing
      that module's role/filler construction, not a reimplementation). No
      projection needed, and zero learned embedding parameters at all.

UNK (vocabulary id 0) is given a single fixed reserved vector in both
schemes (not built from any character) so it never collides with a real
word's encoding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from track_b_holographic_binding.proofs.capacity_proof import (  # noqa: E402
    build_fillers,
    build_roles,
)

KRONECKER_MAX_SLOTS = 32


def build_kronecker_slot_table(words: list[str], char_to_id: dict[str, int], max_slots: int = KRONECKER_MAX_SLOTS) -> np.ndarray:
    alphabet_size = len(char_to_id)
    table = np.zeros((len(words), max_slots * alphabet_size), dtype=np.float32)
    for wi, word in enumerate(words):
        if wi == 0:
            continue  # UNK: all-zero row, distinguishable from every real word
        for pos, ch in enumerate(word[:max_slots]):
            cid = char_to_id[ch]
            table[wi, pos * alphabet_size + cid] = 1.0
    return table


def build_holographic_table(
    words: list[str], char_to_id: dict[str, int], D: int, rng: np.random.Generator, max_len: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (table, roles, fillers) -- roles/fillers are returned
    alongside the table (not just used internally) so that other strings
    (e.g. evaluate.py's synthetic truncation-probe words, which are not
    vocabulary members) can be embedded with the IDENTICAL role/filler
    vectors later, rather than risking a seed mismatch by rebuilding them."""
    natural_max = max((len(w) for w in words), default=1)
    roles = build_roles(max(max_len or natural_max, natural_max), D, rng)
    fillers = build_fillers(len(char_to_id), D, rng)
    unk_vec = rng.normal(0, 1.0 / np.sqrt(D), size=D)  # fixed reserved vector, not built from characters

    table = np.zeros((len(words), D), dtype=np.float32)
    for wi, word in enumerate(words):
        if wi == 0:
            table[wi] = unk_vec
            continue
        fft_word = np.zeros(D, dtype=complex)
        for pos, ch in enumerate(word):
            cid = char_to_id[ch]
            fft_word += np.fft.fft(roles[pos]) * np.fft.fft(fillers[cid])
        table[wi] = np.fft.ifft(fft_word).real
    return table, roles, fillers


class KroneckerWordEmbedding(nn.Module):
    """Arm (a): fixed 32-slot one-hot lookup + a learned linear projection
    down to d_model. This is V1's own mechanism, used here as the baseline
    to beat."""

    def __init__(self, slot_table: np.ndarray, d_model: int):
        super().__init__()
        self.register_buffer("slot_table", torch.tensor(slot_table, dtype=torch.float32), persistent=False)
        self.projection = nn.Linear(slot_table.shape[1], d_model)

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        raw = self.slot_table[word_ids]
        return self.projection(raw)

    def raw_width(self) -> int:
        return self.slot_table.shape[1]


class HolographicWordEmbedding(nn.Module):
    """Arm (b): fixed-dimension lookup, no length cap, zero learned
    embedding parameters -- the table is entirely determined by the fixed
    role/filler vectors, exactly as proven in proofs/capacity_proof.py.

    `dress=True` adds a thin learned linear layer on top of the fixed table.
    This is the controlled ablation for the perplexity gap against the
    Kronecker arm: that arm gets a learned projection (from its 1,088-dim raw
    slot code) while the plain holographic arm gets none, so the comparison
    confounds "which code carries more information" with "which arm is allowed
    to adapt its code". Dressing the holographic table separates the two.
    It costs learned parameters, so it is always reported as its own arm."""

    def __init__(self, holo_table: np.ndarray, dress: bool = False):
        super().__init__()
        self.register_buffer("holo_table", torch.tensor(holo_table, dtype=torch.float32), persistent=False)
        d_model = holo_table.shape[1]
        self.dress = nn.Linear(d_model, d_model) if dress else None

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        out = self.holo_table[word_ids]
        if self.dress is not None:
            out = self.dress(out)
        return out


def build_embedding(
    arm: str,
    id_to_word: list[str],
    char_to_id: dict[str, int],
    d_model: int,
    seed: int,
    max_len: int = 64,
    dress: bool = False,
) -> tuple[nn.Module, dict]:
    """Returns (module, extra_materials). extra_materials is empty for the
    Kronecker arm; for the holographic arm it carries the exact roles/fillers
    used, so evaluate.py can embed out-of-vocabulary probe strings (e.g. the
    truncation-information-loss pairs) with the identical construction."""
    rng = np.random.default_rng(seed)
    if arm == "kronecker":
        table = build_kronecker_slot_table(id_to_word, char_to_id)
        return KroneckerWordEmbedding(table, d_model), {}
    if arm == "holographic":
        table, roles, fillers = build_holographic_table(id_to_word, char_to_id, d_model, rng, max_len=max_len)
        return HolographicWordEmbedding(table, dress=dress), {"roles": roles, "fillers": fillers}
    raise ValueError(f"unknown arm {arm!r}, choose from kronecker, holographic")
