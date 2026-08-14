"""Evaluation helpers for Track B's word-level LM comparison.

Two honestly-separated things are measured:

  1. Perplexity (in-distribution, on held-out natural corpus text), overall
     and bucketed by natural word length. The natural tiny_shakespeare
     corpus tops out at 15 characters (measured in data/corpus.py) -- far
     short of the 32-char cap -- so this alone CANNOT demonstrate a
     truncation cliff, and this module does not claim it does.

  2. A direct, controlled information-loss check: pairs of synthetic words
     that are IDENTICAL in their first 32 characters and differ only after
     position 32. The Kronecker/tensor-product arm truncates at 32 chars by
     construction, so it must embed such a pair identically (cosine
     similarity exactly 1.0) -- that is the truncation cliff, made
     measurable and unambiguous, independent of whether the natural corpus
     happens to contain long words.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(model, windows: np.ndarray, device, batch_size: int = 256) -> float:
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        input_ids = torch.from_numpy(chunk[:, :-1]).to(device)
        target_ids = torch.from_numpy(chunk[:, 1:]).to(device)
        logits = model(input_ids)
        B, T, V = logits.shape
        nll = F.cross_entropy(logits.reshape(B * T, V), target_ids.reshape(B * T), reduction="sum")
        total_nll += float(nll.item())
        total_tokens += B * T
    return float(np.exp(total_nll / total_tokens))


@torch.no_grad()
def perplexity_by_length_bucket(
    model, windows: np.ndarray, id_to_word: list[str], device, buckets: list[tuple[int, int]]
) -> dict:
    input_ids = torch.from_numpy(windows[:, :-1]).to(device)
    target_ids_np = windows[:, 1:]
    target_ids = torch.from_numpy(target_ids_np).to(device)
    logits = model(input_ids)
    B, T, V = logits.shape
    per_token_nll = F.cross_entropy(
        logits.reshape(B * T, V), target_ids.reshape(B * T), reduction="none"
    ).reshape(B, T)
    per_token_nll = per_token_nll.cpu().numpy()

    lengths = np.vectorize(lambda i: len(id_to_word[i]))(target_ids_np)

    results = {}
    for lo, hi in buckets:
        mask = (lengths >= lo) & (lengths <= hi)
        if mask.sum() == 0:
            results[f"{lo}-{hi}"] = None
            continue
        mean_nll = per_token_nll[mask].mean()
        results[f"{lo}-{hi}"] = {"perplexity": float(np.exp(mean_nll)), "n_tokens": int(mask.sum())}
    return results


def make_truncation_pairs(rng: np.random.Generator, alphabet: list[str], n_pairs: int = 100) -> list[dict]:
    """Pairs of strings sharing an identical 32-character prefix, differing
    only after it -- the minimal, controlled probe for information loss
    past the 32-char cap."""
    letters = [c for c in alphabet if c.isalpha()]
    pairs = []
    for _ in range(n_pairs):
        prefix = "".join(rng.choice(letters, size=32))
        suffix_len = int(rng.integers(1, 20))
        suffix_a = "".join(rng.choice(letters, size=suffix_len))
        suffix_b = "".join(rng.choice(letters, size=suffix_len))
        while suffix_b == suffix_a:
            suffix_b = "".join(rng.choice(letters, size=suffix_len))
        pairs.append({"word_a": prefix + suffix_a, "word_b": prefix + suffix_b})
    return pairs


def embed_word_kronecker(word: str, char_to_id: dict[str, int], projection: torch.nn.Module) -> np.ndarray:
    """Embeds a single arbitrary string directly through the Kronecker arm's
    construction (not via the precomputed vocab lookup table, since these
    probe words are not vocabulary members)."""
    from track_b_holographic_binding.model.embeddings import KRONECKER_MAX_SLOTS

    alphabet_size = len(char_to_id)
    vec = np.zeros(KRONECKER_MAX_SLOTS * alphabet_size, dtype=np.float32)
    for pos, ch in enumerate(word[:KRONECKER_MAX_SLOTS]):
        vec[pos * alphabet_size + char_to_id[ch]] = 1.0
    device = next(projection.parameters()).device
    with torch.no_grad():
        out = projection(torch.tensor(vec, dtype=torch.float32, device=device))
    return out.cpu().numpy()


def embed_word_holographic(word: str, char_to_id: dict[str, int], roles: np.ndarray, fillers: np.ndarray) -> np.ndarray:
    """Same construction as build_holographic_table, applied to a single
    arbitrary string -- `roles`/`fillers` must be the SAME arrays used to
    build the model's vocab table, so the two are directly comparable."""
    D = fillers.shape[1]
    spec = np.zeros(D, dtype=complex)
    for pos, ch in enumerate(word):
        cid = char_to_id[ch]
        spec += np.fft.fft(roles[pos]) * np.fft.fft(fillers[cid])
    return np.fft.ifft(spec).real


def truncation_information_loss(pairs: list[dict], embed_fn) -> dict:
    """embed_fn(word) -> np.ndarray. Returns mean cosine similarity between
    embed(word_a) and embed(word_b) across all pairs -- 1.0 means the arm
    cannot tell the pair apart at all (guaranteed for Kronecker/32-slot,
    since both share the same first-32-char prefix by construction)."""
    sims = []
    for pair in pairs:
        va = embed_fn(pair["word_a"])
        vb = embed_fn(pair["word_b"])
        sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
        sims.append(sim)
    return {"mean_cosine_similarity": float(np.mean(sims)), "n_pairs": len(pairs)}
