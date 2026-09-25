"""TinyStories in, a uint16 token stream out.

Three choices worth defending, because they shape every number in the report.

**TinyStories, not FineWeb.** A 21M model trained on 50M tokens of general web text
produces fluent-ish noise. On TinyStories it produces actual stories, so the README can
show samples per variant and a reader can see whether an integrator learned anything -
not just whether its loss curve went down.

**An 8192-token BPE, not GPT-2's 50257.** At 50257 the tied embedding table alone is
19.3M parameters and a "20M model" would be a lookup table with a rounding error attached.
At 8192 the embedding is 3.1M (15%) and the ten transformer blocks get the other 17.9M.

**A ranged download.** The source file is 2.23 GB and we need about a sixth of it. An HTTP
Range request takes the first slice only, which keeps 1.9 GB off the disk and cuts the
download to a couple of minutes.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request

import numpy as np

REPO = "roneneldan/TinyStories"
SOURCE = "TinyStoriesV2-GPT4-train.txt"
DEFAULT_BYTES = 350 * 1024 * 1024        # ~106M tokens at 8k BPE: 50M is under one epoch


def _url() -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(REPO, SOURCE, repo_type="dataset")


def fetch_text(path: str, n_bytes: int = DEFAULT_BYTES, log=print) -> str:
    """Download the first `n_bytes` of the corpus, resuming if interrupted."""
    if os.path.exists(path) and os.path.getsize(path) >= n_bytes * 0.99:
        log(f"  text already present: {os.path.getsize(path)/2**20:.0f} MiB")
        return path
    url, done = _url(), 0
    mode = "wb"
    if os.path.exists(path):
        done = os.path.getsize(path)
        mode = "ab"
        log(f"  resuming at {done/2**20:.0f} MiB")
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-{n_bytes - 1}"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(path, mode) as fh:
        while True:
            chunk = resp.read(8 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if done % (64 << 20) < (8 << 20):
                log(f"  {done/2**20:6.0f} MiB")
    # the slice almost certainly ends mid-story; drop the partial tail
    with open(path, "rb+") as fh:
        fh.seek(max(0, done - 4096))
        tail = fh.read()
        cut = tail.rfind(b"\n")
        if cut > 0:
            fh.seek(max(0, done - 4096) + cut)
            fh.truncate()
    log(f"  text: {os.path.getsize(path)/2**20:.0f} MiB")
    return path


def train_tokenizer(text_path: str, out_path: str, vocab_size: int = 8192,
                    train_bytes: int = 100 << 20, log=print):
    """A byte-level BPE over the first `train_bytes` of the corpus."""
    if os.path.exists(out_path):
        log("  tokenizer already trained")
        return out_path
    from tokenizers import ByteLevelBPETokenizer

    def chunks():
        with io.open(text_path, encoding="utf-8", errors="ignore") as fh:
            read = 0
            for line in fh:
                read += len(line)
                if read > train_bytes:
                    break
                yield line

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(chunks(), vocab_size=vocab_size, min_frequency=2,
                            special_tokens=["<|endoftext|>"])
    tok.save(out_path)
    log(f"  tokenizer: {vocab_size} merges -> {out_path}")
    return out_path


def encode_to_bin(text_path: str, tok_path: str, train_bin: str, val_bin: str,
                  val_tokens: int = 2_000_000, log=print) -> dict:
    """Tokenise the corpus into a flat uint16 stream, train and validation split."""
    meta_path = os.path.join(os.path.dirname(train_bin), "meta.json")
    if os.path.exists(train_bin) and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        log(f"  tokens already packed: {meta['train_tokens']:,} train")
        return meta
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    eot = tok.token_to_id("<|endoftext|>")
    buf, total = [], 0
    with open(train_bin + ".tmp", "wb") as out:
        with io.open(text_path, encoding="utf-8", errors="ignore") as fh:
            block = []
            for line in fh:
                block.append(line)
                if len(block) < 20000:
                    continue
                ids = []
                for enc in tok.encode_batch(block):
                    ids.extend(enc.ids)
                    ids.append(eot)
                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(out)
                total += arr.size
                block = []
                if total % 20_000_000 < 1_000_000:
                    log(f"  {total/1e6:6.1f}M tokens")
            if block:
                ids = []
                for enc in tok.encode_batch(block):
                    ids.extend(enc.ids)
                    ids.append(eot)
                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(out)
                total += arr.size

    data = np.memmap(train_bin + ".tmp", dtype=np.uint16, mode="r")
    np.asarray(data[:val_tokens]).tofile(val_bin)
    np.asarray(data[val_tokens:]).tofile(train_bin)
    del data
    os.remove(train_bin + ".tmp")
    meta = {"train_tokens": int(total - val_tokens), "val_tokens": int(val_tokens),
            "vocab_size": tok.get_vocab_size(), "source": f"{REPO}/{SOURCE}"}
    json.dump(meta, open(meta_path, "w"), indent=2)
    log(f"  packed {meta['train_tokens']:,} train + {val_tokens:,} val tokens")
    return meta


def prepare(data_dir: str = "data", n_bytes: int = DEFAULT_BYTES,
            vocab_size: int = 8192, log=print) -> dict:
    """Everything, idempotently. Safe to re-run; it resumes rather than restarts."""
    os.makedirs(data_dir, exist_ok=True)
    text = os.path.join(data_dir, "tinystories.txt")
    tokf = os.path.join(data_dir, "tokenizer.json")
    trb = os.path.join(data_dir, "train.bin")
    vab = os.path.join(data_dir, "val.bin")
    log("preparing data")
    fetch_text(text, n_bytes, log=log)
    train_tokenizer(text, tokf, vocab_size, log=log)
    meta = encode_to_bin(text, tokf, trb, vab, log=log)
    meta.update({"train_bin": trb, "val_bin": vab, "tokenizer": tokf})
    return meta


class Batches:
    """Deterministic batch sampler, with the host out of the critical path.

    Two things this has to get right.

    **Determinism.** Every run draws from one generator seeded identically, so two runs at
    the same batch size see the same tokens in the same order, and their loss curves are
    comparable step for step rather than merely on average.

    **Staying out of the way, simply.** The token stream is read into RAM once (180 MB)
    and a batch is gathered with a single vectorised index - 0.6 ms against ~250 ms of
    compute, so there is nothing to win by being cleverer.

    Two dead ends are recorded here because both cost real time. A `np.memmap` fancy-index
    per step was blamed for a run coming in at half the benchmarked speed; the real cause
    was the GPU thermally throttling from 2100 to 1560 MHz (see `revlm/bench.py`). And a
    background prefetch thread with `pin_memory()` measured *identically* to the plain
    synchronous path (50.2k against 50.0k tokens/s) while leaking host memory - the process
    reached 5.5 GB and stalled. Neither cleverness was load-bearing; both are gone.
    """

    def __init__(self, bin_path: str, batch_size: int, block_size: int,
                 device: str = "cuda", seed: int = 1234):
        import torch

        raw = np.fromfile(bin_path, dtype=np.uint16)
        self.data = torch.from_numpy(raw.astype(np.int32))
        self.B, self.T, self.device = batch_size, block_size, device
        self.rng = np.random.default_rng(seed)
        self.n = self.data.numel() - block_size - 1
        self._off = torch.arange(block_size + 1, dtype=torch.int64)

    def __len__(self):
        return self.data.numel()

    def next(self):
        import torch

        ix = torch.from_numpy(self.rng.integers(0, self.n, size=self.B).astype(np.int64))
        win = self.data[ix[:, None] + self._off[None, :]].to(torch.int64)
        x, y = win[:, :-1].contiguous(), win[:, 1:].contiguous()
        if self.device == "cuda":
            return x.to("cuda", non_blocking=False), y.to("cuda", non_blocking=False)
        return x, y


def decode(tok_path: str, ids) -> str:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(tok_path).decode(list(ids))
