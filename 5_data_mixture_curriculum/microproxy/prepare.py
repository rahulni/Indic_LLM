# -*- coding: utf-8 -*-
"""
prepare.py - build byte-level train/heldout arrays from the session-4 corpora.

Byte-level (vocab 256) rather than a subword tokenizer, deliberately:

  - It removes the tokenizer as a confound. This experiment measures training
    DYNAMICS at a distribution shift, and a tokenizer whose fertility differs
    5.78x between the two lanes (session-4 measurement: 13.268 tok/word cl100k
    on Telugu vs 2.296 under MuRIL) would make "share of tokens" mean something
    different per lane - which is the exact accounting error section 5.1 of the
    plan exists to prevent.
  - Vocab 256 keeps the embedding table negligible, so nearly all parameters do
    real work at this scale.

Lanes come straight from what we actually hold: Telugu web (indic) and the
English reasoning/SFT mix (reasoning). Held-out slices are the disjoint ones the
session-4 pipeline already froze.
"""

import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "4_model_data", "assignment", "raw_sample")
OUT = os.path.join(HERE, "data")

LANES = [
    dict(id="indic", train="telugu_raw.jsonl", heldout="telugu_heldout.jsonl",
         name="Sangraha unverified / Telugu"),
    dict(id="reasoning", train="reasoning_sft_raw.jsonl",
         heldout="reasoning_sft_heldout.jsonl",
         name="Reasoning / SFT distillation mix"),
]

DOC_SEP = b"\n\n"


def encode(path, limit_mb=None):
    buf = bytearray()
    n_docs = 0
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line).get("text", "")
            if not t:
                continue
            buf += t.encode("utf-8") + DOC_SEP
            n_docs += 1
            if limit_mb and len(buf) > limit_mb * 1_000_000:
                break
    return np.frombuffer(bytes(buf), dtype=np.uint8), n_docs


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for lane in LANES:
        for split in ("train", "heldout"):
            src = os.path.join(SRC, lane[split])
            if not os.path.exists(src):
                print(f"  MISSING {src}")
                return 1
            arr, n_docs = encode(src)
            dst = os.path.join(OUT, f"{lane['id']}_{split}.bin")
            arr.tofile(dst)
            print(f"  {lane['id']:10s} {split:8s} {n_docs:6,d} docs  "
                  f"{len(arr)/1e6:7.2f} MB")
            manifest.append(dict(lane=lane["id"], split=split, docs=n_docs,
                                 bytes=int(len(arr)), source=lane[split],
                                 name=lane["name"]))
    with io.open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    tot = sum(m["bytes"] for m in manifest if m["split"] == "train")
    print(f"\n  total train: {tot/1e6:.1f} MB ({tot/1e6:.1f}M byte-tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
