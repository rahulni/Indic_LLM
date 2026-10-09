from . import md, code

CELLS = [
md(r'''
---
## D · Data: TinyStories, a 4,096-token BPE, and a fixed batch order

* **Corpus.** `TinyStoriesV2-GPT4`: short, simple English stories. A ~20M-parameter model is far
  from saturated on it, so a bigger MoE has room to show a gain. Only the first ~240 MB of the 2.2 GB
  training file is streamed, using the standard library, so nothing beyond `tokenizers` is installed.
* **Tokenizer.** A byte-level BPE with 4,096 tokens, trained here on 30 MB. With GPT-2's 50k vocabulary
  the embeddings alone would be 19M parameters and would dwarf the feed-forward blocks this
  experiment is about. At 4,096 they are 1.6M.
* **Batch order.** The token stream is cut into 257-token windows and shuffled once with a fixed
  seed. Step *s* of any run reads windows `[cursor, cursor + batch)`. Two branches that start from
  the same checkpoint therefore see *identical* batches, so the MoE-vs-dense comparison is paired.
  The main path stays under one epoch.
'''),
code(r'''
TS_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{}.txt"
EOT = "<|endoftext|>"

def fetch(url, part, limit=None, retries=10):
    """Resumable download: on a dropped connection, continue from the bytes already on disk."""
    for attempt in range(retries):
        have_b = part.stat().st_size if part.exists() else 0
        if limit is not None and have_b >= limit:
            return
        rng = f"bytes={have_b}-" + (str(limit - 1) if limit else "")
        try:
            req = urllib.request.Request(url, headers={"Range": rng, "User-Agent": "moe14-notebook"})
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if (have_b and r.status == 206) else "wb"   # server ignored Range: restart
                with open(part, mode) as f:
                    got = have_b if mode == "ab" else 0
                    while limit is None or got < limit:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            return
                        f.write(chunk)
                        got += len(chunk)
            return
        except Exception as e:
            print(f"  download interrupted at {have_b / 2**20:.0f} MB ({type(e).__name__}); resuming ({attempt + 1}/{retries})")
            time.sleep(min(30, 3 * (attempt + 1)))
    raise RuntimeError("Could not download TinyStories from huggingface.co. Check the network and re-run this cell.")


def stream_text(split, max_mb=None):
    """Download (the first max_mb of) a TinyStories file, cut at a story boundary, cache it."""
    out = DATA / (f"ts_{split}.txt" if max_mb is None else f"ts_{split}_{max_mb}mb.txt")
    if out.exists():
        return out
    part = out.with_suffix(".part")
    print(f"downloading TinyStoriesV2-GPT4-{split}" + (f" (first {max_mb} MB)" if max_mb else ""))
    fetch(TS_URL.format(split), part, None if max_mb is None else max_mb * 2**20)
    text = part.read_bytes().decode("utf-8", errors="ignore")
    if max_mb is not None:
        text = text[: text.rfind(EOT) + len(EOT)]
    out.write_text(text, encoding="utf-8", newline="")
    part.unlink()
    return out


def stories(path, max_mb=None):
    text = path.read_text(encoding="utf-8")
    if max_mb is not None:
        text = text[: max_mb * 2**20]
    for s in text.split(EOT):
        s = s.strip()
        if s:
            yield s


def build_tokenizer(train_path):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    f = DATA / f"bpe4096_{BUDGET['bpe_mb']}mb.json"
    if f.exists():
        return Tokenizer.from_file(str(f))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=GCFG.vocab, special_tokens=[EOT], min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(stories(train_path, BUDGET["bpe_mb"]), trainer)
    tok.save(str(f))
    return tok


def encode_file(tok, path, out):
    if out.exists():
        return np.memmap(out, dtype=np.uint16, mode="r")
    parts, batch = [], []
    eot = tok.token_to_id(EOT)
    def flush():
        for e in tok.encode_batch(batch):
            parts.append(np.asarray(e.ids + [eot], dtype=np.uint16))
        batch.clear()
    for s in stories(path):
        batch.append(s)
        if len(batch) == 20000:
            flush()
    flush()
    arr = np.concatenate(parts)
    arr.tofile(out)
    return np.memmap(out, dtype=np.uint16, mode="r")


class Stream:
    """Shuffled, non-overlapping 257-token windows; batch(cursor, B) is a pure function of its args."""
    def __init__(self, tokens, ctx, seed=0, shuffle=True):
        self.tok, self.ctx = tokens, ctx
        self.n = (len(tokens) - 1) // ctx
        self.perm = np.random.default_rng(seed).permutation(self.n) if shuffle else np.arange(self.n)
        self.ar = np.arange(ctx + 1)

    def batch(self, cursor, B):
        idx = self.perm[(cursor + np.arange(B)) % self.n] * self.ctx
        t = torch.from_numpy(self.tok[idx[:, None] + self.ar].astype(np.int64))
        if DEVICE == "cuda":
            t = t.pin_memory().to(DEVICE, non_blocking=True)
        return t[:, :-1], t[:, 1:]
'''),
code(r'''
TOK = TRAIN = VAL = None
if COMPUTE:
    t0 = time.time()
    train_txt = stream_text("train", BUDGET["train_mb"])
    valid_txt = stream_text("valid")
    TOK = build_tokenizer(train_txt)
    assert TOK.get_vocab_size() == GCFG.vocab and TOK.token_to_id(EOT) == 0
    train_ids = encode_file(TOK, train_txt, DATA / f"train_{BUDGET['train_mb']}mb_{BUDGET['bpe_mb']}.bin")
    val_ids = encode_file(TOK, valid_txt, DATA / f"val_{BUDGET['bpe_mb']}.bin")
    TRAIN = Stream(train_ids, GCFG.ctx, seed=0)
    VAL = Stream(val_ids, GCFG.ctx, shuffle=False)
    need = (S1 + S2 + S3) * BUDGET["batch"]
    print(f"train: {len(train_ids) / 1e6:.1f}M tokens ({TRAIN.n:,} windows) | val: {len(val_ids) / 1e6:.2f}M tokens"
          f" | chars/token {train_txt.stat().st_size / len(train_ids):.2f} | {time.time() - t0:.0f}s")
    assert TRAIN.n >= need, f"main path needs {need:,} windows for one epoch; raise BUDGET['train_mb']"
    R["data"] = dict(train_tokens=int(len(train_ids)), val_tokens=int(len(val_ids)),
                     epoch_fraction=need / TRAIN.n,
                     sample=[TOK.decode([i]) for i in TOK.encode("Once upon a time, Lily saw a big red ball.").ids])
    save_results()
if have("data"):
    print("epoch fraction used by the main path:", round(R["data"]["epoch_fraction"], 3))
    print("tokenized:", "|".join(R["data"]["sample"]), f"({len(R['data']['sample'])} tokens)")
'''),
]
