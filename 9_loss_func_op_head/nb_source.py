# %% [markdown]
# # Loss Functions and Output Heads
#
# **ERA V5 — Session 9.** One notebook, one loss harness, and one thing you have to get
# right by reading rather than by guessing.
#
# The assignment is not "train a model". It is: take these four lines
#
# ```python
# hidden = model(tokens)
# logits = output_head(hidden)
# loss   = cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
#                        tokens[:, 1:].reshape(-1))
# ```
#
# and make them **correct and observable**. Every serious bug that lives in these lines
# shares one property: *it does not raise an exception*. A target shift in the wrong
# direction produces a beautiful loss curve and a worthless model.
#
# So every section below prints evidence, and most of them assert on it.
#
# ---
#
# ### How to read this notebook
#
# Each section has the same four beats:
#
# 1. **Why this check exists** — and the failure it catches
# 2. **The code** — short, inline, no helper indirection
# 3. **The printed evidence**
# 4. **What you should be seeing** — so you can tell a pass from a fail
#
# Set `QUICK_RUN = True` to read the whole thing end to end in a couple of minutes.
# Set it to `False` to reproduce the numbers reported in the README.

# %% [markdown]
# ## 0 — Setup
#
# Installs are idempotent, so this runs unchanged on a bare Colab T4 or on a local CUDA box.

# %%
import subprocess, sys, importlib.util

for pkg in ["numpy", "tiktoken", "matplotlib"]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os, math, time, json, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import matplotlib
matplotlib.use("Agg")          # so the same file runs headless and under nbconvert
import matplotlib.pyplot as plt

try:                            # inside a notebook, also show the figure inline
    from IPython.display import Image, display
    def show_png(path):
        display(Image(filename=path))
except ImportError:
    def show_png(path):
        pass

# ---------------------------------------------------------------- run configuration
QUICK_RUN = os.environ.get("QUICK_RUN", "1") == "1"   # 300 steps to read; False for README numbers
VERBOSE   = True                                      # extra shape traces

STEPS      = 300 if QUICK_RUN else 2000
EVAL_EVERY = 25  if QUICK_RUN else 100
SEED       = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# Every headline number lands here; the last cell renders it as the README table.
RESULTS = {}

# dark charts, to match the rest of the project
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
    "savefig.facecolor": "#0d1117", "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3", "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d", "grid.color": "#21262d", "figure.dpi": 130,
})

print(f"torch      : {torch.__version__}")
print(f"device     : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
print(f"QUICK_RUN  : {QUICK_RUN}  ->  {STEPS} steps")

# %% [markdown]
# ## 1 — The model and the data
#
# A nanoGPT-style decoder with the GPT-2 BPE tokenizer. The vocabulary choice is not
# cosmetic: `V = 50257` is what makes the untrained-perplexity check in section 6 a real
# test, and it gives real subword strings for the shift table in section 3.
#
# **Be honest about the shape of this model.** With `n_embd = 256` against `V = 50257`,
# only about 4.7M of its ~17.7M parameters are actual transformer. The rest is two
# enormous lookup tables. That is a deliberately lopsided configuration — it is exactly
# what makes sections 7 and 8 worth measuring — but it is not a well-proportioned
# language model, and the loss curves later should be read with that in mind.

# %%
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.qkv  = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=False)
        self.proj = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, n_head, T, head_dim]: each head gets its own slice of the channels
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # concatenate the heads back
        return self.proj(y)


class MLP(nn.Module):
    """The plain 4x up-project / GELU / down-project feed-forward."""
    def __init__(self, cfg):
        super().__init__()
        self.fc   = nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"], bias=False)
        self.proj = nn.Linear(4 * cfg["n_embd"], cfg["n_embd"], bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-norm: the residual stream is never operated on, only added to."""
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg["n_embd"])
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # what the other tokens say about me
        x = x + self.mlp(self.ln2(x))    # what I make of that on my own
        return x


class GPT(nn.Module):
    def __init__(self, cfg, tie_weights=True, n_heads_out=1, block_cls=None):
        super().__init__()
        block_cls = block_cls or Block
        self.cfg = cfg
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.wpe = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.blocks = nn.ModuleList([block_cls(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.apply(self._init)
        if tie_weights:                       # after init, so the two never diverge
            self.head.weight = self.wte.weight
        # a second output head, used only in Part 2
        self.head2 = None
        if n_heads_out == 2:
            self.head2 = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
            nn.init.normal_(self.head2.weight, mean=0.0, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def hidden_states(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, idx):
        h = self.hidden_states(idx)
        if self.head2 is None:
            return self.head(h)
        return self.head(h), self.head2(h)


cfg = dict(vocab_size=50257, n_embd=256, n_layer=6, n_head=8, block_size=256)
enc = tiktoken.get_encoding("gpt2")
V, D, T_CTX = cfg["vocab_size"], cfg["n_embd"], cfg["block_size"]
PAD_ID = enc.eot_token   # 50256, <|endoftext|>

# ---------------------------------------------------------------- data
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/input.txt"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "data/input.txt")

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

all_ids = torch.tensor(enc.encode(text), dtype=torch.long)
n_split = int(0.9 * len(all_ids))
train_ids, val_ids = all_ids[:n_split], all_ids[n_split:]


def get_batch(split, B=16, T=T_CTX, generator=None):
    """Returns x, y (targets at t+1) and y2 (targets at t+2, for Part 2)."""
    src = train_ids if split == "train" else val_ids
    ix = torch.randint(len(src) - T - 2, (B,), generator=generator)
    x  = torch.stack([src[i     : i + T    ] for i in ix])
    y  = torch.stack([src[i + 1 : i + T + 1] for i in ix])
    y2 = torch.stack([src[i + 2 : i + T + 2] for i in ix])
    return x.to(device), y.to(device), y2.to(device)


def tok(i):
    """Render one token id as bytes. NOT decode([i]) - see section 3."""
    return enc.decode_single_token_bytes(int(i))


print(f"characters      : {len(text):,}")
print(f"tokens (GPT-2)  : {len(all_ids):,}   train {len(train_ids):,} / val {len(val_ids):,}")
print(f"vocab size V    : {V:,}")
print(f"pad token       : {PAD_ID} -> {tok(PAD_ID)!r}")

model = GPT(cfg, tie_weights=True).to(device)
n_params = sum(p.numel() for p in model.parameters())
n_lookup = model.wte.weight.numel() + model.wpe.weight.numel()
print(f"\nparameters      : {n_params:,}  (tied)")
print(f"  transformer   : {n_params - n_lookup:,}  ({100*(n_params-n_lookup)/n_params:.1f}%)")
print(f"  lookup tables : {n_lookup:,}  ({100*n_lookup/n_params:.1f}%)")

# %% [markdown]
# ### A small training helper
#
# Used by several sections below. Nothing clever: AdamW, cosine decay, linear warmup.

# %%
def make_opt(m, lr=3e-4):
    return torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


def lr_at(step, total, base=3e-4, warmup=100):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * prog))


def quick_train(m, steps, loss_fn, log_every=None, seed=SEED):
    """Trains m for `steps`. loss_fn(model, x, y, y2) -> (loss, dict_of_things_to_log)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = make_opt(m)
    hist = []
    m.train()
    for step in range(steps):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, steps)
        x, y, y2 = get_batch("train", generator=g)
        loss, extras = loss_fn(m, x, y, y2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            hist.append({"step": step, **extras})
    return hist


def plain_loss(m, x, y, y2):
    """The standard next-token objective, written out the long way."""
    logits = m(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    return loss, {"train": loss.item()}


# %% [markdown]
# ## 2 — Print every tensor shape, and say what each dimension is
#
# **What this catches.** A `reshape` that silently succeeds on the wrong layout. If
# `logits` and `targets` disagree on how many rows they have, PyTorch raises - but if they
# *agree* for the wrong reason, it does not, and you train a scrambled objective forever.
#
# The four names, once:
#
# - **B** - how many independent documents are in this batch
# - **T** - how many tokens in each one (the context length)
# - **D** - how wide each token vector is
# - **V** - how many distinct tokens exist in the whole dictionary

# %%
x_demo, y_demo, _ = get_batch("train", B=4)

model.eval()
with torch.no_grad():
    hidden_demo = model.hidden_states(x_demo)
    logits_demo = model.head(hidden_demo)

logits_flat  = logits_demo[:, :-1, :].reshape(-1, V)
targets_flat = y_demo[:, :-1].reshape(-1)

rows = [
    ("tokens",              tuple(x_demo.shape),       "B documents, T token ids each"),
    ("hidden",              tuple(hidden_demo.shape),  "per-token meaning vector, D wide"),
    ("logits",              tuple(logits_demo.shape),  "one score per vocabulary entry, per position"),
    ("logits[:, :-1] flat", tuple(logits_flat.shape),  "last position dropped - nothing follows it"),
    ("targets flat",        tuple(targets_flat.shape), "first position dropped - it is nobody target"),
]
w = max(len(r[0]) for r in rows)
print(f"{'tensor'.ljust(w)}  {'shape'.ljust(18)}  what each dimension is")
print("-" * (w + 68))
for name, shape, gloss in rows:
    print(f"{name.ljust(w)}  {str(shape).ljust(18)}  {gloss}")

print(f"\nB = {x_demo.shape[0]}   T = {x_demo.shape[1]}   D = {hidden_demo.shape[2]}   V = {V:,}")

assert logits_flat.shape[0] == targets_flat.shape[0], "row count mismatch - the reshape is wrong"
assert logits_flat.shape[1] == V
print(f"\nOK  {logits_flat.shape[0]:,} rows of logits line up with {targets_flat.shape[0]:,} targets")

RESULTS["1. flattened loss matrix"] = f"`{tuple(logits_flat.shape)}` vs `{tuple(targets_flat.shape)}`"

# %% [markdown]
# **What you should be seeing.** `hidden` and `logits` agree on B and T and differ only in
# the last axis - D became V. The flattened pair must agree on row count, and that row
# count is `B*(T-1)`, not `B*T`: one position at each end of the sequence is unusable.

# %% [markdown]
# ## 3 — Verify the shift by printing the token **strings**
#
# This is the check the whole assignment is built around. You will not catch an
# off-by-one in a wall of integers.
#
# **One trap worth knowing.** Do not render tokens with `enc.decode([i])`. GPT-2 uses
# byte-level BPE, so plenty of token ids are *fragments of a UTF-8 sequence* and decode to
# the replacement character. On an exercise whose entire thesis is *print the strings*,
# having them render as boxes would defeat the point. Use
# `enc.decode_single_token_bytes(i)` and `repr()` the result - that also makes leading
# spaces visible instead of losing them to column alignment.

# %%
seq_in, seq_tgt = x_demo[0], y_demo[0]

print(f"{'pos':>4}  {'input token':<20}  {'target token':<20}")
print("-" * 50)
for p in range(14):
    print(f"{p:>4}  {str(tok(seq_in[p])):<20}  {str(tok(seq_tgt[p])):<20}")

print("\nreading the first row as a sentence:")
print("   input :", repr(b"".join(tok(i) for i in seq_in[:14]).decode("utf-8", "replace")))
print("   target:", repr(b"".join(tok(i) for i in seq_tgt[:14]).decode("utf-8", "replace")))

# the machine version of the same check: targets are inputs advanced by exactly one
matched = int((seq_in[1:] == seq_tgt[:-1]).sum())
total   = len(seq_in) - 1
assert matched == total, f"shift is wrong: only {matched}/{total} positions line up"
print(f"\nOK  targets are inputs advanced by exactly one at {matched}/{total} positions")

RESULTS["2. shift positions verified"] = f"{matched}/{total}"

# %% [markdown]
# **What you should be seeing.** Every target is the input from the *next* row down. The
# target sentence is the input sentence missing its first token and carrying one extra at
# the end. If instead you see `b'First'` sitting opposite `b'First'`, you have the bug
# from section 10 and your loss is about to look wonderful.

# %% [markdown]
# ## 4 — Mask the padding, and watch the contributing-token count change
#
# **What this catches.** Real batches hold documents of different lengths, so short ones
# get filled with a pad token. Pad is trivially predictable - it always follows pad - so
# if you let it into the loss the model learns "just keep saying pad", the mean collapses,
# and the curve looks superb while inference emits pad forever.
#
# The fix is `ignore_index`: set those target positions to `-100` and cross-entropy skips
# them, both in the sum and in the count it divides by.

# %%
lengths = [256, 190, 120, 64]        # four documents of deliberately unequal length
B_pad = len(lengths)

pad_x = torch.full((B_pad, T_CTX), PAD_ID, dtype=torch.long)
pad_y = torch.full((B_pad, T_CTX), PAD_ID, dtype=torch.long)
for r, L in enumerate(lengths):
    start = r * 1000
    pad_x[r, :L] = train_ids[start     : start + L]
    pad_y[r, :L] = train_ids[start + 1 : start + L + 1]
pad_x, pad_y = pad_x.to(device), pad_y.to(device)

with torch.no_grad():
    pad_logits = model(pad_x)

flat_logits  = pad_logits[:, :-1, :].reshape(-1, V)
tgt_unmasked = pad_y[:, :-1].reshape(-1).clone()
tgt_masked   = tgt_unmasked.clone()
tgt_masked[tgt_masked == PAD_ID] = -100          # the one line that matters

loss_unmasked = F.cross_entropy(flat_logits, tgt_unmasked)
loss_masked   = F.cross_entropy(flat_logits, tgt_masked, ignore_index=-100)

n_unmasked = tgt_unmasked.numel()
n_masked   = int((tgt_masked != -100).sum())

print(f"document lengths            : {lengths}  (padded to {T_CTX})")
print()
print(f"{'':<28}{'contributing tokens':>21}{'loss':>12}")
print("-" * 61)
print(f"{'pad counted in the loss':<28}{n_unmasked:>21,}{loss_unmasked.item():>12.4f}")
print(f"{'pad masked out':<28}{n_masked:>21,}{loss_masked.item():>12.4f}")
print("-" * 61)
print(f"{'difference':<28}{n_unmasked - n_masked:>21,}{loss_masked.item()-loss_unmasked.item():>+12.4f}")
print(f"\n{100*(n_unmasked-n_masked)/n_unmasked:.1f}% of this batch was padding")

assert n_masked < n_unmasked, "masking changed nothing - check the pad id"
RESULTS["3. contributing tokens, pad counted -> masked"] = (
    f"{n_unmasked:,} -> {n_masked:,} tokens; loss {loss_unmasked.item():.4f} -> {loss_masked.item():.4f}"
)

# %% [markdown]
# **What you should be seeing.** The token count drops by exactly the number of pad
# positions, and the loss *moves* - here it goes **up** by about 0.24 when pad is removed,
# which is to say pad was quietly making the model look better than it was.
#
# That is worth a second of thought, because the model has not been trained yet. The
# reason is correlation: all 391 pad targets are the *same* token id, so if that one id
# happens to draw a slightly above-average logit out of a random head, all 391 positions
# collect the same small discount at once. Real tokens get no such shared luck. Under
# training the effect stops being luck and becomes a strategy, which is the failure mode
# the session described: the loss falls, and inference emits pad forever.
#
# **A nuance the assignment does not ask for but which matters in practice:** the position
# that predicts the *first* pad is the one teaching the model to stop. Masking every pad
# target, as above, throws that lesson away too. Production setups usually keep the first
# EOS and mask the rest.

# %% [markdown]
# ## 5 — Pack two documents into one sequence, and mask the boundary
#
# **What this catches.** To avoid wasting context on padding, real pipelines *pack*:
# several short documents are concatenated into one row of length T. That creates one
# poisonous position per join. At the last token of document A, the "correct" next token
# is the first token of document B - a completely unrelated document. Nothing in A
# predicts it. You are asking the model to guess the unguessable, and punishing it for
# failing.
#
# **This has to be measured on a trained model, not at initialisation.** At init every
# position costs about `ln(V)` and the boundary is invisible in the noise. Once the model
# has learned something, the boundary stands out precisely because everything else got
# easier.

# %%
t0 = time.time()
hist_warm = quick_train(model, STEPS // 2, plain_loss, log_every=EVAL_EVERY)
print(f"warmed the model up for {STEPS//2} steps in {time.time()-t0:.1f}s "
      f"(train loss {hist_warm[0]['train']:.3f} -> {hist_warm[-1]['train']:.3f})")

# ---------------------------------------------------------------- build packed rows
# One join is an anecdote, so pack 32 independent document pairs and average over them.
HALF, N_PACK = 128, 32
BOUNDARY = HALF - 1          # index in the target row whose target is doc B's first token

gp = torch.Generator().manual_seed(SEED)
starts_a = torch.randint(0, len(train_ids) - HALF - 1, (N_PACK,), generator=gp)
starts_b = torch.randint(0, len(train_ids) - HALF - 1, (N_PACK,), generator=gp)
packed = torch.stack([
    torch.cat([train_ids[a : a + HALF], train_ids[b : b + HALF]])
    for a, b in zip(starts_a, starts_b)
]).to(device)                                                   # [N_PACK, 256]

pk_x, pk_y = packed[:, :-1], packed[:, 1:]

model.eval()
with torch.no_grad():
    pk_logits = model(pk_x)

per_pos = F.cross_entropy(
    pk_logits.reshape(-1, V), pk_y.reshape(-1), reduction="none"
).view(N_PACK, -1)                                              # [N_PACK, 255]

print(f"\none of the {N_PACK} joins, as strings:")
for p in range(BOUNDARY - 2, BOUNDARY + 3):
    marker = "  <-- BOUNDARY (doc A ends, doc B begins)" if p == BOUNDARY else ""
    print(f"  pos {p:>3}  {str(tok(pk_x[0, p])):<16} -> {str(tok(pk_y[0, p])):<16}"
          f"  loss {per_pos[0, p].item():>7.3f}{marker}")

keep = per_pos.mean()
mask = torch.ones_like(per_pos, dtype=torch.bool)
mask[:, BOUNDARY] = False
drop = per_pos[mask].mean()
boundary_mean = per_pos[:, BOUNDARY].mean()

print(f"\naveraged over {N_PACK} packed sequences:")
print(f"\n{'':<34}{'tokens':>9}{'loss':>11}")
print("-" * 54)
print(f"{'boundary counted':<34}{per_pos.numel():>9,}{keep.item():>11.4f}")
print(f"{'boundary masked out':<34}{int(mask.sum()):>9,}{drop.item():>11.4f}")
print("-" * 54)
print(f"{'mean loss AT the boundary':<34}{N_PACK:>9,}{boundary_mean.item():>11.4f}")
print(f"{'mean loss everywhere else':<34}{'':>9}{drop.item():>11.4f}")
print(f"\nthe boundary position costs {boundary_mean.item()/drop.item():.2f}x "
      f"the average position")

RESULTS["4. loss, boundary kept -> masked"] = (
    f"{keep.item():.4f} -> {drop.item():.4f} "
    f"(boundary positions average {boundary_mean.item():.4f}, "
    f"{boundary_mean.item()/drop.item():.2f}x the rest)"
)

# %% [markdown]
# **What you should be seeing.** One position carrying several times the loss of its
# neighbours, and the sequence mean dropping once it is removed. The drop from a single
# masked position out of 255 looks small; scale it up - a 4096-token context packed with
# 1000-token documents has three of these, and every one of them is teaching the model
# that documents bleed into each other.
#
# Masking is one line - the same `-100` from section 4, applied at the join. The reason
# to do it is not really the loss number, it is that the gradient from an unguessable
# target is pure noise pushing the model to hallucinate topic changes.

# %% [markdown]
# ## 6 — Perplexity, and why an untrained model must sit near the vocabulary size
#
# `perplexity = exp(loss)`, and it is the honest unit: it answers *"how many equally
# likely options is the model effectively choosing between?"*
#
# A freshly initialised model knows nothing, so it spreads probability roughly evenly over
# all V tokens. Probability `1/V` gives `loss = ln(V)` and `perplexity = V`. **That is a
# gate, not an observation.** If a fresh model starts anywhere else, something is leaking
# the answer and there is no point training it.
#
# The expected value sits slightly *above* `ln(V)`, and it is worth deriving rather than
# treating as a magic constant. Cross-entropy is `logsumexp(z) - z_target`. For logits
# `z ~ N(0, sigma^2)` the target term averages to zero, while
# `E[logsumexp(z)] ~= ln(V) + sigma^2 / 2`, so `E[loss] ~= ln(V) + sigma^2 / 2`.
#
# With std-0.02 init and `D = 256` the logits arrive with
# `sigma ~= 0.02 * sqrt(256) = 0.32`, giving `10.825 + 0.051 = 10.876`. The measured value
# lands slightly under that, which is Jensen's inequality: `E[ln X] <= ln E[X]`.

# %%
torch.manual_seed(SEED)
fresh = GPT(cfg, tie_weights=True).to(device)      # a brand new model, never trained
fx, fy, _ = get_batch("val", B=8)

fresh.eval()
with torch.no_grad():
    f_logits = fresh(fx)
    init_loss = F.cross_entropy(f_logits[:, :-1].reshape(-1, V), fy[:, :-1].reshape(-1))
    logit_sigma = f_logits.std().item()

init_ppl = math.exp(init_loss.item())

print(f"ln(V)                    = ln({V:,})  = {math.log(V):.4f}")
print(f"logit sigma at init      = {logit_sigma:.4f}")
print(f"predicted  ln(V) + s^2/2 = {math.log(V) + logit_sigma**2/2:.4f}")
print(f"measured   initial loss  = {init_loss.item():.4f}")
print()
print(f"perplexity at init       = {init_ppl:,.0f}")
print(f"vocabulary size V        = {V:,}")
print(f"ratio                    = {init_ppl/V:.3f}")

assert abs(init_loss.item() - math.log(V)) < 0.15, "init is wrong - do not train this"
print("\nOK  the untrained model sits at the vocabulary size, as it must")

RESULTS["5. init loss / perplexity"] = (
    f"{init_loss.item():.4f} / {init_ppl:,.0f}  (`ln(50257)` = {math.log(V):.4f}, V = {V:,})"
)

# %% [markdown]
# **What you should be seeing.** A loss in the 10.7-10.8 band and a perplexity within a
# percent or two of 50,257.
#
# The instructor's rule of thumb, worth committing to memory:
#
# - starts at **~11.7** for a 131k vocab, **~10.8** for GPT-2's 50k -> initialisation is fine
# - starts at **4** -> something is leaking the answer. Do not train. Find it.
# - starts at 11.7 and reaches **1** within a few steps -> too good to be true, and it is
#
# A related warning: **perplexity is not comparable across tokenizers.** A tokenizer that
# splits Hindi into three fragments per word will report a flattering perplexity on Hindi
# precisely because it made each individual prediction easier, not because the model
# understands Hindi better.

# %% [markdown]
# ## 7 — Tied against untied output head
#
# The embedding matrix maps `token id -> vector`. The output head maps `vector -> scores
# over token ids`. They are the same relationship read in opposite directions, so they can
# share one matrix. That is weight tying.
#
# On this configuration it is not a marginal saving, because `V x D` dominates everything.

# %%
torch.manual_seed(SEED); m_tied   = GPT(cfg, tie_weights=True)
torch.manual_seed(SEED); m_untied = GPT(cfg, tie_weights=False)

p_tied   = sum(p.numel() for p in m_tied.parameters())
p_untied = sum(p.numel() for p in m_untied.parameters())
head_sz  = V * D

print(f"{'':<22}{'parameters':>14}{'vs tied':>12}")
print("-" * 48)
print(f"{'tied':<22}{p_tied:>14,}{'-':>12}")
print(f"{'untied':<22}{p_untied:>14,}{f'+{100*(p_untied-p_tied)/p_tied:.1f}%':>12}")
print("-" * 48)
print(f"{'difference':<22}{p_untied - p_tied:>14,}")
print(f"{'V x D':<22}{head_sz:>14,}   ({V:,} x {D})")
print()
print(f"the head is {100*head_sz/p_untied:.1f}% of the untied model")
print(f"tying removes {100*(p_untied-p_tied)/p_untied:.1f}% of the parameters "
      f"and costs zero accuracy at this scale")

assert p_untied - p_tied == head_sz, "the difference must be exactly one V x D matrix"
print("\nOK  the difference is exactly one V x D matrix")

RESULTS["6. tied vs untied parameters"] = (
    f"{p_tied:,} vs {p_untied:,}  (difference {head_sz:,} = V x D)"
)

del m_tied, m_untied

# %% [markdown]
# **What you should be seeing.** A difference equal to `V * D` to the parameter, and an
# untied model roughly 73% larger.
#
# The session's framing: at production scale (`D = 4096`, `V = 131072`) that same matrix is
# **half a billion parameters** on the input side and another half billion on the output
# side - about 1B of an 11B model spent on two lookup tables. Tying is the cheap answer,
# usually applied to small models and dropped for large ones, where the head is given the
# freedom to disagree with the embedding.

# %% [markdown]
# ## 8 — Peak memory: ordinary cross-entropy against a chunked version
#
# `logits` is `[B, T, V]`, and V is enormous. It is almost always the largest tensor in the
# whole model - larger than any activation inside the transformer. At `B=4, T=1024,
# V=50257` in fp32 that single tensor is about **785 MB**, and cross-entropy's backward
# needs room for a log-softmax of the same size plus a gradient of the same size again.
#
# The fix is to never hold all of it: slice the rows into chunks, compute the loss on one
# chunk, and move on. Mathematically identical, materially cheaper, somewhat slower.
#
# **The trap, and it is the whole exercise.** Chunking only the *forward* pass saves
# nothing. Every chunk's logits stay alive in the autograd graph until `backward()` runs,
# so peak memory is unchanged and you have made the code slower for free. The chunked
# version must call `backward()` *inside* the loop so each chunk is freed as it goes, then
# push the accumulated gradient into the trunk in one final step:
#
# ```
# h_det = hidden.detach().requires_grad_(True)   # cut the graph
# for chunk: loss_c.backward()                   # frees this chunk's logits
# hidden.backward(h_det.grad)                    # reattach, one shot
# ```
#
# **Why this is measured on a synthetic hidden state.** The model above has
# `block_size = 256`, so it cannot produce a `T = 1024` activation. Isolating the head and
# the loss is also the more honest comparison: the transformer trunk's activations are
# common to both paths and would only dilute the ratio being reported.
#
# **On sizing.** The batch is deliberately kept small enough that the ordinary path fits
# in VRAM. Push it to `B = 8` on an 8 GB card and the naive path silently spills into
# shared system memory over PCIe - at which point the numbers describe Windows' memory
# manager rather than cross-entropy, and the "chunked is 20x faster" result you get is an
# artifact, not a finding.

# %%
def measure_peak(fn):
    """Extra allocator bytes this call needed, over what was already resident.

    max_memory_allocated() is a process-wide counter, so it also counts the model and
    optimiser state sitting in VRAM from earlier sections. Subtracting the baseline is
    what isolates the cost of the loss computation itself.
    """
    if device != "cuda":
        return fn(), float("nan"), float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return out, (peak - baseline) / 1024**2, peak / 1024**2


def measure_time(fn, reps=5):
    """Warm up, then take the fastest of `reps` runs."""
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.time()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    return best


# Sized so the ordinary path fits in VRAM. On an 8 GB card, 8192 rows makes the naive
# path spill into shared system memory over PCIe, which measures Windows' memory manager
# rather than the thing we are comparing.
B_MEM, T_MEM, CHUNK = 4, 1024, 512
N_ROWS = B_MEM * T_MEM

torch.manual_seed(SEED)
x_mem  = torch.randn(N_ROWS, D, device=device)
y_mem  = torch.randint(0, V, (N_ROWS,), device=device)
trunk  = nn.Linear(D, D, bias=False).to(device)     # stands in for the transformer
head_m = nn.Linear(D, V, bias=False).to(device)


def zero():
    trunk.zero_grad(set_to_none=True); head_m.zero_grad(set_to_none=True)


def naive():
    zero()
    hidden = trunk(x_mem)                            # [N_ROWS, D]
    logits = head_m(hidden)                          # [N_ROWS, V]  <- the big one
    loss = F.cross_entropy(logits, y_mem)
    loss.backward()
    return loss.item(), trunk.weight.grad.clone(), head_m.weight.grad.clone()


def chunked():
    zero()
    hidden = trunk(x_mem)
    h_det = hidden.detach().requires_grad_(True)     # cut the graph in two
    total = torch.zeros((), device=device)
    for s0 in range(0, N_ROWS, CHUNK):
        sl = slice(s0, min(s0 + CHUNK, N_ROWS))
        logits_c = head_m(h_det[sl])                 # only CHUNK rows exist at a time
        loss_c = F.cross_entropy(logits_c, y_mem[sl], reduction="sum")
        (loss_c / N_ROWS).backward()                 # this chunk's logits are freed here
        total += loss_c.detach()
    hidden.backward(h_det.grad)                      # one shot into the trunk
    return (total / N_ROWS).item(), trunk.weight.grad.clone(), head_m.weight.grad.clone()


# ---- correctness first: a memory win that changes the answer is not a win
l_n, gt_n, gh_n = naive()
l_c, gt_c, gh_c = chunked()
assert abs(l_n - l_c) < 1e-4, f"loss differs: {l_n} vs {l_c}"
assert torch.allclose(gh_n, gh_c, atol=1e-4), "output-head gradients differ"
assert torch.allclose(gt_n, gt_c, atol=1e-4), "trunk gradients differ"
del gt_n, gh_n, gt_c, gh_c

# ---- then memory and time, measured separately
(_, peak_naive, abs_naive) = measure_peak(naive)
(_, peak_chunk, abs_chunk) = measure_peak(chunked)
t_naive = measure_time(naive)
t_chunk = measure_time(chunked)

logits_mb = N_ROWS * V * 4 / 1024**2
print(f"rows x vocab             : {N_ROWS:,} x {V:,}")
print(f"that logits tensor alone : {logits_mb:,.0f} MB in fp32")
print(f"chunk size               : {CHUNK:,} rows  ({N_ROWS//CHUNK} chunks)")
print()
print(f"{'':<26}{'loss memory':>14}{'process peak':>15}{'time':>11}")
print("-" * 67)
print(f"{'ordinary cross-entropy':<26}{peak_naive:>11,.1f} MB{abs_naive:>12,.0f} MB{t_naive*1000:>8.1f} ms")
print(f"{'chunked cross-entropy':<26}{peak_chunk:>11,.1f} MB{abs_chunk:>12,.0f} MB{t_chunk*1000:>8.1f} ms")
print("-" * 67)
print(f"{'ratio':<26}{peak_naive/peak_chunk:>13.2f}x{abs_naive/abs_chunk:>14.2f}x{t_chunk/t_naive:>9.2f}x")
print()
print(f"for reference, one logits tensor is {logits_mb:,.0f} MB and the ordinary path needs")
print(f"about four of them alive at once: the logits, the log-softmax cross-entropy")
print(f"saves for backward, and the gradient.")
print()
print(f"loss, naive vs chunked   : {l_n:.6f} vs {l_c:.6f}")
print("gradients match on both the head and the trunk")

RESULTS["7. peak memory, ordinary -> chunked"] = (
    f"{peak_naive:,.0f} MB -> {peak_chunk:,.0f} MB attributable to the loss "
    f"(**{peak_naive/peak_chunk:.2f}x** less, {t_chunk/t_naive:.2f}x the time)"
)

del trunk, head_m, x_mem, y_mem
if device == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# **What you should be seeing.** A large ratio on the *attributable* column, a modest time
# penalty, and
# *identical* numbers out of both paths. That last part is the whole point: chunking is a
# pure memory-for-time trade with no approximation anywhere. If your chunked loss differs
# in the fourth decimal, you have a reduction bug - almost always averaging the averages
# instead of summing then dividing once.
#
# This is what buys long context. The session's framing: a big vocabulary puts so much
# pressure on this one tensor that it, not attention, becomes the thing capping your
# sequence length.

# %% [markdown]
# # Part 2 — a second output head
#
# ## 9 — Predicting t+2 alongside t+1
#
# One trunk, two output heads. Head 1 does the usual next-token job. Head 2 reads the
# *same* hidden state and predicts the token after that.
#
# Training is not complicated: at training time you already know every token, so both
# targets are just different slices of the same row. The model is still fed one token at a
# time - the second head changes the loss, not the input.
#
# The payoff is at inference. If head 2 is often right, you can accept two tokens per
# forward pass instead of one, checking the speculation against the next step and throwing
# it away when it misses. That is multi-token prediction, and it is what makes speculative
# decoding work.

# %%
def two_head_loss(m, x, y, y2):
    logits1, logits2 = m(x)
    loss1 = F.cross_entropy(logits1.reshape(-1, V), y.reshape(-1))
    loss2 = F.cross_entropy(logits2.reshape(-1, V), y2.reshape(-1))
    return loss1 + loss2, {"loss1": loss1.item(), "loss2": loss2.item()}


@torch.no_grad()
def eval_two_head(m, split, iters=20):
    m.eval()
    g = torch.Generator().manual_seed(4242)
    a = b = 0.0
    for _ in range(iters):
        x, y, y2 = get_batch(split, generator=g)
        l1, l2 = m(x)
        a += F.cross_entropy(l1.reshape(-1, V), y.reshape(-1)).item()
        b += F.cross_entropy(l2.reshape(-1, V), y2.reshape(-1)).item()
    m.train()
    return a / iters, b / iters


# ---- prove the alignment with strings before training anything
xa, ya, y2a = get_batch("train", B=1)
print("head 1 predicts t+1, head 2 predicts t+2 - from the same position:\n")
print(f"{'pos':>4}  {'input':<16}{'head 1 target':<18}{'head 2 target':<18}")
print("-" * 58)
for p in range(8):
    print(f"{p:>4}  {str(tok(xa[0,p])):<16}{str(tok(ya[0,p])):<18}{str(tok(y2a[0,p])):<18}")
assert (ya[0, 1:] == y2a[0, :-1]).all(), "head 2 targets are not one beyond head 1"
print("\nOK  head 2's target is head 1's target advanced by one more step")

# ---- train
torch.manual_seed(SEED)
mtp = GPT(cfg, tie_weights=True, n_heads_out=2).to(device)
hist2, val2 = [], []


def logged_two_head(m, x, y, y2):
    loss, extras = two_head_loss(m, x, y, y2)
    return loss, extras


t0 = time.time()
torch.manual_seed(SEED)
gtr = torch.Generator().manual_seed(SEED)
opt = make_opt(mtp)
mtp.train()
for step in range(STEPS):
    for grp in opt.param_groups:
        grp["lr"] = lr_at(step, STEPS)
    x, y, y2 = get_batch("train", generator=gtr)
    loss, extras = two_head_loss(mtp, x, y, y2)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
    opt.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        v1, v2 = eval_two_head(mtp, "val")
        hist2.append({"step": step, **extras})
        val2.append({"step": step, "loss1": v1, "loss2": v2})

print(f"\ntrained {STEPS} steps in {time.time()-t0:.1f}s\n")

f_tr, f_va = hist2[-1], val2[-1]
print(f"{'':<12}{'head 1 (t+1)':>14}{'head 2 (t+2)':>14}{'gap':>9}{'sum':>10}")
print("-" * 60)
print(f"{'train':<12}{f_tr['loss1']:>14.4f}{f_tr['loss2']:>14.4f}"
      f"{f_tr['loss2']-f_tr['loss1']:>9.4f}{f_tr['loss1']+f_tr['loss2']:>10.4f}")
print(f"{'val':<12}{f_va['loss1']:>14.4f}{f_va['loss2']:>14.4f}"
      f"{f_va['loss2']-f_va['loss1']:>9.4f}{f_va['loss1']+f_va['loss2']:>10.4f}")
print()
print(f"perplexity  head 1 {math.exp(f_va['loss1']):>9,.0f}"
      f"      head 2 {math.exp(f_va['loss2']):>9,.0f}")

best_val_step = min(val2, key=lambda r: r["loss1"])["step"]
print(f"\nvalidation loss1 bottoms out at step {best_val_step} of {STEPS}")

RESULTS["8. Part 2 - two heads (val)"] = (
    f"head1 {f_va['loss1']:.4f}, head2 {f_va['loss2']:.4f}, "
    f"sum {f_va['loss1']+f_va['loss2']:.4f}  (gap {f_va['loss2']-f_va['loss1']:.4f})"
)

# ---- plot
fig, ax = plt.subplots(figsize=(7.5, 4.2))
s = [h["step"] for h in hist2]
ax.plot(s, [h["loss1"] for h in hist2], color="#58a6ff", lw=1.8, label="head 1 (t+1) train")
ax.plot(s, [h["loss2"] for h in hist2], color="#f778ba", lw=1.8, label="head 2 (t+2) train")
ax.plot([h["step"] for h in val2], [h["loss1"] for h in val2],
        color="#58a6ff", lw=1.4, ls="--", label="head 1 val")
ax.plot([h["step"] for h in val2], [h["loss2"] for h in val2],
        color="#f778ba", lw=1.4, ls="--", label="head 2 val")
ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
ax.set_title("Two output heads: t+1 and t+2 from the same hidden state", color="#e6edf3")
ax.grid(alpha=0.3); ax.legend(frameon=False, labelcolor="#e6edf3", fontsize=8)
fig.tight_layout(); os.makedirs("assets", exist_ok=True)
fig.savefig("assets/two_heads.png", bbox_inches="tight"); plt.close(fig)
print("\nsaved assets/two_heads.png")

# %% [markdown]
# **What you should be seeing, and why.**
#
# Head 2's loss sits above head 1's and **the gap does not close**. Both curves descend
# together, so head 2 is learning perfectly well - it is simply solving a strictly harder
# problem, and no amount of training makes it as easy.
#
# The reason is information, not capacity. Predicting `t+2` means marginalising over the
# token at `t+1` that you have not seen yet:
#
# ```
# H(t+2 | context)  >  H(t+1 | context)
# ```
#
# That inequality is a property of the data, not of the model. A second head with its own
# 12.9M parameters cannot repeal it. The gap you measure is roughly the conditional
# entropy of one token of English - the price of one step of uncertainty.
#
# Two practical notes from the session:
#
# - Training uses one token at a time regardless. The extra head changes the loss, never
#   the input.
# - People often train with 3-4 heads because the auxiliary signal improves the *main*
#   head, then deploy only 1-2, because acceptance rates fall off fast past the second.

# %% [markdown]
# # Part 3 — the beautiful wrong loss curve
#
# ## 10 — Three shifts, one of them correct
#
# The assignment's warning: *a target shift in the incorrect direction can produce a
# beautiful loss curve*. This section makes that concrete by training the same model three
# times, on the same data, from the same seed. The **only** difference is which slice of
# the sequence is used as the target.
#
# | variant | slicing | what it asks the model to do |
# |---|---|---|
# | correct | `logits[:, :-1]` vs `tokens[:, 1:]` | predict the next token |
# | no shift | `logits` vs `tokens` | predict the token it is currently looking at |
# | reversed | `logits[:, 1:]` vs `tokens[:, :-1]` | predict the token it just saw |
#
# Neither broken variant raises an exception. Both are one character away from correct.

# %%
def loss_correct(m, x):
    logits = m(x)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))


def loss_noshift(m, x):
    logits = m(x)
    return F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))


def loss_reversed(m, x):
    logits = m(x)
    return F.cross_entropy(logits[:, 1:].reshape(-1, V), x[:, :-1].reshape(-1))


VARIANTS = [
    ("correct  (t+1)",   loss_correct,  "#58a6ff"),
    ("no shift (t)",     loss_noshift,  "#f778ba"),
    ("reversed (t-1)",   loss_reversed, "#f0883e"),
]

curves = {}
for name, fn, _ in VARIANTS:
    torch.manual_seed(SEED)                       # identical init for every variant
    m = GPT(cfg, tie_weights=True).to(device)
    g = torch.Generator().manual_seed(SEED)       # identical data order for every variant
    opt = make_opt(m)
    m.train()
    hist = []
    t0 = time.time()
    for step in range(STEPS):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, STEPS)
        x, _, _ = get_batch("train", generator=g)
        loss = fn(m, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            hist.append((step, loss.item()))
    curves[name] = hist
    print(f"{name:<18} final loss {hist[-1][1]:>8.4f}   ppl {math.exp(hist[-1][1]):>10,.1f}"
          f"   ({time.time()-t0:.0f}s)")
    del m, opt
    if device == "cuda":
        torch.cuda.empty_cache()

print()
print(f"{'variant':<18}{'final loss':>12}{'vs correct':>14}")
print("-" * 44)
base = curves["correct  (t+1)"][-1][1]
for name, _, _ in VARIANTS:
    fin = curves[name][-1][1]
    tag = "-" if name.startswith("correct") else f"{fin - base:+.4f}"
    print(f"{name:<18}{fin:>12.4f}{tag:>14}")

RESULTS["9. Part 3 - final loss by shift"] = "; ".join(
    f"{n.split()[0]} {curves[n][-1][1]:.4f}" for n, _, _ in VARIANTS
)

# ---- plot
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for name, _, colour in VARIANTS:
    xs = [h[0] for h in curves[name]]
    ys = [h[1] for h in curves[name]]
    ax.plot(xs, ys, color=colour, lw=1.9, label=name)
ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
ax.set_title("The same model, three target shifts", color="#e6edf3")
ax.grid(alpha=0.3); ax.legend(frameon=False, labelcolor="#e6edf3")
fig.tight_layout()
fig.savefig("assets/wrong_shift.png", bbox_inches="tight"); plt.close(fig)
print("\nsaved assets/wrong_shift.png")

# %% [markdown]
# ### And now the strings, which is the only thing that would have caught it

# %%
xs_demo, _, _ = get_batch("train", B=1)
row = xs_demo[0]

pairs = {
    "correct  (t+1)":  [(row[p],     row[p + 1]) for p in range(6)],
    "no shift (t)":    [(row[p],     row[p])     for p in range(6)],
    "reversed (t-1)":  [(row[p + 1], row[p])     for p in range(6)],
}

for name, ps in pairs.items():
    fin = curves[name][-1][1]
    print(f"\n{name}   final loss {fin:.4f}")
    print(f"  {'input':<18}{'target':<18}")
    print("  " + "-" * 34)
    for a, b in ps:
        flag = "   <-- identical" if int(a) == int(b) else ""
        print(f"  {str(tok(a)):<18}{str(tok(b)):<18}{flag}")

# %% [markdown]
# **What you should be seeing.** The two broken variants reach a *lower* loss than the
# correct one, and they get there faster. On a loss curve alone they are the better runs.
#
# They are also worthless:
#
# - **No shift** asks the model to output the token it is already holding. The residual
#   stream carries the embedding straight to the head, so this is close to learning the
#   identity function. It has learned nothing about language.
# - **Reversed** asks for the token one step back. Attention only has to look at position
#   `t-1`, which one head can do. Also nearly free, also useless.
#
# Both would train to a beautiful number, generate garbage, and never raise an exception.
# The only artefact in this entire notebook that catches them in one glance is the string
# table above - `b' shall'` opposite `b' shall'`.
#
# That is why the instruction is *print the strings*, not *check the loss looks sensible*.

# %% [markdown]
# ## 11 — The block the session actually described
#
# Everything above used the plain LayerNorm + GELU block, deliberately: you do not debug a
# loss harness and a new architecture at the same time. Now that every check passes, here
# is the same model built the way the session described it - **RMSNorm, pre-norm, SwiGLU**
# - run through the same gate.
#
# - **RMSNorm** drops the mean-centering that LayerNorm does and only divides by
#   `sqrt(mean(x^2))`, projecting the vector onto the unit sphere. One pass instead of two.
# - **SwiGLU** is not an activation but a design pattern: two matrices instead of one, one
#   branch deciding *which* dimensions survive and the other *by how much*. Because it adds
#   a third matrix, the expansion drops from 4x to about 8/3x to keep the parameter count
#   level.

# %%
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    """(Swish(x @ W1) * (x @ V)) @ W2 - the gate and the value, then project down."""
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg["n_embd"] / 3) // 32 * 32     # ~2.67x, rounded to 32
        self.w1 = nn.Linear(cfg["n_embd"], hidden, bias=False)   # -> Swish, the gate
        self.v  = nn.Linear(cfg["n_embd"], hidden, bias=False)   # -> the value
        self.w2 = nn.Linear(hidden, cfg["n_embd"], bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.v(x))


class ModernBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg["n_embd"]); self.attn = CausalSelfAttention(cfg)
        self.n2 = RMSNorm(cfg["n_embd"]); self.mlp  = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


torch.manual_seed(SEED)
modern = GPT(cfg, tie_weights=True, block_cls=ModernBlock).to(device)

mx, my, _ = get_batch("val", B=8)
modern.eval()
with torch.no_grad():
    m_logits = modern(mx)
    m_init = F.cross_entropy(m_logits[:, :-1].reshape(-1, V), my[:, :-1].reshape(-1))

p_modern = sum(p.numel() for p in modern.parameters())
print(f"{'':<28}{'parameters':>13}{'init loss':>12}{'init ppl':>12}")
print("-" * 65)
print(f"{'LayerNorm + GELU':<28}{p_tied:>13,}{init_loss.item():>12.4f}{init_ppl:>12,.0f}")
print(f"{'RMSNorm + SwiGLU':<28}{p_modern:>13,}{m_init.item():>12.4f}{math.exp(m_init.item()):>12,.0f}")
print("-" * 65)
print(f"{'difference':<28}{p_modern - p_tied:>+13,}")

assert abs(m_init.item() - math.log(V)) < 0.15, "the modern block fails the init gate"
print("\nOK  the modern block passes the same untrained-perplexity gate")

hist_modern = quick_train(modern, STEPS // 2, plain_loss, log_every=EVAL_EVERY)
print(f"trains normally: {hist_modern[0]['train']:.3f} -> {hist_modern[-1]['train']:.3f} "
      f"over {STEPS//2} steps")
print(f"(LayerNorm + GELU over the same {STEPS//2} steps: "
      f"{hist_warm[0]['train']:.3f} -> {hist_warm[-1]['train']:.3f})")

del modern
if device == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# **What you should be seeing.** Nearly the same parameter count, the same initial loss
# near `ln(V)`, and a comparable training curve. That is the point: swapping the block
# internals does not change anything the loss harness cares about. The four lines between
# `hidden` and the scalar are the same four lines either way, and so are all seven checks.

# %% [markdown]
# ## 12 — Every number, in one table
#
# This cell renders the results as markdown, ready to paste into the README. Generating it
# from the same run that produced the numbers is deliberate - it is the only way to keep a
# write-up from drifting away from its own code.

# %%
print(f"Run: {'QUICK' if QUICK_RUN else 'FULL'}  |  {STEPS} steps  |  {device}"
      f"  |  torch {torch.__version__}\n")
print("| # | measurement | value |")
print("|---|---|---|")
for k, v in RESULTS.items():
    num, _, label = k.partition(". ")
    print(f"| {num} | {label} | {v} |")

with open("assets/results.json", "w", encoding="utf-8") as f:
    json.dump({"quick_run": QUICK_RUN, "steps": STEPS, "device": device,
               "torch": torch.__version__, "results": RESULTS}, f, indent=2)
print("\nsaved assets/results.json")
