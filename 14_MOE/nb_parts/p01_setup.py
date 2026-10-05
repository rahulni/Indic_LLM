from . import md, code

CELLS = [
md(r'''
---
## 1 · Setup

Pick a **mode** in the next cell, then *Run all*.

| `MODE` | Needs | Time | What happens |
|---|---|---|---|
| `"learn"` | CPU is fine | ~1 min | Runs the refresher, the toy, and every gate. Training cells **replay** the saved results (`assets/results.json`, fetched from GitHub if it is not next to the notebook) and only draw. |
| `"quick"` | GPU | ~5 min | Every code path at 1/20 scale. Writes to `assets-quick/` and `ckpt-quick/`, never to `assets/`. |
| `"full"` | GPU | ~30 min on an RTX 3070 Laptop (measured), ~1 h on a Colab T4 (estimate) | The real experiment. Writes `assets/`. |

On Colab: *Runtime → Change runtime type → T4 GPU* before `quick`/`full`. Each finished stage is
checkpointed, so if Colab disconnects, re-run all and finished stages are loaded instead of retrained
(set `USE_DRIVE = True` to keep checkpoints on Google Drive across Colab restarts).
'''),
code(r'''
import os, sys, math, json, time, random, platform, subprocess, urllib.request, copy, re
from dataclasses import dataclass, asdict, field
from pathlib import Path

MODE = os.environ.get("MOE_MODE", "learn").lower()   # <- "learn" | "quick" | "full"
USE_DRIVE = False                                    # Colab only: keep data/checkpoints on Google Drive
RESUME = os.environ.get("MOE_RESUME", "1") == "1"    # reuse finished stages instead of retraining
assert MODE in ("learn", "quick", "full"), MODE
IN_COLAB = "google.colab" in sys.modules
COMPUTE = MODE != "learn"                            # learn mode never trains, it replays

if COMPUTE:
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tokenizers"])

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, Markdown

HERE = Path.cwd()
WORK = HERE
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = Path("/content/drive/MyDrive/moe14")
SUFFIX = "-quick" if MODE == "quick" else ""
DATA = WORK / "data"
CKPT = WORK / f"ckpt{SUFFIX}"
ASSETS = HERE / f"assets{SUFFIX}"
for p in (DATA, CKPT, ASSETS):
    if COMPUTE:
        p.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
if COMPUTE and DEVICE != "cuda":
    raise RuntimeError("MODE='quick'/'full' needs a GPU. On CPU use MODE='learn'.")
if DEVICE == "cuda":
    cap = torch.cuda.get_device_capability()
    PREC = "bf16" if cap[0] >= 8 else "fp16"         # T4 is sm75: no native bf16
else:
    PREC = "fp32"
PREC = os.environ.get("MOE_PRECISION", PREC)          # e.g. force "fp16" to test the T4 path
AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[PREC]

if DEVICE == "cuda" and platform.system() == "Windows":
    # WDDM pages VRAM to host RAM instead of raising OOM, which makes "it fits" a fiction.
    # Capping the allocator turns paging back into a real OutOfMemoryError.
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(free / total * 0.97)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

seed_all(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

GPU = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
print(f"MODE={MODE}  device={GPU}  precision={PREC}  torch={torch.__version__}  colab={IN_COLAB}")
'''),
md(r'''
**Plot style.** Every chart uses one dark surface, hairline grids, 2px lines and a fixed categorical
order (validated for colour-vision deficiency against this surface). Colour always follows the
entity: *dense* is blue and *MoE* is orange everywhere in the notebook.
'''),
code(r'''
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#0d1117", "#e6edf3", "#c3c2b7", "#898781", "#21262d", "#30363d"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
DENSE_C, MOE_C = SERIES[0], SERIES[1]
CRITICAL = "#d03b3b"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#0d1117", "#104281", "#256abf", "#5598e7", "#9ec5f4", "#cde2fb"])
DIV = LinearSegmentedColormap.from_list("div", ["#3987e5", "#383835", "#e66767"])
DIV.set_bad(SURFACE)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "legend.frameon": False,
    "legend.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "font.size": 9.5, "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
})

def savefig(fig, name):
    if COMPUTE:
        fig.savefig(ASSETS / f"{name}.png")
    plt.show()
'''),
md(r'''
**Results store.** Every compute cell writes into one dict, `R`, and every plot cell reads only from
`R`. That is what lets `learn` mode replay the whole notebook without a GPU: it loads `R` from
`assets/results.json` and skips the compute.
'''),
code(r'''
RAW_URL = "https://raw.githubusercontent.com/rahulni/Indic_LLM/main/14_MOE/assets/results.json"

def load_results():
    f = ASSETS / "results.json"
    if f.exists():
        return json.loads(f.read_text())
    if MODE == "learn":
        try:
            with urllib.request.urlopen(RAW_URL, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"(no saved results found locally or at GitHub: {e}) - plots will be skipped")
    return {}

R = load_results() if (RESUME or MODE == "learn") else {}
if MODE != "learn" and R.get("meta", {}).get("mode") not in (None, MODE):
    R = {}                                    # never mix quick and full numbers
R.setdefault("meta", {})
R.setdefault("runs", {})
if COMPUTE:
    R["meta"].update(mode=MODE, gpu=GPU, precision=PREC, torch=torch.__version__,
                     python=platform.python_version(), complete=False)

def save_results():
    if COMPUTE:
        tmp = ASSETS / "results.json.tmp"
        tmp.write_text(json.dumps(R, separators=(",", ":")))
        tmp.replace(ASSETS / "results.json")

def have(*keys):
    """True when a result already exists, so a finished stage is not recomputed."""
    d = R
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return False
        d = d[k]
    return True

GATES = {}
def gate(name, ok, detail=""):
    """An exact check that must hold before anything trains. A failure stops the notebook."""
    GATES[name] = bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    assert ok, name

HOTPLUG = WORK / "hotplug.json"
def read_hotplug():
    """Live knobs, re-read every 50 steps: {"gamma": .., "seq_alpha": .., "lr_mult": .., "freeze_router": bool}."""
    try:
        return json.loads(HOTPLUG.read_text())
    except Exception:
        return {}

print("results loaded:", sorted(k for k in R if k != "meta" and R[k]) or "none yet")
'''),
md(r'''
### Configuration: every size and budget lives here

The token budgets are set so a free Colab T4 finishes `full` in about an hour. The model is
deliberately the same shape in every stage; only the feed-forward block changes.
'''),
code(r'''
@dataclass
class GPTConfig:
    vocab: int = 4096
    d: int = 384          # hidden size
    n_layer: int = 8
    n_head: int = 6       # query heads
    n_kv: int = 2         # key/value heads (GQA: 3 query heads share one K/V head)
    ctx: int = 256
    ffn: int = 1536       # dense SwiGLU inner width F (4 x d)

@dataclass
class MoEConfig:
    n_exp: int            # routed experts per layer, E
    width: int            # inner width of one expert, w
    top_k: int            # experts chosen per token, k
    shared: int = 0       # inner width of the always-on shared expert (0 = none)
    score: str = "sigmoid"        # "sigmoid" | "softmax" | "sqrt_softplus"
    route_scale: float = None     # multiplies the renormalised top-k weights; None -> k
    balance: str = "bias"         # "bias" (loss-free) | "none"
    gamma: float = 1e-3           # bias update speed
    seq_alpha: float = 1e-4       # sequence-level auxiliary loss weight
    switch_alpha: float = 0.0     # Switch-style batch auxiliary loss weight
    family: int = 1               # clones per family after growth (for diagnostics only)

GCFG = GPTConfig()
SCALE = {"full": 1.0, "quick": 0.05, "learn": 1.0}[MODE]
BUDGET = dict(
    batch=64, lab_batch=16, micro_batch=32,    # sequences per step (x 256 tokens); 64 = 2 micro-batches
    T1=25e6, T2=8e6, T3=12e6,                  # dense stage, MoE-8 stage, MoE-32 stage (tokens)
    lab=1.2e6,                                 # tokens per lab run
    peak_lr=1.2e-3, warmup=200, rewarm=50,     # WSD schedule
    eval_every=50, lab_eval_every=25,
    val_batches=16, final_val_batches=64,      # x 32 sequences
    train_mb=240, bpe_mb=30,                   # MB of TinyStories text to stream / to train BPE on
)
if MODE == "quick":
    BUDGET.update(T1=1.3e6, T2=0.45e6, T3=0.6e6, lab=0.12e6, warmup=20, rewarm=10, eval_every=10,
                  lab_eval_every=5, val_batches=4, final_val_batches=8, train_mb=12, bpe_mb=6)
TOK_STEP, LAB_TOK_STEP = BUDGET["batch"] * GCFG.ctx, BUDGET["lab_batch"] * GCFG.ctx
S1 = int(BUDGET["T1"] // TOK_STEP)
S2 = int(BUDGET["T2"] // TOK_STEP)
S3 = int(BUDGET["T3"] // TOK_STEP)
SLAB = int(BUDGET["lab"] // LAB_TOK_STEP)
print(f"steps: dense {S1} | MoE-8 {S2} | MoE-32 {S3} | dense control {S2 + S3} | each lab run {SLAB}")
print(f"tokens: T1 {S1 * TOK_STEP / 1e6:.1f}M -> T2 {(S1 + S2) * TOK_STEP / 1e6:.1f}M -> T3 {(S1 + S2 + S3) * TOK_STEP / 1e6:.1f}M")
'''),
]
