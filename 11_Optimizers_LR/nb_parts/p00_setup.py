# %% [markdown]
r"""
# Optimizers and Learning Rates, Measured

`loss.backward()` gives every weight a gradient. A gradient says which way a weight should move,
but not how far. `optimizer.step()` decides the distance, and this notebook takes that decision
apart. Each claim below is checked by computation or by a controlled training run, not taken on
trust.

Five questions, each answered with a measured number:

| | question | section |
|---|---|---|
| 1 | Does Adam computed by hand on one weight match PyTorch, quantity by quantity? | §1 |
| 2 | What does bias correction do, and after how many steps does it stop mattering? | §2, §4 |
| 3 | How large is each layer's update relative to its weights, and when does warmup stop shaping it? | §5 |
| 4 | Two runs planned for 300 steps, one cosine and one WSD, both stopped at 200: which model do you keep? | §6 |
| 5 | How does the best learning rate move with width, and what would you use at width 4,096? | §7 |

One principle runs through all of them: **tune both sides before accepting a comparison**. A
well-tuned method measured against a badly tuned baseline produces most of the optimizer claims
that later fail to replicate. §6, §M (Muon against AdamW) and §8 measure how large that effect is.

**How to read it.** Every section gives the intuition, then the math, then code that measures it.
Facts that must hold are written as `check(...)`, which stops the notebook if it is wrong. Outcomes
that are measured rather than guaranteed are written as `observe(...)`, which records whether they
held. Each section ends with one line to carry forward.

**Modes.**

| `MODE` | what runs | time |
|---|---|---|
| `learn` (default) | §R–§2 run live; every training result is replayed from `assets/results.json` (fetched once from GitHub, ~10 MB, if missing) | about a minute |
| `quick` | tiny models and short runs through every code path; writes `assets/quick/` | ~10 min on a GPU |
| `full` | the measured runs behind every number in this notebook | ~5 h of training on an RTX 3070 Laptop (resumable) |

**Contents.**

| part | sections |
|---|---|
| setup | §0 setup and decision rules |
| the arithmetic | §R the rules before Adam · §1 Adam by hand · §2 bias correction |
| training-time choices | §3 the testbed · §4 bias correction in a real model · §5 update-to-weight ratio and warmup · §6 cosine vs WSD · §M Muon vs AdamW |
| scaling | §7 learning rate vs width, and muP |
| method | §8 tuning both sides · §9 answers, limitations, references |
"""

# %% [markdown]
r"""
## §0 Setup and decision rules

The mode switch, the device and precision, a results store that makes long runs resumable, the plot
style, and the thresholds every later section uses to call a result.
"""

# %%
MODE = "learn"          # "learn" | "quick" | "full"   (see the table above)
import os
MODE = os.environ.get("OPT_MODE", MODE)                 # lets a headless run choose the mode
FORCE = os.environ.get("FORCE", "0") == "1"             # ignore stored results and redo them
PRECISION_OVERRIDE = os.environ.get("OPT_PRECISION")    # e.g. "fp16" rehearses the T4 path on any GPU
assert MODE in ("learn", "quick", "full"), MODE
print(f"MODE={MODE}  FORCE={FORCE}  precision override={PRECISION_OVERRIDE}")

# %%
# Install only what is missing (Colab already has all of it).
import importlib.util, subprocess, sys
_need = [pkg for pkg, mod in [("tokenizers", "tokenizers"), ("matplotlib", "matplotlib")]
         if importlib.util.find_spec(mod) is None]
if _need:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_need])

# %%
import contextlib, copy, decimal, gc, json, math, platform, random, re, time, urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from cycler import cycler

HERE = Path.cwd()
DATA = HERE / "data"
ASSETS = HERE / "assets" / ("quick" if MODE == "quick" else "")
RESULTS_FILE = ASSETS / "results.json"
REPO_RAW = "https://raw.githubusercontent.com/rahulni/Indic_LLM/main/11_Optimizers_LR"
for _d in (DATA, ASSETS):
    _d.mkdir(parents=True, exist_ok=True)


def _gpu_available():
    # device_count() first: with CUDA_VISIBLE_DEVICES="" is_available() can still say True
    try:
        return torch.cuda.device_count() > 0 and torch.cuda.is_available()
    except Exception:
        return False


DEV = "cuda" if _gpu_available() else "cpu"
CAP = torch.cuda.get_device_capability() if DEV == "cuda" else (0, 0)
if PRECISION_OVERRIDE:
    AMP = PRECISION_OVERRIDE
elif DEV == "cuda":
    AMP = "bf16" if CAP >= (8, 0) else "fp16"           # a T4 (sm75) has no bf16
else:
    AMP = "fp32"
AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[AMP]

if DEV == "cuda" and platform.system() == "Windows" and "_MEM_CAP" not in globals():
    # Windows (WDDM) silently pages GPU memory to host RAM instead of raising OutOfMemoryError.
    # Capping the allocator at what is free now turns paging back into a real error.
    _free, _total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(_free / _total)
    _MEM_CAP = _free
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def free_gpu():
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def autocast():
    if DEV != "cuda" or AMP == "fp32":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=AMP_DTYPE)


def qk(full, quick):
    """Pick a size by mode: quick mode shrinks everything, learn mode reads the full-run results."""
    return quick if MODE == "quick" else full


_gpu = torch.cuda.get_device_name(0) if DEV == "cuda" else "none"
print(f"device={DEV} ({_gpu})  precision={AMP}  torch={torch.__version__}  python={platform.python_version()}")

# %% [markdown]
r"""
**The results store.** Every training experiment goes through `experiment(name, fn)`. In `quick`
and `full` mode it runs `fn` once, rounds the result to 5 significant figures, and saves it to
`results.json` straight away. An interrupted run therefore resumes where it stopped. In `learn`
mode nothing trains, and the stored result is returned (fetched from GitHub if the file is
missing).
"""

# %%
def _clean(x, sig=5):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        x = x.item()
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return float(f"{x:.{sig}g}") if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v, sig) for v in x]
    return x


def save_results():
    if MODE == "learn":
        return
    tmp = RESULTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(R, separators=(",", ":")), encoding="utf-8")
    for attempt in range(20):                     # Windows: an editor or virus scan can hold the file for a moment
        try:
            tmp.replace(RESULTS_FILE)
            return
        except PermissionError:
            time.sleep(0.5)
    tmp.replace(RESULTS_FILE)


def _load_results():
    if MODE == "learn":
        path = HERE / "assets" / "results.json"
        if not path.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"{REPO_RAW}/assets/results.json", path)
            except Exception as e:                              # offline: the live sections still run
                print("could not fetch the stored results:", e)
                return {}
        return json.loads(path.read_text(encoding="utf-8"))
    if FORCE or not RESULTS_FILE.exists():
        return {}
    return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))


R = _load_results()
ANS = R.setdefault("answers", {})                              # the measured answers, read by make_readme.py
_RAN = []                                                       # names run (not reused) in this session


def experiment(name, fn):
    if MODE == "learn":
        if name not in R:
            print(f"[{name}] is not in the stored results; skipped")
        return R.get(name)
    if name in R:
        return R[name]
    t0 = time.time()
    out = _clean(fn())
    R[name] = out
    R.setdefault("_seconds", {})[name] = round(time.time() - t0, 1)
    save_results()
    _RAN.append(name)
    print(f"  ran [{name}] in {time.time() - t0:.0f} s")
    return out


def check(cond, msg):
    assert cond, f"CHECK FAILED: {msg}"
    print(f"✓ {msg}")


def observe(key, cond, msg):
    """Record a measured outcome. It never stops the notebook; the README reports whether it held."""
    R.setdefault("_observations", {})[key] = dict(held=bool(cond), msg=msg)
    print(("✓ observed: " if cond else "✗ not observed: ") + msg)
    save_results()


def env_info():
    info = dict(date=time.strftime("%Y-%m-%d"), torch=torch.__version__, python=platform.python_version(),
                device=DEV, precision=AMP, platform=platform.platform())
    if DEV == "cuda":
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gib=round(p.total_memory / 2**30, 2))
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.sm,clocks.max.sm,temperature.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            drv, sm, smax, temp = [s.strip() for s in q.strip().split(",")]
            info.update(driver=drv, sm_clock_mhz=int(sm), sm_clock_max_mhz=int(smax), gpu_temp_c=int(temp))
        except Exception:
            pass
    if platform.system() == "Windows":
        try:
            ps = "(Get-CimInstance Win32_Battery).BatteryStatus; powercfg /getactivescheme"
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                 capture_output=True, text=True, timeout=20).stdout.splitlines()
            info["on_ac_power"] = out[0].strip() == "2" if out else None
            info["power_plan"] = (re.findall(r"\((.*)\)", out[-1]) or [None])[0] if out else None
        except Exception:
            pass
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                            text=True, timeout=10, cwd=HERE).stdout.strip() or None
    except Exception:
        pass
    return info


if MODE != "learn":
    R.setdefault("env", _clean(env_info()))         # the environment of the first (measuring) run is kept
    R["env_last"] = _clean(env_info())              # and the latest session's is recorded next to it
    save_results()
def show_env(env, title):
    keys = ("date", "gpu", "precision", "torch", "python", "driver", "on_ac_power", "power_plan", "git_commit")
    print(title + ("  " + "  ·  ".join(f"{k}: {env[k]}" for k in keys if env and k in env) if env else "  (none stored)"))


show_env(R.get("env"), "environment of the measured run:" if MODE == "learn" else "environment:")

# %%
# Plot style: dark surface, one y-axis per panel, direct labels instead of legends where possible.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#0d1117", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
C = dict(blue="#3987e5", orange="#d95926", aqua="#199e70", yellow="#c98500",
         magenta="#d55181", green="#008300", violet="#9085e9", red="#e66767")
SERIES = [C[k] for k in ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red")]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titlelocation": "left",
    "legend.frameon": False, "legend.labelcolor": INK2, "axes.prop_cycle": cycler(color=SERIES),
    "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight", "axes.axisbelow": True})


def label_end(ax, x, y, text, color, dx=4, dy=0):
    """Queue a direct label at a line's end; finish() places all of an axis's labels without overlaps."""
    if not hasattr(ax, "_end_labels"):
        ax._end_labels = []
    ax._end_labels.append((x, y, text, color, dx, dy))


def _place_labels(fig, gap_px=12):
    fig.canvas.draw()                                            # limits and scales are final now
    k = fig.dpi / 72
    for ax in fig.axes:
        items = getattr(ax, "_end_labels", [])
        rows = []
        for x, y, text, color, dx, dy in items:
            if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
                continue
            px, py = ax.transData.transform((x, y))
            rows.append([py + dy * k, py, x, y, text, color, dx])
        rows.sort(key=lambda r: r[0])
        for i in range(1, len(rows)):                            # push apart labels closer than gap_px
            rows[i][0] = max(rows[i][0], rows[i - 1][0] + gap_px)
        for ty, py, x, y, text, color, dx in rows:
            ax.annotate(text, (x, y), xytext=(dx * k, ty - py), textcoords="offset pixels", color=color,
                        fontsize=8.5, va="center", annotation_clip=False)
        ax._end_labels = []


def finish(fig, name):
    _place_labels(fig)
    if MODE != "learn":
        fig.savefig(ASSETS / f"{name}.png")
    plt.show()
    plt.close(fig)

# %% [markdown]
r"""
### Decision rules, fixed before any measurement

Several answers below need a threshold: when a difference "stops mattering", when warmup "stops
changing" a ratio, which model to "keep". Choosing a threshold after seeing the data invites
choosing the one that gives the nicer answer. So the rules are written here, before any training
run. They are stored in `results.json` on the first full run, and every later run `check`s that
they have not changed.
"""

# %%
RULES = {
    "bias_correction_stop": dict(
        rule="The per-step multiplier (1-b1^t)/sqrt(1-b2^t) is within p of 1 for every later step. "
             "Headline p = 1%; 5% and 10% also reported. In a real model: the 20-step EMA of the paired "
             "loss gap stays inside +-2 seed s.d. from that step on.",
        p=[0.01, 0.05, 0.10], headline=0.01, ema_steps=20, band_sd=2.0),
    "warmup_stop": dict(
        rule="For each tensor, the 9-step rolling median of its update/weight ratio stays within p of the "
             "no-warmup run's ratio from that step on. Headline p = 10%; 5% and 20% for sensitivity.",
        p=[0.05, 0.10, 0.20], headline=0.10, window=9),
    "keep_at_200": dict(
        rule="Both runs are stopped at step 200. For each, take the best finished model reachable with at "
             "most 20% more compute (cosine: its step-200 or step-240 model; WSD: its step-200 model or a "
             "40-step cooldown from it). Keep the run whose option has the lower paired-mean validation "
             "loss. If they are within 2 paired standard errors, keep WSD, because it can still continue.",
        extra_compute=0.2, tie_se=2.0),
    "lr_minimum": dict(
        rule="Vertex of a parabola in log2(lr) through the five sqrt(2)-spaced points around the argmin of "
             "the seed-mean final loss. 90% interval from 2,000 bootstrap resamples over seeds.",
        points=5, boot=2000, ci=0.90),
    "confidence_4096": dict(
        rule="High: the width-2,048 prediction is off by < 0.5 octave AND the 90% interval at 4,096 is "
             "narrower than 1 octave. Medium: off by < 1 octave. Otherwise low.",
        high_err_oct=0.5, high_ci_oct=1.0, medium_err_oct=1.0),
    "optimizer_claim": dict(
        rule="A speedup is claimed only if the tuned-vs-tuned paired loss difference exceeds 2 standard "
             "errors. Speedup = steps the tuned AdamW run needs to reach its final loss, divided by the "
             "steps the challenger needs to reach that same loss.",
        se=2.0),
}
if "_rules" in R:
    check(R["_rules"] == _clean(RULES), "decision rules are unchanged since the first measurement")
elif MODE != "learn":
    R["_rules"] = _clean(RULES)
    save_results()
    print("decision rules stored before the first measurement")
for k, v in RULES.items():
    print(f"- {k}: {v['rule']}")
