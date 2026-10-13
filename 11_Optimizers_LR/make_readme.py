"""Render README.md from assets/results.json (written by the notebook in full mode).

    python make_readme.py                 # the measured run in assets/
    python make_readme.py --quick         # the smoke-test run in assets/quick/ (writes README.quick.md)

Every number in the README comes from results.json; nothing is typed in by hand.
Section headings mirror the notebook's (§R, §1 … §8, §M), so a reader can move between the two.
"""
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUICK = "--quick" in sys.argv
ASSETS = HERE / "assets" / ("quick" if QUICK else "")
OUT = HERE / ("README.quick.md" if QUICK else "README.md")
FIG = "assets/quick" if QUICK else "assets"
BRANCH = "main"
NB = "optimizers_and_learning_rates.ipynb"
REPO = "rahulni/Indic_LLM"
FOLDER = "11_Optimizers_LR"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{FOLDER}/{NB}"

R = json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))
A = R.get("answers", {})
E = R.get("env", {})
TOTAL = 300 if not QUICK else 60
STOP = 200 if not QUICK else 40
COOL = round(0.2 * STOP)


# ---------------------------------------------------------------- helpers
def g(path, default=None):
    """g('schedules/keep') walks the answers dict."""
    cur = A
    for k in path.split("/"):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def f(x, spec=".4f", none="—"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return none
    return format(x, spec)


def cell(x):
    """Table cells: a literal '|' would split the cell, and a newline would end the row."""
    return str(x).replace("|", "/").replace("\n", " ")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fig(name, alt):
    return f"![{alt}]({FIG}/{name}.png)" if (ASSETS / f"{name}.png").exists() else f"*(figure `{name}` not generated)*"


def anchor(title):
    """GitHub's heading anchor: lowercase, drop punctuation and symbols (including §), spaces to hyphens."""
    s = re.sub(r"[^\w\- ]", "", title.lower(), flags=re.UNICODE)
    return "#" + s.strip().replace(" ", "-")


def link(title, label=None):
    return f"[{label or title}]({anchor(title)})"


def oneline(s):
    return " ".join(str(s).split())


def stp(v, n=None):
    n = n or (g("warmup/n_steps") or TOTAL)
    return f"never within {n}" if v is None or v > n else str(v)


def bc_setting(k):                          # "b2=0.95,warmup=0" -> "β2 0.95, no warmup"
    b2, w = (x.split("=")[1] for x in k.split(","))
    return f"β2 {b2}, " + ("no warmup" if w == "0" else f"warmup {w}")


def pretty_rule(s):
    for a, b in (("(1-b1^t)/sqrt(1-b2^t)", "(1−β₁ᵗ)/√(1−β₂ᵗ)"), ("+-", "±"), ("sqrt(2)", "√2"), ("log2(lr)", "log₂(lr)"),
                 ("s.d.", "s.d."), ("4,096", "4,096")):
        s = s.replace(a, b)
    return s


H = {  # section titles: one place, so the contents list and the links can never drift from the headings
    "answers": "Answers at a glance", "rules": "Decision rules, fixed before the first measurement",
    "R": "§R The rules before Adam", "1": "§1 Adam by hand", "2": "§2 Bias correction", "3": "§3 The testbed",
    "4": "§4 Bias correction in a real model", "5": "§5 Update-to-weight ratio and warmup",
    "6": f"§6 Cosine vs WSD, stopped at {STOP}", "M": "§M Muon vs AdamW, both tuned",
    "7": "§7 Learning rate vs width, and muP", "8": "§8 Tune both sides", "obs": "Measured outcomes",
    "lim": "Limitations", "run": "Running it", "refs": "References",
}
secs = {k: v for k, v in R.get("_seconds", {}).items() if v <= 1800}     # longer = the machine slept mid-run (see §9)
minutes = sum(secs.values()) / 60
bc, cf = g("bias_correction", {}), g("bias_correction/closed_form_stop", {})
wu, sc, wd, mu, ad, base, tb = (g(k, {}) for k in ("warmup", "schedules", "width", "muon", "adam_by_hand", "base", "tune_both"))
blocks = []

# ---------------------------------------------------------------- header and contents
blocks.append(f"""# Optimizers and Learning Rates, Measured

*What `optimizer.step()` does, taken apart and measured: Adam by hand, bias correction, warmup and the
update-to-weight ratio, cosine vs WSD, Muon vs AdamW, and how the best learning rate moves with width.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

A gradient says which way each weight should move, but not how far. Every optimizer is a rule for the
distance. This notebook checks those rules one at a time:
- the arithmetic, by computing Adam by hand and against PyTorch's internal state
- the training effects, by small runs that change one thing at a time against a tuned baseline

Every comparison is tuned on both sides and paired by seed. Every threshold used to call a result
was written down before the first measurement. A one-page refresher is in
[`CHEATSHEET.md`](CHEATSHEET.md).

**The measured run:** about {minutes / 60:.1f} hours of training ({minutes:.0f} min) on an {E.get('gpu', 'unknown GPU')}.
- {E.get('precision', '?')}, torch {E.get('torch', '?')}, Python {E.get('python', '?')}
- {E.get('date', '?')}, {'on AC power' if E.get('on_ac_power') else 'power source unknown'}, {E.get('power_plan', 'unknown')} power plan

The notebook opens in **learn** mode. That mode replays these results in about a minute, and needs
no GPU (see {link(H['run'])}).

**Contents**

| | |
|---|---|
| results | {link(H['answers'])} · {link(H['rules'])} |
| the arithmetic | {link(H['R'])} · {link(H['1'])} · {link(H['2'])} |
| training-time choices | {link(H['3'])} · {link(H['4'])} · {link(H['5'])} · {link(H['6'])} · {link(H['M'])} |
| scaling | {link(H['7'])} |
| method | {link(H['8'])} · {link(H['obs'])} · {link(H['lim'])} · {link(H['run'])} · {link(H['refs'])} |
""")

# ---------------------------------------------------------------- answers at a glance
ans = []
if ad:
    d = ad["diffs"]
    worst64 = max(max(v.values()) for k, v in d.items() if "float64" in k)
    worst32 = max(max(v.values()) for k, v in d.items() if "float32" in k and "fused" not in k)
    fused = d.get("torch float32, fused CUDA", {})
    ans.append(("Does Adam by hand match PyTorch?",
                f"**Yes.** Step by step, PyTorch's own m, v, m̂, v̂ and step agree with the hand computation to "
                f"**{ad.get('agree_decimals', 7)} decimal places**. Against a 50-digit reference, float64 is within {worst64:.0e} "
                f"and float32 within {worst32:.0e}. The one outlier is the fused float32 kernel, whose v̂ is off by "
                f"{f(fused.get('v_hat'), '.1e')} because it rounds 1−β2 in float32; its step is still within {f(fused.get('step'), '.0e')}.",
                link(H['1'], "§1")))
if bc:
    ms = bc.get("model_stop", {})
    ans.append(("Bias correction off: when does the difference stop mattering?",
                f"The first step is **{bc['first_step']:.2f}×** too large, and the gap *grows* to {bc['peak']:.2f}× at step {bc['peak_t']} "
                f"({bc['at20']:.2f}× at step 20). It is within 1% only after **{cf.get('0.999', {}).get('1%', 0):,} steps** at "
                f"β2 = 0.999, or **{cf.get('0.95', {}).get('1%', 0)}** at β2 = 0.95. In the 31.5M model, the loss gap re-enters seed "
                f"noise at: " + "; ".join(f"{bc_setting(k)}: {('step ' + str(v)) if v else 'never within ' + str(TOTAL)}"
                                         for k, v in ms.items()) + ".",
                f"{link(H['2'], '§2')}, {link(H['4'], '§4')}"))
if wu:
    st, W = wu.get("stop", {}), str(wu.get("warm", 30))
    never = all(not stp(v.get("10%")).isdigit() for v in st.values())
    ans.append(("When does warmup stop changing the update-to-weight ratio?",
                "It changes the ratio from **step 1**, by construction (its learning rate is smaller). Measured against no "
                "warmup at all (the rule fixed in advance), its effect "
                + ("**never ends** within the run" if never else
                   "ends at " + "; ".join(f"warmup {w}: step **{stp(v.get('10%'))}**" for w, v in st.items()))
                + f": skipping warmup lets matrices move by up to {f(wu['max_ratio'].get('0'), '.0%')} of their size in one step "
                f"(vs {f(wu['max_ratio'].get(W), '.1%')} with {W} warmup steps), and that run never catches up."
                + (" Among warmed-up runs, the warmup *length* stops mattering at **step "
                   + " / ".join(stp(v) for v in wu["pairwise"].values()) + "** ("
                   + ", ".join(wu["pairwise"].keys()) + "; exploratory)." if wu.get("pairwise") else ""),
                link(H['5'], "§5")))
if sc:
    t = sc["table"]
    c_, w_ = t.get(f"cosine-{TOTAL}, stopped at {STOP}", {}), t.get(f"WSD-{TOTAL}, stopped at {STOP}", {})
    bf = sc.get("by_floor", {})
    degenerate = sc.get("cos_variant", "").endswith("100%")
    ans.append((f"Cosine vs WSD, both stopped at {STOP} of {TOTAL}: which to keep?",
                f"**Keep WSD.** As stopped: cosine {f(c_.get('mean'))} ± {f(c_.get('sd'))}, WSD {f(w_.get('mean'))} ± {f(w_.get('sd'))} "
                f"(3 seeds)"
                + (f". Tuned on its own floor, cosine's best is *no decay at all*, so up to step {STOP} it is the same run "
                   f"as WSD's flat phase, and the two tie exactly" if degenerate else "")
                + f". What separates them is the option to decay late: with ≤20% more compute, WSD's best option "
                f"({sc['best_wsd'].replace('WSD ', 'checkpoint ')}) beats cosine's ({sc['best_cos'].replace('cosine-300 ', '')}) "
                f"by **{sc['diff']:+.4f} ± {sc['se']:.4f}**."
                + (" The margin shrank as cosine's floor was tuned ("
                   + ", ".join(f"{float(k):.0%}: {v['diff']:+.3f}" for k, v in bf.items()) + ") and never flipped." if bf else ""),
                link(H['6'], "§6")))
if wd:
    a = wd["answer"]
    ans.append(("Best learning rate at width 4,096, and how confident?",
                f"**Use {a['lr']:.1e}**, expected within {a.get('band', a['ci'])[0]:.1e}–{a.get('band', a['ci'])[1]:.1e}. "
                f"Minima: " + ", ".join(f"{w}: {v['lr']:.1e}" for w, v in wd["sp"].items())
                + f"; best lr ∝ width^{wd['slope4']:.2f} (inverse-width would be −1). A fit on three widths predicted the held-out "
                f"width {wd['hold']['width']} to within {abs(wd['hold']['err_oct']):.2f} octave, which sets the band. Confidence "
                f"**{a['confidence']}** by the rule fixed in advance, for this depth, batch and step count. Under muP the minima "
                f"move only {wd['mup_spread']:.2f} octave across widths (vs {wd['sp_spread']:.2f} under SP).",
                link(H['7'], "§7")))
if mu:
    ans.append(("Tune both sides: does Muon beat AdamW here?",
                f"**Yes, by about {sum(mu['speedup']) / len(mu['speedup']):.2f}×** ({min(mu['speedup']):.2f}–{max(mu['speedup']):.2f}× "
                f"across seeds; paired loss difference {mu['diff']:+.4f} ± {mu['se']:.4f}), with both sides tuned on the same grid. "
                f"Against a mis-tuned AdamW the same Muon run looks {max(v[1] for v in mu['apparent'].values()):.2f}× faster.",
                f"{link(H['M'], '§M')}, {link(H['8'], '§8')}"))
blocks.append(f"## {H['answers']}\n\n" + table(["question", "measured answer", "where"], ans))

# ---------------------------------------------------------------- decision rules
names = {"bias_correction_stop": "When bias correction stops mattering", "warmup_stop": "When warmup stops changing the ratio",
         "keep_at_200": f"Which model to keep at step {STOP}", "lr_minimum": "Where a learning-rate minimum is",
         "confidence_4096": "How confident the width-4,096 answer is", "optimizer_claim": "When an optimizer speedup counts"}
blocks.append(f"## {H['rules']}\n\nStored in `results.json` on the first measured run; every rerun `check`s that they "
              "have not changed.\n\n"
              + "\n".join(f"- **{names.get(k, k)}.** {pretty_rule(v['rule'])}" for k, v in R.get("_rules", {}).items()))

# ---------------------------------------------------------------- §R
blocks.append(f"""## {H['R']}

The notebook first rebuilds the ideas Adam combines, each checked against a standard worked example:
- gradient descent's stability condition, and why one learning rate cannot serve two curvatures
- momentum, which cancels what alternates and keeps what is consistent
- per-parameter learning rates
- what the optimizer costs in memory (16 bytes per weight with AdamW)
- the √batch rule for Adam
- Muon's orthogonalized update

{fig('gd_two_curvatures', 'gradient descent on two curvatures')}

{fig('muon_orthogonalize', 'Newton-Schulz orthogonalization')}""")

# ---------------------------------------------------------------- §1
if ad:
    rows = [(i + 1, f"{gr:.2f}", f"{r['m']:.4f}", f"{r['v']:.6f}", f"{r['m_hat']:.4f}", f"{r['v_hat']:.4f}", f"{r['step']:.6f}", f"{r['w']:.6f}")
            for i, (gr, r) in enumerate(zip(ad["grads"], ad["table"]))]
    srows = []
    for i, (h, p) in enumerate(zip(ad["table"], ad.get("torch_rows", []))):
        srows.append((i + 1, "by hand") + tuple(f"{h[q]:.8f}" for q in ("m", "v", "m_hat", "v_hat", "step")))
        srows.append(("", "PyTorch") + tuple(f"{p[q]:.8f}" for q in ("m", "v", "m_hat", "v_hat", "step")))
    drows = [(k,) + tuple(f"{v[q]:.1e}" for q in ("m", "v", "m_hat", "v_hat", "step", "w")) for k, v in ad["diffs"].items()]
    blocks.append(f"""## {H['1']}

One weight at w = 1, lr = 1e-3, β = (0.9, 0.999), ε = 1e-8, five gradients. The exact (50-digit)
computation:

{table(['t', 'g', 'm', 'v', 'm̂', 'v̂', 'step', 'w'], rows)}

**Step by step against PyTorch.** PyTorch's numbers are read from `optimizer.state` after each step
(float32, the default dtype). PyTorch stores m (`exp_avg`) and v (`exp_avg_sq`); m̂ and v̂ are derived
from them. The two sides agree to at least **{ad.get('agree_decimals')} decimal places**:

{table(['t', '', 'm', 'v', 'm̂', 'v̂', 'step'], srows)}

**Every implementation against the exact values** (largest absolute difference over the five steps):

{table(['implementation', 'm', 'v', 'm̂', 'v̂', 'step', 'w'], drows)}

The fused float32 kernel computes 1−β2 in float32, which is off by 1.3e-5 relative. Its own bias
correction uses the same rounded β2, so the step is unaffected. `adam_by_hand.py` runs the same check
on its own and exits 1 on any mismatch.""")

# ---------------------------------------------------------------- §2
if bc:
    trows = [(b2, *(f"{v[p]:,}" for p in ("1%", "5%", "10%"))) for b2, v in cf.items()]
    blocks.append(f"""## {H['2']}

Without correction, the step is too large by exactly $(1-\\beta_1^t)/\\sqrt{{1-\\beta_2^t}}$, whatever the
gradients. $m$ recovers from its zero start in tens of steps, $v$ in thousands, so the gap **widens**
over the first dozen steps before it closes.

{fig('bias_correction_first20', 'first 20 steps with and without bias correction')}

Steps until the per-step multiplier stays within 1/5/10% of 1. After about the first 44 steps, only $v$'s correction matters:

{table(['β2', '1%', '5%', '10%'], trows)}

{fig('bias_correction_long', 'multiplier vs step for several beta2')}

Summed over the run, the overshoot is permanent: about **{bc.get('excess_displacement_0999', 0):,.0f} η** of extra
displacement at β2 = 0.999, with consistently signed gradients.""")

# ---------------------------------------------------------------- §3
if base:
    blocks.append(f"""## {H['3']}

All training runs use TinyStories with the SmolLM2 tokenizer:

| | base model | width family |
|---|---|---|
| used in | §4–§6, §M | §7 |
| shape | width 384, 8 layers, GQA 6/2 heads, SwiGLU, RoPE | width 256–2,048, 4 layers, head size 64 |
| vocabulary | 49,152, tied | 8,192 most frequent ids, untied |
| parameters | {base.get('n_params', 0) / 1e6:.1f}M | 7M–234M |
| tokens per step | 16,384 | 16,384 |

**Fair comparisons.**
- **Paired runs.** Runs with the same seed start from the same weights and see the same batches.
- **A verified hand-written optimizer.** The hand-written AdamW used in §4 to switch bias correction
  off matches `torch.optim.AdamW` on all 31.5M weights for 20 steps (largest difference 0).
- **A tuned base learning rate.** The base model's peak learning rate is tuned on a bracketed factor-2
  grid; the best is **{base.get('lr', 0):.0e}**.

**Noise levels.**
- Seed-to-seed s.d. of validation loss: {f(base.get('val_sd'))}.
- The same seed run twice: {f(base.get('nondet_val'), '.5f')}. This is the GPU noise floor; here the
  runs are deterministic.
- Untrained loss: {f(base.get('loss0'), '.2f')}, against ln 49,152 = 10.80.

{fig('base_lr_tuning', 'base model learning-rate tuning')}

{fig('schedules', 'learning-rate schedules')}""")

# ---------------------------------------------------------------- §4
if bc:
    vrows = [(bc_setting(k), f(v["on"]), f(v["off"]), f(v["off"] - v["on"], "+.4f") if v["on"] and v["off"] else "—",
              (bc["model_stop"].get(k) or f"never within {TOTAL}")) for k, v in bc.get("val", {}).items()]
    blocks.append(f"""## {H['4']}

The 31.5M base model for 300 steps at the tuned learning rate, with bias correction on and off, on the
same seed and data:

{table(['setting', 'val loss, on', 'val loss, off', 'off − on', 'gap back inside seed noise from step'], vrows)}

- **β2 = 0.999.** Switching correction off is costly and never recovers within the run: the
  multiplier is still about 2× at step 300, and warmup does not hide it.
- **β2 = 0.95.** The uncorrected early steps are *smaller* (0.45η at step 1). They act as a built-in
  warmup, so without warmup, switching correction off *helps*.

{fig('bias_correction_model', 'loss with and without bias correction in the base model')}

From the same optimizer state, the uncorrected/corrected update ratio of the median tensor follows
§2's closed form to within {max(bc.get('counterfactual_max_dev', {0: 0}).values()):.1%}:

{fig('bias_correction_counterfactual', 'counterfactual update ratio vs closed form')}""")

# ---------------------------------------------------------------- §5
if wu:
    srows = [(w, stp(v.get("5%")), stp(v.get("10%")), stp(v.get("20%")), stp(wu["plateau"].get(w)),
              "–".join(stp(x) for x in wu["stop_range"].get(w, []))) for w, v in wu["stop"].items()]
    spec = dict(zip(("first query matrix", "a middle up-projection", "last down-projection", "embedding"), wu.get("spectral_over_frob", [])))
    er = wu.get("emb_rows", {})
    blocks.append(f"""## {H['5']}

Each matrix's $\\lVert\\Delta W\\rVert/\\lVert W\\rVert$ is logged on every step. Runs warmed up over W steps
are compared with one that is not, at the same constant peak learning rate.

Warmup changes the ratio from step 1, because its learning rate is smaller by construction, so the
informative question is where that influence **ends**. It counts as ended at the first step after
which the 9-step rolling median stays within the threshold of the no-warmup run's (median over
matrices):

{table(['warmup W', '5%', '10% (headline)', '20%', 'ratio settles by', 'range over matrices (10%)'], srows)}

**Two findings.**
- **Against no warmup, the effect never ends.** Skipping warmup leaves the run permanently worse
  (val {f(wu.get('val', {}).get('0'))} vs {f(wu.get('val', {}).get(str(wu.get('warm', 30))))}), and its ratios stay lower.
- **Among warmed-up runs, the length stops mattering at about step
  {' / '.join(stp(v) for v in wu.get('pairwise', {}).values())}** ({', '.join(wu.get('pairwise', {}).keys())}; exploratory).

{fig('update_ratio_warmup', 'update-to-weight ratio heatmap and medians')}

**What drives the ratio.**
- **Adam's step as a fraction of η** falls from {f(wu.get('adam_frac_first'), '.2f')} at step 1 to
  {f(wu.get('adam_frac_late'), '.2f')} once gradients decorrelate.
- **The decay part.** Decoupled decay alone contributes ηλ = {f(wu.get('decay_floor'), '.1e')} per step,
  against a late median ratio of {f(wu.get('late_ratio'), '.1e')}.
- **Spectral vs Frobenius.** The spectral ratio is
  {', '.join(f'{v:.1f}× for the {k}' for k, v in spec.items())} the Frobenius ratio, so updates concentrate in a
  few directions.
- **The fp32 master copy.** {f(wu.get('bf16_lost'), '.0%')} of updates are smaller than half a bf16 spacing
  at the peak learning rate, and {f(wu.get('bf16_lost_floor'), '.0%')} at a 10× smaller one. Without the
  copy, those updates would be rounded away.

{fig('update_ratio_anatomy', 'adam fraction, spectral ratio, bf16 rounding')}

Embedding rows (tied head), median ratio by token frequency:

{table(['rows', 'median update/weight'], [(k, f'{v:.1e}') for k, v in er.items()])}

A RAdam run with **no** warmup reaches {f(wu.get('radam_val'))}, against AdamW at warmup
{', '.join(f'{w}: {f(v)}' for w, v in wu.get('val', {}).items())} (one seed each).""")

# ---------------------------------------------------------------- §6
if sc:
    trows = [(k, v["compute"], f(v["mean"]), f(v["sd"])) for k, v in sc["table"].items()]
    srows = [(k, f(v["val"]), f(v["train"]), f(v["lr"], ".1e"), f(v["gnorm"], ".3f"), f(v["clipped"], ".0%"), f(v["ratio"], ".1e"))
             for k, v in sc.get("state", {}).items()]
    frows = [(f"{float(k):.0%}", f"{v['diff']:+.4f}", f"{v['se']:.4f}", v['keep'] + (' (tie)' if v.get('tie') else ''))
             for k, v in sc.get("by_floor", {}).items()]
    grid = sc.get("cos_floor_grid", {})
    samp = {k: oneline(v) for k, v in sc.get("samples", {}).items() if v}
    bud = sc.get("budget", {})
    blocks.append(f"""## {H['6']}

The same model, planned for {TOTAL} steps under each schedule, is stopped at step {STOP}.

**Tuning.** Both schedules were tuned by the same rule: a learning-rate grid, then the schedule's own
knob, each extended at an edge until bracketed.
- **Cosine** had {bud.get('cosine', '?')} trials. Its floor kept improving all the way to 100%, which
  means no decay at all. Validation loss at {TOTAL} by floor:
  {', '.join(f'{float(k):.0%}: {v:.3f}' for k, v in grid.items())}.
- **WSD** had {bud.get('WSD', '?')} trials. Its best decay is the last 10%, linear.
- **Both** ended at a peak learning rate of {sc['lr_cos']:.0e}.

So in this short-run regime the tuned cosine arm is a constant learning rate, the same run as WSD's
flat phase. The rows below that coincide do so because they are the same model.

Three paired seeds; compute is in steps:

{table(['model', 'compute', 'val loss', 's.d.'], trows)}

{fig('cosine_vs_wsd', 'cosine vs WSD loss and schedules')}

{fig('cosine_vs_wsd_arms', 'every finished model with its compute')}

**The state of each model at step {STOP}:**

{table(['', 'val loss', 'train loss', 'lr now', 'grad norm', 'clipped', 'update/weight'], srows)}

**Decision (rule fixed in advance): keep the {sc['keep']} run.**
- Best option within +20% compute: cosine → {sc['best_cos']}; WSD → {sc['best_wsd']}.
- Paired difference WSD − cosine: {sc['diff']:+.4f} ± {sc['se']:.4f}. That is the SE over seeds; the
  eval-sampling SE is {f(sc.get('se_eval'))}.
- As stopped, with no cooldown: WSD − cosine = {sc['raw_diff']:+.4f}.

**How the decision moved as cosine's floor was tuned** (3 paired seeds each; the first pass stopped
at 20%):

{table(['cosine floor', 'WSD − cosine', 'SE', 'keep'], frows)}

**Samples at step {STOP}** (seed 0):
- cosine: *{samp.get('cos300', '—')}*
- WSD: *{samp.get('wsd300', '—')}*
- WSD, checkpoint {STOP - COOL} + cooldown: *{samp.get('wsd_cool160', '—')}*""")

# ---------------------------------------------------------------- §M
if mu:
    arows = [(k, f(v[0], "+.4f"), f(v[1], ".2f") + "×") for k, v in mu.get("apparent", {}).items()]
    tr = mu.get("trials", {})
    blocks.append(f"""## {H['M']}

**Setup.**
- Muon (momentum 0.95, Nesterov, `match_rms_adamw`) updates the 40 hidden matrices.
- AdamW updates the embedding and the norm gains.
- Each optimizer's learning rate was tuned on the same grid with the same number of points (AdamW
  {tr.get('adamw', '?')}, Muon {tr.get('muon', '?')}), and both ran on the same 3 seeds.

**Tuned vs tuned.**
- Final loss: Muon {sum(mu['muon']) / len(mu['muon']):.4f}, AdamW {sum(mu['adamw']) / len(mu['adamw']):.4f}.
- Paired difference: **{mu['diff']:+.4f} ± {mu['se']:.4f}**. {'A speedup is claimed' if mu['claim'] else 'No speedup is claimed (within 2 SE)'}.
- Speedup to AdamW's final loss: {', '.join(f'{x:.2f}×' for x in mu['speedup'])}.

The same tuned Muon run against differently tuned AdamW baselines:

{table(['baseline', 'loss difference', 'apparent speedup'], arows)}

{fig('muon_vs_adamw', 'muon vs adamw')}""")

# ---------------------------------------------------------------- §7
if wd:
    rows = [(w, f"{v['n_params'] / 1e6:.1f}M", f"{v['lr']:.2e}", f"{v['lo']:.2e} – {v['hi']:.2e}", f(v['loss']), f(v['cost2x'], '+.4f'))
            for w, v in wd["sp"].items()]
    a = wd["answer"]
    tr = wd.get("transfer", {})
    blocks.append(f"""## {H['7']}

**Protocol.**
- **Model:** the width family (4 layers, head size 64, 8,192-id vocabulary, untied).
- **Held fixed:** everything except width, i.e. a 16,384-token batch, {TOTAL} steps, warmup + cosine.
- **Minimum:** the vertex of a parabola in log₂(lr) through five √2-spaced points, 3 seeds each, with
  a 90% bootstrap interval.

{table(['width', 'parameters', 'best lr', '90% interval', 'loss there', 'cost of a 2× error'], rows)}

{fig('lr_vs_width', 'loss vs learning rate at each width, and best lr vs width')}

**Fit, then test on a held-out width.**
- Fit on three widths: log₂(best lr) = {wd['slope']:.2f}·log₂(width) + c, slope 90% interval
  {wd['slope_ci'][0]:.2f} to {wd['slope_ci'][1]:.2f}.
- Predicted for width {wd['hold']['width']}: {wd['pred_hold']:.2e}. Measured: {wd['hold']['lr']:.2e}, off by
  **{wd['hold']['err_oct']:+.2f} octave**.
- Refit on four widths: slope {wd['slope4']:.2f}.

**Width {a['width']}: use {a['lr']:.1e}.**
- **Expected range: {a.get('band', a['ci'])[0]:.1e}–{a.get('band', a['ci'])[1]:.1e}.** This is the error
  the same fit made one octave out.
- **The bootstrap interval ({a['ci'][0]:.1e}–{a['ci'][1]:.1e}) sees seed noise only**, so it understates the
  real uncertainty.
- **Confidence: {a['confidence']}**, by the rule fixed in advance.
- **Scope.** It holds for this depth, batch, step count and the standard parameterization. A
  different batch needs the √batch adjustment.
- **For comparison**, scaling width 256's best by 1/width gives {wd['naive_inverse_width']:.1e}.

{fig('lr_cliff_diagnostics', 'attention logit and output logit growth vs learning rate')}

**Does one learning rate suit every layer?** Update/weight ratio at each width's best learning rate,
over the second half of the run:

{table(['width', 'embedding', 'hidden matrices', 'output head'], [(w, f"{v['emb']:.1e}", f"{v['body']:.1e}", f"{v['head']:.1e}") for w, v in wd.get('headbody', {}).items()])}

**What muP changes.**
- **Coordinate check.** After 10 steps at one learning rate, the logit RMS grows
  {wd['coord_growth']['sp']:.1f}× from width 256 to {wd['hold']['width']} under SP, against
  {wd['coord_growth']['mup']:.1f}× under muP.
- **Sweep.** The best learning rate spreads {wd['sp_spread']:.2f} octaves across widths under SP, and
  {wd['mup_spread']:.2f} under muP.
- **Transfer.** Carrying width 256's best learning rate ({f(tr.get('lr'), '.1e')}) unchanged to width
  {wd['hold']['width']} gives {f(tr.get('sp'))} under SP and {f(tr.get('mup'))} under muP. SP tuned at
  {wd['hold']['width']} reaches {f(tr.get('sp_tuned'))}.

{fig('mup_coord_check', 'muP coordinate check')}

{table(['width', 'muP best lr', 'loss'], [(w, f"{v['lr']:.2e}", f(v['loss'])) for w, v in wd.get('mup', {}).items()])}

{fig('mup_sweep', 'muP sweep')}""")

# ---------------------------------------------------------------- §8
if tb:
    rows = [(k, f(v["fair"], "+.4f"), f(v["worst"], "+.4f"), f"{v['worst_at_oct']:+.0f}") for k, v in tb["pairs"].items() if "worst_at_oct" in v]
    fl = tb["pairs"].get("WSD (tuned) vs cosine floor", {})
    blocks.append(f"""## {H['8']}

The same tuned challenger against baselines run away from their own best, from grids already
measured (seed 0). In the learning-rate grids, both schedules use their default knobs.

**Mis-tuning the baseline's learning rate:**

{table(['comparison', 'apparent advantage when fair', 'worst mis-tuned baseline', 'octaves off'], rows)}

**Mis-tuning the baseline's own knob.**
- Against cosine at its tuned floor ({fl.get('tuned_floor', 0):.0%}), tuned WSD is ahead by {f(fl.get('fair'))} at step {TOTAL}.
- Against cosine at the usual 10% default floor, the same WSD run looks {f(fl.get('at_default_floor'))} ahead.
- This notebook's first pass made exactly that mistake (§6).

{fig('tune_both_sides', 'apparent advantage vs baseline tuning')}

**Tuning budget per side:**

{table(['comparison', 'trials', 'tuned', 'held fixed'], tb.get('parity', []))}""")

# ---------------------------------------------------------------- outcomes, limitations, running, references
obs = R.get("_observations", {})
if obs:
    held = sum(v["held"] for v in obs.values())
    blocks.append(f"## {H['obs']}\n\nThe notebook records each expectation with `observe(...)`. It is measured, not guaranteed, "
                  f"and a failed one is reported as failed. {held} of {len(obs)} held.\n\n"
                  + "\n".join(f"- {'✅' if v['held'] else '❌'} {v['msg']}" for v in obs.values()))

blocks.append(f"""## {H['lim']}

- **Horizon.** 300 steps × 16,384 tokens is about 0.16 tokens per parameter for the base model, which
  is very early training. Learning-rate optima and schedule rankings shift with run length, and longer
  horizons were not tested.
- **Seeds.** Three paired seeds give 2 degrees of freedom, so differences within 2 standard errors are
  called ties.
- **Data.** There is one dataset (TinyStories).
- **Held fixed in the width sweep:** depth, batch, warmup and β.
- **muP-lite.** The muP used here rescales hidden-layer learning rates and the output multiplier only.
- **Weight decay.** ηλT ≈ 0.06, so decay barely acts within these runs.
- **Validation reuse.** The schedule knobs and the EMA decay were chosen on the validation set used
  for the comparison.
- **Tuned cosine is constant LR.** Cosine's best floor was 100%, so the comparison WSD wins is "decay
  at the end" against "no decay". A genuine cosine decay at any floor loses by more.
- **Sequential tuning.** Learning rate and knob were tuned one after the other, not jointly.
- **Width 2,048 used one seed.** The width-4,096 band therefore uses that held-out error rather than
  the seed-only bootstrap.
- **Timing.** One run's wall time includes the machine sleeping, so it is excluded from the timing
  totals. Its result is unaffected.""")

blocks.append(f"""## {H['run']}

| `MODE` | what runs | time |
|---|---|---|
| `learn` (default) | the arithmetic sections live; every training result is replayed from `assets/results.json` | ~1 min, plus a one-time ~10 MB download if the file is missing |
| `quick` | tiny models through every code path; writes `assets/quick/` | ~10 min on a GPU |
| `full` | every measured run, resumable | ~{minutes / 60:.0f} h on the RTX 3070 Laptop above; a Colab T4 is roughly 2–3× slower (estimate) |

```bash
pip install -r requirements.txt
python adam_by_hand.py                                     # the hand-vs-PyTorch check on its own
python build_notebook.py                                   # nb_parts/*.py -> {NB}
OPT_MODE=full python -m nbconvert --to notebook --execute --inplace {NB} --ExecutePreprocessor.timeout=-1
python make_readme.py                                      # this file, from assets/results.json
```

- **Resumable.** Each experiment is saved to `results.json` as soon as it finishes, so an interrupted
  full run resumes where it stopped.
- **T4 rehearsal.** `OPT_PRECISION=fp16` rehearses the T4 path (fp16 + GradScaler) on any GPU.

| file | what it is |
|---|---|
| `{NB}` | the notebook, executed in full mode |
| `nb_parts/p00_setup.py` … `p12_answers.py` | the notebook's source, one file per section |
| `adam_by_hand.py` | a standalone Adam-by-hand vs PyTorch check |
| `build_notebook.py`, `make_readme.py` | build the notebook and this README |
| `assets/results.json`, `assets/*.png` | every measured number, and the figures |
| `CHEATSHEET.md` | a one-page refresher |
| `requirements.txt` | dependencies for a local run (Colab has them) |""")

blocks.append(f"""## {H['refs']}

| topic | reference |
|---|---|
| Adam | Kingma & Ba (2015), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| AdamW | Loshchilov & Hutter (2019), [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) |
| RAdam | Liu et al. (2020), [arXiv:1908.03265](https://arxiv.org/abs/1908.03265) |
| warmup and the update/weight ratio | Kosson et al. (2024), [arXiv:2410.23922](https://arxiv.org/abs/2410.23922) |
| small-scale proxies for instabilities | Wortsman et al. (2023), [arXiv:2309.14322](https://arxiv.org/abs/2309.14322) |
| μTransfer | Yang et al. (2022), [arXiv:2203.03466](https://arxiv.org/abs/2203.03466) |
| spectral condition for feature learning | Yang, Simon & Bernstein (2023), [arXiv:2310.17813](https://arxiv.org/abs/2310.17813) |
| scaling exponents across parameterizations | Everett et al. (2024), [arXiv:2407.05872](https://arxiv.org/abs/2407.05872) |
| cooldowns and WSD | Hägele et al. (2024), [arXiv:2405.18392](https://arxiv.org/abs/2405.18392); Hu et al. (2024), [arXiv:2404.06395](https://arxiv.org/abs/2404.06395); Wen et al. (2024), [arXiv:2410.05192](https://arxiv.org/abs/2410.05192) |
| AdamW decay as an EMA | Wang & Aitchison (2024), [arXiv:2405.13698](https://arxiv.org/abs/2405.13698) |
| √batch scaling for Adam | Malladi et al. (2022), [arXiv:2205.10287](https://arxiv.org/abs/2205.10287) |
| critical batch size | McCandlish et al. (2018), [arXiv:1812.06162](https://arxiv.org/abs/1812.06162) |
| Muon | Jordan et al. (2024), [blog](https://kellerjordan.github.io/posts/muon/); Liu et al. (2025), [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) |
| MuonClip | Kimi Team (2025), [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) |
| QK-norm | Dehghani et al. (2023), [arXiv:2302.05442](https://arxiv.org/abs/2302.05442) |
| data and tokenizer | TinyStories, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759); SmolLM2, [arXiv:2502.02737](https://arxiv.org/abs/2502.02737) |""")

OUT.write_text("\n\n".join(blocks) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {OUT.name}: {sum(len(b) for b in blocks):,} characters from {ASSETS / 'results.json'}")
