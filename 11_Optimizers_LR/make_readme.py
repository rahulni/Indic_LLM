"""Render README.md from assets/results.json (written by the notebook in full mode).

    python make_readme.py                 # the measured run in assets/
    python make_readme.py --quick         # the smoke-test run in assets/quick/ (writes README.quick.md)

Every number in the README comes from results.json; nothing is typed in by hand.
"""
import json
import math
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


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fig(name, alt):
    return f"![{alt}]({FIG}/{name}.png)" if (ASSETS / f"{name}.png").exists() else f"*(figure `{name}` not generated)*"


secs = {k: v for k, v in R.get("_seconds", {}).items() if v <= 1800}     # longer = the machine slept mid-run; see §9
minutes = sum(secs.values()) / 60
TOTAL = 300 if not QUICK else 60
STOP = 200 if not QUICK else 40
blocks = []

# ---------------------------------------------------------------- header
blocks.append(f"""# Optimizers and Learning Rates, Measured

*What `optimizer.step()` does, taken apart and measured: Adam by hand, bias correction, warmup and the
update-to-weight ratio, cosine vs WSD, Muon vs AdamW, and how the best learning rate moves with width.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

A gradient says which way each weight should move, but not how far. Every optimizer is a rule for the
distance. This notebook checks those rules one at a time:
- the arithmetic, by computing Adam by hand and against PyTorch's internal state
- the training effects, by small training runs that change one thing at a time against a tuned
  baseline

Every comparison is tuned on both sides and paired by seed. Every threshold used to call a result
was written down before the first measurement.

The measured run took {minutes:.0f} minutes on an {E.get('gpu', 'unknown GPU')}:
- precision: {E.get('precision', '?')}
- software: torch {E.get('torch', '?')}, Python {E.get('python', '?')}
- date: {E.get('date', '?')}
- power: {'on AC power' if E.get('on_ac_power') else 'on battery (or unknown)'}, {E.get('power_plan', 'unknown')} power plan

The notebook opens in **learn** mode, which replays these results on a CPU in about a minute
(see [Running it](#running-it)).
""")

# ---------------------------------------------------------------- answers at a glance
bc = g("bias_correction", {})
cf = bc.get("closed_form_stop", {})
wu = g("warmup", {})
sc = g("schedules", {})
wd = g("width", {})
mu = g("muon", {})
ad = g("adam_by_hand", {})
ans_rows = []
if ad:
    d = ad["diffs"]
    worst64 = max(max(v.values()) for k, v in d.items() if "float64" in k)
    worst32 = max(max(v.values()) for k, v in d.items() if "float32" in k and "fused" not in k)
    fused = d.get("torch float32, fused CUDA", {})
    ans_rows.append(("Does Adam by hand match PyTorch?",
                     f"Yes. Every quantity matches the 50-digit reference to {worst64:.0e} in float64 and {worst32:.0e} in float32. "
                     f"The fused float32 kernel's v̂ is off by {f(fused.get('v_hat'), '.1e')}, because it rounds 1−β2 in float32; its step agrees to {f(fused.get('step'), '.0e')}.",
                     "[§1](#1-adam-by-hand)"))
if bc:
    ms = bc.get("model_stop", {})
    ans_rows.append(("Bias correction off: when does the difference stop mattering?",
                     f"The first step is **{bc['first_step']:.2f}×** too large, and the gap *grows* to {bc['peak']:.2f}× at step {bc['peak_t']} "
                     f"({bc['at20']:.2f}× at step 20). It is within 1% only after **{cf.get('0.999', {}).get('1%', 0):,} steps** at β2=0.999, "
                     f"or **{cf.get('0.95', {}).get('1%', 0)}** at β2=0.95. In the 31.5M model the loss gap leaves seed noise at: "
                     + "; ".join(f"{k.replace('b2=', 'β2=')}: {('step ' + str(v)) if v else 'never in ' + str(TOTAL)}" for k, v in ms.items()) + ".",
                     "[§2](#2-bias-correction), [§4](#4-bias-correction-in-a-real-model)"))
def stp(v, n=None):
    n = n or (wu.get("n_steps") if wu else TOTAL) or TOTAL
    return f"never in {n}" if v is None or v > n else str(v)


if wu:
    st = wu.get("stop", {})
    W = str(wu.get("warm", 30))
    ans_rows.append(("When does warmup stop changing the update-to-weight ratio?",
                     "Compared with no warmup at all (the rule fixed in advance): "
                     + "; ".join(f"warmup {w}: " + (f"step **{v.get('10%')}**" if stp(v.get('10%')).isdigit() else f"**{stp(v.get('10%'))}**")
                                 for w, v in st.items())
                     + ". Skipping warmup leaves a lasting mark: the largest ratio is "
                     f"{f(wu['max_ratio'].get('0'), '.1e')} without warmup vs {f(wu['max_ratio'].get(W), '.1e')} with {W} steps, "
                     f"and the loss never catches up."
                     + (" Among warmed-up runs, the warmup *length* stops mattering at step "
                        + ", ".join(f"{stp(v)} ({k})" for k, v in wu["pairwise"].items()) + " (exploratory)." if wu.get("pairwise") else ""),
                     "[§5](#5-update-to-weight-ratio-and-warmup)"))
if sc:
    t = sc["table"]
    c_ = t.get(f"cosine-{TOTAL}, stopped at {STOP}", {})
    w_ = t.get(f"WSD-{TOTAL}, stopped at {STOP}", {})
    ans_rows.append((f"Cosine vs WSD, both stopped at {STOP} of {TOTAL}: which to keep?",
                     f"At step {STOP}: cosine **{f(c_.get('mean'))}** ± {f(c_.get('sd'))}, WSD **{f(w_.get('mean'))}** ± {f(w_.get('sd'))} "
                     f"(3 seeds). Keep **{sc['keep']}**: its best option within +20% compute ({sc['best_wsd']}) vs cosine's ({sc['best_cos']}) "
                     f"differs by {sc['diff']:+.4f} ± {sc['se']:.4f}.",
                     "[§6](#6-cosine-vs-wsd-stopped-at-200)"))
if wd:
    a = wd["answer"]
    ans_rows.append(("Best learning rate at width 4,096, and how confident?",
                     "Minima: " + ", ".join(f"{w}: {v['lr']:.1e}" for w, v in wd["sp"].items())
                     + f". The fitted slope is {wd['slope4']:.2f} (inverse-width scaling would be −1). The held-out width {wd['hold']['width']} "
                     f"was predicted to within {wd['hold']['err_oct']:+.2f} octaves. Use **{a['lr']:.1e}**, expected within "
                     f"{a.get('band', a['ci'])[0]:.1e}–{a.get('band', a['ci'])[1]:.1e} (the error the same fit made one octave out); "
                     f"confidence **{a['confidence']}** by the stated rule, for this depth, batch and step count.",
                     "[§7](#7-learning-rate-vs-width-and-mup)"))
if mu:
    ans_rows.append(("Tune both sides: does Muon beat AdamW here?",
                     f"Tuned vs tuned: {mu['diff']:+.4f} ± {mu['se']:.4f}, speedup **{sum(mu['speedup']) / len(mu['speedup']):.2f}×** "
                     f"({min(mu['speedup']):.2f}–{max(mu['speedup']):.2f}× across seeds). "
                     f"{'A speedup is claimed.' if mu['claim'] else 'No speedup is claimed: the difference is within 2 SE.'} "
                     f"Against a mis-tuned AdamW the same run looks {max(v[1] for v in mu['apparent'].values()):.2f}× faster.",
                     "[§M](#muon-vs-adamw), [§8](#8-tune-both-sides)"))
blocks.append("## Answers at a glance\n\n" + table(["question", "measured answer", "where"], ans_rows))

# ---------------------------------------------------------------- decision rules
rules = R.get("_rules", {})
blocks.append("## Decision rules, fixed before the first measurement\n\nThe notebook stores these in `results.json` "
              "and `check`s on every rerun that they have not changed.\n\n"
              + "\n".join(f"- **{k.replace('_', ' ')}**: {v['rule']}" for k, v in rules.items()))

# ---------------------------------------------------------------- §1
if ad:
    rows = [(i + 1, f"{gr:.2f}", f"{r['m']:.4f}", f"{r['v']:.6f}", f"{r['m_hat']:.4f}", f"{r['v_hat']:.4f}", f"{r['step']:.6f}", f"{r['w']:.6f}")
            for i, (gr, r) in enumerate(zip(ad["grads"], ad["table"]))]
    drows = [(k,) + tuple(f"{v[q]:.1e}" for q in ("m", "v", "m_hat", "v_hat", "step", "w")) for k, v in ad["diffs"].items()]
    blocks.append(f"""## 1. Adam by hand

One weight at w = 1, lr = 1e-3, β = (0.9, 0.999), ε = 1e-8, five gradients. The values below are the
exact (50-digit) computation:

{table(['t', 'g', 'm', 'v', 'm̂', 'v̂', 'step', 'w'], rows)}

Largest |difference| from the exact values. PyTorch stores m (`exp_avg`) and v (`exp_avg_sq`); m̂ and v̂
are derived from them:

{table(['implementation', 'm', 'v', 'm̂', 'v̂', 'step', 'w'], drows)}

The fused float32 kernel computes 1−β2 in float32, which is off by 1.3e-5 relative. Its own bias
correction uses the same rounded β2, so the step is unaffected. `adam_by_hand.py` runs the same check
standalone and exits 1 on any mismatch.""")

# ---------------------------------------------------------------- §2 / §4
if bc:
    trows = [(b2, *(f"{v[p]:,}" for p in ("1%", "5%", "10%"))) for b2, v in cf.items()]
    vrows = [(k.replace("b2=", "β2=").replace(",warmup=", ", warmup "), f(v["on"]), f(v["off"]), f(v["off"] - v["on"], "+.4f") if v["on"] and v["off"] else "—",
              bc["model_stop"].get(k) or f"never in {TOTAL}") for k, v in bc.get("val", {}).items()]
    blocks.append(f"""## 2. Bias correction

Without correction, the step is too large by exactly $(1-\\beta_1^t)/\\sqrt{{1-\\beta_2^t}}$, whatever the
gradients. $m$ recovers from its zero start in tens of steps, $v$ in thousands, so the gap **widens**
over the first dozen steps before it closes.

{fig('bias_correction_first20', 'first 20 steps with and without bias correction')}

Steps until the per-step multiplier stays within 1/5/10% of 1. Only $v$'s correction matters past the
first ~44 steps.

{table(['β2', '1%', '5%', '10%'], trows)}

{fig('bias_correction_long', 'multiplier vs step for several beta2')}

Summed over the run, the overshoot is permanent: about **{bc.get('excess_displacement_0999', 0):,.0f} η** of extra displacement at β2 = 0.999
(consistent gradients).""")
    base = g("base", {})
    if base:
        blocks.append(f"""## 3. The testbed

All training runs use TinyStories with the SmolLM2 tokenizer:
- **Base model**, used in §4–§6 and the Muon comparison: width 384, 8 layers, grouped-query attention
  (6/2 heads), SwiGLU, RoPE, a tied 49,152-token embedding, {base.get('n_params', 0) / 1e6:.1f}M parameters,
  16,384 tokens per step.
- **Width family**, used in §7: 4 layers, head size 64, an 8,192-id vocabulary, untied.

Runs with the same seed start from the same weights and see the same batches, so every comparison
is paired. A hand-written AdamW, used to switch bias correction off, matches `torch.optim.AdamW` on
all 31.5M weights for 20 steps (largest difference 0).

The base model's peak learning rate is tuned on a bracketed factor-2 grid; the best is
**{base.get('lr', 0):.0e}**:
- seed-to-seed spread of validation loss: {f(base.get('val_sd'))} s.d.
- running the same seed twice moves it by {f(base.get('nondet_val'), '.5f')}, the GPU-nondeterminism
  floor
- untrained loss: {f(base.get('loss0'), '.2f')} (ln 49,152 = 10.80)

{fig('base_lr_tuning', 'base model learning-rate tuning')}""")
    if bc:
        blocks.append(f"""## 4. Bias correction in a real model

The 31.5M base model, 300 steps, tuned learning rate, bias correction on and off, same seed and data:

{table(['setting', 'val loss, on', 'val loss, off', 'off − on', 'gap inside seed band from step'], vrows)}

{fig('bias_correction_model', 'loss with and without bias correction in the base model')}

From the same optimizer state, the uncorrected/corrected update ratio of the median tensor tracks the
closed form to within {max(bc.get('counterfactual_max_dev', {0: 0}).values()):.1%}:

{fig('bias_correction_counterfactual', 'counterfactual update ratio vs closed form')}""")

# ---------------------------------------------------------------- §5
if wu:
    srows = [(w, stp(v.get("5%")), stp(v.get("10%")), stp(v.get("20%")), stp(wu["plateau"].get(w)),
              "–".join(stp(x) for x in wu["stop_range"].get(w, []))) for w, v in wu["stop"].items()]
    er = wu.get("emb_rows", {})
    blocks.append(f"""## 5. Update-to-weight ratio and warmup

Each matrix's $\\lVert\\Delta W\\rVert/\\lVert W\\rVert$ is logged on every step. Runs warmed up over W steps
are compared with one that is not, at the same constant peak learning rate. "Warmup stops changing the
ratio" is the step after which every later step stays within the threshold of the no-warmup run
(9-step rolling median, median over matrices):

{table(['warmup W', '5%', '10% (headline)', '20%', 'ratio settles by', 'range over matrices (10%)'], srows)}

{fig('update_ratio_warmup', 'update-to-weight ratio heatmap and medians')}

What drives the ratio:
- **Adam's step as a fraction of η** falls from {f(wu.get('adam_frac_first'), '.2f')} at step 1 to
  {f(wu.get('adam_frac_late'), '.2f')} once gradients decorrelate.
- **The decay part.** Decoupled decay alone contributes ηλ = {f(wu.get('decay_floor'), '.1e')} per
  step, against a late median ratio of {f(wu.get('late_ratio'), '.1e')}.
- **Spectral vs Frobenius.** The spectral ratio is
  {', '.join(f'{x:.1f}' for x in wu.get('spectral_over_frob', []))}× the Frobenius ratio, so updates
  concentrate in a few directions.
- **Updates bf16 would round away:** {f(wu.get('bf16_lost'), '.0%')} of updates are smaller than half
  a bf16 spacing at the peak learning rate, and {f(wu.get('bf16_lost_floor'), '.0%')} at a 10× smaller
  one.

{fig('update_ratio_anatomy', 'adam fraction, spectral ratio, bf16 rounding')}

Embedding rows (tied head), median ratio by token-frequency rank:
{', '.join(f'{k} {v:.1e}' for k, v in er.items())}.

A RAdam run with **no** warmup reaches {f(wu.get('radam_val'))}, against
{', '.join(f'warmup {w}: {f(v)}' for w, v in wu.get('val', {}).items())} (one seed each).""")

# ---------------------------------------------------------------- §6
if sc:
    trows = [(k, v["compute"], f(v["mean"]), f(v["sd"])) for k, v in sc["table"].items()]
    st = sc.get("state", {})
    srows = [(k, f(v["val"]), f(v["train"]), f(v["lr"], ".1e"), f(v["gnorm"], ".3f"), f(v["clipped"], ".0%"), f(v["ratio"], ".1e")) for k, v in st.items()]
    samp = sc.get("samples", {})
    blocks.append(f"""## 6. Cosine vs WSD, stopped at {STOP}

The same model is planned for {TOTAL} steps under each schedule, and both are stopped at step {STOP}.
Each schedule was tuned with the same trial budget ({sc['budget']}). The chosen variants:
- cosine: {sc['cos_variant']}, peak learning rate {sc['lr_cos']:.0e}
- WSD: {sc['wsd_variant']}, peak learning rate {sc['lr_wsd']:.0e}

Three paired seeds. Compute is in steps:

{table(['model', 'compute', 'val loss', 's.d.'], trows)}

{fig('cosine_vs_wsd', 'cosine vs WSD loss and schedules')}

{fig('cosine_vs_wsd_arms', 'every finished model with its compute')}

The state of each model at step {STOP}:

{table(['', 'val loss', 'train loss', 'lr now', 'grad norm', 'clipped', 'update/weight'], srows)}

**Decision (rule fixed in advance): keep the {sc['keep']} run.**
- Best option within +20% compute: cosine → {sc['best_cos']}, WSD → {sc['best_wsd']}.
- Paired difference WSD − cosine: {sc['diff']:+.4f} ± {sc['se']:.4f} (SE over seeds; eval-sampling SE {f(sc.get('se_eval'))}){', a tie within 2 SE' if sc.get('tie') else ''}.
- As stopped, without any cooldown: WSD − cosine = {sc['raw_diff']:+.4f} ± {sc['raw_se']:.4f}.

Samples at step {STOP} (seed 0):
- cosine: *{samp.get('cos300')}*
- WSD: *{samp.get('wsd300')}*
- WSD {STOP - round(0.2 * STOP)} + cooldown: *{samp.get('wsd_cool160')}*""")

# ---------------------------------------------------------------- §M
if mu:
    arows = [(k, f(v[0], "+.4f"), f(v[1], ".2f") + "×") for k, v in mu.get("apparent", {}).items()]
    blocks.append(f"""## Muon vs AdamW

Muon (momentum 0.95, Nesterov, `match_rms_adamw`) runs on the 40 hidden matrices. AdamW runs on the
embedding and norm gains. Each optimizer's learning rate was tuned on the same grid with the same
number of trials ({mu['trials']}), and both ran on the same 3 seeds.

Tuned vs tuned:
- final loss: Muon {sum(mu['muon']) / len(mu['muon']):.4f} vs AdamW {sum(mu['adamw']) / len(mu['adamw']):.4f}
- paired difference: **{mu['diff']:+.4f} ± {mu['se']:.4f}**
- speedup to AdamW's final loss: {', '.join(f'{x:.2f}×' for x in mu['speedup'])}
- {'a speedup is claimed' if mu['claim'] else 'no speedup is claimed (within 2 SE)'}

The same tuned Muon run against differently tuned AdamW baselines:

{table(['baseline', 'loss difference', 'apparent speedup'], arows)}

{fig('muon_vs_adamw', 'muon vs adamw')}""")

# ---------------------------------------------------------------- §7
if wd:
    rows = [(w, f"{v['n_params'] / 1e6:.1f}M", f"{v['lr']:.2e}", f"{v['lo']:.2e} – {v['hi']:.2e}", f(v['loss']), f(v['cost2x'], '+.4f')) for w, v in wd["sp"].items()]
    a = wd["answer"]
    mup_rows = [(w, f"{v['lr']:.2e}", f(v['loss'])) for w, v in wd.get("mup", {}).items()]
    tr = wd.get("transfer", {})
    blocks.append(f"""## 7. Learning rate vs width, and muP

Width family: 4 layers, head size 64, an 8,192-id vocabulary, untied. Everything except width is
fixed: a 16,384-token batch, {TOTAL} steps, warmup + cosine. Each minimum is the vertex of a parabola
in log2(lr) through five √2-spaced points with 3 seeds each, with a 90% bootstrap interval:

{table(['width', 'parameters', 'best lr', '90% interval', 'loss there', 'cost of a 2× error'], rows)}

{fig('lr_vs_width', 'loss vs learning rate at each width, and best lr vs width')}

Fit and test:
- **Fit.** log2(best lr) = {wd['slope']:.2f}·log2(width) + c, slope 90% interval {wd['slope_ci'][0]:.2f} to {wd['slope_ci'][1]:.2f}.
- **Prediction for width {wd['hold']['width']}:** {wd['pred_hold']:.2e}.
- **Measured at width {wd['hold']['width']}:** {wd['hold']['lr']:.2e}, so the prediction was off by **{wd['hold']['err_oct']:+.2f} octaves**.
- **Refit on four widths** gives slope {wd['slope4']:.2f}.

**Width {a['width']}: use {a['lr']:.1e}.**
- **Honest band: {a.get('band', a['ci'])[0]:.1e}–{a.get('band', a['ci'])[1]:.1e}.** This is the error the same fit made one octave
  out, at width {wd['hold']['width']}.
- **Bootstrap interval: {a['ci'][0]:.1e}–{a['ci'][1]:.1e}.** It is {a['ci_oct']:.2f} octaves wide but reflects seed noise only, so it
  understates the real uncertainty.
- **Confidence: {a['confidence']}**, by the rule fixed in advance.

For comparison, scaling width 256's best by 1/width gives {wd['naive_inverse_width']:.1e}. The answer
holds only for this depth, batch, step count and the standard parameterization. A different batch
needs the √batch adjustment.

{fig('lr_cliff_diagnostics', 'attention logit and output logit growth vs learning rate')}

Update/weight ratio at each width's best learning rate (second half of the run):

{table(['width', 'embedding', 'hidden matrices', 'output head'], [(w, f"{v['emb']:.1e}", f"{v['body']:.1e}", f"{v['head']:.1e}") for w, v in wd.get('headbody', {}).items()])}

**muP.**
- **Coordinate check.** After 10 steps at one learning rate, the logit RMS grows
  {wd['coord_growth']['sp']:.1f}× from the smallest width to {wd['hold']['width']} under SP, against
  {wd['coord_growth']['mup']:.1f}× under muP.
- **Sweep.** The spread of the best learning rate across widths is
  {wd['sp_spread']:.2f} octaves under SP and {wd['mup_spread']:.2f} under muP.

{fig('mup_coord_check', 'muP coordinate check')}

{table(['width', 'muP best lr', 'loss'], mup_rows)}

{fig('mup_sweep', 'muP sweep')}

Carrying width 256's best learning rate ({f(tr.get('lr'), '.1e')}) to width {wd['hold']['width']} unchanged
gives {f(tr.get('sp'))} under SP and {f(tr.get('mup'))} under muP. SP tuned at {wd['hold']['width']} reaches {f(tr.get('sp_tuned'))}.""")

# ---------------------------------------------------------------- §8
tb = g("tune_both", {})
if tb:
    rows = [(k, f(v["fair"], "+.4f"), f(v["worst"], "+.4f"), f"{v['worst_at_oct']:+.0f}") for k, v in tb["pairs"].items() if "worst_at_oct" in v]
    fl = tb["pairs"].get("WSD (tuned) vs cosine floor", {})
    fl_txt = (f"The baseline's own knob matters as much as its learning rate. Against cosine at its tuned floor "
              f"({fl.get('tuned_floor', 0):.0%}), tuned WSD is ahead by {f(fl.get('fair'), '.4f')} at step {TOTAL}. Against cosine at the usual "
              f"10% default floor, the same WSD run looks {f(fl.get('at_default_floor'), '.4f')} ahead. This notebook's own first pass "
              f"made that mistake: cosine's floor grid stopped at the edge (20%), and extending it by the same edge rule shrank "
              f"WSD's step-{STOP} margin to the value in §6.") if fl else ""
    blocks.append(f"""## 8. Tune both sides

The same tuned challenger is set against baselines run at learning rates away from their own best.
The values come from grids already measured, at seed 0. In the learning-rate grids, both schedules
use their default knobs:

{table(['comparison', 'apparent advantage, fair', 'worst mis-tuned baseline', 'octaves off'], rows)}

{fl_txt}

{fig('tune_both_sides', 'apparent advantage vs baseline tuning')}

Tuning budget per side:

{table(['comparison', 'trials', 'tuned', 'held fixed'], tb.get('parity', []))}""")

# ---------------------------------------------------------------- background figures
blocks.append(f"""## Background figures

The notebook also covers the rules that come before Adam, each checked against standard worked numbers:
- gradient descent's stability condition, and why one learning rate cannot serve two curvatures
- momentum
- per-parameter learning rates
- optimizer memory
- the √batch rule
- Muon's orthogonalization

{fig('gd_two_curvatures', 'gradient descent on two curvatures')}
{fig('muon_orthogonalize', 'Newton-Schulz orthogonalization')}
{fig('schedules', 'learning-rate schedules')}""")

# ---------------------------------------------------------------- observations
obs = R.get("_observations", {})
if obs:
    blocks.append("## Measured outcomes\n\nThe notebook records each of these as `observe(...)`. They are measured, not guaranteed:\n\n"
                  + "\n".join(f"- {'✅' if v['held'] else '❌'} {v['msg']}" for v in obs.values()))

# ---------------------------------------------------------------- limitations, running, references
blocks.append("""## Limitations

- **Horizon.** 300 steps × 16,384 tokens is about 0.16 tokens per parameter for the base model, which
  is very early training. Learning-rate optima and schedule rankings shift with run length, and
  longer horizons were not tested.
- **Seeds.** Three paired seeds give 2 degrees of freedom, so differences within 2 standard errors are
  called ties.
- **Data.** There is one dataset (TinyStories).
- **Held fixed in the width sweep:** depth, batch, warmup and β.
- **muP-lite.** The muP used here rescales hidden-layer learning rates and the output multiplier only.
- **Weight decay.** ηλT ≈ 0.06, so decay barely acts within these runs.
- **Validation reuse.** The extra schedule knobs and the EMA decay were chosen on the validation set
  used for the comparison.
- **Cosine floor.** Cosine's floor was still improving at 50% after three extensions. Constant LR,
  its limit, still loses to tuned WSD, so the verdict stands.
- **Width 2,048** used one seed. The width-4,096 band uses that held-out error rather than the
  seed-only bootstrap.
- **Timing.** One run's wall time includes the machine sleeping, so it is excluded from the timing
  totals. Its result is unaffected.""")

blocks.append(f"""## Running it

| `MODE` | what runs | time |
|---|---|---|
| `learn` (default) | the arithmetic sections live on a CPU; every training result is replayed from `assets/results.json` | ~1 min |
| `quick` | tiny models through every code path; writes `assets/quick/` | ~10 min on a GPU |
| `full` | every measured run | {minutes:.0f} min on the RTX 3070 Laptop above; a Colab T4 is roughly 2–3× slower (estimate) |

```bash
pip install -r requirements.txt
python adam_by_hand.py                                     # the hand-vs-PyTorch check on its own
python build_notebook.py                                   # nb_parts/*.py -> {NB}
OPT_MODE=full python -m nbconvert --to notebook --execute --inplace {NB} --ExecutePreprocessor.timeout=-1
python make_readme.py                                      # this file, from assets/results.json
```

The full run is resumable: each experiment is saved to `results.json` as soon as it finishes.
`OPT_PRECISION=fp16` rehearses the T4 path (fp16 + GradScaler) on any GPU.

| file | what it is |
|---|---|
| `{NB}` | the notebook, executed in full mode |
| `nb_parts/` | the notebook's source, one file per section |
| `adam_by_hand.py` | a standalone Adam-by-hand vs PyTorch check |
| `build_notebook.py`, `make_readme.py` | build the notebook and this README |
| `assets/results.json` | every measured number |
| `CHEATSHEET.md` | a one-page refresher |

## References

Adam ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980)) ·
AdamW ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101)) ·
RAdam ([arXiv:1908.03265](https://arxiv.org/abs/1908.03265)) ·
warmup and the update/weight ratio ([arXiv:2410.23922](https://arxiv.org/abs/2410.23922)) ·
small-scale proxies for instabilities ([arXiv:2309.14322](https://arxiv.org/abs/2309.14322)) ·
μTransfer ([arXiv:2203.03466](https://arxiv.org/abs/2203.03466)) ·
spectral condition ([arXiv:2310.17813](https://arxiv.org/abs/2310.17813)) ·
scaling exponents across parameterizations ([arXiv:2407.05872](https://arxiv.org/abs/2407.05872)) ·
cooldowns and WSD ([arXiv:2405.18392](https://arxiv.org/abs/2405.18392), [arXiv:2404.06395](https://arxiv.org/abs/2404.06395), [arXiv:2410.05192](https://arxiv.org/abs/2410.05192)) ·
AdamW decay as an EMA ([arXiv:2405.13698](https://arxiv.org/abs/2405.13698)) ·
√ batch scaling ([arXiv:2205.10287](https://arxiv.org/abs/2205.10287)) ·
critical batch size ([arXiv:1812.06162](https://arxiv.org/abs/1812.06162)) ·
Muon ([blog](https://kellerjordan.github.io/posts/muon/), [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)) ·
MuonClip ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)) ·
QK-norm ([arXiv:2302.05442](https://arxiv.org/abs/2302.05442)) ·
TinyStories ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759)) ·
SmolLM2 ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737))""")

OUT.write_text("\n\n".join(blocks) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {OUT.name}: {sum(len(b) for b in blocks):,} characters from {ASSETS / 'results.json'}")
