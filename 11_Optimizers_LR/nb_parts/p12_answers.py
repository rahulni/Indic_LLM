# %% [markdown]
r"""
## §9 Answers, limitations, references

The five questions from the top, answered with the numbers measured above. The same table is
generated into the README.
"""

# %%
A = R.get("answers", {})
bc, wu, sc, wd_, mu = A.get("bias_correction", {}), A.get("warmup", {}), A.get("schedules", {}), A.get("width", {}), A.get("muon", {})
rows = []
if bc:
    rows.append(("1. Adam by hand vs PyTorch",
                 "all five steps match the 50-digit reference: float64 to ~1e-16, float32 to ~1e-8; the fused float32 kernel's v differs "
                 "by 1.3e-5 relative (its 1−β2 is rounded in float32), the step itself by ~1e-8"))
    cf = bc["closed_form_stop"]

    def bc_setting(k):                        # "b2=0.95,warmup=0" -> "β2 0.95, no warmup"
        b2, w = (x.split("=")[1] for x in k.split(","))
        return f"β2 {b2}, " + ("no warmup" if w == "0" else f"warmup {w}")

    rows.append(("2. Bias correction off",
                 f"first step {bc['first_step']:.2f}× too large, peaking at {bc['peak']:.2f}× (step {bc['peak_t']}), still {bc['at20']:.2f}× at step 20. "
                 f"Within 1% after {cf['0.999']['1%']:,} steps (β2=0.999) or {cf['0.95']['1%']} steps (β2=0.95). "
                 f"In the 31.5M model the loss gap re-enters seed noise at: "
                 + "; ".join(f"{bc_setting(k)}: {'step ' + str(v) if v else 'never within ' + str(TOTAL)}" for k, v in bc["model_stop"].items())))
if wu:
    st = wu["stop"]
    rows.append(("3. When warmup stops changing the update/weight ratio",
                 "it changes the ratio from step 1 by construction. Against no warmup at all (the rule fixed in §0) its effect "
                 + ("never ends within the run" if all(v['10%'] > TOTAL for v in st.values()) else
                    "ends at " + "; ".join(f"warmup {w}: step {v['10%']}" for w, v in st.items()))
                 + f": the largest ratio is {wu['max_ratio']['0']:.1e} without warmup vs {wu['max_ratio'][str(WARM)]:.1e} with {WARM} steps, "
                 f"and the loss never catches up. Among warmed-up runs the warmup length stops mattering at step "
                 + ", ".join(f"{v} ({k})" for k, v in wu.get("pairwise", {}).items()) + " (exploratory)"))
if sc:
    t = sc["table"]
    c_ = t[f"cosine-{TOTAL}, stopped at {STOP}"]; w_ = t[f"WSD-{TOTAL}, stopped at {STOP}"]
    degenerate = sc.get("cos_variant", "").endswith("100%")
    rows.append((f"4. Cosine vs WSD, stopped at {STOP}",
                 f"cosine {c_['mean']:.4f} ± {c_['sd']:.4f}, WSD {w_['mean']:.4f} ± {w_['sd']:.4f} (3 seeds)"
                 + (" — the same run, because tuned cosine never decays" if degenerate else "")
                 + f". Keep {sc['keep']}: best option within +20% compute, WSD − cosine = {sc['diff']:+.4f} ± {sc['se']:.4f}"))
if wd_:
    a = wd_["answer"]
    rows.append((f"5. Learning rate at width {a['width']}",
                 "minima " + ", ".join(f"{w}: {v['lr']:.1e}" for w, v in wd_["sp"].items())
                 + f"; slope {wd_['slope4']:.2f}; held-out {wd_['hold']['width']} off by {wd_['hold']['err_oct']:+.2f} octaves. "
                 f"Use {a['lr']:.1e}, expected within {a.get('band', a['ci'])[0]:.1e}–{a.get('band', a['ci'])[1]:.1e} "
                 f"(the error the same fit made one octave out); confidence {a['confidence']} by the stated rule"))
if mu:
    rows.append(("Tune both sides: Muon vs AdamW",
                 f"tuned vs tuned {mu['diff']:+.4f} ± {mu['se']:.4f}, speedup {np.mean(mu['speedup']):.2f}× "
                 f"({'claimed' if mu['claim'] else 'not claimed: within 2 SE'}); against a mis-tuned AdamW the same run looks "
                 f"{max(v[1] for v in mu['apparent'].values()):.2f}× faster"))
for q, a_ in rows:
    print(f"{q}\n    {a_}\n")

obs = R.get("_observations", {})
if obs:
    held = sum(v["held"] for v in obs.values())
    print(f"measured outcomes recorded with observe(): {held} of {len(obs)} held")
    for k, v in obs.items():
        if not v["held"]:
            print(f"  ✗ {v['msg']}")

# %% [markdown]
r"""
### Limitations: what could change these answers

- **Horizon.** 300 steps × 16,384 tokens is 4.9M tokens, about 0.16 tokens per parameter for the
  31.5M base model. This is very early training. Schedule rankings and learning-rate optima are
  known to shift with training length, generally toward lower learning rates for longer runs, and
  this notebook does not test longer horizons.
- **Seeds.** Three paired seeds give a usable but wide error bar (2 degrees of freedom). Differences
  within about 2 standard errors are reported as ties.
- **One dataset.** TinyStories is simple and homogeneous, and harder data has noisier gradients.
- **Fixed everything else.** Depth, batch size, warmup and β are held fixed in the width sweep, so
  the width-4,096 answer holds only for those settings. A different batch needs the √batch
  adjustment (R.5), and greater depth needs its own transfer rule.
- **muP-lite.** This muP rescales hidden-layer learning rates and the output multiplier. It does not
  treat depth, the embedding multiplier or weight decay. Its coordinate check passes, but it is not
  the full recipe.
- **Weight decay barely acts.** $\eta\lambda T \approx 0.06$, so the decay timescale $1/(\eta\lambda)$ is
  far longer than any run here. Claims about decay as weight averaging cannot be tested at this
  length.
- **Reuse of the validation set.** The EMA decay and each schedule's extra knob were chosen on the
  same validation set used to compare them, which flatters both sides a little.
- **Tuned cosine is constant LR.** Cosine's best floor was 100%, so the tuned "cosine" arm never
  decays. The comparison WSD wins is therefore "decay at the end" against "no decay". The same holds
  for a genuine cosine decay at any floor, which loses by more (§6).
- **Learning rate and knob were tuned one after the other**, not jointly. A no-decay schedule might
  prefer a slightly different peak learning rate than the cosine grid found.
- **Width 2,048 used one seed**, and the bootstrap interval at 4,096 sees only seed noise. The
  stated band uses the held-out error instead.
- **The machine slept during one run.** That run's result is unaffected but its wall time is not, so
  it is left out of the timing totals.

### References

- Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Liu et al. (2020), *On the Variance of the Adaptive Learning Rate and Beyond* (RAdam). [arXiv:1908.03265](https://arxiv.org/abs/1908.03265)
- Kosson et al. (2024), *Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training*. [arXiv:2410.23922](https://arxiv.org/abs/2410.23922)
- Wortsman et al. (2023), *Small-scale proxies for large-scale Transformer training instabilities*. [arXiv:2309.14322](https://arxiv.org/abs/2309.14322)
- Yang et al. (2022), *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (μTransfer). [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
- Yang, Simon & Bernstein (2023), *A Spectral Condition for Feature Learning*. [arXiv:2310.17813](https://arxiv.org/abs/2310.17813)
- Everett et al. (2024), *Scaling Exponents Across Parameterizations and Optimizers*. [arXiv:2407.05872](https://arxiv.org/abs/2407.05872)
- Hägele et al. (2024), *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (cooldowns, 1−√). [arXiv:2405.18392](https://arxiv.org/abs/2405.18392)
- Hu et al. (2024), *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (WSD). [arXiv:2404.06395](https://arxiv.org/abs/2404.06395)
- Wen et al. (2024), *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective*. [arXiv:2410.05192](https://arxiv.org/abs/2410.05192)
- Wang & Aitchison (2024), *How to set AdamW's weight decay as you scale model and dataset size*. [arXiv:2405.13698](https://arxiv.org/abs/2405.13698)
- Malladi et al. (2022), *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms* (√ batch scaling). [arXiv:2205.10287](https://arxiv.org/abs/2205.10287)
- McCandlish et al. (2018), *An Empirical Model of Large-Batch Training* (critical batch size). [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
- Jordan et al. (2024), *Muon: An optimizer for hidden layers in neural networks*. [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)
- Liu et al. (2025), *Muon is Scalable for LLM Training*. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
- Kimi Team (2025), *Kimi K2: Open Agentic Intelligence* (MuonClip). [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
- Dehghani et al. (2023), *Scaling Vision Transformers to 22 Billion Parameters* (QK-norm against attention-logit growth). [arXiv:2302.05442](https://arxiv.org/abs/2302.05442)
- Eldan & Li (2023), *TinyStories*. [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
- Allal et al. (2025), *SmolLM2* (the tokenizer). [arXiv:2502.02737](https://arxiv.org/abs/2502.02737)
"""

# %%
sec_raw = R.get("_seconds", {})
# No single run here takes more than ~8 minutes. One that "took" longer was paused by the machine sleeping, which leaves
# its result unchanged (training is deterministic) but makes its wall time meaningless, so it is left out of the totals.
SUSPENDED = {k: v for k, v in sec_raw.items() if v > 1800}
sec = {k: v for k, v in sec_raw.items() if k not in SUSPENDED}
groups = {"§3 base tuning + checks": ("base|adamw", "handadamw"), "§4 bias correction": ("base|hand",),
          "§6 WSD arms": ("wsd_arms",), "§M Muon": ("base|muon",), "§5 RAdam": ("base|radam",),
          "§7 width sweeps": tuple(f"w{w}|" for w in WIDTHS + [HOLD]) + tuple(f"w{w}-mup|" for w in WIDTHS + [HOLD]),
          "§7 coordinate check": ("coord_check",)}
tot = 0.0
print(f"{'experiments':<28}{'runs':>6}{'minutes':>9}")
for lab, prefixes in groups.items():
    ks = [k for k in sec if k.startswith(prefixes)]
    m = sum(sec[k] for k in ks) / 60
    tot += m
    print(f"{lab:<28}{len(ks):>6}{m:9.1f}")
print(f"{'total':<28}{len(sec):>6}{sum(sec.values()) / 60:9.1f}")
for k, v in SUSPENDED.items():
    print(f"left out of the totals: [{k}] recorded {v / 3600:.1f} h of wall time; the machine slept during it (its result is unaffected)")
tok_base = TOTAL * BASE_B * BASE_T
print(f"\neach base-model run: {tok_base / 1e6:.1f}M tokens, ≈ {6 * N_BASE * tok_base:.1e} FLOPs (6·N·D)")
if MODE != "learn":
    R["env_end"] = _clean(env_info())
    save_results()
show_env(R.get("env"), "environment of the measured run:")
show_env(R.get("env_end"), "environment at the latest run's end:")
if MODE == "full" and _RAN:
    print(f"\nthis session ran {len(_RAN)} experiments; everything else was reused from {RESULTS_FILE}")
