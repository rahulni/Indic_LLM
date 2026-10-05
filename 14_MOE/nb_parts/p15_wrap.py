from . import md, code
from .predictions import PREDICTIONS, DEFINITIONS, REGISTERED

_pred_literal = repr(PREDICTIONS)

CELLS = [
md(r'''
---
## N · Predictions, open design questions, and a cheat sheet

### N.1 The predictions, scored

P1–P12 were written down (and committed in `predictions.json`) **before** the full run. P13–P15
(marked †) were added **after** the first full run and before the post-hoc runs of Part K.2. Each
one is scored mechanically from `R` below. A MISS is reported as a MISS, and no prediction was
edited after the fact.
'''),
code(r'''
PREDICTIONS = ''' + _pred_literal + r'''

def _score():
    P = R.get("points", {})
    out = {}
    def put(pid, ok, measured):
        out[pid] = ("HELD" if ok else "MISSED", measured)
    try:
        L = R["runs"]["moe8"]
        early = [v for s, v in zip(L["eval_step"], L["val"]) if s <= 0.25 * L["steps"]]
        put("P1", min(early) < R["val_T1"], f"best val in first 25% {min(early):.4f} vs dense T1 {R['val_T1']:.4f}")
    except (KeyError, ValueError):
        pass
    if "moe32_T3" in P:
        ok = P["moe8_T2"] < P["moe8_T1"] and P["moe32_T3"] < P["moe32_T2"] and P["moe32_T3"] < P["dense_T1"]
        put("P2", ok, f"MoE-8 {P['moe8_T1']:.4f}->{P['moe8_T2']:.4f}; MoE-32 {P['moe32_T2']:.4f}->{P['moe32_T3']:.4f}; dense T1 {P['dense_T1']:.4f}")
    if "paired" in R:
        p = R["paired"]
        put("P3", p["mean"] < 0 and p["hi"] < 0, f"MoE-32 - control = {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]")
    if "router_lab" in R:
        rl = R["router_lab"]
        ok = max(rl["none"]["dead"]) >= 1 and max(rl["bias"]["dead"]) == 0 and rl["bias"]["maxvio"] < min(rl["none"]["maxvio"], rl["aux"]["maxvio"])
        put("P4", ok, f"nearly dead none/aux/bias = {sum(rl['none']['dead'])}/{sum(rl['aux']['dead'])}/{sum(rl['bias']['dead'])}; "
                      f"MaxVio {rl['none']['maxvio']:.2f}/{rl['aux']['maxvio']:.2f}/{rl['bias']['maxvio']:.2f}")
        put("P8", rl["sigmoid" if "sigmoid" in rl else "bias"]["maxvio"] < rl["softmax"]["maxvio"],
            f"MaxVio sigmoid {rl['bias']['maxvio']:.3f} vs softmax {rl['softmax']['maxvio']:.3f}")
    g = R.get("growth", {})
    if "main_family_at_0" in g:
        put("P5", g["main_family_at_0"] > 0.5, f"{g['main_family_at_0']:.1%} of tokens")
    if all(f"grow:{k}" in R["runs"] for k in ("copy", "gumbel", "staggered")):
        dc, dg = int(nearly_dead(R["runs"]["grow:copy"], 4).sum()), int(nearly_dead(R["runs"]["grow:gumbel"], 4).sum())
        put("P6", dc > dg, f"nearly dead: copy+hard {dc}, drop+Gumbel {dg}")
        vs, vg = R["runs"]["grow:staggered"]["val"][-1], R["runs"]["grow:gumbel"]["val"][-1]
        put("P7", vs <= vg, f"staggered {vs:.4f} vs drop+Gumbel {vg:.4f}")
    ex = R.get("experts", {})
    if "consecutive" in ex:
        mid = ex["consecutive"][2:6]
        put("P9", all(c["agree"] > c["chance"] for c in mid),
            "; ".join(f"L{c['layer']} {c['agree']:.0%} vs {c['chance']:.0%}" for c in mid))
    if "prune" in ex:
        pr = ex["prune"]
        put("P10", (pr["half"] - pr["base"]) / pr["base"] < 0.05, f"{(pr['half'] - pr['base']) / pr['base']:+.2%}")
        put("P11", pr["top3"] > np.mean(pr["random3"]), f"top-3 {pr['top3']:.4f} vs random {np.mean(pr['random3']):.4f}")
    ph = R.get("posthoc", {})
    if "moe32_copy" in ph and "moe8_cont" in ph:
        c, m = ph["moe32_copy"]["vs_moe32"], ph["moe8_cont"]["vs_moe32"]
        put("P13", c["mean"] < 0 and c["hi"] < 0, f"copy-grown - registered = {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
        put("P14", m["mean"] < 0 and m["hi"] < 0, f"MoE-8 continued - registered = {m['mean']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]")
        cc, mc = ph["moe32_copy"]["vs_control"], ph["moe8_cont"]["vs_control"]
        put("P15", cc["mean"] >= 0 and mc["mean"] >= 0, f"vs dense control: copy-grown {cc['mean']:+.4f}, MoE-8 continued {mc['mean']:+.4f}")
    if "capacity" in R:
        cap = {str(r["cf"]): r for r in R["capacity"]}
        put("P12", cap["1.0"]["val"] > cap["None"]["val"], f"CF 1.0 {cap['1.0']['val']:.4f} vs dropless {cap['None']['val']:.4f} "
                                                           f"({cap['1.0']['dropped']:.1%} dropped)")
    return out

scored = _score()
R["predictions"] = {p["id"]: dict(claim=p["claim"], verdict=scored.get(p["id"], ("NOT RUN", ""))[0],
                                  measured=scored.get(p["id"], ("", "-"))[1]) for p in PREDICTIONS}
for p in PREDICTIONS:
    R["predictions"][p["id"]]["posthoc"] = bool(p.get("posthoc"))
lines = ["| | prediction | verdict | measured |", "|---|---|---|---|"]
for p in PREDICTIONS:
    r = R["predictions"][p["id"]]
    lines.append(f"| {p['id']}{' †' if r['posthoc'] else ''} | {p['claim']} | **{r['verdict']}** | {r['measured']} |")
reg = [r for r in R["predictions"].values() if not r["posthoc"]]
post = [r for r in R["predictions"].values() if r["posthoc"]]
display(Markdown("\n".join(lines) + f"\n\n**Registered before the run: {sum(r['verdict'] == 'HELD' for r in reg)} of {len(reg)} held.** "
                 f"Post-hoc (†): {sum(r['verdict'] == 'HELD' for r in post)} of {len(post)} held."))
'''),
md(r'''
### N.2 Open design questions: what this small run says

Any larger MoE design has to answer these. A 20M-parameter run on one corpus cannot settle any of them
for a production model. What it can do is show which way the evidence points at small scale, and
which experiment to scale up.
'''),
code(r'''
def _v(name):
    return R["runs"][name]["val"][-1] if have("runs", name) else float("nan")

if have("runs", "grow:staggered") and have("paired"):
    rl, p = R["router_lab"], R["paired"]
    qa = [
        ("Dense or MoE?", f"At equal active compute, MoE-32 finished {p['mean']:+.4f} nats vs the dense control "
                          f"(95% CI [{p['lo']:+.4f}, {p['hi']:+.4f}]), with {R['params']['moe32'] / R['params']['dense']:.1f}x the parameters "
                          f"and {R['bench']['MoE-32']['ms'] / R['bench']['dense']['ms']:.1f}x the step time on one GPU."
                          if have("bench") else f"MoE-32 vs dense control: {p['mean']:+.4f} nats."),
        ("From scratch or grown?", f"In the conversion lab, random experts (attention kept) ended at {_v('conv:random'):.4f} "
                                    f"vs {_v('conv:partition'):.4f} for partition-upcycled after the same 1.2M tokens."),
        ("Shared expert or none?", f"Partition with a shared expert {_v('conv:partition'):.4f} vs without {_v('conv:partition_noshared'):.4f} "
                                    f"(same active size; 26M vs 33M total)."),
        ("Softmax or sigmoid?", f"softmax {rl['softmax']['val']:.4f} (MaxVio {rl['softmax']['maxvio']:.2f}), sigmoid {rl['bias']['val']:.4f} "
                                 f"({rl['bias']['maxvio']:.2f}), sqrt-softplus {rl['sqrt_softplus']['val']:.4f} ({rl['sqrt_softplus']['maxvio']:.2f})."),
        ("How to grow the expert count?", f"After the growth window: copy+hard {_v('grow:copy'):.4f}, drop+hard {_v('grow:drop'):.4f}, "
                                           f"drop+Gumbel {_v('grow:gumbel'):.4f}, staggered-bias copy {_v('grow:staggered'):.4f}."
                                           + (f" Full Stage 3 (post-hoc, 64-batch val at T3): copy growth {R['posthoc']['moe32_copy']['val_T3']:.4f}, "
                                              f"drop+Gumbel {np.mean(R['final_pair']['moe32']):.4f}, no growth (MoE-8) "
                                              f"{R['posthoc']['moe8_cont']['val_T3']:.4f}, dense {np.mean(R['final_pair']['control']):.4f}."
                                              if have("posthoc") else "")),
        ("Balancing?", f"Nearly-dead experts after the lab: none {sum(rl['none']['dead'])}, aux loss {sum(rl['aux']['dead'])}, "
                        f"bias {sum(rl['bias']['dead'])}; MaxVio {rl['none']['maxvio']:.2f} / {rl['aux']['maxvio']:.2f} / {rl['bias']['maxvio']:.2f}."),
    ]
    display(Markdown("| question | what this run measured |\n|---|---|\n" + "\n".join(f"| {q} | {a} |" for q, a in qa)))
'''),
md(r'''
### N.3 Cheat sheet

| idea | one line | where |
|---|---|---|
| MoE layer | `y = Σ_{i∈topk} gᵢ·Eᵢ(x) + Shared(x)`; attention untouched | C |
| total vs active | memory follows total (16 B/param to train), compute follows active (6 FLOP/param/token) | A.3 |
| router | `logits = x·Wᵣᵀ` in **fp32**; score = sigmoid / softmax / √softplus; choose on `score + bias`, weight on `score` | C |
| weights | renormalise top-k to 1, × route scale $s = E/c$: **k for pieces** (partition), **1 for copies** | F |
| why copy is exact | every chosen copy holds every neuron, so the weights sum to 1; partition uses each neuron 0/1/2 times (mean 1, var 3/7) | F |
| batch vs learning rate | keep $\eta/B$ fixed when you shrink the batch (the v1 lab bug) | E |
| perplexity | $e^{\ell}$; a gap of Δℓ nats is a factor $e^{\Delta\ell}$ in perplexity (0.02 → 2%) | K |
| fine-grained experts | many small experts give vastly more combinations at the same compute | A.5 |
| what experts learn | kinds of tokens (punctuation, names, function words), not subjects | L |
| collapse | chosen → more gradient → better → chosen more | H |
| capacity / dropping | old: fixed slots, overflow dropped; now: dropless | M.1 |
| aux loss | `α·E·Σ fᵢPᵢ`; its gradient fights the language loss | H |
| loss-free bias | `bᵢ += γ·sign(mean load − loadᵢ)` after every step; γ ≈ 1e-3, 0 at the end | C |
| balancing scope | count the load over the whole batch, not per micro-batch | E |
| upcycling | a FFN is a sum over neurons: copy (exact, identical experts), partition (exact in expectation), drop (redraw half) | F |
| growing | clones tie in the router → top-k piles into one family; sample (Gumbel) or stagger | J |
| expert parallelism | experts spread over GPUs; tokens all-to-all there and back; keep EP inside a node | M.2 |
| Adam state | slice the moments with the same indices as the weights | F |

### N.4 Pitfalls this notebook hit or guards against

1. **`nn.Linear` is `[out, in]`.** Neurons are rows of gate/up and columns of down.
2. **The router in 16-bit diverges** (Switch). Compute it in fp32 with autocast off.
3. **Copies need `route_scale = 1`, pieces need `route_scale = k`.** The wrong one scales the
   output by k or 1/k at conversion.
4. **Top-k has no gradient.** The router learns only through the weights that multiply the chosen
   outputs, so with top-1 and renormalisation (weight always 1) it never learns.
5. **The bias must never enter the weights.** It steers the choice only (gate in C).
6. **The aux loss sneaking back in.** Copied code can quietly re-add it and waste days of compute;
   here a gate checks the main config.
7. **Cloned experts tie in the router.** Hard top-k picks the whole family.
8. **Dropless needs care with memory.** Activations grow with *k*; micro-batch if needed.
9. **Do not trust laptop step times taken at different moments.** The GPU slows as it heats; use
   ABBA ordering.
10. **Never let a quick run overwrite real results.** Quick mode writes to `assets-quick/`.
11. **A smaller batch needs a smaller learning rate.** This notebook's first full run used batch 16
    at the batch-64 learning rate for the labs, and every lab's loss *rose*. Keep lr / batch fixed
    when you shrink a batch.
12. **Each conversion costs tokens.** Budget the recovery: in this run the T2 growth cost 0.17
    nats, 12M tokens did not repay it, and not growing at all ended better (Part K.2).

### N.5 Check yourself

<details><summary>1. A layer has 64 experts of width 512, top-6, plus one shared expert of width 1,024, with d = 2,048. How many FFN parameters are active per token, per layer?</summary>

Each expert is 3 × 2048 × 512 = 3.15M. Active: 6 × 3.15M + 3 × 2048 × 1024 (6.29M) = 25.2M, plus the
router 2048 × 64 = 0.13M. The 58 idle experts (182M) cost memory but no compute.

</details>

<details><summary>2. Why does copy upcycling preserve the function exactly, but partition only in expectation?</summary>

Copies are all equal to the dense block, and the renormalised top-k weights sum to 1, so any
mixture of them is the dense block. Partition experts are *pieces*: the dense output is the sum of
*all* pieces, but a token only gets k of them. Their selection covers each neuron once on average
(scale k), not exactly once.

</details>

<details><summary>3. With bias balancing, an expert has load 0 for 300 steps. What is its bias relative to the busiest expert, at γ = 0.001?</summary>

It rises by about 0.001 per step and the busiest falls by 0.001 per step, so the gap opens by up to
0.6. With sigmoid scores in (0, 1), that is enough to put it into many tokens' top-k. The bias is a
slow, bounded nudge, not a switch.

</details>

<details><summary>4. You grow 16 → 64 experts by cloning, keep top-8, and the loss jumps. What happened, and two fixes?</summary>

The 4 clones of each family get the same router score, so a token's top-8 becomes the clones of its
best two families instead of eight different experts. Fixes: sample the top-k for an early window
(Gumbel), or start clones with staggered selection offsets so the old routing is reproduced
exactly and handed over gradually. A third is to double top-k with the clone count (Microsoft's
orthogonal growth), at extra compute.

</details>

<details><summary>5. Why balance over the whole batch rather than each micro-batch?</summary>

A micro-batch may be all code. Forcing even load inside it makes every expert handle code, which
blocks specialization. Over the whole batch, experts can split the work by kind (Qwen 2025: lower
perplexity and better benchmarks).

</details>

### N.6 References

Shazeer et al. 2017 (sparsely-gated MoE) · Lepikhin et al. 2020 (GShard) · Fedus et al. 2021 (Switch
Transformer) · Zoph et al. 2022 (ST-MoE, router z-loss) · Gale et al. 2022 (MegaBlocks, dropless) ·
Komatsuzaki et al. 2022 (sparse upcycling) · Chen et al. 2022 (towards understanding MoE) ·
Dai et al. 2024 (DeepSeekMoE: fine-grained and shared experts) · Wang et al. 2024 (auxiliary-loss-free
balancing) · DeepSeek-AI 2024 (DeepSeek-V3) · Jiang et al. 2024 (Mixtral) · Muennighoff et al. 2024
(OLMoE) · Krajewski et al. 2024 (scaling laws for fine-grained MoE) · Nakamura et al. 2025
(Drop-Upcycling) · Qwen team 2025 (global-batch load balancing; Qwen3) · Sukhbaatar et al. 2024
(Branch-Train-MiX) · Ainslie et al. 2023 (GQA) · Lightning LM growth notes.
'''),
code(r'''
if COMPUTE:
    needed = ("val_T1", "conversion", "router_lab", "growth", "points", "paired", "posthoc", "experts", "capacity", "ep", "bench", "checklist")
    R["meta"]["complete"] = all(have(k) for k in needed)
    R["meta"]["gates"] = GATES
    R["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M")
    save_results()
print(f"{sum(GATES.values())}/{len(GATES)} gates passed in this run"
      + (f"; results {'complete' if R['meta'].get('complete') else 'INCOMPLETE'} ({R['meta'].get('mode')})" if R.get("meta") else ""))
'''),
]
