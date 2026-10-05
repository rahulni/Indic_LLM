from . import md, code, mathbox

CELLS = [
md(r'''
---
## M · Systems view: capacity, placement, memory and time

> Capacity and token dropping, expert parallelism, choosing a layout, and the rule that compute
> follows the active parameters while memory follows the total.

### M.1 What token dropping would have cost

Old-style MoEs gave every expert a fixed number of slots,
`capacity = tokens × k / E × capacity factor`. Tokens beyond that were **dropped**: they skipped the
expert and went on through the residual. Our model is dropless. Here its real routing is replayed
with capacity limits, using 8,192-token groups as in the reference example, to see how much would
have been dropped and what that would have cost.
'''),
code(r'''
@torch.no_grad()
def capacity_eval(model, cf, n_batches, B=32):
    moes = model.moe_layers()
    for m in moes:
        m.capacity_factor = cf
    model.eval()
    losses, dropped, total = [], 0, 0
    for i in range(n_batches):
        x, y = VAL.batch(i * B, B)
        with amp():
            _, l = model(x, y)
        losses.append(l.item())
        if cf is not None:
            dropped += sum(m.last["dropped"] for m in moes)
        total += x.numel() * moes[0].k * len(moes)
    for m in moes:
        m.capacity_factor = None
    return float(np.mean(losses)), dropped / total

if COMPUTE and moe32 is not None and not have("capacity"):
    R["capacity"] = [dict(cf=cf, val=v, dropped=d) for cf in (None, 2.0, 1.5, 1.25, 1.0)
                     for v, d in [capacity_eval(moe32, cf, BUDGET["val_batches"])]]
    save_results()
if have("capacity"):
    base = R["capacity"][0]["val"]
    display(Markdown("| capacity factor | tokens x experts dropped | val loss | vs dropless |\n|---|---|---|---|\n" + "\n".join(
        f"| {'dropless' if r['cf'] is None else r['cf']} | {r['dropped']:.2%} | {r['val']:.4f} | {r['val'] - base:+.4f} |" for r in R["capacity"])))
'''),
md(r'''
### M.2 If the 32 experts lived on 8 GPUs (EP = 8)

A thought experiment on real routing. Experts 0–3 sit on GPU 0, experts 4–7 on GPU 1, and so on. A
batch of 32 sequences is split 4 per GPU, which is each token's *home*. Every chosen expert that
lives elsewhere means a copy of the token vector goes out (dispatch) and comes back (combine). The
slowest GPU sets the pace, so what matters is the **busiest GPU's load relative to the average**.

🧠 **Intuition.** A relay team runs at the speed of its slowest runner. If one GPU holds the
popular experts, the other seven finish their work and then wait. Imbalance that cost nothing on
one GPU becomes idle hardware on eight.
'''),
mathbox("straggler time and traffic per token", r'''
**Straggler time.** With experts placed on $G$ GPUs and $n_g$ token copies arriving at GPU $g$, the
expert compute of a step takes time proportional to the busiest GPU:

$$t_{\text{step}} \;\propto\; \max_g n_g \;=\; \underbrace{\frac{\max_g n_g}{\bar n}}_{\text{straggler factor}} \cdot \bar n.$$

A straggler factor of 1.3 wastes $1 - 1/1.3 \approx 23\%$ of the expert compute across the cluster.

**Traffic.** A token whose $k$ copies each stay on its home GPU with probability $p_{\text{home}}$
sends, per layer and per direction,

$$k \cdot d \cdot 2\ \text{bytes} \cdot (1 - p_{\text{home}}).$$

Ours: $4 \cdot 384 \cdot 2 \cdot \tfrac78 \approx 2.6$ KiB out, and the same back, which is the 5.25 KiB in the
table. With random placement, $p_{\text{home}} = 1/G = 12.5\%$.
'''),
code(r'''
if COMPUTE and moe32 is not None and not have("ep"):
    ids, tops = collect_routing(moe32, 4)
    G, per = 8, moe32.moe_layers()[0].E // 8
    rows = []
    for li, t in enumerate(tops):
        home = (torch.arange(t.shape[0]) // GCFG.ctx // 4) % G          # 4 sequences per GPU per batch of 32
        dest = t // per                                                 # [N, k] GPU holding each chosen expert
        recv = torch.bincount(dest.reshape(-1), minlength=G).float()
        stay = float((dest == home[:, None]).float().mean())
        rows.append(dict(layer=li + 1, max_over_mean=float(recv.max() / recv.mean()), stay_home=stay,
                         kib_per_token=moe32.moe_layers()[0].k * GCFG.d * 2 * (1 - stay) * 2 / 1024))
    R["ep"] = rows
    save_results()
if have("ep"):
    display(Markdown("| layer | busiest GPU / average | token copies that stay home | dispatch + combine per token (bf16) |\n|---|---|---|---|\n" + "\n".join(
        f"| {r['layer']} | {r['max_over_mean']:.2f}x | {r['stay_home']:.0%} (1/8 = 12.5% if random) | {r['kib_per_token']:.2f} KiB |" for r in R["ep"])))
'''),
md(r'''
### M.3 Memory follows the total, time follows... it depends

Three models, each trained from fresh weights for a few steps on one micro-batch (32 × 256). Timing
uses **ABBA order** (D, M8, M32, M32, M8, D) after a warm-up. A laptop GPU slows down as it heats,
so whichever model ran first would otherwise look fastest. The mirrored order cancels a linear drift.

The training state is analytic (16 bytes per parameter: fp32 weight, fp32 gradient, two fp32 Adam
moments). The activation peak is measured.
'''),
code(r'''
def bench_models():
    cfg8, cfg32 = copy.deepcopy(MAIN_MOE8), copy.deepcopy(MAIN_MOE8)
    cfg32.n_exp, cfg32.family = 32, 4
    return {"dense": GPT(GCFG), "MoE-8": GPT(GCFG, cfg8), "MoE-32": GPT(GCFG, cfg32)}

if COMPUTE and not have("bench"):
    for v in ("dense_opt", "moe8_opt", "control_opt", "moe32_opt"):
        globals()[v] = None
    torch.cuda.empty_cache()
    models = {k: m.to(DEVICE) for k, m in bench_models().items()}
    opts = {k: make_opt(m) for k, m in models.items()}
    x, y = TRAIN.batch(0, BUDGET["micro_batch"])
    def step(k):
        with amp():
            _, l = models[k](x, y)
        l.backward()
        opts[k].step()
        opts[k].zero_grad(set_to_none=True)
    for k in models:                                   # warm-up and memory
        for _ in range(3):
            step(k)
    mem = {}
    for k in models:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        step(k); torch.cuda.synchronize()
        mem[k] = (torch.cuda.max_memory_allocated() - before) / 2**30
    order = ["dense", "MoE-8", "MoE-32", "MoE-32", "MoE-8", "dense"]
    n = 4 if MODE == "quick" else 15
    times = {k: [] for k in models}
    for k in order:
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            step(k)
        torch.cuda.synchronize(); times[k].append((time.time() - t0) / n)
    R["bench"] = {k: dict(ms=1000 * float(np.mean(times[k])), passes=[round(1000 * t, 1) for t in times[k]],
                          act_gib=mem[k], state_gib=n_params(models[k]) * 16 / 2**30,
                          params=n_params(models[k]), active=n_active(models[k])) for k in models}
    del models, opts
    torch.cuda.empty_cache()
    save_results()
if have("bench"):
    b = R["bench"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.2))
    names = list(b)
    yy = np.arange(len(names))
    a1.barh(yy - 0.2, [b[k]["state_gib"] for k in names], height=0.36, color=SERIES[0], label="training state (16 B/param)")
    a1.barh(yy + 0.2, [b[k]["act_gib"] for k in names], height=0.36, color=SERIES[2], label="step peak (activations, grads)")
    a1.set_yticks(yy, names); a1.invert_yaxis()
    a1.set(xlabel="GiB", title="Memory")
    a1.legend(fontsize=8)
    a2.barh(yy, [b[k]["ms"] for k in names], height=0.5, color=[DENSE_C, MOE_C, MOE_C])
    for i, k in enumerate(names):
        a2.text(b[k]["ms"], i, f"  {b[k]['ms']:.0f} ms  ({b[k]['active'] / 1e6:.1f}M active, {b[k]['params'] / 1e6:.0f}M total)",
                va="center", color=INK2, fontsize=8.5)
    a2.set_yticks(yy, names); a2.invert_yaxis()
    a2.set(xlabel="ms per training step (32 x 256 tokens)", title="Time per step (ABBA mean)")
    a2.set_xlim(0, max(b[k]["ms"] for k in names) * 1.9)
    savefig(fig, "bench_memory_time")
'''),
md(r'''
Same active parameters, so the same FLOPs per token, yet the MoE steps are slower on one GPU. The
FLOPs are the same, but they arrive as batched matmuls over padded expert groups plus a sort, a
gather and a scatter. This is the single-GPU face of the general rule that MoE trades compute
for memory *and* data movement. At scale it is the all-to-all; here it is kernel launches and
memory traffic.

### M.4 The calculator, checked against our models, and a layout recipe
'''),
code(r'''
def our_count(E=None, k=None, w=None, shared=0):
    if E is None:
        return count_moe(L=GCFG.n_layer, d=GCFG.d, vocab=GCFG.vocab, n_q=GCFG.n_head, n_kv=GCFG.n_kv, hd=GCFG.d // GCFG.n_head,
                         E=0, k=0, w=0, tied=True, dense_ffn=GCFG.ffn)
    return count_moe(L=GCFG.n_layer, d=GCFG.d, vocab=GCFG.vocab, n_q=GCFG.n_head, n_kv=GCFG.n_kv, hd=GCFG.d // GCFG.n_head,
                     E=E, k=k, w=w, shared=shared, tied=True)

norms = (2 * GCFG.n_layer + 1) * GCFG.d
cfg32 = copy.deepcopy(MAIN_MOE8); cfg32.n_exp = 32
for label, (tot, act), cfg in (("dense", our_count(), None), ("MoE-8", our_count(8, 4, 192, 768), MAIN_MOE8),
                               ("MoE-32", our_count(32, 4, 192, 768), cfg32)):
    m = GPT(GCFG, cfg)
    print(f"{label:<7} calculator {tot + norms:>11,} total, {act + norms:>11,} active | model {n_params(m):>11,} total, {n_active(m):>11,} active")
    assert tot + norms == n_params(m) and act + norms == n_active(m)
    del m
gate("parameter calculator == the models' numel (dense, MoE-8, MoE-32)", True)

# Qwen3-30B-A3B shape on one node with EP = 8
exp_gib = 28.99e9 / 8 * 16 / GiB
dense_gib = 1.54e9 * (4 + 12 / 8) / GiB
act_gib = 8192 * 2048 * 34 * 48 / GiB
print(f"reference model, EP=8: experts {exp_gib:.1f} GiB + dense parts (ZeRO-1 over 8) {dense_gib:.1f} GiB = state "
      f"{exp_gib + dense_gib:.1f} GiB; + activations {act_gib:.1f} GiB = {exp_gib + dense_gib + act_gib:.1f} GiB "
      f"-> fits B200 (167.6), not H100 (74.5)")
dense16 = 1.54e9 * (4 + 12 / 16) / GiB                       # two nodes: ZeRO-1 over 16 copies
print(f"two H100 nodes, EP=16: experts {exp_gib / 2:.1f} + dense {dense16:.1f} = state {exp_gib / 2 + dense16:.1f} GiB; "
      f"with activations {exp_gib / 2 + dense16 + act_gib:.1f} GiB -> fits")
assert round(exp_gib, 1) == 54.0 and round(dense_gib, 1) == 7.9 and round(act_gib, 1) == 25.5
assert round(exp_gib + dense_gib + act_gib, 1) == 87.4
assert round(exp_gib / 2 + dense16, 1) == 33.8 and round(exp_gib / 2 + dense16 + act_gib, 1) == 59.3
'''),
md(r'''
| step | reference model on 8 × B200 | our MoE-32 on one laptop GPU |
|---|---|---|
| 1. count total and active | 30.53B / 3.35B | 68M / 18.9M |
| 2. expert parallelism = GPUs in a node | EP = 8, 16 experts per GPU | EP = 1, all 32 experts local |
| 3. ZeRO-1 for the dense parts | 7.9 GiB per GPU | not needed |
| 4. training state per GPU | 61.9 GiB | ~1.0 GiB (16 B × 68.5M) |
| 5. add activations | +25.5 GiB per 8K-token sequence | see M.3 |
| 6. tokens per expert per step | 4,096 (≥ ~280 to keep matrix units busy) | 8,192 × 4 / 32 = 1,024 per micro-batch |
| 7. all-to-all vs compute | 0.050 s vs 0.073 s per sequence | none (one GPU) |
| 8. cross nodes only if it does not fit | not needed on B200; EP = 16 on H100 | not needed |
| 9. measure the step time | Megatron: EP 8, no TP/PP | M.3 |

**Carry this forward:** dropping tokens is a quality cost you no longer need to pay. Placement turns
routing imbalance into GPU stragglers. On one GPU an MoE has the FLOPs of its active size but runs
slower than a dense model of that size.
'''),
]
