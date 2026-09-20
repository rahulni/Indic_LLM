# ZeRO on 32 Virtual GPUs: Plan

> **Status:** approved and built. The file map in §10 reflects what was actually built.
> **Folder:** `e:/Projects/ERA_V5/12_Distributed_ZeRO/`, in repo `rahulni/Indic_LLM`.
> **Git:** you do it. I make no commits or pushes. At the end I give you a suggested commit message and the final link.
>
> **How to read this:**
> - §0 is the one-minute version.
> - §1–§4 explain the idea with pictures.
> - §5–§7 cover what we predict, how we prove it and what you'll see.
> - §8–§12 are what gets built (notebook, README, files, agent team, verification).
> - §13 lists the choices you might want to change.
> - The appendices hold the engineering detail.

---

## 0. TL;DR

1. **32 virtual GPUs = 32 threads in one Python process.** Each has its own memory ledger and an optional memory limit. They talk only through collectives (all-reduce, reduce-scatter, all-gather), like real GPUs under `torchrun`.
2. **A tiny GPT** (0.81M params, character-level Shakespeare) trains on them under **DDP (ZeRO-0), ZeRO-1, ZeRO-2 and ZeRO-3**.
3. **What we prove, with asserts rather than plots alone:**
   - all four stages produce **bit-identical** training;
   - per-GPU memory matches the ZeRO paper's formulas **to the byte**;
   - communication is 2Ψ / 2Ψ / 2Ψ / 3Ψ;
   - forward and backward compute is unchanged, and optimizer work drops by a factor of N.
4. **Extras:**
   - an out-of-memory ladder where only ZeRO-3 survives;
   - a ring all-reduce built by hand;
   - a check of the ledger against the **real GPU's allocator**;
   - a cross-check on **real `torch.distributed`** (gloo);
   - real-hardware estimates for 7B–70B models on 32 A100s or H100s.
5. **Deliverables:**
   - an executed notebook;
   - a README that teaches the topic (generated from the run's numbers);
   - a one-page `CHEATSHEET.md` for revision;
   - unit tests.

### What changed after my self-review of the first draft

- **Added pictures:** the ZeRO picture (§1), the architecture (§2), the collectives (§2), step timelines (§3), a mock-up of the figures (§7), and a decision flowchart (§10).
- **Found by probing:** PyTorch's FLOP counter **records 0 FLOPs for CPU attention**. The model therefore uses explicit matrix-multiply attention, so compute is fully counted.
- **Added GPU mode.** All 32 virtual GPUs can live on the **real RTX 3070, or a Colab GPU**, which is the "or Colab GPU" you mentioned. The ledger for all 32 ranks is then checked against the CUDA allocator (probe: 32 threads on the GPU work and repeat bit-for-bit).
- **Added "it really trains":** a longer ZeRO-3 run, then **consolidating the 32 shards into one checkpoint** and generating Shakespeare-like text.
- **Added:** a heatmap of all 32 GPUs, a per-rank compute ledger (FLOPs by phase), a glossary, a decision guide, a self-quiz and `CHEATSHEET.md`.
- **Changed:** out-of-memory capacities are *derived from the predicted peaks* rather than hard-coded. The engineering rules moved to Appendix A.

---

## 1. The idea in one picture

Training with Adam in mixed precision costs **16 bytes per parameter** before any activations:

| Item | Bytes per parameter |
|---|---|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam m | 4 |
| Adam v | 4 |

So a 7B model needs 112 GB, which no single GPU has. **DDP keeps a full copy of all 16 bytes on every GPU.** ZeRO removes that duplication one piece at a time.

**Who owns what** (4 GPUs; ■ = holds that quarter, · = does not hold it):

```text
                    GPU0     GPU1     GPU2     GPU3
DDP (ZeRO-0)  W    ■■■■     ■■■■     ■■■■     ■■■■     everything replicated 4×
              G    ■■■■     ■■■■     ■■■■     ■■■■
              OS   ■■■■     ■■■■     ■■■■     ■■■■
ZeRO-1        W    ■■■■     ■■■■     ■■■■     ■■■■
              G    ■■■■     ■■■■     ■■■■     ■■■■
              OS   ■···     ·■··     ··■·     ···■     ← optimizer states split
ZeRO-2        W    ■■■■     ■■■■     ■■■■     ■■■■
              G    ■···     ·■··     ··■·     ···■     ← + gradients split
              OS   ■···     ·■··     ··■·     ···■
ZeRO-3        W    ■···     ·■··     ··■·     ···■     ← + weights split
              G    ■···     ·■··     ··■·     ···■       (gathered just-in-time,
              OS   ■···     ·■··     ··■·     ···■        freed right after use)
W = weights (2Ψ)   G = gradients (2Ψ)   OS = optimizer states: fp32 master + Adam m + v (12Ψ)
```

**How much each GPU holds** (N = 4, one block ≈ 0.5Ψ bytes):

```text
             weights 2Ψ   grads 2Ψ   optimizer states 12Ψ           per GPU
DDP          ████         ▓▓▓▓       ░░░░░░░░░░░░░░░░░░░░░░░░        16.0 Ψ
ZeRO-1       ████         ▓▓▓▓       ░░░░░░                           7.0 Ψ   = 4Ψ + 12Ψ/N
ZeRO-2       ████         ▓          ░░░░░░                           5.5 Ψ   = 2Ψ + 14Ψ/N
ZeRO-3       █            ▓          ░░░░░░                           4.0 Ψ   = 16Ψ/N
```

**Analogy for revision:** 32 students preparing for one exam.
- **DDP:** every student photocopies the whole textbook, notes and answer key.
- **ZeRO-1:** each student keeps only 1/32 of the answer key, the biggest pile.
- **ZeRO-2:** each student also keeps only 1/32 of the notes.
- **ZeRO-3:** each student keeps only 1/32 of the textbook too. Before reading chapter k, everyone briefly borrows the rest of it, then hands it back.

What you pay for all this is **passing pages around** (communication), not extra thinking (compute).

### What changes between stages, in plain words

| | DDP (ZeRO-0) | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| Whole weights on every GPU? | yes | yes | yes | **no**: gather per unit, use, free |
| Whole gradients kept? | yes | yes, until backward ends | **no**: reduce-scattered as each unit finishes | no |
| Optimizer states | whole | **1/N** | 1/N | 1/N |
| Gradient sync | all-reduce | reduce-scatter | reduce-scatter (early) | reduce-scatter |
| After the optimizer step | nothing | all-gather weights | all-gather weights | nothing: weights stay sharded |
| Communication per step | 2Ψ | 2Ψ | 2Ψ | **3Ψ** (1.5×) |
| Forward/backward FLOPs | F | F | F | F (unchanged) |
| Optimizer work per GPU | Ψ | **Ψ/N** | Ψ/N | Ψ/N |
| DeepSpeed / FSDP name | stage 0 / `NO_SHARD` | stage 1 | stage 2 / `SHARD_GRAD_OP` | stage 3 / `FULL_SHARD` |

---

## 2. How we build 32 virtual GPUs

**Why threads.** The machine has only about 1.5 GB of RAM free, and every Python process loads torch at 300–500 MB, so 32 processes can't run. Threads share one copy of torch, and PyTorch releases Python's global interpreter lock (GIL) inside its operations.

```text
┌──────────────────────────── one Python process ─────────────────────────────┐
│  VirtualCluster(world=32).run(train_step)        ≈  torchrun --nproc 32     │
│                                                                             │
│    thread 0            thread 1                          thread 31          │
│   ┌────────────┐      ┌────────────┐                   ┌────────────┐       │
│   │ vGPU 0     │      │ vGPU 1     │        ...        │ vGPU 31    │       │
│   │ • ledger   │      │ • ledger   │                   │ • ledger   │       │
│   │ • capacity │      │ • capacity │                   │ • capacity │       │
│   │ • shard 0  │      │ • shard 1  │                   │ • shard 31 │       │
│   │ • data 0   │      │ • data 1   │                   │ • data 31  │       │
│   └─────┬──────┘      └─────┬──────┘                   └─────┬──────┘       │
│         └───────────────────┴──────────────┬─────────────────┘              │
│                    ThreadComm  (barriers + shared slots)                    │
│        all_reduce · reduce_scatter · all_gather · broadcast · barrier       │
│      charges each GPU's bytes with the ring model: 2(N−1)/N · S, etc.       │
└─────────────────────────────────────────────────────────────────────────────┘
 Tensors live on:  CPU (default, fully reproducible)  or  the real GPU (GPU mode)
```

**The ledger** is each virtual GPU's memory bookkeeping.
- Every buffer the algorithm holds is recorded as name → (category, bytes). Categories: `weights`, `grads`, `master`, `adam_m`, `adam_v`, `activations`, `temp`.
- It tracks the current total, the peak, a timeline, and the category breakdown at the peak.
- With a capacity set, it raises a CUDA-style `VirtualOOMError` **before** allocating.
- Activations are counted automatically, per thread, through PyTorch's `saved_tensors_hooks`.

**The three collectives** (4 GPUs, each starting with a vector of 4 chunks):

```text
 REDUCE-SCATTER                     ALL-GATHER                       ALL-REDUCE  =  RS then AG
 in   r0 [a0 a1 a2 a3]              in   r0 [x0]                     in   r0 [a0 a1 a2 a3]
      r1 [b0 b1 b2 b3]                   r1 [x1]                          r1 [b0 b1 b2 b3]
      r2 [c0 c1 c2 c3]                   r2 [x2]                          r2 [c0 c1 c2 c3]
      r3 [d0 d1 d2 d3]                   r3 [x3]                          r3 [d0 d1 d2 d3]
 out  r0 [Σ0] r1 [Σ1]               out  every rank                  out  every rank
      r2 [Σ2] r3 [Σ3]                    [x0 x1 x2 x3]                    [Σ0 Σ1 Σ2 Σ3]
 Σk = ak+bk+ck+dk
 bytes sent per GPU (ring):  (N−1)/N · S        (N−1)/N · S                 2(N−1)/N · S
```

- ZeRO uses reduce-scatter where DDP uses all-reduce, then an all-gather for the weights. That totals the **same 2Ψ as DDP**. This is the key insight behind ZeRO-1 and ZeRO-2.
- The step-by-step ring all-reduce (§8, section 3) proves those byte counts by passing chunks between neighbouring threads.

---

## 3. One training step under each stage

The model is cut into **6 units**: `E` = embeddings, `B0`–`B3` = transformer blocks, `F` = final norm + head. Communication happens **per unit**.

```text
               ───────── forward ────────►   ◄──────── backward ─────────   ──── optimizer step ────
DDP          [E][B0][B1][B2][B3][F]        [F][B3][B2][B1][B0][E]          Adam on all Ψ
                                            AR  AR  AR  AR  AR  AR
ZeRO-1       [E][B0][B1][B2][B3][F]        [F][B3][B2][B1][B0][E]  RS×6    Adam on Ψ/N ─► AG×6
ZeRO-2       [E][B0][B1][B2][B3][F]        [F][B3][B2][B1][B0][E]          Adam on Ψ/N ─► AG×6
                                            RS  RS  RS  RS  RS  RS   ← each grad freed right after its RS
ZeRO-3       AG  AG  AG  AG  AG  AG         AG  AG  AG  AG  AG  AG          Adam on Ψ/N   (no gather)
             [E][B0][B1][B2][B3][F]        [F][B3][B2][B1][B0][E]
              ↓ weights freed after use     RS  RS  RS  RS  RS  RS   ← weights + grads freed
AR = all-reduce   RS = reduce-scatter   AG = all-gather      collective calls/step: 6 · 12 · 12 · 18
```

- **How ZeRO-3 frees weights without breaking autograd:** this is the trick FSDP uses.
  - After forward, a unit's flat weight buffer is shrunk to 0 bytes with `untyped_storage().resize_(0)`.
  - Before backward, the buffer is regrown and re-gathered.
  - Autograd's saved references point at that same storage, so they see the right values again.
  - The probe confirmed that gradients come out bit-identical.
  - Reading a freed weight **crashes the process** instead of raising an error, so a guard hook raises a readable error first.
- **Code shape.** One executor runs every stage. Stages differ only in five hooks: `before_forward`, `after_forward`, `before_backward`, `after_backward` and `step`. The README prints them side by side, so the *difference between the stages is literally the difference between those hooks*.

---

## 4. The demo model and data

```text
TinyGPT: d=128, 4 layers, 4 heads, context 64, char vocab 65 (tiny Shakespeare), untied head
Attention is written as explicit matmuls, so the FLOP counter sees it (SDPA on CPU counts 0).

 unit      contents                     params    padded (to N·64)   shard per GPU (N=32)
 embed     wte 65×128 + wpe 64×128      16,512        18,432                576
 block0    LN · QKV · proj · LN · MLP  197,120       198,656              6,208
 block1-3  same as block0              197,120 ×3    198,656 ×3           6,208 ×3
 final     LN_f + head 128×65            8,576        10,240                320
 total                              Ψ = 813,568   Ψ′ = 823,296 (+1.2%)   25,728

 block0 weights ──flatten──► [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■·pad·]  198,656 bf16 values
                              │ GPU0 │ GPU1 │ GPU2 │  ...          │ GPU31 │   6,208 each

 data:  global batch = 32 sequences × 64 characters ──► GPU r trains on sequence r
        (data parallel: same model, different data, gradients averaged)
```

- **Precision:** `bf16-mixed`, which is the paper's 2+2+12 accounting. fp32 is used only for the "DDP = one big GPU" check and the gloo check.
- **Quick configuration:** d=64, 2 layers, context 32, 3 steps. It runs in under a minute and is the default on Colab's 2 vCPUs.

---

## 5. Predictions, written before any run (the notebook asserts them)

**Memory per GPU for model states, steady state (N=32).** The ledger must match to the byte.

| Stage | Formula | Bytes | MiB | vs DDP |
|---|---|---:|---:|---:|
| DDP | 16Ψ′ | 13,172,736 | 12.56 | 1× |
| ZeRO-1 | 4Ψ′ + 12Ψ′/N | 3,601,920 | 3.44 | 3.7× smaller |
| ZeRO-2 | 2Ψ′ + 14Ψ′/N | 2,006,784 | 1.91 | 6.6× smaller |
| ZeRO-3 | 16Ψ′/N | 411,648 | 0.39 | 32× smaller |

**Compute and communication per GPU per step (N=32).** The counters must match.

| | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---:|---:|---:|---:|
| Forward/backward FLOPs | F | F | F | F |
| Optimizer elements updated | 823,296 | 25,728 | 25,728 | 25,728 |
| Bytes sent (ring model) | 3.19 MB | 3.19 MB | 3.19 MB | **4.79 MB** |
| Collective calls | 6 | 12 | 12 | 18 |

**Peak memory** = steady state + activations + temporary buffers. It is checked within a tolerance, not exactly.
- ZeRO-2's temporary buffer is one unit's gradient.
- ZeRO-3's is one unit's weights plus its gradient.

At this toy size those temporary buffers dominate ZeRO-3's peak. That is itself a lesson, shown by the wrap-granularity experiment. Notebook sections 12 (scaling with N) and 19 (real hardware) show the scale where ZeRO pays off.

**Preview of the real-hardware table** (model states only, N=32, GB = 10⁹ bytes; activations are added in the notebook):

| Model | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 | Fits in 80 GB with... |
|---|---:|---:|---:|---:|---|
| 1.5B | 24 | 6.6 | 3.7 | 0.75 | anything |
| 7B | 112 | 30.6 | 17.1 | 3.5 | ZeRO-1 or later |
| 13B | 208 | 56.9 | 31.7 | 6.5 | ZeRO-1 (tight), ZeRO-2 |
| 70B | 1120 | 306 | 171 | 35 | **ZeRO-3 only** |

Communication hides behind compute once each GPU processes about `(2/3)·F·MFU / BW` tokens per step, where F is peak FLOP/s, MFU is the fraction of it achieved, and BW is per-GPU network bandwidth. That threshold **does not depend on model size**.
- A100 with HDR InfiniBand: about 3.3k tokens (DDP, ZeRO-1, ZeRO-2) and about 5.0k (ZeRO-3).
- H100 with NDR InfiniBand: about 5.3k and 7.9k.
- Faster GPUs need *more* tokens per GPU to hide communication.

---

## 6. How each claim is proven

| # | Claim | Evidence | Where |
|---|---|---|---|
| 1 | ZeRO changes memory, not the maths | Losses **and** final weights are bit-identical (`torch.equal`) across all 4 stages at N=32 | nb §9 + test |
| 2 | Memory per GPU follows the paper | Ledger steady state == formula, to the byte | nb §10 + test |
| 3 | ZeRO-1/2 communicate as much as DDP; ZeRO-3 1.5× | Byte counters == formula | nb §11 + test |
| 4 | Forward/backward compute unchanged; optimizer work ÷N | Per-rank FLOP counter equal across stages; element counts | nb §11 |
| 5 | DDP on 32 GPUs == one GPU with the whole batch | fp32 `allclose` | nb §9 |
| 6 | The collectives are real algorithms | Hand-built ring == library result; bytes == 2(N−1)/N·S | nb §3 + test |
| 7 | The ledger isn't fiction | Change in allocated memory on the real GPU (all 32 ranks) == ledger, within allocator rounding | nb §17 |
| 8 | The same engine works on real `torch.distributed` | gloo run `allclose` to the thread simulator at the same N | nb §18 |
| 9 | Memory limits bite exactly where predicted | Out-of-memory ladder: predicted fits == observed | nb §16 |
| 10 | It really trains | ZeRO-3 run, shards consolidated into one checkpoint, generated text | nb §15 |
| 11 | The paper's numbers are reproduced | ZeRO paper Figure 1 (7.5B, N=64): 120 / 31.4 / 16.6 / 1.9 GB | nb §1 + test |

---

## 7. The figures (all dark theme, all generated from the run)

| File | Shows | What to look for |
|---|---|---|
| `what_each_gpu_holds.png` | The §1 picture, drawn from the **measured** ledger | Bars shrink stage by stage and hit the formula ticks |
| `cluster_heatmap.png` | 4 panels, each a 4×8 grid of all 32 GPUs, coloured by peak MiB | Every GPU shrinks, not just rank 0 |
| `memory_timeline.png` | Rank-0 memory through one step, per stage | Flat for DDP, sawtooth for ZeRO-3 |
| `comm_and_compute.png` | Bytes by collective, call counts, FLOPs by phase, optimizer elements | ZeRO-3 is the 1.5× bar; FLOP bars equal |
| `scaling_with_N.png` | Memory per GPU for N = 1 … 32 (log-log) plus formula lines | ZeRO-1/2 flatten at 4Ψ and 2Ψ; ZeRO-3 keeps falling |
| `loss_curves.png` | 4 stages overlaid, max \|Δ\| printed | One line, because they are identical |
| `ring_allreduce.png` | N=4, which chunk each GPU holds at each step | Reduce-scatter phase, then all-gather phase |
| `oom_ladder.png` | Capacity × stage grid, predicted vs observed | Only ZeRO-3 survives the smallest GPU |
| `grad_accum_and_ckpt.png` | Communication vs accumulation steps G; activations and FLOPs with checkpointing | ZeRO-2/3 communicate every micro-batch; checkpointing costs 6→8ΨD FLOPs |
| `real_hardware.png` | 1.5B–70B memory per GPU per stage against an 80 GB line | 70B fits only with ZeRO-3 |

**Mock-ups** (illustrative only; the real figures come from data):

```text
memory_timeline (rank 0, one step)          scaling_with_N (log-log)
MiB                                         per-GPU model states
13 ┤DDP ━━━━━━━━━━━━━━━━━━━━━━━━━━          16Ψ ┤●━━━━━━━━━━━━━━━━━━━  DDP (never shrinks)
   │                                            │ ╲
 4 ┤Z1  ━━━━━━━━━━━━━━━━━━━━━━━━━━           4Ψ ┤  ╲━━━━━━━━━━━━━━━━━━  ZeRO-1 → floor 4Ψ
 2 ┤Z2  ━━━━━━━━━━━━━━━━╮╭╮╭╮╭━━━           2Ψ ┤    ╲━━━━━━━━━━━━━━━━  ZeRO-2 → floor 2Ψ
 1 ┤Z3  ╱╲╱╲╱╲╱╲╱╲╱╲  ╱╲╱╲╱╲╱╲╱╲╱╲              │      ╲
   └── forward ──┴── backward ──┴ step          │         ╲             ZeRO-3 = 16Ψ/N
       gather → use → free, per unit            └─┬──┬──┬──┬──┬──┬─► N
                                                  1  2  4  8  16 32
```

---

## 8. Notebook outline: `zero_32_virtual_gpus.ipynb`

Every section follows four steps: **Why → Code → Evidence (asserted) → What you should see**. Each ends with a one-line **Takeaway**, so skimming the takeaways alone is a revision pass.

**Part I: Concepts**
- **0. Setup.**
  - `QUICK_RUN` flag, `SEED=1337`, `DEVICE` (`cpu` or `cuda`), dark theme.
  - Quick runs write only to `assets/quick/`.
  - Fetches `zero_sim.py` and `zero_theory.py` on Colab.
- **1. The memory bill.** Where 16Ψ comes from, and the ZeRO paper's Figure 1 reproduced (asserted).

**Part II: The machine**
- **2. 32 virtual GPUs.**
  - Ledger, capacity and OOM message.
  - "Hello collectives": every rank contributes its rank id, and the results are asserted.
- **3. Ring all-reduce by hand.**
  - Neighbour-to-neighbour chunk passing, the N=4 heatmap, byte counts at N=32.
  - Shows that all-reduce = reduce-scatter + all-gather.

**Part III: The model**
- **4. TinyGPT, the data, and the units.**
  - Parameter table, padding, shard sizes (including N=3, to show uneven splits).
  - Data sharding across GPUs.

**Part IV: The four stages.** Each section renders its hook code inline, runs one step, and shows the ledger.
- **5. DDP.**
- **6. ZeRO-1.**
- **7. ZeRO-2.**
- **8. ZeRO-3.**

**Part V: Evidence**
- **9. Correctness.**
  - 25 steps per stage: bit-identical losses and weights.
  - DDP on 32 GPUs == one GPU (fp32).
- **10. Memory.** Measured vs formula, the cluster heatmap, the timeline, wrap granularity (1 unit vs per-block).
- **11. Compute and communication.**
  - FLOPs by phase per rank, optimizer elements, bytes and calls.
  - Modelled time from the α-β model. Wall-clock time is a footnote only: CPU threads mostly measure lock contention.
- **12. Scaling N = 1, 2, 4, 8, 16, 32.**
- **13. Gradient accumulation (G=4) and activation checkpointing.**
  - Which stages communicate every micro-batch.
  - ZeRO does not shard activations; checkpointing does reduce them.
- **14. When to use which stage.** Decision guide.
- **15. It really trains.**
  - A longer ZeRO-3 run.
  - Consolidate the 32 shards into one checkpoint (what DeepSpeed's `zero_to_fp32.py` does).
  - Generate text.

**Part VI: Stress tests and reality checks**
- **16. Out-of-memory ladder.**
  - Capacities are chosen from the *predicted* peaks with a margin of 15% or more.
  - Prediction first, then the run, then the comparison.
- **17. GPU mode.**
  - All 32 virtual GPUs on the real GPU.
  - Ledger vs change in `torch.cuda.memory_allocated`. Skipped with a stated reason if there is no CUDA.
- **18. Real `torch.distributed`.**
  - `tools/gloo_check.py` runs the same ZeRO-1 engine over gloo.
  - 2 processes here, 4 on Colab; `allclose` to the simulator.
  - Skipped with a recorded reason if RAM is too low.
- **19. Real hardware.**
  - 1.5B, 7B, 13B and 70B on 32 A100s or H100s: memory including activations (with and without recomputation), and communication vs compute time.
  - Hybrid sharding (shard within an 8-GPU node, replicate across nodes) as a bonus row.

**Part VII: Revision**
- **20. Cheat sheet, decision guide and a 12-question self-quiz.**
- **21. Save results.** Writes `results.json` with a run stamp.

**Time budget:** full run about 10–15 min on this machine, quick run under 1 min. Step counts are fixed after timing one step and **before** any result is seen.

---

## 9. README outline, plus `CHEATSHEET.md`

The README is generated by `tools/build_readme.py`. The prose lives there, but every number comes from `assets/results.json`. It follows your session 9 style: centred title, Colab and nbviewer badges, evidence tables, `<details>` for derivations, no course branding.

1. **Title, badges and "Open it"** (Colab / nbviewer / GitHub)
2. **TL;DR:** five lines with the headline numbers
3. **ZeRO in one picture:** `what_each_gpu_holds.png` and the ownership grid
4. **The memory bill:** the 16Ψ derivation; a `<details>` on why the fp32 master copy exists
5. **How 32 virtual GPUs are built:** architecture, why threads, CPU vs GPU mode
6. **Collectives and ring all-reduce:** a `<details>` deriving 2(N−1)/N
7. **The four stages:** for each one, what's sharded, the step timeline, about 15 lines of hook code, memory and communication
8. **Results:** measured vs formula tables, all figures
9. **What changes in computation, and what doesn't**
10. **Scaling, gradient accumulation, checkpointing**
11. **Stress tests:** OOM ladder, allocator audit, gloo check
12. **Real hardware:** which stage fits 7B/13B/70B, and when communication hides
13. **Decision guide** (flowchart below)
14. **Revision:** cheat-sheet table and 12 interview questions with collapsed answers
15. **Honest limitations**
16. **Reproducing**
17. **What's in here** (file map)
18. **References**

**Decision guide** (goes into the README and `CHEATSHEET.md`):

```text
              Does 16Ψ + activations fit on one GPU?
                 │ yes                     │ no
                 ▼                         ▼
           DDP (ZeRO-0)          Does 4Ψ + 12Ψ/N fit? ──yes──► ZeRO-1
           least communication             │ no
                                           ▼
                                 Does 2Ψ + 14Ψ/N fit? ──yes──► ZeRO-2  (mind grad-accum comm)
                                           │ no
                                           ▼
                                 ZeRO-3 / FSDP FULL_SHARD   (1.5× communication)
                                           │ still no?
                                           ▼
                activation checkpointing ─► offload (ZeRO-Offload / Infinity)
                                         ─► tensor / pipeline parallelism
```

**Self-quiz questions** (answers collapsed in the README):
1. Where do the 16 bytes per parameter go?
2. Why does ZeRO-1/2 communicate no more than DDP?
3. Where does ZeRO-3's extra Ψ come from?
4. Why does ZeRO-2 communicate on every micro-batch under gradient accumulation?
5. What does ZeRO *not* shard, and what does?
6. How does FSDP free weights without breaking autograd?
7. Why does all-reduce = reduce-scatter + all-gather matter?
8. How do you clip by global gradient norm when the gradients are sharded?
9. How do you save and load a ZeRO-3 checkpoint?
10. When does communication hide behind compute?
11. How does PyTorch's `ZeroRedundancyOptimizer` differ from ZeRO-1 (3Ψ vs 2Ψ)?
12. ZeRO vs tensor vs pipeline parallelism: what does each split?

---

## 10. Files

```text
12_Distributed_ZeRO/
├── PLAN.md                        this plan (copied in on approval)
├── README.md                      generated, teaches the topic
├── CHEATSHEET.md                  one page for revision
├── zero_32_virtual_gpus.ipynb     THE SUBMISSION: executed, all outputs and figures embedded
├── nb_source.py                   single source of the notebook (percent format)
├── zero_sim.py                    engine: VirtualGPU, ThreadComm, VirtualCluster, ring all-reduce,
│                                  TinyGPT units, DDP / ZeRO-1 / ZeRO-2 / ZeRO-3 policies, AdamW
├── zero_theory.py                 formulas, paper Figure 1, real-hardware estimator, overlap threshold
├── zero_plots.py                  the figures (dark theme, validated palette)
├── references.md                  verified citations
├── tools/
│   ├── build_notebook.py          percent-format source → .ipynb
│   ├── build_readme.py            README from results.json; asserts the run stamp
│   └── gloo_check.py              real torch.distributed cross-check (TorchDistComm adapter lives here)
├── tests/
│   ├── test_zero_sim.py           python -m unittest discover -s tests -v
│   ├── test_comm.py
│   └── test_zero_theory.py
├── assets/                        results.json + figures (full run only)
├── requirements.txt
└── .gitignore                     data/  __pycache__/  .ipynb_checkpoints/  .vscode/  assets/quick/
```

**Why the engine is a module and not inline.** The tests and the gloo subprocesses must import the *same* code. The notebook still reads top to bottom, because each key class or hook is rendered inline with `IPython.display.Code(inspect.getsource(...))` next to the cell that uses it.

**Self-contained:** everything is written fresh inside this folder, and nothing is imported or copied from sibling folders. It follows the project's conventions: the nanoGPT-style GPT, the dark plot theme, the percent-format notebook source, the README generated from `results.json`, and stdlib `unittest`.

---

## 11. Build order and the agent team

```text
 STEP 1 · lead (me)        STEP 2 · two agents in parallel            STEP 3 · lead           STEP 4 · agents
┌───────────────────┐    ┌────────────────────────────────────┐   ┌──────────────────┐   ┌─────────────────────────┐
│ zero_sim.py core  │───►│ INFRA ENGINEER: ring all-reduce,   │──►│ nb_source.py     │──►│ TEACHER: README prose,  │
│ ledger · comm ·   │    │ TorchDistComm, gloo_check.py,      │   │ quick run → fix  │   │ CHEATSHEET.md, quiz     │
│ units · executor ·│    │ comm tests                         │   │ time one step →  │   ├─────────────────────────┤
│ 4 policies · Adam │───►│ THEORY / DATA SCIENTIST:           │──►│ fix step counts  │   │ REVIEWER (starts cold): │
│ → API frozen      │    │ zero_theory.py, hardware table,    │   │ full run → assets│   │ audits every claim vs   │
└───────────────────┘    │ plotting helpers, theory tests     │   └──────────────────┘   │ results.json + the paper│
                         └────────────────────────────────────┘                          └───────────┬─────────────┘
                                                                                                     ▼
                                                                                 fixes → final review → hand-off to you
```

- **Build pipeline:**
  - `nb_source.py` → `build_notebook.py` → `.ipynb`
  - `.ipynb` → `nbconvert --execute` → outputs, `assets/*.png` and `assets/results.json`
  - `build_readme.py` (run-stamp check) → `README.md`
- **Why agents write only their own files:** no two agents edit the same file, so parallel work can't collide.

---

## 12. Verification

1. `python -m unittest discover -s tests -v`. Covers:
   - collectives at N = 3, 4 and 32;
   - ring == library result; bytes == formula;
   - the paper's Figure 1;
   - bit-exact stage equivalence at N=4 and N=32;
   - ledger == formula;
   - the release guard raises;
   - out-of-memory aborts without a deadlock;
   - activations return to 0 after backward.
2. `QUICK_RUN=1 python nb_source.py`: every assertion passes, and output goes only to `assets/quick/`.
3. Build the notebook, then `QUICK_RUN=0 jupyter nbconvert --execute --to notebook --inplace zero_32_virtual_gpus.ipynb`.
4. `python tools/build_readme.py`: the run-stamp assert must pass.
5. `python tools/gloo_check.py --world 2`.
6. The reviewer agent's report. Every finding is either fixed or explained.
7. **Hand-off:** files ready and uncommitted.
   - Suggested branch: `assignment-12-zero`, with a commit message in your usual style.
   - Link once you push to `main`: `https://github.com/rahulni/Indic_LLM/tree/main/12_Distributed_ZeRO`. The Colab and nbviewer badges work after that push.

---

## 13. Decisions I made for you (change any before approving)

1. **"ZeRO" = stage 0**, the plain DDP baseline, as in DeepSpeed's `"stage": 0`.
2. **Virtual GPUs are threads.** CPU is the default because it's reproducible everywhere; GPU mode is used for the allocator audit and runs faster.
3. **The headline run uses bf16-mixed precision** (the paper's 2+2+12). Only two checks use fp32.
4. **Model:** TinyGPT, 0.81M params, micro-batch 1 per GPU. It is small because the RAM and bf16-on-CPU budget is tight; the real-hardware section covers scale.
5. **Per-unit sharding** in FSDP's layout, rather than DeepSpeed's single global partition. Memory and communication maths are identical, and this is noted in the README.
6. **The engine is an importable module**, rendered inline in the notebook.
7. **Explicit matmul attention** instead of SDPA, so FLOPs are counted.
8. **`PLAN.md` goes into the folder**, so it's public once pushed. Delete it if you prefer not.
9. **No git operations.** You push.

---

## 14. Honest limitations (these go into the README)

- **Threads are not GPUs.** Memory is *accounted* per GPU and audited against the real allocator only in GPU mode. Wall-clock time says nothing about GPU speed.
- **No overlap.** Communication and compute don't overlap, and ZeRO-3 doesn't prefetch. Overlap is covered analytically (§5).
- **Toy scale.** At 0.81M params, ZeRO-3's temporary buffers and activations dominate its peak. The scaling and hardware sections show the regime where ZeRO matters.
- **Collective bytes follow the ring model.** The actual data moves through shared memory.

---

## Appendix A: Engineering rules (from a design review plus probes on this machine)

**Probes already run:**

| Probe | Result | Design consequence |
|---|---|---|
| Can 32 processes run? | 1.5 GB RAM free; torch takes 300–500 MB per process | Threads |
| `resize_(0)` then re-gather then backward | Gradients bit-identical; reading a freed weight **segfaults** | FSDP trick plus a guard hook |
| 32 threads running autograd plus `saved_tensors_hooks` at once | Works; hooks are per-thread | Per-GPU activation ledger |
| bf16 matmul on this CPU | About 9× slower than fp32 | Micro-batch 1, short runs, quick mode |
| `FlopCounterMode` on CPU SDPA | **Counts 0 FLOPs** | Explicit matmul attention |
| 32 threads on the RTX 3070 | Works and repeats bit-for-bit; 867 MB held by per-thread cuBLAS workspaces | GPU audit compares the change in allocated memory |
| `dist.reduce_scatter_tensor` | Exists; gloo support at runtime is unverified | Fallback chain inside `gloo_check.py` |

**Rules for bit-exact results**
- `all_reduce` is built as reduce-scatter + all-gather, so every stage sums gradients through one code path: rank order, fp32, then ÷N, then cast to bf16.
- AdamW uses only `mul_`, `add_`, `div_` and `sqrt` (fused ops round differently in the SIMD tail).
- Units are padded to multiples of N·64, and the flat layout is identical in every stage.
- No dropout: the random generator is shared across threads.
- `p.grad` is pre-set to views into the flat gradient buffer, and `data_ptr()` is asserted unchanged after backward.

**Threading safety**
- Every collective has entry and exit barriers with timeouts.
- Counters are kept per rank.
- On any error, all barriers are aborted, and the **lowest-rank** error is re-raised.
- Threads are daemons, and Ctrl-C is handled.
- Ring mailboxes use `get(timeout)` plus an abort flag.
- `torch.set_num_threads(1)` is set inside the run and restored afterwards.
- Activation handles capture their own `gpu`, and a leak check runs after backward.

**Semantics notes for the README**
- ZeRO-1 keeping full gradients matches DeepSpeed stage 1 and Megatron's distributed optimizer.
- PyTorch's `ZeroRedundancyOptimizer` costs 3Ψ.
- DeepSpeed bf16 and Megatron keep fp32 gradient buffers (4Ψ).
- FSDP keeps the root unit gathered after forward. We don't, which keeps exactly 3Ψ.
- The loss all-reduce is tagged separately and excluded from the communication asserts.
- Global-norm clipping under sharding needs one extra all-reduce. It is off by default and explained in a `<details>`.

**gloo check**
- Plain subprocesses, not `mp.spawn`.
- A fresh FileStore in the scratch directory.
- `CUDA_VISIBLE_DEVICES=""`.
- fp32 SUM, then ÷N.
- Fallbacks: `reduce_scatter_tensor`, then list `reduce_scatter`, then all_reduce + slice.
- A timeout kills the processes.
- A skip is recorded in `results.json` with its reason.

## Appendix B: References (arXiv IDs re-checked during implementation)

- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, arXiv:1910.02054
- Ren et al., *ZeRO-Offload*, arXiv:2101.06840
- Rajbhandari et al., *ZeRO-Infinity*, arXiv:2104.07857
- Wang et al., *ZeRO++*, arXiv:2306.10209
- Zhao et al., *PyTorch FSDP*, arXiv:2304.11277
- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, arXiv:2205.05198
- Patarasuk & Yuan, *Bandwidth optimal all-reduce algorithms for clusters of workstations*, JPDC 2009
- NVIDIA DGX A100 / H100 datasheets, for the bandwidth and FLOP/s assumptions
