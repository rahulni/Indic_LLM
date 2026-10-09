# ZeRO cheat sheet

A page for a 5-minute revision. The derivations and the measured evidence are in the [README](README.md).
Ψ = parameters · N = data-parallel GPUs · b<sub>w</sub>, b<sub>g</sub> = bytes per weight / gradient element ·
K = optimizer bytes per parameter · G = micro-batches per optimizer step · U = units (wrapped modules).
Communication is counted in **elements**, as the paper does: bytes = elements × 2 in bf16, and the exact ring cost
adds a factor (N−1)/N.

## 1. Memory per GPU (model states only)

bf16 + Adam: b<sub>w</sub> = b<sub>g</sub> = 2 and K = 12 (fp32 master 4 + Adam m 4 + Adam v 4), so **16 bytes per parameter**.

| Stage (paper name) | Shards | Per GPU, general | bf16 + Adam | Floor as N → ∞ |
|---|---|---|---|---|
| DDP = ZeRO-0 | nothing | (b<sub>w</sub> + b<sub>g</sub> + K)Ψ | 16Ψ | 16Ψ |
| ZeRO-1 (P<sub>os</sub>) | optimizer states | (b<sub>w</sub> + b<sub>g</sub>)Ψ + KΨ/N | 4Ψ + 12Ψ/N | 4Ψ |
| ZeRO-2 (P<sub>os+g</sub>) | + gradients | b<sub>w</sub>Ψ + (b<sub>g</sub> + K)Ψ/N | 2Ψ + 14Ψ/N | 2Ψ |
| ZeRO-3 (P<sub>os+g+p</sub>) | + weights | (b<sub>w</sub> + b<sub>g</sub> + K)Ψ/N | 16Ψ/N | → 0 |

- The paper's check (7.5B, N = 64): **120 / 31.4 / 16.6 / 1.9 GB**. And 1T parameters on 1024 GPUs under ZeRO-3 is ≈ 16 GB per GPU.
- fp32 gradient buffers (DeepSpeed's bf16 optimizer, Megatron-LM) give b<sub>g</sub> = 4: DDP 18Ψ, ZeRO-1 6Ψ + 12Ψ/N, ZeRO-2 2Ψ + 16Ψ/N.
- **Never sharded by ZeRO-DP:** activations, transients (ZeRO-2: one unit's full gradient; ZeRO-3: one unit's gathered weights + its gradient), fragmentation.
- Why the fp32 master: bf16 keeps 8 significant bits (7 stored), so 1.0 + 0.001 = 1.0; small updates vanish without an fp32 copy.

## 2. The three collectives (ring; S = message bytes; cost is bytes *sent per GPU*)

```text
 REDUCE-SCATTER  (N−1)/N·S         ALL-GATHER  (N−1)/N·S          ALL-REDUCE = RS + AG  2(N−1)/N·S
 r0 [a0 a1 a2 a3] → r0 [Σ0]        r0 [x0] → every rank            r0 [a0 a1 a2 a3] → every rank
 r1 [b0 b1 b2 b3] → r1 [Σ1]        r1 [x1]   [x0 x1 x2 x3]         r1 [b0 b1 b2 b3]   [Σ0 Σ1 Σ2 Σ3]
 r2 [c0 c1 c2 c3] → r2 [Σ2]        r2 [x2]                         ...
 r3 [d0 d1 d2 d3] → r3 [Σ3]        r3 [x3]                         Σk = ak + bk + ck + dk
```

The ring takes N−1 steps per half, and every step sends S/N bytes. Time = 2(N−1)·α + 2(N−1)/N · S/β. The bandwidth
term tends to 2S/β, **independent of N**, and is the proven lower bound (Patarasuk & Yuan). The latency term grows with N.

## 3. Communication and compute per step

| Stage | Gradient sync | After the step | Volume | With G micro-batches | Calls |
|---|---|---|---|---|---|
| DDP | AR each unit during backward | nothing | **2Ψ** | 2Ψ (accumulate locally) | U |
| ZeRO-1 | RS after backward | AG weights | **2Ψ** | 2Ψ | 2U |
| ZeRO-2 | RS each unit during backward, then free | AG weights | **2Ψ** | **(G+1)Ψ** (no full buffer to accumulate into) | (G+1)U |
| ZeRO-3 | AG before forward + AG before backward + RS | nothing (stays sharded) | **3Ψ = 1.5×** | **3GΨ** | 3GU |

- **Why ZeRO-1/2 are free:** AR = RS + AG. ZeRO reduce-scatters the gradients and spends the all-gather on updated *weights*.
- **Same maths, same bits?** Only with the same reduction order (e.g. G = 1). Under accumulation, ZeRO-2/3 reduce per micro-batch, while DDP/ZeRO-1 add locally first, so the last bits differ.
- **HSDP** holds N/g × ZeRO-3's states (4× at N = 32, g = 8). It saves *time*: only a 1/g gradient shard crosses the inter-node link.
- **Compute:** forward and backward FLOPs are unchanged (≈ 6Ψ per token; 8Ψ with full recomputation). Optimizer work per GPU is Ψ → Ψ/N.

## 4. Which stage

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

On 32 × 80 GB, counting activations with selective recomputation at 4k context and micro-batch 1: 7B fits from ZeRO-1,
13B from ZeRO-2 (ZeRO-1 needs full recomputation), and 70B only with ZeRO-3 plus full recomputation (states ≈ 34.5 GB).

## 5. Framework names

| | DeepSpeed `"zero_optimization": {"stage": k}` | PyTorch FSDP `ShardingStrategy` | FSDP2 `fully_shard` |
|---|---|---|---|
| DDP | `0` | `NO_SHARD` | (use DDP) |
| ZeRO-1 | `1` | none (`ZeroRedundancyOptimizer` shards the optimizer but costs **3Ψ**) | none |
| ZeRO-2 | `2` | `SHARD_GRAD_OP` (weights gathered for the whole forward and backward) | `reshard_after_forward=False` |
| ZeRO-3 | `3` | `FULL_SHARD` | `reshard_after_forward=True` |
| Hybrid | ZeRO++ hpZ: `zero_hpz_partition_size` | `HYBRID_SHARD`: ZeRO-3 inside a node, DDP across nodes (`_HYBRID_SHARD_ZERO2`) | 2-D `DeviceMesh` (replicate × shard) |

Useful ZeRO-3 knobs: `overlap_comm`, `stage3_prefetch_bucket_size` and `stage3_param_persistence_threshold` (DeepSpeed);
`forward_prefetch`, `backward_prefetch` and `limit_all_gathers` (FSDP). Offload: `offload_optimizer` and `offload_param`
with `{"device": "cpu" | "nvme"}`.

## 6. Activations and overlap

- **Activations per layer** (Korthikanti et al.; 16-bit, no TP): none **sbh(34 + 5as/h)** · selective **34sbh** · full **2sbh**
  (+ one layer's working set). Multiply by L. They depend on the micro-batch and not on the ZeRO stage.
  Example: LLaMA-2 7B, s = 4096, b = 1: 34·4096·4096·32 ≈ 18.3 GB.
- **Overlap threshold:** comm = v·Ψ·b/BW and compute = 6·Ψ·T/(F·MFU), so communication hides when
  **tokens per GPU per step T ≥ v·b·F·MFU / (6·BW)**, with v = 2 (DDP, Z1, Z2) or 3 (Z3) and b = 2. **Ψ cancels.**
  A100 (312 TFLOP/s, 25 GB/s NIC, MFU 0.4): ≈ 3.3k / 5.0k tokens. H100 (989 TFLOP/s, 50 GB/s): ≈ 5.3k / 7.9k.
  Rail-optimized (8 NICs acting as one pipe): 8× fewer. A faster GPU needs *more* tokens.
- Accumulation adds tokens for DDP and ZeRO-1, not for ZeRO-3 (its traffic grows with G too).

## 7. Pitfalls

- `ZeroRedundancyOptimizer` = AR (2Ψ) + broadcast (Ψ) = **3Ψ**, and it partitions by whole parameters.
- **Clip by global norm:** each GPU sums the squares of its own shard → all-reduce one scalar → √ → every GPU scales by the same factor. A local norm under-clips.
- ZeRO does **not** shard activations. Use checkpointing, sequence/context parallelism or ZeRO-R.
- ZeRO-3 = 3 collectives per unit per micro-batch, so it is latency-bound without **prefetch**; wrap too coarsely and the gathered unit sets the peak.
- FSDP keeps the **root** unit gathered after forward (slightly under 3Ψ).
- FSDP shards **every unit** N ways; DeepSpeed 1/2 partition one flat buffer per parameter group; DeepSpeed 3 partitions each parameter. The memory and bytes are the same; the collective sizes and checkpoint layout differ.
- Checkpoints are N shards tied to N and the padding: consolidate (`zero_to_fp32.py`, FSDP full state dict) or reshard (`torch.distributed.checkpoint`).
- ZeRO-2/3 + pipeline parallelism: many micro-batches means per-micro-batch communication. Pair PP with ZeRO-1.

## 8. Beyond ZeRO-DP, one line each

- **ZeRO-R:** trims *residual* memory with partitioned activation checkpoints (split across tensor-parallel ranks, optionally on CPU), constant-size fused buffers, and defragmentation.
- **ZeRO-Offload:** keeps the optimizer states and gradients, and runs the Adam step, on the **CPU**; forward and backward stay on the GPU. This trains 13B parameters on a single V100.
- **ZeRO-Infinity:** ZeRO-3 plus offload of weights, gradients and optimizer states to **CPU and NVMe**, with bandwidth-centric partitioning and an overlap engine, for models too big for aggregate GPU memory.
- **ZeRO++** (cuts ZeRO-3's cross-node volume 3Ψ → 0.75Ψ, 4×):
  - **qwZ** quantizes weights to int8 for the forward all-gather (0.5Ψ).
  - **hpZ** keeps a secondary, node-local weight partition, so the backward all-gather stays on NVLink (0 cross-node).
  - **qgZ** quantizes gradients to int4 and uses a hierarchical all-to-all in place of the ring reduce-scatter (0.25Ψ).
- **vs TP / PP:** ZeRO splits the batch and shards the *states*. Tensor parallelism splits each matmul and exchanges activations every layer (inside a node). Pipeline parallelism splits the layers and pays a bubble.
