"""
zero_theory.py: the closed-form maths of ZeRO, in pure Python (no torch).

zero_sim.py *measures* what each of 32 virtual GPUs holds and sends. This file *predicts*
those numbers from the ZeRO paper's formulas (Rajbhandari et al. 2019, arXiv:1910.02054).
It then applies the same formulas to real models (GPT-2 XL, LLaMA-2 7B/13B/70B) on real
hardware (32 A100s or H100s). The notebook checks the simulator against sections 1-2.
Sections 3-7 answer "what would this cost at full scale?".

Notation (mostly the ZeRO paper's)
----------------------------------
    Ψ (psi)          number of model parameters
    N                data-parallel degree = number of GPUs
    b_w, b_g         bytes per weight / gradient element (bf16: 2 each)
    K                optimizer-state bytes per parameter. Mixed-precision Adam keeps an
                     fp32 master copy (4) + Adam m (4) + Adam v (4), so K = 12
    G                gradient-accumulation steps (micro-batches per optimizer step)
    s, b, h, a, L    sequence length, micro-batch size, hidden size, heads, layers
    g, M             GPUs per node, number of nodes (N = g * M)
    GB = 1e9 bytes   (the ZeRO paper and NVIDIA's datasheets use decimal GB). GiB = 2**30.

Contents
--------
    1. Model-state memory      model_state_bytes, model_state_breakdown,
                               paper_figure1, paper_trillion_example
    2. Communication volume    comm_volume_psi, comm_bytes_per_gpu, collective_calls,
                               grad_accum_comm_bytes, ring_latency_steps
    3. Activation memory       activation_bytes_per_layer, activation_bytes_model
    4. Real models, hardware   MODELS, HARDWARE, gpt2_param_count, llama_param_count
    5. Time (alpha-beta)       effective_bandwidth, step_time, overlap_threshold_tokens
    6. Hybrid sharding         hybrid_zero3 (HSDP), zeropp_hpz (ZeRO++ hpZ)
    7. Table and decision      real_hardware_rows, rows_to_text, decision
"""

import math

GB = 1e9
GiB = 2 ** 30

STAGES = (0, 1, 2, 3)
STAGE_NAMES = {0: "DDP", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}
PAPER_NAMES = {0: "Baseline", 1: "P_os", 2: "P_os+g", 3: "P_os+g+p"}   # the paper's labels


def _check_stage(stage):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES} (0 = DDP), got {stage!r}")


def _check_n(n):
    if n < 1 or int(n) != n:
        raise ValueError(f"the number of GPUs must be a positive integer, got {n!r}")


# =============================================================================================
# 1. Model-state memory
# =============================================================================================

def model_state_breakdown(stage, psi, n_gpus, bytes_w=2, bytes_g=2, k_opt=12) -> dict:
    """Bytes of weights, gradients and optimizer states that ONE GPU holds.

    Each ZeRO stage shards one more of the three model states across the N GPUs:

        stage    weights     grads       optimizer   per GPU                    bf16 + Adam
        0 DDP    b_w Ψ       b_g Ψ       K Ψ         (b_w + b_g + K) Ψ          16Ψ
        1        b_w Ψ       b_g Ψ       K Ψ / N     (b_w + b_g) Ψ + K Ψ / N    4Ψ + 12Ψ/N
        2        b_w Ψ       b_g Ψ / N   K Ψ / N     b_w Ψ + (b_g + K) Ψ / N    2Ψ + 14Ψ/N
        3        b_w Ψ / N   b_g Ψ / N   K Ψ / N     (b_w + b_g + K) Ψ / N      16Ψ/N

    So the whole rule is: stage >= 1 shards the optimizer, >= 2 the gradients, >= 3 the
    weights. The code below is written exactly that way.

    This is the moment at the end of backward, when every stage holds its largest set of
    model states. It counts model states only. Activations, temporary buffers (one
    gathered unit under ZeRO-3, one unit's full gradient under ZeRO-2) and allocator
    fragmentation come on top.
    """
    _check_stage(stage)
    _check_n(n_gpus)
    return {
        "weights": bytes_w * psi / (n_gpus if stage >= 3 else 1),
        "grads": bytes_g * psi / (n_gpus if stage >= 2 else 1),
        "optimizer": k_opt * psi / (n_gpus if stage >= 1 else 1),
    }


def model_state_bytes(stage, psi, n_gpus, bytes_w=2, bytes_g=2, k_opt=12) -> float:
    """Total model-state bytes on one GPU (see model_state_breakdown for the table).

    As N grows, ZeRO-1 falls towards a floor of (b_w + b_g)Ψ = 4Ψ and ZeRO-2 towards
    b_w Ψ = 2Ψ, because the unsharded parts never shrink. Only ZeRO-3 keeps falling as
    exactly 1/N: double the GPUs and each one holds half as much.
    """
    return sum(model_state_breakdown(stage, psi, n_gpus, bytes_w, bytes_g, k_opt).values())


def paper_figure1() -> dict:
    """The ZeRO paper's Figure 1: Ψ = 7.5B parameters, N_d = 64 GPUs, K = 12.

    The paper prints 120 GB (baseline), 31.4 GB (P_os), 16.6 GB (P_os+g) and 1.9 GB
    (P_os+g+p). The exact values are 120, 31.40625, 16.640625 and 1.875 GB:
        16 * 7.5            = 120
        4 * 7.5 + 12*7.5/64 = 30 + 1.40625
        2 * 7.5 + 14*7.5/64 = 15 + 1.640625
        16 * 7.5 / 64       = 1.875
    """
    psi, n, k = 7.5e9, 64, 12
    return {
        "psi": psi, "n_gpus": n, "k_opt": k,
        "gb": {s: model_state_bytes(s, psi, n, k_opt=k) / GB for s in STAGES},
        "paper_gb": {0: 120.0, 1: 31.4, 2: 16.6, 3: 1.9},
        "labels": dict(PAPER_NAMES),
        "trillion": paper_trillion_example(),
    }


def paper_trillion_example() -> dict:
    """The paper's headline: with all three stages, 1 trillion parameters fit on 1024 GPUs.

    Mixed-precision Adam needs 16Ψ = 16 TB of model states. Split 1024 ways, that is
    15.625 GB per GPU. The paper rounds this to "about 16 GB", which fits in a 32 GB
    V100 (the paper's GPU).
    """
    psi, n = 1e12, 1024
    return {"psi": psi, "n_gpus": n,
            "total_tb": model_state_bytes(0, psi, 1) / 1e12,
            "per_gpu_gb": model_state_bytes(3, psi, n) / GB}


# =============================================================================================
# 2. Communication volume
# =============================================================================================

COMM_VOLUME_PSI = {0: 2, 1: 2, 2: 2, 3: 3}
_RING_STEPS = {"all_reduce": 2, "reduce_scatter": 1, "all_gather": 1}   # x (N - 1)


def comm_volume_psi(stage) -> int:
    """Elements each GPU sends per training step, in units of Ψ (the paper's section 7).

    DDP      all-reduce the gradients = reduce-scatter (Ψ) + all-gather (Ψ)          2Ψ
    ZeRO-1   reduce-scatter the gradients (Ψ), update your 1/N, all-gather weights (Ψ)  2Ψ
    ZeRO-2   the same two collectives; the reduce-scatter just happens earlier       2Ψ
    ZeRO-3   all-gather weights for forward (Ψ), again for backward (Ψ, because they
             were freed in between), reduce-scatter gradients (Ψ). Nothing after the
             step, because the weights stay sharded                                  3Ψ

    The key insight: an all-reduce IS a reduce-scatter followed by an all-gather (that is
    how the bandwidth-optimal ring does it). ZeRO-1/2 therefore rearrange DDP's
    communication rather than add to it. Only ZeRO-3 pays extra (1.5x). These are
    large-N volumes; the exact ring cost has an extra factor of (N-1)/N
    (see comm_bytes_per_gpu).
    """
    _check_stage(stage)
    return COMM_VOLUME_PSI[stage]


def ring_factor(n) -> float:
    """(N-1)/N: the fraction of a buffer each GPU sends in a ring reduce-scatter/all-gather."""
    _check_n(n)
    return (n - 1) / n


def comm_bytes_per_gpu(stage, psi, n, bytes_per_elem=2) -> float:
    """Bytes each GPU sends per step under the ring model: volume * (N-1)/N * Ψ * bytes.

    Why (N-1)/N: a ring reduce-scatter of an S-byte buffer cuts it into N chunks. In each
    of N-1 steps, every GPU sends one chunk (S/N bytes) to its neighbour and adds the
    chunk it receives. Each GPU therefore sends (N-1)/N * S. An all-gather is the same
    pattern without the add. An all-reduce is both: 2(N-1)/N * S. Patarasuk & Yuan (2009)
    proved 2(N-1)/N * S is a lower bound for all-reduce, so the ring is bandwidth-optimal
    and the per-GPU cost barely depends on N.
    """
    return comm_volume_psi(stage) * ring_factor(n) * psi * bytes_per_elem


def collective_calls_by_op(stage, n_units, grad_accum=1) -> dict:
    """Collective calls per optimizer step, by type, when the model is cut into n_units.

    One call per unit per collective (FSDP-style per-unit communication):
        DDP      n_units all-reduces (once, after the last micro-batch)
        ZeRO-1   n_units reduce-scatters + n_units all-gathers
        ZeRO-2   G * n_units reduce-scatters (every micro-batch) + n_units all-gathers
        ZeRO-3   2G * n_units all-gathers (forward and backward) + G * n_units
                 reduce-scatters
    With 6 units and G = 1, that is 6 / 12 / 12 / 18 calls.
    """
    _check_stage(stage)
    u, G = n_units, grad_accum
    return {0: {"all_reduce": u, "reduce_scatter": 0, "all_gather": 0},
            1: {"all_reduce": 0, "reduce_scatter": u, "all_gather": u},
            2: {"all_reduce": 0, "reduce_scatter": G * u, "all_gather": u},
            3: {"all_reduce": 0, "reduce_scatter": G * u, "all_gather": 2 * G * u}}[stage]


def collective_calls(stage, n_units, grad_accum=1) -> int:
    """Total collective calls per optimizer step (see collective_calls_by_op)."""
    return sum(collective_calls_by_op(stage, n_units, grad_accum).values())


def ring_latency_steps(stage, n, n_units=1, grad_accum=1) -> int:
    """Ring steps per optimizer step: each reduce-scatter or all-gather takes N-1 hops,
    and each all-reduce 2(N-1). Each hop costs one latency alpha in the alpha-beta model."""
    _check_n(n)
    calls = collective_calls_by_op(stage, n_units, grad_accum)
    return sum(c * _RING_STEPS[op] * (n - 1) for op, c in calls.items())


def grad_accum_volume_psi(stage, G) -> int:
    """Communication per optimizer step, in units of Ψ, with G micro-batches (see below)."""
    _check_stage(stage)
    if G < 1:
        raise ValueError(f"G must be >= 1, got {G!r}")
    return {0: 2, 1: 2, 2: G + 1, 3: 3 * G}[stage]


def grad_accum_comm_bytes(stage, psi, n, G, bytes_per_elem=2) -> float:
    """Bytes each GPU sends per OPTIMIZER step when it accumulates gradients over G
    micro-batches.

        DDP      2Ψ       Gradients add up in the resident full 2Ψ buffer (DDP's
                          no_sync()). One all-reduce after the last micro-batch.
        ZeRO-1   2Ψ       Also keeps the full gradient buffer, so it too accumulates
                          locally. One reduce-scatter, then one all-gather of weights.
        ZeRO-2   (G+1)Ψ   ZeRO-2 saves memory by NOT keeping a full gradient buffer: a
                          unit's gradient is reduce-scattered and freed as soon as it
                          exists. The next micro-batch makes a fresh full gradient with
                          nowhere local to add it to, so it is reduce-scattered again into
                          the owner's shard. That gives G reduce-scatters plus the one
                          all-gather after the step. (FSDP's SHARD_GRAD_OP under no_sync()
                          avoids this by keeping unsharded gradients, which gives back
                          ZeRO-2's memory saving.)
        ZeRO-3   3GΨ      The weights are not resident either. Every micro-batch
                          re-gathers them for forward and for backward, and reduce-scatters
                          its gradients.

    The consequence: for DDP and ZeRO-1, accumulation spreads the communication over
    G micro-batches of compute. For ZeRO-3 it does not: communication per token is the
    same for any G, and only a bigger micro-batch (more tokens per gather) hides it.
    """
    return grad_accum_volume_psi(stage, G) * ring_factor(n) * psi * bytes_per_elem


# =============================================================================================
# 3. Activation memory (Korthikanti et al. 2022, arXiv:2205.05198)
# =============================================================================================

RECOMPUTE_MODES = ("none", "selective", "full")


def activation_bytes_per_layer(seq, micro_bsz, hidden, n_heads, recompute="none") -> float:
    """Activation bytes one transformer layer keeps for backward (Korthikanti et al. 2022,
    section 4, with no tensor or sequence parallelism and 16-bit activations).

        none       s*b*h*(34 + 5*a*s/h)   everything autograd would save
        selective  34*s*b*h               drop the attention-score tensors (the 5*a*s^2*b
                                          term: QK^T, softmax, dropout mask) and recompute
                                          them in backward. FlashAttention never stores
                                          them either, so modern training gets this for free
        full       2*s*b*h                keep only the layer's input and re-run the whole
                                          layer in backward

    Where the 34 comes from (per s*b*h, 2 bytes per value, 1 byte per dropout mask):
    attention 11 + 5*a*s/h, MLP 19, two layernorms 4.

    Assumptions: a GPT-style block (a 4h GELU MLP, dropout). For LLaMA's SwiGLU MLP
    (roughly 8h/3 wide with three matrices, no dropout), this is an approximation, not
    an exact count. It also assumes full multi-head attention, so it slightly
    overestimates the K/V activations of grouped-query attention (LLaMA-2 70B).
    ZeRO shards none of this. Activations are per-GPU and depend only on the
    micro-batch, never on the ZeRO stage.
    """
    s, b, h, a = seq, micro_bsz, hidden, n_heads
    if recompute == "none":
        return s * b * h * (34 + 5 * a * s / h)
    if recompute == "selective":
        return 34 * s * b * h
    if recompute == "full":
        return 2 * s * b * h
    raise ValueError(f"recompute must be one of {RECOMPUTE_MODES}, got {recompute!r}")


def activation_bytes_model(model, micro_bsz=1, seq=None, recompute="none") -> float:
    """Activation bytes for the whole model: L layers x activation_bytes_per_layer.

    With "full" recomputation, add one layer's un-recomputed activations. During the
    backward of layer i, that layer's forward is re-run with autograd on, so for a moment
    it holds everything. This is the working set on top of the 2sbh checkpoints of all L
    layers. Embedding and logits activations are ignored; for LLaMA-2 at s = 4096 the
    fp32 logits are about 0.5 GB per sequence.
    """
    m = _model(model)
    s = seq or m["seq_len"]
    args = (s, micro_bsz, m["hidden"], m["n_heads"])
    total = m["n_layers"] * activation_bytes_per_layer(*args, recompute=recompute)
    if recompute == "full":
        total += activation_bytes_per_layer(*args, recompute="none")
    return total


# =============================================================================================
# 4. Real models and hardware
# =============================================================================================

def gpt2_param_count(n_layers, hidden, vocab=50257, n_ctx=1024) -> int:
    """GPT-2 parameters: token + position embeddings, L blocks of 12h^2 + 13h (QKV, proj,
    4h MLP, two layernorms, with biases), a final layernorm, and an output head tied to
    the token embedding."""
    h = hidden
    return vocab * h + n_ctx * h + n_layers * (12 * h * h + 13 * h) + 2 * h


def llama_param_count(n_layers, hidden, ffn, n_heads, n_kv_heads, vocab=32000) -> int:
    """LLaMA parameters: untied input embedding and output head, L blocks of attention
    (Q and O are h x h; K and V are h x h*kv/heads under grouped-query attention),
    SwiGLU MLP (three h x ffn matrices), two RMSNorms, and a final RMSNorm. No biases."""
    h = hidden
    kv = h * n_kv_heads // n_heads
    per_layer = 2 * h * h + 2 * h * kv + 3 * h * ffn + 2 * h
    return 2 * vocab * h + n_layers * per_layer + h


MODELS = {
    "GPT-2 XL": dict(
        n_params=1_557_611_200, n_layers=48, hidden=1600, n_heads=25, n_kv_heads=25,
        ffn=6400, vocab=50257, seq_len=1024,
        source="OpenAI GPT-2 '1558M' release (Radford et al. 2019); count from the "
               "architecture (48 x 1600, 25 heads, ctx 1024, tied head)"),
    "LLaMA-2 7B": dict(
        n_params=6_738_415_616, n_layers=32, hidden=4096, n_heads=32, n_kv_heads=32,
        ffn=11008, vocab=32000, seq_len=4096,
        source="Touvron et al. 2023, arXiv:2307.09288 (Table 1: 4k context); "
               "count from the released config"),
    "LLaMA-2 13B": dict(
        n_params=13_015_864_320, n_layers=40, hidden=5120, n_heads=40, n_kv_heads=40,
        ffn=13824, vocab=32000, seq_len=4096,
        source="Touvron et al. 2023, arXiv:2307.09288; count from the released config"),
    "LLaMA-2 70B": dict(
        n_params=68_976_648_192, n_layers=80, hidden=8192, n_heads=64, n_kv_heads=8,
        ffn=28672, vocab=32000, seq_len=4096,
        source="Touvron et al. 2023, arXiv:2307.09288 (GQA for 34B/70B); count from the "
               "released config",
        note="Grouped-query attention (8 KV heads). The activation formula assumes full "
             "multi-head attention, so it slightly overestimates K/V activations."),
}

HARDWARE = {
    "A100-80GB": dict(
        name="NVIDIA A100 80GB SXM (DGX A100 node)",
        peak_bf16_flops=312e12,      # dense; 624 TFLOPS is the 2:4-sparsity figure
        hbm_bytes=80e9,
        hbm_bw=2.039e12,
        nvlink_bw=300e9,             # per direction: datasheet 600 GB/s is bidirectional
        inter_node_bw=25e9,          # one 200 Gb/s HDR InfiniBand NIC per GPU
        gpus_per_node=8,
        source="NVIDIA A100 datasheet (BF16 312 TFLOPS dense, 80GB HBM2e, 2,039 GB/s, "
               "NVLink 600 GB/s bidirectional); DGX A100 datasheet (8x single-port "
               "ConnectX-6 200 Gb/s HDR InfiniBand)"),
    "H100-80GB": dict(
        name="NVIDIA H100 80GB SXM (DGX H100 node)",
        peak_bf16_flops=989e12,      # dense; datasheet's 1,979 TFLOPS is with sparsity
        hbm_bytes=80e9,
        hbm_bw=3.35e12,
        nvlink_bw=450e9,             # per direction: datasheet 900 GB/s is bidirectional
        inter_node_bw=50e9,          # one 400 Gb/s NDR InfiniBand NIC per GPU
        gpus_per_node=8,
        source="NVIDIA H100 datasheet (BF16 1,979 TFLOPS with sparsity, so ~989 dense; "
               "80GB, 3.35 TB/s, NVLink 900 GB/s bidirectional); DGX H100 datasheet "
               "(8x single-port ConnectX-7 400 Gb/s)"),
}
# HBM: nvidia-smi reports about 80 GiB (~85.9e9 bytes) on these parts. We budget 80e9
# bytes, which leaves headroom for the CUDA context, NCCL buffers and fragmentation.


def _model(model) -> dict:
    return MODELS[model] if isinstance(model, str) else model


def _hw(hw) -> dict:
    return HARDWARE[hw] if isinstance(hw, str) else hw


# =============================================================================================
# 5. Time: compute vs communication (the alpha-beta model)
# =============================================================================================

NETWORKS = ("per_nic", "rail")


def effective_bandwidth(hw, n, network="per_nic") -> float:
    """The bytes/s each GPU's collective traffic moves at: the beta in the alpha-beta model.

    If N fits in one node (N <= 8), everything goes over NVLink (per direction). Beyond
    one node, the ring must cross InfiniBand, and there are two ways to model it:

      "per_nic" (default, conservative): every byte a GPU sends goes through its own
                  NIC (25 GB/s A100 HDR, 50 GB/s H100 NDR). This is the plan's model. It
                  is exact when every ring hop crosses a node boundary, and an upper
                  bound on time otherwise.
      "rail":     NCCL on a DGX-style cluster runs g rings in parallel, one per NIC, and
                  only one hop per node in each ring leaves the node. The inter-node
                  links then act like one pipe of g x NIC bandwidth (200 GB/s A100,
                  400 GB/s H100), capped by NVLink. Well-tuned multi-node clusters
                  measure bus bandwidth close to this, so it is the optimistic but
                  realistic case.
    """
    hw = _hw(hw)
    _check_n(n)
    if n <= hw["gpus_per_node"]:
        return hw["nvlink_bw"]
    if network == "per_nic":
        return hw["inter_node_bw"]
    if network == "rail":
        return min(hw["nvlink_bw"], hw["gpus_per_node"] * hw["inter_node_bw"])
    raise ValueError(f"network must be one of {NETWORKS}, got {network!r}")


def flops_per_token_factor(recompute=False) -> int:
    """Training FLOPs per parameter per token: 6 (2 forward + 4 backward), or 8 with full
    recomputation (one extra forward). Selective recomputation re-runs only the
    parameter-free attention ops, which 6Ψ does not count, so it stays at 6."""
    return 8 if recompute in (True, "full") else 6


def step_time(stage, psi, n, tokens_per_gpu, hw, mfu=0.40, recompute=False, alpha=10e-6,
              n_units=None, grad_accum=1, bytes_per_elem=2, network="per_nic") -> dict:
    """Modelled time of one optimizer step on one GPU: compute vs communication.

    Compute:  c * Ψ * T / (peak * MFU), with c = 6 (or 8 with full recomputation) and
              T = tokens this GPU processes per optimizer step. 6Ψ FLOPs per token counts
              the weight matmuls only (the 2*s*h attention term per layer is ignored).
              MFU is the fraction of peak actually achieved; 0.4 is typical.
    Communication, the alpha-beta model:  t = alpha * (ring steps) + bytes / beta
              alpha  latency per ring hop (about 10 us inter-node)
              beta   bandwidth, from effective_bandwidth(hw, n, network)
              bytes  grad_accum_comm_bytes(stage, psi, n, G) (comm_bytes_per_gpu when G = 1)
              ring steps: N-1 per reduce-scatter or all-gather, 2(N-1) per all-reduce,
              one set per unit (n_units=None means 1 unit, the smallest latency term)

    comm_over_compute < 1 means communication can hide behind compute, if the
    implementation overlaps them (bucketing, prefetch). > 1 means the GPU waits on the
    network. "step_s_serial" assumes no overlap; "step_s_overlapped" assumes perfect
    overlap.
    """
    hw = _hw(hw)
    compute_s = (flops_per_token_factor(recompute) * psi * tokens_per_gpu
                 / (hw["peak_bf16_flops"] * mfu))
    nbytes = grad_accum_comm_bytes(stage, psi, n, grad_accum, bytes_per_elem)
    bw = effective_bandwidth(hw, n, network)
    latency_s = alpha * ring_latency_steps(stage, n, n_units or 1, grad_accum)
    bandwidth_s = nbytes / bw
    comm_s = latency_s + bandwidth_s
    return dict(compute_s=compute_s, comm_s=comm_s,
                comm_over_compute=comm_s / compute_s if compute_s else math.inf,
                latency_s=latency_s, bandwidth_s=bandwidth_s, comm_bytes=nbytes, bw=bw,
                step_s_serial=compute_s + comm_s,
                step_s_overlapped=max(compute_s, comm_s))


def overlap_threshold_tokens(stage, hw, mfu=0.4, bytes_per_elem=2, n=None, recompute=False,
                             network="per_nic") -> float:
    """Tokens per GPU per step at which communication time equals compute time
    (bandwidth term only). More tokens than this and communication can hide.

    Set the two times equal:
        comm    = v * Ψ * bytes / BW                  v = 2 (DDP, ZeRO-1/2) or 3 (ZeRO-3)
        compute = 6 * Ψ * T / (peak * MFU)
        =>  T*  = v * bytes * peak * MFU / (6 * BW)

    Ψ cancels. Every parameter is sent a fixed number of times per step and costs a
    fixed 6 FLOPs per token, so both sides grow linearly with model size. A bigger model
    does not hide communication any better; only more tokens per GPU does. And a faster
    GPU (bigger peak) needs MORE tokens, because compute shrinks while the network
    stays the same.

    n=None uses the large-N limit (no (N-1)/N factor) and the inter-node bandwidth.
    Passing n applies (N-1)/N and picks NVLink when N fits in one node.

    Caveat: this assumes communication can overlap the whole step. Gradient sync can
    only start once backward produces gradients (about 2/3 of the compute), so in
    practice you need somewhat more tokens.
    """
    hw = _hw(hw)
    if n is None:
        bw, rho = effective_bandwidth(hw, hw["gpus_per_node"] + 1, network), 1.0
    else:
        bw, rho = effective_bandwidth(hw, n, network), ring_factor(n)
    return (comm_volume_psi(stage) * rho * bytes_per_elem * hw["peak_bf16_flops"] * mfu
            / (flops_per_token_factor(recompute) * bw))


# =============================================================================================
# 6. Hybrid sharding: shard inside a node, replicate across nodes
# =============================================================================================

def hybrid_zero3(psi, n, gpus_per_node=8, hw="A100-80GB", tokens_per_gpu=None, mfu=0.4,
                 recompute=False, alpha=10e-6, n_units=None, grad_accum=1, bytes_w=2,
                 bytes_g=2, k_opt=12, bytes_per_elem=2, network="per_nic") -> dict:
    """HSDP (FSDP's HYBRID_SHARD): ZeRO-3 inside each g-GPU node, plain data parallelism
    across the M = N/g nodes.

    Memory:  every node holds one full copy, split g ways: (b_w + b_g + K) Ψ / g = 16Ψ/g.
    Comm:    inside the node, the ZeRO-3 pattern over NVLink:
                 3 * (g-1)/g * Ψ * bytes   (all-gather, all-gather, reduce-scatter)
             across nodes, each GPU all-reduces only its own Ψ/g shard with the GPUs
             that hold the same shard in the other nodes, over its own NIC:
                 2 * (M-1)/M * (Ψ/g) * bytes
    The trade: g times fewer bytes cross the slow network than flat ZeRO-3, in exchange
    for memory that only divides by g, not N. Once a model's 16Ψ/g does not fit, HSDP
    is out (see zeropp_hpz for a hybrid that keeps the 1/N memory).

    With accumulation, the in-node collectives run every micro-batch (as in ZeRO-3). The
    cross-node shard all-reduce runs once per step, because the shard stays resident.
    The two phases are timed back to back (no overlap between them).
    """
    hw = _hw(hw)
    g = gpus_per_node
    if n % g:
        raise ValueError(f"n ({n}) must be a multiple of gpus_per_node ({g})")
    m, u, G = n // g, n_units or 1, grad_accum
    intra_bytes = 3 * G * (g - 1) / g * psi * bytes_per_elem
    inter_bytes = 2 * (m - 1) / m * (psi / g) * bytes_per_elem
    intra_s = intra_bytes / hw["nvlink_bw"] + alpha * 3 * G * u * (g - 1)
    inter_s = inter_bytes / hw["inter_node_bw"] + alpha * 2 * u * (m - 1)
    flat = step_time(3, psi, n, tokens_per_gpu or 1, hw, mfu=mfu, recompute=recompute,
                     alpha=alpha, n_units=n_units, grad_accum=grad_accum,
                     bytes_per_elem=bytes_per_elem, network=network)
    out = dict(memory_bytes=(bytes_w + bytes_g + k_opt) * psi / g,
               intra_node_bytes=intra_bytes, inter_node_bytes=inter_bytes,
               intra_s=intra_s, inter_s=inter_s, comm_s=intra_s + inter_s,
               flat_zero3_memory_bytes=model_state_bytes(3, psi, n, bytes_w, bytes_g, k_opt),
               flat_zero3_comm_s=flat["comm_s"])
    if tokens_per_gpu:
        out["compute_s"] = flat["compute_s"]
        out["comm_over_compute"] = out["comm_s"] / flat["compute_s"]
    return out


def zeropp_hpz(psi, n, gpus_per_node=8, hw="A100-80GB", tokens_per_gpu=None, mfu=0.4,
               recompute=False, alpha=10e-6, n_units=None, grad_accum=1, bytes_w=2,
               bytes_g=2, k_opt=12, bytes_per_elem=2, network="per_nic") -> dict:
    """ZeRO++'s hpZ (hierarchical partitioning; Wang et al. 2023, arXiv:2306.10209).

    All model states stay sharded over all N GPUs, as in ZeRO-3. On top of that, each
    node keeps a SECONDARY copy of the 16-bit weights, split over its g GPUs. The
    backward all-gather reads that copy over NVLink instead of crossing nodes.
    Memory:  16Ψ/N + b_w Ψ / g
    Comm:    across nodes, forward all-gather + gradient reduce-scatter:
                 2 * (N-1)/N * Ψ * bytes      (ZeRO-3 sends 3x this over the network)
             inside the node, the backward all-gather:  (g-1)/g * Ψ * bytes
    Check: the paper's 100B model on 1024 GPUs with 16-GPU nodes needs 114x less memory
    than DP. The formula gives 16Ψ / (16Ψ/1024 + 2Ψ/16) = 113.8.

    Counted conservatively: every micro-batch re-does the forward cross-node gather.
    """
    hw = _hw(hw)
    g, u, G = gpus_per_node, n_units or 1, grad_accum
    inter_bytes = 2 * G * ring_factor(n) * psi * bytes_per_elem
    intra_bytes = G * (g - 1) / g * psi * bytes_per_elem
    inter_s = (inter_bytes / effective_bandwidth(hw, n, network)
               + alpha * 2 * G * u * (n - 1))
    intra_s = intra_bytes / hw["nvlink_bw"] + alpha * G * u * (g - 1)
    flat = step_time(3, psi, n, tokens_per_gpu or 1, hw, mfu=mfu, recompute=recompute,
                     alpha=alpha, n_units=n_units, grad_accum=grad_accum,
                     bytes_per_elem=bytes_per_elem, network=network)
    out = dict(memory_bytes=model_state_bytes(3, psi, n, bytes_w, bytes_g, k_opt)
               + bytes_w * psi / g,
               intra_node_bytes=intra_bytes, inter_node_bytes=inter_bytes,
               intra_s=intra_s, inter_s=inter_s, comm_s=intra_s + inter_s,
               flat_zero3_comm_s=flat["comm_s"])
    if tokens_per_gpu:
        out["compute_s"] = flat["compute_s"]
        out["comm_over_compute"] = out["comm_s"] / flat["compute_s"]
    return out


# =============================================================================================
# 7. The real-hardware table and the decision guide
# =============================================================================================

def _fits_with(states, model, micro_bsz, seq, budget) -> str:
    """The least recomputation that makes states + activations fit, or 'no'."""
    for mode in RECOMPUTE_MODES:
        if states + activation_bytes_model(model, micro_bsz, seq, mode) <= budget:
            return mode
    return "no"


def real_hardware_rows(n_gpus=32, hw="A100-80GB", micro_bsz=1, tokens_per_gpu=None,
                       recompute="selective", mfu=0.4, alpha=10e-6, network="per_nic",
                       models=None, include_hybrid=True) -> list:
    """One row per model x stage: memory per GPU and compute vs communication time.

    Defaults (stated so the table can be reproduced):
      * sequence length = the model's context (1024 for GPT-2 XL, 4096 for LLaMA-2);
        micro-batch 1; tokens_per_gpu = micro_bsz * seq, i.e. one micro-batch per GPU
        per step (G = 1). Passing a larger tokens_per_gpu means G = ceil(tokens /
        (micro_bsz * seq)) accumulation steps.
      * bf16 weights and gradients plus fp32 Adam (16 bytes per parameter).
      * activations from Korthikanti et al. with `recompute` ("selective" is what
        FlashAttention gives you for free); act_gb_none is the same without any
        recomputation.
      * units = L + 2 (embedding, each layer, head) for the latency term; MFU 0.4;
        alpha = 10 us; network model as in effective_bandwidth.
      * fits_80GB compares model states + activations with the HBM budget. Temporary
        buffers and fragmentation are not counted, so "fits" by a few GB is not a
        guarantee.
      * fits_with = the least recomputation that makes it fit ("none" < "selective" <
        "full"), or "no".
    Bonus rows per model (include_hybrid): HSDP (8-GPU shards) and ZeRO++ hpZ.
    """
    hwd = _hw(hw)
    budget = hwd["hbm_bytes"]
    rows = []
    for name in models or MODELS:
        m = _model(name)
        psi, L, seq = m["n_params"], m["n_layers"], m["seq_len"]
        per_micro = micro_bsz * seq
        tokens = tokens_per_gpu or per_micro
        G = max(1, math.ceil(tokens / per_micro))
        act_none = activation_bytes_model(m, micro_bsz, seq, "none")
        act = activation_bytes_model(m, micro_bsz, seq, recompute)
        common = dict(model=name if isinstance(name, str) else m.get("name", "custom"),
                      hw=hw if isinstance(hw, str) else hwd["name"], n_gpus=n_gpus,
                      n_params=psi, seq=seq, micro_bsz=micro_bsz, grad_accum=G,
                      tokens_per_gpu=tokens, recompute=recompute,
                      act_gb_none=act_none / GB, act_gb=act / GB)

        def row(variant, stage, states, compute_s, comm_s):
            return dict(common, variant=variant, stage=stage, states_gb=states / GB,
                        total_gb=(states + act) / GB, fits_80GB=states + act <= budget,
                        fits_with=_fits_with(states, m, micro_bsz, seq, budget),
                        compute_s=compute_s, comm_s=comm_s,
                        comm_over_compute=comm_s / compute_s)

        for stage in STAGES:
            t = step_time(stage, psi, n_gpus, tokens, hwd, mfu=mfu, recompute=recompute,
                          alpha=alpha, n_units=L + 2, grad_accum=G, network=network)
            rows.append(row(STAGE_NAMES[stage], stage, model_state_bytes(stage, psi, n_gpus),
                            t["compute_s"], t["comm_s"]))
        if include_hybrid and n_gpus > hwd["gpus_per_node"]:
            kw = dict(gpus_per_node=hwd["gpus_per_node"], hw=hwd, tokens_per_gpu=tokens,
                      mfu=mfu, recompute=recompute, alpha=alpha, n_units=L + 2,
                      grad_accum=G, network=network)
            for variant, fn in (("HSDP", hybrid_zero3), ("ZeRO++ hpZ", zeropp_hpz)):
                h = fn(psi, n_gpus, **kw)
                rows.append(row(variant, 3, h["memory_bytes"], h["compute_s"], h["comm_s"]))
    return rows


def rows_to_text(rows, columns=("model", "variant", "states_gb", "act_gb", "total_gb",
                                "fits_80GB", "fits_with", "compute_s", "comm_s",
                                "comm_over_compute")) -> str:
    """A plain-ASCII table of real_hardware_rows output (floats to 3 significant digits)."""
    def fmt(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"
        return str(v)
    cells = [list(columns)] + [[fmt(r[c]) for c in columns] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    lines = ["  ".join(c.rjust(w) for c, w in zip(row, widths)) for row in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


DECISIONS = ("DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3", "ZeRO-3 + checkpointing/offload/TP/PP")


def decision(psi, n, gpu_mem_bytes, act_bytes, bytes_w=2, bytes_g=2, k_opt=12) -> str:
    """The plan's decision flowchart: pick the LEAST sharded stage that fits, because each
    step down the list costs something (ZeRO-2 communicates every micro-batch under
    accumulation; ZeRO-3 sends 1.5x the bytes).

        16Ψ + act fits on one GPU?        -> DDP (least communication)
        4Ψ + 12Ψ/N + act fits?            -> ZeRO-1 (same communication as DDP)
        2Ψ + 14Ψ/N + act fits?            -> ZeRO-2
        16Ψ/N + act fits?                 -> ZeRO-3 / FSDP FULL_SHARD
        otherwise                         -> ZeRO-3 plus activation checkpointing, then
                                             offload (ZeRO-Offload/Infinity), then tensor
                                             or pipeline parallelism

    act_bytes is whatever activations you plan to keep (already reduced by checkpointing
    if you use it). ZeRO never shards them.
    """
    for stage in STAGES:
        if model_state_bytes(stage, psi, n, bytes_w, bytes_g, k_opt) + act_bytes <= gpu_mem_bytes:
            return DECISIONS[stage]
    return DECISIONS[-1]


if __name__ == "__main__":
    f1 = paper_figure1()
    print("ZeRO paper Figure 1 (7.5B, N=64), GB:",
          {PAPER_NAMES[s]: round(v, 3) for s, v in f1["gb"].items()})
    print("1T params on 1024 GPUs, ZeRO-3: %.3f GB/GPU" % f1["trillion"]["per_gpu_gb"])
    for hw_name in HARDWARE:
        for net in NETWORKS:
            print(f"\noverlap threshold, {hw_name}, network={net} (tokens/GPU/step): "
                  + ", ".join(f"{STAGE_NAMES[s]} {overlap_threshold_tokens(s, hw_name, network=net):,.0f}"
                              for s in STAGES))
    for hw_name in HARDWARE:
        print(f"\n32x {hw_name}, micro-batch 1, seq = model context, selective recompute, "
              "per-NIC network")
        print(rows_to_text(real_hardware_rows(32, hw_name)))
