"""The three gates. Nothing in this project is allowed to train until these pass.

A reversible backward pass is easy to write and easy to get subtly wrong: the gradients
still flow, the loss still falls, and the model still trains - just to the wrong place. The
only defence is to check the engine against ordinary autograd on the *same recurrence*, in
fp64, where floating point cannot hide a real error.

Gate 1  the forward pass is what we claim it is
Gate 2  the reconstructed gradients equal autograd's, to machine precision
Gate 3  peak memory stops growing with depth
"""
import pytest
import torch

from revlm.model import GPT, GPTConfig, chunked_cross_entropy
from revlm import reversible as R

TINY = GPTConfig(vocab_size=97, block_size=16, n_layer=6, n_head=2, n_embd=16)
# The implicit-Euler gate needs the *real* width. Lip(G) turns out to scale with d, so at
# the toy width above the fixed point contracts happily (Lip ~ 0.04) and at d=384 it does
# not (Lip ~ 80). A narrow test would therefore certify a method that cannot train.
WIDE = GPTConfig(vocab_size=97, block_size=16, n_layer=4, n_head=6, n_embd=384)


def build(mode, cfg=TINY, **kw):
    """A fp64 model with bf16 autocast disabled - otherwise the test's precision is bf16's."""
    torch.manual_seed(0)
    m = GPT(cfg, mode=mode, **kw).double()
    for b in m.blocks:
        b.use_amp = False
        for sub in (getattr(b, "f", None), getattr(b, "g", None)):
            if sub is not None:
                sub.use_amp = False
    return m


def batch(cfg=TINY, n=2):
    torch.manual_seed(1)
    return (torch.randint(0, cfg.vocab_size, (n, cfg.block_size)),
            torch.randint(0, cfg.vocab_size, (n, cfg.block_size)))


# reference recurrences, written out plainly so autograd can differentiate them normally
def ref_euler(m, h0):
    h, v = h0, torch.zeros_like(h0)
    for b in m.blocks:
        v = v + b(h)
        h = h + v
    return h


def ref_midpoint(m, h0):
    hp, hc = h0, h0 + m.blocks[0](h0)
    for l in range(1, len(m.blocks)):
        hp, hc = hc, hp + 2.0 * m.blocks[l](hc)
    return hc


def ref_coupling(m, h0):
    x1, x2 = h0.chunk(2, dim=-1)
    for b in m.blocks:
        y1 = x1 + b.f(x2)
        x1, x2 = y1, x2 + b.g(y1)
    return torch.cat([x1, x2], dim=-1)


REFS = {"euler": ref_euler, "midpoint": ref_midpoint, "coupling": ref_coupling}


def _grads(m, idx, tgt, forward=None):
    m.zero_grad(set_to_none=True)
    if forward is None:
        loss = m(idx, tgt)
    else:
        loss = chunked_cross_entropy(m.ln_f(forward(m, m.embed(idx))), tgt, m.head.weight, 1)
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def _worst_rel(a, b):
    return max((a[k] - b[k]).abs().max().item() / max(a[k].abs().max().item(), 1e-14) for k in a)


# --------------------------------------------------------------------------------------
# gate 1
# --------------------------------------------------------------------------------------
def test_gate1_implicit_euler_forward_is_the_baseline_bit_for_bit():
    """A pre-LN transformer *is* an Euler step, so these two must agree exactly.

    Not `allclose` - `== 0.0`. This is what makes the memory comparison a controlled one:
    the baseline and the implicit-Euler run are the same function, so any difference in
    their loss curves is attributable to the backward pass alone.
    """
    idx, _ = batch()
    a = R.run_stack(build("store"), build("store").embed(idx))
    b = R.run_stack(build("euler_implicit", euler_iters=4),
                    build("euler_implicit", euler_iters=4).embed(idx))
    assert (a - b).abs().max().item() == 0.0


def test_gate1_coupling_module_matches_the_engine_forward():
    idx, _ = batch()
    m = build("coupling")
    h0 = m.embed(idx)
    stacked = h0
    for blk in m.blocks:
        stacked = blk(stacked)
    assert torch.allclose(stacked, R.run_stack(m, h0), atol=1e-12)


# --------------------------------------------------------------------------------------
# gate 2 - the one that matters
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["euler", "midpoint", "coupling"])
def test_gate2_gradients_match_autograd_in_fp64(mode):
    """Reconstructed gradients vs stored gradients on the identical recurrence."""
    idx, tgt = batch()
    m = build(mode)
    lref, gref = _grads(m, idx, tgt, forward=REFS[mode])
    leng, geng = _grads(m, idx, tgt)
    assert abs(lref - leng) < 1e-12, f"{mode}: forward disagrees, {lref} vs {leng}"
    assert _worst_rel(gref, geng) < 1e-9, f"{mode}: gradient error {_worst_rel(gref, geng):.3e}"


@pytest.mark.parametrize("mode", ["euler", "midpoint", "coupling"])
def test_gate2_every_parameter_actually_receives_a_gradient(mode):
    """A reversible engine that silently drops a parameter still trains - just worse."""
    idx, tgt = batch()
    m = build(mode)
    m.zero_grad(set_to_none=True)
    m(idx, tgt).backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"{mode}: no gradient for {missing}"
    dead = [n for n, p in m.named_parameters()
            if p.grad is not None and p.grad.abs().max().item() == 0.0]
    assert not dead, f"{mode}: all-zero gradient for {dead}"


@pytest.mark.parametrize("mode", ["euler", "midpoint", "coupling"])
def test_gate2_reconstruction_returns_to_the_true_input(mode):
    """The backward walk ends on h_0, and the true h_0 was kept. They must agree."""
    idx, tgt = batch()
    m = build(mode)
    m.track_recon = True
    m.zero_grad(set_to_none=True)
    m(idx, tgt).backward()
    assert m.diag["recon_h0"] < 1e-10, f"{mode}: drifted {m.diag['recon_h0']:.3e}"


@pytest.mark.parametrize("mode", ["euler", "midpoint", "coupling"])
def test_reconstruction_drift_is_reported_per_layer(mode):
    """Every exactly-reversible mode must rebuild every layer, not just the ones with a
    bespoke branch. `coupling` once fell through to the implicit-Euler inverse - the wrong
    inverse entirely - and reported a drift of 1.5e+02 while its gradients were exact."""
    m = build(mode)
    idx, _ = batch()
    drift = R.reconstruction_drift(m, m.embed(idx))
    assert len(drift) == TINY.n_layer + 1
    assert max(drift) < 1e-10, f"{mode} drifted {max(drift):.3e}"


def test_forward_states_matches_the_engine_for_every_mode():
    """The per-layer truth used by the drift check must be the stack's own recurrence."""
    idx, _ = batch()
    for mode in ("euler", "midpoint", "coupling"):
        m = build(mode)
        h0 = m.embed(idx)
        states = R.forward_states(m, h0)
        assert len(states) == TINY.n_layer + 1
        assert torch.allclose(states[-1], R.run_stack(m, h0), atol=1e-12), mode


# --------------------------------------------------------------------------------------
# the negative result, asserted so it cannot quietly start "working"
# --------------------------------------------------------------------------------------
def test_implicit_euler_converges_only_when_it_is_a_contraction():
    """The obvious reading of "reversible Euler" works - but only conditionally.

    Inverting h_l = h_{l+1} - G_l(h_l) by fixed point needs Lip(G_l) = gamma * Lip(F_l) < 1.
    This test asserts the *condition*, not a verdict on one setting: above the line the
    iteration does not improve with more steps, below it the reconstruction goes to zero.

    That conditionality is the argument for the explicit two-state integrators, which need
    no such assumption. gamma is a learnable parameter, so a run that starts inside the
    convergent region can leave it, and nothing in the optimiser notices.
    """
    idx, _ = batch(WIDE)

    loose = GPTConfig(**{**WIDE.dict(), "gamma_init": 0.25})
    m = build("euler_implicit", cfg=loose, euler_iters=4)
    torch.manual_seed(7)
    rep = R.euler_implicit_report(m, m.embed(idx), iters=(1, 4, 16))
    assert max(rep["lipschitz"]) > 1.0, f"expected a non-contraction, got {rep['lipschitz']}"
    errs = [e["err"] for e in rep["sweep"]]
    assert min(errs) > errs[0] / 2, f"iterating should not be rescuing it, got {errs}"

    tight = GPTConfig(**{**WIDE.dict(), "gamma_init": 0.02})
    m2 = build("euler_implicit", cfg=tight, euler_iters=4)
    torch.manual_seed(7)
    rep2 = R.euler_implicit_report(m2, m2.embed(idx), iters=(1, 4, 16))
    assert max(rep2["lipschitz"]) < 1.0, f"expected a contraction, got {rep2['lipschitz']}"
    errs2 = [e["err"] for e in rep2["sweep"]]
    assert errs2[-1] < errs2[0] / 10, f"iterating should converge here, got {errs2}"

    # and the knob is the one we claim it is: Lip scales with gamma
    ratio = max(rep["lipschitz"]) / max(rep2["lipschitz"])
    assert 8 < ratio < 17, f"Lip should scale ~linearly with gamma (0.25/0.02), got {ratio:.1f}"


def test_lipschitz_estimator_is_not_fooled_by_bf16():
    """Guards the diagnostic itself against the finite-difference trap.

    An earlier version of `contraction_estimate` used finite differences and reported ~80
    instead of ~1.0 under bf16 autocast: at eps = 1e-3 the subtraction G(x + eps v) - G(x)
    cancels away every significant bf16 digit. A 50x overestimate of Lip would condemn a
    method that in fact sits right on the contraction boundary. Exact VJPs do not care
    about the compute precision, so the two estimates must agree.
    """
    if not torch.cuda.is_available():
        pytest.skip("the trap only bites under CUDA autocast")
    cfg = GPTConfig(n_layer=2)
    torch.manual_seed(0)
    m = GPT(cfg, mode="euler_implicit").cuda()
    h = m.embed(torch.randint(0, cfg.vocab_size, (2, 128), device="cuda"))

    def measure(amp):
        m.blocks[0].use_amp = amp
        torch.manual_seed(11)          # power iteration starts from a random vector
        return R.contraction_estimate(m.blocks[0], h, n_iter=30)

    with_amp, without_amp = measure(True), measure(False)
    assert abs(with_amp - without_amp) / without_amp < 0.02, (
        f"estimator is precision-sensitive: {with_amp:.4f} vs {without_amp:.4f}")


# --------------------------------------------------------------------------------------
# gate 3
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to measure peaks")
def test_gate3_peak_memory_is_flat_in_depth():
    """Stored memory grows with depth; reversible memory does not. The whole point."""
    from revlm import metering as M

    def peak_for(mode, L):
        cfg = GPTConfig(n_layer=L)
        torch.manual_seed(0)
        m = GPT(cfg, mode=mode).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
        x = torch.randint(0, cfg.vocab_size, (16, 512), device="cuda")
        for _ in range(2):
            m(x, x).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = M.allocated_gib()
        m(x, x).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        out = M.peak_gib() - base
        del m, opt, x
        torch.cuda.empty_cache()
        return out

    store_growth = peak_for("store", 24) / peak_for("store", 6)
    assert store_growth > 2.0, f"baseline should grow with depth, got {store_growth:.2f}x"
    for mode in ("midpoint", "euler"):
        ratio = peak_for(mode, 24) / peak_for(mode, 6)
        assert ratio < 1.10, f"{mode}: 4x the depth changed peak memory by {ratio:.2f}x"

    # and the honest comparison: reversibility must also beat plain recomputation, and by
    # a margin that widens with depth (checkpointing still keeps one tensor per layer)
    edge_shallow = peak_for("checkpoint", 6) / peak_for("midpoint", 6)
    edge_deep = peak_for("checkpoint", 24) / peak_for("midpoint", 24)
    assert edge_deep > edge_shallow > 1.0, f"edge {edge_shallow:.2f}x -> {edge_deep:.2f}x"
