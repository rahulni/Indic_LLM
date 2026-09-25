"""The backward strategies: how each stack gets its activations back without storing them.

Read this as four derivations followed by their code. The derivations are the assignment;
the code is short once they are written down.

Notation: the stream is h_0 ... h_L, the blocks are G_0 ... G_{L-1}, and G_l(h) is the
residual *branch* (gamma_l * F_l(h)), not the block output. a_l denotes dL/dh_l.

=======================================================================================
0. The obvious idea, and the condition it depends on  (mode "euler_implicit")
=======================================================================================
A pre-LN transformer already is an explicit Euler step:  h_{l+1} = h_l + G_l(h_l).
So the obvious way to make it reversible is to invert that step directly:

    h_l = h_{l+1} - G_l(h_l)        <- G_l evaluated at the UNKNOWN

which is implicit, so you reach for a fixed point, x <- h_{l+1} - G_l(x). That is a
contraction - and therefore converges at all - only while

    Lip(G_l) = gamma_l * Lip(F_l) < 1

Measured on this model at initialisation (d=384, T=512), Lip(F_l) ~ 10.5, so at the
default gamma = 0.1 we sit at Lip(G_l) ~ 1.05 - almost exactly *on* the boundary. The
iteration therefore neither converges nor blows up: it stalls, and the reconstruction
error sits near 0.09 however many steps it is given. At gamma = 0.02 (Lip ~ 0.2) it
converges to zero in a handful of steps. So implicit Euler is not impossible; it is
*conditional*, and the default configuration happens to sit on the wrong side by a hair.

Lip also depends on width and sequence length, not just gamma, so the safe region is not
a property of the method - it is a property of the model you happen to be training.

That conditionality is the problem, because gamma is a learnable parameter. Nothing in the
optimiser knows about the constraint, so gamma can drift above 1/Lip(F) mid-run, at which
point the reconstruction silently stops matching, the gradients quietly become wrong, and
the loss keeps falling anyway. Damping does not save it either: the damped iteration has
Lipschitz |1-a| + a*L, which still exceeds 1 for every a > 0 once L > 1. An unconditional
implicit solve needs Newton-Krylov - a linear solve per layer, per step.

`contraction_estimate()` measures Lip(G_l) and `train.py` records the worst value seen, so
the condition is monitored rather than assumed. The two-state explicit integrators below
need no such condition, which is why they are the ones that get trained.

    A measurement trap worth repeating: estimating Lip by finite differences,
    (G(x + eps v) - G(x)) / eps, under bf16 autocast returns ~80 rather than ~1.05. bf16
    carries ~3 decimal digits, so at eps = 1e-3 the subtraction is pure cancellation noise
    and dividing by eps amplifies it. `contraction_estimate` uses exact autograd VJPs.

=======================================================================================
1. SYMPLECTIC EULER  (mode "euler")   -- explicit, exact, one extra F per layer
=======================================================================================
Give the stream a velocity and the first-order scheme becomes exactly invertible:

    forward     v_{l+1} = v_l + G_l(h_l)          v_0 = 0
                h_{l+1} = h_l + v_{l+1}

    inverse     h_l = h_{l+1} - v_{l+1}           <- both evaluated at KNOWN quantities
                v_l = v_{l+1} - G_l(h_l)

Explicit in both directions, exact to floating point, one evaluation of G per layer in
backward - and that same evaluation serves the gradient, so nothing is computed twice.
(This is the Momentum-ResNet construction: momentum is what buys the invertibility.)

Writing c = a_{l+1} + b_{l+1} for the total cotangent arriving at v_{l+1},

    a_l = a_{l+1} + J_l^T c        b_l = c        dL/dtheta_l = (dG_l/dtheta_l)^T c

so backward carries two gradient tensors - O(1) in depth, matching the forward.

=======================================================================================
2. MIDPOINT / LEAPFROG  (mode "midpoint")  -- explicit, exact, one extra F per layer
=======================================================================================
    bootstrap   h_1 = h_0 + G_0(h_0)              (h_0 is kept: one tensor, O(1) in depth)
    forward     h_{l+1} = h_{l-1} + 2 G_l(h_l)
    inverse     h_{l-1} = h_{l+1} - 2 G_l(h_l)    <- G_l evaluated at the KNOWN h_l

Differentiating and collecting the two places h_l appears gives a three-term recurrence

    a_l = a_{l+2} + 2 J_l^T a_{l+1}

seeded with a_L = dL/dh_L and a_{L+1} = 0, closed by a_0 = a_2 + (I + J_0^T) a_1. So this
one also carries a rolling window of exactly two gradient tensors.

Leapfrog's known pathology is that the even and odd chains couple only through G and can
drift apart (the parasitic mode). `model.diag['parasitic']` tracks it.

=======================================================================================
3. COUPLING / RevNet  (mode "coupling")
=======================================================================================
    forward     y1 = x1 + F(x2);  y2 = x2 + G(y1)
    inverse     x2 = y2 - G(y1);  x1 = y1 - F(x2)

Exact, no step size, no fixed point, no parasitic mode. The cost is in the shapes: F and G
each see d/2 channels.

---------------------------------------------------------------------------------------
All three walks end on a *reconstructed* h_0 while the true h_0 was kept, so every step
gets a free end-to-end reconstruction check: `model.diag['recon_h0']`. Floating point is
not associative, so (a + b) - b != a exactly, and that residue compounds with depth. The
number is small; it is not zero; and the honest thing is to publish it.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _flat_params(modules):
    """Flatten the blocks' parameters into one list plus per-block spans.

    Parameters go through `autograd.Function` as explicit inputs so their gradients are
    returned by `backward` and land in `.grad` through the normal autograd engine. The
    usual shortcut - writing `p.grad` by hand inside backward - silently bypasses hooks
    and gradient clipping.
    """
    plist, spans = [], []
    for m in modules:
        ps = [p for p in m.parameters() if p.requires_grad]
        spans.append((len(plist), len(plist) + len(ps)))
        plist.extend(ps)
    return plist, spans


def _vjp(block, x, cotangent, params):
    """(G(x), dL/dx, dL/dparams) at one point, with the one-layer graph freed right after."""
    with torch.enable_grad():
        xr = x.detach().requires_grad_(True)
        y = block(xr)
        grads = torch.autograd.grad(y, [xr] + list(params), grad_outputs=cotangent,
                                    allow_unused=True)
    return y.detach(), grads[0], grads[1:]


def _record(model, track, **kw):
    if track:
        model.diag.update(kw)


# --------------------------------------------------------------------------------------
# 1. Symplectic (momentum) Euler
# --------------------------------------------------------------------------------------
class _EulerStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h0, meta, *params):
        blocks, spans, model, track = meta
        with torch.no_grad():
            h, v = h0, torch.zeros_like(h0)
            for b in blocks:
                v = v + b(h)
                h = h + v
        ctx.meta = meta
        ctx.save_for_backward(h, v, h0)      # two boundary states, whatever the depth
        return h

    @staticmethod
    def backward(ctx, grad_out):
        blocks, spans, model, track = ctx.meta
        h, v, h0_true = ctx.saved_tensors
        pgrads = [None] * spans[-1][1]

        a = grad_out                      # dL/dh_L
        b_ = torch.zeros_like(grad_out)   # dL/dv_L: v_L never reaches the loss
        for l in range(len(blocks) - 1, -1, -1):
            blk = blocks[l]
            lo, _ = spans[l]
            h = h - v                     # h_l   = h_{l+1} - v_{l+1}
            c = a + b_                    # total cotangent arriving at v_{l+1}
            y, gx, gth = _vjp(blk, h, c, list(blk.parameters()))
            v = v - y                     # v_l   = v_{l+1} - G_l(h_l)
            a, b_ = a + gx, c
            for i, g in enumerate(gth):
                pgrads[lo + i] = g

        _record(model, track, recon_h0=(h - h0_true).abs().max().item(),
                velocity_norm=v.abs().mean().item())
        return (a, None, *pgrads)


# --------------------------------------------------------------------------------------
# 2. Midpoint / leapfrog
# --------------------------------------------------------------------------------------
class _MidpointStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h0, meta, *params):
        blocks, spans, model, track = meta
        with torch.no_grad():
            h_prev = h0
            h_cur = h0 + blocks[0](h0)                 # bootstrap: one Euler step
            for l in range(1, len(blocks)):
                h_prev, h_cur = h_cur, h_prev + 2.0 * blocks[l](h_cur)
        ctx.meta = meta
        ctx.save_for_backward(h_cur, h_prev, h0)       # three tensors, whatever the depth
        return h_cur

    @staticmethod
    def backward(ctx, grad_out):
        blocks, spans, model, track = ctx.meta
        h_hi, h_cur, h0_true = ctx.saved_tensors       # h_L, h_{L-1}, h_0
        pgrads = [None] * spans[-1][1]

        g_hi = grad_out                                # a_L, complete
        g_mid = torch.zeros_like(grad_out)             # a_{L+1} = 0

        for m in range(len(blocks) - 1, 0, -1):
            blk = blocks[m]
            lo, _ = spans[m]
            # One evaluation of G_m serves both the gradient and the reconstruction.
            # The factor of 2 is applied *after* the VJP and in place: passing
            # `2.0 * g_hi` as the cotangent would allocate a second full-size tensor per
            # layer, which is enough to lose a whole batch-size step at the OOM boundary
            # (midpoint failed at batch 112 where symplectic Euler fitted, for this alone).
            y, gx, gth = _vjp(blk, h_cur, g_hi, list(blk.parameters()))
            a_m = gx.mul_(2.0).add_(g_mid)             # a_m = a_{m+2} + 2 J^T a_{m+1}
            for i, g in enumerate(gth):
                pgrads[lo + i] = g.mul_(2.0) if g is not None else None
            # h_{m-1} = h_{m+1} - 2 G_m(h_m), without materialising 2*y
            h_low = torch.sub(h_hi, y, alpha=2.0)
            g_mid, g_hi = g_hi, a_m
            h_hi, h_cur = h_cur, h_low

        lo, _ = spans[0]                               # bootstrap: h_1 = h_0 + G_0(h_0)
        _, gx, gth = _vjp(blocks[0], h_cur, g_hi, list(blocks[0].parameters()))
        for i, g in enumerate(gth):
            pgrads[lo + i] = g
        a0 = g_mid + g_hi + gx                         # a_0 = a_2 + (I + J_0^T) a_1

        _record(model, track, recon_h0=(h_cur - h0_true).abs().max().item(),
                parasitic=(h_hi - h_cur).abs().mean().item())
        return (a0, None, *pgrads)


# --------------------------------------------------------------------------------------
# 3. RevNet coupling
# --------------------------------------------------------------------------------------
class _CouplingStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h0, meta, *params):
        blocks, spans, model, track = meta
        with torch.no_grad():
            x1, x2 = h0.chunk(2, dim=-1)
            for b in blocks:
                y1 = x1 + b.f(x2)
                x1, x2 = y1, x2 + b.g(y1)
            h = torch.cat([x1, x2], dim=-1)
        ctx.meta = meta
        ctx.save_for_backward(h, h0)
        return h

    @staticmethod
    def backward(ctx, grad_out):
        blocks, spans, model, track = ctx.meta
        h_top, h0_true = ctx.saved_tensors
        pgrads = [None] * spans[-1][1]

        y1, y2 = h_top.chunk(2, dim=-1)
        dy1, dy2 = (t.contiguous() for t in grad_out.chunk(2, dim=-1))

        for l in range(len(blocks) - 1, -1, -1):
            b = blocks[l]
            lo, _ = spans[l]
            gy, dg_x, dg_th = _vjp(b.g, y1, dy2, list(b.g.parameters()))
            x2 = y2 - gy                                   # invert the second half
            dy1 = dy1 + dg_x
            fx, df_x, df_th = _vjp(b.f, x2, dy1, list(b.f.parameters()))
            x1 = y1 - fx                                   # invert the first half
            dy2 = dy2 + df_x
            # _flat_params walked b.parameters(), which yields f's before g's
            for i, g in enumerate(list(df_th) + list(dg_th)):
                pgrads[lo + i] = g
            y1, y2 = x1, x2

        _record(model, track,
                recon_h0=(torch.cat([y1, y2], -1) - h0_true).abs().max().item())
        return (torch.cat([dy1, dy2], dim=-1), None, *pgrads)


# --------------------------------------------------------------------------------------
# 0. the negative result, kept runnable
# --------------------------------------------------------------------------------------
def euler_invert(block, h_next, iters: int, track: bool = False):
    """Fixed point for h = h_next - G(h). Returns (h, last step size). Diverges; see above."""
    x = h_next
    res = float("nan")
    with torch.no_grad():
        for _ in range(iters):
            x_new = h_next - block(x)
            if track:
                res = (x_new - x).abs().max().item()
            x = x_new
    return x, res


class _EulerImplicitStack(torch.autograd.Function):
    """Kept so the failure can be trained and plotted, not merely described."""

    @staticmethod
    def forward(ctx, h0, meta, *params):
        blocks, spans, iters, model, track = meta
        with torch.no_grad():
            h = h0
            for b in blocks:
                h = h + b(h)
        ctx.meta = meta
        ctx.save_for_backward(h, h0)
        return h

    @staticmethod
    def backward(ctx, grad_out):
        blocks, spans, iters, model, track = ctx.meta
        h, h0_true = ctx.saved_tensors
        pgrads = [None] * spans[-1][1]
        a, worst = grad_out, 0.0
        for l in range(len(blocks) - 1, -1, -1):
            b = blocks[l]
            lo, _ = spans[l]
            h, res = euler_invert(b, h, iters, track=track)
            if res == res:
                worst = max(worst, res)
            _, gx, gth = _vjp(b, h, a, list(b.parameters()))
            a = a + gx
            for i, g in enumerate(gth):
                pgrads[lo + i] = g
        _record(model, track, recon_h0=(h - h0_true).abs().max().item(),
                euler_residual=worst)
        return (a, None, *pgrads)


def euler_implicit_report(model, h0, iters=(1, 2, 4, 8, 16, 32)):
    """Lip(G_l) per layer, and reconstruction error vs iteration count."""
    lips, h = [], h0
    for b in model.blocks:
        lips.append(contraction_estimate(b, h))
        with torch.no_grad():
            h = h + b(h)
    with torch.no_grad():
        h1 = h0 + model.blocks[0](h0)
    sweep = []
    for K in iters:
        x, res = euler_invert(model.blocks[0], h1, K, track=True)
        sweep.append({"iters": K, "step": res, "err": (x - h0).abs().max().item()})
    return {"lipschitz": lips, "sweep": sweep, "stream_scale": h0.abs().mean().item()}


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------
def run_stack(model, h):
    """Apply the block stack to `h` using whichever backward strategy `model.mode` names."""
    mode, blocks = model.mode, model.blocks
    track = bool(getattr(model, "track_recon", False))

    if mode == "store":
        for b in blocks:
            h = h + b(h)
        return h

    if mode == "checkpoint":
        from torch.utils.checkpoint import checkpoint

        for b in blocks:
            h = h + checkpoint(b, h, use_reentrant=False)
        return h

    params, spans = _flat_params(blocks)
    if mode == "euler":
        return _EulerStack.apply(h, (blocks, spans, model, track), *params)
    if mode == "midpoint":
        return _MidpointStack.apply(h, (blocks, spans, model, track), *params)
    if mode == "coupling":
        return _CouplingStack.apply(h, (blocks, spans, model, track), *params)
    if mode == "euler_implicit":
        meta = (blocks, spans, model.euler_iters, model, track)
        return _EulerImplicitStack.apply(h, meta, *params)
    raise ValueError(f"unknown mode {mode}")


# --------------------------------------------------------------------------------------
# diagnostics used by the tests, notebook 01 and the training loop
# --------------------------------------------------------------------------------------
def contraction_estimate(block, x, n_iter: int = 20) -> float:
    """Power iteration for ||J|| of G_l at x, using exact autograd VJPs.

    This is the number that decides whether an implicit Euler inverse can converge at all
    (it needs < 1), and whether it still can *later in training*, since gamma is learnable.

    Exact VJPs, not finite differences. A finite-difference estimate under bf16 autocast
    reads ~80 here instead of ~1.4 - the subtraction G(x + eps v) - G(x) cancels away every
    significant bf16 digit and the division by eps turns what is left into noise. That
    50x overestimate is easy to believe and would condemn a method that in fact sits right
    on the contraction boundary.
    """
    v = torch.randn_like(x)
    v = v / v.norm()
    sigma = 0.0
    for _ in range(n_iter):
        xr = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = block(xr)
            (g,) = torch.autograd.grad(y, xr, grad_outputs=v)
        nrm = g.norm()
        if nrm == 0:
            return 0.0
        sigma, v = (nrm / v.norm()).item(), g / nrm
    return sigma


@torch.no_grad()
def forward_states(model, h0):
    """Every true h_l, to check a reconstruction against. O(L) memory - tests only."""
    blocks = model.blocks
    if model.mode == "midpoint":
        states = [h0]
        h_prev, h_cur = h0, h0 + blocks[0](h0)
        states.append(h_cur)
        for l in range(1, len(blocks)):
            h_prev, h_cur = h_cur, h_prev + 2.0 * blocks[l](h_cur)
            states.append(h_cur)
        return states
    if model.mode == "euler":
        states, h, v = [h0], h0, torch.zeros_like(h0)
        for b in blocks:
            v = v + b(h)
            h = h + v
            states.append(h)
        return states
    if model.mode == "coupling":
        # a CoupledBlock's output IS the next state - there is no residual add outside it
        states, h = [h0], h0
        for b in blocks:
            h = b(h)
            states.append(h)
        return states
    states, h = [h0], h0
    for b in blocks:
        h = h + b(h)
        states.append(h)
    return states


@torch.no_grad()
def reconstruction_drift(model, h0, iters: int = 4):
    """max |h_l reconstructed - h_l true| for every l. The claim, measured."""
    blocks, true = model.blocks, forward_states(model, h0)
    L = len(blocks)
    drift = [0.0] * (L + 1)

    if model.mode == "midpoint":
        h_hi, h_cur = true[L], true[L - 1]
        for m in range(L - 1, 0, -1):
            h_low = h_hi - 2.0 * blocks[m](h_cur)
            drift[m - 1] = (h_low - true[m - 1]).abs().max().item()
            h_hi, h_cur = h_cur, h_low
        return drift

    if model.mode == "euler":
        v, h = torch.zeros_like(h0), h0          # replay forward to recover v_L
        for b in blocks:
            v = v + b(h)
            h = h + v
        for l in range(L - 1, -1, -1):
            h = h - v
            v = v - blocks[l](h)
            drift[l] = (h - true[l]).abs().max().item()
        return drift

    if model.mode == "coupling":
        h = true[L]
        for l in range(L - 1, -1, -1):
            y1, y2 = h.chunk(2, dim=-1)
            x2 = y2 - blocks[l].g(y1)
            x1 = y1 - blocks[l].f(x2)
            h = torch.cat([x1, x2], dim=-1)
            drift[l] = (h - true[l]).abs().max().item()
        return drift

    h = true[L]
    for l in range(L - 1, -1, -1):
        h, _ = euler_invert(blocks[l], h, iters)
        drift[l] = (h - true[l]).abs().max().item()
    return drift
