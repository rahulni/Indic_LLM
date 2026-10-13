"""Adam by hand, checked against PyTorch.

One weight, five gradients. Compute m, v, m_hat, v_hat and the step three ways:
  1. exactly, with 50-digit decimals (the reference),
  2. by hand in ordinary float64,
  3. with torch.optim.Adam, reading its internal state after every step.

Run:  python adam_by_hand.py        (CPU is enough; a GPU adds the fused kernel)
Exit code 0 when every implementation agrees with the reference to its tolerance, 1 otherwise.
"""
import decimal
import math
import sys

import torch

GRADS = [0.50, 0.40, 0.60, 0.45, 0.55]
LR, BETA1, BETA2, EPS, W0 = 1e-3, 0.9, 0.999, 1e-8, 1.0
FIELDS = ("m", "v", "m_hat", "v_hat", "step", "w")


def adam_exact(grads=GRADS):
    """The reference: every operation in 50-digit decimal arithmetic."""
    D = decimal.Decimal
    decimal.getcontext().prec = 50
    b1, b2, lr, eps = D("0.9"), D("0.999"), D("0.001"), D("1e-8")
    w, m, v, rows = D(1), D(0), D(0), []
    for t, g in enumerate(grads, 1):
        g = D(str(g))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        step = -lr * m_hat / (v_hat.sqrt() + eps)
        w += step
        rows.append(dict(m=m, v=v, m_hat=m_hat, v_hat=v_hat, step=step, w=w))
    return rows


def adam_by_hand(grads=GRADS, bias_correction=True):
    """The same five lines in ordinary floats - what you would write on paper."""
    w, m, v, rows = W0, 0.0, 0.0, []
    for t, g in enumerate(grads, 1):
        m = BETA1 * m + (1 - BETA1) * g                    # first moment: average gradient
        v = BETA2 * v + (1 - BETA2) * g * g                # second moment: average squared gradient
        m_hat = m / (1 - BETA1 ** t) if bias_correction else m
        v_hat = v / (1 - BETA2 ** t) if bias_correction else v
        step = -LR * m_hat / (math.sqrt(v_hat) + EPS)
        w += step
        rows.append(dict(m=m, v=v, m_hat=m_hat, v_hat=v_hat, step=step, w=w))
    return rows


def adam_torch(grads=GRADS, dtype=torch.float64, device="cpu", **impl):
    """torch.optim.Adam on a one-element parameter, gradient set by hand each step.

    PyTorch stores m (exp_avg), v (exp_avg_sq) and the step count; it never stores
    m_hat or v_hat, so those are derived here from the stored state exactly as Adam uses them.
    """
    p = torch.nn.Parameter(torch.tensor([W0], dtype=dtype, device=device))
    opt = torch.optim.Adam([p], lr=LR, betas=(BETA1, BETA2), eps=EPS, **impl)
    rows = []
    for g in grads:
        before = p.detach().clone()
        p.grad = torch.tensor([g], dtype=dtype, device=device)
        opt.step()
        st = opt.state[p]
        t = int(st["step"])
        m, v = st["exp_avg"].item(), st["exp_avg_sq"].item()
        rows.append(dict(m=m, v=v, m_hat=m / (1 - BETA1 ** t), v_hat=v / (1 - BETA2 ** t),
                         step=(p.detach() - before).item(), w=p.item()))
    return rows


def max_diff(rows, ref):
    return {f: max(abs(float(r[f]) - float(e[f])) for r, e in zip(rows, ref)) for f in FIELDS}


def implementations():
    impls = [("hand, float64", lambda: adam_by_hand(), 1e-12),
             ("torch float64, for-loop", lambda: adam_torch(dtype=torch.float64, foreach=False), 1e-12),
             ("torch float64, foreach", lambda: adam_torch(dtype=torch.float64, foreach=True), 1e-12),
             ("torch float32, for-loop", lambda: adam_torch(dtype=torch.float32, foreach=False), 1e-6),
             ("torch float32, foreach", lambda: adam_torch(dtype=torch.float32, foreach=True), 1e-6)]
    try:
        has_gpu = torch.cuda.device_count() > 0 and torch.cuda.is_available()
    except Exception:
        has_gpu = False
    if has_gpu:
        impls += [("torch float64, fused CUDA", lambda: adam_torch(dtype=torch.float64, device="cuda", fused=True), 1e-12),
                  ("torch float32, fused CUDA", lambda: adam_torch(dtype=torch.float32, device="cuda", fused=True), 1e-5)]
    return impls


def main():
    ref = adam_exact()
    print(f"Adam, one weight, five gradients  (lr={LR}, betas=({BETA1}, {BETA2}), eps={EPS}, w0={W0})\n")
    print(f"{'t':>2} {'g':>5} {'m':>8} {'v':>9} {'m_hat':>7} {'v_hat':>7} {'step':>10} {'w':>9}")
    for t, (g, r) in enumerate(zip(GRADS, ref), 1):
        r = {k: float(x) for k, x in r.items()}
        print(f"{t:>2} {g:5.2f} {r['m']:8.4f} {r['v']:9.6f} {r['m_hat']:7.4f} {r['v_hat']:7.4f} "
              f"{r['step']:10.6f} {r['w']:9.6f}")
    print("\nlargest |difference| from the 50-digit reference, per quantity:")
    print(f"{'implementation':<28}" + "".join(f"{f:>10}" for f in FIELDS) + "   tolerance  ok")
    ok_all = True
    for name, fn, tol in implementations():
        d = max_diff(fn(), ref)
        ok = all(x <= tol for x in d.values())
        ok_all &= ok
        print(f"{name:<28}" + "".join(f"{d[f]:10.1e}" for f in FIELDS) + f"   {tol:9.0e}  {'yes' if ok else 'NO'}")
    print("\nall implementations agree" if ok_all else "\nMISMATCH: see the table above")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
