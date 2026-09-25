# Reversibility on one page

The residual stream is `h_0 ... h_L`. `G_l(h) = gamma_l * F_l(h)` is the residual
**branch** (LN -> causal attention -> LN -> MLP), *not* the block output. `a_l = dL/dh_l`.

## The hinge

A pre-LN transformer block **is** an explicit Euler step on the depth axis:

```
h_{l+1} = h_l + G_l(h_l)              <-  dh/dl = G(h),  step size 1
```

So "make it reversible" is really: **which integrators can be run backwards?**

## The four stacks

| | forward | inverse | exact? | extra F per layer | states kept |
|---|---|---|---|---|---|
| **store** | `h + G(h)` | — (keeps everything) | — | 0 | ~12 per layer |
| **checkpoint** | `h + G(h)` | recompute from a stored input | yes | 1 | 1 per layer |
| **euler_implicit** | `h + G(h)` | `h_l = h_{l+1} - G_l(h_l)`, by fixed point | only if `Lip(G)<1` | K + 1 | 1 total |
| **euler** (symplectic) | `v += G(h); h += v` | `h -= v; v -= G(h)` | yes | 1 | 2 total |
| **midpoint** (leapfrog) | `h_{l+1} = h_{l-1} + 2G(h_l)` | `h_{l-1} = h_{l+1} - 2G(h_l)` | yes | 1 | 3 total |
| **coupling** (RevNet) | `y1 = x1+F(x2); y2 = x2+G(y1)` | `x2 = y2-G(y1); x1 = y1-F(x2)` | yes | 1 | 2 total |

The rule of thumb: **one state cannot be inverted explicitly; two can.** Whether the
second state is a velocity (symplectic) or the previous layer (midpoint) or the other half
of the channels (coupling) is a design choice, not a difference in kind.

## The backward passes

Write `J_l = dG_l/dh` at `h_l`. Each engine walks `l = L-1 ... 0`, rebuilding `h_l` as it
goes and taking one vector-Jacobian product per layer.

```
euler_implicit   a_l = a_{l+1} + J^T a_{l+1}                      theta: (dG/dtheta)^T a_{l+1}
symplectic       c   = a_{l+1} + b_{l+1}
                 a_l = a_{l+1} + J^T c,     b_l = c               theta: (dG/dtheta)^T c
midpoint         a_l = a_{l+2} + 2 J^T a_{l+1}                    theta: 2 (dG/dtheta)^T a_{l+1}
                 seed a_L = dL/dh_L, a_{L+1} = 0
                 close a_0 = a_2 + (I + J_0^T) a_1
coupling         dy1 += (dG/dy1)^T dy2 ;  dx2 = dy2 + (dF/dx2)^T dy1 ;  dx1 = dy1
```

Midpoint's is a **three-term** recurrence, so backward carries a rolling window of exactly
two gradient tensors — `O(1)` in depth, matching its forward.

## The conditions and the costs

```
implicit Euler converges  <=>  Lip(G_l) = gamma_l * Lip(F_l) < 1
```

Measured here at `d=384, T=512`: `Lip(F) ~ 10.5`, so `gamma=0.1` gives `Lip(G) ~ 1.05` —
on the boundary, and the iteration stalls. `gamma=0.02` gives `~0.2` and it converges.
`Lip` depends on **width and sequence length**, so a toy-sized test certifies nothing.

```
speed     ~4 compute units per step vs the baseline's 3   ->  ~25% slower
memory    O(1) in DEPTH, not O(1):
            weights + grads + Adam      fixed
            boundary states             2-3 tensors
            one live layer's graph      O(batch x seq)   <- the slogan omits this
            fp32 logits                 O(batch x seq x vocab)  <- the new bottleneck
```

## The restrictions

- **No dropout.** The backward pass would have to replay the same random mask; without
  that, the reconstruction is of a different function than the forward ran.
- **No weight decay.** Stated in the session; the runs here keep it off *everywhere* so
  the baseline is not quietly given an advantage the reversible runs cannot have.

## Traps

1. **Estimating `Lip` by finite differences under bf16** reports ~80 instead of ~1.05.
   At `eps=1e-3` the subtraction cancels every significant bf16 digit. Use autograd VJPs.
2. **Testing the engines at toy width.** `Lip` scales with `d`; at `d=16` implicit Euler
   looks unconditionally fine.
3. **Comparing an engine's gradients to a *different* recurrence's.** Midpoint is not the
   baseline, and it is not supposed to match it. Compare each engine to autograd on its
   own equations, in fp64.
4. **Forgetting checkpointing exists.** It gets most of the memory saving for the same
   ~25% slowdown. Reversibility's real edge is `(L-3)` stream tensors — it grows with
   depth, and at shallow depth it is thin.
5. **bf16 residual streams.** The reconstruction error compounds with depth; keep the
   stream in fp32 and compute `G` in bf16.
