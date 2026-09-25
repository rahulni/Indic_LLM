"""revlm - a 21M-parameter GPT that can be trained four ways.

The four ways share one `Block` (LN -> causal attention -> LN -> MLP) and one learnable
per-layer scale `gamma`, so they are the *same function* wherever the integrator allows it.
What differs is only how the backward pass gets its activations back:

    STORE       keep every layer's activations, the way autograd normally does
    EULER       throw them away, solve  h_l = h_{l+1} - G_l(h_l)  by fixed point
    MIDPOINT    throw them away, read them off  h_{l-1} = h_{l+1} - 2 G_l(h_l)
    COUPLING    split the channels and invert each half exactly (RevNet)

`STORE` and `EULER` have a bit-identical forward pass. That is deliberate: it makes the
memory saving measurable against a control whose loss curve *must* overlay it.
"""

__all__ = ["model", "reversible", "metering", "data", "train", "ladder", "plots"]
