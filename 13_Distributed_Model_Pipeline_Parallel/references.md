# References

## The session

`ERA V5 Session - 2026_09_19 06_37 IST - Transcript.docx`, in this folder. The parts this
work is built on:

| Time | What |
|---|---|
| 01:15 | "Section 16" - where the session leaves what the internet will teach you and starts on reversibility |
| 01:17 | The reversibility pitch: full forward, delete the activations, rebuild them during backward |
| 01:19 | The two variants named - Euler and midpoint - and the midpoint rule written out as `P_{l+1} = P_{l-1} + 2h f(P_l)` |
| 01:21 | The restrictions: no weight decay, no dropout |
| 01:24 | "at least 30 to 40% slower" - the slowdown figure this report scores itself against |
| 01:28 | 131k sequence: ~23 GB reversible against ~2 TB stored |
| 01:30 | **The assignment** |
| 01:37 | Avnish asks for the cloud cost comparison - answered in the README |

## The ideas, and where they come from

- **Reversible residual networks.** Gomez, Ren, Urtasun, Grosse, *The Reversible Residual
  Network: Backpropagation Without Storing Activations* (NeurIPS 2017), arXiv:1707.04585.
  The additive coupling used here as the `coupling` variant.
- **Reformer.** Kitaev, Kaiser, Levskaya (ICLR 2020), arXiv:2001.04451. Reversible layers
  inside a transformer specifically.
- **Neural ODEs.** Chen, Rubanova, Bettencourt, Duvenaud (NeurIPS 2018), arXiv:1806.07366.
  The reading of depth as time that makes "which integrator?" the right question.
- **Momentum ResNets.** Sander, Ablin, Blondel, Peyré (ICML 2021), arXiv:2102.07870.
  Adding a velocity to make a first-order scheme exactly invertible - the `euler` variant
  here is this construction.
- **Reversible architectures as dynamical systems.** Chang, Meng, Haber, Ruthotto, Begert,
  Holtham (AAAI 2018), arXiv:1709.03698. Leapfrog/midpoint as a reversible network, i.e.
  the `midpoint` variant.
- **Gradient checkpointing.** Chen, Xu, Zhang, Guestrin, *Training Deep Nets with Sublinear
  Memory Cost* (2016), arXiv:1604.06174. The rival every reversibility claim should be
  measured against, and usually is not.
- **LayerScale.** Touvron et al., *Going deeper with Image Transformers* (2021),
  arXiv:2103.17239. The per-layer `gamma` - which here doubles as the contraction knob.
- **Leapfrog's parasitic mode.** Any numerical-methods text; Durran, *Numerical Methods for
  Fluid Dynamics*, ch. 2 covers the even/odd splitting and the Robert-Asselin filter that
  damps it - at the cost of the exact reversibility we need.

## Data

- **TinyStories.** Eldan & Li (2023), arXiv:2305.07759. `roneneldan/TinyStories` on the
  Hugging Face Hub, file `TinyStoriesV2-GPT4-train.txt`, first 350 MiB fetched by HTTP
  range request. CDLA-Sharing-1.0.

## Prices used in the cost section

On-demand USD/hour, list prices as advertised in September 2026, rounded. They move, and
spot/reserved pricing is far lower - the table is there for the *ratio* between variants,
which is what the measurement actually establishes, not for the absolute dollars.

| GPU | USD/hour |
|---|---|
| A100 40GB | 1.29 |
| A100 80GB | 1.79 |
| H100 80GB | 2.99 |
