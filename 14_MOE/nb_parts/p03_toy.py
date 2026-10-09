from . import md, code, mathbox

CELLS = [
md(r'''
---
## B · The whole experiment in 2-D, in ten seconds

Before the real model, here is the whole idea taken literally on a toy problem small enough to
draw. **A linear model** (logistic regression, 3 parameters) is trained, then **converted into a
mixture of 4 linear experts**, then trained further.

The data are four clusters, and each cluster's label boundary points a different way. No single
line separates them; a router that recognises *which cluster* a point is in, plus one line per
cluster, can. This is Chen et al.'s 2022 result in miniature: on clustered data, one
network of a given kind is capped, and a mixture of them trained by gradient descent is not.

The conversion is **copy upcycling**: every expert starts as an exact copy of the linear model,
top-2 routing renormalises the weights to sum to one, and so the converted model computes *exactly*
the same function. The loss curve is continuous through the conversion.

🧠 **Intuition.** One straight line cannot cut four clusters that each need a differently angled
cut. Give the model four lines and a "which cluster am I in?" chooser, and it can.
'''),
mathbox("one line vs a mixture of lines, and why identical copies drift apart", r'''
**A linear model** predicts $\hat y(x) = w^\top x + b$. Its decision boundary is one line,
$w^\top x + b = 0$. Our clusters need boundaries at 0°, 90°, 45° and 135°, so the best single line
gets about 63%.

**A mixture of linear experts** with router weights $g_i(x)$:

$$\hat y(x) = \sum_{i \in \mathcal{T}(x)} g_i(x)\,\big(w_i^\top x + b_i\big).$$

Wherever the router makes the same choice, this is again linear, but it is a *different* line in
each region. The model is piecewise linear, and four regions can carry four orientations.

**Why the copy conversion is exact.** If $w_i = w$ and $b_i = b$ for every expert, then

$$\hat y(x) = \Big(\sum_{i \in \mathcal{T}} g_i\Big)(w^\top x + b) = w^\top x + b,$$

because the renormalised weights sum to 1.

**Why the identical copies still become different.** The gradient for expert $i$ is

$$\frac{\partial \ell}{\partial w_i} = \sum_{x \text{ routed to } i} g_i(x)\,\frac{\partial \ell}{\partial \hat y}\,x.$$

That is a sum over *different points* for different experts, so the updates differ and the copies
drift apart. Breaking the symmetry needs only one thing: that the router sends different points to
different experts. Its small random initialisation does that.
'''),
code(r'''
torch.manual_seed(0)
TOY_C = torch.tensor([[-2.5, -2.5], [2.5, -2.5], [-2.5, 2.5], [2.5, 2.5]])
toy_ang = torch.tensor([0.0, 90.0, 45.0, 135.0]) * math.pi / 180
toy_c = torch.arange(4).repeat_interleave(500)
TX = TOY_C[toy_c] + 0.9 * torch.randn(2000, 2)
TY = (((TX - TOY_C[toy_c]) * torch.stack([toy_ang.cos(), toy_ang.sin()], 1)[toy_c]).sum(1) > 0).float()

class MixtureOfLinear(nn.Module):
    """E linear experts + a router; top-k weights renormalised to 1, so copies reproduce the original."""
    def __init__(self, lin, E=4, k=2):
        super().__init__()
        self.W = nn.Parameter(lin.weight.detach().repeat(E, 1))        # [E, 2]: E copies of the linear model
        self.b = nn.Parameter(lin.bias.detach().repeat(E))             # [E]
        self.router = nn.Linear(2, E)
        nn.init.normal_(self.router.weight, 0, 0.01); nn.init.zeros_(self.router.bias)
        self.k = k
    def forward(self, x):
        s = torch.softmax(self.router(x), -1)
        top = s.topk(self.k, -1).indices
        w = s.gather(1, top); w = w / w.sum(-1, keepdim=True)
        return ((x @ self.W.t() + self.b).gather(1, top) * w).sum(1), top

lin = nn.Linear(2, 1)
opt = torch.optim.Adam(lin.parameters(), lr=0.05)
toy_hist = []
for _ in range(300):
    l = F.binary_cross_entropy_with_logits(lin(TX).squeeze(1), TY)
    opt.zero_grad(); l.backward(); opt.step(); toy_hist.append(l.item())
acc_lin = ((lin(TX).squeeze(1) > 0).float() == TY).float().mean().item()

mol = MixtureOfLinear(lin)
with torch.no_grad():
    gate("toy: copy-converted mixture == the linear model", torch.allclose(mol(TX)[0], lin(TX).squeeze(1), atol=1e-6))
opt = torch.optim.Adam(mol.parameters(), lr=0.05)
for _ in range(600):
    l = F.binary_cross_entropy_with_logits(mol(TX)[0], TY)
    opt.zero_grad(); l.backward(); opt.step(); toy_hist.append(l.item())
with torch.no_grad():
    acc_mol = ((mol(TX)[0] > 0).float() == TY).float().mean().item()
print(f"linear model: loss {toy_hist[299]:.3f}, accuracy {acc_lin:.1%}   ->   mixture of 4 linear experts: loss {toy_hist[-1]:.3f}, accuracy {acc_mol:.1%}")
'''),
code(r'''
fig = plt.figure(figsize=(12.5, 3.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.28)
a0 = fig.add_subplot(gs[0])
a0.plot(range(1, 301), toy_hist[:300], color=DENSE_C, label="linear model")
a0.plot(range(300, len(toy_hist) + 1), toy_hist[299:], color=MOE_C, label="mixture of 4 linear experts")
a0.axvline(300, color=MUTED, lw=1)
a0.text(300, 0.04, "  convert", transform=a0.get_xaxis_transform(), va="bottom", color=INK2, fontsize=8.5)
a0.set(xlabel="step", ylabel="loss (BCE)", title="Convert, keep training, keep dropping")
a0.legend(fontsize=8, loc="upper right")
gx, gy = torch.meshgrid(torch.linspace(-6, 6, 240), torch.linspace(-6, 6, 240), indexing="xy")
G = torch.stack([gx.flatten(), gy.flatten()], 1)
with torch.no_grad():
    z_lin = (lin(G).squeeze(1) > 0).float().view(240, 240)
    z_mol, top = mol(G)
    z_mol = (z_mol > 0).float().view(240, 240)
    regions = top[:, 0].view(240, 240)
two = LinearSegmentedColormap.from_list("two", ["#1c2a3d", "#3d2416"])
for ax, Z, title in ((fig.add_subplot(gs[1]), z_lin, f"linear model ({acc_lin:.0%})"),
                     (fig.add_subplot(gs[2]), z_mol, f"mixture ({acc_mol:.0%})")):
    ax.imshow(Z, extent=[-6, 6, -6, 6], origin="lower", cmap=two, alpha=1.0)
    ax.scatter(TX[:, 0], TX[:, 1], c=[SERIES[0] if y else SERIES[1] for y in TY.tolist()], s=3, alpha=0.7, lw=0)
    ax.set(title=title, xticks=[], yticks=[]); ax.grid(False)
ax = fig.add_subplot(gs[3])
ax.imshow(regions, extent=[-6, 6, -6, 6], origin="lower",
          cmap=LinearSegmentedColormap.from_list("e4", [SERIES[2], SERIES[3], SERIES[4], SERIES[6]], N=4), alpha=0.55)
ax.scatter(TX[:, 0], TX[:, 1], c=INK2, s=1, alpha=0.25, lw=0)
ax.set(title="router's first choice", xticks=[], yticks=[]); ax.grid(False)
savefig(fig, "toy_linear_to_moe")
'''),
md(r'''
The left panel is the shape the real experiment must also show: a curve that keeps falling
through the conversion. In the right panel each colour is the expert the router sends a point to
first. Nobody told it where the clusters are; it found them because that lowers the loss.

**Carry this forward:** "convert to MoE" means keep what the dense model knows, add a router,
and let the experts become different by training. Here the dense model is one line. Below it is a
20M-parameter transformer, and the same three steps apply.
'''),
]
