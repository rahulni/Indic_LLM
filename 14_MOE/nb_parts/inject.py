"""Fill the results-driven text from assets/results.json without re-executing anything.

Patches the markdown cell tagged "glance" in the executed notebook (outputs are left untouched)
and regenerates README.md. Refuses quick-mode or incomplete results, so a casual run can never
end up in the submitted text.
"""
import json

import nbformat

RAW = "https://raw.githubusercontent.com/rahulni/Indic_LLM/main/14_MOE/assets"


def load(here):
    R = json.loads((here / "assets" / "results.json").read_text())
    meta = R.get("meta", {})
    if meta.get("mode") != "full" or not meta.get("complete"):
        raise SystemExit(f"refusing to inject: assets/results.json is mode={meta.get('mode')}, complete={meta.get('complete')}")
    return R


def glance(R, img_base):
    P, Pa, p = R["points"], R["params"], R["paired"]
    reg = [v for v in R["predictions"].values() if not v.get("posthoc")]
    post = [v for v in R["predictions"].values() if v.get("posthoc")]
    held, post_held = sum(v["verdict"] == "HELD" for v in reg), sum(v["verdict"] == "HELD" for v in post)
    ph = R.get("posthoc", {})
    post_rows = ""
    if ph:
        post_rows = ("**Post-hoc (Part K.2), from the same T2 checkpoint to T3:**\n\n"
                     "| T3 model | val loss | minus dense control | minus registered MoE-32 |\n|---|---|---|---|\n"
                     + "\n".join(f"| {label} | {ph[k]['val_T3']:.4f} | {ph[k]['vs_control']['mean']:+.4f} "
                                 f"[{ph[k]['vs_control']['lo']:+.4f}, {ph[k]['vs_control']['hi']:+.4f}] | "
                                 f"{ph[k]['vs_moe32']['mean']:+.4f} [{ph[k]['vs_moe32']['lo']:+.4f}, {ph[k]['vs_moe32']['hi']:+.4f}] |"
                                 for k, label in (("moe32_copy", "MoE-32, plain-copy growth"), ("moe8_cont", "MoE-8, no growth")))
                     + "\n\n")
    rows = [
        "| | dense (Stage 1) | MoE-8 (Stage 2) | MoE-32 (Stage 3) | dense control |",
        "|---|---|---|---|---|",
        f"| total / active params | {Pa['dense'] / 1e6:.1f}M / {Pa['dense_active'] / 1e6:.1f}M | {Pa['moe8'] / 1e6:.1f}M / {Pa['moe8_active'] / 1e6:.1f}M "
        f"| {Pa['moe32'] / 1e6:.1f}M / {Pa['moe32_active'] / 1e6:.1f}M | {Pa['dense'] / 1e6:.1f}M / {Pa['dense_active'] / 1e6:.1f}M |",
        f"| val loss at start of stage | - | {P['moe8_T1']:.4f} (T1, 0 steps) | {P['moe32_T2']:.4f} (T2, 0 steps) | {P['dense_T1']:.4f} (T1) |",
        f"| val loss at end of stage | {P['dense_T1']:.4f} (T1) | {P['moe8_T2']:.4f} (T2) | **{P['moe32_T3']:.4f}** (T3) | {P['control_T3']:.4f} (T3) |",
    ]
    checks = "\n".join(f"- {'✅' if c['ok'] else '❌'} {c['item']}: {c['detail']}" for c in R["checklist"])
    m = R["meta"]
    return (
        "## Results at a glance\n\n"
        f"![The MoE path vs the dense control]({img_base}/result_timeline.png)\n\n"
        + "\n".join(rows)
        + f"\n\n**MoE-32 minus the dense control at T3** (same batches, same compute per token): {p['mean']:+.4f} nats, "
        f"95% CI [{p['lo']:+.4f}, {p['hi']:+.4f}] over {p['n']} paired validation batches, single seed "
        "(negative = MoE better). "
        + ("The MoE path beat the dense control." if p["hi"] < 0 else
           "The MoE path kept training and kept improving after both conversions, as required, but it did "
           "**not** beat a dense model given the same tokens; Part K explains where the gap comes from." if p["lo"] > 0 else
           "The difference is not distinguishable from zero.")
        + "\n\n" + post_rows
        + f"**Predictions registered before the run:** {held} of {len(reg)} held"
        + (f"; post-hoc predictions: {post_held} of {len(post)} held" if post else "") + " (scored in Part N).\n\n"
        f"{checks}\n\n"
        f"_Full run on {m.get('gpu')} ({m.get('precision')}, torch {m.get('torch')}), finished {m.get('finished')}._"
    )


def readme(R, body):
    return (
        "# Dense → Mixture-of-Experts\n\n"
        "[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/rahulni/Indic_LLM/blob/main/14_MOE/14_dense_to_moe.ipynb)\n\n"
        "> **Goal.** Train a dense (\"linear\") model, convert it into a mixture-of-experts, and show that the "
        "converted model keeps training and its loss keeps falling, at a size that fits one laptop or Colab GPU.\n\n"
        "A ~19M-parameter transformer is trained on TinyStories, converted into an MoE with 8 experts "
        "(partition upcycling), grown to 32 experts (clone + redraw half + Gumbel top-k), and compared "
        "against a dense control on identical batches. The notebook doubles as a refresher: every number "
        "it relies on is recomputed and asserted, and every mechanism has an exact gate and a lab.\n\n"
        + body.replace("## Results at a glance", "## Results")
        + "\n\n## Run it\n\n"
        "| `MODE` | needs | time |\n|---|---|---|\n"
        "| `learn` | CPU | ~1 min: refresher, toy, gates, replays these results |\n"
        "| `quick` | GPU | ~5 min: every code path at 1/20 scale, writes `assets-quick/` |\n"
        "| `full` | GPU | ~30 min on an RTX 3070 Laptop (measured), ~1 h on a Colab T4 (estimate) |\n\n"
        "The notebook is generated: edit `nb_parts/*.py`, then `python nb_source.py`.\n"
    )


def inject(nb_path, here):
    R = load(here)
    nb = nbformat.read(nb_path, as_version=4)
    n = 0
    for c in nb.cells:
        if "glance" in c.metadata.get("tags", []):
            c.source = glance(R, RAW)
            n += 1
    assert n == 1, f"expected one 'glance' cell, found {n}"
    nbformat.validate(nb)
    nbformat.write(nb, nb_path)
    (here / "README.md").write_text(readme(R, glance(R, "assets")), encoding="utf-8")
    print(f"patched the glance cell in {nb_path.name} and wrote README.md")
