"""Cell sources for 14_dense_to_moe.ipynb.

Each module in this package defines CELLS, an ordered list of (kind, source, tags)
tuples. nb_source.py stitches them together in PARTS order. Edit the cells here and
rebuild; never hand-edit the generated notebook's code.
"""
from textwrap import dedent


def md(src, *tags):
    return ("markdown", dedent(src).strip("\n"), list(tags))


def code(src, *tags):
    return ("code", dedent(src).strip("\n"), list(tags))


def mathbox(title, body):
    """A collapsible derivation. Blank lines around the body keep markdown + LaTeX rendering
    inside <details> in Jupyter, Colab, GitHub and nbviewer."""
    return ("markdown", f"<details><summary>📐 <b>The math:</b> {title}</summary>\n\n"
                        f"{dedent(body).strip()}\n\n</details>", [])


PARTS = [
    "p00_front",
    "p01_setup",
    "p02_refresher",
    "p03_toy",
    "p04_blocks",
    "p05_data",
    "p06_dense",
    "p07_surgery",
    "p08_conv_lab",
    "p09_router_lab",
    "p10_stage2",
    "p11_growth",
    "p12_result",
    "p12b_posthoc",
    "p13_experts",
    "p14_systems",
    "p15_wrap",
]
