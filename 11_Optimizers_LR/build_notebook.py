"""Assemble optimizers_and_learning_rates.ipynb from nb_parts/p*.py.

Each part is plain Python with cell markers:
    # %%              starts a code cell
    # %% [markdown]   starts a markdown cell; the cell body is one r\"\"\"...\"\"\" string

Because markdown cells are just string literals, the concatenated parts also run as an ordinary
script (`python build_notebook.py --script` writes it to _notebook_as_script.py).

If the notebook already exists with identical cell sources, it is left alone, so an executed
copy keeps its outputs. Pass --force to rewrite it anyway.
"""
import re
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
PARTS = sorted((HERE / "nb_parts").glob("p*.py"))
NB = HERE / "optimizers_and_learning_rates.ipynb"
MARK = re.compile(r"^# %%(.*)$", re.M)


def cells_from(text):
    pieces = MARK.split(text)          # [preamble, tag1, body1, tag2, body2, ...]
    out = []
    for tag, body in zip(pieces[1::2], pieces[2::2]):
        body = body.strip("\n")
        if "[markdown]" in tag:
            m = re.fullmatch(r'\s*r"""\n?(.*?)\n?"""\s*', body, re.S)
            if not m:
                raise SystemExit(f"markdown cell is not a single r-string:\n{body[:200]}")
            out.append(nbformat.v4.new_markdown_cell(m.group(1)))
        elif body.strip():
            out.append(nbformat.v4.new_code_cell(body))
    return out


def build():
    cells = []
    for p in PARTS:
        cells += cells_from(p.read_text(encoding="utf-8"))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
    }
    return nb


def main():
    if "--script" in sys.argv:
        text = "\n\n".join(p.read_text(encoding="utf-8") for p in PARTS)
        (HERE / "_notebook_as_script.py").write_text(text, encoding="utf-8")
        print("wrote _notebook_as_script.py")
        return
    nb = build()
    if NB.exists() and "--force" not in sys.argv:
        old = nbformat.read(NB, as_version=4)
        same = [(c.cell_type, c.source) for c in old.cells] == [(c.cell_type, c.source) for c in nb.cells]
        if same:
            print(f"{NB.name}: cell sources unchanged, kept the existing (executed) copy")
            return
    nbformat.write(nb, NB)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {NB.name}: {len(nb.cells)} cells ({n_code} code) from {len(PARTS)} parts")


if __name__ == "__main__":
    main()
