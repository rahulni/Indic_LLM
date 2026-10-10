r"""Build inside_the_training_loop.ipynb from nb_source.py.

nb_source.py is a plain Python script split into cells by `# %%` lines, so it can be run
directly (`python nb_source.py`) as well as turned into a notebook:

    # %% [markdown]
    r'''
    ## A heading
    Any markdown, with $\LaTeX$ left alone because the string is raw.
    '''

    # %%
    print("a code cell")

A markdown cell's body is the single raw string that follows its marker; Python treats
that string as a no-op expression, which is what keeps the script runnable.

Usage:  python build_notebook.py            (writes the .ipynb next to this file)
"""
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
SRC = HERE / "nb_source.py"
OUT = HERE / "inside_the_training_loop.ipynb"

MARKER = re.compile(r"^# %%(?P<rest>.*)$")
STRING = re.compile(r'^[rR]?("""|\'\'\')\n?(?P<body>.*?)\n?\1\s*$', re.S)


def split_cells(text):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        m = MARKER.match(line)
        if m:
            if kind is not None:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in m.group("rest") else "code"
            buf = []
        elif kind is not None:
            buf.append(line)
    if kind is not None:
        cells.append((kind, buf))
    return cells


def to_cell(kind, lines):
    body = "\n".join(lines).strip("\n")
    if kind == "code":
        return new_code_cell(body) if body.strip() else None
    m = STRING.match(body.strip())
    if not m:
        sys.exit(f"markdown cell is not a single triple-quoted string:\n{body[:300]}")
    return new_markdown_cell(m.group("body").strip("\n"))


def main():
    cells = [c for c in (to_cell(k, l) for k, l in split_cells(SRC.read_text(encoding="utf-8"))) if c]
    nb = new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
    }
    nbformat.validate(nb)
    # Keep outputs of an already-executed notebook only if the sources are unchanged.
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        if [c.source for c in old.cells] == [c.source for c in nb.cells]:
            print(f"{OUT.name}: sources unchanged, leaving executed copy alone")
            return
    nbformat.write(nb, OUT)
    n_md = sum(c.cell_type == "markdown" for c in cells)
    print(f"wrote {OUT.name}: {len(cells)} cells ({n_md} markdown, {len(cells) - n_md} code)")


if __name__ == "__main__":
    main()
