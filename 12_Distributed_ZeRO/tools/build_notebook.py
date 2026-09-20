"""Turn nb_source.py (percent format) into zero_32_virtual_gpus.ipynb.

The source is a runnable .py so it can be executed and checked directly; this script only
splits it on the `# %%` markers and wraps each piece as a notebook cell.
Run from the project folder:  python tools/build_notebook.py
"""
import io
import json
import os
import sys

SRC = "nb_source.py"
OUT = "zero_32_virtual_gpus.ipynb"


def split_cells(text):
    cells, kind, buf = [], "code", []

    def flush():
        body = "\n".join(buf).strip("\n")
        if body:
            cells.append((kind, body))

    for line in text.split("\n"):
        if line.startswith("# %% [markdown]"):
            flush()
            kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush()
            kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def unwrap_markdown(body):
    """Markdown cells are written as `# ...` comment lines; strip the comment prefix."""
    out = []
    for line in body.split("\n"):
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip("\n")


def as_source(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main():
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} not found - run this from the project folder")
    cells = []
    for kind, body in split_cells(io.open(SRC, encoding="utf-8").read()):
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": as_source(unwrap_markdown(body))})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                          "outputs": [], "source": as_source(body)})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    n_md = sum(c["cell_type"] == "markdown" for c in cells)
    print(f"wrote {OUT}: {len(cells)} cells ({n_md} markdown, {len(cells) - n_md} code)")


if __name__ == "__main__":
    main()
