"""Turn nb_source.py (percent format) into loss_and_heads.ipynb.

The source is a runnable .py so the code can be executed and validated directly;
this script only splits it on the `# %%` markers and wraps each piece as a cell.
Run from the assignment folder:  python tools/build_notebook.py
"""
import io
import json
import os
import sys

SRC = "nb_source.py"
OUT = "loss_and_heads.ipynb"


def split_cells(text):
    cells, kind, buf = [], "code", []

    def flush():
        body = "\n".join(buf).strip("\n")
        if body:
            cells.append((kind, body))

    for line in text.split("\n"):
        if line.startswith("# %% [markdown]"):
            flush(); kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def strip_comment_prefix(body):
    """Markdown cells are written as `# ...` comments; unwrap them."""
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
    """nbformat wants a list of lines, each keeping its trailing newline."""
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main():
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} not found - run this from the assignment folder")

    text = io.open(SRC, encoding="utf-8").read()
    cells = []
    for kind, body in split_cells(text):
        if kind == "markdown":
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": as_source(strip_comment_prefix(body)),
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": as_source(body),
            })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"wrote {OUT}: {len(cells)} cells ({md} markdown, {len(cells)-md} code)")


if __name__ == "__main__":
    main()
