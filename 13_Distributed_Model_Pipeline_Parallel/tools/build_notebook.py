"""Turn nbNN_source.py (percent format) into the matching .ipynb.

The sources are runnable .py files so they can be executed and checked directly; this
script only splits them on the `# %%` markers and wraps each piece as a notebook cell.
Markdown cells are written as `# ` comment lines and unwrapped here.

    python tools/build_notebook.py            # build all three
    python tools/build_notebook.py nb01       # build one
"""
import io
import json
import os
import sys

PAIRS = [
    ("nb01_source.py", "01_reversibility_from_scratch.ipynb"),
    ("nb02_source.py", "02_train_20M_on_50M_tokens.ipynb"),
    ("nb03_source.py", "03_results_and_cost.ipynb"),
]


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


def build(src, out):
    cells = []
    stem = os.path.splitext(os.path.basename(src))[0]
    for i, (kind, body) in enumerate(split_cells(io.open(src, encoding="utf-8").read())):
        # stable ids: nbformat >=4.5 requires them, and deriving them from the source
        # position keeps a rebuilt notebook diffable against the previous one
        cid = f"{stem}-{i:03d}"
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "id": cid, "metadata": {},
                          "source": as_source(unwrap_markdown(body))})
        else:
            cells.append({"cell_type": "code", "id": cid, "metadata": {},
                          "execution_count": None, "outputs": [],
                          "source": as_source(body)})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    io.open(out, "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False))
    return len(cells)


def main(argv):
    wanted = argv[1] if len(argv) > 1 else None
    for src, out in PAIRS:
        if wanted and not src.startswith(wanted):
            continue
        if not os.path.exists(src):
            print(f"  skip {src} (not written yet)")
            continue
        n = build(src, out)
        print(f"  {src} -> {out}  ({n} cells)")


if __name__ == "__main__":
    main(sys.argv)
