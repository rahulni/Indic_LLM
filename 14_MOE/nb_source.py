"""Build 14_dense_to_moe.ipynb from the cell sources in nb_parts/.

    python nb_source.py                    # write the notebook and predictions.json
    python nb_source.py --flat out.py      # all code cells as one script (for debugging)
    python nb_source.py --inject-results   # refresh the results-driven markdown in the EXECUTED
                                           # notebook (outputs untouched) and regenerate README.md
    python nb_source.py --sync-markdown    # copy edited markdown cells into the EXECUTED notebook;
                                           # refuses if any code cell differs (then re-execute)
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
NB = HERE / "14_dense_to_moe.ipynb"
sys.path.insert(0, str(HERE))

from nb_parts import PARTS  # noqa: E402
from nb_parts.predictions import PREDICTIONS  # noqa: E402


def cells():
    out = []
    for name in PARTS:
        mod = importlib.import_module(f"nb_parts.{name}")
        out.extend(mod.CELLS)
    return out


def build():
    nb = nbformat.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
    })
    for kind, src, tags in cells():
        c = nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src)
        if tags:
            c.metadata["tags"] = tags
        nb.cells.append(c)
    nbformat.validate(nb)
    nbformat.write(nb, NB)
    (HERE / "predictions.json").write_text(json.dumps(PREDICTIONS, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {NB.name}: {len(nb.cells)} cells; predictions.json: {len(PREDICTIONS)} predictions")


def sync_markdown():
    nb = nbformat.read(NB, as_version=4)
    src = cells()
    assert len(nb.cells) == len(src), f"cell count differs ({len(nb.cells)} vs {len(src)}): rebuild and re-execute"
    changed = 0
    for i, (c, (kind, s, tags)) in enumerate(zip(nb.cells, src)):
        assert c.cell_type == kind, f"cell {i}: type differs"
        if kind == "code":
            assert c.source == s, f"cell {i}: code differs from nb_parts; re-execute instead"
        elif "glance" not in tags and c.source != s:
            c.source = s
            changed += 1
    nbformat.validate(nb)
    nbformat.write(nb, NB)
    print(f"synced {changed} markdown cells into the executed notebook")


def flat(path):
    lines = []
    for i, (kind, src, _) in enumerate(cells()):
        if kind == "code":
            lines.append(f"# %% [cell {i}]\n{src}\n")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat")
    ap.add_argument("--inject-results", action="store_true")
    ap.add_argument("--sync-markdown", action="store_true")
    a = ap.parse_args()
    if a.flat:
        flat(a.flat)
    elif a.sync_markdown:
        sync_markdown()
    elif a.inject_results:
        from nb_parts.inject import inject
        inject(NB, HERE)
    else:
        build()
