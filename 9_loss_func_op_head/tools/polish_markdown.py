"""Normalise section-header dashes in nb_source.py and, if present, in the executed
notebook's markdown cells.

Markdown cells carry no outputs, so patching them in place after execution is safe -
it does not invalidate the run that produced the code cells' results.
"""
import io
import json
import os
import re

HEADER = re.compile(r"^(#{1,3} (?:Part )?\d+) - ", re.M)


def fix(text):
    return HEADER.sub(lambda m: m.group(1) + " — ", text)


def main():
    src = "nb_source.py"
    if os.path.exists(src):
        s = io.open(src, encoding="utf-8").read()
        # in the .py the markdown lines are comments: "# ## 2 - Title"
        out = re.sub(r"^# (#{1,3} (?:Part )?\d+) - ",
                     lambda m: "# " + m.group(1) + " — ", s, flags=re.M)
        if out != s:
            io.open(src, "w", encoding="utf-8").write(out)
            print(f"{src}: headers normalised")
        else:
            print(f"{src}: nothing to do")

    nb_path = "loss_and_heads.ipynb"
    if os.path.exists(nb_path):
        nb = json.load(io.open(nb_path, encoding="utf-8"))
        changed = 0
        for c in nb["cells"]:
            if c["cell_type"] != "markdown":
                continue
            joined = "".join(c["source"])
            new = fix(joined)
            if new != joined:
                lines = new.split("\n")
                c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
                changed += 1
        if changed:
            with io.open(nb_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
                f.write("\n")
            print(f"{nb_path}: {changed} markdown cells normalised (outputs untouched)")
        else:
            print(f"{nb_path}: nothing to do")


if __name__ == "__main__":
    main()
