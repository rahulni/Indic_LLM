"""Markdown-only edits to nb_source.py and the executed notebook.

Markdown cells carry no outputs, so patching them in place after execution is safe - it
does not invalidate the run that produced the code cells' results.

Two jobs:
  1. normalise section-header dashes
  2. relabel Part 3 as an appendix, since the assignment references a "Part 3" in its
     submission list but never defines one - the wrong-shift demo is inferred from the
     warning paragraph, and it should not read as though it were a stated requirement.
"""
import io
import json
import os
import re

HEADER_PY = re.compile(r"^# (#{1,3} (?:Part )?\d+) - ", re.M)
HEADER_MD = re.compile(r"^(#{1,3} (?:Part )?\d+) - ", re.M)

OLD_P3 = "# Part 3 — the beautiful wrong loss curve"
NEW_P3 = """# Appendix — the beautiful wrong loss curve

> Not required by the assignment. The brief asks for the seven numbers of Part 1 and the
> two losses of Part 2; it references a "Part 3" in its submission list but never defines
> one. This is inferred from the warning attached to it — *a target shift in the incorrect
> direction can produce a beautiful loss curve* — because that warning is worth answering
> with evidence."""


def main():
    src = "nb_source.py"
    if os.path.exists(src):
        s = io.open(src, encoding="utf-8").read()
        out = HEADER_PY.sub(lambda m: "# " + m.group(1) + " — ", s)
        out = out.replace(
            "# " + OLD_P3,
            "\n".join("# " + l if l else "#" for l in NEW_P3.split("\n")),
        )
        if out != s:
            io.open(src, "w", encoding="utf-8").write(out)
            print(f"{src}: patched")
        else:
            print(f"{src}: nothing to do")

    nb_path = "loss_and_heads.ipynb"
    if not os.path.exists(nb_path):
        return
    nb = json.load(io.open(nb_path, encoding="utf-8"))
    changed = 0
    for c in nb["cells"]:
        if c["cell_type"] != "markdown":
            continue
        joined = "".join(c["source"])
        new = HEADER_MD.sub(lambda m: m.group(1) + " — ", joined)
        new = new.replace(OLD_P3, NEW_P3)
        if new != joined:
            lines = new.split("\n")
            c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
            changed += 1
    if changed:
        with io.open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"{nb_path}: {changed} markdown cells patched (outputs untouched)")
    else:
        print(f"{nb_path}: nothing to do")


if __name__ == "__main__":
    main()
