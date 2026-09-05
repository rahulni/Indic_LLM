"""Strip course-specific framing so the notebook reads as a standalone technical note.

Applies the same replacements to nb_source.py and to the executed notebook. Every edit
is to prose - markdown cells, or comments inside code cells - so none of it changes what
the code does, and the embedded outputs stay valid. That is the only reason this can run
without a full re-execution.

Run from the assignment folder:  python tools/debrand.py
"""
import io
import json
import sys

# (old, new) on bare prose. Multi-line entries get "# " prefixes added automatically for
# the .py form, where markdown lives inside comments.
EDITS = [
    ("""**ERA V5 — Session 9.** One notebook, one loss harness, and one thing you have to get
right by reading rather than by guessing.

The assignment is not "train a model". It is: take these four lines""",
     """Four lines sit between a model's hidden states and the scalar it trains on, and they
are where a surprising amount goes wrong:"""),

    ("""and make them **correct and observable**. Every serious bug that lives in these lines
shares one property: *it does not raise an exception*. A target shift in the wrong
direction produces a beautiful loss curve and a worthless model.

So every section below prints evidence, and most of them assert on it.""",
     """This makes them **correct and observable**, then adds a second output head that predicts
two tokens ahead.

Every serious bug that lives in these lines shares one property: *it does not raise an
exception*. A target shift in the wrong direction produces a beautiful loss curve and a
worthless model. So every section prints evidence, and most of them assert on it."""),

    ("This is the check the whole assignment is built around. You will not catch an",
     "This is the check everything else rests on. You will not catch an"),

    ("The count changing is what the assignment asks for, and the section above delivers it.",
     "The changing count is the mechanical check, and the section above delivers it."),

    ("the session described: the loss falls, and inference emits pad forever.",
     "the classic one: the loss falls, and inference emits pad forever."),

    ("**A nuance the assignment does not ask for but which matters in practice:** the position",
     "**A nuance that matters in practice:** the position"),

    ("""in document B can still read all of document A through the causal mask. The assignment
does not ask for this, but stopping at loss masking leaves a reader believing packing is
solved when half of it is not.""",
     """in document B can still read all of document A through the causal mask. Stopping at loss
masking leaves you believing packing is solved when half of it is not."""),

    ("The session's framing: at production scale (`D = 4096`, `V = 131072`) that same matrix is",
     "At production scale (`D = 4096`, `V = 131072`) that same matrix is"),

    ("""This is what buys long context. The session's framing: a big vocabulary puts so much
pressure on this one tensor that it, not attention, becomes the thing capping your
sequence length.""",
     """This is what buys long context: a big vocabulary puts so much pressure on this one
tensor that it, not attention, becomes the thing capping your sequence length."""),

    ("# wasted. That is a departure from the form the brief shows, so assert the two agree:",
     "# wasted. That differs from the in-sequence slicing form, so assert the two agree:"),

    ("Two practical notes from the session:", "Two practical notes:"),

    ("""> Not required by the assignment. The brief asks for the seven numbers of Part 1 and the
> two losses of Part 2; it references a "Part 3" in its submission list but never defines
> one. This is inferred from the warning attached to it — *a target shift in the incorrect
> direction can produce a beautiful loss curve* — because that warning is worth answering
> with evidence.""",
     """> A demonstration rather than a measurement. The warning is that *a target shift in the
> incorrect direction can produce a beautiful loss curve*, and a warning like that is
> worth answering with evidence rather than agreement."""),

    ("""The assignment's warning: *a target shift in the incorrect direction can produce a
beautiful loss curve*. This section makes that concrete by training the same model three
times, on the same data, from the same seed.""",
     """*A target shift in the incorrect direction can produce a beautiful loss curve.* This
section makes that concrete by training the same model three times, on the same data,
from the same seed."""),

    ("## 11 — The block the session actually described",
     "## 11 — The same harness on a more modern block"),

    ("""Everything above used the plain LayerNorm + GELU block, deliberately: you do not debug a
loss harness and a new architecture at the same time. Now that every check passes, here
is the same model built the way the session described it - **RMSNorm, pre-norm, SwiGLU**
- run through the same gate.""",
     """Everything above used the plain LayerNorm + GELU block, deliberately: you do not debug a
loss harness and a new architecture at the same time. Now that every check passes, here
is the same model built with the components most current models use - **RMSNorm,
pre-norm, SwiGLU** - run through the same gate."""),

    ('# "Run All" silently replaces the submitted numbers with 300-step ones.',
     '# "Run All" silently replaces the committed numbers with 300-step ones.'),

    ("""Everything above used the plain LayerNorm + GELU block, deliberately: you do not debug a
loss harness and a new architecture at the same time.""",
     """Everything above used the plain LayerNorm + GELU block, deliberately: you do not debug a
loss harness and a new architecture at the same time."""),
]


def apply_all(text, as_python_comments):
    hits = 0
    for old, new in EDITS:
        if as_python_comments and "\n" in old:
            o = "\n".join(("# " + l) if l else "#" for l in old.split("\n"))
            n = "\n".join(("# " + l) if l else "#" for l in new.split("\n"))
        else:
            o, n = old, new
        if o in text:
            text = text.replace(o, n)
            hits += 1
    return text, hits


def main():
    src = "nb_source.py"
    s = io.open(src, encoding="utf-8").read()
    s2, n = apply_all(s, as_python_comments=True)
    io.open(src, "w", encoding="utf-8").write(s2)
    print(f"{src}: {n} edits applied")

    nb_path = "loss_and_heads.ipynb"
    nb = json.load(io.open(nb_path, encoding="utf-8"))
    total = 0
    for c in nb["cells"]:
        joined = "".join(c["source"])
        new, n = apply_all(joined, as_python_comments=False)
        if new != joined:
            lines = new.split("\n")
            c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
            total += n
    with io.open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{nb_path}: {total} edits applied (outputs untouched)")

    leftover = []
    for path in (src, nb_path):
        body = io.open(path, encoding="utf-8").read()
        for term in ("ERA V5", "Session 9", "the assignment", "the brief",
                     "the session", "submitted numbers"):
            if term.lower() in body.lower():
                leftover.append(f"{path}: {term}")
    if leftover:
        print("\nstill present:")
        for l in leftover:
            print("  " + l)
        sys.exit(1)
    print("\nno course-specific framing left in either file")


if __name__ == "__main__":
    main()
