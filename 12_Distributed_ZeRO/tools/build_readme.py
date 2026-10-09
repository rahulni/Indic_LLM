"""Generate README.md from the executed notebook's results.

The README is generated and never edited by hand. The prose lives in this file; every
number comes from assets/results.json, which the notebook's last cell writes. The build
refuses to run unless the executed notebook printed the same run stamp as results.json,
so every number in the README comes from the committed notebook's run.

    python tools/build_readme.py            # assets/results.json       -> README.md
    python tools/build_readme.py --quick    # assets/quick/results.json -> README.quick.md
                                            # (a development preview; never touches README.md)

Environment: README_BRANCH (default "main") sets the branch in the Colab, nbviewer and
GitHub links.

Two other sources are read, so nothing is typed twice:
  * zero_theory.py for hardware constants (FLOP/s, bandwidths) and model defaults, the same
    module the notebook used (pure Python, no torch);
  * zero_sim.py for the code excerpts, cut out with `ast`, so they are always the real code.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import zero_theory as zt  # noqa: E402  (pure Python: constants and closed forms only)

REPO = "rahulni/Indic_LLM"
BRANCH = os.environ.get("README_BRANCH", "main")
FOLDER = "12_Distributed_ZeRO"
NOTEBOOK = "zero_32_virtual_gpus.ipynb"
MiB = 2 ** 20
STAMP_RE = re.compile(r"RUN STAMP (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) · quick=(True|False)")
FIGURES = ["what_each_gpu_holds.png", "ring_allreduce.png", "cluster_heatmap.png",
           "memory_timeline.png", "comm_and_compute.png", "scaling_with_N.png",
           "grad_accum_and_ckpt.png", "loss_curves.png", "oom_ladder.png", "real_hardware.png"]
BANNED = re.compile(r"\b(assignment|session|cohort|homework|grader|submission)s?\b", re.I)
LABEL = {0: "DDP (ZeRO-0)", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}
SHORT = {0: "DDP", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}
FORMULA = {0: "16Ψ′", 1: "4Ψ′ + 12Ψ′/N", 2: "2Ψ′ + 14Ψ′/N", 3: "16Ψ′/N"}
OP_SHORT = {"all_reduce": "AR", "reduce_scatter": "RS", "all_gather": "AG"}


# =============================================================================================
# small helpers
# =============================================================================================

def fail(msg: str):
    sys.exit(f"build_readme: ERROR: {msg}")


def warn(msg: str):
    print(f"build_readme: WARNING: {msg}")


def n(x) -> str:
    """Integer with thousands separators."""
    return f"{int(round(x)):,}"


def a_n(num: str) -> str:
    """'a' or 'an' before a number as it is read aloud (an 813,568 / an 11,000 / a 109,312)."""
    lead = num.split(",")[0]
    return "an" if lead.startswith("8") or lead in ("11", "18") else "a"


def mib(b, d=3) -> str:
    return f"{b / MiB:,.{d}f}"


def mb(b, d=2) -> str:
    """Decimal megabytes (10^6), the unit networks are quoted in."""
    return f"{b / 1e6:,.{d}f}"


def times(v) -> str:
    return f"{v:,.1f}×"


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci(v) -> str:
    if v == 0:
        return "0"
    m, e = f"{v:.1e}".split("e")
    return f"{m} × 10{str(int(e)).translate(_SUP)}"


def gbf(v) -> str:
    """GB with a precision that suits the size."""
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def tick(ok: bool) -> str:
    return "✓" if ok else "✗"


def shield(label: str, message: str, color: str, logo: str | None = None) -> str:
    esc = lambda s: quote(s.replace("-", "--").replace("_", "__"), safe="")  # noqa: E731
    url = f"https://img.shields.io/badge/{esc(label)}-{esc(message)}-{color}"
    if logo:
        url += f"?logo={logo}&amp;logoColor=white"
    return url


def details(summary: str, body: str) -> str:
    return f"<details>\n<summary><b>{summary}</b></summary>\n\n{body.strip()}\n\n</details>"


def figure(C, name: str) -> str:
    return f'<p align="center"><img src="{C.asset_dir}/{name}" width="820"></p>'


def table(header: list, rows: list, align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---:" if a == "c" else "---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fits_phrase(fits_with: str) -> str:
    return {"none": "even without recomputation",
            "selective": "with selective recomputation",
            "full": "only with full activation recomputation",
            "no": "not at all, even with full recomputation"}.get(fits_with, fits_with)


# =============================================================================================
# code excerpts, cut from zero_sim.py (so they cannot drift from the engine)
# =============================================================================================

class Source:
    def __init__(self, path: Path):
        self.text = path.read_text(encoding="utf-8")
        self.lines = self.text.splitlines()
        self.tree = ast.parse(self.text)

    def cls(self, name):
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        fail(f"class {name} not found in zero_sim.py; update tools/build_readme.py")

    def excerpt(self, cls_name: str, methods: list[str]) -> tuple[str, int, int]:
        """The class line plus the named methods, verbatim. Returns (code, first line of the
        methods, last line of the methods)."""
        node = self.cls(cls_name)
        by_name = {m.name: m for m in node.body if isinstance(m, ast.FunctionDef)}
        bodies, first, last = [], None, None
        for m in methods:
            if m not in by_name:
                fail(f"{cls_name}.{m} not found in zero_sim.py; update tools/build_readme.py")
            f = by_name[m]
            start = (f.decorator_list[0].lineno if f.decorator_list else f.lineno)
            bodies.append("\n".join(self.lines[start - 1:f.end_lineno]))
            first = start if first is None else min(first, start)
            last = f.end_lineno if last is None else max(last, f.end_lineno)
        return self.lines[node.lineno - 1] + "\n" + "\n\n".join(bodies), first, last

    def default(self, cls_name: str, method: str, arg: str):
        """The default value of a keyword argument, read from the source."""
        for f in self.cls(cls_name).body:
            if isinstance(f, ast.FunctionDef) and f.name == method:
                names = [a.arg for a in f.args.args]
                defs = dict(zip(names[len(names) - len(f.args.defaults):], f.args.defaults))
                if arg in defs:
                    return ast.literal_eval(defs[arg])
        fail(f"default of {cls_name}.{method}({arg}=...) not found in zero_sim.py")


def ddp_single_tolerances(quick: bool):
    """The (loss, weight) bounds asserted for DDP-on-N-GPUs vs one GPU (`assert dl < X and dw < Y`),
    read from the code that produced the results: the executed notebook's own cells for the full
    run (nb_source.py may have moved on since), nb_source.py for a quick run (run as a script)."""
    text = ""
    nb = ROOT / NOTEBOOK
    if not quick and nb.exists():
        cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
        text = "\n".join(_join(c.get("source", "")) for c in cells if c.get("cell_type") == "code")
    elif (ROOT / "nb_source.py").exists():
        text = (ROOT / "nb_source.py").read_text(encoding="utf-8")
    m = re.search(r"assert\s+dl\s*<\s*([0-9.eE+-]+)\s+and\s+dw\s*<\s*([0-9.eE+-]+)", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def module_constant(path: Path, name: str):
    """A module-level constant (plain or tuple assignment), read with ast, not imported."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.literal_eval(node.value)
                if isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for t, v in zip(tgt.elts, node.value.elts):
                        if isinstance(t, ast.Name) and t.id == name:
                            return ast.literal_eval(v)
    return None


# =============================================================================================
# the run-stamp check: the README's numbers must come from the committed notebook's run
# =============================================================================================

def _join(t) -> str:
    return "".join(t) if isinstance(t, list) else str(t)


def notebook_outputs(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    texts, errors, n_code, n_out = [], [], 0, 0
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        n_code += 1
        outs = c.get("outputs") or []
        n_out += bool(outs)
        for o in outs:
            kind = o.get("output_type")
            if kind == "stream":
                texts.append(_join(o.get("text", "")))
            elif kind in ("execute_result", "display_data"):
                texts.append(_join(o.get("data", {}).get("text/plain", "")))
            elif kind == "error":
                errors.append(f"{o.get('ename')}: {o.get('evalue')}")
    return "\n".join(texts), errors, n_code, n_out


def check_run_stamp(R: dict, quick: bool) -> str:
    want = R["run"]["timestamp"]
    path = ROOT / NOTEBOOK
    if not path.exists():
        if quick:
            warn(f"{NOTEBOOK} not found; run-stamp check skipped (--quick only)")
            return "skipped"
        fail(f"{NOTEBOOK} not found; the README must be built from an executed notebook")
    text, errors, n_code, n_out = notebook_outputs(path)
    stamps = STAMP_RE.findall(text)
    if quick:
        quick_stamps = [ts for ts, q in stamps if q == "True"]
        if n_out == 0 or not quick_stamps:
            why = ("has no outputs (not executed, or mid-execution)" if n_out == 0 else
                   "holds no quick-run stamp" + (f" (it holds {stamps})" if stamps else ""))
            warn(f"{NOTEBOOK} {why}; run-stamp check skipped (--quick only)")
            return "skipped"
        if want not in quick_stamps:
            fail(f"notebook quick stamp(s) {quick_stamps} != results.json timestamp {want!r}")
        print(f"build_readme: run stamp OK (quick): RUN STAMP {want}")
        return "ok"
    if errors:
        fail(f"{NOTEBOOK} contains error outputs: {errors[:3]}")
    if n_out < n_code:
        fail(f"{NOTEBOOK}: only {n_out} of {n_code} code cells have outputs; execute it fully")
    if (want, "False") not in stamps:
        fail(f"{NOTEBOOK} does not print 'RUN STAMP {want} · quick=False' (found {stamps or 'none'}). "
             "The results do not come from the committed notebook's run: re-execute it, or "
             "rebuild the README from the notebook that wrote assets/results.json.")
    print(f"build_readme: run stamp OK: RUN STAMP {want} · quick=False")
    return "ok"


# =============================================================================================
# the context: everything the sections need, derived from results.json
# =============================================================================================

class Ctx:
    def __init__(self, R: dict, asset_dir: str, quick: bool, stamp_status: str):
        self.R, self.asset_dir, self.quick, self.stamp_status = R, asset_dir, quick, stamp_status
        for key in ("model", "stages", "run"):
            if key not in R:
                fail(f"results.json has no '{key}' block")
        m = R["model"]
        self.cfg, self.psi, self.P, self.N = m["config"], m["psi"], m["P"], m["world"]
        self.units = m["units"]
        self.U = len(self.units)
        self.S = {int(k): v for k, v in R["stages"].items()}
        if sorted(self.S) != [0, 1, 2, 3]:
            fail(f"results.json has stages {sorted(self.S)}, expected 0-3")
        self.steps = R["run"]["steps"]
        self.run = R["run"]
        self.src = Source(ROOT / "zero_sim.py")
        self.align = module_constant(ROOT / "zero_sim.py", "ALIGN") or fail("ALIGN not found in zero_sim.py")
        self.abbr = [self.unit_abbr(u["name"]) for u in self.units]

    @staticmethod
    def unit_abbr(name: str) -> str:
        if name == "embed":
            return "E"
        if name == "head":
            return "H"
        if name.startswith("block"):
            return "B" + name[5:]
        return name

    def ms(self, s):
        return self.S[s]["model_state_bytes"]

    def comm_formula(self, s):
        return (3 if s == 3 else 2) * (self.N - 1) * 2 * self.P // self.N

    def calls_formula(self, s):
        return {0: self.U, 1: 2 * self.U, 2: 2 * self.U, 3: 3 * self.U}[s]

    def hops_formula(self, s):
        return {0: 2 * self.U, 1: 2 * self.U, 2: 2 * self.U, 3: 3 * self.U}[s] * (self.N - 1)

    def hops(self, s):
        """Ring hops per step excluding the logging-only loss all-reduce, if the counter
        includes it (it does: 2(N-1) hops), so the number compares with the formula."""
        raw = self.S[s]["ring_steps"]
        if raw == self.hops_formula(s):
            return raw, False
        if raw - 2 * (self.N - 1) == self.hops_formula(s):
            return raw - 2 * (self.N - 1), True
        return raw, False

    def opt_formula(self, s):
        return self.P if s == 0 else self.P // self.N

    def fwd_formula(self):
        return self.R.get("compute", {}).get("fwd_flops_formula")

    def code(self, cls_name, methods):
        code, a, b = self.src.excerpt(cls_name, methods)
        link = f"[`zero_sim.py`, lines {a}–{b}](zero_sim.py#L{a}-L{b})"
        return (f"```python\n{code}\n```\n<sub>`{cls_name}.{'`, `'.join(methods)}`, copied verbatim "
                f"from {link} when this README was built.</sub>")

    # --- a step timeline, drawn from the real unit list -------------------------------------
    def lane(self, names, label=None):
        boxes = "".join(f"[{x}]" for x in names)
        if label is None:
            return boxes, None
        return boxes, "".join(label.center(len(x) + 2) for x in names)

    def timeline(self, s, name_col=True):
        fwd, bwd = self.abbr, list(reversed(self.abbr))
        pad = " " * 14
        head = f"{SHORT[s]:<7s}" if name_col else " " * 7
        out = []
        fb, _ = self.lane(fwd)
        bb, _ = self.lane(bwd)
        if s == 3:
            _, ag = self.lane(fwd, "AG")
            out.append(f"{head}forward   {ag}   all-gather each unit just before it runs")
            out.append(f"{pad}   {fb}   release it right after (storage resized to 0)")
            _, ag2 = self.lane(bwd, "AG")
            _, rs = self.lane(bwd, "RS")
            out.append(f"{' ' * 7}backward  {ag2}   gather it again,")
            out.append(f"{pad}   {bb}")
            out.append(f"{pad}   {rs}   reduce-scatter its gradient; free weights and full gradient")
            out.append(f"{' ' * 7}step      AdamW on this GPU's Ψ′/N slice → w_shard. No all-gather: weights stay sharded")
        else:
            out.append(f"{head}forward   {fb}")
            out.append(f"{' ' * 7}backward  {bb}")
            if s == 0:
                _, ar = self.lane(bwd, "AR")
                out.append(f"{pad}   {ar}   all-reduce each unit's gradient as soon as it exists")
                out.append(f"{' ' * 7}step      AdamW on all Ψ′ elements; bf16 weights ← fp32 master")
            elif s == 1:
                out.append(f"{pad}   {'RS × ' + str(self.U) + ' after backward ends (full gradients kept until then)':s}")
                out.append(f"{' ' * 7}step      AdamW on this GPU's Ψ′/N slice → its bf16 slice → AG × {self.U} (weights)")
            else:
                _, rs = self.lane(bwd, "RS")
                out.append(f"{pad}   {rs}   reduce-scatter each unit's gradient at once, then free it")
                out.append(f"{' ' * 7}step      AdamW on this GPU's Ψ′/N slice → its bf16 slice → AG × {self.U} (weights)")
        return "\n".join(out)


# =============================================================================================
# the sections
# =============================================================================================

def sec_header(C: Ctx) -> str:
    colab = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{FOLDER}/{NOTEBOOK}"
    nbv = f"https://nbviewer.org/github/{REPO}/blob/{BRANCH}/{FOLDER}/{NOTEBOOK}"
    gh = f"https://github.com/{REPO}/tree/{BRANCH}/{FOLDER}"
    torch_v = C.run.get("torch", "?")
    run_msg = f"{C.N} virtual GPUs · {C.steps} steps"
    banner = ""
    if C.quick:
        banner = ("\n> [!WARNING]\n> **Development preview built from a QUICK run** "
                  f"(`assets/quick/results.json`, a smaller model, {C.steps} steps). "
                  "It is not the committed README; `python tools/build_readme.py` builds that "
                  "from the full run.\n")
    return f"""<!-- Generated by tools/build_readme.py from {C.asset_dir}/results.json (RUN STAMP {C.run['timestamp']}). Do not edit by hand: edit the builder and re-run it. -->

<h1 align="center">ZeRO on 32 Virtual GPUs</h1>

<p align="center"><i>One small GPT trained four ways (DDP, ZeRO-1, ZeRO-2, ZeRO-3) on {C.N} virtual GPUs,<br>
with every byte each GPU holds, sends and computes measured and checked against the ZeRO paper.</i></p>

<p align="center">
  <a href="{colab}"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <a href="{nbv}"><img src="{shield('render', 'nbviewer', 'f37726', 'jupyter')}" alt="Render in nbviewer"></a>
  <a href="{gh}"><img src="{shield('view on', 'GitHub', '181717', 'github')}" alt="View on GitHub"></a>
  <img src="{shield('PyTorch', torch_v, 'ee4c2c', 'pytorch')}" alt="PyTorch {torch_v}">
  <img src="{shield('run', run_msg, '2ea44f')}" alt="{run_msg}">
</p>
{banner}
## Open it

**[Open in Colab]({colab}) · [Render in nbviewer]({nbv}) · [View on GitHub]({gh})**

The notebook [`{NOTEBOOK}`]({NOTEBOOK}) is committed with all its outputs and figures, so
nbviewer and GitHub show the complete run without executing anything. This README is
generated from that run: [`tools/build_readme.py`](tools/build_readme.py) reads
`assets/results.json` and refuses to build unless the notebook's printed run stamp
(`RUN STAMP {C.run['timestamp']}`) matches it. On Colab the notebook fetches its three engine
modules from this folder and starts in quick mode (a smaller model, a few minutes on 2 vCPUs);
set `QUICK_RUN = False` to reproduce the numbers below."""


def sec_tldr(C: Ctx) -> str:
    R, N, P = C.R, C.N, C.P
    ms = {s: C.ms(s) for s in range(4)}
    red = {s: ms[0] / ms[s] for s in range(4)}
    comm = {s: C.S[s]["comm_bytes"] for s in range(4)}
    same_comm = comm[0] == comm[1] == comm[2]
    fl = C.S[0]["flops"]
    flops_same = all(C.S[s]["flops"].get("forward") == fl.get("forward") and
                     C.S[s]["flops"].get("backward") == fl.get("backward") for s in range(4))
    eq = R.get("equivalence", {})
    bit = eq.get("bit_identical_losses") and eq.get("bit_identical_weights")
    lines = [
        f"- **Same training, bit for bit.** {N} virtual GPUs (threads in one process) train "
        f"{a_n(n(C.psi))} {n(C.psi)}-parameter GPT for {C.steps} steps under DDP, ZeRO-1, ZeRO-2 and ZeRO-3. "
        + ("With one micro-batch per optimizer step (G = 1), losses and final weights are bit-identical "
           "across all four (`torch.equal`, not `allclose`)."
           if bit else "**The four stages did not produce bit-identical results in this run.**"),
        f"- **Memory follows the paper to the byte.** Model states per GPU: {mib(ms[0], 2)} → "
        f"{mib(ms[1], 2)} → {mib(ms[2], 2)} → {mib(ms[3], 2)} MiB ({times(red[1])}, {times(red[2])} "
        f"and {times(red[3])} smaller than DDP). Each equals its formula (16Ψ′, 4Ψ′ + 12Ψ′/N, "
        f"2Ψ′ + 14Ψ′/N, 16Ψ′/N) exactly, on every one of the {N} GPUs.",
        f"- **The price is communication, not compute.** Each GPU sends {mb(comm[0])} MB per step "
        + ("under DDP, ZeRO-1 and ZeRO-2 alike" if same_comm else "under DDP")
        + f", and {mb(comm[3])} MB ({comm[3] / comm[0]:.1f}×) under ZeRO-3. Forward and backward FLOPs are "
        + ("identical in all four stages" if flops_same else "NOT identical across stages (see below)")
        + f"; optimizer work per GPU falls from {n(P)} to {n(P // N)} elements.",
    ]
    stress = []
    oom = R.get("oom")
    if oom:
        stress.append(f"{oom['matches']} of {oom['total']} out-of-memory outcomes were predicted "
                      "correctly before running")
    gm = R.get("gpu_mode", {})
    if gm.get("stages"):
        ok = all(0 <= a["cuda_bytes"] - a["ledger_bytes"] <= a["rounding_allowance"]
                 for a in gm["stages"].values())
        stress.append("the ledger matched the real CUDA allocator on all "
                      f"{N} ranks within its 512-byte rounding" if ok else
                      "the ledger did NOT match the CUDA allocator (see below)")
    g = R.get("gloo", {})
    if g.get("status") == "ok":
        how = ("bit for bit" if g.get("bit_exact_weights") else
               f"to max |Δw| = {sci(g['max_abs_weight_diff'])}")
        stress.append(f"the unchanged ZeRO-1 engine on real `torch.distributed` (gloo, "
                      f"{g['world']} processes) matched the simulator {how}")
    if stress:
        lines.append("- **Predictions hold under stress.** " + "; ".join(stress) + ".")
    rh = R.get("real_hardware", {}).get("A100-80GB")
    ot = R.get("overlap_tokens", {}).get("A100-80GB/per_nic")
    if rh:
        seventy = {r["variant"]: r for r in rh if r["model"] == "LLaMA-2 70B"}
        if "ZeRO-3" in seventy and "ZeRO-2" in seventy:
            z3 = seventy["ZeRO-3"]
            line = (f"- **At scale.** On 32 × A100-80GB, LLaMA-2 70B needs {gbf(seventy['ZeRO-2']['states_gb'])} GB "
                    f"per GPU of model states under ZeRO-2 and {gbf(z3['states_gb'])} GB under ZeRO-3. "
                    f"Only ZeRO-3 fits, and {fits_phrase(z3['fits_with'])}.")
            if ot:
                nic = zt.HARDWARE["A100-80GB"]["inter_node_bw"] / 1e9
                line += (f" Communication can hide behind compute above about {n(ot['0'])} tokens per GPU "
                         f"per step ({n(ot['3'])} for ZeRO-3) with one {nic:.0f} GB/s NIC per GPU, "
                         "whatever the model size.")
            lines.append(line)
    return "## TL;DR\n\n" + "\n".join(lines)


def sec_picture(C: Ctx) -> str:
    ms = {s: C.ms(s) for s in range(4)}
    ga = C.R.get("grad_accum", {})
    g_txt = f"1 (and {ga['G']} in one experiment)" if ga.get("G") else "1"
    return f"""## ZeRO in one picture

Adam in mixed precision holds **16 bytes per parameter** before a single activation is
stored. Data parallelism (DDP) keeps all 16 on *every* GPU, so adding GPUs adds throughput
but never room. ZeRO (the Zero Redundancy Optimizer) removes that replication one model
state at a time:

```text
                    GPU0     GPU1     GPU2     GPU3        (■ = holds that quarter, · = does not)
DDP (ZeRO-0)  W    ■■■■     ■■■■     ■■■■     ■■■■     everything replicated 4×
              G    ■■■■     ■■■■     ■■■■     ■■■■
              OS   ■■■■     ■■■■     ■■■■     ■■■■
ZeRO-1        W    ■■■■     ■■■■     ■■■■     ■■■■
              G    ■■■■     ■■■■     ■■■■     ■■■■
              OS   ■···     ·■··     ··■·     ···■     ← optimizer states split
ZeRO-2        W    ■■■■     ■■■■     ■■■■     ■■■■
              G    ■···     ·■··     ··■·     ···■     ← + gradients split
              OS   ■···     ·■··     ··■·     ···■
ZeRO-3        W    ■···     ·■··     ··■·     ···■     ← + weights split
              G    ■···     ·■··     ··■·     ···■       (gathered just in time,
              OS   ■···     ·■··     ··■·     ···■        freed right after use)
W = weights (2Ψ)   G = gradients (2Ψ)   OS = optimizer states: fp32 master + Adam m + v (12Ψ)
```

How much each GPU holds (N = 4, one block ≈ 0.5Ψ bytes):

```text
             weights 2Ψ   grads 2Ψ   optimizer states 12Ψ           per GPU
DDP          ████         ▓▓▓▓       ░░░░░░░░░░░░░░░░░░░░░░░░        16.0 Ψ
ZeRO-1       ████         ▓▓▓▓       ░░░░░░                           7.0 Ψ   = 4Ψ + 12Ψ/N
ZeRO-2       ████         ▓          ░░░░░░                           5.5 Ψ   = 2Ψ + 14Ψ/N
ZeRO-3       █            ▓          ░░░░░░                           4.0 Ψ   = 16Ψ/N
```

The same picture, **measured** on GPU 0 of the {C.N} in this run at the end of backward (the
moment the formulas describe), was identical on all {C.N} GPUs:

{figure(C, "what_each_gpu_holds.png")}

**What to look for:** the bars shrink stage by stage and end exactly on the white formula
ticks: {mib(ms[0])} → {mib(ms[1])} → {mib(ms[2])} → {mib(ms[3])} MiB. The optimizer states
(green) are 12 of the 16 bytes, which is why sharding them first (ZeRO-1) already gives
{times(ms[0] / ms[1])}. The diamonds are the *peak*, which also counts activations and
temporary buffers. ZeRO shards neither, and at this toy size they dominate ZeRO-3's peak.

> **An analogy for revision.** {C.N} students prepare for one exam. Under DDP every student
> photocopies the whole textbook, notes and answer key. Under ZeRO-1 each keeps 1/{C.N} of the
> answer key, the biggest pile. Under ZeRO-2 each also keeps 1/{C.N} of the notes. Under ZeRO-3
> each keeps 1/{C.N} of the textbook too: before reading chapter k, everyone briefly borrows the
> rest of it, then hands it back. The price is passing pages around (communication), not
> extra thinking (compute).

{details("Notation used below", f'''
| Symbol | Meaning | In this run |
|---|---|---:|
| Ψ | number of model parameters | {n(C.psi)} |
| Ψ′ | Ψ with each unit padded to a multiple of N·{C.align} elements (what the buffers really hold) | {n(C.P)} |
| N | data-parallel degree (GPUs) | {C.N} |
| K | optimizer bytes per parameter (fp32 master + Adam m + v) | 12 |
| G | gradient-accumulation steps (micro-batches per optimizer step) | {g_txt} |
| unit | the granularity ZeRO shards and gathers at (FSDP: a wrapped module) | {C.U} units |
| AR, RS, AG | all-reduce, reduce-scatter, all-gather | |
| MiB / MB / GB | 2²⁰ bytes / 10⁶ bytes / 10⁹ bytes | |

The paper counts communication in **elements**: "2Ψ" means 2Ψ elements per GPU per step,
which in bf16 is 4Ψ bytes. The measured byte counts below are exact ring bytes,
2 · (N−1)/N · 2Ψ′ for DDP.
''')}"""


def sec_memory_bill(C: Ctx) -> str:
    f1 = C.R.get("paper_figure1_gb")
    fig1 = ""
    if f1:
        fig1 = (f"\n\n**Checked against the paper.** For its Figure 1 (Ψ = 7.5B, N = 64, K = 12) the "
                f"ZeRO paper prints {' / '.join(f'{round(v, 1):g}' for v in f1)} GB per GPU. The formulas "
                f"above give {' / '.join(f'{v:g}' for v in f1)} GB, and the notebook asserts that they "
                "round to the paper's figures. A 7.5B model that needs "
                f"{f1[0]:g} GB per GPU under DDP needs under {math.ceil(f1[3])} GB under ZeRO-3.")
    ex = C.src.excerpt("DDP", ["step"])
    return f"""## The memory bill: 16 bytes per parameter

| What | dtype | Bytes per parameter | Sharded from |
|---|---|---:|---|
| weights (used by forward and backward) | bf16 | 2 | ZeRO-3 |
| gradients | bf16 | 2 | ZeRO-2 |
| master weights (what the optimizer updates) | fp32 | 4 | ZeRO-1 |
| Adam first moment m | fp32 | 4 | ZeRO-1 |
| Adam second moment v | fp32 | 4 | ZeRO-1 |
| **total** | | **16** | |

With b<sub>w</sub>, b<sub>g</sub> bytes per weight and gradient element and K bytes of
optimizer state per parameter, each GPU holds:

| Stage | Weights | Gradients | Optimizer | Per GPU | bf16 + Adam (b<sub>w</sub> = b<sub>g</sub> = 2, K = 12) |
|---|---|---|---|---|---|
| DDP (ZeRO-0) | b<sub>w</sub>Ψ | b<sub>g</sub>Ψ | KΨ | (b<sub>w</sub> + b<sub>g</sub> + K)Ψ | 16Ψ |
| ZeRO-1 | b<sub>w</sub>Ψ | b<sub>g</sub>Ψ | KΨ/N | (b<sub>w</sub> + b<sub>g</sub>)Ψ + KΨ/N | 4Ψ + 12Ψ/N |
| ZeRO-2 | b<sub>w</sub>Ψ | b<sub>g</sub>Ψ/N | KΨ/N | b<sub>w</sub>Ψ + (b<sub>g</sub> + K)Ψ/N | 2Ψ + 14Ψ/N |
| ZeRO-3 | b<sub>w</sub>Ψ/N | b<sub>g</sub>Ψ/N | KΨ/N | (b<sub>w</sub> + b<sub>g</sub> + K)Ψ/N | 16Ψ/N |

The rule behind the whole table: stage ≥ 1 divides the optimizer by N, stage ≥ 2 the
gradients, stage ≥ 3 the weights. As N grows, ZeRO-1 approaches a floor of 4Ψ and ZeRO-2 a
floor of 2Ψ (the parts that are never sharded); only ZeRO-3 keeps falling as 1/N.{fig1}

{details("Why an fp32 master copy exists (and why it is the biggest item)", f'''
bf16 keeps 8 significant bits (7 stored, plus the implicit leading 1). Near 1.0 the gap between neighbouring bf16 numbers is
2⁻⁷ ≈ 0.0078, so any update smaller than half of that (≈ 0.0039) rounds away:
**1.0 + 0.001 = 1.0 in bf16.** Adam's step is roughly the learning rate per element (the
update m̂/√v̂ is of order 1), and learning rates are 10⁻⁴ to 10⁻³, so applied directly to bf16
weights most updates would vanish and training would stall.

So the optimizer updates an **fp32 master copy** (24 significant bits, gap ≈ 1.2 × 10⁻⁷ near 1.0),
and the bf16 weights used by forward and backward are re-derived from it after every step by
rounding. Adam's m and v stay in fp32 for the same reason: they are running averages whose
per-step increments, such as (1 − β₂)·g², are tiny next to the running value.

Mixed precision therefore saves memory on weights, gradients and activations, and gains matmul
throughput. It does not save optimizer memory, which stays at 12 of the 16 bytes. That is why
ZeRO-1 shards the optimizer states first. In this engine the step is literally:

```python
{ex[0]}
```

(fp32 training has no master copy: 4 + 4 + 8 = 16 bytes per parameter again. Keeping fp32
gradient buffers, as DeepSpeed's bf16 path and Megatron-LM do, makes b<sub>g</sub> = 4 and DDP 18Ψ.)
''')}"""


def sec_machine(C: Ctx) -> str:
    R = C.R
    g = R.get("gloo", {})
    procs = ""
    if g.get("status") == "ok" and g.get("worker_peak_mb"):
        pk = max(g["worker_peak_mb"])
        procs = (f" In this run each gloo worker process (one PyTorch each) peaked at {pk:,.0f} MB, "
                 f"so {C.N} of them would need about {pk * C.N / 1000:,.1f} GB before training started.")
    hello = R.get("hello_collectives")
    hello_txt = ""
    if hello:
        S = C.N * 1024 * 4
        rows = []
        exp = {"all_reduce": 2 * (C.N - 1) / C.N * S, "reduce_scatter": (C.N - 1) / C.N * S,
               "all_gather": (C.N - 1) / C.N * S}
        form = {"all_reduce": "2(N−1)/N · S", "reduce_scatter": "(N−1)/N · S", "all_gather": "(N−1)/N · S"}
        for op, v in hello["bytes"].items():
            rows.append([f"`{op}`", n(v), form.get(op, "?"), n(exp.get(op, float('nan'))), tick(v == exp.get(op))])
        hello_txt = f"""
**"Hello, collectives."** Every GPU contributes a tensor filled with its own rank. After an
all-reduce every GPU must hold 0 + 1 + … + {C.N - 1} = {hello['all_reduce_value']}; after a
reduce-scatter each GPU must hold only its slice of that sum; after an all-gather every GPU
must hold every rank's slice in rank order. All three are asserted, and the bytes GPU 0 was
charged for a message of S = {n(S)} bytes ({S // 1024} KiB) are exactly the ring formulas:

{table(["Collective", "Bytes sent by GPU 0", "Ring formula", "Formula value", ""], rows, "lrlrc")}
"""
    oom_demo = R.get("oom_message_demo")
    oom_txt = ""
    if oom_demo:
        oom_txt = f"""
A capacity turns the ledger into a real limit. On a 1 MiB virtual GPU holding 0.75 MiB, a
0.5 MiB gradient allocation fails *before* the tensor is created, with a CUDA-style message:

```text
{oom_demo}
```
"""
    return f"""## How 32 virtual GPUs are built

```text
┌──────────────────────────── one Python process ─────────────────────────────┐
│  VirtualCluster(world=32).run(train_step)        ≈  torchrun --nproc 32     │
│                                                                             │
│    thread 0            thread 1                          thread 31          │
│   ┌────────────┐      ┌────────────┐                   ┌────────────┐       │
│   │ vGPU 0     │      │ vGPU 1     │        ...        │ vGPU 31    │       │
│   │ • ledger   │      │ • ledger   │                   │ • ledger   │       │
│   │ • capacity │      │ • capacity │                   │ • capacity │       │
│   │ • shard 0  │      │ • shard 1  │                   │ • shard 31 │       │
│   │ • data 0   │      │ • data 1   │                   │ • data 31  │       │
│   └─────┬──────┘      └─────┬──────┘                   └─────┬──────┘       │
│         └───────────────────┴──────────────┬─────────────────┘              │
│                    ThreadComm  (barriers + shared slots)                    │
│        all_reduce · reduce_scatter · all_gather · broadcast · barrier       │
│      charges each GPU's bytes with the ring model: 2(N−1)/N · S, etc.       │
└─────────────────────────────────────────────────────────────────────────────┘
 Tensors live on:  CPU (default, reproducible anywhere)  or  the real GPU (GPU mode)
```

**Why threads, not processes.** {C.N} processes would each load their own copy of PyTorch
before holding a single weight, which does not fit on a laptop or a free Colab runtime.{procs}
Threads share one copy of PyTorch, and PyTorch releases Python's global interpreter lock
inside its kernels, so {C.N} threads really do run {C.N} forward and backward passes. Each run pins
PyTorch to one intra-op thread per virtual GPU and restores the setting afterwards.

Three pieces of [`zero_sim.py`](zero_sim.py) make a cluster:

- **`VirtualGPU`: a memory ledger.** Every buffer the algorithm holds is booked by name
  and category (`weights`, `grads`, `master`, `adam_m`, `adam_v`, `activations`, `temp`).
  The ledger tracks the current total, the peak, the category mix at the peak, named
  snapshots (`end_of_backward`) and an optional event timeline. It is charged *before* a
  tensor is created, so with a capacity set an allocation that would not fit raises
  `VirtualOOMError` and never allocates. Activations are booked automatically: a
  `saved_tensors_hooks` pair charges every tensor autograd saves for backward (once per
  storage, skipping weight storages already on the books) and releases it when autograd drops it.
- **`ThreadComm`: collectives.** Every GPU calls the same collective with its own tensor
  (SPMD, like `torch.distributed`). Inputs are published in shared slots, and a barrier
  waits for every GPU. Each GPU then reads what it needs, and a second barrier keeps anyone
  from reusing a buffer early. Reductions sum in rank order in fp32, then divide by N.
  `all_reduce` is implemented as reduce-scatter + all-gather, so all four stages sum
  gradients through one code path. Bytes are charged per GPU with the ring cost model.
- **`VirtualCluster`: the launcher.** It runs `fn(rank, gpu, comm)` on N daemon threads. If
  any GPU raises (an OOM, say), every barrier is aborted so nobody hangs, and the
  lowest-rank error is re-raised, so failures are deterministic.

**The storage-resize trick (how ZeRO-3 frees weights).** Each unit's parameters are
`nn.Parameter` *views* into one flat buffer. To free the weights, the buffer's storage is
shrunk in place, `flat_w.untyped_storage().resize_(0)`; the tensor objects, and the
references autograd saved for backward, stay alive but point at zero bytes. To use the unit
again, the storage is regrown and refilled by an all-gather, and every view sees the right
numbers. This is what PyTorch FSDP does. Reading a freed weight would crash the process rather
than raise, so a forward pre-hook on every unit raises a readable error first.

**CPU mode and GPU mode.** By default every virtual GPU's tensors live in ordinary CPU memory,
which is reproducible anywhere. In GPU mode all 32 virtual GPUs put their tensors on the one real
GPU, so the ledger can be audited against PyTorch's CUDA allocator (see
[Stress tests](#stress-tests)).
{hello_txt}{oom_txt}
{details("Why four different algorithms can produce bit-identical results", f'''
Floating-point addition is not associative, so "the same maths" is not enough for bitwise
equality. The engine removes every source of divergence:

- **One reduction path.** Every gradient sum goes through `reduce_scatter`: slices are added
  in rank order, in fp32, divided by N, then cast to bf16. DDP's `all_reduce` is literally
  `reduce_scatter` followed by `all_gather`.
- **An elementwise optimizer.** AdamW is written with plain `mul_`, `add_`, `div_` and `sqrt_`,
  never a fused kernel whose vectorized tail might round differently. Updating a 1/N slice
  therefore produces exactly the same numbers as updating the whole vector.
- **One flat layout everywhere.** Units are padded to multiples of N·{C.align} elements, the same in
  every stage, so shard boundaries fall on SIMD-aligned offsets.
- **Gradients written in place.** Each `p.grad` is pre-set to a view into the unit's flat
  gradient buffer, and after backward the engine asserts that autograd did not replace it.
- **No randomness in the step.** No dropout; data batches are a pure function of (seed, step).
- **One micro-batch per step.** With G = 1 every stage adds the same numbers in the same order.
  Under gradient accumulation the order differs: ZeRO-2/3 reduce-scatter every micro-batch and add
  the reduced slices, while DDP and ZeRO-1 add the micro-batches locally and reduce once. Addition
  is not associative, so the last bits differ, though the maths is the same.
''')}"""


def sec_collectives(C: Ctx) -> str:
    ring = C.R.get("ring")
    ring_txt = ""
    if ring:
        ok = ring["sent_per_gpu"] == ring["formula"]
        ring_txt = (f"\n**Evidence.** The hand-written ring (`ring_all_reduce`, neighbour-to-neighbour "
                    f"messages only) run at N = {ring['N']} on S = {n(ring['S_bytes'])} bytes: every GPU sent "
                    f"{n(ring['sent_per_gpu'])} bytes, and 2(N−1)/N · S = {n(ring['formula'])} {tick(ok)}. Its "
                    "result equals the library all-reduce on every GPU.")
    return f"""## Collectives, and why the ring costs 2(N−1)/N

ZeRO is built from three collectives (4 GPUs, each starting with a vector of 4 chunks):

```text
 REDUCE-SCATTER                     ALL-GATHER                       ALL-REDUCE  =  RS then AG
 in   r0 [a0 a1 a2 a3]              in   r0 [x0]                     in   r0 [a0 a1 a2 a3]
      r1 [b0 b1 b2 b3]                   r1 [x1]                          r1 [b0 b1 b2 b3]
      r2 [c0 c1 c2 c3]                   r2 [x2]                          r2 [c0 c1 c2 c3]
      r3 [d0 d1 d2 d3]                   r3 [x3]                          r3 [d0 d1 d2 d3]
 out  r0 [Σ0] r1 [Σ1]               out  every rank                  out  every rank
      r2 [Σ2] r3 [Σ3]                    [x0 x1 x2 x3]                    [Σ0 Σ1 Σ2 Σ3]
 Σk = ak+bk+ck+dk
 bytes sent per GPU (ring):  (N−1)/N · S        (N−1)/N · S                 2(N−1)/N · S
```

| Collective | Each GPU ends with | Bytes sent per GPU (ring) | Used by |
|---|---|---:|---|
| all-reduce | the full sum | 2(N−1)/N · S | DDP gradient sync |
| reduce-scatter | its 1/N slice of the sum | (N−1)/N · S | ZeRO-1/2/3 gradient sync |
| all-gather | every GPU's slice, concatenated | (N−1)/N · S | ZeRO-1/2 weights after the step; ZeRO-3 weights before each use |

**The key fact.** A reduce-scatter followed by an all-gather moves exactly the bytes of one
all-reduce, because that is how the bandwidth-optimal ring *implements* all-reduce. ZeRO-1
and ZeRO-2 do the same two halves as DDP: the reduce-scatter on gradients, then the
all-gather on updated weights instead of on gradients. They therefore cost DDP's
communication. Only ZeRO-3, which must also gather weights before they are used, pays more.

{figure(C, "ring_allreduce.png")}

**What to look for** (N = 4; each cell counts how many GPUs' data that chunk holds): the
reduce-scatter steps (RS 1–3) grow one chunk per GPU to 4. GPU r ends up owning the finished
chunk (r + 1) mod N, a shifted diagonal. Which chunk lands where is a convention of this schedule;
any mapping that gives each GPU exactly one chunk works. The all-gather steps (AG 1–3) copy the
finished chunks around the ring until every cell is 4.
{ring_txt}

{details("Derivation: why a ring all-reduce sends 2(N−1)/N · S bytes per GPU", '''
Put the N GPUs in a ring: GPU r sends only to r + 1 and receives only from r − 1. Cut the
S-byte buffer into N chunks of S/N bytes.

1. **Reduce-scatter, N − 1 steps.** In step k, GPU r sends chunk (r − k) mod N to its right
   neighbour, and adds the chunk (r − k − 1) mod N arriving from its left into its own copy.
   A partial sum travels around the ring, picking up one contribution per hop. After
   N − 1 hops, GPU r holds the complete sum of chunk (r + 1) mod N (the (r + 1) is just this
   schedule's convention). Each GPU has sent N − 1
   chunks: **(N − 1)/N · S bytes.**
2. **All-gather, N − 1 steps.** Each finished chunk travels around the ring once more,
   overwriting stale copies: another **(N − 1)/N · S bytes.**

Total per GPU: **2(N − 1)/N · S.** With α the latency per message and β the bandwidth per
link, the time is 2(N − 1)·α + 2(N − 1)/N · S/β. The bandwidth term tends to 2S/β as N
grows, so adding GPUs does not make each GPU send more. Patarasuk & Yuan (2009) prove this
volume is a lower bound for all-reduce, so the ring is bandwidth-optimal. Its weakness is the
latency term, which grows linearly with N; tree and hierarchical algorithms trade some
bandwidth for fewer hops.

**In ZeRO's units.** The paper counts elements and drops the (N − 1)/N factor. DDP's
all-reduce of Ψ gradient elements is "2Ψ", ZeRO-1/2's reduce-scatter plus all-gather is
Ψ + Ψ = "2Ψ", and ZeRO-3's two weight gathers plus one reduce-scatter are "3Ψ". In bytes, with
2-byte bf16 elements, DDP sends 2 · (N − 1)/N · 2Ψ′, exactly what the counters below report.
''')}"""


def sec_model(C: Ctx) -> str:
    cfg = C.cfg
    rows = [[f"`{u['name']}`", n(u["numel"]), n(u["padded"]), n(u["shard"])] for u in C.units]
    rows.append(["**total**", f"**Ψ = {n(C.psi)}**", f"**Ψ′ = {n(C.P)}**", f"**{n(C.P // C.N)}**"])
    pad = (C.P - C.psi) / C.psi
    return f"""## The model, cut into units

A nanoGPT-style decoder trained on tiny Shakespeare, one character per token:
d = {cfg['n_embd']}, {cfg['n_layer']} layers, {cfg['n_head']} heads, context {cfg['block_size']},
vocabulary {cfg['vocab_size']}, untied input and output embeddings, bf16 weights with an fp32
master copy. Attention is written as explicit matrix multiplies: PyTorch's fused CPU attention
kernel is invisible to `FlopCounterMode` and would silently count as 0 FLOPs.

ZeRO shards at the granularity of **units** (FSDP calls them wrapped modules). Each unit's
parameters are flattened into one buffer, padded to a multiple of N·{C.align} elements, and split
into N equal contiguous slices; GPU r owns slice r.

{table(["Unit", "Parameters", f"Padded (multiple of {C.N}·{C.align})", "Slice per GPU"], rows)}

Padding costs {pad:.1%}. The padded size Ψ′ is what the buffers, and therefore the exact formulas,
contain.

```text
 block0 weights ──flatten──► [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■·pad·]   one flat buffer
                              │ GPU0 │ GPU1 │ GPU2 │  ...          │ GPU31 │   one contiguous slice each

 data: global batch = {C.N} sequences × {cfg['block_size']} characters ──► GPU r trains on sequence r
       (data parallel: same model, different data, gradients averaged over the {C.N} GPUs)
```

A shard is a **contiguous slice of a flat buffer**, not a set of whole tensors. That is
what makes reduce-scatter and all-gather map one-to-one onto it, and what makes
consolidating a checkpoint a concatenation."""


def sec_stages(C: Ctx) -> str:
    N, P, U = C.N, C.P, C.U
    overview = "\n".join(C.timeline(s) + ("\n" if s < 3 else "") for s in range(4))
    legend = ", ".join(f"{a} = {u['name']}" for a, u in zip(C.abbr, C.units))
    executor = C.code("Engine", ["_forward_backward"])

    def comm_line(s):
        st = C.S[s]
        by = " + ".join(f"{OP_SHORT.get(op, op)} {n(v)}" for op, v in st["comm_by_op"].items())
        ok = st["comm_bytes"] == C.comm_formula(s) and st["comm_calls"] == C.calls_formula(s)
        return (f"{n(st['comm_bytes'])} B per GPU per step ({by}) in {st['comm_calls']} collective "
                f"calls, {'both equal to' if ok else 'NOT equal to'} the formula {tick(ok)}")

    def mem_line(s):
        st = C.S[s]
        ok = st["model_state_bytes"] == st["formula_bytes"]
        return (f"{FORMULA[s]} = {n(st['formula_bytes'])} B; the ledger holds "
                f"{n(st['model_state_bytes'])} B ({mib(st['model_state_bytes'])} MiB) {tick(ok)}")

    ddp = f"""### DDP (ZeRO-0): everyone holds everything

**Sharded:** nothing. Every GPU holds all weights, all gradients and the whole optimizer.

```text
{C.timeline(0, name_col=False)}
```

As backward finishes each unit, that unit's gradient is all-reduced (averaged over the {N}
GPUs) before backward moves on. This is the per-bucket schedule PyTorch DDP uses to overlap
communication with the rest of backward; here the collective blocks, so nothing overlaps.
Every GPU then runs the identical AdamW update on all Ψ′ elements, which is N-fold redundant
work, and rounds its fp32 master into its bf16 weights. The run asserts that the {N} replicas never
drift apart. (`setup` allocates the full weights and the full optimizer; `begin_backward`
allocates the full gradient buffer.)

{C.code("DDP", ["after_backward", "step"])}

- **Memory:** {mem_line(0)}.
- **Communication:** {comm_line(0)}. That is one all-reduce per unit, 2 · (N−1)/N · 2Ψ′ bytes in total."""

    z1 = f"""### ZeRO-1: shard the optimizer states

**Sharded:** optimizer states (fp32 master, Adam m and v): each GPU keeps 12Ψ′/N. Weights and,
during backward, gradients stay whole.

```text
{C.timeline(1, name_col=False)}
```

`setup` is DDP's with `sharded=True` for the optimizer. After backward, each unit's gradient
is **reduce-scattered**, so GPU r receives the averaged gradient of only its slice
`[lo, hi)`. It runs AdamW on its slice of the master copy and Adam states, writes the
result into its slice of the bf16 weights, and an **all-gather** puts the updated weights
back on every GPU. This is DDP's all-reduce split in half, with the optimizer step in between.

{C.code("ZeRO1", ["end_of_backward", "step"])}

- **Memory:** {mem_line(1)}.
- **Communication:** {comm_line(1)}. These are the same bytes as DDP.
- ZeRO-1 keeping full gradients until the end of backward matches DeepSpeed stage 1 and
  Megatron-LM's distributed optimizer.

{details("The reduce-scatter helper that ZeRO-1, 2 and 3 share", C.code("Engine", ["_reduce_scatter_grads"]))}"""

    z2 = f"""### ZeRO-2: also shard the gradients

**Sharded:** optimizer states and gradients. Each GPU keeps (2 + 12)Ψ′/N of them, plus the full
bf16 weights.

```text
{C.timeline(2, name_col=False)}
```

The only change from ZeRO-1 is **when** the reduce-scatter happens: immediately after each
unit's backward, after which that unit's full-size gradient is freed. A GPU never holds more
than one unit's full gradient at a time (booked as `temp`), plus its own gradient slices.
The step is inherited from ZeRO-1 unchanged.

{C.code("ZeRO2", ["begin_backward", "before_backward", "after_backward", "end_of_backward"])}

- **Memory:** {mem_line(2)}.
- **Communication:** {comm_line(2)}. These are ZeRO-1's collectives, issued earlier. The catch
  comes with gradient accumulation (below): with no full gradient buffer to add into, ZeRO-2
  must reduce-scatter on every micro-batch."""

    z3 = f"""### ZeRO-3: also shard the weights

**Sharded:** everything. Each GPU permanently holds only its 1/N slice of the bf16 weights
(`w_shard`), of the gradients and of the optimizer: 16Ψ′/N.

```text
{C.timeline(3, name_col=False)}
```

A unit's full weights exist only while it runs. Before its forward they are
**all-gathered** into the unit's flat buffer, and afterwards the buffer is **released** by
shrinking its storage to 0 bytes. Before its backward they are gathered **again**, and a full
gradient buffer is allocated. After its backward the gradient is **reduce-scattered** into
this GPU's slice, and both full buffers are dropped. The optimizer updates the fp32 master
slice and rounds it into `w_shard`. Nothing is gathered after the step, because no GPU holds
full weights between uses. (The last unit is released after its forward and re-gathered at
once for its backward; FSDP skips that round trip for its root unit, which is why its ZeRO-3
traffic sits slightly below 3Ψ.)

{C.code("ZeRO3", ["_gather", "_release", "before_forward", "after_forward", "before_backward", "after_backward"])}

- **Memory:** {mem_line(3)}.
- **Communication:** {comm_line(3)}. AG in forward (Ψ) + AG in backward (Ψ) + RS (Ψ) = 3Ψ,
  {C.S[3]['comm_bytes'] / C.S[0]['comm_bytes']:.1f}× DDP.
- The weights were released between forward and backward, but autograd saved references to
  those same parameter views. Regrowing and refilling the storage before backward brings them
  back to life, and the gradients come out bit-identical to DDP's (next section)."""

    return f"""## The four stages

**One executor runs all four.** Forward runs unit by unit; each unit's input is detached, so
each unit's backward can run on its own, in reverse. That leaves room to act *between* units,
which is where ZeRO lives. The stages differ only in what they do in a handful of hooks:
`setup`, `before_forward`, `after_forward`, `begin_backward`, `before_backward`,
`after_backward`, `end_of_backward` and `step`.

```text
{overview}
{legend}
AR = all-reduce   RS = reduce-scatter   AG = all-gather
collective calls per step: {U} · {2 * U} · {2 * U} · {3 * U}
```

{details("The executor all four stages share", executor)}

{ddp}

{z1}

{z2}

{z3}

> **Takeaway.** Each stage is a small change to the hooks of one executor: *when* the gradient is
> reduced, *what* is kept afterwards, and *whether* weights are gathered before use."""


def sec_results(C: Ctx) -> str:
    N, P, S = C.N, C.P, C.S
    R = C.R
    ms0 = C.ms(0)
    est = R.get("oom", {}).get("estimates")
    # --- memory table
    rows = []
    for s in range(4):
        st = S[s]
        ok = st["model_state_bytes"] == st["formula_bytes"]
        row = [LABEL[s], FORMULA[s], n(st["formula_bytes"]), n(st["model_state_bytes"]),
               tick(ok), mib(st["model_state_bytes"]), times(ms0 / st["model_state_bytes"]),
               mib(st["peak_bytes"], 2)]
        if est:
            row.append(mib(est[s], 2))
        rows.append(row)
    hdr = ["Stage", "Formula", "Formula (B)", "Measured (B)", "", "MiB", "× smaller than DDP",
           "Peak MiB (measured)"]
    if est:
        hdr.append("Peak MiB (predicted bound)")
    mem_table = table(hdr, rows, "llrrcrrr" + ("r" if est else ""))
    # --- category table
    cats = ["weights", "grads", "master", "adam_m", "adam_v"]
    crow = []
    for s in range(4):
        e = S[s]["end_of_backward"]
        crow.append([LABEL[s]] + [n(e[c]) for c in cats] + [n(sum(e[c] for c in cats))])
    cat_table = table(["Stage", "weights", "gradients", "fp32 master", "Adam m", "Adam v", "total"], crow)
    unit_vals = (f"2Ψ′ = {n(2 * P)}, 4Ψ′ = {n(4 * P)}, 2Ψ′/N = {n(2 * P // N)} and 4Ψ′/N = {n(4 * P // N)}")
    # --- communication and compute table
    fwd_f = C.fwd_formula()
    crows = []
    note_loss = False
    for s in range(4):
        st = S[s]
        hops, adj = C.hops(s)
        note_loss |= adj
        fl = st["flops"]
        fok = fwd_f is not None and fl.get("forward") == fwd_f and fl.get("backward") == 2 * fwd_f
        crows.append([
            LABEL[s], n(st["comm_bytes"]), n(C.comm_formula(s)), tick(st["comm_bytes"] == C.comm_formula(s)),
            f"{st['comm_calls']} / {C.calls_formula(s)}", f"{n(hops)} / {n(C.hops_formula(s))}",
            f"{n(st['opt_elements'])} / {n(C.opt_formula(s))}",
            f"{fl.get('forward', 0) / 1e6:,.2f}", f"{fl.get('backward', 0) / 1e6:,.2f}",
            tick(fok)])
    comm_table = table(["Stage", "Bytes sent (measured)", "Formula", "", "Calls (measured / formula)",
                        "Ring hops (measured / formula)", "Optimizer elements (measured / formula)",
                        "Forward MFLOP", "Backward MFLOP", "FLOPs = closed form"], crows, "lrrcrrrrrc")
    hop_note = (" Bytes, calls and ring hops all exclude the one logging-only all-reduce of the loss per step"
                + (f" (its 2(N−1) = {2 * (N - 1)} hops are subtracted here from the raw hop counter)." if note_loss else "."))
    return f"""## Results: measured vs formula

Every stage trained the same model on the same data for {C.steps} steps on {N} virtual GPUs.
The ledger snapshot is taken at the end of backward on the last step, the moment every stage
holds its largest set of model states. The notebook asserts that it is identical on all {N}
GPUs and equal to the formula; communication and FLOPs are per GPU per step.

**Memory per GPU (model states), Ψ′ = {n(P)}, N = {N}**

{mem_table}

The *peak* adds activations and transient buffers to the model states, so it has no exact
formula. The predicted bound is the one the out-of-memory ladder uses (model states +
activations + the largest transient buffer), written down before any capacity was set.

**Where the bytes are (GPU 0, end of backward, in bytes)**

{cat_table}

Every cell is one of {unit_vals}: each stage divides exactly one more category by N.

**Communication and compute per GPU per step**

{comm_table}

Bytes follow the ring model: 2 · (N−1)/N · 2Ψ′ for DDP, ZeRO-1 and ZeRO-2 and
3 · (N−1)/N · 2Ψ′ for ZeRO-3. Calls are one collective per unit per use: {C.U} all-reduces, or
{C.U} + {C.U}, or {C.U} + {C.U} + {C.U}.{hop_note} Each hop costs one latency α on a real network, so
ZeRO-3 pays 1.5× in latency as well as in bandwidth, and with the smallest messages.

{sec_correctness(C)}

{sec_memory_figs(C)}"""


def sec_correctness(C: Ctx) -> str:
    eq = C.R.get("equivalence")
    if not eq:
        return "### Correctness\n\n> The equivalence check is missing from this run's results."
    L = {int(k): v for k, v in eq["losses"].items()}
    steps = len(L[0])
    maxd = {s: max(abs(a - b) for a, b in zip(L[s], L[0])) for s in range(4)}
    show = sorted(set(list(range(min(3, steps))) + [steps - 1]))
    rows = [[str(i + 1)] + [repr(L[s][i]) for s in range(4)] for i in show]
    rows.append(["largest difference from DDP"] + [repr(maxd[s]) if maxd[s] else "0" for s in range(4)])
    full = [[str(i + 1)] + [repr(L[s][i]) for s in range(4)] for i in range(steps)]
    one = eq.get("ddp_vs_single_gpu")
    one_txt = ""
    if one:
        T = C.cfg["block_size"]
        tol = ddp_single_tolerances(C.quick)
        tol_txt = ""
        if tol:
            mut = any(re.search(r"mutation", p.read_text(encoding="utf-8"), re.I)
                      for p in (ROOT / "tests").glob("test_*.py"))
            adam = ("Adam is almost blind to the scale of the gradient: multiplying every gradient by c turns "
                    "m̂/(√v̂ + ε) into m̂/(√v̂ + ε/c). So forgetting the ÷N in the gradient average only "
                    "shrinks ε by N, and shows up only in elements whose gradients are tiny.")
            tight = tol[0] <= 1e-5 and tol[1] <= 1e-4
            tol_txt = f" The notebook asserts loss < {sci(tol[0])} and weights < {sci(tol[1])}."
            if tight:
                tol_txt += (" That is tight on purpose. " + adam + " A loose tolerance would let that bug through"
                            + ("; a mutation test in `tests/` removes the ÷N and checks that these bounds catch it."
                               if mut else "."))
            else:
                tol_txt += (" Those bounds are loose. " + adam + " A bound this loose may not catch that bug; "
                            "the measured differences above are the stronger evidence.")
        one_txt = f"""
**And DDP on {C.N} GPUs equals one big GPU.** In fp32, {C.N} GPUs × 1 sequence must match
1 GPU × {C.N} sequences, up to the order of floating-point summation ({C.N} means of {T}
tokens vs one mean of {n(C.N * T)} tokens round differently). Over {one['steps']} steps: max |loss difference|
= {sci(one['max_loss_diff'])}, max |weight difference| = {sci(one['max_weight_diff'])}.{tol_txt}"""
    bit = eq.get("bit_identical_losses") and eq.get("bit_identical_weights")
    return f"""### Correctness: ZeRO changes memory, not maths

Every stage sums gradients through the same code (rank order, fp32, ÷N), the optimizer is
elementwise (a slice update equals the full update) and shard boundaries are aligned. So, with
one micro-batch per optimizer step (G = 1), the claim is not "close": it is **bit-identical**.
{'The losses of all ' + str(steps) + ' steps and the final consolidated fp32 weights are equal under `torch.equal` for all four stages.' if bit else '**This run did not reach bit-identity; see the notebook.**'}
(Under gradient accumulation the stages add the micro-batches in different orders; see
[gradient accumulation](#gradient-accumulation-and-activation-checkpointing).)

{table(["Step"] + [SHORT[s] for s in range(4)], rows)}

{details(f"All {steps} losses", table(["Step"] + [SHORT[s] for s in range(4)], full))}
{one_txt}

> **Takeaway.** ZeRO is a *layout* change. With the same reduction order (here, one micro-batch
> per step) it computes the same training run, bit for bit."""


def sec_memory_figs(C: Ctx) -> str:
    S, N = C.S, C.N
    ms3, pk3 = C.ms(3), S[3]["peak_bytes"]
    w = C.R.get("wrap")
    wrap_txt = ""
    if w:
        wrap_txt = f"""
**Wrap granularity.** ZeRO-3 materializes one unit at a time, so the largest unit sets the
floor of its peak. Wrap the whole model as a single unit and ZeRO-3 must gather everything at once:

{table(["ZeRO-3 wrapping", "Largest temporary (MiB)", "Peak per GPU (MiB)"], [
    ["one unit per block (as above)", mib(w['per_block_temp']), mib(w['per_block_peak'])],
    ["whole model as one unit", mib(w['whole_model_temp']), mib(w['whole_model_peak'])]])}

That is {times(w['whole_model_peak'] / w['per_block_peak'])} the peak for (padding aside) the same model
states. FSDP auto-wraps every transformer block for this reason: the gathered unit plus its full
gradient is the transient ZeRO-3 cannot shard."""
    return f"""### Memory, on every GPU and through one step

{figure(C, "cluster_heatmap.png")}

**What to look for:** each panel is all {N} GPUs of one stage, coloured by peak memory. Within
a panel every cell is the same: ZeRO is symmetric, so every GPU shrinks, not just rank 0.

{figure(C, "memory_timeline.png")}

**What to look for** (GPU 0, one step, every allocation and free):
- **DDP:** a flat plateau dominated by the optimizer states (green). The full gradient buffer
  (orange) appears at once when backward starts and stays until the step.
- **ZeRO-1:** the optimizer band is thin, but the full gradient block still spans backward.
- **ZeRO-2:** no gradient block. Each unit's full gradient is a short `temp` spike (pink),
  freed as soon as it is reduce-scattered.
- **ZeRO-3:** the weight band has almost vanished; forward and backward are a sawtooth of
  *gather → use → release*. Activations (amber) are now the largest item.

At this toy size ZeRO-3's peak ({mib(pk3, 2)} MiB) is mostly activations and one gathered unit.
Model states are only {ms3 / pk3:.0%} of it. The ZeRO formulas describe model states; the
peak also depends on the micro-batch and on how big one gathered unit is.
{wrap_txt}"""


def sec_compute(C: Ctx) -> str:
    cfg, S, N, P = C.cfg, C.S, C.N, C.P
    fwd = C.fwd_formula()
    fl = S[0]["flops"]
    step_zero = all(S[s]["flops"].get("step", 0) == 0 for s in range(4))
    comm = {s: S[s]["comm_bytes"] for s in range(4)}
    rows = [
        ["Forward FLOPs per GPU", "no", f"{fl.get('forward', 0) / 1e6:,.2f} MFLOP in all four stages"],
        ["Backward FLOPs per GPU", "no", f"{fl.get('backward', 0) / 1e6:,.2f} MFLOP = 2 × forward"],
        ["The numbers computed", "no", "losses and weights bit-identical (G = 1)"],
        ["Activation memory", "no", "set by micro-batch × context; every OOM prediction assumed this and held"],
        ["Optimizer elements updated per GPU", f"÷ N", f"{n(S[0]['opt_elements'])} → {n(S[1]['opt_elements'])}"],
        ["Bytes sent per GPU", "ZeRO-3 only: +50%",
         f"{mb(comm[0])} / {mb(comm[1])} / {mb(comm[2])} / {mb(comm[3])} MB"],
        ["Collective calls per step", "more, and smaller",
         f"{S[0]['comm_calls']} / {S[1]['comm_calls']} / {S[2]['comm_calls']} / {S[3]['comm_calls']}"],
        ["Model-state memory per GPU", "the point",
         f"{mib(C.ms(0), 2)} / {mib(C.ms(1), 2)} / {mib(C.ms(2), 2)} / {mib(C.ms(3), 2)} MiB"],
    ]
    step_note = (" `FlopCounterMode` counts matrix multiplies only, so the elementwise AdamW step "
                 "registers 0 FLOPs; optimizer work is therefore measured in elements updated."
                 if step_zero else "")
    return f"""## What changes in computation, and what doesn't

{figure(C, "comm_and_compute.png")}

{table(["Quantity", "Changes with the stage?", "Measured (DDP / ZeRO-1 / ZeRO-2 / ZeRO-3)"], rows, "lll")}

**Forward and backward FLOPs are the model's, not the stage's.** Per GPU per step (micro-batch
B = 1, T = {cfg['block_size']}, d = {cfg['n_embd']}, L = {cfg['n_layer']}, V = {cfg['vocab_size']}):

```text
forward  = 2·B·T·(12·L·d² + d·V)  +  4·L·B·T²·d  =  {n(fwd) if fwd else '?'} FLOPs   (counted: {n(fl.get('forward', 0))})
           └ weight matmuls:        └ attention: QKᵀ and att·V,
             QKV 3d², proj d²,        2·T²·d each per layer
             MLP 8d², head dV
backward = 2 × forward                  (a gradient for the input and one for the weights, each a matmul of the same size)
```

That is the familiar 6Ψ FLOPs per token (2 forward + 4 backward) for the weight matmuls, plus
the attention term. The embedding lookup is not a matmul and costs nothing here.{step_note}

**Optimizer work is where ZeRO saves compute.** Under DDP all {N} GPUs run the identical
update on all {n(P)} elements, so {N - 1} of every {N} updates are redundant. Under ZeRO each
element is updated exactly once, by its owner: {n(P // N)} per GPU.

> **Takeaway.** ZeRO trades memory for communication, not compute: 0% more FLOPs, 0% more
> bytes for stages 1–2, +50% bytes for stage 3, and N× less optimizer work."""


def sec_scaling(C: Ctx) -> str:
    sc = C.R.get("scaling")
    if not sc:
        return "## Scaling with N\n\n> The scaling experiment is missing from this run's results."
    Ns = sorted({int(k.split("/")[0]) for k in sc})
    rows = []
    all_ok = True
    for Nn in Ns:
        row = [str(Nn)]
        Pn = sc[f"{Nn}/0"]["P"]
        row.append(n(Pn))
        for s in range(4):
            e = sc[f"{Nn}/{s}"]
            form = {0: 16 * Pn, 1: 4 * Pn + 12 * Pn // Nn, 2: 2 * Pn + 14 * Pn // Nn, 3: 16 * Pn // Nn}[s]
            ok = e["model_state_bytes"] == form
            all_ok &= ok
            row.append(f"{e['model_state_bytes'] / e['P']:.3f}")
        rows.append(row)
    return f"""## Scaling with N

The formulas say ZeRO-1 and ZeRO-2 approach a floor (4Ψ and 2Ψ: the parts that are never
sharded), while ZeRO-3 keeps falling as 1/N. Measured with one step at each N:

{figure(C, "scaling_with_N.png")}

{table(["N", "Ψ′ at this N", "DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"], rows)}

Bytes of model state per padded parameter, per GPU. {'Every entry equals its formula exactly (16, 4 + 12/N, 2 + 14/N, 16/N).' if all_ok else '**Some entries differ from the formula; see the notebook.**'}
Ψ′ changes slightly with N because units are padded to a multiple of N·{C.align}.

**What to look for:** at N = 1 every stage is 16 bytes per parameter (sharding across one GPU
is no sharding). DDP stays at 16; ZeRO-1 bends toward 4, ZeRO-2 toward 2; ZeRO-3 halves every
time N doubles.

> **Takeaway.** Only ZeRO-3 turns "more GPUs" into "a bigger model" without limit. ZeRO-1 and
> ZeRO-2 hit the floor of the replicated weights (and gradients)."""


def _stage_map(d: dict) -> dict:
    return {int(k): v for k, v in d.items() if str(k).isdigit()}


def accum_diffs(C: Ctx, G: int) -> str:
    """Accumulation changes the order of additions, so bit-identity no longer holds; show by how much."""
    why = ("**Accumulation also ends bit-identity.** DDP and ZeRO-1 add the G micro-batch gradients "
           "locally and then reduce once. ZeRO-2 and ZeRO-3 reduce-scatter each micro-batch and add the "
           "reduced slices. The maths is the same, but floating-point addition is not associative, so "
           "the last bits differ.")
    d = C.R.get("grad_accum_diffs")
    if not d:
        return why
    d = _stage_map(d)
    rows = []
    for s in range(1, 4):
        if s not in d:
            continue
        v = d[s]
        loss, weight = (v, None) if not isinstance(v, dict) else (v.get("max_loss_diff"),
                                                                  v.get("max_weight_diff"))
        rows.append([LABEL[s], "0 (bit-identical)" if loss == 0 else sci(loss),
                     "—" if weight is None else ("0 (bit-identical)" if weight == 0 else sci(weight))])
    return (why + f" Measured against DDP after 2 steps at G = {G}:\n\n"
            + table(["Stage", "largest loss difference", "largest weight difference"], rows, "lrr"))


def ckpt_losses(entry: dict) -> list:
    v = entry.get("losses", entry.get("loss"))
    return v if isinstance(v, list) else [v]


def sec_accum_ckpt(C: Ctx) -> str:
    ga, ck = C.R.get("grad_accum"), C.R.get("checkpointing")
    parts = ["## Gradient accumulation and activation checkpointing"]
    if ga:
        G = ga["G"]
        vol = {0: "2Ψ", 1: "2Ψ", 2: f"(G+1)Ψ = {G + 1}Ψ", 3: f"3GΨ = {3 * G}Ψ"}
        rows = []
        for s in range(4):
            one, many = ga[str(s)]
            rows.append([LABEL[s], n(one), n(many), times(many / one), vol[s]])
        parts.append(f"""**Gradient accumulation (G = {G} micro-batches per optimizer step).** DDP and ZeRO-1 keep a
full gradient buffer, accumulate into it locally, and communicate **once per step**. ZeRO-2
has no full buffer to add into: each unit's gradient is reduce-scattered and freed as soon as
it exists, so it must reduce-scatter **every micro-batch**. ZeRO-3 must also re-gather the
weights every micro-batch.

{table(["Stage", "Bytes per step, G = 1", f"Bytes per step, G = {G}", "Ratio", "Volume"], rows, "lrrrl")}

This is why ZeRO-2 and ZeRO-3 combine poorly with pipeline parallelism, which relies on many
micro-batches, and why accumulation hides communication for DDP and ZeRO-1 but not for ZeRO-3.
(FSDP's `no_sync()` avoids the per-micro-batch reduce-scatter by keeping unsharded gradients,
which gives the gradient memory back.)

{accum_diffs(C, G)}""")
    if ck:
        no, yes = ck["no"], ck["yes"]
        tot = {k: sum(ck[k]["flops"].get(p, 0) for p in ("forward", "recompute", "backward"))
               for k in ("no", "yes")}
        l_no, l_yes = ckpt_losses(no), ckpt_losses(yes)
        rows = [["keep activations", mib(no["act"]), mib(no["peak"], 2),
                 f"{no['flops'].get('forward', 0) / 1e6:,.2f}", "0",
                 f"{no['flops'].get('backward', 0) / 1e6:,.2f}", f"{tot['no'] / 1e6:,.2f}", repr(l_no[-1])],
                ["activation checkpointing", mib(yes["act"]), mib(yes["peak"], 2),
                 f"{yes['flops'].get('forward', 0) / 1e6:,.2f}",
                 f"{yes['flops'].get('recompute', 0) / 1e6:,.2f}",
                 f"{yes['flops'].get('backward', 0) / 1e6:,.2f}", f"{tot['yes'] / 1e6:,.2f}", repr(l_yes[-1])]]
        same_loss = l_no == l_yes
        n_steps = ck.get("steps", len(l_no))
        if "weights_equal" in ck:
            eq_txt = (f"over {n_steps} steps the losses are {'identical' if same_loss else 'NOT identical'} "
                      f"and the final weights are {'bit-identical' if ck['weights_equal'] else 'NOT bit-identical'}"
                      + (": recomputation rebuilds the same activations, so it yields the same gradients"
                         if ck["weights_equal"] and same_loss else ""))
        else:
            eq_txt = (f"the loss of the {'single-step' if n_steps == 1 else f'{n_steps}-step'} run is "
                      f"{'identical' if same_loss else 'NOT identical'} (weights were not compared in this run)")
        parts.append(f"""**Activation checkpointing (on ZeRO-3).** ZeRO shards model states, never activations.
Checkpointing runs each unit's forward without saving anything but the unit's input, then
re-runs that unit's forward during backward to rebuild what backward needs.

{table(["", "Peak activations (MiB)", "Peak (MiB)", "Forward MFLOP", "Recompute MFLOP",
        "Backward MFLOP", "Total MFLOP", "Last loss"], [r for r in rows])}

Activations fall to {yes['act'] / no['act']:.0%} of what they were, and compute rises by exactly one
forward ({tot['yes'] / tot['no']:.3f}× ≈ 8/6). Also, {eq_txt}. It composes with every stage.""")
    parts.append(figure(C, "grad_accum_and_ckpt.png"))
    parts.append("> **Takeaway.** Choose the stage with your micro-batching in mind, and remember that\n"
                 "> activations are a separate problem with a separate tool.")
    return "\n\n".join(parts)


def sec_training(C: Ctx) -> str:
    t = C.R.get("training")
    if not t:
        return "## It really trains\n\n> The long training run is missing from this run's results."
    total_mib = t["shard_kib"] * C.N / 1024
    chars = t["steps"] * C.N * C.cfg["block_size"]
    sample = t["sample"]
    prompt = sample.split("\n", 1)[0] + "\n"
    temp = C.src.default("TinyGPT", "generate", "temperature")
    gen = len(sample) - len(prompt)
    return f"""## It really trains: ZeRO-3 end to end, then consolidate the shards

A ZeRO-3 checkpoint is {C.N} shard files, and no GPU ever holds the whole model. To use the
model, the shards are stitched back together, which is what DeepSpeed's `zero_to_fp32.py` does.
Here ZeRO-3 trains for {t['steps']} steps; then the {C.N} fp32 master slices of
{t['shard_kib']:.1f} KiB each are concatenated in rank order into one {total_mib:.2f} MiB fp32
model, loaded into an ordinary `nn.Module`, and asked to continue a prompt.

{figure(C, "loss_curves.png")}

**What to look for:** on the left, four stages drawn over each other form one line, because
they are identical. On the right, the ZeRO-3 loss falls from {t['loss_first']:.3f} (uniform over
{C.cfg['vocab_size']} characters is ln {C.cfg['vocab_size']} = {math.log(C.cfg['vocab_size']):.3f})
to {t['loss_last']:.3f}.

A sample from the consolidated model (prompt `{prompt.strip()}`, temperature {temp}, {gen} new characters):

```text
{sample.rstrip()}
```

This is {a_n(n(C.psi))} {n(C.psi)}-parameter character model after {n(chars)} training characters: judge it by
the shape of the text (speaker names, line breaks, short words), not its sense. The point is
that a model no single virtual GPU ever held was trained, consolidated and run.

> **Takeaway.** Sharded training needs a consolidation step to become an ordinary checkpoint.
> Because a shard is a contiguous slice, consolidation is concatenation."""


def sec_stress(C: Ctx) -> str:
    R, N = C.R, C.N
    parts = ["## Stress tests"]
    # ---- OOM ladder
    oom = R.get("oom")
    if oom:
        caps, est = oom["capacities"], oom["estimates"]
        rows = []
        for i, c in enumerate(caps):
            row = [f"{mib(c, 2)} MiB"]
            for s in range(4):
                obs = oom["observed"].get(f"{s}/{i}")
                pred = oom["predicted"].get(f"{s}/{i}")
                word = "fits" if obs else "OOM"
                row.append(f"{word} {tick(obs == pred)}")
            rows.append(row)
        rows.append(["*predicted peak*"] + [f"*{mib(e, 2)} MiB*" for e in est])
        parts.append(f"""### Out-of-memory ladder: predict first, then run

The strongest test of a memory model is to predict failures. Each of the {N} virtual GPUs gets
a fixed capacity; the prediction is written down **before** running:

```text
predicted peak = model states (formula) + max(activations + largest transient, optimizer scratch)
                 activations: {mib(oom['act_bytes'])} MiB, the same for every stage (measured once, one device)
                 transient:   none (DDP, ZeRO-1) · one unit's full gradient (ZeRO-2) · + its gathered weights (ZeRO-3)
```

Capacities sit {caps[0] / est[0] - 1:.0%} above the DDP prediction, at the geometric midpoints between
consecutive predictions, and {1 - caps[-1] / est[3]:.0%} below the ZeRO-3 prediction.

{figure(C, "oom_ladder.png")}

{table(["Capacity per GPU"] + [SHORT[s] for s in range(4)], rows, "lcccc")}

✓ = observed outcome equals the prediction: **{oom['matches']} of {oom['total']}**. It is a
staircase: each step down knocks out one more stage, DDP first, until only ZeRO-3 trains, and on
the smallest GPU even ZeRO-3 runs out. The first failure, as the ledger reported it:

```text
{oom['example_message']}
```""")
    # ---- GPU mode
    gm = R.get("gpu_mode", {})
    if gm.get("stages"):
        prec = gm.get("precision", "bf16")
        rows = []
        for s in range(4):
            a = gm["stages"][str(s)]
            diff = a["cuda_bytes"] - a["ledger_bytes"]
            pred = allocator_rounding(s, C.units, N, prec)
            rows.append([LABEL[s], n(a["ledger_bytes"]), n(a["cuda_bytes"]), n(diff), n(pred),
                         n(a["rounding_allowance"]), tick(0 <= diff <= a["rounding_allowance"])])
        exact = all(gm["stages"][str(s)]["cuda_bytes"] - gm["stages"][str(s)]["ledger_bytes"]
                    == allocator_rounding(s, C.units, N, prec) for s in range(4))
        z3 = gm["stages"].get("3", {})
        gtxt = ""
        if "gather_ledger" in z3:
            gtxt = (f"\n\n**ZeRO-3's gather and release, on real memory.** Gathering one block on all {N} GPUs "
                    f"added {n(z3['gather_ledger'])} B to the ledgers and {n(z3['gather_cuda'])} B to the CUDA "
                    f"allocator; releasing it (`resize_(0)`) returned the allocator to "
                    f"{z3['released_cuda']:+,} B of where it started. Shrinking a storage to zero really gives the "
                    "memory back, which is the whole mechanism behind ZeRO-3 and FSDP parameter freeing.")
        tl = gm.get("train_losses", {})
        ttxt = ""
        if tl.get("0") and tl.get("3"):
            same = tl["0"] == tl["3"]
            cpu = C.S[0].get("losses", [])[:len(tl["0"])]
            cpu_txt = ""
            if cpu and cpu != tl["0"]:
                d = max(abs(a - b) for a, b in zip(cpu, tl["0"]))
                cpu_txt = (f" They differ from the CPU run by up to {sci(d)}, because CUDA and CPU kernels "
                           "round differently; within one device, the stages agree bit for bit.")
            ttxt = (f"\n\nTraining for {len(tl['0'])} steps with all {N} virtual GPUs on the real GPU: DDP and "
                    f"ZeRO-3 losses {'identical' if same else 'differ'} "
                    f"({', '.join(f'{x:.5f}' for x in tl['0'])}).{cpu_txt}")
        parts.append(f"""### GPU mode: is the ledger real?

The ledger is bookkeeping, and a skeptic should ask whether real memory agrees. With all {N} virtual
GPUs on one real GPU ({gm.get('device', 'CUDA')}, {prec}), the change in
`torch.cuda.memory_allocated()` after every GPU has built its model states is compared with the
sum of the {N} ledgers. At that moment no gradients exist yet, so each GPU holds weights +
optimizer: 14Ψ′ (DDP), 2Ψ′ + 12Ψ′/N (ZeRO-1 and ZeRO-2 alike) or 14Ψ′/N (ZeRO-3).

{table(["Stage", "Ledger, all GPUs (B)", "CUDA allocator (B)", "Difference (B)",
        "512-byte rounding, computed (B)", "Allowance (B)", ""], rows, "lrrrrrc")}

{'The difference is **exactly** the allocator rounding each small tensor up to a multiple of 512 bytes, computed from the slice sizes; DDP holds only large, aligned buffers, so its difference is 0.' if exact else 'Every difference is within the allowance of 512 bytes per tensor.'}{gtxt}{step_audit_txt(gm, N)}{ttxt}""")
    else:
        parts.append(f"""### GPU mode: is the ledger real?

> **Skipped in this run:** {gm.get('reason', 'no GPU-mode results recorded')}. On a machine with CUDA the
> notebook puts all {N} virtual GPUs on the real GPU and compares the ledgers with the CUDA
> allocator, byte for byte up to its 512-byte rounding.""")
    # ---- gloo
    g = R.get("gloo", {})
    status = g.get("status")
    intro = (f"Threads and shared slots could hide a mistake that real collectives would expose. "
             f"[`tools/gloo_check.py`](tools/gloo_check.py) runs the *unchanged* `ZeRO1` engine in "
             f"separate OS processes over PyTorch's gloo backend, through an adapter with the same "
             f"communicator interface, then runs the thread simulator with the same model, data, seed "
             f"and learning rate, and compares. Gloo adds contributions in its own order, so the check is "
             f"`allclose`, not bit-equality.")
    if status == "ok":
        prims = ", ".join(f"{k} → `{v}`" for k, v in g.get("primitives", {}).items())
        fb = g.get("fallbacks") or {}
        cfg = g.get("config", {})
        ring = g.get("ring_all_reduce_on_gloo")
        rtol = module_constant(ROOT / "tools" / "gloo_check.py", "RTOL")
        atol = module_constant(ROOT / "tools" / "gloo_check.py", "ATOL")
        rows = [["processes × steps", f"{g['world']} × {g['steps']}"],
                ["configuration", f"stage {cfg.get('stage', 1)}, {cfg.get('precision', 'fp32')}, d = {cfg.get('n_embd')}, "
                                  f"{cfg.get('n_layer')} layers"],
                ["gloo primitives used", prims],
                ["fallbacks needed", ", ".join(f"{k}: {v}" for k, v in fb.items()) if fb else "none"],
                ["largest loss difference, gloo vs simulator", sci(g["max_abs_loss_diff"])],
                ["largest weight difference (consolidated fp32)", sci(g["max_abs_weight_diff"])],
                [f"allclose (rtol {sci(rtol)}, atol {sci(atol)})" if rtol is not None else "allclose",
                 tick(g["allclose"])],
                ["bit-exact weights", "yes" if g.get("bit_exact_weights") else "no (different summation order)"],
                ["per-rank byte counters, gloo == simulator (same collective sequence)",
                 tick(g.get("comm_bytes_per_rank", {}).get("equal", False))]]
        if ring:
            rows.append(["hand-written ring all-reduce over gloo point-to-point",
                         f"allclose {tick(ring['allclose'])}, {n(ring['bytes_expected'])} B sent per rank = 2(N−1)/N · S "
                         f"{tick(all(b == ring['bytes_expected'] for b in ring['bytes_sent_per_rank']))}"])
        exact = ""
        if g.get("bit_exact_weights") and g["world"] == 2:
            exact = (" With 2 processes each element is the sum of just two numbers, so the summation "
                     "order cannot differ, and the result here is bit-exact as well.")
        elif g.get("bit_exact_weights"):
            exact = " In this run the result is bit-exact as well."
        parts.append(f"""### The same engine on real `torch.distributed` (gloo)

{intro}{exact} The gloo check trains its own small configuration, because each process loads
its own PyTorch.

{table(["Check", "Result"], rows, "ll")}

Read the rows for what they prove. The evidence is the losses and consolidated weights: real
collectives in real processes produce the simulator's numbers. The byte counters are weaker: both
sides charge the same ring cost model, so equal counts confirm that the engine issued the same
*sequence* of collectives over gloo, not the bytes on the wire. The communicator interface is the
only thing that changed between threads and processes; the ZeRO code is identical.""")
    else:
        why = g.get("reason") or g.get("error") or (f"return codes {g.get('returncodes')}" if g.get("returncodes") else "no details recorded")
        word = {"skipped": "Skipped", "failed": "Failed", "timeout": "Timed out",
                "mismatch": "Ran but did not match"}.get(status, "Not run")
        extra = ""
        if status == "mismatch":
            extra = (f" max |Δ loss| = {sci(g.get('max_abs_loss_diff', 0))}, "
                     f"max |Δ weight| = {sci(g.get('max_abs_weight_diff', 0))}.")
        parts.append(f"""### The same engine on real `torch.distributed` (gloo)

{intro}

> **{word} in this run:** {why}.{extra} Run `python tools/gloo_check.py --world 2` to reproduce it
> on its own.""")
    return "\n\n".join(parts)


def _flatten(d: dict, prefix="") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + " / "))
        else:
            out[key] = v
    return out


def _cell(v) -> str:
    if isinstance(v, bool):
        return tick(v)
    if isinstance(v, int):
        return n(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.4g}" if abs(v) >= 1e-3 else sci(v)
    if isinstance(v, list):
        return ", ".join(_cell(x) for x in v)
    return str(v)


def step_audit_txt(gm: dict, N: int) -> str:
    """The mid-training allocator audit (results['gpu_mode']['step_audit']), rendered as a table
    whatever its exact keys are: one row per stage, one column per (flattened) field."""
    sa = gm.get("step_audit")
    if not sa or not isinstance(sa, dict):
        return ""
    stages = _stage_map(sa) or {}
    if not stages:
        flat = _flatten(sa)
        rows = [[k.replace("_", " "), _cell(v)] for k, v in flat.items()]
        tbl = table(["Field", "Value"], rows, "lr")
        bools = [v for v in flat.values() if isinstance(v, bool)]
    else:
        flats = {s: _flatten(v) if isinstance(v, dict) else {"value": v} for s, v in stages.items()}
        cols = list(dict.fromkeys(k for f in flats.values() for k in f))
        rows = [[LABEL.get(s, str(s))] + [_cell(flats[s].get(c, "")) for c in cols] for s in sorted(flats)]
        tbl = table(["Stage"] + [c.replace("_", " ") for c in cols], rows, "l" + "r" * len(cols))
        bools = [v for f in flats.values() for v in f.values() if isinstance(v, bool)]
    verdict = ""
    if bools:
        verdict = (" Every check passes." if all(bools) else " **Not every check passes; see the notebook.**")
    return (f"\n\n**Mid-training audit.** At a step in the middle of training, the change in "
            f"`torch.cuda.memory_allocated()` is compared with the change in the {N} ledgers at the end of "
            f"forward (activations) and at the end of backward (gradients). This tests the dynamic part of the "
            f"ledger, not just the model states built at setup.{verdict}\n\n{tbl}")


def allocator_rounding(stage: int, units: list, N: int, precision: str) -> int:
    """Bytes the CUDA caching allocator adds by rounding each tensor up to 512 bytes, summed
    over all N virtual GPUs, for the tensors a stage holds right after setup."""
    r = lambda b: -b % 512  # noqa: E731
    bw = 2 if precision == "bf16" else 4
    n_opt = 3 if precision == "bf16" else 2
    total = 0
    for u in units:
        full, shard = u["padded"], u["padded"] // N
        total += r(full * bw) if stage < 3 else r(shard * bw)
        total += n_opt * (r(full * 4) if stage == 0 else r(shard * 4))
    return total * N


def sec_hardware(C: Ctx) -> str:
    R = C.R
    rh, ot = R.get("real_hardware"), R.get("overlap_tokens")
    if not rh:
        return "## Real hardware\n\n> The real-hardware estimates are missing from this run's results."
    sig = inspect.signature(zt.real_hardware_rows).parameters
    mfu, alpha = sig["mfu"].default, sig["alpha"].default
    first = next(iter(rh.values()))[0]
    hw0 = zt.HARDWARE[next(iter(rh))]
    gpn, n_gpus = hw0["gpus_per_node"], first["n_gpus"]
    hbm = hw0["hbm_bytes"] / 1e9

    def hw_table(rows):
        out = []
        for r in rows:
            out.append([r["model"], r["variant"], gbf(r["states_gb"]), gbf(r["act_gb"]), gbf(r["total_gb"]),
                        "yes" if r["fits_80GB"] else "no", r["fits_with"], f"{r['comm_over_compute']:.2f}"])
        return table(["Model", "Variant", "States GB", "Activations GB", "Total GB", "Fits 80 GB",
                      "Least recompute that fits", "comm / compute"], out, "llrrrccr")

    hws = list(rh)
    tables = []
    for i, hw in enumerate(hws):
        spec = zt.HARDWARE.get(hw, {})
        title = (f"{n_gpus} × {hw}: {spec.get('peak_bf16_flops', 0) / 1e12:,.0f} TFLOP/s bf16 dense, "
                 f"{spec.get('inter_node_bw', 0) / 1e9:,.0f} GB/s NIC per GPU")
        body = hw_table(rh[hw])
        if i == 0:
            tables.append(f"**{title}**\n\n{body}")
        else:
            same_mem = all(a["states_gb"] == b["states_gb"] and a["act_gb"] == b["act_gb"]
                           for a, b in zip(rh[hws[0]], rh[hw]))
            note = (f"The memory columns are identical to the table above (same {spec.get('hbm_bytes', 0) / 1e9:.0f} GB, "
                    "same formulas); only comm / compute changes." if same_mem else "")
            tables.append(details(title, f"{note}\n\n{body}"))
    # ratios for the prose, from the first hardware
    a100 = {(r["model"], r["variant"]): r for r in rh[hws[0]]}
    models = list(dict.fromkeys(r["model"] for r in rh[hws[0]]))
    picks = []
    for m in models:
        cand = [a100.get((m, SHORT[s])) for s in range(4)]
        cand = [c for c in cand if c]
        fit = next((c for c in cand if c["fits_80GB"]), None)
        if fit:
            extra = "" if fit["fits_with"] in ("none", "selective") else f" ({fit['fits_with']} recomputation)"
            picks.append(f"**{m}** → {fit['variant']}{extra}")
        else:
            full = next((c for c in cand if c["fits_with"] == "full"), None)
            picks.append(f"**{m}** → {full['variant']} + full recomputation" if full else
                         f"**{m}** → none of the four: offload, tensor or pipeline parallelism")
    hsdp_txt = ""
    has = [m for m in models if (m, "HSDP") in a100 and (m, "ZeRO-3") in a100 and (m, "DDP") in a100]
    m7 = "LLaMA-2 7B" if "LLaMA-2 7B" in has else (has[0] if has else None)
    if m7:
        h, z3, d = a100[(m7, "HSDP")], a100[(m7, "ZeRO-3")], a100[(m7, "DDP")]
        hsdp_txt = (f" For {m7}, HSDP holds N/g = {h['states_gb'] / z3['states_gb']:.1f}× the model states "
                    f"of ZeRO-3 (16Ψ/{gpn} = {gbf(h['states_gb'])} GB vs 16Ψ/{n_gpus} = {gbf(z3['states_gb'])} GB). What "
                    f"it cuts is modelled communication *time*: {h['comm_s']:.2f} s against {z3['comm_s']:.2f} s "
                    f"for flat ZeRO-3 ({z3['comm_s'] / h['comm_s']:.1f}× less) and {d['comm_s']:.2f} s for DDP. "
                    "The gathers and reduce-scatters stay on NVLink inside the node, and only each GPU's 1/g "
                    "shard of the gradients crosses the slow inter-node link.")
        if (m7, "ZeRO++ hpZ") in a100:
            p = a100[(m7, "ZeRO++ hpZ")]
            hsdp_txt += (f" ZeRO++ hpZ keeps ZeRO-3's 1/N sharding plus a node-local copy of the bf16 weights "
                         f"(16Ψ/{n_gpus} + 2Ψ/{gpn} = {gbf(p['states_gb'])} GB) and cuts the modelled communication "
                         f"time to {p['comm_s'] / z3['comm_s']:.0%} of ZeRO-3's.")
    # overlap thresholds
    ov = ""
    if ot:
        rows = []
        for key, v in ot.items():
            hw, net = key.split("/")
            bw = zt.effective_bandwidth(hw, zt.HARDWARE[hw]["gpus_per_node"] + 1, net) if hw in zt.HARDWARE else float("nan")
            rows.append([hw, f"`{net}`", f"{bw / 1e9:,.0f} GB/s", n(v["0"]), n(v["3"])])
        ratio = ""
        k1, k2 = "A100-80GB/per_nic", "H100-80GB/per_nic"
        if k1 in ot and k2 in ot:
            a, h = zt.HARDWARE["A100-80GB"], zt.HARDWARE["H100-80GB"]
            ratio = (f" An H100 needs {ot[k2]['0'] / ot[k1]['0']:.2f}× the tokens of an A100: "
                     f"{h['peak_bf16_flops'] / a['peak_bf16_flops']:.1f}× the FLOP/s against "
                     f"{h['inter_node_bw'] / a['inter_node_bw']:.0f}× the NIC bandwidth.")
        ov = f"""**When does communication hide behind compute?** Set the two times equal:

```text
communication = v · Ψ · b / BW            v = 2 (DDP, ZeRO-1, ZeRO-2) or 3 (ZeRO-3),  b = 2 bytes
compute       = 6 · Ψ · T / (F · MFU)     T = tokens per GPU per optimizer step,  F = peak FLOP/s
         ⇒    T*  =  v · b · F · MFU / (6 · BW)          (Ψ cancels)
```

{table(["Hardware", "Network model", "BW per GPU", "T*: DDP, ZeRO-1, ZeRO-2", "T*: ZeRO-3"], rows, "llrrr")}

Ψ cancels because every parameter is sent a fixed number of times and costs a fixed 6 FLOPs
per token: a bigger model does not hide communication any better; only more tokens per GPU do.
A **faster** GPU needs **more** tokens, because compute shrinks while the network does not.{ratio}

The two network models bracket reality. `per_nic` (conservative, and the one used in the tables
above) assumes every byte a GPU sends crosses its own single NIC. That is exact when every ring hop
crosses a node boundary. `rail` assumes NCCL runs one ring per NIC in parallel (rail-optimized),
so only one hop per ring leaves each node, and the node's {gpn} NICs act as one pipe of
{gpn} × NIC bandwidth, capped by NVLink. That is up to {gpn}× the bandwidth, and so up to {gpn}× fewer
tokens. Well-tuned DGX clusters measure all-reduce bus bandwidth close to the rail figure
(see [`references.md`](references.md)). Two caveats: T* counts the bandwidth term only, and
gradient traffic can only start once backward produces gradients (DDP buckets, ZeRO-3
prefetch), so in practice you want some margin above T*."""
    rc = first["recompute"]
    rc_txt = {"selective": "selective recomputation (attention scores are recomputed in backward "
                           "instead of stored; FlashAttention never stores them anyway)",
              "none": "no recomputation", "full": "full recomputation"}.get(rc, f"`{rc}` recomputation")
    return f"""## Real hardware: which stage fits 7B, 13B, 70B on {n_gpus} GPUs?

The toy model proves the mechanics, and the same formulas carry them to real sizes: {n_gpus} GPUs =
{n_gpus // gpn} nodes × {gpn}, {hbm:.0f} GB each. The budget is {hbm:.0f} × 10⁹ bytes, which leaves the
rest of the physical HBM for the CUDA context, NCCL buffers and fragmentation. Activations follow
Korthikanti et al. (2022) at micro-batch {first['micro_bsz']} and each model's own context
(G = {first['grad_accum']}), with {rc_txt}. Time per step uses the α-β model: compute 6ΨT at
{mfu:.0%} MFU against communication at α = {alpha * 1e6:.0f} µs per ring hop, plus the bytes over
the per-GPU link (`per_nic`).

{figure(C, "real_hardware.png")}

{chr(10).join(t + chr(10) for t in tables)}
**Reading the table.** *States* is model states from the formulas. *Activations* assumes
{rc} recomputation. *Least recompute that fits* is the cheapest of none < selective < full
that brings states + activations under {hbm:.0f} GB, or "no" if even full recomputation cannot.
*comm / compute* below 1 means communication can hide behind compute, *if* the implementation
overlaps them. The HSDP row is FSDP's `HYBRID_SHARD`: ZeRO-3 inside each {gpn}-GPU node, plain data
parallelism across nodes. ZeRO++ hpZ keeps a secondary, node-local weight partition, so the
backward all-gather stays on NVLink.{hsdp_txt}

**Applying the decision guide to {n_gpus} × {hws[0]}:** {' · '.join(picks)}.

{ov}

> **Takeaway.** Memory decides which stage is *possible*; tokens per GPU per step decide
> whether its communication is *free*."""


def sec_decision(C: Ctx) -> str:
    return """## Which stage to use

Pick the **least** sharded stage that fits, because each step down the list costs something:
ZeRO-2 communicates every micro-batch under accumulation, and ZeRO-3 sends 1.5× the bytes in
3 collectives per unit.

```text
              Does 16Ψ + activations fit on one GPU?
                 │ yes                     │ no
                 ▼                         ▼
           DDP (ZeRO-0)          Does 4Ψ + 12Ψ/N fit? ──yes──► ZeRO-1
           least communication             │ no
                                           ▼
                                 Does 2Ψ + 14Ψ/N fit? ──yes──► ZeRO-2  (mind grad-accum comm)
                                           │ no
                                           ▼
                                 ZeRO-3 / FSDP FULL_SHARD   (1.5× communication)
                                           │ still no?
                                           ▼
                activation checkpointing ─► offload (ZeRO-Offload / Infinity)
                                         ─► tensor / pipeline parallelism
```

"Activations" means whatever you plan to keep after checkpointing. ZeRO never shards them.
Two refinements. With many GPUs across slow inter-node links, **HSDP** (ZeRO-3 inside a node,
replication across nodes) buys much of ZeRO-3's memory saving for a fraction of its
cross-node traffic, provided 16Ψ/g fits (g = GPUs per node). And ZeRO-1 is almost always worth
turning on over DDP: the same bytes, and 4Ψ + 12Ψ/N instead of 16Ψ."""


def sec_revision(C: Ctx) -> str:
    R, S, N, P = C.R, C.S, C.N, C.P
    ga = R.get("grad_accum")
    ck = R.get("checkpointing")
    t = R.get("training")
    ot = R.get("overlap_tokens", {})
    comm = {s: S[s]["comm_bytes"] for s in range(4)}
    by3 = S[3]["comm_by_op"]
    cheat = table(["", "DDP (ZeRO-0)", "ZeRO-1", "ZeRO-2", "ZeRO-3"], [
        ["sharded", "nothing", "optimizer states", "+ gradients", "+ weights"],
        ["memory per GPU", "16Ψ", "4Ψ + 12Ψ/N", "2Ψ + 14Ψ/N", "16Ψ/N"],
        [f"measured here (MiB, N = {N})"] + [mib(C.ms(s), 2) for s in range(4)],
        ["gradient sync", "all-reduce", "reduce-scatter (after backward)", "reduce-scatter (per unit, during backward)", "reduce-scatter (per unit)"],
        ["after the step", "nothing", "all-gather weights", "all-gather weights", "nothing (weights stay sharded)"],
        ["weights before use", "resident", "resident", "resident", "all-gather, twice per step"],
        ["traffic per step", "2Ψ", "2Ψ", "2Ψ", "3Ψ"],
        ["measured here (MB)"] + [mb(comm[s]) for s in range(4)],
        ["with G micro-batches", "2Ψ", "2Ψ", "(G+1)Ψ", "3GΨ"],
        ["collective calls (U units; G micro-batches)", "U", "2U", "2U; (G+1)·U", "3U; 3G·U"],
        ["forward/backward FLOPs", "F", "F", "F", "F"],
        ["optimizer work per GPU", "Ψ", "Ψ/N", "Ψ/N", "Ψ/N"],
        ["DeepSpeed `zero_optimization.stage`", "0", "1", "2", "3"],
        ["PyTorch FSDP `ShardingStrategy`", "`NO_SHARD`", "(none; `ZeroRedundancyOptimizer` is 3Ψ)", "`SHARD_GRAD_OP`", "`FULL_SHARD` (`HYBRID_SHARD`: full shard inside a node, replicate across nodes)"],
    ], "lllll")
    pitfalls = f"""- **PyTorch's `ZeroRedundancyOptimizer` is not ZeRO-1 in traffic.** It shards the optimizer
  states, but runs under ordinary DDP: gradients are all-reduced (2Ψ), each rank updates the
  parameters it owns, and the updated parameters are broadcast from their owners (Ψ). That is 3Ψ,
  not ZeRO-1's 2Ψ. It also assigns whole parameters to ranks, so shards are uneven.
- **Gradient buffers are often fp32.** DeepSpeed's bf16 optimizer and Megatron-LM's bf16 training
  accumulate gradients in fp32 for accuracy, so b<sub>g</sub> = 4: DDP becomes 18Ψ, ZeRO-1 6Ψ + 12Ψ/N
  (Megatron's distributed optimizer) and ZeRO-2 2Ψ + 16Ψ/N. Use the b<sub>w</sub>, b<sub>g</sub>, K
  form, not the 16Ψ shorthand.
- **Gradient clipping needs one extra all-reduce.** With sharded gradients no GPU sees the global norm.
  Each GPU sums the squares of its own slice, the scalars are all-reduced, and every GPU scales its
  slice by the same factor. Clipping by a local norm would under-clip, and differently on each GPU.
- **ZeRO does not shard activations**, temporary buffers or fragmentation. At long context,
  activations, not model states, decide what fits: use checkpointing, sequence/context parallelism
  or ZeRO-R's partitioned activation checkpoints.
- **ZeRO-3 is latency-bound without prefetch.** It issues 3 collectives per unit per micro-batch, each
  small. Real implementations prefetch the next unit's all-gather while the current unit computes
  (FSDP `forward_prefetch` / `backward_prefetch`; DeepSpeed `stage3_prefetch_bucket_size`), and keep
  tiny parameters unsharded (`stage3_param_persistence_threshold`). This engine does neither.
- **FSDP keeps the root unit gathered** after forward, since it is needed again at once for backward,
  so its ZeRO-3 traffic sits slightly below 3Ψ. This engine releases every unit, which keeps exactly 3Ψ.
- **Per-unit vs global partitioning.** FSDP (and this engine) flattens each wrapped unit and splits
  *every unit* N ways. DeepSpeed stages 1 and 2 flatten each parameter group into one buffer split into N
  contiguous ranges, so a GPU's range may cover a few whole layers, and DeepSpeed stage 3 partitions each
  parameter. Memory and bytes are the same; collective sizes, padding and what one checkpoint shard
  contains differ.
- **Unit size sets ZeRO-3's floor.** Wrap too coarsely and the gathered unit plus its full gradient
  dominates the peak{f" (the wrap experiment above: {times(R['wrap']['whole_model_peak'] / R['wrap']['per_block_peak'])} the peak when the whole model is one unit)" if R.get('wrap') else ""}.
- **Checkpoints are sharded.** A ZeRO-3 save is N files whose boundaries depend on N and on the padding;
  resuming at a different N, or serving the model, needs consolidation or a resharding loader."""
    ga_txt = ""
    if ga:
        G = ga["G"]
        ga_txt = (f" Measured here with G = {G}: DDP and ZeRO-1 {ga['0'][1] / ga['0'][0]:.1f}×, "
                  f"ZeRO-2 {ga['2'][1] / ga['2'][0]:.1f}× (= (G+1)/2), ZeRO-3 {ga['3'][1] / ga['3'][0]:.1f}× (= G).")
    ck_txt = (f" (measured here: {mib(ck['no']['act'])} → {mib(ck['yes']['act'])} MiB of activations)"
              if ck else "")
    ck2 = ""
    if t:
        ck2 = (f" Here: {C.N} slices of {t['shard_kib']:.1f} KiB concatenated into one "
               f"{t['shard_kib'] * C.N / 1024:.2f} MiB fp32 model that generated text.")
    thr = ""
    if "A100-80GB/per_nic" in ot:
        a = ot["A100-80GB/per_nic"]
        mfu = inspect.signature(zt.overlap_threshold_tokens).parameters["mfu"].default
        nic = zt.HARDWARE["A100-80GB"]["inter_node_bw"] / 1e9
        thr = (f" With A100s at {mfu:.0%} MFU and one {nic:.0f} GB/s NIC per GPU, "
               f"T* ≈ {n(a['0'])} tokens (DDP, ZeRO-1/2) and {n(a['3'])} (ZeRO-3).")
    qa = [
        ("Where do the 16 bytes per parameter go?",
         f"2 bytes of bf16 weights + 2 of bf16 gradients + 12 of optimizer state: a 4-byte fp32 master "
         f"copy, 4 for Adam's m and 4 for Adam's v (K = 12). Activations and temporary buffers come on "
         f"top; they scale with micro-batch × context, not with Ψ. Here DDP's ledger holds exactly "
         f"16 × {n(P)} = {n(C.ms(0))} bytes per GPU."),
        ("Why does ZeRO-1/2 communicate no more than DDP?",
         f"Because a ring all-reduce *is* a reduce-scatter followed by an all-gather, each moving "
         f"(N−1)/N·Ψ elements per GPU. DDP spends both halves on gradients. ZeRO-1/2 reduce-scatter the "
         f"gradients (a GPU needs the summed gradient only for the slice it updates), then all-gather the "
         f"updated *weights* instead of gradients. Same two halves, same bytes: measured {n(comm[0])} B for "
         f"DDP and {n(comm[1])} B for ZeRO-1 and ZeRO-2."),
        ("Where does ZeRO-3's extra Ψ come from?",
         f"Weights are not resident, so every unit is all-gathered twice per step: before its forward, and "
         f"again before its backward, because it was released in between to save memory. Add the gradient "
         f"reduce-scatter: 3Ψ. Nothing is gathered after the step, since weights stay sharded. Measured: "
         f"all-gather {n(by3.get('all_gather', 0))} B + reduce-scatter {n(by3.get('reduce_scatter', 0))} B "
         f"= {comm[3] / comm[0]:.1f}× DDP. Keeping weights gathered from forward through backward (FSDP "
         f"`SHARD_GRAD_OP`, `reshard_after_forward=False`) drops the second gather and returns to 2Ψ, at the cost "
         f"of holding full weights for the whole step."),
        ("Why does ZeRO-2 communicate on every micro-batch under gradient accumulation?",
         f"It never keeps a full gradient buffer: each unit's gradient is reduce-scattered and freed as soon as "
         f"backward produces it. The next micro-batch creates a fresh full gradient with nowhere local to add it "
         f"to, so it is reduce-scattered again into the owner's slice: G reduce-scatters + 1 all-gather = "
         f"(G+1)Ψ. ZeRO-3 also re-gathers weights every micro-batch: 3GΨ. DDP and ZeRO-1 accumulate into their "
         f"resident full gradients and communicate once: 2Ψ.{ga_txt}"),
        ("What does ZeRO *not* shard, and what does?",
         f"ZeRO-DP shards only model states: optimizer states (stage 1), gradients (2), weights (3). It does not "
         f"shard activations, temporary buffers (the gathered unit, one unit's full gradient, communication "
         f"buckets) or fragmentation. Activations depend on micro-batch × context and are identical under every "
         f"stage. Tools for them: activation checkpointing{ck_txt}, ZeRO-R's partitioned activation checkpoints "
         f"and constant-size buffers, sequence/context parallelism, tensor parallelism with sequence "
         f"parallelism, and CPU offload."),
        ("How does FSDP free weights without breaking autograd?",
         "Parameters are views into one flat buffer per unit. After forward, the buffer's *storage* is resized "
         "to 0 bytes (`untyped_storage().resize_(0)`). The tensor objects, including the references autograd "
         "saved for backward, stay alive but point at empty storage. Before backward the storage is resized "
         "back and refilled by an all-gather, so the saved references see valid values again. A pre-forward "
         "guard catches use-while-freed, which would otherwise crash. In GPU mode here, the release returned "
         "the CUDA allocator exactly to where it started."),
        ("Why does all-reduce = reduce-scatter + all-gather matter?",
         "(1) It is why ZeRO-1/2 cost nothing extra: ZeRO stops after the reduce-scatter when a GPU only needs "
         "its slice, and spends the all-gather half on updated weights. (2) It is how the bandwidth-optimal ring "
         "implements all-reduce, so 2(N−1)/N·S per GPU is a floor, not an artefact. (3) Here it also gives "
         "every stage one reduction code path (rank order, fp32, ÷N), which is why the four stages are "
         "bit-identical with one micro-batch per step."),
        ("How do you clip by global gradient norm when the gradients are sharded?",
         "After the reduce-scatter (so each slice holds averaged gradients) and before the optimizer step: each "
         "GPU computes the sum of squares of its own slice (padding is zero and adds nothing), all-reduces that "
         "one scalar with SUM, takes the square root to get the global norm, and scales its slice by "
         "min(1, max_norm / (norm + ε)). That is one extra, tiny all-reduce. Using the local slice's norm "
         "would be about √N too small, so it would under-clip, and by a different factor on each GPU. Under "
         "DDP, where gradients are replicated, every GPU can compute the norm locally. (Clipping is off in "
         "this engine.)"),
        ("How do you save and load a ZeRO-3 checkpoint?",
         "Each rank saves what it owns: its fp32 master slice, its Adam m and v slices, and the layout "
         "metadata (unit order, padding, N). To resume at the same N, each rank loads its own file. For "
         "inference or a different N, consolidate: concatenate the slices of each unit in rank order, drop "
         "the padding, and unflatten into parameter shapes (DeepSpeed's `zero_to_fp32.py`; in PyTorch, FSDP's "
         "full state dict gathered to rank 0, or `torch.distributed.checkpoint`, which reshards on load)."
         + ck2),
        ("When does communication hide behind compute?",
         "When the compute per step outlasts the communication: 6ΨT/(F·MFU) ≥ v·Ψ·b/BW, i.e. "
         "T ≥ T* = v·b·F·MFU/(6·BW) tokens per GPU per step, with v = 2 (DDP, ZeRO-1/2) or 3 (ZeRO-3). Ψ "
         "cancels, so model size does not help; only more tokens per GPU per step do. Accumulation adds tokens for "
         "DDP and ZeRO-1 but not for ZeRO-3, whose traffic grows with G too. It also needs an implementation "
         "that overlaps (DDP buckets, ZeRO-3 prefetch)." + thr),
        ("How does PyTorch's `ZeroRedundancyOptimizer` differ from ZeRO-1 (3Ψ vs 2Ψ)?",
         "Both shard the optimizer states, so optimizer memory is the same. `ZeroRedundancyOptimizer` wraps an "
         "optimizer under unchanged DDP: DDP all-reduces the full gradients (2Ψ), each rank updates the "
         "parameters it owns, then broadcasts them (Ψ), for 3Ψ in total. ZeRO-1 replaces the all-reduce with a "
         "reduce-scatter, because a rank only needs its own slice's gradient, then all-gathers the updated "
         "weights: Ψ + Ψ = 2Ψ. It also partitions by whole parameters rather than flat slices."),
        ("ZeRO vs tensor vs pipeline parallelism: what does each split?",
         "**ZeRO (data parallel)** splits the batch, and shards the model *states*. Every GPU still computes "
         "the whole model on its own micro-batch, and communication is weights and gradients, ∝ Ψ, once per "
         "step. **Tensor parallelism** splits each layer's matrices: every GPU computes a slice of every matmul "
         "and exchanges activations with all-reduces inside every layer, so it needs NVLink and stays inside a "
         "node, and it also divides activation memory. **Pipeline parallelism** splits the layers into stages "
         "and sends activations point-to-point between them. It pays a pipeline bubble and needs many "
         "micro-batches, which is why it pairs with ZeRO-1 rather than ZeRO-2/3. Large runs combine all three "
         "(3D parallelism): TP inside a node, PP across nodes, ZeRO-1 across data-parallel replicas."),
    ]
    quiz = "\n\n".join(details(f"{i + 1}. {q}", a) for i, (q, a) in enumerate(qa))
    return f"""## Revision

A one-page version lives in [`CHEATSHEET.md`](CHEATSHEET.md).

### Cheat sheet

{cheat}

### Pitfalls worth remembering

{pitfalls}

### Self-quiz

Answer each out loud before opening it.

{quiz}"""


def sec_limitations(C: Ctx) -> str:
    gm = C.R.get("gpu_mode", {})
    audited = ("audited against the CUDA allocator in GPU mode in this run" if gm.get("stages")
               else "audited against the CUDA allocator only in GPU mode, which was skipped in this run")
    g = C.R.get("gloo", {})
    gcfg = g.get("config", {})
    return f"""## Honest limitations

- **Threads are not GPUs.** Memory is *accounted* per virtual GPU by the ledger, and {audited}.
  Everything else about a real GPU (kernels, streams, NCCL) is absent.
- **No communication/compute overlap, no prefetch.** Collectives block, and ZeRO-3 gathers each unit
  only when it is needed. Overlap is treated analytically (the α-β model and T*), not demonstrated.
- **Toy scale.** At {n(C.psi)} parameters, ZeRO-3's peak is dominated by activations and the one gathered
  unit, not by model states. The scaling and real-hardware sections cover the regime where ZeRO matters.
- **Bytes follow the ring cost model.** The counters charge each GPU what a ring would send; the data
  itself moves through shared memory. The gloo check {'(ZeRO-1, ' + str(gcfg.get('precision', 'fp32')) + ', ' + str(g.get('world')) + ' processes) confirms the numerics and the sequence of collectives on real `torch.distributed`' if g.get('status') == 'ok' else 'did not complete in this run, so real collectives were not exercised'}. It does not measure bytes on the wire or timing.
- **What the ledger does not book.** Each collective's fp32 scratch (one slice per reduce-scatter),
  autograd's transient input-gradient buffers between units, and allocator fragmentation. So `estimate_peak_bytes` is an upper bound of *this ledger's*
  peak, not of a real GPU's. The GPU-mode audit compares ledger and allocator at chosen moments; it
  does not prove the peaks agree.
- **CPU wall-clock is not a GPU proxy.** {C.N} threads on a CPU mostly measure lock contention
  and slow CPU bf16 kernels, so no timing claim here comes from wall-clock. Real-hardware times are
  modelled (α-β, fixed MFU), not measured.
- **Simplifications in the engine.** Micro-batch 1 per GPU; no dropout; AdamW applies weight decay to
  every element (norms included); no gradient clipping; per-unit (FSDP-style) sharding rather than
  DeepSpeed's global partition; every unit is released after forward (FSDP keeps the root gathered).
- **Real-hardware activations are approximate.** Korthikanti et al.'s count assumes a GPT-style MLP
  and full multi-head attention; LLaMA's SwiGLU and 70B's grouped-query attention differ slightly.
  "Fits" ignores fragmentation and communication buffers."""


def sec_reproduce(C: Ctx) -> str:
    run = C.run
    return f"""## Reproducing

```bash
pip install -r requirements.txt jupyter        # torch, matplotlib, psutil (+ jupyter to execute)

python -m unittest discover -s tests -t . -v   # collectives, ring, ledger, bit-exact stages, formulas
QUICK_RUN=1 python nb_source.py                # the whole notebook as a script, small model -> assets/quick/
python tools/build_notebook.py                 # nb_source.py -> zero_32_virtual_gpus.ipynb
QUICK_RUN=0 python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 zero_32_virtual_gpus.ipynb
python tools/build_readme.py                   # assets/results.json -> README.md (checks the run stamp)
python tools/gloo_check.py --world 2           # the ZeRO-1 engine on real torch.distributed (gloo)
```

On Windows PowerShell set the variable first (`$env:QUICK_RUN = "0"`), then run the same command.
Quick runs write only to `assets/quick/` (git-ignored), so exploring can never overwrite the committed
numbers, and `build_readme.py` refuses to build `README.md` from a quick run.

This README's run: Python {run.get('python')}, PyTorch {run.get('torch')}, {run.get('cpu_threads')} CPU threads,
seed {run.get('seed')}, {run.get('seconds_total', 0) / 60:.1f} minutes end to end, stamped `{run.get('timestamp')}`."""


def sec_files(C: Ctx) -> str:
    tree = """12_Distributed_ZeRO/
├── README.md                    this page, generated by tools/build_readme.py
├── CHEATSHEET.md                one page for revision
├── zero_32_virtual_gpus.ipynb   the executed notebook: every output and figure embedded
├── nb_source.py                 the notebook's single source (percent format; also runs as a script)
├── zero_sim.py                  the engine: VirtualGPU ledger, ThreadComm, ring all-reduce,
│                                VirtualCluster, TinyGPT units, DDP / ZeRO1 / ZeRO2 / ZeRO3, AdamW
├── zero_theory.py               closed forms: memory, communication, activations, α-β time,
│                                overlap threshold, HSDP, ZeRO++ hpZ, real-hardware table
├── zero_plots.py                the figures (dark theme)
├── tools/
│   ├── build_notebook.py        nb_source.py -> .ipynb
│   ├── build_readme.py          results.json -> README.md, with the run-stamp check
│   └── gloo_check.py            the same ZeRO-1 engine over real torch.distributed (gloo)
├── tests/
│   ├── test_comm.py             collectives at N = 3, 4, 32; byte counters; ring; ledger; failure handling
│   ├── test_zero_sim.py         bit-exact stages; memory = formula; comm; FLOPs; ZeRO-3 mechanics
│   └── test_zero_theory.py      paper Figure 1; formulas; activations; hardware; hybrid; decision guide
├── assets/                      results.json and the figures of the committed run
├── references.md                sources, and what was checked against each
└── requirements.txt"""
    return "## What is in here\n\n" + details("File map", f"```text\n{tree}\n```\n\n"
                                              "`data/` (tiny Shakespeare, downloaded on first run) and `assets/quick/` "
                                              "are git-ignored.")


def sec_refs(C: Ctx) -> str:
    return """## References

- Rajbhandari, Rasley, Ruwase, He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* 2019. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- Ren et al. *ZeRO-Offload: Democratizing Billion-Scale Model Training.* 2021. [arXiv:2101.06840](https://arxiv.org/abs/2101.06840)
- Rajbhandari, Ruwase, Rasley, Smith, He. *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning.* 2021. [arXiv:2104.07857](https://arxiv.org/abs/2104.07857)
- Wang et al. *ZeRO++: Extremely Efficient Collective Communication for Giant Model Training.* 2023. [arXiv:2306.10209](https://arxiv.org/abs/2306.10209)
- Zhao et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.* 2023. [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)
- Korthikanti et al. *Reducing Activation Recomputation in Large Transformer Models.* 2022. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
- Shoeybi et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* 2019. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models.* 2023. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- Patarasuk, Yuan. *Bandwidth optimal all-reduce algorithms for clusters of workstations.* J. Parallel Distrib. Comput. 69 (2009) 117–124. [doi:10.1016/j.jpdc.2008.09.002](https://doi.org/10.1016/j.jpdc.2008.09.002)
- Radford et al. *Language Models are Unsupervised Multitask Learners.* 2019 (GPT-2).
- NVIDIA [A100](https://www.nvidia.com/en-us/data-center/a100/) and [H100](https://www.nvidia.com/en-us/data-center/h100/) datasheets; [DGX A100](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf) and [DGX H100](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-h100-datasheet.pdf) datasheets.
- [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/) · [PyTorch `ZeroRedundancyOptimizer`](https://docs.pytorch.org/docs/stable/distributed.optim.html)

[`references.md`](references.md) records what was checked against each source (arXiv IDs,
the paper's Figure 1, parameter counts, hardware constants)."""


# =============================================================================================
# main
# =============================================================================================

def build(R: dict, asset_dir: str, quick: bool, stamp_status: str) -> str:
    C = Ctx(R, asset_dir, quick, stamp_status)
    sections = [sec_header(C), sec_tldr(C), sec_picture(C), sec_memory_bill(C), sec_machine(C),
                sec_collectives(C), sec_model(C), sec_stages(C), sec_results(C), sec_compute(C),
                sec_scaling(C), sec_accum_ckpt(C), sec_training(C), sec_stress(C), sec_hardware(C),
                sec_decision(C), sec_revision(C), sec_limitations(C), sec_reproduce(C), sec_files(C),
                sec_refs(C)]
    md = "\n\n---\n\n".join(s.strip() for s in sections) + "\n"
    return re.sub(r"\n{3,}", "\n\n", md)


def check_branding(md: str, sample: str | None):
    text = md.replace(sample, "") if sample else md
    hits = sorted({m.group(0) for m in BANNED.finditer(text)})
    if hits:
        fail(f"course branding words in the README: {hits}; rewrite the prose in the builder")


def check_links(md: str, quick: bool):
    """Every local file the README links to or shows must exist."""
    missing = []
    for target in re.findall(r'(?:src="|\]\()([^")#]+)', md):
        if target.startswith(("http://", "https://", "mailto:")) or not target.strip():
            continue
        if not (ROOT / target).exists():
            missing.append(target)
    missing = sorted(set(missing))
    if missing:
        msg = f"the README references missing local files: {missing}"
        if quick:
            warn(msg)
        else:
            fail(msg)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):                 # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="build README.quick.md from assets/quick/results.json (development only)")
    args = ap.parse_args(argv)
    if args.quick:
        asset_dir, out = "assets/quick", ROOT / "README.quick.md"
    else:
        asset_dir, out = "assets", ROOT / "README.md"
    res_path = ROOT / asset_dir / "results.json"
    if not res_path.exists():
        fail(f"{res_path.relative_to(ROOT)} not found; execute the notebook first")
    R = json.loads(res_path.read_text(encoding="utf-8"))
    if "run" not in R or "timestamp" not in R["run"]:
        fail(f"{res_path.relative_to(ROOT)} has no run stamp; it was not written by the notebook's last cell")
    if not args.quick and R["run"].get("quick_run"):
        fail("assets/results.json comes from a QUICK run (run.quick_run == true); refusing to build "
             "README.md from it. Re-execute the notebook with QUICK_RUN=0.")
    if args.quick and not R["run"].get("quick_run"):
        warn("assets/quick/results.json is not marked as a quick run")
    stamp = check_run_stamp(R, args.quick)
    md = build(R, asset_dir, args.quick, stamp)
    check_branding(md, R.get("training", {}).get("sample"))
    check_links(md, args.quick)
    out.write_text(md, encoding="utf-8", newline="\n")
    n_lines = md.count("\n")
    print(f"build_readme: wrote {out.name} ({n_lines:,} lines, {len(md):,} characters) from "
          f"{asset_dir}/results.json, RUN STAMP {R['run']['timestamp']} "
          f"(quick={R['run'].get('quick_run')}, stamp check: {stamp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
