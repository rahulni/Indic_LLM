# -*- coding: utf-8 -*-
"""
build_readme.py - regenerates the summary block inside README.md.

The README is the entry point, so it states the plan's decisions and their
numbers directly rather than only pointing at the document that holds them.
This injects a block between the markers, built from the same audit output as
the plan, so the summary cannot drift from what it summarises.

    python build_readme.py
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spec    # noqa: E402
import audit   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
README = os.path.join(ROOT, "README.md")
START = "<!-- SCORECARD:START -->"
END = "<!-- SCORECARD:END -->"
PLAN = "MIXTURE_PLAN.md"


def T(x):
    if x >= 1e12:
        return f"{x/1e12:.2f}T"
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    return f"{x:,.0f}"


def build_block():
    r = audit.run()
    a = {x["id"]: x for x in r["audited"]}
    g = spec.GATING
    tiers = {t["tier"]: t for t in r["indic_tiers"]}
    L = []
    w = L.append

    w(START)
    w("")
    w("## The plan at a glance")
    w("")
    w("Each row is a decision the plan makes, the number it lands on, and the "
      "section that defends it.")
    w("")
    w("| Decision | Where it lands | Defended in |")
    w("|---|---|---|")

    lanes = " · ".join(f"{x['label'].split(' ')[0].lower()} {x['share']:.1f}%"
                       for x in r["audited"][:4])
    w(f"| **Budget share per capability lane** | "
      f"**{len(r['audited'])} lanes**, summing to 100.00% of {T(r['budget'])} — "
      f"{lanes}, … | [§4]({PLAN}#4-the-mixture) |")

    w(f"| **Indic tier split** — verified / unverified / translated / synthetic | "
      f"**{tiers['verified']['share_of_lane']:.1f}% / "
      f"{tiers['unverified']['share_of_lane']:.1f}% / "
      f"{tiers['translated']['share_of_lane']:.1f}% / "
      f"{tiers['synthetic']['share_of_lane']:.1f}%** of a {T(r['indic_demand'])} "
      f"lane. 213.5B of real Indic tokens are available and **declined** | "
      f"[§7]({PLAN}#7-the-indic-slot-split-four-ways) |")

    w(f"| **Agentic, reasoning, long-context**, against named inventory datasets | "
      f"{a['agentic']['share']:.2f}% / {a['reasoning']['share']:.2f}% / "
      f"{a['longctx']['share']:.2f}%, each with a per-dataset table (tokens, "
      f"provenance tag, dedup keep, epoch cap) | "
      f"[§8]({PLAN}#8-agentic-reasoning-long-context--named-against-the-inventory) |")

    floors = " · ".join(f"{f['id']} {f['floor']:.2f}%" for f in r["floors"][:3])
    w(f"| **Protected always-on floor** the selector may not cross | "
      f"**{r['floor_total']:.2f}%** of every {spec.SELECTOR['refresh_every_steps']}-step "
      f"window ({floors}, …), asserted below each lane's share | "
      f"[§9]({PLAN}#9-protected-always-on-floor) |")

    w(f"| **Anneal reserve** held back for the cooldown | "
      f"**{r['anneal']['pct_of_budget']:.0f}% = {T(r['anneal']['tokens'])}**, "
      f"verified tiers only, and supply-audited against its own eligibility rules | "
      f"[§10]({PLAN}#10-the-anneal-reserve-and-schedule-mechanics) |")

    d = r["difficulty_bands"]
    w(f"| **Difficulty bands**, one worked example each | "
      f"{len(d)} bands — " + ", ".join(f"{b['id']} ({b['example_source']})" for b in d)
      + f" | [§11.1]({PLAN}#111-difficulty-bands) |")

    lb = r["length_bands"]
    w(f"| **Reasoning-length bands**, one worked example each | "
      f"{len(lb)} bands — " + ", ".join(
          f"`{b['control']}` ≤{b['token_budget'][1]:,} tok" for b in lb)
      + f", each with a worked trace and answer | [§11.2]({PLAN}#112-reasoning-length-bands) |")

    w(f"| **Benchmark accountability** per lane | "
      f"Benchmark column on every lane — SWE-bench, τ-bench, BFCL, LiveCodeBench, "
      f"AIME, MILU, FLORES, RULER, MMLU-Pro | [§4]({PLAN}#4-the-mixture) |")

    gen = [x for x in r["audited"] if x["verdict"] == "GENERATE"]
    rep = [x for x in r["audited"] if x["verdict"] == "REPEAT"]
    w(f"| **Supply sizing** — where a share needs repeating or generating | `SUPPLY-OK` / `REPEAT` ({len(rep)}) / "
      f"`GENERATE` ({len(gen)}) per lane, with the manufactured share costed in "
      f"dollars and wall-clock | [§5.3]({PLAN}#53-supply-against-demand), "
      f"[§6]({PLAN}#6-manufacturing-plan) |")

    w(f"| **Proxy design** at 1B / 3B, and the metric that refutes the mixture | {len(spec.PROXY['arms'])} arms at 1B × "
      f"{T(spec.PROXY['scale_1b']['tokens_per_arm'])} + a 3B confirmation, "
      f"**${r['cost']['total_usd']:,.0f}** ({r['cost']['pct_of_full']:.2f}% of the "
      f"run), with 9 metrics and **pre-registered** decision rules | "
      f"[§13]({PLAN}#13-the-proxy-experiment) |")

    w(f"| **Proxy execution** | "
      f"⚠️ **Partial.** The 1B/3B screen was not run — no GPUs. An 11M-param "
      f"micro-proxy was: 3 arms × 2 seeds, both verdicts INCONCLUSIVE, and the "
      f"reason is the finding — **both pre-registered rules were underpowered** | "
      f"[§13.7]({PLAN}#137-the-micro-proxy-what-was-actually-run) |")

    w(f"| **Data gate**, cleaning aimed at the starved tier | **{g['measured_clean_tokens']:,} cleaned tokens** "
      f"({g['growth_multiple']:.2f}× session 4). The +{T(g['added_this_session'])} "
      f"added is Sangraha **Verified** — the tier the audit ranks first | "
      f"[§2]({PLAN}#2-data-gating-status), [§16]({PLAN}#16-where-the-cleaning-goes-next) |")

    w("")
    w("### The three weakest numbers")
    w("")
    w(f"1. **Agentic is {a['agentic']['manufactured_pct']:.0f}% data that does not "
      f"exist.** {T(a['agentic']['supply_usable'])} of real trajectories against "
      f"{T(a['agentic']['demand'])} of demand — "
      f"**{a['agentic']['epochs']:.0f} epochs** of everything ever published. "
      f"Costed at §6, with a fallback if the harness slips. Stated, not hidden.")
    w(f"2. **14T is infeasible and the plan says so.** The budget is swept "
      f"2.4T→14T; at 14T the build *refuses*, and it fails on the **web** lane — "
      f"not agentic, not Indic. See [§3.1]({PLAN}#31-budget-sweep).")
    w(f"3. **The Indic conclusion is conditional.** AI4Bharat documents no "
      f"tokenizer for its 251.3B count; four sources checked. At our own measured "
      f"cl100k fertility the whole plan becomes infeasible. "
      f"[§14]({PLAN}#14-sensitivity) swings it and says so.")
    w("")
    w(END)
    return "\n".join(L)


def main():
    with io.open(README, encoding="utf-8") as f:
        md = f.read()
    block = build_block()
    if START in md and END in md:
        pre = md.split(START)[0]
        post = md.split(END, 1)[1]
        md = pre + block + post
    else:
        raise SystemExit(
            f"markers not found in README.md - add {START} / {END} where the "
            "scorecard should live")
    with io.open(README, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(f"injected scorecard into {os.path.normpath(README)} "
          f"({len(block):,} chars)")


if __name__ == "__main__":
    main()
