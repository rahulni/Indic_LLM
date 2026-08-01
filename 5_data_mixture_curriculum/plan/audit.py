# -*- coding: utf-8 -*-
"""
audit.py - turns spec.py into numbers, and refuses to produce numbers that
do not close.

Every invariant here is a claim the plan makes about itself. If one fails the
build stops, because a mixture plan whose arithmetic does not close is exactly
the wishful accounting this assignment is designed to catch.
"""

import copy
import math

import spec


class InvariantError(AssertionError):
    pass


def effective_multiplier(epochs):
    """[REVIEW FIX 7] Muennighoff et al. 2023, eq. 6:

        D' = U + U * R* * (1 - exp(-R / R*)),   R = epochs - 1

    Returns D'/U: the effective-token yield of `epochs` passes over a corpus,
    where the first pass is worth 1.0 and every later one is worth less.

    At the 4.0 cap this returns ~3.73, not 4.0 - so the last epoch of a capped
    lane is worth roughly a quarter of a fresh one. The previous draft priced
    all four at par."""
    if epochs <= 1.0:
        return max(0.0, epochs)
    rs = spec.EPOCH_DECAY["r_star_data"]
    r = epochs - 1.0
    return 1.0 + rs * (1.0 - math.exp(-r / rs))


def _check(cond, msg):
    if not cond:
        raise InvariantError(msg)


# ---------------------------------------------------------------------------
# Phases and lanes
# ---------------------------------------------------------------------------

def phase_table(budget):
    rows = []
    for i, (pid, label, weight, why) in enumerate(spec.PHASES):
        rows.append(dict(idx=i, id=pid, label=label, weight=weight,
                         tokens=weight * budget, why=why))
    total = sum(r["weight"] for r in rows)
    _check(abs(total - 1.0) < 1e-9, f"phase weights sum to {total}, not 1.0")
    return rows


def lane_selector_multiplier(lane_id, lane_share):
    """[REVIEW FIX 4] How many CANDIDATE tokens the corpus must supply per
    trained token, for a lane holding `lane_share` percent of the budget.

    Data admitted under the protected floor bypasses the selector and is
    consumed 1:1. Everything above the floor is drawn at 1/keep_fraction,
    because the selector discards the remainder and never re-offers it
    (session 5, lines 463-465). A lane that is mostly floor barely notices;
    a lane with no floor pays the full 2x."""
    if lane_share <= 0:
        return 1.0
    kf = spec.SELECTOR["keep_fraction"]
    floor = min(spec.FLOOR.get(lane_id, {}).get("pct", 0.0), lane_share)
    above = lane_share - floor
    return (floor + above / kf) / lane_share


def lane_table(phases, budget):
    """Whole-run share is DERIVED as the token-weighted average of the phase
    shares. It is an output of the curriculum, never an independent input."""
    weights = [p["weight"] for p in phases]

    # Each phase column must sum to 100.
    for i, p in enumerate(phases):
        col = sum(v["phases"][i] for v in spec.LANES.values())
        _check(abs(col - 100.0) < 1e-6,
               f"phase {p['id']} lane shares sum to {col}, not 100")

    rows = []
    for lid, lane in spec.LANES.items():
        share = sum(lane["phases"][i] * weights[i] for i in range(len(phases)))
        tokens = share / 100.0 * budget
        lb = spec.LOSS_BEARING[lid]
        mult = lane_selector_multiplier(lid, share)
        rows.append(dict(
            id=lid, label=lane["label"], phases=lane["phases"],
            share=share, tokens=tokens,
            loss_bearing_frac=lb, loss_bearing_tokens=tokens * lb,
            selector_mult=mult, candidate_tokens=tokens * mult,
            benchmarks=spec.LANE_BENCHMARKS[lid],
        ))

    total_share = sum(r["share"] for r in rows)
    _check(abs(total_share - 100.0) < 1e-6,
           f"whole-run lane shares sum to {total_share}, not 100")
    total_tokens = sum(r["tokens"] for r in rows)
    _check(abs(total_tokens - budget) < 1e3,
           f"lane tokens sum to {total_tokens}, not the {budget} budget")

    rows.sort(key=lambda r: -r["share"])
    return rows


# ---------------------------------------------------------------------------
# Supply
# ---------------------------------------------------------------------------

def resolve_units(item):
    """[REVIEW FIX 3] Which tokenizer and content class a published count is in.

    Anything we DERIVED or ESTIMATED ourselves is already in our own units - we
    invented the mean tokens-per-sample, so we declare the tokenizer. Only
    PUBLISHED counts need converting."""
    src_default, cls_default = spec.LANE_UNITS_DEFAULT[item["lane"]]
    cls = item.get("content_class", cls_default)
    if item["provenance"] in spec.SELF_UNIT_PROVENANCE:
        return "ours", cls
    return item.get("src_tokenizer", src_default), cls


def to_our_tokens(item):
    """Convert a published token count into our tokenizer's units via words.

        words        = tokens_src / fertility[src][class]
        tokens_ours  = words * fertility[ours][class]

    Epoch counts are only meaningful when supply and demand share units, and
    the 14T budget is denominated in our tokens."""
    src, cls = resolve_units(item)
    f_src = spec.TOKENIZER_FERTILITY[src][cls]
    f_ours = spec.TOKENIZER_FERTILITY["ours"][cls]
    factor = f_ours / f_src
    words = item["tokens"] / f_src
    return dict(tokens_src=item["tokens"], src_tokenizer=src, content_class=cls,
                fertility_src=f_src, fertility_ours=f_ours,
                factor=factor, words=words, tokens_ours=item["tokens"] * factor)


def supply_by_lane():
    """Usable supply = converted tokens x cross-corpus dedup survival.
       Capped supply = usable x the per-source epoch cap.

    Everything downstream of here is in OUR tokenizer's units."""
    out = {}
    for item in spec.INVENTORY:
        lane = item["lane"]
        conv = to_our_tokens(item)
        usable = conv["tokens_ours"] * item["dedup_keep"]
        capped = usable * item["epoch_cap"]
        d = out.setdefault(lane, dict(raw=0.0, raw_src=0.0, usable=0.0,
                                      capped=0.0, sources=[]))
        d["raw"] += conv["tokens_ours"]
        d["raw_src"] += item["tokens"]
        d["usable"] += usable
        d["capped"] += capped
        d["sources"].append(dict(item, **conv, usable=usable, capped=capped))
    return out


def audit_lanes(lanes, supply):
    """The core supply-vs-demand audit. Produces a verdict per lane and, where
    the verdict is GENERATE, the exact number of tokens that must be
    manufactured rather than found."""
    rows = []
    for lane in lanes:
        s = supply.get(lane["id"], dict(raw=0.0, usable=0.0, capped=0.0, sources=[]))
        # [REVIEW FIX 4] Supply must cover CANDIDATE tokens, not trained tokens.
        # The selector discards what it does not keep and never re-offers it, so
        # a lane that trains D tokens draws D x its selector multiplier from the
        # corpus. Auditing against trained tokens understates every lane.
        demand = lane["candidate_tokens"]
        trained = lane["tokens"]
        usable = s["usable"]
        capped = s["capped"]

        epochs = demand / usable if usable else float("inf")
        manufactured = max(0.0, demand - capped)
        found = demand - manufactured

        if usable == 0:
            verdict = "GENERATE"
        elif epochs <= 1.0:
            verdict = "SUPPLY-OK"
        elif manufactured <= 0:
            verdict = "REPEAT"
        else:
            verdict = "GENERATE"

        # [REVIEW FIX 7] Physical feasibility is unchanged - you still consume
        # real passes over the corpus. What the decay changes is the VALUE those
        # passes deliver.
        #
        # Epochs actually RUN on real data is capped: past the cap the shortfall
        # is manufactured, not repeated further. Manufactured tokens are fresh
        # by construction, so they carry no repetition discount.
        lane_cap = (capped / usable) if usable else 0.0
        epochs_run = min(epochs, lane_cap) if usable else 0.0
        effective_real = usable * effective_multiplier(epochs_run) if usable else 0.0
        effective = effective_real + manufactured
        eff_ratio = effective / demand if demand else 1.0

        rows.append(dict(
            **lane,
            supply_raw=s["raw"], supply_usable=usable, supply_capped=capped,
            demand=demand, trained=trained, epochs=epochs,
            effective_tokens=effective, effective_ratio=eff_ratio,
            repetition_loss=max(0.0, found - effective),
            found=found, manufactured=manufactured,
            manufactured_pct=100.0 * manufactured / demand if demand else 0.0,
            verdict=verdict, sources=s["sources"],
        ))
    return rows


# ---------------------------------------------------------------------------
# Indic tiers
# ---------------------------------------------------------------------------

def indic_tiers(audited):
    """The four-tier split, allocated rather than assumed.

    Lane demand is filled tier by tier in INDIC_TIER_PRIORITY order - real
    human-authored text before machine translation, synthetic last. Each tier
    contributes at most its own capped supply. Whatever demand survives all
    four is the synthetic tier, which is the only honest way to size a tier
    that does not exist yet.

    This replaces an earlier version that assumed scarcity and treated
    synthetic as the residual unconditionally. At the 3T budget the lane is in
    surplus, and an audit that cannot represent surplus would have forced a
    manufactured tier the plan does not need."""
    lane = next(r for r in audited if r["id"] == "indic")
    demand = lane["demand"]

    capped_by_tier = {}
    srcs_by_tier = {}
    for src in lane["sources"]:
        t = src["tier"]
        capped_by_tier[t] = capped_by_tier.get(t, 0.0) + src["capped"]
        srcs_by_tier.setdefault(t, []).append(src)

    rows = []
    remaining = demand
    for t in spec.INDIC_TIER_PRIORITY:
        if t == "synthetic":
            take = remaining
            manufactured = True
        else:
            avail = capped_by_tier.get(t, 0.0)
            take = min(avail, remaining)
            manufactured = False
        remaining -= take
        rows.append(dict(
            tier=t, tokens=take,
            available=capped_by_tier.get(t, 0.0),
            unused=max(0.0, capped_by_tier.get(t, 0.0) - take),
            share_of_lane=100.0 * take / demand if demand else 0.0,
            manufactured=manufactured, sources=srcs_by_tier.get(t, []),
        ))

    _check(abs(remaining) < 1e3,
           f"Indic tier allocation left {remaining:.3e} tokens unassigned")
    total = sum(r["share_of_lane"] for r in rows)
    _check(abs(total - 100.0) < 1e-6, f"Indic tiers sum to {total}, not 100")
    return rows, demand


# ---------------------------------------------------------------------------
# Floor and reserve
# ---------------------------------------------------------------------------

def floor_table(lanes):
    by_id = {l["id"]: l for l in lanes}
    rows = []
    for lid, f in spec.FLOOR.items():
        lane = by_id[lid]
        _check(f["pct"] <= lane["share"] + 1e-9,
               f"floor for {lid} ({f['pct']}%) exceeds its whole-run share "
               f"({lane['share']:.2f}%) - the floor would be unreachable")
        rows.append(dict(id=lid, label=lane["label"], floor=f["pct"],
                         whole_run=lane["share"], headroom=lane["share"] - f["pct"],
                         note=f["note"]))
    total = sum(r["floor"] for r in rows)
    rows.sort(key=lambda r: -r["floor"])
    return rows, total


def anneal_supply_audit(phases, lanes, supply, audited):
    """[REVIEW FIX 2] Does enough data actually pass the reserve's own rules?

    The first draft declared eligibility rules for the anneal and never checked
    them. Here the reserve's per-lane demand is tested against the supply that
    survives those rules, at a maximum of one epoch - repeating data during a
    cooldown defeats the purpose of a cooldown."""
    pd = phases[-1]
    by_id = {a["id"]: a for a in audited}
    rows = []
    for lane in lanes:
        share = lane["phases"][pd["idx"]]
        if share <= 0:
            continue
        demand = share / 100.0 * pd["tokens"]
        y = spec.ANNEAL_YIELD[lane["id"]]
        a = by_id[lane["id"]]

        if lane["id"] == "indic":
            # The tier restriction IS the filter: verified only, one epoch.
            base = sum(s["usable"] for s in a["sources"] if s.get("tier") == "verified")
        elif lane["id"] == "agentic":
            # Manufactured rollouts are verifier-gated at generation, so they
            # clear at a higher rate than scraped trajectories.
            base = a["supply_usable"] + a["manufactured"]
        else:
            base = a["supply_usable"] + a["manufactured"]

        eligible = base * y["frac"] * spec.ANNEAL_MAX_EPOCHS
        ok = eligible >= demand
        rows.append(dict(id=lane["id"], label=lane["label"], share=share,
                         demand=demand, pool=base, yield_frac=y["frac"],
                         eligible=eligible, ok=ok,
                         shortfall=max(0.0, demand - eligible),
                         basis=y["basis"]))
    rows.sort(key=lambda r: -r["demand"])
    failures = [r for r in rows if not r["ok"]]
    _check(not failures,
           "anneal reserve cannot be stocked under its own eligibility rules: "
           + "; ".join(f"{f['id']} short {f['shortfall']/1e9:.1f}B "
                       f"(demand {f['demand']/1e9:.1f}B vs eligible "
                       f"{f['eligible']/1e9:.1f}B)" for f in failures))
    return dict(rows=rows, failures=failures,
                total_shortfall=sum(r["shortfall"] for r in rows))


def proxy_pools(audited, budget):
    """[REVIEW FIX 1] Per-lane candidate-pool sizes for the proxy, subsampled by
    the same factor as the budget so proxy epochs match full-run epochs."""
    p = spec.PROXY["scale_1b"]
    factor = p["tokens_per_arm"] / budget
    rows = []
    for a in audited:
        pool = a["supply_usable"] * factor
        demand = a["demand"] * factor
        rows.append(dict(id=a["id"], label=a["label"],
                         pool=pool, demand=demand,
                         epochs=demand / pool if pool else float("inf"),
                         full_epochs=a["epochs"]))
        # The whole point: proxy epochs must equal full-run epochs.
        if pool:
            _check(abs(rows[-1]["epochs"] - a["epochs"]) < 1e-6,
                   f"proxy epochs for {a['id']} ({rows[-1]['epochs']:.3f}) do not "
                   f"match full-run epochs ({a['epochs']:.3f}) - corpus scaling is broken")
    return dict(factor=factor, rows=rows,
                tokens_per_param_proxy=p["tokens_per_arm"] / p["params"],
                tokens_per_param_full=budget / spec.MODEL_PARAMS)


def anneal_table(phases, lanes):
    """The anneal reserve is Phase D. Its composition is not declared
    separately - it IS the Phase D column of the lane table, which is what
    stops the reserve from being a second, unreconciled set of numbers."""
    pd = phases[-1]
    _check(pd["id"] == "D", "last phase is not the anneal")
    rows = []
    for lane in lanes:
        share = lane["phases"][pd["idx"]]
        if share <= 0:
            continue
        rows.append(dict(id=lane["id"], label=lane["label"], share=share,
                         tokens=share / 100.0 * pd["tokens"],
                         pct_of_lane_lifetime=100.0 * (share / 100.0 * pd["tokens"]) / lane["tokens"]))
    rows.sort(key=lambda r: -r["share"])
    return dict(tokens=pd["tokens"], pct_of_budget=pd["weight"] * 100.0, rows=rows)


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

def count_4digit_digitsum(target=20):
    """The D-high worked example's answer, computed rather than asserted."""
    n = 0
    for d1 in range(1, 10):
        for d2 in range(10):
            for d3 in range(10):
                d4 = target - d1 - d2 - d3
                if 0 <= d4 <= 9:
                    n += 1
    return n


def band_tables(phases):
    for b in spec.DIFFICULTY_BANDS:
        _check(len(b["phase_mix"]) == len(phases), "difficulty band phase_mix length mismatch")
    for i, p in enumerate(phases):
        col = sum(b["phase_mix"][i] for b in spec.DIFFICULTY_BANDS)
        _check(abs(col - 100.0) < 1e-6,
               f"difficulty bands in phase {p['id']} sum to {col}, not 100")

    share = sum(b["share_of_reasoning_lane"] for b in spec.LENGTH_BANDS)
    _check(abs(share - 100.0) < 1e-6, f"reasoning-length bands sum to {share}, not 100")

    bands = [dict(b) for b in spec.LENGTH_BANDS]
    for b in bands:
        if b["example_a"] == "COMPUTED_AT_BUILD":
            b["example_a"] = str(count_4digit_digitsum(20))
    return spec.DIFFICULTY_BANDS, bands


# ---------------------------------------------------------------------------
# Manufacturing
# ---------------------------------------------------------------------------

def manufacturing_table(audited):
    """Prices every shortfall. A lane that must generate tokens and cannot say
    what that costs has not been planned, only wished for."""
    c = spec.PROXY["cluster"]
    rows = []
    for a in audited:
        need = a["manufactured"]
        if need <= 0:
            continue
        m = spec.MANUFACTURING.get(a["id"])
        _check(m is not None,
               f"lane {a['id']} needs {need:.3e} manufactured tokens but has no "
               "entry in MANUFACTURING - the plan would be wishing")

        units = need / m["mean_tokens_per_unit"]
        generated = need * m["generated_frac"]

        if m["gpus"] and m["generator_params"]:
            rate = m["gpus"] * c["peak_bf16_flops"] * c["mfu"]
            flops = 2.0 * m["generator_params"] * generated   # inference: 2*N*T
            gpu_hours = flops / rate / 3600.0 * m["gpus"]
            usd = gpu_hours * c["usd_per_gpu_hour"]
            infer_days = flops / rate / 86400.0
        else:
            gpu_hours = usd = infer_days = 0.0

        env_days = (units * m["env_seconds_per_unit"] / m["parallel_workers"] / 86400.0
                    if m["parallel_workers"] else 0.0)

        rows.append(dict(
            id=a["id"], label=a["label"], kind=m["kind"], method=m["method"],
            need=need, need_pct=a["manufactured_pct"], units=units,
            generated=generated, gpu_hours=gpu_hours, usd=usd,
            infer_days=infer_days, env_days=env_days,
            wall_days=max(infer_days, env_days),
            risk=m["risk"], fallback=m["fallback"],
        ))
    rows.sort(key=lambda r: -r["need"])
    total_usd = sum(r["usd"] for r in rows)
    return rows, total_usd


# ---------------------------------------------------------------------------
# Proxy cost
# ---------------------------------------------------------------------------

def _flops(params, tokens):
    return 6.0 * params * tokens


def proxy_cost(budget):
    c = spec.PROXY["cluster"]
    rate = c["gpus"] * c["peak_bf16_flops"] * c["mfu"]

    def arm(scale):
        f = _flops(scale["params"], scale["tokens_per_arm"])
        secs = f / rate
        gpu_hours = secs / 3600.0 * c["gpus"]
        return dict(flops=f, hours=secs / 3600.0, gpu_hours=gpu_hours,
                    usd=gpu_hours * c["usd_per_gpu_hour"])

    a1 = arm(spec.PROXY["scale_1b"])
    a3 = arm(spec.PROXY["scale_3b"])
    n1 = len(spec.PROXY["arms"])
    n3 = 2  # baseline + winner only

    fr = spec.PROXY["full_run"]
    full_flops = _flops(spec.MODEL_PARAMS, budget)
    full_rate = fr["gpus"] * c["peak_bf16_flops"] * fr["mfu"]
    full_secs = full_flops / full_rate
    full_gpu_hours = full_secs / 3600.0 * fr["gpus"]
    full_usd = full_gpu_hours * fr["usd_per_gpu_hour"]

    total = a1["usd"] * n1 + a3["usd"] * n3
    return dict(
        arm_1b=a1, arm_3b=a3, n_1b=n1, n_3b=n3,
        total_usd=total,
        full_run=dict(flops=full_flops, days=full_secs / 86400.0,
                      gpu_hours=full_gpu_hours, usd=full_usd),
        pct_of_full=100.0 * total / full_usd,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(budget=None):
    budget = budget or spec.BUDGET_TOKENS
    phases = phase_table(budget)
    lanes = lane_table(phases, budget)
    supply = supply_by_lane()
    audited = audit_lanes(lanes, supply)
    tiers, indic_demand = indic_tiers(audited)
    floors, floor_total = floor_table(lanes)
    anneal = anneal_table(phases, lanes)
    dbands, lbands = band_tables(phases)
    cost = proxy_cost(budget)
    manu, manu_usd = manufacturing_table(audited)
    an_supply = anneal_supply_audit(phases, lanes, supply, audited)
    pools = proxy_pools(audited, budget)

    _check(floor_total < 100.0, "protected floor consumes the entire batch")

    # Lanes sitting within 5% of their epoch ceiling are one bad assumption
    # away from needing manufactured tokens. Surface them rather than letting
    # the REPEAT verdict imply comfort.
    at_ceiling = []
    for a in audited:
        cap = max((s["epoch_cap"] for s in a["sources"]), default=0)
        if cap and a["epochs"] > 0.95 * cap and a["manufactured"] <= 0:
            at_ceiling.append(dict(id=a["id"], label=a["label"],
                                   epochs=a["epochs"], cap=cap))

    ladder = seq_ladder_table()

    return dict(budget=budget, phases=phases, lanes=lanes, audited=audited,
                seq_ladder=ladder,
                indic_tiers=tiers, indic_demand=indic_demand,
                floors=floors, floor_total=floor_total,
                anneal=anneal, anneal_supply=an_supply,
                difficulty_bands=dbands, length_bands=lbands,
                cost=cost, manufacturing=manu, manufacturing_usd=manu_usd,
                proxy_pools=pools, at_ceiling=at_ceiling)


def _apply_sensitivity(s, value):
    """Mutate spec in place for one sensitivity bound. Returns an undo thunk."""
    if s["kind"] == "fertility":
        tok, cls = s["target"]
        old = spec.TOKENIZER_FERTILITY[tok][cls]
        spec.TOKENIZER_FERTILITY[tok][cls] = value
        return lambda: spec.TOKENIZER_FERTILITY[tok].__setitem__(cls, old)
    if s["kind"] == "dedup":
        olds = [(i, i["dedup_keep"]) for i in spec.INVENTORY if i["lane"] == s["target"]]
        for item, _ in olds:
            item["dedup_keep"] = value
        return lambda: [item.__setitem__("dedup_keep", o) for item, o in olds]
    if s["kind"] == "source_tokens":
        item = next(i for i in spec.INVENTORY if i["name"] == s["target"])
        old = item["tokens"]
        item["tokens"] = value
        return lambda: item.__setitem__("tokens", old)
    raise ValueError(f"unknown sensitivity kind {s['kind']}")


def sensitivity_analysis(budget=None):
    """[REVIEW ITEM] Re-run the whole plan at each assumption's bounds.

    Prose reasoning about 'this estimate could be wrong' is worth nothing. This
    actually swings each parameter to its plausible extremes and reports which
    conclusions move and which hold."""
    budget = budget or spec.BUDGET_TOKENS
    out = []
    for s in spec.SENSITIVITY:
        row = dict(id=s["id"], label=s["label"], basis=s["basis"],
                   breaks=s["breaks"], base=s["base"], low=s["low"],
                   high=s["high"], results={})
        for bound in ("low", "base", "high"):
            undo = _apply_sensitivity(s, s[bound])
            try:
                r = run(budget)
                lanes = {a["id"]: a for a in r["audited"]}
                lane_id = ("indic" if "sangraha" in s["id"] else
                           "web" if "web" in s["id"] else
                           "agentic" if "agentic" in s["id"] else "civic")
                a = lanes[lane_id]
                row["results"][bound] = dict(
                    feasible=True, lane=lane_id, epochs=a["epochs"],
                    manufactured_pct=a["manufactured_pct"], verdict=a["verdict"],
                    manufacturing_usd=r["manufacturing_usd"])
            except InvariantError as e:
                row["results"][bound] = dict(feasible=False, error=str(e)[:160])
            finally:
                undo()
        lo = row["results"]["low"]
        hi = row["results"]["high"]
        row["swings_conclusion"] = (
            not lo.get("feasible", True) or not hi.get("feasible", True)
            or lo.get("verdict") != hi.get("verdict")
            or abs(lo.get("manufactured_pct", 0) - hi.get("manufactured_pct", 0)) > 20.0
        )
        out.append(row)
    return out


def seq_ladder_table():
    """[TRANSCRIPT REQUIREMENT] Validate the sequence-length ladder."""
    rows = [dict(r) for r in spec.SEQ_LADDER]
    for r in rows:
        _check(r["seq_len"] >= spec.SEQ_MIN,
               f"sequence length {r['seq_len']} is below the {spec.SEQ_MIN} floor - "
               "padding shorter samples wastes compute")
        r["tokens_per_batch"] = r["seq_len"] * r["batch_examples"]
    by_phase = {}
    for r in rows:
        by_phase.setdefault(r["phase"], 0)
        by_phase[r["phase"]] += r["share_of_phase"]
    for ph, tot in by_phase.items():
        _check(abs(tot - 100.0) < 1e-6,
               f"sequence ladder shares in phase {ph} sum to {tot}, not 100")
    if spec.SEQ_CONSTANT_TOKENS_PER_BATCH:
        tpb = {r["tokens_per_batch"] for r in rows}
        _check(len(tpb) == 1,
               f"tokens-per-batch is not constant across the ladder ({sorted(tpb)}); "
               "every length change would also be a batch-size change and a loss "
               "spike could not be attributed to either")
    return rows


def sweep():
    """[REVIEW FIX 4/5] Audit the same mixture across every candidate budget.

    A scenario that raises InvariantError is not a bug - it is the finding. It
    means the mixture cannot be financed at that budget under its own rules, and
    the sweep records why instead of crashing the build."""
    out = []
    for budget, note in spec.BUDGET_SCENARIOS:
        try:
            r = run(budget)
            worst = max(r["audited"], key=lambda a: a["manufactured_pct"])
            out.append(dict(
                budget=budget, note=note, feasible=True, error=None,
                manufacturing_usd=r["manufacturing_usd"],
                full_run_usd=r["cost"]["full_run"]["usd"],
                full_run_days=r["cost"]["full_run"]["days"],
                generate_lanes=sum(1 for a in r["audited"] if a["verdict"] == "GENERATE"),
                at_ceiling=len(r["at_ceiling"]),
                worst_lane=worst["id"], worst_pct=worst["manufactured_pct"],
                epochs={a["id"]: a["epochs"] for a in r["audited"]},
            ))
        except InvariantError as e:
            out.append(dict(budget=budget, note=note, feasible=False,
                            error=str(e), manufacturing_usd=None,
                            full_run_usd=None, full_run_days=None,
                            generate_lanes=None, at_ceiling=None,
                            worst_lane=None, worst_pct=None, epochs={}))
    return out


if __name__ == "__main__":
    r = run()
    print("invariants: OK")
    print(f"lanes: {len(r['lanes'])}  floor total: {r['floor_total']:.2f}%  "
          f"anneal: {r['anneal']['pct_of_budget']:.1f}%  "
          f"proxy: ${r['cost']['total_usd']:,.0f} "
          f"({r['cost']['pct_of_full']:.3f}% of the full run)")
    for a in r["audited"]:
        print(f"  {a['id']:10s} {a['share']:5.2f}%  {a['verdict']:10s} "
              f"epochs={a['epochs']:.2f}  manufactured={a['manufactured_pct']:.1f}%")
