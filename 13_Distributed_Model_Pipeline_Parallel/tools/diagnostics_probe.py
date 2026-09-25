"""Short runs whose purpose is the diagnostics, not the loss.

The full 50M-token runs answer "which integrator trains better". They do not explain *why*
midpoint trails, or whether implicit Euler's contraction condition holds up under training.
Those need the per-step diagnostics tracked densely through a run, which is cheap - so they
get their own 5M-token probes rather than another two hours of GPU.

Labelled as probes everywhere they are reported: a tenth of the token budget.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revlm import data as D
from revlm.train import RunSpec, train_one

meta = D.prepare("data", log=lambda *a: None)
out = {}
for mode in ("euler", "midpoint", "euler_implicit", "coupling"):
    spec = RunSpec(name=f"probe_{mode}", mode=mode, batch_size=48, tokens=5_000_000,
                   euler_iters=4, note="diagnostics probe, 5M tokens")
    r = train_one(spec, meta, "assets")
    out[mode] = {"history": r["history"], "diagnostics": r["diagnostics"],
                 "final_val_loss": r["final_val_loss"], "tokens": r["tokens_seen"]}
json.dump(out, open("assets/diagnostics.json", "w"), indent=2)
print("\nwrote assets/diagnostics.json")
for m, v in out.items():
    dg = v["diagnostics"]
    rec = [d["recon_h0"] for d in dg if "recon_h0" in d]
    print(f"  {m:<16} {len(dg)} samples, reconstruction {min(rec):.2e} -> {max(rec):.2e}")
