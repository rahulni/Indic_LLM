"""Sweep how much in-domain (related-article) data to add, to find the point that
minimises English (and Telugu) fertility before budget-dilution reverses it."""
import sys
from train_all import run, CANDIDATES, NAMES

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def P(**kw):
    base = {"name": "x", "unit_mode": "akshara", "strip_joiners": True, "cap_chars": None,
            "augment": False, "aug_chars": None, "aug_indomain": False,
            "min_pair_freq": 2, "morph": False, "translit": False,
            "key": "x", "label": "x", "kind": "honest", "note": ""}
    base.update(kw)
    return base


SWEEP = [0, 15000, 30000, 60000, 120000]

for cand in ["es", "kn"]:   # representative: best candidate + smallest 4th
    print(f"\n#### candidate = {NAMES[cand]}")
    print(f"{'aug_chars':>9s} {'Eng':>6s} {'Hin':>6s} {'Tel':>6s} {'4th':>6s} {'spread':>7s} {'score':>8s}")
    for n in SWEEP:
        prof = P(augment=(n > 0), aug_chars=(n if n > 0 else None))
        r = run(cand, prof, keep_tokens=False)
        x = {l["code"]: l["fertility"] for l in r["languages"]}
        print(f"{n:9d} {x['en']:6.3f} {x['hi']:6.3f} {x['te']:6.3f} {x[cand]:6.3f} "
              f"{r['spread']:7.3f} {str(r['score']):>8s}")
