import json, sys
sys.stdout.reconfigure(encoding="utf-8")
d = json.load(open("results.json", encoding="utf-8"))
for cand in ["es", "kn"]:
    print(f"\n=== {cand} ===")
    print(f'{"profile":14s} {"kind":10s} {"en":>6s} {"hi":>6s} {"te":>6s} {"4th":>6s} {"score":>8s}')
    for p in d["results"][cand]["profiles"]:
        x = p["langX"]
        print(f'{p["key"]:14s} {p["kind"]:10s} {x["en"]:6.3f} {x["hi"]:6.3f} {x["te"]:6.3f} {x[cand]:6.3f} {str(p["score"]):>8s}')
# primary (submitted) per-language words + fertility + baseline delta
print("\n-- SUBMITTED (te_morph) primary, per language, es candidate --")
r = d["results"]["es"]
print(f'{"lang":8s} {"words":>7s} {"X":>6s} {"baselineX":>9s}')
for l in r["languages"]:
    print(f'{l["name"]:8s} {l["words"]:7d} {l["fertility"]:6.3f} {l["baseline_fertility"]:9.3f}')
print("best:", d["meta"]["best_candidate"], d["meta"]["best_score"], "primary=", d["meta"]["primary_profile"])
