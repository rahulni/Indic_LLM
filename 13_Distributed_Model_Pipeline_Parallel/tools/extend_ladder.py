"""Re-probe the boundary after the midpoint allocation fix, and find the real chunked ceiling."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revlm.ladder import _run_probe, ladder

lad = json.load(open("assets/ladder.json"))
print("boundary re-probe after removing midpoint's extra temporary")
rep = {}
for mode, b in (("midpoint", 112), ("midpoint", 128), ("euler", 128)):
    hits = [_run_probe(mode, b, 10, 1) is not None for _ in range(3)]
    rep[f"{mode}@{b}"] = hits
    print(f"  {mode}@{b}: fitted {sum(hits)}/3")
    if any(hits):
        r = _run_probe(mode, b, 10, 1)
        if r:
            lad["rows"].append(r)
            lad["max_batch"][mode] = max(lad["max_batch"].get(mode, 0), b)

print("\nchunked CE, extending until it really stops")
ext = ladder(["euler", "midpoint"], [384, 448, 512, 576], ce_chunks=4)
for r in ext["rows"]:
    lad["chunked"]["rows"].append(r)
for m, b in ext["max_batch"].items():
    lad["chunked"]["max_batch"][m] = max(lad["chunked"]["max_batch"].get(m, 0), b)

lad["boundary_repeats_after_fix"] = rep
json.dump(lad, open("assets/ladder.json", "w"), indent=2)
print("\nunchunked max:", lad["max_batch"])
print("chunked   max:", lad["chunked"]["max_batch"])
