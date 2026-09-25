"""Second ladder: the same walk with a chunked cross entropy, plus a flakiness re-probe.

Once activations stop scaling with depth, the fp32 logits are the largest single tensor in
the step. If that is really what caps the batch size, then chunking the loss - recomputing
each chunk's logits in backward so only one is ever live - should move the cap a long way.
This script runs that comparison and appends it to assets/ladder.json.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revlm.ladder import _run_probe, ladder

lad = json.load(open("assets/ladder.json"))

print("re-probing the boundary (the OOM edge has fragmentation noise, so repeat it)")
repeats = {}
for mode, batch in (("midpoint", 112), ("euler", 112), ("euler", 128), ("midpoint", 128)):
    hits = [_run_probe(mode, batch, 10, 1) is not None for _ in range(3)]
    repeats[f"{mode}@{batch}"] = hits
    print(f"  {mode}@{batch}: fitted {sum(hits)}/3 attempts")

print("\nchunked cross entropy (ce_chunks=4)")
grid = [48, 96, 128, 160, 192, 224, 256, 320]
chunked = ladder(["store", "euler", "midpoint"], grid, ce_chunks=4)

lad["boundary_repeats"] = repeats
lad["chunked"] = chunked
json.dump(lad, open("assets/ladder.json", "w"), indent=2)
print("\nmax batch with chunked CE:", chunked["max_batch"])
print("max batch without        :", lad["max_batch"])
