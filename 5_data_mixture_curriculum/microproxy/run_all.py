# -*- coding: utf-8 -*-
"""run_all.py - every arm x every seed, sequentially, on one GPU."""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "..", ".venv", "Scripts", "python.exe")

ARMS = ["A0", "A2", "A6"]
SEEDS = [1337, 2024]
STEPS = 2000


def main():
    t0 = time.time()
    todo = [(a, s) for s in SEEDS for a in ARMS]
    for i, (arm, seed) in enumerate(todo, 1):
        out = os.path.join(HERE, "runs", f"{arm}_seed{seed}.json")
        if os.path.exists(out) and os.path.getsize(out) > 10000:
            print(f"[{i}/{len(todo)}] {arm} seed {seed}: already done, skipping",
                  flush=True)
            continue
        print(f"[{i}/{len(todo)}] {arm} seed {seed} starting "
              f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
        r = subprocess.run(
            [PY, os.path.join(HERE, "train.py"), "--arm", arm,
             "--steps", str(STEPS), "--seed", str(seed), "--eval-every", "200"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAILED {arm} seed {seed}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}",
                  flush=True)
            return 1
        for line in r.stdout.strip().splitlines()[-2:]:
            print("   ", line.strip(), flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
