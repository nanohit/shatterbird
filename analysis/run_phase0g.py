"""Serial CPU experiment driver (concurrent runs would contaminate timings).

Build: cargo build --release -p experiments --bin phase0g
Run: python3 analysis/run_phase0g.py rebase
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
group = sys.argv[1] if len(sys.argv) > 1 else "rebase"
if group == "rebase":
    tasks = [(case, seed, scale, "rebase")
             for case in ("cantilever", "span", "wall")
             for seed in (1, 2) for scale in (1, .5)]
    tasks += [("wall_fast", 1, 1, "rebase"), ("wall_coarse", 1, 1, "rebase"),
              ("contact", 1, 1, "rebase"), ("span", 1, .25, "reference"),
              ("span", 1, .125, "reference")]
elif group == "validation":
    tasks = [("cantilever_pulse", seed, scale, "rebase")
             for seed in (1, 2) for scale in (1, .5)]
elif group == "midpulse":
    tasks = [("cantilever_midpulse", seed, scale, "rebase")
             for seed in (1, 2) for scale in (1, .5)]
elif group == "timing":
    tasks = [("wall", 1, 1), ("wall", 1, .5), ("wall", 2, 1), ("wall", 2, .5),
             ("cantilever", 1, 1), ("cantilever", 1, .5),
             ("span", 1, 1), ("span", 1, .5)]
else:
    raise SystemExit("rebase | timing | validation | midpulse")

out = ROOT / "results" / "phase0g"
out.mkdir(exist_ok=True)
with (out / f"{group}.log").open("w", buffering=1) as log:
    for task in tasks:
        args = list(map(str, task))
        print("START", args, flush=True)
        log.write("START " + str(args) + "\n")
        p = subprocess.run([str(ROOT / "target/release/phase0g"), *args],
                           cwd=ROOT, stdout=log, stderr=log)
        print("END", args, p.returncode, flush=True)
        if p.returncode:
            raise SystemExit(p.returncode)
