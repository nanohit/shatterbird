# Phase 1d and 1e: bodies at rest on the GPU, and the price of interfaces

2026-11-11. Pre-registrations: `prereg-1d.md`, `prereg-1e.md`. Results: `results/phase1/d_*.json`,
`e_v4_*.json`. Code: `gpu-probe/rest.js` (integer specification, WGSL port, lab `lab-rest.cjs`),
`crates/experiments/src/bin/phase1e.rs`. M3 Pro only; 3D equal spheres without rotation (1d), the 2D
beam (1e).

## 1d. Rest as an exact fixed point

**Solver that came out of the lab** (differs from the pre-registered sketch, see "changed after
seeing data"): velocity-level Jacobi with mass splitting; every contact keeps an impulse vector,
antisymmetric between the two bodies; velocities and impulses share one unit (1/32 of a position
step per substep), so nothing is rounded when impulses act; stick friction with a static cone
(0.5) that drops to kinetic (0.375) when it slips; penetration is pushed out on positions only, and
only while one of the two bodies is moving or the overlap exceeds 2.5 % of a diameter; a supported
body slower than 0.18 mm/s is set to rest and its momentum goes to a ledger. Four Jacobi iterations
per substep, 8 substeps per frame.

| id | pre-registered | result | verdict |
|---|---|---|---|
| D1 | 64-layer stack, cold start: final < 2 %, transient < 5 % | final 0.52 %; transient 11.5 % / 5.8 % / 2.9 % / 1.4 % at 1 / 2 / 4 / 8 iterations | fails as pre-registered (1 iteration), passes from 4 |
| D2 | stack and 20k pile reach an exact fixed point within 10 s | 22k pile: zero state changes from substep 1692 (3.5 s) at 4 iterations, never within 8000 at 2; 64-stack: 34 s; 125k pile on the GPU (23 diameters deep): still moving after 3000 substeps | pass for the 22k pile, fail for the tall stack and the deep pile |
| D3 | a kick wakes < 5 % of 20k; wake volume grows < 1.3x from 20k to 80k | default quantum: 17.5 % / 52 % / 95 % of 22k for 0.25 / 1 / 4 m/s, quiet after 50 / 75 / 240 substeps; the 80k pile did not settle in the JS lab. Quantum 4x coarser: 8 % / 19 % / 31 % of 22k, and 0.7 % / 3.7 % / 13.6 % of 80k with wake volume x0.29 / x0.68 / x1.32 of the 22k one | fail at the default quantum; with the coarser one the wake does not grow with the pile |
| D4 | change-driven stepping bit-identical | identical every step through settle, rest and a kick (full state incl. impulses) | pass |
| D5 | WGSL bit-identical; cost <= 3x Phase 1b | identical (8000 spheres x 600 substeps); 18.2 ns per sphere-substep at 125k, 22.0 at 1M = 3.9x / 3.4x (2.9x / 2.6x at 2 iterations) | identity pass, cost fail at the iteration count that rests |

What this says about the engine:

- **Rest can be an exact fixed point of an integer map**, and then skipping is exact: a body whose
  neighbourhood (iterations + 1 cells) did not change reproduces its own state, so any superset of
  the changed set gives bit-identical results. Sleep needs no rule in the specification.
- **The information a resting pile needs is its contact impulses** (16 B per contact per side). With
  them a stack has no steady compression at all (0.4–0.5 % is the slop); without them the Phase 1b
  pipeline sat at 19–28 %.
- **A local solver pays for height.** Transient sag of a cold stack is about 0.18 % x layers /
  iterations; time to exact rest grows as layers^2 / iterations. Holding a 64-stack within 5 % needs
  4 iterations per substep; a 23-diameter pile does not rest in 6 s even then. This is the non-local
  influence the hypothesis talks about, and no local iteration removes it.
- **How local a change is, is set by the quantum.** At the default quantum a 1 m/s kick to one
  sphere changes half of a 22k pile. With a 4x coarser quantum (impulse changes below 2 pos-steps
  ignored, rest below 0.7 mm/s) the wake is a bounded region: 2.9k bodies of 80k for the same kick,
  no larger than in the 22k pile, and change-driven stepping evaluates 5–20x fewer body-steps than
  full stepping while it lasts. The price is on the ledger: 1.6 % of the weight instead of 0.4 %.
- **Honest frame cost** (full stepping, M3 Pro): 125k spheres 18 ms, 1M spheres 176 ms per frame at 8
  substeps and 4 iterations; 1664 B of buffers per body. The earlier "100k bodies in 7 ms" estimate
  does not survive a solver that holds stacks.

Changed after seeing data (all post-hoc): one iteration -> four; impulses in the velocity unit
instead of 32x finer (the finer unit produced idle limit cycles); hairline gaps (2/960 D) keep their
impulse (releasing it re-launched resting bodies); push-out on positions instead of velocities, then
only for moving partners (it kept deep piles sliding at the friction limit); kinetic friction; the
rest quantum with a ledger. Not done: rotation and rigid clumps, re-sort, change-driven stepping on
the GPU, other devices (the probe page now carries this solver as a stage).

Drop: position-based Jacobi without stored impulses for anything stacked; velocity-level push-out
with stored impulses (runaway); impulse units finer than velocity units; "wake region is small" as
a design assumption.

## 1e. The price of interfaces (member/port stepper, 2D beam)

| id | pre-registered | result | verdict |
|---|---|---|---|
| E1 | one span: thresholds within 3 %, D within 10 %, flops < 0.30 | no support collar, contact collar 4: thresholds -0.2 % / 0.0 %, D within 7.1 %, fragments equal, flops 0.22–0.37 (v3: 0.43–0.51); contact collar 3: +1.0 % / -0.2 %, D within 8.3 %, flops 0.19–0.28 | pass with collar 3, one case of four above 0.30 with collar 4 |
| E2 | eight spans, members cut at interior supports: D within 10 %, flops < 0.15 | D -4.0 %, flops 0.081, wall 0.14 (v3: 0.45) | pass |
| control | attachment shapes on/off | shapes change D by a few percent either way and cost 10–20 % more | the shapes are not what was missing |

- **Collars are needed only at the impulsive port.** Members can sit directly on supports and
  against each other; the cut facets are ordinary ports with the full law.
- **The hypothesis about attachment shapes is not supported**: removing the support collars works
  without them. `SB_ATT` stays in the code as a control.
- **Most of the remaining cost was the interior damage check, not the interfaces.** A rigorous
  screen (band norms of the modal amplitudes bound every point's opening; a point below its
  threshold cannot change) makes quiet members nearly free and changes no result: eight spans
  0.169 -> 0.081 flops with identical numbers.
- **Not robust between the extremes**: at two and four spans the energy error is +20 % and -15 %
  (v3: -3 % and -11 %), and it moves by +-15 % with small parameter changes. Holding the model
  detailed longer repairs four spans (+2.5 % at a 4 ms window, at 0.91 cost), so the error comes
  from freezing damage after regrouping, not from the interfaces.

Next experiment that cuts the most uncertainty: let frozen members keep damaging (the same
pseudo-force route pristine members use, on the secant basis) and rerun the span series; pass =
D within 10 % at 1, 2, 4, 8 spans with flops below 0.25 / 0.15 at one / eight spans. For bodies:
a coarse level for the weight (the layers^2 law is the cost driver, and it is the same non-local
problem as the members solve for elastic structures).
