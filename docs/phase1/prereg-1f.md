# Phase 1f pre-registration: one canonical contact system, computed fully or incrementally, on two GPUs

Written 2026-11-11 after the unchanged Phase 1d solver was reproduced on a second vendor
(`results/phase1/xdev/baseline_*.json`: both signatures, full state dumps, body-order permutation —
identical on Apple M3 Pro and NVIDIA GTX 1660 SUPER) and before any of the changes below were run.

## What is being decided

1. **Physics of the rest state.** In 1d a resting pile leaks 0.4 % (1.6 % at the coarse quantum) of
   its weight per substep into a ledger: stored impulses do not balance gravity, the floor carries
   less than the pile weighs. Observables and tolerances are fixed here, before any tuning.
2. **Where the slow settling comes from** (the layers^2 law is so far one fit through four points).
3. **No silent drops.** Cell capacity 8 filled in race order makes the neighbour set
   device-dependent once a cell holds more than 8 bodies; contact capacity 16 likewise.
4. **Incremental execution on the GPU with its real scheduling cost**, bit-identical to full stepping.

## Observables and tolerances (fixed before optimisation)

Scenes, all equal spheres of 10 cm, 8 substeps per frame: `column(L)` cold start, L = 8…128;
`pile(n)` cloud of n^3 collapsing in a box; `slump`: the 8000 pile at rest, then one side wall moved
20 cells away; `kick`: one surface sphere of a resting pile given 0.25 / 1 / 4 m/s.

Reference for physics: the same contact law with 32 iterations per substep and the smallest rest
quantum; scatter from three seeds of the reference.

| observable | tolerance |
|---|---|
| weight carried by the floor and walls at rest | >= 99.9 % (ledger flux <= 0.1 % of weight per substep) |
| column: final loss of height; transient | < 2 %; < 5 % |
| slump: runout (95th percentile of x), shift of the centre of mass, bodies displaced > 1 D | within 10 % of the reference or twice its seed scatter |
| slump: surface profile h(x), mean absolute difference | < 10 % of the mean pile height |
| kick: displacement of the kicked sphere, sum of displacements | within 25 % or twice the scatter |
| determinism | positions, velocities, contact sets and impulses identical word for word: JS, M3 Pro, GTX 1660 SUPER; identical under a permutation of the body order |

## Hypotheses

| id | claim | pass | kill |
|---|---|---|---|
| F1 | Routing: a supported, slow body hands its whole residual momentum to its supporting contacts (exact integer split, remainder to one designated support) instead of to the ledger; the reaction reaches the lower body one iteration later. Rest then has zero flux. | flux at rest <= 0.1 % of weight on every scene; pile(22k) rests no later than in 1d | flux > 0.4 % or piles stop resting |
| F2 | The same routing makes settling of a column linear in its height. | 64-column: exact rest within 200 substeps, transient < 5 %; exponent of time-to-rest vs L below 1.3 | exponent > 1.7 |
| F3 | Routing does not change the physically meaningful consequences. | all slump and kick observables within tolerance against the reference | runout or profile off by more than twice the tolerance |
| F4 | Grid as per-cell linked lists (no capacity) and a loud, canonical contact-overflow rule: a forced overflow (more than 8 bodies in a cell, more than K contacts) gives the same state on both GPUs and a nonzero fault counter. | identical states, fault reported | any device-dependent state |
| F5 | Ledger accumulated on the GPU as exact 64-bit sums equals the JS ledger. | equal on both GPUs | — |
| F6 | Incremental execution on the GPU (frontier of non-quiet bodies, dilation over cells, compacted active lists per iteration, indirect dispatch) reproduces full stepping word for word on both GPUs through settle, rest, kick and slump. | identical at every checkpoint | any difference |
| F7 | Cost of deciding what to do plus doing it is well below full stepping when little changes, and not much above it when everything changes. | GPU time: at rest < 5 % of full; averaged over the wake of a 1 m/s kick in a pile of >= 500k bodies < 20 %; all-active overhead < 30 % | at rest > 20 % of full, or all-active overhead > 100 % |
| F8 | Memory per body. | <= 600 B | > 1000 B |

Not in scope: rotation, rigid clumps, multigrid, the 1e member stepper (its open questions are
listed in the report, not answered here).
