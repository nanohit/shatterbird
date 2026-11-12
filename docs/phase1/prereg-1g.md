# Phase 1g pre-registration: is the rest state a physical equilibrium, and what does the quantum cost

Written 2026-11-12, after the review of Phase 1f and before the experiments below. Two smoke runs
on 512 bodies preceded this text and are stated where they bear on a hypothesis.

## What was wrong with the Phase 1f physics check

1. **Unequal horizons.** After a kick each variant ran until it was exactly at rest or 4000 substeps.
   The reference (32 iterations, eps 1) never rests, so it was observed at 4000 substeps; variants
   with a quantum were observed after tens to hundreds. The reference "response" contains 4000
   substeps of its own creep.
2. **No control.** There was no run without the kick, so creep and response were not separated.
3. **Unequal initial states.** Every variant settled its own pile.
4. **Reference not shown to be converged.** 32 iterations are not a converged solve of a pile
   10–20 layers deep (Jacobi needs of the order of layers^2).

The numbers "responses suppressed 3–20x" are therefore withdrawn until remeasured.

## Design of the remeasurement

* One geometry per pile: the pile at exact rest under the default law (4 iterations, eps 32).
* One impulse state: the exact integer equilibrium on that geometry (below). Velocities zero.
* Every variant starts from that same state. Its control run (no kick) must not move at all; if it
  moves, the response is the difference to the control at the same substep.
* Observables at fixed substeps 6, 12, 25, 50, 100, 200, 400, 800 for every variant.
* Reference: the ladder 4, 8, 16, 32, 64, 128, 256 iterations with eps 0. An observable counts as
  resolved by the model only if 128 and 256 iterations agree within the tolerance; otherwise it is
  reported as not resolved (sensitive to solver detail) and only ensemble statistics are judged.
* Ensemble: 3 piles (seeds 11–13) x 6 surface spheres x kicks {0.25, 1, 4 m/s downward at 27 degrees
  from vertical; 1 m/s horizontal}. Statistics: geometric mean and median over the 18 kicks.

| observable (per kick, at equal substeps) | tolerance against the converged reference |
|---|---|
| displacement of the kicked sphere; sum of displacements | geometric-mean ratio within 1/1.25 … 1.25, or inside the 128-vs-256 scatter of the reference |
| bodies displaced > 0.1 D, summed over the ensemble | within 25 % or 2 bodies |
| kinetic energy left at 25 / 100 substeps (fraction of the kick) | within a factor 2 |
| slump at 240 … 3840 substeps: x95, centre of mass, bodies displaced > 1 D, profile | 10 % (as in 1f) |

Analytic checks (no reference needed), single bodies and small stacks, gravity tilted by theta:
stick for tan(theta) <= 0.9 mu_s, slide for tan(theta) >= 1.1 mu_s; acceleration when sliding
g (sin theta − mu_k cos theta) within 5 %; stopping distance of a body launched along the floor
v^2 / (2 mu_k g) within 10 % once it exceeds 0.1 D; removal of a support: every body that rested on
it starts to fall within 2 substeps.

## Hypotheses

| id | claim | pass | kill |
|---|---|---|---|
| G1 | With equal initial state, equal horizon and a control, the default quantum (eps 32, 4 iterations) does not suppress kick responses by the factors reported in 1f. | geometric-mean ratio to the converged reference inside 1/1.5 … 1.5 for all four kicks at 100 and 800 substeps | ratio below 1/3 for any kick: the 1f claim stands |
| G2 | Next to the anchored rest state there is an exact integer equilibrium: impulses inside their cones that balance every body exactly, with no anchors. (Smoke run, 512 bodies: 128 sweeps warm-started, integer imbalance after rounding <= 2 units, fixed point of the integer law with eps 0.) | three piles of 4096: float imbalance < 1e-6 in < 2000 sweeps warm-started; exact after projection; the state does not change under the integer law with eps 0 at 4 and 32 iterations | no equilibrium on a pile, or the projected state moves |
| G3 | The force network is not unique: an equilibrium found from zero impulses differs substantially from the one next to the history. | relative L1 difference of the two networks > 10 % | < 1 %: the network is determined by the geometry and need not be stored |
| G4 | Handover: the sub-threshold residual of a supported body goes to its designated support through the contact (antisymmetric, cone applied), not to the world frame. Rest is then an exact balance with zero anchors, and piles still come to exact rest. | three piles of 4096: exact rest no later than 1.5x the default law; floor plus walls carry the weight exactly; anchors zero | piles do not rest within 3x, or any anchor remains |
| G5 | Handover leaves the consequences alone. | kick ensemble and slump within tolerance against the converged reference; analytic checks pass | slump off by more than 20 %, or an analytic check fails |
| G6 | A layered two-pass propagation (bottom-up against immovable lower bodies, then top-down handing the deferred reactions), used as a correction with the Jacobi law kept on every contact, settles a 64-column within 300 substeps without changing the slump. | both | slump off by more than 20 % (as routing was): not ported to the GPU |
| G7 | The cost of finding the equilibrium online is small against the settling itself: from the anchored rest state the default law with handover reaches the exact balance within 200 substeps. | <= 200 | > 2000 |
| G8 | If G4 and G5 pass, the changed law runs on both GPUs with JS = M3 Pro = GTX 1660 SUPER, full and incremental, and rest / wake GPU time within 1.5x of Phase 1f. | identity at all checkpoints, costs within 1.5x | any difference in state |
| G9 | Exact coincidence of two bodies and ties in contact selection are resolved without the array index: a canonical id is part of the state. A world with coincident bodies gives the same state under a permutation of the arrays. | identical | any difference |
| G10 | Contact overflow is outside the valid domain of the model for equal spheres: more than 16 contacts within the search margin need mutual overlaps of at least 12 % of the diameter (Tammes: 17 points on a sphere have angular separation <= 51.09 degrees). | no overflow in any scene of this phase; a forced compression overflows only after pair overlap exceeds 10 % | overflow at an overlap below 5 % |

Not relaxed: determinism, exact momentum accounting, no silent drops.

## Changes after the experiments started (marked post hoc)

* The reference ladder showed that the kicked-sphere displacement does not converge (256 vs 128
  iterations: 0.51–0.76). Per the rule above it is reported as not resolved by the model. For the
  final-law comparisons the reference became 64 iterations with the fine cone and no quantum; 128
  iterations are shown next to it.
* The fine friction cone (`cone: 1`) was added after the analytic incline check failed for the cone
  quantised in steps of 4F. It is part of v3.
* G4 was first tested as "hand the whole residual over"; the adopted rule (only bodies that arrived at
  rest hand over, designation by share of the load) came out of four failed variants, listed in the report.
* The two-pass prototype (G6) was tested with one local sweep per body; the variants tp1, tp3, tp4 are
  defined in the report.
* Integer projection of the equilibrium (G2): the maximum-load forest and the minimum kept load of
  64 units replaced the breadth-first forest after the first version failed.

Results and verdicts: [`rest-physics.md`](rest-physics.md).
