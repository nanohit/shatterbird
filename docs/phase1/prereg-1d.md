# Phase 1d pre-registration: bodies on the GPU — rest as an exact fixed point

Written before any run (2026-11-11). Short on purpose; results go to `results/phase1/d_*.json`.

**Idea.** The Phase 1b contact pipeline (position-based Jacobi) compresses a 64-layer stack by 19–28 %
and never stops jittering. A resting pile carries information only in its contact impulses. If those
are stored (one integer vector per contact, antisymmetric between the two bodies) and the solver has
no dead band, rest should be an *exact fixed point of the integer map*: zero velocity, constant
impulses. Then skipping a body whose neighbourhood did not change is bit-identical to stepping it —
sleep needs no thresholds and no rule in the specification, and cost follows change.

**Solver under test.** Velocity-level Jacobi with mass splitting; per contact a persistent impulse
vector in units 32x finer than momentum (kills the truncation dead band); stick friction with a cone
(mu = 0.5), spheres do not rotate; Baumgarte push-out above a slop of 4/960 of a diameter; one
iteration per substep, 8 substeps per 60 Hz frame at a 10 cm diameter (gravity 111 units per substep^2).
All reductions are integer sums over a body's contacts.

| id | claim | pass | kill |
|---|---|---|---|
| D1 | 64-layer stack, cold start | final compression < 2 %, transient < 5 % | final > 2 % |
| D2 | stack and a 20k random pile reach an exact fixed point | zero state changes within 10 s after settling starts | > 1 % of bodies still changing after 20 s |
| D3 | a sphere dropped on a resting pile wakes a bounded region | bodies ever changed < 5 % of 20k; wake volume (sum over time of changed bodies) grows < 1.3x from 20k to 80k | wake volume grows in proportion to N |
| D4 | change-driven stepping (only bodies whose neighbourhood changed) | bit-identical to full stepping | any difference |
| D5 | WGSL port | bit-identical to the JS specification; cost <= 3x the Phase 1b pipeline per sphere-substep | > 10x |

Then, if D1–D5 hold: rigid clumps (rotation in integers) and a deterministic spatial re-sort.
