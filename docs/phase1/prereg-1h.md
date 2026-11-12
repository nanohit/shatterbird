# Phase 1h pre-registration: long-range dependencies inside a substep, fully parallel

Written 2026-11-13, after Phase 1g ([`rest-physics.md`](rest-physics.md)) and before any code below existed.

## Why

Phase 1g located the remaining physics error in the Jacobi iteration count, not in the rest rule:
at 4 iterations a pile is too soft (a downward kick sinks 3–6x too far, 3.5–4x too many bodies move by
more than 1 mm). Settling of a column needs ~L²/iterations substeps, and a column of 8–32 spheres shears
under a sideways load below the friction limit at any practical iteration count. Change-driven
iterations made extra iterations cheap where little changes, not where everything moves. Two ways
of carrying information further per iteration were rejected: multigrid (by decision) and sequential
passes over the support order (they break piles, 1g section 6).

## Method under test

Chebyshev semi-iterative acceleration of the Jacobi impulse update (Wang 2015, used for position-based
dynamics on GPUs). For iteration k ≥ max(1, S):

    P_{k+1} = Π( P_{k-1} + ω_{k+1} · (P̂_{k+1} − P_{k-1}) )

P̂ is the ordinary projected Jacobi update from P_k, and Π the friction-cone projection of the solve.
ω follows the Chebyshev recurrence for an assumed spectral radius ρ (ω_1 = 1, ω_2 = 2/(2 − ρ²),
ω_{k+1} = 4/(4 − ρ² ω_k)), rounded to 1/256. Every contact stays independent within an iteration.
P_{k-1} is the value the ping-pong destination copy holds before it is overwritten, so there is **no
extra memory**. A fixed point stays a fixed point. The change-driven skip of 1g stays valid: a body
whose own and partners' impulses did not change reproduces its value. Integer arithmetic, so it stays deterministic.

## Protocol

* Law: v3 (`PRM3`) plus `cheb: ρ`, `chebs: S`. Reference: the same contact law without acceleration,
  128 iterations, fine cone, no quantum (`i128c1`); 256 iterations on two piles to state what is resolved.
* **Tuning set:** pile seed 11 only, plus columns and stacks. ρ ∈ {0.9, 0.95, 0.98, 0.99}, S ∈ {0, 2},
  at 8 and 16 iterations. The chosen (ρ, S) is frozen before the evaluation set is run.
* **Evaluation set:** piles seeds 12–16 (five piles of 1728; 14–16 never used before), all 1g events at
  equal horizons from the exact equilibrium, with controls: kicks (6 spheres x 0.25/1/4 m/s down, 1 m/s
  side), slump, tilt 0.2–0.5, removal of a buried sphere.
* Observables resolved by the model (1g): bodies moved > 0.1 D and > 0.01 D, slump x95 / centre of mass /
  bodies > 1 D, tilt centre of mass, removal counts. The kicked-sphere displacement is reported, but it is
  not used for a verdict (it does not converge with iterations).

## Hypotheses

| id | claim | pass | kill |
|---|---|---|---|
| H1 | Accelerated v3 at N iterations is as accurate as plain v3 at 2N or more. | for N = 8 and 16, every resolved observable at least as close to the reference as plain 2N (geometric mean over the evaluation set) | worse than plain N on any observable class |
| H2 | It carries long-range dependencies. | a column of 32 holds at tan θ = 0.45 with ≤ 32 iterations; time to rest of a cold column grows slower than L^1.5 | the 8-column still shears at 32 iterations |
| H3 | It does no harm. | every pile of the evaluation set comes to exact rest; controls stay still; no kick moves more bodies than plain v3 at 4 iterations; slump within 10 % of the reference | any pile never rests, or energy grows after a kick |
| H4 | Its cost is that of the iterations it replaces. | change-driven evaluations per wake at N accelerated ≤ plain 2N; GPU time per iteration within 10 % of plain; JS = M3 Pro = GTX 1660 SUPER, full and incremental | any state difference across devices or modes |

Not relaxed: determinism, exact momentum accounting (antisymmetric impulses), no silent drops.

## Decisions on the tuning set (post hoc, before the evaluation set was run)

Recorded 2026-11-13 after the runs on pile 11, columns and stacks, before any run on piles 12–16.

* **Plain acceleration breaks exact rest.** ω > 1 near the fixed point makes integer limit cycles: 7 of 36
  events never rest at 8 iterations (ρ 0.95), 35 of 36 at 32 iterations (ρ 0.9875). Added a **gate**: a
  contact is accelerated only if one of its bodies has a residual above the rest threshold (32 units per
  axis, the handover threshold). It is symmetric in the pair, so impulses stay antisymmetric. With it every event comes to
  exact rest (36 of 36 at 8, 16, 32 iterations).
* **Accelerating the tangential impulse harms frictional loads.** With the gate, at 16 and 32 iterations,
  a pile under a sideways body force (tilt 0.4–0.5) creeps more than plain Jacobi at the same iteration
  count, and the slump is slightly worse. Over-relaxation pushes tangential impulses past the static cone,
  and the contacts then slide at kinetic friction. Added **normal-only acceleration** (`chebn`): only the
  normal part of the Chebyshev correction is kept, and friction stays plain Jacobi.
* **Frozen for evaluation:** `cheb = 1 − 0.4/N` (0.95, 0.975, 0.9875 for N = 8, 16, 32), `chebs = 0`,
  `chebg = 32`, `chebn = 1`. On pile 11 this gives, at N iterations, bodies moved > 1 mm after a 4 m/s kick
  50 / 34 / 20 (plain 2N: 47 / 33, reference 26), and tilt and slump at or better than plain 2N.
* Two GPU findings on the way, both fixed in JS and WGSL alike. The change-driven skip of 1g has to
  treat a body as changed when its unaccelerated update moves; otherwise it is not neutral under
  acceleration. Values the acceleration leaves unchanged are not reprojected, because the integer cone
  projection is not idempotent on the static boundary.
* The tuning compared against the 1g reference at 64 iterations; the 128-iteration reference was still running.

Results and verdicts: [`long-range.md`](long-range.md).
