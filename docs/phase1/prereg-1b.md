# Phase 1b — pre-registration

Written 2026-11-11, before any run of the experiments below. Results go to
`docs/phase1/sufficiency.md`; this file is not edited after the first run except for the
dated addenda at the bottom.

## What is being tested

The Phase 1 report claimed that the microsecond contact pulse carries no outcome
information. That claim rested on the two-coordinate model, not on the explicit reference;
Phase 0c had stated the opposite ("the peak force sets the threshold") from runs where
numerical dissipation was 35–88 % of the impact energy. Neither is a measurement of the
reference. Experiment A measures it.

Working definition used throughout. A reduced state `R(s)` is **ε-sufficient** for an
observable set `O` (capacity, onset, crack region, dissipated energy, large fragments) and
an input class `U` (what the rest of the world can still do to the subsystem) if replacing
`s` by `R(s)` changes no observable in `O` by more than ε for any continuation in `U`.
The quantity to minimise is bytes of state read and written per unit of simulated time,
subject to ε-sufficiency. Experiment A3 tests sufficiency directly: replace the state,
continue with the full model, compare consequences against the reference's own noise.

## A1. Contact stiffness and striker mass in the explicit reference

Scene: Phase 0d S2 (`beam_impact(seed, 0.025, v0, mp, 1)`, explicit, all cells detailed),
seeds 1 and 2, broken = at least two fragments within 50 ms.

- Contact stiffness: `kn`, `kt` of the striker facets scaled by 0.01, 0.1, 1, 10.
  Threshold `v_crit` by bisection in log v (10 steps, 0.2–1.5 m/s).
- Striker mass 5, 50, 500 kg at nominal stiffness; the two-coordinate prediction for each
  mass is computed before the explicit runs.

Predictions:
1. ×0.1 and ×10 change `v_crit` by less than 5 %. Above 10 % refutes "the contact pulse is
   irrelevant" in this regime.
2. ×0.01 (contact period ≈ 3.6 ms, comparable to a quarter of the 10 ms beam period)
   raises `v_crit` by 5–20 %: a soft contact excites fewer high modes, and damage ratchets
   on transient curvature.
3. Two-coordinate model error versus striker mass: within 10 % at 500 kg, 13–19 % at 50 kg
   (known), above 20 % at 5 kg, where the failure pattern is expected to stop being a single
   mid-span flexural crack.

## A2. Consequences above the threshold

At 1.25 and 2.0 × `v_crit(nominal)`, for each stiffness scale: dissipated energy D, fragment
count and masses, crack position, transferred impulse, contact duration and peak force.
Noise floor: the same run with `v0·(1 ± 10⁻³)` and with a different macro-step chunking.

Prediction: for ×0.1 and ×10 every consequence stays within max(10 %, 2 × noise). For ×0.01
D changes by more than noise.

## A3. Modal content and state truncation

Modes: lowest k eigenpairs of the intact beam (striker detached, supports bilateral).
At t_c = 1, 2, 4 ms after first contact, at 1.25 and 2.0 × `v_crit`: energy of the
displacement and velocity relative to the static state in the lowest k modes; then the
kinematic state of the cells is replaced by its projection on k modes (damage history
kept, striker kept) and the explicit run continues.

Predictions:
1. The lowest 10 modes hold at least 90 % of the vibration energy at t_c = 2 ms.
2. Truncation to k = 10 preserves break / no break, fragment count and D within
   max(15 %, 2 × noise).
3. k = 1 preserves the break outcome at 2.0 × `v_crit`; not guaranteed at 1.25.

Refutation of "the sufficient state is low-dimensional" for this member: k = 40 (3 % of
1362 DOF) fails prediction 2.

## C. Deterministic contact pipeline on the GPU

Workload: N equal spheres in 3D, integer state, uniform grid built with atomic counters
(fixed slots per cell), position-based contact with antisymmetric integer corrections
summed per sphere, PBD velocity update, gravity, box walls. Every reduction is an integer
sum or a count, so the unordered slot order cannot change the result. JS reference with
the same integer specification.

Predictions:
1. GPU state equals the JS reference bit for bit after 64 substeps (20 000 spheres, dense
   pile).
2. Dense pile (about 1.4 spheres per cell, 27 cells): 1.3–1.7 KB of gathers per
   sphere-substep, 15–50 ns on the M3 Pro, i.e. 100 000 spheres × 8 substeps = 12–40 ms.
   The Phase 1 estimate (7 ms for 100 000 bodies) is expected to be optimistic by 2–5×.
3. The count of cells whose population exceeds the slot capacity is a deterministic
   quantity; it is reported, and must be zero for the result to be inside the
   specification.

Not covered by C and stated as such: non-spherical bodies, rotation, friction, joints.

## B. What a reduction excludes and what its upkeep costs

Pre-registered separately below once A has shown which information matters.

## Addendum 1 (2026-11-11, after A3, before A3b) — post-hoc experiment

A3 projected on modes of the **intact** beam. Once damage has localised (t_c ≥ 1.2 ms, or
0.3 ms at 2 × v_crit) that projection is not merely inaccurate but wrong: it closes the
crack, and D rises by 60–2800 % for every k up to 160. The intact basis cannot hold the
state of a member whose topology has changed.

A3b repeats the truncation with the lowest k modes of the **current damaged operator**
(secant stiffness: (1 − d)·k_n on open points, k_n on closed ones, (1 − d)·k_t; d capped so
that separated parts stay weakly connected and their rigid motions appear as the lowest
modes), recomputed at t_c.

Predictions, stated before the run:
1. k = 41 preserves fragment count and D within 15 % at every t_c (0.3, 1.2, 2.0, 4.0 ms)
   for both 1.25 and 2.0 × v_crit.
2. At t_c = 4 ms (fracture complete) k = 11 preserves fragment count and fragment
   velocities within 0.1·v0.
3. Threshold under truncation at t_c = 1.2 ms with k = 21: within 5 % of the untruncated one.

If prediction 1 fails, a modal state is insufficient during fracture even with an updated
basis, and the member has to be simulated in full while it is breaking.

## Addendum 2 (2026-11-11, after A and C, before P) — experiment P: a port model

A3b showed which state is sufficient; it did not show that a reduced model can *evolve*
through a contact. P tests that. The beam is replaced by k modal coordinates of its current
(intact) operator plus one quasi-static attachment shape per port (the static response to
the port load minus its modal part, "residual compliance"). The striker and the contact
spring stay as in the reference. Reference for P: the explicit run with elastic interior
facets (no damage), v0 = 0.5 m/s, so only the reduction is tested.

Compared over the first 3 ms: contact force peak, transferred impulse, striker velocity
after contact, peak mid-span deflection; and the damage-onset velocity (first interior
facet point reaching its onset opening), reference by bisection with the damage law,
port model from recovered facet openings.

Predictions:
1. With the attachment shape, k = 21: impulse and striker velocity within 5 %, mid-span
   deflection within 5 %, peak force within 15 %, onset velocity within 15 %.
2. Without it, the peak force is off by more than 30 % for every k ≤ 21: truncated modes
   make the contact point too stiff.

Refutation of "boundary response with a few internal coordinates is enough for a member
that is not breaking": prediction 1 fails at k = 41 as well.
