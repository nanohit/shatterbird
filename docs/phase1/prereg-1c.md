# Phase 1c — pre-registration: members with ports, end to end

Written 2026-11-11 before the first run. Results: `docs/phase1/members.md`.

## Model under test

One stepper, one representation. The world is a set of **members** joined by **ports**.

- A member is a connected set of cells whose mutual facets are undamaged. It carries a
  rigid frame (finite rotation) and k free-free elastic modes of its own linear operator.
  A single cell is a member with k = 0, so a region where every cell is its own member is
  exactly the explicit rigid-body-spring network.
- A port is every other facet: supports, contacts, damaged facets, facets between members.
  Ports are evaluated with the full nonlinear law, including damage growth, on cell
  positions reconstructed from member coordinates. Nothing unilateral is inside a linear
  operator.
- Escalation: a bound on undamaged interior openings, sum over modes of |η_k|·σ_k, costs k
  operations per step. When it reaches χ = 0.8 the openings are recovered exactly (every
  8th step); if the largest reaches χ of its onset the member is dissolved into single
  cells with the reconstructed state.
- De-escalation: when no damage has grown anywhere for T_q = 1 ms, single cells are
  regrouped by undamaged connectivity; groups of at least 10 cells get a basis (k = 40)
  and become members, unless their recovered openings are already above χ.
- Every decision is a function of the state only. Step 2 µs throughout.

## Runs

Beam impact (Phase 0d S2), seeds 1 and 2, striker 50 kg, 50 ms.
0. Validation: the stepper with every cell single, against the Phase 0d reference
   (thresholds from A1, consequences from A2).
1. Hybrid: threshold by bisection; consequences at 1.25 and 2.0 × v_crit; a sub-onset
   impact at 0.15 m/s.
2. Work: flop model (120 per facet point, 6k + 12 per port cell of a member, 4k + 20 per
   member, 9 per single body, 2k per recovered point) and wall time, hybrid over
   all-single, same code.

## Predictions

1. Validation: threshold within 3 %, D within 5 % of the Phase 0d reference.
2. Hybrid: threshold within 5 %; D within 10 %; same fragment count; fragment velocities
   and striker velocity within 0.1 v0; crack position within 0.1 m.
3. Escalation within 0.15 ms of first contact for every velocity above 0.2 m/s; none at
   0.15 m/s, with striker velocity within 5 % of v0 of the all-single run.
4. Work ratio 0.10–0.25 over 50 ms; at most 2 escalations per run.

Pass: 10 % on every consequence. Kill: any consequence beyond 20 %, or work above 0.5.
Known risk, stated in advance: damage may keep trickling after the break, so the quiet
condition is never met, or members re-escalate immediately after regrouping.

Scenes with several members and the support-removal cantilever follow only if this passes.

## Addendum (2026-11-11, after the first runs, before v2) — post-hoc redesign

What the first runs showed (details in members.md): the all-single stepper equals the
Phase 0d reference to ten digits; the hybrid needs a collar of single cells around loaded
ports; and the escalation trigger "an undamaged facet approaches onset" is wrong for this
material. Concrete micro-cracks almost everywhere a wave passes, so every span of a
multi-span beam escalates, and after the break the fragments ring near their thresholds
and the model cycles between reduced and full (up to 45 escalations in 50 ms).

v2, to be tested next:
- The basis of a member is that of its **intact elastic** operator and is rebuilt only when
  its cell set changes.
- Damage inside a member is evaluated from the modal openings every 8th step and enters as
  a held pseudo-force (law force minus linear force). Micro-damage therefore evolves inside
  the member; nothing faster than the highest retained mode exists there.
- Escalation trigger is **localisation**: an interior facet reaches damage d_esc = 0.3.
  Facets above d_esc are ports.

Predictions for v2 (k = 20, collar 4, quiet window 2 ms):
1. At most 3 escalations per run on the single span; consequences within 10 %.
2. Work ratio below 0.45 on the single span.
3. On 8 spans with collars that separate the spans (8 rings): far spans do not escalate;
   work ratio below 0.35.
