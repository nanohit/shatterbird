# Phase 1c — Members with ports, end to end

2026-11-11. Pre-registration and two dated addenda: [prereg-1c.md](prereg-1c.md).
Code: `crates/experiments/src/bin/phase1e.rs`. Raw: `results/phase1/e_*.json`.
CPU only, 2D, one beam geometry (1–8 spans), two meshes. The design changed twice after
seeing results; every such change is marked post-hoc.

## 0. Verdict

The hypothesis of [sufficiency.md](sufficiency.md) §7 was run as a whole for the first time.

| | result |
|---|---|
| One stepper for every level of detail | Works. With every cell its own member it reproduces the Phase 0d reference to ten digits at the same step. |
| Consequences, final variant, 40 modes | Threshold −1.4 % / −0.2 %; D within 7.7 %; fragment count and masses equal (≤ 3.3 %); striker velocity within 0.3 % of v0; crack within 0.03 m; fragment velocity within 0.1 v0 in three cases of four (0.11 in one). |
| Cost on the single span | 0.43–0.51 of the all-cells run. Predicted 0.10–0.25. The kill line (0.5) is touched in one case. |
| Cost versus structure size | 0.74 → 0.45 from 1 to 8 spans. It falls, but far slower than "cost follows activity" requires. |
| Pre-registered variant (no collar, onset trigger) | Failed: D +10 %, horizontal fragment velocity 60 % low, up to 21 escalations per run. |

By the pre-registered criteria this is **not a pass**: consequences are mostly inside 10 %,
cost is not. What it bought is a list of mechanisms that do and do not work, each with a
number.

## 1. Model

- **Member:** a connected set of cells with a rigid frame (finite rotation) and k
  free-free elastic modes. Interior facets live in its linear operator.
- **Port:** every other facet — supports, contacts, broken facets, facets between members.
  Ports are evaluated with the full nonlinear law on cell positions reconstructed from
  member coordinates.
- **A single cell is a member with k = 0.** A region of single cells is the explicit
  rigid-body-spring network.
- Step 2 µs everywhere. Decisions depend on state only.

Work model: 120 flops per facet point, 6k + 12 per port cell of a member, 5k + 20 per
member, 9 per single body, plus the interior evaluation every 8th step. Wall time is
reported next to it; both are ratios to the same code with every cell single.

## 2. What happened, in order

**Validation.** All cells single: thresholds −0.4 % / −0.3 % against Phase 0d (different
bisection brackets), D within 0.4 %; at an identical step size the run matches the
reference World to ten digits (D = 6.1649132220 J).

**v1, as pre-registered** (escalate when an undamaged facet approaches 0.8 of onset;
regroup after 1 ms without damage):
- The beam escalates 0.06–0.09 ms after first contact, during the pulse. The state handed
  to the cells lacks the dent under the striker. D +10 %, horizontal fragment velocity
  0.047 instead of 0.125 m/s.
- **Collar** (post-hoc): cells within n rings of a support or contact facet stay single.
  With 4 rings (0.1 m, the width of the loaded pad) D is +2.8 %, fragment velocities within
  0.02 m/s. With 2 rings: D +11 %. The collar is 23 % of the cells of this beam.
- After the break nothing could be regrouped: 250 of 1221 facets carry some damage at
  1.25 v_crit (56 above d = 0.5, 13 broken), 489 at 2 v_crit. Allowing damaged facets into
  member operators (secant stiffness) regroups the fragments, but the onset trigger then
  fires on their ringing: 4–21 escalations per run, D between −5 % and +24 % depending on
  the quiet time.
- Eight spans, spans separated by collars: every span escalates within 4 ms as the flexural
  wave passes, 45 escalations in 50 ms. The far spans do reach 0.8 of onset; the reference
  dissipates 1.2 J in them without breaking anything.

Reading: **onset is the wrong trigger for a quasi-brittle material.** Micro-cracking
happens wherever a wave passes; what has consequences is localisation.

**v2** (post-hoc, addendum): basis of the intact elastic operator; interior damage
evaluated from modal openings every 8th step and applied as a held pseudo-force; escalate
when an interior facet reaches d = 0.3.
- Before the break this works: one escalation, at 0.13–0.19 ms.
- After the break it does not. Regrouped fragments get spurious damage (D +17 % at
  2 v_crit, 5 escalations): a smooth basis spreads the opening that the true field
  concentrates at the already softened facets, and the neighbours crack instead.

**v3** (post-hoc): two kinds of member.
- *Pristine* (no damage inside at build time): v2 behaviour.
- *Frozen* (built on a damaged set after a quiet window): secant operator of the committed
  damage, no damage growth inside, dissolved only if an opening exceeds 1.5 × its history.
- Quiet = interior dissipation grew by less than 0.5 % over a 2 ms window.

## 3. Results of v3

Single span, 4-ring collar, against the all-cells run of the same code:

| | k = 20 | k = 40 |
|---|---|---|
| threshold, seeds 1 / 2 | +1.0 % / −2.7 % | −1.4 % / −0.2 % |
| D at 1.25 v_crit | +2.9 % / +5.8 % | +3.4 % / +1.4 % |
| D at 2.0 v_crit | +12.6 % / +1.5 % | +7.7 % / −3.6 % |
| fragment velocity error / v0, worst | 0.098 | 0.110 |
| striker velocity error / v0, worst | 0.091 | 0.003 |
| crack position error, worst | 0.046 m | 0.034 m |
| flops ratio | 0.38–0.46 | 0.43–0.51 |
| wall ratio | 0.43–0.50 | 0.50–0.57 |
| escalations / regroupings per run | 1 / 2 | 1 / 2 |

- Sub-onset impact (0.15 m/s): no escalation; striker velocity after 50 ms differs by
  8 % of v0 from the all-cells run (several bounces; predicted 5 %); flops ratio 0.34–0.40.
- Where the cost goes on the single span: 23 % of the cells are collar and never reduce;
  the member is dissolved for 4–5 of 50 ms; the interior evaluation is 5–13 % of the total.

Structure size, v0 = 1.25 v_crit of the single span, k = 40 (the multi-span beams do not
break at this velocity):

| spans | cells | D error, one member for all spans (collar 4) | flops | D error, one member per span (collar 8) | flops |
|---|---|---|---|---|---|
| 1 | 454 | +3.4 % | 0.44 | −0.3 % | 0.74 |
| 2 | 863 | +8.3 % | 0.38 | −3.0 % | 0.65 |
| 4 | 1694 | +55 % | 0.34 | −11 % | 0.55 |
| 8 | 3361 | +1013 % | 0.27 | −1.4 % | 0.45 |

- **A fixed number of modes on a growing member is wrong, not merely coarse:** 40 modes on
  eight spans cannot resolve the impact and put ten times the damage into the beam.
  Resolution has to follow the member's size, or members have to stay bounded.
- With one member per span the answer holds (within 11 %) and the ratio falls with size,
  but collars at nine supports keep a third of the cells single.

## 4. What this establishes

1. **Continuity of representation is real.** One code path spans rigid fragments, modal
   members and the full network, and its fine limit is the reference itself.
2. **Loaded interfaces need real cells** over a depth comparable to the loaded patch.
   Modes alone are too stiff there, and the error goes straight into the consequences.
3. **The trigger is localisation, not onset.** Distributed micro-damage can evolve inside
   a member at a slow rate (every 8th step here) through pseudo-forces.
4. **After fracture the basis must be the damaged one and the damage must be frozen.** The
   fragments ring near their thresholds forever in an undamped model; chasing that costs
   more than it returns (the reference adds 3.5 % to D after 4 ms).
5. **Mode count is a resolution, not a constant.**
6. **The cost floor is the interfaces, not the fracture.** On this beam the fracture itself
   is 9 % of the run; the collars are 23–34 %.

Approaches to drop, in addition to the list in sufficiency.md §7:
- Onset-based escalation in softening materials.
- Regrouping damaged fragments on an intact basis.
- One member for a whole connected structure.
- Permanent collars at every support.

## 5. Not verified

- 2D, one geometry, two meshes; no scene with separate bodies colliding.
- Parameters (collar 4, d_esc 0.3, history factor 1.5, quiet window 2 ms, 8-step interior
  rate) were chosen after seeing sweeps; a sweep of each is not reported, only the cases
  listed in §2–3.
- The flop model is mine; wall time of an unoptimised implementation is the only check.
- Frozen members forgo late micro-damage by construction.
- Threshold of the multi-span beams, the support-removal scene and anything with two
  members were not run (the pre-registration made them conditional on a pass).
- Work ratios include no cost for basis construction in flops; in wall time it is
  0.03–0.35 s per run and is included.

## 6. Next

The open quantity is interface cost. Two candidates, one experiment:
- carry slowly loaded ports (supports under gravity, resting contacts) by one attachment
  shape each instead of a collar, and keep collars only for ports whose force changes
  faster than the member's highest mode;
- bound member size by structure (one member per span, panel or block) with k per member
  set by its size.

Test on the 8-span beam and one two-body scene. Pass: consequences within 10 % and flops
ratio below 0.15 on eight spans. Kill: ratio above 0.3. If it is killed, reduced members
are a factor-of-two device and the runtime's leverage has to come from elsewhere: larger
steps in the detailed zone and sleeping of quiet regions.

## 7. Reproduce

```
cargo build --release -p experiments --bin phase1e
./target/release/phase1e validate
SB_COLLAR=4 SB_TQ=0.002 SB_K=40 ./target/release/phase1e hybrid
SB_NSPAN=8 SB_COLLAR=8 SB_TQ=0.002 SB_K=40 ./target/release/phase1e one 1 0.6076 h
```
The v1 and v2 variants are not reachable from the final binary; their numbers above come
from runs made on 2026-11-11 before the changes and were not stored as files, except
`results/phase1/e_scaling_spans_v1.json`.
