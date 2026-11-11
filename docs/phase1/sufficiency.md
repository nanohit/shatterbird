# Phase 1b — What state is sufficient, and what a contact pipeline costs

2026-11-11. Pre-registration: [prereg-1b.md](prereg-1b.md) (two addenda are marked
post-hoc there). Raw results: `results/phase1/a1_*.json`, `a2_*.json`, `a3*.json`,
`p_port_model*.json`, `b_basis_cost.json`, `det_chrome_metal_m3pro_v2.json`.
The mechanics (§2–5) was measured on one machine. The GPU part (§6) now has three devices:
M3 Pro (Chrome 157 / Dawn / Metal), iPhone (Safari 18.6 / WebKit / Metal) and GTX 1660 Super
(Chrome 157 on Windows); see §6.1.

This document supersedes two things in [runtime-reframing.md](runtime-reframing.md):
- §3, the tier ladder T0–T3. It fixed model classes too early and is withdrawn.
- §4, "the microsecond contact pulse carries no outcome information". That sentence was
  about the two-coordinate model. The reference says something narrower (§2 below).

## 0. Summary

| question | answer from the runs |
|---|---|
| Does the contact pulse matter in the explicit reference? | The **threshold** moves by at most 6.2 % when contact stiffness changes 1000×. The **consequences** do not survive softening: at 2 × v_crit a 100× softer contact dissipates 55–60 % less energy. A stiffer contact than nominal changes nothing (≤ 3 %). |
| How much state decides what happens next? | About 20 modal coordinates right after the pulse, 6–11 while the crack runs, 6 (two rigid fragments) when it is over — **if the modes are those of the current damaged operator**. |
| Can the basis be fixed in advance? | No. The intact-beam basis is wrong once damage has localised: D is off by 60–2800 % for every k up to 160. |
| What does basis upkeep cost? | One rebuild equals 0.2–0.8 ms of explicit simulation of the same member. During a 4 ms fracture that is more than simulating it; afterwards it pays back within a small fraction of one frame. |
| Does a reduced model evolve correctly through contact? | With 40 modes: striker velocity within 1–4 % of v0, mid-span deflection within 4–9 %, damage onset 10–12 % early. Only after the unilateral supports were taken out of the linear operator; with them inside, the deflection is 24 % off at every k. |
| What does a deterministic contact pipeline cost? | 4.7 ns per sphere-substep at 125 000 spheres, 6.5 ns at 10⁶ on the M3 Pro, bit-identical to the CPU reference. With memory order decoupled from space: 8 ns and 41 ns. Tall stacks compress 8–28 % with this solver. |
| Is it the same on other GPUs? | Yes for the bits: M3 Pro, iPhone and GTX 1660 Super agree on every integer workload, and disagree on every float one across vendors. Coherent speed transfers; the penalty for incoherent memory order is 3–6× at 125 000 spheres on the other two devices. |

## 1. What is being minimised

A subsystem exchanges canonical quantities (impulse, force, displacement of a few boundary
ports) with the rest of the world. Its interior representation is free. Call a reduced
state `R(s)` **ε-sufficient** for an observable set `O` and an input class `U` if replacing
`s` by `R(s)` changes no observable in `O` by more than ε under any continuation in `U`.

- `O`: capacity, onset, crack region, dissipated energy, large fragments and their motion.
- `U`: what the neighbours can still do to the subsystem, bounded in amplitude and in
  bandwidth (a contact of duration τ injects content up to about 1/τ).
- Objective: bytes of state read and written per unit of simulated time, subject to
  ε-sufficiency, plus the cost of the test that says sufficiency is about to be lost.

This makes "minimal state" a measurable quantity: truncate the state, continue with the
full model, compare `O` against the reference's own noise. A3 and A3b do exactly that.
It is not a claim that cost depends only on the number of changed elements: when a change
has a non-local effect, the sufficient state grows, and the cost should grow with it.

## 2. A1–A2: the contact pulse in the explicit reference

Scene: Phase 0d S2, beam 2.0 × 0.2 m, 449–454 cells, explicit step 2.1 µs, striker on top
at mid-span. Broken = two fragments within 50 ms. Threshold noise (explicit step × 0.9):
≤ 1.4 % at 50 kg, up to 5.8 % at 5 kg.

**Thresholds, striker 50 kg, contact stiffness scaled:**

| scale | seed 1 (m/s) | seed 2 (m/s) |
|---|---|---|
| ×0.01 | 0.462 (−5.0 %) | 0.511 (−5.7 %) |
| ×0.1 | 0.482 (−0.9 %) | 0.508 (−6.2 %) |
| ×1 | 0.486 | 0.542 |
| ×10 | 0.501 (+3.1 %) | 0.543 (+0.3 %) |

Predicted: under 5 % for ×0.1 and ×10 (missed once, −6.2 %); +5…+20 % for ×0.01 (wrong
sign: softer contact breaks the beam slightly more easily).

**Thresholds versus striker mass, and the two-coordinate model computed beforehand:**

| mass | reference | kinetic energy at threshold | two-coordinate | error |
|---|---|---|---|---|
| 5 kg | 2.55 / 2.47 m/s | 16.3 / 15.2 J | 2.71 / 2.87 | +6 % / +16 % |
| 50 kg | 0.486 / 0.542 | 5.9 / 7.3 J | 0.413 / 0.436 | −15 % / −20 % |
| 500 kg | 0.135 / 0.141 | 4.6 / 5.0 J | 0.127 / 0.136 | −6 % / −3 % |

Static work of fracture is 3.9 / 4.5 J. The error changes sign with the mass ratio, so no
single correction factor fixes a one-shape model. Predicted: within 10 % at 500 kg (held),
above 20 % at 5 kg (did not hold; the model is better there than expected).

**Consequences at fixed velocity (noise ≤ 3 %):**

| contact | D at 1.25 v_crit | D at 2.0 v_crit | impulse at 1.25 v_crit |
|---|---|---|---|
| ×10 | +3 % / +1 % | +1 % / +1 % | −0.5 % / −1 % |
| ×1 | 6.15 / 6.79 J | 10.4 / 12.8 J | 30.1 / 32.9 N·s |
| ×0.1 | +2 % / +8 % | −12 % / −13 % | −16 % / −14 % |
| ×0.01 | −16 % / −6 % | −55 % / −60 % | +16 % / +17 % |

Predicted: ×0.1 within 10 % on every consequence — failed (D −12 %, impulse −16 %).

What the histories show (seed 1, 1.25 v_crit):
- The nominal pulse peaks at 190 kN (static capacity 14.5 kN), lasts about 0.15 ms, and a
  second contact follows until 1.06 ms. The beam is broken by 4 ms.
- At ×0.01 the contact is one 0.94 ms push of 58 kN.
- With the stiff contact at 2 × v_crit the damage spreads over the whole length; with the
  soft one it stays a single flexural crack. That is where the missing 55–60 % of D is.
- The 5 kg striker dissipates 5.5 J within 0.1 ms, during contact: a local, wave-driven
  failure, final D 17 J.

Reading. The nominal contact is already in the impulsive limit: making it shorter changes
nothing. What matters is whether the pulse is shorter than the periods of the modes that
can do damage. "Impulse only" is a sufficient boundary quantity for the threshold to about
6 %, not for the damage pattern.

## 3. A3: how much state decides the future

Lowest k modes of the beam (striker detached). At time t_c the cell positions and
velocities are replaced by their projection on k modes; damage history and striker are
kept; the explicit run continues. Deviation from the pre-registration: t_c = 0.3, 1.2,
2.0, 4.0 ms after release instead of 1, 2, 4 ms after contact, because the fracture is
complete at 4 ms.

**Intact-beam basis** (pre-registered). Threshold change, seeds 1 / 2:

| elastic modes | highest frequency | t_c = 0.3 ms | t_c = 1.2 ms |
|---|---|---|---|
| 1 | 149 Hz | +52 % / +23 % | +21 % / +15 % |
| 5 | 1.3 kHz | +17 % / +9 % | +13 % / +13 % |
| 10 | 3.0 kHz | +8 % / +1.5 % | −4.5 % / −5.1 % |
| 20 | 6.0 kHz | +1.0 % / −1.4 % | −3.7 % / −10.6 % |
| 40 | 8.7 kHz | −0.1 % / −0.3 % | −0.4 % / −5.5 % |

At 0.3 ms (D = 0.4 of 6.2 J, member essentially intact) the lowest 10 modes hold 78–81 % of
the kinetic and 66–76 % of the strain energy, 20 modes 96 % and 72–86 %. Predicted: 10
modes hold 90 % — failed.

From t_c = 1.2 ms on, and at 0.3 ms for 2 × v_crit, the projection closes the crack. D
after truncation is +61…+178 % (1.2 ms), +82…+187 % (2 ms), +75…+2800 % (4 ms) for
k = 10…160, with spurious third and fourth fragments. The refutation criterion (k = 40
fails) is met: **a basis of the intact member cannot hold the state of a damaged one.**

**Basis of the current damaged operator** (post-hoc, predictions in addendum 1). Secant
stiffness of the committed damage, recomputed at t_c:

| modes | threshold, t_c = 0.3 ms | threshold, t_c = 1.2 ms | D, worst case over t_c and both velocities |
|---|---|---|---|
| 6 | +16 % / +9 % | +2.6 % / +4.0 % | −44 % (0.3 ms), within 13 % later |
| 11 | +8 % / +3 % | −1.2 % / +2.1 % | −25 % (0.3 ms), within 8 % later |
| 21 | +0.2 % / +0.2 % | +0.2 % / −1.0 % | within 6 % |
| 41 | −1.3 % / +0.5 % | +0.1 % / +3.2 % | within 5 % |

Fragment count is preserved for every k ≥ 6; fragment velocities stay within 0.13 v0.
At 4 ms, 6 modes reproduce D and the fragments, 4 modes do not (+300…+700 % D): six is
the number of rigid degrees of freedom of two fragments in 2D. All three addendum
predictions held.

Reading.
- Right after the pulse the sufficient state is about 20 coordinates, up to 6 kHz, which
  is the bandwidth of a 0.15 ms pulse. While the crack runs it is 6–11. When it is over it
  is the rigid motion of the fragments.
- The coordinates must belong to the operator as it is now. Damage is part of the state
  that defines the basis, not something the basis can absorb.

## 4. B: what the upkeep costs

Same CPU, one thread, 449 cells. Explicit: 83.8 µs per 2.12 µs step (7 860 steps per 60 Hz
frame at this cell size).

| modes | Lanczos steps | wall time | equal to explicit simulation of |
|---|---|---|---|
| 6 | 16 | 8.4 ms | 0.21 ms |
| 21 | 46 | 11.9 ms | 0.30 ms |
| 41 | 114 | 30.8 ms | 0.78 ms |

- **During fracture** (4 ms, damage changing throughout) a basis valid to the tolerances
  above would need rebuilding many times; a rebuild every 0.1 ms already costs more than
  the explicit run it replaces. Reduction does not pay while a member is breaking.
- **After fracture** one rebuild per fragment is repaid within 1/20–1/80 of the first frame.
- **Static condensation alone** removes degrees of freedom but not bytes: a block of m × m
  cells has an interface operator of (12 m)² entries against about 190 m² for its sparse
  interior in 2D, and it is larger than the interior in 3D for m > 3 (arithmetic, not
  measured). It helps only when the interface itself is reduced to a few port coordinates.

The CPU numbers use a general sparse factorisation and full re-orthogonalisation; they are
an order of magnitude, not a tuned implementation.

## 5. P: a port model against the elastic reference

The beam is replaced by k modes plus one quasi-static attachment shape for the striker
port. Striker and contact spring as in the reference. Reference: explicit run with elastic
interior, v0 = 0.5 m/s, 3 ms; damage-onset velocity by bisection (0.178 / 0.185 m/s).

| variant | striker velocity error / v0 | mid-span deflection | onset velocity | peak force |
|---|---|---|---|---|
| pads inside the operator, 40 modes | 2–4 % | −24 % / −23 % | +20 % / +16 % | −2 % / +3 % |
| pads as inner-edge pins, 20 modes | 6–7 % | +13 % / +7 % | −12 % | +15 % / +19 % |
| pads as inner-edge pins, 40 modes | 1–4 % | +8.5 % / +3.6 % | −12 % | −2 % / +4 % |
| same, 160 modes | 1–2 % | +8 % / +2 % | −10 % / −12 % | +20 % / +24 % |
| 40 modes, no attachment shape | 1.5 % | +8.5 % / +1.5 % | −10 % | +72 % / +76 % |

- The first run (pre-registered) kept the support pads in the linear operator as bilateral
  springs. The result did not converge with k: deflection −24 % at 160 modes. The pads are
  unilateral; under a downward load each carries only at its inner edge. With that contact
  state linearised (post-hoc) the first frequency drops from 149 to 111–118 Hz and the
  error falls to single digits.
- Pre-registered prediction 1 (k = 21 within 5 %) failed in both variants. At k = 41 with
  pins everything is within 12 %; the 5 % line for deflection is missed in one seed.
- Prediction 2 held: without the attachment shape the contact point is too stiff, peak
  force +130 % at 20 modes.
- The peak force does not converge with k and depends on sampling in the reference
  (150 kN at 20 µs, 159 kN at 4 µs). It is not a usable observable.

Reading. The error of a reduced member is dominated by the state of its unilateral
interfaces, not by the number of modes. Supports and contacts have to stay outside the
reduced operator as explicit ports; the member should be reduced free.

## 6. C: a deterministic contact pipeline on the GPU

Workload (`gpu-probe/spheres.js`): equal spheres in 3D, positions and velocities as i32 in
1/256 units, diameter 1024. Per substep four dispatches: integrate, clear grid, insert
(atomic counter per cell, 8 slots), solve. The solve gathers 27 cells and sums
antisymmetric integer corrections per sphere; velocity is the position difference. All
products fit 32 bits. Every reduction is an integer sum, so slot order does not matter.

- **Bitwise:** 21 952 spheres, 64 substeps, GPU equals the JS reference on every word
  (`738b4865d8570d24`). Maximum cell population 4 of 8 slots. Total momentum is conserved
  exactly between wall contacts (checked on the CPU).
- **Cost** (M3 Pro, dense state, 8–11 contacts per sphere):

| spheres | memory order | ns per sphere-substep | solve share | 8 substeps per frame |
|---|---|---|---|---|
| 125 000 | follows space | 4.7 | 4.9 | 4.7 ms |
| 125 000 | shuffled | 8.0 | 7.8 | 8.0 ms |
| 1 000 000 | follows space | 6.5 | 5.5 | 52 ms |
| 1 000 000 | shuffled | 41 | 40 | 330 ms |

  Predicted 15–50 ns: too pessimistic for a coherent layout, right for an incoherent one at
  10⁶. The Phase 1 estimate (7 ms for 100 000 bodies and 16 sweeps) corresponds to 7.4 ms
  measured here for spheres.
- **Layout is a first-order term.** The same work costs 1.7× more at 125 000 and 6.3× more
  at 10⁶ when memory order stops following space. A runtime has to keep bodies spatially
  sorted, deterministically. Not built.
- **Solver quality is the open problem, not throughput.** Settled columns in the integer
  reference: 16 / 32 / 64 layers compress by 8 / 20 / 28 % of a diameter (relaxation 1/8 per
  contact), 4 / 9 / 19 % at 1/4, and keep jittering at 0.5–2 % of a diameter per substep.
  One Jacobi iteration per substep moves information one layer per substep.

### 6.1 Three devices (added 2026-11-11, probe v2, full size)

Raw: `results/phase1/devices/`. The iPhone and the GTX were run by hand from the published
page, once each; their load during the run is unknown.

**Bits.**

| workload | M3 Pro, Chrome | iPhone, Safari | GTX 1660 Super, Chrome |
|---|---|---|---|
| integer facet kernel, 32-bit and 64-bit variants | `99b34cc3` / `2ae7e6fc` | same | same |
| block automaton, 64 steps | `5065ab0a` | same | same |
| sphere contacts, 21 952 × 64 substeps | `738b4865d8570d24` | same | same |
| f32 facet kernel, plain | `e4992206` | `e4992206` | `84b787f7` |
| f32, explicit fma | `8131c96b` | `8131c96b` | `460cd228` |
| f32, re-associated | `ff370890` | `ff370890` | `e7021c46` |

Every integer workload, including the grid built with unordered atomic counters, gives the
same bits on two GPU vendors, two shader compilers (Tint, WebKit) and two graphics APIs.
Every float variant gives different bits on NVIDIA than on Apple. On NVIDIA the plain and
the fma variants differ in 78 % of elements (4 % on Apple), so the two stacks do not even
agree on which operations get fused. This replaces the synthetic census of Phase 1 with a
real cross-vendor one, for one kernel.

**Speed.**

| | M3 Pro | iPhone | GTX 1660 Super |
|---|---|---|---|
| copy, read + write | 114 GB/s | 47 GB/s | 139 GB/s |
| facet kernel, memory-bound, ns per facet (F / N / W) | 1.74 / 1.73 / 1.69 | 3.81 / 3.81 / 3.93 | 1.30 / 1.31 / 1.34 |
| integer cost, compute-bound (N / W over F) | 1.65 / 4.57 | 1.34 / 3.15 | 1.17 / 3.00 |
| automaton, 16.7 M cells, ms per step | 1.39 | 3.19 | 0.64 |
| spheres 125 000, coherent / shuffled, ns | 4.7 / 8.0 | 9.9 / 30.4 | 5.0 / 29.2 |
| spheres 10⁶, coherent / shuffled, ns | 6.5 / 41 | 14.8 / 79 | 6.0 / 68 |

- The determinism cost stays at 1.00–1.03 when memory-bound on all three, and is lower
  than on the M3 Pro when compute-bound.
- Streaming and coherent workloads transfer between devices roughly with bandwidth: the
  phone is 2.1–2.4× slower than the M3 Pro, not the 5× assumed in Phase 1.
- Incoherent gathers do not transfer. Shuffling costs 1.7× on the M3 Pro at 125 000
  spheres, 3.1× on the iPhone and 5.9× on the GTX; at 10⁶ it is 6.3×, 5.3× and 11.4×.
  The M3 Pro was the most forgiving device, so its numbers understated the layout penalty.
- Safari reports time in 1 ms steps; the iPhone dispatch overhead is not resolved.

Not covered: non-spherical bodies, rotation, friction, joints, sleeping, slot overflow
handling (the count is deterministic, so a fault is the same on every peer; it was zero).

## 7. What follows

Hypothesis after these runs, not a design:
- A solid subsystem is a free member with ports. Supports, contacts and joints are
  explicit, unilateral where they are, and stay outside any reduced operator.
- Its interior is whatever is ε-sufficient now: rigid motion; rigid motion plus k modes of
  the current operator and one attachment shape per loaded port; or every cell.
- Escalation is decided by a cheap bound on facet openings from the reduced coordinates
  (onset was predicted 10–12 % early, on the safe side). De-escalation happens when damage
  stops, per fragment, with one basis rebuild.
- Detail is paid for while a member is breaking, and only there.

Approaches to drop:
1. The wave-resolving network as the universal base. It stays as the interior of members in
   active fracture and as the oracle.
2. Models with one or two coordinates for impacts: ±15–20 % with a sign that depends on the
   mass ratio.
3. A reduced basis fixed in advance, and any basis kept alive during fracture.
4. Reduced operators that contain unilateral supports or contacts.
5. Peak contact force as a criterion.
6. Static condensation without interface reduction.
7. Cost estimates from streaming bandwidth for irregular workloads.
8. One-iteration Jacobi contact where stacks matter.

## 8. Not verified

- Three devices, two vendors. No AMD, Intel, Adreno or Mali GPU, no Vulkan backend, no Firefox.
- The cross-vendor float comparison is one kernel; the iPhone and GTX runs are single runs.
- GPU timings under foreign load on the same machine; ±25 %.
- A, B, P: one beam geometry, two meshes, 2D. No scene with several members.
- The hybrid (reduced → full → reduced) was not run end to end. A3b truncates once and
  continues in full; P runs the reduced model only on an elastic member.
- The onset bound was evaluated on every facet, not as a cheap certificate.
- P uses an inner-edge pin as the support contact state; the residual +8 % in one seed is
  unexplained.
- No rigid bodies with rotation or friction on the GPU; no spatial re-sort; no sleeping.

## 9. Next experiment

**Free-interface port member with escalation, end to end.** Reduce the free beam (rigid
modes, k free modes, one attachment shape per port). Supports and striker act on ports as
unilateral contacts. Run reduced until the opening bound fires, hand the state to the
explicit model, re-reduce each fragment when damage stops. Compare threshold, D, fragments
and their velocities with the explicit reference on the beam impact, the support-removal
cantilever and one scene with two members. Pass: 10 % on every consequence. Kill: 20 %, or
total work above half of the explicit run. This is the cheapest run that can refute the
hypothesis of §7 as a whole rather than piece by piece.

Second, independent: rigid clumps of cells on the GPU pipeline, with a solver whose settled
compression at 64 layers is under 2 %, and a deterministic spatial re-sort.

## 10. Reproduce

```
cargo build --release -p experiments --bin phase1
./target/release/phase1 a1 && ./target/release/phase1 a1n && ./target/release/phase1 a2
./target/release/phase1 a3 && ./target/release/phase1 a3b
./target/release/phase1 p && ./target/release/phase1 p2 && ./target/release/phase1 b
DET_OPTS='{"nofallback":true}' node gpu-probe/run-det.mjs chrome_metal_m3pro_v2
sh gpu-probe/publish.sh        # probe page: https://nanohit.me/shatterbird/
```
