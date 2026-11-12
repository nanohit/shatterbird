# Phase 1g: what a resting pile has to remember, and what the solver gets wrong

Pre-registration: [`prereg-1g.md`](prereg-1g.md). Previous phase: [`canonical-system.md`](canonical-system.md) (1f).
Code: `gpu-probe/rest2.js` (JS specification and WGSL, laws v2 and v3), `gpu-probe/equil.cjs` (static
equilibrium on a frozen geometry), `gpu-probe/lab3.cjs` (responses at equal horizons, slump, tilt, support
removal, analytic checks, stacks, contact capacity), `gpu-probe/g_summary.py`, `g_events.py`, `g_time.py`
(tables), `gpu-probe/xdev2.js` (GPU identity and timing). Results: `results/phase1/g/`, `results/phase1/xdev3/`;
the per-run records behind the tables are in `results/phase1/g/raw_runs.tar.gz` (unpack into a directory and
pass it to `g_summary.py` / `g_events.py`; the displacement fields are not included, so the field column stays empty).

Names used below. **v2** is the law of 1f: a supported body whose leftover velocity is below eps = 32 is set
to rest and the leftover is booked as an anchor towards the world frame. **v3** is the law adopted here
(section 4). `iN` is the number of Jacobi iterations per substep, `eN` the rest quantum, `c1` the fine
friction cone (section 2.5). The reference is the same contact law with many iterations and no quantum.

## 1. The 1f physics check was not a measurement

In 1f each variant was kicked and run until it reached exact rest or 4000 substeps, and displacements were
read at that moment. The reference (32 iterations, eps 1) never rests, so it was read after 4000
substeps of its own creep, the variants with a quantum after tens or hundreds. There was no run without a
kick, every variant had settled its own pile, and nobody had checked that 32 iterations are a converged
solve of a pile 12–20 layers deep. **The 1f statement "small responses are suppressed 3–20x" is
withdrawn.** What follows replaces it.

Design of the remeasurement (pre-registered): one geometry per pile (three piles of 1728, seeds 11–13, at
exact rest under v2); on it one impulse state, the exact integer equilibrium (section 3); velocities zero.
Every variant starts from that state. Its control run (no event) must not move: it does not, for every
variant, to the last bit of position. Observables at fixed substeps 6 … 800. Events: 18 kicks per speed
(three piles x six exposed surface spheres) at 0.25, 1, 4 m/s downward and 1 m/s sideways; the +x wall
moved 20 cells away (slump); gravity tilted by tan θ = 0.2 … 0.5; one buried sphere removed (18 events).

## 2. What is wrong with the physics, measured

### 2.1 The reference does not converge for a kick into the pile

| kicked sphere, geometric-mean ratio | down 0.25 | down 1 | down 4 | side 1 |
|---|---|---|---|---|
| 128 / 64 iterations (substep 800, 18 kicks) | 0.82 | 0.82 | 0.89 | 1.01 |
| 256 / 128 iterations (substep 400, 6 kicks) | 0.51 | 0.63 | 0.76 | 1.00 |

How far a sphere kicked into a pile moves keeps falling with the iteration count, without a limit in sight.
The contact law has no compliance: a fully converged rigid frictional pile absorbs a small kick without
moving, and what the solver produces instead is a compliance set by its iteration count. **This observable
is not defined by the model**, so no rest rule can be right or wrong about it. A sideways kick (the sphere
slides over the surface; the model has no rotation) is resolved: 0.99–1.01 between 64, 128 and 256 iterations. The sum of
displacements of all bodies is resolved within 15 %.

Consequence for the project: a physically meaningful response of a pile to a small push needs a physical
compliance in the contact law (a stiffness of the contact, then a finite converged answer that does not
depend on iterations). Nothing below adds one; it is the main open item.

### 2.2 The error is the iteration count, not the quantum

Kick ensemble at substep 800 against 128 iterations without quantum (`kick_ensemble_vs_128_iterations.json`).
Each cell: kicked-sphere ratio / sum-of-displacements ratio / bodies moved more than 0.01 D (1 mm), variant vs reference.

| variant | down 0.25 | down 1 | down 4 | side 1 |
|---|---|---|---|---|
| 4 it, eps 0 | 3.02 / 0.91 / 1 vs 3 | 5.83 / 1.81 / 39 vs 11 | 2.99 / 2.89 / 310 vs 73 | 1.10 / 1.29 / 70 vs 54 |
| 4 it, eps 32 (1f default) | 3.14 / 0.26 / 1 vs 3 | 5.82 / 1.36 / 39 vs 11 | 3.00 / 2.77 / 319 vs 73 | 1.12 / 1.09 / 69 vs 54 |
| 4 it, eps 128 | 2.96 / 0.17 / 1 vs 3 | 5.82 / 1.25 / 39 vs 11 | 3.00 / 2.81 / 318 vs 73 | 1.10 / 1.07 / 68 vs 54 |
| 16 it, eps 0 | 1.68 / 0.92 / 1 vs 3 | 2.31 / 1.23 / 17 vs 11 | 1.74 / 1.66 / 163 vs 73 | 1.06 / 1.11 / 63 vs 54 |
| 32 it, eps 0 | 1.58 / 0.95 / 1 vs 3 | 1.73 / 1.14 / 12 vs 11 | 1.37 / 1.35 / 120 vs 73 | 1.05 / 1.11 / 59 vs 54 |
| 32 it, eps 32 | 1.40 / 0.08 / 1 vs 3 | 1.72 / 0.41 / 12 vs 11 | 1.38 / 1.11 / 120 vs 73 | 1.05 / 0.84 / 58 vs 54 |
| 128 it, eps 32 | 0.66 / 0.04 / 1 vs 3 | 0.99 / 0.22 / 7 vs 11 | 1.01 / 0.70 / 67 vs 73 | 0.99 / 0.77 / 54 vs 54 |

* **Too few iterations make the pile too soft.** At 4 iterations a sphere kicked downward sinks 3–6 times
  too far, and after kicks of 1–4 m/s 3.5–4.2 times too many bodies move by more than a millimetre.
  Nothing is suppressed; the opposite.
* **The quantum leaves the millimetre scale alone.** At equal iterations eps 0 … 128 change neither the
  kicked sphere nor the count of bodies above 0.01 D.
* **What the quantum does remove is the far field.** The sum of displacements drops 3–25x at 0.25 m/s.
  That sum is made of about 1700 bodies moving 5·10⁻⁵ D each (5 µm on a 10 cm sphere, 12 position units).
  In the reference those bodies keep creeping for the whole run: about 350 of 1728 still change at
  substep 800, and the sum grows by up to 30 % between substeps 100 and 800. A rigid model has no length scale that
  would make that motion physical; at game tolerance it is nothing.

### 2.3 The same holds for the other events

From `events_and_cost.json`, three piles per variant. Reference: 64 iterations, fine cone, no quantum.

| event | 4 it (v3) | 16 it (v3) | 32 it (v3) | reference |
|---|---|---|---|---|
| slump at 3840: bodies moved > 1 D, per pile | 8, 18, 14 | 8, 14, 21 | 8, 16, 17 | 8, 15, 17 |
| slump: shift of the centre of mass, D | 0.0113 | 0.0105 | 0.0096 | 0.0095 |
| tilt 0.2: bodies moved > 0.1 D, per pile | 14, 15, 14 | 5, 9, 10 | 3, 8, 10 | 1, 9, 8 |
| tilt 0.5: shift of the centre of mass, D | 0.14, 0.08, 0.09 | 0.06, 0.04, 0.05 | 0.045, 0.028, 0.042 | 0.036, 0.024, 0.037 |
| support removal: bodies > 0.01 D, all 18 events | 104 | 76 | 70 | 66 |
| support removal: moving within 6 substeps | 18 of 18 | 18 of 18 | 18 of 18 | 18 of 18 |

The slump (a large rearrangement) is right at every iteration count. Under a sideways body force the
pile creeps too far: at tan θ = 0.5 its centre of mass moves 2.5–3.9 times the reference at 4 iterations,
1.1–1.25 times at 32. Removal of a support is felt at once by
every variant; the extent of the response follows the iteration count.

### 2.4 Stacks: a long-range dependency that Jacobi does not resolve

A column of L spheres at rest, gravity then tilted suddenly; the static friction coefficient is 0.5
(`stack_tilt_*.json`). Below the limit (tan θ = 0.45) the column must hold, above (0.6) it must slide as one block.

| law | L = 3 | 8 | 16 | 32 |
|---|---|---|---|---|
| v2 / v3, 4 it, tan 0.45 | shears | shears | shears | shears |
| v3, 62 it, tan 0.45 | holds | shears | shears | shears |
| 128 it, no quantum, tan 0.45 | holds | shears | shears | shears |
| two-pass prototype with rigid groups (section 6), tan 0.45 / 0.6 | holds / block | holds / block | holds / falls apart | holds / falls apart |

The friction that keeps a column together has to travel its whole height within one substep. Jacobi
moves it one contact per iteration, and the column shears at any practical iteration count. This is the
dependency the two-pass idea addresses, and the prototype does hold every column; section 6 says why it
is still not usable.

### 2.5 Single bodies: the friction cone was too coarse

The tangential impulse was quantised in steps of 4F (a velocity of four position units per substep).
A sphere sliding on a slope accelerated 3–10 % too fast just above the static limit. Resolving the cone
relatively (the tangential part scaled by a power of two to 14 bits, `cone: 1`) brings it to 0.6 %
(`analytic_v2.json`, `analytic_v3.json`). Stopping distance of a sphere launched along the floor: within
2 % from 0.4 m/s, 9 % short at 0.09 m/s, half at 0.012 m/s, where one substep of the quantum is a large
share of the motion.

## 3. Equilibrium: it exists, it is not unique, and it is the memory

On the frozen geometry of a resting pile, projected Gauss–Seidel in float64 finds contact impulses that
balance every body, inside a cone narrowed to 0.45 (warm-started from the stored impulses: 128 sweeps for
512 bodies, 240–256 for 1728, 430–460 for 4096; on one pile of 4096 it stalls at an imbalance of 77 units, 2 % of a
weight). Rounding to integers leaves an imbalance of a few units per body. **It is carried to the world
frame along a maximum-load spanning forest** (Kruskal by normal load, the world as one more vertex), after
dropping the contacts that carry less than 64 units. The result balances every body exactly in integers,
and it is a fixed point of v3 and of the fine-cone law: zero bodies change in three substeps, on every pile
tried (one of 512, three of 1728, three of 4096), including the one where the float solve had stalled
(`equilibrium_pile*_seed*.json`). The coarse cone of v2 still moves 72–185 bodies on two of the three
piles of 4096. The narrowed cone matters too: with the full static cone (0.5) the solution lies on the
boundary, rounding pushes 2–16 contacts outside, and 45–550 bodies move.

Two simpler versions failed, and how they failed matters for any future solver of this kind. With a
breadth-first forest the rounding imbalance collected on lightly loaded wall contacts at the roots and
pushed them out of their cones. Without the minimum load, contacts carrying 1–16 units (a weight is 3552)
sat anywhere relative to their cones after rounding. Either way tens to hundreds of bodies started to move.

**The force network is not a function of the geometry.** Started from zero impulses on the same piles of
1728, the solve does not converge in 200 000 sweeps: the imbalance stays near one body weight on all three piles, and
the network it reaches differs from the one next to the stored impulses by 58 % (relative L1). On 512 bodies
it does converge, and there **two exact integer equilibria of the same geometry, both fixed points of v3
(zero bodies change), differ by 54 %**. The coarse cone of v2 holds neither of them. A resting pile is statically
indeterminate: which of the many balanced networks it holds is its history. That network is the physical
memory of the world at rest, and it is exactly what the contact slots store. A runtime that rebuilt it
from the geometry would not find the same pile, and would struggle to find one at all.

## 4. Law v3: rest as an exact balance of contact impulses

> **Corrections (Phase 1h, 2026-11-13, [`long-range.md`](long-range.md)).** (1) Not every event of this phase
> came to exact rest: plain v3 at 32 iterations never rested on one tilt event (pile 12, tan 0.5); the data had it,
> the text did not. (2) The rest rule below has a trap. A body just above the threshold that sits on a body just
> below it slides forever, because the anchor the end of the substep applies to the lower body is invisible
> to the iterations. A lab fix (`reff`) is described in 1h and is not yet part of the law. (3) The integer projection of the
> equilibrium was replaced at the end of this phase; the ensemble runs of this report all used the older one, consistently.

What changed against v2, all in `rest2.js` (JS and WGSL):

* **Handover.** A body below the threshold (32 units per axis) that arrived at rest relaxes no contact
  with another body below the threshold. Instead its whole residual goes, through one contact, to its
  designated support, antisymmetrically; the support feels it one iteration later, and the cone applies as
  always. The residual of a body that arrived at rest is a force imbalance, and the impulse that takes it
  rightly persists.
* **Designated support:** the lowest touching partner among the contacts that carry a real share of the
  body's load (at least four thresholds along the normal and at least a quarter of the largest), else the
  lowest touching one. The floor comes first, a side wall counts as level, the ceiling never. Its key is
  a 16-bit hash of its position that must be unique in the body's list, published in the flag word. Two
  bodies that designate each other exchange nothing.
* **Momentum is never stored in an impulse.** A body that arrived with a velocity hands nothing on. At the
  end of the substep a supported body slower than the threshold is set to rest. The world frame takes that
  momentum once, recorded as before, and the record vanishes when the body rests on balanced contacts.

Each rule exists because the previous version failed on the GPU or on a pile. Handing on momentum as well
left a body bouncing forever between two contacts. A support chosen by height alone could have a
saturated cone, so the handed residual was clipped and returned each substep. A support whose key
collided with another's in the same list received nothing. Mutual designation cancelled out. A body with
no partner below kept its residual.

| result | v2 (1f) | v3 |
|---|---|---|
| share of the weight on the world frame at rest, 1728 / 4096 | 0.39–0.47 % | 0 on all 18 piles of 1728 (4, 16, 32 it); 0 on 5 of 6 runs of 4096 (4, 16 it), 0.0003 % (2 bodies) on one |
| floor plus walls carry the weight | 100.00–100.03 % | 100.0000 % |
| exact rest, piles of 1728 (18 piles, 4/16/32 it) | 703 (seed 11) | 611–849, every pile |
| exact rest, piles of 4096 at 4 it / 16 it | 696–875 | 673–770 / 692–814 |
| 64-sphere column at 4 / 16 / 32 it, substeps to rest | 16 412 | 10 195 / 2 415 / 1 177 |
| from the anchored rest state of v2 to an exact balance | – | 8–10 substeps, 6–7 full-substep equivalents of work, largest motion 4·10⁻⁴ D |

The quantum is still there for momentum: below 32 units a supported body stops at the end of a substep.
What changed is that at rest nothing leans on the world frame; the stored impulses are the balance. v3
does not change the consequences: kick ensemble 2.33 / 3.78 / 2.56 / 1.06 of the reference at 4
iterations against 2.58 / 4.80 / 2.65 / 1.12 for v2; slump closer to the reference (bodies > 1 D 8, 18,
14 against 9, 22, 27). Its gain is exactness and a little speed, not a different response.

## 5. Change-driven iterations: more iterations at the price of fewer

The physics asks for more iterations (section 2). Inside a substep most bodies of an active region stop
changing after a few. On the GPU an iteration of a body now copies its input to its output when neither
it nor any partner changed an impulse in the previous iteration (bit 13 of the flag word, set by the
match for every body it recomputed). That would reproduce the same values anyway, so the result is
bitwise unchanged: the v2 signatures of 1f are reproduced word for word. The test also replaces the
distance test of the solve.

Body evaluations per kick (`events_and_cost.json`, mean over 18 kicks, change-driven count):

| | down 0.25 | down 1 | down 4 | side 1 | substeps to exact rest after down 1 (median) |
|---|---|---|---|---|---|
| v2, 4 it | 0.30 M | 0.82 M | 1.32 M | 0.89 M | 102 |
| v3, 4 it | 0.23 M | 0.58 M | 1.04 M | 0.64 M | 89 |
| v3, 16 it | 0.16 M | 0.42 M | 0.89 M | 0.58 M | 29 |
| v3, 32 it | 0.15 M | 0.37 M | 0.90 M | 0.59 M | 19 |
| one full substep of the 1728 pile at 4 it | 0.010 M | | | | |

A wake at 32 iterations costs no more body evaluations than at 4: it ends five times sooner, and most
iterations touch few bodies. Where everything moves there is nothing to skip, and the cost is
proportional to the iterations.

## 6. Two-pass propagation over the support order: prototype, not adopted

Following Xiong et al. 2025, but bodies stay Jacobi bodies (JS only, sequential, by height). The upward
pass brings each body to rest against the bodies below and defers the reaction. The downward pass hands
the deferred reactions down. Then the Jacobi iterations run as usual. Variants: tp1 both passes; tp3 only
contacts that stay inside the static cone; tp4 as tp3, plus trees of bodies stuck to their most loaded
support treated as rigid groups.

| | stacks (section 2.4) | pile of 1728: kick 1 m/s | slump | tilt 0.3 |
|---|---|---|---|---|
| tp1 | – | about 1500 bodies in motion at substep 800 (energy grows) | 671 bodies > 1 D | 1675 bodies > 0.1 D |
| tp3 | hold at 0.45, fall apart at 0.6 | fine, rests in 25–100 substeps | fine (9 bodies > 1 D) | **872 bodies > 0.1 D (reference about 20)** |
| tp4 | hold at 0.45, block slide for L ≤ 8 | about 1100 bodies in motion | 247 bodies > 1 D | 1534 bodies > 0.1 D |

Sequential passes solve the chain that Jacobi cannot: a column holds. In a pile the same passes either
pump energy or let a pile below its angle of repose flow. With one local sweep per body this is not a
physics anyone would want, and it is not ported to the GPU. The routing of 1f failed the same way:
faster settling, changed consequences. Long-range dependencies need a method that is checked against the
pile, not only against the column.

## 7. Determinism

| | M3 Pro | GTX 1660 SUPER |
|---|---|---|
| v2, 1728 x 400 / 8000 x 600 (signatures of 1f) | b5ab16ed / b9d80fd5 | b5ab16ed / b9d80fd5 |
| v2, incremental through settle, kick, wake: 22 checkpoints | 0ef4fc07, 0 mismatches | 0ef4fc07, 0 mismatches |
| v3, 1728 x 400 / 8000 x 600, GPU = JS | 6b4b0e21 / 6fbbfca2 | 6b4b0e21 / 6fbbfca2 |
| v3, incremental, 22 checkpoints | fb1c1fc6, 0 mismatches | fb1c1fc6, 0 mismatches |
| v3, fuzz of 1240 host edits, three configurations | c58c3ab3, 0 mismatches of 75 | c58c3ab3, 0 mismatches of 75 |
| v3, forced overflow (243 268 contacts dropped, counted) | 82efb59b | 82efb59b |
| v3 at 16 iterations: 1728 x 400 / 8000 x 600 / incremental | 42c1ca8e / 7002da71 / 56c657fc | same |
| v3, permutation of the arrays with 40 pairs of coincident bodies | 0 words differ | 0 words differ |
| same without canonical ids | 222 words differ | 222 words differ |

**The array index is no longer part of the physics.** Bits 1–31 of `pos.w` hold a canonical id assigned
by the host. It travels with the body and is the only thing that breaks an exact tie: the order in
contact selection and the normal of two coincident centres. Without ids the old behaviour remains, and
the last row shows that it was real.

## 8. Contact overflow: outside the valid domain

On a sphere, 17 points are at most 51.09° apart (Tammes). Seventeen partners within the search margin
therefore force two of them to overlap by at least 12.3 % of the diameter; thirteen force 2.8 %. Compressing a
settled pile uniformly (`contact_capacity.json`) gives the first body with more than 16 partners at a
pair overlap of 30 %. The piles here have at most 8–10 contacts, and no scene of this phase dropped a
contact. Overflow can only occur after a failure that is already visible: interpenetration of tens of percent. The
deterministic counted drop stays as the guard for that case.

## 9. A cheap error indicator, tried

The impulse taken by the world frame during a wake, relative to the momentum of the kick, orders the
iteration counts correctly on average (2.1, 1.2, 0.65, 0.36 for 4, 8, 16, 32). Kick by kick it does not
predict the error of the response: the correlation with the log error of the kicked sphere is 0.15 over
all 288 kicks, and between −0.23 and −0.35 within one iteration count. As an indicator of a single event
it is not usable.

## 10. GPU time

Timestamp queries, ms per substep, the same wide cloud of 8 layers as in 1f (`results/phase1/xdev3/time_*.json`,
summary `results/phase1/g/gpu_time_summary.json`). "Full" steps every body every substep; "inc" is the
incremental pipeline; both are run on the same world, and their states are identical at rest and after
the wake on both devices (hashes in the summary, equal across devices). Wake: a sphere on the surface kicked at
1 m/s down and 0.5 m/s sideways, summed over windows of 20 substeps until nothing is evaluated.

| bodies | device | law, iterations | all moving: full / inc | at rest: full / inc | wake, total ms: full / inc (exact rest after) |
|---|---|---|---|---|---|
| 204 800 | M3 Pro | v2, 4 | 1.65 / 1.48 | 2.95 / 0.087 | 296 / 18.2 (100) |
| | | v3, 4 | 1.66 / 1.50 | 3.31 / 0.090 | 258 / 16.9 (80) |
| | | v3, 8 | 2.11 / 2.13 | 4.28 / 0.101 | 254 / 16.8 (60) |
| | | v3, 16 | 3.43 / 3.29 | 6.63 / 0.160 | 386 / 19.3 (60) |
| | | v3, 32 | 5.65 / 5.56 | 10.41 / 0.251 | 418 / 27.7 (40) |
| 204 800 | GTX 1660 SUPER | v2, 4 | 1.22 / 1.12 | 3.31 / 0.126 | 329 / 23.0 (100) |
| | | v3, 4 | 1.30 / 1.15 | 3.62 / 0.126 | 288 / 19.7 (80) |
| | | v3, 8 | 1.66 / 1.65 | 4.45 / 0.151 | 266 / 18.9 (60) |
| | | v3, 16 | 2.34 / 2.29 | 6.19 / 0.201 | 369 / 24.2 (60) |
| | | v3, 32 | 3.51 / 3.16 | 9.69 / 0.303 | 384 / 49.1 (40) |
| 524 288 | M3 Pro | v2, 4 | 3.76 / 3.55 | 7.53 / 0.118 | 1047 / 33.4 (140) |
| | | v3, 4 | 3.81 / 3.75 | 8.74 / 0.102 | 1062 / 25.4 (120) |
| | | v3, 16 | 9.28 / 9.17 | 16.89 / 0.160 | 1058 / 21.1 (60) |
| | | v3, 32 | 14.57 / 14.63 | 27.89 / 0.256 | 1682 / 44.5 (60) |
| 524 288 | GTX 1660 SUPER | v2, 4 | 2.57 / 2.47 | 8.11 / 0.126 | 1132 / 36.8 (140) |
| | | v3, 4 | 2.69 / 2.59 | 8.88 / 0.127 | 1064 / 28.8 (120) |
| | | v3, 16 | 4.57 / 4.58 | 14.86 / 0.196 | 888 / 26.7 (60) |
| | | v3, 32 | 6.93 / 6.96 | 22.89 / 0.292 | 1371 / 80.5 (60) |

* **The cheap rest of 1f survives the better physics.** At 4 iterations v3 rests at 0.090–0.127 ms per
  substep against 0.087–0.126 for v2: 2.7–3.5 % of full stepping at 205k bodies, 1.2–1.4 % at 524k. Its wakes are cheaper:
  16.9 / 19.7 ms instead of 18.2 / 23.0 at 205k, 25.4 / 28.8 instead of 33.4 / 36.8 at 524k, and they reach exact rest sooner (80 substeps instead of
  100, 120 instead of 140).
* **Iterations are cheap where little changes, and only there.** A wake at 16 iterations costs 0.83–1.23
  times the wake at 4: it ends twice as soon, and change-driven iterations skip most bodies. At 32 the
  factor is 1.6–2.8. Rest costs about 6 µs more per iteration; that is the fixed price of a dispatch over an
  empty list. Where everything moves there is nothing to skip: full stepping grows with the iteration count,
  2.6–3.8 times from 4 to 32.
* GTX: at 524k with 16 and 32 iterations the first runs ended with a lost device. A command buffer of 50
  full substeps runs for about a second there, close to the Windows driver watchdog. With at most 10
  substeps per command buffer (`batch: 10`, the trajectory is unchanged) both runs completed and gave the
  same hashes as the M3 Pro. A runtime has to bound the GPU time of one submission, not only its average.

## 11. Verdict against the pre-registration

| id | verdict |
|---|---|
| G1 | **neither pass nor kill, the question was wrong.** The default does not suppress kick responses; it exaggerates a downward kick 3–6x, and the cause is the iteration count. The quantum removes only micrometre motion of the far field. The 1f claim is withdrawn. |
| G2 | **pass** (after two failed projection schemes, section 3): exact integer equilibrium inside a cone narrowed by 10 %, fixed point of v3 on every pile tried (1728 and 4096); with the full cone it is not |
| G3 | **pass**: two exact equilibria of one geometry, both at rest under v3, differ by 54 %; from zero impulses the solve does not even converge on 1728 bodies; the network is history and must be stored |
| G4 | **pass**: zero anchors at rest (one pile of 4096 keeps 0.0003 %), floor and walls carry 100.0000 %, every pile rests, no slower than v2 |
| G5 | **pass for v3 against v2**: consequences unchanged or closer. The analytic checks pass for single bodies with the fine cone, **fail for stacks** at any Jacobi iteration count; that failure is inherited from Jacobi, not from handover |
| G6 | **kill**: a 64-column needs 10 195 substeps at 4 it (target 300). Two-pass propagation holds columns but breaks piles; not ported |
| G7 | **pass**: 8–10 substeps from anchored rest to exact balance |
| G8 | **pass**: JS = M3 Pro = GTX 1660 SUPER everywhere; at 4 iterations rest 0.090–0.127 ms and wakes 17–29 ms, both at or below 1f (section 10) |
| G9 | **pass**: canonical ids, permutation with coincident bodies identical |
| G10 | **pass**: overflow needs about 30 % overlap; it never occurred |

## 12. What follows

* **The model lacks a compliance.** How far a small push moves a pile is the iteration count of the
  solver, not physics. Before any further optimisation of rest or wake, the contact law needs a physical
  stiffness (a soft-constraint formulation whose converged answer is finite), and the reference has to
  converge for the kick.
* **The physics error is the iteration count, and change-driven iterations make iterations cheap where
  little changes.** 16–32 iterations cost the same per wake as 4. They cost proportionally more only when
  everything moves.
* **Rest is now an exact balance, and the balance is the stored history.** Nothing leans on the world
  frame. The memory of a resting world is its impulse network; it has to be kept, not recomputed.
* **Long-range friction chains (stacks) are unsolved.** Sequential propagation fixes columns and breaks
  piles. Any successor must pass the pile tests of this phase, not only the column.
* Not done: memory per body (663 B at 524k bodies, 676 B at 205k; v3 adds nothing, the 600 B target of 1f is still missed), rotation and rigid clumps,
  1e as a subsystem of the runtime, phones.
