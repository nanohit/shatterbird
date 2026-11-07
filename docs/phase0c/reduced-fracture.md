# Phase 0c — Fracture with reduced intact regions

*Started 2026-11-07. Sections 0–2 were written before any Phase 0c run. Any later change to them is marked **post-hoc**.*

## 0. Scope

**Question.** Can fracture be obtained correctly without continuously computing every internal oscillation of the material?

**Hypothesis H.** Whole structures and strong flying fragments can be represented coarsely. The detailed RBSN network and the cohesive law are kept only where damage, contact or dangerous stress develops. The size of these detailed zones is measured, including the worst case of mass collapse.

**What fracture must get right.** These are mechanical consequences:

| required | not required |
|---|---|
| carrying capacity, load redistribution, significant structural vibrations | equal instantaneous forces everywhere in an undamped network |
| onset load and the region where the crack starts | χ of every facet equal to 0.01 at the same instant |
| no systematic spurious fracture; event order preserved; visible reaction within a game step | onset time to fractions of a millisecond |
| fracture character, broken volume, distribution of large fragments | exact dynamic crack path, identical damaged facets |
| whole regions coarse, detail where needed | every cell with independent elastic dynamics at all times |
| fast waves and reflections approximated or excluded, with the supported regimes stated | fast waves, reflections and their phase reproduced |

**Kept without relaxation:**
- cross-device determinism;
- mass and momentum accounting;
- energy consistency;
- strength and fracture tied to geometry and material.

Numerical damping is allowed. It must be measured and kept apart from fracture energy, and it must not become a hidden way to make a test pass.

**Tolerance.** About 10 % on loads and energies (game tolerance).

### Relation to Phase 0b/3

Phase 0b/3 judged every candidate against one standard: the full undamped network, compared pointwise in time. This was the §4.4 table of `docs/phase0b/architecture-decision.md`, covering force histories, onset time, and χ per facet on a union set. That answered a question nobody asked. Its finding "8 ms not met at ≥ 1k cells" holds only for the formulation "every cell, every wave, all the time".

Phase 0b/3's measurements stay valid as measurements:
- GPU throughput;
- bit-exact integer kernel;
- energy behaviour of the integrators.

Its acceptance and rejection verdicts are not used here. They are re-derived from the criteria above.

---

## 1. Formulation: reduced far regions coupled to a detailed zone

**Partition.** The cells are split into a detailed set 𝒟 and far regions R₁, R₂, … Within a far region, displacements are restricted to a basis:

  u_R = T_R ξ_R

**Facets entirely inside R** are assumed elastic. Their energy ½ ξᵀ K̂ ξ uses K̂ = T_Rᵀ K_RR T_R, a precomputed dense matrix of size m × m.

**All other facets** are evaluated with the full nonlinear corotated SB-IL1 law on the recovered positions of R's boundary cells:
- R–𝒟 facets;
- R–support facets;
- R–R′ facets.

**Consistency.** There is one potential and one Newton system. The reduced problem is the Galerkin restriction of the full one. Transfer of forces, displacements and energy is therefore consistent by construction: no multipliers and no separate interface law. External loads on R enter as T_Rᵀ f.

**Basis.** B denotes the boundary cells of R (those touching any non-internal facet) and I the interior cells. The interior follows by static condensation:

  T_I = −K_II⁻¹ K_IB T_B

| basis | T_B | m | role |
|---|---|---|---|
| **M0**, exact condensation | identity | ∝ D/d in 2D, ∝ (D/d)² in 3D | exact for a linear R; control that isolates the linearisation error |
| **M1**, section model | interface cells: rigid section (axial = tension, transverse = shear, rotation = bending) plus polynomial warping in the section coordinate up to degree P; support-pad cells keep exact DOFs | independent of the mesh | the candidate |

**Validity monitor.** After every accepted step, recover the R-internal facets and compute χ = δ_eq/δ₀ for the facets that are damageable in the reference. The run is valid only if χ < 1: the region it assumed elastic must stay elastic. The monitor's cost is counted.

**Stated limitation of the version used in Exp 1.** R is geometrically linear, which is valid for small rotations. Using it for large rotations creates spurious strain of order ½φ² at the interface. A corotated (floating-frame) region is required for Exp 2 and 3.

---

## 2. Exp 1 — notched beam (pre-registered)

**Specimen and solver.** This is the E6 setup unchanged:
- specimen 500 × 100 × 100 mm, span 400 mm, notch 50 mm;
- concrete with SB-IL1, damage allowed for |x − x_c| < 80 mm;
- unilateral pads and supports.

S0 bordered Newton runs under CMOD control to 1 mm. A step is accepted when it converges and Δd ≤ 0.1; otherwise it is halved. History is committed on acceptance only.

Quasi-static is the physically correct regime here, since T_load ≫ L/c. Neither model computes oscillations. The comparison isolates the spatial reduction. Full and reduced runs use the same Newton code and the same linear solver.

**Instances.**

| instance | d_min | seeds | purpose |
|---|---|---|---|
| lab beam | 5.6 mm | 1, 2, 3 | accuracy |
| lab beam | 4 mm | 1 | mesh trend |
| long beam: same section, L = 1.7 m, span 1.6 m (L/D = 17), damage zone ±80 mm | 5.6 mm | 1 | potential |

**Configurations.** w is the half-width of the 𝒟 zone, measured by cell centroid. The reference damage lies within |x − x_c| ≤ 21 mm on all nine E6 runs.

| | basis | w [mm] |
|---|---|---|
| primary | M1, P = 3 | 50 (0.5 D) |
| control | M0 | 25 |
| sweep (seed 1) | M1 with P ∈ {rigid, 1, 3} | 25, 50, 100 |

**Accuracy criteria.** Applied for each seed, reduced against full on the same tessellation:

1. Peak load within ±10 %.
2. Onset load, taken at the first committed material damage, within ±10 %. Onset location within 2 d_min.
3. Load–CMOD curve: max |ΔP|/P_peak ≤ 10 % over CMOD ∈ [0, 1 mm]. Initial stiffness within ±10 %.
4. Energy at CMOD = 1 mm: W_ext and the fracture dissipation D within ±10 %. Ledger |W − U − D| ≤ 10⁻³ W, which checks that the coupling neither creates nor loses energy.
5. Crack:
   - path x̄(y), dissipation-weighted in 5 mm bands: mean |Δx̄| ≤ 2 d_min;
   - height reached by the d ≥ 0.99 facets within ±10 % of the ligament;
   - spurious dissipation ≤ 5 % of D (dissipation in facets farther than 2 d_min from every facet that dissipates in the reference).
6. Validity: χ_R < 1 throughout.

**Work criterion.** The potential gain must be an order of magnitude or more, not 2×.

Total work is counted in flops:
- facet-point evaluations: 200 flops each with tangent, 60 without;
- factorisation: a pattern-based estimate (symbolic Cholesky flop count of the actual matrix), the same estimator for both models;
- triangular solves;
- coupling projection and K̂ products;
- the validity monitor;
- one-time precomputation, reported separately and amortised over the run.

Single-thread wall time is reported as well.

The potential is judged on the long beam with the primary configuration; the lab beam is reported as measured. Expectations written before running:
- lab beam: 4–8× fewer unknowns, 10–30× fewer factorisation flops;
- long beam: ≥ 20× fewer unknowns.

**Predictions.**
- M0 matches the full model within 0.5 % on loads (linearisation of R only).
- M1 with P = 3 and w = 50 mm stays within 5 %.
- M1 with the rigid section and w = 25 mm may fail: a stiff interface next to the process zone.

---

## 2R. Exp 1 — results

**Run:** `cargo build --release && ./target/release/reduced exp1` (about 9 min, 10 threads). Output: `results/phase0c/exp1/`:
- `summary_exp1.json`;
- `curves_*.csv`;
- `damage_*.csv`;
- `log_exp1.txt`.

A development run on the 8 mm mesh (`reduced quick`) was used to fix one coupling inefficiency before the matrix: the M1 projection is now done once per cell instead of per facet point. It changed work counts only.

### Primary and control configurations: all criteria pass on every instance

| instance | cells | config | detailed cells | unknowns (full → red.) | peak | load curve max ΔP/P | D | crack (Jaccard) | χ_R max | work ratio | wall ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lab 5.6 mm s1 | 1 091 | M1-P3 w=50 | 212 | 3 273 → 684 | +0.09 % | 0.34 % | +0.01 % | 1.00 | 0.40 | 5.3× | 3.7× |
| | | M0 w=25 | 101 | → 418 | +0.05 % | 0.05 % | +0.02 % | 1.00 | 0.66 | 9.2× | 8.0× |
| lab 5.6 mm s2 | 1 089 | M1-P3 w=50 | 211 | 3 267 → 675 | +0.05 % | 0.25 % | −0.01 % | 1.00 | 0.37 | 6.2× | 4.3× |
| | | M0 w=25 | 108 | → 427 | +0.05 % | 0.05 % | +0.01 % | 1.00 | 0.72 | 10.2× | 7.3× |
| lab 5.6 mm s3 | 1 100 | M1-P3 w=50 | 213 | 3 300 → 678 | +0.07 % | 0.27 % | +0.01 % | 1.00 | 0.31 | 6.7× | 4.5× |
| | | M0 w=25 | 109 | → 433 | +0.05 % | 0.06 % | +0.03 % | 1.00 | 0.84 | 10.5× | 7.5× |
| lab 4 mm s1 | 2 134 | M1-P3 w=50 | 420 | 6 402 → 1 314 | +0.05 % | 0.39 % | −0.02 % | 1.00 | 0.38 | 7.4× | 4.1× |
| | | M0 w=25 | 208 | → 778 | +0.05 % | 0.05 % | +0.03 % | 1.00 | 0.76 | 11.3× | 6.1× |
| **long 5.6 mm** (L/D = 17) | 3 705 | **M1-P3 w=50** | 207 | 11 115 → 666 | +0.03 % | 0.29 % | −0.02 % | 1.00 | 0.41 | **22.4×** | 13.2× |
| | | **M0 w=25** | 101 | → 421 | +0.04 % | 0.06 % | +0.02 % | 1.00 | 0.81 | **30.2×** | 28.9× |

Other quantities, on all of these runs:
- initial stiffness: within 0.9 %;
- W_ext: within 0.06 %;
- first fully broken facet: within 0.4 % in load, same facet;
- ledger: within 3.6·10⁻⁴;
- spurious dissipation: 0;
- the coupling facets never damaged.

The work ratio includes coupling, K̂ products and the monitor. Precomputation, amortised over the run, changes it by at most 2 % on the lab beam and 6 % on the long one: 30.2× → 28.3×.

### Sweep (lab 5.6 mm, seed 1): what the reduced model must keep

| w [mm] | rigid section (3 modes) | + linear warping (P1) | + cubic warping (P3) | M0 (exact) |
|---|---|---|---|---|
| 25 | **fails**: curve 15 %, K +36 % | **fails**: 15 %, +34 % | passes: 2.7 %, +5.7 % | passes: 0.05 % |
| 50 | passes: 2.5 %, +5.1 % | passes: 1.7 %, +3.3 % | passes: 0.34 %, +0.65 % | passes: 0.04 % |
| 100 | passes: 0.12 % | passes: 0.06 % | passes: 0.05 % | — |

Peak load and dissipation pass in every cell of the sweep: |Δpeak| ≤ 3.3 % and |ΔD| ≤ 0.4 %. The failures are in the elastic part of the curve.

A rigid-section ("plane sections remain plane") far field is enough once the interface is half a depth away. At a quarter depth, cubic warping is needed. Exact condensation works down to the damage zone itself.

### Pre-registered predictions against outcome

| prediction | outcome |
|---|---|
| M0 within 0.5 % | met: ≤ 0.06 % |
| M1-P3 at w = 50 within 5 % | met: ≤ 0.4 % |
| M1-rigid at w = 25 may fail | it failed |
| expected work 4–8× on the lab beam | 5–11× |
| expected ≥ 20× fewer unknowns on the long beam | 17× (M1-P3) and 26× (M0); work 22× and 30× |

The pre-registered "onset = first committed damage" happens at the first load step in every run, full and reduced: the notch-tip facet is loaded at once. That criterion is resolution-limited, though it passes trivially. Two post-hoc substitutes are reported:
- the first fully broken facet, within 0.5 %;
- the load at a 5 % secant-stiffness loss, within 3.2 % for every passing configuration and 12–13 % for the failing ones.

### Interpretation

The beam's fracture consequences are reproduced to well under 1 %:
- capacity;
- load–CMOD curve;
- onset;
- energy;
- the identical crack.

This holds when only the zone that actually damages (±21 mm, plus a margin) is a full RBSN. The rest becomes 20–60 generalised coordinates per side.

The cost of the full model grows with the structure. The cost of the reduced model grows with the damage zone. On the lab beam (L/D = 5) the gain is 5–11×. On a member with L/D = 17 it is 22–30×. It keeps growing linearly with the intact length.

**Limits stated:**
- 2D, quasi-static, small rotation (geometrically linear regions).
- The monitor recovers all region facets every step. That is 20 % of the reduced work on the long beam with M0 and should be screened, for example by stress modes on candidate facets.
- In 3D, M0's interface grows as (D/d)², so a section model is the scalable form. In 2D it already holds at w = 0.5 D.

---

## 4. Exp 2 — support removal (pre-registered after development runs, before the runs reported)

**Scene.** Concrete cantilever:
- 4 × 0.4 m, b = 0.3 m, under gravity;
- clamped on the left edge, with a prop on the last 0.2 m of the bottom edge;
- clamp and prop facets elastic; SB-IL1 everywhere else;
- d_min = 0.04 m, about 700 cells; seeds 1 and 2.

The prop force is released by a cosine ramp over t_r = 5 ms. A static free cantilever under self-weight would reach 0.92 of the elastic-limit root moment. Only the dynamic overshoot can break it, so this scene tests inertia and bending, not statics.

**Methods.**

| method | description |
|---|---|
| reference | full RBSN, leapfrog at h = min(0.3 d_min/c, 0.8·2/√λ_max(M⁻¹K)); history committed every substep |
| full BDF2 | full model, h = 1/240 |
| reduced, a priori zone | detailed x < 0.4 m (one depth); the rest is one floating-frame region (§1; basis M0 + 2 body-force modes + 4 fixed-interface modes); h = 1/240 and 1/480 |
| automatic zone | no detailed cell at the start. After each step, recover χ in every region; if χ ≥ χ_trig anywhere, reject the step, move every cell within r_e = 0.15 m of the hot facets into the detailed set, rebuild the regions, refit and redo the step. χ_trig = 0.8; h = 1/240 and 1/480 |

All implicit runs use BDF2 with a backward-Euler restart on step change and event control: accept a step if Newton converges and Δd ≤ 0.1, otherwise halve it.

**Development runs (d_min = 0.08 m, and elastic at 0.04 m).** These were done before this section was written.
- The reduced model matches full BDF2 at equal h to 4 digits.
- BDF2 damps the higher modes. Peak root moment against the reference: −7.5 % at h = 1/240, −4.3 % at 1/480, −2.1 % at 1/960. The first-mode overshoot is kept.
- At unit gravity, the coarse case broke in the reference but arrested in BDF2 at 1/240. A single near-critical case is therefore not a meaningful test, so the criterion is posed on the threshold.

**Criteria.**

1. **Inertia and bending (elastic, unit gravity).** First downward tip peak, oscillation period and mean deflection within ±10 %. No energy creation: E_tot(t) − E_tot(0) ≤ 10⁻³ KE_max. Numerical dissipation is reported.
2. **Fracture threshold.** Gravity is scaled by λ. λ_crit is the smallest λ at which the fragment separates by t = 0.3 s. It is found by bisection on [0.6, 1.6], 6 steps (±0.8 %), for each method and seed. Criterion: |Δλ_crit|/λ_crit,ref ≤ 10 %.
3. **Consequences at λ = 1.25 λ_crit,ref.**
   - separation in both;
   - onset and separation times within one 60 Hz frame;
   - crack location: dissipation-weighted x within 0.1 m;
   - fracture energy D within ±10 %;
   - fragment centre-of-mass velocity within 10 % of |v_ref|;
   - angular velocity within 10 %;
   - fragment kinetic energy within ±10 %;
   - no energy creation.

   The numerical dissipation of BDF2 is reported next to D and is never counted as fracture energy.
4. **Zone size.** Maximum detailed cells over total. Reported, not gated.

**Work.** The same flop model as Exp 1, plus:
- region inertia: O(m²) per iteration, O(n_R m) per step;
- the monitor;
- runtime rebuilds of regions, counted in full.

Reported against the explicit reference and against full BDF2 at equal h.

---

## 4R. Exp 2 — results

**Runs:**
- `./target/release/reduced_dyn elastic 1` (seed 1 only);
- `./target/release/reduced_dyn exp2 1 2`;
- post-hoc `SB_HFRAC=0.00026 ./target/release/reduced_dyn exp2 1 2`;
- diagnosis `./target/release/reduced_dyn conv 1`.

Output: `results/phase0c/exp2/`.

### Inertia and bending (elastic, unit gravity, seed 1)

| method | first tip peak | period | mean | peak root moment | energy lost / KE_max over 0.3 s | work vs explicit |
|---|---|---|---|---|---|---|
| full BDF2 1/240 | −5.2 % | +4.1 % | +2.5 % | −7.5 % | 0.47 | 29× |
| reduced, zone x < 0.4 m, 1/240 | −5.2 % | +4.1 % | +2.5 % | −7.5 % | 0.47 | **72×** |
| reduced 1/480 | −1.1 % | +1.1 % | +0.5 % | −4.3 % | 0.10 | 46× |
| reduced 1/120 | **−16 %** | **+14 %** | +4.4 % | −17 % | 1.1 | 104× |

- At equal step the reduced model reproduces full BDF2 to four digits: the spatial reduction is exact here.
- 1/240 and 1/480 pass. 1/120 fails.
- No run created energy.
- The missing 4–8 % of the peak moment is the higher-mode part: BDF2 at 1/960 still misses 2.1 %.

### Pre-registered criteria (fracture, seeds 1 and 2)

| | seed 1 | seed 2 |
|---|---|---|
| λ_crit reference | 0.920 | 0.905 |
| λ_crit, all BDF2 variants | +1.7 … +6.8 % | +3.5 … +6.9 % |

The threshold criterion passes for every method.

Consequences at 1.25 λ_crit,ref:
- Fracture and separation: yes in all.
- Crack location: same within 2 mm.
- D: −5 %.
- Onset: within 2 ms.
- Separation: 13.5–17.3 ms late. That is within one frame, except 1/480 on seed 2.
- **Fragment velocity and angular velocity fail in every BDF2 variant, full model included.** Reference v = (0.52, −2.52) m/s; BDF2 gives (−0.16, −2.34). |Δv|/|v| = 0.27, Δω = +27 %.
- Numerical dissipation: 0.1–0.2 % of the fragment's KE.
- The a priori zone x < 0.4 m reached χ_R = 1.15. Formally it was too small (the region should have damaged slightly). Results were identical to the full model.

### Diagnosis

The cause is the time resolution of the separation event, not the reduction or the threshold dynamics.

- Explicit at h_E and h_E/2: identical.
- Full BDF2 at 1/960: still on the wrong branch.
- Full BDF2 at 1/3840 (0.26 ms):
  - separation at 40.1 ms vs 39.8 ms;
  - v = (0.530, −2.523) vs (0.525, −2.519);
  - ω within 0.7 %.

### Post-hoc policy (marked post-hoc)

Step ≤ 0.26 ms while damage evolves or an undamaged facet is within 5 % of onset; otherwise the game step.

All pre-registered criteria then pass on both seeds for every variant:

| | value |
|---|---|
| threshold | +5.1 % and +5.2 % |
| fragment \|Δv\|/\|v\| | 1.1–1.9 % |
| fragment Δω | 0.3–2.6 % |
| D | −1 … −3 % |
| separation | 1–9 ms late |
| numerical dissipation | ≤ 0.12 % KE |

| post-hoc variant | detailed cells (of 713–716) | work vs explicit |
|---|---|---|
| full BDF2 + window | all | 1.4–1.5× |
| **reduced, zone x < 0.4 + window** | 69–71 | **5.1–5.6×** |
| automatic χ_trig = 1.0 + window | 68–88 | 1.3–2.1× |
| automatic χ_trig = 0.8 + window | 110–115 | 0.5–0.7× |

The automatic zone finds the right place and gives the right mechanics with no prior knowledge of the crack. Its cost is the problem: in the run without the window, 85 % of its work is in factorisation and coupling.

The zone grows along the stressed top fibre. Exact condensation (M0) on that long, irregular interface makes a dense block of several hundred unknowns. Each expansion also rebuilds the region.

This is the same lesson as Exp 1: the interface must itself be reduced, by section or patch modes, not condensed exactly. It was not done here.

## 5. Exp 3 — impact (pre-registered after development runs, before the runs reported)

**Scene.**
- Simply supported concrete beam 2.0 × 0.2 m, b = 0.2 m, span 1.8 m; d_min = 0.025 m, about 450 cells; seeds 1 and 2.
- Unilateral support pads 0.1 m wide.
- Gravity on the beam.
- Rigid flat impactor at mid-span: pad 0.1 m, mass 50 kg, moving in y only, unilateral normal-only pad facets, so it can rebound and cannot pull. It starts touching at speed v₀.
- Run to t = 0.05 s.

**Methods.**

| method | description |
|---|---|
| reference | full explicit leapfrog |
| full BDF2 | game step 1/240, at most 10⁻⁴ s while the impactor pad is in contact or closing within one game step, Δd ≤ 0.1 |
| a priori zone | detailed \|x − x_c\| < 0.2 m |
| automatic zone | χ_trig = 0.8, r_e = 0.075 m |

**Development runs.** These were done before this section was written. At d_min = 0.05 m and 0.025 m, v₀ = 3, 5 and 10 m/s:
- Full BDF2 and the automatic zone reproduce the reference fragments, the impactor exit velocity (within 1 %) and the impulse.
- The a priori ±0.2 m zone misses cracks outside it and is wrong.
- The automatic zone grows to almost the whole short beam.
- The beam already breaks at 1 m/s: 50 kg gives about 40 kN against a capacity of about 9 kN. The search range was moved to [0.05, 10] m/s.
- At 10 m/s the beam shatters into 6 pieces. That is fine dynamic fragmentation, outside the supported regime.

**Criteria.** Thresholds and consequences use the methods above.

1. **Threshold.** "Broken" means at least two fragments with ≥ 5 % of the cell mass each at t_end. v_crit is found by log-bisection on [0.05, 10] m/s, 7 steps (±2 %). Criterion: |Δv_crit|/v_crit,ref ≤ 10 %.
2. **Consequences at 0.75, 1.25 and 2.0 v_crit,ref.**
   - impulse M_p(v_out − v_in) within 10 %;
   - impactor rebound: |Δv_out| ≤ 0.1 v₀;
   - fragments: same count; each mass within 10 %; each centre-of-mass velocity within 0.1 v₀;
   - crack location within 0.1 m;
   - D within ±10 %;
   - no energy creation: E_tot rise ≤ 10⁻³ KE₀.

   Numerical dissipation is reported apart.
3. **Zone size.** Reported.

---

## 5R. Exp 3 — results

**Runs:**
- `./target/release/reduced_dyn exp3 1 2`;
- diagnosis `SB_HC=… SB_V0=0.6 ./target/release/reduced_dyn impact_one 1` (`results/phase0c/exp3/diag_contact_step.txt`).

Output: `results/phase0c/exp3/`.

| | seed 1 | seed 2 |
|---|---|---|
| v_crit reference (m/s) | 0.477 | 0.540 |
| full BDF2 (contact step 0.1 ms) | +39 % | +34 % |
| a priori zone | +34 % | +28 % |
| automatic zone | +39 % | +28 % |

**The threshold criterion fails.** Consequences:
- 0.75 v_crit and 1.25 v_crit: impulse, rebound and D fail. At 1.25 v_crit the BDF2 runs do not break, while the reference does.
- 2.0 v_crit: fragment count, masses and impactor exit velocity match, the latter within 0.01–0.1 m/s. Impulse is off by 4–15 % and D by 40–60 %.
- **Numerical dissipation is 35–88 % of the impact kinetic energy.** That is exactly the hidden energy sink the scope forbids.
- No run created energy.

**Diagnosis (v₀ = 0.6, seed 1).** Full BDF2 converges to the reference as the contact step shrinks:

| contact step | outcome |
|---|---|
| 0.1 ms | no break, rebound +0.21 m/s |
| 25 µs | breaks, fragment velocities halved |
| 6 µs | matches: v_out −0.004 vs −0.005 m/s; same fragments; velocities within 10 % |

At 6 µs the implicit run costs 2× the explicit one. The contact pulse of a stiff impactor on concrete lasts tens of µs. It sets the peak force, and the peak force sets the threshold. This is §6.7 of the context doc.

**Conclusion for impacts.** The implicit large-step path is the wrong tool during contact. The reduction itself is not at fault: the reduced variants match full BDF2 wherever full BDF2 is right.

The consistent design is explicit sub-stepping of the small detailed zone during a contact event. The reduced far field moves with the game step, coupled through the interface. This multi-rate coupling was not built or tested.

**Zone size.**
- On this short member the automatic zone covered 16–34 % below and near the threshold.
- At 2 v_crit it covered 68–77 % (up to 347 of 449 cells).
- When the member itself is the impact zone, there is little to reduce.

---

## 6. Synthesis

**Hypothesis H (coarse intact structures, detail only where damage develops).**

| regime | verdict | evidence |
|---|---|---|
| quasi-static loading (Exp 1) | confirmed | <1 % on every consequence; 5–11× less work on a lab beam, 22–30× on L/D = 17, growing with intact length |
| support removal (Exp 2) | confirmed for the mechanics; the work gain is only 5× and needs the post-hoc fracture-window step | threshold +5 %; fragments within 2 % |
| impact (Exp 3) | not shown | the time integrator fails during contact before the reduction matters |

**Zone sizes measured.** Detailed cells over total:

| case | detailed / total |
|---|---|
| notched beam, M0 | 9–10 % (long beam 2.7 %) |
| cantilever, automatic χ_trig = 1.0 | 10–12 % |
| short beam under impact | 16–77 % |

The worst case of mass collapse was **not measured**. Two parts are missing:
- a scene with many simultaneous failures;
- a re-reduction policy that returns detailed cells of separated, unloaded fragments to regions. The detailed set only grows in this implementation.

Without re-reduction, a collapse makes the detailed set grow to everything that ever cracked or collided.

**Supported regimes, as of now:**
- quasi-static and slow loading;
- support loss and structural dynamics, with game steps 1/240–1/480 and a ≤ 0.26 ms window while fracture is active.

**Not supported:** stiff impacts with large-step implicit integration.

**Next, in order:**
1. Interface reduction for automatic zones (section or patch modes on arbitrary interfaces), so the automatic zone keeps the a priori zone's 5× (and Exp 1's 20–30× at scale).
2. Explicit sub-stepping of the detailed zone during contact, coupled to the reduced field. Exp 3 repeated with it.
3. Re-reduction of separated fragments, and a mass-collapse scene measuring the peak detailed fraction.
4. A corotated (floating-frame) region in the quasi-static path, and 3D.

## Appendix — original outline of Exp 2 and Exp 3 (written before any run, superseded by §4 and §5)

**Exp 2: support removal.** Check:
- inertia;
- bending;
- fracture location;
- fragment motion;
- energy balance.

Oscillograms do not have to match. After that, the fracture zone is selected automatically by the χ monitor, without a known crack location.

**Exp 3: impact.** Check:
- fracture threshold;
- impulse;
- rebound;
- fragments;
- no artificial energy.

Exact shock waves and fine dynamic fragmentation are excluded.

**Both experiments report zone sizes:**
- detailed cells over total;
- reduced DOFs;
- the worst case of mass collapse.
