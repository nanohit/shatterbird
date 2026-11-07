# Phase 0d — Adaptive multirate engine: patches, explicit active zones, re-reduction

*Started 2026-11-07. Sections 0–3 were written before any Phase 0d run. Later changes to them are marked **post-hoc**. Success criteria are the ones of Phase 0c §0 (mechanical consequences); nothing here adds pointwise-accuracy gates.*

## 0. What Phase 0c left open, and why it needs one system, not three fixes

Phase 0c showed that intact material can be coarse: quasi-static loading and support removal pass every consequence criterion. Three things blocked a general result.

| blocker in 0c | root cause | what a system has to do instead |
|---|---|---|
| automatic zone cost 0.5–2× explicit | the reduced region was "everything not detailed", condensed exactly (M0) on an irregular interface: dense blocks of hundreds of unknowns, full rebuild at every expansion | the reduced description must not depend on where the zone is: **fixed patches** with precomputed bases; activation removes patches, it rebuilds nothing |
| fragment velocity −27 % without a post-hoc 0.26 ms window; impact thresholds +28–39 %, 35–88 % of impact energy lost | BDF2 at a game step integrates the fast parts (crack release, contact pulse of tens of µs) | fast physics must be integrated at its own time scale, and only where it happens: **explicit active zones sub-cycled inside the implicit game step**, coupled without losing momentum or energy |
| no worst case of mass collapse; detailed set only grows | no way back from detailed to reduced | **re-reduction**: quiet detailed material returns to patches, its projection loss is booked as a separate energy term |

All three are the same statement: *the variables used to describe a piece of material, and the time step it is integrated with, are chosen locally and change with the state.* Phase 0d builds that engine and tests it on the 0c scenes, on a scaling family and on a collapse.

## 1. Design

### 1.1 State: patches and active cells

At scene build the cells are partitioned into **patches** of diameter ℓ_p (square bins of side ℓ_p, split into facet-connected components, components under 4 cells merged into the neighbour they share most facets with). Deterministic, no dependence on load or zone.

Every cell is in one of two states:

- **R (reduced)**: its patch is closed. The patch moves as a floating-frame body (0c §4 formulation): x_i = c + R(φ)(P_i + U_i η), θ_i = φ + W_i η.
- **E (active)**: its patch is open; every cell has its own (x, y, θ).

Free plates (impactors) are always E bodies.

**Patch basis (multiscale, interface-reduced).** B = cells of the patch with any facet leaving the patch (or to a support, or loaded); I = the rest.
- Boundary traces: u_x, u_y and θ of the B cells follow polynomials of total degree ≤ P in the local coordinates;
- interior by static condensation T_I = −K_II⁻¹ K_IB T_B (exact response of the patch interior, so stress concentrations inside a patch are kept for these traces);
- plus 2 body-force modes;
- rigid modes removed; M-orthonormal (T_eᵀ M T_e = I, mean-axis conditions).

This is the interface reduction asked for in 0c: an interface between two closed patches, or between a patch and an active zone, carries only polynomial traces, whatever its shape. The basis is computed once per patch and cached; a patch rebuilt after damage (re-reduction, §1.5) gets a fresh basis.

**Facets.**

| facet | evaluated where | how |
|---|---|---|
| inside a closed patch | R | ½ ηᵀ K̂ η (must stay undamaged: monitored) |
| between two closed patches, or patch–support | R | full nonlinear SB-IL1 / contact on recovered cell positions |
| touching an E cell | E | full law at every explicit sub-step |

So damage can only evolve in E or on R–R facets; the monitor (§1.4) makes R–R damage an activation event, so **all damage evolution is integrated explicitly**.

### 1.2 Time integration

> **Post-hoc:** the final engine (§5) uses variable-step BDF2 for everything with local-error control, and explicit windows chosen by cost; §1.2–1.4 describe the multirate design as first built.

- **R**: BDF2 incremental potential at the macro step H (game step, H = 1/240 s), one Newton system over all closed patches. The Jacobian is lagged: one factorisation is reused over iterations and steps and refreshed only when the R set changes, the step changes, or convergence slows. Backward-Euler restart after any change of the R set.
- **E**: velocity Verlet at h_E = 0.8 · 2/√λ_max(M_E⁻¹K_E), λ_max by power iteration on the active zone with its interface facets, recomputed when the zone changes. History is committed at every sub-step, as in the explicit reference. N_s = ⌈H/h_E⌉ sub-steps per macro step.

Explicit integration of the active zone costs no numerical dissipation, needs no factorisation, and is the GPU-friendly, bit-reproducible kernel of the earlier phases.

### 1.3 Multirate coupling (impulse-matched)

> **Post-hoc.** As built (§4.2), the coupling differs from this outline:
> - R uses the implicit midpoint, not BDF2;
> - the Robin term is the Schur complement of the active zone;
> - the fixed point uses IQN-ILS, not Aitken;
> - acceptance is on the interface energy jump;
> - the interface force is taken at the midpoint of each boundary jump.
>
> The engine used for the final results is the implicit adaptive mode of §5.

Over one macro step [t_n, t_n + H], with interface facets F_I between E cells and R cells:

1. The E zone is sub-cycled with the R side moving linearly in its generalised coordinates z (c, φ, η) from z_n to a trial end state z⁽ᵏ⁾. Cell positions are recovered exactly from z(t).
2. The interface forces on R cells are accumulated as a generalised impulse J⁽ᵏ⁾ = Σ_s h_E G(t_s)ᵀ f_I(t_s).
3. R is solved implicitly with the constant generalised force J⁽ᵏ⁾/H, plus a Robin term ½(z − z⁽ᵏ⁾)ᵀ K_I (z − z⁽ᵏ⁾). K_I is the interface facet stiffness projected on R; it stabilises the iteration and vanishes at convergence.
4. Aitken-relaxed fixed point on z until the interface cells' end positions change by less than 5 % of the smallest onset opening δ₀ of the interface facets. A last R solve without the Robin term then makes the impulse R receives equal and opposite to what E received.

Properties by construction:
- linear momentum: E's interface impulse and R's are the same sum (R in the BDF2 momentum p̃ = M(3vₙ₊₁ − vₙ)/2; equal to M v at BE steps);
- interface work: the work done on E by R's motion equals minus the work done on R (to first order in h_E), because both are Σ_s f · Δq with the same Δq;
- the only numerical dissipation is BDF2's, in R, where nothing breaks. It is measured per step from R's own energy balance and booked apart.

If the coupling iteration does not converge in 12 iterations, the macro step is halved.

### 1.4 Activation (R → E)

After a macro step, before accepting it:
- **material**: any R-internal or R–R facet point with χ_rel = δ_eq / max(κ_c, δ₀) ≥ χ_trig;
- **contact**: any contact or unilateral pad point on an R cell that is open at t_n and is predicted to close within 2H at current closing speed (impactor, support pads, ground, crack faces);
- **coupling**: the coupling iteration fails twice in a row at the minimum macro step.

If any of these fires, the step is rejected. Every patch containing a hot point, and every patch whose centroid lies within r_e = ℓ_p of it, is opened: its cells get the exact prolongation of the R state (position and velocity). The step is then redone. Opening only removes patches from the R system; nothing is rebuilt.

### 1.5 Re-reduction (E → R)

After each accepted macro step, an open patch is a candidate when:
- it has been open for ≥ 3 macro steps;
- no point on its facets has damage growth over the last 3 steps;
- every point on its facets has χ_rel < χ_rr = 0.5;
- no contact touching its cells is closing.

Its cells are split into components connected by **undamaged** facets (κ_c ≤ δ₀). Each component becomes a new closed patch: the cached basis if it equals the original patch, otherwise a fresh one, built in the reference configuration since undamaged facets are linear. A single cell is a rigid patch with no deformation modes. Damaged and broken facets become R–R facets with the full law.

**Projection.**
- rigid part (c, φ) by the M-orthogonal mean-axis fit;
- deformation η by the K-orthogonal fit in the body frame, so internal strain energy can only drop;
- velocities by M-orthogonal projection, which keeps linear momentum exactly and kinetic energy by Pythagoras.

The total energy of the affected terms is evaluated before and after. Re-reduction is accepted only if it does not create energy and the loss is ≤ ε_rr = 5 % of the component's own (kinetic + strain) energy, or ≤ 10⁻⁴ of the scene energy scale. The loss is booked as **D_red**, never as fracture energy.

### 1.6 Energy ledger

E_tot = KE + U + V_g + D_frac + D_red + D_fric − W_ext.

The drift E_tot(t) − E_tot(0) is the numerical error. It is split into:
- R's BDF2 dissipation, from R's per-step balance with the interface work;
- the rest: E integrator oscillation and coupling mismatch.

No-creation gate: E_tot(t) − E_tot(0) ≤ 10⁻³ of the scene energy scale.

### 1.7 Cost model

Same flop model as 0c:
- 200 flops per facet point with tangent, 60 force-only;
- factorisation from the symbolic Cholesky;
- region, coupling, recovery and monitor terms.

New terms:
- E sub-steps, at 60 per facet point, plus body updates;
- interface recovery and impulse accumulation;
- coupling iterations, counted in full;
- re-reduction and basis builds.

Wall-clock is reported next to it, single thread for both engine and reference.

Expected scaling, for a structure of N cells with active zones of N_E cells, per unit simulated time:

  W_engine ≈ n_it · N_E · c_E / h_E + N · c_R / H + (factorisations, amortised)

  W_explicit ≈ N · c_E / h_E

The marginal cost of one more intact cell is ≈ (c_R/c_E)(h_E/H) of explicit, with H/h_E ≈ 10³. This is the scaling claim the experiments test.

## 2. Development protocol

Free parameters: ℓ_p, P, χ_trig, ε_rr, coupling tolerance.
- They are chosen in development runs on seed 1 and on coarse meshes, and then frozen.
- Final runs use seeds 1 and 2 (S1, S2), or the fixed scene (S3, S4).
- Any parameter changed after a final run is marked post-hoc, and the run is reported both ways.

## 3. Experiments and pre-registered criteria

**S0 — component checks (gates for correctness of the implementation, not results).**
- S0a: the engine with every patch open equals the 0c explicit reference on the support-removal scene: tip trajectory and D to ≤ 10⁻⁸ relative.
- S0b: every patch closed, elastic cantilever under prop release, same H: patch model vs full BDF2. Static tip deflection, first period and first peak each within 2 %. This chooses ℓ_p and P.
- S0c: elastic cantilever with an a priori open zone (x < 0.4 m) coupled to closed patches, vs the explicit reference:
  - first peak, period and mean within 5 %;
  - no energy creation;
  - linear momentum of the whole system consistent with the external impulse to 10⁻⁶ relative (BDF2 momentum).

**S1 — support removal (0c Exp 2 scene, d_min = 0.04 m, seeds 1, 2).** The engine starts with every patch closed and activates automatically. There is no fracture window and no step rule other than §1.2–1.4.

Criteria are the 0c §4 criteria 2–4, unchanged:
- threshold λ_crit (bisection [0.6, 1.6], 6 steps) within 10 %;
- at 1.25 λ_crit,ref:
  - separation in both;
  - onset and separation within one 60 Hz frame;
  - crack x within 0.1 m;
  - D within ±10 %;
  - fragment v, ω and KE within 10 %;
  - no energy creation.

Numerical dissipation and D_red are reported apart. Work vs explicit is reported; the 0c a priori zone with its window achieved 5×.

**S2 — impact on a short beam (0c Exp 3 scene, d_min = 0.025 m, seeds 1, 2).** Criteria are the 0c §5 criteria, unchanged:
- threshold v_crit (log-bisection on [0.05, 10], 7 steps) within 10 %;
- at 0.75, 1.25 and 2.0 v_crit,ref:
  - impulse within 10 %;
  - |Δv_out| ≤ 0.1 v₀;
  - same fragment count, each mass within 10 %, each fragment velocity within 0.1 v₀;
  - crack x within 0.1 m;
  - D within ±10 %;
  - no energy creation.

Expectation, written now: little or no cost gain on this member, because the impact loads most of it (0c: zone 16–77 %).

**S3 — scaling: the same impact on continuous beams of n = 1, 2, 4, 8, 16 spans.** Depth 0.2 m, spans 1.8 m, unilateral support pads at every support, impact at the middle span, v₀ = 1.5 m/s, d_min = 0.025 m, N ≈ 450 n cells.

Criteria at every n against the explicit reference:
- impulse and rebound as S2;
- D within ±10 %;
- crack x within 0.1 m.

Scaling criterion: fit W(n) = a + b·N.
- The engine's marginal cost per cell, b_engine/b_explicit, must be ≤ 0.1.
- The work ratio at n = 16 must be ≥ 10×.

**S4 — collapse onto the ground (worst case).** A 6 × 0.3 m concrete beam, b = 0.3 m, d_min = 0.04 m:
- it rests on two supports 1.0 m above a rigid ground;
- one support is removed at t = 0, the other 20 ms later;
- the beam falls tilted, hits the ground (frictional contact, μ = 0.5) and breaks;
- run to 0.6 s.

Contacts:
- ground: cell circles against the half-plane;
- crack faces: through broken facets;
- broken facets are dropped when opened beyond half their length;
- then circle contact between cells not joined by a facet.

The reference uses the same contact model, all explicit. Because multi-fragmentation is chaotic, the reference is also run at h_E/2: its difference from itself is the noise floor.

Criteria:
- D within ±10 %, or 1.5 × the noise floor if that is larger;
- large-fragment count (≥ 5 % mass) within ±1;
- max difference of the cumulative large-fragment mass distribution ≤ 0.1;
- no energy creation;
- D_red and numerical dissipation reported, each as a fraction of the impact kinetic energy.

Reported, not gated:
- peak and time-averaged active fraction;
- work against the reference.

This is the worst case asked for in 0c: everything touches the ground at once, and only re-reduction brings the zone back down.

---

## 4. Development log: what was built, what changed, and why

Sections 0–3 above were written before any run. Everything from here on is after the fact; design changes are marked **post-hoc**.

### 4.1 Component checks (S0)

**S0a — the active zone is the 0c explicit reference.** Every patch open, the same sub-step: tip trajectory identical bit for bit over 0.06 s of fracture (max difference 0), D to 9·10⁻¹⁵ relative. `results/phase0d/s0a.json`.

**S0b — patch accuracy** (elastic cantilever, all patches closed, against single-cell patches = full implicit at the same H, BDF2):

| ℓ_p | P | unknowns (full: 2148) | first peak | period | mean |
|---|---|---|---|---|---|
| 0.1–0.4 m | 1 | 110–1306 | **+10…+90 %** (locking) | −4…−58 % | +9…+87 % |
| 0.1–0.4 m | 2 | 200–2047 | ≤ 0.31 % | 0 | ≤ 0.36 % |
| 0.1–0.4 m | 3 | 320–2122 | ≤ 0.11 % | 0 | ≤ 0.13 % |

Affine traces lock in bending; quadratic traces are exact enough at every patch size up to the full depth. P = 2 is used everywhere.

A finding that shaped the rest: at H = 1/240 even the *unreduced* implicit model costs 58× less per simulated second than explicit; patches add at most 1–3× on top. **The decisive gain is temporal** (large steps for quiet material), the spatial reduction is secondary in 2D. Patches matter for factorisation size (decisive in 3D) and make opening and closing free (precomputed bases).

**S0c — multirate coupling** (elastic cantilever, open zone x < 0.4 m, closed patches elsewhere, against explicit):
- first peak +0.41 %, period 0 %, mean +0.08 %;
- energy rise 0, interface work on E and on R equal to 10⁻⁴ J;
- 2 coupling iterations per step, work 3.9× below explicit.

`results/phase0d/s0c.json`.

### 4.2 The multirate coupling: four corrections needed before it conserved energy

| symptom (development, S0c scene) | cause | fix (in §1.3 now) |
|---|---|---|
| energy rise ≫ KE, oscillation of period 2H | the impulse enters R as the step's mean force (trapezoidal), BDF2 treats internal forces at the step end; a stiff interface mode is then undamped and the BDF2 velocity reconstruction amplifies it | R uses the **implicit midpoint** in impulse form (x̂ = x + H v/2, α = 4): the natural partner of a mean impulse |
| contraction 0.97 per iteration | Robin term = interface springs only; the active zone (clamped) is much stiffer than R | Robin term = **Schur complement of the active zone** (secant stiffness + 4M/H²) projected on R: contraction 10⁻³ |
| energy created at "converged" steps | 10⁻⁸ m of position mismatch against a stiff active zone is kN of force; the momentum-exact final solve then jumps the interface by A⁻¹·rf | accept on the **energy of the interface jump** (≤ 10⁻⁶ of the scene scale per step), not on positions |
| residual creation ∝ h_E (6·10⁻³ → 3·10⁻³ → 1.5·10⁻³ J for h, h/2, h/4) | the R boundary moves in jumps between sub-steps; the force before the jump does ½ S Δb² too little work | interface force evaluated at the **midpoint of each boundary jump**, the same for E's kick and R's impulse: energy 0 |

### 4.3 Where the multirate coupling failed (S1 development, seed 1, λ = 1.15)

With fracture, the coupling met three problems that no tuning removed:
- **Strong, non-smooth coupling.** The tip fragment hangs on the crack while it releases, and the clamped stub rings at kHz. An E–R interface through such material needs 4–12 iterations, or does not converge, because damage growth makes the fixed-point map non-smooth.
- **The explicit zone is never quiet.** Without damping, ringing material keeps χ up and the zone cannot close.
- **Chattering.** Patches close and reopen.

Mitigations built and kept:
- closing by connected component with a joint energy check;
- separate thresholds for damaged facets (only χ_rel ≥ 1 reopens, since reloading to their own history is not damage);
- exponential back-off;
- a predictor from the exact interface force;
- opening the interface patches when the coupling does not converge.

With them the mechanics were right: fragment v within 1 %, ω 2 %, D 0.9 %. But the zone spread to most of the beam (85 % explicit on average) and the cost was 2–6× the explicit reference.

**Post-hoc design change.** The engine was redesigned as the *implicit adaptive mode* of §5. The multirate machinery is kept:
- as the explicit reference;
- as the explicit window of §5;
- in a local-impact test (§6.6) that shows where it pays and what it still lacks.

## 5. Implicit adaptive mode (post-hoc design; parameters frozen before the final runs)

**One Newton system, two kinds of patches.**
- **Coarse patches:** the §1.1 basis.
- **Fine patches:** single cells, plates as single bodies (partly fixed DOFs held as generalised Dirichlet conditions).
- **Split.** Opening a coarse patch replaces it by its cells; cell histories persist, so BDF2 continues.
- **Merge.** A quiet group of split cells is merged back:
  - components through undamaged facets;
  - K-orthogonal position fit and M-orthogonal velocity fit;
  - joint energy check, loss booked as D_red.
- There is no partitioned coupling and no iteration between subsystems.

**Split triggers** (on the end state of every step; the step is redone):
- an undamaged point of a coarse patch at χ ≥ 0.9;
- a damaged point at χ_rel ≥ 1, i.e. damage would grow;
- a contact on a coarse cell predicted to close within 2H faster than χ_lte·σ_t/(ρc).

**Time integration.**
- **Integrator.** Variable-step BDF2 (ω = h_n/h_{n−1}, no restarts on step changes) with event control Δd ≤ 0.1.
- **Step control by local error** — the difference between the implicit solution and the Taylor predictor (O(h²) after a restart, O(h³) otherwise):
  - measured on facet and contact **openings**, not positions, because that is what drives force and fracture;
  - only where fracture or impact can happen: material points with χ_rel ≥ 0.5 and contacts closing faster than χ_lte·σ_t/(ρc) ≈ 0.18 m/s (the speed whose impact stress ρcΔv is half the strength);
  - tolerance 0.1 δ₀ + 2 % of the step's opening change.
- **Unresolved oscillations.** Ringing of intact material does not limit the step; BDF2 damps it, and the loss is booked (D_num).

**Cost switch.** The work per simulated second of the implicit mode is compared, every step, with explicit on the whole model.
- **Measure:** mean work per step over the last 12 steps, divided by the current step.
- **Entry:** when implicit is dearer for 2 consecutive steps, the whole model runs explicitly (a *window*).
- **Exit:** when fracture dissipation over the last 1 ms falls below 1 % of the total. The cells re-enter as fine patches and merge as they quiet.
- **Re-entry:** a window reopened within 5 ms lasts twice as long.

By construction, the engine is not much dearer than explicit.

**Contact (for S4):**
- cell circles, radius half the smallest neighbour-centroid distance, so neighbours just touch at rest;
- penalty k_c = E b scaled by cell size;
- the ground against polygon vertices, with the stiffness of a support facet;
- Coulomb friction: lagged normal force and regularised in the implicit mode, stick-limited impulse in the explicit mode;
- broken facets handed to contact when their cells have separated, slid or turned, and only when that is energy-neutral.

**Energy ledger:**

E_tot = KE + U (facets + contact) + V_g + D_frac + D_red + D_fric (+ D_rad) − W_ext.

Reported apart:
- D_num (BDF2);
- D_red (projection);
- D_fric.

**Frozen parameters:**

| parameter | value |
|---|---|
| H | 1/240 s |
| ℓ_p | 0.2 m at d_min 0.04, 0.1 m at d_min 0.025 (≈ 4–5 d_min) |
| P | 2 |
| χ_trig | 0.9 |
| r_e | ℓ_p |
| χ_rr | 0.7 (0.9 for damaged) |
| ε_rr | 5 % |
| local error | 0.1 δ₀ + 2 % |
| χ_lte | 0.5 |
| cost switch | 1.0 |

They were chosen on seed 1 of S1 and S2 in development.

**Post-hoc fix (§6):** the first step without history is H/16, not H.

## 6. Results (implicit adaptive mode, frozen parameters)

**Runs:** `./target/release/phase0d s1 | s2 | s3 | s3b | s4 | s4b | s5 | s1noise`. Outputs are in `results/phase0d/<exp>/`; tables come from `python3 analysis/phase0d_tables.py`. Work uses the §1.7 flop model; wall-clock is single thread for both engine and reference.

**Post-hoc fix: the first step.** With no history the implicit mode started with a backward-Euler step of the full H. When a support is released at t = 0, that step damps the initial jerk. It was found by S5:
- the 2 m cantilever did not break (D −99 %);
- starting at H/16 and growing the step fixed it.

Every experiment was rerun with the fix. The first-run results are kept in `results/phase0d/v1_before_start_fix/` and differ as noted per experiment. S4 was not rerun: the reference itself is degenerate there (§6.7).

### 6.1 S1 — support removal (pre-registered criteria of 0c §4)


| | seed 1 | seed 2 |
|---|---|---|
| λ_crit ref / engine | 0.920 / 0.936 (+1.7 %) | 0.905 / 0.920 (+1.7 %) |
| separation, onset, separation time | yes | yes |
| crack x ref / engine | 0.229 / 0.236 m | 0.178 / 0.170 m |
| D | -0.3 % | -2.0 % |
| fragment v ref / engine | (0.232, -2.936) / (0.183, -2.964): 1.9 % | (0.136, -2.931) / (0.517, -2.499): 19.6 % ✗ |
| fragment ω ref / engine | -0.305 / -0.246 ✗ | -0.209 / -0.901 ✗ |
| fragment KE | +1.7 % | -24.4 % ✗ |
| energy creation | 0.00e+00 J | 0.00e+00 J |
| numerical dissipation / D_red | 11.7 J / 7.1e-04 J | 11.0 J / 2.6e-04 J |
| work / wall-clock vs explicit | 9.1× / 5.2× | 7.7× / 5.0× |
| cells with own DOFs (mean) | 13 % | 27 % |


**Passes on both seeds:**
- threshold +1.7 % (first run: +5.1 and +6.9 %);
- separation, onset and separation time;
- crack location;
- D;
- no energy creation.

**Fails:** fragment motion on seed 2 (v 20 %, ω 4×, KE −24 %) and fragment ω on seed 1 (−19 %).

**Cost:** 9.1× and 7.7× less work, 5.2× and 5.0× less wall-clock, on a 716-cell structure. 13–27 % of cells carry their own DOFs on average.

**Diagnosis: crack-path branches** (§6.5). The reference itself switches between fragment motions that differ by 20 % in velocity and 4× in rotation, within fractions of a per mille of load. The engine lands on one of the reference's own branches, but with other frequencies.

The pre-registered ±10 % criterion on fragment velocity is ill-posed at such a bifurcation. Phase 0c's criterion had the same flaw: its reference at λ = 1.15 sat on branch A. It is reported as failed.

### 6.2 S2 — impact on the short beam (pre-registered criteria of 0c §5)


| seed | v_crit ref / engine | factor | all criteria | D | impulse | rebound ref / engine | fragments | num. dissipation / KE₀ | work / wall |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.477 / 0.477 (+0.0 %) | 0.75 | pass | -1.3 % | -0.0 % | 0.2310 / 0.2307 | 1 / 1 | 10 % | 0.96 / 0.96 |
| 1 | 0.477 / 0.477 (+0.0 %) | 1.25 | pass | -0.2 % | +0.0 % | -0.0043 / -0.0041 | 2 / 2 | 7 % | 0.94 / 0.92 |
| 1 | 0.477 / 0.477 (+0.0 %) | 2.0 | pass | -2.0 % | -0.1 % | -0.3150 / -0.3154 | 2 / 2 | 4 % | 0.95 / 0.94 |
| 2 | 0.540 / 0.540 (+0.0 %) | 0.75 | pass | -1.8 % | -0.0 % | 0.2711 / 0.2710 | 1 / 1 | 16 % | 0.95 / 0.93 |
| 2 | 0.540 / 0.540 (+0.0 %) | 1.25 | pass | -0.2 % | -0.1 % | -0.0185 / -0.0192 | 2 / 2 | 7 % | 0.95 / 0.93 |
| 2 | 0.540 / 0.540 (+0.0 %) | 2.0 | pass | -2.3 % | -0.0 % | -0.3433 / -0.3436 | 2 / 2 | 3 % | 0.95 / 0.94 |


**Every pre-registered criterion passes on both seeds, and the thresholds are exact.** This is the regime Phase 0c failed: thresholds were then +28–39 %, with 35–88 % of KE₀ lost to numerical dissipation.

The contact pulse runs in the explicit window. Afterwards BDF2 damps the beam's ringing. That loss (3–16 % of KE₀) is booked and does not change the outcome, which was decided inside the window.

There is no cost gain on a member the impact loads entirely, as written in §3.

### 6.3 S3 — scaling, impact on continuous beams (pre-registered)


| spans | cells | consequences | work ratio | wall ratio | cells with own DOFs |
|---|---|---|---|---|---|
| 1 | 454 | pass | 0.89 | 0.80 | 100 % |
| 2 | 863 | pass | 0.86 | 0.77 | 100 % |
| 4 | 1694 | pass | 0.87 | 0.70 | 100 % |
| 8 | 3361 | ✗ | 0.87 | 0.79 | 100 % |
| 16 | 6609 | pass | 0.87 | 0.81 | 100 % |

marginal cost ratio b_engine/b_explicit = 1.16


At every size:
- D within 3 %;
- impulse within 0.2 %;
- same fragments and rebound.

At 8 spans the dissipation-weighted crack centroid is 0.117 m off against 0.1 m, with the same fragments; that criterion is marked failed.

**The scaling criterion fails**: the marginal cost ratio is 1.16 against ≤ 0.1. At 1.5 m/s the impact is not local: the flexural wave runs through all spans, and the beam breaks over a support several spans away. The cost switch keeps the whole model explicit, and the engine is 11–14 % dearer in work and 20–30 % dearer in wall-clock.

### 6.4 S3b — local impact on walls of growing width (**post-hoc**)

Wall 1 m high, width 1–16 m, on a foundation, 50 kg at 1.5 m/s at the top centre. Local cracking under the impactor, no fragments.


| width (m) | cells | consequences | work ratio | wall ratio | cells with own DOFs |
|---|---|---|---|---|---|
| 1.0 | 425 | pass | 0.91 | 0.90 | 100 % |
| 2.0 | 858 | pass | 0.94 | 0.92 | 100 % |
| 4.0 | 1679 | pass | 0.85 | 0.77 | 100 % |
| 8.0 | 3329 | pass | 0.82 | 0.74 | 100 % |
| 16.0 | 6620 | pass | 0.81 | 0.78 | 100 % |

marginal cost ratio b_engine/b_explicit = 1.25


**No gain either.**
- During the contact pulse the implicit step must be of the explicit one's order. One Newton iteration over all coarse patches (projection of their inter-patch facets) then costs more than one explicit sub-step of the whole wall.
- The impactor keeps loading the crack, so the window does not close within 30 ms.

**The multirate coupling of §1.3** — an explicit zone around the impact, the quiet far field implicit — is what would pay here. On the 8 m wall, development run:
- 3.3× less work, 2.2× less wall-clock, 6 % of cells active;
- but D **2.4× too high** and the impactor rebound 1.09 against 0.85 m/s.

The reduced far field at H cannot carry the stress waves, and the interface reflects them back into the active zone. A Lysmer dashpot (ρcA) on the interface facets brought D to +28 %. But it created 17 J (31 % of KE₀) and raised the rebound to 1.27 m/s: the naive dashpot is not consistent with the impulse-matched coupling.

### 6.5 Fragment branches near the S1 threshold (post-hoc noise study)

Loads 1.25 λ_crit,ref × (1 + k·0.0025), k = −4 … +4 (±1 %). `summary_s1_noise_wide.json`.

The tip fragment leaves on one of three motions:

| branch | v_x (m/s) | ω (rad/s) |
|---|---|---|
| A | ≈ 0.52 | ≈ −0.9 |
| B | ≈ 0.23 | ≈ −0.30 |
| C | ≈ 0.14–0.18 | ≈ −0.21…−0.24 |


| | seed 1 | seed 2 |
|---|---|---|
| reference | B ×3, C ×6 | C ×9 |

- **Reference elsewhere:** branch A at λ = 1.15 (seed 1) and λ = 1.13 (seed 2).
- **Common to all runs:** D and separation; the engine's D is within −0.3…−2 %.
- **Seed 1:** the engine now sits on the reference's dominant branch C in all 9 runs. The first run, before the start fix, was spread over A, B and C.
- **Seed 2:** it prefers A, 8 of 9.
- **On a shared branch:** the engine agrees to 6 % in v and 2 % in ω.

**Not reproduced near threshold: the frequency of the crack-path branches.** This is a real limitation for fragment motion close to the fracture threshold. Further above it (S2, S5, S4b seed 2) the fragment outcome is the same.

Its likely cause is the time integration of the release: BDF2 at a controlled but finite step on the fine cells, against the undamped explicit reference. It was not isolated here.

### 6.6 Mass failure without rubble: S4b (**post-hoc**)

Continuous beam 12 × 0.4 m, b = 0.3 m, 2124 cells, on 7 supports. The 5 intermediate supports are released at t = 0 (2 ms ramp), and the beam fails at several sections at once. Run to 0.5 s; references at h_E and h_E/2.


| | seed 1 | seed 2 |
|---|---|---|
| large fragments ref / engine (kg) | 3 / 2 (1044, 422, 1971 / 1466, 1968) | 2 / 2 (1581, 1850 / 1581, 1850) |
| D ref / ref h/2 / engine | 54.5 / 51.8 / 48.8 J (-10.5 %, noise 5.0 %) ✗ | 59.8 / 59.2 / 56.5 J (-5.4 %, noise 1.0 %) |
| mass distribution distance (noise) | 0.12 (0.12) ✗ | 0.00 (0.00) |
| energy creation | 0.0e+00 J | 0.0e+00 J |
| num. dissipation / D_red (× KE_max) | 0.08 % / 3e-07 | 0.07 % / 8e-08 |
| cells with own DOFs mean / peak | 71 % / 100 % | 92 % / 100 % |
| work / wall-clock vs explicit | 1.89× / 1.13× | 1.18× / 1.08× |


- **Seed 2:** the engine reproduces the mass failure exactly (same fragments to the kilogram, D −5 %).
- **Seed 1:** it misses one of three cracks (1466 = 1044 + 422 kg) and D is −10.5 %. The reference at h_E/2 differs from itself by the same mass-distribution distance (0.12) and by 5 % in D: multi-section failure is chaotic.
- **First run (before the start fix):** the seeds were the other way round. Seed 1 was exact; seed 2 had one crack too many (+14 %).
- **Zone size, the worst case asked for:** during the failure the whole beam is detailed (100 %). The mean is 71–92 %, and the cost gain is small (1.2–1.9× work, 1.1× wall).
- **Mass failure is the regime where the reduction buys least:** most material is either cracking or ringing at the strength level.

### 6.7 S4 — collapse onto the ground (pre-registered scene; support timing and contact geometry post-hoc)

The beam swings about the right pad and breaks near it. It falls 1 m and hits the ground.

**In the reference, at h_E and at h_E/2, the beam is ground to single cells.** D reaches the total facet capacity (2045–2067 J), and no fragment of 5 % mass remains. Friction does 9.5 kJ of work. The reference's own energy balance loses 560–690 J (9–11 % of KE_max) that nothing books: the explicit contact and friction treatment is itself not energy-consistent at this level.

Development runs with circle-only ground contact, with frictionless ground, and with polygon-vertex ground contact all pulverise too. The cause is the cell–cell contact by circles smaller than the cells: rubble behaves like a granular flow in which every collision breaks more facets. This is a model limitation, not a property of concrete. The engine reproduces the same pulverisation and is 2× dearer: it is mostly explicit, and returns to the implicit mode cost it.

**S4 cannot be judged with this contact model.** A polygon–polygon contact is a prerequisite. The first-run engine numbers are in `results/phase0d/s4/`.

### 6.8 S5 — support removal on growing cantilevers (**post-hoc**)

Cantilevers 2, 4, 8 m (depth 0.4 m) with the same root moment (λ = 1.15·(4/L)²), run over the same number of first-mode periods (t_end ∝ L²). The fracture is local in space at the root and global in time.


| length | cells | separated ref / engine | D | fragment Δv | ω ref / engine | work: explicit / engine (Gflop) | work ratio | wall ratio | cells with own DOFs |
|---|---|---|---|---|---|---|---|---|---|
| 2 m | 360 | True / True | -1.6 % | 4.1 % | -2.167 / -2.057 | 5.5 / 2.00 | 2.8× | 2.4× | 46 % |
| 4 m | 716 | True / True | -0.3 % | 1.2 % | -0.305 / -0.265 | 43.0 / 5.71 | 7.5× | 4.2× | 21 % |
| 8 m | 1428 | True / True | +0.1 % | 0.1 % | -0.065 / -0.064 | 341.6 / 17.34 | 19.7× | 8.3× | 8 % |


**The gain grows with size.**
- Explicit work grows like L³ (cells × time).
- The engine's grows like L^1.5 (5.7 → 17.3 Gflop per doubling: × 3.0).
- The share of cells with their own DOFs falls from 46 % to 8 %.

Consequences: D within 1.6 %, fragment velocity within 4 %, rotation within 13 %. The 13 % is at 4 m, which is the S1 seed-1 case of §6.5.

This is the scaling the reduction was built for: a local event in a structure that is otherwise quiet.

### 6.9 Determinism

Two runs of the engine on the S1 scene (fracture, splits, merges, an explicit window) give **bit-identical** trajectories: every recorded quantity, 17 digits. The engine is single-threaded, its loops are ordered, and there are no hash maps on the computation path.

Cross-device bit-exactness was not tested. It needs:
- the same operation order;
- no FMA contraction;
- deterministic transcendental functions (sin, cos, atan2 in the floating frames).

The explicit windows are the integer GPU kernel of Phase 0b. The implicit part is a small sparse Cholesky on the CPU.

## 7. Synthesis

**The question.** Can fracture be obtained correctly without continuously computing every internal oscillation of the material?

**Answer.** Yes for the consequences asked of it, in every regime tested except fragment-motion statistics near threshold and rubble. Where it pays, it pays because quiet material is integrated with large steps and coarse variables.

| regime | consequences | cost against explicit | not computed (booked) |
|---|---|---|---|
| quasi-static loading (0c Exp 1) | < 1 % | 5–30× | — |
| support removal (S1, 716 cells) | threshold +1.7 %, D ≤ 2 %, timing, crack location; fragment motion on the reference's own branches, other frequencies near threshold ✗ | **7.7–9.1× work, 5.0–5.2× wall** | ringing (≈ 11 J against ≈ 4800 J fragment KE) |
| support removal, growing size (S5) | D ≤ 1.6 %, v ≤ 4 % | **2.8× → 7.5× → 19.7×** work for 2 → 4 → 8 m; engine ∝ L^1.5, explicit ∝ L³ | ringing |
| impact, short member (S2) | **all pre-registered criteria pass**; thresholds exact | ≈ 0.95× | post-pulse ringing, 3–16 % of KE₀ |
| impact, continuous beams 1–16 spans (S3) | pass (one crack-centroid 0.117 m) | 0.87× (explicit fallback); **scaling ✗** | — |
| local impact, walls 1–16 m (S3b) | pass | 0.8–0.9× (explicit fallback) | — |
| mass failure, no rubble (S4b) | one seed exact, one seed one crack short; the reference differs from itself as much | 1.2–1.9× | ringing, < 0.1 % KE |
| collapse onto ground (S4) | not judgeable: the contact model pulverises the reference | 0.5× | — |

**What the engine is, as a system.** The variables of each piece of material and its time step are chosen locally, by three measured quantities:
- **χ against strength** — split, merge, and whether a point needs time accuracy;
- **local error of the openings that drive force and fracture** — the step;
- **cost** — implicit or explicit.

Every non-physical sink is booked apart from fracture energy: D_num, D_red, D_fric. No run of the engine created energy.

**Scaling, honestly:**
- **Won:**
  - slow events in large quiet structures, such as support removal and gravity-driven failure: S5, with the gain growing as L^1.5;
  - quiet material in general: the implicit mode costs 1/60–1/160 of explicit per cell per second.
- **Not won yet:** fast events — contact pulses of tens of µs — and events engaging the whole structure.
  - During a pulse the implicit step must be of explicit size, and a Newton iteration over coarse patches costs more than an explicit sub-step of the whole structure.
  - The cost switch then runs everything explicitly. This is correct, with 5–15 % overhead, but it scales like explicit.
- **Worst case (S4b):** the zone reaches 100 % during the failure and averages 71–92 %. Mass failure is where reduction buys least.

**Next, in order:**
1. **Energy-consistent absorbing interface for local multirate zones.** The far field's radiation impedance belongs inside the impulse-matched coupling, with an automatic choice between local multirate and global explicit. This is what turns local impacts into a size-independent cost; the coupling already gave 3.3× on the 8 m wall, with the reflection error as the blocker.
2. **Cheaper coarse iterations:**
   - precomputed linear patch–patch couplings in the corotated frame;
   - matrix-free residuals;
   - so the implicit mode wins already at moderate steps, and explicit windows shrink.
3. **Branch statistics near threshold (§6.5):** isolate whether BDF2 on the fine cells biases the crack release. Candidates: a tighter local-error tolerance on damaging points, or explicit sub-steps on them alone.
4. **Rubble:** polygon–polygon contact, energy-consistent with friction, before any collapse onto the ground can be judged.
5. **3D:** factorisation scaling is where patches decide; the explicit windows map onto the Phase 0b GPU kernel.

**Unrelated.** The pre-existing unit test `solvers::twolevel::gs_and_pcg_converge` (Phase 0b) fails on the committed tree as well. It was left as is.
