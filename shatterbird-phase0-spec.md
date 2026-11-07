# Shatterbird — Phase 0 Empirical Spec: Mechanics and Solver Truth (CPU, f64)

Read `shatterbird-context.md` first; §6 (mechanics) and §2–3 (constraints, determinism) are the background for everything here. This document defines the first empirical program: what to build, what to measure, what each result decides.

---

## 0. Agent handoff: first assignment and progression

**Current implementation assignment: E1 only.** Reproduce the exactly specified scalar ALM iteration and its analytic contraction, then measure displacement, force-distribution, constitutive and energy errors. This requires no Voronoi generator, damage implementation, S2 coupling, GPU code or full multi-crate engine. A negative convergence result is a valid result; it must not be presented as proof that a global solver is mandatory.

Completion evidence for this assignment: reproducible run/config, agreement with the analytic solution and contraction within declared f64 tolerances, error-versus-iteration outputs for the prescribed regimes, and a short interpretation stating the limits of the counterexample. Record assumptions and unresolved choices instead of silently filling them with an implementation default.

The rest of this document is the Phase 0 roadmap, not one undivided implementation task. Later assignments proceed through elastic E4/E2, constitutive closure and E5, then the remaining scaling, fracture, dynamics and 3D experiments. Before fracture experiments, complete the law and trial/committed-history contract in §3.2; before using S2, complete its coupling and factorization contract in §4.3.

Notation: $\Delta t$ = physical time step; $\ell_e$ = distance between the generators of facet $e$; $l$ = 2D facet length; $b$ = thickness; $A_e=lb$ = geometric facet area; $\ell_{\mathrm{cb}}$ = crack-band width defined by the selected fracture model. Do not use the same symbol for time and length or assume $\ell_{\mathrm{cb}}=\ell_e$ for every fracture formulation.

---

## 1. What Phase 0 must answer

- **Q1 — Mechanics.** Does RBSN on Voronoi cells with an energetic cohesive interface law, solved accurately, reproduce elasticity, strength and fracture energy with macroscopic response stable under mesh refinement and across random tessellations? If it fails with an accurate solver, nothing downstream (integers, GPU) can fix it.
- **Q2 — Force truth under a budget (the architectural fork).** How fast do internal forces converge in redundant intact structures with (a) compliant-ALM block descent, (b) ALM motion + static global force reconstruction with conditional factor reuse, (c) variants (per-bond ρ, penalty VBD)? Measured as error vs. work. Context §6.4 shows arbitrarily slow force-distribution convergence for one shared-ρ ALM iteration when $k\gg\rho$. It does not establish a universal limit for local solvers or require S2. Measure both force accuracy and the consistency of motion, work, energy and damage.
- **Q3 — Time discretization.** What implicit integration removes at real-time substeps (damping, dispersion, impact peak force), and whether another integrator is needed.
- **Q4 — 2D → 3D transfer.** Do conclusions hold for 6-DOF cells; which normalization makes per-cell 6×6 blocks well-conditioned.
- **Q5 — Numeric ranges.** Observed dynamic ranges of every quantity (positions, deltas, rotations, λ, $s$, $1-s$, $k_\text{eff}$, normalized Hessians, failure strains) — input for fixed-point / Flock IR design, not a proof of bounds outside the tested scenes.

The main mechanics track uses CPU f64 and excludes fixed point, WGSL/GPU, cross-machine determinism, networking, rendering and speed optimization. It establishes a reference and work/accuracy curves; it does not settle later numerical-format or hardware feasibility. Compare work counts (sweeps, cell/facet updates, factorization flops/nnz), memory and dependency depth; CPU wall time is not evidence of GPU speed. GPU device measurements are a separate Track B (§8), outside the first assignment.

---

## 2. Principles of the setup

These exist because each one prevents a specific way of getting a convincing but wrong answer:

- **Same equations, different solvers.** Compare iterative results to an accurate solution of the *same* discrete problem (S0). If S2 changes the equations or uses an auxiliary elastic field, report that as a model/coupling approximation, separately from under-convergence. Compare the discrete model to analytic solutions and literature independently. Keep model/coupling, algebraic, time/load-step, mesh-refinement and tessellation-randomness errors apart.
- **CPU update order mirrors the intended parallel algorithm.** In colored GS, cells of one color read a snapshot at the start of that color; the next color sees the preceding color's updates. A snapshot frozen across all colors is a different algorithm. Jacobi uses separate read/accumulate/apply buffers. Record the schedule explicitly (§4.2).
- **Trial damage may change inside a nonlinear solve; committed history may not.** Compute each trial from the last accepted history and current trial displacement. Commit only on acceptance of the physical load/time step or event substep; rejected trials leave history unchanged. Solver iterations are not elapsed physical time. A one-shot damage update after a frozen-damage solve is a separate lagged approximation.
- **Softening runs are displacement- or CMOD-controlled** (arc-length if needed); force control cannot follow descending branches.
- **Energy ledger** on every run with fracture or dynamics; independently identified numerical losses and the unexplained signed balance residual are separate lines.
- **Reproducibility:** seed, config, source revision (commit, or a source hash when unversioned), model-law version and solver schedule stored with every output.
- **Targets are written before runs** (§7); revising them later is fine with a recorded reason.

---

## 3. The model (baseline definitions)

### 3.1 Geometry (2D first, then a small 3D test)
- Plane stress with thickness $b$ (record the choice; plane strain as a variant). Geometric facet area is $A_e=lb$; fracture-energy normalization follows the selected law (§3.2), which may use a projected area.
- Seeds by random sequential addition with a minimum distance (controls size and regularity); Voronoi clipped to the domain polygon; notches by removing/marking facets.
- Each cell is a rigid body with DOFs $(x,y,\theta)$ at its centroid; mass, centroid and polar inertia from the polygon and density.
- Generators are kept: uniformity results are stated relative to generators. Facet kinematics use rigid arms; the transform between generator-based and centroid-based quantities is explicit, not a substitution of distances.
- Facet data: length $l$, generator distance $\ell_e$, facet centroid, normal $n$ (generator-to-generator direction), tangent $t$, rest displacement jump.

### 3.2 Interface law (per facet)
- Displacement jump at the facet centroid from both cells' rigid motions, expressed in the **corotated facet frame**, minus the rest jump computed the same way (objectivity: rigid motion of an unloaded block gives zero force).
- Elastic: $k_n=Ebl/\ell_e$, $k_t=\gamma k_n$ ($\gamma=1$ baseline → $\nu=0$), $k_\phi=k_nl^2/12$.
- **Pure-normal starting candidate:** linear tensile softening with $\sigma_t$, $G_c$ and $w_c=2G_c/\sigma_t$; scalar $d\in[0,1]$, irreversible accepted history, secant unloading/reloading and an undamaged compressive branch. Distinguish total jump $\delta$ from crack opening $w$; specify how the elastic displacement and $w$ combine. Full-separation dissipation is $G_cA_e$ only for the specified pure-normal law using geometric area.
- **Required closure before E5:** write the exact traction/opening and damage/history functions, onset and final openings, stored-energy function, dissipation increments and unilateral branch rules. Define the corotated frame and the trial/commit procedure. Show that the pure-normal cycle gives the intended energy as load steps refine; do not infer that from the value of $w_c$ alone.
- **Required closure before E6:** define equivalent opening under combined normal/shear/rotation loading, degradation of shear and moment, and their energy contributions. Macroscopic mode I on a random lattice is not pure-normal loading of each facet. Degrading every spring with normal-driven $d$ does not automatically dissipate $G_cA_e$ under combined loading. Either reproduce a named literature formulation with its geometric scaling, or label the chosen simpler law as a separate candidate.
- **Area convention:** record whether $G_c$ is normalized by actual facet area or projected crack area, and how $\ell_{\mathrm{cb}}$ is obtained. Bolander–Berton use resultant direction, projected area and a corresponding band width; do not transfer their objectivity claim to a different law without testing it.
- After full damage the candidate retains compressive response while the original facet pair remains a valid local interface; this is not a general nonpenetration/contact algorithm. Friction and general contact after large separation/sliding are later tasks. Keep early specimens within the stated local-interface regime.
- Crack band: report $\ell_{\mathrm{cb}}/l_{ch}$ for every fracture run. $\sigma_\text{eff}=\min(\sigma_t,\sqrt{2EG_c/\ell_{\mathrm{cb}}})$ is an explicit coarse-graining option, off in mesh-objectivity runs. It reaches the local nonnegative-slope bound, with equality degenerate; specify treatment before use. It changes strength and is not a general stability guarantee.

### 3.3 Stepping
- Quasi-static: incremental equilibrium without inertia, displacement control.
- Dynamic: implicit Euler on the incremental potential (baseline) plus one less dissipative alternative (BDF2, implicit midpoint or Newmark) for comparison.

### 3.4 Reference parameters and two scale regimes
Concrete-like baseline: $E=30$ GPa, $\sigma_t=3$ MPa, $G_c=100$ N/m, density 2400 kg/m³, $\nu=0$; $l_{ch}=EG_c/\sigma_t^2\approx0.33$ m.
- **Lab scale** (cells ≪ $l_{ch}$, e.g. 3–20 mm): validates mechanics against literature behavior.
- **Game scale** (cells ~0.3–1 m, band widths comparable to $l_{ch}$): the proposed engine regime; report actual $\ell_{\mathrm{cb}}$, since cell size alone does not define it. Crack-band effects and snap-back risk live here.

---

## 4. Solvers

### 4.1 S0 — Reference

Analytic solution for E1; assembled linear solve for linear elasticity; Newton with suitable sparse direct factorization for nonlinear problems. Report equilibrium and constitutive residuals, scaling and tolerances. For softening use displacement/CMOD control (arc-length if needed), trial history from the last accepted state, and load-step refinement.

Before softening runs, specify continuation, step acceptance, branch selection and event rules. Sub-stepping to the first threshold crossing does not resolve exact simultaneous events; deterministic tie-breaking is reproducibility, not proof of physical branch selection. A low Newton residual alone does not identify the physically admissible equilibrium. State whether S0 reproduces a sequential literature damage algorithm or solves a coupled damage law, and compare budgeted methods against that same definition.

### 4.2 S1 — Compliant-ALM block descent

For fixed trial spring coefficients and multipliers, define the reduced primal objective

$$\Phi(q;\lambda)=E_0(q)+\sum_e\left[s_e\lambda_e C_e(q)+\tfrac12 k_{\mathrm{eff},e}C_e(q)^2\right],\qquad s_e=\frac{k_e}{k_e+\rho_e},\quad k_{\mathrm{eff},e}=\rho_es_e.$$

$E_0$ contains the chosen dynamic incremental term and load potential, or only the load potential in a quasi-static problem. The index $e$ here ranges over scalar spring components. With $J_e=\partial C_e/\partial q$, the spring gradient is $J_e^\top(s_e\lambda_e+k_{\mathrm{eff},e}C_e)$; the positive-semidefinite Hessian approximation is $k_{\mathrm{eff},e}J_e^\top J_e$. Exact nonlinear Hessians also contain geometric terms. Declare the approximation, local-update rule and any damping/line search before comparison; the shortened source-code sketch is not the full mathematical specification.

- Per-cell 3×3 block (6×6 in 3D), deterministic greedy coloring and a recorded color order. A **primal sweep** visits every color once: snapshot at the start of a color, update its independent cells, then make those updates visible to the next color. Freezing across all colors is a separately labeled variant.
- Baseline **ALM sweep**: one primal sweep followed by one simultaneous dual update $\lambda_e^+=s_e(\lambda_e+\rho_eC_e(q^+))$, using the completed primal state. Freeze multipliers during the primal sweep. Other primal/dual frequencies are distinct variants with their work counted.
- Warm-start multipliers across accepted steps; declare initialization and handling of changed stiffness or inactive rows. They are not material-history increments. Damage trial/commit follows §2 and the completed law, not the sweep count.
- Dynamic penalty policies: translational $\rho$ scaled to $m_\text{eff}/\Delta t^2$, rotational $\rho$ to the corresponding $I_\text{eff}/\Delta t^2$; define each constraint's units and effective metric. Compare shared, per-spring and adaptive policies. In quasi-statics choose and report a stiffness-normalized penalty explicitly; do not introduce an unstated physical time step.
- E1 uses an **exact scalar primal minimization**, followed by the dual update. Its analytic contraction applies to that schedule and shared ρ, not to every block-descent variant.

### 4.3 S2 — Motion plus static force reconstruction [candidate]

Motion comes from S1. In the linear setting reconstruct $Ku=f_\text{ext}-Ma$ using its acceleration. A precise solve with inaccurate acceleration is not a precise solution of the coupled dynamics; define the rotational/large-motion extension separately.

Before using S2 in a comparison, specify:

- whether reconstructed $u$ corrects motion or is an auxiliary field, how forces feed back, and which displacement determines external work and measured response;
- when it solves the same equations as S0 and when it is a reduced/partitioned approximation;
- load compatibility and rigid-mode/null-space treatment for free clusters, including component changes and possible mechanisms after separation; no artificial supports to make factorization succeed;
- the fixed-reference/linearization assumptions under which numerical factors are reusable. Fixed topology alone preserves a sparsity pattern, not matrix values; relative rotations, continuous damage, opening/closure and changed constraints may invalidate factors;
- refactorization versus low-rank updates/downdates, including loss of definiteness or rank. Count symbolic analysis, numerical factorization and triangular solves separately.

Check displacement, acceleration, force/moment resultants, work, energy and damage history against S0. Force-only agreement cannot qualify S2 for the core. Fracture decisions from reconstructed forces are an experimental variant only after the constitutive and coupling definitions are complete.

### 4.4 S3 — Control

Penalty VBD without ALM, retaining the same physical stiffness and equations, to measure its work/accuracy tradeoff. A Soft-Step-style velocity-level bond solver is an optional later control with its equation/time-discretization differences stated explicitly.

---

## 5. Experiments

Suggested order: **E1 → E2 → E4 → E5 → E3 → E8 → E6 → E7 → E9 → E10.** E1–E3 answer the architectural fork early; E4–E5 validate the model before E6.

**E1 — Parallel springs (analytic).** One DOF, springs $k_1,k_2$ with ratios {1:2, 1:10}, $k_i/\rho\in\{1,10,10^3\}$, applied force $P$ (force applied, displacement unknown — prescribing displacement would make the test trivial). Regimes: cold start ($\lambda=0$); slow load ramp with warm start; step change in $P$; step change in $k_2$ (stands in for damage). Measure vs. iterations: $x$ error, each $F_i$, $\sum\lambda-P$, $\lambda_i-k_iC_i$, elastic energy. First reproduce the analytic contraction $r=2/(2+\rho/k_1+\rho/k_2)$ for shared ρ (validates the implementation), then evaluate per-spring $\rho_i$ and S2.

**E2 — Redundant RBSN block.** Small clamped Voronoi block under top load (statically indeterminate). Facet force fields (normal, shear, moment) from S1/S2/S3 vs. S0 as a function of work; same three regimes as E1 (cold, ramp, sudden change incl. removing a support facet).

**E3 — Scaling.** Two different experiments, kept separate:
(a) structure grows at fixed cell size (column/wall N cells high);
(b) fixed structure, refined tessellation (masses, stiffnesses and sizes change consistently).
Output: work required to reach a given force error (base reaction and facet forces) as a function of N. The full curve is the result; it decides whether S2 or a hierarchy is required and at what cost.

**E4 — Elastic patch tests.** Uniform strain on Voronoi blocks (several tessellations): stress uniformity and effective $E$, $\nu$. Rigid rotation 0–180° and large translation of unloaded and loaded blocks: spurious facet forces must be at round-off level. Long runs for drift.

**E5 — Single facet / small specimen.** Load–unload–reload, full separation, compression after opening. Dissipated energy vs. $G_c\,l\,b$ as load steps refine; no fracture dissipation during unloading without new damage; partial damage dissipates proportionally less.

**E6 — Notched beam, three-point bending (lab scale).** Typical geometry: depth ~100 mm, span/depth ≈ 4, notch ≈ half depth (prefer a configuration with published RBSN or experimental data). CMOD/displacement control. At least 3 resolutions × 3 tessellations, more if the spread is large; parameters fixed across resolutions; physical heterogeneity (if any) keeps its physical length scale and statistics instead of fresh per-facet randomness at each resolution. Outputs: P–δ, P–CMOD, peak load, dissipated energy vs. $G_c\times$ ligament area, damaged-zone width. S0 first (model truth), then S1/S2 under budgets. Bolander–Berton fitted their softening parameters by inverse analysis, so matching their curve is a sanity check, not independent validation. Fragment statistics are not judged here (beams barely fragment).

**E7 — Game-scale structure.** Wall/column/frame with cells ~0.3–1 m. Scenarios: support removal, lateral push, free fall of a detached cluster (must show zero self-weight stress — tests d'Alembert loads in S2). Compare S0 / S1 / S2; test the crack-band option; track timing of load redistribution after support loss.

**E8 — Time discretization.** SDOF oscillator: damping vs. $\omega h$ against the analytic $1/\sqrt{1+(\omega h)^2}$ per step for implicit Euler, and the alternative integrator. 1D chain/bar under impulse: arrival time, amplitude, peak stress vs. $h\in\{1/240,1/480,1/960\}$ s. Output: which structural frequency band survives at which substep, per integrator.

**E9 — Impact.** Rigid projectile on block/beam; vary $h$ and contact stiffness. Measure sensitivity of damage outcome to $h$. Prototype impulse-to-force via a Hertz-based contact duration (Hahn & Wojtan) feeding fracture loading; compare against a finely time-resolved S0 run.

**E10 — Small 3D.** Two cells, then a ~100-cell block: 6-DOF cells, 3+3 facet springs, full inertia tensors, torsion. Per-cell Hessian normalized separately for translation ($m/h^2$) and rotation ($I/h^2$): observed eigenvalue range vs. the bound $[1,\,1+\sum_e k_{\text{eff},e}\|v_e\|^2]$; Voronoi valence distribution (max, tail). Rerun E1/E2-style force convergence to check that 2D conclusions transfer.

---

## 6. Measurements and logging

- **Errors (relative to S0 or analytic):** positions; facet forces (L2 and max relative; normal, shear, moment separately); per-cell resultants; constitutive residual $\lambda-kC$; energy.
- **Work:** sweeps, cell-updates, facet-updates, number of passes (dispatch proxy), factorization nnz/flops and update cost for S2.
- **Energy ledger:** kinetic, elastic, gravitational, external work, fracture dissipation, friction, numerical dissipation (residual of the balance).
- **Range log (for Q5):** min/max |value| and required dynamic range for positions, deltas, rotations, λ, $s$ and $1-s$, $k_\text{eff}$, normalized Hessian entries and condition numbers, strains/openings at failure.
- **Outputs:** CSV/JSON per run with config; plots of error vs. work, P–δ, P–CMOD, energy vs. time.

---

## 7. Preliminary targets (written before runs, revisable with a recorded reason)

- **Force distribution (E1, E2):** ≤ 1% facet-force error within a realistic per-substep budget (order of 8–16 sweeps). Whichever method reaches it is the candidate; if none does, the size of the gap and its scaling (E3) are the result.
- **Mesh objectivity (E6):** peak load and dissipated energy within ~10–15% across resolutions with fixed parameters; S0 solver error well below that, so the source of differences is identifiable.
- **Patch tests (E4):** uniform stress up to tessellation noise; spurious forces under rigid motion at round-off level.
- **Single facet (E5):** dissipated energy → $G_c\,l\,b$ as load steps refine.
- **E8, E9:** quantify first; the decision (integrator, impact model) comes from the numbers.

---

## 8. Track B — device measurements (parallel, independent)

Small WebGPU microbenchmarks on target devices (iPhone, Adreno, Apple M, NVIDIA, AMD/Intel):
1. cost per dispatch and per frame for chains of 50–200 small dispatches;
2. integer `atomicAdd` throughput under contention (many writers per address), with and without workgroup pre-aggregation;
3. achieved bandwidth for gather patterns typical of cell–facet traversal;
4. sustained 10-minute runs (thermal steady state).

These numbers bound how many passes per substep the real-time solver can afford and feed back into the S1/S2 choice.

---

## 9. Decision map

- **S2 meets force targets, S1 does not** → static global force solve enters the core. Next research: fixed-point factorization with iterative refinement on an exact integer residual; GPU sparse triangular solves; low-rank updates under mass fracture; continuous $K$ changes under softening.
- **S1 meets targets with some ρ policy** → simpler architecture; confirm it under the sudden-change regime (damage, support loss) before accepting.
- **Neither meets** → crack-aware hierarchy/multigrid or per-cluster block solves; E3 curves size the problem.
- **E6 fails with S0** → mechanics problem; identify which part (interface law, geometric scaling, heterogeneity statistics, crack-band treatment) before any change of goal.
- **E8 shows unacceptable damping** → change integrator before further solver work.
- **E9 shows strong $h$-dependence** → impact duration model becomes part of the core model.
- **E10 contradicts 2D** → repeat the key tests in 3D before Phase 1.

Phase 1 (after Phase 0): integer CPU reference bit-exact to a chosen semantics → WGSL → scale and network.

---

## 10. Deliverables

- Rust workspace, e.g.: `geometry` (seeding, Voronoi, clipping), `model` (cells, facets, interface law, energy ledger), `solvers` (S0–S3), `experiments` (one runnable per E#), `analysis` (plots, tables).
- Per experiment: a short findings note — what was run, plots, interpretation, implications for the decision map.
- Range report for fixed-point / Flock IR design.
- A Phase 0 summary answering Q1–Q5.

Library suggestions (any equivalent is fine): `faer` (dense and sparse factorizations), `spade` or `voronoice` (2D Delaunay/Voronoi), `nalgebra` (small linear algebra), `plotters` or CSV + Python for plots.
