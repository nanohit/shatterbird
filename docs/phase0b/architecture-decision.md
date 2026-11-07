# Phase 0b / 3 — Comparison for the computational architecture

> **Phase 0c note (2026-11-07).** The acceptance criteria of §4.4 here compared every candidate pointwise against the full undamped network and are not used for decisions any more. Decisions are re-derived from mechanical consequences in `docs/phase0c/reduced-fracture.md`. Measurements in this file stay valid as measurements.


*Assignment: `shatterbird-phase0b-step3.md`. This file is the final artefact of the phase.*
- *§0–§2 (status, manifest, pre-registration) are written before the measurements they govern.*
- *Results are appended per stage. Changes after a result are marked **[rev]** / post-hoc.*
- *No architecture is chosen until §8.*

## 0. Status of the stages

| stage | content | status |
|---|---|---|
| §1 | corrections to Phase 0b / 2: cut, metrics, counters, certified Chebyshev, closed loop with a true cut | **done** (two-level.md §8, v2 series) |
| §3 | stale preconditioner, cheap $\Delta A_c$, update policy | **done** (§3): RD chosen, post-hoc revision of the rule stated |
| §6.1 | early WebGPU probe (desktop) | **done** (§4): run 3 on an idle GPU is the reference; run 2 was contended |
| §4 | equal-error dynamics A1, A2, R1′, R2, R3, nonlinear bridge | **done** (§5), with revisions V1–V7 stated |
| §5 | scaling of I | **not run**: I did not survive §4 (§6) |
| §6.2 | finalist's chain on the GPU (E with history) | **done** (§7), including 10 minutes of load without degradation |
| §7 | numeric-contract prototype (E) | **done as a bounded prototype** (§8): bit-exact on one device; the format needs one more iteration |
| §8 | decision | **done** (§9): only the confirmed part is accepted (E kernel family); budget and mobile not met or not measured |

**Device profile available:** one desktop.
- Apple GPU (WebGPU adapter: vendor `apple`, architecture `metal-3`) in Chromium 152 inside the Claude desktop app.
- Features include `timestamp-query`, `subgroups`, `shader-f16`. maxStorageBufferBindingSize 4 GiB, 1 024 invocations per workgroup, 32 KiB workgroup storage.
- **No mobile device is available.** Every hardware conclusion is desktop-only; the mobile column stays "not measured".

## 1. Manifest (accuracy and cost)

**Accuracy.** The candidate is measured against a converged reference. Criteria are copied from the assignment §4.4 and not changed after results:

| quantity | tolerance |
|---|---|
| force / reaction in time, impact peak and impulse | ≤ 5 % (physical normalisation, not by near-zero instantaneous values) |
| damage onset time | ≤ max(0.1 ms, 2 % of the load time scale) |
| local χ before the first crack branching | max abs. error ≤ 0.01 on the union of points with χ_ref ≥ 0.8 or χ_cand ≥ 0.8 |
| dissipated energy, quasi-static peak load | ≤ 5 % |
| energy-balance error | ≤ 1 % of the declared available energy; physical and numerical dissipation separate |
| algebraic error vs a direct solve of the same discretisation | ≤ 1 % in forces and ≤ 0.01 in χ, with margin to the total error |

**Cost.**
- **Research bound:** 8 ms per physical frame of 1/60 s, p95, with a separate maximum during fracture. Also reported: N_max for 2 / 4 / 8 / 16.67 ms.
- **Active cells:** 1 k / 10 k / 30 k / 100 k.
- **Counted work:** full sequences (apply + update + retries). Work counters:
  - A-equivalents (flops, diagnostic only);
  - dependent phases;
  - every computed global reduction;
  - vector passes;
  - coarse solves.
- **Not covered:** rendering, broad-phase contact and networking. A full-game FPS is never claimed.

## 2. Pre-registration

### 2.1 §1 corrections (two-level v2)

Corrected series go to `results/phase0b/two-level-v2/`; v1 files stay untouched.

**Cut.**
- Each cell's layer = number of planes at or below its centroid height (a tie goes up); every facet between different layers is cut.
- Component audit before any solve: count, sizes, support attachment. Free rigid motions must be null vectors of $K$.

**Chebyshev and work counters.**
- Fine-level Chebyshev on the certified interval [0.25, 2] ($A\preceq2D$). The Lanczos estimate is used only for the SA prolongator weight.
- Counters: every norm and inner product is a reduction (and a phase). Vector passes are counted. Fixed-k PCG applies no preconditioner after its last iteration.
- The coarse solve counts as one phase only if it can be a single-workgroup dense solve ($n_c\le384$); otherwise $n_c$ is reported and its depth is open.

**Closed loop.**
- Every step counts. Reported separately: step 0 (push), steps 1–2, routine ≥ 3, the event window (event step + 2), all steps.
- Local χ on the union set. False and missed threshold crossings at 0.8 and 0.9.
- Includes the trajectory with a true cut (η = 0 at t = 0.05 s), with the coarse policies "P kept" ($A_c$ re-assembled and refactored) and "strength-aware rebuild".

### 2.2 §3 stale preconditioner

**Problems.**
- 2D wall 1 × 4 m, $d_{\min}$ 0.1, seed 1, and 3D cube 2 m, $d_{\min}$ 0.2, seed 1 (≈ 690 cells), both on contract meshes.
- BDF2, $h$ = 1/480 (the first step is IE). Push: $v_x$ = 1 m/s on cells with centroid within 0.25 m of the top. 60 steps.
- The replay uses the reference history (direct solves with the current operator). Initial guess = the previous reference solution.

**Scenarios** (stiffness multipliers $\eta_f(n)$):

| id | scenario |
|---|---|
| S0 | no change |
| S1 | the 1 % most stressed interior facets (max over the intact reference trajectory of normal force per area) soften: $\eta=\max(0.1,\,1-0.9n/40)$ |
| S2 | the same for 10 % |
| S3 | growing crack along the mid-height cut, facets ordered by centroid $x$ (then $y$, id): each facet goes to η = 0.5 and then 0 on two consecutive steps; the crack starts at step 10 and is complete by step 50 |
| S4 | true mid-height cut at step 20 (η = 0) |
| S5 | three cuts (¼, ½, ¾ height) at steps 15 / 25 / 35 |

**Preconditioner policies** (outer PCG always with the current $A_t$; B fixed within a solve):

| id | policy |
|---|---|
| U | full rebuild at every change: strength-aware aggregation, P, $A_c$, symbolic and numeric factorisation, current fine smoother |
| F | completely frozen B, built from the first BDF2 operator: old aggregates, P, $A_c$, factor and old fine blocks inside B |
| R8 | as F, but rebuilt every 8 steps |
| RD | as F, rebuilt when a solve fails the budget of 8 iterations; the solve is then repeated from the same guess and the first attempt's work is counted |
| PA | P fixed (initial), $A_c$ updated by $\Delta A_c=\sum_e\Delta k_e(P^Tj_e)(P^Tj_e)^T$ and refactored numerically, fine smoother current |

**Variants and stopping.**
- PTL-SA-s8-CH (chosen) and PTL-PR-s8-GS (control).
- Stopping on $\|r\|\le\tau\|r_0\|$ with τ = 1e-3, counted as reductions.
- Calibration rule fixed now: if U on S0 does not reach F_traj ≤ 1 % (and χ_union ≤ 0.01 in 2D) on every step, τ becomes 1e-4 for all runs, before the policies are compared.

**$\Delta A_c$ check.** It must equal the full triple product to ≤ 1e-12 relative (Frobenius). Its cost is reported against the fraction of changed components.

**Reported per scenario × policy × variant:**
- PCG iterations, refreshes, retries;
- A-eq, phases, reductions, coarse solves;
- update CPU time and coarse factor flops;
- quality (max F_traj, E_traj; χ_union in 2D).

**Policy choice for §4–6.** Among policies that meet the quality on all scenarios with SA-s8-CH, the smallest total CPU time (apply + update + retries) summed over S1–S5. Within 15 %, the simpler one wins, in the order F < R8 < RD < PA < U.

### 2.3 §6.1 early WebGPU probe (desktop only)

**Meshes.** rbsn3 contract meshes (constrained RSA, $d_{\min}$ 0.2, clamped bottom) in cubes of side 2.25 / 4.85 / 7.0 m, giving ≈ 1 k / 10 k / 30 k cells. Real facets and buffer bytes are reported.

**Kernels** (f32; linear network; no damage history — stated):

| id | kernel |
|---|---|
| E-A | facet pass (each facet's 6 component values C, generalized forces for both cells → facet buffer) + cell pass (gather through a fixed signed facet list, leapfrog / velocity-Verlet with full inverse inertia) |
| E-B | one dispatch per substep. Each cell evaluates its facets in canonical orientation (lower id = side a) and applies **its own** Jacobian row, with ping-pong state buffers. Checks: $F_a+F_b=0$ bitwise; moment balance $\tau_a+\tau_b+(x_a-o)\times F_a+(x_b-o)\times F_b$ to f32 rounding |
| C | empty dispatches of the same count (control, never subtracted) |
| I | matvec (cell-gather form), 6 × 6 block solve (precomputed inverse), global dot product (two-pass reduction), aggregate restriction / prolongation (PR, s = 8) |

**Chains and timing.**
- Chains of 250 / 400 / 700 / 800 substeps without readback.
- Timestamps at chain start and end. Wall time per chain via `onSubmittedWorkDone`.
- 3 warm-up chains, then 10 measured: median and p95.

**Verification.** After the chain, states against a CPU f64 run of the same leapfrog, with a relative tolerance of 1e-3 for f32 over 800 steps (declared now). Bitwise pair-force symmetry for E-B.

### 2.4 §4 criteria

The table of §1 applies unchanged. The design of R1–R3 and the nonlinear bridge are pre-registered in §2.5 before their runs.

### 2.5 §4 design (written after §3 and §6.1, before any §4 run)

**Model.** The 2D corotational RBSN with SB-IL1 (`model::structure`), because the 3D rbsn3 has no damage law. GPU cost is mapped from the 3D probe in units of facet passes (§4.4). This 2D → 3D mapping is a declared assumption, not a measurement.

**Schemes.** Every scheme evaluates forces with the trial history of the current configuration. History is committed only after an accepted (sub)step, with $\kappa\leftarrow\max(\kappa,\delta_{eq}(q_{n+1}))$.

| id | scheme | velocity position | step equation |
|---|---|---|---|
| E | leapfrog (kick–drift) | $v_{n+1/2}$; energy uses $v_n=\tfrac12(v_{n-1/2}+v_{n+1/2})$; start $v_{1/2}=v_0+\tfrac h2M^{-1}F(q_0)$ | explicit; history committed at every substep |
| I-BDF2 | BDF2, first step implicit Euler | whole steps | $\min\Phi$, $\alpha/h^2=9/(4h^2)$ |
| I-NM (control) | Newmark β = ¼, γ = ½ | whole steps; consistent $a_0=-M^{-1}\nabla\Pi(q_0)$ | $\min\Phi$, $4/h^2$ |
| I-GA (new candidate) | generalized-α, ρ∞ = 0.5, Chung–Hulbert convention: $\alpha_m=(2\rho_\infty-1)/(\rho_\infty+1)=0$, $\alpha_f=\rho_\infty/(\rho_\infty+1)=1/3$, $\gamma=\tfrac12-\alpha_m+\alpha_f=5/6$, $\beta=\tfrac14(1-\alpha_m+\alpha_f)^2=4/9$. In OpenSees' complementary convention, alphaM = 1, alphaF = 2/3. Equal to HHT with α = −1/3. | whole steps; consistent $a_0$ | force averaging: $Ma_{n+1}+(1-\alpha_f)r(q_{n+1})+\alpha_f r(q_n)=0$, $r=\nabla\Pi-f$; as $\min\Phi$ with $\alpha/h^2=1/((1-\alpha_f)\beta h^2)$ and $\hat q$ shifted by $-\tfrac{\alpha_f}{1-\alpha_f}M^{-1}r_n/(\alpha/h^2)$ |

**GA state and history.**
- Intermediate state: the force at $n+1-\alpha_f$ is the average of the end forces. Each end force uses the history valid at that end: committed $\kappa_n$ for $r_n$, trial $\kappa$ for $r(q_{n+1})$. No history is evaluated at an intermediate configuration.
- $r_{n+1}$ is recomputed after commit (one extra force pass, counted).
- **Admission gate for damage problems:** the one-element energy cycle below must show no energy creation.

**Step sizes.**
- E: the declared safe step $h_E\le0.38\,d_{\min}/c$, chosen as an integer fraction of the problem's sample grid; then $h_E/2$, $h_E/4$.
- Implicit: 1/240, 1/480, 1/960, 1/1920.
- Equal physical interval, loads and initial state for all.

**Tests.**

**A1 — one-element energy cycle** (admission for GA, check for all).
- One cell (0.1 × 0.1 × 0.3 m concrete) on a damageable SB-IL1 facet to the ground, pulled away by an initial velocity. Two cases: $KE_0$ = 0.5 × and 3 × the facet's full dissipation $G_cA$ (partial damage then oscillation; full separation).
- Exact reference from energy conservation: $KE_0=U(\kappa_{\max})+D(\kappa_{\max})$ for the partial case, $D=G_cA$ for full separation.
- Pass: numerical energy $E_{\rm num}=E_0-(K+U+D)$ never < $-10^{-6}E_0$ (no creation), and $|D-D_{\rm exact}|\to0$ under $h$-refinement.

**A2 — R1 elastic against the modal reference.** Exact solution of the linearised semi-discrete system: $u(t)=u_s+\sum_i\phi_i\,\phi_i^TM(u_0-u_s)\cos\omega_it$ (dense M-orthonormal eigenpairs, residuals reported). The nonlinear code runs are compared with it; the geometric nonlinearity is quantified by E at $h_E/4$ against the modal solution.

**R1 — prop removal.** Cantilever 8 × 1.2 × 0.3 m, $d_{\min}$ 0.3, contract seeding (exclusion 0.5 $d_{\min}$ from the clamped edge and from the prop segment [7.4, 8] extended by $d_{\min}$). Clamp and exact prop are elastic, non-damageable fixture facets; gravity. Static propped equilibrium (Newton), then the prop is removed at t = 0.
- Window 0.2 s, then 5 s of free motion for variants that pass.
- Elastic, then SB-IL1 on internal facets.
- Seeds 7, 8 (development), 9 (held out).

**R2 — notched beam.** E6 geometry, law, unilateral pads and central damage zone; contract rule 2 added to the seeding (generators ≥ 0.5 $d_{\min}$ from the support and pad segments ± $d_{\min}$).
- The plate displacement is prescribed: $d(t)=vt^2/(2t_a)$ for $t<t_a$, then $v(t-t_a/2)$, with $v$ = 1 mm/s, $t_a$ = 2 ms, to 0.25 mm (0.251 s). Then 0.25 s hold.
- No gauge: mass removes the horizontal null mode.
- Meshes: $d_{\min}$ 8 mm seed 1 (representative), 5.6 mm seed 1 (refinement), 8 mm seed 2 (held out).

**R3 — impact.** E9 block 1 × 0.5 × 0.2 m, $d_{\min}$ 0.05 contract seeding, clamped elastic base, SB-IL1 inside. Steel disk R = 5 cm, 12.3 kg, $v_0$ = 20 m/s, $k_c=10^9$ N/m ($t_c$ = 0.35 ms); unilateral frictionless contact facets at closest points; gravity.
- The disk's rotation is free: there is no torque through its centre. This is needed for uniform 3-DOF blocks in the hierarchy.
- E rebuilds contacts every substep; I at every (sub)step start.
- I's pre-declared event policy: substeps $h_s$ ≈ $t_c/20$ while contact is predicted (gap < $|v|h$ + 5 mm) or active, and for 10 wave transits ($10\,L/c$ = 2.8 ms) after the last contact. Their cost counts.
- Window 6 ms, then 5 s of free motion for variants that pass.
- Seeds 9, 10 (development), 11 (held out).

**Reference.**
- E at $h_E/4$, accepted when E at $h_E/2$ differs by < 1 % in every main metric. Otherwise refine to $h_E/8$; if that also fails, the reference is "not converged" and is reported as such.
- Cross-check: I-BDF2 with direct Newton at the finest implicit step. Disagreement is reported, not resolved by choice.

**Metrics** (sampled at the candidate's own step times, which lie on a common grid: 1/7680 s for R1 and R2, 1/192000 s for R3):
- **Force.** R1 clamp reaction, scale = prop reaction at 0⁻. R2 plate load, scale = reference peak. R3 contact force: peak and impulse relative to the reference.
- **Damage onset.** First material point with $\kappa>\delta_0$. Tolerance max(0.1 ms, 2 % of the load time scale): R1 0.1 ms; R2 2 % of the reference time to peak; R3 0.1 ms.
- **χ.** $\chi=\min(\delta_{eq}/\delta_0,1)$ on material points. Max abs error on the union {χ_ref ≥ 0.8 or χ_cand ≥ 0.8}, until the reference's first fully damaged facet (d ≥ 0.99), which stands in for "first branching". False/missed crossings at 0.8, 0.9 and 1.0.
- **D.** At the end of the window, plus the Jaccard overlap of facets with d > 0.5.
- **Energy.** $E_{\rm num}=W_{\rm ext}-(\Delta K+\Delta\Pi+D)$, physical $D$ separate. Scale: R1 = static energy released by the prop removal; R2 = plate work; R3 = $KE_0$.

**Pass** = every §1 tolerance for that problem. The minimal passing $h$ per family is reported, with all seeds; held-out runs only after the choice is fixed on development seeds.

**Nonlinear bridge (I-PCG)** for the passing I scheme:
- Outer iteration on $\Phi$ (trial history frozen within the step).
- Direction: $S\,p=-\nabla\Phi$, with $S$ = secant ($J^TD_{\rm sec}J$, PSD) + $(\alpha/h^2)M$.
- Linear solve: PCG with two-level SA-s8-CH. B under policy RD (§3.3): frozen between refreshes; refresh and repeat when a solve needs more than 8 iterations. Inner tolerance $\|r\|\le0.05\|\nabla\Phi\|$.
- Armijo backtracking on $\Phi$, with a residual-decrease fallback as in S0.
- Runtime stop: $\|\nabla\Phi\|_\infty\le\varepsilon_r F_{\min}$ (failure force of the smallest material facet) **and** max $|\Delta\chi|$ ≤ ε_χ and max relative change of the problem's reported force ≤ ε_F between the last two outer iterates. Starting values $\varepsilon_r=\varepsilon_\chi=\varepsilon_F=10^{-3}$; at most 30 outer iterations, otherwise the step is invalid.
- Calibration (development seeds only): the algebraic error against I-direct at the same $h$, along the whole closed trajectory, must be ≤ 1 % in force and ≤ 0.01 in χ. If not, the ε values are tightened by 10× once and fixed before the held-out run.
- Counted per step: outer iterations, PCG iterations, force passes (gradients and line-search energies), B refreshes and retries.
- If the bridge fails, the cause is localised. One alternative may be declared in advance: consistent tangent with secant fallback, as in S0, solved by the same PCG.

**What is not done.** No R2/R3 for GA if A1 fails. No mobile. 3D damage, contact between fragments and large rotations stay outside: the decision states them as conditions.

## 3. Results: stale preconditioner (§3)

*Data: `results/phase0b/architecture-selection/reuse/` (`summary.json`, `steps.csv`, `run_log.txt`). Command: `two_level reuse` (454 s).*

**Calibration.** U on S0 reached F_traj ≤ 2.3e-5 (2D, χ_union ≤ 1.1e-5) and 3.5e-5 (3D) on every step, so τ = 1e-3 stayed.

**Quality is never the issue.** Every policy, scenario and variant ends with max F_traj ≤ 5.4e-5 and χ_union ≤ 1.1e-5. This is what the construction promises: the outer PCG always uses the current $A_t$ and stops on its own residual, so a stale B can only cost iterations. The comparison is therefore a pure cost comparison.

**$\Delta A_c$ check.** Exact to rounding in all six checks (S0 step 1, S2 step 30 with 10 % of the facets changed, S4 at the cut): relative Frobenius error ≤ 2.3e-14, same sparsity pattern. Its cost depends on the width of $P^Tj$, not only on the changed fraction:

| update | PR-s8 ($P^Tj$: ≤ 2 aggregates) | SA-s8 ($P^Tj$: all aggregates touched by the smoothed rows) |
|---|---|---|
| 3D, 427 facets (10 %) | 1.7e5 madds per update | 4.2e6 madds per update (≈ 40 ms CPU, 75 % of a full SA rebuild) |
| 3D, 342 facets (cut) | 4.3e4 madds | 1.4e6 madds |

So with SA the "cheap" low-rank update is not cheap: the smoothed prolongator spreads one facet over a dozen aggregates.

### 3.1 Totals over the sequence (SA-s8-CH)

Entries: PCG iterations over 60 steps / total CPU time (apply + update + retries). Refreshes in brackets where they occur.

| scenario | U | F | R8 | RD | PA |
|---|---|---|---|---|---|
| **3D** S1 soften 1 % | 240 / 2816 ms (41) | 371 / 620 | 244 / 755 (7) | 371 / 629 (0) | 240 / 745 |
| S2 soften 10 % | 286 / 2661 (41) | 495 / 816 | 295 / 872 (7) | 354 / 699 (1) | 304 / 2130 |
| S3 growing crack | 258 / 2576 (42) | 1738 / 2765 | 516 / 1176 (7) | 434 / 2204 (15 retries) | 768 / 1498 |
| S4 true cut | 202 / 563 (3) | 2038 / 3339 | 382 / 1157 (7) | 210 / 544 (1) | 1040 / 1692 |
| S5 three cuts | 196 / 655 (5) | 2743 / 4243 | 235 / 767 (7) | 229 / 511 (1) | 1773 / 2868 |
| **2D** S1–S5 sum | 1430 / 326 ms | 6395 / 577 | 1702 / 216 | 1695 / 202 | 3144 / 315 |
| **sum S1–S5, 2D + 3D** | 9597 ms | 12360 | **4943** | **4789** | 9248 |

**Pre-registered rule.** RD is the cheapest; R8 is within 3.2 % of it and simpler, so the rule selects **R8**.

### 3.2 Per step: what the totals hide

The manifest's cost criterion is per frame (p95, plus a separate maximum during fracture), not a sum. Per step, CPU ms of SA-s8-CH (apply + update), 3D cube, as p95 / max over the 60 steps:

| scenario | U | F | R8 | RD | PA |
|---|---|---|---|---|---|
| S1 | 84 / 134 | 12 / 12 | 52 / 53 | 13 / 13 | 15 / 17 |
| S2 | 65 / 157 | 19 / 19 | 56 / 59 | 12 / 14 | 51 / 56 |
| S3 | 61 / 92 | 83 / 91 | 54 / 60 | 26 / 26 | 38 / 42 |
| S4 | 7 / 55 | 79 / 79 | 87 / 113 | 8 / 8 | 37 / 74 |
| S5 | 52 / 95 | 98 / 98 | 54 / 62 | 7 / 7 | 70 / 107 |

Iterations per step in S4 (3D): 4 before the cut. After it:
- F: **48–49** for every remaining step;
- PA: 24;
- R8: 48–49 until its next scheduled refresh (4 steps here, up to 7 in general), then 3;
- U and RD: 3–4 (RD after one failed attempt and a rebuild).

**What this shows.**
1. **A topology change invalidates the coarse space, not just $A_c$.** Aggregates that straddle the cut tie both sides to one rigid motion. Even with the exact $A_c$ (PA), the relative motion across the cut is missing from the coarse space, and iterations grow 6× (12× with F). Softening without separation costs F only 1.5–2×, and PA equals U there. This repeats the v2 finding "P kept 7–11 vs rebuild 2–3" (two-level.md §8) in a dynamic sequence.
2. **R8 is event-blind.** After every cut it runs as a frozen B for up to 7 steps, and in calm periods it pays a full rebuild every 8 steps. That is 12.5 % of all steps, inside the p95. RD rebuilds exactly when its canonical runtime metric (the iteration budget) fails: once per cut, 15 times during the growing 3D crack.
3. **CPU update time is not a method cost.** A full SA rebuild of the 688-cell cube takes ≈ 54 ms in this research implementation (aggregation, Lanczos weight, smoothed P, triple product, factorisation). It is reported, but it is not a forecast; §5 measures update cost against size.
4. **PR control with RD is degenerate.** PR needs ≈ 10 iterations per step even with a fresh B, so the 8-iteration budget fails every step (2D: 60 retries in S0). The budget was set for SA and is not meaningful for PR. Recorded as a design flaw of the pre-registration, not as a PR result.

### 3.3 Policy for §4–6 — **[rev] post-hoc, stated as such**

The pre-registered rule summed time over the sequence. That contradicts the manifest it was meant to serve: the budget is per frame, at p95 and as a maximum during fracture. Under that criterion RD dominates R8 in every 3D scenario: p95 7–26 vs 52–87 ms, max 7–26 vs 53–113 ms. R8 also has the structural defect of item 2.

**Decision.** §4–6 use **RD**: B frozen between refreshes; refresh and repeat the solve when the iteration budget fails; the first attempt's work is counted. R8 stays the formal outcome of the pre-registered rule and is replayed beside RD in §6.2, so the effect of this revision stays visible.

In the nonlinear bridge (§4), RD also defines B within a step: B is fixed across the outer iterations and refreshed only on budget failure. The budget remains 8 iterations per linear solve, for SA only.

## 4. Results: early WebGPU probe (§6.1, desktop only)

*Data: `results/phase0b/architecture-selection/gpu/` (`export.json`, `probe_run2_contended.json`). Commands: `gpu_export <dir>` for the meshes; `gpu-probe/probe.html` + `runProbe()` in the browser pane, served by `gpu-probe/serve.py`.*

**Device.** Apple M3 Pro (14-core GPU, 18 GiB unified memory, nominal ≈ 150 GB/s), macOS 14.6, Chromium 152 (Claude desktop app), WebGPU `metal-3` with `timestamp-query`.

**Linear probe only.** The probe runs the linear 6-DOF rbsn3 network: no damage history, no contact, f32. Velocity Verlet with velocities at whole steps (symplectic-Euler form: v ← v + h M⁻¹F(x), then x ← x + h v), identical to the CPU f64 reference.

**Meshes:**

| label | cells | facets | aggregates | bytes |
|---|---:|---:|---:|---:|
| 1k | 991 | 6 336 | 111 | 0.86 MB |
| 10k | 8 975 | 62 414 | 913 | 8.3 MB |
| 30k | 25 920 | 184 543 | 2 527 | 24.4 MB |

### 4.1 Two runs and a confounder

**Run 1** (session transcript only). The pane closed before its JSON was saved, so only medians quoted during the session survive, for 800-substep chains:

| | 1k | 10k | 30k |
|---|---:|---:|---:|
| E-A | 8.98 ms | 72.9 | 306.7 |
| E-B | 14.75 | 68.2 | 322.0 |
| matvec | 13.2 | 63.8 | 297 |

**Run 2** (saved) is 1.3–1.8× slower. Immediately after it, with the probe page idle, the GPU's own counter (IOAccelerator "Device Utilization %") read **86–92 %**. A foreign process was capturing the screen (`replayd`, 73 % CPU). Run 2 therefore measures a shared GPU.

**Consequence:** absolute milliseconds from this device carry at least a factor-1.5 uncertainty. Ratios within a run are more robust.

**Run 3** (`probe_run3_idle.json`): an idle GPU (counter 2–3 % before the run), same code. Per substep at 800-substep chains:

| | 1k | 10k | 30k |
|---|---:|---:|---:|
| E-A | 11.7 µs | 76.0 | 366 |
| E-B | 18.3 | 83.0 | 377 |
| matvec | 17.4 | 78.2 | 336 |
| block solve | 2.5 | 6.7 | 17.9 |
| dot | 5.1 | 9.8 | 20.4 |
| restrict + prolong | 8.4 | 11.1 | 18.2 |

- Within 5–10 % of run 1. p95/median ≤ 1.04.
- All checks are identical to run 2, bit for bit.
- **Run 3 is the reference measurement.** Run 2 stays reported as the contended one.
- On an idle GPU, E-A is faster than E-B at every size (1.6× at 1k): the facet-parallel pass fills a small GPU better than 1k cell threads each looping over ≈ 14 facets.

### 4.2 Per-substep cost (run 2, GPU timestamps)

Median of 10 chains after 3 warm-ups, divided by the chain length:

| kernel | 1k | 10k | 30k |
|---|---:|---:|---:|
| E-A (2 dispatches) | 20.3 µs | 124 µs | 599 µs |
| E-B (1 dispatch) | 28.8 | 123 | 530 |
| C empty dispatch | 0.7 | 1.0 | 2.9 |
| I matvec | 28.2 | 117 | 523 |
| I 6×6 block solve | 5.5 | 12.8 | 32 |
| I dot (2-pass reduction) | 8.8 | 18 | 38 |
| I restrict + prolong (PR) | 15 | 19 | 33 |

- **Chain length.** Cost per substep is flat in the chain length (250 / 400 / 700 / 800 agree within 5 % for E-A, 10 % for E-B). There is no hidden start-up term.
- **Dispatch overhead.** Below 1 µs at 1k and 10k. Removing dispatches cannot win much.
- **p95.** Within 10 % of the median at 10k and 30k. At 1k the p95 of short chains is up to 1.5× the median (scheduling noise).

### 4.3 Verification

| check | 1k | 10k | 30k |
|---|---|---|---|
| E-A vs CPU f64, 800 substeps (max position error / max displacement) | 4.4e-6 | 1.6e-6 | 2.4e-6 |
| E-B vs CPU f64 | identical to E-A in all printed digits | ← | ← |
| E-B pair forces $F_a=-F_b$ bitwise (after 100 substeps, non-zero forces up to 6e5 N) | 6 244 / 6 244 | 62 019 / 62 019 | 183 763 / 183 763 |
| moment balance $\lvert\tau_a+\tau_b+x_a\times F_a+x_b\times F_b\rvert$ / scale | 5.1e-7 | 1.6e-6 | 1.8e-6 |

- The declared f32 tolerance (1e-3) is met by three orders of magnitude.
- The bitwise antisymmetry holds on this compiler and device. That is an observation for one Metal compiler, not a guarantee (track-b.md §2, caveat i).
- Moment balance uses reference positions (linearised rbsn3). It holds to f32 rounding, not bitwise: the two moments are different expressions.
- An earlier zero count ("0/6244") came from a signed-integer bug in the JavaScript check (`x ^ y` is signed). It was fixed with `>>> 0` before run 2.

### 4.4 What the probe says

1. **One dispatch does not pay on this desktop.** E-B equals E-A at 10k, is 11 % faster at 30k (run 1: 5 % slower) and 40 % slower at 1k. Dispatch overhead is ≈ 1 µs against substeps of 20–600 µs. The decisive cost is elsewhere.
2. **The substep is memory-bound.** A byte count of E-A gives ≈ 48 MB per substep at 30k:
   - facet pass ≈ 184 B per facet (indices, 20 constants, two states, 12 results);
   - cell pass ≈ 550 B per cell (≈ 14 list entries with 6 values each, cell data, state).

   At run 1's 383 µs that is ≈ 125 GB/s, close to the nominal 150 GB/s. E-B reads the same facet constants twice but writes nothing per facet, so its bytes, and its time, are about the same.

   The arithmetic of the pair (≈ 150 flop) is not the limit. The lever is **bytes per facet**, not dispatch count or flops: compressed facet frames, narrower constants, shared reads. This matters for §7: a fixed-point format of the same width costs the same bandwidth.
3. **A matvec costs one explicit substep.** Both are one pass over all facets with the same traffic. The implicit path's arithmetic is counted in substep equivalents below. Block solve, dot and transfers are 5–15 % of a matvec at 30k.

### 4.5 First frame-cost bounds (formula only, before §4 fixes h)

$T_{\rm frame}=\lceil(1/60)/h\rceil\,T_{\rm substep}$, at the contract step $h=0.38\,d_{\min}/c=2.15\cdot10^{-5}$ s for $d_{\min}$ = 0.2 m: **776 substeps per frame.**

| path | 1k | 10k | 30k (25 920 cells) |
|---|---:|---:|---:|
| E (run 1 / run 2) | 8.7 / 15.8 ms | 66 / 95 ms | 297 / 411 ms |
| I, BDF2 h = 1/480, 3.5 PCG its, ≈ 6 matvec-eq per iteration, coarse solve excluded (run 1 / run 2 matvec) | 2.8 / 4.7 ms | 13 / 20 ms | 62 / 88 ms |

- **E.** Cost per cell-substep ≈ 15 ns (run 1, 30k). With 776 substeps, $N_{\max}(8\ {\rm ms})\approx 700$ cells at $d_{\min}$ = 0.2. With $d_{\min}$ = 0.5 (310 substeps), ≈ 1.7k. **E cannot carry 10k active cells within 8 ms on this desktop at any tested resolution.** It is excluded neither by accuracy nor by determinism, but by bandwidth × stability step.
- **I.** Its cost depends on what §4 must still decide: the admissible h at equal error, the nonlinear iteration multiplier, and the coarse solve (2 527 aggregates × 6 = 15k coarse DOF at 30k, not a single-workgroup solve: §5).
  - Per frame, I currently costs about **168 substep-equivalents** (8 × 3.5 × 6) against E's 776, before nonlinear iterations.
  - At h = 1/120 it would be about 42.
  - One I step at 30k is ≈ 7.8 ms (run 1), so 8 ms at 30k needs h = 1/60 with one outer iteration. At 10k one step is ≈ 1.7 ms, so h = 1/240 (four steps) fits.

These are bounds from kernels, not chain measurements (§6.2). The coarse level and the nonlinear multiplier can only add to I. Mobile is not measured.

## 5. Results: equal-error dynamics (§4)

*Code: `crates/experiments/src/bin/arch_dyn.rs`. Data: `results/phase0b/architecture-selection/dynamics/`. Commands are listed with each subsection.*

*Superseded development files, kept and not cited:*
- `r1_7.json`: after V3, before V4/V6;
- `r3_9.json`: identical to seed 9 in `r3_9_10.json`;
- `numeric/*_s800*` and `fixed_ref_1k_s50`: before the format fields were stored; same hashes as `A800`.

*`bridge_r2_480_1e-3_consistent.json` holds the final run with all three globalisation fixes.*

### 5.0 Revisions made while executing §4 (all after seeing data)

| # | revision | trigger | effect |
|---|---|---|---|
| V1 | The A1 gate "E_num never < −10⁻⁶E₀" is unsatisfiable for any conservative scheme. Leapfrog's energy error oscillates boundedly, O((ωh)²), with no net creation. Decomposed against an elastic control. | A1 | GA still fails the gate literally and stays off damage problems, as pre-registered |
| V2 | R1 with instantaneous prop removal is kept as a reported failure. **R1′** replaces the prop by its reaction, released over 10 ms by s(t) = ½(1 + cos πt/t_r); the external work is booked. | E itself fails against the exact modal solution (§5.2) | all R1 rows below are R1′ unless marked |
| V3 | The implicit predictor q̂ + (h²/α)M⁻¹f (inherited from E7) is replaced by the net-force predictor q̂ − M⁻¹r(q_n)/(α/h²) | spurious damage in the first implicit step | applies to direct Newton and to the bridge |
| V4 | Force criterion window: whole window if the reference pair converges there, otherwise up to the reference's first fully damaged facet (the χ window). The full-window value is always reported. | R1′ damage: post-failure hinge dynamics do not converge, even for E | consistent with the assignment: "после разветвления не требовать идентичности" |
| V5 | 1/3840 added for I as a labelled extra step; R3 event substeps of 10.4 µs and 5.2 µs as a labelled exploration | I failed every declared step | used only to locate where I would pass |
| V6 | R1′ damage reference: E at $h_E/8$ with the $h_E/4$ partner | D differed between $h_E/2$ and $h_E/4$ | — |
| V7 | Convergence of the reference pair in χ: difference ≤ 0.002 (one fifth of the tolerance) | "< 1 %" is undefined for χ | — |

### 5.1 A1 — one-element energy cycle (`arch_dyn a1`)

One 0.1 × 0.1 × 0.3 m cell on one SB-IL1 facet ($G_cA$ = 3 J, undamaged period T = 126 µs), pulled off with $KE_0$:
- 0.05 $G_cA$ (elastic control);
- 0.5 $G_cA$ (partial damage, exact $D$ = 0.7944 J from energy conservation);
- 3 $G_cA$ (full separation, $D$ = 3 J).

Values are relative to $KE_0$: "creation" = max(−E_num), "end" = E_num at 10 T.

| scheme | elastic control: creation / loss at T/20 | partial: D error at T/5 · T/10 · T/20 · T/80 | partial: creation at T/10 · T/80 |
|---|---|---|---|
| E | 2.5e-2 / 0 (bounded, net −1.7e-3) | — · — · +2.7 % · +0.1 % | — · 1.0e-3 |
| BDF2 | 0 / 0.60 | **−51 % · −30 %** · −11 % · −0.8 % | 0 · 0 |
| Newmark | 1e-10 / 1e-10 (exact) | +45 % · +17 % · +2.3 % · +0.05 % | **0.12** · 3.8e-4 |
| GA(0.5) | 0 / 0.074 | +41 % · +14 % · +2.2 % · +0.04 % | **0.095** · 3.2e-4 |

- **D converges to the exact value for every scheme.** The history definitions (end-point histories, commit after acceptance) are consistent.
- **Newmark and GA create energy in steps where damage grows:** up to 0.31 $KE_0$ at T/5, 0.12 at T/10. In elastic motion Newmark conserves to 1e-10 and GA only dissipates. The creation comes from averaging end forces across a softening step, and GA inherits it from Newmark. **The gate fails for GA, so GA is not run on damage problems** (pre-registered consequence).
- BDF2 never creates energy, but at coarse steps it removes the kinetic energy before it can reach the facet: D −51 % at T/5.
- Implicit steps in R1–R3 are 10–40× the local facet period. That is the left end of this table.

### 5.2 A2 — R1 elastic against the exact modal solution (`arch_dyn a2 7`, `SB_TR=0.01 arch_dyn a2 7|8|9`, `a2long`)

**Modal reference.** Dense M-orthonormal modes of the linearised prop-free system: 252 DOF, eigen-residual ≤ 3.5e-10, $u_0$ reconstructed to 6e-18 m. Fundamental 10.6 Hz, $\omega_{\max}$ = 3.7e4 rad/s. $h_E$ = 26 µs ≤ 0.38 $d_{\min}/c$.

**Instantaneous removal (pre-registered R1): nobody passes, the reference included.**
- Force error against the modal solution: E 64 / 47 / 23 % at $h_E$ / $h_E/2$ / $h_E/4$; implicit 74–162 %.
- The clamp reaction carries mesh-scale ringing (> 200 Hz) of up to 1.12 × the prop reaction. All ~250 undamped modes are excited and never decay.
- E's error is pure dephasing of these modes. It is second order (19 / 6 / 1.5 % at 20 ms) and grows linearly in time (0.3 % at 5 ms, 23 % at 200 ms for $h_E/4$).
- Pointwise agreement over 0.2 s would need $h\approx0.15$ µs.
- **The criterion asks for mesh-scale dynamics no real structure has.** This is a property of the undamped discrete model with an infinitely fast event, not of either method.

**R1′ (10 ms release), window 0.2 s, three seeds** (7, 8 development; 9 held out — identical verdicts):

| method | force error | χ error | energy | pass |
|---|---|---|---|---|
| E $h_E$ | 0.10–0.14 % | 0.003–0.004 (geometric-nonlinearity floor, same at $h_E/4$) | 1.8e-4 | **yes** |
| BDF2 1/240 · 1/480 · 1/960 · 1/1920 · 1/3840 | 42–46 · 27–30 · 30–34 · 17–19 · 5.1–5.9 % | 0.21 → 0.009 | 0.13 → 3.4e-4 | no (1/3840 fails only force) |
| Newmark same steps | 36–41 · 39–44 · 18–21 · 5.1–5.7 · **2.0–2.1 %** | 0.08 → 0.005 | ≤ 3e-6 | **only 1/3840** |
| GA(0.5) same steps | 31–35 · 34–39 · 24–27 · 7.4–8.5 · **2.3–2.6 %** | 0.12 → 0.006 | 0.019 → 4.8e-5 | **only 1/3840** |

- 97.5 % of the reaction's variance is the fundamental (10.6 Hz). The implicit error is phase and amplitude error of the global modes, not the linear solver.
- At h = 1/480 the first dynamic peak (≈ 47 ms) is already 21 % off for BDF2.
- **5 s free motion** (seed 7): E at $h_E$ 1.07 % force, 1.8e-4 energy, pass. Every implicit variant fails, 1/3840 included: 29–37 % force after 53 periods.

### 5.3 R1′ with SB-IL1 (`arch_dyn r1 7 8`, `arch_dyn r1 9`)

Reference E at $h_E/8$ (3.3 µs). Onset at 29.1–29.9 ms, first fully damaged facet 0.4–1.0 ms later. A root crack runs through the depth in ≈ 10 ms; D = 42–50 J; 12–18 facets broken.

**Reference convergence ($h_E/4$ vs $h_E/8$):**
- Force, χ and onset converge: force 1e-4–4e-4, χ ≤ 1e-4, onset ≤ 3 µs.
- **D converges for seed 7 (0 %) but not for 8 and 9 (3.9 %, 22.5 %).** D at $h_E$ / $h_E/2$ / $h_E/4$ / $h_E/8$:
  - seed 8: 46.6 / 53.7 / 48.4 / 50.4 J;
  - seed 9: 40.0 / 40.9 / 51.2 / 41.8 J.
- Once the crack runs, its path branches with the step size. The force history after the first broken facet (hinge rotation with contact-like compression, peaks up to 450 kN) does not converge either: E $h_E/2$ vs $h_E/4$ differ by 11 % at 35 ms and 100 % at 50 ms.
- D of dynamic fracture is a distribution over discretisations, not a converged number. The ±25 % spread of E7 is of the same nature.

**Candidates** (force on the pre-failure window, V4):

| method | seed 7 | seed 8 | seed 9 (held out) |
|---|---|---|---|
| E $h_E$ | pass (force 0.3 %, χ 2e-4, D +0.5 %) | D +7.4 % against an unconverged reference; all else pass | pass (D +4.3 %) |
| BDF2 1/240 … 1/1920 | fail: onset (0.3–26 ms early), χ 0.38–1.0, energy 0.3–0.7 | fail, Newton invalid at 1/240 | fail |
| BDF2 1/3840 | fail: force 55 %, χ 0.22, energy 0.80 | fail (also onset) | fail |
| Newmark 1/240 … 1/960 | **energy explosion**: D 374–1395 J, E_num up to 3.7e6 × scale; Newton invalid | same | same |
| Newmark 1/1920 · 1/3840 | fail: χ 0.33 / 0.16, energy 0.13 / 0.05 | 1/1920: force 0.9 %, χ 1e-3, but onset 0.48 ms, D −19 % | fail |

- **Before V3** the inherited predictor put the prop cells ≈ 9 mm off (a 500 m/s² free-fall prediction under a 32 kN released force). Every implicit run then damaged facets in its first step: onset 25.7 ms early at 1/240.
- With V3, BDF2's onset at 1/960–1/1920 is still 0.3 ms early (tolerance 0.2 ms).
- χ near crack initiation is the hardest criterion. The crack front moves from χ = 0.8 to 1 in tens of µs, so a 30 µs shift in onset is a 0.2 error in χ.

### 5.4 R3 — impact (`arch_dyn r3 9 10`, `arch_dyn r3 11`, `SB_SUB=2|1 arch_dyn r3 9`)

Block 1 × 0.5 m, $d_{\min}$ 0.05 contract mesh (≈ 180 cells), disk 12.3 kg at 20 m/s, $k_c=10^9$ N/m.
- The reference (E $h_E/4$, 1.3 µs) shatters the block: D = 250–254 J (10 % of $KE_0$), ≈ 370 facets broken, peak contact force 1.27–1.46 MN, impulse 355–373 N·s.
- Pair $h_E/2$ vs $h_E/4$: force, peak, impulse and D within 0.5 %; χ within 0.003.

| method | seed 9 | seed 10 | seed 11 (held out) |
|---|---|---|---|
| E $h_E$ (5.2 µs) | fail **χ 0.015** only (force 0.1 %, D +0.06 %) | **pass** | fail χ 0.013 only |
| E $h_E/2$ | **pass** | pass | **pass** |
| BDF2, declared event substeps 20.8 µs (≈ $t_c/17$) | fail: χ 0.40, D −10.5 %, energy 5.9 % | fail: Newton invalid, D −91 % | fail: χ 0.36, D −12 % |
| Newmark, same | fail: invalid, χ 0.021 | fail: invalid, D −95 % | fail: invalid, χ 0.24, D −13 % |
| BDF2 / Newmark with 10.4 µs substeps (labelled) | fail: χ 0.29 / 0.026, D −5.3 % / +0.8 % | — | — |
| BDF2 / Newmark with 5.2 µs = E's step (labelled) | fail: χ 0.027 / 0.018, energy 1.3 % / 0.06 % | — | — |

- During the impact the block is in contact for the whole 6 ms window. The declared event policy therefore runs I in substeps throughout, and its outer step does not matter.
- **Even at E's own step, implicit integration with per-step contact update does not reach the χ tolerance.** Newmark at 5.2 µs lands where E at $h_E$ lands (χ 0.018 vs 0.015).
- I would need at least E's step count during impacts, and each of its steps is a nonlinear solve.

### 5.5 R2 — notched beam, slow prescribed displacement (`arch_dyn r2`, `r2 refine`, `r2 heldout`)

C¹ plate ramp at 1 mm/s to 0.25 mm, then a 0.25 s hold.
- Reference (E $h_E/4$; 2.3 M substeps on 544 cells, 3.3 M on the refined 1 090): peak 2.41 kN (8 mm) / 2.84 kN (5.6 mm) at ≈ 0.1 s.
- Onset at 18.6 / 18.1 ms; D = 0.312 / 0.384 J.
- Pair $h_E/2$ vs $h_E/4$: force 7e-5, D 6e-8. Converged on the whole window.

| method | 8 mm, seed 1 | 5.6 mm, seed 1 |
|---|---|---|
| E $h_E$ (0.86 / 0.60 µs) | **pass** (force 1.4e-4) | **pass** |
| BDF2 1/240 | fail **onset only**: 2.27 ms late vs tolerance 1.89 ms (2 % of $t_{\rm peak}$); force 0.18 %, χ 0.006, D +0.02 % | same, onset 2.70 ms |
| **BDF2 1/480** | **pass**: force 0.19 %, χ 0.0042, D +0.015 %, onset 0.18 ms, energy 7e-5 | **pass** |
| BDF2 1/960 … 1/3840 | pass | pass |
| Newmark 1/240 … 1/1920 | fail: Newton invalid, D −66 … −99 % | same |
| Newmark 1/3840 | pass | pass |

**This is the regime where the implicit path works at game steps.** The load time scale (0.1 s to peak) is long against h. The criteria that bind are onset quantisation (1/240) and, for Newmark, Newton robustness.

### 5.6 The nonlinear bridge (`arch_dyn bridge <prob> <1/h> <eps> [consistent]`)

**Elastic (R1′, BDF2 1/480):**
- Algebraic error against direct Newton at the same h: force 2e-5, χ 4e-6.
- Per step: 2.3 outer iterations, 7.0 PCG iterations, 43 matvec-equivalents.

**R2 (BDF2 1/480), pre-registered secant direction: fails.**
- Before damage: 4–6 outer iterations.
- After onset: 15–27 outer iterations. The Picard (secant) iteration converges only linearly when facets soften.
- Invalid at 77 ms (30 outer iterations).

**Declared alternative: consistent tangent + the same PCG** (B from the SPD secant; secant fallback on a non-descent direction). Three globalisation defects surfaced and were localised. All three were fixed in the bridge only; S0 is unchanged.
1. **Cycling at a unilateral kink.** A crack-face point toggles between tension and compression (Δχ = 0.88 per iteration). S0's acceptance on residual decrease accepted a real increase of Φ. Fix: accept on residual decrease only within Φ's round-off.
2. **Round-off of Φ.** Φ is built from absolute coordinates, so its relative noise is ≈ ε·L/δ ≈ 1e-10, not S0's 1e3·ε. A predicted decrease of 2.8e-12·Φ could not be arbitrated. Fix: take the full step below 1e-10·|Φ|.
3. **Non-smooth point during the hold.** Armijo fails with g·p = −5.1, because the gradient is not the one-sided derivative there. Fix: the declared secant fallback also triggers on a failed line search.

**Result over the whole R2 trajectory:** valid. Algebraic error against direct Newton: force 4.8e-6, χ 2.7e-6, D 3e-10.

**Work per step:**

| quantity | mean | max |
|---|---:|---:|
| outer iterations | 4.9 | 12 |
| PCG iterations | 20.4 | 73 |
| fine-operator applications | 107 | |
| force passes (incl. line search) | 13.3 | |
| global reductions | 66 | |
| coarse solves | 20 | |

Also 13 B refreshes over 241 steps (12 after a failed budget), 3 secant fallbacks and 2 line-search fallbacks.

**Sensitivity (labelled, not used): ε = 1e-2 is more expensive.** It takes 9.3 outer and 62 PCG iterations per step, with 451 refreshes. The less converged states push later steps into more non-convex regions. The pre-registered ε = 1e-3 stands; its algebraic error is 2000× below the requirement.

### 5.7 Equal-error cost per 1/60 s frame

Mapped to the 3D GPU kernels of run 3 (idle GPU, §6.1) at game resolution $d_{\min}$ = 0.2 m.

| regime | E: minimal passing h | E: work per frame | I: minimal passing h | I: work per frame |
|---|---|---|---|---|
| dynamic, elastic (R1′) | $h_E$ | 776 force passes (κ step) | Newmark/GA 1/3840; BDF2 none | ≥ 64 steps × ≥ 43 ≈ **2 750+** matvec-eq |
| dynamic fracture (R1′) | $h_E$ | 776 | none tested passes | — |
| impact (R3) | $h_E/2$ (χ) | 1 552 during the event | none, even at E's step | > E per substep, × nonlinear iterations |
| slow loading (R2) | $h_E$ | 776 | BDF2 1/480 | 8 × 120 ≈ **960** matvec-eq + 530 reductions + 160 coarse solves |

- **At game resolution, E is the cheaper path in every regime, including the one where I passes.** E's cost per frame is fixed by the stability step ∝ $d_{\min}/c$. I's is fixed by physical time accuracy and nonlinear iterations, independent of $d_{\min}$.
- The two are equal near $d_{\min}$ ≈ 0.16 m for slow loading (776 · 0.2/$d_{\min}$ = 960). Below that, e.g. the 8 mm laboratory mesh with 19 400 E substeps per frame, I is up to 20× cheaper in quasi-static loading. In dynamic events I never is.
- **Caveats.**
  - I's counts are 2D. 3D two-level iterations were lower in v2 (2–3 vs 3–4), so I could gain up to ~30 %. That does not change the ordering.
  - The coarse solve at 30k (15k coarse DOF) is not a single-workgroup operation and is not costed here (§5 not run, see §6).

## 6. §5 (scaling I) — not run, with the reason

The assignment scales "only the surviving variant I".
- After §4, I is not the surviving variant:
  - it fails every dynamic regime (R1′, R3) at game steps;
  - at equal error it needs ≥ E's step during impacts;
  - in its one passing regime (slow loading, R2 at 1/480) it costs ≈ 960 matvec-equivalents per frame at game resolution, against E's 776 force passes (§5.7).
- A recursive hierarchy would lower the coarse-solve cost, which §5.7 did not even count. It cannot remove the 120 fine-operator applications per step that already exceed E.
- **Per the assignment ("не расходовать время на порт уже отвергнутого варианта") §5 is skipped.**
- What I would need to come back: a regime with $d_{\min}$ well below 0.16 m and slow loading only, where I is up to 20× cheaper (R2's laboratory mesh). That is a different product profile from the one fixed here.

## 7. §6.2 — the finalist's chain on the GPU (E)

*Data: `results/phase0b/architecture-selection/gpu/probe_h_e_chain.json`, `sustain_30k.json`. Page: `gpu-probe/probe.html`, `runH()`, `runSustain()`.*

**The chain.** E-B-H is one dispatch per substep:
- every cell evaluates its facets in canonical orientation with a synthetic SB-IL1 history on the 3D facets: $\delta_{eq}=\sqrt{\langle\delta_n\rangle_+^2+\gamma|\delta_t|^2}$, $\delta_0=10^{-4}\ell$, $w_c=2G_c/\sigma_t$, linear softening on all six components, undamaged compression;
- κ is ping-ponged: read from the old buffer, the new value written only by the canonical owner (side a);
- velocity Verlet as in §6.1.

The chain contains the complete E step, with no reductions, coarse solves, stopping tests or updates. That is the point of E.

| | 1k | 10k | 30k |
|---|---:|---:|---:|
| E-B linear (same session) | 18.0 µs | 82.7 | 370 |
| **E-B-H with history** | **21.9 µs** | **93.3** | **396** |
| history overhead | +22 % | +13 % | +7 % |
| damaged facets GPU / CPU f64 (800 substeps) | 981 / 981 | 4 680 / 4 680 | 9 734 / 9 734 |
| broken facets GPU / CPU f64 | 119 / 119 | 440 / 440 | 600 / 600 |
| onset disagreements; max abs damage difference | 0; 4.6e-5 | 0; 1.4e-5 | 0; 9.9e-6 |
| position error vs f64 | 4.8e-7 | 3.6e-6 | 8.5e-7 |
| pair forces bitwise opposite; κ of both sides bitwise equal | 6 244 / 6 244 | 62 019 / 62 019 | 183 763 / 183 763 |

Measured with the GPU counter at 33–40 % from the compositor before the run; the linear E-B column of the same session matches run 3 within 2 %.

**Frame cost of the finalist** (776 κ-certified substeps per frame at $d_{\min}$ 0.2 m):
- E-B-H: 1k 17.0 ms, 10k 72 ms, 30k 307 ms. Event frames cost the same (no event spikes, no update).
- **N_max:**
  - 2 / 4 / 8 ms: below 1k cells. The one-dispatch form has a ≈ 13 µs floor at 1k (each thread loops serially over its facets at low occupancy). E-A, at 11.7 µs for 1k (linear), would give ≈ 9 ms.
  - 16.67 ms: ≈ 1k cells.
- With the 0.8·Sylvester step instead of the κ certificate (625 substeps; needs a per-asset eigenvalue bound), every figure drops by 19 %.

**10 minutes of sustained load** (E-B-H, 30k cells, chains of 800 substeps back to back; `sustain_30k.json`): 1 843 chains, i.e. 1.47 M substeps.

| window | median per chain | p95 | max |
|---|---|---|---|
| all | 320 ms | 336 ms | 438 ms (one outlier) |
| first minute | 328 ms | 339 ms | — |
| last minute | 311 ms | 317 ms | — |

No thermal degradation on this device (macOS reported no thermal or performance warning).

## 8. §7 — numeric-contract prototype for E

*Code: `gpu-probe/fixed.js` (constants + WGSL), `crates/experiments/src/bin/fixed_ref.rs` (independent CPU reference). Data: `results/phase0b/architecture-selection/numeric/`. Command: `runFixed(labels, 800, 10, 3, tag[, formats])` in `fixed.html`, then `fixed_ref <dir> <label> 800 <tag>`.*

**Contract (format A).** All state and arithmetic are integers.

| quantity | format (LSB) | range |
|---|---|---|
| u | i32, $2^{-28}$ m | ±8 m |
| θ | $2^{-30}$ rad | |
| v | $2^{-20}$ m/s | ±2 048 m/s |
| w | $2^{-22}$ rad/s | |
| facet force and moment | $2^{-6}$ N, N·m | ±3.4·10⁷ N |
| unit vectors | $2^{-30}$ | |
| arms, κ, δ₀, $w_c$ | $2^{-28}$ m | |
| $k_n$, $k_t$ | $2^{5}$ N/m | |
| rotational springs | $2^{0}$ N·m/rad | |
| γ, ω = 1 − d | $2^{-16}$ | |
| integration constants $h/m$, $hI^{-1}$, $h$ | per-mesh shifts chosen so that every constant < $2^{30}$ (SM 34, SI 23, SH 37) | |

**Operations.**
- Every product is exact 64-bit. In WGSL it is emulated with 16-bit limbs (WGSL has no 64-bit integers).
- Every narrowing is round-half-up, $\lfloor x2^{-s}+\tfrac12\rfloor$. A value that does not fit i32 saturates and raises a global flag.
- $\delta_{eq}$ uses the bit-by-bit integer square root of a u32 (components clamped at $2^{15}$ LSB > $w_c$). ω uses 16-step long division.
- Facet forces are accumulated per cell by integer addition, so the result is independent of order.
- Constants are computed once (f64 from the mesh, `Math.round`) and shipped to both GPU and CPU.
- The CPU reference is native i64, wrapping exactly where the emulation wraps.
- There is no physical-parameter scaling: masses, stiffnesses and $G_c$ enter as given.

**Results after 800 substeps with real damage growth:**

| | 1k | 10k | 30k |
|---|---|---|---|
| **GPU vs CPU integer reference** | **0 of 18 228 words differ** (FNV-1a `fa66298e` both) | **0 of 170 114** (`bc4685d7`) | **0 of 495 583** (`1ce4a957`) |
| repeated GPU run, same hash | yes | yes | yes |
| pair forces exactly opposite; κ equal on both sides | 6 244 / 6 244 | 62 019 / 62 019 | 183 763 / 183 763 |
| overflow flag | **set**: first at the normal-force narrowing (shift 17), one stiff support facet in compression needing 3.36·10⁷ N | clear | clear |
| displacement error vs f64 (max / max) | 4.4e-4 | 6.0e-3 | 3.7e-4 |
| velocity error vs f64 | 8.2e-3 | 1.8e-2 | 7.1e-3 |
| damage: max abs difference; onset disagreements | 0.0077; 1 of 981 | 0.0090; 6 of 4 680 | 0.0069; 6 of 9 734 |
| broken facets fixed / f64 | 119 / 119 | 440 / 440 | 600 / 600 |
| GPU per substep (vs f32 E-B-H) | 181 µs (8.3×) | 288 µs (3.1×) | 735 µs (1.9×) |

For comparison, f32 has position error 5e-7–4e-6, damage difference ≤ 5e-5 and no onset disagreements.

**Localising the difficult operations:**
1. **Wide multiplication.** All cost comes from emulating 32×32→64 products: ≈ 60 per facet evaluation. The kernel becomes ALU-bound.
   - At 30k the memory traffic still hides part of it (1.9×). At 1k it does not (8.3×).
   - On this device the integer E costs 1.9–8.3× the f32 E. That is a named overhead, not a rounding error.
2. **Range of the force format.** ±3.4·10⁷ N was exceeded once at 1k by a support facet ($k_n$ = 3.3·10¹⁰ N/m). Saturation is deterministic and identical on CPU and GPU, but the result is wrong there.
   - A wider force LSB ($2^{-4}$, ±1.3·10⁸ N) or per-class scales are needed. The second iteration of the contract must derive the format from the certified stiffness and step bounds, not by hand.
3. **Precision of the displacement vs its range.** One i32 for absolute displacement (±8 m) quantises the jump at 3.7 nm, i.e. 2·10⁻⁴ δ₀.
   - Within the χ tolerance on its own (0.01), but it flips damage onset at facets sitting within that margin of χ = 1: 0.06–0.13 % of damaged facets. Trajectories then diverge locally (velocity errors ~1 %).
   - **Hypothesis tested and rejected (format B):** a 64× finer velocity quantum (v at $2^{-26}$ m/s, w at $2^{-28}$ rad/s, range ±32 m/s) gives no improvement (v error 0.7–2.3 %) and removes the w headroom (0.2 bit). The limit is the displacement/jump quantum, not the velocity increment.
   - The named alternative is positions relative to a per-fragment or per-tile origin (jumps between neighbours stay small while fragments travel), or i64 positions at a further cost.

**Status of the numeric path.**
- **Bit-exact on one device:** demonstrated, including a saturation event.
- **Cross-vendor:** not verified, since only one GPU is available.
- Accuracy against f64 is acceptable for χ (resolution 2·10⁻⁴) but 3–4 orders worse than f32 in this impulsive test, with one range failure. The format needs one more iteration (items 2–3) before it can be called the contract.
- The cost overhead is measured: 1.9–8.3×.

## 9. Decision (§8 of the assignment)

### 9.1 What the measurements say

| question | answer | where |
|---|---|---|
| Which family reaches the §1 accuracy in dynamic events (prop removal, dynamic fracture, impact)? | **E only**, at its stability step ($h_E/2$ for χ in impacts in 2 of 3 seeds). I fails at every game step; it needs 1/3840 elastic, and with damage passes nowhere tested. | §5.2–5.4 |
| …in slow loading? | both: E at $h_E$, I (BDF2) at 1/480 | §5.5 |
| Is the new candidate GA admissible for damage? | no: it creates energy in softening steps (inherited from Newmark), so the gate fails | §5.1 |
| Can I's step be solved with the hierarchy? | yes, only with the declared consistent-tangent alternative (secant fails). It costs 4.9 outer / 20 PCG iterations, 120 matvec-eq and 66 reductions per step. | §5.6 |
| Cost at equal error, game resolution ($d_{\min}$ 0.2 m)? | **E is cheaper in every regime**, including I's own. Its frame cost is flat, with no updates, retries or reductions. | §5.7, §7 |
| Budget 8 ms p95 (M3 Pro, WebGPU)? | **missed by both at ≥ 1k active cells.** E: 17 ms at 1k, 72 at 10k, 307 at 30k. I is more expensive. | §4, §7 |
| Determinism? | E: bit-exact integer prototype on one device; force range and displacement precision need one more format iteration; ×1.9–8.3 cost. I: not attempted (rejected earlier). | §8 |
| Mobile? | not measured | — |

### 9.2 Applying the pre-registered rule

- "E passes physics, the hardware profile and the numeric prototype; I does not pass or is more expensive":
  - physics: holds;
  - I being more expensive: holds;
  - **the hardware profile (8 ms at the target sizes) does not hold, and the numeric prototype is not finished.**
- "Neither passes": holds for the budget. **No combination of the measured paths reaches 8 ms at ≥ 1k active cells, $d_{\min}$ 0.2 m, on an Apple M3 Pro desktop under WebGPU.** For 16.67 ms, E reaches ≈ 1k cells. 10k / 30k / 100k fail by 4× / 18× / > 60×.
- "Mobile, arithmetic or nonlinear measurements missing": mobile missing, arithmetic partial.

**Therefore only the confirmed part is accepted, with conditions. It is not a full architecture for the original goal.**

### 9.3 Accepted (the computational kernel family)

| element | choice |
|---|---|
| integrator | explicit leapfrog / velocity Verlet on physical masses; v at half steps; step = the contract's κ-certified 0.38 $d_{\min}/c$ ($h_E/2$ while a contact is active, if χ ≤ 0.01 is required during impacts) |
| nonlinear closure | none needed: forces from the SB-IL1 law with the history of the current configuration, committed every substep; no Newton, no line search, no branch selection |
| operator / hierarchy / updates | matrix-free facet pass; no hierarchy, no coarse solve, no reductions, no update schedule |
| GPU form | canonical pairs (lower id = side a). History is read from the old buffer; the new κ is written only by the owner (both sides compute it bit-identically, verified). E-A (facet pass + cell pass) was faster than one-dispatch E-B at every size on an idle GPU; the history variant of E-A is not yet measured. |
| numeric contract | integer format A of §8 as the starting point, with two named corrections: the force range derived from certified stiffness, and positions relative to a fragment or tile origin. Then one more bit-exact check and a second vendor. |
| scene limit (measured) | ≈ 1k active cells at $d_{\min}$ 0.2 m per 16.67 ms frame on an M3 Pro desktop (WebGPU, f32); 8 ms not reached |

**Rejected, with reasons:**
- I (BDF2 + hierarchical PCG): accuracy in dynamic regimes; cost at game resolution.
- GA(0.5): energy gate.
- Newmark: energy creation and Newton failures with damage.
- The secant bridge: linear convergence under softening.
- A hybrid E/I: I wins only fine-mesh quasi-static loading, outside the game profile. A hybrid would need its own transfer check, and it does not pay here.

### 9.4 Not decided — conditions for any further step

1. **The budget gap is the main open problem.**
   - E's cost is (stability substeps ∝ c/$d_{\min}$) × (cells) × (≈ 1.9 KB per cell-substep at ≈ 125 GB/s).
   - Measures that change the product, none of them tested here:
     - coarser active resolution: $d_{\min}$ 0.5 m gives 311 substeps and 15× fewer cells per volume;
     - an active set (sleeping islands, which is a different load profile);
     - fewer bytes per facet (compressed frames, narrower constants);
     - local time stepping (a new behaviour contract with its own accuracy check).
   - Each needs a pre-registered test. No FPS for a full game is claimed.
2. **3D cohesive law.** The probe's 3D history is synthetic, and the 3D model has no large rotations. The rbsn3 nonlinear law and corotation remain conditions.
3. **General contact.** Only 2D penalty contact with $k_c$ under the stability certificate has been checked. A per-cell certificate including contact springs is required.
4. **Mobile and cross-vendor bit-exactness** are unmeasured.
5. **Two properties of dynamic fracture are part of the contract, not bugs to tune away:**
   - D after crack branching does not converge under step refinement (R1′ seeds 8–9: ±22 %);
   - an instantaneous event in the undamped lattice is unresolvable pointwise (R1).

   Any future accuracy claim must use event windows and distributions, as here.

### 9.5 Revisions log (all post-hoc changes of this phase)

- §3.3: RD instead of R8 (rule metric vs manifest metric).
- §5.0 V1–V7 (A1 gate, R1′, predictor, force window, extra steps, reference refinement, χ pair threshold).
- §5.6: bridge globalisation fixes (kink cycling, Φ round-off 1e-10, secant fallback on a failed line search).
- §8: format B tested and rejected.

Every pre-registered verdict is reported next to the revised one.
