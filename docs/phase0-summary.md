# Phase 0 — Summary (Q1–Q5)

*CPU f64 mechanics and solver-truth program of `shatterbird-phase0-spec.md`. All ten experiments were run: E1 validated against closed forms, E2–E10 against S0 references. Per-experiment notes are in `docs/findings/`; theory, closures and pre-registered predictions (with revisions) in `docs/00-theory-and-design.md`.*

## Short answer

1. **The mechanics works at lab scale.** The RBSN with the two-point corotated facet and the energetic law SB-IL1 reproduces elasticity exactly (uniform stress to 2·10⁻¹¹, objective under 180° rotations), dissipates exactly $G_cA$ per facet on any path, and gives a notched-beam response that is **mesh-objective**: mean peak load and dissipation differ by ≤ 3 % across 3 resolutions, with ±4–7 % tessellation spread (target 10–15 %). Dissipation is 1.05–1.21 $G_c A_{\rm lig}$ (crack tortuosity).
2. **No local solver delivers force truth within a realistic budget.** Compliant-ALM block descent (any tested ρ policy), penalty block descent, and their S2 hybrids miss 1 % facet-force error in 8–16 sweeps by 1–2 orders of magnitude (E1, E2, E3, E10). The gap grows ≈ $N^4$ for slender structures (E3). In dynamics they are **unstable** under budget: they diverge, create energy and fracture spuriously (E7). What decides the cost is the stiffness number $\pi_h=(c\,h/\ell)^2$ (E3), not ρ.
3. **Time discretisation is part of the model, not a detail.** Implicit Euler at game substeps removes ~10 Hz structural modes and halves the peak force of sudden loads (E8 → BDF2 baseline). Impacts at ≥ 1 ms steps produce almost no fracture (E9). Breaking a facet within one step leaves an energy-ledger residual of up to 84 % of the dissipation (E7), and event sub-stepping cuts it to 8 %.

## Q1 — Mechanics

| check | result | experiment |
|---|---|---|
| uniform strain, $\gamma=1$ | interior tractions = $E\varepsilon n$ to 1.8·10⁻¹¹ for any tessellation, *if* the rigid motion follows the generators (centroid substitution: 6–8 % error) | E4 |
| uniaxial $E$, ν | $E_{\rm eff}=E$, ν = 0 exactly at $\gamma=1$; ν > 0 for γ < 1, at the cost of 2–15 % stress noise | E4 |
| objectivity | spurious force ≤ 10⁻¹⁰ of failure for rotations ≤ 180°; no drift over 3 600 incremental rotations; grows ∝ $T\varepsilon_{64}$ with absolute translation (10⁻⁴ at 10⁶ m) | E4 |
| interface-law energy | $D=G_cA$ exactly on monotonic, cyclic, compression-after-opening and mixed-mode (proportional and non-proportional) paths; zero dissipation on unloading; ledger residual → 0 with step refinement | E5 |
| structural snap-back | correctly detected for $L>2l_{ch}$; opening control traces it | E5 |
| notched beam (lab scale, $\ell_e/l_{ch}<0.06$) | mean peak 2.85 / 2.77 / 2.79 kN at 8 / 5.6 / 4 mm (≤ 3 % apart; ±4–7 % across tessellations); $D$ = 1.05–1.21 $G_cA_{\rm lig}$; localised process zone 2–8 mm; step-converged; ledger ≤ 6·10⁻⁵; target (10–15 %) met. S1 under displacement control does not follow (reaction 120–200× too large even at 1 024 sweeps/step) | E6 |
| game scale ($\ell_e/l_{ch}$ up to 1.8) | dissipation spread ±25 % across $d_{\min}$, $h$, integrator and solver branch; not an objectivity result | E7 |

**Model changes found necessary during Phase 0** (all recorded in the design note):
- Rigid supports must use $\ell_e=d$ (generator to support), not the mirror $2d$.
- Facets against a support or plate must be measured in the support's frame (otherwise a "roller" resists sliding once the cell rotates).
- The predictor must include external forces.

**Named limitations that showed up:**
- SB-IL1 ties shear strength to tensile strength and has no pressure dependence, which gives spurious bearing/splitting cracks at concentrated supports. In E6 these were excluded by a declared central damage zone; a real fix is a pressure-dependent shear branch.
- $\nu=0$ at uniformity.
- Geometric-area normalisation (tortuosity → 1.1–1.2 $G_c$).
- Under softening the S0 step problem is non-convex. Its solution depends on the initial guess (38 vs 61 J in E7), so **branch selection needs a physical rule**, not only determinism.

## Q2 — Force truth under a budget (the architectural fork)

| evidence | result |
|---|---|
| E1 (closed form) | shared-ρ ALM: force distribution contracts at $r\approx1-\tfrac12\rho/k$; fast only for $\rho\gtrsim k$, which in multi-DOF problems moves the stiffness into the primal. Force balance and displacement are not convergence proxies. |
| E2 (145 cells, quasi-static) | best local policy (per-spring $\rho=0.1k$) needs 75–100 sweeps; 30 % error at 16. Slow ramps are fine (0.16 %); cold start and support loss are not. |
| E3 (scaling) | local methods ≈ $N^4$ for slender walls (213 k sweeps at 8 m height); PCG ≈ $N$; direct factorization ≈ linear in $N$ for strips, one factorization ≈ 2 sweeps. Dynamics: cost governed by $\pi_h$; ≈1 000 sweeps per 60 Hz frame for 1 % at any substep. |
| E7 (dynamic, fracture) | budgeted S1 diverges or creates energy (elastic case: per-spring needs 256 sweeps/step to track S0 to 0.03 %); spurious fracture up to 25× S0's dissipation; S2(A/B) cannot repair a wrong acceleration and destabilises the dual. |
| E10 (3D) | same ordering and magnitudes; 6×6 blocks are well conditioned (≤ 7), so local conditioning is not the bottleneck. |

**Decision map:**
- *"S1 meets targets with some ρ policy"* → **rejected**.
- *"S2 meets force targets, S1 does not"* → **rejected in the specified form** (S2 driven by S1 motion). The *static global solve* itself is cheap and exact (quasi-static S2 = S0). What fails is coupling it to unconverged motion.
- *"Neither meets"* → **applies.**

The next architecture candidates, in order of evidence:
- (a) Solve the **whole implicit step globally** (S0-like: a sparse direct or factor-reusing solve of $M/h^2+K$) instead of using S1 for motion.
- (b) A crack-aware **hierarchy/multigrid** (untested here).
- (c) **Wave-resolved local stepping** ($\pi_h\lesssim1$, i.e. $h\lesssim\ell/c$: ~130 substeps per frame at 0.5 m cells, 1–2 sweeps each). E3 shows this is the only regime where local sweeps reach 1 % cheaply, and it also removes the integrator damping of E8.

Which of these fits a phone GPU is Track B's question: sparse triangular-solve depth vs. dispatch count vs. bandwidth.

## Q3 — Time discretisation

- Implicit Euler at 1/480 s keeps 67 % of a 10 Hz mode per period (1.7 % after 1 s) and gives 1.3 F₀ instead of 2.55 F₀ for a step load. That is **unacceptable** for fracture under sudden loads. BDF2 (variational, L-stable) keeps 10 Hz at 99.7 %/period and reaches 2 F₀ at 1/960 s. **Baseline changed to BDF2** (E8).
- Impact (E9): coarse steps miss fracture almost entirely; contact stiffness changes the outcome 4.6×. **A contact-duration / impact model becomes part of the core model** (decision map). Event-triggered substeps during contact recover 47–86 %; the remainder needs refinement of the structural response for a few transit times. The Hertz-duration prototype was not built.
- Fracture events (E7): one-step facet breakage is energetically inconsistent (ledger ε ≈ 84 % of $D$); event sub-stepping (Δd ≤ 0.2) reduces it to 8 % at +48 % steps.

## Q4 — 2D → 3D

Conclusions transfer (E10). Normalising translation by $m/h^2$ and rotation by the full $I/h^2$ gives 6×6 block eigenvalues in [1.9, 13.2] (bound $1+\sum k_{\rm eff}\lVert v\rVert^2$ holds, ≈3× loose). Valence: mean 15, max 19.

## Q5 — Numeric ranges

See `docs/range-report.md`. Headlines:
- Base+delta positions are required (E4).
- Per-step deltas are mm, not m.
- Facet stiffness spans 12 bits within one mesh (slivers).
- Lab-scale onset stretch is 0.5–1 µm (~135 quanta at Q4.28).
- Elastic jumps vs crack openings need ~32 bits or two representations.
- Slowly contracting solvers amplify quantisation by $1/(1-r)$.
- Range proofs presuppose a stable solver, and the budgeted local solvers are not stable (E7).

## Not done / open

- Track B (device measurements).
- Hertz contact-duration prototype (E9).
- S2 low-rank factor updates under fracture.
- Multigrid/aggregation.
- Pressure-dependent shear in the law; Bolander–Berton projected-area formulation.
- A branch-selection rule for non-convex S0 steps.
- Game-scale objectivity with more tessellations.
- 3D fracture.
- General contact after separation (only local-interface regime and per-step point contact were used).

## Recommendation

Proceed to Phase 1 only after a **Phase 0b on the solver fork**. Measure (a) a global implicit-step solve with factor reuse/low-rank updates under fracture and (b) wave-resolved local stepping, both against S0 on E7/E9-type scenes, together with Track B numbers for sparse solves and dispatch chains on target phones. The mechanics (E4–E6) is not the blocker; the force-truth budget is.
