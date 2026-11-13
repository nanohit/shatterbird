# Phase 2a — state, memory, locality and certificates

Written 2026-11-13. CPU float64 only: nothing in the integer specification or the GPU pipeline changed.
Code: `lab/` (Python), `gpu-probe/lab4.cjs`. Data: `results/phase2/`.

**How this phase was run.** There was no pre-registration file. The phase was exploratory by
instruction, and the hypotheses were formed while working. Section 2 lists what was predicted before
each measurement and marks which predictions failed. Section 9 pre-registers the next phase.

## 0. Why

Three criticisms of Phases 1g and 1h, all accepted:

* The reference of 1g and 1h was the same rigid algorithm with more iterations. It has no limit: the
  displacement of a kicked sphere at 256 iterations is 0.51–0.76 of the value at 128. The verdicts of
  1h measured the distance to a moving target.
* 1g section 12 said the law needs a physical stiffness before any further optimisation. 1h optimised
  convergence anyway and arrived at the same sentence.
* "The impulse network is history that has to be stored" (G3) did not follow from the data. Two
  equilibria that differ by 54 % show static indeterminacy, which is a known property of frictional
  packings. Nobody had measured whether the difference changes anything.

Two more: the rigid law had accumulated threshold rules (handover, designated support, hash key, gate,
normal-only acceleration, `reff`, μk < μs, narrowed cone), and friction without rotation is a different
mechanical system from the one the tests asked about.

## 1. Theory

### 1.1 What information the future needs

World state `x`, deterministic step `F`, admissible future inputs `u` (player actions, bodies arriving),
observables `O` with a tolerance ε. Two states are **future-equivalent** if no admissible input sequence
separates their observables by more than ε at any later time. The information the world needs is the
set of equivalence classes (the Nerode quotient), nothing more. For a subsystem that talks to the rest
of the world through ports this is a statement about one operator: the map from past port inputs to
future port outputs (the Hankel operator). Its singular values σ₁ ≥ σ₂ ≥ … are the answer to "how many
variables":

* any linear model with r internal variables has a worst-case error of at least σ_{r+1};
* balanced truncation reaches 2·(σ_{r+1} + σ_{r+2} + …).

This is the linear, quantitative form of Mori–Zwanzig. Eliminating interior variables exactly turns
them into a memory kernel on the boundary; the number of exponentials needed to represent that kernel
to ε is the number of Hankel singular values above ε.

Excluded information falls into four classes. Each has its own criterion, and the experiments below
measure one each.

| class | the excluded variable … | criterion | measured in |
|---|---|---|---|
| slaved | relaxes to a function of the kept ones faster than any input changes | spectral gap between the interior and the input band; count of σ_i above ε | section 5 |
| screened | acts on a decision only through an operator whose inverse decays with distance | condition number κ of the step operator: influence ≤ C·q^d, q = (√κ−1)/(√κ+1) | section 4 |
| within a margin | changes a quantity that feeds a threshold by less than the distance to the threshold | computable error bound against the margin | section 6 |
| undefined | is not determined by the model at all | uniqueness of the step problem | section 3 |

The fourth class is the one the rigid law lived in. The self-stress component of a frictional contact
network is not unnecessary information, it is information the rigid model does not define. The solver
history filled it in, and the response depended on what it filled in (section 3).

Information that cannot be dropped: lightly damped modes inside the input band (they are the memory),
states close to a threshold, and anything near a loss of stiffness (a mechanism, a floating cluster),
where κ → ∞ and the screening length diverges.

### 1.2 The axis ℓ

ℓ = ω_c·h is the number of layers a contact signal crosses in one substep (ω_c² = k_n/m).

* ℓ < 1: explicit elastic network. 776 substeps per frame for concrete.
* ℓ = ∞: rigid contact. Every substep is a global problem. Jacobi needs of the order of L² iterations for
  a depth L, and an under-iterated rigid solver is in fact compliant with a stiffness set by the
  iteration count (1g: the kick sinks 3–6 times too far at 4 iterations).
* ℓ of a few units: implicit step of a compliant contact. Influence inside a substep decays over about ℓ
  layers; beyond that the load travels as a physical wave.

With a finite ℓ the step is the minimisation of a strictly convex function,

    Φ(u) = ½ (u − u*)ᵀ M (u − u*) + h Σ_c ψ_c(J_c u),

`u` the velocities and angular velocities, `u*` the free velocities, ψ_c the contact potentials. Three
consequences follow from M alone, with no spectral estimate:

1. **Uniqueness.** Φ is strongly convex with respect to M, so the step has one solution. Contact forces
   are functions of the state. Nothing is left for the solver history to choose.
2. **Certificate.** For any iterate with gradient g, ½‖u − u_opt‖²_M ≤ ½ gᵀM⁻¹g. The right side is a sum
   over bodies of |g_i|²/2m_i: a bound in joules, computed locally.
3. **Parallel descent.** A contact couples two bodies, so the Hessian is bounded by twice its block
   diagonal: with D_i = M_i + 2 Σ_c J_ciᵀ G_c J_ci the step u ← u − D⁻¹g never increases Φ. This is the
   mass splitting of the rigid law seen from the primal side. The preconditioned Hessian has its
   spectrum in [λ_lo, 1] with the computable bound λ_lo ≥ min_i λ_min(D_i⁻¹M_i), so the iteration count
   depends on ℓ and on the number of contacts per body, and not on the size or depth of the pile.

The price of a finite ℓ is softness. In the units of the integer world (D = 94 mm, h = 1/480 s) the
static overlap at the bottom of a column of L spheres is L·g·h²/ℓ² = 4.5·10⁻⁴·L/ℓ² diameters: 0.7 % for
L = 64 at ℓ = 2. An impact at 4 m/s penetrates about v·h/(√2·ℓ) = 3 % of a diameter.

### 1.3 State, cache, certificate

The impulse network of the rigid law played three roles at once. In the compliant law they separate:

* **State**: positions, velocities, angular velocities, and per contact the accumulated tangential slip
  (and rolling angle). The slip is real history; it is where the static indeterminacy of friction lives.
* **Cache**: the iterate of the solver. Recomputable from the state; worth only iterations.
* **Certificate**: gᵀM⁻¹g. Zero at rest, so "nothing will change here" is a statement the arithmetic can check.

### 1.4 Memory of a body behind its ports

Seen through its ports a linear elastic body is G(s) = Σ_i φ_i φ_iᵀ / (s² + 2ζω_i s + ω_i²) restricted
to the port coordinates. A mode with port participation b_i contributes a Hankel singular value
|b_i|²/(4ζω_i) (force to velocity). Modes far above the band of the port forces respond statically:
they need no variable, only their compliance. The declared softness of the contacts sets that band:
no body can be loaded faster than a contact period. So the same parameter that makes the contact
problem well posed also bounds the memory of every body, and the count is a property of the ports, the
band and the damping, not of the mesh.

### 1.5 Fracture as a change of operator

K = Σ_e k_e b_e b_eᵀ. Removing bond e is a rank-one change. With T = B A⁻¹ Bᵀ (A = M/h² + K) and the
leverage ρ_e = k_e T_ee ∈ [0, 1), the strains after the removal are exactly

    ε' = ε + T[:, e] · k_e ε_e / (1 − ρ_e).

ρ_e → 1 marks a bond whose removal frees a mechanism; 1/(1 − ρ_e) is the amplification of what the bond
carried. The influence T decays with distance at the rate of section 1.1 for the dynamic operator, so
both ρ_e and the redistribution are local computations.

## 2. Predictions made before the measurements, and what happened

| | prediction | outcome |
|---|---|---|
| P1 | The explicit reference of the compliant law converges with the time step (the model has a limit). | **held** (section 4.1) |
| P2 | In the rigid law another exact equilibrium gives the same collapse load but a different response below the limit (limit analysis). | **held** (section 3) |
| P3 | Solver iterations per step do not grow with the depth of a column; they grow like ℓ with acceleration and ℓ² without. | **held** (section 4.3); the time to rest still grows like the depth squared |
| P4 | The response of one step decays geometrically with the contact-graph distance, faster than the bound from κ. | **held** (section 4.4) |
| P5 | Chebyshev acceleration over the certified interval is the right parallel accelerator. | **failed**: it diverges when contacts open and close (section 4.2) |
| P6 | The implicit game step at ℓ = 2 is within 10 % of the explicit reference. | **failed for kicks**, held for collapse (section 4.2) |
| P7 | The number of memory variables of a body does not depend on the mesh. | **held** (section 5) |
| P8 | A per-step certificate makes an approximate solver reproduce the fracture sequence of the exact one. | **failed as stated** (section 6) |
| P9 | The leverage of a bond predicts which bond breaks next. | **failed**: it predicts the consequence, not the order (section 6.3) |
| P10 | A purely viscous rolling resistance is enough for rest. | **failed**: the pile creeps for ever (section 4.1) |

## 3. The rigid law: is the impulse network a state?

`gpu-probe/lab4.cjs`, law v3, four piles of 512 at rest, 4 / 16 / 64 / 256 iterations
(`results/phase2/rigid_network_state_or_cache.json`). One geometry, four impulse networks on it:

* **A**: exact integer equilibrium from the float solve warm-started with the stored impulses;
* **A′**: the same float solution plus noise of one unit per component, projected again (noise control;
  the networks differ by 0.025 %);
* **B**: exact integer equilibrium from a cold float solve (differs from A by 52–56 %);
* **Z**: all impulses zero.

The cold solve converged on piles 11 and 12 only. On piles 13 and 14 it stopped at an imbalance of
364–547 units, so B is not an equilibrium there (it moves by itself, 0.4 D in total on pile 13). The table
below averages all four piles; restricting B to piles 11 and 12 does not change any conclusion.

Distance between the displacement fields of two responses, L1 of the difference over the mean L1 of the
two responses, mean over piles and events:

| iterations | load | A′ (noise) | B (other equilibrium) | Z (amnesia) |
|---|---|---|---|---|
| 4 | kick 1 m/s | 0.11 | 1.21 | 1.85 |
| 4 | removal of a buried sphere | 0.03 | 1.11 | 1.70 |
| 4 | slump | 0.16 | 0.46 | 0.63 |
| 16 | kick 1 m/s | 0.10 | 1.04 | 1.71 |
| 16 | removal | 0.05 | 1.07 | 1.51 |
| 16 | slump | 0.05 | 0.32 | 0.34 |
| 64 | kick 1 m/s | 0.17 | 0.86 | 1.45 |
| 64 | removal | 0.25 | 1.16 | 1.42 |
| 64 | slump | 0.11 | 0.24 | 0.35 |
| 256 | kick 1 m/s | 0.43 | 0.93 | 1.20 |
| 256 | removal | 0.33 | 1.13 | 1.19 |
| 256 | slump | 0.10 | 0.25 | 0.25 |

Slow tilt to failure (gravity rotated by 1/111 every 24 substeps): the tangent at which 5 % of the bodies
have moved 0.1 D, per pile, for A / A′ / B / Z at 256 iterations: 0.59 / 0.64 / 0.62 / 0.59,
0.54 / 0.53 / 0.50 / 0.50, 0.50 / 0.49 / 0.49 / 0.48, 0.51 / 0.52 / 0.52 / 0.52. B and Z are inside the spread of the noise
control.

Amnesia by itself (control run from Z): 146–176 bodies move more than 0.01 D at 4 iterations, 7–12 at 16,
0–5 at 64, 0–4 at 256, where the pile finds a balance in 7–16 substeps on three piles of four.

What this says:

* **Below the limit the network matters.** For kicks and removals the response from another equilibrium
  differs from the response from A by 2–11 times more than the noise control does. The claim of G3 was
  not supported by its own data, but it is true for these loads.
* **At the limit it does not.** The failure angle and the slump do not distinguish the networks beyond
  noise. This is the static theorem of limit analysis: the load a rigid-plastic assembly can carry does
  not depend on its self-stress.
* **Storing the network is a matter of iterations, not of physics.** With enough iterations amnesia costs
  no motion and behaves like one more equilibrium.
* **The rigid response is ill conditioned.** The effect of one quantum of noise grows with the iteration
  count (0.11 → 0.43 for the 1 m/s kick). A better-converged rigid solver is a more sensitive one. This is
  the same fact as the missing limit of the reference.

So the rigid law has a state-like quantity that it does not define: which equilibrium the pile is in.
The answer to "state or cache" is "neither": it is an undetermined part of the model that behaves like
state for small loads and like cache for collapse.

## 4. The compliant contact

`lab/soft.py`, `lab/soft_exp.py`, `lab/soft_local.py`, tables by `lab/soft_summary.py`
(`soft_ell*.json`, `soft_budget_vs_converged_*.json`). Equal spheres with rotation, D = 94 mm, substep
h = 1/480 s as in the integer world. Normal: one-sided spring with a damper. Tangential: spring with a
slider, one μ = 0.5. Rolling: spring with a slider, μ_r = 0.1. Stiffness from ℓ (k_n = m ℓ²/h²),
k_t = 2/7 k_n, damping ratio 0.5. Piles of 576 spheres in a box of 8 × 8 diameters, about ten layers,
settled by the explicit integrator. Two integrators of the same law: explicit (the reference as the
step goes to zero, written e8 … e128 for 8 … 128 steps per substep) and one implicit step per substep.

### 4.1 The model has a limit and a rest state

Mean |log ratio| to e128 per class of observables (0.01 is about 1 %), ℓ = 2 (`soft_ell2.json`):

| | kicked sphere | fields after kicks | removal | slump | slow tilt |
|---|---|---|---|---|---|
| e32 | 0.031 | 0.008 | 0.001 | 0.019 | 0.003 |
| e64 | 0.011 | 0.002 | 0.002 | 0.028 | 0.001 |
| e64, positions perturbed by 10⁻⁹ D | 0.011 | 0.002 | 0.002 | 0.049 | 0.001 |

The displacement of a sphere kicked at 4 m/s is 0.1231, 0.1253, 0.1274 D at e32, e64, e128. Every
observable converges; the slump sits at its own chaos floor of 3–5 %. The rigid reference of 1g moved
by 24–49 % between 128 and 256 iterations.

Rest. With the spring-slider rolling law the pile decays to 10⁻¹⁶ J per body within two seconds and on
to 10⁻²² J in total, and that state is also a fixed point of the implicit step (certificate 10⁻¹⁵ J with zero
iterations). Two cheaper rolling laws fail, and they fail the way the rest quantum did:

* no rolling resistance: single spheres roll on the floor for ever (0.4 mm/s after 8 s);
* a viscous resistance capped by a Coulomb limit: the whole pile creeps at 4–9 mm/s for ever, because a
  viscous law has no static branch. Rest needs a variable that can hold a load at zero velocity.

The explicit integrator is unstable at 4 and 6 steps per substep (the pile at rest drifts by 290 D and
7 D in total) and stable from 8.

### 4.2 The game step against the reference

Same table, ℓ = 2. "BE" is the backward-Euler step; "n" the restarted momentum of section 4.3:

| | evaluations per substep | kicked sphere | fields after kicks | removal | slump | slow tilt |
|---|---|---|---|---|---|---|
| explicit e8 | 8 | 0.162 | 0.021 | 0.008 | 0.027 | 0.057 |
| explicit e16 | 16 | 0.069 | 0.044 | 0.004 | 0.038 | 0.005 |
| BE, converged | 12.8 on average | 0.724 | 0.727 | 0.120 | 0.057 | 0.066 |
| BE, 16 n | 16 | 0.791 | 0.711 | 0.120 | 0.062 | 0.069 |
| BE, 16 plain Jacobi | 16 | 0.468 | 0.278 | 0.116 | 0.254 | 0.056 |
| implicit midpoint, converged | 10.0 on average | 0.860 | 0.414 | 0.010 | 0.046 | 0.043 |

Converged BE at other settings: ℓ = 1: 0.30 / 0.64 / 0.18 / 0.06 / 0.03; ℓ = 4: 0.47 / 1.11 / 0.35 / 0.04 / 0.08;
ℓ = 2 with critical damping: 0.41 / 0.62 / 0.05 / 0.10 / 0.10.

* **Collapse is reproduced by the explicit step and by every converged implicit step** at ℓ ≤ 2: slump and
  slow tilt within 3–7 %. With 16 plain Jacobi iterations the slump is off by 25 %.
* **An impact is not reproduced by a one-step implicit rule.** Backward Euler removes the rebound: the
  sphere kicked at 4 m/s travels 0.041 D instead of 0.127 D, and the displacement field after a kick is
  half of the reference. The midpoint rule errs the other way (1.08 D). The contact lasts 1.1 substeps
  at ℓ = 2; no rule with one force evaluation point per substep resolves it, at any ℓ from 1 to 4 and at
  either damping.
* **The explicit step at 8–16 evaluations per substep reproduces everything** to 0.4–6 %, except the
  kicked sphere (16 % at 8, 7 % at 16). It has no solver, no cache and no iteration count. Its weakness is the
  stability limit, which depends on the stiffest and lightest thing in the world.
* Chebyshev acceleration over the certified interval diverged in the slump (kinetic energy 10⁴ to 10⁴⁸ J)
  and at 8 iterations sent a kicked sphere 26 times further than the converged step. A fixed three-term recurrence is not safe when contacts
  open and close. 1h had to add its gate for the same reason.

So the implicit step is a law of its own, defined at the scale of the substep like the rigid law was:
inelastic contact, unique solution. Its own reference is the converged step.

### 4.3 Iterations

Momentum with a per-body restart: the gradient is taken at the extrapolated point, and a body whose
step points uphill drops its momentum. No global quantity is needed. Certificate after 8 / 16 / 32
iterations of the first substep after a 4 m/s kick, from 1.9·10³ J: plain Jacobi 1.7 / 0.21 / 0.024,
Chebyshev 190 / 11 / 0.16, restarted momentum 0.15 / 0.007 / 7·10⁻⁵.

A fixed budget against the converged step of the same law (mean |log ratio|):

| | kicked sphere | fields after kicks | removal | slump | slow tilt |
|---|---|---|---|---|---|
| ℓ = 1, 8 n | 0.048 | 0.027 | 0.001 | 0.020 | 0.003 |
| ℓ = 1, 16 n | 0.002 | 0.001 | 0.001 | 0.030 | 0.020 |
| ℓ = 2, 8 n | 0.275 | 0.303 | 0.003 | 0.036 | 0.046 |
| ℓ = 2, 16 n | 0.103 | 0.034 | 0.002 | 0.017 | 0.003 |
| ℓ = 2, 32 n | 0.005 | 0.001 | 0.001 | 0.008 | 0.047 |
| ℓ = 2, 16 plain Jacobi | 0.448 | 0.486 | 0.004 | 0.197 | 0.050 |
| ℓ = 4, 16 n | 0.230 | 0.420 | 0.011 | 0.085 | 0.063 |
| ℓ = 4, 32 n | 0.041 | 0.025 | 0.003 | 0.010 | 0.004 |

About 8ℓ iterations for 5–10 %.

Depth (`soft_depth.json`): columns of 8 to 128 spheres released at zero overlap, every substep solved to
10⁻¹² J per body. Largest iteration count of any substep:

| ℓ | L = 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 1 | 22 | 23 | 23 | 22 | 21 |
| 2 | 38 | 43 | 44 | 42 | 42 |
| 4 | 48 | 84 | 89 | 87 | 81 |
| 8 | 89 | 134 | 164 | 191 | 179 |
| 1, plain Jacobi | 78 | 85 | 88 | 89 | – |
| 2, plain Jacobi | 209 | 248 | 263 | 268 | – |

The count saturates once the column is a few ℓ deep and is then about 20ℓ. Plain Jacobi needs 4–6 times
more.
With a budget of 16 the 64-column at ℓ = 2 ends at a compression of 0.2349 D against the static 0.2350 D.

**What still grows with depth is the time to rest.** Substeps until the kinetic energy is below 10⁻¹⁴ J
per body: ℓ = 2: 150, 404, 2028 for L = 8, 16, 32 and more than 2400 for 64; ℓ = 4: 60, 174, 427, 1519;
ℓ = 8: 18, 52, 104, 615, 2141. That is about 6–8 (L/ℓ)² substeps. It is the ringing of the lowest mode of
the column, which a damper in the contacts damps in proportion to its frequency squared. The exponent
is the one of the rigid law (layers² over iterations); the cause is now the declared law and no longer
the solver.

### 4.4 Locality of one step

`soft_locality_ell2.json`, `soft_locality_ell4.json`: one body of the pile at rest gets 0.2 m/s; the
step is solved to convergence with and without it. Share of the kinetic energy of the response by
contact-graph distance from the source (three sources):

| hops | ℓ = 2 | ℓ = 4 |
|---|---|---|
| 0 | 0.47–0.85 | 0.37–0.71 |
| 1 | 0.13–0.44 | 0.20–0.23 |
| 2 | 0.015–0.093 | 0.056–0.23 |
| 4 | 0.002–0.005 | 0.007–0.049 |
| 6 | 6·10⁻⁵–8·10⁻⁴ | 4·10⁻⁴–7·10⁻³ |
| 8 | 2·10⁻⁶–4·10⁻⁵ | 7·10⁻⁵–2·10⁻³ |

The amplitude falls by about 0.45 per hop at ℓ = 2 and 0.6 at ℓ = 4; the bound from the certified
condition number (110 and 395) is 0.83 and 0.90. Solving only the bodies within r hops and leaving the
rest untouched changes the response by (energy of the error over energy of the response):

| r | bodies solved | ℓ = 2 | ℓ = 4 |
|---|---|---|---|
| 2 | 10–16 | 0.9–4.5 % | 7.5–24 % |
| 3 | 26–43 | 0.3–1.1 % | 1.6–12 % |
| 4 | 52–101 | 0.06–0.4 % | 0.4–4.9 % |
| 6 | 136–271 | 0.002–0.03 % | 0.05–1.0 % |

### 4.5 State and cache in the compliant law

`soft_state_ell2.json`, ℓ = 2, 32 n. The same loads as in section 3 from: **A** the pile at rest; **A′** its
slips perturbed by 10⁻⁶; **B** the pile after a history (gravity tilted to tan 0.15 for 0.3 s and back,
then rest: the bodies moved by 0.0012 D on average, one by 0.2 D); **Z** slips and rolling angles set to
zero; **Zslip** only the slips.

| load | A′ | B | Z | Zslip |
|---|---|---|---|---|
| kick 1 m/s (two spheres) | 0.000 | 0.24–0.32 | 1.81–1.86 | 1.78–1.83 |
| kick 4 m/s | 0.000 | 0.06 | 1.01 | 0.95 |
| side kick 1 m/s | 0.000 | 0.10 | 1.65 | 1.58 |
| removal | 0.000 | 0.002 | 0.47 | 0.40 |
| tilt to tan 0.4 | 0.000 | 0.20 | 0.38 | 0.18 |

(field distance to the response from A, as in section 3)

* **The response is well conditioned.** A perturbation of the state does not show in four digits. In the
  rigid law the same test gave 0.03–0.43.
* **The slips are state.** Forgetting them makes the pile at rest move (0.99 D in total, two bodies by
  more than 0.01 D), and every response is then dominated by that settling.
* **Another history is another state, but a mild one**: 6–32 % of the field for kicks against 86–121 % in
  the rigid law, and that includes the small difference in geometry.
* **The solver iterate is a cache.** Converged steps from a cold start (the free velocity) and from a warm
  start differ by 2·10⁻⁴ of the field; the cold start costs 50 % more iterations. At a budget of 16 the
  two differ by 27 % of the field: an under-converged step turns the cache into state again.

### 4.6 How far a kick reaches

`soft_wake_ell2.json`, `soft_radius_ell2.json`. Bodies that ever exceed a speed after a kick (576 in the
pile), 16 n:

| kick | above 10 mm/s | above 1 mm/s | above 0.1 mm/s | last substep above 0.1 mm/s |
|---|---|---|---|---|
| 1 m/s | 29 | 370 | 569 | 60 |
| 4 m/s | 172 | 531 | 573 | 337 |

A compliant pile carries sound. Finite signal speed does not make the wake small; in this pile the wave
is everywhere within eight substeps. What limits the wake is its amplitude against a threshold.

Whole events with only the bodies within r hops of the source moving and the rest held:

| event | r = 4 | r = 6 | r = 8 | all |
|---|---|---|---|---|
| kick 1 m/s: kicked sphere, D | 0.0072 | 0.0081 | 0.0082 | 0.0082 |
| kick 1 m/s: sum of displacements, D | 0.040 | 0.050 | 0.055 | 0.058 |
| kick 4 m/s: kicked sphere | 0.016 | 0.038 | 0.038 | 0.039 |
| kick 4 m/s: sum of displacements | 0.33 | 0.59 | 0.66 | 0.93 |
| kick 4 m/s: bodies moved more than 0.01 D | 8 | 8 | 8 | 10 |
| removal: sum of displacements | 1.35 | 1.37 | 1.38 | 1.38 |
| bodies moving | 58–75 | 172–214 | 305–393 | 576 |

The kicked sphere and the removal are settled by six hops (within 3 %). The sum of displacements is not:
after the 4 m/s kick a third of it lies in many small displacements far away. Holding the far field
rigid is the crudest boundary model. The better one is the far field as a linear passive system behind
its boundary, which is section 5 applied to a pile; it was not built.

### 4.7 The pass criterion proposed for this experiment

"Reference: the same law with an explicit step. Pass: every observable converges, the game step is within
10 %, iterations do not grow with depth and fit in 16–32. Fail: no ℓ gives both the softness tolerance and
the budget."

* Observables converge: **pass**.
* Game step within 10 %: **pass for the explicit step** at 16 evaluations per substep (8 for everything but
  the kicked sphere); **fail for the implicit step** on impacts, pass on collapse.
* Iterations do not grow with depth: **pass**. In 16–32: **pass** for ℓ ≤ 2 (16) and ℓ = 4 (32) against the
  converged step of the same law, with the restarted momentum; fail with plain Jacobi.
* The fail condition is not met: ℓ = 2 gives 0.7 % overlap under a column of 64 and fits the budget.


## 5. Memory of an elastic body behind its ports

`lab/memory.py`. A beam (triangular lattice 48 × 6, 576 degrees of freedom, modal damping 2 %), free in
the plane. Ports: three nodes on the top face and two on the bottom. This is a stand-in for the Phase 1b
beam with the same construction (rigid frame plus modes, everything unilateral outside); it is not the
RBSN code.

### 5.1 How many variables any linear model needs

Hankel singular values of the port map (force to velocity, flexible part, by Gramians;
`memory_hankel.json`). Number of internal modes with σ_i above a fraction of σ₁:

| port forces limited to | 10 % | 3 % | 1 % |
|---|---|---|---|
| ω₁ (first flexible frequency) | 1 | 3 | 4 |
| 3 ω₁ | 2 | 4 | 6 |
| 10 ω₁ | 4 | 8 | 13 |
| 30 ω₁ | 4 | 15 | 38 |
| 100 ω₁ | 4 | 35 | 83 |
| unlimited | 4 | 39 | 95 (of 573) |

The same body and the same ports at three mesh densities (144, 576 and 1296 degrees of freedom): 2–3 / 4 /
5–6 at 3 ω₁, 3–4 / 8 / 12–13 at 10 ω₁, 3–4 / 15–17 / 33–38 at 30 ω₁. **The count does not depend on the
number of elements.** Beams of 24, 48 and 96 cells in length at one absolute band: 16, 13 and 14 at 1 %.

### 5.2 A sequence of impacts

Six strikers (each 5 % of the beam mass) fall on three ports within seven periods of the first mode. Two
of them arrive while the beam still rings from the previous one, two hit the same port twice. The beam
lies on two unilateral supports under its own weight. Supports and strikers are outside the body and
see it only through port forces. Reference: all 573 flexible modes. A run with 400 modes reproduces it to
0.0–0.6 % except at a contact time of 0.1 T₁, where it differs by 1.9 % in the rebound: there the scene
itself is sensitive (the beam chatters on its supports), and 3–4 % is the floor of that row.

Error of the rebound velocities of the strikers / of the peak strain in the bottom fibre
(`memory_impacts.json`):

| contact time | rigid body | static compliance, no memory | 1 mode + static | 2 modes + static | 4 + static | 8 + static | 16 + static | 16 lowest modes, no static part |
|---|---|---|---|---|---|---|---|---|
| T₁ | 13 % / 100 % | 14 % / 43 % | 0.0 % / 0.2 % | 0.0 % / 0.3 % | 0.0 % / 0.1 % | 0 / 0 | 0 / 0 | 0.1 % / 0.8 % |
| 0.3 T₁ | 28 % / 100 % | 21 % / 17 % | 4.5 % / 0.9 % | 0.5 % / 1.2 % | 0.6 % / 1.2 % | 0.2 % / 0.4 % | 0.2 % / 0.3 % | 0.8 % / 1.6 % |
| 0.1 T₁ | 41 % / 100 % | 23 % / 3.4 % | 17 % / 12 % | 19 % / 11 % | 3.2 % / 1.8 % | 5.2 % / 3.3 % | 3.9 % / 1.7 % | 6.3 % / 2.7 % |
| 0.03 T₁ | 84 % / 100 % | 73 % / 9.7 % | 54 % / 0.6 % | 54 % / 1.4 % | 43 % / 7.0 % | 23 % / 10 % | 6.0 % / 2.5 % | 13 % / 8.9 % |

* A body without memory is wrong by 14–73 % in what it gives back to the strikers, at every contact time.
  The instantaneous description of a body is not enough, however complete it is statically.
* The number of memory variables follows the contact time: 1, 2, 4, 16 for T₁, 0.3 T₁, 0.1 T₁, 0.03 T₁,
  as the Hankel count of section 5.1 says it should (the corner frequency is π over the contact time).
* The static compliance of the dropped modes is worth a factor of 2–4 in error at equal count.
* Ranking modes by their Hankel singular value did not beat ranking by frequency here: for this beam
  and these ports the two orders nearly coincide.

**Reading of Phase 1b and 1c.** Their members were "rigid frame plus k lowest modes", and their cost floor
was the collar of single cells around every loaded port. The contact there lasted about 0.36 ms against
a 10 ms beam period, a contact time of 0.04 T₁. The table says such a load needs of the order of 16–40
memory variables per member and that most of the response is a local wave near the port. That is what
the collar was computing. A softer declared contact lowers the count, and it also changes the physics:
1b measured that a contact 100 times softer leaves the damage threshold within 6 % and halves the
damage at twice the threshold. The declared softness is a physical parameter with a price, not a free
numerical one.

## 6. Fracture decisions and the operator

`lab/events.py`. A plate (triangular lattice 40 × 24, 1920 degrees of freedom, 2753 bonds) with brittle
bonds of random strength (±30 %), held at the bottom and pulled at the top at a constant rate until it
has parted. Implicit steps; after each solve the most overloaded bond above its limit breaks and the
step is solved again. Reference: a direct solve. Two regimes: ℓ = 4 (condition number about 10²) and
ℓ = 16 (about 2·10³). Three plates each (`events_ell4.json`, `events_ell16.json`).

### 6.1 An under-converged solver is a different material

Peak load (the capacity of the plate) against the reference, worst of three plates:

| solver | ℓ = 4 | ℓ = 16 |
|---|---|---|
| residual 10⁻⁶ | 0.04 % | 0.01 % |
| residual 10⁻⁴ | 4.0 % | 0.5 % |
| residual 10⁻³ | 6.4 % | 4.8 % |
| residual 10⁻² | 58 % | 270 % |
| 64 iterations per solve | 0.02 % | 93 % |
| 32 iterations | 0.6 % | 370 % |
| 16 iterations | 190 % | 810 % |
| 8 iterations | 200 % | 1180 % |

With a fixed budget the plate is up to 13 times too strong: the displacement lags, the strains are too
small, the bonds break late. This is the fracture counterpart of the rigid pile that is too soft at 4
iterations. The iteration count is a material parameter unless something bounds the error.

### 6.2 Certified decisions

The solver iterates until every decision of the step is certain: no bond can be above its limit, or the
leader is certainly above it and certainly the leader. The bound uses only the residual and the masses
(section header of `events.py`).

| | ℓ = 4 | ℓ = 16 |
|---|---|---|
| iterations, share of a solve to 10⁻¹⁰ | 22 % | 36 % |
| the same for a residual of 10⁻⁶ | 32 % | 39 % |
| peak load error | 0.09 % | 0.01 % |
| plates with the same set of broken bonds | 2 of 3 | 3 of 3 |
| plates with the identical sequence of breaks | 1 of 3 | 3 of 3 |

* The certificate picks the tolerance by itself and lands near a residual of 10⁻⁶, a little cheaper. It
  removes a tuning parameter; it does not remove much work. The bound is loose by up to the condition
  number, and it uses the residual of the whole plate, so it would get worse with the size of the world.
  A bound localised with the decay of section 6.3 is the missing piece.
* **A certificate per step does not certify the trajectory.** On one plate two bonds are within 1 % of
  each other at the eighth break. Every earlier decision was certified, and the order still flips,
  because each certificate is relative to the exact solution of a step whose history already carries
  the earlier errors. A residual of 10⁻⁶ flips the same pair. The consequence is another crack path
  (overlap of the broken sets 0.92 with the certificate, 0.63 with the residual 10⁻⁶) and the same
  capacity (0.09 %).
* The rigorous way to carry the bound along the trajectory (the implicit step is non-expansive in the
  energy norm, so step bounds add) gives a sum of 0.9 against a strain limit of 0.01: useless. A floor on
  the accuracy of the state (10⁻² … 10⁻⁵) on top of the certificate changed nothing.
* After the peak, 15 bonds break within one step and even a residual of 10⁻⁶ leaves the reference
  sequence at break 92 of 106. The order of breaks in an avalanche is not a reproducible quantity of
  the model. The set of broken bonds and the capacity are.

So four properties that were tied together come apart here. Determinism is a property of the arithmetic
and holds for any budget. Convergence is what the certificate measures. Physical accuracy at the level
of consequences (capacity) needs a residual of about 10⁻⁴ or a certificate. Identity of individual
outcomes is not available even from a tight solver once two candidates are within its error, and it
is not needed for the capacity.

### 6.3 Leverage and influence

`critical_ell4.json`, `critical_damaged_ell4.json`.

* The rank-one formula of section 1.5 reproduces a re-solve to 8·10⁻¹⁵: breaking a bond is an update of
  the operator, not a new problem.
* Influence between bonds (normalised |T_je|) against distance, median: 0.10 at one spacing, 6·10⁻³ at
  3–5, 4·10⁻⁴ at 8–12, 5·10⁻⁶ at 18–30 for the dynamic operator (a decade per five spacings at ℓ = 4).
  For the static operator: 0.11, 8·10⁻³, 1.2·10⁻³, 1.9·10⁻⁴, an algebraic tail. The mass term is what
  makes the redistribution local.
* The leverage of a bond from a solve restricted to a disc around it: 1 % error with 44 nodes (radius 4),
  0.03 % with 162 (radius 9).
* In the intact plate the leverage is 0.64 in the bulk (the mean is fixed by the ratio of degrees of
  freedom to bonds). During the failure one bond reaches static leverage 1.0000 (it alone holds a piece),
  and after it ten bonds are above 0.95.
* **The leverage does not say which bond breaks next.** The next twelve bonds to break rank around 2500
  of 2750 by leverage during the main failure; by overload most of them are within the first 50. Leverage measures the
  consequence of a break (the amplification 1/(1 − ρ), the freeing of a mechanism), not its likelihood.

## 7. What follows for the design

A system built from what was measured here, layer by layer. Each layer states what is established and
what is not.

**Arithmetic (established in 1f–1h).** Integer state, order-free sums, canonical identifiers. Determinism
lives here and nowhere else: any local integer map gives the same bits on both GPUs, and skipping a body
whose inputs did not change is exact for any such map. Neither fact depends on what the map computes, so
both carry over to a new law unchanged.

**Law (this phase, float64 only).** A strictly convex step: compliant contact with a declared softness,
a spring with a slider for sliding and for rolling, one friction coefficient, rotation. The state holds
positions, velocities, the slips of the contacts, damage, and nothing that depends on the solver
history. The test for that is the one of sections 3 and 4.5: a cold solver start has to give the warm
result.

**Solver.** The majorised block-Jacobi step with momentum that is dropped per body when it points
uphill. No global reduction, no spectral estimate, monotone without momentum. Its iteration count is an
accuracy setting and no longer a material parameter, because the certificate says how far the iterate
is from the solution of the step.

**Rest.** A fixed point of the map. In float64 the pile decays to machine zero; in integers this has to
become an exact fixed point without a rest quantum. Not shown yet.

**Bodies.** A stiff interior is never time-resolved. A body is its rigid motion, the static compliance of
its ports and r memory coordinates, with r read off the port map for the declared contact band. Galerkin
projection keeps M and K symmetric and positive, so a reduced body stays passive and enters the same
convex step as everything else. No separate coupling scheme is needed for bodies that share one step.

**Events.** A threshold decision is taken when it is certain. Breaking a bond is a rank-one update with
a local footprint. Ties are settled by the canonical order, which is a matter of determinism, not of
accuracy.

**Cost.** Per active body and frame the work is about ω_c·T contact evaluations times a constant: a
property of the declared softness, not of the depth. What decides the total is the number of active
bodies, and there the result of this phase is mixed (section 4.6).

## 8. What this phase did not show

* Nothing was run in integers or on a GPU. Exact rest, bit identity and cost of the compliant law are
  open. The slips add state per contact, and the memory target of 600 B per body was already missed.
* The pile has 576 spheres. The reach of a kick in a large pile is a prediction from 576, not a
  measurement.
* The wake of the physical wave is large. Holding the far field rigid costs 10–40 % of the displacement
  sum at six hops. A sleeping region as a linear passive system behind a boundary (section 1.4 applied
  to a pile) was not built. The energy argument that would certify it is in section 9.
* A column still needs of the order of (L/ℓ)² substeps to come to rest. It is now the ringing of the
  declared law and no longer an artefact of the iteration count, but it is the same exponent.
* The certificate is loose by up to the condition number and global. A local one needs the decay bound
  inside it.
* Bodies with different internal step sizes and an energy contract between them were not tested. Inside
  one convex step the question does not arise; across different rates it does, and Phase 0d already
  found it hard.
* A coarse space was not tried. On this operator it is no longer needed for the iteration count at ℓ of a
  few units. It would matter for stiff bonded structures (ℓ of hundreds) and possibly for the trajectory
  drift of section 6.2, because the error an iterative solver leaves is smooth and the implicit step
  does not damp smooth error.
* Experiment A used a lattice beam, not the RBSN member of Phase 1b, and no damage.

## 9. Next phase, pre-registered

Integer version of the compliant law, JS specification first, then WGSL, both devices.

| id | claim | pass | kill |
|---|---|---|---|
| I1 | The integer map has exact rest without a rest quantum or a handover rule. | piles of 1728 and 4096 reach a state that no further substep changes; control runs stay put | rest needs a threshold rule of the kind listed in section 0 |
| I2 | It reproduces the float64 lab. | collapse observables (slump, ramp, removal) within 10 % of `lab/soft.py` at 16 iterations on three piles | worse than law v4 at 16 iterations against the same lab |
| I3 | Determinism is untouched. | JS = M3 Pro = GTX 1660 SUPER, full and incremental, fuzz of host edits | any difference |
| I4 | It costs what it replaces. | GPU time per substep at 16 iterations ≤ 1.5 × v4 at 16; rest ≤ 0.2 ms per substep | more than 3 × |
| I5 | The reach of a kick does not grow with the pile. | bodies re-evaluated after a 1 m/s kick in piles of 22k and 205k differ by less than 2 × | proportional to the pile |
| I6 | A sleeping region can be held while the work done on it is below its margin. | with E = min over its contacts of (cone margin)²/2k, no contact of a held region would have slipped in the full run whenever the work delivered through its boundary stayed below E | a slip below the bound (the bound is wrong), or the bound is never above the work (it is useless) |

Not relaxed: determinism, exact momentum accounting, no silent drops.
