# Shatterbird — Project Context, Theory and Decisions

*Snapshot: 2026-11-06. Working names: **Shatterbird** (engine), **Flock IR** (semantic IR / determinism-checking layer), *Murmuration* (tentative name for the P2P layer).*

This document condenses a long design session: Pavel (author) plus two AI assistants (ChatGPT, Claude) used adversarially, including several explicit "try to kill the idea" rounds and independent reviews of each other's conclusions. It records the problem, how it was reformulated, what was settled, what was rejected and why, and what remains open. It is the context for the empirical work in `shatterbird-phase0-spec.md`.

Status tags: **[settled]** — established within stated assumptions; applicability of those assumptions must still be checked; **[candidate]** — a proposal to test, not an approved architecture; **[open]** — unresolved; **[rejected]** — considered and dropped within the stated scope, reason given.

This file is background, not an instruction to implement every candidate. The current empirical assignment and its boundaries are in §0 of the Phase 0 spec. Notation throughout both files: $\Delta t$ is a physical time step, $\ell_e$ is generator distance for facet $e$, and $\ell_{\mathrm{cb}}$ is the crack-band width prescribed by the selected fracture model; these lengths need not coincide.

---

## 0. Session path (how we got here)

1. Started as "cross-GPU determinism" in the LLM-inference sense (reproducible accumulators, NVIDIA CCCL `gpu_to_gpu`/RFA, exact superaccumulators). Discarded as a different problem class; the only surviving idea is "define the arithmetic contract instead of hoping hardware agrees".
2. Reframed by Pavel to **physics (and shaders) on consumer GPUs in the browser**, with destruction, P2P, bit-exactness, 60+ FPS on phones (§1).
3. First architecture (ChatGPT): integer authoritative state + float rendering, explicit peridynamics, edge-pass/gather-pass force assembly, canonical contact keys, input-only lockstep.
4. DSL question → "fix semantics first, syntax later" (§9).
5. Kill attempt (ChatGPT): prior art and the wave-speed time step. It partly attacked claims never made ("first bit-exact GPU physics") and its own proposals; the time-step argument was valid but only against explicit wave-resolved models.
6. Fundamental analysis (Claude): problem split into determinism / ordering / physics time scale / scale & lockstep / network; four hard constraints (§2); WGSL float rules checked; Teardown multiplayer found as key prior art.
7. Pavel's numeric formalizations (base+delta positions, Q-formats, dimensionless inertia, ownership by size, canonical hashing, pre-colored bond graph) and several "mine-sweeping" rounds (rotation quantization, feature IDs, friction basis, ID layout). Valuable, but convergence was declared prematurely: the mines were computational while the mechanics was untouched.
8. Review #1 (ChatGPT): mechanics not closed — λ indeterminacy in redundant rigid constraints, threshold ≠ fracture energy, quasi-static-by-FPS is the wrong criterion, several factual errors → move to **RBSN** (§6.2).
9. Review #2: stiffness need not be softened; **compliant ALM** separates physical and numerical stiffness; energy must be audited at system level; RBSN needs a full interface law.
10. Review #3: **two-parallel-springs counterexample** — the examined ALM update with shared moderate ρ can recover force distribution extremely slowly, even with an exact primal solve. ALM motion + static global force reconstruction became a candidate (§6.5), not a proven necessity. Whether different ALM policies, global solves or hierarchies meet the budget is a Phase 0 question.
11. Handoff review: corrected the overgeneralization from that counterexample; distinguished trial damage from committed history; made the interface-law, S1/S2 consistency and energy-audit requirements explicit. Begin with E1, then expand through the empirical stages rather than implementing the entire roadmap at once.

---

## 1. Ambition

Original intent: a GPU-native, massively parallel, dynamically destructible physics engine that simulates fracture of complex structures with physically grounded damage propagation, dynamic fragment formation and interaction of hundreds of thousands of elements; entirely in the browser via WebGPU, no simulation server, on consumer hardware (desktop GPUs, recent iPhones, Snapdragon/Adreno, Apple M), 60+ FPS; **bit-identical evolution across GPU vendors**; P2P multiplayer with minimal traffic. "Physically grounded fracture, not deterministic animation." "Beat Photon Quantum and Rapier" — on the axis that matters (deterministic GPU-scale destruction), not on feature breadth.

Working philosophy: a pet project is only worth doing if it aims at being the best/only thing of its kind. Ambition is not scope.

Sharpened target after adversarial review — no confirmed equivalent was found in the sources reviewed:

> A destructible physical world whose evolution is a pure function of (initial state, ordered inputs), bit-identical across supported WebGPU devices satisfying the same computation contract, with material-grade fracture (RBSN-class: calibrated elasticity, strength and fracture energy, objective w.r.t. discretization) at game rates and GPU scale, synchronized peer-to-peer by inputs only.

Every ingredient has precedents (§10); the review did not establish an equivalent combination, but it is not a proof of worldwide novelty. Success criteria stay explicit: 100k visual debris particles ≠ 100k authoritative interacting elements. Simplifications are allowed only when named; never as silent substitution of the success criterion.

Consequences worth building on (if achieved): replays = initial state + input journal (kilobytes); verifiable/citable simulations; spectators without a server; world = link + journal.

---

## 2. Hard constraints that shape the design

1. **Time scale of stiff materials.** Elastic wave speed in concrete ≈ 3.5–4.5 km/s. Explicit wave-resolving integration at 1 cm resolution needs ~2.5 µs steps (~6700 steps per 1/60 s) → infeasible in real time at the intended scale. Implicit integration is the current candidate. Its price is numerical damping/dispersion, and it is not limited to invisible kHz modes: implicit Euler multiplies an undamped oscillator's amplitude by $1/\sqrt{1+(\omega\Delta t)^2}$ per step; a 50 Hz mode at $\Delta t=1/480$ s keeps ~17% after one period. Quasi-static treatment is valid when $T_\text{load}\gg L/c$ (load time vs. wave transit); frame rate is not the criterion. Collapse after support loss is dynamic.
2. **Lockstep is paced by the weakest peer.** Computation is replicated, not distributed. Max world complexity / support for weak devices / inputs-only sync: pick two (mitigation: sessions by device tier). Physical LOD must be a function of shared simulation state or of a shared simulation event, never of local FPS or hardware. Visual LOD is free.
3. **Memory bandwidth.** Phones ≈ 60–85 GB/s shared CPU+GPU vs 546 GB/s on M4 Max (7–9×). All throughput estimates in the session were illustrative; phone capacity is unknown until measured (which bodies, which contacts, which quality, sustained thermal state).
4. **Network latency.** Input delay or rollback; rolling back a heavy GPU world multiplies its cost by rollback depth ⇒ delay-based lockstep for the world, local prediction for avatars. Two-way avatar↔world coupling has an inherent cost (delay, correction or rollback).
5. **Unprovable stacks.** WGSL → Tint / WebKit's compiler / Naga → MSL/HLSL/SPIR-V → vendor drivers cannot be proven correct, only tested. Defense: startup self-tests, periodic state hashes, checkpoint resync. A resync does not fix a systematic backend bug (divergence recurs) → device exclusion policy.

---

## 3. Determinism layer [settled in principle]

**Floats are out of the authoritative path.** WGSL §15.7 permits reassociation outright, fusion when at least as accurate, rounding intermediates up or down, flush-to-zero of subnormals, and assuming infinities/NaN are absent; `fma` may be unfused. Observed in practice: reassociation of f32 expressions measured on Chromium/SwiftShader; luma.gl switches its double-float arithmetic to an integer-controlled path on Apple WebGPU. Box2D v3 achieves cross-platform float determinism on CPU only because the compiler can be controlled (FMA off, custom trig) — WebGPU offers no such control.

**Integers are in.** WGSL integer semantics are fully specified (wrap-around, defined division by zero, masked shifts). Authoritative state is integer/fixed-point; rendering may use floats as long as nothing rendered feeds back into simulation.

**Determinism by construction.** Tick map $S_{t+1}=F_v(S_t,I_t)$ ($v$ = semantics version). If every stage is built only from:
1. pure functions of canonically indexed inputs writing only their own slots;
2. exact commutative/associative integer reductions — atomic adds whose return value is unused; modular wrap yields the exact final sum whenever the true sum fits; saturation only per term before summation (SunaEngine's "per-term clamped" sums);
3. sorts/scans with a total order on stable keys;

then results are independent of thread scheduling, and induction over ticks gives identical trajectories.

Forbidden: using an atomic's return value as a slot without later canonicalization; any dependence on subgroup size (differs by vendor); data races; reading and applying accumulated deltas in the same pass (turns Jacobi into schedule-dependent asynchronous iteration); floats in the authoritative path; overflow inside term computation (only the final modular sum is protected).

**Everything is state.** $S_t$ includes position remainders, warm-start multipliers, damage, deferred queues, sleep state, hysteresis memory, spawn counters. With finite iteration counts warm start changes the trajectory, so it is state and is hashed.

Scope of the guarantee: reproduction of the chosen discrete model on a race-free restricted subset. Physical meaningfulness is a separate question (§6).

---

## 4. Numerics [candidate; formats deliberately not frozen]

From Pavel's formalization, refined in review:

- **Positions.** Per-body base $X_\text{base}$ in Q20.12 (0.244 mm) plus per-tick delta $\Delta X$ in Q4.28 (±8 m, 3.7 nm), initialized at tick start from a stored sub-quantum remainder; all geometry uses base+delta; rebase on simulation ticks (never render frames). Information-wise a 48-bit position; the gain is i32 hot math. Matches Catto's `deltaPosition` change in Box2D v3/Solver2D (all solvers moved to deltas because of round-off far from origin). All intermediate relative quantities must be range-bounded, not only $\Delta X$.
- **Rotation.** Q2.30. Rule: *wide math per body, narrow math per pair*. Mixed `narrow × wide` products (16×32) cost two native multiplies; WGSL has no mul-high, a full 32×32→64 costs ~10–15 ops.
- **Rotation-dependent geometry in narrowphase.** Rotating local vertices by a coarse (Q1.14) matrix makes a 1 m plate's corners step by ~0.06 mm → stack jitter. Fix: wide rotation; Gottschalk-style OBB SAT in body A's frame ($C=R_A^TR_B$, $t=R_A^Td$ computed wide once, then extents × |C| narrow×wide); manifold once per tick with local anchors updated per substep.
- **Dimensionless inertia.** $\tilde r=r/R$, $\tilde I^{-1}=mR^2I^{-1}$ → effective mass depends only on mass ratio and shape numbers. Needs an aspect-ratio cap (slivers: $mR^2/I_\min\sim2(R/t)^2$). $\Delta\omega$ still scales as $1/(mR)$; velocity increments are not dimensionless; a tiny mass ratio can still quantize to zero.
- **Quantization principle.** Time-varying quantities wide. Static data may be coarse **only if** every deformation measure is taken relative to the identically quantized rest state in a corotational frame (static error cancels; rigid rotation of an unloaded structure produces no stress). Mechanics sets the strict scale: in the reference concrete bond the elastic stretch at failure is ~50 µm ($\varepsilon=\sigma_t/E=10^{-4}$, $\ell=0.5$ m), so deformation errors must be ≪ 50 µm. Q4.28 deltas (3.7 nm) and Q2.30 rotations (~1e-9 rad) pass if this principle holds; Q1.14 anchors (~30 µm on a 0.5 m cell) would not without it.
- **Coefficients matter as much as coordinates.** E.g. ALM's $s=1/(1+\rho/k)$ can round to exactly 1 and silently delete compliance. Formats must cover update coefficients, multiplier increments, divisions and local-system conditioning. Phase 0 (f64) records the real ranges.

---

## 5. Discrete topology and contacts [candidate]

- **Identity ≠ priority ≠ storage.** Stable entity ID; processing-priority key (SizeClass, StableID); physical slot. Make slot allocation deterministic too (accidental storage-order dependence becomes harmless; hashes debuggable). Periodic deterministic spatial re-sort of storage is allowed and probably needed for bandwidth.
- **Spawned IDs.** Canonical from (parent ID, fragment index) via prefix sum over spawn events sorted canonically. No wrapping tick fields (a 12-bit SpawnTick wraps after ~68 s and produces equal IDs → ambiguous pair ownership). A global monotonic spawn counter is a contract on total spawns per session; a wider logical ID is cheap (comparisons, not arithmetic).
- **Pair ownership.** The lower priority key owns the pair (small/young owns). For touching compact bodies, packing bounds the owned list; this does *not* bound broadphase candidates, margins, or deep interpenetration during fracture. Top-K truncation is an approximation (a symmetric drop is still a dropped contact) — part of a scene contract, not a correctness guarantee.
- **Impulse transfer.** Owner computes the pair and atomically adds to the partner; the owner's own contributions also go through the accumulator (no overwrite); separate read and apply buffers.
- **Spatial hash without a global radix sort.** count (atomic counts are deterministic) → scan → scatter → per-bucket canonicalization; or block histograms + local rank. Sorting is one way to get canonical semantics, not a requirement when downstream is order-independent. Multi-level grids by size class.
- **Manifolds (Box3D / Soft Step practice).** Contact points stored in local coordinates; separation updated per substep from body deltas (Catto: sub-stepping without recomputing contacts); feature IDs with type and flip bit for warm-start matching; reduction ties broken by ID. Limitation: anchor reuse does not discover new contacts within a tick → speculative contacts / CCD policy for fast debris and rotating beams.
- **Friction basis.** Duff's orthonormal basis jumps at the $n_z$ sign change (90° or 180° depending on direction). Prefer bases from reference geometry (face edges / edge directions); on basis change re-project the stored tangential impulse vector rather than dropping it. Box (pyramid) friction is an anisotropic change of the friction law — acceptable only as a named simplification.
- **Coloring.** A proper coloring of the static bond graph stays proper under edge deletion (closed result). New debris contacts need separate handling; AVBD's "rare same-color conflicts" relies on coloring that includes current interactions. Dispatch count = colors × iterations × substeps; workgroup-local coloring sweeps for clusters that fit one workgroup are a mitigation (with inter-cluster coupling handled separately, which changes convergence).
- **Collision geometry.** OBB proxies of Voronoi cells overlap neighbors → false contacts once a bond breaks; colliders must be the convex cells (more expensive than OBB SAT). A broken facet gives initial contact geometry, not a permanent manifold.

---

## 6. Mechanics — the core

### 6.1 Reasoning path

| Step | Proposal | Fate |
|---|---|---|
| 1 | Explicit bond-based peridynamics on integer GPU; polynomial bond energy $U=\frac\kappa4(r^2-r_0^2)^2$ (no sqrt) | **[rejected]** wave-speed time step (§2.1); polynomial potential loses stiffness under compression (zero force as $r\to0$); bond-based PD restricts ν to 1/4 in 3D. GPU PD is not new (PeriPy). |
| 2 | Quasi-static fracture analysis at impact + rigid fragments (Bao et al. 2007; Hahn & Wojtan 2015/16) | Kept as precedent for decoupling rigid motion from elastic state. |
| 3 | Rigid chunks + stiff breakable joints, break on solver multiplier λ | **[rejected as stated]** internal forces in redundant rigid constraints are indeterminate (solver/order dependent); a strength threshold is not a fracture-energy law: $U_c/A=\sigma_c^2\ell/(2E)$ depends on element size. |
| 4 | **RBSN** on Voronoi cells with energetic cohesive interfaces | **[candidate, core]** reconciles "rigid-body engine" with material mechanics. |
| 5 | Soften stiffness ~1000× to fit the solver | **[rejected — error]** an oscillator-accuracy guideline (constraint Hertz well below the step rate, Nyquist) was mistaken for a stiffness limit. Implicit schemes give correct static equilibrium at physical stiffness; the cost moves into solving. The resulting "energy explosion at break" was an artifact. |
| 6 | Compliant ALM: physical compliance + numerical penalty ρ | **[candidate]** physical stiffness kept at convergence; scalar $k_\text{eff}$ bounded; full-block conditioning still needs analysis. |
| 7 | Two-parallel-springs counterexample | The examined shared-ρ ALM update recovers force *distribution* very slowly when $k\gg\rho$; no impossibility theorem for local solvers (§6.4). |
| 8 | ALM motion + static global elastic force solve on intact clusters, conditionally prefactorized | **[candidate, untested]** (§6.5); consistency and cost must both be established. |

### 6.2 RBSN baseline

Rigid-body-spring concept (Kawai). Bolander & Saito (1998): stiffness scaled by the Voronoi tessellation gives elastic uniformity under uniform strain and reduces directional bias. Bolander et al. (2000): energetic fracture formulation addressing network density and random geometry. Berton & Bolander: 3D RBSN, notched beam in three-point bending. 2D facet springs (thickness $b$, facet length $l$, generator distance $\ell_e$): $k_n=Ebl/\ell_e$, $k_t=k_n$ for uniformity, $k_\phi=k_nl^2/12$. Uniformity with $k_t=k_n$ corresponds to $\nu=0$; nonzero ν needs an additional scheme (e.g. Bolander's iterative Poisson treatment). $\nu=0$ is fine as an early control model, not an agreed material property.

"RBSN" is a family, not a guarantee: simple element-removal criteria are not mesh-objective (Sofianos & Koumousis 2017 state this explicitly); crack patterns depend chaotically on the network while macroscopic response should not. The energetic formulation must be chosen deliberately.

Required specifics:
- 3D: three translational + three rotational springs per facet with facet geometric properties.
- Generator vs. centroid: uniformity results refer to generators; DOFs at centroids require a consistent transform of arms/operators, not substituting distances.
- Damage history: irreversibility, unloading/reloading, compression after opening, mixed mode (e.g. max-history equivalent opening as in Grassl–Bolander).
- Constitutive closure before fracture tests: distinguish total displacement jump $\delta$ from crack opening $w$; specify equivalent opening, damage/history functions, stored energy of all spring components, and the area used to normalize $G_c$. Degrading shear and moment with a damage variable driven only by normal opening does not automatically dissipate exactly $G_cA$ under combined loading.
- Macroscopic mode I on a random lattice generally loads facets in combined normal/shear modes. Bolander–Berton use resultant direction, projected facet area and a corresponding crack-band width. A simpler per-geometric-facet law is a separate candidate; it does not inherit that formulation's mesh-objectivity results.
- One interface law covering cohesion, compression and post-damage contact/friction. A rigid contact on top of an intact compliant bond would block or double its compliance; an opened crack must not pull; closed faces must transmit pressure and friction.
- Corotational objectivity under large rigid motion (rotate an unloaded structure → no stress, no damage).

### 6.3 Fracture energy, crack band, energy balance

1D linear cohesive softening $\sigma(w)=\sigma_t(1-w/w_c)$, $w_c=2G_c/\sigma_t$; with $\delta=\ell_{\mathrm{cb}}\sigma/E+w$: $d\delta/dw=1-\ell_{\mathrm{cb}}\sigma_t^2/(2EG_c)$. A strictly increasing local total-opening branch requires $\ell_{\mathrm{cb}}<2l_{ch}$, $l_{ch}=EG_c/\sigma_t^2$; this does not exclude structural snap-back. Reference concrete ($E=30$ GPa, $G_c=100$ N/m, $\sigma_t=3$ MPa): $l_{ch}\approx0.33$ m; $\ell_{\mathrm{cb}}=0.5$ m satisfies this local condition, near the limit. For larger bands $\sigma_\text{eff}=\min(\sigma_t,\sqrt{2EG_c/\ell_{\mathrm{cb}}})$ reaches the nonnegative-slope bound; equality is degenerate and needs explicit treatment. This changes initiation and capacity — a coarse-graining model to validate, not a general stability guarantee. Three separate claims: (1) admissible law shape, (2) correct dissipated energy, (3) the mesh reproduces the material. (1)+(2) do not imply (3).

Energy is audited at system level:

$$\Delta(T+U_\text{el}+V_g)+D_\text{fracture}+D_\text{friction}+D_\text{physical}+D_\text{numerical}=W_\text{ext}+\varepsilon$$

Here $D_\text{numerical}$ is an independently identified integrator/algorithm loss where available; $\varepsilon$ is an unexplained **signed** balance residual. Never define the whole residual as numerical dissipation: it may contain energy creation, a constitutive error or inconsistent motion/force coupling. Deleting a spring without changing velocities deletes its potential energy (artificial dissipation); energy available to fragments depends on unloading of the whole structure; crack creation consumes energy according to the chosen area and mode convention. A local cap like $F^2/(2k)$ is not an evolution law and determines neither velocity directions nor momentum.

### 6.4 Compliant ALM: a specific slow-convergence counterexample

Compliant ALM (Macklin's Newton VBD/AVBD note, 2026): user parameter = physical stiffness $k$ (compliance $\alpha=1/k$, hard limit $\alpha=0$); solver parameter $\rho$ = penalty/dual step controlling conditioning, not material; $s=k/(k+\rho)$, $k_\text{eff}=\rho s$, $\lambda^+=s(\lambda+\rho C)$; λ is a warm-started multiplier, equal to physical spring force at the fixed point $\lambda=kC$. For a translational dynamic constraint the note recommends ρ ≈ or moderately above $m_\text{eff}/\Delta t^2$; rotational rows require the corresponding inertia scale and units. Quasi-static experiments need an explicitly specified penalty scale rather than an implicit physical time step. Contact limitations reported in that research branch are historical observations, not a guarantee about every ALM implementation.

At fixed multipliers and spring coefficients, the reduced primal contribution is $s\lambda C+\tfrac12 k_\text{eff}C^2$ (up to a constant). With $J=\partial C/\partial q$, its gradient is $J^\top(s\lambda+k_\text{eff}C)$; the positive-semidefinite approximation to its Hessian is $k_\text{eff}J^\top J$. For nonlinear $C$, the exact Hessian also has geometric terms. The objective and update schedule, not a shortened source-code sketch, define the experiment.

**Counterexample.** One DOF, two parallel springs $k_1,k_2$, load $P$, shared ρ, exact primal solve each iteration. The λ-iteration matrix is $\frac{s_1s_2}{\sigma}\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$, $\sigma=s_1+s_2$, with eigenvalues 0 (force balance satisfied after one update) and
$$r=\frac{2s_1s_2}{s_1+s_2}=\frac{2}{2+\rho/k_1+\rho/k_2}$$
for the distribution error. With $k_1=1000\rho$, $k_2=2000\rho$: $r\approx0.99925$, ~6100 iterations per 100× error reduction; from cold start the split is ≈ P/2 : P/2 instead of P/3 : 2P/3.

**Interpretation and limits.** This disproves a general fast-force-convergence claim for the examined shared-ρ ALM iteration. It does not prove that every local solver fails within 8–16 sweeps or that a separate global force solve is mandatory. The original static problem here is scalar, $(k_1+k_2)x=P$, with matrix condition number 1: slow convergence of the split iteration must not be confused with poor conditioning of that primal system. Large $k\Delta t^2/m$ describes stiffness relative to a time step, not by itself the condition number of a full system. Different penalty policies, blocks, preconditioners and update rules remain candidates. Phase 0 decides whether a global or hierarchical method is needed for the target scenes. Positions, forces and constitutive residual ($\lambda-kC$) are separate errors and are all measured.

Also: bounded $k_\text{eff}$ improves but does not by itself bound the condition number of per-cell 6×6 blocks. For the dynamic positive-semidefinite Hessian approximation, whiten by the full mass/inertia metric ($m/\Delta t^2$, $I/\Delta t^2$); then $\hat H=I+\sum_e k_{\text{eff},e}v_ev_e^T$ has eigenvalues in $[1,\,1+\sum_e k_{\text{eff},e}\|v_e\|^2]$, with $v_e$ the whitened constraint Jacobians. This needs explicit geometric and valence bounds (Voronoi valence has a tail; "14–16 faces" is an average, not a maximum). The bound does not automatically apply to an exact geometric Hessian, a softening tangent or an inertia-free quasi-static system.

### 6.5 Candidate: motion/force decoupling with a static global solve

- **Motion:** ALM block descent with ρ at inertial scale; fast, sufficiently accurate motion is a hypothesis to measure, not a consequence of bounded $k_\text{eff}$.
- **Forces:** in a linear setting reconstruct $Ku=f_\text{ext}-Ma$ using acceleration from the motion solution. This is exact only with consistent motion and loads; an accurate reconstruction cannot repair an incorrect acceleration by itself. Rotational and large-motion versions need the corresponding inertial terms and frame transformations. A cluster initially unstressed and nonrotating, with uniform initial velocity in uniform gravity, must acquire no self-weight stress. A previously loaded cluster can retain stress and vibrate after release.
- **Null space and conditioning:** a free intact cluster has rigid translation/rotation modes; establish load compatibility and remove only those modes without adding artificial supports. Recompute component/null-space handling after separation; additional mechanisms may appear. Neither removing inertia nor having similar material moduli guarantees a well-conditioned $K$.
- **Conditional prefactorization:** a fixed graph preserves sparsity, not necessarily matrix values. Reuse numerical factors only while coefficients, geometry in the chosen frame and boundary/active-set treatment remain valid. Whole-cluster rigid motion can be transformed out under a fixed-reference approximation; relative cell rotations, damage softening and opening/closure can change $K$ before any edge disappears. Distinguish reusable symbolic analysis from numerical factors.
- **Updates:** local stiffness changes may permit low-rank updates/downdates; benchmark them against refactorization. Connectivity changes, loss of definiteness and changing constraints require explicit handling, not unconditional Cholesky downdates. Historical precomputation/cutting references in §13 are leads to verify, not evidence that this cost model is already established.
- **Consistency:** define whether reconstructed displacements are corrections to motion or an auxiliary elastic field, how forces feed back, and which field determines external work and measured displacement. Check force and moment resultants, displacement, work, energy and damage history against S0. Passing a force-only test is insufficient.
- **Open:** fixed-point factorization accuracy (candidate: iterative refinement against an exact integer residual); GPU parallelism and memory cost of sparse triangular solves; update cost under widespread damage; validity of the motion/force coupling. Alternatives include crack-aware multigrid/aggregation, preconditioned Krylov, per-bond ρ policies and cluster block solves.
- A unified physical RBSN solved accurately is the reference. State explicitly when S2 solves the same discrete equations and when it is a reduced/partitioned approximation whose additional error must be measured.

### 6.6 Damage evolution

Separate **trial damage** from **committed material history**. During a nonlinear load/time step, trial damage may change with trial displacements; compute it from the last accepted history, not by irreversibly accumulating every solver iterate. Commit history only when the physical step or event substep is accepted; rejected trials leave it unchanged. Solver sweeps are not elapsed physical time. Freezing damage for a whole step and updating it once afterwards without restoring equilibrium is a separate lagged scheme, not the reference by default.

Simultaneous damage on a stale stress field can over-break; Bolander–Berton change at most one spring set per cycle and re-equilibrate (sequential, using resultant direction and projected facet area). Reproducing that algorithm and solving a coupled damage law are distinct choices: document which model/path the reference follows. Sub-stepping to an event does not resolve exact ties or select a physically admissible branch after instability. A parallel scheme needs justification against the specified reference, including step refinement and branch/event rules. Rate/viscous regularization introduces a new time scale and can raise dissipated energy above $G_c$ (COMSOL warns about excessive delay). **[rejected]** the earlier $\Delta d\le\Delta t\,v_c/\ell_e$ (≈4.17 with the given numbers, i.e. no limit; $d$ is not a crack-front position). Deterministic tie-breaks give reproducibility; plausibility must come from the mechanics.

### 6.7 Dynamics and impacts

- Implicit damping must be measured (§2.1); BDF2, implicit midpoint or Newmark-type schemes are candidates if it is unacceptable.
- $J=\int F\,dt$ does not determine peak force. If the actual contact duration is below a ~2 ms substep, interpreting $J/\Delta t$ as the peak makes fracture time-step dependent. An explicit duration model is a candidate (Hahn & Wojtan estimate it with Hertz contact and point out the limitation of Bao's impulse treatment). Arbitrary smoothing of forces over frames itself introduces a time dependence.
- Unresolved at the proposed coarse time steps: sub-millisecond wave effects (spall, dynamic fragmentation statistics). Rate-dependent phenomenology can cover a bounded class of impacts; it does not reproduce wave propagation. The heuristic "$c\,\Delta t\approx8$ m per substep, so arrival errors are small" is not a bound (an implicit elliptic solve is not wave propagation).

---

## 7. Solver history

| Solver | Why considered | Status |
|---|---|---|
| AVBD (Giles, Diaz, Yuksel, SIGGRAPH 2025) | implicit, handles large stiffness ratios, quality at low iteration counts, already in WGSL (three-avbd, webphysics) | Dropped once over fixed-point 6×6 concerns; **back** via compliant ALM. |
| Soft Step / TGS Soft (Catto, Box2D v3) | substeps + soft constraints + relax; robust to mass ratios, long chains, stacks; deterministic lineage | Reference for the rigid contact pipeline. Its constraint softness is a stabilization choice, not physical compliance. |
| Substepped integer XPBD | scalar math | **[rejected]** Solver2D: TGS Soft beats XPBD in nearly all tests; patent concern claimed by Avian's author (Avian moved to TGS Soft). |
| Jacobi + mass splitting for debris | GPU-friendly, one thread per contact | **[avoid]** NVIDIA patent US10776532B2 (Tonge, Benevolenski, Voroshilov), active to 2033; hybrid convergence unproven. |
| Pre-colored GS for structural bonds | coloring survives edge deletion | Property kept; dispatch cost is the issue. |
| Compliant-ALM VBD + static global force solve | §6.4–6.5 | **[candidate]**; consistency and cost untested |

Box3D (Catto, MIT) is the semantic reference for the rigid contact pipeline: manifolds once per step with local anchors, delta positions, feature IDs, relax iterations, CCD.

---

## 8. Networking [candidate]

- Input-only lockstep with input delay; avatars predicted locally against the delayed deterministic world; avatar influence enters as delayed inputs. Physics-affecting delays in game design (fuses, projectile flight, wind-ups) hide input delay.
- Checkpoints for late join (one-time snapshot transfer, MBs) and recovery; hashes detect divergence but do not arbitrate who is right; determinism detects, it does not prevent, cheating (input validation and acceptance rules needed); rewind needs checkpoints/replay (dissipative fracture is not reversible).
- WebRTC needs signaling and, for some NATs, TURN — compatible with "no simulation server". **[rejected]** using third-party call infrastructure (e.g. VK calls) as TURN: neither legitimate nor reliable; own/commodity TURN is cheap.
- Deferring a fracture by k ticks changes causality if the object keeps interacting; any such queue is part of the model semantics, not a visual trick.
- Teardown multiplayer (Mar 2026): fixed-point deterministic destruction commands + state sync of moving objects, ~1 Mbit/s per client budget, a player acts as host; the author revised his earlier skepticism about full determinism. It neither proves nor disproves the full-lockstep approach; it shows the industry leader stopped at a hybrid.

---

## 9. Flock IR

Purpose: fix semantics — units, number formats, rounding, ranges, memory-access patterns, accumulation modes — and statically reject non-deterministic patterns (atomic return values as slots, floats in the authoritative path, subgroup-size dependence); generate WGSL and a Rust reference from one source. Start as an embedded Rust layer (no parser). Codegen prior art: CubeCL (Rust-like kernels → WGSL/CUDA/HIP/CPU, no determinism contract), wgsl-rs (Rust subset → WGSL, CPU-runnable), vyre-conform (CPU reference as specification, determinism and workgroup-size-independence checks). Shared codegen proves consistency between backends, not correctness of the model → independent oracles (analytic tests, high-precision references) are mandatory. Range proofs only under explicit scene contracts (bounded shapes, velocities, interaction counts, defined out-of-range behavior). Naming: `flock` collides with flock(2)/flock(1); crate name `flock-ir`. A full language is not a condition of feasibility; a strictly defined semantics is.

---

## 10. Prior art map

| Project | What it shows | Gap vs. Shatterbird |
|---|---|---|
| SunaEngine (2026) | Integer-only WebGPU PBF fluid: Q16.16, exact 32×32→64 with round-half-even, saturating overflow, per-term clamped integer sums, canonicalized neighbor truncation, 256-bit state hashes; cross-device identity claimed; ~40k particles at 80 fps (RTX 2070 S) / 60 fps (Apple Silicon) | particles; no rigid bodies, no fracture |
| three-avbd | AVBD in WGSL, hundreds of thousands of bodies, contacts, joints, fracture suites | deterministic on one device only |
| webphysics | AVBD in WebGPU, LBVH broadphase, colored solves | prototype; no cross-vendor guarantee |
| Rapier (WASM) | cross-platform deterministic snapshots | CPU; scale |
| Box2D v3 | float cross-platform determinism on CPU; Soft Step; graph coloring | CPU; 2D |
| dbox2d | Box2D v3 port with Q32.32 mode (~3× slower than float64 on a pyramid), probes incl. Q20.12 grids | CPU; 2D |
| Maul3D | Soft Step lineage, deterministic breaking, voxel destruction inside rollback, cross-platform hash gating, ~5000-body city benchmark | CPU; IEEE float |
| Teardown MP | deterministic fixed-point destruction commands + state sync | host-player server, ~1 Mbit/s/client |
| The Finals | destruction synchronized server-side | server |
| Newton / Warp | GPU physics, deterministic contacts | run-to-run guarantee only |
| PeriPy | GPU peridynamics, bond forces + node reduction | not real-time/browser/deterministic |
| Unreal Chaos | connection graphs, strain, clustering | not material-grade, not cross-vendor deterministic |
| Photon Quantum 3 | deterministic netcode + fixed point | CPU; relay servers |
| yacineMTB demo (2026) | claimed deterministic WebGPU rigid-body multiplayer | cross-vendor unconfirmed |

---

## 11. Corrected errors (do not regress)

- Novelty attacks on claims never made; the claim is the combination (§1).
- "Integers unnecessary since Box2D is float-deterministic" — false for WebGPU (§3).
- Maul3D is IEEE float, not fixed-point; "fixed point + Soft Step + breaking already proven together" was a conflation.
- Convergence declared from computational mines while mechanics was untouched.
- Quasi-static justified by frame rate; "implicit dynamics fully resolves structural motion"; "$c\,\Delta t$ bounds arrival error".
- ~1000× softening as solver necessity and the derived energy-explosion mine.
- "λ closes force indeterminacy" — only at convergence (§6.4).
- The shared-ρ counterexample treated as proof that all local solvers fail or that a global force solve is mandatory; large $k\Delta t^2/m$ equated with a matrix condition number.
- Fixed bond graph equated with fixed numerical $K$; free-cluster null spaces omitted; force-only agreement treated as sufficient for S2.
- $\Delta d\le\Delta t\,v_c/\ell_e$; committing trial damage per solver iterate; the opposite blanket ban on recomputing trial damage inside a nonlinear solve.
- Degrading all spring components assumed to preserve a pure-mode $G_cA$ budget automatically; geometric and projected fracture areas conflated.
- Whole energy-balance residual labeled numerical dissipation; every freely falling cluster expected to become instantaneously stress-free.
- Duff basis "always 180°" (can be 90°); recommending mass splitting while citing patent risk for XPBD; "anti-cheat by construction"; "per-cell ranking requires radix sort"; "velocity increments are dimensionless"; "top-K is bounded by packing" (only for touching compact bodies).

---

## 12. Open questions (priority order)

1. Force distribution in intact redundant structures within a real-time budget: ALM alone vs. static global solve vs. hierarchy.
2. Damage evolution under simultaneous overload; path dependence; energy objectivity across resolutions and random tessellations.
3. Time discretization: damping, impact peak force, support-loss dynamics.
4. Contacts in the same formulation (stacks, mass ratios, CCD), cohesion↔contact transition.
5. 3D rotational blocks, normalization, fixed-point ranges (from Phase 0 range logs).
6. Performance on target devices: dispatches per frame, atomics under contention (Adreno/Apple), achieved bandwidth, thermal steady state.
7. Network details: input acceptance, checkpoint cadence, late join, device exclusion on systematic divergence.

---

## 13. References

**Determinism / WebGPU**
- WGSL spec, §15.7 Floating Point Evaluation — https://www.w3.org/TR/WGSL/
- typeshade #378 (f32 reassociation observed) — https://github.com/typeshade/typeshade/issues/378
- luma.gl fp64 (integer path on Apple WebGPU) — https://luma.gl/docs/api-reference/shadertools/shader-modules/fp64
- Box2D determinism — https://box2d.org/posts/2024/08/determinism/
- Rapier determinism (JS/WASM) — https://rapier.rs/docs/user_guides/javascript/determinism/
- SunaEngine — https://engine.sunabox.dev/ · https://github.com/DARIENBATHALTER/sunaengine
- Newton: deterministic simulation — https://newton-physics.github.io/newton/latest/concepts/determinism.html

**Solvers / contacts**
- Catto, Solver2D — https://box2d.org/posts/2024/02/solver2d/
- Catto, Releasing Box2D 3.0 — https://box2d.org/posts/2024/08/releasing-box2d-3.0
- Box3D — https://github.com/erincatto/box3d · contacts docs https://box2d.org/documentation3d/md_simulation.html
- Catto, Soft Constraints (GDC 2011) — https://box2d.org/files/ErinCatto_SoftConstraints_GDC2011.pdf
- Macklin et al., Small Steps in Physics Simulation (2019) — https://mmacklin.com/smallsteps.pdf
- Chen et al., Vertex Block Descent (2024) — https://arxiv.org/abs/2403.06321
- Giles, Diaz, Yuksel, Augmented VBD (SIGGRAPH 2025) — https://graphics.cs.utah.edu/research/projects/avbd/
- Compliant ALM for High-Stiffness Equality Constraints (Newton VBD/AVBD note, 2026) — https://reports.mmacklin.com/alm-high-stiffness/
- Zheng, Luo, Li, Penetration-Free Elastodynamics without Barriers (SIGGRAPH 2026) — https://simulation-intelligence.github.io/barrier-free/
- three-avbd — https://github.com/sbobyn/three-avbd
- webphysics — https://github.com/jure/webphysics
- dbox2d — https://pkg.go.dev/github.com/dhannyell/dbox2d
- Maul3D — https://github.com/Rawframe-Project/maul3d
- Tonge et al., Mass Splitting for Jitter-Free Parallel Rigid Body Simulation (2012) — https://www.richardtonge.com/papers/Tonge-2012-MassSplittingForJitterFreeParallelRigidBodySimulation-preprint.pdf
- US10776532B2, Modified effective mass for parallel rigid body simulation — https://patents.google.com/patent/US10776532B2/en
- Avian: XPBD → TGS Soft — https://github.com/Jondolf/avian/issues/346 · https://joonaa.dev/blog/06/avian-0-1
- Duff et al., Building an Orthonormal Basis, Revisited — https://graphics.pixar.com/library/OrthonormalB/paper.pdf

**Fracture mechanics**
- Bao, Hong, Teran, Fedkiw, Fracturing Rigid Materials (2007) — https://math.ucdavis.edu/~jteran/papers/BHTF07.pdf
- Hahn & Wojtan (2015) — https://visualcomputing.ist.ac.at/publications/2015/HRBFSwBE/
- Hahn & Wojtan, Fast Approximations for BEM-based Brittle Fracture (2016) — https://pub.ista.ac.at/~wojtan/projects/2016_Hahn_FastFracture/download/2016_Hahn_FastFracture_small.pdf
- Bolander & Saito, Fracture analyses using spring networks with random geometry, Eng. Fract. Mech. (1998)
- Bolander, Hong, Yoshitake, Structural concrete analysis using RBSN (2000) — https://onlinelibrary.wiley.com/doi/pdf/10.1111/0885-9507.00177
- Bolander & Yip, RBSN modeling of cement-based composites (FraMCoS-4) — https://framcos.org/FraMCoS-4/773.pdf
- Bolander & Berton, cohesive zone modeling with RBSN (FraMCoS-5) — https://framcos.org/wp-content/uploads/framcos-papers/Cohesive_Zone_Modeling_of_Fracture_in_Irregular_Lattices.pdf
- Berton, Bolander, Hikosaka, 3D RBSN notched beam — https://ctresources.info/ccp/paper.html?id=3540
- Bolander & Sukumar, irregular lattice model (PRB 2005) — https://escholarship.org/content/qt9qn7n4sb/qt9qn7n4sb.pdf
- Grassl & Bolander — https://arxiv.org/html/1510.05184
- Bolander, Yoshitake, Thomure (JSCE 1999) — https://www.jstage.jst.go.jp/article/jscej1984/1999/633/1999_633_25/_article
- Sofianos & Koumousis, RBSN for plasticity and fracture (2017) — https://avestia.com/CSEE2017_Proceedings/files/paper/ICSENM/ICSENM_146.pdf
- Lattice/RBSN mesh dependency (ECF17) — https://gruppofrattura.it/ocs/index.php/esis/ECF17/paper/download/6566/3181
- Červenka et al., crack band (2018) — https://www.cervenkaconsulting.com/assets/files/papers/crack-band_2018_engineering-fract.pdf
- Discrete elements and size effects — https://www.sciencedirect.com/science/article/abs/pii/S0013794417308573
- Reactions with redundant constraints — https://www.sciencedirect.com/science/article/pii/S0094114X09001384 · https://doi.org/10.1115/1.4006958
- COMSOL, Decohesion (viscous regularization warning) — https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_theory.06.092.html
- Abaqus, cohesive behavior/contact — https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-cohesivebehavior.htm
- Kaczmarczyk et al., multigrid for fracture (2010) — https://www.sciencedirect.com/science/article/abs/pii/S0045782509003934
- Real-time FEM with precomputation and low-rank updates for cuts: Bro-Nielsen & Cotin (1996); Cotin, Delingette, Ayache (1999) — cited from memory, verify the exact methods and assumptions before using as support for S2.

**Networking**
- Gustafsson, The unlikely story of Teardown Multiplayer (2026) — https://blog.voxagon.se/2026/03/13/teardown-multiplayer.html
- MDN, WebRTC connectivity — https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Connectivity

**Tooling**
- CubeCL — https://github.com/tracel-ai/cubecl
- wgsl-rs — https://docs.rs/crate/wgsl-rs/latest
- vyre-conform — https://docs.rs/crate/vyre-conform/latest

**Other prior art mentioned (no link captured):** PeriPy (OpenCL peridynamics), Unreal Chaos destruction, Photon Quantum 3, The Finals; yacineMTB WebGPU demo — https://digg.com/tech/8zw74ln1
