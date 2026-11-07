# Phase 0b — Track B probe plan (WebGPU kernel cost) — *pre-registered, not run*

*Assignment: `shatterbird-phase0b-step2.md` §7, plus the project lead's one-dispatch variant. This file fixes the probe design before any device measurement. No GPU numbers exist yet.*

## 1. What the probe measures

The cost of representative kernels on an available WebGPU device, as dependent chains without readback.

**In scope:**
- the linear 6-DOF RBSN network of `model::rbsn3` (contract meshes, full inertia tensors, real valences, constant facet coefficients, dynamic state);
- after `docs/phase0b/two-level.md`, the smoothing / residual / transfer kernels of the chosen two-level variant.

**Out of scope:** the cost of damage, finite rotations or contact. A synthetic history law, if added, is reported separately. f32 is acceptable for this probe; it is not the cost of the later integer implementation. A desktop result is never presented as mobile.

## 2. Variants of one explicit substep

**A — two dispatches per substep (baseline from the assignment).**
1. Facet pass: each facet reads its two cells' states and writes its force/moment pair (12 floats) to a facet buffer.
2. Cell pass: each cell gathers its facets' entries through a fixed per-cell facet list with signs (no atomics, no scatter) and integrates.

**B — one dispatch per substep (proposed by the project lead).** Each cell loops over its own facet list, evaluates every facet force itself and integrates in the same dispatch.
- Cell states are ping-ponged: read the previous substep's buffer, write the next, so a dispatch boundary is the only global barrier needed.
- Arithmetic per facet doubles: both cells evaluate it.
- Dispatches per substep halve. No facet buffer, no intermediate traffic, no atomics.

**Determinism and Newton's third law in B.**
- Every pair is evaluated in canonical orientation: the lower cell ID is side *a*, inputs ordered (state_a, state_b, facet constants). The side-*b* cell negates the result.
- Both cells run the same function on the same inputs in the same compiled code path, so they obtain bit-identical $f_{ab}$ on a given device. Negation is exact, so action = −reaction bit for bit at the pair level.
- Caveats to verify, not assume:
  - (i) The guarantee needs one code path. If the compiler specialises the function differently for the two roles, for example through different FMA contraction after inlining into different branches, bitwise equality can break. The probe checks $f_{ab}+f_{ba}=0$ bitwise over all pairs after a run.
  - (ii) Global momentum is still subject to the per-cell summation rounding. That is identical in A and B and deterministic for a fixed facet order per cell.
  - (iii) Nothing here makes f32 results equal across vendors. Cross-device determinism remains a fixed-point question.

**C — empty dispatches** with the same chain length: a separate control value for dispatch overhead, never subtracted from A or B.

## 3. Sizes, chains, measurements

- **Cells:** 1 000 / 10 000 / 30 000, then 100 000 if memory allows. Report real facet counts and buffer bytes, and cover working sets inside and outside cache.
- **Chains of 250 / 400 / 700 / 800 substeps:** A = 500 / 800 / 1 400 / 1 600 dispatches, B = 250 / 400 / 700 / 800, C = matched. These counts come from the stability bounds of geometry-spectrum (safe steps for $d_{\min}$ 0.5 / 0.35 / 0.2 m and the κ guarantee).
- **Timing:**
  - GPU timestamp queries where available, plus observed series time;
  - warm-up, then median and p95 over repeated chains;
  - batch size, adapter / OS / browser, limits, memory.
- **Verification:** final cell states against a CPU reference of the same arithmetic. The tolerance is declared per precision, and the pair antisymmetry is checked bitwise (B).
- **Thermal:** a long run only after a promising short result.
- **Formula:** $T_{\rm frame}(N,h)=\lceil(1/60)/h\rceil\,T_{\rm substep}(N)$, only for the measured scheme. The headroom for contacts, rendering and integer arithmetic is reported as unknown, not zero.

## 4. Implicit-path kernels (after two-level)

Measured work of the chosen variant PTL-SA-s8-CH per routine step, from `docs/phase0b/two-level.md`: ≈3 PCG iterations.

**Per iteration:**
- 1 A-application;
- two Chebyshev(2) smoothings (4 A-applications + 4 block-Jacobi solves);
- 1 residual;
- restriction and prolongation;
- 1 coarse solve;
- 2 global reductions.

**Per step:** ≈37 dependent phases, 3 coarse solves, ≈6 reductions; per event, an $A_c$ update and numeric refactorisation.

**Chain to time:** this kernel sequence with the coarse solve both on CPU (counting the CPU↔GPU transfers) and as GPU triangular solves (counting the elimination-tree depth). The coarse cost does not disappear from the budget because the GPU passes are fast.

## 5. Decision boundary

The probe can refute a direct implementation on a device. Its success does not confirm the cost of the full engine. The architecture is chosen only after §8 of the step-2 assignment: equal-error comparison on corrected E7/E9.
