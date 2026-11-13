# Shatterbird — Phase 0 (mechanics and solver truth, CPU f64)

**Phase 2a (current): state, memory, locality and certificates (CPU float64)** — [`docs/phase2/state-memory-certificates.md`](docs/phase2/state-memory-certificates.md).
- **The rigid law leaves part of the model undefined.** On one geometry another exact equilibrium of the impulses changes the response to kicks and removals 2–11 times more than one quantum of noise does, and leaves the failure angle and the slump unchanged. The sensitivity to noise grows with the iteration count.
- **A compliant contact with rotation has a limit, a rest state and a unique step.** The explicit reference converges to 1–3 %; rest needs a spring with a slider for rolling (a viscous law creeps for ever). Iterations per step do not grow with depth (about 20ℓ to convergence, 8ℓ for 5–10 %), the response of a step decays by about half per contact hop, a perturbed state gives the same response to four digits. The slips are state, the solver iterate is cache.
- **Failed:** a one-step implicit rule does not reproduce an impact (backward Euler halves the response, midpoint overshoots); an explicit step at 8–16 evaluations per substep does. Chebyshev acceleration diverges when contacts open; momentum with a per-body restart replaces it. The time to rest of a column still grows like depth squared, now as ringing of the declared law.
- **Memory of a body behind its ports** is set by the band of the port forces and not by the mesh (2–6 internal modes at 3 ω₁, the same at 144 and 1296 degrees of freedom); a body without memory is wrong by 14–73 % in a sequence of impacts.
- **Fracture:** a fixed iteration budget makes a plate up to 13 times too strong; certified decisions keep the capacity within 0.1 % at 22–36 % of the work of a tight solve, but do not certify the trajectory (a near-tie flips). Bond leverage predicts the consequence of a break, not its order.
- Nothing ran in integers or on a GPU; the next phase is pre-registered in section 9.

**Phase 1h: carrying load further per iteration, fully parallel** — [`docs/phase1/long-range.md`](docs/phase1/long-range.md).
- **Chebyshev-accelerated Jacobi (law v4), no extra memory:** the previous level of the impulses is what the ping-pong copy already holds. Two rules were forced by measurement: accelerate only contacts with a body above the rest threshold (otherwise integer limit cycles prevent exact rest), and only the normal part of the impulse (otherwise friction slips).
- **Evaluated on five held-out piles against 128 iterations:** v4 at N iterations matches plain Jacobi at 2N on kicks and support removal, lies between N and 2N under sustained frictional load; columns settle 2.7–4.8x faster but still ~L^1.8; stacks under a sudden tilt still shear.
- **Same bits on M3 Pro and GTX 1660 SUPER** after two exactness rules (the change-driven skip must test the unaccelerated update; no reprojection of unchanged values). GPU time per iteration +8–56 % when everything moves, wakes at 8–16 iterations cheaper than plain 2N.
- **Found:** a trap in the rest rule of v3 (a body just above the threshold slides forever over an anchored one; lab fix `reff`), and that a sudden load on a column, like a small kick, is not defined without a physical compliance.

**Phase 1g: what a resting pile has to remember, and what the solver gets wrong** — [`docs/phase1/rest-physics.md`](docs/phase1/rest-physics.md).
- **Remeasured at equal horizons from one common equilibrium, with controls:** the rest quantum does not suppress responses on the millimetre scale (the 1f claim is withdrawn); the dominant error is the Jacobi iteration count, which makes a pile too soft (a downward kick sinks 3–6x too far at 4 iterations, 1.0–1.35x at 32).
- **The model lacks a compliance:** how far a kicked sphere sinks into a pile keeps falling with iterations (256 vs 128: 0.5–0.76); it is not defined by a rigid contact law.
- **Rest as an exact balance (law v3):** residuals below the threshold are handed to a designated support through the contact; at rest nothing leans on the world frame, floor and walls carry 100.0000 % of the weight. The impulse network is history (two exact equilibria of one geometry, both at rest, differ by 54 %; from zero impulses the equilibrium of a 1728 pile is not even found), and it is the memory that has to be stored.
- **Change-driven iterations on the GPU** (bitwise unchanged results): a wake at 16–32 iterations costs about what it costs at 4. Same bits on M3 Pro and GTX 1660 SUPER, also with coincident bodies renumbered (canonical ids).
- **Not solved:** stacks under sideways load shear at any Jacobi iteration count; two-pass propagation holds them but breaks piles (not ported).

**Phase 1f: one canonical contact system, full or incremental, on two GPUs** — [`docs/phase1/canonical-system.md`](docs/phase1/canonical-system.md).
- **Same bits on Apple M3 Pro and NVIDIA GTX 1660 SUPER**, stepped fully or incrementally: positions, velocities, contact sets, impulses; also with the bodies renumbered and with a forced overflow. Remote worker: `gpu-probe/worker.mjs`.
- **Incremental execution with its real scheduling cost:** a resting world costs 0.11–0.13 ms per substep at any size (0.85–1.4 % of full stepping at 524k bodies), the wake of a kick 1.8–2.5 %, nothing extra when everything moves.
- **The rest quantum is a force threshold (0.9 % of a body's weight):** large rearrangements do not notice; the "small responses suppressed 3–20x" of this phase was mismeasured and is withdrawn in 1g. Routing of momentum to supports settles stacks 60x faster but changes consequences; not adopted.

**Phase 1d/1e: bodies at rest on the GPU, and the price of interfaces** — [`docs/phase1/bodies-and-interfaces.md`](docs/phase1/bodies-and-interfaces.md).
- **Rest as an exact integer fixed point:** a contact solver with stored impulses (JS specification and WGSL, bit-identical) brings a 22k-sphere pile to zero state changes; skipping unchanged neighbourhoods is then bit-exact. Cost 18 ns per sphere-substep at 125k; tall stacks and deep piles settle slowly (layers^2 / iterations); a 1 m/s kick still wakes half of a 22k pile.
- **Collars only at the impulsive port:** members sit directly on supports and on each other; with a rigorous screen for the interior damage check the eight-span beam costs 0.081 of the full run (was 0.45). Not robust at two and four spans.

**Phase 1c: members with ports, end to end** — [`docs/phase1/members.md`](docs/phase1/members.md).
- **One stepper for every level of detail:** rigid fragments, modal members and single cells; with every cell single it equals the Phase 0d reference to ten digits.
- **Result:** consequences mostly within 10 % of the reference (threshold within 1.4 %), cost 0.43–0.51 of the full run on one span and 0.45 on eight; not a pass by the pre-registered cost criterion.
- **Learned:** loaded interfaces need real cells; escalate on localisation, not onset; after fracture use the damaged basis with frozen damage; mode count must follow member size.

**Phase 1b: what state is sufficient, and what a contact pipeline costs** — [`docs/phase1/sufficiency.md`](docs/phase1/sufficiency.md).
- **Reference:** a 1000× change of contact stiffness moves the fracture threshold by at most 6 %, but changes the dissipated energy by up to 60 %.
- **State:** about 20 modal coordinates after an impact, 6–11 while a crack runs, rigid fragments afterwards — only in the basis of the current damaged operator; a basis fixed in advance fails.
- **GPU:** an integer sphere-contact pipeline with an atomics-built grid matches the CPU bit for bit at 4.7–6.5 ns per sphere-substep; memory layout changes that by up to 6×; tall stacks are the open problem.
- **Probe for other devices:** https://nanohit.me/shatterbird/ (`gpu-probe/det.html`, `gpu-probe/spheres.js`, `sh gpu-probe/publish.sh`).

**Phase 1: Shatterbird as a deterministic runtime** — [`docs/phase1/runtime-reframing.md`](docs/phase1/runtime-reframing.md).
- **Question:** what has to be computed at all, and what determinism costs, before asking how to compute an elastic network cheaply.
- **Measured on one GPU:** integer kernels are bit-identical to their reference and cost the same as f32 when memory-bound; bytes, not arithmetic, set the tick budget.
- **Mechanics:** wave-free predictors reproduce the explicit support-removal threshold within 5 % and the impact threshold 13–19 % low.
- **Code:** `gpu-probe/det.html` (portable probe), `cargo run --release -p experiments --bin phase1 -- wavefree`. Outputs go to `results/phase1/`.

**Phase 0d: adaptive engine for fracture without computing every internal oscillation** — [`docs/phase0d/adaptive-engine.md`](docs/phase0d/adaptive-engine.md).
- **What it is:** one implicit Newton system over coarse patches (multiscale P2 bases) and single-cell patches. Cells are split where damage, stress or impact demands it and merged back when quiet, with the projection loss booked.
- **Time integration:** variable-step BDF2, with the step set by the local error of fracture- and contact-relevant openings. A cost switch runs the whole model explicitly whenever that is cheaper.
- **Results:** pre-registered support-removal and impact criteria, scaling and mass-failure experiments, plus the multirate explicit/implicit coupling and why it was set aside.
- **Code:** library `crates/experiments/src/engine/`, binary `cargo run --release --bin phase0d -- <s0a|s0b|s0c|s1|s2|s3|s3b|s4|s4b|s5|s1noise>`. Outputs go to `results/phase0d/`.

**Phase 0c:** reduced intact regions — notched beam, support removal, impact. See [`docs/phase0c/reduced-fracture.md`](docs/phase0c/reduced-fracture.md).

**Current continuation:** [Phase 0b context and assignment 1](shatterbird-phase0b.md) — geometry contract and spectrum. It records the post-Phase-0 corrections and supersedes the old “E1 only” handoff.

**Phase 0b / 1 result:** [`docs/phase0b/geometry-spectrum.md`](docs/phase0b/geometry-spectrum.md). It contains the theory, the pre-registration (written before any comparison), the results and the answers to §4.5. Run with `cargo run --release --bin geo_spectrum` (≈3 min), plot with `python3 analysis/plot_geo_spectrum.py`; outputs go to `results/phase0b/geometry-spectrum/`. Spectral tools: `crates/solvers/src/spectrum.rs`; linearised 3D model: `crates/model/src/rbsn3.rs`.

**Phase 0b / 2 result:** [`docs/phase0b/two-level.md`](docs/phase0b/two-level.md) — coarse correction inside the physical step: two-level solver, reliability fixes (E7/E9 invalid Newton steps, exact partial supports), routine-step closed loop. Run `cargo run --release --bin two_level` (≈2.5 min), plot `python3 analysis/plot_two_level.py`; library `crates/solvers/src/twolevel.rs`. Track B probe plan (not run): [`docs/phase0b/track-b.md`](docs/phase0b/track-b.md).

The sections below describe the completed Phase 0 program.

Phase 0 empirical program from `shatterbird-phase0-spec.md`, with background in `shatterbird-context.md`.

- **Start here:** [`docs/phase0-summary.md`](docs/phase0-summary.md) answers Q1–Q5 and evaluates the decision map.
- **Theory, closures, pre-registered predictions:** [`docs/00-theory-and-design.md`](docs/00-theory-and-design.md). Revisions made after runs are marked **[rev]**.
- **Per-experiment notes:** `docs/findings/E1.md` … `E10.md`.
- **Range report for fixed point / Flock IR:** [`docs/range-report.md`](docs/range-report.md).

## Layout

```
crates/geometry     RSA seeding, 2D/3D Voronoi by convex clipping, mass properties, deterministic RNG
crates/model        2D RBSN: two-point corotated facets, interface law SB-IL1, supports/plates, range log;
                    linearised 3D RBSN (rbsn3, Phase 0b)
crates/solvers      S0 (Newton + faer sparse Cholesky/LU, bordered CMOD control), S1 (compliant-ALM block
                    descent; S3 = penalty limit), S2 (static force reconstruction), PCG, variational integrators;
                    spectrum (Lanczos, bounds, Sylvester certificate; Phase 0b)
crates/experiments  one binary per experiment (e1…e10, ranges, geo_spectrum); provenance (source hash) in every output
analysis/           matplotlib scripts → docs/figures/
results/            generated outputs (JSON + CSV per experiment)
```

## Reproduce

```bash
cargo test --release -p geometry -p model -p solvers
```

```bash
for e in e1 e2 e4 e5 e8 e9 e10 ranges; do cargo run --release --bin $e; done
```

```bash
cargo run --release --bin e3   # ≈1.5 h (slow local methods are the point)
```

```bash
cargo run --release --bin e6   # ≈40 min (finest resolution dominates)
```

```bash
cargo run --release --bin e7   # ≈10 min; SB_E7_DIAG=1 for the stability diagnostic
```

```bash
cd analysis && for p in plot_e*.py; do python3 $p; done
```

Every `results/*/summary.json` records the experiment, run, source hash (FNV-1a over all `.rs`/`.toml` files; the tree is not under version control), law version, seed, solver schedule and config.

## Scope

CPU f64 only. No fixed point, GPU, determinism, networking or rendering. Work counts (sweeps, point evaluations, factorization nnz/flops) stand in for cost; CPU wall time is not evidence of GPU speed. Track B (WebGPU device microbenchmarks, spec §8) is separate and not part of this program.
