# Shatterbird — Phase 0 (mechanics and solver truth, CPU f64)

**Phase 1 (current): Shatterbird as a deterministic runtime** — [`docs/phase1/runtime-reframing.md`](docs/phase1/runtime-reframing.md).
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
