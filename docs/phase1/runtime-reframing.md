# Phase 1 — Shatterbird as a deterministic runtime: cost model, determinism tax, minimal world state

*2026-11-11. Status tags: **[measured]** on one device (Apple M3 Pro, Chrome 157 / Dawn / Metal),
**[derived]** from measured coefficients, **[proposal]** not yet tested. Post-hoc changes are marked.*

## 0. What changed

Phases 0–0g asked how to compute an elastic RBSN network cheaply enough. This document asks
the prior question: what has to be computed at all. The target is restated as

> a deterministic runtime for large interacting worlds on WebGPU: bit-identical state across
> supported GPUs, at a cost close to what the device allows, growing with current activity
> rather than world size. Fracture is the hardest workload, not the definition of the runtime.

Three experiments were run for it. None of them tunes the Phase 0d engine.

| | question | result |
|---|---|---|
| D | What does determinism cost, per kernel class? | Nothing when the kernel is memory-bound; 1.6–2.1× when compute-bound with native-width integer products; 4.4–6.3× with generic 64-bit products. |
| L | What limits a tick on this device? | Bytes. 119–134 GB/s streamed, about 75 GB/s with gathers, 1.3–2.2 µs per dispatch. |
| M | Do the reference fracture thresholds need resolved waves? | Support removal: no (within 5 %). Impact: a two-coordinate model is 13–19 % low and does not depend on contact stiffness. |

## 1. Cost model [measured coefficients, derived budgets]

A tick costs approximately

    T = bytes_touched / BW + evaluations × c_op + dispatches × c_d

| coefficient | value on M3 Pro | source |
|---|---|---|
| BW, streaming copy (read + write) | 119–134 GB/s | probe §3, 64 MB buffers |
| BW, block automaton | 73–117 GB/s at 16.7 M cells, up to 168 GB/s at 1 M | probe §5 |
| BW, pair kernel with two gathers | 75 GB/s (112 B in 1.50 ns) | probe §2, repetition 1 |
| c_op, one pair-kernel evaluation, f32 | 0.033–0.048 ns | slope of time against repetitions |
| c_op, same kernel, native i32 | 0.070–0.076 ns | same |
| c_op, same kernel, 64-bit products | 0.207–0.209 ns | same |
| c_d, dispatch | 1.3–1.6 µs in one pass, 1.9–2.2 µs in separate passes | 2 000 small dispatches |

The GPU carried 54 % foreign load during the runs, so each timing is good to about ±25 %.

Consequences:

- **Arithmetic is nearly free; bytes are the cost.** One evaluation of the pair kernel (about
  25 multiplications, a square root, two divisions) costs 3 % of the time needed to move its
  112 bytes. Compute overtakes memory only after about 30 f32 or 20 i32 evaluations per element.
- **Dependent stages are cheap on this device.** A thousand dispatches cost about 2 ms. The
  number of solver colours or iterations is limited by the bytes each sweep touches, not by
  dispatch count. This must be re-measured on phones.
- **A tick is a byte budget.** A quarter of a 60 Hz frame is 4.2 ms, or about 0.5 GB at 125 GB/s.

What blocks 100 k and 1 M elements [derived]:

| model | bytes per element per frame | elements in a whole 60 Hz frame |
|---|---|---|
| explicit 3D RBSN: 1.9 KB per cell-substep (Phase 0b/3) × 776 substeps | 1.47 MB | about 1 400 |
| wave-free tiers of §3, about 0.5–1 KB per *active* element per tick | 0.5–1 KB | 2–4 M active |

- 776 substeps per frame is the Phase 0b/3 contract step (d_min = 0.2 m, h = 21.5 µs); the 25 mm 2D beams of Phase 0d need about 2 450.
- The explicit row agrees with the direct 0b/3 measurement (1 k cells, 21.9 µs per substep, 17 ms per frame).
- The gap to 100 k cells is two orders of magnitude and sits in *steps × bytes*, not in the kernel.
  No arithmetic optimisation closes it. Phases 0d–0g confirm this from the other side: multirate
  coupling, ports, the causal carrier and rigid freezing returned at most 1.1–1.4× on impacts.
- Removing the wave time scale (steps 776 → 1–8), packing state (1.9 KB → about 100 B per pass)
  and touching only active elements are each worth one to two orders.

## 2. Determinism tax [measured]

**Kernel.** A 2D bonded-facet interaction: relative displacement and rotation of two cells,
projection on the facet frame, equivalent opening with a square root, history maximum,
softening law with two divisions, forces and torques. Inputs are integers; 48 B per facet,
two 16 B cell gathers, 32 B output.

**Variants.** All compute from the same integer state.
- `N`: integer specification; every product fits 32 bits by range design. A product of a wide
  and a narrow value is split into two native multiplies. The square root is a float guess
  corrected by integer comparison, so the result is the exact floor.
- `W`: the same specification written with a generic 32×32→64 multiply for every product
  (the style of the Phase 0b fixed-point prototype). Bit-identical to `N`.
- `F`, `Ffma`, `Falt`: f32 versions of the real-valued function in three evaluation orders that
  WGSL permits (plain, explicit `fma`, reassociated).

**Bitwise results.**
- `N` and `W` equal the JavaScript reference on every word (65 536 facets, 1 and 3 repetitions;
  hashes `99b34cc3`, `2ae7e6fc`). The block automaton equals its reference (`5065ab0a`).
- Float outputs differ between legal evaluation orders in 4.4 % of elements (`fma`) and
  57 % (reassociation); with three accumulated repetitions, 10 % and 73 %. This census is
  synthetic: it shows what a conforming compiler may do, not what two vendors did.

**Timing** (ns per facet per dispatch, 2²⁰ facets, `r` evaluations per facet):

| r | F | N | W | N / F | W / F |
|---|---|---|---|---|---|
| 1 | 1.50 | 1.51 | 1.49 | 1.01 | 0.99 |
| 8 | 1.51 | 1.50 | 1.75 | 1.00 | 1.16 |
| 32 | 1.72 | 2.52 | 6.74 | 1.47 | 3.93 |
| 128 | 6.14 | 9.78 | 26.66 | 1.59 | 4.34 |

Compute-bound slope ratios across four runs: `N` 1.6–2.1, `W` 4.4–6.3.

**Reading.**
- The tax is not a constant. It is 1 + (compute share) × (c_spec − 1), where c_spec is set by
  how the integer specification is written, not by the fact that it is integer.
- The 1.9–8.3× of Phase 0b/3 was the price of 64-bit products at a small, compute-bound size.
  At the kernel's natural intensity (one evaluation per 112 B) the tax measured here is 1.00.
- Rules that follow: state, differences and sums are integers; products are kept within 31 bits
  by range analysis; a float may propose a value (square root, quotient) only if integers verify it.
- A float fast path guarded by a rounding-margin test ("filter") was analysed and not built.
  It cannot shortcut a specification that rounds intermediate values: the real-valued function
  differs from `N` by up to 18 000 force units here. It remains a candidate for kernels whose
  specification is a single final rounding of a smooth function (rotations, normalisation).
- Flock IR's first job is therefore range analysis: prove every product fits, choose shifts,
  and emit the WGSL kernel and its reference from one source.

**Block automaton.** A Margolus 2×2 rule (sand, water, wall) with an integer hash is race-free
and conserves cells by construction. 16.7 M cells update in 1.15–1.85 ms; 1 M in 0.05–0.075 ms.
A cell world of this size is not limited by determinism or by throughput on this device.

## 3. Minimal world state [proposal]

The Phase 0 record and experiment M support treating elastic displacement as a derived
quantity, not as state.

| tier | state | update | game class it carries |
|---|---|---|---|
| T0 cells | material id per cell | integer block automaton, active tiles only | falling-sand worlds |
| T1 aggregates | pose, velocity, a few generalised coordinates per connected body; contacts | impulse solver, 1–8 substeps per tick | machines, joints, rubble |
| T2 bonds | damage history per bond | stress recovered from the aggregate's loads by static solves; fracture as an event | structural destruction with load paths and fracture energy |
| T3 resolved waves | full RBSN | explicit, microsecond steps | offline oracle; optional local zone |

- Activation, substep counts and tier changes are functions of canonical integer state only.
- T3 stays as the reference that T2 is tested against. It is not the base representation.
- Stress enters T2 in two ways that M tests: a pseudo-static energy balance for sudden load
  changes, and an impulse onto generalised coordinates for impacts.

## 4. Experiment M — wave-free predictors against the explicit reference

Pre-registration: [`prereg-wavefree.md`](prereg-wavefree.md), written before the first run.
Scenes, seeds and failure criteria are those of Phase 0d S1 and S2. No constant is fitted.

**Support removal** (propped cantilever 4 × 0.4 m, λ scales gravity):

| | seed 1 | seed 2 | predicted |
|---|---|---|---|
| explicit reference λ_crit | 0.920 | 0.905 | |
| static capacity only | 1.903 (+107 %) | 1.903 (+110 %) | +60–100 % |
| dynamic factor 2 on the removed reaction | 0.889 (−3.4 %) | 0.947 (+4.7 %) | ±15 % |
| pseudo-static energy balance | 0.921 (+0.1 %) | 0.947 (+4.7 %) | ±10 % |

- The two dynamic predictors pass. They need one static load path and no time integration.
- "Static only" fails by slightly more than predicted: a plain static check would let a
  structure stand at twice the load that breaks it after a sudden support loss.
- Bisection resolution is 0.5 % (static) and 0.25 % (others); both seeds landing in the same
  static bin is within that resolution.

**Impact** (50 kg striker on a 1.8 m span; reference criterion: two fragments within 50 ms):

| | seed 1 | seed 2 | predicted |
|---|---|---|---|
| explicit reference v_crit [m/s] | 0.477 | 0.540 | |
| two coordinates (striker, beam with Rayleigh mass 81 kg and the static curve) | 0.413 (−13.5 %) | 0.436 (−19.3 %) | ±20 % |
| all striker energy into static work of fracture | 0.394 (−17 %) | 0.423 (−22 %) | lower bound |
| perfectly plastic impact (Cox) | 0.638 (+34 %) | 0.686 (+27 %) | upper bound |
| static work of fracture | 3.88 J | 4.48 J | about 5 J |
| two-coordinate v_crit with contact stiffness ×0.01 … ×10 | 0.415 … 0.413 | 0.439 … 0.436 | not registered |

- The two-coordinate model is inside the registered ±20 % and outside the 10 % game tolerance.
  Its bias is one-sided: it breaks the beam too early.
- At the reference threshold 61–68 % of the striker's energy equals the static work of
  fracture. The rest goes to motion the single coordinate cannot hold (higher modes, fragment
  and striker kinetic energy). More coordinates are the obvious next test; it was not run.
- **[post-hoc]** The contact-stiffness sweep was added after the first result. Over three
  decades the threshold moves by 0.5–0.8 %. In this regime the microsecond contact pulse
  carries no information about the outcome; only the exchanged impulse does. This is the
  time scale that forced Phase 0d into explicit windows.
- **[correction]** The first run assumed contact at zero displacement; the striker starts
  26 µm above the sagged beam, which gave an undefined mass and no result. `wavefree.json`
  keeps that run's support-removal numbers; the impact numbers are in `wavefree_impact.json`.

**Limits of M.**
- Two seeds, one geometry per event class, and both are beams. The predictors used 148–168
  static solves of the full fine model; they are a test of what information is needed, not an engine.
- Load control cannot follow the softening branch, so the energy-balance predictor is conservative.
- Fragment velocities, crack position and dissipated energy were not compared.

## 5. What would refute this direction

| claim | cheapest refuting observation |
|---|---|
| integer kernels are bit-identical across vendors | any hash other than `99b34cc3`, `2ae7e6fc`, `5065ab0a` from `gpu-probe/det.html` on another GPU |
| the tax is about 1 at natural kernel intensity | N / F above 1.3 at repetition 1 on a phone or a discrete GPU |
| bytes set the budget | dispatch cost above 50 µs, or pair-kernel throughput below 10 GB/s, on a target device |
| wave-free tiers carry structural fracture | a T2 model with up to 8 coordinates per member more than 30 % off the explicit threshold on a non-beam scene (frame, arch, wall with a projectile) |
| 100 k active bodies fit | an integer contact solver needing more than about 100 B × 16 sweeps per contact for a stable pile |

Estimates not yet measured [derived]:
- Contacts: about 100 B per contact-sweep at 75 GB/s is 1.5 ns. 100 k bodies × 3 contacts × 16
  sweeps is 7 ms per frame on this device; a phone with a fifth of the bandwidth carries about 20 k.
- Wave-free solids: 0.5–1 KB per active element per tick gives 0.5–1 M active elements in a
  quarter frame here and 100–200 k on such a phone.

## 6. Not verified

- One GPU, one browser, one WGSL compiler. Firefox 127 and Safari on macOS 14.7 have no WebGPU
  enabled here, and Chrome on macOS has no software adapter, so no second stack was available.
- The float divergence is between hand-written legal orders, not between vendors.
- Timings were taken under 54 % foreign GPU load.
- The game classes in §3 are assigned from general knowledge of those games, not from captured workloads.
- T1 and T2 do not exist as engines. Activity-proportional scheduling was not measured.

## 7. Next experiments [proposal]

1. Run `det.html` on the iPhone 16 Pro, the GTX 1660 Super, a Radeon and an older Android.
   It needs an https origin. Pass: identical hashes; record BW, c_d and the tax per device.
2. T1 on GPU: integer rigid bodies with contacts, 10 k–100 k bodies. Pass: bit-identical
   across devices, a resting pile stays at rest, cost within 2× of the byte estimate.
3. T2 against the explicit oracle on scenes that are not beams. Pass: thresholds within 15 %,
   cost proportional to active aggregates. Kill: the 30 % line of §5.
4. T0 with T1: rigid aggregates made of cells in the block automaton, with tile sleeping.

## 8. Reproduce

    node gpu-probe/run-det.mjs <tag>          # headless Chrome, isolated profile -> results/phase1/det_<tag>.json
    python3 gpu-probe/serve.py 8771           # then open http://127.0.0.1:8771/det.html and press Run
    cargo run --release -p experiments --bin phase1 -- wavefree   # support removal + impact (8 min)
    cargo run --release -p experiments --bin phase1 -- impact     # impact only (15 s)

Code: `gpu-probe/det.js`, `crates/experiments/src/bin/phase1.rs`. Data: `results/phase1/`.
