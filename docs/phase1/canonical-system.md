# Phase 1f: one canonical contact system, full or incremental, on two GPUs

2026-11-12. Pre-registration: `prereg-1f.md`. Code: `gpu-probe/rest2.js` (specification v2 in JS, WGSL
port), `worker.mjs`, `xdev.js`, `xdev2.js`, `lab2.cjs`. Results: `results/phase1/xdev/*.json`,
`f_*.json`, `f_physics/`. Devices: Apple M3 Pro (Metal, Chrome 157) and NVIDIA GTX 1660 SUPER
(driver 610.88 / 32.0.16.1088, D3D12, Chrome 157, Windows 11), both headless with the hardware
adapter (the worker refuses a fallback adapter). 3D equal spheres without rotation.

## 1. Remote worker

`node gpu-probe/worker.mjs <local|win> <page> <function> <out.json> [opts]` runs a page function in
headless Chrome on this Mac or on the NVIDIA machine. For `win` the page is served from this checkout
through an SSH reverse tunnel and Chrome is driven through a forwarded debugging port, so the worker
needs nothing but Chrome. The result carries the adapter, the driver as the OS reports it, the Chrome
version; console output is echoed as progress; state dumps are POSTed back as binaries. Chrome on the
worker sometimes starts without the GPU right after a previous instance was killed; the runner waits
and relaunches until the hardware adapter is there. GPU time comes from timestamp queries.

## 2. The unchanged 1d solver on the second vendor

| check | M3 Pro | GTX 1660 SUPER |
|---|---|---|
| signature, 1728 x 400 substeps | 9ff33c42 | 9ff33c42 |
| signature, 8000 x 600 substeps | c82e2c76 | c82e2c76 |
| full state dumps, M3 against GTX | positions, velocities, contact sets, impulses identical word for word; the order of slots differs for 84 and 601 bodies |
| bodies stored in a permuted order | identical | identical |
| largest cell count / capacity; contacts / capacity | 4 / 8; 10 / 16 | same |
| GPU time, 125k bodies, per sphere-substep | 18.5 ns (match 4.9, solve 10.4, finish 2.4) | 37.4 ns (9.3, 25.2, 2.4) |

No overflow occurred in these scenes, which is why the race in the 8-slot cells never showed.

## 3. Specification v2: what changed and why

- **Nothing is dropped silently.** The grid is a linked list per cell (one atomic exchange per body, no
  capacity). If more than K = 16 contacts are in range the K deepest are kept, ties by position, and
  every dropped contact is counted. Impulse, velocity and position clamps are counted too. A forced
  overflow (20 bodies in one cell, 242 574 dropped contacts) gives the same state and the same counters
  in JS and on both GPUs, also with the bodies renumbered.
- **Every tie-break is independent of the numbering** (position order; the designated support of
  routing is published as a hash of its position).
- **The ledger became state.** A supported body whose velocity after the last iteration is within
  eps is set to rest, and the impulse that took (the anchor) is stored with the body (7 bits per axis
  in the word that holds the contact count) and recomputed every substep. The share of the weight that
  rests on the world frame instead of on contacts is then a property of the current state, identical
  for full and incremental stepping, and part of the state hash. The pre-registered 64-bit running sums
  on the GPU (F5) were dropped for this reason: a running sum differs between full and incremental
  stepping as long as resting bodies leak.
- **State in place.** Impulses live in the buffer the last iteration writes; residuals ping-pong.
  `quiet` means a strong fixed point (no impulse changed in any iteration), so both residual copies of
  a quiet body hold its rest value and a skipped body can be read by its neighbours at any level.
- **Routing** (option, off by default): a slow body resting on slow supports hands its residual to
  those supports instead of solving the contact by Jacobi. See section 6.
- With routing and the rest quantum off, v2 reproduces v1 word for word.

## 4. Incremental execution on the GPU

Per substep: a one-invocation pass reads the seed counters (cells touched by bodies that were not
quiet in the previous substep, written by the end-of-step pass into 64 striped lists) and decides
full or incremental; an exact Chebyshev distance transform over cells in three passes (x, y, z;
atomic min, each pass appends newly reached cells to a list); the grid is kept between substeps and
only the lists of the seed cells are rebuilt (every cell a body left or entered is a seed cell);
bodies of the reached cells are compacted into an active list; contact search for distance <= 1,
iteration k for distance <= k + 2, end of step for distance <= iterations + 1; marks are reset through
the list. All dispatches are indirect; list overflow or too many seeds switches that substep to full
(with a full grid rebuild). 25 dispatches per substep, all in one compute pass (a pass per stage
costs about 4x more at rest on Metal).

Identity: JS incremental = GPU full = GPU incremental at 22 checkpoints through falling, settling,
rest, a kick and its wake (4096 bodies), on both GPUs, including the fault counters; the number of
evaluated body-substeps on the GPU equals the JS count exactly (3 212 058), so the distance transform
is exact. A fuzz run (1240 random host edits on a settled pile: single kicks and bursts of 300, five
configurations including list capacities small enough to force the overflow path and a setting that
never switches to full) matches JS full stepping at all 25 checkpoints of every configuration on both
GPUs. At 32k, 205k and 524k bodies the hashes of full and incremental runs agree at rest and after the
wake, and the M3 Pro and GTX hashes agree with each other.

GPU time per substep, Jacobi variant (`f_gpu_time_summary.json`):

| bodies | device | all moving, full / incr. | at rest, full / incr. | wake of a 1 m/s kick, full / incr. |
|---|---|---|---|---|
| 524 288 | M3 Pro | 3.67 / 3.70 ms | 9.15 / 0.131 ms (1.4 %) | 1275 / 32.0 ms over 140 substeps (2.5 %) |
| 524 288 | GTX 1660 S | 2.70 / 2.70 ms | 14.73 / 0.126 ms (0.85 %) | 2062 / 36.9 ms (1.8 %) |
| 204 800 | M3 Pro | 1.53 / 1.54 ms | 3.56 / 0.111 ms (3.1 %) | 361 / 20.4 ms (5.6 %) |
| 204 800 | GTX 1660 S | 1.17 / 1.17 ms | 5.83 / 0.125 ms (2.1 %) | 583 / 22.9 ms (3.9 %) |
| 32 768 | M3 Pro | — | 0.66 / 0.113 ms (17 %) | 79 / 43.2 ms (55 %) |
| 32 768 | GTX 1660 S | 0.38 / 0.38 ms | 1.07 / 0.124 ms (12 %) | 128 / 60.4 ms (47 %) |

- **A resting world costs the same at any size**: 0.11–0.13 ms per substep on both GPUs for 32k,
  205k and 524k bodies. That is the fixed price of 25 dispatches; no term proportional to the number
  of bodies or cells is left. With a full grid rebuild every substep the same state cost 0.36 ms at
  524k on the M3 Pro (0.22 ms of it the rebuild).
- **Deciding costs nothing measurable when everything moves**: the substep switches to full.
- At 32k the kick wakes the whole pile (it is only 64 cells wide), so there is little to gain; at
  524k it wakes 0.5–0.7 % of the bodies.
- During settling the incremental run is not cheaper until bodies start to rest (ratios 1.0, then
  0.05–0.1 in the last few hundred substeps).

Memory: 663 B per body at 524k (596 B per body, the rest per cell; the grid has 3 cells per body
because of the head room). Pre-registered <= 600 B: failed. Packing a slot into 12 B would give about
535 B; not done.

## 5. Physics of the rest state and of the quantum

Reference: same contact law, 32 iterations, eps = 1. It never reaches an exact fixed point.
Three piles of 4096 per variant (`f_physics_summary.json`).

| variant | exact rest | anchors, % of weight | kick: sum of displacements, D (0.25 / 1 / 4 m/s) | kicked sphere at 4 m/s, D | slump: bodies moved > 1 D |
|---|---|---|---|---|---|
| reference | never | 0.007 | 0.23 / 0.41 / 1.46 | 0.23 | 110 |
| 4 iterations, eps 1 | never | 0.007 | 0.24 / 0.59 / 2.02 | 0.21 | 126 |
| 4 iterations, eps 8 | not within 6000 | 0.18 | 0.22 / 1.11 / 2.50 | 0.19 | 113 |
| 4 iterations, eps 16 | 790–850 | 0.27 | 0.08 / 0.62 / 2.13 | 0.26 | 131 |
| 4 iterations, eps 32 (default) | 700–880 | 0.45 | 0.08 / 0.37 / 1.64 | 0.07 | 109 |
| 32 iterations, eps 32 | 670–2700 | 0.48 | 0.01 / 0.07 / 0.43 | 0.03 | 113 |
| 4 iterations, coarse (eps 128) | 670 | 1.94 | 0.03 / 0.70 / 1.94 | 0.18 | 105 |
| routing, eps 32 | 750–830 | 0.02 | 0.13 / 0.39 / 3.20 | 0.10 | 173 |

- **The quantum is a force threshold.** eps per substep is an acceleration: 32 units = 0.9 % of the
  body's weight, 128 = 3.5 %. Imbalances below it produce no motion. Large rearrangements do not
  notice (the slump is within the scatter of the reference for every eps), small responses do: with a
  converged solver and eps 32 the displacement field of a kick is 3–20 times smaller than the
  reference, and the kicked sphere moves 0.03 D instead of 0.23 D. The observation "not one of 79 507
  bodies moved by 0.1 D" was this effect.
- At 4 iterations the under-converged solver partly hides it (errors of opposite sign), which is why
  the default looked closer to the reference than it deserves.
- Exact rest needs eps >= 16 at 4 iterations; at 8 some piles rest and some do not; at <= 4 none do.
  So exact sleep and sub-percent force resolution exclude each other in this solver.
- Anchors: at the default 0.45 % of the weight rests on the world frame (per axis, absolute sum;
  the net vertical part is about 0.01 %, the floor and walls carry 100.0 %).

Pre-registered tolerances: slump — bodies displaced and vertical shift of the centre of mass within
tolerance for every Jacobi variant, horizontal shift +26 % at the default (outside 10 %); the profile
metric turned out not to discriminate (different reference piles differ by as much). Kicks — sum of
displacements at 0.25 m/s outside tolerance for eps >= 16.

## 6. Routing: what it buys and why it is not adopted

Routing is shock propagation in Jacobi form: the upper body's whole residual goes to its supports
(exact integer split, remainder to one designated support), the reaction reaches the lower body one
iteration later, momentum is conserved exactly.

- Rest with almost nothing on the world frame: anchors 0.02 % of the weight (F1 passed).
- A cold 64-column rests in 244 substeps instead of 16 412, 16 layers in 8 instead of 1143. Not a
  clean linear law: 4, 8, 112, 244 substeps for 8, 16, 32, 64 layers and 9158 for 100; above 16
  layers the unsupported part falls freely until the support wave arrives and the overlap it leaves
  is pushed out slowly (F2 failed as stated: 244 > 200, no single exponent).
- Wakes are 2–5 times smaller.
- **It changes consequences** (F3 failed): in the slump 173 bodies move more than 1 D instead of
  110, the centre of mass shifts twice as far; a 4 m/s kick displaces twice as much. Switching routing
  on only after the pile has formed gives the same excess (181), switching it off restores the
  reference (90): the cause is the response, not the pile. Routing only the vertical part does not
  help (170); routing along the contact normal does not come to rest.

## 7. Where slow settling comes from

- **Single column, Jacobi:** time to exact rest 278, 1143, 4272, 16 412 substeps for 8, 16, 32, 64
  layers: exponent 1.90–2.04 over four doublings, and inversely proportional to the iteration count.
  The transient sag is linear in the height (0.35, 0.72, 1.44, 2.89 %). The tail is the positional
  push-out of that sag, a diffusion.
- **Piles are different.** Fixed footprint, depth 3.7 / 5.8 / 10.2 / 20.7 D: rest at 628 / 703 / 859 /
  1578 substeps (986 at 8 iterations). After subtracting the fall, the remainder is about 430
  substeps up to 10 D and 1000 at 20 D. During that time most bodies are being pushed out of the
  overlap the impact left (5288 of 6909 at substep 600 in the deepest pile), then a tail of a few
  hundred substeps in which only impulses change. So for piles the cost driver is the push-out of
  impact overlap, not a layers^2 law; "layers^2" is established for stacks only.
- **Four iterations are all used.** Among bodies that are not quiet, 98 % still change an impulse in
  the fourth iteration (`f_iterations_needed.json`); there is nothing to skip inside the active set.

## 8. Verdict against the pre-registration

| id | result |
|---|---|
| F1 routing, zero-flux rest | passed (0.02 %), but see F3 |
| F2 linear settling of columns | failed as stated |
| F3 routing keeps consequences | failed; routing is off by default |
| F4 no silent drops, canonical overflow | passed on both GPUs |
| F5 ledger on the GPU | replaced: the anchor is state, compared bit for bit as part of it |
| F6 incremental = full, both GPUs | passed at every checkpoint |
| F7 cost of scheduling plus work | passed: rest 0.85–1.4 %, wake 1.8–2.5 % of full at 524k; no overhead when everything moves; rest cost independent of size |
| F8 memory <= 600 B | failed (663 B) |

## 9. Open, not answered here

- **1e and this runtime are not yet two subsystems of one system.** The member stepper is CPU f64 with
  its own 2 µs step. What a fracture must hand over to make them one: when one object splits, each
  fragment needs its contacts re-attributed (a contact belongs to a pair of surface elements, not to
  the parent body), the stored impulse of a contact that survives must stay with it, and the damage
  history must stay with the facets (it already does in 1e). In the integer runtime this is a
  renumbering problem: the specification is now independent of numbering, so a split can be expressed
  as a host edit of the state followed by seeding the touched cells; this is untested.
- **0.081 on eight spans** says nothing about a fracture that propagates through many members.
- **Long-range dependencies.** Routing is the Jacobi form of shock propagation and fails on
  consequences. The two-pass scheme (Xiong et al. 2025: an upward pass that records the impulses the
  lower body could not take, a downward pass that applies them) needs layers and ordered passes; the
  distance-transform lists built here are the same machinery (layers of the support graph instead of
  cells), and a deterministic layer assignment is possible because every tie-break is already
  canonical. The multi-layer XPBD solver (Mercier-Aubin and Kry 2024) treats groups as rigid on coarse
  layers, which is the idea of the 1e members. Next experiment: two-pass propagation over support
  layers as an option of the same specification, judged by the same slump and kick observables; pass
  = consequences within tolerance and a 64-column at rest within 200 substeps.
- Rotation and rigid clumps, 12-byte slots, phones (the v2 pipeline needs 8 storage buffers per stage,
  the default limit), host edits that move bodies (they must seed the old and the new cell).
