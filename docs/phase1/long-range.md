# Phase 1h: carrying load further per iteration, fully parallel

Pre-registration: [`prereg-1h.md`](prereg-1h.md), including the decisions taken on the tuning set before
the evaluation. Previous phase: [`rest-physics.md`](rest-physics.md) (1g).
Code: `gpu-probe/rest2.js` (parameters `cheb`, `chebs`, `chebg`, `chebn`; lab-only `fstate`), `gpu-probe/lab3.cjs`,
`gpu-probe/xdev2.js` (new test `trace`). Results: `results/phase1/h/`, `results/phase1/xdev4/`.

## 1. Question

1g located the remaining physics error in the number of Jacobi iterations per substep. At 4 iterations
a pile is too soft, and a column needs ~L²/iterations substeps to settle. Multigrid is excluded by
decision, and sequential passes over the support order broke piles (1g section 6). Can a method that
keeps every contact independent within an iteration carry load further per iteration, without extra
memory, without losing exact rest, determinism or the cheap incremental execution?

## 2. Method

Chebyshev semi-iterative acceleration of the impulse update (Wang 2015):
P_{k+1} = Π(P_{k−1} + ω_{k+1}(P̂_{k+1} − P_{k−1})). Here P̂ is the plain projected Jacobi update and ω
follows the Chebyshev recurrence for a spectral radius ρ, in 1/256. P_{k−1} is what the ping-pong
destination copy still holds before it is overwritten, so no memory is added.

What the measurements forced, in order (details in the pre-registration):

1. **Gate.** Unrestricted acceleration makes integer limit cycles near the fixed point: 7 of 36 events
   never rest at 8 iterations, 35 of 36 at 32. A contact is now accelerated only if one of its bodies has
   a residual above the rest threshold. The rule is symmetric in the pair, so impulses stay antisymmetric.
2. **Normal part only.** Accelerating the tangential impulse pushes it past the static cone; the contact
   then slides at kinetic friction, and a pile under a sideways load creeps more than with plain Jacobi
   at the same iteration count. Only the normal part of the correction is kept.
3. **Two exactness rules for the GPU.** (a) The change-driven skip of 1g is neutral only if "changed"
   also covers a body whose *unaccelerated* update moves; with acceleration an update can stand still
   while the body is not stationary. (b) A value the acceleration leaves unchanged is not reprojected,
   because the integer cone projection is not idempotent on the static boundary. Found by a new
   substep-by-substep trace (`xdev2` test `trace`): the first difference was ±1 unit of one impulse at substep 121.

Frozen for the evaluation (law **v4** = v3 + this): ρ = 1 − 0.4/N (0.95, 0.975, 0.9875 for N = 8, 16, 32),
gate at 32 units, normal part only.

## 3. Evaluation: five piles never used for tuning

Piles 12–16 (1728 spheres each), every 1g event at equal horizons from one exact equilibrium, with controls.
Reference: the same contact law without acceleration, 128 iterations, fine cone, no quantum (`i128c1`).
Error per observable: |log((v + d)/(r + d))| (d = 1 for counts, 0.001 D for shifts), averaged over the
piles; per class, the mean over its observables (`verdict_vs_i128.json`, script `h_verdict.py`).

| variant | kicks | slump | tilt | support removal | events never at exact rest (of 180) |
|---|---|---|---|---|---|
| v3, 4 it | 0.571 | 0.299 | 1.155 | 0.320 | 1 |
| v3, 8 it | 0.421 | 0.280 | 0.746 | 0.274 | 1 |
| v3, 16 it | 0.275 | 0.154 | 0.469 | 0.294 | 0 |
| v3, 32 it | 0.181 | 0.119 | 0.262 | 0.122 | 0 |
| v3, 64 it | 0.120 | 0.064 | 0.137 | 0.067 | 0 |
| **v4, 8 it** | 0.270 | 0.198 | 0.604 | 0.270 | 0 |
| **v4, 16 it** | 0.131 | 0.119 | 0.347 | 0.107 | 0 |
| **v4, 32 it** | 0.084 | 0.085 | 0.202 | 0.198 | 0 |

Kicks are counted as bodies moved by more than 0.01 D and 0.1 D; slump as the centre-of-mass shift and the counts above 0.1 D and 1 D;
tilt as the centre-of-mass shift and the count above 0.1 D, at tan θ = 0.2 … 0.5; removal as the counts after a buried sphere is removed.

* **Kicks:** v4 at N iterations is as good as v3 at 2N or better (0.270 vs 0.275, 0.131 vs 0.181, 0.084 vs 0.120).
  After a 4 m/s kick, the bodies that move by more than 1 mm: v3 at 32 iterations 25–45 per pile, v4 at 16
  iterations 19–42, reference 8–38.
* **Tilt (a sustained frictional load):** between N and 2N, never at 2N.
* **Slump:** between N and 2N at 8 iterations, equal to 2N at 16.
* **Support removal:** at 2N for 8 and 16 iterations. At 32 it is worse than plain 32, from one extra local
  collapse on pile 12 (bodies > 0.1 D: 10 against 6; 64 and 128 iterations agree on 6).
  Whether an arch holds after a support goes is a threshold consequence. Softer solvers cross it,
  and v4 at 32 crosses it here like plain v3 at 8 and 16 does.

## 4. Columns and stacks

Cold columns, substeps to exact rest (`columns_stacks_tuning.json`):

| layers | v3 8 it | v3 16 it | v3 32 it | v4 8 it | v4 16 it | v4 32 it |
|---|---|---|---|---|---|---|
| 8 | 111 | 57 | 29 | 39 | 15 | 7 |
| 16 | 391 | 196 | 100 | 140 | 52 | 21 |
| 32 | 1383 | 687 | 346 | 509 | 183 | 74 |
| 64 | 5011 | 2415 | 1177 | 1847 | 667 | 246 |

v4 settles columns 2.7–4.8 times faster, but the law stays ~L^1.8: Chebyshev lowers the constant, not the
exponent (pre-registered target: below L^1.5).

Stacks under a sudden tilt still shear at tan θ = 0.45 (static limit 0.5) for 8 spheres and more at every
setting tried, except one ungated configuration at 32 iterations that holds 8.

A further hypothesis, tested and rejected (`fstate`, JS only): the static or kinetic friction regime chosen
once per substep from the relative velocity at its start, instead of at every iteration. A single body on a
slope then accelerates within 0.1–0.6 % of g(sin θ − μ_k cos θ), better than before. Columns do not improve,
and one gets worse (32 spheres at tan 0.3, 16 iterations). The cause is not the switch inside the iterations.
The first substep after the tilt does not converge over the height, the base acquires a slip velocity, and a
load between μ_k = 0.375 and μ_s = 0.5 then keeps it sliding under any reasonable law.

That makes the stack test itself questionable. "The column must hold" is the answer of a rigid model. An
elastic column under a step load overshoots its static load (up to twice) and can slip too. As with the
kick of 1g, the answer is not defined until the model has a physical compliance.

## 5. Exact rest, and a trap in the rest rule of v3

On the evaluation piles v4 brings every one of 540 events to exact rest. Plain v3 brings 898 of 900; the two misses
are pile 14, tilt 0.5, at 4 and 8 iterations. On the GPU, the 524 288-body cloud at 32 accelerated iterations never comes to rest on
either device (same hashes), while 8 and 16 do. The `stuck` test plus a JS replay of the dumped state
(`trace4.cjs`) show the mechanism; it belongs to the rest rule of v3, and acceleration only reaches it more often:

* a body j resting below, with a residual (−30, 19, 23): below the threshold, so the end of every substep
  sets it to rest (an anchor);
* a body i on top of it with a residual (−33, 22, 26): just above the threshold, so it is not set to rest;
* during the iterations the pair has a relative residual of (−3, 3, 3): the anchor that the end of the
  substep will apply to j is invisible to them. Divided by the Jacobi denominator and truncated, the
  correction is zero, and i slides over the stationary j at 33 units per substep forever.

Lab option `reff` (JS only, not adopted): a supported body whose residual is below the threshold counts as at
rest in the relative velocity of its contacts, which is exactly what the end of the substep will make it. The
rule is symmetric in the pair. With it the dumped 524k state comes to exact rest in 4 substeps, and so does
pile 14 at 4 iterations. Pile 14 at 8 iterations does not, because it is a different case: a body sliding over
the floor at 688 units (20 times the threshold) under a tilt equal to the static coefficient. That is flow at the
friction limit, and the reference does not rest there either. Adopting `reff` changes the law of v3, so it needs the
full cycle (ensembles, both GPUs), not done here.

## 6. Cost

Body evaluations of change-driven iterations per wake, mean over 30 kicks (`events_and_cost_eval.json`):

| | down 0.25 | down 1 | down 4 | side 1 |
|---|---|---|---|---|
| v3 16 / 32 / 64 it | 0.16 / 0.15 / 0.14 M | 0.44 / 0.40 / 0.40 M | 0.89 / 0.89 / 1.00 M | 0.55 / 0.55 / 0.60 M |
| v4 8 / 16 / 32 it | 0.16 / 0.13 / 0.12 M | 0.41 / 0.35 / 0.31 M | 0.71 / 0.64 / 0.65 M | 0.50 / 0.50 / 0.50 M |

A wake costs v4 at N fewer evaluations than v3 at 2N, and it reaches exact rest about as soon (median after a
4 m/s kick: 57 / 32 / 28 substeps against 48 / 35 / 28).

GPU time, timestamp queries, ms per substep; wake = total ms of the wake of a 1 m/s kick (exact rest after
how many substeps). The same session, 10 substeps per command buffer (`results/phase1/xdev4/time_*.json`).

| bodies | device | | v3 8 | v3 16 | v3 32 | v4 8 | v4 16 | v4 32 |
|---|---|---|---|---|---|---|---|---|
| 204 800 | M3 Pro | all moving | 2.44 | 3.70 | 5.57 | 2.63 | 4.29 | 7.39 |
| | | at rest | 0.153 | 0.154 | 0.183 | 0.125 | 0.152 | 0.183 |
| | | wake | 18.7 (60) | 18.6 (60) | 27.7 (40) | 16.1 (60) | 23.2 (60) | 21.4 (40) |
| 204 800 | GTX 1660 SUPER | all moving | 1.67 | 2.35 | 3.48 | 2.06 | 3.34 | 5.44 |
| | | at rest | 0.149 | 0.196 | 0.293 | 0.149 | 0.195 | 0.291 |
| | | wake | 19.0 (60) | 24.3 (60) | 49.3 (40) | 16.1 (60) | 30.0 (60) | 36.3 (40) |
| 524 288 | M3 Pro | all moving | 5.72 | 8.78 | 14.45 | 6.18 | 10.00 | 17.27 |
| | | at rest | 0.100 | 0.176 | 0.240 | 0.100 | 0.153 | 0.734 |
| | | wake | 21.0 (80) | 20.2 (60) | 44.0 (60) | 15.2 (60) | 15.7 (40) | 413 (never) |
| 524 288 | GTX 1660 SUPER | all moving | 3.29 | 4.62 | 7.01 | 4.29 | 6.44 | 10.96 |
| | | at rest | 0.150 | 0.197 | 0.294 | 0.150 | 0.198 | 2.013 |
| | | wake | 22.6 (80) | 27.2 (60) | 80.8 (60) | 18.8 (60) | 18.5 (40) | 946 (never) |

* **An accelerated iteration costs more when everything moves:** +8–33 % on the M3 Pro, +23–56 % on the GTX.
  It reads the previous level of the impulses and projects twice. Against plain 2N, v4 at N costs 0.69–0.77 of
  it on the M3 Pro and 0.88–0.96 on the GTX.
* **Rest is not dearer, and wakes are cheaper at 8–16 iterations.** At 524k bodies and 16 iterations a wake
  costs 15.7 / 18.5 ms against 44.0 / 80.8 for plain 32.
* At 32 accelerated iterations on 524k bodies the trap of section 5 keeps one body awake, so rest and the
  wake never end (0.73 / 2.0 ms per substep).

## 7. Determinism

v4 at 8, 16, 32 iterations: GPU = JS on both devices, full and incremental (22 checkpoints), and at 16 iterations also
fuzz (1240 host edits, 3 configurations), permutation with 40 coincident pairs, forced overflow.
Hashes are equal across devices: 3e744ae3 / 0751c2f6 / 0ea0505f (8), d2dc3621 / 3ba85f3a / b9656385,
fuzz f3ce09f7, overflow 6e86d9d0 (16), 51a7b049 / 56a8306d / 62265962 (32). The timing runs give equal
hashes for every common configuration as well (`gpu_time_summary.json`). v2 and v3 signatures are unchanged.

## 8. A confound in the comparison, found and removed

At the end of 1g the integer projection of the equilibrium was replaced (breadth-first forest → maximum-load
forest). Positions are the same, but the initial impulse network, which is history, is not. The first 1h comparison on piles 12–13
mixed plain v3 runs from 1g (old network) with v4 and the reference (new network). One "never at rest" event of
plain v3 at 32 iterations (pile 12, tilt 0.5) came from that mismatch: rerun from the new network, it rests at
substep 379. All plain v3 runs on piles 11–13 were rerun; every number above comes from one network per pile.
1g itself is consistent: its variants and its reference started from the same, older network. The tuning
decisions of section 2 were taken against the older plain runs and the 64-iteration reference. Recomputed on
pile 11 against the 128-iteration reference and the rerun plain runs, the choice of normal-only over full
acceleration still holds (16 iterations: slump 0.016 against 0.104, tilt 0.52 against 0.63;
`verdict_tuning_pile11.json`).

Two corrections to 1g follow from this phase. The 1g statement that every pile comes to rest held for the
kicks; plain v3 at 32 iterations missed it on one tilt event (in the 1g data, unreported). And the rest rule
of v3 has the trap of section 5.

## 9. Verdict against the pre-registration

| id | verdict |
|---|---|
| H1 | **fails as stated; not killed at 8 and 16.** v4 at N is as good as v3 at 2N on kicks and support removal, but only between N and 2N on tilt (and on slump at N = 8). At N = 32 support removal is worse than plain 32 (kill for N = 32). |
| H2 | **killed.** The 8-column still shears at 32 iterations; the settling law stays ~L^1.8, though 2.7–4.8 times faster. |
| H3 | **mixed.** All 540 events on the evaluation piles reach exact rest (plain v3: 898 of 900), and controls do not move. No kick moves more than plain v3 at 4. But the 524k cloud at 32 iterations never rests (trap of section 5), and slump within 10 % of the reference is met by nothing below 64 iterations, v4 included. |
| H4 | **mixed.** Wake work at N is below plain 2N; JS = M3 Pro = GTX 1660 SUPER everywhere. GPU time per iteration when everything moves is +8–56 %, not within 10 %. |

## 10. What follows

* **What v4 buys:** roughly a factor 1.5–2 in iterations on the observables the model resolves, columns 3–5
  times faster, wakes that cost fewer evaluations, with no extra memory and with the bits unchanged across
  devices. Not a change of the scaling law. Recommended range 8–16 iterations, where no trap was seen.
* **The rest rule of v3 has a trap** at the threshold between "set to rest" and "moving" (section 5). `reff` closes it
  in both trap cases seen; it has to go through the full cycle before it becomes the law.
* **Two observables are undefined in a rigid model:** how far a small push moves a pile (1g) and whether a
  column survives a sudden load (here). Both need a physical compliance in the contact law. That is the next physics
  question, and it decides what "converged" even means for the solver.
* Not done: memory per body (unchanged), rotation, 1e, phones.
