# Phase 1e pre-registration: the price of interfaces

Written before any run (2026-11-11). Follows `members.md`: the cost floor of the member/port stepper
was the collars of single cells kept around every port (23–34 % of all cells), not the fracture.

**Hypothesis.** A collar is needed only where a port is loaded impulsively (the striker contact).
Where a port is loaded slowly (supports, cuts between members) the missing local compliance of a
truncated free-free basis can be supplied by residual-attachment shapes: the static response of the
member to resultant loads on each port cluster, orthogonalised against the kept modes and
diagonalised, then integrated like any other mode. Members can then touch supports and each other
directly, and a long structure can be cut into size-bounded members without collars.

**Implementation.** `SB_ATT=a`: per port cluster, loads Fx, Fy and torque times s^0..s^(a-1) along the
cluster's long axis (3a shapes). `SB_COLLAR_S`: collar depth at supports (0 = none). `SB_CUT=1`:
groups are cut at interior supports; the cut facets are ordinary ports with the full nonlinear law.
Everything else as in v3 (k = 40, contact collar 4, T_q = 2 ms, localisation trigger 0.3).

| id | scene | pass | kill |
|---|---|---|---|
| E1 | one span, seeds 1 and 2, thresholds and 2 v_crit | thresholds within 3 %, D within 10 %, fragments equal, flops ratio < 0.30 (v3: 0.43–0.51) | D error > 20 % or ratio > 0.45 |
| E2 | eight spans, v0 = 0.6076, members cut at interior supports | D within 10 %, flops ratio < 0.15 | ratio > 0.30 or D error > 20 % |
| control | same with `SB_ATT=0` | shows what the shapes buy | — |
