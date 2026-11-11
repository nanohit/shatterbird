# Phase 1 / M — pre-registration (written before the first run, 2026-11-11)

Question: do the fracture thresholds of the explicit wave-resolving RBSN reference need the
waves, or are they reproduced by a model whose only dynamic state is a few generalised
coordinates and whose stress comes from static solves?

Reference values (Phase 0d, explicit, unchanged scenes and criteria):
- S1 support removal, propped cantilever 4 x 0.4 m: lambda_crit = 0.9203 (seed 1), 0.9047 (seed 2).
- S2 impact, 50 kg striker on a 1.8 m span: v_crit = 0.477 m/s (seed 1), 0.540 m/s (seed 2).

Wave-free predictors (no fitted constants):
- M1a plain static capacity (gravity only, prop absent).
- M1b sudden removal = static analysis under gravity minus the prop reaction (dynamic factor 2 on the removed reaction).
- M1c pseudo-static energy balance: survive iff the static path reaches a state where the work of the
  suddenly applied unbalanced load equals the absorbed work.
- M2 two generalised coordinates: striker mass, unilateral contact spring, beam coordinate with the
  Rayleigh mass of the first static shape and the static force-deflection curve to failure (secant unloading).

Predictions:
- M1a over-predicts capacity by 60-100 % (so "static only" is refuted as a model of sudden loading).
- M1b within +-15 %, M1c within +-10 % of the reference lambda_crit.
- M2 within +-20 % of the reference v_crit; static work of fracture about 5 J.

Refutation: an error above 30 % for M1c or M2 means the wave-free tier cannot carry this class of
events and a resolved hot zone is required.
