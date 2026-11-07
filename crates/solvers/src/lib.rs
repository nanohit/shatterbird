//! Phase 0 solvers: S0 (reference Newton + sparse direct), S1 (compliant-ALM
//! block descent; S3 = penalty limit), S2 (static force reconstruction), PCG.

pub mod assembly;
pub mod cg;
pub mod integrator;
pub mod s0;
pub mod s1;
pub mod s2;
pub mod sparse;
pub mod spectrum;
pub mod twolevel;
