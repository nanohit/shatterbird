//! Phase 0d adaptive multirate engine (docs/phase0d/adaptive-engine.md).
//!
//! Cells live either in closed **patches** (floating-frame bodies with a precomputed
//! multiscale basis, integrated implicitly at the macro step H) or in the **active
//! zone** (every cell free, velocity Verlet at the zone's own stable step). The two
//! are coupled by an impulse-matched multirate iteration; patches open when the
//! monitor or a contact prediction fires and close again when their material is quiet.

pub mod contact;
pub mod carrier;
pub mod patch;
pub mod rsys;
pub mod scenes;
pub mod world;

use geometry::V2;
use serde::Serialize;

pub const G: f64 = 9.81;
/// Flop model (as Phase 0c): one facet point with tangent / force only.
pub const C_PT_TANGENT: f64 = 200.0;
pub const C_PT_FORCE: f64 = 60.0;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Work {
    /// R-system facet points evaluated with / without tangent
    pub pts_tangent: u64,
    pub pts_force: u64,
    /// active-zone facet points (force only, every sub-step)
    pub e_pts: u64,
    pub e_body_flops: f64,
    pub iface_flops: f64,
    pub coupling_flops: f64,
    pub region_flops: f64,
    pub fact_flops: f64,
    pub solve_flops: f64,
    pub step_flops: f64,
    pub monitor_flops: f64,
    pub rebuild_flops: f64,
    pub contact_flops: f64,
    pub newton_iters: u64,
    pub factorizations: u64,
    pub e_substeps: u64,
    pub coupling_iters: u64,
    pub macro_steps: u64,
    pub rejected: u64,
    pub activations: u64,
    pub rereductions: u64,
    /// Dynamic solves stopped at the explicit |H|*ULP roundoff floor, rather than
    /// the static absolute force tolerance. These are reported, never hidden.
    pub roundoff_accepts: u64,
}

impl Work {
    pub fn total(&self) -> f64 {
        self.pts_tangent as f64 * C_PT_TANGENT
            + self.pts_force as f64 * C_PT_FORCE
            + self.e_pts as f64 * C_PT_FORCE
            + self.e_body_flops
            + self.iface_flops
            + self.coupling_flops
            + self.region_flops
            + self.fact_flops
            + self.solve_flops
            + self.step_flops
            + self.monitor_flops
            + self.rebuild_flops
            + self.contact_flops
    }
}

#[inline]
pub fn rot(v: V2, phi: f64) -> V2 {
    v.rot(phi)
}

#[inline]
pub fn jperp(v: V2) -> V2 {
    V2::new(-v.y, v.x)
}

/// Cosine release ramp: 1 before t = 0, 0 after t_r.
pub fn release(t: f64, t_r: f64) -> f64 {
    if t <= 0.0 {
        1.0
    } else if t >= t_r {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * t / t_r).cos())
    }
}
