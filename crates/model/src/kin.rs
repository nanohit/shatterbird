//! Corotated jump at one facet point and its Jacobian (docs §3.2).
//!
//! Δ = (x_b + R(θ_b) arm_b) − (x_a + R(θ_a) arm_a),  δ = R(−θ̄)Δ − δ_rest,
//! θ̄ = (1−w)θ_a + wθ_b: w = ½ between cells (mean frame), w = 1 against a support or
//! plate body (the support's own frame, so a roller stays a roller when the cell rotates).
//! Returned Jacobians are w.r.t. (x, y, θ) of body a and body b.

use geometry::V2;

#[derive(Clone, Copy, Debug)]
pub struct Jump {
    pub dn: f64,
    pub dt: f64,
    /// d(dn)/d(q_a), d(dt)/d(q_a), d(dn)/d(q_b), d(dt)/d(q_b)
    pub jna: [f64; 3],
    pub jta: [f64; 3],
    pub jnb: [f64; 3],
    pub jtb: [f64; 3],
}

#[inline]
pub fn jump(qa: [f64; 3], qb: [f64; 3], arm_a: V2, arm_b: V2, n0: V2, t0: V2, rest: V2, w: f64) -> Jump {
    let ra = arm_a.rot(qa[2]);
    let rb = arm_b.rot(qb[2]);
    let delta = V2::new(qb[0] + rb.x - qa[0] - ra.x, qb[1] + rb.y - qa[1] - ra.y);
    let th = (1.0 - w) * qa[2] + w * qb[2];
    let raw = delta.rot(-th);
    let d = raw - rest;
    // current facet frame
    let nc = n0.rot(th);
    let tc = t0.rot(th);
    // dδ/dθ_a = -(1-w) J raw - R(-θ̄) J R(θ_a) arm_a ;  R(-θ̄) J R(θ_a) a = J R(θ_a-θ̄) a
    let jr = raw.perp();
    let ga = (ra.rot(-th)).perp();
    let gb = (rb.rot(-th)).perp();
    let dth_a = -(jr * (1.0 - w)) - ga;
    let dth_b = -(jr * w) + gb;
    Jump {
        dn: n0.dot(d),
        dt: t0.dot(d),
        jna: [-nc.x, -nc.y, n0.dot(dth_a)],
        jta: [-tc.x, -tc.y, t0.dot(dth_a)],
        jnb: [nc.x, nc.y, n0.dot(dth_b)],
        jtb: [tc.x, tc.y, t0.dot(dth_b)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rigid_motion_gives_zero_jump() {
        let xa = V2::new(0.1, 0.2);
        let xb = V2::new(0.35, 0.25);
        let p = V2::new(0.22, 0.21);
        let n0 = (xb - xa).normalized();
        let t0 = n0.perp();
        for &ang in &[0.0, 0.7, 3.0, -2.5] {
            let c = V2::new(5.0, -3.0);
            let ya = c + xa.rot(ang);
            let yb = c + xb.rot(ang);
            let j = jump([ya.x, ya.y, ang], [yb.x, yb.y, ang], p - xa, p - xb, n0, t0, V2::ZERO, 0.5);
            assert!(j.dn.abs() < 1e-14 && j.dt.abs() < 1e-14, "{} {}", j.dn, j.dt);
        }
    }

    #[test]
    fn jacobian_matches_fd() {
        let xa = V2::new(0.1, 0.2);
        let xb = V2::new(0.35, 0.25);
        let p = V2::new(0.22, 0.21);
        let n0 = (xb - xa).normalized();
        let t0 = n0.perp();
        let qa = [0.11, 0.19, 0.3];
        let qb = [0.36, 0.27, 0.45];
        for w in [0.5, 1.0] {
        let j = jump(qa, qb, p - xa, p - xb, n0, t0, V2::ZERO, w);
        let h = 1e-7;
        for k in 0..3 {
            let mut a1 = qa;
            a1[k] += h;
            let mut a0 = qa;
            a0[k] -= h;
            let jp = jump(a1, qb, p - xa, p - xb, n0, t0, V2::ZERO, w);
            let jm = jump(a0, qb, p - xa, p - xb, n0, t0, V2::ZERO, w);
            assert!(((jp.dn - jm.dn) / (2.0 * h) - j.jna[k]).abs() < 1e-8);
            assert!(((jp.dt - jm.dt) / (2.0 * h) - j.jta[k]).abs() < 1e-8);
            let mut b1 = qb;
            b1[k] += h;
            let mut b0 = qb;
            b0[k] -= h;
            let jp = jump(qa, b1, p - xa, p - xb, n0, t0, V2::ZERO, w);
            let jm = jump(qa, b0, p - xa, p - xb, n0, t0, V2::ZERO, w);
            assert!(((jp.dn - jm.dn) / (2.0 * h) - j.jnb[k]).abs() < 1e-8);
            assert!(((jp.dt - jm.dt) / (2.0 * h) - j.jtb[k]).abs() < 1e-8);
        }
        }
    }
}
