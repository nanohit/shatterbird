//! S0 — reference solver: Newton on the incremental trial potential with sparse
//! direct solves (docs §4.1). History is never written here; callers commit.

use crate::assembly::{assemble, axpy_q, energy, DofMap, Inertia, Tangent};
use crate::sparse::{csc_from_triplets, lu_solve, Spd};
use model::Model;

#[derive(Clone, Copy, Debug)]
pub struct NewtonOpts {
    /// Absolute tolerance on ‖∇Φ‖∞ (N or N·m).
    pub tol: f64,
    pub max_iter: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NewtonReport {
    pub iters: usize,
    pub converged: bool,
    pub res: f64,
    pub lu_fallbacks: usize,
    pub secant_fallbacks: usize,
    pub factorizations: usize,
    pub nnz_l: usize,
    pub flops: f64,
    /// Stopped because the Newton step reached the round-off of absolute positions
    /// (|Δq| ≤ 8 ε·L): the residual floor is then ≈ ε·k·L, not `tol`.
    pub roundoff_limited: bool,
}

fn position_scale(q: &[[f64; 3]]) -> f64 {
    q.iter().fold(1.0f64, |s, r| s.max(r[0].abs()).max(r[1].abs()))
}

/// Solve ∇Φ(q) = 0 for the free DOFs (prescribed DOFs keep their values in q).
pub fn newton(m: &Model, q: &mut Vec<[f64; 3]>, lf: f64, inertia: Option<&Inertia>, opts: NewtonOpts) -> NewtonReport {
    let map = DofMap::new(m);
    let mut rep = NewtonReport::default();
    if map.n == 0 {
        rep.converged = true;
        return rep;
    }
    let mut res0 = f64::NAN;
    let mut best = f64::INFINITY;
    let mut stall = 0;
    for it in 0..opts.max_iter {
        let a = assemble(m, q, &map, lf, inertia, true, Tangent::Consistent);
        let res = a.grad.iter().fold(0.0f64, |s, &x| s.max(x.abs()));
        rep.res = res;
        rep.iters = it;
        if res <= opts.tol {
            rep.converged = true;
            return rep;
        }
        if it == 0 {
            res0 = res;
        }
        // Stagnation at the round-off floor: residual already reduced ≥1e8× and
        // no longer halving over three Newton steps.
        if res < 0.5 * best {
            best = res;
            stall = 0;
        } else {
            stall += 1;
            if stall >= 3 && best <= 1e-8 * res0 {
                rep.converged = true;
                rep.roundoff_limited = true;
                return rep;
            }
        }
        let rhs: Vec<f64> = a.grad.iter().map(|x| -x).collect();
        let mat = csc_from_triplets(map.n, a.trip);
        let mut p = rhs.clone();
        let mut ok = false;
        let mut spd = Spd::analyze(&mat);
        if spd.factor(&mat).is_ok() {
            spd.solve(&mut p);
            ok = p.iter().all(|x| x.is_finite());
            rep.factorizations += 1;
            rep.nnz_l = spd.stats.nnz_l;
            rep.flops = spd.stats.flops_factor;
        }
        if !ok {
            if let Ok(x) = lu_solve(&mat, &rhs) {
                p = x;
                ok = p.iter().all(|x| x.is_finite());
                rep.lu_fallbacks += 1;
            }
        }
        let gp: f64 = a.grad.iter().zip(&p).map(|(g, x)| g * x).sum();
        if !ok || gp >= 0.0 {
            // Not a descent direction (indefinite tangent): use the PSD secant.
            let s = assemble(m, q, &map, lf, inertia, true, Tangent::Secant);
            let mat = csc_from_triplets(map.n, s.trip);
            let mut spd = Spd::analyze(&mat);
            p = rhs.clone();
            if spd.factor(&mat).is_ok() {
                spd.solve(&mut p);
            } else if let Ok(x) = lu_solve(&mat, &rhs) {
                p = x;
            } else {
                return rep;
            }
            rep.secant_fallbacks += 1;
        }
        let gp: f64 = a.grad.iter().zip(&p).map(|(g, x)| g * x).sum();
        let pmax = p.iter().fold(0.0f64, |s, &x| s.max(x.abs()));
        if pmax <= 8.0 * f64::EPSILON * position_scale(q) {
            rep.converged = true;
            rep.roundoff_limited = true;
            return rep;
        }
        // Armijo backtracking on Φ; round-off guard: accept if the residual drops.
        let e0 = a.energy;
        let mut alpha = 1.0;
        let mut accepted = false;
        // Predicted decrease below the round-off of Φ: the energy cannot arbitrate,
        // take the full Newton step (the residual check above governs convergence).
        if gp.abs() <= 1e3 * f64::EPSILON * e0.abs() {
            axpy_q(q, &map, 1.0, &p);
            continue;
        }
        // Accept on sufficient energy decrease, or on residual decrease: at small
        // strains Φ carries round-off ~ε·L/δ relative (absolute coordinates), which
        // can bury the predicted decrease of a good Newton step.
        for _ in 0..40 {
            let mut qt = q.clone();
            axpy_q(&mut qt, &map, alpha, &p);
            let e1 = energy(m, &qt, lf, inertia);
            if e1 <= e0 + 1e-4 * alpha * gp {
                *q = qt;
                accepted = true;
                break;
            }
            let at = assemble(m, &qt, &map, lf, inertia, false, Tangent::Consistent);
            let rt = at.grad.iter().fold(0.0f64, |s, &x| s.max(x.abs()));
            if rt < (1.0 - 1e-4 * alpha) * res {
                *q = qt;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            return rep;
        }
    }
    let a = assemble(m, q, &map, lf, inertia, false, Tangent::Consistent);
    rep.res = a.grad.iter().fold(0.0f64, |s, &x| s.max(x.abs()));
    rep.converged = rep.res <= opts.tol;
    rep
}

/// Reaction (generalised force) the facets exert on body `i` at q (sign: on the body).
pub fn body_reaction(m: &Model, q: &[[f64; 3]], i: usize) -> [f64; 3] {
    let mut r = [0.0; 3];
    for f in m.facets.iter().filter(|f| f.active && (f.a == i || f.b == i)) {
        for p in 0..2 {
            let (j, e) = m.eval_point(q, f, p);
            let (jn, jt) = if f.a == i { (j.jna, j.jta) } else { (j.jnb, j.jtb) };
            for k in 0..3 {
                r[k] -= jn[k] * e.fnorm + jt[k] * e.ftan;
            }
        }
    }
    r
}

/// Linear(ised) control functional c(q) = Σ w·dir·(x_b + R(θ_b) arm) — CMOD,
/// local opening or load-point displacement.
#[derive(Clone, Debug, Default)]
pub struct Control {
    pub terms: Vec<(usize, geometry::V2, geometry::V2, f64)>,
}

impl Control {
    pub fn value(&self, q: &[[f64; 3]]) -> f64 {
        self.terms
            .iter()
            .map(|&(b, arm, dir, w)| {
                let p = geometry::V2::new(q[b][0], q[b][1]) + arm.rot(q[b][2]);
                w * dir.dot(p)
            })
            .sum()
    }
    fn grad(&self, q: &[[f64; 3]], map: &DofMap) -> Vec<(usize, f64)> {
        let mut g = Vec::new();
        for &(b, arm, dir, w) in &self.terms {
            let ra = arm.rot(q[b][2]).perp();
            let d = [w * dir.x, w * dir.y, w * dir.dot(ra)];
            for k in 0..3 {
                if let Some(i) = map.idx[b][k] {
                    g.push((i, d[k]));
                }
            }
        }
        g
    }
}

/// Newton on the bordered system [H −p; cᵀ 0][Δu; Δλ] = [−r; c* − c]: the load
/// factor λ (scaling `m.pattern`) is an unknown, the control c(q) = target.
pub fn newton_controlled(
    m: &Model,
    q: &mut Vec<[f64; 3]>,
    lf: &mut f64,
    ctrl: &Control,
    target: f64,
    opts: NewtonOpts,
    ctol: f64,
    tangent: Tangent,
) -> NewtonReport {
    let map = DofMap::new(m);
    let mut rep = NewtonReport::default();
    let n = map.n;
    let mut pvec = vec![0.0; n];
    for (i, row) in map.idx.iter().enumerate() {
        for k in 0..3 {
            if let Some(g) = row[k] {
                pvec[g] = m.pattern[i][k];
            }
        }
    }
    let mut res0 = f64::NAN;
    let mut best = f64::INFINITY;
    let mut stall = 0;
    for it in 0..opts.max_iter {
        let a = assemble(m, q, &map, *lf, None, true, tangent);
        let res = a.grad.iter().fold(0.0f64, |s, &x| s.max(x.abs()));
        let cv = ctrl.value(q);
        rep.res = res;
        rep.iters = it;
        if res <= opts.tol && (cv - target).abs() <= ctol {
            rep.converged = true;
            return rep;
        }
        if it == 0 {
            res0 = res;
        }
        if res < 0.5 * best {
            best = res;
            stall = 0;
        } else {
            stall += 1;
            if stall >= 3 && best <= 1e-8 * res0 && (cv - target).abs() <= ctol {
                rep.converged = true;
                rep.roundoff_limited = true;
                return rep;
            }
        }
        let mut trip = a.trip;
        for (g, &p) in pvec.iter().enumerate() {
            if p != 0.0 {
                trip.push((g, n, -p));
            }
        }
        for (g, v) in ctrl.grad(q, &map) {
            trip.push((n, g, v));
        }
        let mat = csc_from_triplets(n + 1, trip);
        let mut rhs: Vec<f64> = a.grad.iter().map(|x| -x).collect();
        rhs.push(target - cv);
        let Ok(x) = lu_solve(&mat, &rhs) else { return rep };
        if !x.iter().all(|v| v.is_finite()) {
            return rep;
        }
        rep.factorizations += 1;
        let pmax = x[..n].iter().fold(0.0f64, |s, &v| s.max(v.abs()));
        axpy_q(q, &map, 1.0, &x[..n]);
        *lf += x[n];
        if pmax <= 8.0 * f64::EPSILON * position_scale(q) && (cv - target).abs() <= ctol {
            rep.converged = true;
            rep.roundoff_limited = true;
            return rep;
        }
    }
    let a = assemble(m, q, &map, *lf, None, false, Tangent::Consistent);
    rep.res = a.grad.iter().fold(0.0f64, |s, &x| s.max(x.abs()));
    rep.converged = rep.res <= opts.tol && (ctrl.value(q) - target).abs() <= ctol;
    rep
}

/// Largest trial damage increment over all points (event-size check for acceptance).
pub fn max_damage_increment(m: &Model, q: &[[f64; 3]]) -> f64 {
    let mut mx: f64 = 0.0;
    for f in m.facets.iter().filter(|f| f.active) {
        for p in 0..2 {
            let (_, e) = m.eval_point(q, f, p);
            mx = mx.max(e.d - f.pts[p].law.damage(f.pts[p].kappa_c));
        }
    }
    mx
}

/// Work of the pattern load over an increment (trapezoidal in the load factor).
pub fn pattern_work(m: &Model, q0: &[[f64; 3]], q1: &[[f64; 3]], lf0: f64, lf1: f64) -> f64 {
    let mut w = 0.0;
    for i in 0..m.bodies.len() {
        for k in 0..3 {
            w += 0.5 * (lf0 + lf1) * m.pattern[i][k] * (q1[i][k] - q0[i][k]);
        }
    }
    w
}
