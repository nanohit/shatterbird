//! Global gradient / Gauss–Newton Hessian of the incremental potential
//!   Φ(q) = α/(2h²)‖q − q̂‖²_M + Σ Ψ_trial(δ(q)) − (loads + lf·pattern)·(q − q_ref)

use model::Model;

/// Inertial term of a variational integrator: α/h² and the prediction q̂.
#[derive(Clone, Debug)]
pub struct Inertia {
    pub alpha_h2: f64,
    pub qhat: Vec<[f64; 3]>,
}

impl Inertia {
    pub fn weight(&self, m: &Model, i: usize, k: usize) -> f64 {
        let b = &m.bodies[i];
        self.alpha_h2 * if k < 2 { b.mass } else { b.inertia }
    }
}

pub struct DofMap {
    pub idx: Vec<[Option<usize>; 3]>,
    pub n: usize,
}

impl DofMap {
    pub fn new(m: &Model) -> Self {
        let (idx, n) = m.dof_map();
        DofMap { idx, n }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tangent {
    /// Consistent material tangent (may be indefinite under softening).
    Consistent,
    /// Secant (always PSD).
    Secant,
}

pub struct Assembled {
    pub grad: Vec<f64>,
    pub trip: Vec<(usize, usize, f64)>,
    pub energy: f64,
}

pub fn energy(m: &Model, q: &[[f64; 3]], lf: f64, inertia: Option<&Inertia>) -> f64 {
    let mut e = m.load_potential(q, lf);
    for f in m.facets.iter().filter(|f| f.active) {
        for p in 0..2 {
            let pt = &f.pts[p];
            let j = model::kin::jump(q[f.a], q[f.b], pt.arm_a, pt.arm_b, f.n0, f.t0, pt.rest, f.frame_w);
            e += pt.law.potential(j.dn, j.dt, pt.kappa_c);
        }
    }
    if let Some(ine) = inertia {
        for i in 0..m.bodies.len() {
            for k in 0..3 {
                if !m.bodies[i].fixed[k] {
                    let d = q[i][k] - ine.qhat[i][k];
                    e += 0.5 * ine.weight(m, i, k) * d * d;
                }
            }
        }
    }
    e
}

pub fn assemble(
    m: &Model,
    q: &[[f64; 3]],
    map: &DofMap,
    lf: f64,
    inertia: Option<&Inertia>,
    want_h: bool,
    tangent: Tangent,
) -> Assembled {
    let mut g = vec![0.0; map.n];
    let mut trip = Vec::with_capacity(if want_h { m.facets.len() * 2 * 36 + map.n } else { 0 });
    let mut energy = m.load_potential(q, lf);
    for (i, b) in m.bodies.iter().enumerate() {
        for k in 0..3 {
            if let Some(gi) = map.idx[i][k] {
                g[gi] -= m.loads[i][k] + lf * m.pattern[i][k];
                if let Some(ine) = inertia {
                    let w = ine.weight(m, i, k);
                    let d = q[i][k] - ine.qhat[i][k];
                    g[gi] += w * d;
                    energy += 0.5 * w * d * d;
                    if want_h {
                        trip.push((gi, gi, w));
                    }
                }
                let _ = b;
            }
        }
    }
    for f in m.facets.iter().filter(|f| f.active) {
        let ia = map.idx[f.a];
        let ib = map.idx[f.b];
        let dofs = [ia[0], ia[1], ia[2], ib[0], ib[1], ib[2]];
        for p in 0..2 {
            let (j, e) = m.eval_point(q, f, p);
            energy += f.pts[p].law.potential(j.dn, j.dt, f.pts[p].kappa_c);
            let jn = [j.jna[0], j.jna[1], j.jna[2], j.jnb[0], j.jnb[1], j.jnb[2]];
            let jt = [j.jta[0], j.jta[1], j.jta[2], j.jtb[0], j.jtb[1], j.jtb[2]];
            for r in 0..6 {
                if let Some(gr) = dofs[r] {
                    g[gr] += jn[r] * e.fnorm + jt[r] * e.ftan;
                }
            }
            if want_h {
                let d = match tangent {
                    Tangent::Consistent => e.tan,
                    Tangent::Secant => {
                        let (kn, kt) = f.pts[p].law.secant(j.dn, e.d);
                        [kn, 0.0, kt]
                    }
                };
                for r in 0..6 {
                    let Some(gr) = dofs[r] else { continue };
                    // (Jᵀ D J)_{rc}
                    let ar = d[0] * jn[r] + d[1] * jt[r];
                    let br = d[1] * jn[r] + d[2] * jt[r];
                    for c in 0..6 {
                        let Some(gc) = dofs[c] else { continue };
                        let v = ar * jn[c] + br * jt[c];
                        if v != 0.0 {
                            trip.push((gr, gc, v));
                        }
                    }
                }
            }
        }
    }
    Assembled { grad: g, trip, energy }
}

/// Scatter a free-DOF vector into body DOFs (adds `alpha * x`).
pub fn axpy_q(q: &mut [[f64; 3]], map: &DofMap, alpha: f64, x: &[f64]) {
    for (i, row) in map.idx.iter().enumerate() {
        for k in 0..3 {
            if let Some(g) = row[k] {
                q[i][k] += alpha * x[g];
            }
        }
    }
}
