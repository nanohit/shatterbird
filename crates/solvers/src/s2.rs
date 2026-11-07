//! S2 — static force reconstruction (contract: docs §4.3).
//!
//! K u = f_ext − M a(q_S1), K = reference-frame linear stiffness with the trial
//! secant law at the S1 state. u is an auxiliary elastic field: it defines the
//! reported facet forces only; motion/work/measured displacement stay with S1.
//! Floating components: load projected to self-equilibrium (mass-weighted), gauge
//! R^T M u = 0 imposed by a bordered system (not a support); their loads are
//! taken in the component's mean rotated frame (fixed-reference approximation).

use crate::assembly::DofMap;
use crate::s1::Alm;
use crate::sparse::{csc_from_triplets, lu_solve, FactorStats, Spd};
use model::{FacetForce, Model};

pub struct S2 {
    spd: Option<Spd>,
    last_k: Vec<f64>,
    last_active: Vec<bool>,
    pub stats: FactorStats,
    pub lu_solves: u64,
    pub reused: u64,
    pub mechanism_failures: u64,
}

pub struct S2Out {
    /// Auxiliary displacement relative to the reference configuration.
    pub u: Vec<[f64; 3]>,
    /// Point forces (normal, tangential) per facet point: [f][p] = (fn, ft).
    pub point: Vec<[(f64, f64); 2]>,
    pub forces: Vec<FacetForce>,
    pub floating_components: usize,
}

impl Default for S2 {
    fn default() -> Self {
        Self::new()
    }
}

impl S2 {
    pub fn new() -> Self {
        S2 { spd: None, last_k: Vec::new(), last_active: Vec::new(), stats: FactorStats::default(), lu_solves: 0, reused: 0, mechanism_failures: 0 }
    }

    /// `accel`: integrator acceleration of the S1 motion per body (None = quasi-static).
    pub fn reconstruct(&mut self, m: &Model, q_motion: &[[f64; 3]], lf: f64, accel: Option<&[[f64; 3]]>) -> Result<S2Out, String> {
        let map = DofMap::new(m);
        let qref: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
        // Secant stiffness at the S1 trial state; Jacobians at the reference.
        let mut ks = Vec::with_capacity(m.facets.len() * 4);
        for f in &m.facets {
            for p in 0..2 {
                if f.active {
                    let (j, e) = m.eval_point(q_motion, f, p);
                    // (1−d) < 1e-3: treated as broken in S2 (declared approximation; keeps
                    // near-mechanisms from making K numerically singular)
                    let (kn, kt) = if e.d > 1.0 - 1e-3 {
                        (if j.dn <= 0.0 { f.pts[p].law.kn } else { 0.0 }, 0.0)
                    } else {
                        f.pts[p].law.secant(j.dn, e.d)
                    };
                    ks.push(kn);
                    ks.push(kt);
                } else {
                    ks.push(0.0);
                    ks.push(0.0);
                }
            }
        }
        let active: Vec<bool> = m.facets.iter().map(|f| f.active).collect();
        let mut trip = Vec::new();
        let mut rhs = vec![0.0; map.n];
        let jac: Vec<[model::kin::Jump; 2]> = m
            .facets
            .iter()
            .map(|f| [0, 1].map(|p| model::kin::jump(qref[f.a], qref[f.b], f.pts[p].arm_a, f.pts[p].arm_b, f.n0, f.t0, f.pts[p].rest, f.frame_w)))
            .collect();
        // prescribed displacements
        let up: Vec<[f64; 3]> = (0..m.bodies.len())
            .map(|i| {
                let mut u = [0.0; 3];
                for k in 0..3 {
                    if m.bodies[i].fixed[k] {
                        u[k] = q_motion[i][k] - qref[i][k];
                    }
                }
                u
            })
            .collect();
        for (fi, f) in m.facets.iter().enumerate() {
            if !f.active {
                continue;
            }
            let dofs = [map.idx[f.a], map.idx[f.b]];
            for p in 0..2 {
                let j = &jac[fi][p];
                let rows = [
                    ([j.jna, j.jnb], ks[4 * fi + 2 * p]),
                    ([j.jta, j.jtb], ks[4 * fi + 2 * p + 1]),
                ];
                for (jr, k) in rows {
                    if k == 0.0 {
                        continue;
                    }
                    // prescribed part of C
                    let mut cp = 0.0;
                    for (side, bi) in [(0, f.a), (1, f.b)] {
                        for d in 0..3 {
                            if dofs[side][d].is_none() {
                                cp += jr[side][d] * up[bi][d];
                            }
                        }
                    }
                    for sr in 0..2 {
                        for dr in 0..3 {
                            let Some(gr) = dofs[sr][dr] else { continue };
                            rhs[gr] -= k * jr[sr][dr] * cp;
                            for sc in 0..2 {
                                for dc in 0..3 {
                                    let Some(gc) = dofs[sc][dc] else { continue };
                                    trip.push((gr, gc, k * jr[sr][dr] * jr[sc][dc]));
                                }
                            }
                        }
                    }
                }
            }
        }
        // loads
        let mut b = vec![[0.0; 3]; m.bodies.len()];
        for i in 0..m.bodies.len() {
            for k in 0..3 {
                b[i][k] = m.loads[i][k] + lf * m.pattern[i][k];
                if let Some(a) = accel {
                    let mm = if k < 2 { m.bodies[i].mass } else { m.bodies[i].inertia };
                    b[i][k] -= mm * a[i][k];
                }
            }
        }
        // Components among free bodies (through facets with stiffness).
        let nb = m.bodies.len();
        let mut comp = vec![usize::MAX; nb];
        let mut ncomp = 0;
        let mut grounded = Vec::new();
        let adj = m.neighbours();
        for s in 0..nb {
            if !m.is_free(s) || comp[s] != usize::MAX {
                continue;
            }
            let mut stack = vec![s];
            comp[s] = ncomp;
            let mut gnd = m.bodies[s].fixed.iter().any(|&x| x);
            while let Some(i) = stack.pop() {
                for &fi in &adj[i] {
                    // only facets that still carry stiffness connect (detached fragments float)
                    if (0..4).all(|c| ks[4 * fi + c] == 0.0) {
                        continue;
                    }
                    let f = &m.facets[fi];
                    let o = if f.a == i { f.b } else { f.a };
                    if !m.is_free(o) || m.bodies[o].fixed.iter().any(|&x| x) {
                        gnd = true;
                        if !m.is_free(o) {
                            continue;
                        }
                    }
                    if comp[o] == usize::MAX {
                        comp[o] = ncomp;
                        stack.push(o);
                    }
                }
            }
            grounded.push(gnd);
            ncomp += 1;
        }
        let floating: Vec<usize> = (0..ncomp).filter(|&c| !grounded[c]).collect();
        // Rigid modes and self-equilibrated loads for floating components.
        let mut gauge_rows: Vec<Vec<(usize, f64)>> = Vec::new();
        let mut rot_c = vec![0.0; ncomp];
        for &c in &floating {
            let members: Vec<usize> = (0..nb).filter(|&i| comp[i] == c).collect();
            let mt: f64 = members.iter().map(|&i| m.bodies[i].mass).sum();
            let xc = members.iter().fold([0.0, 0.0], |s, &i| {
                [s[0] + m.bodies[i].mass * m.bodies[i].x0.x, s[1] + m.bodies[i].mass * m.bodies[i].x0.y]
            });
            let xc = [xc[0] / mt, xc[1] / mt];
            let th: f64 = members.iter().map(|&i| m.bodies[i].mass * q_motion[i][2]).sum::<f64>() / mt;
            rot_c[c] = th;
            // rotate loads into the reference frame
            for &i in &members {
                let (s, co) = th.sin_cos();
                let (x, y) = (b[i][0], b[i][1]);
                b[i][0] = co * x + s * y;
                b[i][1] = -s * x + co * y;
            }
            // modes: tx, ty, rotation about xc
            let mode = |i: usize, r: usize| -> [f64; 3] {
                let x = m.bodies[i].x0;
                match r {
                    0 => [1.0, 0.0, 0.0],
                    1 => [0.0, 1.0, 0.0],
                    _ => [-(x.y - xc[1]), x.x - xc[0], 1.0],
                }
            };
            let mass = |i: usize, k: usize| if k < 2 { m.bodies[i].mass } else { m.bodies[i].inertia };
            // G = Rᵀ M R (3×3), r = Rᵀ b
            let mut gm = [[0.0; 3]; 3];
            let mut rb = [0.0; 3];
            for &i in &members {
                for r in 0..3 {
                    let mr = mode(i, r);
                    for k in 0..3 {
                        rb[r] += mr[k] * b[i][k];
                        for s in 0..3 {
                            gm[r][s] += mr[k] * mass(i, k) * mode(i, s)[k];
                        }
                    }
                }
            }
            let coef = crate::s1::solve3(gm, rb).unwrap_or([0.0; 3]);
            for &i in &members {
                for k in 0..3 {
                    let mut rig = 0.0;
                    for r in 0..3 {
                        rig += mode(i, r)[k] * coef[r];
                    }
                    b[i][k] -= mass(i, k) * rig;
                }
            }
            for r in 0..3 {
                let mut row = Vec::new();
                for &i in &members {
                    for k in 0..3 {
                        if let Some(g) = map.idx[i][k] {
                            row.push((g, mode(i, r)[k] * mass(i, k)));
                        }
                    }
                }
                gauge_rows.push(row);
            }
        }
        for i in 0..nb {
            for k in 0..3 {
                if let Some(g) = map.idx[i][k] {
                    rhs[g] += b[i][k];
                }
            }
        }
        let mut x = rhs.clone();
        if gauge_rows.is_empty() {
            let same_pattern = self.spd.is_some() && self.last_active == active;
            let same_values = same_pattern && self.last_k == ks;
            let mat = csc_from_triplets(map.n, trip);
            if !same_pattern {
                let s = Spd::analyze(&mat);
                self.stats.symbolic += 1;
                self.stats.nnz_l = s.stats.nnz_l;
                self.stats.flops_factor = s.stats.flops_factor;
                self.spd = Some(s);
            }
            let spd = self.spd.as_mut().unwrap();
            if !same_values {
                if let Err(e) = spd.factor(&mat) {
                    // mechanism (zero pivot): reported, never regularised away
                    self.spd = None;
                    self.mechanism_failures += 1;
                    return Err(format!("S2 mechanism: {e}"));
                }
                self.stats.numeric += 1;
            } else {
                self.reused += 1;
            }
            spd.solve(&mut x);
            self.stats.solves += 1;
            self.last_active = active;
            self.last_k = ks.clone();
        } else {
            let ng = gauge_rows.len();
            for (r, row) in gauge_rows.iter().enumerate() {
                for &(g, v) in row {
                    trip.push((map.n + r, g, v));
                    trip.push((g, map.n + r, v));
                }
            }
            let mat = csc_from_triplets(map.n + ng, trip);
            let mut rr = rhs.clone();
            rr.extend(std::iter::repeat(0.0).take(ng));
            x = match lu_solve(&mat, &rr) {
                Ok(v) if v.iter().all(|z| z.is_finite()) => v,
                _ => {
                    self.mechanism_failures += 1;
                    return Err("S2 mechanism in a floating component".into());
                }
            };
            x.truncate(map.n);
            self.lu_solves += 1;
            self.spd = None;
        }
        // sanity of the auxiliary field: a near-mechanism shows up as |u| far beyond the motion
        let umax = x.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let qmax = (0..nb).fold(0.0f64, |a, i| a.max((q_motion[i][0] - qref[i][0]).abs()).max((q_motion[i][1] - qref[i][1]).abs()));
        if !(umax <= 10.0 * qmax + 0.01) {
            self.mechanism_failures += 1;
            return Err(format!("S2 ill-conditioned: |u| = {umax:e} vs motion {qmax:e}"));
        }
        let mut u = up.clone();
        for i in 0..nb {
            for k in 0..3 {
                if let Some(g) = map.idx[i][k] {
                    u[i][k] = x[g];
                }
            }
        }
        // rotate floating components' u back to the current frame
        for &c in &floating {
            let th = rot_c[c];
            let (s, co) = th.sin_cos();
            for i in 0..nb {
                if comp[i] == c {
                    let (ux, uy) = (u[i][0], u[i][1]);
                    u[i][0] = co * ux - s * uy;
                    u[i][1] = s * ux + co * uy;
                }
            }
        }
        // point forces from the reference-frame field (rotation-invariant scalars)
        let mut point = vec![[(0.0, 0.0); 2]; m.facets.len()];
        let mut forces = vec![FacetForce::default(); m.facets.len()];
        for (fi, f) in m.facets.iter().enumerate() {
            if !f.active {
                continue;
            }
            for p in 0..2 {
                let j = &jac[fi][p];
                let mut ua = u[f.a];
                let mut ub = u[f.b];
                // back to the reference frame for floating components
                for (uu, bi) in [(&mut ua, f.a), (&mut ub, f.b)] {
                    if comp[bi] != usize::MAX && !grounded[comp[bi]] {
                        let th = rot_c[comp[bi]];
                        let (s, co) = th.sin_cos();
                        let (x, y) = (uu[0], uu[1]);
                        uu[0] = co * x + s * y;
                        uu[1] = -s * x + co * y;
                    }
                }
                let cn: f64 = (0..3).map(|k| j.jna[k] * ua[k] + j.jnb[k] * ub[k]).sum();
                let ct: f64 = (0..3).map(|k| j.jta[k] * ua[k] + j.jtb[k] * ub[k]).sum();
                let fnn = ks[4 * fi + 2 * p] * cn;
                let ftt = ks[4 * fi + 2 * p + 1] * ct;
                point[fi][p] = (fnn, ftt);
                forces[fi].n += fnn;
                forces[fi].v += ftt;
                forces[fi].m += f.pts[p].s * fnn;
            }
        }
        Ok(S2Out { u, point, forces, floating_components: floating.len() })
    }
}

/// Dual injection (coupling B): λ ← S2 point forces.
pub fn inject(alm: &mut Alm, out: &S2Out) {
    for (fi, pts) in out.point.iter().enumerate() {
        for p in 0..2 {
            alm.lambda[4 * fi + 2 * p] = pts[p].0;
            alm.lambda[4 * fi + 2 * p + 1] = pts[p].1;
        }
    }
}
