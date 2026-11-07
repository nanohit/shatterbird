//! S1 — compliant-ALM block descent (spec §4.2, docs §4.2); S3 = penalty limit.
//!
//! Scalar components c = 4·facet + 2·point + {0: normal, 1: tangential}; every
//! component is a length (two-point facets have no separate rotational spring),
//! so all penalties are in N/m.
//!
//! Primal sweep: colours in fixed order; each free cell takes one Gauss–Newton
//! step on Φ(q; λ) with its neighbours as left by previous colours (a colour has
//! no internal facets, so sequential = parallel snapshot semantics).
//! Dual update: λ_c ← s_c(λ_c + ρ_c C_c(q)) on the completed primal state.

use crate::assembly::Inertia;
use model::{FacetForce, Model};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RhoPolicy {
    /// ρ = c · mean(k_n) for every component.
    Shared(f64),
    /// ρ_c = c · k_c (initial elastic stiffness of the component).
    PerSpring(f64),
    /// ρ_c = c · m_eff,c / h² (requires inertia).
    Dynamic(f64),
    /// S3: no multipliers, physical stiffness in the primal (ρ → ∞).
    Penalty,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Work {
    pub sweeps: u64,
    pub cell_updates: u64,
    pub point_evals: u64,
    pub passes: u64,
    pub dual_updates: u64,
}

pub struct Alm {
    pub policy: RhoPolicy,
    pub lambda: Vec<f64>,
    pub rho: Vec<f64>,
    pub colors: Vec<Vec<usize>>,
    adj: Vec<Vec<usize>>,
    pub work: Work,
}

pub fn greedy_coloring(m: &Model) -> Vec<Vec<usize>> {
    let adj = m.neighbours();
    let nb = m.bodies.len();
    let mut color = vec![usize::MAX; nb];
    let mut ncol = 0;
    for i in 0..nb {
        if !m.is_free(i) {
            continue;
        }
        let mut used = Vec::new();
        for &fi in &adj[i] {
            let f = &m.facets[fi];
            let o = if f.a == i { f.b } else { f.a };
            if color[o] != usize::MAX {
                used.push(color[o]);
            }
        }
        let mut c = 0;
        while used.contains(&c) {
            c += 1;
        }
        color[i] = c;
        ncol = ncol.max(c + 1);
    }
    let mut colors = vec![Vec::new(); ncol];
    for i in 0..nb {
        if color[i] != usize::MAX {
            colors[color[i]].push(i);
        }
    }
    colors
}

impl Alm {
    pub fn new(m: &Model, policy: RhoPolicy, inertia: Option<&Inertia>) -> Self {
        let nc = 4 * m.facets.len();
        let mut alm = Alm {
            policy,
            lambda: vec![0.0; nc],
            rho: vec![0.0; nc],
            colors: greedy_coloring(m),
            adj: m.neighbours(),
            work: Work::default(),
        };
        alm.set_rho(m, inertia);
        alm
    }

    /// Rebuild colouring/adjacency after facets were (de)activated; multipliers of
    /// surviving components are kept (they are state), removed ones are zeroed.
    pub fn refresh_topology(&mut self, m: &Model) {
        self.colors = greedy_coloring(m);
        self.adj = m.neighbours();
        for (fi, f) in m.facets.iter().enumerate() {
            if !f.active {
                for c in 0..4 {
                    self.lambda[4 * fi + c] = 0.0;
                }
            }
        }
    }

    /// (Re)compute penalties; call after topology or stiffness changes.
    pub fn set_rho(&mut self, m: &Model, inertia: Option<&Inertia>) {
        let nc = 4 * m.facets.len();
        self.lambda.resize(nc, 0.0);
        self.rho.resize(nc, 0.0);
        let act: Vec<_> = m.facets.iter().filter(|f| f.active).collect();
        let kbar = act.iter().map(|f| f.pts[0].law.kn).sum::<f64>() / act.len().max(1) as f64;
        for (fi, f) in m.facets.iter().enumerate() {
            for p in 0..2 {
                for c in 0..2 {
                    let k = if c == 0 { f.pts[p].law.kn } else { f.pts[p].law.kt };
                    let r = match self.policy {
                        RhoPolicy::Shared(cf) => cf * kbar,
                        RhoPolicy::PerSpring(cf) => cf * if k > 0.0 { k } else { f.pts[p].law.kn },
                        RhoPolicy::Dynamic(cf) => {
                            let ine = inertia.expect("dynamic ρ needs inertia");
                            cf * self.meff(m, ine, fi, p, c) * ine.alpha_h2
                        }
                        RhoPolicy::Penalty => f64::INFINITY,
                    };
                    self.rho[4 * fi + 2 * p + c] = r;
                }
            }
        }
    }

    /// Effective mass of a scalar component: 1/m_eff = Σ_bodies J M⁻¹ Jᵀ (free DOFs only).
    fn meff(&self, m: &Model, _ine: &Inertia, fi: usize, p: usize, c: usize) -> f64 {
        let f = &m.facets[fi];
        let (j, _) = m.eval_point(&m.q, f, p);
        let (ja, jb) = if c == 0 { (j.jna, j.jnb) } else { (j.jta, j.jtb) };
        let mut inv = 0.0;
        for (bi, jr) in [(f.a, ja), (f.b, jb)] {
            let b = &m.bodies[bi];
            for k in 0..3 {
                if !b.fixed[k] {
                    let mm = if k < 2 { b.mass } else { b.inertia };
                    inv += jr[k] * jr[k] / mm;
                }
            }
        }
        1.0 / inv
    }

    #[inline]
    fn coeffs(&self, c: usize, k: f64) -> (f64, f64) {
        match self.policy {
            RhoPolicy::Penalty => (0.0, k),
            _ => {
                let r = self.rho[c];
                let s = k / (k + r);
                (s, r * s)
            }
        }
    }

    /// One primal sweep over all colours.
    pub fn sweep(&mut self, m: &Model, q: &mut [[f64; 3]], lf: f64, inertia: Option<&Inertia>) {
        for col in 0..self.colors.len() {
            for k in 0..self.colors[col].len() {
                let i = self.colors[col][k];
                self.update_cell(m, q, i, lf, inertia);
            }
            self.work.passes += 1;
        }
        self.work.sweeps += 1;
    }

    fn update_cell(&mut self, m: &Model, q: &mut [[f64; 3]], i: usize, lf: f64, inertia: Option<&Inertia>) {
        let b = &m.bodies[i];
        let mut g = [0.0; 3];
        let mut h = [[0.0; 3]; 3];
        for k in 0..3 {
            g[k] -= m.loads[i][k] + lf * m.pattern[i][k];
            if let Some(ine) = inertia {
                let w = ine.weight(m, i, k);
                g[k] += w * (q[i][k] - ine.qhat[i][k]);
                h[k][k] += w;
            }
        }
        for &fi in &self.adj[i] {
            let f = &m.facets[fi];
            for p in 0..2 {
                let (j, e) = m.eval_point(q, f, p);
                self.work.point_evals += 1;
                let (kn, kt) = f.pts[p].law.secant(j.dn, e.d);
                let (rn, rt) = if f.a == i { (j.jna, j.jta) } else { (j.jnb, j.jtb) };
                for (c, cval, k, row) in [(4 * fi + 2 * p, j.dn, kn, rn), (4 * fi + 2 * p + 1, j.dt, kt, rt)] {
                    let (s, keff) = self.coeffs(c, k);
                    let force = s * self.lambda[c] + keff * cval;
                    for r in 0..3 {
                        g[r] += row[r] * force;
                        for cc in 0..3 {
                            h[r][cc] += keff * row[r] * row[cc];
                        }
                    }
                    if model::range::enabled() {
                        model::range::rec("s", s);
                        model::range::rec("1-s", 1.0 - s);
                        model::range::rec("k_eff", keff);
                    }
                }
            }
        }
        // Fixed DOFs of a partially fixed body are left alone.
        for k in 0..3 {
            if b.fixed[k] {
                g[k] = 0.0;
                for c in 0..3 {
                    h[k][c] = 0.0;
                    h[c][k] = 0.0;
                }
                h[k][k] = 1.0;
            }
        }
        if let Some(dx) = solve3(h, g) {
            for k in 0..3 {
                q[i][k] -= dx[k];
            }
        }
        self.work.cell_updates += 1;
    }

    /// Simultaneous dual update on the completed primal state.
    pub fn dual(&mut self, m: &Model, q: &[[f64; 3]]) {
        for (fi, f) in m.facets.iter().enumerate() {
            if !f.active {
                continue;
            }
            for p in 0..2 {
                let (j, e) = m.eval_point(q, f, p);
                self.work.point_evals += 1;
                let (kn, kt) = f.pts[p].law.secant(j.dn, e.d);
                for (c, cval, k) in [(4 * fi + 2 * p, j.dn, kn), (4 * fi + 2 * p + 1, j.dt, kt)] {
                    self.lambda[c] = match self.policy {
                        RhoPolicy::Penalty => k * cval,
                        _ => {
                            let (s, _) = self.coeffs(c, k);
                            s * (self.lambda[c] + self.rho[c] * cval)
                        }
                    };
                    if model::range::enabled() {
                        model::range::rec("lambda", self.lambda[c]);
                    }
                }
            }
        }
        self.work.passes += 1;
        self.work.dual_updates += 1;
    }

    /// One ALM iteration as specified: primal sweep then dual update.
    pub fn iterate(&mut self, m: &Model, q: &mut [[f64; 3]], lf: f64, inertia: Option<&Inertia>) {
        self.sweep(m, q, lf, inertia);
        self.dual(m, q);
    }

    /// Facet resultants from the multipliers.
    pub fn dual_forces(&self, m: &Model) -> Vec<FacetForce> {
        m.facets
            .iter()
            .enumerate()
            .map(|(fi, f)| {
                if !f.active {
                    return FacetForce::default();
                }
                let l = &self.lambda[4 * fi..4 * fi + 4];
                FacetForce { n: l[0] + l[2], v: l[1] + l[3], m: f.pts[0].s * l[0] + f.pts[1].s * l[2] }
            })
            .collect()
    }

    /// Set multipliers from given point forces (dual injection / warm start).
    pub fn set_lambda_from_q(&mut self, m: &Model, q: &[[f64; 3]]) {
        for (fi, f) in m.facets.iter().enumerate() {
            if !f.active {
                continue;
            }
            for p in 0..2 {
                let (_, e) = m.eval_point(q, f, p);
                self.lambda[4 * fi + 2 * p] = e.fnorm;
                self.lambda[4 * fi + 2 * p + 1] = e.ftan;
            }
        }
    }
}

pub fn solve3(h: [[f64; 3]; 3], g: [f64; 3]) -> Option<[f64; 3]> {
    let det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1]) - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
    let scale = h[0][0].abs().max(h[1][1].abs()).max(h[2][2].abs());
    if !(det.abs() > 1e-14 * scale * scale * scale) {
        return None;
    }
    let inv = |r: usize, c: usize| -> f64 {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
        h[c1][r1] * h[c2][r2] - h[c1][r2] * h[c2][r1]
    };
    let mut x = [0.0; 3];
    for r in 0..3 {
        for c in 0..3 {
            x[r] += inv(r, c) * g[c];
        }
        x[r] /= det;
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    #[test]
    fn solve3_works() {
        let h = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let g = [1.0, 2.0, 3.0];
        let x = super::solve3(h, g).unwrap();
        for r in 0..3 {
            let s: f64 = (0..3).map(|c| h[r][c] * x[c]).sum();
            assert!((s - g[r]).abs() < 1e-12);
        }
    }
}
