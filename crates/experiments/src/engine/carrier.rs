//! Experimental causal constitutive mask. All wave DOFs and the explicit clock
//! remain intact. Promotions are permanent and decided from the present state,
//! never a reference run. Optional local relinearization follows large rotations
//! without changing the physical rest geometry or erasing stored strain.
use model::{kin::Jump, law::LawEval, LawKind, Model};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Config {
    pub chi: f64,
    /// Taylor remainder budget, as a fraction of the onset opening.
    /// None is an intentionally unsafe, strain-only ablation.
    pub geometry: Option<f64>,
    pub rebase: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            chi: 0.8,
            geometry: Some(0.05),
            rebase: false,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub full_points: u64,
    pub linear_points: u64,
    pub guarded_facets: u64,
    /// Exact point evaluations spent refreshing a still-elastic cold facet.
    /// Included in full_points too, so runtime work is not hidden.
    pub refresh_points: u64,
    pub promoted_strain: usize,
    pub promoted_geometry: usize,
    pub promoted_over_onset: usize,
    pub switch_energy_added: f64,
    pub switch_energy_absolute: f64,
}
#[derive(Clone)]
struct Cached {
    // Full point Jacobians beat reconstructing shared fields in the CPU probe.
    j: [Jump; 2],
    equivalent_bound_weight: [f64; 2],
    center_distance: f64,
    arm_a: [f64; 2],
    arm_b: [f64; 2],
}
pub struct Carrier {
    pub config: Config,
    pub hot: Vec<bool>,
    pub stats: Stats,
    q0: Vec<[f64; 3]>,
    /// Only moving-frame runs pay for per-facet reference coordinates.
    references: Option<Vec<([f64; 3], [f64; 3])>>,
    cache: Vec<Cached>,
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn frozen_energy(m: &Model, fi: usize, p: usize, j: Jump) -> f64 {
    let pt = &m.facets[fi].pts[p];
    let l = &pt.law;
    let om = 1.0 - l.damage(pt.kappa_c);
    0.5 * (l.kn * j.dn.min(0.0).powi(2) + om * (l.kn * j.dn.max(0.0).powi(2) + l.kt * j.dt * j.dt))
}
impl Carrier {
    pub fn new(m: &Model, q: &[[f64; 3]], config: Config) -> Self {
        assert!(config.chi > 0.0 && config.chi < 1.0);
        assert!(config.geometry.is_none_or(|x| x > 0.0 && x.is_finite()));
        assert!(!config.rebase || config.geometry.is_some());
        let cache = m
            .facets
            .iter()
            .map(|f| Cached {
                j: std::array::from_fn(|p| m.eval_point(q, f, p).0),
                equivalent_bound_weight: std::array::from_fn(|p| {
                    f.pts[p].law.gamma.max(1.0).sqrt()
                }),
                center_distance: ((q[f.a][0] - q[f.b][0]).powi(2)
                    + (q[f.a][1] - q[f.b][1]).powi(2))
                .sqrt(),
                arm_a: std::array::from_fn(|p| f.pts[p].arm_a.norm()),
                arm_b: std::array::from_fn(|p| f.pts[p].arm_b.norm()),
            })
            .collect();
        let hot = m
            .facets
            .iter()
            .map(|f| {
                f.tag != 0
                    || f.pts
                        .iter()
                        .any(|p| p.law.kind != LawKind::Damage || p.kappa_c > p.law.delta0)
            })
            .collect();
        Self {
            config,
            hot,
            stats: Stats::default(),
            q0: q.to_vec(),
            references: config
                .rebase
                .then(|| m.facets.iter().map(|f| (q[f.a], q[f.b])).collect()),
            cache,
        }
    }
    fn deltas(&self, m: &Model, fi: usize, q: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
        let f = &m.facets[fi];
        let (qa, qb) = match self.references.as_ref() {
            Some(r) => r[fi],
            None => (self.q0[f.a], self.q0[f.b]),
        };
        (
            std::array::from_fn(|r| q[f.a][r] - qa[r]),
            std::array::from_fn(|r| q[f.b][r] - qb[r]),
        )
    }
    /// Bound on the Euclidean error in the corotated two-component jump.
    /// Uses ||R(a)-I-aJ|| <= a²/2 and ||R(a)-I|| <= |a|.
    pub fn remainder(&self, m: &Model, fi: usize, p: usize, q: &[[f64; 3]]) -> f64 {
        let f = &m.facets[fi];
        let c = &self.cache[fi];
        let (a, b) = self.deltas(m, fi, q);
        let phi = (1.0 - f.frame_w) * a[2] + f.frame_w * b[2];
        let du = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        0.5 * phi * phi * c.center_distance
            + phi.abs() * du
            + 0.5 * (b[2] - phi).powi(2) * c.arm_b[p]
            + 0.5 * (a[2] - phi).powi(2) * c.arm_a[p]
    }
    pub fn linear(&self, m: &Model, fi: usize, p: usize, q: &[[f64; 3]]) -> (Jump, LawEval) {
        let (a, b) = self.deltas(m, fi, q);
        let mut j = self.cache[fi].j[p];
        j.dn += dot(j.jna, a) + dot(j.jnb, b);
        j.dt += dot(j.jta, a) + dot(j.jtb, b);
        let law = &m.facets[fi].pts[p].law;
        (
            j,
            LawEval {
                fnorm: law.kn * j.dn,
                ftan: law.kt * j.dt,
                psi: 0.5 * (law.kn * j.dn * j.dn + law.kt * j.dt * j.dt),
                tan: [law.kn, 0.0, law.kt],
                ..Default::default()
            },
        )
    }
    /// Fused production path: one displacement gather and one geometric guard
    /// per facet, reusing the affine results if the facet stays cold.
    pub fn evaluate(
        &mut self,
        m: &Model,
        fi: usize,
        q: &[[f64; 3]],
    ) -> ([(Jump, LawEval); 2], f64) {
        let f = &m.facets[fi];
        if self.hot[fi] {
            self.stats.full_points += 2;
            return (std::array::from_fn(|p| m.eval_point(q, f, p)), 0.0);
        }
        self.stats.guarded_facets += 1;
        let (a, b) = self.deltas(m, fi, q);
        let cache = &self.cache[fi];
        let lin = std::array::from_fn(|p| {
            let mut j = cache.j[p];
            j.dn += dot(j.jna, a) + dot(j.jnb, b);
            j.dt += dot(j.jta, a) + dot(j.jtb, b);
            let l = &f.pts[p].law;
            (
                j,
                LawEval {
                    fnorm: l.kn * j.dn,
                    ftan: l.kt * j.dt,
                    psi: 0.5 * (l.kn * j.dn * j.dn + l.kt * j.dt * j.dt),
                    tan: [l.kn, 0.0, l.kt],
                    ..Default::default()
                },
            )
        });
        let (base, aa, ab) = if self.config.geometry.is_some() {
            let phi = (1.0 - f.frame_w) * a[2] + f.frame_w * b[2];
            let du = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            (
                0.5 * phi * phi * cache.center_distance + phi.abs() * du,
                0.5 * (a[2] - phi).powi(2),
                0.5 * (b[2] - phi).powi(2),
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        let mut strain = false;
        let mut geometry = false;
        for p in 0..2 {
            let l = &f.pts[p].law;
            let j = lin[p].0;
            let rem = base + aa * cache.arm_a[p] + ab * cache.arm_b[p];
            geometry |= self.config.geometry.is_some_and(|eps| rem > eps * l.delta0);
            let safe =
                (self.config.chi * l.delta0 - rem * cache.equivalent_bound_weight[p]).max(0.0);
            strain |= j.dn.max(0.0).powi(2) + l.gamma * j.dt * j.dt >= safe * safe;
        }
        if !strain && !geometry {
            self.stats.linear_points += 2;
            return (lin, 0.0);
        }
        let exact = std::array::from_fn(|p| m.eval_point(q, f, p));
        let mut de = 0.0;
        let mut late = false;
        let mut close_to_onset = false;
        for p in 0..2 {
            let (j, _) = exact[p];
            let l = &f.pts[p].law;
            de += frozen_energy(m, fi, p, j) - lin[p].1.psi;
            late |= j.dn.max(0.0).powi(2) + l.gamma * j.dt * j.dt > l.delta0 * l.delta0;
            close_to_onset |= j.dn.max(0.0).powi(2) + l.gamma * j.dt * j.dt
                >= (self.config.chi * l.delta0).powi(2);
        }
        self.stats.full_points += 2;
        self.stats.switch_energy_added += de;
        self.stats.switch_energy_absolute += de.abs();
        if self.config.rebase && !close_to_onset {
            // This changes the Taylor expansion point, NOT the material rest
            // opening, stiffness, history, masses, positions, or velocities.
            let c = &mut self.cache[fi];
            c.j = [exact[0].0, exact[1].0];
            self.references.as_mut().unwrap()[fi] = (q[f.a], q[f.b]);
            c.center_distance =
                ((q[f.a][0] - q[f.b][0]).powi(2) + (q[f.a][1] - q[f.b][1]).powi(2)).sqrt();
            self.stats.refresh_points += 2;
            return (exact, de);
        }
        self.hot[fi] = true;
        if geometry {
            self.stats.promoted_geometry += 1;
        } else {
            self.stats.promoted_strain += 1;
        }
        self.stats.promoted_over_onset += late as usize;
        (exact, de)
    }
    pub fn energy(&self, m: &Model, fi: usize, p: usize, q: &[[f64; 3]]) -> f64 {
        if self.hot[fi] {
            m.eval_point(q, &m.facets[fi], p).1.psi
        } else {
            self.linear(m, fi, p, q).1.psi
        }
    }
    pub fn hot_cells(&self, m: &Model) -> usize {
        let mut used = vec![false; m.bodies.len()];
        for (fi, f) in m.facets.iter().enumerate() {
            if f.active && f.tag == 0 && self.hot[fi] {
                used[f.a] = true;
                used[f.b] = true;
            }
        }
        used.iter().filter(|&&x| x).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geometry::V2;
    fn fixture() -> Model {
        let t = crate::rect_tess(V2::ZERO, V2::new(1.0, 0.5), 0.16, 37);
        Model::from_tessellation(
            &t,
            model::Material::concrete(0.2),
            model::LawOptions::damage(),
            &[],
        )
    }
    #[test]
    fn affine_force_is_the_gradient_of_its_energy() {
        let m = fixture();
        let c = Carrier::new(&m, &m.q, Config::default());
        let fi = 0;
        let f = &m.facets[fi];
        let mut q = m.q.clone();
        q[f.a][0] += 1e-6;
        q[f.b][2] += 2e-5;
        let (j, e) = c.linear(&m, fi, 0, &q);
        for (body, jn, jt) in [(f.a, j.jna, j.jta), (f.b, j.jnb, j.jtb)] {
            for r in 0..3 {
                let eps = 1e-8;
                let mut plus = q.clone();
                let mut minus = q.clone();
                plus[body][r] += eps;
                minus[body][r] -= eps;
                let fd = (c.linear(&m, fi, 0, &plus).1.psi - c.linear(&m, fi, 0, &minus).1.psi)
                    / (2.0 * eps);
                let analytic = jn[r] * e.fnorm + jt[r] * e.ftan;
                assert!((fd - analytic).abs() < 1e-6 * analytic.abs().max(1.0));
            }
        }
    }
    #[test]
    fn geometric_remainder_bounds_true_jump_including_rigid_rotation() {
        let mut m = fixture();
        // The reference itself need not be the undeformed, unrotated geometry.
        for (i, q) in m.q.iter_mut().enumerate() {
            q[0] += 1e-5 * (i as f64).sin();
            q[2] = 0.003 * (i as f64).cos();
        }
        let c = Carrier::new(&m, &m.q, Config::default());
        for k in 0..30 {
            let mut q = m.q.clone();
            let angle = 0.0001 * 1.35f64.powi(k);
            for (i, x) in q.iter_mut().enumerate() {
                let rotated = V2::new(x[0], x[1]).rot(angle);
                x[0] = rotated.x + 0.02;
                x[1] = rotated.y - 0.01;
                x[2] += angle;
                x[0] += 0.003 * angle * (i as f64).sin();
                x[2] += 0.2 * angle * (i as f64).cos();
            }
            for fi in 0..m.facets.len() {
                for p in 0..2 {
                    let (lin, _) = c.linear(&m, fi, p, &q);
                    let (exact, _) = m.eval_point(&q, &m.facets[fi], p);
                    let err = ((lin.dn - exact.dn).powi(2) + (lin.dt - exact.dt).powi(2)).sqrt();
                    assert!(
                        err <= c.remainder(&m, fi, p, &q) * (1.0 + 1e-10) + 1e-14,
                        "{err} exceeds bound"
                    );
                }
            }
        }
    }
    #[test]
    fn every_certified_cold_point_remains_below_damage_onset() {
        let m = fixture();
        let mut c = Carrier::new(&m, &m.q, Config::default());
        for k in 0..50 {
            let mut q = m.q.clone();
            for (i, x) in q.iter_mut().enumerate() {
                x[0] += k as f64 * 1e-7 * (i as f64).sin();
                x[2] += k as f64 * 1e-5 * (i as f64).cos();
            }
            for fi in 0..m.facets.len() {
                c.evaluate(&m, fi, &q);
                if !c.hot[fi] {
                    for p in 0..2 {
                        let (j, _) = m.eval_point(&q, &m.facets[fi], p);
                        let l = &m.facets[fi].pts[p].law;
                        assert!((j.dn.max(0.0).powi(2) + l.gamma * j.dt * j.dt).sqrt() < l.delta0);
                    }
                }
            }
        }
        assert!(c.stats.promoted_strain + c.stats.promoted_geometry > 0);
        assert_eq!(m.dissipated, 0.0); // monitoring never commits material history
    }
    #[test]
    fn all_hot_carrier_matches_unmodified_world() {
        use crate::engine::{
            scenes,
            world::{Open0, Params, World},
        };
        let sc = scenes::wall_impact(1, 0.25, 2.0, 1.0, 1.5, 50.0);
        let mut p = Params::new(0.001, 1e9);
        p.open0 = Open0::All;
        p.rereduce = false;
        let mut full = World::new(sc.clone(), p.clone());
        let mut masked = World::new(sc, p);
        masked.enable_carrier(Config::default());
        masked.carrier.as_mut().unwrap().hot.fill(true);
        for _ in 0..3 {
            assert!(full.macro_step(0.003));
            assert!(masked.macro_step(0.003));
        }
        assert_eq!(full.q, masked.q);
        assert_eq!(full.v, masked.v);
        assert_eq!(full.sc.m.dissipated, masked.sc.m.dissipated);
        assert_eq!(full.energies(), masked.energies());
        assert_eq!(masked.led.d_model, 0.0);
    }
    #[test]
    fn moving_linearization_preserves_strain_under_rigid_rotation() {
        let mut m = fixture();
        let fi = 0;
        let f = &m.facets[fi];
        m.q[f.a][0] += 0.1 * f.pts[0].law.delta0 * f.n0.x;
        m.q[f.a][1] += 0.1 * f.pts[0].law.delta0 * f.n0.y;
        let mut c = Carrier::new(
            &m,
            &m.q,
            Config {
                rebase: true,
                ..Config::default()
            },
        );
        let reference: [Jump; 2] = std::array::from_fn(|p| m.eval_point(&m.q, &m.facets[fi], p).0);
        let rest = m.facets[fi].pts[0].rest;
        for k in 1..300 {
            let mut q = m.q.clone();
            let angle = k as f64 * 0.002;
            for x in &mut q {
                let v = V2::new(x[0], x[1]).rot(angle);
                x[0] = v.x;
                x[1] = v.y;
                x[2] += angle;
            }
            let (ev, _) = c.evaluate(&m, fi, &q);
            assert!(!c.hot[fi], "rigid rotation activated nonlinear damage");
            for p in 0..2 {
                let err = ((ev[p].0.dn - reference[p].dn).powi(2)
                    + (ev[p].0.dt - reference[p].dt).powi(2))
                .sqrt();
                assert!(err <= 0.051 * m.facets[fi].pts[p].law.delta0);
            }
        }
        assert!(c.stats.refresh_points > 0);
        assert_eq!(m.facets[fi].pts[0].rest, rest);
        assert_eq!(m.dissipated, 0.0);
    }
    #[test]
    fn promotion_does_not_book_damage_as_representation_energy() {
        let m = fixture();
        let fi = 0;
        let f = &m.facets[fi];
        let mut c = Carrier::new(&m, &m.q, Config::default());
        let mut q = m.q.clone();
        q[f.b][0] += 1.2 * f.pts[0].law.delta0 * f.n0.x;
        q[f.b][1] += 1.2 * f.pts[0].law.delta0 * f.n0.y;
        let (ev, de) = c.evaluate(&m, fi, &q);
        assert!(c.hot[fi]);
        assert!(ev[0].1.d > 0.0);
        // With translation alone the affine and corotated *undamaged* energies
        // agree. Subsequent physical damage must be charged only to D, once.
        assert!(de.abs() < 1e-8, "{de}");
        assert_eq!(m.dissipated, 0.0);
    }
}
