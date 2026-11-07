//! 2D RBSN model: rigid Voronoi cells (DOFs at centroids), two-point facets,
//! ground/plate bodies for supports and loading. All lengths in metres, SI units.

use crate::kin::{jump, Jump};
use crate::law::{LawEval, LawKind, PointLaw};
use geometry::polygon::segments_intersect;
use geometry::voronoi2d::{EdgeTag, Tessellation};
use geometry::V2;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Material {
    pub e: f64,
    pub sigma_t: f64,
    pub gc: f64,
    pub density: f64,
    /// k_t / k_n (1 → ν = 0 uniformity).
    pub gamma: f64,
    /// Plane-stress thickness b.
    pub thickness: f64,
}

impl Material {
    /// Reference concrete of spec §3.4.
    pub fn concrete(thickness: f64) -> Self {
        Self { e: 30e9, sigma_t: 3e6, gc: 100.0, density: 2400.0, gamma: 1.0, thickness }
    }
    pub fn lch(&self) -> f64 {
        self.e * self.gc / (self.sigma_t * self.sigma_t)
    }
    pub fn wave_speed(&self) -> f64 {
        (self.e / self.density).sqrt()
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct LawOptions {
    pub kind: LawKind,
    /// Crack-band strength cap σ_eff = min(σ_t, η sqrt(2 E G_c/ℓ)); None = off.
    pub crack_band_eta: Option<f64>,
    pub d_max: f64,
}

impl LawOptions {
    pub fn elastic() -> Self {
        Self { kind: LawKind::Elastic, crack_band_eta: None, d_max: 1.0 }
    }
    pub fn damage() -> Self {
        Self { kind: LawKind::Damage, crack_band_eta: None, d_max: 1.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum BodyKind {
    Cell,
    Ground,
    Plate,
}

#[derive(Clone, Debug)]
pub struct Body {
    pub kind: BodyKind,
    /// Reference centroid (DOF reference point).
    pub x0: V2,
    pub generator: V2,
    pub mass: f64,
    /// Polar inertia about the centroid.
    pub inertia: f64,
    pub fixed: [bool; 3],
    /// Reference polygon (absolute coordinates); empty for ground.
    pub poly: Vec<V2>,
}

#[derive(Clone, Copy, Debug)]
pub struct FacetPoint {
    pub arm_a: V2,
    pub arm_b: V2,
    /// Offset along t0 from the facet centroid.
    pub s: f64,
    pub law: PointLaw,
    /// Committed history (only written by `commit`).
    pub kappa_c: f64,
    pub rest: V2,
}

#[derive(Clone, Debug)]
pub struct Facet {
    pub a: usize,
    pub b: usize,
    pub pc: V2,
    pub n0: V2,
    pub t0: V2,
    pub len: f64,
    /// Generator distance ℓ_e (mirror distance for boundary facets).
    pub h: f64,
    pub pts: [FacetPoint; 2],
    pub active: bool,
    /// 0 = interior; otherwise a user tag (support / plate id).
    pub tag: u32,
    pub ends: [V2; 2],
    /// Frame weight w of the corotated jump (½ cell–cell, 1 cell–support/plate).
    pub frame_w: f64,
}

impl Facet {
    pub fn area(&self, b: f64) -> f64 {
        self.len * b
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct FacetForce {
    /// Resultant normal force (tension +), shear force, moment about the facet centroid.
    pub n: f64,
    pub v: f64,
    pub m: f64,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub mat: Material,
    pub law_opts: LawOptions,
    pub bodies: Vec<Body>,
    pub facets: Vec<Facet>,
    /// Current configuration (x, y, θ) per body.
    pub q: Vec<[f64; 3]>,
    /// Dead loads (force x, force y, moment) per body.
    pub loads: Vec<[f64; 3]>,
    /// Proportional load pattern (scaled by a load factor where used).
    pub pattern: Vec<[f64; 3]>,
    /// Cumulative committed fracture dissipation [J].
    pub dissipated: f64,
}

pub struct BoundaryAttach {
    /// Domain edge index.
    pub edge: usize,
    /// Only edges whose midpoint lies in [lo, hi] along the domain edge's dominant axis.
    pub range: Option<(f64, f64)>,
    /// Body to connect to (ground or plate index).
    pub body: usize,
    /// false → normal-only (roller): tangential stiffness 0.
    pub shear: bool,
    pub tag: u32,
}

impl Model {
    pub fn point_law(&self, len: f64, h: f64, shear: bool) -> PointLaw {
        let m = &self.mat;
        let ap = 0.5 * len * m.thickness;
        let kn = m.e * ap / h;
        let sig = match self.law_opts.crack_band_eta {
            Some(eta) => m.sigma_t.min(eta * (2.0 * m.e * m.gc / h).sqrt()),
            None => m.sigma_t,
        };
        let delta0 = sig * h / m.e;
        let wc = 2.0 * m.gc / sig;
        if self.law_opts.kind == LawKind::Damage {
            assert!(
                delta0 < wc,
                "facet violates ℓ_cb < 2 l_ch (ℓ={h}, 2l_ch={}); enable the crack-band option",
                2.0 * m.lch()
            );
        }
        let gamma = if shear { m.gamma } else { 0.0 };
        PointLaw { kind: self.law_opts.kind, kn, kt: gamma * kn, gamma, delta0, wc, d_max: self.law_opts.d_max }
    }

    fn make_facet(&self, a: usize, b: usize, p0: V2, p1: V2, n0: V2, h: f64, shear: bool, tag: u32) -> Facet {
        let len = (p1 - p0).norm();
        let pc = (p0 + p1) * 0.5;
        let t0 = n0.perp();
        let law = self.point_law(len, h, shear);
        let s = len / (2.0 * 3f64.sqrt());
        let xa = self.bodies[a].x0;
        let xb = self.bodies[b].x0;
        let mk = |sg: f64| {
            let p = pc + t0 * sg;
            FacetPoint { arm_a: p - xa, arm_b: p - xb, s: sg, law, kappa_c: 0.0, rest: V2::ZERO }
        };
        let frame_w = if self.bodies[b].kind == BodyKind::Cell { 0.5 } else { 1.0 };
        Facet { a, b, pc, n0, t0, len, h, pts: [mk(-s), mk(s)], active: true, tag, ends: [p0, p1], frame_w }
    }

    /// Build from a tessellation. Body 0..ncells are cells; body `ncells` is the ground.
    /// Facets crossing any `notches` segment are not created.
    pub fn from_tessellation(t: &Tessellation, mat: Material, law_opts: LawOptions, notches: &[(V2, V2)]) -> Model {
        let mut bodies: Vec<Body> = t
            .cells
            .iter()
            .map(|c| Body {
                kind: BodyKind::Cell,
                x0: c.props.centroid,
                generator: c.seed,
                mass: mat.density * c.props.area * mat.thickness,
                inertia: mat.density * c.props.polar * mat.thickness,
                fixed: [false; 3],
                poly: c.verts.clone(),
            })
            .collect();
        bodies.push(Body {
            kind: BodyKind::Ground,
            x0: V2::ZERO,
            generator: V2::ZERO,
            mass: 0.0,
            inertia: 0.0,
            fixed: [true; 3],
            poly: Vec::new(),
        });
        let nb = bodies.len();
        let mut m = Model {
            mat,
            law_opts,
            bodies,
            facets: Vec::new(),
            q: Vec::new(),
            loads: vec![[0.0; 3]; nb],
            pattern: vec![[0.0; 3]; nb],
            dissipated: 0.0,
        };
        for f in &t.facets {
            if notches.iter().any(|&(c, d)| segments_intersect(f.p0, f.p1, c, d)) {
                continue;
            }
            let ga = t.cells[f.a].seed;
            let gb = t.cells[f.b].seed;
            let n0 = (gb - ga).normalized();
            let h = (gb - ga).norm();
            // Orient p0→p1 along t0 = perp(n0) (irrelevant for symmetric points).
            let fac = m.make_facet(f.a, f.b, f.p0, f.p1, n0, h, true, 0);
            m.facets.push(fac);
        }
        m.reset_q();
        m
    }

    pub fn ground(&self) -> usize {
        self.bodies.iter().position(|b| b.kind == BodyKind::Ground).unwrap()
    }

    pub fn reset_q(&mut self) {
        self.q = self.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
        self.loads.resize(self.bodies.len(), [0.0; 3]);
        self.pattern.resize(self.bodies.len(), [0.0; 3]);
    }

    /// Add a rigid plate body (prescribed or free) with reference point `x0`.
    pub fn add_plate(&mut self, x0: V2, mass: f64, inertia: f64, fixed: [bool; 3]) -> usize {
        self.bodies.push(Body { kind: BodyKind::Plate, x0, generator: x0, mass, inertia, fixed, poly: Vec::new() });
        self.q.push([x0.x, x0.y, 0.0]);
        self.loads.push([0.0; 3]);
        self.pattern.push([0.0; 3]);
        self.bodies.len() - 1
    }

    /// Connect cells' boundary edges on a domain edge to a ground/plate body.
    /// Normal points from the cell into the support; ℓ_e = distance from the
    /// generator to the support line (the material between generator and a *rigid*
    /// support; the mirror distance 2d would be right only for a symmetry plane).
    pub fn attach_boundary(&mut self, t: &Tessellation, att: &BoundaryAttach) -> usize {
        let d0 = t.domain[att.edge];
        let d1 = t.domain[(att.edge + 1) % t.domain.len()];
        let dir = (d1 - d0).normalized();
        let out = V2::new(dir.y, -dir.x); // outward for CCW domain
        let mut added = 0;
        for (i, c) in t.cells.iter().enumerate() {
            let k = c.verts.len();
            for e in 0..k {
                if c.tags[e] != EdgeTag::Boundary(att.edge) {
                    continue;
                }
                let p0 = c.verts[e];
                let p1 = c.verts[(e + 1) % k];
                let mid = (p0 + p1) * 0.5;
                if let Some((lo, hi)) = att.range {
                    let s = if dir.x.abs() > dir.y.abs() { mid.x } else { mid.y };
                    if s < lo || s > hi {
                        continue;
                    }
                }
                let h = (c.seed - d0).dot(-out).abs();
                let f = self.make_facet(i, att.body, p0, p1, out, h, att.shear, att.tag);
                self.facets.push(f);
                added += 1;
            }
        }
        added
    }

    /// Like `attach_boundary`, but a ranged support covers exactly [lo, hi]: each
    /// boundary edge is clipped to the interval (along the domain edge's dominant axis)
    /// and the facet (points, area, arms) is built on the clipped segment. The
    /// midpoint rule of `attach_boundary` makes the supported length depend on the
    /// tessellation; this one does not.
    pub fn attach_boundary_exact(&mut self, t: &Tessellation, att: &BoundaryAttach) -> usize {
        let d0 = t.domain[att.edge];
        let d1 = t.domain[(att.edge + 1) % t.domain.len()];
        let dir = (d1 - d0).normalized();
        let out = V2::new(dir.y, -dir.x);
        let mut added = 0;
        for (i, c) in t.cells.iter().enumerate() {
            let k = c.verts.len();
            for e in 0..k {
                if c.tags[e] != EdgeTag::Boundary(att.edge) {
                    continue;
                }
                let Some((p0, p1)) = clip_to_range(c.verts[e], c.verts[(e + 1) % k], dir, att.range) else { continue };
                let h = (c.seed - d0).dot(-out).abs();
                let f = self.make_facet(i, att.body, p0, p1, out, h, att.shear, att.tag);
                self.facets.push(f);
                added += 1;
            }
        }
        added
    }

    pub fn add_gravity(&mut self, g: f64) {
        for (i, b) in self.bodies.iter().enumerate() {
            if b.kind == BodyKind::Cell {
                self.loads[i][1] -= b.mass * g;
            }
        }
    }

    /// Evaluate one facet point at configuration q with committed history.
    #[inline]
    pub fn eval_point(&self, q: &[[f64; 3]], f: &Facet, p: usize) -> (Jump, LawEval) {
        let pt = &f.pts[p];
        let j = jump(q[f.a], q[f.b], pt.arm_a, pt.arm_b, f.n0, f.t0, pt.rest, f.frame_w);
        let e = pt.law.eval(j.dn, j.dt, pt.kappa_c);
        (j, e)
    }

    pub fn facet_force(&self, q: &[[f64; 3]], fi: usize) -> FacetForce {
        let f = &self.facets[fi];
        if !f.active {
            return FacetForce::default();
        }
        let mut r = FacetForce::default();
        for p in 0..2 {
            let (_, e) = self.eval_point(q, f, p);
            r.n += e.fnorm;
            r.v += e.ftan;
            r.m += f.pts[p].s * e.fnorm;
        }
        r
    }

    pub fn facet_forces(&self, q: &[[f64; 3]]) -> Vec<FacetForce> {
        (0..self.facets.len()).map(|i| self.facet_force(q, i)).collect()
    }

    /// Stored elastic energy at q (trial history from committed state).
    pub fn elastic_energy(&self, q: &[[f64; 3]]) -> f64 {
        let mut u = 0.0;
        for f in self.facets.iter().filter(|f| f.active) {
            for p in 0..2 {
                u += self.eval_point(q, f, p).1.psi;
            }
        }
        u
    }

    /// Potential of the dead loads (−f·(q − q_ref)).
    pub fn load_potential(&self, q: &[[f64; 3]], lf: f64) -> f64 {
        let mut v = 0.0;
        for (i, b) in self.bodies.iter().enumerate() {
            let d = [q[i][0] - b.x0.x, q[i][1] - b.x0.y, q[i][2]];
            for k in 0..3 {
                v -= (self.loads[i][k] + lf * self.pattern[i][k]) * d[k];
            }
        }
        v
    }

    /// Accept the step: κ_c ← max(κ_c, δ_eq(q)); returns the dissipation increment.
    pub fn commit(&mut self, q: &[[f64; 3]]) -> f64 {
        let mut dd = 0.0;
        for fi in 0..self.facets.len() {
            if !self.facets[fi].active {
                continue;
            }
            for p in 0..2 {
                let (_, e) = self.eval_point(q, &self.facets[fi], p);
                let pt = &mut self.facets[fi].pts[p];
                if e.kappa > pt.kappa_c {
                    dd += pt.law.dissipation(e.kappa) - pt.law.dissipation(pt.kappa_c);
                    pt.kappa_c = e.kappa;
                }
            }
        }
        self.dissipated += dd;
        dd
    }

    pub fn max_damage(&self) -> f64 {
        self.facets
            .iter()
            .filter(|f| f.active)
            .flat_map(|f| f.pts.iter())
            .map(|p| p.law.damage(p.kappa_c))
            .fold(0.0, f64::max)
    }

    pub fn n_cells(&self) -> usize {
        self.bodies.iter().filter(|b| b.kind == BodyKind::Cell).count()
    }

    /// Free-DOF numbering: idx[body][k] = Some(global) if free.
    pub fn dof_map(&self) -> (Vec<[Option<usize>; 3]>, usize) {
        let mut n = 0;
        let map = self
            .bodies
            .iter()
            .map(|b| {
                let mut r = [None; 3];
                for k in 0..3 {
                    if !b.fixed[k] {
                        r[k] = Some(n);
                        n += 1;
                    }
                }
                r
            })
            .collect();
        (map, n)
    }

    /// Adjacency of bodies through active facets.
    pub fn neighbours(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.bodies.len()];
        for (fi, f) in self.facets.iter().enumerate() {
            if f.active {
                adj[f.a].push(fi);
                adj[f.b].push(fi);
            }
        }
        adj
    }

    pub fn is_free(&self, i: usize) -> bool {
        self.bodies[i].fixed.iter().any(|&f| !f)
    }

    /// Range of ℓ_e / l_ch over active interior facets.
    pub fn band_ratio(&self) -> (f64, f64) {
        let lch = self.mat.lch();
        let mut lo = f64::INFINITY;
        let mut hi: f64 = 0.0;
        for f in self.facets.iter().filter(|f| f.active) {
            lo = lo.min(f.h / lch);
            hi = hi.max(f.h / lch);
        }
        (lo, hi)
    }
}

/// Part of segment p0→p1 whose coordinate along the dominant axis of `dir` lies in
/// `range` (None = the whole segment). None if the overlap has zero length.
pub fn clip_to_range(p0: V2, p1: V2, dir: V2, range: Option<(f64, f64)>) -> Option<(V2, V2)> {
    let Some((lo, hi)) = range else { return Some((p0, p1)) };
    let s = |p: V2| if dir.x.abs() > dir.y.abs() { p.x } else { p.y };
    let (s0, s1) = (s(p0), s(p1));
    if s1 == s0 {
        return None;
    }
    let t0 = ((lo - s0) / (s1 - s0)).clamp(0.0, 1.0);
    let t1 = ((hi - s0) / (s1 - s0)).clamp(0.0, 1.0);
    let (ta, tb) = (t0.min(t1), t0.max(t1));
    if tb - ta <= 1e-12 {
        return None;
    }
    Some((p0 + (p1 - p0) * ta, p0 + (p1 - p0) * tb))
}
