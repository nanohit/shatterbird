//! Phase 0b / 2 — coarse correction inside the physical step (shatterbird-phase0b-step2.md
//! §4–6; pre-registration and manifest: docs/phase0b/two-level.md §4).
//!
//! Linear step systems A u = b, A = μM + K, on contract meshes; one-level controls
//! (coloured block GS, PCG–block-Jacobi) and two-level cycles (piecewise-rigid or
//! smoothed aggregation, aggregates of 8 / 16 cells, block-GS or Chebyshev smoothing;
//! stationary or as PCG preconditioner), all against a sparse direct reference with
//! the same matrix and the same force map. Accuracy targets: facet forces, energy
//! norm, support reaction (force and moment separately), SB-IL1 onset indicator χ.

#![recursion_limit = "512"]

use experiments::{edge_traction_exact, meta, write_json, Csv};
use faer::linalg::solvers::Solve;
use geometry::rng::Rng;
use geometry::rsa::{rsa_box_with, rsa_rect_with};
use geometry::voronoi2d::{rect, voronoi_convex, Tessellation};
use geometry::voronoi3d::voronoi_box;
use geometry::{V2, V3};
use model::rbsn3::{Mat3D, Model3};
use model::{BodyKind, BoundaryAttach, LawOptions, Material, Model};
use serde_json::{json, Value};
use solvers::sparse::{csc_from_triplets, Spd};
use solvers::twolevel::{aggregate, cycle, pcg, rigid_p, smooth_p, AggInput, Aggregates, Coarse, Csr, Precond, Smoother, System, Work};
use model::rbsn3;
use std::time::Instant;

/// v2 (Phase 0b / 3 §1): corrected cut, certified Chebyshev interval, complete work
/// counters, closed loop with a true cut and local χ metrics. v1 results stay in
/// results/phase0b/two-level.
const EXP: &str = "phase0b/two-level-v2";
/// Chebyshev interval on D⁻¹A: the certified upper bound 2 (A ⪯ 2D), ratio 8.
const CH_HI: f64 = 2.0;
const CH_LO: f64 = 0.25;
const CAP1: usize = 1024;
const CAP2: usize = 32;
const TOL: f64 = 0.01;
const CHI_TOL: f64 = 0.01;
const THETA_S: f64 = 0.1;
const CHECKPOINTS: [usize; 11] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

// ======================================================================= problems

/// Linearised facet point (2D): jump Jacobians on free DOFs, ground-side Jacobian for
/// support points, stiffnesses and the SB-IL1 onset scale.
struct Pt {
    facet: usize,
    jn: Vec<(usize, f64)>,
    jt: Vec<(usize, f64)>,
    gn: [f64; 3],
    gt: [f64; 3],
    kn: f64,
    kt: f64,
    s: f64,
    delta0: f64,
    gamma: f64,
}

/// A linear structure in a form common to 2D and 3D: stiffness components with
/// multipliers, mass blocks, cell geometry for aggregation, force/reaction maps.
struct Problem {
    name: String,
    dim: usize,
    k: usize,
    nb: usize,
    n: usize,
    /// per-cell centroid, mass
    pos: Vec<[f64; 3]>,
    cmass: Vec<f64>,
    ell: f64,
    /// interior cell–cell adjacency: (a, b, facet)
    edges: Vec<(usize, usize, usize)>,
    nfacets: usize,
    /// stiffness multiplier per facet (1 = intact)
    eta: Vec<f64>,
    mass_t: Vec<(usize, usize, f64)>,
    load: Vec<f64>,
    // 2D
    pts: Vec<Pt>,
    support_facet: Vec<bool>,
    facet_pc: Vec<[f64; 3]>,
    // 3D: per component (dofs/values, k, facet, family, support, ground Jacobian)
    comps3: Vec<(Vec<(usize, f64)>, f64, usize, usize, bool, [f64; 6])>,
    base_centre: [f64; 3],
    width: f64,
    height: f64,
    /// generator distance ℓ_e per facet
    facet_ell: Vec<f64>,
}

fn contract_seeds2(lo: V2, hi: V2, dmin: f64, seed: u64) -> Vec<V2> {
    let keep = move |p: V2| p.y - lo.y >= 0.5 * dmin;
    rsa_rect_with(lo, hi, dmin, &mut Rng::new(seed), 20_000, &keep, Vec::new())
}

/// 2D block/wall on the contract mesh, full base support, elastic.
fn problem2(name: &str, w: f64, h: f64, dmin: f64, seed: u64, thickness: f64, loads: &[(Option<(f64, f64)>, V2)]) -> (Problem, Model, Tessellation) {
    let lo = V2::new(0.0, 0.0);
    let hi = V2::new(w, h);
    let seeds = contract_seeds2(lo, hi, dmin, seed);
    let t = voronoi_convex(&seeds, &rect(lo, hi));
    assert!(t.max_mismatch < 1e-9 * (hi - lo).norm());
    let mut m = Model::from_tessellation(&t, Material::concrete(thickness), LawOptions::elastic(), &[]);
    let g = m.ground();
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: None, body: g, shear: true, tag: 1 });
    for &(range, tr) in loads {
        edge_traction_exact(&mut m, &t, 2, range, tr, false);
    }
    let (idx, n) = m.dof_map();
    let nb = m.n_cells();
    assert_eq!(n, 3 * nb, "cells must carry all DOFs");
    let q0: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
    let mut pts = Vec::new();
    let mut edges = Vec::new();
    let mut support_facet = vec![false; m.facets.len()];
    let mut facet_pc = Vec::new();
    for (fi, f) in m.facets.iter().enumerate() {
        facet_pc.push([f.pc.x, f.pc.y, 0.0]);
        let ground = m.bodies[f.b].kind != BodyKind::Cell;
        support_facet[fi] = ground;
        if !ground {
            edges.push((f.a, f.b, fi));
        }
        for p in 0..2 {
            let (j, _) = m.eval_point(&q0, f, p);
            let mut jn = Vec::new();
            let mut jt = Vec::new();
            for r in 0..3 {
                if let Some(gi) = idx[f.a][r] {
                    jn.push((gi, j.jna[r]));
                    jt.push((gi, j.jta[r]));
                }
                if let Some(gi) = idx[f.b][r] {
                    jn.push((gi, j.jnb[r]));
                    jt.push((gi, j.jtb[r]));
                }
            }
            let law = &f.pts[p].law;
            pts.push(Pt { facet: fi, jn, jt, gn: if ground { j.jnb } else { [0.0; 3] }, gt: if ground { j.jtb } else { [0.0; 3] }, kn: law.kn, kt: law.kt, s: f.pts[p].s, delta0: law.delta0, gamma: law.gamma });
        }
    }
    let mut mass_t = Vec::new();
    let mut pos = Vec::new();
    let mut cmass = Vec::new();
    let mut load = vec![0.0; n];
    for i in 0..nb {
        let b = &m.bodies[i];
        mass_t.push((3 * i, 3 * i, b.mass));
        mass_t.push((3 * i + 1, 3 * i + 1, b.mass));
        mass_t.push((3 * i + 2, 3 * i + 2, b.inertia));
        pos.push([b.x0.x, b.x0.y, 0.0]);
        cmass.push(b.mass);
        for c in 0..3 {
            load[3 * i + c] = m.loads[i][c];
        }
    }
    let area: f64 = t.cells.iter().map(|c| c.props.area).sum();
    let nf = m.facets.len();
    let p = Problem {
        name: name.into(),
        dim: 2,
        k: 3,
        nb,
        n,
        pos,
        cmass,
        ell: (area / nb as f64).sqrt(),
        edges,
        nfacets: nf,
        eta: vec![1.0; nf],
        mass_t,
        load,
        pts,
        support_facet,
        facet_pc,
        comps3: Vec::new(),
        base_centre: [0.5 * w, 0.0, 0.0],
        width: w,
        height: h,
        facet_ell: m.facets.iter().map(|f| f.h).collect(),
    };
    (p, m, t)
}

/// 3D block on the contract mesh (rbsn3), full bottom support, top traction.
fn problem3(name: &str, ext: V3, dmin: f64, seed: u64, traction: V3) -> Problem {
    let lo = V3::ZERO;
    let keep = move |p: V3| p.z - lo.z >= 0.5 * dmin;
    let seeds = rsa_box_with(lo, ext, dmin, &mut Rng::new(seed), 5000, &keep, Vec::new());
    let cells3 = voronoi_box(&seeds, lo, ext);
    let mut m = Model3::build(&cells3, &[4], Mat3D::concrete());
    m.traction(&cells3, 5, traction);
    let nb = m.cells.len();
    let n = 6 * nb;
    let mut mass_t = Vec::new();
    for (i, c) in m.cells.iter().enumerate() {
        for d in 0..3 {
            mass_t.push((6 * i + d, 6 * i + d, c.mass));
            for e in 0..3 {
                mass_t.push((6 * i + 3 + d, 6 * i + 3 + e, c.inertia[d][e]));
            }
        }
    }
    let mut comps3 = Vec::new();
    let mut fam = vec![0usize; m.facets.len()];
    for c in &m.comps {
        let mut ent: Vec<(usize, f64)> = (0..6).filter(|&q| c.ja[q] != 0.0).map(|q| (6 * c.a + q, c.ja[q])).collect();
        if let Some(b) = c.b {
            ent.extend((0..6).filter(|&q| c.jb[q] != 0.0).map(|q| (6 * b + q, c.jb[q])));
        }
        let f = fam[c.facet];
        fam[c.facet] += 1;
        comps3.push((ent, c.k, c.facet, f, c.b.is_none(), c.jb));
    }
    let edges = m.facets.iter().enumerate().filter_map(|(fi, f)| f.b.map(|b| (f.a, b, fi))).collect();
    let vol: f64 = m.cells.iter().map(|c| c.volume).sum();
    Problem {
        name: name.into(),
        dim: 3,
        k: 6,
        nb,
        n,
        pos: m.cells.iter().map(|c| [c.centroid.x, c.centroid.y, c.centroid.z]).collect(),
        cmass: m.cells.iter().map(|c| c.mass).collect(),
        ell: (vol / nb as f64).cbrt(),
        edges,
        nfacets: m.facets.len(),
        eta: vec![1.0; m.facets.len()],
        mass_t,
        load: m.loads.iter().flatten().copied().collect(),
        pts: Vec::new(),
        support_facet: m.facets.iter().map(|f| f.b.is_none()).collect(),
        facet_pc: m.facets.iter().map(|f| [f.centroid.x, f.centroid.y, f.centroid.z]).collect(),
        comps3,
        base_centre: [0.5 * ext.x, 0.5 * ext.y, 0.0],
        width: ext.x,
        height: ext.z,
        facet_ell: m.facets.iter().map(|f| f.ell).collect(),
    }
}

impl Problem {
    fn k_triplets(&self) -> Vec<(usize, usize, f64)> {
        let mut t = Vec::new();
        let outer = |e: &[(usize, f64)], k: f64, t: &mut Vec<(usize, usize, f64)>| {
            for &(r, vr) in e {
                for &(c, vc) in e {
                    t.push((r, c, k * vr * vc));
                }
            }
        };
        for p in &self.pts {
            let eta = self.eta[p.facet];
            outer(&p.jn, eta * p.kn, &mut t);
            outer(&p.jt, eta * p.kt, &mut t);
        }
        for c in &self.comps3 {
            outer(&c.0, self.eta[c.2] * c.1, &mut t);
        }
        t
    }

    fn matrix(&self, mu: f64) -> Csr {
        let mut t = self.k_triplets();
        t.extend(self.mass_t.iter().map(|&(r, c, v)| (r, c, mu * v)));
        Csr::from_triplets(self.n, self.n, t)
    }

    fn mass_csr(&self) -> Csr {
        Csr::from_triplets(self.n, self.n, self.mass_t.clone())
    }

    /// Facet resultants per family: 2D (N, V, M); 3D 6 component families.
    fn forces(&self, u: &[f64]) -> Vec<Vec<f64>> {
        let dot = |e: &[(usize, f64)]| e.iter().map(|&(d, v)| v * u[d]).sum::<f64>();
        if self.dim == 2 {
            let mut f = vec![vec![0.0; self.nfacets]; 3];
            for p in &self.pts {
                let eta = self.eta[p.facet];
                let fnn = eta * p.kn * dot(&p.jn);
                let ft = eta * p.kt * dot(&p.jt);
                f[0][p.facet] += fnn;
                f[1][p.facet] += ft;
                f[2][p.facet] += p.s * fnn;
            }
            f
        } else {
            let mut f = vec![vec![0.0; self.nfacets]; 6];
            for c in &self.comps3 {
                f[c.3][c.2] = self.eta[c.2] * c.1 * dot(&c.0);
            }
            f
        }
    }

    /// Support reaction on the ground: force vector and moment about the base centre.
    fn reaction(&self, u: &[f64]) -> ([f64; 3], [f64; 3]) {
        let dot = |e: &[(usize, f64)]| e.iter().map(|&(d, v)| v * u[d]).sum::<f64>();
        let (mut fr, mut mo) = ([0.0; 3], [0.0; 3]);
        if self.dim == 2 {
            // ground reference point is the origin; shift the moment to the base centre
            for p in self.pts.iter().filter(|p| self.support_facet[p.facet]) {
                let eta = self.eta[p.facet];
                let fnn = eta * p.kn * dot(&p.jn);
                let ft = eta * p.kt * dot(&p.jt);
                for d in 0..2 {
                    fr[d] -= p.gn[d] * fnn + p.gt[d] * ft;
                }
                mo[2] -= p.gn[2] * fnn + p.gt[2] * ft;
            }
            let c = self.base_centre;
            mo[2] -= c[0] * fr[1] - c[1] * fr[0];
        } else {
            for c in self.comps3.iter().filter(|c| c.4) {
                let f = -self.eta[c.2] * c.1 * dot(&c.0);
                let pc = self.facet_pc[c.2];
                let r = [pc[0] - self.base_centre[0], pc[1] - self.base_centre[1], pc[2] - self.base_centre[2]];
                let fv = [f * c.5[0], f * c.5[1], f * c.5[2]];
                for d in 0..3 {
                    fr[d] += fv[d];
                    mo[d] += f * c.5[3 + d];
                }
                mo[0] += r[1] * fv[2] - r[2] * fv[1];
                mo[1] += r[2] * fv[0] - r[0] * fv[2];
                mo[2] += r[0] * fv[1] - r[1] * fv[0];
            }
        }
        (fr, mo)
    }

    /// SB-IL1 onset indicator per point of intact facets (2D); empty in 3D.
    fn chi(&self, u: &[f64]) -> Vec<f64> {
        let dot = |e: &[(usize, f64)]| e.iter().map(|&(d, v)| v * u[d]).sum::<f64>();
        self.pts
            .iter()
            .map(|p| {
                if self.eta[p.facet] != 1.0 {
                    return f64::NAN;
                }
                let dn = dot(&p.jn).max(0.0);
                let dt = dot(&p.jt);
                (dn * dn + p.gamma * dt * dt).sqrt() / p.delta0
            })
            .collect()
    }

    fn agg_input<'a>(&'a self, target: usize, strength_aware: bool, edges: &'a mut Vec<(usize, usize, bool)>) -> AggInput<'a> {
        edges.clear();
        edges.extend(self.edges.iter().map(|&(a, b, f)| (a, b, !strength_aware || self.eta[f] >= THETA_S)));
        AggInput { pos: &self.pos, mass: &self.cmass, edges, target, dim: self.dim, ell: self.ell }
    }

    /// v1 selection (kept only for the audit): facets whose segment crosses y = y0. This
    /// is NOT a cut of the cell graph (Phase 0b / 3 §1.2).
    fn band_v1(&self, y0: f64, m: Option<&Model>) -> Vec<usize> {
        match m {
            Some(m) => m.facets.iter().enumerate().filter(|(fi, f)| !self.support_facet[*fi] && (f.ends[0].y - y0) * (f.ends[1].y - y0) < 0.0).map(|x| x.0).collect(),
            None => Vec::new(),
        }
    }

    /// Diagnostic cut: each cell gets the layer = number of planes at or below its centroid
    /// height (y in 2D, z in 3D; a centroid exactly on a plane goes up); every facet that
    /// joins cells of different layers is cut.
    fn cut(&self, planes: &[f64]) -> Vec<usize> {
        let ax = if self.dim == 2 { 1 } else { 2 };
        let layer: Vec<usize> = self.pos.iter().map(|x| planes.iter().filter(|&&z| x[ax] >= z).count()).collect();
        let mut f: Vec<usize> = self.edges.iter().filter(|&&(a, b, _)| layer[a] != layer[b]).map(|e| e.2).collect();
        f.sort_unstable();
        f
    }

    /// Connected components of the cell graph without `removed` facets, and whether each
    /// touches a support.
    fn components(&self, removed: &[usize]) -> (Vec<usize>, Vec<(usize, bool)>) {
        let mut parent: Vec<usize> = (0..self.nb).collect();
        fn find(p: &mut Vec<usize>, i: usize) -> usize {
            let mut r = i;
            while p[r] != r {
                r = p[r];
            }
            let mut j = i;
            while p[j] != r {
                let nx = p[j];
                p[j] = r;
                j = nx;
            }
            r
        }
        let rem: std::collections::HashSet<usize> = removed.iter().copied().collect();
        for &(a, b, f) in &self.edges {
            if !rem.contains(&f) {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
        let roots: Vec<usize> = (0..self.nb).map(|i| find(&mut parent, i)).collect();
        let mut ids: Vec<usize> = roots.clone();
        ids.sort_unstable();
        ids.dedup();
        let comp: Vec<usize> = roots.iter().map(|r| ids.binary_search(r).unwrap()).collect();
        let mut info = vec![(0usize, false); ids.len()];
        for i in 0..self.nb {
            info[comp[i]].0 += 1;
        }
        // support facets: 2D points / 3D components on ground carry the cell index in `a`
        let mut sup_cells = std::collections::HashSet::new();
        for p in self.pts.iter().filter(|p| self.support_facet[p.facet]) {
            if let Some(&(d, _)) = p.jn.first() {
                sup_cells.insert(d / self.k);
            }
        }
        for c in self.comps3.iter().filter(|c| c.4) {
            if let Some(&(d, _)) = c.0.first() {
                sup_cells.insert(d / self.k);
            }
        }
        for i in sup_cells {
            info[comp[i]].1 = true;
        }
        (comp, info)
    }

    /// Audit of a cut: components (size, supported), and the rigid motions of every free
    /// component as null vectors of K with the cut facets at η = 0.
    fn audit_cut(&mut self, facets: &[usize]) -> Value {
        let (comp, info) = self.components(facets);
        let saved = self.eta.clone();
        for &f in facets {
            self.eta[f] = 0.0;
        }
        let k = self.matrix(0.0);
        self.eta = saved;
        let kinf = (0..k.nr).map(|r| (k.ptr[r]..k.ptr[r + 1]).map(|q| k.val[q].abs()).sum::<f64>()).fold(0.0, f64::max);
        let mut null_rel: f64 = 0.0;
        for (ci, &(_, sup)) in info.iter().enumerate() {
            if sup {
                continue;
            }
            let cells: Vec<usize> = (0..self.nb).filter(|&i| comp[i] == ci).collect();
            let mut cen = [0.0; 3];
            for &i in &cells {
                for d in 0..3 {
                    cen[d] += self.pos[i][d] / cells.len() as f64;
                }
            }
            let nmodes = if self.dim == 2 { 3 } else { 6 };
            for mode in 0..nmodes {
                let mut r = vec![0.0; self.n];
                for &i in &cells {
                    let x = [self.pos[i][0] - cen[0], self.pos[i][1] - cen[1], self.pos[i][2] - cen[2]];
                    if self.dim == 2 {
                        match mode {
                            0 => r[3 * i] = 1.0,
                            1 => r[3 * i + 1] = 1.0,
                            _ => {
                                r[3 * i] = -x[1];
                                r[3 * i + 1] = x[0];
                                r[3 * i + 2] = 1.0;
                            }
                        }
                    } else if mode < 3 {
                        r[6 * i + mode] = 1.0;
                    } else {
                        let mut e = [0.0; 3];
                        e[mode - 3] = 1.0;
                        let u = [e[1] * x[2] - e[2] * x[1], e[2] * x[0] - e[0] * x[2], e[0] * x[1] - e[1] * x[0]];
                        r[6 * i..6 * i + 3].copy_from_slice(&u);
                        r[6 * i + mode] = 1.0;
                    }
                }
                let mut kr = vec![0.0; self.n];
                k.matvec(&r, &mut kr);
                let rinf = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                null_rel = null_rel.max(kr.iter().fold(0.0f64, |m, v| m.max(v.abs())) / (kinf * rinf));
            }
        }
        json!({"cut_facets": facets.len(), "components": info.len(), "sizes": info.iter().map(|x| x.0).collect::<Vec<_>>(),
            "supported": info.iter().map(|x| x.1).collect::<Vec<_>>(), "free_parts": info.iter().filter(|x| !x.1).count(), "free_rigid_null_rel": null_rel})
    }
}

// ======================================================================= reference & metrics

struct Reference {
    u: Vec<f64>,
    forces: Vec<Vec<f64>>,
    fnorm: Vec<f64>,
    react: ([f64; 3], [f64; 3]),
    chi: Vec<f64>,
    chi_scale: f64,
    a_norm: f64,
    /// ‖u_ref − u_prev‖_A for routine steps (diagnostic normalisation)
    inc_norm: Option<f64>,
    scaled_residual: f64,
    dense_rel: Option<f64>,
    nnz_l: usize,
    flops: f64,
    t_factor: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    f: f64,
    e: f64,
    rf: f64,
    rm: f64,
    chi_hi: f64,
    chi_all: f64,
    e_inc: f64,
    worst_facet: usize,
}

impl Metrics {
    fn ok(&self) -> bool {
        self.f <= TOL && self.e <= TOL && self.rf <= TOL && self.rm <= TOL && !(self.chi_hi > CHI_TOL)
    }
    fn json(&self) -> Value {
        json!({"F": self.f, "E": self.e, "Rf": self.rf, "Rm": self.rm, "chi_hi": if self.chi_hi.is_nan() { Value::Null } else { json!(self.chi_hi) }, "chi_all": self.chi_all, "E_inc": self.e_inc, "worst_facet": self.worst_facet})
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn a_norm(a: &Csr, v: &[f64]) -> f64 {
    let mut av = vec![0.0; v.len()];
    a.matvec(v, &mut av);
    av.iter().zip(v).map(|(x, y)| x * y).sum::<f64>().max(0.0).sqrt()
}

fn direct(a: &Csr, b: &[f64]) -> (Vec<f64>, usize, f64, f64) {
    let t0 = Instant::now();
    let m = csc_from_triplets(a.nr, a.triplets());
    let mut s = Spd::analyze(&m);
    s.factor(&m).expect("A SPD");
    let mut u = b.to_vec();
    s.solve(&mut u);
    (u, s.stats.nnz_l, s.stats.flops_factor, t0.elapsed().as_secs_f64())
}

fn reference(p: &Problem, a: &Csr, b: &[f64], u_prev: Option<&[f64]>, chi_scale: Option<f64>) -> Reference {
    let (u, nnz_l, flops, t_factor) = direct(a, b);
    let mut au = vec![0.0; a.nr];
    a.matvec(&u, &mut au);
    let inf = |v: &[f64]| v.iter().fold(0.0f64, |s, x| s.max(x.abs()));
    let anorm_inf = (0..a.nr).map(|r| (a.ptr[r]..a.ptr[r + 1]).map(|q| a.val[q].abs()).sum::<f64>()).fold(0.0, f64::max);
    let res: Vec<f64> = au.iter().zip(b).map(|(x, y)| y - x).collect();
    let scaled_residual = inf(&res) / (anorm_inf * inf(&u) + inf(b));
    let dense_rel = (a.nr <= 1500).then(|| {
        let dm = faer::Mat::<f64>::from_fn(a.nr, a.nr, |_, _| 0.0);
        let mut dm = dm;
        for r in 0..a.nr {
            for q in a.ptr[r]..a.ptr[r + 1] {
                dm[(r, a.idx[q])] = a.val[q];
            }
        }
        let rhs = faer::Mat::<f64>::from_fn(a.nr, 1, |i, _| b[i]);
        let x = dm.partial_piv_lu().solve(&rhs);
        let d: Vec<f64> = (0..a.nr).map(|i| x[(i, 0)] - u[i]).collect();
        norm(&d) / norm(&u)
    });
    let forces = p.forces(&u);
    let fnorm = forces.iter().map(|f| norm(f)).collect();
    let chi = p.chi(&u);
    let cmax = chi.iter().cloned().filter(|x| x.is_finite()).fold(0.0, f64::max);
    let chi_scale = chi_scale.unwrap_or(if cmax > 0.0 { 0.95 / cmax } else { 1.0 });
    let inc_norm = u_prev.map(|up| {
        let d: Vec<f64> = u.iter().zip(up).map(|(a, b)| a - b).collect();
        a_norm(a, &d)
    });
    Reference { a_norm: a_norm(a, &u), react: p.reaction(&u), forces, fnorm, chi, chi_scale, inc_norm, scaled_residual, dense_rel, nnz_l, flops, t_factor, u }
}

/// Reference scale per force family: force families (2D N, V; 3D translations) share
/// the joint force norm; moment families use max(own norm, force norm · ℓ̄/√12), so a
/// family that is ≈ 0 in the reference (uniform states) is not divided by round-off.
fn family_scales(p: &Problem, r: &Reference) -> Vec<f64> {
    let nf = if p.dim == 2 { 2 } else { 3 };
    let fj = r.fnorm[..nf].iter().map(|x| x * x).sum::<f64>().sqrt();
    (0..r.fnorm.len()).map(|f| if f < nf { fj } else { r.fnorm[f].max(fj * p.ell / 12f64.sqrt()) }).collect()
}

fn metrics(p: &Problem, a: &Csr, r: &Reference, u: &[f64]) -> Metrics {
    let f = p.forces(u);
    let sc = family_scales(p, r);
    let mut fe: f64 = 0.0;
    let mut worst = (0.0, 0usize);
    for (fam, (fv, fr)) in f.iter().zip(&r.forces).enumerate() {
        let d: Vec<f64> = fv.iter().zip(fr).map(|(a, b)| a - b).collect();
        if sc[fam] > 0.0 {
            fe = fe.max(norm(&d) / sc[fam]);
            for (i, x) in d.iter().enumerate() {
                let v = x.abs() / sc[fam];
                if v > worst.0 {
                    worst = (v, i);
                }
            }
        }
    }
    let e: Vec<f64> = u.iter().zip(&r.u).map(|(a, b)| a - b).collect();
    let en = a_norm(a, &e);
    let (fr, mo) = p.reaction(u);
    let (rf0, mo0) = r.react;
    let rfn = norm(&rf0);
    let dfr = norm(&[fr[0] - rf0[0], fr[1] - rf0[1], fr[2] - rf0[2]]);
    let mscale = norm(&mo0).max(rfn * p.width);
    let dmo = norm(&[mo[0] - mo0[0], mo[1] - mo0[1], mo[2] - mo0[2]]);
    let (mut chi_hi, mut chi_all) = (f64::NAN, 0.0f64);
    if p.dim == 2 {
        let c = p.chi(u);
        for (x, x0) in c.iter().zip(&r.chi) {
            if x0.is_finite() {
                let d = (x - x0).abs() * r.chi_scale;
                chi_all = chi_all.max(d);
                if x0 * r.chi_scale >= 0.8 {
                    chi_hi = if chi_hi.is_nan() { d } else { chi_hi.max(d) };
                }
            }
        }
    }
    Metrics {
        f: fe,
        e: en / r.a_norm,
        rf: if rfn > 0.0 { dfr / rfn } else { dfr },
        rm: if mscale > 0.0 { dmo / mscale } else { dmo },
        chi_hi,
        chi_all,
        e_inc: r.inc_norm.map(|x| en / x).unwrap_or(f64::NAN),
        worst_facet: worst.1,
    }
}

// ======================================================================= methods

#[derive(Clone, Copy, Debug, PartialEq)]
enum PKind {
    Pr,
    Sa,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Method {
    Gs1,
    PcgJ,
    Tl { p: PKind, s: usize, cheb: bool, pcg: bool },
}

impl Method {
    fn name(&self) -> String {
        match self {
            Method::Gs1 => "GS1".into(),
            Method::PcgJ => "PCG-J".into(),
            Method::Tl { p, s, cheb, pcg } => format!("{}TL-{}-s{}-{}", if *pcg { "P" } else { "" }, if *p == PKind::Pr { "PR" } else { "SA" }, s, if *cheb { "CH" } else { "GS" }),
        }
    }
    fn two_level() -> Vec<Method> {
        let mut v = Vec::new();
        for pcg in [false, true] {
            for p in [PKind::Pr, PKind::Sa] {
                for s in [8, 16] {
                    for cheb in [false, true] {
                        v.push(Method::Tl { p, s, cheb, pcg });
                    }
                }
            }
        }
        v
    }
    fn all() -> Vec<Method> {
        let mut v = vec![Method::Gs1, Method::PcgJ];
        v.extend(Method::two_level());
        v
    }
}

/// Coarse levels of one system: (s, kind) → Coarse, plus setup costs.
struct Levels {
    coarse: Vec<(usize, PKind, Coarse, Value)>,
    lam: f64,
    aggs: Vec<(usize, Aggregates, f64)>,
}

fn build_levels(p: &Problem, sys: &System, strength_aware: bool) -> Levels {
    let lam = (1.05 * sys.lambda_dinv_a()).min(2.0);
    let mut coarse = Vec::new();
    let mut aggs = Vec::new();
    for s in [8usize, 16] {
        let t0 = Instant::now();
        let mut e = Vec::new();
        let agg = aggregate(&p.agg_input(s, strength_aware, &mut e));
        let t_agg = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let ppr = rigid_p(&agg, &p.pos, &p.cmass, p.k, p.ell);
        let t_ppr = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let psa = smooth_p(sys, &ppr, 4.0 / (3.0 * lam));
        let t_psa = t2.elapsed().as_secs_f64();
        for (kind, pm, tp) in [(PKind::Pr, ppr.clone(), t_ppr), (PKind::Sa, psa, t_ppr + t_psa)] {
            let co = Coarse::new(sys, pm).expect("A_c SPD");
            let st = &co.stats;
            let info = json!({"s": s, "kind": format!("{kind:?}"), "n_agg": agg.n, "nc": st.nc, "nnz_P": st.nnz_p, "nnz_Ac": st.nnz_ac, "nnz_Lc": st.nnz_l,
                "flops_factor": st.flops_factor, "t_aggregate": t_agg, "t_P": tp, "t_Ac": st.t_ac, "t_symbolic": st.t_symbolic, "t_numeric": st.t_numeric,
                "mem_bytes": 12 * (st.nnz_p + st.nnz_ac + st.nnz_l)});
            coarse.push((s, kind, co, info));
        }
        aggs.push((s, agg, t_agg));
    }
    Levels { coarse, lam, aggs }
}

struct RunOut {
    method: Method,
    iters: Option<usize>,
    censored: bool,
    work_at_target: Work,
    a_eq_at_target: f64,
    checkpoints: Vec<(usize, Metrics)>,
    initial: Metrics,
    last: Metrics,
    t_apply: f64,
    /// first iteration meeting F, E, R (both parts), χ separately
    first: [Option<usize>; 4],
}

impl RunOut {
    fn note_first(&mut self, it: usize, m: &Metrics) {
        let c = [m.f <= TOL, m.e <= TOL, m.rf <= TOL && m.rm <= TOL, !(m.chi_hi > CHI_TOL)];
        for k in 0..4 {
            if self.first[k].is_none() && c[k] {
                self.first[k] = Some(it);
            }
        }
    }
    fn json(&self) -> Value {
        let w = &self.work_at_target;
        json!({"method": self.method.name(), "iters_to_target": self.iters, "censored": self.censored,
            "work_at_target": {"a_apps": w.a_apps, "block_passes": w.block_passes, "phases": w.phases, "reductions": w.reductions, "coarse_solves": w.coarse_solves, "transfers": w.transfers, "A_equivalents": self.a_eq_at_target},
            "initial": self.initial.json(), "final": self.last.json(), "t_apply_s": self.t_apply,
            "first_iter": {"F": self.first[0], "E": self.first[1], "R": self.first[2], "chi": self.first[3]},
            "checkpoints": self.checkpoints.iter().map(|(k, m)| json!({"iter": k, "m": m.json()})).collect::<Vec<_>>()})
    }
}

/// Run one method from x0. Two-level methods always run CAP2 iterations (errors at
/// every checkpoint); one-level methods stop at max(target, 32) or CAP1.
fn run(p: &Problem, sys: &System, lv: &mut Levels, r: &Reference, b: &[f64], x0: &[f64], method: Method, cap2: usize, min_iters: usize) -> RunOut {
    let mut x = x0.to_vec();
    let x_zero = x0.iter().all(|v| *v == 0.0);
    let initial = metrics(p, &sys.a, r, &x);
    let mut out = RunOut { method, iters: None, censored: true, work_at_target: Work::default(), a_eq_at_target: f64::NAN, checkpoints: Vec::new(), initial, last: initial, t_apply: 0.0, first: [None; 4] };
    out.note_first(0, &initial);
    if initial.ok() {
        out.iters = Some(0);
        out.censored = false;
        out.a_eq_at_target = 0.0;
        if min_iters == 0 {
            return out;
        }
    }
    let mut w = Work::default();
    let nnz_a = sys.a.nnz() as f64;
    let t0 = Instant::now();
    let mut lfac = 0.0;
    let record = |it: usize, x: &[f64], w: &Work, out: &mut RunOut, lfac: f64| -> bool {
        let m = metrics(p, &sys.a, r, x);
        out.last = m;
        out.note_first(it, &m);
        if CHECKPOINTS.contains(&it) {
            out.checkpoints.push((it, m));
        }
        if out.iters.is_none() && m.ok() {
            out.iters = Some(it);
            out.censored = false;
            out.work_at_target = *w;
            out.a_eq_at_target = w.a_apps + w.coarse_solves * lfac;
        }
        out.iters.is_some() && it >= min_iters
    };
    match method {
        Method::Gs1 => {
            for it in 1..=CAP1 {
                sys.gs_sweep(b, &mut x, false, &mut w);
                if record(it, &x, &w, &mut out, 0.0) {
                    break;
                }
            }
        }
        Method::PcgJ => {
            pcg(sys, &mut Precond::Jacobi, b, &mut x, CAP1, None, &mut w, &mut |it, xx, ww| record(it, xx, ww, &mut out, 0.0));
        }
        Method::Tl { p: kind, s, cheb, pcg: use_pcg } => {
            let co = &mut lv.coarse.iter_mut().find(|c| c.0 == s && c.1 == kind).unwrap().2;
            lfac = 4.0 * co.stats.nnz_l as f64 / (2.0 * nnz_a);
            let sm = if cheb { Smoother::Cheb { deg: 2, lo: CH_LO, hi: CH_HI } } else { Smoother::Gs };
            if use_pcg {
                pcg(sys, &mut Precond::TwoLevel(co, sm), b, &mut x, cap2, None, &mut w, &mut |it, xx, ww| record(it, xx, ww, &mut out, lfac));
            } else {
                for it in 1..=cap2 {
                    cycle(sys, co, sm, b, &mut x, x_zero && it == 1, &mut w);
                    if record(it, &x, &w, &mut out, lfac) {
                        break;
                    }
                }
            }
        }
    }
    let _ = lfac;
    out.t_apply = t0.elapsed().as_secs_f64();
    out
}


/// CPU timings of the building blocks without metric evaluation (median of 15 runs):
/// matvec, GS sweep, Chebyshev degree 2, coarse correction and full cycle per coarse
/// space, numeric refactorisation of A_c. CPU time is a CPU measure, not a GPU forecast.
fn time_ops(sys: &System, lv: &mut Levels, b: &[f64]) -> Value {
    let med = |f: &mut dyn FnMut()| {
        let mut t: Vec<f64> = (0..15)
            .map(|_| {
                let t0 = Instant::now();
                f();
                t0.elapsed().as_secs_f64()
            })
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        t[7]
    };
    let n = sys.n;
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut w = Work::default();
    let t_mv = med(&mut || sys.a.matvec(b, &mut y));
    let t_gs = med(&mut || sys.gs_sweep(b, &mut x, false, &mut w));
    let _ = lv.lam;
    let t_ch = med(&mut || solvers::twolevel::chebyshev(sys, b, &mut x, 2, CH_LO, CH_HI, false, &mut w));
    let mut per = Vec::new();
    for (s0, kind, co, _) in lv.coarse.iter_mut() {
        let t_cc = med(&mut || co.correct(b, &mut x, &mut w));
        let t_cyc_gs = med(&mut || cycle(sys, co, Smoother::Gs, b, &mut x, false, &mut w));
        let t_cyc_ch = med(&mut || cycle(sys, co, Smoother::Cheb { deg: 2, lo: CH_LO, hi: CH_HI }, b, &mut x, false, &mut w));
        let t_ref = med(&mut || co.update(sys).unwrap());
        per.push(json!({"s": s0, "kind": format!("{kind:?}"), "t_coarse_correction": t_cc, "t_cycle_GS": t_cyc_gs, "t_cycle_CH": t_cyc_ch, "t_update_Ac_plus_numeric": t_ref}));
    }
    json!({"t_matvec": t_mv, "t_gs_sweep": t_gs, "t_cheb2": t_ch, "coarse": per})
}

// ======================================================================= output

struct Out {
    audits: Vec<Value>,
    runs: Csv,
    dyn_open: Csv,
    dyn_closed: Csv,
    json: Vec<Value>,
}

fn emit(out: &mut Out, case: &str, system: &str, rhs: &str, p: &Problem, sys: &System, lv: &Levels, r: &Reference, res: &[RunOut], extra: Value) {
    for ro in res {
        let get = |k: usize| ro.checkpoints.iter().find(|c| c.0 == k).map(|c| c.1);
        let f = |k: usize, sel: fn(&Metrics) -> f64| get(k).map(|m| format!("{:.4e}", sel(&m))).unwrap_or_default();
        let w = &ro.work_at_target;
        out.runs.row_s(&[
            case.into(),
            system.into(),
            rhs.into(),
            ro.method.name(),
            p.nb.to_string(),
            ro.iters.map(|x| x.to_string()).unwrap_or_default(),
            (ro.censored as u8).to_string(),
            format!("{}", w.a_apps),
            format!("{}", w.phases),
            format!("{}", w.reductions),
            format!("{}", w.coarse_solves),
            format!("{:.4}", ro.a_eq_at_target),
            format!("{:.4e}", ro.initial.f),
            f(1, |m| m.f),
            f(2, |m| m.f),
            f(4, |m| m.f),
            f(8, |m| m.f),
            f(16, |m| m.f),
            f(32, |m| m.f),
            f(8, |m| m.e),
            f(8, |m| m.chi_hi),
            format!("{:.4e}", ro.last.f),
            format!("{:.4e}", ro.last.chi_all),
        ]);
    }
    out.json.push(json!({"case": case, "system": system, "rhs": rhs, "cells": p.nb, "dofs": p.n, "nnz_A": sys.a.nnz(), "colors": sys.colors.len(),
        "lambda_hat_DinvA": lv.lam, "levels": lv.coarse.iter().map(|c| c.3.clone()).collect::<Vec<_>>(),
        "aggregates": lv.aggs.iter().map(|(s, a, _)| json!({"s": s, "n": a.n, "size_min": a.sizes.iter().min(), "size_max": a.sizes.iter().max(),
            "size_mean": p.nb as f64 / a.n as f64, "max_detour": a.max_detour, "reseeds": a.reseeds, "merged": a.merged})).collect::<Vec<_>>(),
        "reference": {"scaled_residual": r.scaled_residual, "dense_rel": r.dense_rel, "nnz_L": r.nnz_l, "flops": r.flops, "t_factor": r.t_factor, "chi_scale": r.chi_scale,
            "chi_hi_points": r.chi.iter().filter(|c| c.is_finite() && *c * r.chi_scale >= 0.8).count()},
        "runs": res.iter().map(|x| x.json()).collect::<Vec<_>>(), "extra": extra}));
    let s: Vec<String> = res.iter().map(|x| format!("{}:{}", x.method.name(), x.iters.map(|v| v.to_string()).unwrap_or("-".into()))).collect();
    eprintln!("{case} | {system} | {rhs}: {}", s.join(" "));
}

fn mu_list(rep: bool) -> Vec<(String, f64)> {
    let mut v = vec![("static".to_string(), 0.0)];
    for hinv in [60.0, 480.0] {
        v.push((format!("IE h=1/{hinv}"), hinv * hinv));
    }
    if rep {
        for hinv in [60.0, 480.0] {
            v.push((format!("BDF2 h=1/{hinv}"), 2.25 * hinv * hinv));
            v.push((format!("Newmark h=1/{hinv}"), 4.0 * hinv * hinv));
        }
    }
    v
}

/// Cold and post-event (band at mid-height, η = 1e-2) systems of one 2D/3D problem.
fn study(out: &mut Out, case: &str, p: &mut Problem, m: Option<&Model>, rep: bool, post_event: bool, methods: &[Method]) {
    for (sname, mu) in mu_list(rep) {
        p.eta.iter_mut().for_each(|e| *e = 1.0);
        let a = p.matrix(mu);
        let sys = System::new(a.clone(), p.k);
        let r = reference(p, &a, &p.load, None, None);
        let mut lv = build_levels(p, &sys, false);
        let zero = vec![0.0; p.n];
        let res: Vec<RunOut> = methods.iter().map(|&me| run(p, &sys, &mut lv, &r, &p.load, &zero, me, CAP2, 32)).collect();
        let timing = time_ops(&sys, &mut lv, &p.load);
        emit(out, case, &sname, "cold", p, &sys, &lv, &r, &res, json!({"timing": timing}));
        if post_event && (sname.starts_with("static") || sname.starts_with("IE")) {
            // band weakens: warm start from the intact solution; P kept vs strength-aware
            let band = p.cut(&[0.5 * p.height]);
            let _ = m;
            for &f in &band {
                p.eta[f] = 1e-2;
            }
            let a2 = p.matrix(mu);
            let sys2 = System::new(a2.clone(), p.k);
            let r2 = reference(p, &a2, &p.load, None, None);
            // P kept: same P, A_c re-assembled and refactored
            let mut lv_keep = lv;
            let mut upd = Vec::new();
            for c in lv_keep.coarse.iter_mut() {
                c.2.update(&sys2).expect("A_c SPD");
                upd.push(json!({"s": c.0, "kind": format!("{:?}", c.1), "t_Ac": c.2.stats.t_ac, "t_numeric": c.2.stats.t_numeric, "flops_factor": c.2.stats.flops_factor}));
            }
            lv_keep.lam = (1.05 * sys2.lambda_dinv_a()).min(2.0);
            let res2: Vec<RunOut> = methods.iter().map(|&me| run(p, &sys2, &mut lv_keep, &r2, &p.load, &r.u, me, CAP2, 32)).collect();
            emit(out, case, &sname, "post-event, P kept", p, &sys2, &lv_keep, &r2, &res2, json!({"band_facets": band.len(), "eta": 1e-2, "update": upd}));
            let mut lv_new = build_levels(p, &sys2, true);
            let tl: Vec<Method> = methods.iter().cloned().filter(|m| matches!(m, Method::Tl { .. })).collect();
            let res3: Vec<RunOut> = tl.iter().map(|&me| run(p, &sys2, &mut lv_new, &r2, &p.load, &r.u, me, CAP2, 32)).collect();
            emit(out, case, &sname, "post-event, strength-aware rebuild", p, &sys2, &lv_new, &r2, &res3, json!({"band_facets": band.len(), "eta": 1e-2}));
            p.eta.iter_mut().for_each(|e| *e = 1.0);
        }
    }
}

// ======================================================================= checks (§5.1.1)

fn checks_small(out: &mut Vec<Value>) {
    let (mut p, _, _) = problem2("check block 1x0.5 d0.1", 1.0, 0.5, 0.1, 1, 0.1, &[(Some((0.0, 0.4)), V2::new(0.5e6, -2.0e6))]);
    for (sname, mu) in [("static", 0.0), ("IE h=1/480", 480.0 * 480.0)] {
        p.eta.iter_mut().for_each(|e| *e = 1.0);
        let a = p.matrix(mu);
        let at = a.transpose();
        let amax = a.val.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let asym = a.triplets().iter().zip(at.triplets().iter()).map(|(x, y)| if x.0 == y.0 && x.1 == y.1 { (x.2 - y.2).abs() } else { f64::INFINITY }).fold(0.0, f64::max) / amax;
        let sys = System::new(a.clone(), p.k);
        let r = reference(&p, &a, &p.load, None, None);
        let mut lv = build_levels(&p, &sys, false);
        // static: support reaction balances the applied load
        let (fr, _) = r.react;
        let ltot = [(0..p.nb).map(|i| p.load[3 * i]).sum::<f64>(), (0..p.nb).map(|i| p.load[3 * i + 1]).sum::<f64>()];
        // force exerted on the ground = applied load (statics)
        let balance = if mu == 0.0 { Some(((fr[0] - ltot[0]).powi(2) + (fr[1] - ltot[1]).powi(2)).sqrt() / (ltot[0].powi(2) + ltot[1].powi(2)).sqrt()) } else { None };
        let mut per = Vec::new();
        let mm = p.mass_csr();
        for (s0, kind, co, _) in lv.coarse.iter_mut() {
            let nc = co.p.nc;
            let z: Vec<f64> = (0..nc).map(|i| ((i as f64) * 0.731).sin()).collect();
            let mut pz = vec![0.0; p.n];
            co.p.matvec(&z, &mut pz);
            let e_fine = a_norm(&a, &pz).powi(2);
            let e_c = a_norm(&co.ac, &z).powi(2);
            // exact elimination of an error in range(P) by one coarse correction
            let mut u: Vec<f64> = r.u.iter().zip(&pz).map(|(x, y)| x + y).collect();
            let mut res = vec![0.0; p.n];
            let mut w = Work::default();
            sys.residual(&p.load, &u, &mut res, &mut w);
            co.correct(&res, &mut u, &mut w);
            let elim = norm(&u.iter().zip(&r.u).map(|(x, y)| x - y).collect::<Vec<_>>()) / norm(&pz);
            // global rigid modes in range(P): least squares in the M-norm (dense normal
            // equations; the condition number of PᵀMP is reported)
            let ptm = co.pt.matmul(&mm);
            let g = ptm.matmul(&co.p);
            let mut gd = faer::Mat::<f64>::zeros(nc, nc);
            for (r0, c0, v) in g.triplets() {
                gd[(r0, c0)] += v;
            }
            let ev = gd.self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
            let (emin, emax) = (ev.iter().cloned().fold(f64::INFINITY, f64::min), ev.iter().cloned().fold(0.0, f64::max));
            let rank_ok = emin > 1e-12 * emax;
            let lu = gd.partial_piv_lu();
            let mut rig = Vec::new();
            for mode in 0..3 {
                let rv: Vec<f64> = (0..p.nb).flat_map(|i| {
                    let (x, y) = (p.pos[i][0], p.pos[i][1]);
                    match mode {
                        0 => [1.0, 0.0, 0.0],
                        1 => [0.0, 1.0, 0.0],
                        _ => [-y, x, 1.0],
                    }
                }).collect();
                let mut rhs = vec![0.0; nc];
                ptm.matvec(&rv, &mut rhs);
                let sol = lu.solve(&faer::Mat::<f64>::from_fn(nc, 1, |i, _| rhs[i]));
                let z: Vec<f64> = (0..nc).map(|i| sol[(i, 0)]).collect();
                let mut fit = vec![0.0; p.n];
                co.p.matvec(&z, &mut fit);
                let d: Vec<f64> = fit.iter().zip(&rv).map(|(x, y)| x - y).collect();
                rig.push(a_norm(&mm, &d) / a_norm(&mm, &rv));
            }
            let g_cond = emax / emin;
            // direct image of the summed translation columns (no least squares): P z_t vs r
            let kc = if p.k == 3 { 3 } else { 6 };
            let zt: Vec<f64> = (0..nc).map(|c| if c % kc == 0 { 1.0 } else { 0.0 }).collect();
            let mut img = vec![0.0; p.n];
            co.p.matvec(&zt, &mut img);
            let rx: Vec<f64> = (0..p.n).map(|d| if d % p.k == 0 { 1.0 } else { 0.0 }).collect();
            let dimg: Vec<f64> = img.iter().zip(&rx).map(|(a, b)| a - b).collect();
            let direct_tx = a_norm(&mm, &dimg) / a_norm(&mm, &rx);
            per.push(json!({"s": s0, "kind": format!("{kind:?}"), "energy_identity_rel": (e_fine - e_c).abs() / e_fine, "range_P_error_after_one_coarse_correction": elim,
                "rank_full": rank_ok, "PtMP_condition": g_cond, "rigid_mode_fit_rel_M": rig, "translation_image_rel": direct_tx}));
        }
        // cycle as preconditioner: symmetric and positive, both smoothers
        let mut cyc = Vec::new();
        for (s0, kind) in [(8usize, PKind::Pr), (8, PKind::Sa)] {
            for cheb in [false, true] {
                let _ = lv.lam;
                let co = &mut lv.coarse.iter_mut().find(|c| c.0 == s0 && c.1 == kind).unwrap().2;
                let sm = if cheb { Smoother::Cheb { deg: 2, lo: CH_LO, hi: CH_HI } } else { Smoother::Gs };
                let mut apply = |rv: &[f64]| {
                    let mut x = vec![0.0; p.n];
                    let mut w = Work::default();
                    cycle(&sys, co, sm, rv, &mut x, true, &mut w);
                    x
                };
                let x1: Vec<f64> = (0..p.n).map(|i| (i as f64 * 0.37).sin()).collect();
                let y1: Vec<f64> = (0..p.n).map(|i| (i as f64 * 1.13).cos()).collect();
                let cx = apply(&x1);
                let cy = apply(&y1);
                let d = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
                let (a1, a2) = (d(&y1, &cx), d(&x1, &cy));
                cyc.push(json!({"s": s0, "kind": format!("{kind:?}"), "smoother": if cheb { "CH" } else { "GS" }, "symmetry_rel": (a1 - a2).abs() / a1.abs().max(a2.abs()), "xCx_over_xx": d(&x1, &cx) / d(&x1, &x1)}));
            }
        }
        eprintln!("checks {sname}: asym {asym:.1e} balance {balance:?} {}", serde_json::to_string(&per).unwrap());
        out.push(json!({"system": sname, "A_symmetry_rel": asym, "A_spd_direct": true, "reference_scaled_residual": r.scaled_residual, "dense_rel": r.dense_rel,
            "static_reaction_balance_rel": balance, "lambda_hat_DinvA": lv.lam, "coarse": per, "cycle": cyc}));
    }
}

// ======================================================================= bands (§4.3)

fn band_study(out: &mut Out, seed: u64) {
    let (mut p, m, _) = problem2(&format!("band wall 1x4 d0.1 s{seed}"), 1.0, 4.0, 0.1, seed, 0.2, &[(None, V2::new(2.5e3, -1.0e4))]);
    let h = p.height;
    // audit: v1 bands (not a cut) vs the corrected cut
    let mut v1_mid = p.band_v1(0.5 * h, Some(&m));
    let mut v1_three = v1_mid.clone();
    v1_three.extend(p.band_v1(0.25 * h, Some(&m)));
    v1_three.extend(p.band_v1(0.75 * h, Some(&m)));
    v1_mid.sort_unstable();
    v1_three.sort_unstable();
    let mid = p.cut(&[0.5 * h]);
    let frag = p.cut(&[0.25 * h, 0.5 * h, 0.75 * h]);
    let audit = json!({"cells": p.nb, "v1_band_one": p.audit_cut(&v1_mid), "v1_band_three": p.audit_cut(&v1_three),
        "cut_one_plane": p.audit_cut(&mid), "cut_three_planes": p.audit_cut(&frag)});
    eprintln!("cut audit s{seed}: {}", serde_json::to_string(&audit).unwrap());
    out.audits.push(json!({"seed": seed, "audit": audit}));
    let cases: Vec<(String, Vec<usize>, f64)> = vec![
        ("cut eta=1e-2".into(), mid.clone(), 1e-2),
        ("cut eta=1e-4".into(), mid.clone(), 1e-4),
        ("cut eta=0".into(), mid.clone(), 0.0),
        ("3 cuts eta=0".into(), frag, 0.0),
    ];
    let methods = Method::all();
    for (cname, facets, eta) in cases {
        for (sname, mu) in mu_list(false) {
            if eta == 0.0 && mu == 0.0 {
                continue;
            }
            p.eta.iter_mut().for_each(|e| *e = 1.0);
            let sys0 = System::new(p.matrix(mu), p.k);
            for &f in &facets {
                p.eta[f] = eta;
            }
            let a = p.matrix(mu);
            let sys = System::new(a.clone(), p.k);
            let r = reference(&p, &a, &p.load, None, None);
            let zero = vec![0.0; p.n];
            // update costs from the intact system: P kept (A_c + numeric) per coarse space
            let mut upd = Vec::new();
            {
                let lv0 = build_levels(&p, &sys0, false);
                for (s0, kind, mut co, _) in lv0.coarse {
                    co.update(&sys).expect("A_c SPD");
                    upd.push(json!({"s": s0, "kind": format!("{kind:?}"), "P_kept_t_Ac": co.stats.t_ac, "P_kept_t_numeric": co.stats.t_numeric, "flops_factor": co.stats.flops_factor, "nnz_Lc": co.stats.nnz_l}));
                }
            }
            for (aname, strong) in [("old aggregates", false), ("strength-aware", true)] {
                let mut lv = build_levels(&p, &sys, strong);
                let ms: Vec<Method> = if strong { methods.iter().cloned().filter(|m| matches!(m, Method::Tl { .. })).collect() } else { methods.clone() };
                let res: Vec<RunOut> = ms.iter().map(|&me| run(&p, &sys, &mut lv, &r, &p.load, &zero, me, CAP2, 32)).collect();
                emit(out, &p.name.clone(), &sname, &format!("{cname}, {aname}"), &p, &sys, &lv, &r, &res, json!({"band_facets": facets.len(), "eta": eta, "update_P_kept": upd}));
            }
        }
    }
    p.eta.iter_mut().for_each(|e| *e = 1.0);
}


/// Errors normalised by trajectory-wide reference scales (max over steps of the family
/// scales and of ‖u_ref‖_A): robust where a step's own reference is near zero.
fn traj_errors(p: &Problem, a: &Csr, uref: &[f64], u: &[f64], fam_scale: &[f64], a_scale: f64) -> (f64, f64) {
    let f = p.forces(u);
    let fr = p.forces(uref);
    let mut fe: f64 = 0.0;
    for k in 0..f.len() {
        let d: Vec<f64> = f[k].iter().zip(&fr[k]).map(|(x, y)| x - y).collect();
        if fam_scale[k] > 0.0 {
            fe = fe.max(norm(&d) / fam_scale[k]);
        }
    }
    let e: Vec<f64> = u.iter().zip(uref).map(|(x, y)| x - y).collect();
    (fe, a_norm(a, &e) / a_scale)
}

fn traj_scales(p: &Problem, sols: &[Vec<f64>], a: &Csr) -> (Vec<f64>, f64) {
    let nf = if p.dim == 2 { 2 } else { 3 };
    let mut sc: Vec<f64> = Vec::new();
    let mut amax: f64 = 0.0;
    for u in sols {
        let f = p.forces(u);
        let fn_: Vec<f64> = f.iter().map(|x| norm(x)).collect();
        let fj = fn_[..nf].iter().map(|x| x * x).sum::<f64>().sqrt();
        let s: Vec<f64> = (0..fn_.len()).map(|k| if k < nf { fj } else { fn_[k].max(fj * p.ell / 12f64.sqrt()) }).collect();
        if sc.is_empty() {
            sc = s;
        } else {
            for k in 0..sc.len() {
                sc[k] = sc[k].max(s[k]);
            }
        }
        amax = amax.max(a_norm(a, u));
    }
    (sc, amax)
}

// ======================================================================= routine dynamic steps

#[derive(Clone, Copy, PartialEq, Debug)]
enum Sch {
    Bdf2,
    Newmark,
}

/// Linear-dynamics state on DOF vectors, formulas of `solvers::integrator`
/// (BDF2 starts with one implicit-Euler step).
#[derive(Clone)]
struct LinDyn {
    sch: Sch,
    h: f64,
    u: Vec<f64>,
    u_prev: Vec<f64>,
    u_prev2: Vec<f64>,
    v: Vec<f64>,
    v_prev: Vec<f64>,
    a: Vec<f64>,
    steps: usize,
}

impl LinDyn {
    fn new(sch: Sch, h: f64, n: usize, v0: Vec<f64>) -> Self {
        LinDyn { sch, h, u: vec![0.0; n], u_prev: vec![0.0; n], u_prev2: vec![0.0; n], v: v0, v_prev: vec![0.0; n], a: vec![0.0; n], steps: 0 }
    }
    fn alpha(&self) -> f64 {
        match self.sch {
            Sch::Bdf2 if self.steps == 0 => 1.0,
            Sch::Bdf2 => 2.25,
            Sch::Newmark => 4.0,
        }
    }
    fn qhat(&self) -> Vec<f64> {
        let h = self.h;
        (0..self.u.len())
            .map(|i| match (self.sch, self.steps) {
                (Sch::Bdf2, 0) => self.u[i] + h * self.v[i],
                (Sch::Bdf2, _) => 4.0 / 3.0 * self.u[i] - 1.0 / 3.0 * self.u_prev[i] + 2.0 * h / 3.0 * (4.0 / 3.0 * self.v[i] - 1.0 / 3.0 * self.v_prev[i]),
                (Sch::Newmark, _) => self.u[i] + h * self.v[i] + 0.25 * h * h * self.a[i],
            })
            .collect()
    }
    /// Initial guess: "extrap" = BDF2 polynomial extrapolation / Newmark predictor,
    /// "qhat" = the integrator target.
    fn guess(&self, kind: &str) -> Vec<f64> {
        if kind == "qhat" {
            return self.qhat();
        }
        if kind == "prev" {
            return self.u.clone();
        }
        let h = self.h;
        (0..self.u.len())
            .map(|i| match (self.sch, self.steps) {
                (Sch::Newmark, _) => self.u[i] + h * self.v[i] + 0.5 * h * h * self.a[i],
                (Sch::Bdf2, 0) => self.u[i] + h * self.v[i],
                (Sch::Bdf2, 1) => 2.0 * self.u[i] - self.u_prev[i],
                (Sch::Bdf2, _) => 3.0 * self.u[i] - 3.0 * self.u_prev[i] + self.u_prev2[i],
            })
            .collect()
    }
    fn rhs(&self, mass: &Csr) -> Vec<f64> {
        let q = self.qhat();
        let mut b = vec![0.0; q.len()];
        mass.matvec(&q, &mut b);
        let c = self.alpha() / (self.h * self.h);
        b.iter_mut().for_each(|x| *x *= c);
        b
    }
    fn finish(&mut self, u_new: &[f64]) {
        let h = self.h;
        let c = self.alpha() / (h * h);
        let qh = self.qhat();
        let n = u_new.len();
        let a_new: Vec<f64> = (0..n).map(|i| c * (u_new[i] - qh[i])).collect();
        let v_new: Vec<f64> = (0..n)
            .map(|i| match (self.sch, self.steps) {
                (Sch::Bdf2, 0) => (u_new[i] - self.u[i]) / h,
                (Sch::Bdf2, _) => 1.5 / h * (u_new[i] - 4.0 / 3.0 * self.u[i] + 1.0 / 3.0 * self.u_prev[i]),
                (Sch::Newmark, _) => self.v[i] + 0.5 * h * (self.a[i] + a_new[i]),
            })
            .collect();
        self.v_prev = std::mem::replace(&mut self.v, v_new);
        self.a = a_new;
        self.u_prev2 = std::mem::replace(&mut self.u_prev, self.u.clone());
        self.u = u_new.to_vec();
        self.steps += 1;
    }
    fn energy(&self, mass: &Csr, k: &Csr) -> f64 {
        0.5 * a_norm(mass, &self.v).powi(2) + 0.5 * a_norm(k, &self.u).powi(2)
    }
}

struct Traj {
    name: String,
    sch: Sch,
    h: f64,
    steps: usize,
    event: Option<(usize, f64)>,
}

/// One step's system: operator (α/h²)M + K(η), its Levels (old / strength-aware) and the
/// reference solution from the reference history.
fn dynamics(out: &mut Out, json_out: &mut Vec<Value>) {
    let (mut p, m, _) = problem2("dyn wall 1x4 d0.1 s1", 1.0, 4.0, 0.1, 1, 0.2, &[]);
    let mass = p.mass_csr();
    let _ = &m;
    let mid = p.cut(&[0.5 * p.height]);
    let mut v0 = vec![0.0; p.n];
    for i in 0..p.nb {
        if p.pos[i][1] > p.height - 0.25 {
            v0[3 * i] = 1.0;
        }
    }
    let trajs = vec![
        Traj { name: "BDF2 h=1/480".into(), sch: Sch::Bdf2, h: 1.0 / 480.0, steps: 120, event: None },
        Traj { name: "BDF2 h=1/60".into(), sch: Sch::Bdf2, h: 1.0 / 60.0, steps: 30, event: None },
        Traj { name: "Newmark h=1/480".into(), sch: Sch::Newmark, h: 1.0 / 480.0, steps: 120, event: None },
        Traj { name: "BDF2 h=1/480 cut eta=0 at t=0.05".into(), sch: Sch::Bdf2, h: 1.0 / 480.0, steps: 60, event: Some((24, 0.0)) },
    ];
    let tl = Method::two_level();
    for tr in &trajs {
        // ---- reference trajectory (direct solves) and per-step systems
        p.eta.iter_mut().for_each(|e| *e = 1.0);
        let mut st = LinDyn::new(tr.sch, tr.h, p.n, v0.clone());
        let mut hist: Vec<LinDyn> = Vec::new();
        let mut sols: Vec<Vec<f64>> = Vec::new();
        let mut energies = Vec::new();
        let kmat_intact = p.matrix(0.0);
        for n in 0..tr.steps {
            if let Some((ne, eta)) = tr.event {
                if n == ne {
                    for &f in &mid {
                        p.eta[f] = eta;
                    }
                }
            }
            let a = p.matrix(st.alpha() / (tr.h * tr.h));
            let b = st.rhs(&mass);
            let (u, _, _, _) = direct(&a, &b);
            hist.push(st.clone());
            st.finish(&u);
            energies.push(st.energy(&mass, &p.matrix(0.0)));
            sols.push(u);
        }
        let _ = &kmat_intact;
        let a_last = p.matrix(st.alpha() / (tr.h * tr.h));
        let (fam_traj, a_traj) = traj_scales(&p, &sols, &a_last);
        // trajectory-wide χ scale
        let cmax = sols.iter().map(|u| p.chi(u).into_iter().filter(|x| x.is_finite()).fold(0.0, f64::max)).fold(0.0, f64::max);
        let chi_scale = 0.95 / cmax;
        // ---- open loop: every step from the exact history
        // "prev" (previous solution as guess) is post-hoc
        let preds: Vec<&str> = if tr.sch == Sch::Bdf2 { vec!["extrap", "qhat", "prev"] } else { vec!["extrap", "prev"] };
        let mut summary_steps = Vec::new();
        p.eta.iter_mut().for_each(|e| *e = 1.0);
        let mut lv_cache: Vec<(f64, bool, Levels, System)> = Vec::new();
        let aggs: Vec<(&str, bool)> = if tr.event.is_some() { vec![("P kept", false), ("strength-aware", true)] } else { vec![("intact", false)] };
        for n in 0..tr.steps {
            if let Some((ne, eta)) = tr.event {
                if n == ne {
                    for &f in &mid {
                        p.eta[f] = eta;
                    }
                }
                if n + 2 < ne {
                    continue; // only the steps around and after the event
                }
            }
            let stn = &hist[n];
            let mu = stn.alpha() / (tr.h * tr.h);
            let a = p.matrix(mu);
            let b = stn.rhs(&mass);
            let r = reference(&p, &a, &b, Some(&stn.u), Some(chi_scale));
            let event_now = tr.event.map(|(ne, _)| n >= ne).unwrap_or(false);
            for &(aname, strong) in &aggs {
                // levels per (μ, aggregation, event state); "P kept" reuses intact P with
                // A_c re-assembled and refactored at the event
                let key_eta = if event_now { 1.0 } else { 0.0 };
                let pos = lv_cache.iter().position(|c| c.0 == mu + key_eta * 1e-9 && c.1 == strong);
                let idx = match pos {
                    Some(i) => i,
                    None => {
                        let sys = System::new(a.clone(), p.k);
                        let lv = if strong && event_now {
                            build_levels(&p, &sys, true)
                        } else if event_now {
                            // P from the intact operator, updated to the weakened one
                            let saved = p.eta.clone();
                            p.eta.iter_mut().for_each(|e| *e = 1.0);
                            let sys0 = System::new(p.matrix(mu), p.k);
                            p.eta = saved;
                            let mut lv = build_levels(&p, &sys0, false);
                            for c in lv.coarse.iter_mut() {
                                c.2.update(&sys).expect("A_c SPD");
                                c.3["event_update_t_Ac"] = json!(c.2.stats.t_ac);
                                c.3["event_update_t_numeric"] = json!(c.2.stats.t_numeric);
                            }
                            lv.lam = (1.05 * sys.lambda_dinv_a()).min(2.0);
                            lv
                        } else {
                            build_levels(&p, &sys, false)
                        };
                        lv_cache.push((mu + key_eta * 1e-9, strong, lv, sys));
                        lv_cache.len() - 1
                    }
                };
                let (_, _, lv, sys) = &mut lv_cache[idx];
                for &pk in &preds {
                    let x0 = stn.guess(pk);
                    let ms: Vec<Method> = if strong { tl.clone() } else { Method::all() };
                    for &me in &ms {
                        let ro = run(&p, sys, lv, &r, &b, &x0, me, CAP2, 2);
                        let g = |k: usize| ro.checkpoints.iter().find(|c| c.0 == k).map(|c| c.1);
                        let fmt = |v: Option<f64>| v.map(|x| format!("{x:.4e}")).unwrap_or_default();
                        out.dyn_open.row_s(&[tr.name.clone(), n.to_string(), format!("{}", (n + 1) as f64 * tr.h), pk.into(), aname.into(), me.name(),
                            ro.iters.map(|x| x.to_string()).unwrap_or_default(), format!("{:.4e}", ro.initial.f), fmt(g(1).map(|m| m.f)), fmt(g(2).map(|m| m.f)),
                            fmt(g(4).map(|m| m.f)), fmt(g(8).map(|m| m.f)), format!("{:.4e}", ro.initial.e), fmt(g(1).map(|m| m.e)), fmt(g(2).map(|m| m.e)),
                            format!("{:.4e}", ro.initial.e_inc), fmt(g(1).map(|m| m.e_inc)), fmt(g(2).map(|m| m.e_inc)), fmt(g(1).map(|m| m.chi_hi)),
                            format!("{:.4e}", traj_errors(&p, &sys.a, &r.u, &x0, &fam_traj, a_traj).0)]);
                        summary_steps.push(json!({"step": n, "pred": pk, "agg": aname, "method": me.name(), "iters": ro.iters, "censored": ro.censored,
                            "a_eq": ro.a_eq_at_target, "phases": ro.work_at_target.phases, "F0": ro.initial.f, "E0": ro.initial.e, "Einc0": ro.initial.e_inc}));
                    }
                }
            }
        }
        // ---- closed loop: k cycles / PCG iterations per step from the variant's own
        // history. Every step counts. At an event the fine operator is always current; the
        // coarse level follows the policy: "P kept" (A_c re-assembled and refactored) or
        // "strength-aware" (rebuilt).
        let mut closed = Vec::new();
        let kmat_end = p.matrix(0.0);
        let policies: Vec<&str> = if tr.event.is_some() { vec!["P kept", "strength-aware"] } else { vec!["intact"] };
        let cl_methods = [
            (Method::Tl { p: PKind::Sa, s: 8, cheb: true, pcg: true }, vec![1usize, 2, 3, 4, 6, 8]),
            (Method::Tl { p: PKind::Pr, s: 8, cheb: false, pcg: true }, vec![2, 3, 4, 6, 8]),
            (Method::Tl { p: PKind::Sa, s: 8, cheb: true, pcg: false }, vec![2, 4, 8, 16]),
            (Method::Tl { p: PKind::Sa, s: 8, cheb: false, pcg: false }, vec![2, 4, 8, 16]),
            (Method::Tl { p: PKind::Pr, s: 8, cheb: false, pcg: false }, vec![2, 4, 8, 16, 32]),
        ];
        for &pol in &policies {
            let mut levels: Vec<(String, Levels, System)> = Vec::new();
            for (me, ks) in &cl_methods {
                for guess in ["extrap", "prev"] {
                    for &kc in ks {
                        p.eta.iter_mut().for_each(|e| *e = 1.0);
                        let mut sa = LinDyn::new(tr.sch, tr.h, p.n, v0.clone());
                        let mut rows: Vec<(usize, f64, f64, f64)> = Vec::new(); // (step, F_traj, E_traj, chi_union)
                        let (mut fal8, mut mis8, mut fal9, mut mis9) = (0usize, 0usize, 0usize, 0usize);
                        let mut wsum = Work::default();
                        let mut ratio_last = f64::NAN;
                        for n in 0..tr.steps {
                            let event_now = tr.event.map(|(ne, _)| n >= ne).unwrap_or(false);
                            if let Some((ne, eta)) = tr.event {
                                if n == ne {
                                    for &f in &mid {
                                        p.eta[f] = eta;
                                    }
                                }
                            }
                            let mu = sa.alpha() / (tr.h * tr.h);
                            let key = format!("{mu}-{event_now}");
                            if !levels.iter().any(|l| l.0 == key) {
                                let sys = System::new(p.matrix(mu), p.k);
                                let lv = if event_now && pol == "strength-aware" {
                                    build_levels(&p, &sys, true)
                                } else if event_now {
                                    let saved = p.eta.clone();
                                    p.eta.iter_mut().for_each(|e| *e = 1.0);
                                    let sys0 = System::new(p.matrix(mu), p.k);
                                    p.eta = saved;
                                    let mut lv = build_levels(&p, &sys0, false);
                                    for c in lv.coarse.iter_mut() {
                                        c.2.update(&sys).expect("A_c SPD");
                                    }
                                    lv
                                } else {
                                    build_levels(&p, &sys, false)
                                };
                                levels.push((key.clone(), lv, sys));
                            }
                            let (_, lv, sys) = levels.iter_mut().find(|l| l.0 == key).unwrap();
                            let b = sa.rhs(&mass);
                            let mut x = sa.guess(guess);
                            let Method::Tl { p: kind, s: s0, cheb, pcg: use_pcg } = *me else { unreachable!() };
                            let co = &mut lv.coarse.iter_mut().find(|c| c.0 == s0 && c.1 == kind).unwrap().2;
                            let sm = if cheb { Smoother::Cheb { deg: 2, lo: CH_LO, hi: CH_HI } } else { Smoother::Gs };
                            if use_pcg {
                                pcg(sys, &mut Precond::TwoLevel(co, sm), &b, &mut x, kc, None, &mut wsum, &mut |_, _, _| false);
                            } else {
                                for _ in 0..kc {
                                    cycle(sys, co, sm, &b, &mut x, false, &mut wsum);
                                }
                            }
                            sa.finish(&x);
                            let (ft, et) = traj_errors(&p, &sys.a, &sols[n], &x, &fam_traj, a_traj);
                            // local onset indicator: union of reference and candidate points ≥ 0.8
                            let (cr, cc) = (p.chi(&sols[n]), p.chi(&x));
                            let mut cu: f64 = 0.0;
                            for (a, c) in cr.iter().zip(&cc) {
                                if !a.is_finite() {
                                    continue;
                                }
                                let (a, c) = (a * chi_scale, c * chi_scale);
                                if a >= 0.8 || c >= 0.8 {
                                    cu = cu.max((a - c).abs());
                                }
                                fal8 += (c >= 0.8 && a < 0.8) as usize;
                                mis8 += (a >= 0.8 && c < 0.8) as usize;
                                fal9 += (c >= 0.9 && a < 0.9) as usize;
                                mis9 += (a >= 0.9 && c < 0.9) as usize;
                            }
                            ratio_last = sa.energy(&mass, &kmat_end) / energies[n];
                            out.dyn_closed.row_s(&[tr.name.clone(), pol.into(), format!("{} {guess}", me.name()), kc.to_string(), n.to_string(), format!("{}", (n + 1) as f64 * tr.h),
                                format!("{:.4e}", ft), format!("{:.4e}", et), format!("{:.4e}", cu), format!("{:.6}", ratio_last)]);
                            rows.push((n, ft, et, cu));
                            if !et.is_finite() || et > 1e12 {
                                break;
                            }
                        }
                        let ne = tr.event.map(|e| e.0);
                        let mx = |sel: &dyn Fn(usize) -> bool, k: usize| rows.iter().filter(|r| sel(r.0)).map(|r| [r.1, r.2, r.3][k]).fold(0.0, f64::max);
                        let in_event = |n: usize| ne.map(|e| n >= e && n < e + 3).unwrap_or(false);
                        let summ = json!({"policy": pol, "method": me.name(), "guess": guess, "k": kc,
                            "F_traj": {"step0": mx(&|n| n == 0, 0), "steps1_2": mx(&|n| n == 1 || n == 2, 0), "routine_ge3": mx(&|n| n >= 3 && !in_event(n), 0),
                                "event_window": if ne.is_some() { json!(mx(&|n| in_event(n), 0)) } else { Value::Null }, "all_steps": mx(&|_| true, 0)},
                            "E_traj_all": mx(&|_| true, 1), "chi_union_max_all": mx(&|_| true, 2),
                            "chi_crossings": {"false_0.8": fal8, "missed_0.8": mis8, "false_0.9": fal9, "missed_0.9": mis9},
                            "steps_done": rows.len(), "energy_ratio_end": ratio_last,
                            "work_total": {"a_apps": wsum.a_apps, "phases": wsum.phases, "reductions": wsum.reductions, "coarse_solves": wsum.coarse_solves, "vec_ops": wsum.vec_ops}});
                        eprintln!("closed {} [{pol}] {} {guess} k={kc}: F all {:.3e} routine {:.3e} step0 {:.3e} χ {:.3e} false/missed0.8 {fal8}/{mis8}", tr.name, me.name(),
                            summ["F_traj"]["all_steps"].as_f64().unwrap(), summ["F_traj"]["routine_ge3"].as_f64().unwrap(), summ["F_traj"]["step0"].as_f64().unwrap(), summ["chi_union_max_all"].as_f64().unwrap());
                        closed.push(summ);
                    }
                }
            }
        }
        p.eta.iter_mut().for_each(|e| *e = 1.0);
        let lv_info: Vec<Value> = lv_cache.iter().map(|(mu, strong, lv, sys)| json!({"mu": mu, "strength_aware": strong, "colors": sys.colors.len(), "levels": lv.coarse.iter().map(|c| c.3.clone()).collect::<Vec<_>>()})).collect();
        eprintln!("trajectory {} done", tr.name);
        json_out.push(json!({"trajectory": tr.name, "h": tr.h, "steps": tr.steps, "event": tr.event.map(|(n, e)| json!({"step": n, "eta": e, "cut_facets": mid.len()})),
            "chi_scale": chi_scale, "energy_ref": energies, "levels": lv_info, "open_loop": summary_steps, "closed_loop": closed}));
    }
}


// ======================================================================= §3 reuse study (Phase 0b / 3)

const REUSE_EXP: &str = "phase0b/architecture-selection/reuse";

impl Problem {
    /// Stress proxy per facet: 2D max χ of its points; 3D E·|C_n|/ℓ_e of its normal component.
    fn facet_stress(&self, u: &[f64]) -> Vec<f64> {
        let mut s = vec![0.0f64; self.nfacets];
        if self.dim == 2 {
            for (p, c) in self.pts.iter().zip(self.chi(u)) {
                if c.is_finite() {
                    s[p.facet] = s[p.facet].max(c);
                }
            }
        } else {
            for c in self.comps3.iter().filter(|c| c.3 == 0) {
                let v: f64 = c.0.iter().map(|&(d, x)| x * u[d]).sum();
                s[c.2] = s[c.2].max(rbsn3::Mat3D::concrete().e * v.abs() / self.facet_ell[c.2]);
            }
        }
        s
    }

    /// Stiffness components of one facet as (entries, k).
    fn facet_comps(&self, f: usize) -> Vec<(&[(usize, f64)], f64)> {
        let mut v: Vec<(&[(usize, f64)], f64)> = Vec::new();
        for p in self.pts.iter().filter(|p| p.facet == f) {
            v.push((&p.jn, p.kn));
            v.push((&p.jt, p.kt));
        }
        for c in self.comps3.iter().filter(|c| c.2 == f) {
            v.push((&c.0, c.1));
        }
        v
    }
}

fn build_one(p: &Problem, sys: &System, strong: bool, kind: PKind) -> (Coarse, f64) {
    let t0 = Instant::now();
    let mut e = Vec::new();
    let agg = aggregate(&p.agg_input(8, strong, &mut e));
    let ppr = rigid_p(&agg, &p.pos, &p.cmass, p.k, p.ell);
    let pm = if kind == PKind::Sa {
        let lam = (1.05 * sys.lambda_dinv_a()).min(2.0);
        smooth_p(sys, &ppr, 4.0 / (3.0 * lam))
    } else {
        ppr
    };
    let co = Coarse::new(sys, pm).expect("A_c SPD");
    (co, t0.elapsed().as_secs_f64())
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Pol {
    U,
    F,
    R8,
    Rd,
    Pa,
}

fn reuse_problem(name: &str, mut p: Problem, out: &mut Vec<Value>, steps_csv: &mut Csv) {
    let h = 1.0 / 480.0;
    let nsteps = 60;
    let mass = p.mass_csr();
    let mut v0 = vec![0.0; p.n];
    let ax = if p.dim == 2 { 1 } else { 2 };
    for i in 0..p.nb {
        if p.pos[i][ax] > p.height - 0.25 {
            v0[p.k * i] = 1.0;
        }
    }
    let hgt = p.height;
    // ---- reference trajectory with a given η schedule (direct solves)
    let run_ref = |p: &mut Problem, sched: &dyn Fn(usize, &mut Vec<f64>)| -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        let mut st = LinDyn::new(Sch::Bdf2, h, p.n, v0.clone());
        let (mut etas, mut bs, mut us, mut mus) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        p.eta.iter_mut().for_each(|e| *e = 1.0);
        for n in 0..nsteps {
            sched(n, &mut p.eta);
            let mu = st.alpha() / (h * h);
            let a = p.matrix(mu);
            let b = st.rhs(&mass);
            let (u, _, _, _) = direct(&a, &b);
            st.finish(&u);
            etas.push(p.eta.clone());
            bs.push(b);
            us.push(u);
            mus.push(mu);
        }
        p.eta.iter_mut().for_each(|e| *e = 1.0);
        (etas, bs, us, mus)
    };
    // stress ranking on the intact trajectory
    let (_, _, us0, _) = run_ref(&mut p, &|_, _| {});
    let mut smax = vec![0.0f64; p.nfacets];
    for u in &us0 {
        for (a, b) in smax.iter_mut().zip(p.facet_stress(u)) {
            *a = a.max(b);
        }
    }
    let interior: Vec<usize> = (0..p.nfacets).filter(|&f| !p.support_facet[f]).collect();
    let mut ranked = interior.clone();
    ranked.sort_by(|&a, &b| smax[b].partial_cmp(&smax[a]).unwrap().then(a.cmp(&b)));
    let top1: Vec<usize> = ranked[..(interior.len() / 100).max(1)].to_vec();
    let top10: Vec<usize> = ranked[..(interior.len() / 10).max(1)].to_vec();
    let mid = p.cut(&[0.5 * hgt]);
    let mut crack = mid.clone();
    crack.sort_by(|&a, &b| {
        let (pa, pb) = (p.facet_pc[a], p.facet_pc[b]);
        pa[0].partial_cmp(&pb[0]).unwrap().then(pa[1].partial_cmp(&pb[1]).unwrap()).then(a.cmp(&b))
    });
    let c1 = p.cut(&[0.25 * hgt]);
    let c2 = mid.clone();
    let c3 = p.cut(&[0.75 * hgt]);
    let ncr = crack.len();
    type Sched = Box<dyn Fn(usize, &mut Vec<f64>)>;
    let scen: Vec<(&str, Sched)> = vec![
        ("S0 none", Box::new(|_, _| {})),
        ("S1 soften 1%", { let s = top1.clone(); Box::new(move |n, e: &mut Vec<f64>| for &f in &s { e[f] = (1.0 - 0.9 * n as f64 / 40.0).max(0.1); }) }),
        ("S2 soften 10%", { let s = top10.clone(); Box::new(move |n, e: &mut Vec<f64>| for &f in &s { e[f] = (1.0 - 0.9 * n as f64 / 40.0).max(0.1); }) }),
        ("S3 growing crack", { let s = crack.clone(); Box::new(move |n, e: &mut Vec<f64>| for (j, &f) in s.iter().enumerate() {
            let st = 10 + j * 39 / ncr.max(1);
            e[f] = if n > st { 0.0 } else if n == st { 0.5 } else { 1.0 };
        }) }),
        ("S4 cut at step 20", { let s = mid.clone(); Box::new(move |n, e: &mut Vec<f64>| if n >= 20 { for &f in &s { e[f] = 0.0; } }) }),
        ("S5 three cuts", { let (a, b, c) = (c1.clone(), c2.clone(), c3.clone()); Box::new(move |n, e: &mut Vec<f64>| {
            for (set, at) in [(&a, 15usize), (&b, 25), (&c, 35)] {
                if n >= at {
                    for &f in set.iter() { e[f] = 0.0; }
                }
            }
        }) }),
    ];
    let mu2 = 2.25 / (h * h);
    let mut tau = 1e-3;
    let mut calib = Value::Null;
    let mut results = Vec::new();
    for (sname, sched) in &scen {
        let (etas, bs, us, mus) = run_ref(&mut p, sched.as_ref());
        let a_last = p.matrix(mu2);
        let (fam, asc) = traj_scales(&p, &us, &a_last);
        let cmax = us.iter().map(|u| p.chi(u).into_iter().filter(|x| x.is_finite()).fold(0.0, f64::max)).fold(0.0, f64::max);
        let chi_scale = if cmax > 0.0 { 0.95 / cmax } else { 1.0 };
        let pols = [Pol::U, Pol::F, Pol::R8, Pol::Rd, Pol::Pa];
        for &kind in &[PKind::Sa, PKind::Pr] {
            let sm = if kind == PKind::Sa { Smoother::Cheb { deg: 2, lo: CH_LO, hi: CH_HI } } else { Smoother::Gs };
            for &pol in &pols {
                // calibration (pre-registered): U on S0 with SA must meet F ≤ 1 % and χ ≤ 0.01 every step
                p.eta.iter_mut().for_each(|e| *e = 1.0);
                let sys_init = System::new(p.matrix(mu2), p.k);
                let (mut co, t_build0) = build_one(&p, &sys_init, false, kind);
                let mut sys_b = sys_init;
                let ptmp = co.pt.matmul(&mass).matmul(&co.p);
                let mut mu_b = mu2;
                let mut eta_b = vec![1.0; p.nfacets];
                let (mut iters_tot, mut retries, mut rebuilds, mut lowrank) = (0usize, 0usize, 0usize, 0usize);
                let (mut t_apply, mut t_update) = (0.0f64, t_build0);
                let mut w = Work::default();
                let (mut fmax, mut emax, mut cmax_err) = (0.0f64, 0.0f64, 0.0f64);
                let mut lowrank_check: Vec<Value> = Vec::new();
                let mut madds_tot = 0.0;
                let nnz_a = sys_b.a.nnz() as f64;
                let mut aeq = 0.0;
                for n in 0..nsteps {
                    p.eta = etas[n].clone();
                    let a = p.matrix(mus[n]);
                    let sys = System::new(a, p.k);
                    let changed: Vec<usize> = (0..p.nfacets).filter(|&f| etas[n][f] != eta_b[f]).collect();
                    let mu_changed = mus[n] != mu_b;
                    // ---- update policy
                    let mut tu = 0.0;
                    match pol {
                        Pol::U => {
                            if n == 0 || !changed.is_empty() || mu_changed {
                                let (c, t) = build_one(&p, &sys, true, kind);
                                co = c;
                                tu = t;
                                rebuilds += 1;
                                eta_b = etas[n].clone();
                                mu_b = mus[n];
                            }
                        }
                        Pol::R8 => {
                            if n > 0 && n % 8 == 0 {
                                let (c, t) = build_one(&p, &sys, true, kind);
                                co = c;
                                tu = t;
                                sys_b = System::new(sys.a.clone(), p.k);
                                rebuilds += 1;
                            }
                        }
                        Pol::Pa => {
                            if !changed.is_empty() || mu_changed {
                                let t0 = Instant::now();
                                let ch: Vec<(Vec<(usize, f64)>, f64)> = changed.iter().flat_map(|&f| {
                                    let de = etas[n][f] - eta_b[f];
                                    p.facet_comps(f).into_iter().map(move |(j, k)| (j.to_vec(), de * k)).collect::<Vec<_>>()
                                }).collect();
                                let refs: Vec<(&[(usize, f64)], f64)> = ch.iter().map(|(j, k)| (j.as_slice(), *k)).collect();
                                madds_tot += co.add_lowrank(&refs);
                                if mu_changed {
                                    co.add_scaled(&ptmp, mus[n] - mu_b);
                                }
                                co.refactor().expect("A_c SPD");
                                tu = t0.elapsed().as_secs_f64();
                                lowrank += 1;
                                eta_b = etas[n].clone();
                                mu_b = mus[n];
                                if (sname.starts_with("S2") && n == 30) || (sname.starts_with("S4") && n == 20) || (sname.starts_with("S0") && n == 1) {
                                    let full = co.pt.matmul(&sys.a.matmul(&co.p));
                                    let d: f64 = full.val.iter().zip(&co.ac.val).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt();
                                    let nf: f64 = full.val.iter().map(|x| x * x).sum::<f64>().sqrt();
                                    let same = full.idx == co.ac.idx;
                                    lowrank_check.push(json!({"step": n, "rel_frobenius": d / nf, "same_pattern": same, "changed_facets": changed.len(), "changed_fraction": changed.len() as f64 / interior.len() as f64}));
                                }
                            }
                        }
                        _ => {}
                    }
                    t_update += tu;
                    // ---- solve
                    let x0 = if n == 0 { vec![0.0; p.n] } else { us[n - 1].clone() };
                    let mut x = x0.clone();
                    let mut ws = Work::default();
                    let t0 = Instant::now();
                    let cap = if pol == Pol::Rd { 8 } else { 64 };
                    let (mut it, mut rel) = match pol {
                        Pol::U | Pol::Pa => pcg(&sys, &mut Precond::TwoLevel(&mut co, sm), &bs[n], &mut x, cap, Some(tau), &mut ws, &mut |_, _, _| false),
                        _ => pcg(&sys, &mut Precond::Stale(&sys_b, &mut co, sm), &bs[n], &mut x, cap, Some(tau), &mut ws, &mut |_, _, _| false),
                    };
                    if pol == Pol::Rd && rel > tau {
                        // refresh and repeat from the same guess; the first attempt is counted
                        let (c, t) = build_one(&p, &sys, true, kind);
                        co = c;
                        t_update += t;
                        sys_b = System::new(sys.a.clone(), p.k);
                        rebuilds += 1;
                        retries += 1;
                        iters_tot += it;
                        x = x0.clone();
                        let r2 = pcg(&sys, &mut Precond::Stale(&sys_b, &mut co, sm), &bs[n], &mut x, 64, Some(tau), &mut ws, &mut |_, _, _| false);
                        it = r2.0;
                        rel = r2.1;
                    }
                    t_apply += t0.elapsed().as_secs_f64();
                    iters_tot += it;
                    let lfac = 4.0 * co.stats.nnz_l as f64 / (2.0 * nnz_a);
                    aeq += ws.a_apps + ws.coarse_solves * lfac;
                    w.add(&ws);
                    // ---- quality against the direct solution of the same step
                    let (ft, et) = traj_errors(&p, &sys.a, &us[n], &x, &fam, asc);
                    let (cr, cc) = (p.chi(&us[n]), p.chi(&x));
                    let mut cu: f64 = 0.0;
                    for (a0, c0) in cr.iter().zip(&cc) {
                        if a0.is_finite() && (a0 * chi_scale >= 0.8 || c0 * chi_scale >= 0.8) {
                            cu = cu.max((a0 - c0).abs() * chi_scale);
                        }
                    }
                    fmax = fmax.max(ft);
                    emax = emax.max(et);
                    cmax_err = cmax_err.max(cu);
                    steps_csv.row_s(&[name.into(), (*sname).into(), format!("{kind:?}"), format!("{pol:?}"), n.to_string(), it.to_string(), format!("{rel:.3e}"),
                        format!("{tu:.6e}"), format!("{ft:.4e}"), format!("{et:.4e}"), format!("{cu:.4e}"), changed.len().to_string()]);
                }
                let r = json!({"problem": name, "scenario": sname, "variant": format!("{kind:?}-s8"), "policy": format!("{pol:?}"), "tau": tau,
                    "iters_total": iters_tot, "retries": retries, "rebuilds": rebuilds, "lowrank_updates": lowrank, "lowrank_madds": madds_tot, "lowrank_check": lowrank_check,
                    "t_apply_s": t_apply, "t_update_s": t_update, "t_total_s": t_apply + t_update, "A_eq": aeq,
                    "work": {"a_apps": w.a_apps, "phases": w.phases, "reductions": w.reductions, "coarse_solves": w.coarse_solves, "vec_ops": w.vec_ops},
                    "max_F_traj": fmax, "max_E_traj": emax, "max_chi_union": cmax_err});
                eprintln!("{name} | {sname} | {kind:?} | {pol:?}: iters {iters_tot} retries {retries} rebuilds {rebuilds} t_apply {:.1} ms t_update {:.1} ms | F {:.2e} E {:.2e} χ {:.2e}", t_apply * 1e3, t_update * 1e3, fmax, emax, cmax_err);
                if sname.starts_with("S0") && pol == Pol::U && kind == PKind::Sa && calib.is_null() {
                    let ok = fmax <= 0.01 && (p.dim == 3 || cmax_err <= 0.01);
                    calib = json!({"tau_tried": tau, "pass": ok, "max_F_traj": fmax, "max_chi_union": cmax_err});
                    if !ok {
                        tau = 1e-4;
                        eprintln!("calibration: τ → 1e-4 (pre-registered rule)");
                    }
                }
                results.push(r);
            }
        }
    }
    out.push(json!({"problem": name, "cells": p.nb, "dofs": p.n, "interior_facets": interior.len(), "set_sizes": {"top1": top1.len(), "top10": top10.len(), "crack": crack.len(),
        "cut_mid": mid.len(), "cuts3": [c1.len(), c2.len(), c3.len()]}, "calibration": calib, "tau_final": tau, "runs": results}));
}

fn reuse_main() {
    let t0 = Instant::now();
    let mut steps = Csv::new(REUSE_EXP, "steps", &["problem", "scenario", "variant", "policy", "step", "iters", "rel_res", "t_update", "F_traj", "E_traj", "chi_union", "changed_facets"]);
    let mut out = Vec::new();
    let (p2, _, _) = problem2("wall 1x4 d0.1 s1", 1.0, 4.0, 0.1, 1, 0.2, &[]);
    reuse_problem("2D wall 1x4 d0.1 s1", p2, &mut out, &mut steps);
    let p3 = problem3("3D cube 2m d0.2 s1", V3::new(2.0, 2.0, 2.0), 0.2, 1, V3::new(0.0, 0.0, 0.0));
    reuse_problem("3D cube 2m d0.2 s1", p3, &mut out, &mut steps);
    write_json(REUSE_EXP, "summary", &json!({"meta": meta(REUSE_EXP, "reuse", Some(1), "PCG(current A_t) with B per policy U/F/R8/RD/PA; SA-s8-CH and PR-s8-GS; stop ||r|| <= tau ||r0||", json!({"h": 1.0 / 480.0, "steps": 60})),
        "problems": out, "seconds": t0.elapsed().as_secs_f64()}));
    eprintln!("reuse done in {:.0} s", t0.elapsed().as_secs_f64());
}

// ======================================================================= main

fn choose_variant(results: &[Value]) -> (String, Value) {
    let mut score: std::collections::BTreeMap<String, (usize, f64, usize)> = Default::default();
    for r in results {
        let case = r["case"].as_str().unwrap_or("");
        let sys = r["system"].as_str().unwrap_or("");
        if !(case.starts_with("E3") && r["rhs"] == json!("cold") && (sys.starts_with("IE") || sys.starts_with("BDF2") || sys.starts_with("Newmark"))) {
            continue;
        }
        for ro in r["runs"].as_array().unwrap() {
            let name = ro["method"].as_str().unwrap().to_string();
            if !name.contains("TL-") {
                continue;
            }
            let e = score.entry(name).or_insert((0, 0.0, 0));
            if ro["censored"] == json!(true) {
                e.0 += 1;
            } else {
                e.1 += ro["work_at_target"]["A_equivalents"].as_f64().unwrap();
            }
            e.2 += 1;
        }
    }
    let best = score.iter().min_by(|a, b| a.1 .0.cmp(&b.1 .0).then(a.1 .1.partial_cmp(&b.1 .1).unwrap())).map(|x| x.0.clone()).unwrap();
    (best, json!(score.iter().map(|(k, v)| json!({"method": k, "censored": v.0, "sum_A_eq": v.1, "systems": v.2})).collect::<Vec<_>>()))
}

fn manifest() -> Value {
    json!({
        "written": "before any solver run of this binary (main writes it first)",
        "geometry": {"construction": "constrained RSA from scratch, generator-support distance >= 0.5 dmin on supported edges",
            "E2": {"size": [1.0, 0.5], "dmin": 0.05, "b": 0.1, "seeds": [1, 2], "loads": ["uniform 1 MPa", "patch (0.5,-2) MPa on [0,0.4] exact"]},
            "E3_heights": {"width": 1.0, "heights": [1.0, 2.0, 4.0, 8.0], "dmin": 0.1, "b": 0.2, "seeds": [1, 2], "load": "top traction (2.5,-10) kPa"},
            "E3_refine": {"size": [1.0, 4.0], "dmin": [0.2, 0.1, 0.05], "seeds": [1, 2]},
            "representative": "1x4 d0.1 s1: + BDF2 and Newmark operators for h = 1/60, 1/480",
            "held_out": {"seeds": [3001, 3002], "heights": [4.0, 8.0], "dmin": 0.1},
            "3D": {"cube": [2.0, 2.0, 2.0], "dmin": [0.35, 0.2], "column": [1.0, 1.0, 4.0], "column_dmin": 0.2, "seeds": [1, 2], "load": "top traction (10,0,0) kPa"}},
        "operators": "mu = 0; mu = 1/h^2 (h = 1/60, 1/480); representative also 2.25/h^2, 4/h^2",
        "rhs": ["cold: load, u0 = 0", "post-event: mid-height band eta = 1e-2, u0 = intact solution (same mu); P kept and strength-aware rebuild"],
        "bands": {"wall": "1x4 d0.1, seeds 1, 2", "cases": ["band H/2 eta 1e-2", "1e-4", "0 (mu>0)", "3 bands H/4,H/2,3H/4 eta 0 (mu>0)"], "strong_threshold": THETA_S},
        "dynamics": {"wall": "1x4 d0.1 s1", "push": "v_x = 1 m/s on cells with centroid y > H - 0.25", "trajectories": ["BDF2 1/480 120 steps", "BDF2 1/60 30 steps", "Newmark 1/480 120 steps", "BDF2 1/480 with band eta 1e-2 from step 24 (steps 22..35)"],
            "predictors": {"BDF2": ["extrap: 3u_n-3u_{n-1}+u_{n-2} (lower order at start)", "qhat"], "Newmark": ["u+hv+h^2/2 a"]}, "closed_loop_k": [1, 2, 4]},
        "methods": {"one_level": ["GS1 (coloured block GS, forward)", "PCG-J"], "two_level": "P in {PR, SA}, s in {8, 16}, smoother in {GS fwd/bwd, Chebyshev deg 2 on [lam/8, lam]}, stationary or PCG",
            "lambda_hat": "1.05 x 15-step Lanczos of D^-1/2 A D^-1/2, capped at 2", "SA_omega": "4/(3 lambda_hat)"},
        "caps": {"one_level": CAP1, "two_level": CAP2},
        "targets": {"F": TOL, "E": TOL, "R_force": TOL, "R_moment": TOL, "chi_hi": CHI_TOL, "chi_threshold": 0.8, "chi_max_ref": 0.95},
        "metric_revision": "[rev, before the main runs, found on the E2 smoke test] force families: N and V share the joint force norm, M uses max(||M||, ||(N,V)|| l/sqrt12); 3D: translations share, rotations max(own, trans*l/sqrt12)",
        "chosen_variant_rule": "fewest censored, then smallest sum of A-equivalents (a_apps + coarse_solves*4nnz(Lc)/(2nnz(A))) to target over E3 dev walls, cold RHS, dynamic operators",
    })
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("reuse") {
        reuse_main();
        return;
    }
    if std::env::var("SB_TL_CHECKS_ONLY").is_ok() {
        let mut c = Vec::new();
        checks_small(&mut c);
        return;
    }
    let t_start = Instant::now();
    write_json(EXP, "manifest", &manifest());
    let mut out = Out {
        audits: Vec::new(),
        runs: Csv::new(EXP, "runs", &["case", "system", "rhs", "method", "cells", "iters", "censored", "a_apps", "phases", "reductions", "coarse_solves", "A_eq", "F0", "F1", "F2", "F4", "F8", "F16", "F32", "E8", "chi8", "F_last", "chi_all_last"]),
        dyn_open: Csv::new(EXP, "dyn_open", &["traj", "step", "t", "predictor", "aggregation", "method", "iters", "F0", "F1", "F2", "F4", "F8", "E0", "E1", "E2", "Einc0", "Einc1", "Einc2", "chi1", "Ftraj0"]),
        dyn_closed: Csv::new(EXP, "dyn_closed", &["traj", "policy", "method", "k", "step", "t", "F_traj", "E_traj", "chi_union", "energy_ratio"]),
        json: Vec::new(),
    };
    let methods = Method::all();
    let mut checks = Vec::new();
    checks_small(&mut checks);
    // ---------------- E2 blocks: implementation check
    for seed in [1u64, 2] {
        for (lname, loads) in [("uniform 1 MPa", vec![(None, V2::new(0.0, -1.0e6))]), ("patch (0.5,-2) MPa on [0,0.4]", vec![(Some((0.0, 0.4)), V2::new(0.5e6, -2.0e6))])] {
            let (mut p, m, _) = problem2(&format!("E2 s{seed} {lname}"), 1.0, 0.5, 0.05, seed, 0.1, &loads);
            study(&mut out, &p.name.clone(), &mut p, Some(&m), false, false, &methods);
        }
    }
    // ---------------- E3 walls: heights and refinement
    let load3 = [(None, V2::new(2.5e3, -1.0e4))];
    let mut walls: Vec<(f64, f64)> = vec![(1.0, 0.1), (2.0, 0.1), (4.0, 0.1), (8.0, 0.1), (4.0, 0.2), (4.0, 0.05)];
    walls.dedup();
    for seed in [1u64, 2] {
        for &(h, d) in &walls {
            let rep = seed == 1 && h == 4.0 && d == 0.1;
            let (mut p, m, _) = problem2(&format!("E3 wall 1x{h} d{d} s{seed}"), 1.0, h, d, seed, 0.2, &load3);
            study(&mut out, &p.name.clone(), &mut p, Some(&m), rep, true, &methods);
        }
    }
    let (chosen, scores) = choose_variant(&out.json);
    eprintln!("chosen variant for 3D: {chosen}");
    // ---------------- bands and fragments
    for seed in [1u64, 2] {
        band_study(&mut out, seed);
    }
    // ---------------- routine dynamic steps
    let mut dyn_json = Vec::new();
    dynamics(&mut out, &mut dyn_json);
    // ---------------- held-out
    for seed in [3001u64, 3002] {
        for h in [4.0, 8.0] {
            let (mut p, m, _) = problem2(&format!("held-out wall 1x{h} d0.1 s{seed}"), 1.0, h, 0.1, seed, 0.2, &load3);
            study(&mut out, &p.name.clone(), &mut p, Some(&m), false, true, &methods);
        }
    }
    // ---------------- 3D transfer (chosen variant fixed above; others supplementary)
    for seed in [1u64, 2] {
        for (name, ext, d) in [("cube 2m", V3::new(2.0, 2.0, 2.0), 0.35), ("cube 2m", V3::new(2.0, 2.0, 2.0), 0.2), ("column 1x1x4", V3::new(1.0, 1.0, 4.0), 0.2)] {
            let mut p = problem3(&format!("3D {name} d{d} s{seed}"), ext, d, seed, V3::new(1.0e4, 0.0, 0.0));
            study(&mut out, &p.name.clone(), &mut p, None, false, false, &methods);
        }
    }
    let doc = json!({
        "meta": meta(EXP, "all", None, "direct reference; GS1, PCG-J (cap 1024); two-level PR/SA x s8/s16 x GS/CH, stationary or PCG (cap 32)", manifest()),
        "cut_audits": out.audits, "checks": checks, "chosen_variant": chosen, "variant_scores": scores, "dynamics": dyn_json, "results": out.json,
        "seconds": t_start.elapsed().as_secs_f64(),
    });
    write_json(EXP, "summary", &doc);
    eprintln!("done in {:.0} s", t_start.elapsed().as_secs_f64());
}
