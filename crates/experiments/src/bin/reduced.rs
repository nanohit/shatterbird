//! Phase 0c Exp 1 — notched beam with reduced far regions coupled to a detailed
//! notch zone (docs/phase0c/reduced-fracture.md §1–2).
//!
//! One potential, one Newton system. Cells of a far region move as u = T ξ (static
//! modes of a boundary basis); facets inside the region contribute the precomputed
//! ½ ξᵀ K̂ ξ; every other facet is evaluated with the full nonlinear law on the
//! recovered positions. Full and reduced runs share the Newton code and the linear
//! solver, so the work comparison isolates the spatial reduction.

use experiments::{meta, write_json, Csv};
use geometry::rng::Rng;
use geometry::rsa::rsa_rect;
use geometry::voronoi2d::{rect, voronoi_convex, Tessellation};
use geometry::V2;
use model::{BodyKind, BoundaryAttach, LawKind, LawOptions, Material, Model};
use serde::Serialize;
use serde_json::{json, Value};
use solvers::assembly::Tangent;
use solvers::sparse::{csc_from_triplets, lu_solve, Spd};
use std::time::Instant;

const EXP: &str = "phase0c/exp1";
const DEPTH: f64 = 0.1;
const NOTCH: f64 = 0.05;
const B: f64 = 0.1;
const PAD: f64 = 0.01;
const CENTRAL: f64 = 0.08;

// ---------------------------------------------------------------- specimen (E6)

fn tess(len: f64, dmin: f64, seed: u64) -> Tessellation {
    let xc = len / 2.0;
    let mut rng = Rng::new(seed);
    let band = 2.0 * dmin;
    let ytop = NOTCH + dmin;
    let all = rsa_rect(V2::new(0.0, 0.0), V2::new(len, DEPTH), dmin, &mut rng, 20_000);
    let mut seeds: Vec<V2> = all.into_iter().filter(|p| !((p.x - xc).abs() < band && p.y < ytop)).collect();
    let mut fails = 0;
    let mut half = Vec::new();
    while fails < 5000 {
        let p = V2::new(rng.range(xc - band, xc - 0.5 * dmin), rng.range(0.0, ytop));
        let mirror = V2::new(2.0 * xc - p.x, p.y);
        let ok = seeds.iter().chain(half.iter()).all(|s: &V2| (*s - p).norm() >= dmin && (*s - mirror).norm() >= dmin);
        if ok {
            half.push(p);
            fails = 0;
        } else {
            fails += 1;
        }
    }
    for p in &half {
        seeds.push(*p);
        seeds.push(V2::new(2.0 * xc - p.x, p.y));
    }
    let t = voronoi_convex(&seeds, &rect(V2::new(0.0, 0.0), V2::new(len, DEPTH)));
    assert!(t.max_mismatch < 1e-12);
    t
}

type Ctrl = Vec<(usize, V2, V2, f64)>;

struct Beam {
    m: Model,
    plate: usize,
    cmod: Ctrl,
    xc: f64,
}

/// E6 notched beam, generalised to any length/span (same section, notch, pads and
/// central damage zone).
fn beam(len: f64, span: f64, dmin: f64, seed: u64) -> Beam {
    let t = tess(len, dmin, seed);
    let xc = len / 2.0;
    let mut m = Model::from_tessellation(&t, Material::concrete(B), LawOptions::damage(), &[]);
    for f in m.facets.iter_mut() {
        let on_plane = (f.ends[0].x - xc).abs() < 1e-12 && (f.ends[1].x - xc).abs() < 1e-12;
        if on_plane && f.pc.y < NOTCH {
            f.active = false;
        }
    }
    let g = m.ground();
    let x1 = (len - span) / 2.0;
    let x2 = x1 + span;
    m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: Some((x1 - PAD, x1 + PAD)), body: g, shear: false, tag: 1 });
    m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: Some((x2 - PAD, x2 + PAD)), body: g, shear: false, tag: 2 });
    let plate = m.add_plate(V2::new(xc, DEPTH), 0.0, 0.0, [true, false, true]);
    m.attach_boundary(&t, &BoundaryAttach { edge: 2, range: Some((xc - PAD, xc + PAD)), body: plate, shear: false, tag: 3 });
    for f in m.facets.iter_mut().filter(|f| f.tag != 0) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Damage;
            p.law.delta0 *= 1e-4;
            p.law.wc *= 1e-4;
        }
    }
    for f in m.facets.iter_mut().filter(|f| f.tag == 0) {
        if (f.pc.x - xc).abs() >= CENTRAL {
            for p in f.pts.iter_mut() {
                p.law.kind = LawKind::Elastic;
            }
        }
    }
    let gauge = (0..t.cells.len())
        .min_by(|&a, &b| (m.bodies[a].x0 - V2::new(0.05, 0.05)).norm().partial_cmp(&(m.bodies[b].x0 - V2::new(0.05, 0.05)).norm()).unwrap())
        .unwrap();
    m.bodies[gauge].fixed[0] = true;
    m.pattern[plate][1] = -1.0;
    let mouth = V2::new(xc, 0.0);
    let find = |left: bool| -> usize {
        (0..t.cells.len())
            .find(|&i| {
                let c = &t.cells[i];
                (if left { c.seed.x < xc } else { c.seed.x > xc }) && c.verts.iter().any(|v| (*v - mouth).norm() < 1e-12)
            })
            .expect("mouth cell")
    };
    let (l, r) = (find(true), find(false));
    let cmod = vec![(r, mouth - m.bodies[r].x0, V2::new(1.0, 0.0), 1.0), (l, mouth - m.bodies[l].x0, V2::new(1.0, 0.0), -1.0)];
    Beam { m, plate, cmod, xc }
}

// ---------------------------------------------------------------- reduced system

#[derive(Clone, Copy, Debug, PartialEq)]
enum Basis {
    Full,
    /// Exact static condensation onto all boundary-cell DOFs.
    M0,
    /// Section model: rigid section + polynomial warping to degree P (−1 = rigid only)
    /// on interface cells; support-pad cells keep exact DOFs.
    M1(i32),
}

impl Basis {
    fn label(&self) -> String {
        match self {
            Basis::Full => "full".into(),
            Basis::M0 => "M0".into(),
            Basis::M1(-1) => "M1-rigid".into(),
            Basis::M1(p) => format!("M1-P{p}"),
        }
    }
}

/// How a body's DOFs map into the unknown vector x.
#[derive(Clone, Debug)]
enum Rep {
    /// Unit coefficient on x[idx] (None = fixed DOF).
    Direct([Option<usize>; 3]),
    /// u_k = Σ_j rows[k][j] x[off + j] (boundary cell of a section-model region).
    /// Only the nonzero columns are stored: rows[k][c] multiplies x[off + cols[c]].
    Proj { r: usize, off: usize, cols: Vec<usize>, rows: [Vec<f64>; 3] },
    /// Region interior: recovered only for output and the validity monitor.
    Interior,
}

struct Region {
    off: usize,
    m: usize,
    /// region-local free DOFs (body, k); boundary first
    dofs: Vec<(usize, usize)>,
    nbd: usize,
    /// T: dofs.len() × m, row-major
    t: Vec<f64>,
    khat: Vec<f64>,
    internal: Vec<usize>,
    cells: usize,
    iface_cells: usize,
    pad_cells: usize,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct Work {
    newton_iters: u64,
    pts_tangent: u64,
    pts_force: u64,
    commit_pts: u64,
    coupling_flops: f64,
    khat_flops: f64,
    fact_flops: f64,
    solve_flops: f64,
    monitor_flops: f64,
}

/// Flop model: a facet point with tangent 200, force only 60 (jump + law + scatter).
const C_PT_TANGENT: f64 = 200.0;
const C_PT_FORCE: f64 = 60.0;

impl Work {
    fn total(&self) -> f64 {
        self.pts_tangent as f64 * C_PT_TANGENT
            + (self.pts_force + self.commit_pts) as f64 * C_PT_FORCE
            + self.coupling_flops
            + self.khat_flops
            + self.fact_flops
            + self.solve_flops
            + self.monitor_flops
    }
    fn facet_flops(&self) -> f64 {
        self.pts_tangent as f64 * C_PT_TANGENT + (self.pts_force + self.commit_pts) as f64 * C_PT_FORCE
    }
}

struct Sys {
    m: Model,
    n: usize,
    nd: usize,
    q0: Vec<[f64; 3]>,
    rep: Vec<Rep>,
    eval_facets: Vec<usize>,
    regions: Vec<Region>,
    pattern: Vec<f64>,
    dead: Vec<f64>,
    fact_flops: f64,
    nnz_l: usize,
    pre_flops: f64,
    pre_wall: f64,
    direct_cells: usize,
}

fn gram_schmidt(cands: Vec<Vec<f64>>, scale: &[f64]) -> Vec<Vec<f64>> {
    // orthonormalise in the scaled space (θ entries × h), drop dependent vectors
    let mut out: Vec<Vec<f64>> = Vec::new();
    for c in cands {
        let mut v: Vec<f64> = c.iter().zip(scale).map(|(a, s)| a * s).collect();
        let n0 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if n0 == 0.0 {
            continue;
        }
        for _ in 0..2 {
            for u in &out {
                let d: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
                for (a, b) in v.iter_mut().zip(u) {
                    *a -= d * b;
                }
            }
        }
        let n1 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if n1 > 1e-8 * n0 {
            out.push(v.iter().map(|a| a / n1).collect());
        }
    }
    out.into_iter().map(|v| v.iter().zip(scale).map(|(a, s)| a / s).collect()).collect()
}

impl Sys {
    fn new(m: Model, zone: Option<(f64, f64)>, basis: Basis) -> Sys {
        let t0 = Instant::now();
        let nb = m.bodies.len();
        let mut reg: Vec<Option<usize>> = vec![None; nb];
        let nreg = if let Some((xc, w)) = zone {
            for (i, b) in m.bodies.iter().enumerate() {
                if b.kind == BodyKind::Cell && (b.x0.x - xc).abs() >= w {
                    reg[i] = Some(if b.x0.x < xc { 0 } else { 1 });
                }
            }
            2
        } else {
            0
        };
        let mut eval_facets = Vec::new();
        let mut internal = vec![Vec::new(); nreg];
        for (fi, f) in m.facets.iter().enumerate() {
            if !f.active {
                continue;
            }
            match (reg[f.a], reg[f.b]) {
                (Some(ra), Some(rb)) if ra == rb => internal[ra].push(fi),
                _ => eval_facets.push(fi),
            }
        }
        let (mut bnd, mut iface, mut pad) = (vec![false; nb], vec![false; nb], vec![false; nb]);
        for &fi in &eval_facets {
            let f = &m.facets[fi];
            for (s, o) in [(f.a, f.b), (f.b, f.a)] {
                if reg[s].is_some() {
                    bnd[s] = true;
                    if m.bodies[o].kind == BodyKind::Cell {
                        iface[s] = true;
                    } else {
                        pad[s] = true;
                    }
                }
            }
        }
        let mut rep: Vec<Rep> = vec![Rep::Interior; nb];
        let mut n = 0;
        for i in 0..nb {
            if reg[i].is_none() {
                let mut idx = [None; 3];
                for k in 0..3 {
                    if !m.bodies[i].fixed[k] {
                        idx[k] = Some(n);
                        n += 1;
                    }
                }
                rep[i] = Rep::Direct(idx);
            }
        }
        let nd = n;
        let direct_cells = (0..nb).filter(|&i| reg[i].is_none() && m.bodies[i].kind == BodyKind::Cell).count();
        let q0: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
        let mut regions = Vec::new();
        let mut pre_flops = 0.0;
        for r in 0..nreg {
            let cells: Vec<usize> = (0..nb).filter(|&i| reg[i] == Some(r)).collect();
            let mut dofs = Vec::new();
            for &i in cells.iter().filter(|&&i| bnd[i]) {
                for k in 0..3 {
                    if !m.bodies[i].fixed[k] {
                        dofs.push((i, k));
                    }
                }
            }
            let nbd = dofs.len();
            for &i in cells.iter().filter(|&&i| !bnd[i]) {
                for k in 0..3 {
                    if !m.bodies[i].fixed[k] {
                        dofs.push((i, k));
                    }
                }
            }
            let nr = dofs.len();
            let mut loc = vec![[usize::MAX; 3]; nb];
            for (d, &(i, k)) in dofs.iter().enumerate() {
                loc[i][k] = d;
            }
            // linear stiffness of the region's internal facets at the reference state
            let mut trip = Vec::new();
            for &fi in &internal[r] {
                let f = &m.facets[fi];
                for p in 0..2 {
                    let (j, _) = m.eval_point(&q0, f, p);
                    let (kn, kt) = (f.pts[p].law.kn, f.pts[p].law.kt);
                    let jn = [j.jna[0], j.jna[1], j.jna[2], j.jnb[0], j.jnb[1], j.jnb[2]];
                    let jt = [j.jta[0], j.jta[1], j.jta[2], j.jtb[0], j.jtb[1], j.jtb[2]];
                    let ids = [loc[f.a][0], loc[f.a][1], loc[f.a][2], loc[f.b][0], loc[f.b][1], loc[f.b][2]];
                    for a in 0..6 {
                        if ids[a] == usize::MAX {
                            continue;
                        }
                        for c in 0..6 {
                            if ids[c] == usize::MAX {
                                continue;
                            }
                            let v = kn * jn[a] * jn[c] + kt * jt[a] * jt[c];
                            if v != 0.0 {
                                trip.push((ids[a], ids[c], v));
                            }
                        }
                    }
                }
            }
            // boundary basis T_B: columns over the nbd boundary DOFs
            let cols: Vec<Vec<f64>> = match basis {
                Basis::M0 => (0..nbd).map(|j| (0..nbd).map(|d| if d == j { 1.0 } else { 0.0 }).collect()).collect(),
                Basis::M1(pdeg) => {
                    let is_sec = |i: usize| iface[i] && !pad[i];
                    let sec: Vec<usize> = cells.iter().copied().filter(|&i| bnd[i] && is_sec(i)).collect();
                    let nsec = sec.len() as f64;
                    let xb = sec.iter().map(|&i| m.bodies[i].x0.x).sum::<f64>() / nsec;
                    let yb = sec.iter().map(|&i| m.bodies[i].x0.y).sum::<f64>() / nsec;
                    let h = sec.iter().map(|&i| (m.bodies[i].x0.y - yb).abs()).fold(0.0, f64::max);
                    let scale: Vec<f64> = dofs[..nbd].iter().map(|&(_, k)| if k == 2 { h } else { 1.0 }).collect();
                    let mut cands: Vec<Vec<f64>> = Vec::new();
                    let field = |f: &dyn Fn(usize, usize) -> f64| -> Vec<f64> {
                        dofs[..nbd].iter().map(|&(i, k)| if is_sec(i) { f(i, k) } else { 0.0 }).collect()
                    };
                    cands.push(field(&|_, k| if k == 0 { 1.0 } else { 0.0 }));
                    cands.push(field(&|_, k| if k == 1 { 1.0 } else { 0.0 }));
                    cands.push(field(&|i, k| {
                        let p = m.bodies[i].x0;
                        [-(p.y - yb), p.x - xb, 1.0][k]
                    }));
                    for p in 0..=pdeg.max(-1) {
                        for kk in 0..3 {
                            cands.push(field(&|i, k| if k == kk { ((m.bodies[i].x0.y - yb) / h).powi(p) } else { 0.0 }));
                        }
                    }
                    let mut cols = gram_schmidt(cands, &scale);
                    // pad cells: exact DOFs
                    for d in 0..nbd {
                        if !is_sec(dofs[d].0) {
                            cols.push((0..nbd).map(|e| if e == d { 1.0 } else { 0.0 }).collect());
                        }
                    }
                    cols
                }
                Basis::Full => unreachable!(),
            };
            let mr = cols.len();
            let ni = nr - nbd;
            let mut tii = Vec::new();
            let mut kib = Vec::new();
            for &(a, c, v) in &trip {
                if a >= nbd && c >= nbd {
                    tii.push((a - nbd, c - nbd, v));
                } else if a >= nbd {
                    kib.push((a - nbd, c, v));
                }
            }
            let kii = csc_from_triplets(ni, tii);
            let mut spd = Spd::analyze(&kii);
            spd.factor(&kii).expect("K_II not SPD: region interior not anchored by its boundary");
            pre_flops += spd.stats.flops_factor;
            let mut t = vec![0.0; nr * mr];
            for (j, col) in cols.iter().enumerate() {
                for d in 0..nbd {
                    t[d * mr + j] = col[d];
                }
                let mut rhs = vec![0.0; ni];
                for &(a, c, v) in &kib {
                    rhs[a] -= v * col[c];
                }
                spd.solve(&mut rhs);
                for a in 0..ni {
                    t[(nbd + a) * mr + j] = rhs[a];
                }
                pre_flops += 4.0 * spd.stats.nnz_l as f64 + 2.0 * kib.len() as f64;
            }
            let mut kt_ = vec![0.0; nr * mr];
            for &(a, c, v) in &trip {
                for j in 0..mr {
                    kt_[a * mr + j] += v * t[c * mr + j];
                }
            }
            let mut khat = vec![0.0; mr * mr];
            for a in 0..nr {
                for i in 0..mr {
                    let ta = t[a * mr + i];
                    if ta != 0.0 {
                        for j in 0..mr {
                            khat[i * mr + j] += ta * kt_[a * mr + j];
                        }
                    }
                }
            }
            for i in 0..mr {
                for j in 0..i {
                    let s = 0.5 * (khat[i * mr + j] + khat[j * mr + i]);
                    khat[i * mr + j] = s;
                    khat[j * mr + i] = s;
                }
            }
            pre_flops += 2.0 * (trip.len() * mr) as f64 + 2.0 * (nr * mr * mr) as f64;
            // boundary cells: direct indices (identity rows, M0) or projected rows (M1)
            for &i in cells.iter().filter(|&&i| bnd[i]) {
                let mut unit = [None; 3];
                let mut all_unit = true;
                let mut cols: Vec<usize> = Vec::new();
                for k in 0..3 {
                    let d = loc[i][k];
                    if d == usize::MAX {
                        continue;
                    }
                    let nz: Vec<usize> = (0..mr).filter(|&j| t[d * mr + j] != 0.0).collect();
                    if nz.len() == 1 && t[d * mr + nz[0]] == 1.0 {
                        unit[k] = Some(n + nz[0]);
                    } else {
                        all_unit = false;
                    }
                    for j in nz {
                        if !cols.contains(&j) {
                            cols.push(j);
                        }
                    }
                }
                cols.sort_unstable();
                let rows: [Vec<f64>; 3] = std::array::from_fn(|k| {
                    let d = loc[i][k];
                    cols.iter().map(|&j| if d == usize::MAX { 0.0 } else { t[d * mr + j] }).collect()
                });
                rep[i] = if all_unit { Rep::Direct(unit) } else { Rep::Proj { r, off: n, cols, rows } };
            }
            let ncell = cells.len();
            let iface_cells = cells.iter().filter(|&&i| bnd[i] && iface[i] && !pad[i]).count();
            let pad_cells = cells.iter().filter(|&&i| bnd[i] && pad[i]).count();
            regions.push(Region { off: n, m: mr, dofs, nbd, t, khat, internal: internal[r].clone(), cells: ncell, iface_cells, pad_cells });
            n += mr;
        }
        let mut pattern = vec![0.0; n];
        let mut dead = vec![0.0; n];
        for i in 0..nb {
            if let Rep::Direct(idx) = &rep[i] {
                if reg[i].is_none() {
                    for k in 0..3 {
                        if let Some(g) = idx[k] {
                            pattern[g] += m.pattern[i][k];
                            dead[g] += m.loads[i][k];
                        }
                    }
                }
            }
        }
        for rg in &regions {
            for (d, &(i, k)) in rg.dofs.iter().enumerate() {
                for j in 0..rg.m {
                    pattern[rg.off + j] += rg.t[d * rg.m + j] * m.pattern[i][k];
                    dead[rg.off + j] += rg.t[d * rg.m + j] * m.loads[i][k];
                }
            }
        }
        let mut s = Sys { m, n, nd, q0, rep, eval_facets, regions, pattern, dead, fact_flops: 0.0, nnz_l: 0, pre_flops, pre_wall: 0.0, direct_cells };
        let mut w = Work::default();
        let (_, trip) = s.assemble(&vec![0.0; n], 0.0, true, Tangent::Consistent, &mut w);
        let sp = Spd::analyze(&csc_from_triplets(n, trip));
        s.fact_flops = sp.stats.flops_factor;
        s.nnz_l = sp.stats.nnz_l;
        s.pre_wall = t0.elapsed().as_secs_f64();
        s
    }

    fn q_of(&self, x: &[f64]) -> Vec<[f64; 3]> {
        let mut q = self.q0.clone();
        for (i, r) in self.rep.iter().enumerate() {
            match r {
                Rep::Direct(idx) => {
                    for k in 0..3 {
                        if let Some(g) = idx[k] {
                            q[i][k] += x[g];
                        }
                    }
                }
                Rep::Proj { off, cols, rows, .. } => {
                    for k in 0..3 {
                        q[i][k] += rows[k].iter().zip(cols).map(|(c, &j)| c * x[off + j]).sum::<f64>();
                    }
                }
                Rep::Interior => {}
            }
        }
        q
    }

    /// All bodies, region interiors recovered through T.
    fn q_all(&self, x: &[f64]) -> Vec<[f64; 3]> {
        let mut q = self.q_of(x);
        for rg in &self.regions {
            for d in rg.nbd..rg.dofs.len() {
                let (i, k) = rg.dofs[d];
                q[i][k] = self.q0[i][k] + (0..rg.m).map(|j| rg.t[d * rg.m + j] * x[rg.off + j]).sum::<f64>();
            }
        }
        q
    }

    fn assemble(&self, x: &[f64], lf: f64, want_h: bool, tang: Tangent, w: &mut Work) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
        let q = self.q_of(x);
        let mut g = vec![0.0; self.n];
        let mut trip = Vec::new();
        for i in 0..self.n {
            g[i] -= self.dead[i] + lf * self.pattern[i];
        }
        let mut hreg: Vec<Vec<f64>> = self.regions.iter().map(|r| vec![0.0; r.m * r.m]).collect();
        // 3×3 diagonal blocks of projected cells, summed over their facet points and
        // projected once per cell (Tᵢᵀ Hᵢᵢ Tᵢ) after the facet loop
        let mut dblk: Vec<[[f64; 3]; 3]> = Vec::new();
        let mut dbody: Vec<usize> = Vec::new();
        let mut dslot = vec![usize::MAX; self.rep.len()];
        for &fi in &self.eval_facets {
            let f = &self.m.facets[fi];
            let sides = [&self.rep[f.a], &self.rep[f.b]];
            for p in 0..2 {
                let (j, e) = self.m.eval_point(&q, f, p);
                let jn = [j.jna[0], j.jna[1], j.jna[2], j.jnb[0], j.jnb[1], j.jnb[2]];
                let jt = [j.jta[0], j.jta[1], j.jta[2], j.jtb[0], j.jtb[1], j.jtb[2]];
                let gl: Vec<f64> = (0..6).map(|r| jn[r] * e.fnorm + jt[r] * e.ftan).collect();
                for s in 0..2 {
                    match sides[s] {
                        Rep::Direct(idx) => {
                            for k in 0..3 {
                                if let Some(gi) = idx[k] {
                                    g[gi] += gl[3 * s + k];
                                }
                            }
                        }
                        Rep::Proj { off, cols, rows, .. } => {
                            for k in 0..3 {
                                for (c, &jj) in rows[k].iter().zip(cols) {
                                    g[off + jj] += c * gl[3 * s + k];
                                }
                            }
                            w.coupling_flops += 6.0 * cols.len() as f64;
                        }
                        Rep::Interior => unreachable!(),
                    }
                }
                if !want_h {
                    w.pts_force += 1;
                    continue;
                }
                w.pts_tangent += 1;
                let d = match tang {
                    Tangent::Consistent => e.tan,
                    Tangent::Secant => {
                        let (kn, kt) = f.pts[p].law.secant(j.dn, e.d);
                        [kn, 0.0, kt]
                    }
                };
                let mut h = [[0.0; 6]; 6];
                for r in 0..6 {
                    let ar = d[0] * jn[r] + d[1] * jt[r];
                    let br = d[1] * jn[r] + d[2] * jt[r];
                    for c in 0..6 {
                        h[r][c] = ar * jn[c] + br * jt[c];
                    }
                }
                let bodies = [f.a, f.b];
                for sa in 0..2 {
                    for sb in 0..2 {
                        let blk = |k: usize, l: usize| h[3 * sa + k][3 * sb + l];
                        if sa == sb && matches!(sides[sa], Rep::Proj { .. }) {
                            let b = bodies[sa];
                            if dslot[b] == usize::MAX {
                                dslot[b] = dblk.len();
                                dblk.push([[0.0; 3]; 3]);
                                dbody.push(b);
                            }
                            for k in 0..3 {
                                for l in 0..3 {
                                    dblk[dslot[b]][k][l] += blk(k, l);
                                }
                            }
                            continue;
                        }
                        match (sides[sa], sides[sb]) {
                            (Rep::Direct(ia), Rep::Direct(ib)) => {
                                for k in 0..3 {
                                    for l in 0..3 {
                                        if let (Some(r), Some(c)) = (ia[k], ib[l]) {
                                            let v = blk(k, l);
                                            if v != 0.0 {
                                                trip.push((r, c, v));
                                            }
                                        }
                                    }
                                }
                            }
                            (Rep::Proj { off, cols, rows, .. }, Rep::Direct(ib)) => {
                                for l in 0..3 {
                                    let Some(c) = ib[l] else { continue };
                                    for (jj, &col) in cols.iter().enumerate() {
                                        let v: f64 = (0..3).map(|k| rows[k][jj] * blk(k, l)).sum();
                                        trip.push((off + col, c, v));
                                    }
                                }
                                w.coupling_flops += 18.0 * cols.len() as f64;
                            }
                            (Rep::Direct(ia), Rep::Proj { off, cols, rows, .. }) => {
                                for k in 0..3 {
                                    let Some(r) = ia[k] else { continue };
                                    for (jj, &col) in cols.iter().enumerate() {
                                        let v: f64 = (0..3).map(|l| blk(k, l) * rows[l][jj]).sum();
                                        trip.push((r, off + col, v));
                                    }
                                }
                                w.coupling_flops += 18.0 * cols.len() as f64;
                            }
                            (Rep::Proj { r: ra, off: oa, cols: ca, rows: rowa }, Rep::Proj { r: rb, off: ob, cols: cb, rows: rowb }) => {
                                // two different projected cells (not produced by the beam partition)
                                for (ii, &ci) in ca.iter().enumerate() {
                                    for (jj, &cj) in cb.iter().enumerate() {
                                        let v: f64 = (0..3).map(|k| (0..3).map(|l| rowa[k][ii] * blk(k, l) * rowb[l][jj]).sum::<f64>()).sum();
                                        if ra == rb {
                                            hreg[*ra][ci * self.regions[*ra].m + cj] += v;
                                        } else {
                                            trip.push((oa + ci, ob + cj, v));
                                        }
                                    }
                                }
                                w.coupling_flops += 18.0 * (ca.len() * cb.len()) as f64;
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }
        for (slot, &b) in dbody.iter().enumerate() {
            let Rep::Proj { r, cols, rows, .. } = &self.rep[b] else { unreachable!() };
            let mr = self.regions[*r].m;
            let d = &dblk[slot];
            let mc = cols.len();
            let mut ht = vec![[0.0; 3]; mc];
            for jj in 0..mc {
                for k in 0..3 {
                    ht[jj][k] = (0..3).map(|l| d[k][l] * rows[l][jj]).sum();
                }
            }
            for ii in 0..mc {
                for jj in 0..mc {
                    hreg[*r][cols[ii] * mr + cols[jj]] += (0..3).map(|k| rows[k][ii] * ht[jj][k]).sum::<f64>();
                }
            }
            w.coupling_flops += 18.0 * mc as f64 + 6.0 * (mc * mc) as f64;
        }
        for (ri, rg) in self.regions.iter().enumerate() {
            let xi = &x[rg.off..rg.off + rg.m];
            for a in 0..rg.m {
                g[rg.off + a] += (0..rg.m).map(|b| rg.khat[a * rg.m + b] * xi[b]).sum::<f64>();
            }
            w.khat_flops += 2.0 * (rg.m * rg.m) as f64;
            if want_h {
                for a in 0..rg.m {
                    for b in 0..rg.m {
                        trip.push((rg.off + a, rg.off + b, rg.khat[a * rg.m + b] + hreg[ri][a * rg.m + b]));
                    }
                }
                w.khat_flops += (rg.m * rg.m) as f64;
            }
        }
        (g, trip)
    }

    /// Elastic energy: evaluated facets + ½ ξᵀ K̂ ξ.
    fn elastic_energy(&self, x: &[f64]) -> f64 {
        let q = self.q_of(x);
        let mut u = 0.0;
        for &fi in &self.eval_facets {
            for p in 0..2 {
                u += self.m.eval_point(&q, &self.m.facets[fi], p).1.psi;
            }
        }
        for rg in &self.regions {
            let xi = &x[rg.off..rg.off + rg.m];
            for a in 0..rg.m {
                for b in 0..rg.m {
                    u += 0.5 * xi[a] * rg.khat[a * rg.m + b] * xi[b];
                }
            }
        }
        u
    }

    fn material_dd(&self, x: &[f64], w: &mut Work) -> f64 {
        let q = self.q_of(x);
        let mut mx: f64 = 0.0;
        for &fi in &self.eval_facets {
            let f = &self.m.facets[fi];
            if f.tag != 0 {
                continue;
            }
            for p in 0..2 {
                let (_, e) = self.m.eval_point(&q, f, p);
                mx = mx.max(e.d - f.pts[p].law.damage(f.pts[p].kappa_c));
                w.commit_pts += 1;
            }
        }
        mx
    }

    fn commit(&mut self, x: &[f64], w: &mut Work) -> f64 {
        let q = self.q_of(x);
        let mut dd = 0.0;
        for &fi in &self.eval_facets {
            for p in 0..2 {
                let (_, e) = self.m.eval_point(&q, &self.m.facets[fi], p);
                let pt = &mut self.m.facets[fi].pts[p];
                if e.kappa > pt.kappa_c {
                    dd += pt.law.dissipation(e.kappa) - pt.law.dissipation(pt.kappa_c);
                    pt.kappa_c = e.kappa;
                }
                w.commit_pts += 1;
            }
        }
        self.m.dissipated += dd;
        dd
    }

    /// Validity monitor: max χ = δ_eq/δ₀ over region-internal facet points that are
    /// damageable in the reference (and over all of them), at the recovered state.
    fn monitor(&self, x: &[f64], w: &mut Work) -> (f64, f64, V2) {
        if self.regions.is_empty() {
            return (0.0, 0.0, V2::ZERO);
        }
        let q = self.q_all(x);
        let (mut cd, mut ca, mut at) = (0.0f64, 0.0f64, V2::ZERO);
        for rg in &self.regions {
            w.monitor_flops += 2.0 * (rg.dofs.len() * rg.m) as f64;
            for &fi in &rg.internal {
                let f = &self.m.facets[fi];
                for p in 0..2 {
                    let pt = &f.pts[p];
                    let j = model::kin::jump(q[f.a], q[f.b], pt.arm_a, pt.arm_b, f.n0, f.t0, pt.rest, f.frame_w);
                    let dn = j.dn.max(0.0);
                    let chi = (dn * dn + pt.law.gamma * j.dt * j.dt).sqrt() / pt.law.delta0;
                    ca = ca.max(chi);
                    if pt.law.kind == LawKind::Damage && chi > cd {
                        cd = chi;
                        at = f.pc;
                    }
                    w.monitor_flops += C_PT_FORCE;
                }
            }
        }
        (cd, ca, at)
    }
}

fn ctrl_value(c: &Ctrl, q: &[[f64; 3]]) -> f64 {
    c.iter().map(|&(b, arm, dir, wt)| wt * dir.dot(V2::new(q[b][0], q[b][1]) + arm.rot(q[b][2]))).sum()
}

fn ctrl_grad(s: &Sys, c: &Ctrl, q: &[[f64; 3]]) -> Vec<(usize, f64)> {
    let mut g = Vec::new();
    for &(b, arm, dir, wt) in c {
        let ra = arm.rot(q[b][2]).perp();
        let d = [wt * dir.x, wt * dir.y, wt * dir.dot(ra)];
        let Rep::Direct(idx) = &s.rep[b] else { panic!("control on a non-direct body") };
        for k in 0..3 {
            if let Some(i) = idx[k] {
                g.push((i, d[k]));
            }
        }
    }
    g
}

/// Bordered Newton [H −p; cᵀ 0][Δx; Δλ] = [−r; c* − c] (as solvers::s0::newton_controlled).
fn newton_ctrl(s: &Sys, x: &mut Vec<f64>, lf: &mut f64, ctrl: &Ctrl, target: f64, tol: f64, max_iter: usize, tang: Tangent, w: &mut Work) -> bool {
    let n = s.n;
    let ctol = 1e-12;
    let mut res0 = f64::NAN;
    let mut best = f64::INFINITY;
    let mut stall = 0;
    for it in 0..max_iter {
        let (g, mut trip) = s.assemble(x, *lf, true, tang, w);
        let res = g.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let q = s.q_of(x);
        let cv = ctrl_value(ctrl, &q);
        if res <= tol && (cv - target).abs() <= ctol {
            return true;
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
                return true;
            }
        }
        for (gi, &p) in s.pattern.iter().enumerate() {
            if p != 0.0 {
                trip.push((gi, n, -p));
            }
        }
        for (gi, v) in ctrl_grad(s, ctrl, &q) {
            trip.push((n, gi, v));
        }
        let mat = csc_from_triplets(n + 1, trip);
        let mut rhs: Vec<f64> = g.iter().map(|v| -v).collect();
        rhs.push(target - cv);
        let Ok(dx) = lu_solve(&mat, &rhs) else { return false };
        if !dx.iter().all(|v| v.is_finite()) {
            return false;
        }
        w.newton_iters += 1;
        w.fact_flops += s.fact_flops;
        w.solve_flops += 4.0 * s.nnz_l as f64;
        for i in 0..n {
            x[i] += dx[i];
        }
        *lf += dx[n];
        let pmax = dx[..n].iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let scale = q.iter().fold(1.0f64, |a, r| a.max(r[0].abs()).max(r[1].abs()));
        if pmax <= 8.0 * f64::EPSILON * scale && (cv - target).abs() <= ctol {
            return true;
        }
    }
    let (g, _) = s.assemble(x, *lf, false, tang, w);
    let res = g.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    res <= tol && (ctrl_value(ctrl, &s.q_of(x)) - target).abs() <= ctol
}

// ---------------------------------------------------------------- one run

#[derive(Clone, Serialize)]
struct RunOut {
    label: String,
    instance: String,
    w_mm: f64,
    // sizes
    n: usize,
    n_direct: usize,
    direct_cells: usize,
    cells: usize,
    region_m: Vec<usize>,
    region_cells: Vec<usize>,
    region_iface_cells: Vec<usize>,
    region_pad_cells: Vec<usize>,
    eval_facets: usize,
    facets: usize,
    nnz_l: usize,
    fact_flops_per_iter: f64,
    // results
    steps: usize,
    cutbacks: usize,
    secant_steps: usize,
    failed_at_cmod: Option<f64>,
    peak_load_n: f64,
    cmod_at_peak: f64,
    onset_load_n: Option<f64>,
    onset_cmod: Option<f64>,
    onset_at: Option<[f64; 2]>,
    /// post-hoc: first fully broken material facet (d ≥ 0.99) and 5 % secant drop
    break_load_n: Option<f64>,
    break_at: Option<[f64; 2]>,
    nonlin_load_n: Option<f64>,
    k_cmod: f64,
    k_disp: f64,
    w_ext: f64,
    u_end: f64,
    d_end: f64,
    ledger_max_rel: f64,
    chi_r_damageable: f64,
    chi_r_all: f64,
    chi_r_at: [f64; 2],
    coupling_dmax: f64,
    // work
    work: Work,
    work_total: f64,
    pre_flops: f64,
    pre_wall_s: f64,
    run_wall_s: f64,
    #[serde(skip)]
    curve: Vec<[f64; 6]>,
    #[serde(skip)]
    fdiss: Vec<f64>,
    #[serde(skip)]
    fdmax: Vec<f64>,
    #[serde(skip)]
    fpc: Vec<V2>,
}

fn run(bm: Beam, inst: &str, zone_w: Option<f64>, basis: Basis, dmin: f64) -> RunOut {
    let zone = zone_w.map(|w| (bm.xc, w));
    let mut s = Sys::new(bm.m, zone, basis);
    let fscale = s.m.mat.sigma_t * DEPTH * B;
    let tol = 1e-9 * fscale;
    let dd_max = 0.1;
    let cmod_end = 1.0e-3;
    let plate_y = match &s.rep[bm.plate] {
        Rep::Direct(idx) => idx[1].unwrap(),
        _ => unreachable!(),
    };
    let mut w = Work::default();
    let mut x = vec![0.0; s.n];
    let c0 = ctrl_value(&bm.cmod, &s.q_of(&x));
    let (mut lf, mut c) = (0.0f64, 0.0f64);
    let (ds_max, mut ds) = (4e-6f64, 0.5e-6f64);
    let (mut wext, mut steps, mut cuts, mut secant) = (0.0, 0usize, 0usize, 0usize);
    let (mut peak, mut cpeak) = (0.0f64, 0.0);
    let mut onset: Option<(f64, f64, V2)> = None;
    let mut brk: Option<(f64, V2)> = None;
    let mut nonlin: Option<f64> = None;
    let mut failed = None;
    let (mut ledger, mut chid, mut chia, mut chiat) = (0.0f64, 0.0f64, 0.0f64, V2::ZERO);
    let (mut kc, mut kd) = (0.0, 0.0);
    let mut curve = vec![[0.0; 6]];
    let t0 = Instant::now();
    while c < cmod_end {
        let tgt = (c + ds).min(cmod_end);
        let mut ok = false;
        let (mut xt, mut lft) = (x.clone(), lf);
        for tang in [Tangent::Consistent, Tangent::Secant] {
            xt = x.clone();
            lft = lf;
            let it = if tang == Tangent::Secant { 200 } else { 25 };
            if newton_ctrl(&s, &mut xt, &mut lft, &bm.cmod, c0 + tgt, tol, it, tang, &mut w) && s.material_dd(&xt, &mut w) <= dd_max {
                ok = true;
                if tang == Tangent::Secant {
                    secant += 1;
                }
                break;
            }
        }
        if !ok {
            ds *= 0.5;
            cuts += 1;
            if ds < 1e-11 {
                failed = Some(c);
                break;
            }
            continue;
        }
        wext += (0..s.n).map(|i| 0.5 * (lf + lft) * s.pattern[i] * (xt[i] - x[i])).sum::<f64>();
        let before: Vec<f64> = s.m.facets.iter().map(|f| f.pts.iter().map(|p| p.law.dissipation(p.kappa_c)).sum()).collect();
        s.commit(&xt, &mut w);
        if onset.is_none() && s.m.dissipated > 0.0 {
            let (mut sw, mut sp) = (0.0, V2::ZERO);
            for (fi, f) in s.m.facets.iter().enumerate() {
                let dd: f64 = f.pts.iter().map(|p| p.law.dissipation(p.kappa_c)).sum::<f64>() - before[fi];
                if dd > 0.0 {
                    sw += dd;
                    sp = sp + f.pc * dd;
                }
            }
            onset = Some((lft, tgt, sp * (1.0 / sw)));
        }
        x = xt;
        lf = lft;
        c = tgt;
        steps += 1;
        if steps == 1 {
            kc = lf / c;
            kd = lf / (-x[plate_y]);
        }
        if brk.is_none() {
            if let Some(&fi) = s.eval_facets.iter().find(|&&fi| {
                let f = &s.m.facets[fi];
                f.tag == 0 && f.pts.iter().any(|p| p.law.damage(p.kappa_c) >= 0.99)
            }) {
                brk = Some((lf, s.m.facets[fi].pc));
            }
        }
        if nonlin.is_none() && steps > 1 && lf / c < 0.95 * kc {
            nonlin = Some(lf);
        }
        if lf > peak {
            peak = lf;
            cpeak = c;
        }
        let u = s.elastic_energy(&x);
        ledger = ledger.max((wext - u - s.m.dissipated).abs() / wext.max(1e-300));
        let (cd, ca, at) = s.monitor(&x, &mut w);
        if cd > chid {
            chid = cd;
            chiat = at;
        }
        chia = chia.max(ca);
        curve.push([c, -x[plate_y], lf, wext, u, s.m.dissipated]);
        ds = (ds * 1.3).min(ds_max);
    }
    let run_wall = t0.elapsed().as_secs_f64();
    let u_end = s.elastic_energy(&x);
    let fdiss: Vec<f64> = s.m.facets.iter().map(|f| if f.active && f.tag == 0 { f.pts.iter().map(|p| p.law.dissipation(p.kappa_c)).sum() } else { 0.0 }).collect();
    let fdmax: Vec<f64> = s.m.facets.iter().map(|f| if f.active && f.tag == 0 { f.pts.iter().map(|p| p.law.damage(p.kappa_c)).fold(0.0, f64::max) } else { 0.0 }).collect();
    let fpc: Vec<V2> = s.m.facets.iter().map(|f| f.pc).collect();
    // largest damage among evaluated facets touching a region cell (coupling facets)
    let coupling_dmax = s
        .eval_facets
        .iter()
        .filter(|&&fi| {
            let f = &s.m.facets[fi];
            matches!(s.rep[f.a], Rep::Proj { .. }) || matches!(s.rep[f.b], Rep::Proj { .. }) || s.regions.iter().any(|rg| rg.dofs[..rg.nbd].iter().any(|&(i, _)| i == f.a || i == f.b))
        })
        .map(|&fi| fdmax[fi])
        .fold(0.0, f64::max);
    let _ = dmin;
    RunOut {
        label: basis.label(),
        instance: inst.to_string(),
        w_mm: zone_w.map(|v| v * 1e3).unwrap_or(f64::NAN),
        n: s.n,
        n_direct: s.nd,
        direct_cells: s.direct_cells,
        cells: s.m.n_cells(),
        region_m: s.regions.iter().map(|r| r.m).collect(),
        region_cells: s.regions.iter().map(|r| r.cells).collect(),
        region_iface_cells: s.regions.iter().map(|r| r.iface_cells).collect(),
        region_pad_cells: s.regions.iter().map(|r| r.pad_cells).collect(),
        eval_facets: s.eval_facets.len(),
        facets: s.m.facets.iter().filter(|f| f.active).count(),
        nnz_l: s.nnz_l,
        fact_flops_per_iter: s.fact_flops,
        steps,
        cutbacks: cuts,
        secant_steps: secant,
        failed_at_cmod: failed,
        peak_load_n: peak,
        cmod_at_peak: cpeak,
        onset_load_n: onset.map(|o| o.0),
        onset_cmod: onset.map(|o| o.1),
        onset_at: onset.map(|o| [o.2.x, o.2.y]),
        break_load_n: brk.map(|b| b.0),
        break_at: brk.map(|b| [b.1.x, b.1.y]),
        nonlin_load_n: nonlin,
        k_cmod: kc,
        k_disp: kd,
        w_ext: wext,
        u_end,
        d_end: s.m.dissipated,
        ledger_max_rel: ledger,
        chi_r_damageable: chid,
        chi_r_all: chia,
        chi_r_at: [chiat.x, chiat.y],
        coupling_dmax,
        work_total: w.total(),
        work: w,
        pre_flops: s.pre_flops,
        pre_wall_s: s.pre_wall,
        run_wall_s: run_wall,
        curve,
        fdiss,
        fdmax,
        fpc,
    }
}

// ---------------------------------------------------------------- comparison

fn interp(curve: &[[f64; 6]], c: f64, col: usize) -> f64 {
    let k = curve.partition_point(|r| r[0] < c);
    if k == 0 {
        return curve[0][col];
    }
    if k >= curve.len() {
        return curve[curve.len() - 1][col];
    }
    let (a, b) = (&curve[k - 1], &curve[k]);
    let t = (c - a[0]) / (b[0] - a[0]);
    a[col] + t * (b[col] - a[col])
}

fn crack_path(r: &RunOut, xc: f64) -> Vec<Option<f64>> {
    // dissipation-weighted mean x per 5 mm band of height, from the notch tip to the top
    let nb = ((DEPTH - NOTCH) / 0.005).round() as usize;
    let mut acc = vec![(0.0, 0.0); nb];
    for (fi, &d) in r.fdiss.iter().enumerate() {
        if d > 0.0 {
            let y = r.fpc[fi].y;
            if y >= NOTCH && y < DEPTH {
                let b = (((y - NOTCH) / 0.005) as usize).min(nb - 1);
                acc[b].0 += d;
                acc[b].1 += d * (r.fpc[fi].x - xc);
            }
        }
    }
    acc.iter().map(|&(w, s)| if w > 0.0 { Some(s / w) } else { None }).collect()
}

fn crack_height(r: &RunOut) -> f64 {
    r.fdmax.iter().enumerate().filter(|(_, &d)| d >= 0.99).map(|(fi, _)| r.fpc[fi].y).fold(NOTCH, f64::max)
}

fn compare(full: &RunOut, red: &RunOut, xc: f64, dmin: f64) -> Value {
    let rel = |a: f64, b: f64| (a - b) / b;
    let p_ref = full.peak_load_n;
    let mut max_dp = 0.0f64;
    for row in &full.curve[1..] {
        if row[0] <= red.curve.last().unwrap()[0] {
            max_dp = max_dp.max((interp(&red.curve, row[0], 2) - row[2]).abs() / p_ref);
        }
    }
    let pa = crack_path(full, xc);
    let pb = crack_path(red, xc);
    let (mut sdx, mut nbands, mut missing) = (0.0, 0, 0);
    for (a, b) in pa.iter().zip(&pb) {
        match (a, b) {
            (Some(a), Some(b)) => {
                sdx += (a - b).abs();
                nbands += 1;
            }
            (Some(_), None) | (None, Some(_)) => missing += 1,
            _ => {}
        }
    }
    let mean_dx = if nbands > 0 { sdx / nbands as f64 } else { f64::NAN };
    let ref_set: Vec<V2> = full.fdiss.iter().enumerate().filter(|(_, &d)| d > 0.0).map(|(i, _)| full.fpc[i]).collect();
    let spurious: f64 = red
        .fdiss
        .iter()
        .enumerate()
        .filter(|(i, &d)| d > 0.0 && ref_set.iter().all(|p| (*p - red.fpc[*i]).norm() > 2.0 * dmin))
        .map(|(_, &d)| d)
        .sum::<f64>()
        / red.d_end.max(1e-300);
    let (sa, sb): (Vec<usize>, Vec<usize>) = (
        (0..full.fdiss.len()).filter(|&i| full.fdiss[i] > 0.0).collect(),
        (0..red.fdiss.len()).filter(|&i| red.fdiss[i] > 0.0).collect(),
    );
    let inter = sa.iter().filter(|i| sb.contains(i)).count();
    let jacc = inter as f64 / (sa.len() + sb.len() - inter).max(1) as f64;
    let lig = DEPTH - NOTCH;
    let (ha, hb) = (crack_height(full), crack_height(red));
    let onset_dist = match (full.onset_at, red.onset_at) {
        (Some(a), Some(b)) => ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt(),
        _ => f64::NAN,
    };
    let d_onset = match (red.onset_load_n, full.onset_load_n) {
        (Some(a), Some(b)) => rel(a, b),
        _ => f64::NAN,
    };
    let crit = json!({
        "1_peak": rel(red.peak_load_n, p_ref).abs() <= 0.10,
        "2_onset": d_onset.abs() <= 0.10 && onset_dist <= 2.0 * dmin,
        "3_curve": max_dp <= 0.10 && rel(red.k_cmod, full.k_cmod).abs() <= 0.10,
        "4_energy": rel(red.w_ext, full.w_ext).abs() <= 0.10 && rel(red.d_end, full.d_end).abs() <= 0.10 && red.ledger_max_rel <= 1e-3,
        "5_crack": mean_dx <= 2.0 * dmin && ((hb - ha) / lig).abs() <= 0.10 && spurious <= 0.05,
        "6_valid": red.chi_r_damageable < 1.0 && red.failed_at_cmod.is_none(),
    });
    let all = crit.as_object().unwrap().values().all(|v| v.as_bool() == Some(true));
    json!({
        "label": red.label, "w_mm": red.w_mm,
        "peak_rel": rel(red.peak_load_n, p_ref),
        "onset_rel": d_onset, "onset_dist_mm": onset_dist * 1e3,
        "posthoc_break_rel": match (red.break_load_n, full.break_load_n) { (Some(a), Some(b)) => rel(a, b), _ => f64::NAN },
        "posthoc_break_dist_mm": match (red.break_at, full.break_at) { (Some(a), Some(b)) => ((a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2)).sqrt() * 1e3, _ => f64::NAN },
        "posthoc_nonlin_rel": match (red.nonlin_load_n, full.nonlin_load_n) { (Some(a), Some(b)) => rel(a, b), _ => f64::NAN },
        "curve_max_dP_over_Ppeak": max_dp,
        "k_cmod_rel": rel(red.k_cmod, full.k_cmod), "k_disp_rel": rel(red.k_disp, full.k_disp),
        "W_rel": rel(red.w_ext, full.w_ext), "D_rel": rel(red.d_end, full.d_end), "ledger": red.ledger_max_rel,
        "crack_mean_dx_mm": mean_dx * 1e3, "crack_bands_missing": missing,
        "crack_height_ref_mm": ha * 1e3, "crack_height_red_mm": hb * 1e3,
        "spurious_D_frac": spurious, "jaccard_dissipating": jacc,
        "dissipating_facets": [sa.len(), sb.len()],
        "chi_R": red.chi_r_damageable, "chi_R_all": red.chi_r_all, "coupling_dmax": red.coupling_dmax,
        "work_ratio_total": full.work_total / red.work_total,
        "work_ratio_total_incl_precompute": full.work_total / (red.work_total + red.pre_flops),
        "work_ratio_facets": full.work.facet_flops() / (red.work.facet_flops() + red.work.coupling_flops + red.work.khat_flops + red.work.monitor_flops),
        "work_ratio_factor": full.work.fact_flops / red.work.fact_flops,
        "unknowns_ratio": full.n as f64 / red.n as f64,
        "wall_ratio": full.run_wall_s / red.run_wall_s,
        "wall_ratio_incl_precompute": full.run_wall_s / (red.run_wall_s + red.pre_wall_s),
        "criteria": crit, "all_pass": all,
    })
}

// ---------------------------------------------------------------- driver

#[derive(Clone)]
struct Inst {
    name: String,
    len: f64,
    span: f64,
    dmin: f64,
    seed: u64,
}

fn main() {
    faer::set_global_parallelism(faer::Par::Seq);
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("exp1");
    let lab = |dmin: f64, seed: u64| Inst { name: format!("lab_d{}_s{seed}", (dmin * 1e4).round() / 10.0), len: 0.5, span: 0.4, dmin, seed };
    let long = |dmin: f64, seed: u64| Inst { name: format!("long_d{}_s{seed}", (dmin * 1e4).round() / 10.0), len: 1.7, span: 1.6, dmin, seed };
    // (instance, zone half-width, basis)
    let mut tasks: Vec<(Inst, Option<f64>, Basis)> = Vec::new();
    match mode {
        "exp1" => {
            for seed in [1u64, 2, 3] {
                let i = lab(0.0056, seed);
                tasks.push((i.clone(), None, Basis::Full));
                tasks.push((i.clone(), Some(0.05), Basis::M1(3)));
                tasks.push((i.clone(), Some(0.025), Basis::M0));
            }
            let i = lab(0.0056, 1);
            for w in [0.025, 0.05, 0.1] {
                for b in [Basis::M1(-1), Basis::M1(1), Basis::M1(3)] {
                    if !(w == 0.05 && b == Basis::M1(3)) {
                        tasks.push((i.clone(), Some(w), b));
                    }
                }
            }
            tasks.push((i.clone(), Some(0.05), Basis::M0));
            for i in [lab(0.004, 1), long(0.0056, 1)] {
                tasks.push((i.clone(), None, Basis::Full));
                tasks.push((i.clone(), Some(0.05), Basis::M1(3)));
                tasks.push((i.clone(), Some(0.025), Basis::M0));
            }
        }
        "quick" => {
            let i = lab(0.008, 1);
            tasks.push((i.clone(), None, Basis::Full));
            tasks.push((i.clone(), Some(0.05), Basis::M1(3)));
            tasks.push((i.clone(), Some(0.025), Basis::M0));
        }
        _ => panic!("modes: exp1 | quick"),
    }
    // longest first
    tasks.sort_by(|a, b| {
        let cost = |t: &(Inst, Option<f64>, Basis)| t.0.len / t.0.dmin.powi(2) * if t.2 == Basis::Full { 100.0 } else { 1.0 };
        cost(b).partial_cmp(&cost(a)).unwrap()
    });
    let nt: usize = std::env::var("SB_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let queue = std::sync::Mutex::new(tasks.clone().into_iter().enumerate().collect::<Vec<_>>());
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..nt {
            sc.spawn(|| loop {
                let job = queue.lock().unwrap().pop();
                let Some((k, (inst, w, basis))) = job else { break };
                let bm = beam(inst.len, inst.span, inst.dmin, inst.seed);
                let r = run(bm, &inst.name, w, basis, inst.dmin);
                eprintln!(
                    "[{k}] {} {} w={:?} n={} peak={:.1} D={:.4} chiR={:.3} steps={} iters={} work={:.3e} wall={:.1}s",
                    inst.name, r.label, w, r.n, r.peak_load_n, r.d_end, r.chi_r_damageable, r.steps, r.work.newton_iters, r.work_total, r.run_wall_s
                );
                results.lock().unwrap().push((k, inst, r));
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| r.0);
    let mut out = Vec::new();
    let mut names: Vec<String> = results.iter().map(|r| r.1.name.clone()).collect();
    names.dedup();
    names.sort();
    names.dedup();
    for name in &names {
        let runs: Vec<&(usize, Inst, RunOut)> = results.iter().filter(|r| &r.1.name == name).collect();
        let inst = &runs[0].1;
        let xc = inst.len / 2.0;
        let full = runs.iter().find(|r| r.2.label == "full").map(|r| &r.2);
        let mut cmp = Vec::new();
        let mut csv = Csv::new(EXP, &format!("curves_{name}"), &["config", "w_mm", "cmod", "disp", "load", "W", "U", "D"]);
        let mut dcsv = Csv::new(EXP, &format!("damage_{name}"), &["config", "w_mm", "x", "y", "D_J", "d_max"]);
        for (ci, r) in runs.iter().enumerate() {
            for row in &r.2.curve {
                csv.row(&[ci as f64, r.2.w_mm, row[0], row[1], row[2], row[3], row[4], row[5]]);
            }
            for (fi, &d) in r.2.fdiss.iter().enumerate() {
                if d > 0.0 {
                    dcsv.row(&[ci as f64, r.2.w_mm, r.2.fpc[fi].x - xc, r.2.fpc[fi].y, d, r.2.fdmax[fi]]);
                }
            }
            if let Some(f) = full {
                if r.2.label != "full" {
                    let c = compare(f, &r.2, xc, inst.dmin);
                    eprintln!("{name} {c}");
                    cmp.push(c);
                }
            }
        }
        out.push(json!({
            "instance": name, "len": inst.len, "span": inst.span, "dmin": inst.dmin, "seed": inst.seed,
            "configs_in_curves_csv": runs.iter().map(|r| format!("{} w={}", r.2.label, r.2.w_mm)).collect::<Vec<_>>(),
            "runs": runs.iter().map(|r| serde_json::to_value(&r.2).unwrap()).collect::<Vec<_>>(),
            "comparison": cmp,
        }));
    }
    let res = json!({
        "meta": meta(EXP, mode, None, "S0 bordered Newton under CMOD control to 1 mm (consistent tangent, secant fallback), accept if converged and max Δd ≤ 0.1 over material facets, else halve; history committed on acceptance only; same code and linear solver (faer sparse LU, sequential) for full and reduced",
            json!({"specimen": "E6 notched beam generalised (depth 0.1, notch 0.05, thickness 0.1, pads ±0.01, damage |x-xc|<0.08)",
                   "reduced": "far regions u = T ξ, T_I = -K_II^-1 K_IB T_B; M0: T_B = I; M1(P): rigid section + polynomial warping to degree P on interface cells, pads exact; region-internal facets linear elastic (monitored)",
                   "flop_model": {"facet_point_with_tangent": C_PT_TANGENT, "facet_point_force_only": C_PT_FORCE, "factorisation": "symbolic Cholesky flop estimate of the Newton-matrix pattern per Newton iteration", "solve": "4 nnz(L)"},
                   "prereg": "docs/phase0c/reduced-fracture.md §2"})),
        "instances": out,
    });
    write_json(EXP, &format!("summary_{mode}"), &res);
}
