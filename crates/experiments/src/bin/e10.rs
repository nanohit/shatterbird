//! E10 — small 3D: 6-DOF Voronoi cells, 3 translational + 3 rotational springs per
//! facet (distributed-spring integrals: k_n = EA/ℓ, k_t = γk_n, bending k_n·I/A about
//! the principal in-plane axes, torsion k_t·I_p/A), linearised kinematics.
//! (1) two-cell checks; (2) per-cell 6×6 blocks normalised by the mass/inertia metric vs
//! the bound [1, 1+Σ k_eff‖v‖²]; (3) valence distribution; (4) E2-style force
//! convergence of S1/S3 vs S0 on a clamped ~100-cell block.

use experiments::{meta, write_json, Csv};
use geometry::rng::Rng;
use geometry::rsa::rsa_box;
use geometry::voronoi3d::{planar_props, voronoi_box, FaceTag, VCell3};
use geometry::V3;
use serde_json::json;
use faer::linalg::solvers::DenseSolveCore;
use solvers::sparse::{csc_from_triplets, Spd};

const EXP: &str = "E10";
const E: f64 = 30e9;
const RHO_D: f64 = 2400.0;

#[derive(Clone)]
struct Cell {
    c: V3,
    m: f64,
    inertia: [[f64; 3]; 3],
    fixed: bool,
}

/// One scalar spring component: C = j·[u_a ω_a u_b ω_b].
#[derive(Clone)]
struct Comp {
    a: usize,
    b: Option<usize>, // None = ground
    ja: [f64; 6],
    jb: [f64; 6],
    k: f64,
    facet: usize,
}

struct Model3 {
    cells: Vec<Cell>,
    comps: Vec<Comp>,
    loads: Vec<[f64; 6]>,
    nfacets: usize,
    valence: Vec<usize>,
}

fn cross(a: V3, b: V3) -> V3 {
    a.cross(b)
}

fn facet_comps(a: usize, b: Option<usize>, ca: V3, cb: Option<V3>, verts: &[V3], n: V3, h: f64, gamma: f64, fi: usize, out: &mut Vec<Comp>) {
    let (area, pc, e1, e2, i2) = planar_props(verts, n);
    // principal axes of the in-plane second moments
    let (iuu, ivv, iuv) = (i2[0], i2[1], i2[2]);
    let th = 0.5 * (2.0 * iuv).atan2(iuu - ivv);
    let (s, c) = th.sin_cos();
    let p1 = e1 * c + e2 * s;
    let p2 = e2 * c - e1 * s;
    let i_p1 = iuu * c * c + 2.0 * iuv * s * c + ivv * s * s; // ∫ (coord along p1)²
    let i_p2 = iuu * s * s - 2.0 * iuv * s * c + ivv * c * c; // ∫ (coord along p2)²
    let kn = E * area / h;
    let kt = gamma * kn;
    let ra = pc - ca;
    let rb = cb.map(|x| pc - x).unwrap_or(V3::ZERO);
    let tr = |d: V3, k: f64, out: &mut Vec<Comp>| {
        let rad = cross(ra, d);
        let rbd = cross(rb, d);
        out.push(Comp { a, b, ja: [-d.x, -d.y, -d.z, -rad.x, -rad.y, -rad.z], jb: [d.x, d.y, d.z, rbd.x, rbd.y, rbd.z], k, facet: fi });
    };
    tr(n, kn, out);
    tr(p1, kt, out);
    tr(p2, kt, out);
    let rot = |e: V3, k: f64, out: &mut Vec<Comp>| {
        out.push(Comp { a, b, ja: [0.0, 0.0, 0.0, -e.x, -e.y, -e.z], jb: [0.0, 0.0, 0.0, e.x, e.y, e.z], k, facet: fi });
    };
    // rotation about p1 makes the normal jump vary along p2: stiffness k_n ∫(p2)²/A
    rot(p1, kn * i_p2 / area, out);
    rot(p2, kn * i_p1 / area, out);
    rot(n, kt * (i_p1 + i_p2) / area, out);
}

fn build(cells3: &[VCell3], clamp_face: Option<usize>, gamma: f64) -> Model3 {
    let mut cells: Vec<Cell> = cells3
        .iter()
        .map(|c| Cell { c: c.props.centroid, m: RHO_D * c.props.volume, inertia: c.props.inertia.map(|r| r.map(|x| x * RHO_D)), fixed: false })
        .collect();
    let _ = &mut cells;
    let mut comps = Vec::new();
    let mut nf = 0;
    let mut valence = vec![0; cells3.len()];
    for (i, c) in cells3.iter().enumerate() {
        for f in &c.poly.faces {
            match f.tag {
                FaceTag::Cell(j) => {
                    valence[i] += 1;
                    if i < j {
                        let gi = c.seed;
                        let gj = cells3[j].seed;
                        let n = (gj - gi).normalized();
                        facet_comps(i, Some(j), cells[i].c, Some(cells[j].c), &f.verts, n, (gj - gi).norm(), gamma, nf, &mut comps);
                        nf += 1;
                    }
                }
                FaceTag::Boundary(k) if Some(k) == clamp_face => {
                    let h = (f.verts[0] - c.seed).dot(f.normal).abs();
                    facet_comps(i, None, cells[i].c, None, &f.verts, f.normal, h, gamma, nf, &mut comps);
                    nf += 1;
                }
                _ => {}
            }
        }
    }
    let n = cells.len();
    Model3 { cells, comps, loads: vec![[0.0; 6]; n], nfacets: nf, valence }
}

fn top_traction(m: &mut Model3, cells3: &[VCell3], face: usize, t: V3) -> V3 {
    let mut tot = V3::ZERO;
    for (i, c) in cells3.iter().enumerate() {
        for f in &c.poly.faces {
            if f.tag == FaceTag::Boundary(face) {
                let (area, pc, _, _, _) = planar_props(&f.verts, f.normal);
                let fv = t * area;
                let mo = cross(pc - m.cells[i].c, fv);
                for k in 0..3 {
                    m.loads[i][k] += fv[k];
                    m.loads[i][3 + k] += mo[k];
                }
                tot += fv;
            }
        }
    }
    tot
}


fn comp_value(c: &Comp, u: &[f64]) -> f64 {
    let mut v = 0.0;
    for k in 0..6 {
        v += c.ja[k] * u[6 * c.a + k];
        if let Some(b) = c.b {
            v += c.jb[k] * u[6 * b + k];
        }
    }
    v
}

fn s0(m: &Model3) -> (Vec<f64>, usize) {
    let n = 6 * m.cells.len();
    let mut t = Vec::new();
    for c in &m.comps {
        let mut ent: Vec<(usize, f64)> = (0..6).map(|k| (6 * c.a + k, c.ja[k])).collect();
        if let Some(b) = c.b {
            ent.extend((0..6).map(|k| (6 * b + k, c.jb[k])));
        }
        for &(r, vr) in &ent {
            for &(cc, vc) in &ent {
                if vr * vc != 0.0 {
                    t.push((r, cc, c.k * vr * vc));
                }
            }
        }
    }
    let k = csc_from_triplets(n, t);
    let mut spd = Spd::analyze(&k);
    spd.factor(&k).expect("3D K SPD");
    let mut u: Vec<f64> = m.loads.iter().flatten().copied().collect();
    spd.solve(&mut u);
    (u, spd.stats.nnz_l)
}

fn chol6(a: &mut [[f64; 6]; 6]) -> bool {
    for j in 0..6 {
        let mut s = a[j][j];
        for k in 0..j {
            s -= a[j][k] * a[j][k];
        }
        if s <= 0.0 {
            return false;
        }
        a[j][j] = s.sqrt();
        for i in j + 1..6 {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            a[i][j] = s / a[j][j];
        }
    }
    true
}

fn solve6(mut a: [[f64; 6]; 6], b: [f64; 6]) -> Option<[f64; 6]> {
    if !chol6(&mut a) {
        return None;
    }
    let mut y = b;
    for i in 0..6 {
        for k in 0..i {
            y[i] -= a[i][k] * y[k];
        }
        y[i] /= a[i][i];
    }
    for i in (0..6).rev() {
        for k in i + 1..6 {
            y[i] -= a[k][i] * y[k];
        }
        y[i] /= a[i][i];
    }
    Some(y)
}

fn sym_eig_range(a: [[f64; 6]; 6]) -> (f64, f64) {
    let mat = faer::Mat::<f64>::from_fn(6, 6, |i, j| a[i][j]);
    let ev = mat.self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
    (ev.iter().cloned().fold(f64::INFINITY, f64::min), ev.iter().cloned().fold(0.0, f64::max))
}

/// D^{-1/2} for the 6×6 metric blockdiag(m/h² I, I_tensor/h²).
fn metric_inv_sqrt(cell: &Cell, h: f64) -> [[f64; 6]; 6] {
    let mut r = [[0.0; 6]; 6];
    let sm = (h * h / cell.m).sqrt();
    for k in 0..3 {
        r[k][k] = sm;
    }
    let it = faer::Mat::<f64>::from_fn(3, 3, |i, j| cell.inertia[i][j]);
    let evd = it.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let u = evd.U();
    let s = evd.S();
    for i in 0..3 {
        for j in 0..3 {
            let mut v = 0.0;
            for k in 0..3 {
                v += u[(i, k)] * (h * h / s[k]).sqrt() * u[(j, k)];
            }
            r[3 + i][3 + j] = v;
        }
    }
    r
}

enum Pol {
    Shared(f64),
    PerSpring(f64),
    Dynamic(f64, f64),
    Penalty,
}

fn s1_run(m: &Model3, pol: &Pol, inertia_h: Option<f64>, fstar: &[f64], max_it: usize) -> (Option<usize>, f64, f64) {
    let nc = m.cells.len();
    let kbar = m.comps.iter().filter(|c| c.facet == c.facet).map(|c| c.k).sum::<f64>() / m.comps.len() as f64;
    // effective mass of each component (for the dynamic policy)
    let meff = |c: &Comp| -> f64 {
        let mut inv = 0.0;
        for (bi, j) in [(Some(c.a), c.ja), (c.b, c.jb)] {
            if let Some(b) = bi {
                let cell = &m.cells[b];
                if cell.fixed {
                    continue;
                }
                for k in 0..3 {
                    inv += j[k] * j[k] / cell.m;
                }
                // rotational part via inverse inertia
                let it = faer::Mat::<f64>::from_fn(3, 3, |i, jj| cell.inertia[i][jj]);
                let inv_it = it.full_piv_lu().inverse();
                for p in 0..3 {
                    for q in 0..3 {
                        inv += j[3 + p] * inv_it[(p, q)] * j[3 + q];
                    }
                }
            }
        }
        1.0 / inv
    };
    let rho: Vec<f64> = m
        .comps
        .iter()
        .map(|c| match pol {
            Pol::Shared(x) => x * kbar,
            Pol::PerSpring(x) => x * c.k,
            Pol::Dynamic(x, h) => x * meff(c) / (h * h),
            Pol::Penalty => f64::INFINITY,
        })
        .collect();
    let coef = |i: usize| -> (f64, f64) {
        match pol {
            Pol::Penalty => (0.0, m.comps[i].k),
            _ => {
                let s = m.comps[i].k / (m.comps[i].k + rho[i]);
                (s, rho[i] * s)
            }
        }
    };
    // colouring
    let mut adj = vec![Vec::new(); nc];
    let mut by_cell = vec![Vec::new(); nc];
    for (ci, c) in m.comps.iter().enumerate() {
        by_cell[c.a].push(ci);
        if let Some(b) = c.b {
            by_cell[b].push(ci);
            adj[c.a].push(b);
            adj[b].push(c.a);
        }
    }
    let mut color = vec![usize::MAX; nc];
    let mut ncol = 0;
    for i in 0..nc {
        let mut c = 0;
        while adj[i].iter().any(|&o| color[o] == c) {
            c += 1;
        }
        color[i] = c;
        ncol = ncol.max(c + 1);
    }
    let mut u = vec![0.0; 6 * nc];
    let mut lam = vec![0.0; m.comps.len()];
    let fnorm = fstar.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut hit = None;
    let (mut e8, mut e16) = (f64::NAN, f64::NAN);
    for it in 1..=max_it {
        for col in 0..ncol {
            for i in (0..nc).filter(|&i| color[i] == col) {
                let mut g = [0.0; 6];
                let mut hmat = [[0.0; 6]; 6];
                for k in 0..6 {
                    g[k] = -m.loads[i][k];
                }
                if let Some(h) = inertia_h {
                    let cell = &m.cells[i];
                    for k in 0..3 {
                        hmat[k][k] += cell.m / (h * h);
                        g[k] += cell.m / (h * h) * u[6 * i + k];
                    }
                    for p in 0..3 {
                        for q in 0..3 {
                            hmat[3 + p][3 + q] += cell.inertia[p][q] / (h * h);
                            g[3 + p] += cell.inertia[p][q] / (h * h) * u[6 * i + 3 + q];
                        }
                    }
                }
                for &ci in &by_cell[i] {
                    let c = &m.comps[ci];
                    let j = if c.a == i { c.ja } else { c.jb };
                    let cv = comp_value(c, &u);
                    let (s, ke) = coef(ci);
                    let f = s * lam[ci] + ke * cv;
                    for p in 0..6 {
                        g[p] += j[p] * f;
                        for q in 0..6 {
                            hmat[p][q] += ke * j[p] * j[q];
                        }
                    }
                }
                if let Some(dx) = solve6(hmat, g) {
                    for k in 0..6 {
                        u[6 * i + k] -= dx[k];
                    }
                }
            }
        }
        for (ci, c) in m.comps.iter().enumerate() {
            let cv = comp_value(c, &u);
            lam[ci] = match pol {
                Pol::Penalty => c.k * cv,
                _ => {
                    let (s, _) = coef(ci);
                    s * (lam[ci] + rho[ci] * cv)
                }
            };
        }
        let err = (lam.iter().zip(fstar).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()).sqrt() / fnorm;
        if it == 8 {
            e8 = err;
        }
        if it == 16 {
            e16 = err;
        }
        if err <= 0.01 {
            hit = Some(it);
            break;
        }
    }
    (hit, e8, e16)
}

fn main() {
    let mut out = serde_json::Map::new();
    // (1) two cells: rank of the free pair stiffness = 6, torsion check
    {
        let seeds = [V3::new(0.25, 0.5, 0.5), V3::new(0.75, 0.5, 0.5)];
        let cells3 = voronoi_box(&seeds, V3::ZERO, V3::new(1.0, 1.0, 1.0));
        let m = build(&cells3, None, 1.0);
        let n = 12;
        let mut k = vec![vec![0.0; n]; n];
        for c in &m.comps {
            let mut j = [0.0; 12];
            j[..6].copy_from_slice(&c.ja);
            j[6..].copy_from_slice(&c.jb);
            for p in 0..12 {
                for q in 0..12 {
                    k[p][q] += c.k * j[p] * j[q];
                }
            }
        }
        let mat = faer::Mat::<f64>::from_fn(n, n, |i, j| k[i][j]);
        let ev = mat.self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
        let emax = ev.iter().cloned().fold(0.0, f64::max);
        let zero_modes = ev.iter().filter(|&&x| x.abs() < 1e-10 * emax).count();
        // facet: unit square at x=0.5, h = 0.5 → k_n = E·1/0.5; torsion k_t I_p/A = k_n (1/6)
        let kn = E * 1.0 / 0.5;
        let tw = m.comps.iter().find(|c| (c.ja[3] + 1.0).abs() < 1e-12 && c.ja[4].abs() < 1e-12).map(|c| c.k).unwrap_or(f64::NAN);
        out.insert("two_cells".into(), json!({"zero_modes_free_pair": zero_modes, "expected": 6, "twist_stiffness_over_kn": tw / kn, "expected_twist": 1.0 / 6.0,
            "eigs_rel": ev.iter().map(|x| x / emax).collect::<Vec<_>>()}));
    }
    // (2)-(4) ~100-cell block, clamped bottom (face 4), top traction
    let mut csv = Csv::new(EXP, "blocks", &["cell", "valence_cells", "boundary", "eig_min", "eig_max", "bound"]);
    let mut blocks = Vec::new();
    for &(dmin, seed) in &[(0.2, 1u64), (0.2, 2), (0.15, 3)] {
        let mut rng = Rng::new(seed);
        let seeds = rsa_box(V3::ZERO, V3::new(1.0, 1.0, 1.0), dmin, &mut rng, 5000);
        let cells3 = voronoi_box(&seeds, V3::ZERO, V3::new(1.0, 1.0, 1.0));
        let vol: f64 = cells3.iter().map(|c| c.props.volume).sum();
        let mut m = build(&cells3, Some(4), 1.0);
        top_traction(&mut m, &cells3, 5, V3::new(2.5e3, 1.0e3, -1.0e4));
        let (u, nnz) = s0(&m);
        let fstar: Vec<f64> = m.comps.iter().map(|c| c.k * comp_value(c, &u)).collect();
        // valence
        let boundary: Vec<bool> = cells3.iter().map(|c| c.poly.faces.iter().any(|f| matches!(f.tag, FaceTag::Boundary(_)))).collect();
        let interior_val: Vec<usize> = (0..cells3.len()).filter(|&i| !boundary[i]).map(|i| m.valence[i]).collect();
        let all_val = m.valence.clone();
        let mut hist = std::collections::BTreeMap::new();
        for &v in &all_val {
            *hist.entry(v).or_insert(0) += 1;
        }
        // (2) normalised blocks with the dynamic policy at game substep, and the penalty limit
        let h = 1.0 / 480.0;
        let mut block_stats = Vec::new();
        for (pname, keff_of) in [("dynamic rho=m_eff/h^2", 0usize), ("penalty k_eff=k", 1usize)] {
            let mut worst_ratio: f64 = 0.0;
            let (mut emin_all, mut emax_all, mut bmax_all) = (f64::INFINITY, 0.0f64, 0.0f64);
            let mut tight = Vec::new();
            let dinvs: Vec<[[f64; 6]; 6]> = m.cells.iter().map(|c| metric_inv_sqrt(c, h)).collect();
            let whiten = |d: &[[f64; 6]; 6], j: &[f64; 6]| -> [f64; 6] {
                let mut v = [0.0; 6];
                for p in 0..6 {
                    for q in 0..6 {
                        v[p] += d[p][q] * j[q];
                    }
                }
                v
            };
            for i in 0..m.cells.len() {
                let mut hh = [[0.0; 6]; 6];
                for k in 0..6 {
                    hh[k][k] = 1.0;
                }
                let mut bound = 1.0;
                for c in m.comps.iter().filter(|c| c.a == i || c.b == Some(i)) {
                    let (j, other) = if c.a == i { (c.ja, c.b.map(|b| (b, c.jb))) } else { (c.jb, Some((c.a, c.ja))) };
                    let v = whiten(&dinvs[i], &j);
                    let vv: f64 = v.iter().map(|x| x * x).sum();
                    let keff = if keff_of == 1 {
                        c.k
                    } else {
                        // ρ = m_eff/h², 1/m_eff from both sides (h² cancels in whitened norms)
                        let vo: f64 = other.map(|(b, jo)| whiten(&dinvs[b], &jo).iter().map(|x| x * x).sum()).unwrap_or(0.0);
                        let rho = 1.0 / (vv + vo);
                        rho * c.k / (c.k + rho)
                    };
                    for p in 0..6 {
                        for q in 0..6 {
                            hh[p][q] += keff * v[p] * v[q];
                        }
                    }
                    bound += keff * vv;
                }
                let (emin, emax) = sym_eig_range(hh);
                emin_all = emin_all.min(emin);
                emax_all = emax_all.max(emax);
                bmax_all = bmax_all.max(bound);
                worst_ratio = worst_ratio.max(emax / bound);
                tight.push(emax / bound);
                if keff_of == 0 && dmin == 0.2 && seed == 1 {
                    csv.row(&[i as f64, m.valence[i] as f64, boundary[i] as u8 as f64, emin, emax, bound]);
                }
            }
            let mean_tight = tight.iter().sum::<f64>() / tight.len() as f64;
            block_stats.push(json!({"policy": pname, "eig_min": emin_all, "eig_max": emax_all, "bound_max": bmax_all,
                "max_eig_over_bound": worst_ratio, "mean_eig_over_bound": mean_tight, "cond_max": emax_all / emin_all}));
        }
        // (4) force convergence, quasi-static cold start
        let mut conv = Vec::new();
        for (name, pol) in [("S1 perspring c=0.1", Pol::PerSpring(0.1)), ("S1 shared c=0.1", Pol::Shared(0.1)), ("S3 penalty", Pol::Penalty)] {
            let (hit, e8, e16) = s1_run(&m, &pol, None, &fstar, 40_000);
            conv.push(json!({"method": name, "iters_to_1pct": hit, "err8": e8, "err16": e16}));
            eprintln!("3D dmin {dmin} seed {seed}: {name} -> {hit:?}");
        }
        // dynamic first step at h = 1/480 (block 1 m, cells ~0.2 m)
        for (name, pol) in [("S1 dynamic c=1 (h=1/480)", Pol::Dynamic(1.0, h)), ("S1 perspring c=0.1 (h=1/480)", Pol::PerSpring(0.1)), ("S3 penalty (h=1/480)", Pol::Penalty)] {
            // reference dynamic step from rest: (M/h² + K) u = f
            let nn = 6 * m.cells.len();
            let mut t = Vec::new();
            for c in &m.comps {
                let mut ent: Vec<(usize, f64)> = (0..6).map(|k| (6 * c.a + k, c.ja[k])).collect();
                if let Some(b) = c.b {
                    ent.extend((0..6).map(|k| (6 * b + k, c.jb[k])));
                }
                for &(r, vr) in &ent {
                    for &(cc, vc) in &ent {
                        if vr * vc != 0.0 {
                            t.push((r, cc, c.k * vr * vc));
                        }
                    }
                }
            }
            for (i, cell) in m.cells.iter().enumerate() {
                for k in 0..3 {
                    t.push((6 * i + k, 6 * i + k, cell.m / (h * h)));
                }
                for p in 0..3 {
                    for q in 0..3 {
                        t.push((6 * i + 3 + p, 6 * i + 3 + q, cell.inertia[p][q] / (h * h)));
                    }
                }
            }
            let kk = csc_from_triplets(nn, t);
            let mut spd = Spd::analyze(&kk);
            spd.factor(&kk).unwrap();
            let mut ud: Vec<f64> = m.loads.iter().flatten().copied().collect();
            spd.solve(&mut ud);
            let fd: Vec<f64> = m.comps.iter().map(|c| c.k * comp_value(c, &ud)).collect();
            let (hit, e8, e16) = s1_run(&m, &pol, Some(h), &fd, 40_000);
            conv.push(json!({"method": name, "iters_to_1pct": hit, "err8": e8, "err16": e16}));
            eprintln!("3D dyn dmin {dmin} seed {seed}: {name} -> {hit:?}");
        }
        let ell = m.comps.iter().map(|_| 0.0).sum::<f64>();
        let _ = ell;
        blocks.push(json!({"dmin": dmin, "seed": seed, "cells": cells3.len(), "volume_check": vol, "facets": m.nfacets, "components": m.comps.len(), "nnz_L": nnz,
            "valence_interior": {"mean": interior_val.iter().sum::<usize>() as f64 / interior_val.len().max(1) as f64, "max": interior_val.iter().max(), "min": interior_val.iter().min(), "n": interior_val.len()},
            "valence_all_hist": hist, "valence_all_max": all_val.iter().max(),
            "normalised_blocks": block_stats, "convergence": conv}));
    }
    out.insert("blocks".into(), json!(blocks));
    let res = json!({
        "meta": meta(EXP, "all", None, "linearised 3D RBSN; S0 sparse Cholesky; S1 6x6 block GS (colour order) + simultaneous dual; error = L2 of all spring forces",
            json!({"E": E, "density": RHO_D, "gamma": 1.0, "box": "1x1x1 m, clamped bottom face, top traction (2.5, 1.0, -10) kPa"})),
        "results": out,
    });
    write_json(EXP, "summary", &res);
    println!("{}", serde_json::to_string_pretty(&res["results"]).unwrap());
}
