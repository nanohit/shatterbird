//! Patch partition and the multiscale patch basis.
//!
//! A patch moves as x_i = c + R(φ)(P_i + U_i η), θ_i = φ + W_i η. The columns of
//! [U; W] are built from polynomial traces of total degree ≤ P on the patch's
//! boundary cells (each of u_x, u_y, θ), extended into the interior by static
//! condensation T_I = −K_II⁻¹ K_IB T_B, plus two body-force modes; rigid modes are
//! removed and the rest made M-orthonormal (mean-axis frame).

use super::Work;
use geometry::V2;
use model::Model;
use solvers::sparse::{csc_from_triplets, Spd};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Patch {
    pub cells: Vec<usize>,
    /// reference offsets from the patch centroid
    pub p: Vec<V2>,
    pub mi: Vec<f64>,
    pub ii: Vec<f64>,
    pub mass: f64,
    pub it: f64,
    pub xc: V2,
    pub me: usize,
    /// (3 n) × me, row-major; T_eᵀ M T_e = I, M-orthogonal to rigid modes
    pub te: Vec<f64>,
    pub khat: Vec<f64>,
    /// Cholesky factor of K̂ (lower, row-major) for K-orthogonal fits
    pub khat_l: Vec<f64>,
    pub s1: Vec<f64>,
    pub w: Vec<f64>,
    /// undamaged material facets inside the patch (carried by K̂)
    pub internal: Vec<usize>,
    pub radius: f64,
}

impl Patch {
    pub fn ny(&self) -> usize {
        3 + self.me
    }
    /// Body-frame offset P_i + U_i η and θ-part W_i η of local cell li.
    pub fn local(&self, li: usize, eta: &[f64]) -> (V2, f64) {
        let me = self.me;
        let (mut ux, mut uy, mut uw) = (0.0, 0.0, 0.0);
        let r = 3 * li * me;
        for j in 0..me {
            ux += self.te[r + j] * eta[j];
            uy += self.te[r + me + j] * eta[j];
            uw += self.te[r + 2 * me + j] * eta[j];
        }
        (self.p[li] + V2::new(ux, uy), uw)
    }
}

fn find(p: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while p[r] != r {
        p[r] = p[p[r]];
        r = p[r];
    }
    r
}

/// Square bins of side `lp`, split into components connected through undamaged
/// material facets; components under `min_cells` merged into the neighbour patch
/// sharing most facets. Deterministic (ordered by smallest cell index).
pub fn partition(m: &Model, adj: &[Vec<usize>], lp: f64, min_cells: usize) -> Vec<Vec<usize>> {
    let nc = m.n_cells();
    let (mut x0, mut y0) = (f64::INFINITY, f64::INFINITY);
    for b in &m.bodies[..nc] {
        x0 = x0.min(b.x0.x);
        y0 = y0.min(b.x0.y);
    }
    let bin: Vec<(i64, i64)> = (0..nc).map(|i| (((m.bodies[i].x0.x - x0) / lp).floor() as i64, ((m.bodies[i].x0.y - y0) / lp).floor() as i64)).collect();
    let mut parent: Vec<usize> = (0..nc).collect();
    for f in m.facets.iter() {
        if !f.active || f.tag != 0 || f.a >= nc || f.b >= nc || bin[f.a] != bin[f.b] {
            continue;
        }
        if f.pts.iter().any(|p| p.law.damage(p.kappa_c) > 0.0) {
            continue;
        }
        let (a, b) = (find(&mut parent, f.a), find(&mut parent, f.b));
        if a != b {
            parent[a.max(b)] = a.min(b);
        }
    }
    let mut lab = vec![0usize; nc];
    for i in 0..nc {
        lab[i] = find(&mut parent, i);
    }
    // merge small components into the neighbour with most shared facets
    loop {
        let mut size: BTreeMap<usize, usize> = BTreeMap::new();
        for &l in &lab {
            *size.entry(l).or_default() += 1;
        }
        let small: Vec<usize> = size.iter().filter(|(_, &s)| s < min_cells).map(|(&l, _)| l).collect();
        if small.is_empty() || size.len() == 1 {
            break;
        }
        let mut changed = false;
        for l in small {
            let mut shared: BTreeMap<usize, usize> = BTreeMap::new();
            for i in 0..nc {
                if lab[i] != l {
                    continue;
                }
                for &fi in &adj[i] {
                    let f = &m.facets[fi];
                    if f.tag != 0 || !f.active {
                        continue;
                    }
                    let o = if f.a == i { f.b } else { f.a };
                    if o < nc && lab[o] != l {
                        *shared.entry(lab[o]).or_default() += 1;
                    }
                }
            }
            if let Some((&to, _)) = shared.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))) {
                for v in lab.iter_mut() {
                    if *v == l {
                        *v = to;
                    }
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..nc {
        groups.entry(lab[i]).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = groups.into_values().collect();
    out.sort_by_key(|g| g[0]);
    out
}

/// Linear stiffness triplets (local DOFs 3 li + k) of the given facets at the
/// reference state.
fn facet_stiffness(m: &Model, facets: &[usize], loc: &[usize]) -> Vec<(usize, usize, f64)> {
    let q0: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
    let mut trip = Vec::new();
    for &fi in facets {
        let f = &m.facets[fi];
        for pt in 0..2 {
            let (j, _) = m.eval_point(&q0, f, pt);
            let (kn, kt) = (f.pts[pt].law.kn, f.pts[pt].law.kt);
            let jn = [j.jna[0], j.jna[1], j.jna[2], j.jnb[0], j.jnb[1], j.jnb[2]];
            let jt = [j.jta[0], j.jta[1], j.jta[2], j.jtb[0], j.jtb[1], j.jtb[2]];
            let ids = [3 * loc[f.a], 3 * loc[f.a] + 1, 3 * loc[f.a] + 2, 3 * loc[f.b], 3 * loc[f.b] + 1, 3 * loc[f.b] + 2];
            for a in 0..6 {
                for c in 0..6 {
                    let v = kn * jn[a] * jn[c] + kt * jt[a] * jt[c];
                    if v != 0.0 {
                        trip.push((ids[a], ids[c], v));
                    }
                }
            }
        }
    }
    trip
}

/// Dense Cholesky (lower, row-major) of an SPD matrix; None if not SPD.
pub fn chol(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

pub fn chol_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// Build the patch basis. `boundary_extra[i]` forces cell i onto the trace set
/// (loaded cells). Facets internal to the patch that are damaged, or not material,
/// are left out of K̂ and their cells put on the trace set.
pub fn build_patch(m: &Model, adj: &[Vec<usize>], cells_in: &[usize], deg: usize, nvib: usize, boundary_extra: &[bool], w: &mut Work) -> Patch {
    let nb = m.bodies.len();
    let mut inp = vec![false; nb];
    for &i in cells_in {
        inp[i] = true;
    }
    let mut internal = Vec::new();
    let mut bnd = vec![false; nb];
    let mut seen = std::collections::BTreeSet::new();
    for &i in cells_in {
        for &fi in &adj[i] {
            if !seen.insert(fi) {
                continue;
            }
            let f = &m.facets[fi];
            if !f.active {
                continue;
            }
            let (ia, ib) = (inp[f.a], inp[f.b]);
            let undamaged = f.tag == 0 && f.pts.iter().all(|p| p.law.damage(p.kappa_c) == 0.0);
            if ia && ib && undamaged {
                internal.push(fi);
            } else {
                if ia {
                    bnd[f.a] = true;
                }
                if ib {
                    bnd[f.b] = true;
                }
            }
        }
    }
    internal.sort_unstable();
    for &i in cells_in {
        if boundary_extra[i] {
            bnd[i] = true;
        }
    }
    let mut cells: Vec<usize> = cells_in.iter().copied().filter(|&i| bnd[i]).collect();
    if cells.is_empty() {
        cells = cells_in.to_vec();
        for &i in cells_in {
            bnd[i] = true;
        }
    }
    let nbc = cells.len();
    cells.extend(cells_in.iter().copied().filter(|&i| !bnd[i]));
    let nr = cells.len();
    let mut loc = vec![usize::MAX; nb];
    for (li, &i) in cells.iter().enumerate() {
        loc[i] = li;
    }
    let mi: Vec<f64> = cells.iter().map(|&i| m.bodies[i].mass).collect();
    let ii: Vec<f64> = cells.iter().map(|&i| m.bodies[i].inertia).collect();
    let mass: f64 = mi.iter().sum();
    let xc = cells.iter().zip(&mi).fold(V2::ZERO, |s, (&i, &mm)| s + m.bodies[i].x0 * mm) * (1.0 / mass);
    let p: Vec<V2> = cells.iter().map(|&i| m.bodies[i].x0 - xc).collect();
    let it: f64 = ii.iter().sum();
    let radius = p.iter().fold(0.0f64, |a, v| a.max(v.norm()));
    let trip = facet_stiffness(m, &internal, &loc);
    let nbd = 3 * nbc;
    let ndof = 3 * nr;
    let ni = ndof - nbd;
    let mdiag: Vec<f64> = (0..ndof).map(|d| if d % 3 == 2 { ii[d / 3] } else { mi[d / 3] }).collect();
    // polynomial traces on the boundary cells
    let sc = if radius > 0.0 { radius } else { 1.0 };
    let monos: Vec<(i32, i32)> = (0..=deg as i32).flat_map(|t| (0..=t).map(move |b| (t - b, b))).collect();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for k in 0..3 {
        for &(a, b) in &monos {
            let mut col = vec![0.0; ndof];
            for li in 0..nbc {
                let (xi, ze) = (p[li].x / sc, p[li].y / sc);
                col[3 * li + k] = xi.powi(a) * ze.powi(b);
            }
            cols.push(col);
        }
    }
    if ni > 0 {
        let mut kii_t = Vec::new();
        let mut kib = Vec::new();
        for &(a, c, v) in &trip {
            if a >= nbd && c >= nbd {
                kii_t.push((a - nbd, c - nbd, v));
            } else if a >= nbd {
                kib.push((a - nbd, c, v));
            }
        }
        let kii = csc_from_triplets(ni, kii_t.clone());
        let mut spd = Spd::analyze(&kii);
        spd.factor(&kii).expect("K_II not SPD");
        w.rebuild_flops += spd.stats.flops_factor;
        let solve_cost = 4.0 * spd.stats.nnz_l as f64;
        // condensation of the traces
        for col in cols.iter_mut() {
            let mut rhs = vec![0.0; ni];
            for &(a, c, v) in &kib {
                rhs[a] -= v * col[c];
            }
            spd.solve(&mut rhs);
            w.rebuild_flops += solve_cost + 2.0 * kib.len() as f64;
            col[nbd..].copy_from_slice(&rhs);
        }
        // body-force modes (boundary fixed)
        for k in 0..2 {
            let mut rhs: Vec<f64> = (0..ni).map(|a| if (nbd + a) % 3 == k { mdiag[nbd + a] } else { 0.0 }).collect();
            spd.solve(&mut rhs);
            w.rebuild_flops += solve_cost;
            let mut col = vec![0.0; ndof];
            col[nbd..].copy_from_slice(&rhs);
            cols.push(col);
        }
        // fixed-interface vibration modes (optional) by subspace iteration
        if nvib > 0 && ni > 3 * nvib {
            let pb = (nvib + 4).min(ni);
            let mut x: Vec<Vec<f64>> = (0..pb).map(|k| (0..ni).map(|a| ((a as f64 + 1.0) * (k as f64 + 1.0) * 0.7123).sin()).collect()).collect();
            for _ in 0..30 {
                for v in x.iter_mut() {
                    let mut y: Vec<f64> = v.iter().enumerate().map(|(a, &s)| s * mdiag[nbd + a]).collect();
                    spd.solve(&mut y);
                    *v = y;
                }
                for k in 0..pb {
                    for l in 0..k {
                        let d: f64 = (0..ni).map(|a| x[k][a] * mdiag[nbd + a] * x[l][a]).sum();
                        let xl = x[l].clone();
                        for a in 0..ni {
                            x[k][a] -= d * xl[a];
                        }
                    }
                    let nn: f64 = (0..ni).map(|a| x[k][a] * x[k][a] * mdiag[nbd + a]).sum::<f64>().sqrt();
                    for a in 0..ni {
                        x[k][a] /= nn;
                    }
                }
                w.rebuild_flops += (pb as f64) * (solve_cost + (pb * ni * 4) as f64);
            }
            for v in x.into_iter().take(nvib) {
                let mut col = vec![0.0; ndof];
                col[nbd..].copy_from_slice(&v);
                cols.push(col);
            }
        }
    }
    // M-orthonormalise against the rigid modes, then among themselves
    let rig: Vec<Vec<f64>> = (0..3)
        .map(|k| {
            (0..ndof)
                .map(|d| {
                    let (li, kk) = (d / 3, d % 3);
                    match k {
                        0 => (kk == 0) as u8 as f64,
                        1 => (kk == 1) as u8 as f64,
                        _ => [-p[li].y, p[li].x, 1.0][kk],
                    }
                })
                .collect()
        })
        .collect();
    let mdot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).zip(&mdiag).map(|((x, y), mm)| x * y * mm).sum() };
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for (ci, c) in rig.into_iter().chain(cols).enumerate() {
        let n0 = mdot(&c, &c).sqrt();
        if n0 == 0.0 {
            continue;
        }
        let mut v = c;
        for _ in 0..2 {
            for u in &basis {
                let d = mdot(&v, u);
                for (a, b) in v.iter_mut().zip(u) {
                    *a -= d * b;
                }
            }
        }
        let n1 = mdot(&v, &v).sqrt();
        if n1 > 1e-7 * n0 || ci < 3 {
            basis.push(v.iter().map(|a| a / n1).collect());
        }
    }
    w.rebuild_flops += (basis.len() * basis.len() * ndof * 4) as f64;
    let te_cols: Vec<Vec<f64>> = basis.into_iter().skip(3).collect();
    let me = te_cols.len();
    let mut te = vec![0.0; ndof * me];
    for (j, col) in te_cols.iter().enumerate() {
        for d in 0..ndof {
            te[d * me + j] = col[d];
        }
    }
    let mut kt_ = vec![0.0; ndof * me];
    for &(a, c, v) in &trip {
        for j in 0..me {
            kt_[a * me + j] += v * te[c * me + j];
        }
    }
    let mut khat = vec![0.0; me * me];
    for a in 0..ndof {
        for i in 0..me {
            let ta = te[a * me + i];
            if ta != 0.0 {
                for j in 0..me {
                    khat[i * me + j] += ta * kt_[a * me + j];
                }
            }
        }
    }
    for i in 0..me {
        for j in 0..i {
            let s = 0.5 * (khat[i * me + j] + khat[j * me + i]);
            khat[i * me + j] = s;
            khat[j * me + i] = s;
        }
    }
    w.rebuild_flops += 2.0 * (trip.len() * me) as f64 + 2.0 * (ndof * me * me) as f64;
    // K̂ is SPD on the deformation subspace unless the patch is internally disconnected
    // (then a few modes carry no stiffness); regularise those for the K-fit only
    let khat_l = chol(&khat, me).unwrap_or_else(|| {
        let mut kr = khat.clone();
        let dmax = (0..me).map(|i| khat[i * me + i]).fold(0.0f64, f64::max).max(1.0);
        for i in 0..me {
            kr[i * me + i] += 1e-10 * dmax;
        }
        chol(&kr, me).expect("regularised K̂")
    });
    let s1: Vec<f64> = (0..me).map(|j| (0..nr).map(|li| mi[li] * (p[li].x * te[(3 * li) * me + j] + p[li].y * te[(3 * li + 1) * me + j])).sum()).collect();
    let wv: Vec<f64> = (0..me).map(|j| (0..nr).map(|li| ii[li] * te[(3 * li + 2) * me + j]).sum()).collect();
    Patch { cells, p, mi, ii, mass, it, xc, me, te, khat, khat_l, s1, w: wv, internal, radius }
}

/// K d over the patch's internal facets (reference linear stiffness), local DOFs.
pub fn k_apply(m: &Model, pa: &Patch, d: &[f64]) -> Vec<f64> {
    let nb = m.bodies.len();
    let mut loc = vec![usize::MAX; nb];
    for (li, &i) in pa.cells.iter().enumerate() {
        loc[i] = li;
    }
    let mut out = vec![0.0; d.len()];
    for (a, c, v) in facet_stiffness(m, &pa.internal, &loc) {
        out[a] += v * d[c];
    }
    out
}
