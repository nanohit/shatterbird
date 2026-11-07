//! Spectral diagnostics of K φ = λ M φ for linearised RBSN models (Phase 0b).
//!
//! Works on the symmetric form B = M^{-1/2} K M^{-1/2} = Σ_c k_c v_c v_cᵀ with
//! v_c = M^{-1/2} j_c (M block-diagonal per cell, K a sum of rank-one spring
//! components). Provides Lanczos with full re-orthogonalisation (top end, and the low
//! end by shift-invert), a dense reference, lower bounds (Ritz, per-cell blocks) and
//! upper bounds on λ_max: Gershgorin, Collatz–Wielandt on |B|, the per-cell splitting
//! certificate, and a Sylvester-inertia certificate (sparse Cholesky of σM − K).

use crate::sparse::{csc_from_triplets, Spd};
use faer::{Mat, Side};

use geometry::rng::Rng;
use model::rbsn3::Model3;
use model::{BodyKind, Model};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompClass {
    Interior,
    Support,
    Contact,
}

impl CompClass {
    pub fn name(self) -> &'static str {
        match self {
            CompClass::Interior => "interior",
            CompClass::Support => "support",
            CompClass::Contact => "contact",
        }
    }
}

/// One rigid body's free DOFs and its mass sub-matrix (row-major).
#[derive(Clone, Debug)]
pub struct Block {
    /// Model body / cell index.
    pub body: usize,
    pub dofs: Vec<usize>,
    pub rot: Vec<bool>,
    pub mass: Vec<f64>,
}

/// Rank-one spring component: energy ½ k (j·u)².
#[derive(Clone, Debug)]
pub struct Comp {
    pub ent: Vec<(usize, f64)>,
    pub k: f64,
    pub class: CompClass,
    pub facet: usize,
}

pub struct Spectral {
    pub n: usize,
    pub blocks: Vec<Block>,
    pub dof_block: Vec<usize>,
    pub comps: Vec<Comp>,
    /// Whitened components grouped per block: (block, local vector of length dofs.len()).
    pub wsub: Vec<Vec<(usize, Vec<f64>)>>,
    msqrt: Vec<Vec<f64>>,
    // B in CSR (symmetric)
    ptr: Vec<usize>,
    idx: Vec<usize>,
    val: Vec<f64>,
}

/// Cyclic Jacobi for a small symmetric matrix (row-major): eigenvalues and
/// eigenvectors (columns of the returned row-major matrix). High relative accuracy.
pub fn jacobi_eig(a: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a: Vec<f64> = (0..k * k).map(|p| 0.5 * (a[p] + a[(p % k) * k + p / k])).collect();
    let mut v = vec![0.0; k * k];
    for i in 0..k {
        v[i * k + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..k).flat_map(|i| (0..k).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i * k + j].powi(2)).sum();
        let diag: f64 = (0..k).map(|i| a[i * k + i].powi(2)).sum();
        if off <= 1e-34 * diag || off == 0.0 {
            break;
        }
        for p in 0..k {
            for q in p + 1..k {
                let apq = a[p * k + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * k + q] - a[p * k + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for r in 0..k {
                    let (arp, arq) = (a[r * k + p], a[r * k + q]);
                    a[r * k + p] = c * arp - s * arq;
                    a[r * k + q] = s * arp + c * arq;
                }
                for r in 0..k {
                    let (apr, aqr) = (a[p * k + r], a[q * k + r]);
                    a[p * k + r] = c * apr - s * aqr;
                    a[q * k + r] = s * apr + c * aqr;
                }
                for r in 0..k {
                    let (vrp, vrq) = (v[r * k + p], v[r * k + q]);
                    v[r * k + p] = c * vrp - s * vrq;
                    v[r * k + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    ((0..k).map(|i| a[i * k + i]).collect(), v)
}

/// Largest eigenvalue of a small symmetric matrix.
pub fn sym_max_eig(a: &[f64], k: usize) -> f64 {
    jacobi_eig(a, k).0.into_iter().fold(f64::NEG_INFINITY, f64::max)
}

fn mat_fn(a: &[f64], k: usize, f: impl Fn(f64) -> f64) -> Vec<f64> {
    let (s, u) = jacobi_eig(a, k);
    let mut r = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            r[i * k + j] = (0..k).map(|l| u[i * k + l] * f(s[l]) * u[j * k + l]).sum();
        }
    }
    r
}

/// Implicit QL with Wilkinson shifts for a symmetric tridiagonal matrix (diagonal
/// `d`, off-diagonal `e[i] = T[i][i+1]`). Returns the eigenvalues; each tracked row
/// r of the identity becomes row r of the eigenvector matrix (column j = vector j).
/// (faer's dense symmetric eigensolver returned Ritz vectors with residuals up to
/// 4e-6 on Lanczos tridiagonals, so T is solved here.)
pub fn tridiag_eig(d: &[f64], e: &[f64], rows: &mut [Vec<f64>]) -> Vec<f64> {
    let n = d.len();
    let mut d = d.to_vec();
    let mut e: Vec<f64> = (0..n).map(|i| if i + 1 < n { e[i] } else { 0.0 }).collect();
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            assert!(iter < 100, "tridiag_eig: no convergence");
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut early = false;
            let mut i = m;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                for z in rows.iter_mut() {
                    let f = z[i + 1];
                    z[i + 1] = s * z[i] + c * f;
                    z[i] = c * z[i] - s * f;
                }
            }
            if early {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    d
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// Σ_c k_c e_c e_cᵀ as triplets, summed per run of consecutive components of the same
/// facet and class (≤ 12 DOFs each), which keeps the triplet count per facet at
/// (DOFs)² instead of (components × DOFs²).
pub fn grouped_triplets(comps: &[Comp], ents: &[Vec<(usize, f64)>]) -> Vec<(usize, usize, f64)> {
    let mut t = Vec::new();
    let mut i = 0;
    while i < comps.len() {
        let mut j = i + 1;
        while j < comps.len() && comps[j].facet == comps[i].facet && comps[j].class == comps[i].class {
            j += 1;
        }
        let mut dofs: Vec<usize> = Vec::new();
        for e in &ents[i..j] {
            for &(d, _) in e {
                if !dofs.contains(&d) {
                    dofs.push(d);
                }
            }
        }
        let k = dofs.len();
        let mut loc = vec![0.0; k * k];
        for (c, e) in comps[i..j].iter().zip(&ents[i..j]) {
            let pos: Vec<(usize, f64)> = e.iter().map(|&(d, v)| (dofs.iter().position(|&x| x == d).unwrap(), v)).collect();
            for &(r, vr) in &pos {
                for &(cc, vc) in &pos {
                    loc[r * k + cc] += c.k * vr * vc;
                }
            }
        }
        for r in 0..k {
            for cc in 0..k {
                if loc[r * k + cc] != 0.0 {
                    t.push((dofs[r], dofs[cc], loc[r * k + cc]));
                }
            }
        }
        i = j;
    }
    t
}

/// Top-`nev` Ritz pairs (largest first) of a symmetric operator.
pub struct Ritz {
    pub vals: Vec<f64>,
    pub vecs: Vec<Vec<f64>>,
    /// Lanczos residual estimates β_k |s_k,i| (absolute).
    pub est: Vec<f64>,
    pub steps: usize,
    pub matvecs: usize,
}

/// Symmetric Lanczos with full re-orthogonalisation (classical Gram–Schmidt, twice).
/// Stops when the top `nev` estimates are ≤ tol·|θ_1|, on an invariant subspace, or
/// after `max_steps`.
pub fn lanczos(n: usize, op: &mut dyn FnMut(&[f64], &mut [f64]), start: &[f64], nev: usize, tol: f64, max_steps: usize) -> Ritz {
    let max_steps = max_steps.min(n);
    let s0 = norm(start);
    let mut v: Vec<Vec<f64>> = vec![start.iter().map(|x| x / s0).collect()];
    let (mut alpha, mut beta): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut w = vec![0.0; n];
    let mut matvecs = 0;
    let mut next_check = (nev + 10).min(max_steps);
    loop {
        let j = v.len() - 1;
        op(&v[j], &mut w);
        matvecs += 1;
        let a = dot(&v[j], &w);
        axpy(-a, &v[j], &mut w);
        if j > 0 {
            axpy(-beta[j - 1], &v[j - 1], &mut w);
        }
        for _ in 0..2 {
            for q in &v {
                let h = dot(q, &w);
                axpy(-h, q, &mut w);
            }
        }
        alpha.push(a);
        let b = norm(&w);
        let m = alpha.len();
        let last = m >= max_steps;
        if m >= next_check || last || b == 0.0 {
            let mut lastrow = vec![vec![0.0; m]];
            lastrow[0][m - 1] = 1.0;
            let s = tridiag_eig(&alpha, &beta, &mut lastrow);
            let mut order: Vec<usize> = (0..m).collect();
            order.sort_by(|&x, &y| s[y].partial_cmp(&s[x]).unwrap());
            let k = nev.min(m);
            let top = &order[..k];
            let est: Vec<f64> = top.iter().map(|&i| b * lastrow[0][i].abs()).collect();
            let scale = s[order[0]].abs();
            let done = est.iter().all(|&e| e <= tol * scale) || b <= 1e-14 * scale || last;
            if done {
                let mut z: Vec<Vec<f64>> = (0..m).map(|r| (0..m).map(|c| if r == c { 1.0 } else { 0.0 }).collect()).collect();
                tridiag_eig(&alpha, &beta, &mut z);
                let vecs = top
                    .iter()
                    .map(|&i| {
                        let mut y = vec![0.0; n];
                        for (l, q) in v.iter().enumerate() {
                            axpy(z[l][i], q, &mut y);
                        }
                        y
                    })
                    .collect();
                return Ritz { vals: top.iter().map(|&i| s[i]).collect(), vecs, est, steps: m, matvecs };
            }
            next_check = m + (m / 8).max(5);
        }
        beta.push(b);
        v.push(w.iter().map(|x| x / b).collect());
    }
}

pub fn random_start(n: usize, seed: u64) -> Vec<f64> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.range(-1.0, 1.0)).collect()
}

/// Eigenpairs with explicit normalised residuals η = ‖By − λy‖/(λ‖y‖).
pub struct Pairs {
    pub vals: Vec<f64>,
    pub vecs: Vec<Vec<f64>>,
    pub resid: Vec<f64>,
    pub steps: usize,
    pub matvecs: usize,
}

/// Upper bounds on λ_max(B) and the per-cell quantities behind them.
#[derive(Clone, Debug)]
pub struct Bounds {
    pub gershgorin: f64,
    pub collatz_wielandt: f64,
    /// Splitting certificate with t_c = 1 (G_i ⪯ 2 B_ii).
    pub split_t1: f64,
    /// Splitting certificate with balanced t_c (min over iterates).
    pub split: f64,
    /// Per-cell U_i of the best splitting iterate.
    pub u_cell: Vec<f64>,
    /// Per-cell lower bounds L_i = λ_max(B_ii).
    pub l_cell: Vec<f64>,
}

/// Result of the Sylvester-inertia bisection: λ_max ∈ [lower, upper].
#[derive(Clone, Debug)]
pub struct Sylvester {
    pub lower: f64,
    pub upper: f64,
    pub factorizations: usize,
    /// σ = θ(1 + 1e-6) was already positive definite.
    pub first_try: bool,
}

/// Relative margin added to a Cholesky-certified bound (f64 backward error).
pub const SYL_MARGIN: f64 = 1e-9;

impl Spectral {
    pub fn new(n: usize, blocks: Vec<Block>, comps: Vec<Comp>) -> Self {
        let mut dof_block = vec![usize::MAX; n];
        let mut local = vec![usize::MAX; n];
        for (bi, b) in blocks.iter().enumerate() {
            for (l, &d) in b.dofs.iter().enumerate() {
                dof_block[d] = bi;
                local[d] = l;
            }
        }
        assert!(dof_block.iter().all(|&b| b != usize::MAX), "every DOF needs a mass block");
        let minvsqrt: Vec<Vec<f64>> = blocks.iter().map(|b| mat_fn(&b.mass, b.dofs.len(), |s| 1.0 / s.sqrt())).collect();
        let msqrt: Vec<Vec<f64>> = blocks.iter().map(|b| mat_fn(&b.mass, b.dofs.len(), |s| s.sqrt())).collect();
        let mut wsub = Vec::with_capacity(comps.len());
        let mut flats: Vec<Vec<(usize, f64)>> = Vec::with_capacity(comps.len());
        for c in &comps {
            let mut subs: Vec<(usize, Vec<f64>)> = Vec::new();
            for &(d, v) in &c.ent {
                let bi = dof_block[d];
                let k = blocks[bi].dofs.len();
                let pos = match subs.iter().position(|s| s.0 == bi) {
                    Some(p) => p,
                    None => {
                        subs.push((bi, vec![0.0; k]));
                        subs.len() - 1
                    }
                };
                subs[pos].1[local[d]] += v;
            }
            for (bi, j) in subs.iter_mut() {
                let k = j.len();
                let mi = &minvsqrt[*bi];
                *j = (0..k).map(|r| (0..k).map(|l| mi[r * k + l] * j[l]).sum()).collect();
            }
            flats.push(subs.iter().flat_map(|(bi, j)| blocks[*bi].dofs.iter().copied().zip(j.iter().copied())).filter(|x| x.1 != 0.0).collect());
            wsub.push(subs);
        }
        let mut trip = grouped_triplets(&comps, &flats);
        trip.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let mut ptr = vec![0usize; n + 1];
        let mut idx = Vec::new();
        let mut val: Vec<f64> = Vec::new();
        let mut last: Option<(usize, usize)> = None;
        for (r, c, v) in trip {
            if last == Some((r, c)) {
                *val.last_mut().unwrap() += v;
            } else {
                idx.push(c);
                val.push(v);
                ptr[r + 1] += 1;
                last = Some((r, c));
            }
        }
        for r in 0..n {
            ptr[r + 1] += ptr[r];
        }
        Spectral { n, blocks, dof_block, comps, wsub, msqrt, ptr, idx, val }
    }

    pub fn apply_b(&self, x: &[f64], y: &mut [f64]) {
        for r in 0..self.n {
            let mut s = 0.0;
            for p in self.ptr[r]..self.ptr[r + 1] {
                s += self.val[p] * x[self.idx[p]];
            }
            y[r] = s;
        }
    }

    fn apply_msqrt(&self, x: &[f64], y: &mut [f64]) {
        for (bi, b) in self.blocks.iter().enumerate() {
            let k = b.dofs.len();
            for r in 0..k {
                y[b.dofs[r]] = (0..k).map(|l| self.msqrt[bi][r * k + l] * x[b.dofs[l]]).sum();
            }
        }
    }

    /// Normalised residual η = ‖By − λy‖/(|λ|‖y‖).
    pub fn residual(&self, y: &[f64], lam: f64) -> f64 {
        let mut by = vec![0.0; self.n];
        self.apply_b(y, &mut by);
        let r: f64 = by.iter().zip(y).map(|(a, b)| (a - lam * b).powi(2)).sum::<f64>().sqrt();
        r / (lam.abs() * norm(y))
    }

    fn pairs(&self, r: Ritz, invert: bool) -> Pairs {
        let vals: Vec<f64> = r.vals.iter().map(|&v| if invert { 1.0 / v } else { v }).collect();
        let vecs: Vec<Vec<f64>> = r.vecs.into_iter().map(|y| {
            let s = norm(&y);
            y.into_iter().map(|x| x / s).collect()
        }).collect();
        let resid = vals.iter().zip(&vecs).map(|(&l, y)| self.residual(y, l)).collect();
        Pairs { vals, vecs, resid, steps: r.steps, matvecs: r.matvecs }
    }

    /// Largest `nev` eigenpairs of B.
    pub fn top(&self, nev: usize, start_seed: u64, tol: f64, max_steps: usize) -> Pairs {
        let start = random_start(self.n, start_seed);
        let mut op = |x: &[f64], y: &mut [f64]| self.apply_b(x, y);
        let r = lanczos(self.n, &mut op, &start, nev, tol, max_steps);
        self.pairs(r, false)
    }

    /// K = Σ k j jᵀ (unwhitened) as triplets.
    pub fn k_triplets(&self) -> Vec<(usize, usize, f64)> {
        let ents: Vec<Vec<(usize, f64)>> = self.comps.iter().map(|c| c.ent.clone()).collect();
        grouped_triplets(&self.comps, &ents)
    }

    fn m_triplets(&self, scale: f64) -> Vec<(usize, usize, f64)> {
        let mut t = Vec::new();
        for b in &self.blocks {
            let k = b.dofs.len();
            for r in 0..k {
                for c in 0..k {
                    t.push((b.dofs[r], b.dofs[c], scale * b.mass[r * k + c]));
                }
            }
        }
        t
    }

    /// Smallest `nev` eigenpairs by shift-invert (K must be positive definite).
    pub fn low(&self, nev: usize, start_seed: u64, tol: f64, max_steps: usize) -> Result<Pairs, String> {
        let kmat = csc_from_triplets(self.n, self.k_triplets());
        let mut spd = Spd::analyze(&kmat);
        spd.factor(&kmat)?;
        let start = random_start(self.n, start_seed);
        let mut tmp = vec![0.0; self.n];
        let mut op = |x: &[f64], y: &mut [f64]| {
            self.apply_msqrt(x, &mut tmp);
            spd.solve(&mut tmp);
            self.apply_msqrt(&tmp, y);
        };
        let r = lanczos(self.n, &mut op, &start, nev, tol, max_steps);
        Ok(self.pairs(r, true))
    }

    /// All eigenvalues of B (descending), dense.
    pub fn dense_eigs(&self) -> Vec<f64> {
        let mut a = Mat::<f64>::zeros(self.n, self.n);
        for r in 0..self.n {
            for p in self.ptr[r]..self.ptr[r + 1] {
                a[(r, self.idx[p])] = self.val[p];
            }
        }
        let mut ev: Vec<f64> = a.self_adjoint_eigenvalues(Side::Lower).expect("dense eig").to_vec();
        ev.sort_by(|a, b| b.partial_cmp(a).unwrap());
        ev
    }

    fn cell_g(&self, weight: &dyn Fn(usize, usize) -> f64) -> Vec<f64> {
        let mut g: Vec<Vec<f64>> = self.blocks.iter().map(|b| vec![0.0; b.dofs.len().pow(2)]).collect();
        for (ci, subs) in self.wsub.iter().enumerate() {
            let k = self.comps[ci].k;
            for (side, (bi, v)) in subs.iter().enumerate() {
                let w = k * weight(ci, side);
                let kk = v.len();
                for r in 0..kk {
                    for c in 0..kk {
                        g[*bi][r * kk + c] += w * v[r] * v[c];
                    }
                }
            }
        }
        g.iter().zip(&self.blocks).map(|(a, b)| sym_max_eig(a, b.dofs.len())).collect()
    }

    pub fn bounds(&self) -> Bounds {
        let mut gersh: f64 = 0.0;
        for r in 0..self.n {
            gersh = gersh.max((self.ptr[r]..self.ptr[r + 1]).map(|p| self.val[p].abs()).sum());
        }
        // Collatz–Wielandt: every positive x gives ρ(|B|) ≤ max (|B|x)_i / x_i
        let mut x = vec![1.0; self.n];
        let mut cw = f64::INFINITY;
        for _ in 0..60 {
            let mut y = vec![0.0; self.n];
            for r in 0..self.n {
                y[r] = (self.ptr[r]..self.ptr[r + 1]).map(|p| self.val[p].abs() * x[self.idx[p]]).sum();
            }
            let ub = (0..self.n).map(|i| y[i] / x[i]).fold(0.0, f64::max);
            cw = cw.min(ub);
            let mx = y.iter().cloned().fold(0.0, f64::max);
            if !(mx > 0.0) || y.iter().any(|&v| !(v > 0.0)) {
                break;
            }
            x = y.iter().map(|v| v / mx).collect();
        }
        let l_cell = self.cell_g(&|_, _| 1.0);
        let two_sided = |ci: usize| self.wsub[ci].len() == 2;
        let u1 = self.cell_g(&|ci, _| if two_sided(ci) { 2.0 } else { 1.0 });
        let split_t1 = u1.iter().cloned().fold(0.0, f64::max);
        // balance t_c: more weight on the side whose cell bound is lower
        let mut t = vec![1.0f64; self.comps.len()];
        let (mut best, mut best_u) = (split_t1, u1.clone());
        let mut u = u1;
        for _ in 0..15 {
            for ci in 0..self.comps.len() {
                if two_sided(ci) {
                    let (a, b) = (self.wsub[ci][0].0, self.wsub[ci][1].0);
                    t[ci] = (t[ci] * (u[b] / u[a]).sqrt()).clamp(1e-3, 1e3);
                }
            }
            u = self.cell_g(&|ci, side| if !two_sided(ci) { 1.0 } else if side == 0 { 1.0 + t[ci] } else { 1.0 + 1.0 / t[ci] });
            let mx = u.iter().cloned().fold(0.0, f64::max);
            if mx < best {
                best = mx;
                best_u = u.clone();
            }
        }
        Bounds { gershgorin: gersh, collatz_wielandt: cw, split_t1, split: best, u_cell: best_u, l_cell }
    }

    /// Certified bracket of λ_max from positive definiteness of σM − K.
    pub fn sylvester(&self, theta: f64, hi: f64, rel_tol: f64) -> Sylvester {
        let mut t = self.m_triplets(1.0);
        let nm = t.len();
        t.extend(self.k_triplets().into_iter().map(|(r, c, v)| (r, c, -v)));
        let mut factorizations = 0;
        let pattern = csc_from_triplets(self.n, t.clone());
        let mut spd = Spd::analyze(&pattern);
        let mut pd = |sigma: f64| {
            let mut tt = t.clone();
            for e in tt.iter_mut().take(nm) {
                e.2 *= sigma;
            }
            factorizations += 1;
            spd.factor(&csc_from_triplets(self.n, tt)).is_ok()
        };
        let s1 = theta * (1.0 + 1e-6);
        if pd(s1) {
            return Sylvester { lower: theta, upper: s1 * (1.0 + SYL_MARGIN), factorizations, first_try: true };
        }
        let (mut a, mut b) = (s1, hi);
        while !pd(b) {
            a = b;
            b *= 2.0;
        }
        while (b - a) > rel_tol * b {
            let mid = 0.5 * (a + b);
            if pd(mid) {
                b = mid;
            } else {
                a = mid;
            }
        }
        Sylvester { lower: a, upper: b * (1.0 + SYL_MARGIN), factorizations, first_try: false }
    }

    /// Per-block kinetic shares (total, translation, rotation) of a unit vector y.
    pub fn kinetic_shares(&self, y: &[f64]) -> Vec<[f64; 3]> {
        let s = dot(y, y);
        self.blocks
            .iter()
            .map(|b| {
                let (mut tr, mut ro) = (0.0, 0.0);
                for (l, &d) in b.dofs.iter().enumerate() {
                    if b.rot[l] {
                        ro += y[d] * y[d] / s;
                    } else {
                        tr += y[d] * y[d] / s;
                    }
                }
                [tr + ro, tr, ro]
            })
            .collect()
    }

    /// Per-component elastic shares k_c (v_c·y)²/(yᵀBy).
    pub fn elastic_shares(&self, y: &[f64]) -> Vec<f64> {
        let e: Vec<f64> = self
            .wsub
            .iter()
            .zip(&self.comps)
            .map(|(subs, c)| {
                let p: f64 = subs.iter().map(|(bi, v)| self.blocks[*bi].dofs.iter().zip(v).map(|(&d, x)| y[d] * x).sum::<f64>()).sum();
                c.k * p * p
            })
            .collect();
        let tot: f64 = e.iter().sum();
        e.into_iter().map(|x| x / tot).collect()
    }

    /// Spectral problem of a 2D model at its reference configuration (elastic
    /// stiffness of every active facet point; fixed DOFs removed).
    pub fn from_model2d(m: &Model) -> Spectral {
        let (idx, n) = m.dof_map();
        let q0: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
        let mut blocks = Vec::new();
        for (i, b) in m.bodies.iter().enumerate() {
            let dofs: Vec<(usize, usize)> = (0..3).filter_map(|k| idx[i][k].map(|g| (k, g))).collect();
            if dofs.is_empty() {
                continue;
            }
            let kk = dofs.len();
            let mut mass = vec![0.0; kk * kk];
            for (l, &(k, _)) in dofs.iter().enumerate() {
                mass[l * kk + l] = if k < 2 { b.mass } else { b.inertia };
            }
            blocks.push(Block { body: i, dofs: dofs.iter().map(|x| x.1).collect(), rot: dofs.iter().map(|x| x.0 == 2).collect(), mass });
        }
        let mut comps = Vec::new();
        for (fi, f) in m.facets.iter().enumerate().filter(|(_, f)| f.active) {
            let class = if m.bodies[f.b].kind == BodyKind::Cell {
                CompClass::Interior
            } else if m.is_free(f.b) {
                CompClass::Contact
            } else {
                CompClass::Support
            };
            for p in 0..2 {
                let (j, _) = m.eval_point(&q0, f, p);
                let law = &f.pts[p].law;
                for (ja, jb, k) in [(j.jna, j.jnb, law.kn), (j.jta, j.jtb, law.kt)] {
                    if k == 0.0 {
                        continue;
                    }
                    let mut ent = Vec::new();
                    for r in 0..3 {
                        if let Some(g) = idx[f.a][r] {
                            ent.push((g, ja[r]));
                        }
                        if let Some(g) = idx[f.b][r] {
                            ent.push((g, jb[r]));
                        }
                    }
                    comps.push(Comp { ent, k, class, facet: fi });
                }
            }
        }
        Spectral::new(n, blocks, comps)
    }

    /// Spectral problem of the linearised 3D model (support facets → ground).
    pub fn from_model3(m: &Model3) -> Spectral {
        let blocks = m
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut mass = vec![0.0; 36];
                for k in 0..3 {
                    mass[k * 6 + k] = c.mass;
                    for l in 0..3 {
                        mass[(3 + k) * 6 + 3 + l] = c.inertia[k][l];
                    }
                }
                Block { body: i, dofs: (6 * i..6 * i + 6).collect(), rot: vec![false, false, false, true, true, true], mass }
            })
            .collect();
        let comps = m
            .comps
            .iter()
            .map(|c| {
                let mut ent: Vec<(usize, f64)> = (0..6).filter(|&k| c.ja[k] != 0.0).map(|k| (6 * c.a + k, c.ja[k])).collect();
                if let Some(b) = c.b {
                    ent.extend((0..6).filter(|&k| c.jb[k] != 0.0).map(|k| (6 * b + k, c.jb[k])));
                }
                Comp { ent, k: c.k, class: if c.b.is_some() { CompClass::Interior } else { CompClass::Support }, facet: c.facet }
            })
            .collect();
        Spectral::new(m.ndof(), blocks, comps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Chain of n unit masses between two walls, springs k: λ_j = 2k(1 − cos(jπ/(n+1))).
    fn chain(n: usize, kspring: f64, mass: f64) -> Spectral {
        let blocks = (0..n).map(|i| Block { body: i, dofs: vec![i], rot: vec![false], mass: vec![mass] }).collect();
        let mut comps = Vec::new();
        for i in 0..=n {
            let mut ent = Vec::new();
            if i > 0 {
                ent.push((i - 1, -1.0));
            }
            if i < n {
                ent.push((i, 1.0));
            }
            let class = if i == 0 || i == n { CompClass::Support } else { CompClass::Interior };
            comps.push(Comp { ent, k: kspring, class, facet: i });
        }
        Spectral::new(n, blocks, comps)
    }

    #[test]
    fn chain_spectrum_and_bounds() {
        let (n, k, m) = (60, 3.0, 2.0);
        let s = chain(n, k, m);
        let exact = |j: usize| 2.0 * k / m * (1.0 - (j as f64 * std::f64::consts::PI / (n + 1) as f64).cos());
        let top = s.top(4, 11, 1e-12, 200);
        for (i, &v) in top.vals.iter().enumerate() {
            assert!((v - exact(n - i)).abs() < 1e-9 * v, "{v} vs {}", exact(n - i));
            assert!(top.resid[i] < 1e-8, "resid {}", top.resid[i]);
        }
        let low = s.low(3, 12, 1e-12, 200).unwrap();
        for (i, &v) in low.vals.iter().enumerate() {
            assert!((v - exact(i + 1)).abs() < 1e-8 * v, "low {v} vs {}", exact(i + 1));
        }
        let dense = s.dense_eigs();
        assert!((dense[0] - exact(n)).abs() < 1e-10 * dense[0]);
        let b = s.bounds();
        let lmax = exact(n);
        assert!(b.gershgorin >= lmax && b.collatz_wielandt >= lmax && b.split >= lmax && b.split_t1 >= lmax);
        // uniform chain: interior block 2k/m, split bound 4k/m (= Gershgorin)
        assert!((b.split_t1 - 4.0 * k / m).abs() < 1e-12);
        assert!(b.l_cell.iter().cloned().fold(0.0, f64::max) <= lmax);
        let syl = s.sylvester(top.vals[0], b.split, 1e-10);
        assert!(syl.first_try && syl.upper >= lmax && syl.upper <= lmax * (1.0 + 2e-6));
        // a deliberately low guess must be repaired by bisection
        let syl2 = s.sylvester(0.5 * lmax, b.split, 1e-10);
        assert!(!syl2.first_try && syl2.lower <= lmax && syl2.upper >= lmax && syl2.upper < lmax * (1.0 + 1e-8));
    }

    #[test]
    fn tridiagonal_ql() {
        let n = 50;
        let d = vec![2.0; n];
        let e = vec![-1.0; n - 1];
        let mut z: Vec<Vec<f64>> = (0..n).map(|r| (0..n).map(|c| if r == c { 1.0 } else { 0.0 }).collect()).collect();
        let s = tridiag_eig(&d, &e, &mut z);
        let mut ex: Vec<f64> = (1..=n).map(|k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n + 1) as f64).cos()).collect();
        let mut got = s.clone();
        ex.sort_by(|a, b| a.partial_cmp(b).unwrap());
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in got.iter().zip(&ex) {
            assert!((a - b).abs() < 1e-13, "{a} {b}");
        }
        for j in 0..n {
            let mut r: f64 = 0.0;
            for i in 0..n {
                let mut tu = d[i] * z[i][j];
                if i > 0 {
                    tu += e[i - 1] * z[i - 1][j];
                }
                if i + 1 < n {
                    tu += e[i] * z[i + 1][j];
                }
                r = r.max((tu - s[j] * z[i][j]).abs());
            }
            assert!(r < 1e-13, "vector {j}: {r}");
        }
    }

    #[test]
    fn whitening_full_inertia_block() {
        // one 2-DOF body with a non-diagonal mass block, springs to ground on each DOF
        let blocks = vec![Block { body: 0, dofs: vec![0, 1], rot: vec![false, true], mass: vec![2.0, 0.5, 0.5, 1.0] }];
        let comps = vec![
            Comp { ent: vec![(0, 1.0)], k: 3.0, class: CompClass::Support, facet: 0 },
            Comp { ent: vec![(1, 1.0)], k: 5.0, class: CompClass::Support, facet: 1 },
        ];
        let s = Spectral::new(2, blocks, comps);
        // generalized eigenvalues of K = diag(3,5), M = [[2,.5],[.5,1]]
        let ev = s.dense_eigs();
        for &l in &ev {
            let det = (3.0 - 2.0 * l) * (5.0 - l) - (0.5 * l) * (0.5 * l);
            assert!(det.abs() < 1e-10, "{l} {det}");
        }
    }
}
