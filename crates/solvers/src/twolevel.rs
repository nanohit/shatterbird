//! Phase 0b / 2: two-level solution of the linear step system A u = b, A = μM + K,
//! on cell blocks (3 DOF per cell in 2D, 6 in 3D; DOFs of cell i are k·i … k·i+k−1).
//!
//! Smoothers: coloured block Gauss–Seidel (forward / backward) and Chebyshev on the
//! block-Jacobi-preconditioned operator D⁻¹A. Coarse spaces: piecewise-rigid motions
//! of deterministic compact aggregates, optionally smoothed (SA). The coarse system
//! A_c = PᵀAP is solved exactly by sparse Cholesky. The symmetric cycle is used as a
//! stationary iteration or as the PCG preconditioner. Every application is counted in
//! `Work` so that GPU-relevant quantities (A-applications, dependent phases, global
//! reductions, coarse solves) can be compared at equal accuracy.

use crate::sparse::{csc_from_triplets, Spd};
use crate::spectrum::lanczos;
use std::collections::BinaryHeap;
use std::time::Instant;

/// Scalar CSR matrix (structural zeros are kept: a stiffness scaled by η = 0 keeps
/// its pattern, so numeric refactorisations can reuse the symbolic analysis).
#[derive(Clone, Debug)]
pub struct Csr {
    pub nr: usize,
    pub nc: usize,
    pub ptr: Vec<usize>,
    pub idx: Vec<usize>,
    pub val: Vec<f64>,
}

impl Csr {
    pub fn from_triplets(nr: usize, nc: usize, mut t: Vec<(usize, usize, f64)>) -> Csr {
        t.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let mut ptr = vec![0usize; nr + 1];
        let mut idx = Vec::with_capacity(t.len());
        let mut val: Vec<f64> = Vec::with_capacity(t.len());
        let mut last: Option<(usize, usize)> = None;
        for (r, c, v) in t {
            if last == Some((r, c)) {
                *val.last_mut().unwrap() += v;
            } else {
                idx.push(c);
                val.push(v);
                ptr[r + 1] += 1;
                last = Some((r, c));
            }
        }
        for r in 0..nr {
            ptr[r + 1] += ptr[r];
        }
        Csr { nr, nc, ptr, idx, val }
    }

    pub fn nnz(&self) -> usize {
        self.idx.len()
    }

    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for r in 0..self.nr {
            let mut s = 0.0;
            for p in self.ptr[r]..self.ptr[r + 1] {
                s += self.val[p] * x[self.idx[p]];
            }
            y[r] = s;
        }
    }

    pub fn row_dot(&self, r: usize, x: &[f64]) -> f64 {
        let mut s = 0.0;
        for p in self.ptr[r]..self.ptr[r + 1] {
            s += self.val[p] * x[self.idx[p]];
        }
        s
    }

    pub fn transpose(&self) -> Csr {
        let mut t = Vec::with_capacity(self.nnz());
        for r in 0..self.nr {
            for p in self.ptr[r]..self.ptr[r + 1] {
                t.push((self.idx[p], r, self.val[p]));
            }
        }
        Csr::from_triplets(self.nc, self.nr, t)
    }

    /// self · b with a sparse accumulator; the pattern is the structural product.
    pub fn matmul(&self, b: &Csr) -> Csr {
        assert_eq!(self.nc, b.nr);
        let mut acc = vec![0.0; b.nc];
        let mut mark = vec![usize::MAX; b.nc];
        let mut ptr = vec![0usize; self.nr + 1];
        let (mut idx, mut val) = (Vec::new(), Vec::new());
        let mut cols: Vec<usize> = Vec::new();
        for r in 0..self.nr {
            cols.clear();
            for p in self.ptr[r]..self.ptr[r + 1] {
                let (k, a) = (self.idx[p], self.val[p]);
                for q in b.ptr[k]..b.ptr[k + 1] {
                    let c = b.idx[q];
                    if mark[c] != r {
                        mark[c] = r;
                        acc[c] = 0.0;
                        cols.push(c);
                    }
                    acc[c] += a * b.val[q];
                }
            }
            cols.sort_unstable();
            for &c in &cols {
                idx.push(c);
                val.push(acc[c]);
            }
            ptr[r + 1] = idx.len();
        }
        Csr { nr: self.nr, nc: b.nc, ptr, idx, val }
    }

    pub fn triplets(&self) -> Vec<(usize, usize, f64)> {
        let mut t = Vec::with_capacity(self.nnz());
        for r in 0..self.nr {
            for p in self.ptr[r]..self.ptr[r + 1] {
                t.push((r, self.idx[p], self.val[p]));
            }
        }
        t
    }
}

/// Work counters. One A-application = all facets once (a matvec, a residual or a
/// block-GS sweep). A phase is a dependent stage (one GPU dispatch at best).
#[derive(Clone, Copy, Debug, Default)]
pub struct Work {
    pub a_apps: f64,
    pub block_passes: f64,
    /// dependent stages: matvec / sweep colour / Chebyshev step / restriction / coarse
    /// solve / prolongation / global reduction (vector updates are assumed fused)
    pub phases: f64,
    /// every global inner product or norm actually computed
    pub reductions: f64,
    pub coarse_solves: f64,
    pub transfers: f64,
    /// full-length vector updates (axpy-type passes)
    pub vec_ops: f64,
}

impl Work {
    pub fn add(&mut self, o: &Work) {
        self.a_apps += o.a_apps;
        self.block_passes += o.block_passes;
        self.phases += o.phases;
        self.reductions += o.reductions;
        self.coarse_solves += o.coarse_solves;
        self.transfers += o.transfers;
        self.vec_ops += o.vec_ops;
    }
}

/// k×k Cholesky (row-major, lower) of an SPD block.
fn chol(a: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; k * k];
    for j in 0..k {
        let mut s = a[j * k + j];
        for p in 0..j {
            s -= l[j * k + p] * l[j * k + p];
        }
        if !(s > 0.0) {
            return None;
        }
        l[j * k + j] = s.sqrt();
        for i in j + 1..k {
            let mut s = a[i * k + j];
            for p in 0..j {
                s -= l[i * k + p] * l[j * k + p];
            }
            l[i * k + j] = s / l[j * k + j];
        }
    }
    Some(l)
}

fn lsolve(l: &[f64], k: usize, x: &mut [f64]) {
    for i in 0..k {
        for p in 0..i {
            x[i] -= l[i * k + p] * x[p];
        }
        x[i] /= l[i * k + i];
    }
}

fn ltsolve(l: &[f64], k: usize, x: &mut [f64]) {
    for i in (0..k).rev() {
        for p in i + 1..k {
            x[i] -= l[p * k + i] * x[p];
        }
        x[i] /= l[i * k + i];
    }
}

/// The fine system: A, block-diagonal Cholesky factors and a colouring of the cells.
pub struct System {
    pub n: usize,
    pub k: usize,
    pub nb: usize,
    pub a: Csr,
    pub dl: Vec<Vec<f64>>,
    pub colors: Vec<Vec<usize>>,
    pub adj: Vec<Vec<usize>>,
}

impl System {
    pub fn new(a: Csr, k: usize) -> System {
        let n = a.nr;
        assert_eq!(n % k, 0);
        let nb = n / k;
        let mut dl = Vec::with_capacity(nb);
        let mut adj = vec![Vec::new(); nb];
        for i in 0..nb {
            let mut d = vec![0.0; k * k];
            for r in 0..k {
                let row = k * i + r;
                for p in a.ptr[row]..a.ptr[row + 1] {
                    let c = a.idx[p];
                    let j = c / k;
                    if j == i {
                        d[r * k + (c - k * i)] = a.val[p];
                    } else if !adj[i].contains(&j) {
                        adj[i].push(j);
                    }
                }
            }
            dl.push(chol(&d, k).expect("diagonal block not SPD"));
        }
        for v in adj.iter_mut() {
            v.sort_unstable();
        }
        // greedy colouring in cell-ID order (deterministic)
        let mut color = vec![usize::MAX; nb];
        let mut ncol = 0;
        for i in 0..nb {
            let mut c = 0;
            while adj[i].iter().any(|&j| color[j] == c) {
                c += 1;
            }
            color[i] = c;
            ncol = ncol.max(c + 1);
        }
        let mut colors = vec![Vec::new(); ncol];
        for i in 0..nb {
            colors[color[i]].push(i);
        }
        System { n, k, nb, a, dl, colors, adj }
    }

    /// Operator-only system (no block factors, no colouring): enough for the outer Krylov
    /// method when the preconditioner carries its own System. Accepts indefinite A.
    pub fn operator(a: Csr, k: usize) -> System {
        let n = a.nr;
        assert_eq!(n % k, 0);
        System { n, k, nb: n / k, a, dl: Vec::new(), colors: Vec::new(), adj: Vec::new() }
    }

    pub fn apply(&self, x: &[f64], y: &mut [f64], w: &mut Work) {
        self.a.matvec(x, y);
        w.a_apps += 1.0;
        w.phases += 1.0;
    }

    pub fn residual(&self, b: &[f64], x: &[f64], r: &mut [f64], w: &mut Work) {
        self.apply(x, r, w);
        for i in 0..self.n {
            r[i] = b[i] - r[i];
        }
    }

    /// y = D⁻¹ x.
    pub fn dinv(&self, x: &[f64], y: &mut [f64], w: &mut Work) {
        let k = self.k;
        for i in 0..self.nb {
            let s = &mut y[k * i..k * i + k];
            s.copy_from_slice(&x[k * i..k * i + k]);
            lsolve(&self.dl[i], k, s);
            ltsolve(&self.dl[i], k, s);
        }
        w.block_passes += 1.0;
        w.phases += 1.0;
    }

    fn block_update(&self, i: usize, b: &[f64], x: &mut [f64]) {
        let k = self.k;
        let mut r = vec![0.0; k];
        for c in 0..k {
            r[c] = b[k * i + c] - self.a.row_dot(k * i + c, x);
        }
        lsolve(&self.dl[i], k, &mut r);
        ltsolve(&self.dl[i], k, &mut r);
        for c in 0..k {
            x[k * i + c] += r[c];
        }
    }

    /// One coloured block-GS sweep; `backward` reverses the colour (and block) order.
    pub fn gs_sweep(&self, b: &[f64], x: &mut [f64], backward: bool, w: &mut Work) {
        if backward {
            for col in self.colors.iter().rev() {
                for &i in col.iter().rev() {
                    self.block_update(i, b, x);
                }
            }
        } else {
            for col in &self.colors {
                for &i in col {
                    self.block_update(i, b, x);
                }
            }
        }
        w.a_apps += 1.0;
        w.block_passes += 1.0;
        w.phases += self.colors.len() as f64;
    }

    /// λ_max(D⁻¹A) by 15 Lanczos steps on L⁻¹AL⁻ᵀ (deterministic start); ≤ 2 holds
    /// exactly for pairwise positive spring contributions (A ⪯ 2D).
    pub fn lambda_dinv_a(&self) -> f64 {
        let k = self.k;
        let n = self.n;
        let mut tmp = vec![0.0; n];
        let mut op = |x: &[f64], y: &mut [f64]| {
            tmp.copy_from_slice(x);
            for i in 0..self.nb {
                ltsolve(&self.dl[i], k, &mut tmp[k * i..k * i + k]);
            }
            self.a.matvec(&tmp, y);
            for i in 0..self.nb {
                lsolve(&self.dl[i], k, &mut y[k * i..k * i + k]);
            }
        };
        let start = crate::spectrum::random_start(n, 21);
        let r = lanczos(n, &mut op, &start, 1, 1e-6, 15);
        r.vals[0]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Smoother {
    /// one forward coloured block-GS sweep before, one backward after the correction
    Gs,
    /// Chebyshev of the given degree on D⁻¹A over [hi/ratio, hi]
    Cheb { deg: usize, lo: f64, hi: f64 },
}

/// Chebyshev smoothing of A x = b (x updated in place). `x_zero` skips the first
/// residual when x is known to be zero.
pub fn chebyshev(sys: &System, b: &[f64], x: &mut [f64], deg: usize, lo: f64, hi: f64, x_zero: bool, w: &mut Work) {
    let n = sys.n;
    let theta = 0.5 * (hi + lo);
    let delta = 0.5 * (hi - lo);
    let sigma = theta / delta;
    let mut rho = 1.0 / sigma;
    let mut r = vec![0.0; n];
    if x_zero {
        r.copy_from_slice(b);
    } else {
        sys.residual(b, x, &mut r, w);
    }
    let mut z = vec![0.0; n];
    sys.dinv(&r, &mut z, w);
    let mut d: Vec<f64> = z.iter().map(|v| v / theta).collect();
    for i in 0..n {
        x[i] += d[i];
    }
    w.vec_ops += 1.0;
    for _ in 1..deg {
        sys.residual(b, x, &mut r, w);
        sys.dinv(&r, &mut z, w);
        let rho_new = 1.0 / (2.0 * sigma - rho);
        for i in 0..n {
            d[i] = rho_new * rho * d[i] + 2.0 * rho_new / delta * z[i];
            x[i] += d[i];
        }
        w.vec_ops += 1.0;
        rho = rho_new;
    }
}

// ---------------------------------------------------------------- aggregation

/// Input of the deterministic aggregation: cell centroids (z = 0 in 2D), masses,
/// facet graph edges with a strong flag, target size, dimension, mean cell size.
pub struct AggInput<'a> {
    pub pos: &'a [[f64; 3]],
    pub mass: &'a [f64],
    pub edges: &'a [(usize, usize, bool)],
    pub target: usize,
    pub dim: usize,
    pub ell: f64,
}

#[derive(Clone, Debug)]
pub struct Aggregates {
    pub of: Vec<usize>,
    pub n: usize,
    pub sizes: Vec<usize>,
    /// max over cells of (path distance to seed)/(Euclidean distance to seed + ℓ̄),
    /// before merging
    pub max_detour: f64,
    pub reseeds: usize,
    pub merged: usize,
}

#[derive(PartialEq)]
struct Item(f64, usize, usize); // (distance, seed id order, cell)

impl Eq for Item {}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // min-heap on (distance, seed, cell)
        o.0.partial_cmp(&self.0).unwrap().then(o.1.cmp(&self.1)).then(o.2.cmp(&self.2))
    }
}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Seeds in ID order at hard-core distance R, multi-source Dijkstra over strong edges,
/// detour re-seeding, merge of small aggregates (docs/phase0b/two-level.md §2).
pub fn aggregate(inp: &AggInput) -> Aggregates {
    let nb = inp.pos.len();
    let s = inp.target as f64;
    let r = if inp.dim == 2 { 0.834 * s.sqrt() * inp.ell } else { 0.899 * s.cbrt() * inp.ell };
    let mut nbr: Vec<Vec<usize>> = vec![Vec::new(); nb];
    for &(a, b, st) in inp.edges {
        if st {
            nbr[a].push(b);
            nbr[b].push(a);
        }
    }
    for v in nbr.iter_mut() {
        v.sort_unstable();
        v.dedup();
    }
    let mut seeds: Vec<usize> = Vec::new();
    for i in 0..nb {
        if seeds.iter().all(|&s0| dist(inp.pos[i], inp.pos[s0]) >= r) {
            seeds.push(i);
        }
    }
    let mut reseeds = 0;
    let (owner, pdist) = loop {
        // Dijkstra from all seeds; ties by (distance, seed order)
        let mut owner = vec![usize::MAX; nb];
        let mut pd = vec![f64::INFINITY; nb];
        let mut heap = BinaryHeap::new();
        for (k, &s0) in seeds.iter().enumerate() {
            pd[s0] = 0.0;
            owner[s0] = k;
            heap.push(Item(0.0, k, s0));
        }
        let mut done = vec![false; nb];
        while let Some(Item(d, k, i)) = heap.pop() {
            if done[i] {
                continue;
            }
            if d > pd[i] || (d == pd[i] && k != owner[i]) {
                continue;
            }
            done[i] = true;
            for &j in &nbr[i] {
                let nd = d + dist(inp.pos[i], inp.pos[j]);
                if !done[j] && (nd < pd[j] || (nd == pd[j] && k < owner[j])) {
                    pd[j] = nd;
                    owner[j] = k;
                    heap.push(Item(nd, k, j));
                }
            }
        }
        // unreachable component → new seed; detour → new seed (lowest ID first)
        if let Some(i) = (0..nb).find(|&i| owner[i] == usize::MAX) {
            seeds.push(i);
            reseeds += 1;
            continue;
        }
        if let Some(i) = (0..nb).find(|&i| pd[i] > 2.0 * dist(inp.pos[i], inp.pos[seeds[owner[i]]]) + inp.ell) {
            seeds.push(i);
            reseeds += 1;
            continue;
        }
        break (owner, pd);
    };
    let max_detour = (0..nb).map(|i| pdist[i] / (dist(inp.pos[i], inp.pos[seeds[owner[i]]]) + inp.ell)).fold(0.0, f64::max);
    let mut of = owner;
    // merge small aggregates into the neighbour with the most strong edges
    let mut merged = 0;
    let nag = seeds.len();
    loop {
        let mut size = vec![0usize; nag];
        for &a in &of {
            size[a] += 1;
        }
        let small = (0..nag).find(|&a| size[a] > 0 && size[a] * 4 < inp.target && {
            (0..nb).any(|i| of[i] == a && nbr[i].iter().any(|&j| of[j] != a))
        });
        let Some(a) = small else { break };
        let mut cnt: std::collections::BTreeMap<usize, usize> = Default::default();
        for i in (0..nb).filter(|&i| of[i] == a) {
            for &j in &nbr[i] {
                if of[j] != a {
                    *cnt.entry(of[j]).or_default() += 1;
                }
            }
        }
        let best = cnt.iter().max_by(|x, y| x.1.cmp(y.1).then(y.0.cmp(x.0))).map(|x| *x.0).unwrap();
        for v in of.iter_mut() {
            if *v == a {
                *v = best;
            }
        }
        merged += 1;
    }
    // renumber in order of first appearance by cell ID
    let mut map = vec![usize::MAX; nag];
    let mut n = 0;
    for i in 0..nb {
        if map[of[i]] == usize::MAX {
            map[of[i]] = n;
            n += 1;
        }
    }
    for v in of.iter_mut() {
        *v = map[*v];
    }
    let mut sizes = vec![0usize; n];
    for &a in &of {
        sizes[a] += 1;
    }
    Aggregates { of, n, sizes, max_detour, reseeds, merged }
}

/// Piecewise-rigid prolongator: 3 (2D) or 6 (3D) rigid motions per aggregate about its
/// mass centre; rotation columns scaled by 1/ρ_a (radius of gyration, ≥ ℓ̄/2).
pub fn rigid_p(agg: &Aggregates, pos: &[[f64; 3]], mass: &[f64], k: usize, ell: f64) -> Csr {
    let nb = pos.len();
    let kc = if k == 3 { 3 } else { 6 };
    let mut cen = vec![[0.0; 3]; agg.n];
    let mut msum = vec![0.0; agg.n];
    for i in 0..nb {
        let a = agg.of[i];
        for d in 0..3 {
            cen[a][d] += mass[i] * pos[i][d];
        }
        msum[a] += mass[i];
    }
    for a in 0..agg.n {
        for d in 0..3 {
            cen[a][d] /= msum[a];
        }
    }
    let mut rho = vec![0.0; agg.n];
    for i in 0..nb {
        let a = agg.of[i];
        rho[a] += mass[i] * dist(pos[i], cen[a]).powi(2);
    }
    for a in 0..agg.n {
        rho[a] = (rho[a] / msum[a]).sqrt().max(0.5 * ell);
    }
    let mut t = Vec::new();
    for i in 0..nb {
        let a = agg.of[i];
        let r = [pos[i][0] - cen[a][0], pos[i][1] - cen[a][1], pos[i][2] - cen[a][2]];
        let s = 1.0 / rho[a];
        if k == 3 {
            t.push((3 * i, 3 * a, 1.0));
            t.push((3 * i + 1, 3 * a + 1, 1.0));
            t.push((3 * i, 3 * a + 2, -r[1] * s));
            t.push((3 * i + 1, 3 * a + 2, r[0] * s));
            t.push((3 * i + 2, 3 * a + 2, s));
        } else {
            for d in 0..3 {
                t.push((6 * i + d, 6 * a + d, 1.0));
            }
            for m in 0..3 {
                // u = e_m × r, θ = e_m
                let mut e = [0.0; 3];
                e[m] = 1.0;
                let u = [e[1] * r[2] - e[2] * r[1], e[2] * r[0] - e[0] * r[2], e[0] * r[1] - e[1] * r[0]];
                for d in 0..3 {
                    if u[d] != 0.0 {
                        t.push((6 * i + d, 6 * a + 3 + m, u[d] * s));
                    }
                }
                t.push((6 * i + 3 + m, 6 * a + 3 + m, s));
            }
        }
    }
    Csr::from_triplets(k * nb, kc * agg.n, t)
}

/// Smoothed aggregation: P = (I − ω D⁻¹A) P_t.
pub fn smooth_p(sys: &System, pt: &Csr, omega: f64) -> Csr {
    let ap = sys.a.matmul(pt);
    let k = sys.k;
    let mut t = pt.triplets();
    // D⁻¹ applied block-row-wise to AP: gather the k rows of block i per column
    for i in 0..sys.nb {
        let mut cols: std::collections::BTreeMap<usize, Vec<f64>> = Default::default();
        for r in 0..k {
            let row = k * i + r;
            for p in ap.ptr[row]..ap.ptr[row + 1] {
                cols.entry(ap.idx[p]).or_insert_with(|| vec![0.0; k])[r] += ap.val[p];
            }
        }
        for (c, mut v) in cols {
            lsolve(&sys.dl[i], k, &mut v);
            ltsolve(&sys.dl[i], k, &mut v);
            for r in 0..k {
                t.push((k * i + r, c, -omega * v[r]));
            }
        }
    }
    Csr::from_triplets(pt.nr, pt.nc, t)
}

#[derive(Clone, Debug, Default)]
pub struct CoarseStats {
    pub nc: usize,
    pub nnz_p: usize,
    pub nnz_ac: usize,
    pub nnz_l: usize,
    pub flops_factor: f64,
    pub t_ac: f64,
    pub t_symbolic: f64,
    pub t_numeric: f64,
}

/// Coarse level: P, Pᵀ and the Cholesky factor of A_c = PᵀAP.
pub struct Coarse {
    pub p: Csr,
    pub pt: Csr,
    pub ac: Csr,
    pub spd: Spd,
    pub stats: CoarseStats,
}

impl Coarse {
    pub fn new(sys: &System, p: Csr) -> Result<Coarse, String> {
        let t0 = Instant::now();
        let pt = p.transpose();
        let ac = pt.matmul(&sys.a.matmul(&p));
        let t_ac = t0.elapsed().as_secs_f64();
        let m = csc_from_triplets(ac.nr, ac.triplets());
        let t1 = Instant::now();
        let mut spd = Spd::analyze(&m);
        let t_symbolic = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        spd.factor(&m)?;
        let t_numeric = t2.elapsed().as_secs_f64();
        let stats = CoarseStats { nc: ac.nr, nnz_p: p.nnz(), nnz_ac: ac.nnz(), nnz_l: spd.stats.nnz_l, flops_factor: spd.stats.flops_factor, t_ac, t_symbolic, t_numeric };
        Ok(Coarse { p, pt, ac, spd, stats })
    }

    /// New fine operator, same P: re-assemble A_c and refactor numerically (symbolic
    /// analysis reused — the pattern is structural).
    pub fn update(&mut self, sys: &System) -> Result<(), String> {
        let t0 = Instant::now();
        self.ac = self.pt.matmul(&sys.a.matmul(&self.p));
        self.stats.t_ac = t0.elapsed().as_secs_f64();
        let m = csc_from_triplets(self.ac.nr, self.ac.triplets());
        let t2 = Instant::now();
        self.spd.factor(&m)?;
        self.stats.t_numeric = t2.elapsed().as_secs_f64();
        self.stats.t_symbolic = 0.0;
        Ok(())
    }

    /// A_c += Σ Δk_c g_c g_cᵀ with g_c = Pᵀ j_c for fixed P (exact for the Galerkin
    /// operator of the changed fine stiffness). Entries are updated in place inside the
    /// structural pattern of A_c. Returns the number of scalar multiply-adds. The factor is
    /// not touched: call `refactor`.
    pub fn add_lowrank(&mut self, changes: &[(&[(usize, f64)], f64)]) -> f64 {
        let mut madds = 0.0;
        let mut g: Vec<(usize, f64)> = Vec::new();
        for &(j, dk) in changes {
            if dk == 0.0 {
                continue;
            }
            g.clear();
            for &(d, v) in j {
                for q in self.p.ptr[d]..self.p.ptr[d + 1] {
                    let c = self.p.idx[q];
                    match g.iter_mut().find(|x| x.0 == c) {
                        Some(x) => x.1 += v * self.p.val[q],
                        None => g.push((c, v * self.p.val[q])),
                    }
                }
            }
            for &(c1, g1) in &g {
                let row = &self.ac.idx[self.ac.ptr[c1]..self.ac.ptr[c1 + 1]];
                for &(c2, g2) in &g {
                    let pos = row.binary_search(&c2).expect("ΔA_c entry outside the pattern of A_c");
                    self.ac.val[self.ac.ptr[c1] + pos] += dk * g1 * g2;
                    madds += 1.0;
                }
            }
        }
        madds
    }

    /// A_c += s · X (same pattern assumed, e.g. Δμ · PᵀMP).
    pub fn add_scaled(&mut self, x: &Csr, s: f64) {
        for r in 0..x.nr {
            let row = &self.ac.idx[self.ac.ptr[r]..self.ac.ptr[r + 1]];
            for q in x.ptr[r]..x.ptr[r + 1] {
                let pos = row.binary_search(&x.idx[q]).expect("pattern");
                self.ac.val[self.ac.ptr[r] + pos] += s * x.val[q];
            }
        }
    }

    /// Numeric refactorisation of the current A_c (symbolic analysis reused).
    pub fn refactor(&mut self) -> Result<f64, String> {
        let t = Instant::now();
        let m = csc_from_triplets(self.ac.nr, self.ac.triplets());
        self.spd.factor(&m)?;
        Ok(t.elapsed().as_secs_f64())
    }

    /// u += P A_c⁻¹ Pᵀ r.
    pub fn correct(&mut self, r: &[f64], u: &mut [f64], w: &mut Work) {
        let mut rc = vec![0.0; self.pt.nr];
        self.pt.matvec(r, &mut rc);
        self.spd.solve(&mut rc);
        let mut e = vec![0.0; self.p.nr];
        self.p.matvec(&rc, &mut e);
        for i in 0..u.len() {
            u[i] += e[i];
        }
        w.transfers += 2.0;
        w.coarse_solves += 1.0;
        w.phases += 3.0;
    }
}

/// One symmetric two-level cycle on A x = b (x updated in place).
pub fn cycle(sys: &System, co: &mut Coarse, sm: Smoother, b: &[f64], x: &mut [f64], x_zero: bool, w: &mut Work) {
    let n = sys.n;
    match sm {
        Smoother::Gs => sys.gs_sweep(b, x, false, w),
        Smoother::Cheb { deg, lo, hi } => chebyshev(sys, b, x, deg, lo, hi, x_zero, w),
    }
    let mut r = vec![0.0; n];
    sys.residual(b, x, &mut r, w);
    co.correct(&r, x, w);
    match sm {
        Smoother::Gs => sys.gs_sweep(b, x, true, w),
        Smoother::Cheb { deg, lo, hi } => chebyshev(sys, b, x, deg, lo, hi, false, w),
    }
}

/// Preconditioner of PCG: one cycle from zero, or block Jacobi.
pub enum Precond<'a> {
    Jacobi,
    TwoLevel(&'a mut Coarse, Smoother),
    /// A two-level cycle built on another (older) operator: the cycle smooths and
    /// corrects with its own System and coarse level, while the outer Krylov method works
    /// with the current operator. It is a fixed SPD operator for the whole solve.
    Stale(&'a System, &'a mut Coarse, Smoother),
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// PCG from x0 with a fixed SPD preconditioner (it must not change during one solve).
/// Stops after `max_iter` iterations, when ‖r‖ ≤ rel_tol·‖r₀‖ (if given; every norm is a
/// counted reduction), or when `after(iter, x, work)` returns true. No preconditioner is
/// applied after the final iteration. Returns (iterations, final ‖r‖/‖r₀‖ if computed).
pub fn pcg(sys: &System, pre: &mut Precond, b: &[f64], x: &mut [f64], max_iter: usize, rel_tol: Option<f64>, w: &mut Work, after: &mut dyn FnMut(usize, &[f64], &Work) -> bool) -> (usize, f64) {
    let n = sys.n;
    let mut r = vec![0.0; n];
    sys.residual(b, x, &mut r, w);
    w.vec_ops += 1.0;
    let apply_pre = |r: &[f64], z: &mut [f64], w: &mut Work, pre: &mut Precond| match pre {
        Precond::Jacobi => sys.dinv(r, z, w),
        Precond::TwoLevel(co, sm) => {
            z.iter_mut().for_each(|v| *v = 0.0);
            cycle(sys, co, *sm, r, z, true, w);
        }
        Precond::Stale(bsys, co, sm) => {
            z.iter_mut().for_each(|v| *v = 0.0);
            cycle(bsys, co, *sm, r, z, true, w);
        }
    };
    let reduce = |w: &mut Work| {
        w.reductions += 1.0;
        w.phases += 1.0;
    };
    let r0 = if rel_tol.is_some() {
        reduce(w);
        dot(&r, &r).sqrt()
    } else {
        f64::NAN
    };
    if let Some(t) = rel_tol {
        if r0 == 0.0 || r0 <= t * r0 {
            return (0, 0.0);
        }
    }
    let mut z = vec![0.0; n];
    apply_pre(&r, &mut z, w, pre);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    reduce(w);
    let mut ap = vec![0.0; n];
    let mut rel = f64::NAN;
    let mut done = 0;
    for it in 1..=max_iter {
        done = it;
        sys.apply(&p, &mut ap, w);
        let pap = dot(&p, &ap);
        reduce(w);
        if !(pap > 0.0) {
            break;
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        w.vec_ops += 2.0;
        if after(it, x, w) {
            break;
        }
        if let Some(t) = rel_tol {
            reduce(w);
            rel = dot(&r, &r).sqrt() / r0;
            if rel <= t {
                break;
            }
        }
        if it == max_iter {
            break;
        }
        apply_pre(&r, &mut z, w, pre);
        let rz_new = dot(&r, &z);
        reduce(w);
        if rz_new == 0.0 {
            break;
        }
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        w.vec_ops += 1.0;
    }
    (done, rel)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D chain of 2-DOF blocks: translations form a Laplacian clamped at the left end
    /// (free right end); the second DOF of each block is coupled to its translation.
    fn chain(nb: usize) -> System {
        let mut t = Vec::new();
        for i in 0..nb {
            t.push((2 * i, 2 * i, if i + 1 == nb { 1.0 } else { 2.0 }));
            if i + 1 < nb {
                t.push((2 * i, 2 * i + 2, -1.0));
                t.push((2 * i + 2, 2 * i, -1.0));
            }
            // spring ½k(u_t − u_s)² plus a ground spring on u_s: PSD by construction
            t.push((2 * i, 2 * i, 0.5));
            t.push((2 * i + 1, 2 * i + 1, 0.5 + 3.0));
            t.push((2 * i, 2 * i + 1, -0.5));
            t.push((2 * i + 1, 2 * i, -0.5));
        }
        System::new(Csr::from_triplets(2 * nb, 2 * nb, t), 2)
    }

    #[test]
    fn csr_products() {
        let a = Csr::from_triplets(2, 3, vec![(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (0, 0, 1.0)]);
        let b = Csr::from_triplets(3, 2, vec![(0, 0, 1.0), (1, 1, 1.0), (2, 0, 4.0)]);
        let c = a.matmul(&b);
        assert_eq!(c.triplets(), vec![(0, 0, 10.0), (1, 1, 3.0)]);
        assert_eq!(a.transpose().transpose().triplets(), a.triplets());
    }

    #[test]
    fn gs_and_pcg_converge() {
        let sys = chain(30);
        let b: Vec<f64> = (0..sys.n).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut x = vec![0.0; sys.n];
        let mut w = Work::default();
        let mut hist = Vec::new();
        pcg(&sys, &mut Precond::Jacobi, &b, &mut x, 200, Some(1e-12), &mut w, &mut |it, xx, _| {
            let mut r = vec![0.0; sys.n];
            sys.a.matvec(xx, &mut r);
            hist.push((it, r.iter().zip(&b).map(|(a, c)| (a - c).abs()).fold(0.0, f64::max)));
            false
        });
        let mut r = vec![0.0; sys.n];
        sys.residual(&b, &x, &mut r, &mut w);
        assert!(hist.len() <= 60, "{} iterations", hist.len());
        assert!(r.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-10);
        let lam = sys.lambda_dinv_a();
        assert!(lam > 0.5 && lam <= 2.0 + 1e-12, "{lam}");
        // two-level with 5 aggregates of the chain: symmetric, positive preconditioner,
        // exact on range(P)
        let pos: Vec<[f64; 3]> = (0..30).map(|i| [i as f64, 0.0, 0.0]).collect();
        let mass = vec![1.0; 30];
        let agg = Aggregates { of: (0..30).map(|i| i / 6).collect(), n: 5, sizes: vec![6; 5], max_detour: 1.0, reseeds: 0, merged: 0 };
        // 2-DOF blocks: column 0 = translation, column 1 = the coupled DOF
        let mut t = Vec::new();
        for i in 0..30 {
            t.push((2 * i, 2 * agg.of[i], 1.0));
            t.push((2 * i + 1, 2 * agg.of[i] + 1, 1.0));
        }
        let _ = (&pos, &mass);
        let mut co = Coarse::new(&sys, Csr::from_triplets(60, 10, t)).unwrap();
        let mut ww = Work::default();
        let z: Vec<f64> = (0..10).map(|i| (i as f64).cos()).collect();
        let mut e = vec![0.0; 60];
        co.p.matvec(&z, &mut e);
        let mut ae = vec![0.0; 60];
        sys.a.matvec(&e, &mut ae);
        let mut u = vec![0.0; 60];
        co.correct(&ae, &mut u, &mut ww);
        assert!(u.iter().zip(&e).map(|(a, c)| (a - c).abs()).fold(0.0, f64::max) < 1e-12, "coarse correction not exact on range(P)");
        let apply = |r: &[f64], co: &mut Coarse| {
            let mut x = vec![0.0; 60];
            let mut w = Work::default();
            cycle(&sys, co, Smoother::Gs, r, &mut x, true, &mut w);
            x
        };
        let x1: Vec<f64> = (0..60).map(|i| (i as f64 * 0.7).sin()).collect();
        let y1: Vec<f64> = (0..60).map(|i| (i as f64 * 1.3).cos()).collect();
        let (cx, cy) = (apply(&x1, &mut co), apply(&y1, &mut co));
        let (a1, a2) = (dot(&y1, &cx), dot(&x1, &cy));
        assert!((a1 - a2).abs() < 1e-12 * a1.abs().max(1.0), "cycle not symmetric: {a1} {a2}");
        assert!(dot(&x1, &cx) > 0.0);
    }
}
