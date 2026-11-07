//! Thin wrapper over faer's sparse Cholesky (SPD) and LU (indefinite/bordered),
//! exposing the work counts S2 must report: symbolic analyses, numeric
//! factorizations, triangular solves, nnz(L) and a flop estimate.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::prelude::*;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LltRef, SymbolicCholesky, SymbolicCholeskyRaw,
    SymmetricOrdering,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Conj, Par, Side};

#[derive(Clone, Copy, Debug, Default)]
pub struct FactorStats {
    pub symbolic: u64,
    pub numeric: u64,
    pub solves: u64,
    pub nnz_l: usize,
    pub flops_factor: f64,
}

/// Sum duplicate triplets (faer requires no duplicates in some paths; we make it explicit).
pub fn csc_from_triplets(n: usize, mut t: Vec<(usize, usize, f64)>) -> SparseColMat<usize, f64> {
    t.sort_unstable_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
    let mut merged: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(t.len());
    for (r, c, v) in t {
        if let Some(last) = merged.last_mut() {
            if last.row == r && last.col == c {
                last.val += v;
                continue;
            }
        }
        merged.push(Triplet::new(r, c, v));
    }
    SparseColMat::try_new_from_triplets(n, n, &merged).expect("triplets")
}

pub struct Spd {
    pub n: usize,
    sym: SymbolicCholesky<usize>,
    vals: Vec<f64>,
    pub stats: FactorStats,
    factored: bool,
}

impl Spd {
    pub fn analyze(a: &SparseColMat<usize, f64>) -> Self {
        let sym = factorize_symbolic_cholesky(
            a.symbolic(),
            Side::Lower,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .expect("symbolic cholesky");
        let (nnz_l, flops) = l_stats(&sym);
        let n = a.nrows();
        let len = sym.len_val();
        Spd {
            n,
            sym,
            vals: vec![0.0; len],
            stats: FactorStats { symbolic: 1, numeric: 0, solves: 0, nnz_l, flops_factor: flops },
            factored: false,
        }
    }

    /// Numeric factorization reusing the symbolic analysis (same pattern required).
    pub fn factor(&mut self, a: &SparseColMat<usize, f64>) -> Result<(), String> {
        let par = Par::Seq;
        let mut mem = MemBuffer::new(self.sym.factorize_numeric_llt_scratch::<f64>(par, Default::default()));
        let stack = MemStack::new(&mut mem);
        self.sym
            .factorize_numeric_llt(
                &mut self.vals,
                a.as_ref(),
                Side::Lower,
                LltRegularization::default(),
                par,
                stack,
                Default::default(),
            )
            .map_err(|e| format!("{e:?}"))?;
        self.stats.numeric += 1;
        self.factored = true;
        Ok(())
    }

    pub fn solve(&mut self, rhs: &mut [f64]) {
        assert!(self.factored);
        let par = Par::Seq;
        let llt = LltRef::new(&self.sym, &self.vals);
        let mut mem = MemBuffer::new(self.sym.solve_in_place_scratch::<f64>(1, par));
        let stack = MemStack::new(&mut mem);
        let mut m = Mat::<f64>::from_fn(self.n, 1, |i, _| rhs[i]);
        llt.solve_in_place_with_conj(Conj::No, m.as_mut(), par, stack);
        for i in 0..self.n {
            rhs[i] = m[(i, 0)];
        }
        self.stats.solves += 1;
    }
}

fn l_stats(sym: &SymbolicCholesky<usize>) -> (usize, f64) {
    match sym.raw() {
        SymbolicCholeskyRaw::Simplicial(s) => {
            let cp = s.col_ptr();
            let mut nnz = 0usize;
            let mut fl = 0.0;
            for j in 0..s.ncols() {
                let c = (cp[j + 1] - cp[j]) as f64 + 1.0;
                nnz += c as usize;
                fl += c * c;
            }
            (nnz, fl)
        }
        SymbolicCholeskyRaw::Supernodal(s) => {
            let b = s.supernode_begin();
            let e = s.supernode_end();
            let cp = s.col_ptr_for_row_idx();
            let mut nnz = 0usize;
            let mut fl = 0.0;
            for k in 0..s.n_supernodes() {
                let c = e[k] - b[k];
                let r = cp[k + 1] - cp[k];
                for j in 0..c {
                    let cnt = (c - j + r) as f64;
                    nnz += cnt as usize;
                    fl += cnt * cnt;
                }
            }
            (nnz, fl)
        }
    }
}

/// One-shot sparse LU solve (indefinite or unsymmetric systems).
pub fn lu_solve(a: &SparseColMat<usize, f64>, rhs: &[f64]) -> Result<Vec<f64>, String> {
    let lu = a.sp_lu().map_err(|e| format!("{e:?}"))?;
    let b = Mat::<f64>::from_fn(rhs.len(), 1, |i, _| rhs[i]);
    let x = lu.solve(&b);
    Ok((0..rhs.len()).map(|i| x[(i, 0)]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_and_lu_agree() {
        // 1D Laplacian + identity, with duplicate triplets
        let n = 50;
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i, 1.0));
            t.push((i, i, 1.0));
            if i + 1 < n {
                t.push((i, i, 1.0));
                t.push((i + 1, i + 1, 1.0));
                t.push((i, i + 1, -1.0));
                t.push((i + 1, i, -1.0));
            }
        }
        let a = csc_from_triplets(n, t);
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let mut x = rhs.clone();
        let mut s = Spd::analyze(&a);
        s.factor(&a).unwrap();
        s.solve(&mut x);
        let y = lu_solve(&a, &rhs).unwrap();
        for i in 0..n {
            assert!((x[i] - y[i]).abs() < 1e-12);
        }
        // residual
        let mut r = 0.0f64;
        for i in 0..n {
            let mut ax = 2.0 * x[i];
            if i > 0 {
                ax += x[i] - x[i - 1];
            }
            if i + 1 < n {
                ax += x[i] - x[i + 1];
            }
            r = r.max((ax - rhs[i]).abs());
        }
        assert!(r < 1e-12, "{r}");
        assert!(s.stats.nnz_l >= n);
    }
}
