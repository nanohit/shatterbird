//! Block-Jacobi preconditioned CG on the linear reference stiffness (+ optional
//! mass/h² term): a "global Krylov" reference curve for E3, not a candidate core.

use crate::assembly::{assemble, DofMap, Inertia, Tangent};
use crate::s1::solve3;
use model::Model;

pub struct Pcg {
    pub map: DofMap,
    rows: Vec<Vec<(usize, f64)>>,
    blocks: Vec<([usize; 3], [[f64; 3]; 3])>,
    pub x: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    rz: f64,
    pub iters: u64,
}

impl Pcg {
    /// Linearise at m.q (reference); solve H Δ = −∇Φ(q_ref).
    pub fn new(m: &Model, lf: f64, inertia: Option<&Inertia>) -> Self {
        let map = DofMap::new(m);
        let a = assemble(m, &m.q, &map, lf, inertia, true, Tangent::Secant);
        let mut rows = vec![Vec::new(); map.n];
        for (r, c, v) in a.trip {
            rows[r].push((c, v));
        }
        for row in rows.iter_mut() {
            row.sort_by_key(|e| e.0);
            let mut merged: Vec<(usize, f64)> = Vec::new();
            for &(c, v) in row.iter() {
                match merged.last_mut() {
                    Some(l) if l.0 == c => l.1 += v,
                    _ => merged.push((c, v)),
                }
            }
            *row = merged;
        }
        let mut blocks = Vec::new();
        for i in 0..m.bodies.len() {
            let idx = map.idx[i];
            if idx.iter().all(|x| x.is_some()) {
                let g = [idx[0].unwrap(), idx[1].unwrap(), idx[2].unwrap()];
                let mut h = [[0.0; 3]; 3];
                for r in 0..3 {
                    for &(c, v) in &rows[g[r]] {
                        if let Some(k) = g.iter().position(|&x| x == c) {
                            h[r][k] = v;
                        }
                    }
                }
                blocks.push((g, h));
            }
        }
        let b: Vec<f64> = a.grad.iter().map(|x| -x).collect();
        let n = map.n;
        let mut s = Pcg { map, rows, blocks, x: vec![0.0; n], r: b, z: vec![0.0; n], p: vec![0.0; n], rz: 0.0, iters: 0 };
        s.precond();
        s.p = s.z.clone();
        s.rz = dot(&s.r, &s.z);
        s
    }

    fn precond(&mut self) {
        for (g, h) in &self.blocks {
            let rr = [self.r[g[0]], self.r[g[1]], self.r[g[2]]];
            let z = solve3(*h, rr).unwrap_or(rr);
            for k in 0..3 {
                self.z[g[k]] = z[k];
            }
        }
    }

    pub fn step(&mut self) {
        let n = self.map.n;
        let mut ap = vec![0.0; n];
        for i in 0..n {
            ap[i] = self.rows[i].iter().map(|&(c, v)| v * self.p[c]).sum();
        }
        let alpha = self.rz / dot(&self.p, &ap);
        for i in 0..n {
            self.x[i] += alpha * self.p[i];
            self.r[i] -= alpha * ap[i];
        }
        self.precond();
        let rz_new = dot(&self.r, &self.z);
        let beta = rz_new / self.rz;
        self.rz = rz_new;
        for i in 0..n {
            self.p[i] = self.z[i] + beta * self.p[i];
        }
        self.iters += 1;
    }

    /// Configuration q_ref + x.
    pub fn q(&self, m: &Model) -> Vec<[f64; 3]> {
        let mut q = m.q.clone();
        crate::assembly::axpy_q(&mut q, &self.map, 1.0, &self.x);
        q
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
