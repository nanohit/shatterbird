//! Phase 0b / 3 §6.1 — export of rbsn3 contract meshes for the WebGPU probe.
//!
//! Cubes of side 2.25 / 4.85 / 7.0 m at d_min 0.2 (constrained RSA, generator–support
//! distance ≥ 0.5 d_min, clamped bottom) → ≈ 1 k / 10 k / 30 k cells. One little-endian
//! binary per mesh (layout in `write_mesh`), plus a JSON index. The probe page reads
//! these files, runs the kernels and checks against its own f64 CPU reference.

use experiments::{meta, write_json};
use faer::linalg::solvers::DenseSolveCore;
use geometry::rng::Rng;
use geometry::rsa::rsa_box_with;
use geometry::voronoi3d::voronoi_box_grid;
use geometry::V3;
use model::rbsn3::{Mat3D, Model3};
use serde_json::json;
use solvers::twolevel::{aggregate, AggInput};
use std::io::Write;

const DMIN: f64 = 0.2;
const MU: f64 = 2.25 * 480.0 * 480.0;

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(b: &mut Vec<u8>, v: f64) {
    b.extend_from_slice(&(v as f32).to_le_bytes());
}

fn inv3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a = faer::Mat::<f64>::from_fn(3, 3, |i, j| m[i][j]);
    let inv = a.full_piv_lu().inverse();
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = inv[(i, j)];
        }
    }
    r
}

/// Layout: header [magic "SBM1", ncells, nfacets, nagg, ncsr] (u32);
/// cells 25 f32 (centroid 3, mass, I 9, I⁻¹ 9, v0 3);
/// facet index pairs 2 u32 (a, b; b = 0xFFFFFFFF for ground);
/// facet data 20 f32 (n 3, p1 3, p2 3, r_a 3, r_b 3, k_n, k_t, k_r1, k_r2, k_tor);
/// cell→facet CSR: ptr (ncells+1 u32), entries u32 (facet | side_b << 31);
/// D⁻¹ blocks 36 f32 per cell (A = μM + K, μ = 2.25/h², h = 1/480);
/// aggregates: agg id per cell (u32), per aggregate centre 3 + ρ 1 (f32), CSR ptr
/// (nagg+1) and cell list.
fn write_mesh(path: &std::path::Path, m: &Model3, push_z: f64) -> serde_json::Value {
    let nc = m.cells.len();
    let nf = m.facets.len();
    // per-facet data recovered from the 6 components (order n, p1, p2, rot p1, rot p2, rot n)
    let mut fdata = vec![[0.0f64; 20]; nf];
    let mut seen = vec![0usize; nf];
    for c in &m.comps {
        let f = c.facet;
        let k = seen[f];
        seen[f] += 1;
        let d = [-c.ja[0], -c.ja[1], -c.ja[2]];
        match k {
            0 => fdata[f][0..3].copy_from_slice(&d),
            1 => fdata[f][3..6].copy_from_slice(&d),
            2 => fdata[f][6..9].copy_from_slice(&d),
            _ => {}
        }
        // k_n, k_t (shared by p1 and p2), k_r1, k_r2, k_tor
        match k {
            0 => fdata[f][15] = c.k,
            1 => fdata[f][16] = c.k,
            2 => assert_eq!(fdata[f][16], c.k),
            _ => fdata[f][14 + k] = c.k,
        }
    }
    for (fi, fc) in m.facets.iter().enumerate() {
        let ra = fc.centroid - m.cells[fc.a].centroid;
        let rb = fc.b.map(|b| fc.centroid - m.cells[b].centroid).unwrap_or(V3::ZERO);
        fdata[fi][9..12].copy_from_slice(&[ra.x, ra.y, ra.z]);
        fdata[fi][12..15].copy_from_slice(&[rb.x, rb.y, rb.z]);
    }
    // cell→facet lists
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nc];
    for (fi, fc) in m.facets.iter().enumerate() {
        lists[fc.a].push(fi as u32);
        if let Some(b) = fc.b {
            lists[b].push(fi as u32 | (1 << 31));
        }
    }
    let ncsr: usize = lists.iter().map(|l| l.len()).sum();
    // D⁻¹: diagonal 6×6 blocks of μM + K
    let mut dblk = vec![[[0.0f64; 6]; 6]; nc];
    for (i, c) in m.cells.iter().enumerate() {
        for d in 0..3 {
            dblk[i][d][d] += MU * c.mass;
            for e in 0..3 {
                dblk[i][3 + d][3 + e] += MU * c.inertia[d][e];
            }
        }
    }
    for c in &m.comps {
        for (cell, j) in [(Some(c.a), c.ja), (c.b, c.jb)] {
            if let Some(i) = cell {
                for p in 0..6 {
                    for q in 0..6 {
                        dblk[i][p][q] += c.k * j[p] * j[q];
                    }
                }
            }
        }
    }
    // aggregates of 8 (intact graph)
    let pos: Vec<[f64; 3]> = m.cells.iter().map(|c| [c.centroid.x, c.centroid.y, c.centroid.z]).collect();
    let masses: Vec<f64> = m.cells.iter().map(|c| c.mass).collect();
    let edges: Vec<(usize, usize, bool)> = m.facets.iter().filter_map(|f| f.b.map(|b| (f.a, b, true))).collect();
    let vol: f64 = m.cells.iter().map(|c| c.volume).sum();
    let ell = (vol / nc as f64).cbrt();
    let agg = aggregate(&AggInput { pos: &pos, mass: &masses, edges: &edges, target: 8, dim: 3, ell });
    let mut cen = vec![[0.0f64; 3]; agg.n];
    let mut msum = vec![0.0; agg.n];
    for i in 0..nc {
        for d in 0..3 {
            cen[agg.of[i]][d] += masses[i] * pos[i][d];
        }
        msum[agg.of[i]] += masses[i];
    }
    for a in 0..agg.n {
        for d in 0..3 {
            cen[a][d] /= msum[a];
        }
    }
    let mut rho = vec![0.0f64; agg.n];
    for i in 0..nc {
        let a = agg.of[i];
        rho[a] += masses[i] * ((pos[i][0] - cen[a][0]).powi(2) + (pos[i][1] - cen[a][1]).powi(2) + (pos[i][2] - cen[a][2]).powi(2));
    }
    for a in 0..agg.n {
        rho[a] = (rho[a] / msum[a]).sqrt().max(0.5 * ell);
    }
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); agg.n];
    for i in 0..nc {
        members[agg.of[i]].push(i as u32);
    }
    // ---- write
    let mut b: Vec<u8> = Vec::new();
    for v in [0x3153_4253u32, nc as u32, nf as u32, agg.n as u32, ncsr as u32] {
        put_u32(&mut b, v);
    }
    for c in &m.cells {
        let inv = inv3(c.inertia);
        for v in [c.centroid.x, c.centroid.y, c.centroid.z, c.mass] {
            put_f32(&mut b, v);
        }
        for r in c.inertia {
            for v in r {
                put_f32(&mut b, v);
            }
        }
        for r in inv {
            for v in r {
                put_f32(&mut b, v);
            }
        }
        let vx = if c.centroid.z > push_z { 1.0 } else { 0.0 };
        for v in [vx, 0.0, 0.0] {
            put_f32(&mut b, v);
        }
    }
    for f in &m.facets {
        put_u32(&mut b, f.a as u32);
        put_u32(&mut b, f.b.map(|x| x as u32).unwrap_or(u32::MAX));
    }
    for d in &fdata {
        for &v in d {
            put_f32(&mut b, v);
        }
    }
    let mut ptr = 0u32;
    put_u32(&mut b, 0);
    for l in &lists {
        ptr += l.len() as u32;
        put_u32(&mut b, ptr);
    }
    for l in &lists {
        for &e in l {
            put_u32(&mut b, e);
        }
    }
    for blk in &dblk {
        let a = faer::Mat::<f64>::from_fn(6, 6, |i, j| blk[i][j]);
        let inv = a.full_piv_lu().inverse();
        for i in 0..6 {
            for j in 0..6 {
                put_f32(&mut b, inv[(i, j)]);
            }
        }
    }
    for i in 0..nc {
        put_u32(&mut b, agg.of[i] as u32);
    }
    for a in 0..agg.n {
        for v in [cen[a][0], cen[a][1], cen[a][2], rho[a]] {
            put_f32(&mut b, v);
        }
    }
    let mut ap = 0u32;
    put_u32(&mut b, 0);
    for l in &members {
        ap += l.len() as u32;
        put_u32(&mut b, ap);
    }
    for l in &members {
        for &e in l {
            put_u32(&mut b, e);
        }
    }
    std::fs::File::create(path).unwrap().write_all(&b).unwrap();
    json!({"cells": nc, "facets": nf, "support_facets": m.facets.iter().filter(|f| f.b.is_none()).count(), "csr_entries": ncsr, "aggregates": agg.n, "bytes": b.len()})
}

fn main() {
    let audit = std::env::args().any(|s| s == "--phase0e");
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("output directory"));
    std::fs::create_dir_all(&dir).unwrap();
    let mut index = Vec::new();
    for (label, side) in [("1k", 2.25), ("10k", 4.85), ("30k", 7.0)] {
        if audit && label != "1k" { continue; }
        let t0 = std::time::Instant::now();
        let hi = V3::new(side, side, side);
        let keep = |p: V3| p.z >= 0.5 * DMIN;
        let seeds = rsa_box_with(V3::ZERO, hi, DMIN, &mut Rng::new(1), 5000, &keep, Vec::new());
        let cells3 = voronoi_box_grid(&seeds, V3::ZERO, hi, DMIN);
        let m = Model3::build(&cells3, &[4], Mat3D::concrete());
        let info = write_mesh(&dir.join(format!("mesh_{label}.bin")), &m, side - 0.25);
        eprintln!("{label}: {info} in {:.1} s", t0.elapsed().as_secs_f64());
        index.push(json!({"label": label, "side_m": side, "file": format!("mesh_{label}.bin"), "info": info}));
    }
    let c = Mat3D::concrete().wave_speed();
    let doc = json!({"meta": meta("phase0b/architecture-selection/gpu", "export", Some(1), "rbsn3 contract meshes for the WebGPU probe", json!({"dmin": DMIN, "mu": MU})),
        "dt_contract_s": 0.38 * DMIN / c, "meshes": index});
    std::fs::write(dir.join("index.json"), serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    write_json(if audit { "phase0e/gpu" } else { "phase0b/architecture-selection/gpu" }, "export", &doc);
}
