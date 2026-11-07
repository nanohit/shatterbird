//! Phase 0b / 1 — geometry contract and spectrum (shatterbird-phase0b.md §4,
//! pre-registration in docs/phase0b/geometry-spectrum.md §3).
//!
//! For each configuration: top eigenpairs of B = M^{-1/2}KM^{-1/2} (Lanczos, two start
//! vectors, dense check), certified upper bounds, step sizes, localisation of the top
//! modes against per-cell geometric features, and mechanics checks after geometry
//! changes (coverage, mass moments, clamped uniaxial patch test, compliance, low
//! frequencies). Variants: control, supports removed (diagnostic), support exclusion
//! α·d_min by repair; then the contract is fixed and tested on held-out 3D seeds.

#![recursion_limit = "512"]

use experiments::{edge_traction, meta, write_json, Csv};
use geometry::polygon::{closest_on_segment, poly_props};
use geometry::rng::Rng;
use geometry::rsa::{rsa_box, rsa_box_with, rsa_rect, rsa_rect_with};
use geometry::voronoi2d::{rect, voronoi_convex, Tessellation};
use geometry::voronoi3d::{voronoi_box, FaceTag, VCell3};
use geometry::{V2, V3};
use model::rbsn3::{Mat3D, Model3};
use model::structure::{Facet, FacetPoint};
use model::{BodyKind, BoundaryAttach, LawKind, LawOptions, Material, Model, PointLaw};
use serde_json::{json, Value};
use solvers::sparse::{csc_from_triplets, Spd};
use solvers::spectrum::{CompClass, Pairs, Spectral};

const EXP: &str = "phase0b/geometry-spectrum";
const ALPHAS: [f64; 3] = [0.25, 0.35, 0.5];
const NEV: usize = 10;
const LANCZOS_TOL: f64 = 1e-10;
const MAX_STEPS: usize = 800;
const DENSE_MAX: usize = 3000;
const SAFETY: f64 = 0.8;
const HELD_OUT_SEEDS: [u64; 3] = [1001, 1002, 1003];
const GAME_SIZES: [f64; 3] = [0.2, 0.35, 0.5];
const R_PROJ: f64 = 0.05;
// post-hoc only
const PH_ALPHAS: [f64; 2] = [0.6, 0.75];
const PH_SEEDS: [u64; 3] = [2001, 2002, 2003];

// ---------------------------------------------------------------- per-cell features

/// Geometric features of one block, lengths in units of d_min.
#[derive(Clone, Debug, Default)]
struct Feat {
    body: usize,
    cell: bool,
    support: bool,
    boundary: bool,
    /// Generator–support distance (NaN if no support facet).
    d_sup: f64,
    vol: f64,
    /// Minimum principal rotational inertia / (m d_min²).
    jmin: f64,
    elong: f64,
    /// Nearest neighbour generator distance.
    nn: f64,
    max_face: f64,
    max_arm: f64,
    valence: usize,
    /// Σ k_n of support/contact facets over Σ k_n of all facets of the cell.
    ext_share: f64,
}

fn eig2_ratio(a: f64, b: f64, c: f64) -> f64 {
    // eigenvalues of [[a, c], [c, b]]
    let m = 0.5 * (a + b);
    let r = (0.25 * (a - b).powi(2) + c * c).sqrt();
    ((m + r) / (m - r)).sqrt()
}

/// 2D second moments ∫x², ∫y², ∫xy about the centroid of a polygon.
fn second_moments(v: &[V2]) -> (f64, f64, f64) {
    let pp = poly_props(v);
    let o = pp.centroid;
    let n = v.len();
    let (mut ixx, mut iyy, mut ixy) = (0.0, 0.0, 0.0);
    for k in 0..n {
        let p = v[k] - o;
        let q = v[(k + 1) % n] - o;
        let c = p.cross(q);
        ixx += (p.x * p.x + p.x * q.x + q.x * q.x) * c;
        iyy += (p.y * p.y + p.y * q.y + q.y * q.y) * c;
        ixy += (p.x * q.y + 2.0 * p.x * p.y + 2.0 * q.x * q.y + q.x * p.y) * c;
    }
    (ixx / 12.0, iyy / 12.0, ixy / 24.0)
}

fn feats2(m: &Model, sp: &Spectral, dmin: f64, lo: V2, hi: V2) -> Vec<Feat> {
    sp.blocks
        .iter()
        .map(|blk| {
            let i = blk.body;
            let b = &m.bodies[i];
            if b.kind != BodyKind::Cell {
                return Feat { body: i, d_sup: f64::NAN, nn: f64::NAN, ..Default::default() };
            }
            let mut f = Feat { body: i, cell: true, d_sup: f64::NAN, nn: f64::INFINITY, ..Default::default() };
            let (mut kext, mut kall) = (0.0, 0.0);
            for fa in m.facets.iter().filter(|fa| fa.active && (fa.a == i || fa.b == i)) {
                let other = if fa.a == i { fa.b } else { fa.a };
                let kn = fa.pts[0].law.kn;
                kall += kn;
                if m.bodies[other].kind == BodyKind::Cell {
                    f.nn = f.nn.min(fa.h / dmin);
                    f.valence += 1;
                } else {
                    kext += kn;
                    if !m.is_free(other) {
                        f.support = true;
                        f.d_sup = if f.d_sup.is_nan() { fa.h / dmin } else { f.d_sup.min(fa.h / dmin) };
                    }
                }
                f.max_face = f.max_face.max(fa.len / dmin);
                for p in &fa.pts {
                    let arm = if fa.a == i { p.arm_a } else { p.arm_b };
                    f.max_arm = f.max_arm.max(arm.norm() / dmin);
                }
            }
            f.ext_share = kext / kall;
            let pp = poly_props(&b.poly);
            f.vol = pp.area / (dmin * dmin);
            f.jmin = b.inertia / (b.mass * dmin * dmin);
            let (a, bb, c) = second_moments(&b.poly);
            f.elong = eig2_ratio(a, bb, c);
            let tol = 1e-9 * (hi - lo).norm();
            f.boundary = b.poly.iter().any(|p| (p.x - lo.x).abs() < tol || (p.x - hi.x).abs() < tol || (p.y - lo.y).abs() < tol || (p.y - hi.y).abs() < tol);
            f
        })
        .collect()
}

fn sym3_eigs(a: [[f64; 3]; 3]) -> Vec<f64> {
    let m = faer::Mat::<f64>::from_fn(3, 3, |i, j| a[i][j]);
    let mut e: Vec<f64> = m.self_adjoint_eigenvalues(faer::Side::Lower).unwrap().to_vec();
    e.sort_by(|x, y| x.partial_cmp(y).unwrap());
    e
}

fn feats3(m: &Model3, cells3: &[VCell3], dmin: f64) -> Vec<Feat> {
    let mut fs: Vec<Feat> = (0..m.cells.len()).map(|i| Feat { body: i, cell: true, d_sup: f64::NAN, nn: f64::INFINITY, ..Default::default() }).collect();
    let mut kext = vec![0.0; m.cells.len()];
    let mut kall = vec![0.0; m.cells.len()];
    for fa in &m.facets {
        let kn = m.mat.e * fa.area / fa.ell;
        for (i, other) in [(fa.a, fa.b), (fa.b.unwrap_or(usize::MAX), Some(fa.a))] {
            if i == usize::MAX {
                continue;
            }
            let f = &mut fs[i];
            kall[i] += kn;
            match other {
                Some(_) if fa.b.is_some() => {
                    f.nn = f.nn.min(fa.ell / dmin);
                    f.valence += 1;
                }
                _ => {
                    kext[i] += kn;
                    f.support = true;
                    f.d_sup = if f.d_sup.is_nan() { fa.ell / dmin } else { f.d_sup.min(fa.ell / dmin) };
                }
            }
            f.max_face = f.max_face.max(fa.area / (dmin * dmin));
            f.max_arm = f.max_arm.max((fa.centroid - m.cells[i].centroid).norm() / dmin);
        }
    }
    for (i, f) in fs.iter_mut().enumerate() {
        let c = &m.cells[i];
        f.ext_share = kext[i] / kall[i];
        f.vol = c.volume / dmin.powi(3);
        let ie = sym3_eigs(c.inertia);
        f.jmin = ie[0] / (c.mass * dmin * dmin);
        // second-moment tensor C = tr(I)/2·1 − I
        let tr = 0.5 * (c.inertia[0][0] + c.inertia[1][1] + c.inertia[2][2]);
        let mut cm = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                cm[a][b] = if a == b { tr } else { 0.0 } - c.inertia[a][b];
            }
        }
        let ce = sym3_eigs(cm);
        f.elong = (ce[2] / ce[0]).sqrt();
        f.boundary = cells3[i].poly.faces.iter().any(|fc| matches!(fc.tag, FaceTag::Boundary(_)));
    }
    fs
}

// ---------------------------------------------------------------- statistics

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let x = q * (sorted.len() - 1) as f64;
    let (i, t) = (x.floor() as usize, x - x.floor());
    if i + 1 < sorted.len() {
        sorted[i] * (1.0 - t) + sorted[i + 1] * t
    } else {
        sorted[i]
    }
}

fn stats(v: impl IntoIterator<Item = f64>) -> Value {
    let mut s: Vec<f64> = v.into_iter().filter(|x| x.is_finite()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let mean = if n > 0 { s.iter().sum::<f64>() / n as f64 } else { f64::NAN };
    json!({"n": n, "min": s.first(), "p05": quantile(&s, 0.05), "median": quantile(&s, 0.5), "p95": quantile(&s, 0.95), "max": s.last(), "mean": mean})
}

// ---------------------------------------------------------------- spectral analysis

struct Out {
    cases: Csv,
    cells: Csv,
    modes: Csv,
    polys: Csv,
}

struct Res {
    json: Value,
    theta1: f64,
    /// max L̂ over support cells / max L̂ over other cells.
    support_ratio: f64,
    top_rot: f64,
    top5: f64,
    u_hat: Vec<f64>,
    low_hz: Vec<f64>,
    compliance: f64,
}

fn hat(lam: f64, dmin: f64, c: f64) -> f64 {
    lam * dmin * dmin / (c * c)
}

fn dt(lam: f64) -> f64 {
    2.0 / lam.sqrt()
}

fn n_safe(dt_safe: f64) -> f64 {
    (1.0 / (60.0 * dt_safe)).ceil()
}

/// Spectral analysis of one configuration. `polys`: 2D cell polygons to dump with the
/// top-mode kinetic share (for figures).
fn analyse(case: &str, variant: &str, sp: &Spectral, feats: &[Feat], dmin: f64, c: f64, polys: Option<&Model>, out: &mut Out) -> Res {
    let t0 = std::time::Instant::now();
    let nev = NEV.min(sp.n);
    let runs: Vec<Pairs> = [11u64, 12].iter().map(|&s| sp.top(nev, s, LANCZOS_TOL, MAX_STEPS)).collect();
    let theta1 = runs.iter().map(|r| r.vals[0]).fold(0.0, f64::max);
    let start_rel = (runs[0].vals[0] - runs[1].vals[0]).abs() / theta1;
    let dense = (sp.n <= DENSE_MAX).then(|| sp.dense_eigs());
    let dense_rel = dense.as_ref().map(|d| (d[0] - theta1).abs() / d[0]);
    for (si, r) in runs.iter().enumerate() {
        for k in 0..r.vals.len() {
            let dv = dense.as_ref().map(|d| hat(d[k], dmin, c)).unwrap_or(f64::NAN);
            out.modes.row_s(&[case.into(), variant.into(), format!("{}", 11 + si), k.to_string(), format!("{:.12e}", hat(r.vals[k], dmin, c)), format!("{:.3e}", r.resid[k]), format!("{dv:.12e}"), r.steps.to_string(), r.matvecs.to_string()]);
        }
    }
    let b = sp.bounds();
    let syl = sp.sylvester(theta1, b.split, 1e-9);
    let lmax_block = b.l_cell.iter().cloned().fold(0.0, f64::max);
    // localisation of the top mode (and of the near-degenerate cluster around it)
    let p = &runs[0];
    let cluster: Vec<usize> = (0..p.vals.len()).filter(|&k| (p.vals[0] - p.vals[k]) / p.vals[0] < 1e-3).collect();
    let mut kin = vec![[0.0; 3]; sp.blocks.len()];
    let mut el = vec![0.0; sp.comps.len()];
    for &k in &cluster {
        for (a, s) in kin.iter_mut().zip(sp.kinetic_shares(&p.vecs[k])) {
            for j in 0..3 {
                a[j] += s[j] / cluster.len() as f64;
            }
        }
        for (a, s) in el.iter_mut().zip(sp.elastic_shares(&p.vecs[k])) {
            *a += s / cluster.len() as f64;
        }
    }
    let mut order: Vec<usize> = (0..kin.len()).collect();
    order.sort_by(|&x, &y| kin[y][0].partial_cmp(&kin[x][0]).unwrap());
    let topk = |k: usize| order.iter().take(k).map(|&i| kin[i][0]).sum::<f64>();
    let part = 1.0 / kin.iter().map(|s| s[0] * s[0]).sum::<f64>();
    let rot: f64 = kin.iter().map(|s| s[2]).sum();
    let mut class_share = [0.0; 3];
    let mut facet_share: std::collections::HashMap<(usize, usize), f64> = Default::default();
    for (ci, comp) in sp.comps.iter().enumerate() {
        let k = match comp.class {
            CompClass::Interior => 0,
            CompClass::Support => 1,
            CompClass::Contact => 2,
        };
        class_share[k] += el[ci];
        *facet_share.entry((k, comp.facet)).or_default() += el[ci];
    }
    let top_facet = facet_share.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(k, v)| (*k, *v)).unwrap();
    let ftop = &feats[order[0]];
    // support-cell outlier ratio of the per-cell lower bounds
    let lmax_of = |sup: bool| feats.iter().zip(&b.l_cell).filter(|(f, _)| f.cell && f.support == sup).map(|(_, l)| *l).fold(0.0, f64::max);
    let (ls, lo) = (lmax_of(true), lmax_of(false));
    let support_ratio = if ls > 0.0 { ls / lo } else { 0.0 };
    for (k, f) in feats.iter().enumerate() {
        out.cells.row_s(&[
            case.into(),
            variant.into(),
            f.body.to_string(),
            (f.cell as u8).to_string(),
            (f.support as u8).to_string(),
            (f.boundary as u8).to_string(),
            format!("{:.6e}", f.d_sup),
            format!("{:.6e}", f.vol),
            format!("{:.6e}", f.jmin),
            format!("{:.6e}", f.elong),
            format!("{:.6e}", f.nn),
            format!("{:.6e}", f.max_face),
            format!("{:.6e}", f.max_arm),
            f.valence.to_string(),
            format!("{:.6e}", f.ext_share),
            format!("{:.6e}", hat(b.l_cell[k], dmin, c)),
            format!("{:.6e}", hat(b.u_cell[k], dmin, c)),
            format!("{:.6e}", kin[k][0]),
            format!("{:.6e}", kin[k][2]),
        ]);
    }
    if let Some(m) = polys {
        for (k, blk) in sp.blocks.iter().enumerate() {
            let bd = &m.bodies[blk.body];
            if bd.kind == BodyKind::Cell {
                let v: Vec<String> = bd.poly.iter().map(|p| format!("{:.6} {:.6}", p.x, p.y)).collect();
                out.polys.row_s(&[case.into(), variant.into(), blk.body.to_string(), format!("{:.6e}", kin[k][0]), format!("{:.6e}", hat(b.l_cell[k], dmin, c)), v.join(";")]);
            }
        }
    }
    let resid_max = runs.iter().flat_map(|r| r.resid.iter()).cloned().fold(0.0, f64::max);
    let u_best = syl.upper;
    let dt_crit = dt(theta1);
    let dt_safe_split = SAFETY * dt(b.split);
    let dt_safe_syl = SAFETY * dt(u_best);
    out.cases.row_s(&[
        case.into(),
        variant.into(),
        sp.blocks.len().to_string(),
        sp.n.to_string(),
        format!("{:.6e}", hat(theta1, dmin, c)),
        format!("{:.6e}", dt_crit * c / dmin),
        format!("{:.6e}", dt_crit * 1e6),
        format!("{:.6e}", hat(lmax_block, dmin, c)),
        format!("{:.6e}", hat(b.split, dmin, c)),
        format!("{:.6e}", hat(b.split_t1, dmin, c)),
        format!("{:.6e}", hat(b.collatz_wielandt, dmin, c)),
        format!("{:.6e}", hat(b.gershgorin, dmin, c)),
        format!("{:.6e}", hat(u_best, dmin, c)),
        format!("{:.6e}", dt_safe_split * 1e6),
        format!("{:.6e}", dt_safe_syl * 1e6),
        format!("{}", n_safe(dt_safe_split)),
        format!("{}", n_safe(dt_safe_syl)),
        format!("{:.4}", topk(1)),
        format!("{:.4}", topk(5)),
        format!("{:.4}", rot),
        format!("{:.4}", class_share[1]),
        format!("{:.4e}", ftop.d_sup),
        format!("{:.4}", support_ratio),
        format!("{:.3e}", resid_max),
    ]);
    let tf_class = ["interior", "support", "contact"][top_facet.0 .0];
    let json = json!({
        "case": case, "variant": variant, "cells": feats.iter().filter(|f| f.cell).count(), "dofs": sp.n, "components": sp.comps.len(),
        "lanczos": runs.iter().zip([11, 12]).map(|(r, s)| json!({"start_seed": s, "steps": r.steps, "matvecs": r.matvecs,
            "lam_hat": r.vals.iter().map(|&v| hat(v, dmin, c)).collect::<Vec<_>>(), "resid": r.resid})).collect::<Vec<_>>(),
        "start_vectors_rel_diff": start_rel,
        "dense_top_lam_hat": dense.as_ref().map(|d| d.iter().take(nev).map(|&v| hat(v, dmin, c)).collect::<Vec<_>>()),
        "dense_rel_diff": dense_rel,
        "dense_zero_modes": dense.as_ref().map(|d| d.iter().filter(|&&x| x.abs() < 1e-10 * d[0]).count()),
        "theta1_hat": hat(theta1, dmin, c),
        "bounds_hat": {"block_lower": hat(lmax_block, dmin, c), "split_t1": hat(b.split_t1, dmin, c), "split": hat(b.split, dmin, c),
            "collatz_wielandt": hat(b.collatz_wielandt, dmin, c), "gershgorin": hat(b.gershgorin, dmin, c),
            "sylvester_lower": hat(syl.lower, dmin, c), "sylvester_upper": hat(syl.upper, dmin, c)},
        "sylvester": {"factorizations": syl.factorizations, "first_try": syl.first_try, "margin": solvers::spectrum::SYL_MARGIN},
        "gap_ratios": {"split_over_theta": b.split / theta1, "split_t1_over_theta": b.split_t1 / theta1, "cw_over_theta": b.collatz_wielandt / theta1,
            "gershgorin_over_theta": b.gershgorin / theta1, "theta_over_block_lower": theta1 / lmax_block},
        "dt_us": {"crit_ritz": dt_crit * 1e6, "block_upper": dt(lmax_block) * 1e6, "safe_split": dt_safe_split * 1e6, "safe_sylvester": dt_safe_syl * 1e6},
        "dt_hat": {"crit_ritz": dt_crit * c / dmin, "safe_split": dt_safe_split * c / dmin, "safe_sylvester": dt_safe_syl * c / dmin},
        "n_safe_per_60hz_frame": {"split": n_safe(dt_safe_split), "sylvester": n_safe(dt_safe_syl), "ritz_unsafe": n_safe(dt_crit)},
        "top_mode": {"cluster_size": cluster.len(), "kin_top1": topk(1), "kin_top5": topk(5), "kin_top10": topk(10), "participation_cells": part,
            "rotation_share": rot, "elastic_share": {"interior": class_share[0], "support": class_share[1], "contact": class_share[2]},
            "top_facet": {"class": tf_class, "share": top_facet.1},
            "top_cell": {"body": ftop.body, "cell": ftop.cell, "support": ftop.support, "boundary": ftop.boundary, "d_sup": ftop.d_sup, "vol": ftop.vol, "jmin": ftop.jmin,
                "elong": ftop.elong, "nn": ftop.nn, "max_face": ftop.max_face, "max_arm": ftop.max_arm, "valence": ftop.valence, "ext_share": ftop.ext_share,
                "L_hat": hat(b.l_cell[order[0]], dmin, c), "U_hat": hat(b.u_cell[order[0]], dmin, c)}},
        "support_outlier_ratio": support_ratio,
        "max_resid": resid_max,
        "seconds": t0.elapsed().as_secs_f64(),
    });
    eprintln!(
        "{case:>28} {variant:>12}: n={:5} θ̂1={:8.3} Δt={:7.3} µs (ĉ {:.3}) U_split/θ={:.2} sup-ratio {:.2} top1 {:.2} rot {:.2} supE {:.2} d_sup {:.3} resid {:.1e} dense {:?}",
        sp.n, hat(theta1, dmin, c), dt_crit * 1e6, dt_crit * c / dmin, b.split / theta1, support_ratio, topk(1), rot, class_share[1], ftop.d_sup, resid_max, dense_rel
    );
    Res {
        json,
        theta1,
        support_ratio,
        top_rot: rot,
        top5: topk(5),
        u_hat: feats.iter().zip(&b.u_cell).filter(|(f, _)| f.cell).map(|(_, &u)| hat(u, dmin, c)).collect(),
        low_hz: Vec::new(),
        compliance: f64::NAN,
    }
}

fn low_freqs(sp: &Spectral) -> Option<(Vec<f64>, f64)> {
    let p = sp.low(6.min(sp.n), 13, 1e-12, 400).ok()?;
    let mut f: Vec<f64> = p.vals.iter().map(|&l| l.sqrt() / (2.0 * std::f64::consts::PI)).collect();
    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((f, p.resid.iter().cloned().fold(0.0, f64::max)))
}

fn solve_k(sp: &Spectral, f: &[f64]) -> Vec<f64> {
    let k = csc_from_triplets(sp.n, sp.k_triplets());
    let mut s = Spd::analyze(&k);
    s.factor(&k).expect("K SPD");
    let mut u = f.to_vec();
    s.solve(&mut u);
    u
}

// ---------------------------------------------------------------- 2D cases

#[derive(Clone)]
struct Case2 {
    name: String,
    lo: V2,
    hi: V2,
    dmin: f64,
    seed: u64,
    thickness: f64,
    supports: Vec<(usize, Option<(f64, f64)>)>,
    /// Comparator load (edge, range, traction) for compliance.
    load: Option<(usize, Option<(f64, f64)>, V2)>,
    contact: Option<f64>,
}

fn edge_line(lo: V2, hi: V2, e: usize) -> (bool, f64) {
    // (horizontal?, line coordinate)
    match e {
        0 => (true, lo.y),
        1 => (false, hi.x),
        2 => (true, hi.y),
        _ => (false, lo.x),
    }
}

fn excl2(c: &Case2, alpha: f64) -> impl Fn(V2) -> bool + '_ {
    move |p: V2| {
        c.supports.iter().all(|&(e, range)| {
            let (horiz, line) = edge_line(c.lo, c.hi, e);
            let (dist, s) = if horiz { ((p.y - line).abs(), p.x) } else { ((p.x - line).abs(), p.y) };
            let inside = range.map(|(a, b)| s >= a - c.dmin && s <= b + c.dmin).unwrap_or(true);
            !(inside && dist < alpha * c.dmin)
        })
    }
}

/// Seeds: None = original RSA; Some(α) = repair with support exclusion α·d_min.
fn seeds2(c: &Case2, alpha: Option<f64>) -> Vec<V2> {
    let mut rng = Rng::new(c.seed);
    let orig = rsa_rect(c.lo, c.hi, c.dmin, &mut rng, 20_000);
    match alpha {
        None => orig,
        Some(a) => {
            let keep = excl2(c, a);
            let kept: Vec<V2> = orig.into_iter().filter(|&p| keep(p)).collect();
            rsa_rect_with(c.lo, c.hi, c.dmin, &mut rng, 20_000, &keep, kept)
        }
    }
}

fn add_projectile(m: &mut Model, kc: f64) -> usize {
    let mp = 7850.0 * std::f64::consts::PI * R_PROJ * R_PROJ * 0.2;
    let cen = V2::new(0.5, 0.5 + R_PROJ);
    let proj = m.add_plate(cen, mp, 0.5 * mp * R_PROJ * R_PROJ, [false, false, true]);
    for i in 0..m.bodies.len() {
        if m.bodies[i].kind != BodyKind::Cell {
            continue;
        }
        let poly = m.bodies[i].poly.clone();
        let k = poly.len();
        let (mut dbest, mut cp) = (f64::INFINITY, V2::ZERO);
        for e in 0..k {
            let (d, p) = closest_on_segment(cen, poly[e], poly[(e + 1) % k]);
            if d < dbest {
                dbest = d;
                cp = p;
            }
        }
        // E9: contacts within the 2 mm start gap of the projectile surface
        if dbest - R_PROJ > 2e-3 {
            continue;
        }
        let nrm = (cen - cp) / dbest;
        let sb = cen - nrm * R_PROJ;
        let law = PointLaw { kind: LawKind::Elastic, kn: 0.5 * kc, kt: 0.0, gamma: 0.0, delta0: 1.0, wc: 2.0, d_max: 1.0 };
        let pt = FacetPoint { arm_a: cp - m.bodies[i].x0, arm_b: sb - cen, s: 0.0, law, kappa_c: 0.0, rest: V2::ZERO };
        m.facets.push(Facet { a: i, b: proj, pc: cp, n0: nrm, t0: nrm.perp(), len: 1.0, h: 1.0, pts: [pt, pt], active: true, tag: 99, ends: [cp, cp], frame_w: 1.0 });
    }
    proj
}

fn model2(c: &Case2, t: &Tessellation, with_supports: bool) -> Model {
    let mut m = Model::from_tessellation(t, Material::concrete(c.thickness), LawOptions::elastic(), &[]);
    if with_supports {
        let g = m.ground();
        for (k, &(edge, range)) in c.supports.iter().enumerate() {
            m.attach_boundary(t, &BoundaryAttach { edge, range, body: g, shear: true, tag: 1 + k as u32 });
        }
    }
    if let Some(kc) = c.contact {
        add_projectile(&mut m, kc);
    }
    m
}

/// Free-DOF vector of the model's dead loads.
fn load_vec2(m: &Model) -> Vec<f64> {
    let (idx, n) = m.dof_map();
    let mut f = vec![0.0; n];
    for (i, row) in idx.iter().enumerate() {
        for k in 0..3 {
            if let Some(g) = row[k] {
                f[g] = m.loads[i][k];
            }
        }
    }
    f
}

/// Coverage + clamped uniaxial patch test + comparator compliance + low frequencies.
fn mech2(c: &Case2, t: &Tessellation, m: &Model, sp: &Spectral) -> (Value, f64, Vec<f64>) {
    let area: f64 = t.cells.iter().map(|x| x.props.area).sum();
    let dom = (c.hi.x - c.lo.x) * (c.hi.y - c.lo.y);
    let min_mass = m.bodies.iter().filter(|b| b.kind == BodyKind::Cell).map(|b| b.mass).fold(f64::INFINITY, f64::min);
    let min_inertia = m.bodies.iter().filter(|b| b.kind == BodyKind::Cell).map(|b| b.inertia).fold(f64::INFINITY, f64::min);
    let mut j = json!({"area_rel_err": (area - dom).abs() / dom, "facet_mismatch": t.max_mismatch, "min_mass": min_mass, "min_inertia": min_inertia});
    let base_only = c.supports.len() == 1 && c.supports[0] == (0, None);
    if base_only {
        let mut mp = m.clone();
        let p = 1e4;
        edge_traction(&mut mp, t, 2, None, V2::new(0.0, -p), false);
        let u = solve_k(sp, &load_vec2(&mp));
        let (idx, _) = m.dof_map();
        let eps = -p / m.mat.e;
        let mut uex = vec![0.0; sp.n];
        for (i, b) in m.bodies.iter().enumerate() {
            if b.kind == BodyKind::Cell {
                uex[idx[i][1].unwrap()] = eps * (b.generator.y - c.lo.y);
            }
        }
        let du = u.iter().zip(&uex).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / uex.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let fval = |cmp: &solvers::spectrum::Comp, v: &[f64]| cmp.k * cmp.ent.iter().map(|&(d, x)| x * v[d]).sum::<f64>();
        let fmax = sp.comps.iter().map(|cmp| fval(cmp, &uex).abs()).fold(0.0, f64::max);
        let df = sp.comps.iter().map(|cmp| (fval(cmp, &u) - fval(cmp, &uex)).abs()).fold(0.0, f64::max) / fmax;
        j["patch_test"] = json!({"u_rel_err": du, "force_rel_err": df});
    }
    let mut comp = f64::NAN;
    if c.supports.is_empty() {
        return (j, comp, Vec::new());
    }
    if let Some((edge, range, tr)) = c.load {
        let mut mp = m.clone();
        edge_traction(&mut mp, t, edge, range, tr, false);
        let f = load_vec2(&mp);
        let u = solve_k(sp, &f);
        comp = f.iter().zip(&u).map(|(a, b)| a * b).sum();
        j["compliance_J"] = json!(comp);
    }
    let mut low = Vec::new();
    if let Some((f, r)) = low_freqs(sp) {
        j["low_hz"] = json!(f);
        j["low_resid"] = json!(r);
        low = f;
    }
    (j, comp, low)
}

fn sizes2(t: &Tessellation, c: &Case2, m: &Model) -> Value {
    let d = c.dmin;
    let nn: Vec<f64> = t.cells.iter().enumerate().map(|(i, x)| t.cells.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, y)| (y.seed - x.seed).norm()).fold(f64::INFINITY, f64::min) / d).collect();
    let dsup: Vec<f64> = m.facets.iter().filter(|f| f.active && m.bodies[f.b].kind == BodyKind::Ground).map(|f| f.h / d).collect();
    json!({"cells": t.cells.len(), "equiv_size_sqrtA_over_dmin": stats(t.cells.iter().map(|x| x.props.area.sqrt() / d)),
        "nn_generator_over_dmin": stats(nn), "support_d_over_dmin": stats(dsup.clone()),
        "support_facets_below": ALPHAS.iter().map(|a| dsup.iter().filter(|&&x| x < *a).count()).collect::<Vec<_>>()})
}

fn cases2() -> Vec<Case2> {
    let o = V2::new(0.0, 0.0);
    let base = vec![(0usize, None)];
    let mut v = Vec::new();
    for s in 1..=3 {
        v.push(Case2 { name: format!("E2 block s{s}"), lo: o, hi: V2::new(1.0, 0.5), dmin: 0.05, seed: s, thickness: 0.1, supports: base.clone(), load: Some((2, Some((0.0, 0.4)), V2::new(0.5e6, -2.0e6))), contact: None });
    }
    for &(w, h, d) in &[(1.0, 2.0, 0.1), (1.0, 8.0, 0.1), (4.0, 8.0, 0.4)] {
        v.push(Case2 { name: format!("E3 wall {w}x{h} d{d}"), lo: o, hi: V2::new(w, h), dmin: d, seed: 1, thickness: 0.2, supports: base.clone(), load: Some((2, None, V2::new(2.5e3, -1.0e4))), contact: None });
    }
    v.push(Case2 { name: "E7 cantilever".into(), lo: o, hi: V2::new(8.0, 1.2), dmin: 0.3, seed: 7, thickness: 0.3, supports: vec![(3, None), (0, Some((7.4, 8.0)))], load: Some((2, None, V2::new(0.0, -1.0e4))), contact: None });
    v.push(Case2 { name: "E7 column".into(), lo: o, hi: V2::new(1.0, 4.0), dmin: 0.25, seed: 3, thickness: 0.3, supports: base.clone(), load: Some((2, None, V2::new(1.0e4, 0.0))), contact: None });
    v.push(Case2 { name: "E7 block C2".into(), lo: o, hi: V2::new(3.0, 1.0), dmin: 0.3, seed: 5, thickness: 0.3, supports: vec![(3, None)], load: Some((2, None, V2::new(0.0, -1.0e4))), contact: None });
    v.push(Case2 { name: "E7 block C1 free".into(), lo: o, hi: V2::new(3.0, 1.0), dmin: 0.3, seed: 5, thickness: 0.3, supports: vec![], load: None, contact: None });
    v.push(Case2 { name: "E9 block".into(), lo: o, hi: V2::new(1.0, 0.5), dmin: 0.05, seed: 9, thickness: 0.2, supports: base.clone(), load: Some((2, None, V2::new(0.0, -1.0e4))), contact: None });
    v
}

// ---------------------------------------------------------------- 3D cases

#[derive(Clone)]
struct Case3 {
    name: String,
    lo: V3,
    hi: V3,
    dmin: f64,
    seed: u64,
    supports: Vec<usize>,
}

fn face_dist(lo: V3, hi: V3, f: usize, p: V3) -> f64 {
    match f {
        0 => p.x - lo.x,
        1 => hi.x - p.x,
        2 => p.y - lo.y,
        3 => hi.y - p.y,
        4 => p.z - lo.z,
        _ => hi.z - p.z,
    }
}

fn excl3(c: &Case3, alpha: f64) -> impl Fn(V3) -> bool + '_ {
    move |p: V3| c.supports.iter().all(|&f| face_dist(c.lo, c.hi, f, p) >= alpha * c.dmin)
}

#[derive(Clone, Copy)]
enum Gen3 {
    Original,
    Repair(f64),
    Scratch(f64),
}

fn seeds3(c: &Case3, g: Gen3) -> Vec<V3> {
    let mut rng = Rng::new(c.seed);
    match g {
        Gen3::Original => rsa_box(c.lo, c.hi, c.dmin, &mut rng, 5000),
        Gen3::Repair(a) => {
            let orig = rsa_box(c.lo, c.hi, c.dmin, &mut rng, 5000);
            let keep = excl3(c, a);
            let kept: Vec<V3> = orig.into_iter().filter(|&p| keep(p)).collect();
            rsa_box_with(c.lo, c.hi, c.dmin, &mut rng, 5000, &keep, kept)
        }
        Gen3::Scratch(a) => {
            let keep = excl3(c, a);
            rsa_box_with(c.lo, c.hi, c.dmin, &mut rng, 5000, &keep, Vec::new())
        }
    }
}

fn mech3(c: &Case3, cells3: &[VCell3], m: &Model3, sp: &Spectral) -> (Value, f64, Vec<f64>) {
    let ext = c.hi - c.lo;
    let vdom = ext.x * ext.y * ext.z;
    let vol: f64 = m.cells.iter().map(|x| x.volume).sum();
    let mtot: f64 = m.cells.iter().map(|x| x.mass).sum();
    let cen = (c.lo + c.hi) * 0.5;
    let mut first = V3::ZERO;
    let mut second = [[0.0; 3]; 3];
    let mut min_j = f64::INFINITY;
    for x in &m.cells {
        first += (x.centroid - cen) * x.mass;
        let r = x.centroid - cen;
        for a in 0..3 {
            for b in 0..3 {
                second[a][b] += x.inertia[a][b] + x.mass * (if a == b { r.norm2() } else { 0.0 } - r[a] * r[b]);
            }
        }
        min_j = min_j.min(sym3_eigs(x.inertia)[0]);
    }
    let md = m.mat.density * vdom;
    let exact = [md * (ext.y * ext.y + ext.z * ext.z) / 12.0, md * (ext.x * ext.x + ext.z * ext.z) / 12.0, md * (ext.x * ext.x + ext.y * ext.y) / 12.0];
    let mut ierr: f64 = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            let ex = if a == b { exact[a] } else { 0.0 };
            ierr = ierr.max((second[a][b] - ex).abs() / exact[2]);
        }
    }
    // facet areas seen from both sides
    let mut area: std::collections::HashMap<(usize, usize), f64> = Default::default();
    for (i, x) in cells3.iter().enumerate() {
        for f in &x.poly.faces {
            if let FaceTag::Cell(j) = f.tag {
                *area.entry((i, j)).or_default() += geometry::voronoi3d::poly_area(&f.verts, f.normal);
            }
        }
    }
    let mut amis: f64 = 0.0;
    let mut unmatched = 0;
    for (&(i, j), &a) in &area {
        match area.get(&(j, i)) {
            Some(&b) => amis = amis.max((a - b).abs() / a.max(b)),
            None => unmatched += 1,
        }
    }
    let mut j = json!({"volume_rel_err": (vol - vdom).abs() / vdom, "mass_rel_err": (mtot - md).abs() / md, "first_moment_rel": first.norm() / (mtot * ext.norm()),
        "second_moment_rel": ierr, "min_mass": m.cells.iter().map(|x| x.mass).fold(f64::INFINITY, f64::min), "min_principal_inertia": min_j,
        "facet_area_mismatch": amis, "unmatched_facets": unmatched});
    let base_only = c.supports == vec![4];
    if base_only {
        // clamped uniaxial patch test: exact u_i = ε (g_z − z0) e_z, ω_i = 0
        let mut mp = m.clone();
        let p = 1e4;
        mp.traction(cells3, 5, V3::new(0.0, 0.0, -p));
        let f: Vec<f64> = mp.loads.iter().flatten().copied().collect();
        let u = solve_k(sp, &f);
        let eps = -p / m.mat.e;
        let mut uex = vec![0.0; sp.n];
        for (i, x) in m.cells.iter().enumerate() {
            uex[6 * i + 2] = eps * (x.generator.z - c.lo.z);
        }
        let du = u.iter().zip(&uex).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / uex.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let fval = |cmp: &solvers::spectrum::Comp, v: &[f64]| cmp.k * cmp.ent.iter().map(|&(d, x)| x * v[d]).sum::<f64>();
        let fmax = sp.comps.iter().map(|cmp| fval(cmp, &uex).abs()).fold(0.0, f64::max);
        let df = sp.comps.iter().map(|cmp| (fval(cmp, &u) - fval(cmp, &uex)).abs()).fold(0.0, f64::max) / fmax;
        j["patch_test"] = json!({"u_rel_err": du, "force_rel_err": df});
    }
    let mut comp = f64::NAN;
    if !c.supports.is_empty() {
        let mut mp = m.clone();
        mp.traction(cells3, 5, V3::new(2.5e3, 1.0e3, -1.0e4));
        let f: Vec<f64> = mp.loads.iter().flatten().copied().collect();
        let u = solve_k(sp, &f);
        comp = f.iter().zip(&u).map(|(a, b)| a * b).sum();
        // area-weighted mean displacement of the top face
        let (mut s, mut at) = (V3::ZERO, 0.0);
        for (i, x) in cells3.iter().enumerate() {
            for fc in x.poly.faces.iter().filter(|fc| fc.tag == FaceTag::Boundary(5)) {
                let (a, pc, _, _, _) = geometry::voronoi3d::planar_props(&fc.verts, fc.normal);
                let w = V3::new(u[6 * i + 3], u[6 * i + 4], u[6 * i + 5]);
                s += (V3::new(u[6 * i], u[6 * i + 1], u[6 * i + 2]) + w.cross(pc - m.cells[i].centroid)) * a;
                at += a;
            }
        }
        let ut = s / at;
        j["compliance_J"] = json!(comp);
        j["top_mean_disp_m"] = json!([ut.x, ut.y, ut.z]);
    }
    let mut low = Vec::new();
    if !c.supports.is_empty() {
        if let Some((f, r)) = low_freqs(sp) {
            j["low_hz"] = json!(f);
            j["low_resid"] = json!(r);
            low = f;
        }
    } else {
        // free block: rigid rotations must be exact null vectors of K
        let k = csc_from_triplets(sp.n, sp.k_triplets());
        let kn: f64 = k.val().iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut worst: f64 = 0.0;
        for ax in 0..3 {
            let mut w = V3::ZERO;
            w[ax] = 1.0;
            let mut r = vec![0.0; sp.n];
            for (i, x) in m.cells.iter().enumerate() {
                let t = w.cross(x.centroid - cen);
                r[6 * i..6 * i + 3].copy_from_slice(&[t.x, t.y, t.z]);
                r[6 * i + 3 + ax] = 1.0;
            }
            let kr: f64 = sp.comps.iter().map(|cmp| (cmp.k * cmp.ent.iter().map(|&(d, x)| x * r[d]).sum::<f64>()).powi(2)).sum::<f64>().sqrt();
            let rn: f64 = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            worst = worst.max(kr / (kn * rn));
        }
        j["rigid_rotation_null_rel"] = json!(worst);
    }
    (j, comp, low)
}

fn sizes3(c: &Case3, cells3: &[VCell3], m: &Model3) -> Value {
    let d = c.dmin;
    let nn: Vec<f64> = cells3.iter().enumerate().map(|(i, x)| cells3.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, y)| (y.seed - x.seed).norm()).fold(f64::INFINITY, f64::min) / d).collect();
    let dsup: Vec<f64> = m.facets.iter().filter(|f| f.b.is_none()).map(|f| f.ell / d).collect();
    let mut dsup_cells: Vec<f64> = Vec::new();
    for i in 0..m.cells.len() {
        let v = m.facets.iter().filter(|f| f.b.is_none() && f.a == i).map(|f| f.ell / d).fold(f64::INFINITY, f64::min);
        if v.is_finite() {
            dsup_cells.push(v);
        }
    }
    let vmean = m.cells.iter().map(|x| x.volume).sum::<f64>() / m.cells.len() as f64;
    json!({"cells": cells3.len(), "mean_cell_size_cbrtV": vmean.cbrt(), "mean_cell_size_over_dmin": vmean.cbrt() / d,
        "equiv_size_cbrtV_over_dmin": stats(m.cells.iter().map(|x| x.volume.cbrt() / d)),
        "equiv_sphere_diam_over_dmin": stats(m.cells.iter().map(|x| (6.0 * x.volume / std::f64::consts::PI).cbrt() / d)),
        "nn_generator_over_dmin": stats(nn), "support_cells_d_over_dmin": stats(dsup_cells.clone()),
        "support_cells_below": ALPHAS.iter().map(|a| dsup_cells.iter().filter(|&&x| x < *a).count()).collect::<Vec<_>>(),
        "support_facets": dsup.len()})
}

fn run3(c: &Case3, gen: Gen3, label: &str, with_supports: bool, rot_scale: f64, out: &mut Out) -> (Res, Value) {
    let seeds = seeds3(c, gen);
    let cells3 = voronoi_box(&seeds, c.lo, c.hi);
    let sup = if with_supports { c.supports.clone() } else { vec![] };
    let mut m = Model3::build(&cells3, &sup, Mat3D::concrete());
    if rot_scale != 1.0 {
        for x in m.cells.iter_mut() {
            x.inertia = x.inertia.map(|r| r.map(|v| v * rot_scale));
        }
    }
    let sp = Spectral::from_model3(&m);
    let feats = feats3(&m, &cells3, c.dmin);
    let cc = Mat3D::concrete().wave_speed();
    let mut r = analyse(&c.name, label, &sp, &feats, c.dmin, cc, None, out);
    let cc3 = Case3 { supports: sup, ..c.clone() };
    let (mj, comp, low) = mech3(&cc3, &cells3, &m, &sp);
    r.compliance = comp;
    r.low_hz = low;
    let mut j = r.json.clone();
    j["mechanics"] = mj;
    j["sizes"] = sizes3(c, &cells3, &m);
    j["n_seeds"] = json!(seeds.len());
    j["rot_inertia_scale"] = json!(rot_scale);
    j["total_rot_inertia_about_centroids"] = json!(m.cells.iter().map(|x| x.inertia[0][0] + x.inertia[1][1] + x.inertia[2][2]).sum::<f64>());
    (r, j)
}

fn run2(c: &Case2, alpha: Option<f64>, label: &str, with_supports: bool, dump: bool, out: &mut Out) -> (Res, Value) {
    let seeds = seeds2(c, alpha);
    let t = voronoi_convex(&seeds, &rect(c.lo, c.hi));
    assert!(t.max_mismatch < 1e-9 * (c.hi - c.lo).norm());
    let m = model2(c, &t, with_supports);
    let sp = Spectral::from_model2d(&m);
    let feats = feats2(&m, &sp, c.dmin, c.lo, c.hi);
    let cc = m.mat.wave_speed();
    let mut r = analyse(&c.name, label, &sp, &feats, c.dmin, cc, dump.then_some(&m), out);
    let mut j = r.json.clone();
    let cs = Case2 { supports: if with_supports { c.supports.clone() } else { vec![] }, ..c.clone() };
    if c.contact.is_none() {
        let (mj, comp, low) = mech2(&cs, &t, &m, &sp);
        r.compliance = comp;
        r.low_hz = low;
        j["mechanics"] = mj;
    } else {
        let mp = m.bodies.iter().find(|b| b.kind == BodyKind::Plate).map(|b| b.mass).unwrap();
        let kc = c.contact.unwrap();
        let contact_cells: Vec<f64> = m.facets.iter().filter(|f| f.tag == 99).map(|f| m.bodies[f.a].mass).collect();
        j["contact"] = json!({"kc": kc, "projectile_mass": mp, "contact_cells": contact_cells.len(), "contact_cell_masses": contact_cells,
            "omega_estimate": contact_cells.iter().map(|&mc| (kc * (1.0 / mp + 1.0 / mc)).sqrt()).collect::<Vec<_>>(), "omega_ritz": r.theta1.sqrt()});
    }
    j["sizes"] = sizes2(&t, c, &m);
    (r, j)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

fn main() {
    let t_start = std::time::Instant::now();
    let mut out = Out {
        cases: Csv::new(EXP, "cases", &["case", "variant", "blocks", "dofs", "theta1_hat", "dt_crit_hat", "dt_crit_us", "block_lower_hat", "U_split_hat", "U_split_t1_hat", "U_cw_hat", "U_gersh_hat", "U_syl_hat", "dt_safe_split_us", "dt_safe_syl_us", "N_safe_split", "N_safe_syl", "kin_top1", "kin_top5", "rot_share", "support_elastic_share", "top_cell_d_sup", "support_ratio", "max_resid"]),
        cells: Csv::new(EXP, "cells", &["case", "variant", "body", "cell", "support", "boundary", "d_sup", "vol", "jmin", "elong", "nn", "max_face", "max_arm", "valence", "ext_share", "L_hat", "U_hat", "kin_top", "rot_top"]),
        modes: Csv::new(EXP, "modes", &["case", "variant", "start_seed", "rank", "lam_hat", "resid", "dense_lam_hat", "steps", "matvecs"]),
        polys: Csv::new(EXP, "polys2d", &["case", "variant", "body", "kin_top", "L_hat", "verts"]),
    };
    let mut res = serde_json::Map::new();

    // ---------------- development cases: 3D (E10)
    let e10: Vec<Case3> = [(0.2, 1u64), (0.2, 2), (0.15, 3)]
        .iter()
        .map(|&(d, s)| Case3 { name: format!("E10 d{d} s{s}"), lo: V3::ZERO, hi: V3::new(1.0, 1.0, 1.0), dmin: d, seed: s, supports: vec![4] })
        .collect();
    let mut dev3 = Vec::new();
    let mut dev_res: Vec<(String, String, Res)> = Vec::new();
    for c in &e10 {
        let mut runs = Vec::new();
        let (r0, j0) = run3(c, Gen3::Original, "V0", true, 1.0, &mut out);
        let (rf, jf) = run3(c, Gen3::Original, "V0-free", false, 1.0, &mut out);
        runs.push(j0);
        runs.push(jf);
        let mut cmp = Vec::new();
        for &a in &ALPHAS {
            let (ra, ja) = run3(c, Gen3::Repair(a), &format!("V1 a{a}"), true, 1.0, &mut out);
            cmp.push(json!({"alpha": a, "theta_rel_to_free": ra.theta1 / rf.theta1, "dt_gain_vs_control": (r0.theta1 / ra.theta1).sqrt(),
                "compliance_rel_diff": rel(ra.compliance, r0.compliance),
                "low_hz_rel_diff": ra.low_hz.iter().zip(&r0.low_hz).map(|(a, b)| rel(*a, *b)).collect::<Vec<_>>()}));
            runs.push(ja);
            dev_res.push((c.name.clone(), format!("{a}"), ra));
        }
        dev_res.push((c.name.clone(), "free".into(), rf));
        dev_res.push((c.name.clone(), "V0".into(), r0));
        dev3.push(json!({"case": c.name, "runs": runs, "variant_vs_control": cmp}));
    }
    res.insert("dev3d".into(), json!(dev3));

    // ---------------- development cases: 2D (E2/E3/E7/E9)
    let mut dev2 = Vec::new();
    for c in cases2() {
        let mut runs = Vec::new();
        let dump = c.name == "E2 block s1" || c.name.starts_with("E7 cantilever") || c.name.starts_with("E3 wall 1x2");
        let (r0, j0) = run2(&c, None, "V0", true, dump, &mut out);
        runs.push(j0);
        let rf = if !c.supports.is_empty() {
            let (rf, jf) = run2(&c, None, "V0-free", false, false, &mut out);
            runs.push(jf);
            Some(rf)
        } else {
            None
        };
        let mut cmp = Vec::new();
        if !c.supports.is_empty() {
            for &a in &ALPHAS {
                let (ra, ja) = run2(&c, Some(a), &format!("V1 a{a}"), true, dump && a == 0.5, &mut out);
                let rff = rf.as_ref().unwrap();
                cmp.push(json!({"alpha": a, "theta_rel_to_free": ra.theta1 / rff.theta1, "dt_gain_vs_control": (r0.theta1 / ra.theta1).sqrt(),
                    "compliance_rel_diff": rel(ra.compliance, r0.compliance),
                    "low_hz_rel_diff": ra.low_hz.iter().zip(&r0.low_hz).map(|(a, b)| rel(*a, *b)).collect::<Vec<_>>()}));
                runs.push(ja);
                dev_res.push((c.name.clone(), format!("{a}"), ra));
            }
            dev_res.push((c.name.clone(), "free".into(), rf.unwrap()));
        }
        dev_res.push((c.name.clone(), "V0".into(), r0));
        dev2.push(json!({"case": c.name, "dmin": c.dmin, "seed": c.seed, "runs": runs, "variant_vs_control": cmp}));
    }
    res.insert("dev2d".into(), json!(dev2));

    // ---------------- α decision (pre-registered rule)
    let mut decision = Vec::new();
    let mut alpha_final = None;
    for &a in &ALPHAS {
        let tag = format!("{a}");
        let mut ok = true;
        let mut detail = Vec::new();
        for (case, v, r) in dev_res.iter().filter(|x| x.1 == tag) {
            let free = dev_res.iter().find(|x| &x.0 == case && x.1 == "free").map(|x| x.2.theta1).unwrap();
            let pass = r.support_ratio <= 1.1 && r.theta1 / free <= 1.1;
            ok &= pass;
            detail.push(json!({"case": case, "variant": v, "support_ratio": r.support_ratio, "theta_over_free": r.theta1 / free, "pass": pass}));
        }
        decision.push(json!({"alpha": a, "all_pass": ok, "cases": detail}));
        if ok && alpha_final.is_none() {
            alpha_final = Some(a);
        }
    }
    let alpha_rule = alpha_final;
    let alpha_final = alpha_final.unwrap_or(0.5);
    eprintln!("α decision: rule → {alpha_rule:?}; final α = {alpha_final}");
    // contract constant from the 3D development cases under the final α
    let kappa_dev = dev_res
        .iter()
        .filter(|(case, v, _)| case.starts_with("E10") && *v == format!("{alpha_final}"))
        .flat_map(|(_, _, r)| r.u_hat.iter().map(|u| u.sqrt()))
        .fold(0.0, f64::max);
    let kappa = (kappa_dev * 1.1 * 100.0).ceil() / 100.0;
    let dt_hat_contract = SAFETY * 2.0 / kappa;
    eprintln!("contract: κ = {kappa} (dev max {kappa_dev:.4}); Δt_safe ≥ {dt_hat_contract:.4} d_min/c");
    res.insert("alpha_decision".into(), json!({"rule": "smallest α with support_ratio ≤ 1.1 and θ1/θ1(free) ≤ 1.1 in every development case; else 0.5",
        "rule_result": alpha_rule, "final_alpha": alpha_final, "per_alpha": decision}));
    res.insert("contract".into(), json!({"kappa": kappa, "kappa_dev_max": kappa_dev, "dt_safe_hat_guaranteed": dt_hat_contract,
        "statement": "asset valid iff RSA spacing ≥ d_min, generator–support distance ≥ α d_min, Voronoi consistent, and every cell Û_i ≤ κ² (Û = U d_min²/c²)"}));

    // ---------------- E9 contact (declared k_c 1e8, 1e9; 1e10, 1e11 to locate the crossover)
    let mut e9c = Vec::new();
    let base9 = cases2().into_iter().find(|c| c.name == "E9 block").unwrap();
    for &kc in &[1e8, 1e9, 1e10, 1e11] {
        for (alpha, lab) in [(None, "V0"), (Some(alpha_final), "V1 final")] {
            let c = Case2 { name: format!("E9 contact kc{kc:.0e}"), contact: Some(kc), ..base9.clone() };
            let (_, j) = run2(&c, alpha, lab, true, false, &mut out);
            e9c.push(j);
        }
    }
    res.insert("e9_contact".into(), json!(e9c));

    // ---------------- inertia-scaling diagnostic (only if top modes stay local and rotational)
    let mut scaling = Vec::new();
    for c in &e10 {
        let r = dev_res.iter().find(|x| x.0 == c.name && x.1 == format!("{alpha_final}")).map(|x| &x.2).unwrap();
        let triggered = r.top_rot > 0.5 && r.top5 >= 0.8;
        let mut runs = vec![json!({"case": c.name, "top_rot_share": r.top_rot, "kin_top5": r.top5, "triggered": triggered})];
        if triggered {
            for &s in &[2.0, 4.0] {
                let (rs, mut js) = run3(c, Gen3::Repair(alpha_final), &format!("V1 J×{s}"), true, s, &mut out);
                js["dt_gain_vs_physical"] = json!((r.theta1 / rs.theta1).sqrt());
                js["low_hz_rel_diff_vs_physical"] = json!(rs.low_hz.iter().zip(&r.low_hz).map(|(a, b)| (a - b) / b).collect::<Vec<_>>());
                runs.push(js);
            }
        }
        scaling.push(json!(runs));
    }
    res.insert("inertia_scaling_diagnostic".into(), json!(scaling));

    // ---------------- held-out 3D: cube 2 m, clamped base, game sizes × seeds
    let mut held = Vec::new();
    for &d in &GAME_SIZES {
        let mut per = Vec::new();
        let (mut comp0, mut comp1, mut low0, mut low1) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &s in &HELD_OUT_SEEDS {
            let c = Case3 { name: format!("cube2 d{d} s{s}"), lo: V3::ZERO, hi: V3::new(2.0, 2.0, 2.0), dmin: d, seed: s, supports: vec![4] };
            let (r0, j0) = run3(&c, Gen3::Original, "V0", true, 1.0, &mut out);
            let (r1, j1) = run3(&c, Gen3::Scratch(alpha_final), "V-final", true, 1.0, &mut out);
            let umax = r1.u_hat.iter().cloned().fold(0.0, f64::max);
            let fails = r1.u_hat.iter().filter(|&&u| u > kappa * kappa).count();
            let fails0 = r0.u_hat.iter().filter(|&&u| u > kappa * kappa).count();
            comp0.push(r0.compliance);
            comp1.push(r1.compliance);
            low0.push(r0.low_hz.clone());
            low1.push(r1.low_hz.clone());
            per.push(json!({"seed": s, "control": j0, "variant": j1, "contract": {"U_hat_max": umax, "kappa2": kappa * kappa, "cells_failing": fails, "control_cells_failing": fails0},
                "dt_gain_vs_control": (r0.theta1 / r1.theta1).sqrt()}));
        }
        let spread = |v: &[f64]| {
            let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = v.iter().cloned().fold(0.0, f64::max);
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            json!({"mean": mean, "min": mn, "max": mx})
        };
        let low_by_rank = |l: &[Vec<f64>], k: usize| l.iter().filter_map(|x| x.get(k).copied()).collect::<Vec<_>>();
        held.push(json!({"dmin": d, "seeds": per, "compliance": {"control": spread(&comp0), "variant": spread(&comp1)},
            "low_hz": (0..6).map(|k| json!({"rank": k, "control": spread(&low_by_rank(&low0, k)), "variant": spread(&low_by_rank(&low1, k))})).collect::<Vec<_>>()}));
    }
    res.insert("held_out".into(), json!(held));

    // ================= POST-HOC (not pre-registered; added after the results above) =================
    // PH1: larger exclusion distances, same rule and mechanics comparison as the pre-registered α.
    let mut ph1 = Vec::new();
    for c in &e10 {
        let (r0, _) = run3(c, Gen3::Original, "PH V0", true, 1.0, &mut out);
        let free = dev_res.iter().find(|x| x.0 == c.name && x.1 == "free").map(|x| x.2.theta1).unwrap();
        for &a in &PH_ALPHAS {
            let (ra, ja) = run3(c, Gen3::Repair(a), &format!("PH V1 a{a}"), true, 1.0, &mut out);
            ph1.push(json!({"case": c.name, "alpha": a, "support_ratio": ra.support_ratio, "theta_over_free": ra.theta1 / free,
                "pass": ra.support_ratio <= 1.1 && ra.theta1 / free <= 1.1, "compliance_rel": ra.compliance / r0.compliance - 1.0,
                "low_hz_rel": ra.low_hz.iter().zip(&r0.low_hz).map(|(a, b)| a / b - 1.0).collect::<Vec<_>>(), "run": ja}));
        }
    }
    for c in cases2().into_iter().filter(|c| !c.supports.is_empty()) {
        let (r0, _) = run2(&c, None, "PH V0", true, false, &mut out);
        let free = dev_res.iter().find(|x| x.0 == c.name && x.1 == "free").map(|x| x.2.theta1).unwrap();
        for &a in &PH_ALPHAS {
            let (ra, ja) = run2(&c, Some(a), &format!("PH V1 a{a}"), true, false, &mut out);
            ph1.push(json!({"case": c.name, "alpha": a, "support_ratio": ra.support_ratio, "theta_over_free": ra.theta1 / free,
                "pass": ra.support_ratio <= 1.1 && ra.theta1 / free <= 1.1, "compliance_rel": ra.compliance / r0.compliance - 1.0,
                "low_hz_rel": ra.low_hz.iter().zip(&r0.low_hz).map(|(a, b)| a / b - 1.0).collect::<Vec<_>>(), "run": ja}));
        }
    }
    // PH2/PH3: refinement — does the support-layer change of compliance / low frequencies vanish,
    // and which construction is closer to the converged value?
    let mut refine = Csv::new(EXP, "posthoc_refinement", &["dim", "dmin", "seed", "gen", "cells", "compliance", "f0", "f1", "f2", "f3", "f4", "f5", "low_resid"]);
    let mut ph2 = Vec::new();
    for &d in &[0.5, 0.35, 0.25, 0.2, 0.15, 0.1] {
        for &s in &PH_SEEDS {
            let c = Case3 { name: format!("cube2 d{d} s{s}"), lo: V3::ZERO, hi: V3::new(2.0, 2.0, 2.0), dmin: d, seed: s, supports: vec![4] };
            for (gname, g) in [("control", Gen3::Original), ("a0.5", Gen3::Scratch(0.5)), ("a0.6", Gen3::Scratch(0.6))] {
                let seeds = seeds3(&c, g);
                let cells3 = voronoi_box(&seeds, c.lo, c.hi);
                let mut m = Model3::build(&cells3, &c.supports, Mat3D::concrete());
                let sp = Spectral::from_model3(&m);
                m.traction(&cells3, 5, V3::new(2.5e3, 1.0e3, -1.0e4));
                let f: Vec<f64> = m.loads.iter().flatten().copied().collect();
                let u = solve_k(&sp, &f);
                let comp: f64 = f.iter().zip(&u).map(|(a, b)| a * b).sum();
                let (low, lr) = low_freqs(&sp).unwrap();
                let mut row = vec![3.0, d, s as f64, ["control", "a0.5", "a0.6"].iter().position(|x| *x == gname).unwrap() as f64, m.cells.len() as f64, comp];
                row.extend(low.iter().copied());
                row.push(lr);
                refine.row(&row);
                eprintln!("refine 3D d{d} s{s} {gname}: cells {} f0 {:.2} C {:.5e}", m.cells.len(), low[0], comp);
                ph2.push(json!({"dim": 3, "dmin": d, "seed": s, "gen": gname, "cells": m.cells.len(), "compliance": comp, "low_hz": low, "low_resid": lr}));
            }
        }
    }
    for &d in &[0.3, 0.2, 0.15, 0.1, 0.075, 0.05] {
        for &s in &PH_SEEDS {
            let c = Case2 { name: format!("cantilever clamped d{d} s{s}"), lo: V2::new(0.0, 0.0), hi: V2::new(8.0, 1.2), dmin: d, seed: s, thickness: 0.3,
                supports: vec![(3, None)], load: Some((2, None, V2::new(0.0, -1.0e4))), contact: None };
            for (gname, a) in [("control", None), ("a0.5", Some(0.5)), ("a0.6", Some(0.6))] {
                // post-hoc 2D constructions use constrained RSA from scratch, like the held-out 3D set
                let seeds = match a {
                    None => rsa_rect(c.lo, c.hi, c.dmin, &mut Rng::new(s), 20_000),
                    Some(al) => rsa_rect_with(c.lo, c.hi, c.dmin, &mut Rng::new(s), 20_000, &excl2(&c, al), Vec::new()),
                };
                let t = voronoi_convex(&seeds, &rect(c.lo, c.hi));
                let m = model2(&c, &t, true);
                let sp = Spectral::from_model2d(&m);
                let mut mp = m.clone();
                edge_traction(&mut mp, &t, 2, None, V2::new(0.0, -1.0e4), false);
                let f = load_vec2(&mp);
                let u = solve_k(&sp, &f);
                let comp: f64 = f.iter().zip(&u).map(|(a, b)| a * b).sum();
                let (low, lr) = low_freqs(&sp).unwrap();
                let mut row = vec![2.0, d, s as f64, ["control", "a0.5", "a0.6"].iter().position(|x| *x == gname).unwrap() as f64, m.n_cells() as f64, comp];
                row.extend(low.iter().copied());
                row.push(lr);
                refine.row(&row);
                eprintln!("refine 2D d{d} s{s} {gname}: cells {} f0 {:.3} C {:.5e}", m.n_cells(), low[0], comp);
                ph2.push(json!({"dim": 2, "dmin": d, "seed": s, "gen": gname, "cells": m.n_cells(), "compliance": comp, "low_hz": low, "low_resid": lr}));
            }
        }
    }
    res.insert("posthoc".into(), json!({"note": "added after the pre-registered runs; not used to fix α or κ",
        "larger_alpha": ph1, "refinement": ph2, "seeds": PH_SEEDS}));

    let doc = json!({
        "meta": meta("Phase0b-geometry-spectrum", "all", None,
            "Lanczos (full reorth, 2 start vectors), dense check n≤3000, bounds (Gershgorin, Collatz–Wielandt, cell splitting, Sylvester/Cholesky), shift-invert low modes",
            json!({"alphas": ALPHAS, "nev": NEV, "lanczos_tol": LANCZOS_TOL, "max_steps": MAX_STEPS, "dense_max": DENSE_MAX, "safety": SAFETY,
                "held_out_seeds": HELD_OUT_SEEDS, "game_sizes_dmin": GAME_SIZES, "material": "E 30 GPa, rho 2400, gamma 1",
                "residual_definition": "eta = ||B y - lambda y||_2 / (|lambda| ||y||_2), B = M^-1/2 K M^-1/2",
                "dimensionless": "lam_hat = lambda d_min^2 / c^2, dt_hat = dt c / d_min, c = sqrt(E/rho)"})),
        "results": res,
        "seconds": t_start.elapsed().as_secs_f64(),
    });
    write_json(EXP, "summary", &doc);
    eprintln!("done in {:.0} s", t_start.elapsed().as_secs_f64());
}
