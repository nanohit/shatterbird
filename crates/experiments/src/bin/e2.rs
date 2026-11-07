//! E2 — redundant (statically indeterminate) RBSN block: facet force convergence
//! of S1 (ALM policies), S3 (penalty), PCG and S2 against S0, as a function of work.

use experiments::{edge_traction, force_err, meta, rect_tess, support_reaction, write_json, Csv};
use geometry::V2;
use model::{BoundaryAttach, FacetForce, LawOptions, Material, Model};
use serde_json::{json, Value};
use solvers::cg::Pcg;
use solvers::s0::{newton, NewtonOpts};
use solvers::s1::{Alm, RhoPolicy};
use solvers::s2::S2;

const EXP: &str = "E2";

pub struct Setup {
    pub m: Model,
    pub qs: Vec<[f64; 3]>,
    pub fs: Vec<FacetForce>,
    pub rs: V2,
}

fn setup(seed: u64) -> (Model, geometry::voronoi2d::Tessellation) {
    let t = rect_tess(V2::new(0.0, 0.0), V2::new(1.0, 0.5), 0.05, seed);
    let mut m = Model::from_tessellation(&t, Material::concrete(0.1), LawOptions::elastic(), &[]);
    let g = m.ground();
    m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: None, body: g, shear: true, tag: 1 });
    // eccentric patch load on the top-left 40 %: −2 MPa normal, +0.5 MPa shear
    edge_traction(&mut m, &t, 2, Some((0.0, 0.4)), V2::new(0.5e6, -2.0e6), false);
    (m, t)
}

fn solve_ref(m: &Model) -> Setup {
    let mut q = m.q.clone();
    let rep = newton(m, &mut q, 0.0, None, NewtonOpts { tol: 1e-6, max_iter: 20 });
    assert!(rep.converged, "S0 failed {:?}", rep);
    let fs = m.facet_forces(&q);
    let rs = support_reaction(m, &q, 1);
    Setup { m: m.clone(), qs: q, fs, rs }
}

struct Errs {
    f_dual: f64,
    f_const: f64,
    f_dual_max: f64,
    disp: f64,
    constit: f64,
    reaction: f64,
    energy: f64,
}

fn errs(m: &Model, q: &[[f64; 3]], alm: Option<&Alm>, s: &Setup) -> Errs {
    let mask: Vec<bool> = m.facets.iter().map(|f| f.active).collect();
    let fc = m.facet_forces(q);
    let ec = force_err(&fc, &s.fs, &mask);
    let ed = alm.map(|a| force_err(&a.dual_forces(m), &s.fs, &mask)).unwrap_or(ec);
    let umax = s.qs.iter().zip(&m.q).fold(0.0f64, |acc, (a, r)| acc.max((a[0] - r[0]).abs()).max((a[1] - r[1]).abs()));
    let du = q.iter().zip(&s.qs).fold(0.0f64, |acc, (a, b)| acc.max((a[0] - b[0]).abs()).max((a[1] - b[1]).abs()));
    let fnorm = s.fs.iter().map(|f| f.n * f.n + f.v * f.v).sum::<f64>().sqrt();
    let constit = alm
        .map(|a| {
            let fd = a.dual_forces(m);
            (fd.iter().zip(&fc).map(|(x, y)| (x.n - y.n).powi(2) + (x.v - y.v).powi(2)).sum::<f64>()).sqrt() / fnorm
        })
        .unwrap_or(0.0);
    let r = support_reaction(m, q, 1);
    let rd = alm.map(|a| {
        let fd = a.dual_forces(m);
        // reaction on the support from multipliers: −Σ (N n + V t), reference frame
        let mut v = V2::ZERO;
        for (fi, f) in m.facets.iter().enumerate() {
            if f.active && f.tag == 1 {
                v -= f.n0 * fd[fi].n + f.t0 * fd[fi].v;
            }
        }
        v
    });
    let rr = rd.unwrap_or(r);
    let us = m.elastic_energy(&s.qs);
    Errs {
        f_dual: ed.l2,
        f_const: ec.l2,
        f_dual_max: ed.max,
        disp: du / umax,
        constit,
        reaction: (rr - s.rs).norm() / s.rs.norm(),
        energy: (m.elastic_energy(q) - us).abs() / us,
    }
}

fn policies() -> Vec<(String, RhoPolicy)> {
    let mut v = Vec::new();
    for c in [1e-3, 1e-2, 1e-1, 1.0, 10.0] {
        v.push((format!("shared c={c}"), RhoPolicy::Shared(c)));
    }
    for c in [1e-3, 1e-2, 1e-1, 1.0, 10.0] {
        v.push((format!("perspring c={c}"), RhoPolicy::PerSpring(c)));
    }
    v.push(("S3 penalty".into(), RhoPolicy::Penalty));
    v
}

const HEAD: [&str; 12] = [
    "method", "iter", "sweeps", "passes", "point_evals", "f_dual_l2", "f_const_l2", "f_dual_max", "disp", "constit", "reaction", "energy",
];

fn row(csv: &mut Csv, mi: usize, it: usize, alm: &Alm, e: &Errs) {
    csv.row(&[
        mi as f64,
        it as f64,
        alm.work.sweeps as f64,
        alm.work.passes as f64,
        alm.work.point_evals as f64,
        e.f_dual,
        e.f_const,
        e.f_dual_max,
        e.disp,
        e.constit,
        e.reaction,
        e.energy,
    ]);
}

fn first_below(v: &[(usize, f64)], thr: f64) -> Option<usize> {
    v.iter().find(|x| x.1 <= thr).map(|x| x.0)
}

fn main() {
    model::range::enable();
    let seeds = [1u64, 2, 3];
    let nmax = 3000usize;
    let pol = policies();
    let mut res = Vec::new();
    let mut csv = Csv::new(EXP, "cold", &HEAD);
    let mut info = Vec::new();
    for &seed in &seeds {
        let (m, _t) = setup(seed);
        let s = solve_ref(&m);
        let alm0 = Alm::new(&m, RhoPolicy::Penalty, None);
        info.push(json!({"seed": seed, "cells": m.n_cells(), "facets": m.facets.len(), "colors": alm0.colors.len(),
            "support_reaction_N": [s.rs.x, s.rs.y]}));
        // --- S1 / S3 cold start
        for (pi, (name, p)) in pol.iter().enumerate() {
            let mut alm = Alm::new(&m, *p, None);
            let mut q = m.q.clone();
            let mut hist = Vec::new();
            let mut at = serde_json::Map::new();
            for it in 1..=nmax {
                alm.iterate(&m, &mut q, 0.0, None);
                let log = it <= 32 || it % 25 == 0 || it == 8 || it == 16;
                if log {
                    let e = errs(&m, &q, Some(&alm), &s);
                    if seed == 1 {
                        row(&mut csv, pi, it, &alm, &e);
                    }
                    let best = e.f_dual.min(if matches!(p, RhoPolicy::Penalty) { e.f_const } else { e.f_dual });
                    hist.push((it, best));
                    if it == 8 || it == 16 || it == 100 || it == 1000 || it == nmax {
                        at.insert(format!("it{it}"), json!({"f_dual": e.f_dual, "f_const": e.f_const, "disp": e.disp, "reaction": e.reaction, "energy": e.energy, "constit": e.constit}));
                    }
                }
            }
            res.push(json!({"seed": seed, "method": name, "iters_to_1pct": first_below(&hist, 0.01), "iters_to_10pct": first_below(&hist, 0.1), "at": at}));
        }
        // --- PCG
        {
            let mut pcg = Pcg::new(&m, 0.0, None);
            let mut hist = Vec::new();
            let mut at = serde_json::Map::new();
            let mask: Vec<bool> = m.facets.iter().map(|f| f.active).collect();
            for it in 1..=nmax {
                pcg.step();
                if it <= 400 || it % 25 == 0 {
                    let q = pcg.q(&m);
                    let e = force_err(&m.facet_forces(&q), &s.fs, &mask);
                    hist.push((it, e.l2));
                    if seed == 1 {
                        csv.row(&[pol.len() as f64, it as f64, it as f64, it as f64, 0.0, e.l2, e.l2, e.max, 0.0, 0.0, 0.0, 0.0]);
                    }
                    if it == 8 || it == 16 || it == 100 {
                        at.insert(format!("it{it}"), json!({"f": e.l2}));
                    }
                }
            }
            res.push(json!({"seed": seed, "method": "PCG block-Jacobi", "iters_to_1pct": first_below(&hist, 0.01), "iters_to_10pct": first_below(&hist, 0.1), "at": at}));
        }
        // --- S2 (quasi-static linear = S0) and its cost
        {
            let mut s2 = S2::new();
            let out = s2.reconstruct(&m, &m.q, 0.0, None).unwrap();
            let mask: Vec<bool> = m.facets.iter().map(|f| f.active).collect();
            let e = force_err(&out.forces, &s.fs, &mask);
            let alm = Alm::new(&m, RhoPolicy::Shared(1e-2), None);
            let mut q = m.q.clone();
            let mut a2 = alm;
            a2.sweep(&m, &mut q, 0.0, None);
            let per_sweep = a2.work.point_evals;
            res.push(json!({"seed": seed, "method": "S2 (quasi-static, = S0 equations)", "force_err_l2": e.l2,
                "nnz_L": s2.stats.nnz_l, "factor_flops_est": s2.stats.flops_factor, "dofs": 3 * m.n_cells(),
                "point_evals_per_sweep": per_sweep, "solve_flops_est": 4.0 * s2.stats.nnz_l as f64}));
        }
    }
    let ranges = experiments::ranges_json();

    // --- ramp (seed 1): load factor 0→1 over n steps, m iterations per step, warm start
    let (mut m, _t) = setup(1);
    m.pattern = m.loads.clone();
    m.loads.iter_mut().for_each(|l| *l = [0.0; 3]);
    let mut mref = m.clone();
    mref.loads = mref.pattern.clone();
    let s = solve_ref(&mref);
    let mut ramp = Vec::new();
    for (name, p) in [("shared c=0.01", RhoPolicy::Shared(1e-2)), ("shared c=1", RhoPolicy::Shared(1.0)), ("perspring c=1", RhoPolicy::PerSpring(1.0)), ("S3 penalty", RhoPolicy::Penalty)] {
        for &(n, mm) in &[(60usize, 8usize), (60, 16), (600, 8), (600, 16)] {
            let mut alm = Alm::new(&m, p, None);
            let mut q = m.q.clone();
            for k in 1..=n {
                let lf = k as f64 / n as f64;
                for _ in 0..mm {
                    alm.iterate(&m, &mut q, lf, None);
                }
            }
            let e = errs(&mref, &q, Some(&alm), &s);
            ramp.push(json!({"method": name, "steps": n, "iters_per_step": mm, "f_dual": e.f_dual, "f_const": e.f_const, "disp": e.disp, "reaction": e.reaction}));
        }
    }

    // --- sudden change: remove the middle 40 % of the support facets from a converged state
    let (m, _t) = setup(1);
    let s_before = solve_ref(&m);
    let mut m2 = m.clone();
    let mut removed = 0;
    for f in m2.facets.iter_mut() {
        if f.tag == 1 && f.pc.x > 0.3 && f.pc.x < 0.7 {
            f.active = false;
            removed += 1;
        }
    }
    let s_after = solve_ref(&m2);
    let mut csv2 = Csv::new(EXP, "support_removal", &HEAD);
    let mut sudden = Vec::new();
    for (pi, (name, p)) in pol.iter().enumerate() {
        let mut alm = Alm::new(&m, *p, None);
        alm.set_lambda_from_q(&m, &s_before.qs);
        let mut q = s_before.qs.clone();
        // new topology: inactive facets drop out of S1 (adjacency rebuilt)
        let mut alm2 = Alm::new(&m2, *p, None);
        alm2.lambda = alm.lambda.clone();
        let mut hist = Vec::new();
        let e0 = errs(&m2, &q, Some(&alm2), &s_after);
        let mut at = serde_json::Map::new();
        at.insert("it0".into(), json!({"f_dual": e0.f_dual, "f_const": e0.f_const}));
        for it in 1..=nmax {
            alm2.iterate(&m2, &mut q, 0.0, None);
            if it <= 32 || it % 25 == 0 || it == 8 || it == 16 {
                let e = errs(&m2, &q, Some(&alm2), &s_after);
                row(&mut csv2, pi, it, &alm2, &e);
                hist.push((it, if matches!(p, RhoPolicy::Penalty) { e.f_const } else { e.f_dual }));
                if it == 8 || it == 16 || it == 100 || it == 1000 {
                    at.insert(format!("it{it}"), json!({"f_dual": e.f_dual, "f_const": e.f_const, "reaction": e.reaction}));
                }
            }
        }
        sudden.push(json!({"method": name, "iters_to_1pct": first_below(&hist, 0.01), "at": at}));
    }

    let out = json!({
        "meta": meta(EXP, "all", Some(1), "S1: colour-sequential primal sweep (1 GN step per cell) + simultaneous dual; S3: rho=inf; PCG: block-Jacobi on reference K",
            json!({"block": "1.0 x 0.5 m, b=0.1, dmin=0.05", "load": "top-left 40%: (+0.5, -2.0) MPa", "support": "bottom edge clamped (ground facets with shear)",
                   "elastic": true, "seeds": seeds, "nmax": nmax, "support_facets_removed": removed})),
        "info": info,
        "cold": res,
        "ramp": ramp,
        "support_removal": sudden,
        "methods": pol.iter().map(|p| p.0.clone()).chain(std::iter::once("PCG block-Jacobi".to_string())).collect::<Vec<_>>(),
        "ranges": ranges,
    });
    write_json(EXP, "summary", &out);
    let short: Vec<Value> = out["cold"].as_array().unwrap().iter().filter(|r| r["seed"] == 1).cloned().collect();
    println!("{}", serde_json::to_string_pretty(&short).unwrap());
    println!("{}", serde_json::to_string_pretty(&out["support_removal"]).unwrap());
    println!("{}", serde_json::to_string_pretty(&out["ramp"]).unwrap());
}
