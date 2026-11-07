//! E3 — scaling of the work needed for 1 % facet-force error.
//! (a) wall height N at fixed cell size; (b) fixed wall, refined tessellation;
//! (c) dynamic first step after a sudden load vs (c·h/ℓ)² (labelled E3-dyn).

use experiments::{edge_traction, force_err, meta, rect_tess, write_json, Csv};
use geometry::V2;
use model::{BoundaryAttach, FacetForce, LawOptions, Material, Model};
use serde_json::json;
use solvers::assembly::Inertia;
use solvers::cg::Pcg;
use solvers::integrator::{Dyn, Scheme};
use solvers::s0::{newton, NewtonOpts};
use solvers::s1::{Alm, RhoPolicy};
use solvers::s2::S2;

const EXP: &str = "E3";

fn wall(w: f64, h: f64, dmin: f64, seed: u64) -> Model {
    let t = rect_tess(V2::new(0.0, 0.0), V2::new(w, h), dmin, seed);
    let mut m = Model::from_tessellation(&t, Material::concrete(0.2), LawOptions::elastic(), &[]);
    let g = m.ground();
    m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: None, body: g, shear: true, tag: 1 });
    // top load: −10 kPa normal + 2.5 kPa shear over the full top edge. Small on
    // purpose: geometric nonlinearity stays ≲1e-4, so the linear PCG/S2 and the
    // corotated S0/S1 solve effectively the same equations up to 16 m height.
    edge_traction(&mut m, &t, 2, None, V2::new(2.5e3, -1.0e4), false);
    m
}

#[derive(Clone, Copy)]
enum Method {
    S1(RhoPolicy),
    Pcg,
}

fn name(m: Method) -> String {
    match m {
        Method::S1(RhoPolicy::Penalty) => "S3 penalty".into(),
        Method::S1(RhoPolicy::Shared(c)) => format!("S1 shared c={c}"),
        Method::S1(RhoPolicy::PerSpring(c)) => format!("S1 perspring c={c}"),
        Method::S1(RhoPolicy::Dynamic(c)) => format!("S1 dynamic c={c}"),
        Method::Pcg => "PCG block-Jacobi".into(),
    }
}

/// Iterations until the facet-force error first drops below each threshold.
fn iters_to(m: &Model, fs: &[FacetForce], meth: Method, inertia: Option<&Inertia>, max_it: usize) -> (Option<usize>, Option<usize>, f64, f64) {
    let mask: Vec<bool> = m.facets.iter().map(|f| f.active).collect();
    let mut hit1 = None;
    let mut hit10 = None;
    let mut e8 = f64::NAN;
    let mut e16 = f64::NAN;
    let check = |i: usize| i <= 64 || i % (i / 64).max(1) == 0;
    match meth {
        Method::S1(p) => {
            let mut alm = Alm::new(m, p, inertia);
            let mut q = m.q.clone();
            for it in 1..=max_it {
                alm.iterate(m, &mut q, 0.0, inertia);
                if check(it) {
                    let f = if matches!(p, RhoPolicy::Penalty) { m.facet_forces(&q) } else { alm.dual_forces(m) };
                    let e = force_err(&f, fs, &mask).l2;
                    if it == 8 {
                        e8 = e;
                    }
                    if it == 16 {
                        e16 = e;
                    }
                    if hit10.is_none() && e <= 0.1 {
                        hit10 = Some(it);
                    }
                    if e <= 0.01 {
                        hit1 = Some(it);
                        break;
                    }
                }
            }
        }
        Method::Pcg => {
            let mut pcg = Pcg::new(m, 0.0, inertia);
            for it in 1..=max_it {
                pcg.step();
                if check(it) {
                    let q = pcg.q(m);
                    let e = force_err(&m.facet_forces(&q), fs, &mask).l2;
                    if it == 8 {
                        e8 = e;
                    }
                    if it == 16 {
                        e16 = e;
                    }
                    if hit10.is_none() && e <= 0.1 {
                        hit10 = Some(it);
                    }
                    if e <= 0.01 {
                        hit1 = Some(it);
                        break;
                    }
                }
            }
        }
    }
    (hit1, hit10, e8, e16)
}

fn reference(m: &Model, inertia: Option<&Inertia>) -> Vec<FacetForce> {
    let mut q = m.q.clone();
    let fmax = m.loads.iter().flatten().fold(0.0f64, |s, x| s.max(x.abs()));
    let rep = newton(m, &mut q, 0.0, inertia, NewtonOpts { tol: 1e-10 * fmax, max_iter: 30 });
    assert!(rep.converged, "S0 reference failed: {rep:?} (fmax {fmax})");
    m.facet_forces(&q)
}

fn direct_cost(m: &Model) -> (usize, f64, usize) {
    let mut s2 = S2::new();
    s2.reconstruct(m, &m.q, 0.0, None).unwrap();
    (s2.stats.nnz_l, s2.stats.flops_factor, 3 * m.n_cells())
}

fn main() {
    let static_methods = [
        Method::S1(RhoPolicy::PerSpring(0.1)),
        Method::S1(RhoPolicy::Shared(0.1)),
        Method::S1(RhoPolicy::Penalty),
        Method::Pcg,
    ];
    let mut out = serde_json::Map::new();

    // (a) height scaling at fixed cell size
    let mut csv = Csv::new(EXP, "height", &["method", "height_m", "cells", "dofs", "iters_1pct", "iters_10pct", "err8", "err16", "nnz_L", "factor_flops", "point_evals_per_sweep"]);
    let mut a = Vec::new();
    for &h in &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] {
        let m = wall(1.0, h, 0.1, 1);
        let fs = reference(&m, None);
        let (nnz, fl, dofs) = direct_cost(&m);
        let pe = {
            let mut alm = Alm::new(&m, RhoPolicy::Penalty, None);
            let mut q = m.q.clone();
            alm.sweep(&m, &mut q, 0.0, None);
            alm.work.point_evals
        };
        for (mi, &meth) in static_methods.iter().enumerate() {
            let local = !matches!(meth, Method::Pcg);
            if local && h > 8.0 {
                continue;
            }
            let cap = if local { 300_000 } else { 200_000 };
            let (i1, i10, e8, e16) = iters_to(&m, &fs, meth, None, cap);
            csv.row(&[mi as f64, h, m.n_cells() as f64, dofs as f64, i1.map(|x| x as f64).unwrap_or(f64::NAN), i10.map(|x| x as f64).unwrap_or(f64::NAN), e8, e16, nnz as f64, fl, pe as f64]);
            a.push(json!({"method": name(meth), "height_m": h, "cells": m.n_cells(), "iters_1pct": i1, "iters_10pct": i10, "err8": e8, "err16": e16, "nnz_L": nnz, "factor_flops": fl, "point_evals_per_sweep": pe}));
            eprintln!("height {h} {} -> {:?}", name(meth), i1);
        }
    }
    out.insert("height".into(), json!(a));

    // (b) refinement of a fixed 1 m × 2 m wall
    let mut csv = Csv::new(EXP, "refine", &["method", "dmin", "cells", "dofs", "iters_1pct", "iters_10pct", "err8", "err16", "nnz_L", "factor_flops", "point_evals_per_sweep"]);
    let mut b = Vec::new();
    for &d in &[0.2, 0.14, 0.1, 0.07, 0.05, 0.035] {
        let m = wall(1.0, 2.0, d, 1);
        let fs = reference(&m, None);
        let (nnz, fl, dofs) = direct_cost(&m);
        let pe = {
            let mut alm = Alm::new(&m, RhoPolicy::Penalty, None);
            let mut q = m.q.clone();
            alm.sweep(&m, &mut q, 0.0, None);
            alm.work.point_evals
        };
        for (mi, &meth) in static_methods.iter().enumerate() {
            let cap = if matches!(meth, Method::Pcg) { 200_000 } else { 60_000 };
            let (i1, i10, e8, e16) = iters_to(&m, &fs, meth, None, cap);
            csv.row(&[mi as f64, d, m.n_cells() as f64, dofs as f64, i1.map(|x| x as f64).unwrap_or(f64::NAN), i10.map(|x| x as f64).unwrap_or(f64::NAN), e8, e16, nnz as f64, fl, pe as f64]);
            b.push(json!({"method": name(meth), "dmin": d, "cells": m.n_cells(), "iters_1pct": i1, "iters_10pct": i10, "err8": e8, "err16": e16, "nnz_L": nnz, "factor_flops": fl}));
            eprintln!("refine {d} {} -> {:?}", name(meth), i1);
        }
    }
    out.insert("refine".into(), json!(b));

    // (c) E3-dyn: first implicit-Euler step after a sudden load, game-scale cells
    let mut csv = Csv::new(EXP, "dyn", &["method", "h", "dmin", "pi_h", "cells", "iters_1pct", "iters_10pct", "err8", "err16"]);
    let mut c = Vec::new();
    let dyn_methods = [
        Method::S1(RhoPolicy::Dynamic(1.0)),
        Method::S1(RhoPolicy::PerSpring(0.1)),
        Method::S1(RhoPolicy::Penalty),
        Method::Pcg,
    ];
    for &(wd, ht, dmin) in &[(4.0, 8.0, 0.4), (1.0, 2.0, 0.1)] {
        let m = wall(wd, ht, dmin, 1);
        let mat = m.mat;
        // mean generator distance as ℓ
        let ell = m.facets.iter().filter(|f| f.b < m.n_cells()).map(|f| f.h).sum::<f64>() / m.facets.iter().filter(|f| f.b < m.n_cells()).count() as f64;
        for &hinv in &[60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0, 7680.0, 15360.0] {
            let h = 1.0 / hinv;
            let dynst = Dyn::new(Scheme::ImplicitEuler, h, &m);
            let ine = dynst.inertia(&m.q);
            let fs = reference(&m, Some(&ine));
            let pi_h = (mat.wave_speed() * h / ell).powi(2);
            for (mi, &meth) in dyn_methods.iter().enumerate() {
                let (i1, i10, e8, e16) = iters_to(&m, &fs, meth, Some(&ine), 100_000);
                csv.row(&[mi as f64, h, dmin, pi_h, m.n_cells() as f64, i1.map(|x| x as f64).unwrap_or(f64::NAN), i10.map(|x| x as f64).unwrap_or(f64::NAN), e8, e16]);
                c.push(json!({"method": name(meth), "h": h, "dmin": dmin, "ell_mean": ell, "pi_h": pi_h, "cells": m.n_cells(), "iters_1pct": i1, "iters_10pct": i10, "err8": e8, "err16": e16,
                    "sweeps_per_frame_60Hz_at_1pct": i1.map(|x| x as f64 * hinv / 60.0)}));
                eprintln!("dyn d={dmin} h=1/{hinv} {} -> {:?}", name(meth), i1);
            }
        }
    }
    out.insert("dyn".into(), json!(c));

    let res = json!({
        "meta": meta(EXP, "all", Some(1), "as E2; dynamic: implicit Euler, first step from rest under sudden load",
            json!({"wall": "width 1 m (dyn: 4 m), b=0.2, clamped base, top traction (+0.25,-1.0) MPa", "threshold": "L2 facet-force error 1%", "caps": "local methods: 300k (height, skipped above 8 m), 60k (refine); PCG 200k"})),
        "methods_static": static_methods.iter().map(|&m| name(m)).collect::<Vec<_>>(),
        "methods_dyn": dyn_methods.iter().map(|&m| name(m)).collect::<Vec<_>>(),
        "results": out,
    });
    write_json(EXP, "summary", &res);
}
