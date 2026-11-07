//! E8 — time discretisation: per-step amplification of the implemented integrators
//! (extracted as the one-step map G of the actual code, compared with closed forms),
//! surviving frequency bands, and a bar under a step load (dynamic amplification).

use experiments::{meta, support_reaction, write_json, Csv};
use faer::Mat;
use geometry::voronoi2d::{rect, voronoi_convex};
use geometry::V2;
use model::{BoundaryAttach, LawOptions, Material, Model};
use serde_json::json;
use solvers::integrator::{Dyn, Scheme};
use solvers::s0::{newton, NewtonOpts};

const EXP: &str = "E8";
const SCHEMES: [Scheme; 3] = [Scheme::ImplicitEuler, Scheme::Bdf2, Scheme::Newmark];

fn sdof(ell: f64) -> Model {
    // centred at the origin: no absolute-coordinate cancellation in the extraction
    let t = voronoi_convex(&[V2::new(0.0, 0.0)], &rect(V2::new(-ell / 2.0, -ell / 2.0), V2::new(ell / 2.0, ell / 2.0)));
    let mut m = Model::from_tessellation(&t, Material::concrete(0.1), LawOptions::elastic(), &[]);
    let g = m.ground();
    m.attach_boundary(&t, &BoundaryAttach { edge: 3, range: None, body: g, shear: true, tag: 1 });
    m.bodies[0].fixed = [false, true, true];
    m
}

fn analytic(s: Scheme, om: f64) -> (f64, f64) {
    use num_complex::Complex64 as C;
    let i = C::new(0.0, 1.0);
    let z = match s {
        Scheme::ImplicitEuler => 1.0 / (1.0 - i * om),
        Scheme::Bdf2 => (2.0 + (1.0 + 2.0 * i * om).sqrt()) / (3.0 - 2.0 * i * om),
        Scheme::Newmark => (1.0 + i * om / 2.0) / (1.0 - i * om / 2.0),
    };
    (z.norm(), z.arg())
}

/// Principal root of the one-step map extracted from the implementation.
fn measured(s: Scheme, h: f64, m: &Model) -> (f64, f64) {
    let x0 = m.bodies[0].x0.x;
    let mut g = Mat::<f64>::zeros(5, 5);
    for j in 0..5 {
        let mut st = Dyn::new(s, h, m);
        st.steps = 1;
        // perturbation scaled to each state's units so every column moves q by ~1e-6 m
        let sc = [1e-6, 1e-6, 1e-6 / h, 1e-6 / h, 1e-6 / (h * h)];
        let mut e = [0.0; 5];
        e[j] = sc[j];
        let mut q = m.q.clone();
        q[0][0] = x0 + e[0];
        st.q_prev = m.q.clone();
        st.q_prev[0][0] = x0 + e[1];
        st.v[0][0] = e[2];
        st.v_prev[0][0] = e[3];
        st.a[0][0] = e[4];
        let ine = st.inertia(&q);
        let mut qn = q.clone();
        newton(m, &mut qn, 0.0, Some(&ine), NewtonOpts { tol: 1e-16, max_iter: 3 });
        st.finish(&q, &qn);
        let out = [qn[0][0] - x0, st.q_prev[0][0] - x0, st.v[0][0], st.v_prev[0][0], st.a[0][0]];
        for r in 0..5 {
            // D⁻¹ G D with D = diag(sc): same eigenvalues, well scaled
            g[(r, j)] = out[r] / sc[r];
        }
    }
    let ev = g.eigenvalues().unwrap();
    let p = ev.iter().filter(|z| z.im > 0.0).max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap()).copied();
    match p {
        Some(z) => (z.norm(), z.arg()),
        None => {
            // overdamped to real roots: report the dominant one
            let z = ev.iter().max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap()).unwrap();
            (z.norm(), z.arg())
        }
    }
}

fn bar(n: usize, ell: f64) -> Model {
    let seeds: Vec<V2> = (0..n).map(|i| V2::new((i as f64 + 0.5) * ell, 0.5 * ell)).collect();
    let t = voronoi_convex(&seeds, &rect(V2::new(0.0, 0.0), V2::new(n as f64 * ell, ell)));
    let mut m = Model::from_tessellation(&t, Material::concrete(ell), LawOptions::elastic(), &[]);
    let g = m.ground();
    m.attach_boundary(&t, &BoundaryAttach { edge: 3, range: None, body: g, shear: true, tag: 1 });
    m
}

fn main() {
    let mut out = serde_json::Map::new();
    // (a) amplification: implementation vs closed form
    let m = sdof(0.5);
    let k: f64 = m.facets.iter().flat_map(|f| f.pts.iter()).map(|p| p.law.kn).sum();
    let omega = (k / m.bodies[0].mass).sqrt();
    let mut csv = Csv::new(EXP, "amplification", &["scheme", "Omega", "abs_meas", "arg_meas", "abs_an", "arg_an"]);
    let mut worst = serde_json::Map::new();
    for (si, &s) in SCHEMES.iter().enumerate() {
        let mut w: f64 = 0.0;
        for i in 0..=60 {
            let om = 10f64.powf(-2.0 + 3.5 * i as f64 / 60.0);
            let h = om / omega;
            let (am, pm) = measured(s, h, &m);
            let (aa, pa) = analytic(s, om);
            w = w.max((am - aa).abs() / aa).max((pm - pa).abs() / pa.abs().max(1e-300));
            csv.row(&[si as f64, om, am, pm, aa, pa]);
        }
        worst.insert(s.name().into(), json!(w));
    }
    out.insert("amplification_max_rel_dev".into(), json!(worst));

    // (b) surviving band per scheme and substep
    let mut band = Vec::new();
    let mut csv = Csv::new(EXP, "band", &["scheme", "h", "f_hz", "amp_per_period", "amp_after_1s", "period_ratio"]);
    for (si, &s) in SCHEMES.iter().enumerate() {
        for &hinv in &[240.0, 480.0, 960.0, 1920.0] {
            let h = 1.0 / hinv;
            let mut f50 = 0.0;
            let mut f90 = 0.0;
            for i in 0..=200 {
                let f = 10f64.powf(-0.5 + 3.0 * i as f64 / 200.0);
                let om = 2.0 * std::f64::consts::PI * f * h;
                let (a, p) = analytic(s, om);
                let per = (1.0 / (f * h)) as f64;
                let app = a.powf(per);
                let a1 = a.powf(hinv);
                csv.row(&[si as f64, h, f, app, a1, om / p]);
                if app >= 0.5 {
                    f50 = f;
                }
                if app >= 0.9 {
                    f90 = f;
                }
            }
            let at = |f: f64| {
                let om = 2.0 * std::f64::consts::PI * f * h;
                let (a, p) = analytic(s, om);
                json!({"f": f, "amp_per_period": a.powf(1.0 / (f * h)), "amp_after_1s": a.powf(hinv), "period_ratio": om / p})
            };
            band.push(json!({"scheme": s.name(), "h": h, "f_max_keeping_90pct_per_period": f90, "f_max_keeping_50pct_per_period": f50,
                "at_10Hz": at(10.0), "at_50Hz": at(50.0), "at_100Hz": at(100.0)}));
        }
    }
    out.insert("band".into(), json!(band));

    // (c) bar 10 m (20 cells of 0.5 m), clamped, step force at the free end
    let n = 20;
    let ell = 0.5;
    let mut mb = bar(n, ell);
    let f0 = 1.0e5;
    mb.loads[n - 1][0] = f0;
    let c = mb.mat.wave_speed();
    let l = n as f64 * ell;
    let t_end = 0.03;
    let mut csv = Csv::new(EXP, "bar", &["run", "t", "reaction_over_F0"]);
    let mut runs = Vec::new();
    let cases: Vec<(Scheme, f64)> = std::iter::once((Scheme::Newmark, 2e-6))
        .chain(SCHEMES.iter().flat_map(|&s| [240.0, 480.0, 960.0, 1920.0, 3840.0, 7680.0].into_iter().map(move |hi| (s, 1.0 / hi))))
        .collect();
    for (ri, &(s, h)) in cases.iter().enumerate() {
        let mut st = Dyn::new(s, h, &mb);
        let mut q = mb.q.clone();
        let steps = (t_end / h).round() as usize;
        let mut peak: f64 = 0.0;
        let mut arrival = None;
        let mut series = Vec::new();
        for k in 1..=steps {
            let ine = st.inertia(&q);
            let mut qn = q.clone();
            newton(&mb, &mut qn, 0.0, Some(&ine), NewtonOpts { tol: 1e-9 * f0, max_iter: 10 });
            st.finish(&q, &qn);
            q = qn;
            let r = support_reaction(&mb, &q, 1).x / f0;
            let t = k as f64 * h;
            peak = peak.max(r);
            if arrival.is_none() && r > 0.5 {
                arrival = Some(t);
            }
            if ri == 0 {
                if k % 50 == 0 {
                    csv.row(&[ri as f64, t, r]);
                }
            } else {
                csv.row(&[ri as f64, t, r]);
            }
            series.push(r);
        }
        runs.push(json!({"run": ri, "scheme": s.name(), "h": h, "steps": steps, "peak_reaction_over_F0": peak,
            "arrival_time_s": arrival, "reaction_at_end_over_F0": series.last(), "c_h_over_ell": c * h / ell}));
    }
    out.insert("bar".into(), json!({"L": l, "cells": n, "c": c, "transit_L_over_c_s": l / c, "f1_hz": c / (4.0 * l), "runs": runs}));

    let res = json!({
        "meta": meta(EXP, "all", None, "variational integrators (docs §4.4); S0 Newton per step",
            json!({"sdof": {"cell": 0.5, "omega_rad_s": omega}, "bar": {"cells": n, "ell": ell, "F0": f0}})),
        "results": out,
    });
    write_json(EXP, "summary", &res);
    println!("{}", serde_json::to_string_pretty(&res["results"]["amplification_max_rel_dev"]).unwrap());
    println!("{}", serde_json::to_string_pretty(&res["results"]["band"]).unwrap());
    for r in res["results"]["bar"]["runs"].as_array().unwrap() {
        println!("{} h={:.2e} peak={:.3} arrival={:?} end={:.3}", r["scheme"], r["h"].as_f64().unwrap(), r["peak_reaction_over_F0"].as_f64().unwrap(), r["arrival_time_s"], r["reaction_at_end_over_F0"].as_f64().unwrap());
    }
}
