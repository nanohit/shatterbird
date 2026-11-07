//! E4 — elastic patch tests: uniform strain exactness, uniaxial E/ν, rigid
//! motions (objectivity), incremental rotation drift.

use experiments::{meta, rect_tess, write_json, Csv};
use geometry::voronoi2d::{EdgeTag, Tessellation};
use geometry::V2;
use model::{BoundaryAttach, LawOptions, Material, Model};
use serde_json::{json, Value};
use solvers::s0::{body_reaction, newton, NewtonOpts};

const EXP: &str = "E4";

fn boundary_cells(t: &Tessellation) -> Vec<bool> {
    t.cells.iter().map(|c| c.tags.iter().any(|g| matches!(g, EdgeTag::Boundary(_)))).collect()
}

/// Max deviation of facet traction from σn over interior facets, relative to ‖σ‖.
fn traction_dev(m: &Model, q: &[[f64; 3]], sig: [[f64; 2]; 2], mask: &[bool]) -> (f64, f64) {
    let smax = sig.iter().flatten().fold(0.0f64, |a, &b| a.max(b.abs()));
    let mut worst: f64 = 0.0;
    let mut worst_m: f64 = 0.0;
    for (fi, f) in m.facets.iter().enumerate() {
        if !mask[fi] {
            continue;
        }
        let ff = m.facet_force(q, fi);
        let a = f.len * m.mat.thickness;
        // reference-frame traction (q is a small perturbation of the reference)
        let tr = (f.n0 * ff.n + f.t0 * ff.v) / a;
        let ex = V2::new(sig[0][0] * f.n0.x + sig[0][1] * f.n0.y, sig[1][0] * f.n0.x + sig[1][1] * f.n0.y);
        worst = worst.max((tr - ex).norm() / smax);
        worst_m = worst_m.max(ff.m.abs() / (smax * a * f.len));
    }
    (worst, worst_m)
}

fn opts(scale: f64) -> NewtonOpts {
    NewtonOpts { tol: 1e-11 * scale, max_iter: 30 }
}

fn uniform_strain(seed: u64, gamma: f64, at_centroid: bool) -> Value {
    let lo = V2::new(0.0, 0.0);
    let hi = V2::new(1.0, 1.0);
    let t = rect_tess(lo, hi, 0.06, seed);
    let mut mat = Material::concrete(0.1);
    mat.gamma = gamma;
    let mut m = Model::from_tessellation(&t, mat, LawOptions::elastic(), &[]);
    let eps = [[1e-4, 0.4e-4], [0.4e-4, -0.6e-4]];
    let bc = boundary_cells(&t);
    let apply = |p: V2| V2::new(eps[0][0] * p.x + eps[0][1] * p.y, eps[1][0] * p.x + eps[1][1] * p.y);
    for (i, &b) in bc.iter().enumerate() {
        if b {
            m.bodies[i].fixed = [true; 3];
            let src = if at_centroid { m.bodies[i].x0 } else { m.bodies[i].generator };
            let u = apply(src);
            m.q[i] = [m.bodies[i].x0.x + u.x, m.bodies[i].x0.y + u.y, 0.0];
        }
    }
    let fscale = mat.e * 1e-4 * 0.06 * 0.1;
    let mut q = m.q.clone();
    let rep = newton(&m, &mut q, 0.0, None, opts(fscale));
    let interior: Vec<bool> = m.facets.iter().map(|f| f.b < t.cells.len() && !bc[f.a] && !bc[f.b]).collect();
    let sig = [[mat.e * eps[0][0], mat.e * eps[0][1]], [mat.e * eps[1][0], mat.e * eps[1][1]]];
    let (dev, devm) = traction_dev(&m, &q, sig, &interior);
    // interior DOF error vs the generator-following field
    let mut qerr: f64 = 0.0;
    for i in 0..t.cells.len() {
        if !bc[i] {
            let u = apply(m.bodies[i].generator);
            let e = V2::new(q[i][0] - m.bodies[i].x0.x - u.x, q[i][1] - m.bodies[i].x0.y - u.y).norm();
            qerr = qerr.max(e / (1e-4 * 1.0));
        }
    }
    json!({"seed": seed, "gamma": gamma, "drive": if at_centroid {"centroid (naive)"} else {"generator rigid motion"},
        "cells": t.cells.len(), "newton_iters": rep.iters, "res_over_force_scale": rep.res / fscale, "converged": rep.converged, "roundoff_limited": rep.roundoff_limited,
        "max_traction_dev_rel": dev, "max_facet_moment_rel": devm, "max_interior_dof_err_rel": qerr})
}

/// Uniaxial tension: left roller wall, right roller plate pulled by u; one y gauge.
fn uniaxial(seed: u64, gamma: f64, dmin: f64) -> (Value, Model, Vec<[f64; 3]>, Tessellation) {
    let t = rect_tess(V2::new(0.0, 0.0), V2::new(1.0, 1.0), dmin, seed);
    let mut mat = Material::concrete(0.1);
    mat.gamma = gamma;
    let mut m = Model::from_tessellation(&t, mat, LawOptions::elastic(), &[]);
    let g = m.ground();
    m.attach_boundary(&t, &BoundaryAttach { edge: 3, range: None, body: g, shear: false, tag: 1 });
    let plate = m.add_plate(V2::new(1.0, 0.5), 0.0, 0.0, [true; 3]);
    m.attach_boundary(&t, &BoundaryAttach { edge: 1, range: None, body: plate, shear: false, tag: 2 });
    // y gauge on the cell nearest the origin
    let i0 = (0..t.cells.len()).min_by(|&a, &b| m.bodies[a].x0.norm().partial_cmp(&m.bodies[b].x0.norm()).unwrap()).unwrap();
    m.bodies[i0].fixed[1] = true;
    let u = 1e-4;
    m.q[plate][0] += u;
    let mut q = m.q.clone();
    let rep = newton(&m, &mut q, 0.0, None, opts(mat.e * 1e-4 * dmin * 0.1));
    let fx = -body_reaction(&m, &q, plate)[0];
    let sxx = fx / (1.0 * mat.thickness);
    let e_eff = sxx / (u / 1.0);
    // regression of generator displacement
    let nc = t.cells.len();
    let (mut sy, mut syy, mut suy, mut syuy, mut sx, mut sxx2, mut sux, mut sxux) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..nc {
        let gx = m.bodies[i].generator;
        let arm = gx - m.bodies[i].x0;
        let th = q[i][2];
        let pos = V2::new(q[i][0], q[i][1]) + arm.rot(th);
        let d = pos - gx;
        sy += gx.y;
        syy += gx.y * gx.y;
        suy += d.y;
        syuy += gx.y * d.y;
        sx += gx.x;
        sxx2 += gx.x * gx.x;
        sux += d.x;
        sxux += gx.x * d.x;
    }
    let n = nc as f64;
    let eyy = (n * syuy - sy * suy) / (n * syy - sy * sy);
    let exx = (n * sxux - sx * sux) / (n * sxx2 - sx * sx);
    let nu = -eyy / exx;
    let nu_voigt = (1.0 - gamma) / (3.0 + gamma);
    let e_voigt = mat.e * (3.0 + gamma) * (1.0 - nu_voigt * nu_voigt) / 4.0;
    // cell stress σ_xx coefficient of variation (Love–Weber), interior cells
    let bc = boundary_cells(&t);
    let mut sv = Vec::new();
    for i in 0..nc {
        if bc[i] {
            continue;
        }
        let mut s = 0.0;
        for f in m.facets.iter().filter(|f| f.a == i || f.b == i) {
            for p in 0..2 {
                let (j, e) = m.eval_point(&q, f, p);
                let _ = j;
                let pt = f.pts[p];
                let arm = if f.a == i { pt.arm_a } else { pt.arm_b };
                let fv = (f.n0 * e.fnorm + f.t0 * e.ftan) * if f.a == i { 1.0 } else { -1.0 };
                s += arm.x * fv.x;
            }
        }
        sv.push(s / (t.cells[i].props.area * mat.thickness));
    }
    let mean = sv.iter().sum::<f64>() / sv.len() as f64;
    let sd = (sv.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / sv.len() as f64).sqrt();
    (
        json!({"seed": seed, "gamma": gamma, "dmin": dmin, "cells": nc, "newton_iters": rep.iters, "converged": rep.converged,
        "E_eff_over_E": e_eff / mat.e, "nu_eff": nu, "eps_xx_regression": exx, "E_voigt_over_E": e_voigt / mat.e, "nu_voigt": nu_voigt,
        "cell_sxx_mean_over_Eeps": mean / (mat.e * u), "cell_sxx_cov": sd / mean.abs()}),
        m,
        q,
        t,
    )
}

fn rigidly(m: &Model, q: &[[f64; 3]], c: V2, alpha: f64, tr: V2) -> Vec<[f64; 3]> {
    (0..m.bodies.len())
        .map(|i| {
            let p = V2::new(q[i][0], q[i][1]);
            let r = c + (p - c).rot(alpha) + tr;
            [r.x, r.y, q[i][2] + alpha]
        })
        .collect()
}

fn main() {
    let mut out = serde_json::Map::new();

    // (a) uniform strain exactness
    let mut a = Vec::new();
    for seed in [1u64, 2, 3] {
        a.push(uniform_strain(seed, 1.0, false));
        a.push(uniform_strain(seed, 1.0, true));
        a.push(uniform_strain(seed, 0.5, false));
    }
    out.insert("uniform_strain".into(), json!(a));

    // (b) uniaxial tension: E_eff, ν_eff vs γ
    let mut b = Vec::new();
    for &gamma in &[1.0, 0.75, 0.5, 0.25, 0.1] {
        for seed in [1u64, 2, 3] {
            b.push(uniaxial(seed, gamma, 0.05).0);
        }
    }
    out.insert("uniaxial".into(), json!(b));

    // (c) rigid motions without solve
    let t = rect_tess(V2::new(0.0, 0.0), V2::new(1.0, 1.0), 0.05, 1);
    let m = Model::from_tessellation(&t, Material::concrete(0.1), LawOptions::elastic(), &[]);
    let fail = m.mat.sigma_t * 0.05 * 0.1 / 2.0; // ~failure force of a point
    let (_, ml, ql, _) = uniaxial(1, 0.5, 0.05);
    let fl = ml.facet_forces(&ql);
    let fmax = fl.iter().fold(0.0f64, |s, f| s.max(f.n.abs()).max(f.v.abs()));
    let mut csv = Csv::new(EXP, "rigid", &["alpha_deg", "translation_m", "unloaded_max_force_over_failure", "loaded_max_dev_over_maxforce"]);
    let mut c = Vec::new();
    for &deg in &[0.0, 1e-3, 1.0, 10.0, 45.0, 90.0, 135.0, 179.0, 180.0] {
        for &tr in &[0.0, 1.0, 1e3, 1e6] {
            let alpha = deg * std::f64::consts::PI / 180.0;
            let tv = V2::new(1.0, 1.0).normalized() * tr;
            let q = rigidly(&m, &m.q, V2::new(0.5, 0.5), alpha, tv);
            let f = m.facet_forces(&q);
            let un = f.iter().fold(0.0f64, |s, x| s.max(x.n.abs()).max(x.v.abs()));
            let ql2 = rigidly(&ml, &ql, V2::new(0.5, 0.5), alpha, tv);
            let f2 = ml.facet_forces(&ql2);
            let dev = f2.iter().zip(&fl).fold(0.0f64, |s, (x, y)| s.max((x.n - y.n).abs()).max((x.v - y.v).abs()));
            csv.row(&[deg, tr, un / fail, dev / fmax]);
            c.push(json!({"alpha_deg": deg, "translation_m": tr, "unloaded_max_force_over_failure": un / fail, "loaded_max_dev_over_maxforce": dev / fmax}));
        }
    }
    out.insert("rigid_motion".into(), json!(c));

    // (d) incremental rotation with Newton solves: unloaded and pre-stretched, 10 cycles 0→180°→0
    let mut d = Vec::new();
    let mut csv = Csv::new(EXP, "incremental", &["case", "step", "alpha_deg", "max_force_err_rel"]);
    for (case, stretch) in [(0.0, 0.0), (1.0, 1e-4)] {
        let mut m = Model::from_tessellation(&t, Material::concrete(0.1), LawOptions::elastic(), &[]);
        let lp = m.add_plate(V2::new(0.0, 0.5), 0.0, 0.0, [true; 3]);
        m.attach_boundary(&t, &BoundaryAttach { edge: 3, range: None, body: lp, shear: true, tag: 1 });
        let rp = m.add_plate(V2::new(1.0, 0.5), 0.0, 0.0, [true; 3]);
        m.attach_boundary(&t, &BoundaryAttach { edge: 1, range: None, body: rp, shear: false, tag: 2 });
        let c0 = V2::new(0.5, 0.5);
        let place = |m: &mut Model, alpha: f64| {
            let l = c0 + (V2::new(0.0, 0.5) - c0).rot(alpha);
            let r = c0 + (V2::new(1.0 + stretch, 0.5) - c0).rot(alpha);
            m.q[lp] = [l.x, l.y, alpha];
            m.q[rp] = [r.x, r.y, alpha];
        };
        place(&mut m, 0.0);
        let fs = m.mat.e * 1e-4 * 0.05 * 0.1;
        let mut q = m.q.clone();
        newton(&m, &mut q, 0.0, None, opts(fs));
        let f0 = m.facet_forces(&q);
        let scale = if stretch > 0.0 { f0.iter().fold(0.0f64, |s, f| s.max(f.n.abs())) } else { fail };
        let mut worst: f64 = 0.0;
        let mut iters = 0usize;
        let mut step = 0u64;
        for _cycle in 0..10 {
            for k in (1..=180).chain((0..180).rev()) {
                let alpha = k as f64 * std::f64::consts::PI / 180.0;
                place(&mut m, alpha);
                for b in [lp, rp] {
                    q[b] = m.q[b];
                }
                let rep = newton(&m, &mut q, 0.0, None, opts(fs));
                iters += rep.iters;
                let f = m.facet_forces(&q);
                let dev = f.iter().zip(&f0).fold(0.0f64, |s, (x, y)| s.max((x.n - y.n).abs()).max((x.v - y.v).abs()));
                worst = worst.max(dev / scale);
                step += 1;
                if step % 10 == 0 {
                    csv.row(&[case, step as f64, k as f64, dev / scale]);
                }
            }
        }
        d.push(json!({"case": if stretch > 0.0 {"pre-stretched 1e-4"} else {"unloaded"}, "steps": step, "newton_iters_total": iters,
            "max_force_dev_rel": worst, "scale": if stretch > 0.0 {"max facet normal force"} else {"point failure force"}}));
    }
    out.insert("incremental_rotation".into(), json!(d));

    let res = json!({
        "meta": meta(EXP, "all", Some(1), "S0 Newton (GN tangent), sparse Cholesky", json!({"block": "1x1 m, b=0.1 m", "material": Material::concrete(0.1)})),
        "results": Value::Object(out),
    });
    write_json(EXP, "summary", &res);
    println!("{}", serde_json::to_string_pretty(&res["results"]).unwrap());
}
