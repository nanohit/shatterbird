//! E5 — single facet and small specimens: SB-IL1 energy closure.
//! (a) one facet under prescribed jump paths (monotonic, cyclic with compression,
//!     mixed-mode proportional/non-proportional) with load-step refinement;
//! (b) series bars (chains) under end-displacement and local-opening control,
//!     including structural snap-back (L > 2 l_ch).

use experiments::{meta, write_json, Csv};
use geometry::voronoi2d::{rect, voronoi_convex};
use geometry::V2;
use model::{BoundaryAttach, LawOptions, Material, Model};
use serde_json::{json, Value};
use solvers::s0::{body_reaction, max_damage_increment, newton_controlled, pattern_work, Control, NewtonOpts};

const EXP: &str = "E5";

/// Two rectangular cells sharing one facet at x = 0 (generator distance `ell`, facet length `l`).
fn pair(ell: f64, l: f64, law: LawOptions) -> Model {
    let seeds = [V2::new(-ell / 2.0, 0.0), V2::new(ell / 2.0, 0.0)];
    let t = voronoi_convex(&seeds, &rect(V2::new(-ell, -l / 2.0), V2::new(ell, l / 2.0)));
    let mut m = Model::from_tessellation(&t, Material::concrete(0.1), law, &[]);
    m.bodies[0].fixed = [true; 3];
    m.bodies[1].fixed = [true; 3];
    m
}

/// Drive body 1 through jump targets (normal, tangential) in the facet frame.
fn drive(m: &mut Model, path: &[(f64, f64)], csv: Option<(&mut Csv, f64)>) -> Value {
    let n0 = m.facets[0].n0;
    let t0 = m.facets[0].t0;
    let x1 = m.bodies[1].x0;
    let mut w_ext = 0.0;
    let mut prev_q = m.q.clone();
    let mut prev_r = [0.0; 3];
    let mut w_unload_new_d = 0.0f64;
    let mut max_eps: f64 = 0.0;
    let mut csv = csv;
    let mut prev_target = (0.0, 0.0);
    for (k, &(dn, dt)) in path.iter().enumerate() {
        let p = x1 + n0 * dn + t0 * dt;
        let mut q = m.q.clone();
        q[1] = [p.x, p.y, 0.0];
        // agent force = −(force of facets on body 1)
        let r = body_reaction(m, &q, 1);
        let ag = [-r[0], -r[1], -r[2]];
        let agp = [-prev_r[0], -prev_r[1], -prev_r[2]];
        let mut dw = 0.0;
        for i in 0..3 {
            dw += 0.5 * (ag[i] + agp[i]) * (q[1][i] - prev_q[1][i]);
        }
        w_ext += dw;
        let dd = m.commit(&q);
        let unloading = dn * dn + dt * dt < prev_target.0 * prev_target.0 + prev_target.1 * prev_target.1;
        if unloading {
            w_unload_new_d += dd;
        }
        m.q = q.clone();
        let u = m.elastic_energy(&q);
        let eps = w_ext - u - m.dissipated;
        max_eps = max_eps.max(eps.abs());
        if let Some((c, id)) = csv.as_mut() {
            let fnorm = ag[0] * n0.x + ag[1] * n0.y;
            c.row(&[*id, k as f64, dn, dt, fnorm, w_ext, u, m.dissipated, eps]);
        }
        prev_q = q;
        prev_r = r;
        prev_target = (dn, dt);
    }
    let gca = m.mat.gc * m.facets[0].len * m.mat.thickness;
    json!({"W_ext": w_ext, "D": m.dissipated, "U_end": m.elastic_energy(&m.q), "D_over_GcA": m.dissipated / gca,
        "max_abs_balance_residual_over_GcA": max_eps / gca, "dissipation_on_unloading_steps": w_unload_new_d})
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (1..=n).map(|i| a + (b - a) * i as f64 / n as f64).collect()
}

/// Series bar of `n` square cells (size ell), left end clamped, right end to a plate.
fn chain(n: usize, ell: f64, law: LawOptions) -> (Model, usize, usize) {
    let seeds: Vec<V2> = (0..n).map(|i| V2::new((i as f64 + 0.5) * ell, 0.5 * ell)).collect();
    let t = voronoi_convex(&seeds, &rect(V2::new(0.0, 0.0), V2::new(n as f64 * ell, ell)));
    let mut m = Model::from_tessellation(&t, Material::concrete(0.1), law, &[]);
    let g = m.ground();
    m.attach_boundary(&t, &BoundaryAttach { edge: 3, range: None, body: g, shear: true, tag: 1 });
    let plate = m.add_plate(V2::new(n as f64 * ell, 0.5 * ell), 1.0, 1.0, [false, true, true]);
    m.attach_boundary(&t, &BoundaryAttach { edge: 1, range: None, body: plate, shear: true, tag: 2 });
    // weak facet in the middle (σ_t × 0.99, same G_c) to localise deterministically
    let mid = n / 2;
    let wf = m.facets.iter().position(|f| f.a.min(f.b) == mid - 1 && f.a.max(f.b) == mid).unwrap_or(0);
    for p in 0..2 {
        let l = &mut m.facets[wf].pts[p].law;
        l.delta0 *= 0.99;
        l.wc /= 0.99;
    }
    m.pattern[plate][0] = 1.0; // unit force in +x
    (m, plate, wf)
}

fn run_chain(n: usize, ell: f64, by_opening: bool, csv: &mut Csv, id: f64) -> Value {
    let (mut m, plate, wf) = chain(n, ell, LawOptions::damage());
    let lch = m.mat.lch();
    let f = m.facets[wf].clone();
    let mut ctrl = Control::default();
    if by_opening {
        // opening of the weak facet at its centroid along n0
        ctrl.terms.push((f.b, f.pc - m.bodies[f.b].x0, f.n0, 1.0));
        ctrl.terms.push((f.a, f.pc - m.bodies[f.a].x0, f.n0, -1.0));
    } else {
        ctrl.terms.push((plate, V2::ZERO, V2::new(1.0, 0.0), 1.0));
    }
    let x0 = ctrl.value(&m.q);
    let wc = f.pts[0].law.wc;
    let fscale = m.mat.sigma_t * ell * m.mat.thickness;
    let opts = NewtonOpts { tol: 1e-9 * fscale, max_iter: 40 };
    let target_end = if by_opening { 1.05 * wc } else { 1.05 * wc + n as f64 * ell * m.mat.sigma_t / m.mat.e * 1.2 };
    let mut q = m.q.clone();
    let mut lf = 0.0;
    let mut s = 0.0;
    let mut ds = target_end / 200.0;
    let mut w_ext = 0.0;
    let mut steps = 0;
    let mut cut = 0;
    let mut failed_at: Option<f64> = None;
    let mut max_eps: f64 = 0.0;
    let gca = m.mat.gc * f.len * m.mat.thickness;
    while s < target_end - 1e-15 && steps < 20000 {
        let tgt = (s + ds).min(target_end);
        let mut qt = q.clone();
        let mut lft = lf;
        let rep = newton_controlled(&m, &mut qt, &mut lft, &ctrl, x0 + tgt, opts, 1e-12 * wc, solvers::assembly::Tangent::Consistent);
        let dd = if rep.converged { max_damage_increment(&m, &qt) } else { f64::INFINITY };
        if !rep.converged || dd > 0.05 {
            ds *= 0.5;
            cut += 1;
            if ds < target_end * 1e-7 {
                failed_at = Some(s);
                break;
            }
            continue;
        }
        w_ext += pattern_work(&m, &q, &qt, lf, lft);
        m.commit(&qt);
        q = qt;
        lf = lft;
        s = tgt;
        steps += 1;
        let u = m.elastic_energy(&q);
        let eps = w_ext - u - m.dissipated;
        max_eps = max_eps.max(eps.abs());
        let end = q[plate][0] - m.bodies[plate].x0.x;
        let open = {
            let (j0, _) = m.eval_point(&q, &m.facets[wf], 0);
            let (j1, _) = m.eval_point(&q, &m.facets[wf], 1);
            0.5 * (j0.dn + j1.dn)
        };
        csv.row(&[id, steps as f64, end, open, lf, w_ext, u, m.dissipated, eps]);
        ds = (ds * 1.5).min(target_end / 200.0);
    }
    // where did damage localise?
    let damaged: Vec<usize> = m.facets.iter().enumerate().filter(|(_, f)| f.pts.iter().any(|p| p.law.damage(p.kappa_c) > 0.0)).map(|(i, _)| i).collect();
    json!({"cells": n, "ell": ell, "L_over_2lch": n as f64 * ell / (2.0 * lch), "control": if by_opening {"weak-facet opening"} else {"end displacement"},
        "steps": steps, "cutbacks": cut, "failed_at_control": failed_at, "completed": failed_at.is_none(),
        "D_over_GcA_weak": m.dissipated / gca, "damaged_facets": damaged.len(), "weak_facet": wf,
        "max_abs_balance_residual_over_GcA": max_eps / gca})
}

fn main() {
    let mut out = serde_json::Map::new();
    let lab = (0.02, 0.02);
    // ---- (a) single facet, monotonic, load-step refinement
    let mut csv = Csv::new(EXP, "facet_paths", &["case", "step", "dn", "dt", "force_n", "W_ext", "U", "D", "residual"]);
    let mut mono = Vec::new();
    let m0 = pair(lab.0, lab.1, LawOptions::damage());
    let wc = m0.facets[0].pts[0].law.wc;
    let d0 = m0.facets[0].pts[0].law.delta0;
    for (i, &n) in [4usize, 8, 16, 32, 64, 128, 1024].iter().enumerate() {
        let mut m = pair(lab.0, lab.1, LawOptions::damage());
        let path: Vec<(f64, f64)> = linspace(0.0, 1.2 * wc, n).into_iter().map(|x| (x, 0.0)).collect();
        let mut r = drive(&mut m, &path, if n == 1024 { Some((&mut csv, 0.0)) } else { None });
        r["steps"] = json!(n);
        let _ = i;
        mono.push(r);
    }
    out.insert("monotonic_refinement".into(), json!(mono));

    // partial damage: D(κ) linear in κ
    let mut partial = Vec::new();
    for &frac in &[0.25, 0.5, 0.75] {
        let mut m = pair(lab.0, lab.1, LawOptions::damage());
        let kmax = d0 + frac * (wc - d0);
        let mut path: Vec<(f64, f64)> = linspace(0.0, kmax, 400).into_iter().map(|x| (x, 0.0)).collect();
        path.extend(linspace(kmax, 0.0, 100).into_iter().map(|x| (x, 0.0)));
        let r = drive(&mut m, &path, None);
        partial.push(json!({"kappa_fraction_of_softening": frac, "D_over_GcA": r["D_over_GcA"], "expected": frac,
            "W_minus_D_after_unload_over_GcA": (r["W_ext"].as_f64().unwrap() - r["D"].as_f64().unwrap()) / (m.mat.gc * lab.1 * 0.1)}));
    }
    out.insert("partial_damage".into(), json!(partial));

    // cyclic with compression after opening
    {
        let mut m = pair(lab.0, lab.1, LawOptions::damage());
        let seg = |a: f64, b: f64| linspace(a, b, 200).into_iter().map(|x| (x, 0.0)).collect::<Vec<_>>();
        let mut path = seg(0.0, 0.3 * wc);
        path.extend(seg(0.3 * wc, 0.0));
        path.extend(seg(0.0, 0.6 * wc));
        path.extend(seg(0.6 * wc, -2.0 * d0));
        path.extend(seg(-2.0 * d0, 1.2 * wc));
        let r = drive(&mut m, &path, Some((&mut csv, 1.0)));
        // compressive stiffness after damage: force at −2δ0 vs undamaged k_n δ
        out.insert("cyclic".into(), r);
    }

    // mixed mode: proportional at angle φ in (δn, sqrt(γ) δt) space, and non-proportional
    let mut mixed = Vec::new();
    for (ci, &phi) in [0.0f64, 30.0, 60.0, 90.0].iter().enumerate() {
        let mut m = pair(lab.0, lab.1, LawOptions::damage());
        let g = m.facets[0].pts[0].law.gamma;
        let r = phi.to_radians();
        let path: Vec<(f64, f64)> = linspace(0.0, 1.2 * wc, 1000).into_iter().map(|s| (s * r.cos(), s * r.sin() / g.sqrt())).collect();
        let mut res = drive(&mut m, &path, Some((&mut csv, 2.0 + ci as f64)));
        res["phi_deg"] = json!(phi);
        mixed.push(res);
    }
    {
        let mut m = pair(lab.0, lab.1, LawOptions::damage());
        let mut path: Vec<(f64, f64)> = linspace(0.0, 0.5 * wc, 500).into_iter().map(|s| (0.0, s)).collect();
        path.extend(linspace(0.0, 1.2 * wc, 500).into_iter().map(|s| (s, 0.5 * wc)));
        let mut res = drive(&mut m, &path, Some((&mut csv, 6.0)));
        res["phi_deg"] = json!("non-proportional: shear 0.5 wc, then normal");
        mixed.push(res);
    }
    out.insert("mixed_mode".into(), json!(mixed));

    // game-scale facets: ℓ = 0.5 m (ℓ/l_ch = 1.5, admissible) and ℓ = 0.7 m with crack band option
    let mut game = Vec::new();
    for &(ell, eta) in &[(0.5, None), (0.7, Some(0.95))] {
        let law = LawOptions { crack_band_eta: eta, ..LawOptions::damage() };
        let mut m = pair(ell, 0.5, law);
        let l = m.facets[0].pts[0].law;
        let path: Vec<(f64, f64)> = linspace(0.0, 1.2 * l.wc, 2000).into_iter().map(|x| (x, 0.0)).collect();
        let mut r = drive(&mut m, &path, None);
        r["ell"] = json!(ell);
        r["ell_over_lch"] = json!(ell / m.mat.lch());
        r["crack_band_eta"] = json!(eta);
        r["sigma_eff_over_sigma_t"] = json!(l.delta0 * m.mat.e / ell / m.mat.sigma_t);
        r["softening_slope_beta"] = json!(l.beta());
        game.push(r);
    }
    out.insert("game_scale_facet".into(), json!(game));

    // ---- (b) chains
    let mut csv2 = Csv::new(EXP, "chains", &["case", "step", "end_disp", "weak_opening", "load", "W_ext", "U", "D", "residual"]);
    let mut chains = Vec::new();
    let mut id = 0.0;
    for &n in &[3usize, 9, 31] {
        for by_open in [false, true] {
            let r = run_chain(n, 0.03, by_open, &mut csv2, id);
            eprintln!("{r}");
            chains.push(r);
            id += 1.0;
        }
    }
    out.insert("chains".into(), json!(chains));

    let m = pair(lab.0, lab.1, LawOptions::damage());
    let res = json!({
        "meta": meta(EXP, "all", None, "prescribed paths: evaluate + commit per step; chains: bordered Newton (indirect control), accept if converged and Δd ≤ 0.05, else halve",
            json!({"facet": {"ell": lab.0, "l": lab.1, "b": 0.1}, "delta0": d0, "wc": wc, "lch": m.mat.lch(), "GcA": m.mat.gc * lab.1 * 0.1})),
        "results": Value::Object(out),
    });
    write_json(EXP, "summary", &res);
    println!("{}", serde_json::to_string_pretty(&res["results"]).unwrap());
}
