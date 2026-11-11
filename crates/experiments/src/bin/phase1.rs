//! Phase 1 / M: wave-free predictors of the Phase 0d fracture thresholds.
//! cargo run --release -p experiments --bin phase1 -- wavefree
use experiments::engine::scenes::{beam_impact, cantilever};
use experiments::{meta, write_json};
use model::{BodyKind, Model};
use serde_json::{json, Value};
use solvers::s0::{newton, NewtonOpts};

const G: f64 = 9.81;

fn mg(m: &Model) -> f64 {
    m.bodies.iter().map(|b| if b.kind == BodyKind::Cell { b.mass } else { 0.0 }).sum::<f64>() * G
}

/// Static path of the cantilever under gravity·λ and prop force × c, c: 1 → −1.
/// Returns (a_limit, energy_survives, path) with a = 1 − c the unbalanced-load factor.
fn cantilever_path(seed: u64, lam: f64) -> (f64, bool, Vec<(f64, f64)>) {
    let sc = cantilever(seed, 0.04, true, 0.005, lam);
    let mut m = sc.m.clone();
    for &(i, f) in &sc.releases[0].forces {
        m.pattern[i] = f;
    }
    let q0 = m.q.clone();
    let mut q = q0.clone();
    let tol = 1e-8 * mg(&m);
    let u_of = |q: &Vec<[f64; 3]>| -> f64 { sc.releases[0].forces.iter().map(|&(i, f)| -(0..3).map(|r| f[r] * (q[i][r] - q0[i][r])).sum::<f64>()).sum() };
    let (mut a, mut da) = (0.0f64, 0.05f64);
    let (mut w, mut u_prev, mut a_prev) = (0.0, 0.0, 0.0);
    let mut path = vec![(0.0, 0.0)];
    let mut survives = false;
    while a < 2.0 - 1e-12 {
        let an = (a + da).min(2.0);
        let mut qt = q.clone();
        let rep = newton(&m, &mut qt, 1.0 - an, None, NewtonOpts { tol, max_iter: 80 });
        if !rep.converged {
            da *= 0.5;
            if da < 1e-4 {
                break;
            }
            continue;
        }
        q = qt;
        m.commit(&q);
        a = an;
        let u = u_of(&q);
        w += 0.5 * (a + a_prev) * (u - u_prev);
        u_prev = u;
        a_prev = a;
        path.push((a, u));
        if w >= u && a > 0.0 {
            survives = true; // absorbed work ≥ work of the suddenly applied unit load
        }
    }
    (a, survives, path)
}

fn bisect(mut lo: f64, mut hi: f64, n: usize, fails: impl Fn(f64) -> bool) -> f64 {
    for _ in 0..n {
        let mid = 0.5 * (lo + hi);
        if fails(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

fn m1(seed: u64, lam_ref: f64) -> Value {
    let stat = bisect(0.6, 3.2, 9, |l| cantilever_path(seed, l).0 < 1.0);
    let daf2 = bisect(0.4, 1.6, 9, |l| cantilever_path(seed, l).0 < 2.0 - 1e-9);
    let en = bisect(0.4, 1.6, 9, |l| !cantilever_path(seed, l).1);
    eprintln!("M1 seed {seed}: ref {lam_ref:.4} static {stat:.4} daf2 {daf2:.4} energy {en:.4}");
    json!({"seed": seed, "lambda_ref_explicit": lam_ref, "static_only": stat, "daf2": daf2, "energy": en,
        "rel_error": {"static_only": stat / lam_ref - 1.0, "daf2": daf2 / lam_ref - 1.0, "energy": en / lam_ref - 1.0}})
}

struct Push {
    /// (plate displacement, gauge displacement, plate force)
    pts: Vec<(f64, f64, f64)>,
    m_eff: f64,
    newton_solves: usize,
    cells: usize,
}

fn push_curve(seed: u64) -> Push {
    let sc = beam_impact(seed, 0.025, 0.0, 50.0, 1);
    let mut m = sc.m.clone();
    let plate = sc.impactor.unwrap();
    let gauge = sc.tip_cells[0];
    m.bodies[plate].fixed[1] = true;
    m.bodies[gauge].fixed[0] = true;
    let q0 = m.q.clone();
    let mut q = q0.clone();
    let tol = 1e-8 * mg(&m);
    let force = |m: &Model, q: &Vec<[f64; 3]>| -> f64 {
        let mut p = 0.0;
        for f in m.facets.iter().filter(|f| f.tag == 3 && f.active) {
            for pt in 0..2 {
                let (j, e) = m.eval_point(q, f, pt);
                p -= if f.b == plate { j.jnb[1] * e.fnorm + j.jtb[1] * e.ftan } else { j.jna[1] * e.fnorm + j.jta[1] * e.ftan };
            }
        }
        p
    };
    let p0 = force(&m, &q);
    let mut pts = vec![(0.0, 0.0, p0)];
    let (mut u, mut du) = (0.0f64, 4e-6f64);
    let mut peak = p0;
    let mut m_eff = 0.0;
    let mut solves = 0;
    while u < 4e-3 {
        let mut qt = q.clone();
        qt[plate][1] = q0[plate][1] - (u + du);
        let rep = newton(&m, &mut qt, 0.0, None, NewtonOpts { tol, max_iter: 80 });
        solves += 1;
        if !rep.converged {
            du *= 0.5;
            if du < 1e-8 {
                break;
            }
            continue;
        }
        let pn = force(&m, &qt);
        if m_eff == 0.0 && pts.last().unwrap().2 - p0 > 200.0 && pn > pts.last().unwrap().2 {
            // Rayleigh mass of the first loaded static increment, normalised at the gauge
            let dg = qt[gauge][1] - q[gauge][1];
            m_eff = (0..m.bodies.len()).filter(|&i| m.bodies[i].kind == BodyKind::Cell).map(|i| {
                let d = [qt[i][0] - q[i][0], qt[i][1] - q[i][1], qt[i][2] - q[i][2]];
                (m.bodies[i].mass * (d[0] * d[0] + d[1] * d[1]) + m.bodies[i].inertia * d[2] * d[2]) / (dg * dg)
            }).sum();
        }
        q = qt;
        m.commit(&q);
        u += du;
        du = (du * 1.5).min(4e-6);
        let p = force(&m, &q);
        peak = peak.max(p);
        pts.push((u, -(q[gauge][1] - q0[gauge][1]), p));
        if p < 0.02 * peak && u > 5e-5 {
            break;
        }
    }
    Push { pts, m_eff, newton_solves: solves, cells: m.n_cells() }
}

/// Two generalised coordinates: striker and beam (Rayleigh mass, static curve, secant unloading).
fn two_dof_breaks(p: &Push, mp: f64, v0: f64, kc_scale: f64) -> bool {
    let (_, ug_fail, _) = *p.pts.last().unwrap();
    let p0 = p.pts[0].2;
    // first loaded points: contact gap, contact stiffness, initial beam stiffness
    let i0 = p.pts.iter().position(|x| x.2 - p0 > 200.0).unwrap();
    let (a, b) = (p.pts[i0], p.pts[i0 + 3]);
    let kc = (b.2 - a.2) / ((b.0 - b.1) - (a.0 - a.1));
    let gap = (a.0 - a.1) - (a.2 - p0) / kc;
    let kc = kc * kc_scale;
    let k0 = (b.2 - a.2) / (b.1 - a.1);
    let ug0 = a.1 - (a.2 - p0) / k0;
    let curve = |ug: f64| -> f64 {
        if ug <= a.1 {
            return (k0 * (ug - ug0)).max(0.0);
        }
        let k = p.pts.partition_point(|x| x.1 < ug);
        if k >= p.pts.len() {
            return 0.0;
        }
        let (c, d) = (p.pts[k - 1], p.pts[k]);
        c.2 - p0 + (d.2 - c.2) * (ug - c.1) / (d.1 - c.1).max(1e-15)
    };
    let (mut xp, mut vp, mut xg, mut vg, mut umax) = (0.0f64, v0, ug0.max(0.0), 0.0f64, 0.0f64);
    let h = 1e-7;
    for _ in 0..500_000 {
        let fc = (kc * (xp - xg - gap)).max(0.0);
        let fb = if xg >= umax { curve(xg) } else if xg > 0.0 && umax > 0.0 { curve(umax) * xg / umax } else { k0 * (xg - ug0).min(0.0) };
        vp += h * (-fc / mp);
        vg += h * (fc - fb) / p.m_eff;
        xp += h * vp;
        xg += h * vg;
        umax = umax.max(xg);
        if umax >= ug_fail {
            return true;
        }
    }
    false
}

fn m2(seed: u64, v_ref: f64) -> Value {
    let t0 = std::time::Instant::now();
    let p = push_curve(seed);
    let wall = t0.elapsed().as_secs_f64();
    let mp = 50.0;
    let p0 = p.pts[0].2;
    let wf: f64 = p.pts.windows(2).map(|w| 0.5 * (w[0].2 + w[1].2 - 2.0 * p0) * (w[1].0 - w[0].0)).sum();
    let peak = p.pts.iter().map(|x| x.2 - p0).fold(0.0, f64::max);
    let v2 = bisect(0.05f64.ln(), 10f64.ln(), 14, |lv| two_dof_breaks(&p, mp, lv.exp(), 1.0)).exp();
    let sens: Vec<Value> = [0.01, 0.1, 10.0].iter().map(|&sc| { let v = bisect(0.05f64.ln(), 10f64.ln(), 14, |lv| two_dof_breaks(&p, mp, lv.exp(), sc)).exp(); eprintln!("   contact stiffness x{sc}: v_crit {v:.3}"); json!({"kc_scale": sc, "v_crit": v}) }).collect();
    let v_el = (2.0 * wf / mp).sqrt();
    let v_pl = (2.0 * wf * (mp + p.m_eff) / (mp * mp)).sqrt();
    let (kc, gap) = { let i0 = p.pts.iter().position(|x| x.2 - p0 > 200.0).unwrap(); let (a, b) = (p.pts[i0], p.pts[i0 + 3]); let kc = (b.2 - a.2) / ((b.0 - b.1) - (a.0 - a.1)); (kc, (a.0 - a.1) - (a.2 - p0) / kc) };
    eprintln!("   kc {kc:.3e} N/m gap {gap:.2e} m");
    eprintln!("M2 seed {seed}: ref {v_ref:.3} two-dof {v2:.3} full-transfer {v_el:.3} plastic {v_pl:.3}  Wf {wf:.2} J peak {peak:.0} N m_eff {:.1} kg u_fail {:.2e} ({} pts, {} solves, {wall:.1}s)", p.m_eff, p.pts.last().unwrap().0, p.pts.len(), p.newton_solves);
    json!({"seed": seed, "v_ref_explicit": v_ref, "two_dof": v2, "energy_full_transfer": v_el, "energy_plastic_impact": v_pl,
        "rel_error": {"two_dof": v2 / v_ref - 1.0, "energy_full_transfer": v_el / v_ref - 1.0, "energy_plastic_impact": v_pl / v_ref - 1.0},
        "contact_stiffness_sensitivity": sens, "static_work_of_fracture_J": wf, "peak_force_N": peak, "rayleigh_mass_kg": p.m_eff, "cells": p.cells,
        "newton_solves": p.newton_solves, "wall_s": wall,
        "curve": p.pts.iter().map(|x| json!([x.0, x.1, x.2 - p0])).collect::<Vec<_>>()})
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("wavefree") => {
            let r2: Vec<Value> = [(1u64, 0.4772054908999464), (2, 0.5403010638961208)].iter().map(|&(s, v)| m2(s, v)).collect();
            let r1: Vec<Value> = [(1u64, 0.9203), (2, 0.9047)].iter().map(|&(s, l)| m1(s, l)).collect();
            write_json("phase1", "wavefree", &json!({"meta": meta("phase1", "wavefree", None, "static Newton paths; two-coordinate ODE", json!({})), "support_removal": r1, "impact": r2}));
        }
        Some("impact") => {
            let r2: Vec<Value> = [(1u64, 0.4772054908999464), (2, 0.5403010638961208)].iter().map(|&(s, v)| m2(s, v)).collect();
            write_json("phase1", "wavefree_impact", &json!({"meta": meta("phase1", "wavefree_impact", None, "static Newton path; two-coordinate ODE", json!({})), "impact": r2}));
        }
        _ => eprintln!("usage: phase1 wavefree|impact"),
    }
}
