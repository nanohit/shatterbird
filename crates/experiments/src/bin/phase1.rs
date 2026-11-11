//! Phase 1 / M: wave-free predictors of the Phase 0d fracture thresholds.
//! cargo run --release -p experiments --bin phase1 -- wavefree
use experiments::engine::scenes::{beam_impact, cantilever, Scene};
use experiments::engine::world::{crack_stats, fragments, Open0, Params, World};
use experiments::{meta, write_json};
use model::{BodyKind, Model};
use serde_json::{json, Value};
use solvers::s0::{body_reaction, newton, NewtonOpts};
use solvers::spectrum::Spectral;

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

// ---------------------------------------------------------------- Phase 1b / A: the explicit reference under changed contact

fn par_map<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    let nthreads: usize = std::env::var("SB_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let items: Vec<(usize, T)> = items.into_iter().enumerate().collect();
    let queue = std::sync::Mutex::new(items);
    let out = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..nthreads {
            s.spawn(|| loop {
                let job = queue.lock().unwrap().pop();
                let Some((k, it)) = job else { break };
                let r = f(it);
                out.lock().unwrap().push((k, r));
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort_by_key(|x| x.0);
    v.into_iter().map(|x| x.1).collect()
}

/// S2 scene with the striker facets' stiffness scaled by `ks` (they are open at t = 0, so the
/// static state is unchanged).
fn imp_scene(seed: u64, v0: f64, mp: f64, ks: f64) -> Scene {
    let mut sc = beam_impact(seed, 0.025, v0, mp, 1);
    for f in sc.m.facets.iter_mut().filter(|f| f.tag == 3) {
        for p in f.pts.iter_mut() {
            p.law.kn *= ks;
            p.law.kt *= ks;
        }
    }
    sc
}

#[derive(Clone, Debug, Default)]
struct ImpOut {
    v0: f64,
    broken: bool,
    frag_mass: Vec<f64>,
    frag_v: Vec<[f64; 2]>,
    d_end: f64,
    crack: [f64; 6],
    v_imp: f64,
    impulse: f64,
    peak_f: f64,
    t_c0: f64,
    t_c1: f64,
    t90: f64,
    onset_t: f64,
    he: f64,
    wall_s: f64,
    /// [t, contact force, striker velocity, dissipated]
    hist: Vec<[f64; 4]>,
}

impl ImpOut {
    fn json(&self) -> Value {
        json!({"v0": self.v0, "broken": self.broken, "frag_mass": self.frag_mass, "frag_v": self.frag_v, "D": self.d_end, "crack_x": self.crack[0], "crack_x0": self.crack[2], "crack_x1": self.crack[3],
            "v_imp": self.v_imp, "impulse": self.impulse, "peak_f": self.peak_f, "t_c0": self.t_c0, "t_c1": self.t_c1, "t90": self.t90, "onset_t": self.onset_t, "he": self.he, "wall_s": self.wall_s})
    }
}

/// Explicit reference run (all cells detailed) in chunks of `h`; `hook(world)` runs after every chunk.
fn run_imp(sc: Scene, h: f64, t_end: f64, hook: impl FnMut(&mut World)) -> ImpOut {
    run_imp_s(sc, h, 1.0, t_end, hook)
}

fn run_imp_s(sc: Scene, h: f64, he_scale: f64, t_end: f64, mut hook: impl FnMut(&mut World)) -> ImpOut {
    let t0 = std::time::Instant::now();
    let plate = sc.impactor.unwrap();
    let v0 = -sc.v_init[plate][1];
    let mp = sc.m.bodies[plate].mass;
    let wy = sc.m.loads[plate][1];
    let mut p = Params::new(h, 1e9);
    p.open0 = Open0::All;
    p.he_scale = he_scale;
    let mut wd = World::new(sc, p);
    let mut o = ImpOut { v0, t_c0: f64::NAN, t_c1: f64::NAN, t90: f64::NAN, onset_t: f64::NAN, ..Default::default() };
    let mut jh: Vec<(f64, f64)> = Vec::new();
    while wd.t < t_end - 1e-12 {
        wd.apply_removals();
        if !wd.macro_step(t_end) {
            eprintln!("reference run failed at t = {:.6}", wd.t);
            break;
        }
        wd.drop_open_cracks();
        wd.rereduce();
        hook(&mut wd);
        let f = body_reaction(&wd.sc.m, &wd.q, plate)[1].abs();
        let j = mp * (wd.v[plate][1] + v0) - wy * wd.t;
        if f > 1.0 {
            if o.t_c0.is_nan() {
                o.t_c0 = wd.t;
            }
            o.t_c1 = wd.t;
        }
        o.peak_f = o.peak_f.max(f);
        if o.onset_t.is_nan() && wd.sc.m.dissipated > 0.0 {
            o.onset_t = wd.t;
        }
        jh.push((wd.t, j));
        if h < 1e-3 {
            o.hist.push([wd.t, f, wd.v[plate][1], wd.sc.m.dissipated]);
        }
    }
    let jmax = jh.iter().fold(0.0f64, |a, x| a.max(x.1));
    o.t90 = jh.iter().find(|x| x.1 >= 0.9 * jmax).map_or(f64::NAN, |x| x.0);
    o.impulse = jmax;
    let fr = fragments(&wd.sc.m, &wd.q, &wd.v, 0.05);
    o.broken = fr.len() >= 2;
    o.frag_mass = fr.iter().map(|f| f.mass).collect();
    o.frag_v = fr.iter().map(|f| f.v).collect();
    o.d_end = wd.sc.m.dissipated;
    o.crack = crack_stats(&wd.sc.m);
    o.v_imp = wd.v[plate][1];
    o.he = wd.he;
    o.wall_s = t0.elapsed().as_secs_f64();
    o
}

const H240: f64 = 1.0 / 240.0;

fn v_crit_ref(seed: u64, mp: f64, ks: f64, lo: f64, hi: f64, n: usize) -> (f64, Vec<(f64, bool)>) {
    v_crit_ref_s(seed, mp, ks, lo, hi, n, 1.0)
}

fn v_crit_ref_s(seed: u64, mp: f64, ks: f64, lo: f64, hi: f64, n: usize, he_scale: f64) -> (f64, Vec<(f64, bool)>) {
    let (mut lo, mut hi) = (lo.ln(), hi.ln());
    let mut hist = Vec::new();
    for _ in 0..n {
        let mid = 0.5 * (lo + hi);
        let b = run_imp_s(imp_scene(seed, mid.exp(), mp, ks), H240, he_scale, 0.05, |_| {}).broken;
        hist.push((mid.exp(), b));
        if b {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    ((0.5 * (lo + hi)).exp(), hist)
}

fn two_dof_vcrit(p: &Push, mp: f64) -> f64 {
    bisect(0.01f64.ln(), 100f64.ln(), 16, |lv| two_dof_breaks(p, mp, lv.exp(), 1.0)).exp()
}

/// A1: thresholds of the explicit reference versus contact stiffness and striker mass.
fn a1() {
    let seeds = [1u64, 2];
    let pushes: Vec<Push> = par_map(seeds.to_vec(), push_curve);
    let masses = [5.0, 50.0, 500.0];
    let pred: Vec<Vec<f64>> = pushes.iter().map(|p| masses.iter().map(|&mp| two_dof_vcrit(p, mp)).collect()).collect();
    println!("two-coordinate predictions (before the explicit runs): {pred:?}");
    let mut jobs: Vec<(u64, f64, f64)> = Vec::new();
    for &s in &seeds {
        for ks in [0.01, 0.1, 1.0, 10.0] {
            jobs.push((s, 50.0, ks));
        }
        for mp in [5.0, 500.0] {
            jobs.push((s, mp, 1.0));
        }
    }
    let res = par_map(jobs.clone(), |(s, mp, ks)| {
        let (lo, hi) = if mp < 20.0 { (0.3, 30.0) } else if mp > 200.0 { (0.03, 3.0) } else { (0.1, 10.0) };
        let r = v_crit_ref(s, mp, ks, lo, hi, 12);
        println!("A1 seed {s} mp {mp} ks {ks}: v_crit {:.4}", r.0);
        r
    });
    let rows: Vec<Value> = jobs.iter().zip(&res).map(|(&(s, mp, ks), r)| {
        let si = seeds.iter().position(|&x| x == s).unwrap();
        let two = masses.iter().position(|&m| m == mp).map(|mi| pred[si][mi]);
        json!({"seed": s, "mp": mp, "ks": ks, "v_crit": r.0, "bisection": r.1, "two_coordinate": two, "ke_crit": 0.5 * mp * r.0 * r.0, "p_crit": mp * r.0})
    }).collect();
    write_json("phase1", "a1_contact_thresholds", &json!({"meta": meta("phase1", "a1", None, "explicit reference, all cells detailed, h_macro 1/240", json!({})), "rows": rows}));
}

fn a1_configs() -> Vec<(u64, f64, f64)> {
    let mut jobs = Vec::new();
    for s in [1u64, 2] {
        for ks in [0.01, 0.1, 1.0, 10.0] {
            jobs.push((s, 50.0, ks));
        }
        for mp in [5.0, 500.0] {
            jobs.push((s, mp, 1.0));
        }
    }
    jobs
}

/// A1 noise: the same thresholds with the explicit step scaled by 0.9.
fn a1n() {
    let jobs = a1_configs();
    let res = par_map(jobs.clone(), |(s, mp, ks)| {
        let (lo, hi) = if mp < 20.0 { (0.3, 30.0) } else if mp > 200.0 { (0.03, 3.0) } else { (0.1, 10.0) };
        let r = v_crit_ref_s(s, mp, ks, lo, hi, 12, 0.9);
        println!("A1n seed {s} mp {mp} ks {ks}: v_crit {:.4}", r.0);
        r
    });
    let rows: Vec<Value> = jobs.iter().zip(&res).map(|(&(s, mp, ks), r)| json!({"seed": s, "mp": mp, "ks": ks, "v_crit": r.0, "bisection": r.1})).collect();
    write_json("phase1", "a1_contact_thresholds_he09", &json!({"meta": meta("phase1", "a1n", None, "explicit reference, explicit step x0.9", json!({})), "rows": rows}));
}

/// A2: consequences above the threshold for every contact variant, with a noise floor.
fn a2() {
    let a1: Value = serde_json::from_str(&std::fs::read_to_string("results/phase1/a1_contact_thresholds.json").unwrap()).unwrap();
    let vc = |s: u64, mp: f64, ks: f64| a1["rows"].as_array().unwrap().iter().find(|r| r["seed"] == s && r["mp"] == mp && r["ks"] == ks).unwrap()["v_crit"].as_f64().unwrap();
    let mut jobs = Vec::new();
    for (s, mp, ks) in a1_configs() {
        // stiffness variants are compared at the same velocity (factor x nominal threshold);
        // mass variants at factor x their own threshold
        let base = if mp == 50.0 { vc(s, 50.0, 1.0) } else { vc(s, mp, 1.0) };
        for fac in [1.25, 2.0] {
            for (var, h, dv) in [("base", H240, 0.0), ("chunk", 2e-5, 0.0), ("v+", H240, 1e-3), ("v-", H240, -1e-3)] {
                jobs.push((s, mp, ks, fac, var, h, base * fac * (1.0 + dv)));
            }
        }
    }
    let res = par_map(jobs.clone(), |(s, mp, ks, _, _, h, v0)| run_imp(imp_scene(s, v0, mp, ks), h, 0.05, |_| {}));
    let rows: Vec<Value> = jobs.iter().zip(&res).map(|(&(s, mp, ks, fac, var, _, _), o)| {
        let mut j = o.json();
        j["seed"] = json!(s);
        j["mp"] = json!(mp);
        j["ks"] = json!(ks);
        j["factor"] = json!(fac);
        j["variant"] = json!(var);
        j
    }).collect();
    let hists: Vec<Value> = jobs.iter().zip(&res).filter(|(j, _)| j.4 == "chunk" && j.0 == 1 && j.3 == 1.25).map(|(&(s, mp, ks, fac, _, _, _), o)| json!({"seed": s, "mp": mp, "ks": ks, "factor": fac, "cols": ["t", "contact_force", "v_striker", "D"], "rows": o.hist.iter().take(400).collect::<Vec<_>>()})).collect();
    write_json("phase1", "a2_contact_consequences", &json!({"meta": meta("phase1", "a2", None, "explicit reference; variants: base (h 1/240), chunk (h 20 us), v0 x (1 +- 1e-3)", json!({})), "rows": rows, "histories": hists}));
}

// ---------------------------------------------------------------- A3: modal content and state truncation

/// Lowest modes of the intact beam (striker detached, supports bilateral), whitened.
struct Modes {
    /// (body, dof) -> global index
    idx: Vec<[Option<usize>; 3]>,
    sqm: Vec<f64>,
    lam: Vec<f64>,
    y: Vec<Vec<f64>>,
    sp: Spectral,
    qs: Vec<[f64; 3]>,
    plate: usize,
    /// (cell, facet length) under the striker
    pad: Vec<(usize, f64)>,
}

fn modes(sc: &Scene, nev: usize) -> Modes {
    modes_of(&sc.m, None, sc.impactor.unwrap(), sc.m.q.clone(), nev, true)
}

/// Lowest modes of `m0` with the striker detached. With `state = Some(q)` the operator is the
/// secant stiffness of the committed damage at configuration q (open points and shear scaled
/// by 1 - d, floor 1e-3), so separated parts appear as near-rigid low modes.
fn modes_of(m0: &Model, state: Option<&[[f64; 3]]>, plate: usize, qs: Vec<[f64; 3]>, nev: usize, verbose: bool) -> Modes {
    let mut m = m0.clone();
    if let Some(q) = state {
        for fi in 0..m.facets.len() {
            if !m.facets[fi].active {
                continue;
            }
            for p in 0..2 {
                let (j, _) = m0.eval_point(q, &m0.facets[fi], p);
                let pt = &mut m.facets[fi].pts[p];
                let om = (1.0 - pt.law.damage(pt.kappa_c)).max(1e-3);
                if j.dn > 0.0 {
                    pt.law.kn *= om;
                }
                pt.law.kt *= om;
            }
        }
    }
    m.bodies[plate].fixed = [true; 3];
    for f in m.facets.iter_mut().filter(|f| f.tag == 3) {
        f.active = false;
    }
    // the supports carry no shear, so horizontal translation is a zero-energy mode; a
    // 1e-6 shear spring makes K definite and keeps that motion in the basis as mode 1
    for f in m.facets.iter_mut().filter(|f| f.tag >= 10) {
        for p in f.pts.iter_mut() {
            if p.law.kt == 0.0 {
                p.law.kt = 1e-6 * p.law.kn;
            }
        }
    }
    let (idx, n) = m.dof_map();
    let mut sqm = vec![0.0; n];
    for (i, b) in m.bodies.iter().enumerate() {
        for k in 0..3 {
            if let Some(g) = idx[i][k] {
                sqm[g] = (if k < 2 { b.mass } else { b.inertia }).sqrt();
            }
        }
    }
    let sp = Spectral::from_model2d(&m);
    let pr = sp.low(nev, 7, 1e-13, n).expect("shift-invert");
    let mut ord: Vec<usize> = (0..pr.vals.len()).collect();
    ord.sort_by(|&a, &b| pr.vals[a].partial_cmp(&pr.vals[b]).unwrap());
    let worst = ord.iter().map(|&i| pr.resid[i]).fold(0.0f64, f64::max);
    if verbose {
        eprintln!("modes: n {n}, {} pairs, {} Lanczos steps, worst residual {:.1e}, f1..f5 = {:?} Hz", ord.len(), pr.steps, worst, ord.iter().take(5).map(|&i| (pr.vals[i].sqrt() / (2.0 * std::f64::consts::PI)).round()).collect::<Vec<_>>());
    }
    Modes { idx, sqm, lam: ord.iter().map(|&i| pr.vals[i]).collect(), y: ord.iter().map(|&i| pr.vecs[i].clone()).collect(), sp, qs, plate, pad: m0.facets.iter().filter(|f| f.tag == 3).map(|f| (f.a, f.len)).collect() }
}

impl Modes {
    /// whitened vector of a cell field x (displacement or velocity)
    fn whiten(&self, x: &[[f64; 3]]) -> Vec<f64> {
        let mut w = vec![0.0; self.sqm.len()];
        for (i, r) in self.idx.iter().enumerate() {
            for k in 0..3 {
                if let Some(g) = r[k] {
                    w[g] = self.sqm[g] * x[i][k];
                }
            }
        }
        w
    }
    fn coeffs(&self, w: &[f64], k: usize) -> Vec<f64> {
        self.y[..k].iter().map(|y| y.iter().zip(w).map(|(a, b)| a * b).sum()).collect()
    }
    /// Project the world's cell state on the lowest k modes. Returns
    /// (kinetic fraction kept, elastic fraction kept, vibration energy before [J], energy removed [J]).
    fn truncate(&self, wd: &mut World, k: usize, apply: bool) -> [f64; 4] {
        let u: Vec<[f64; 3]> = wd.q.iter().zip(&self.qs).map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]]).collect();
        let (wu, wv) = (self.whiten(&u), self.whiten(&wd.v));
        let (a, ad) = (self.coeffs(&wu, k), self.coeffs(&wv, k));
        let ke: f64 = 0.5 * wv.iter().map(|x| x * x).sum::<f64>();
        let mut bw = vec![0.0; wu.len()];
        self.sp.apply_b(&wu, &mut bw);
        let pe: f64 = 0.5 * wu.iter().zip(&bw).map(|(x, y)| x * y).sum::<f64>();
        let ke_k: f64 = 0.5 * ad.iter().map(|x| x * x).sum::<f64>();
        let pe_k: f64 = 0.5 * a.iter().zip(&self.lam).map(|(x, l)| l * x * x).sum::<f64>();
        if apply {
            let (mut q, mut v) = (wd.q.clone(), wd.v.clone());
            for (i, r) in self.idx.iter().enumerate() {
                for d in 0..3 {
                    if let Some(g) = r[d] {
                        let (mut su, mut sv) = (0.0, 0.0);
                        for m in 0..k {
                            su += a[m] * self.y[m][g];
                            sv += ad[m] * self.y[m][g];
                        }
                        q[i][d] = self.qs[i][d] + su / self.sqm[g];
                        v[i][d] = sv / self.sqm[g];
                    }
                }
            }
            // keep the striker gap: move the striker with the mean displacement change of the pad
            // cells (the local dent is high-mode content and disappears in the projection)
            let wsum: f64 = self.pad.iter().map(|x| x.1).sum();
            let shift: f64 = self.pad.iter().map(|&(c, w)| w * (q[c][1] - wd.q[c][1])).sum::<f64>() / wsum;
            q[self.plate][1] += shift;
            wd.set_kinematics(q, v);
        }
        [ke_k / ke, pe_k / pe, ke + pe, ke + pe - ke_k - pe_k]
    }
}

/// mode counts including the quasi-rigid horizontal translation (elastic modes = k - 1)
const KS: [usize; 9] = [2, 3, 4, 6, 11, 21, 41, 81, 161];
const TCS: [f64; 4] = [0.3e-3, 1.2e-3, 2.0e-3, 4.0e-3];

/// A3: energy content of the lowest modes, consequences after truncation, thresholds under truncation.
fn a3() {
    let a1: Value = serde_json::from_str(&std::fs::read_to_string("results/phase1/a1_contact_thresholds.json").unwrap()).unwrap();
    let vc = |s: u64| a1["rows"].as_array().unwrap().iter().find(|r| r["seed"] == s && r["mp"] == 50.0 && r["ks"] == 1.0).unwrap()["v_crit"].as_f64().unwrap();
    let seeds = [1u64, 2];
    let md: Vec<Modes> = seeds.iter().map(|&s| modes(&imp_scene(s, 0.5, 50.0, 1.0), 161)).collect();
    let freqs: Vec<Vec<f64>> = md.iter().map(|m| m.lam.iter().map(|l| l.sqrt() / (2.0 * std::f64::consts::PI)).collect()).collect();
    // content + consequences
    let mut jobs = Vec::new();
    for (si, &s) in seeds.iter().enumerate() {
        for fac in [1.25, 2.0] {
            jobs.push((si, s, fac, f64::NAN, 0usize));
            for tc in TCS {
                for k in KS {
                    jobs.push((si, s, fac, tc, k));
                }
            }
        }
    }
    let res = par_map(jobs.clone(), |(si, s, fac, tc, k)| {
        let mut info = None;
        let mut content = Vec::new();
        let m = &md[si];
        let o = run_imp(imp_scene(s, fac * vc(s), 50.0, 1.0), 1e-4, 0.05, |wd| {
            if tc.is_nan() {
                // untruncated run: record the modal content at every t_c
                for t in TCS {
                    if (wd.t - t).abs() < 0.5e-4 {
                        let d = wd.sc.m.dissipated;
                        content.push(json!({"t": t, "D": d, "kept": KS.iter().map(|&k| { let r = m.truncate(wd, k, false); json!({"k": k, "ke": r[0], "pe": r[1], "e_vib": r[2]}) }).collect::<Vec<_>>()}));
                    }
                }
            } else if info.is_none() && wd.t >= tc - 0.5e-4 {
                let d = wd.sc.m.dissipated;
                let r = m.truncate(wd, k, true);
                info = Some(json!({"t": wd.t, "D_at": d, "ke_kept": r[0], "pe_kept": r[1], "e_vib": r[2], "e_removed": r[3]}));
            }
        });
        (o, info, content)
    });
    let rows: Vec<Value> = jobs.iter().zip(&res).map(|(&(_, s, fac, tc, k), (o, info, content))| {
        let mut j = o.json();
        j["seed"] = json!(s);
        j["factor"] = json!(fac);
        j["t_c"] = if tc.is_nan() { Value::Null } else { json!(tc) };
        j["k"] = json!(k);
        j["truncation"] = json!(info);
        j["content"] = json!(content);
        j
    }).collect();
    write_json("phase1", "a3_truncation", &json!({"meta": meta("phase1", "a3", None, "explicit reference, chunks 0.1 ms; cell state projected on k intact-beam modes at t_c", json!({})), "freq_hz": freqs, "rows": rows}));
    // thresholds under truncation
    let mut tj = Vec::new();
    for (si, &s) in seeds.iter().enumerate() {
        tj.push((si, s, f64::NAN, 0usize));
        for tc in [0.3e-3, 1.2e-3] {
            for k in [2usize, 4, 6, 11, 21, 41] {
                tj.push((si, s, tc, k));
            }
        }
    }
    let thr = par_map(tj.clone(), |(si, s, tc, k)| {
        let m = &md[si];
        let (mut lo, mut hi) = (0.1f64.ln(), 10f64.ln());
        for _ in 0..12 {
            let mid = 0.5 * (lo + hi);
            let mut done = false;
            let b = run_imp(imp_scene(s, mid.exp(), 50.0, 1.0), 1e-4, 0.05, |wd| {
                if !tc.is_nan() && !done && wd.t >= tc - 0.5e-4 {
                    m.truncate(wd, k, true);
                    done = true;
                }
            }).broken;
            if b {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let v = (0.5 * (lo + hi)).exp();
        println!("A3 threshold seed {s} t_c {tc} k {k}: {v:.4}");
        v
    });
    let trows: Vec<Value> = tj.iter().zip(&thr).map(|(&(_, s, tc, k), v)| json!({"seed": s, "t_c": if tc.is_nan() { Value::Null } else { json!(tc) }, "k": k, "v_crit": v})).collect();
    write_json("phase1", "a3_truncation_thresholds", &json!({"meta": meta("phase1", "a3t", None, "explicit reference, chunks 0.1 ms; threshold when the cell state is projected on k modes at t_c", json!({})), "rows": trows}));
}

/// A3b (post-hoc): truncation on the lowest modes of the current damaged operator.
fn a3b() {
    let a1: Value = serde_json::from_str(&std::fs::read_to_string("results/phase1/a1_contact_thresholds.json").unwrap()).unwrap();
    let vc = |s: u64| a1["rows"].as_array().unwrap().iter().find(|r| r["seed"] == s && r["mp"] == 50.0 && r["ks"] == 1.0).unwrap()["v_crit"].as_f64().unwrap();
    let ks = [4usize, 6, 11, 21, 41, 81];
    let trunc = |wd: &mut World, qs: &Vec<[f64; 3]>, k: usize| -> Value {
        let plate = wd.sc.impactor.unwrap();
        let m = modes_of(&wd.sc.m, Some(&wd.q), plate, qs.clone(), k, false);
        let d = wd.sc.m.dissipated;
        let f: Vec<f64> = m.lam.iter().take(8).map(|l| (l.max(0.0).sqrt() / (2.0 * std::f64::consts::PI)).round()).collect();
        let r = m.truncate(wd, k, true);
        json!({"t": wd.t, "D_at": d, "ke_kept": r[0], "pe_kept": r[1], "e_vib": r[2], "e_removed": r[3], "f_low_hz": f})
    };
    let mut jobs = Vec::new();
    for s in [1u64, 2] {
        for fac in [1.25, 2.0] {
            jobs.push((s, fac, f64::NAN, 0usize));
            for tc in TCS {
                for k in ks {
                    jobs.push((s, fac, tc, k));
                }
            }
        }
    }
    let res = par_map(jobs.clone(), |(s, fac, tc, k)| {
        let sc = imp_scene(s, fac * vc(s), 50.0, 1.0);
        let qs = sc.m.q.clone();
        let mut info = None;
        let o = run_imp(sc, 1e-4, 0.05, |wd| {
            if !tc.is_nan() && info.is_none() && wd.t >= tc - 0.5e-4 {
                info = Some(trunc(wd, &qs, k));
            }
        });
        (o, info)
    });
    let rows: Vec<Value> = jobs.iter().zip(&res).map(|(&(s, fac, tc, k), (o, info))| {
        let mut j = o.json();
        j["seed"] = json!(s);
        j["factor"] = json!(fac);
        j["t_c"] = if tc.is_nan() { Value::Null } else { json!(tc) };
        j["k"] = json!(k);
        j["truncation"] = json!(info);
        j
    }).collect();
    write_json("phase1", "a3b_truncation_damaged_basis", &json!({"meta": meta("phase1", "a3b", None, "explicit reference, chunks 0.1 ms; cell state projected at t_c on k modes of the current damaged (secant) operator", json!({})), "rows": rows}));
    let mut tj = Vec::new();
    for s in [1u64, 2] {
        for tc in [0.3e-3, 1.2e-3] {
            for k in [6usize, 11, 21, 41] {
                tj.push((s, tc, k));
            }
        }
    }
    let thr = par_map(tj.clone(), |(s, tc, k)| {
        let (mut lo, mut hi) = (0.1f64.ln(), 10f64.ln());
        for _ in 0..12 {
            let mid = 0.5 * (lo + hi);
            let sc = imp_scene(s, mid.exp(), 50.0, 1.0);
            let qs = sc.m.q.clone();
            let mut done = false;
            let b = run_imp(sc, 1e-4, 0.05, |wd| {
                if !done && wd.t >= tc - 0.5e-4 {
                    trunc(wd, &qs, k);
                    done = true;
                }
            }).broken;
            if b {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let v = (0.5 * (lo + hi)).exp();
        println!("A3b threshold seed {s} t_c {tc} k {k}: {v:.4}");
        v
    });
    let trows: Vec<Value> = tj.iter().zip(&thr).map(|(&(s, tc, k), v)| json!({"seed": s, "t_c": tc, "k": k, "v_crit": v})).collect();
    write_json("phase1", "a3b_truncation_damaged_basis_thresholds", &json!({"meta": meta("phase1", "a3bt", None, "threshold when the cell state is projected at t_c on k modes of the current damaged operator", json!({})), "rows": trows}));
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
        Some("a1") => a1(),
        Some("a1n") => a1n(),
        Some("a2") => a2(),
        Some("a3") => a3(),
        Some("a3b") => a3b(),
        Some("one") => {
            let a: Vec<f64> = std::env::args().skip(2).map(|x| x.parse().unwrap()).collect();
            let o = run_imp(imp_scene(a[0] as u64, a[1], a[2], a[3]), a[4], 0.05, |_| {});
            println!("{}", o.json());
        }
        _ => eprintln!("usage: phase1 wavefree|impact|a1|one seed v0 mp ks h"),
    }
}
