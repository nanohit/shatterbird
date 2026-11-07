//! Phase 0d — adaptive multirate engine (docs/phase0d/adaptive-engine.md).
//!
//! Modes: s0a, s0b, s0c (component checks), s1, s2, s3, s4 (experiments), one-off
//! runs for development (cant, imp).

use experiments::engine::scenes::{beam_impact, cantilever, cantilever_len, collapse_beam, span_collapse, wall_impact, Scene};
use experiments::engine::world::{run, Open0, Params, Traj, World, ROW};
use experiments::engine::{Work, C_PT_FORCE};
use experiments::{meta, write_json, Csv};
use serde_json::{json, Value};
use std::time::Instant;

const EXP: &str = "phase0d";

fn env_f(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_u(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn save_traj(dir: &str, tr: &Traj) {
    let mut c = Csv::new(dir, &format!("traj_{}_{}", tr.scene, tr.label), &ROW);
    for r in &tr.rows {
        c.row(r);
    }
}

fn summary(tr: &Traj) -> Value {
    let mut v = serde_json::to_value(tr).unwrap();
    v["events"] = json!(tr.events.len());
    v
}


/// Development overrides of the engine parameters (documented in each run's meta).
fn apply_env(p: &mut Params) {
    p.coupling_ftol = env_f("SB_FTOL", p.coupling_ftol);
    p.coupling_tol = env_f("SB_CTOL", p.coupling_tol);
    p.chi_trig = env_f("SB_TRIG", p.chi_trig);
    p.r_e = env_f("SB_RE", p.r_e);
    p.eps_rr = env_f("SB_EPSRR", p.eps_rr);
    p.chi_rr = env_f("SB_CHIRR", p.chi_rr);
    p.quiet_steps = env_u("SB_QUIET", p.quiet_steps);
    p.deg = env_u("SB_DEG", p.deg);
    p.he_scale = env_f("SB_HESCALE", p.he_scale);
    p.coupling_ebudget = env_f("SB_EBUD", p.coupling_ebudget);
    p.implicit = env_u("SB_IMPL", p.implicit as usize) == 1;
    p.lte_atol = env_f("SB_ATOL", p.lte_atol);
    p.lte_rtol = env_f("SB_RTOL", p.lte_rtol);
    p.chi_lte = env_f("SB_CHILTE", p.chi_lte);
    if env_u("SB_SWITCH", 1) == 0 {
        p.cost_switch = None;
    }
}

// ---------------------------------------------------------------- explicit reference (0c run_explicit)

/// The Phase 0c explicit reference (leapfrog, history committed every step), with
/// the scene's released patterns. Returns (tip trajectory at every `every`-th step, D).
fn ref_explicit(sc: &Scene, h: f64, nsteps: usize, every: usize) -> (Vec<(f64, f64, f64)>, Work) {
    let mut m = sc.m.clone();
    let nb = m.bodies.len();
    let mut q = m.q.clone();
    let mut vh = sc.v_init.clone();
    let minv: Vec<[f64; 3]> = m.bodies.iter().map(|b| if b.mass > 0.0 { [1.0 / b.mass, 1.0 / b.mass, 1.0 / b.inertia] } else { [0.0; 3] }).collect();
    let mut f = vec![[0.0f64; 3]; nb];
    let mut out = Vec::new();
    let mut w = Work::default();
    for n in 0..=nsteps {
        let t = n as f64 * h;
        for i in 0..nb {
            f[i] = m.loads[i];
        }
        for (i, fe) in sc.ext(t) {
            for r in 0..3 {
                f[i][r] += fe[r];
            }
        }
        for fa in m.facets.iter().filter(|f| f.active) {
            for pt in 0..2 {
                let (j, e) = m.eval_point(&q, fa, pt);
                for r in 0..3 {
                    f[fa.a][r] -= j.jna[r] * e.fnorm + j.jta[r] * e.ftan;
                    f[fa.b][r] -= j.jnb[r] * e.fnorm + j.jtb[r] * e.ftan;
                }
            }
            w.e_pts += 2;
        }
        m.commit(&q);
        if n % every == 0 {
            out.push((t, experiments::engine::world::probe1(sc, &m, &q, &vh), m.dissipated));
        }
        if n == nsteps {
            break;
        }
        let dt = if n == 0 { 0.5 * h } else { h };
        for i in 0..nb {
            for r in 0..3 {
                if !m.bodies[i].fixed[r] {
                    vh[i][r] = vh[i][r] + dt * minv[i][r] * f[i][r];
                }
            }
        }
        for i in 0..nb {
            for r in 0..3 {
                if !m.bodies[i].fixed[r] {
                    let dq = h * vh[i][r];
                    q[i][r] += dq;
                }
            }
        }
    }
    (out, w)
}

fn lambda_h(sc: &Scene) -> f64 {
    // the engine's own stable step for the whole model (all open)
    let mut p = Params::new(1.0, 1e9);
    p.open0 = Open0::All;
    let wd = World::new(sc.clone(), p);
    wd.he
}

// ---------------------------------------------------------------- S0a

fn s0a() {
    let sc = cantilever(1, 0.04, true, 0.005, 1.25 * 0.92);
    let h = lambda_h(&sc);
    let k = 40;
    let hm = k as f64 * h;
    let nmacro = (0.06 / hm).round() as usize;
    let nsteps = nmacro * k;
    let t0 = Instant::now();
    let (rf, _) = ref_explicit(&sc, h, nsteps, k);
    let t_ref = t0.elapsed().as_secs_f64();
    let mut p = Params::new(hm, 0.2);
    p.open0 = Open0::All;
    p.h_e_fixed = Some(h);
    let wd = World::new(sc.clone(), p);
    let t1 = Instant::now();
    let (tr, _) = run(wd, "all_open", nsteps as f64 * h, hm);
    let t_eng = t1.elapsed().as_secs_f64();
    let mut max_tip: f64 = 0.0;
    let mut max_d: f64 = 0.0;
    let mut dmax_ref: f64 = 0.0;
    let mut tipmax: f64 = 0.0;
    for (a, b) in rf.iter().zip(&tr.rows) {
        max_tip = max_tip.max((a.1 - b[1]).abs());
        tipmax = tipmax.max(a.1.abs());
        max_d = max_d.max((a.2 - b[5]).abs());
        dmax_ref = dmax_ref.max(a.2.abs());
    }
    let res = json!({
        "h": h, "H": hm, "rows": rf.len().min(tr.rows.len()),
        "tip_maxdiff_rel": max_tip / tipmax, "d_maxdiff_rel": if dmax_ref > 0.0 { max_d / dmax_ref } else { max_d },
        "d_end_ref": rf.last().unwrap().2, "d_end_engine": tr.d_end,
        "wall_ref_s": t_ref, "wall_engine_s": t_eng,
        "pass": max_tip / tipmax <= 1e-8 && (dmax_ref == 0.0 || max_d / dmax_ref <= 1e-8),
    });
    println!("S0a {}", serde_json::to_string_pretty(&res).unwrap());
    write_json(EXP, "s0a", &json!({"meta": meta(EXP, "s0a", Some(1), "engine all-open vs 0c explicit", json!({})), "result": res}));
}

// ---------------------------------------------------------------- elastic cantilever metrics

fn first_min(rows: &[[f64; 16]], t_from: f64) -> (f64, f64) {
    for k in 1..rows.len() - 1 {
        if rows[k][0] > t_from && rows[k][1] <= rows[k - 1][1] && rows[k][1] < rows[k + 1][1] {
            return (rows[k][0], rows[k][1]);
        }
    }
    (f64::NAN, f64::NAN)
}

fn period(rows: &[[f64; 16]], t_from: f64) -> f64 {
    let mut mins = Vec::new();
    for k in 1..rows.len() - 1 {
        if rows[k][0] > t_from && rows[k][1] <= rows[k - 1][1] && rows[k][1] < rows[k + 1][1] {
            mins.push(rows[k][0]);
        }
    }
    if mins.len() < 2 {
        return f64::NAN;
    }
    (mins[mins.len() - 1] - mins[0]) / (mins.len() - 1) as f64
}

fn mean_tip(rows: &[[f64; 16]], t_from: f64) -> f64 {
    let r: Vec<f64> = rows.iter().filter(|r| r[0] > t_from).map(|r| r[1]).collect();
    r.iter().sum::<f64>() / r.len() as f64
}

fn elastic_metrics(tr: &Traj) -> Value {
    let (tp, yp) = first_min(&tr.rows, 0.005);
    json!({"peak_t": tp, "peak": yp, "period": period(&tr.rows, 0.005), "mean": mean_tip(&tr.rows, 0.005),
           "e_rise": tr.e_max_rise, "ke_max": tr.ke_max, "d_num_r": tr.ledger.d_num_r, "work": tr.work_total, "wall": tr.wall_s,
           "unknowns_patches": tr.patches})
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b) / b.abs()
}

// ---------------------------------------------------------------- S0b

fn s0b() {
    let sc = cantilever(1, 0.04, false, 0.005, 1.0);
    let hm = 1.0 / 240.0;
    let t_end = env_f("SB_TEND", 0.3);
    // full BDF2 = single-cell patches
    let mut pf = Params::new(hm, 1e-6);
    pf.open0 = Open0::Never;
    pf.min_cells = 1;
    let (rf, _) = run(World::new(sc.clone(), pf), "full_bdf2", t_end, 0.0005);
    let mf = elastic_metrics(&rf);
    println!("full BDF2: {mf}");
    let mut rows = vec![json!({"label": "full_bdf2", "m": mf})];
    for &lp in &[0.1, 0.15, 0.2, 0.3, 0.4] {
        for &deg in &[1usize, 2, 3] {
            let mut p = Params::new(hm, lp);
            p.open0 = Open0::Never;
            p.deg = deg;
            let wd = World::new(sc.clone(), p);
            let np = wd.patches.len();
            let n = wd.rs.n;
            let (tr, _) = run(wd, &format!("lp{lp}_P{deg}"), t_end, 0.0005);
            let mm = elastic_metrics(&tr);
            let d = json!({
                "peak": rel(mm["peak"].as_f64().unwrap(), mf["peak"].as_f64().unwrap()),
                "period": rel(mm["period"].as_f64().unwrap(), mf["period"].as_f64().unwrap()),
                "mean": rel(mm["mean"].as_f64().unwrap(), mf["mean"].as_f64().unwrap()),
                "work_ratio_vs_full_bdf2": mf["work"].as_f64().unwrap() / mm["work"].as_f64().unwrap(),
            });
            println!("lp {lp} P {deg}: patches {np} unknowns {n} rel {d}");
            rows.push(json!({"lp": lp, "deg": deg, "patches": np, "unknowns": n, "m": mm, "rel": d}));
        }
    }
    write_json(EXP, "s0b", &json!({"meta": meta(EXP, "s0b", Some(1), "patch model vs full BDF2, elastic cantilever", json!({"H": hm, "t_end": t_end})), "rows": rows}));
}

// ---------------------------------------------------------------- S0c

fn s0c() {
    let sc = cantilever(1, 0.04, false, 0.005, 1.0);
    let hm = 1.0 / 240.0;
    let t_end = env_f("SB_TEND", 0.3);
    let lp = env_f("SB_LP", 0.2);
    let deg = env_u("SB_DEG", 2);
    let mut pr = Params::new(hm, lp);
    pr.open0 = Open0::All;
    let (rf, _) = run(World::new(sc.clone(), pr), "explicit", t_end, 0.0005);
    let mr = elastic_metrics(&rf);
    println!("explicit: {mr}");
    let mut p = Params::new(hm, lp);
    p.open0 = Open0::XLess(0.4);
    p.rereduce = false;
    p.deg = deg;
    apply_env(&mut p);
    let wd = World::new(sc.clone(), p);
    println!("open cells {} of {}", wd.e_cells(), wd.sc.m.n_cells());
    let (tr, _) = run(wd, "coupled", t_end, 0.0005);
    let mm = elastic_metrics(&tr);
    println!("coupled: {mm}");
    let d = json!({
        "peak": rel(mm["peak"].as_f64().unwrap(), mr["peak"].as_f64().unwrap()),
        "period": rel(mm["period"].as_f64().unwrap(), mr["period"].as_f64().unwrap()),
        "mean": rel(mm["mean"].as_f64().unwrap(), mr["mean"].as_f64().unwrap()),
        "e_rise_over_ke_max": tr.e_max_rise / tr.ke_max,
        "work_ratio": rf.work_total / tr.work_total,
        "coupling_iters_per_step": tr.work.coupling_iters as f64 / tr.work.macro_steps as f64,
        "ledger": tr.ledger,
    });
    println!("S0c rel {d}");
    save_traj(EXP, &rf);
    save_traj(EXP, &tr);
    write_json(EXP, "s0c", &json!({"meta": meta(EXP, "s0c", Some(1), "a priori open root + closed patches vs explicit", json!({"H": hm, "lp": lp, "deg": deg})), "ref": summary(&rf), "coupled": summary(&tr), "rel": d}));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    faer::set_global_parallelism(faer::Par::Seq);
    let _ = C_PT_FORCE;
    match a.get(1).map(|s| s.as_str()) {
        Some("s0a") => s0a(),
        Some("s0b") => s0b(),
        Some("s0c") => s0c(),
        Some("cant") => dev_cant(),
        Some("imp") => dev_imp(),
        Some("col") => dev_col(),
        Some("s1") => s1(),
        Some("s2") => s2(),
        Some("s3") => s3(),
        Some("s3b") => s3b(),
        Some("s1noise") => s1_noise(),
        Some("s4b") => s4b(),
        Some("s4") => s4(),
        Some("s5") => s5(),
        Some("span") => dev_span(),
        Some("wall") => dev_wall(),
        _ => eprintln!("usage: phase0d s0a|s0b|s0c|..."),
    }
}

#[allow(dead_code)]
fn dev_cant() {
    let sc = cantilever_len(env_u("SB_SEED", 1) as u64, env_f("SB_DMIN", 0.04), env_u("SB_DAMAGE", 0) == 1, 0.005, env_f("SB_G", 1.0), env_f("SB_LEN", 4.0));
    let mut p = Params::new(env_f("SB_H", 1.0 / 240.0), env_f("SB_LP", 0.2));
    p.deg = env_u("SB_DEG", 2);
    p.min_cells = env_u("SB_MINC", 4);
    p.open0 = match std::env::var("SB_OPEN").as_deref() {
        Ok("all") => Open0::All,
        Ok("never") => Open0::Never,
        Ok(x) if x.starts_with("x") => Open0::XLess(x[1..].parse().unwrap()),
        _ => Open0::None,
    };
    p.rereduce = env_u("SB_RR", 1) == 1;
    apply_env(&mut p);
    let wd = World::new(sc, p);
    eprintln!("patches {} unknowns {} open cells {}", wd.patches.len(), wd.rs.n, wd.e_cells());
    let label = std::env::var("SB_LABEL").unwrap_or("dev".into());
    let (tr, _) = run(wd, &label, env_f("SB_TEND", 0.3), 0.0005);
    println!("{}", serde_json::to_string_pretty(&json!({"work": tr.work, "total": tr.work_total, "wall": tr.wall_s, "e_rise": tr.e_max_rise, "ke_max": tr.ke_max, "ledger": tr.ledger, "d_end": tr.d_end, "frags": tr.frags, "fragment": tr.fragment, "sep_t": tr.sep_t, "onset": tr.onset_t, "max_e_cells": tr.max_e_cells, "mean_e_frac": tr.mean_e_frac, "chi_r_max": tr.chi_r_max, "failed": tr.failed_at})).unwrap());
    save_traj(EXP, &tr);
}

#[allow(dead_code)]
fn dev_imp() {
    let sc = beam_impact(env_u("SB_SEED", 1) as u64, env_f("SB_DMIN", 0.025), env_f("SB_V0", 0.6), 50.0, env_u("SB_NSPAN", 1));
    let mut p = Params::new(env_f("SB_H", 1.0 / 240.0), env_f("SB_LP", 0.1));
    p.deg = env_u("SB_DEG", 2);
    p.open0 = match std::env::var("SB_OPEN").as_deref() {
        Ok("all") => Open0::All,
        Ok("never") => Open0::Never,
        _ => Open0::None,
    };
    p.rereduce = env_u("SB_RR", 1) == 1;
    apply_env(&mut p);
    let wd = World::new(sc, p);
    eprintln!("patches {} unknowns {} open cells {}", wd.patches.len(), wd.rs.n, wd.e_cells());
    let (tr, _) = run(wd, "dev", env_f("SB_TEND", 0.05), 0.0001);
    println!("{}", serde_json::to_string_pretty(&json!({"work": tr.work, "total": tr.work_total, "wall": tr.wall_s, "e_rise": tr.e_max_rise, "ke_max": tr.ke_max, "ledger": tr.ledger, "d_end": tr.d_end, "frags": tr.frags, "fragment": tr.fragment, "sep_t": tr.sep_t, "onset": tr.onset_t, "max_e_cells": tr.max_e_cells, "mean_e_frac": tr.mean_e_frac, "chi_r_max": tr.chi_r_max, "failed": tr.failed_at, "v_impactor": tr.v_impactor, "crack": tr.crack})).unwrap());
    save_traj(EXP, &tr);
}

#[allow(dead_code)]
fn dev_col() {
    let sc = collapse_beam(env_u("SB_SEED", 1) as u64, env_f("SB_DMIN", 0.04), env_f("SB_LEN", 6.0), env_f("SB_DEPTH", 0.3), env_f("SB_HEIGHT", 1.0), env_f("SB_MU", 0.5), env_f("SB_T2", 0.5));
    let mut p = Params::new(env_f("SB_H", 1.0 / 240.0), env_f("SB_LP", 0.2));
    p.open0 = match std::env::var("SB_OPEN").as_deref() {
        Ok("all") => Open0::All,
        Ok("never") => Open0::Never,
        _ => Open0::None,
    };
    p.rereduce = env_u("SB_RR", 1) == 1;
    p.he_scale = env_f("SB_HESCALE", 1.0);
    apply_env(&mut p);
    let wd = World::new(sc, p);
    eprintln!("cells {} patches {} unknowns {} open cells {} contact pairs {}", wd.sc.m.n_cells(), wd.patches.len(), wd.rs.n, wd.e_cells(), wd.cpairs.len());
    let label = std::env::var("SB_LABEL").unwrap_or("dev".into());
    let (tr, wd) = run(wd, &label, env_f("SB_TEND", 1.2), 0.001);
    let frags = experiments::engine::world::fragments(&wd.sc.m, &wd.q, &wd.v, 0.05);
    println!("{}", serde_json::to_string_pretty(&json!({"work": tr.work, "total": tr.work_total, "wall": tr.wall_s, "e_rise": tr.e_max_rise, "ke_max": tr.ke_max, "ledger": tr.ledger, "d_end": tr.d_end, "frags": frags, "fragment": tr.fragment, "sep_t": tr.sep_t, "onset": tr.onset_t, "max_e_cells": tr.max_e_cells, "mean_e_frac": tr.mean_e_frac, "chi_r_max": tr.chi_r_max, "failed": tr.failed_at, "v_impactor": tr.v_impactor, "crack": tr.crack, "windows": wd.windows, "ymin": (0..wd.sc.m.n_cells()).map(|i| wd.q[i][1]).fold(f64::INFINITY, f64::min), "below": (0..wd.sc.m.n_cells()).filter(|&i| wd.q[i][1] < -env_f("SB_HEIGHT", 1.0)).count()})).unwrap());
    save_traj(EXP, &tr);
}

// ---------------------------------------------------------------- frozen engine parameters

/// The implicit adaptive engine as frozen after development (docs §5): patches of
/// ℓ_p ≈ 5 d_min, P = 2, χ_trig 0.9, local error 0.1 δ₀ + 2 %, cost switch at 1.
fn engine_params(lp: f64) -> Params {
    let mut p = Params::new(1.0 / 240.0, lp);
    p.implicit = true;
    p.lte_atol = 0.1;
    p.lte_rtol = 0.02;
    apply_env(&mut p);
    p
}

fn reference_params() -> Params {
    let mut p = Params::new(env_f("SB_HREF", 1.0 / 240.0), 1e9);
    p.open0 = Open0::All;
    p
}

fn par_map<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    let nthreads = env_u("SB_THREADS", 8);
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

// ---------------------------------------------------------------- S1

fn s1_run(seed: u64, lam: f64, engine: bool, t_end: f64) -> Traj {
    let sc = cantilever(seed, 0.04, true, 0.005, lam * 0.92 / 0.92);
    let p = if engine { engine_params(0.2) } else { reference_params() };
    let (tr, _) = run(World::new(sc, p), if engine { "engine" } else { "reference" }, t_end, 0.001);
    tr
}

fn s1() {
    let seeds: Vec<u64> = vec![1, 2];
    let t_end = 0.3;
    // thresholds: bisection on λ ∈ [0.6, 1.6], 6 steps, per seed and method
    let jobs: Vec<(u64, bool)> = seeds.iter().flat_map(|&s| [(s, false), (s, true)]).collect();
    let crit: Vec<(u64, bool, f64, Vec<(f64, bool)>)> = par_map(jobs, |(seed, engine)| {
        let (mut lo, mut hi) = (0.6, 1.6);
        let mut hist = Vec::new();
        for _ in 0..6 {
            let mid = 0.5 * (lo + hi);
            let tr = s1_run(seed, mid, engine, t_end);
            let broken = tr.fragment.separated;
            hist.push((mid, broken));
            if broken {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        (seed, engine, 0.5 * (lo + hi), hist)
    });
    let lam_ref = |s: u64| crit.iter().find(|c| c.0 == s && !c.1).unwrap().2;
    // consequences at 1.25 λ_crit,ref
    let jobs: Vec<(u64, bool)> = seeds.iter().flat_map(|&s| [(s, false), (s, true)]).collect();
    let trs: Vec<(u64, bool, Traj)> = par_map(jobs, |(seed, engine)| (seed, engine, s1_run(seed, 1.25 * lam_ref(seed), engine, t_end)));
    let mut out = Vec::new();
    for &seed in &seeds {
        let rf = &trs.iter().find(|x| x.0 == seed && !x.1).unwrap().2;
        let en = &trs.iter().find(|x| x.0 == seed && x.1).unwrap().2;
        let lc_r = lam_ref(seed);
        let lc_e = crit.iter().find(|c| c.0 == seed && c.1).unwrap().2;
        let fr = &rf.fragment;
        let fe = &en.fragment;
        let vr = (fr.v[0].powi(2) + fr.v[1].powi(2)).sqrt();
        let dv = ((fe.v[0] - fr.v[0]).powi(2) + (fe.v[1] - fr.v[1]).powi(2)).sqrt() / vr;
        let ke = |f: &experiments::engine::world::Fragment| 0.5 * f.mass * (f.v[0].powi(2) + f.v[1].powi(2));
        let frame = 1.0 / 60.0;
        let c = json!({
            "threshold": {"ref": lc_r, "engine": lc_e, "rel": (lc_e - lc_r) / lc_r, "pass": ((lc_e - lc_r) / lc_r).abs() <= 0.10},
            "separation": {"ref": fr.separated, "engine": fe.separated, "pass": fr.separated && fe.separated},
            "onset": {"ref": rf.onset_t, "engine": en.onset_t, "pass": match (rf.onset_t, en.onset_t) { (Some(a), Some(b)) => (a - b).abs() <= frame, _ => false }},
            "sep_t": {"ref": rf.sep_t, "engine": en.sep_t, "pass": match (rf.sep_t, en.sep_t) { (Some(a), Some(b)) => (a - b).abs() <= frame, _ => false }},
            "crack_x": {"ref": rf.crack[0], "engine": en.crack[0], "pass": (rf.crack[0] - en.crack[0]).abs() <= 0.1},
            "D": {"ref": rf.d_end, "engine": en.d_end, "rel": (en.d_end - rf.d_end) / rf.d_end, "pass": ((en.d_end - rf.d_end) / rf.d_end).abs() <= 0.10},
            "frag_v": {"ref": fr.v, "engine": fe.v, "rel": dv, "pass": dv <= 0.10},
            "frag_omega": {"ref": fr.omega, "engine": fe.omega, "rel": (fe.omega - fr.omega) / fr.omega.abs(), "pass": ((fe.omega - fr.omega) / fr.omega.abs()).abs() <= 0.10},
            "frag_ke": {"ref": ke(fr), "engine": ke(fe), "rel": (ke(fe) - ke(fr)) / ke(fr), "pass": ((ke(fe) - ke(fr)) / ke(fr)).abs() <= 0.10},
            "no_creation": {"e_rise": en.e_max_rise, "ke_max": en.ke_max, "pass": en.e_max_rise <= 1e-3 * en.ke_max},
            "numerical_dissipation": en.ledger.d_num_r, "d_red": en.ledger.d_red, "d_fric": en.ledger.d_fric,
            "work_ratio": rf.work_total / en.work_total, "wall_ratio": rf.wall_s / en.wall_s,
            "zone": {"max": en.max_e_cells, "mean_frac": en.mean_e_frac, "cells": en.cells},
        });
        println!("S1 seed {seed}: {}", serde_json::to_string(&c).unwrap());
        save_traj("phase0d/s1", rf);
        save_traj("phase0d/s1", en);
        out.push(json!({"seed": seed, "criteria": c, "ref": summary(rf), "engine": summary(en)}));
    }
    let hist: Vec<Value> = crit.iter().map(|c| json!({"seed": c.0, "engine": c.1, "lambda_crit": c.2, "bisection": c.3})).collect();
    write_json("phase0d/s1", "summary_s1", &json!({"meta": meta(EXP, "s1", None, "implicit adaptive engine vs explicit reference", serde_json::to_value(engine_params(0.2)).unwrap()), "thresholds": hist, "results": out}));
}

// ---------------------------------------------------------------- S2 / S3

fn imp_run(seed: u64, v0: f64, nspan: usize, engine: bool) -> Traj {
    let sc = beam_impact(seed, 0.025, v0, 50.0, nspan);
    let p = if engine { engine_params(0.1) } else { reference_params() };
    let (tr, _) = run(World::new(sc, p), if engine { "engine" } else { "reference" }, 0.05, 0.0001);
    tr
}

fn broken(tr: &Traj) -> bool {
    tr.frags.len() >= 2
}

fn imp_compare(rf: &Traj, en: &Traj, v0: f64) -> Value {
    let mp = 50.0;
    let imp_r = mp * (rf.v_impactor.unwrap() + v0);
    let imp_e = mp * (en.v_impactor.unwrap() + v0);
    let same_count = rf.frags.len() == en.frags.len();
    let mut frag_ok = same_count;
    let mut fdetail = Vec::new();
    if same_count {
        for (a, b) in rf.frags.iter().zip(&en.frags) {
            let dm = (b.mass - a.mass).abs() / a.mass;
            let dv = ((b.v[0] - a.v[0]).powi(2) + (b.v[1] - a.v[1]).powi(2)).sqrt();
            frag_ok &= dm <= 0.10 && dv <= 0.1 * v0;
            fdetail.push(json!({"mass_ref": a.mass, "mass": b.mass, "dv": dv}));
        }
    }
    let d_rel = if rf.d_end > 0.0 { (en.d_end - rf.d_end) / rf.d_end } else { 0.0 };
    let ke0 = 0.5 * mp * v0 * v0;
    let crack_ok = if rf.crack[0].is_nan() && en.crack[0].is_nan() { true } else { (rf.crack[0] - en.crack[0]).abs() <= 0.1 };
    json!({
        "impulse": {"ref": imp_r, "engine": imp_e, "rel": (imp_e - imp_r) / imp_r, "pass": ((imp_e - imp_r) / imp_r).abs() <= 0.10},
        "rebound": {"ref": rf.v_impactor, "engine": en.v_impactor, "pass": (rf.v_impactor.unwrap() - en.v_impactor.unwrap()).abs() <= 0.1 * v0},
        "fragments": {"ref": rf.frags.len(), "engine": en.frags.len(), "detail": fdetail, "pass": frag_ok},
        "crack_x": {"ref": rf.crack[0], "engine": en.crack[0], "pass": crack_ok},
        "D": {"ref": rf.d_end, "engine": en.d_end, "rel": d_rel, "pass": d_rel.abs() <= 0.10},
        "no_creation": {"e_rise": en.e_max_rise, "ke0": ke0, "pass": en.e_max_rise <= 1e-3 * ke0},
        "numerical_dissipation_over_ke0": en.ledger.d_num_r / ke0, "d_red": en.ledger.d_red,
        "work_ratio": rf.work_total / en.work_total, "wall_ratio": rf.wall_s / en.wall_s,
        "zone": {"max": en.max_e_cells, "mean_frac": en.mean_e_frac, "cells": en.cells},
    })
}

fn s2() {
    let seeds: Vec<u64> = vec![1, 2];
    let jobs: Vec<(u64, bool)> = seeds.iter().flat_map(|&s| [(s, false), (s, true)]).collect();
    let crit: Vec<(u64, bool, f64, Vec<(f64, bool)>)> = par_map(jobs, |(seed, engine)| {
        let (mut lo, mut hi) = (0.05f64.ln(), 10f64.ln());
        let mut hist = Vec::new();
        for _ in 0..7 {
            let mid = 0.5 * (lo + hi);
            let tr = imp_run(seed, mid.exp(), 1, engine);
            let b = broken(&tr);
            hist.push((mid.exp(), b));
            if b {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        (seed, engine, (0.5 * (lo + hi)).exp(), hist)
    });
    let vref = |s: u64| crit.iter().find(|c| c.0 == s && !c.1).unwrap().2;
    let jobs: Vec<(u64, f64, bool)> = seeds.iter().flat_map(|&s| [0.75, 1.25, 2.0].into_iter().flat_map(move |f| [(s, f, false), (s, f, true)])).collect();
    let trs: Vec<(u64, f64, bool, Traj)> = par_map(jobs, |(seed, f, engine)| (seed, f, engine, imp_run(seed, f * vref(seed), 1, engine)));
    let mut out = Vec::new();
    for &seed in &seeds {
        let vc_r = vref(seed);
        let vc_e = crit.iter().find(|c| c.0 == seed && c.1).unwrap().2;
        let thr = json!({"ref": vc_r, "engine": vc_e, "rel": (vc_e - vc_r) / vc_r, "pass": ((vc_e - vc_r) / vc_r).abs() <= 0.10});
        println!("S2 seed {seed} threshold {thr}");
        let mut cons = Vec::new();
        for f in [0.75, 1.25, 2.0] {
            let rf = &trs.iter().find(|x| x.0 == seed && x.1 == f && !x.2).unwrap().3;
            let en = &trs.iter().find(|x| x.0 == seed && x.1 == f && x.2).unwrap().3;
            let c = imp_compare(rf, en, f * vc_r);
            println!("S2 seed {seed} at {f} v_crit: {}", serde_json::to_string(&c).unwrap());
            save_traj("phase0d/s2", rf);
            save_traj("phase0d/s2", en);
            cons.push(json!({"factor": f, "v0": f * vc_r, "criteria": c, "ref": summary(rf), "engine": summary(en)}));
        }
        out.push(json!({"seed": seed, "threshold": thr, "consequences": cons}));
    }
    let hist: Vec<Value> = crit.iter().map(|c| json!({"seed": c.0, "engine": c.1, "v_crit": c.2, "bisection": c.3})).collect();
    write_json("phase0d/s2", "summary_s2", &json!({"meta": meta(EXP, "s2", None, "implicit adaptive engine vs explicit reference", serde_json::to_value(engine_params(0.1)).unwrap()), "thresholds": hist, "results": out}));
}

fn s3() {
    let ns: Vec<usize> = std::env::var("SB_NS").ok().map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1, 2, 4, 8, 16]);
    let v0 = env_f("SB_V0", 1.5);
    let jobs: Vec<(usize, bool)> = ns.iter().flat_map(|&n| [(n, false), (n, true)]).collect();
    let trs: Vec<(usize, bool, Traj)> = par_map(jobs, |(n, engine)| (n, engine, imp_run(1, v0, n, engine)));
    let mut out = Vec::new();
    for &n in &ns {
        let rf = &trs.iter().find(|x| x.0 == n && !x.1).unwrap().2;
        let en = &trs.iter().find(|x| x.0 == n && x.1).unwrap().2;
        let c = imp_compare(rf, en, v0);
        println!("S3 n={n} cells {}: {}", rf.cells, serde_json::to_string(&c).unwrap());
        out.push(json!({"nspan": n, "cells": rf.cells, "work_ref": rf.work_total, "work_engine": en.work_total, "wall_ref": rf.wall_s, "wall_engine": en.wall_s, "criteria": c, "ref": summary(rf), "engine": summary(en)}));
    }
    // marginal cost per cell: least-squares slope of work vs cells
    let fit = |sel: &dyn Fn(&Value) -> f64| -> (f64, f64) {
        let xs: Vec<f64> = out.iter().map(|o| o["cells"].as_f64().unwrap()).collect();
        let ys: Vec<f64> = out.iter().map(|o| sel(o)).collect();
        let n = xs.len() as f64;
        let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
        let b = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
        (my - b * mx, b)
    };
    let (_, b_ref) = fit(&|o| o["work_ref"].as_f64().unwrap());
    let (_, b_eng) = fit(&|o| o["work_engine"].as_f64().unwrap());
    let res = json!({"b_ref": b_ref, "b_engine": b_eng, "marginal_ratio": b_eng / b_ref, "pass_marginal": b_eng / b_ref <= 0.1});
    println!("S3 scaling {res}");
    write_json("phase0d/s3", &format!("summary_s3_v{v0}"), &json!({"meta": meta(EXP, "s3", Some(1), "continuous beams, impact at mid span", json!({"v0": v0, "engine": engine_params(0.1)})), "runs": out, "scaling": res}));
}

/// S3b (post-hoc): the same local impact on walls of growing width.
fn s3b() {
    let ws: Vec<f64> = std::env::var("SB_WS").ok().map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1.0, 2.0, 4.0, 8.0, 16.0]);
    let v0 = env_f("SB_V0", 3.0);
    let mp = env_f("SB_MP", 50.0);
    let t_end = env_f("SB_TEND", 0.03);
    let jobs: Vec<(f64, bool)> = ws.iter().flat_map(|&w| [(w, false), (w, true)]).collect();
    let trs: Vec<(f64, bool, Traj)> = par_map(jobs, |(w, engine)| {
        let sc = wall_impact(1, 0.04, w, 1.0, v0, mp);
        let p = if engine { engine_params(0.2) } else { reference_params() };
        let (tr, _) = run(World::new(sc, p), if engine { "engine" } else { "reference" }, t_end, 0.0001);
        (w, engine, tr)
    });
    let mut out = Vec::new();
    for &w in &ws {
        let rf = &trs.iter().find(|x| x.0 == w && !x.1).unwrap().2;
        let en = &trs.iter().find(|x| x.0 == w && x.1).unwrap().2;
        let c = imp_compare(rf, en, v0);
        println!("S3b W={w} cells {}: {}", rf.cells, serde_json::to_string(&c).unwrap());
        out.push(json!({"width": w, "cells": rf.cells, "work_ref": rf.work_total, "work_engine": en.work_total, "wall_ref": rf.wall_s, "wall_engine": en.wall_s, "criteria": c, "ref": summary(rf), "engine": summary(en)}));
    }
    let fit = |sel: &dyn Fn(&Value) -> f64| -> f64 {
        let xs: Vec<f64> = out.iter().map(|o| o["cells"].as_f64().unwrap()).collect();
        let ys: Vec<f64> = out.iter().map(|o| sel(o)).collect();
        let n = xs.len() as f64;
        let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
        xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>()
    };
    let b_ref = fit(&|o| o["work_ref"].as_f64().unwrap());
    let b_eng = fit(&|o| o["work_engine"].as_f64().unwrap());
    let res = json!({"b_ref": b_ref, "b_engine": b_eng, "marginal_ratio": b_eng / b_ref});
    println!("S3b scaling {res}");
    write_json("phase0d/s3", &format!("summary_s3b_v{v0}"), &json!({"meta": meta(EXP, "s3b", Some(1), "post-hoc: wall on a foundation, impact at the top centre", json!({"v0": v0, "mp": mp, "engine": engine_params(0.2)})), "runs": out, "scaling": res}));
}

/// S1 noise floor (post-hoc): fragment outcome under λ perturbations of ±0.05 %, ±0.1 %,
/// for the reference and the engine, at 1.25 λ_crit,ref.
fn s1_noise() {
    let lam_crit = [(1u64, env_f("SB_LC1", 0.9203125)), (2u64, env_f("SB_LC2", 0.9046875))];
    let ks: Vec<f64> = std::env::var("SB_KS").ok().map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let jobs: Vec<(u64, f64, f64, bool)> = lam_crit.iter().flat_map(|&(s, lc)| ks.clone().into_iter().flat_map(move |k| [(s, lc, k, false), (s, lc, k, true)])).collect();
    let trs: Vec<(u64, f64, bool, Traj)> = par_map(jobs, |(seed, lc, k, engine)| {
        let lam = 1.25 * lc * (1.0 + 0.0005 * k);
        (seed, k, engine, s1_run(seed, lam, engine, 0.3))
    });
    let mut out = Vec::new();
    for (seed, k, engine, tr) in &trs {
        let f = &tr.fragment;
        println!("seed {seed} k {k:+} {} sep {} v [{:.4}, {:.4}] omega {:.4} D {:.3} mass {:.1}", if *engine { "engine" } else { "ref   " }, f.separated, f.v[0], f.v[1], f.omega, tr.d_end, f.mass);
        out.push(json!({"seed": seed, "k": k, "engine": engine, "fragment": f, "d": tr.d_end, "work": tr.work_total}));
    }
    let tag = std::env::var("SB_NOISETAG").unwrap_or_default();
    write_json("phase0d/s1", &format!("summary_s1_noise{tag}"), &json!({"meta": meta(EXP, "s1_noise", None, "post-hoc: λ perturbations ±0.05 %, ±0.1 % at 1.25 λ_crit,ref", json!({})), "runs": out}));
}

/// S4b (post-hoc): mass failure of a long continuous beam whose intermediate supports
/// are all released at once; reference at h_E and h_E/2 (noise floor) and the engine.
fn s4b() {
    let len = env_f("SB_LEN", 12.0);
    let nsup = env_u("SB_NSUP", 5);
    let t_end = env_f("SB_TEND", 0.5);
    s4_common("s4b", t_end, &|seed| span_collapse(seed, 0.04, len, 0.4, nsup), json!({"len": len, "nsup": nsup}));
}

/// S4 (pre-registered scene, post-hoc support timing and contact geometry): collapse
/// onto the ground.
fn s4() {
    let t_end = env_f("SB_TEND", 1.2);
    s4_common("s4", t_end, &|seed| collapse_beam(seed, 0.04, 6.0, 0.3, 1.0, 0.5, 0.5), json!({"len": 6.0, "depth": 0.3, "height": 1.0, "mu": 0.5, "t2": 0.5}));
}

fn s4_common(tag: &str, t_end: f64, scene: &(dyn Fn(u64) -> Scene + Sync), cfg: Value) {
    let jobs: Vec<(&str, u64)> = vec![("reference", 1), ("reference_half", 1), ("engine", 1), ("reference", 2), ("reference_half", 2), ("engine", 2)];
    let trs: Vec<(String, u64, Traj, Vec<experiments::engine::world::Fragment>)> = par_map(jobs, |(kind, seed)| {
        let sc = scene(seed);
        let mut p = if kind == "engine" { engine_params(0.2) } else { reference_params() };
        if kind != "engine" {
            p.h_macro = 1e-3;
        }
        if kind == "reference_half" {
            p.he_scale = 0.5;
        }
        let (tr, wd) = run(World::new(sc, p), kind, t_end, 0.001);
        let fr = experiments::engine::world::fragments(&wd.sc.m, &wd.q, &wd.v, 0.05);
        (kind.to_string(), seed, tr, fr)
    });
    let mut out = Vec::new();
    for seed in [1u64, 2] {
        let get = |k: &str| trs.iter().find(|x| x.0 == k && x.1 == seed).unwrap();
        let (rf, rh, en) = (get("reference"), get("reference_half"), get("engine"));
        let cdf = |fr: &Vec<experiments::engine::world::Fragment>| -> Vec<f64> {
            let mut m: Vec<f64> = fr.iter().map(|f| f.mass).collect();
            m.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let tot: f64 = rf.2.cells as f64; let _ = tot;
            let total_mass: f64 = m.iter().sum::<f64>().max(1e-30);
            let mut acc = 0.0;
            m.iter().map(|x| { acc += x; acc / total_mass }).collect()
        };
        let dist = |a: &Vec<f64>, b: &Vec<f64>| -> f64 {
            let n = a.len().max(b.len());
            (0..n).map(|k| (a.get(k).copied().unwrap_or(1.0) - b.get(k).copied().unwrap_or(1.0)).abs()).fold(0.0, f64::max)
        };
        let (cr, ch, ce) = (cdf(&rf.3), cdf(&rh.3), cdf(&en.3));
        let noise_d = ((rh.2.d_end - rf.2.d_end) / rf.2.d_end).abs();
        let d_rel = (en.2.d_end - rf.2.d_end) / rf.2.d_end;
        let res = json!({
            "D": {"ref": rf.2.d_end, "ref_half": rh.2.d_end, "engine": en.2.d_end, "rel": d_rel, "noise": noise_d, "pass": d_rel.abs() <= 0.10f64.max(1.5 * noise_d)},
            "large_fragments": {"ref": rf.3.len(), "ref_half": rh.3.len(), "engine": en.3.len(), "pass": (rf.3.len() as i64 - en.3.len() as i64).abs() <= 1},
            "mass_cdf_distance": {"engine": dist(&cr, &ce), "noise": dist(&cr, &ch), "pass": dist(&cr, &ce) <= 0.1},
            "no_creation": {"e_rise": en.2.e_max_rise, "ke_max": en.2.ke_max, "pass": en.2.e_max_rise <= 1e-3 * en.2.ke_max},
            "numerical_dissipation_over_ke_max": en.2.ledger.d_num_r / en.2.ke_max, "d_red_over_ke_max": en.2.ledger.d_red / en.2.ke_max,
            "work_ratio": rf.2.work_total / en.2.work_total, "wall_ratio": rf.2.wall_s / en.2.wall_s,
            "zone": {"max": en.2.max_e_cells, "mean_frac": en.2.mean_e_frac, "cells": en.2.cells},
            "fragments": {"ref": rf.3, "engine": en.3},
        });
        println!("{tag} seed {seed}: {}", serde_json::to_string(&res).unwrap());
        save_traj("phase0d/s4", &rf.2);
        save_traj("phase0d/s4", &en.2);
        out.push(json!({"seed": seed, "criteria": res, "ref": summary(&rf.2), "ref_half": summary(&rh.2), "engine": summary(&en.2)}));
    }
    write_json("phase0d/s4", &format!("summary_{tag}"), &json!({"meta": meta(EXP, tag, None, "mass failure: reference at h_E and h_E/2 and the engine", json!({"scene": cfg, "t_end": t_end, "engine": engine_params(0.2)})), "results": out}));
}

#[allow(dead_code)]
fn dev_span() {
    let sc = span_collapse(env_u("SB_SEED", 1) as u64, 0.04, env_f("SB_LEN", 12.0), 0.4, env_u("SB_NSUP", 5));
    let p = if std::env::var("SB_OPEN").as_deref() == Ok("all") { reference_params() } else { engine_params(0.2) };
    let label = std::env::var("SB_LABEL").unwrap_or("dev".into());
    let wd = World::new(sc, p);
    eprintln!("cells {} patches {}", wd.sc.m.n_cells(), wd.patches.len());
    let (tr, wd) = run(wd, &label, env_f("SB_TEND", 0.5), 0.001);
    let fr = experiments::engine::world::fragments(&wd.sc.m, &wd.q, &wd.v, 0.05);
    let m: Vec<f64> = fr.iter().map(|f| (f.mass * 10.0).round() / 10.0).collect();
    println!("D {:.2} frags {:?} work {:.2}G wall {:.1}s mean_e {:.3} max_e {} e_rise {:.3e} ke_max {:.1} dnum {:.2} dred {:.3} windows {:?}", tr.d_end, m, tr.work_total / 1e9, tr.wall_s, tr.mean_e_frac, tr.max_e_cells, tr.e_max_rise, tr.ke_max, tr.ledger.d_num_r, tr.ledger.d_red, wd.windows);
}

#[allow(dead_code)]
fn dev_wall() {
    let sc = wall_impact(1, 0.04, env_f("SB_W", 8.0), 1.0, env_f("SB_V0", 1.5), 50.0);
    let mut p = match std::env::var("SB_MODE").as_deref() {
        Ok("ref") => reference_params(),
        Ok("multirate") => {
            let mut p = Params::new(1.0 / 240.0, 0.2);
            apply_env(&mut p);
            p.implicit = false;
            p.absorbing = env_u("SB_ABS", 1) == 1;
            p
        }
        _ => engine_params(0.2),
    };
    p.rereduce = env_u("SB_RR", 1) == 1;
    let label = std::env::var("SB_LABEL").unwrap_or("dev".into());
    let wd = World::new(sc, p);
    let (tr, wd) = run(wd, &label, env_f("SB_TEND", 0.03), 0.0001);
    let _ = wd;
    println!("D {:.3} v_imp {:.4} frags {} work {:.2}G wall {:.1}s mean_e {:.3} max_e {} e_rise {:.3e} dnum {:.3} drad {:.3} it/step {:.2} failed {:?}", tr.d_end, tr.v_impactor.unwrap(), tr.frags.len(), tr.work_total / 1e9, tr.wall_s, tr.mean_e_frac, tr.max_e_cells, tr.e_max_rise, tr.ledger.d_num_r, tr.ledger.d_rad, tr.work.coupling_iters as f64 / tr.work.macro_steps.max(1) as f64, tr.failed_at);
}

/// S5 (post-hoc): support removal on cantilevers of growing length with the same root
/// moment (λ = λ₀ (4/L)²) and time scaled with the period (t_end ∝ L²): a local event in
/// space, global in time.
fn s5() {
    let ls: Vec<f64> = std::env::var("SB_LS").ok().map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![2.0, 4.0, 8.0]);
    let lam0 = env_f("SB_LAM0", 1.15);
    let jobs: Vec<(f64, bool)> = ls.iter().flat_map(|&l| [(l, false), (l, true)]).collect();
    let trs: Vec<(f64, bool, Traj)> = par_map(jobs, |(len, engine)| {
        let lam = lam0 * (4.0 / len).powi(2);
        let sc = cantilever_len(1, 0.04, true, 0.005, lam, len);
        let p = if engine { engine_params(0.2) } else { reference_params() };
        let t_end = 0.3 * (len / 4.0).powi(2);
        let (tr, _) = run(World::new(sc, p), if engine { "engine" } else { "reference" }, t_end, 0.001 * (len / 4.0).powi(2));
        (len, engine, tr)
    });
    let mut out = Vec::new();
    for &len in &ls {
        let rf = &trs.iter().find(|x| x.0 == len && !x.1).unwrap().2;
        let en = &trs.iter().find(|x| x.0 == len && x.1).unwrap().2;
        let (fr, fe) = (&rf.fragment, &en.fragment);
        let vr = (fr.v[0].powi(2) + fr.v[1].powi(2)).sqrt();
        let dv = ((fe.v[0] - fr.v[0]).powi(2) + (fe.v[1] - fr.v[1]).powi(2)).sqrt() / vr.max(1e-12);
        let r = json!({"len": len, "cells": rf.cells, "separated": [fr.separated, fe.separated], "D_rel": (en.d_end - rf.d_end) / rf.d_end.max(1e-12), "frag_dv_rel": dv, "omega": [fr.omega, fe.omega], "work_ref": rf.work_total, "work_engine": en.work_total, "work_ratio": rf.work_total / en.work_total, "wall_ref": rf.wall_s, "wall_engine": en.wall_s, "wall_ratio": rf.wall_s / en.wall_s, "zone_mean": en.mean_e_frac, "zone_max": en.max_e_cells, "e_rise_engine": en.e_max_rise, "ke_max": en.ke_max, "d_num": en.ledger.d_num_r});
        println!("S5 {}", serde_json::to_string(&r).unwrap());
        out.push(r);
    }
    write_json("phase0d/s5", "summary_s5", &json!({"meta": meta(EXP, "s5", Some(1), "post-hoc: cantilevers of growing length, same root moment", json!({"lam0": lam0, "engine": engine_params(0.2)})), "runs": out}));
}
