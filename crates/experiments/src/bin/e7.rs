//! E7 — game-scale structures, dynamic with fracture: S0 vs S1 (budgets) vs S1+S2.
//! A: cantilever loses its propping support (load redistribution, root cracking);
//! B: column under a sudden lateral push; C1: free fall of an unloaded block
//! (zero self-weight stress); C2: release of a pre-loaded block (vibration).

use experiments::{meta, rect_tess, support_reaction, write_json, Csv};
use geometry::V2;
use model::{BoundaryAttach, FacetForce, LawOptions, Material, Model};
use serde_json::{json, Value};
use solvers::integrator::{Dyn, Scheme};
use solvers::s0::{newton, NewtonOpts};
use solvers::s1::{Alm, RhoPolicy};
use solvers::s2::{inject, S2};

const EXP: &str = "E7";
const G: f64 = 9.81;

#[derive(Clone, Copy, Debug)]
enum Solver {
    S0,
    /// S0 with event sub-stepping (implicit Euler only): halve h while max Δd > 0.2.
    S0Event,
    /// S1 with the ALM note's dynamic policy ρ = m_eff/h².
    S1 { sweeps: usize },
    /// Same schedule with another penalty policy (S3 = Penalty: no multipliers).
    S1P { sweeps: usize, policy: RhoPolicy },
    /// S1 motion, S2 forces reported only (coupling A: no feedback).
    S1S2A { sweeps: usize, policy: RhoPolicy },
    S1S2 { pre: usize, post: usize, policy: RhoPolicy },
}

fn sname(s: Solver) -> String {
    match s {
        Solver::S0 => "S0".into(),
        Solver::S0Event => "S0 + event sub-stepping".into(),
        Solver::S1S2A { sweeps, policy } => format!("S1 {policy:?} {sweeps} sweeps + S2(A) forces"),
        Solver::S1 { sweeps } => format!("S1 dyn-rho, {sweeps} sweeps"),
        Solver::S1P { sweeps, policy } => format!("{policy:?}, {sweeps} sweeps"),
        Solver::S1S2 { pre, post, policy } => format!("S1 {policy:?} + S2(B) {pre}+{post} sweeps"),
    }
}

struct Rec {
    t: Vec<f64>,
    react: Vec<f64>,
    tip: Vec<f64>,
    diss: Vec<f64>,
    energy: Vec<f64>,
    max_force_over_fail: Vec<f64>,
    damaged: Vec<usize>,
    dnum: Vec<f64>,
}

struct Scenario {
    m: Model,
    /// Steps at which events fire and what they do.
    events: Vec<(usize, Box<dyn Fn(&mut Model)>)>,
    tip_body: usize,
    react_tag: u32,
}

fn total_energy(m: &Model, q: &[[f64; 3]], d: &Dyn) -> (f64, f64, f64) {
    let t = d.kinetic(m);
    let u = m.elastic_energy(q);
    let vg = m.load_potential(q, 0.0);
    (t, u, vg)
}

/// Implicit-Euler numerical dissipation of one accepted step (after commit, so the
/// history is d_{n+1}): ½‖v₁−v₀‖²_M + [∇Π(q₁)·Δq − (Π(q₁) − Π(q₀))].
fn ie_dnum(m: &Model, q0: &[[f64; 3]], q1: &[[f64; 3]], v0: &[[f64; 3]], v1: &[[f64; 3]]) -> f64 {
    let map = solvers::assembly::DofMap::new(m);
    let a = solvers::assembly::assemble(m, q1, &map, 0.0, None, false, solvers::assembly::Tangent::Consistent);
    let mut gd = 0.0;
    for (i, row) in map.idx.iter().enumerate() {
        for c in 0..3 {
            if let Some(gi) = row[c] {
                gd += a.grad[gi] * (q1[i][c] - q0[i][c]);
            }
        }
    }
    let pi1 = m.elastic_energy(q1) + m.load_potential(q1, 0.0);
    let pi0 = m.elastic_energy(q0) + m.load_potential(q0, 0.0);
    let mut kin = 0.0;
    for (i, b) in m.bodies.iter().enumerate() {
        kin += 0.5 * b.mass * ((v1[i][0] - v0[i][0]).powi(2) + (v1[i][1] - v0[i][1]).powi(2)) + 0.5 * b.inertia * (v1[i][2] - v0[i][2]).powi(2);
    }
    kin + gd - (pi1 - pi0)
}

fn max_force_ratio(m: &Model, f: &[FacetForce]) -> f64 {
    let fail = |fi: usize| m.mat.sigma_t * m.facets[fi].len * m.mat.thickness;
    f.iter().enumerate().filter(|(i, _)| m.facets[*i].active && m.facets[*i].tag == 0).map(|(i, x)| x.n.abs().max(x.v.abs()) / fail(i)).fold(0.0, f64::max)
}

fn simulate(sc: Scenario, scheme: Scheme, h: f64, t_end: f64, solver: Solver, csv: &mut Csv, id: f64) -> (Value, Rec) {
    simulate_opt(sc, scheme, h, t_end, solver, csv, id, 60)
}

/// One trajectory. A reference step (S0, S0Event) whose Newton solve fails is not
/// accepted: nothing of that step is committed (coordinates, damage history,
/// integrator, time, counters) and the trajectory stops, marked invalid with the reason.
fn simulate_opt(mut sc: Scenario, scheme: Scheme, h: f64, t_end: f64, solver: Solver, csv: &mut Csv, id: f64, newton_max_iter: usize) -> (Value, Rec) {
    let m = &mut sc.m;
    let mut dynst = Dyn::new(scheme, h, m);
    let mut q = m.q.clone();
    let mut alm = Alm::new(m, RhoPolicy::Dynamic(1.0), Some(&dynst.inertia(&q)));
    let mut s2 = S2::new();
    let fscale = m.mat.sigma_t * 0.3 * m.mat.thickness;
    let nsteps = (t_end / h).round() as usize;
    let mut rec = Rec { t: vec![], react: vec![], tip: vec![], diss: vec![], energy: vec![], max_force_over_fail: vec![], damaged: vec![], dnum: vec![] };
    let (t0, u0, v0) = total_energy(m, &q, &dynst);
    let e0 = t0 + u0 + v0;
    let mut dnum_ie = 0.0;
    let mut max_dd: f64 = 0.0;
    let mut newton_fail = 0;
    let mut substeps_total = 0usize;
    let mut ref_err = Vec::new();
    let tip0 = m.bodies[sc.tip_body].x0;
    let mut invalid: Option<Value> = None;
    let mut steps_done = 0usize;
    for k in 0..nsteps {
        for (ks, ev) in &sc.events {
            if *ks == k {
                ev(m);
                alm.refresh_topology(m);
            }
        }
        let q_old = q.clone();
        let v_before = dynst.v.clone();
        let ine = dynst.inertia(&q_old);
        // initial guess: inertial prediction plus the external-force term (h²/α) M⁻¹ f.
        // (A first run used q̂ alone: colour-sequential S1 then strained a free-falling block
        // by ~h²g/ℓ ≈ the failure strain and broke it.)
        for i in 0..q.len() {
            for c in 0..3 {
                if !m.bodies[i].fixed[c] {
                    let mm = if c < 2 { m.bodies[i].mass } else { m.bodies[i].inertia };
                    q[i][c] = ine.qhat[i][c] + if mm > 0.0 { m.loads[i][c] / (mm * ine.alpha_h2) } else { 0.0 };
                }
            }
        }
        let mut s2_forces: Option<Vec<FacetForce>> = None;
        let mut substeps_here: usize;
        match solver {
            Solver::S0 => {
                let rep = newton(m, &mut q, 0.0, Some(&ine), NewtonOpts { tol: 1e-9 * fscale, max_iter: newton_max_iter });
                if !rep.converged {
                    newton_fail += 1;
                    invalid = Some(json!({"step": k, "t": k as f64 * h, "reason": "S0 Newton not converged", "residual": rep.res, "iters": rep.iters}));
                    break;
                }
            }
            Solver::S0Event => {
                // advance through [t, t+h] in IE substeps, committing each accepted substep
                let mut tleft = h;
                let mut hs = h;
                let mut qs = q_old.clone();
                let mut vs = dynst.v.clone();
                substeps_here = 0;
                while tleft > 1e-15 * h {
                    hs = hs.min(tleft);
                    let mut dd = Dyn::new(Scheme::ImplicitEuler, hs, m);
                    dd.v = vs.clone();
                    let ine_s = dd.inertia(&qs);
                    let mut qt = qs.clone();
                    let rep = newton(m, &mut qt, 0.0, Some(&ine_s), NewtonOpts { tol: 1e-9 * fscale, max_iter: newton_max_iter });
                    let ddmax = m.facets.iter().filter(|f| f.active).flat_map(|f| (0..2).map(move |p| (f, p))).map(|(f, p)| {
                        let (_, e) = m.eval_point(&qt, f, p);
                        e.d - f.pts[p].law.damage(f.pts[p].kappa_c)
                    }).fold(0.0, f64::max);
                    if (ddmax > 0.2 || !rep.converged) && hs > h / 256.0 {
                        hs *= 0.5;
                        continue;
                    }
                    if !rep.converged {
                        newton_fail += 1;
                        invalid = Some(json!({"step": k, "t": k as f64 * h + (h - tleft), "reason": "S0Event Newton not converged at the h/256 floor", "residual": rep.res, "iters": rep.iters}));
                        break;
                    }
                    m.commit(&qt);
                    dd.finish(&qs, &qt);
                    dnum_ie += ie_dnum(m, &qs, &qt, &vs, &dd.v);
                    vs = dd.v.clone();
                    qs = qt;
                    tleft -= hs;
                    substeps_here += 1;
                    hs = (hs * 2.0).min(h);
                }
                if invalid.is_some() {
                    break;
                }
                q = qs;
                substeps_total += substeps_here;
                // integrator velocity from the substepped motion
                dynst.v = vs.clone();
            }
            Solver::S1P { sweeps, policy } => {
                if alm.policy != policy {
                    alm = Alm::new(m, policy, Some(&ine));
                }
                alm.set_rho(m, Some(&ine));
                for _ in 0..sweeps {
                    alm.iterate(m, &mut q, 0.0, Some(&ine));
                }
            }
            Solver::S1S2A { sweeps, policy } => {
                if alm.policy != policy {
                    alm = Alm::new(m, policy, Some(&ine));
                }
                alm.set_rho(m, Some(&ine));
                for _ in 0..sweeps {
                    alm.iterate(m, &mut q, 0.0, Some(&ine));
                }
                let a = dynst.accel_of(&q_old, &q);
                if let Ok(out) = s2.reconstruct(m, &q, 0.0, Some(&a)) {
                    s2_forces = Some(out.forces);
                }
            }
            Solver::S1 { sweeps } => {
                alm.set_rho(m, Some(&ine));
                for _ in 0..sweeps {
                    alm.iterate(m, &mut q, 0.0, Some(&ine));
                }
            }
            Solver::S1S2 { pre, post, policy } => {
                if alm.policy != policy {
                    alm = Alm::new(m, policy, Some(&ine));
                }
                alm.set_rho(m, Some(&ine));
                for _ in 0..pre {
                    alm.iterate(m, &mut q, 0.0, Some(&ine));
                }
                let a = dynst.accel_of(&q_old, &q);
                if let Ok(out) = s2.reconstruct(m, &q, 0.0, Some(&a)) {
                    inject(&mut alm, &out);
                }
                for _ in 0..post {
                    alm.iterate(m, &mut q, 0.0, Some(&ine));
                }
            }
        }
        // implicit-Euler numerical dissipation (independent identification): kinetic
        // part ½‖Δv‖²_M + Bregman part ∇Π(q₁)·Δq − ΔΠ at d_{n+1}
        let dd = {
            let mut mm = 0.0f64;
            for f in m.facets.iter().filter(|f| f.active) {
                for p in 0..2 {
                    let (_, e) = m.eval_point(&q, f, p);
                    mm = mm.max(e.d - f.pts[p].law.damage(f.pts[p].kappa_c));
                }
            }
            mm
        };
        max_dd = max_dd.max(dd);
        let v_old = if matches!(solver, Solver::S0Event) { v_before.clone() } else { dynst.v.clone() };
        if !matches!(solver, Solver::S0Event) {
            m.commit(&q);
            dynst.finish(&q_old, &q);
        } else {
            dynst.q_prev = q_old.clone();
            dynst.steps += 1;
        }
        if scheme == Scheme::ImplicitEuler && !matches!(solver, Solver::S0Event) {
            dnum_ie += ie_dnum(m, &q_old, &q, &v_old, &dynst.v);
        }
        let t = (k + 1) as f64 * h;
        let (tk, uk, vk) = total_energy(m, &q, &dynst);
        let react = support_reaction(m, &q, sc.react_tag).y;
        let tip = V2::new(q[sc.tip_body][0], q[sc.tip_body][1]) - tip0;
        let ff = m.facet_forces(&q);
        let mfr = max_force_ratio(m, &ff);
        if let Some(fs2) = &s2_forces {
            ref_err.push(max_force_ratio(m, fs2));
        }
        let damaged = m.facets.iter().filter(|f| f.active && f.pts.iter().any(|p| p.law.damage(p.kappa_c) > 0.99)).count();
        let bal = tk + uk + vk + m.dissipated - e0; // W_ext = 0 (gravity in V_g, supports fixed)
        rec.t.push(t);
        rec.react.push(react);
        rec.tip.push(tip.y);
        rec.diss.push(m.dissipated);
        rec.energy.push(bal);
        rec.max_force_over_fail.push(mfr);
        rec.damaged.push(damaged);
        rec.dnum.push(dnum_ie);
        csv.row(&[id, t, react, tip.x, tip.y, tk, uk, vk, m.dissipated, dnum_ie, bal, mfr, damaged as f64]);
        steps_done += 1;
    }
    let w = alm.work;
    let kappa_sum: f64 = m.facets.iter().flat_map(|f| f.pts.iter()).map(|p| p.kappa_c).sum();
    (
        json!({"solver": sname(solver), "scheme": scheme.name(), "h": h, "steps": nsteps, "newton_failures": newton_fail,
            "valid": invalid.is_none(), "invalid": invalid, "steps_completed": steps_done, "committed_kappa_sum": kappa_sum, "integrator_steps": dynst.steps,
            "D_final_J": m.dissipated, "fully_damaged_facets": rec.damaged.last(), "max_step_damage_increment": max_dd,
            "energy_balance_final_J": rec.energy.last(), "Dnum_IE_J": dnum_ie,
            "unexplained_residual_IE_J": if scheme == Scheme::ImplicitEuler { rec.energy.last().map(|e| e + dnum_ie) } else { None },
            "policy_note": "S1 dyn-rho = RhoPolicy::Dynamic(1.0)",
            "sweeps_total": w.sweeps, "s2_numeric_factorizations": s2.stats.numeric, "s2_symbolic": s2.stats.symbolic, "s2_reused": s2.reused, "s2_lu": s2.lu_solves, "s2_mechanism_failures": s2.mechanism_failures,
            "tip_final_m": rec.tip.last(), "max_force_over_failure": rec.max_force_over_fail.iter().cloned().fold(0.0, f64::max),
            "substeps_total": substeps_total, "s2A_max_force_over_failure": ref_err.iter().cloned().fold(0.0, f64::max)}),
        rec,
    )
}

/// A: 8 m × 1.2 m beam, clamped left end, propped at the right end; prop removed at t=0.
fn cantilever(dmin: f64, eta: Option<f64>) -> Scenario {
    let t = rect_tess(V2::new(0.0, 0.0), V2::new(8.0, 1.2), dmin, 7);
    let law = LawOptions { crack_band_eta: eta, ..LawOptions::damage() };
    let mut m = Model::from_tessellation(&t, Material::concrete(0.3), law, &[]);
    let g = m.ground();
    m.attach_boundary(&t, &BoundaryAttach { edge: 3, range: None, body: g, shear: true, tag: 1 });
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((7.4, 8.0)), body: g, shear: true, tag: 2 });
    m.add_gravity(G);
    // static propped state (S0), committed (no damage expected)
    let mut q = m.q.clone();
    let r = newton(&m, &mut q, 0.0, None, NewtonOpts { tol: 1e-6, max_iter: 50 });
    assert!(r.converged);
    m.commit(&q);
    m.q = q;
    let tip = (0..m.n_cells()).max_by(|&a, &b| m.bodies[a].x0.x.partial_cmp(&m.bodies[b].x0.x).unwrap()).unwrap();
    Scenario {
        m,
        events: vec![(0, Box::new(|m: &mut Model| m.facets.iter_mut().filter(|f| f.tag == 2).for_each(|f| f.active = false)))],
        tip_body: tip,
        react_tag: 1,
    }
}

/// B: column 1 m × 4 m clamped at the base, sudden lateral load at the top.
fn column() -> Scenario {
    let t = rect_tess(V2::new(0.0, 0.0), V2::new(1.0, 4.0), 0.25, 3);
    let mut m = Model::from_tessellation(&t, Material::concrete(0.3), LawOptions::damage(), &[]);
    let g = m.ground();
    m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: None, body: g, shear: true, tag: 1 });
    m.add_gravity(G);
    let mut q = m.q.clone();
    newton(&m, &mut q, 0.0, None, NewtonOpts { tol: 1e-6, max_iter: 50 });
    m.commit(&q);
    m.q = q;
    let top: Vec<usize> = (0..m.n_cells()).filter(|&i| m.bodies[i].x0.y > 3.75).collect();
    let tip = top[0];
    // static cracking estimate: M = σ_t b h²/6 at the base → H = M / 4 m ≈ 37.5 kN; apply 60 kN
    let f_each = 60e3 / top.len() as f64;
    Scenario {
        m,
        events: vec![(0, Box::new(move |m: &mut Model| {
            for &i in &top {
                m.loads[i][0] += f_each;
            }
        }))],
        tip_body: tip,
        react_tag: 1,
    }
}

/// C1: unloaded block in free fall. C2: block pre-loaded on a clamp, released at t=0.
fn block(preloaded: bool) -> Scenario {
    let t = rect_tess(V2::new(0.0, 0.0), V2::new(3.0, 1.0), 0.3, 5);
    let mut m = Model::from_tessellation(&t, Material::concrete(0.3), LawOptions::damage(), &[]);
    m.add_gravity(G);
    if preloaded {
        let g = m.ground();
        m.attach_boundary(&t, &BoundaryAttach { edge: 3, range: None, body: g, shear: true, tag: 1 });
        let mut q = m.q.clone();
        newton(&m, &mut q, 0.0, None, NewtonOpts { tol: 1e-6, max_iter: 50 });
        m.commit(&q);
        m.q = q;
    }
    let tip = (0..m.n_cells()).max_by(|&a, &b| m.bodies[a].x0.x.partial_cmp(&m.bodies[b].x0.x).unwrap()).unwrap();
    Scenario {
        m,
        events: if preloaded { vec![(0, Box::new(|m: &mut Model| m.facets.iter_mut().filter(|f| f.tag == 1).for_each(|f| f.active = false)))] } else { vec![] },
        tip_body: tip,
        react_tag: 1,
    }
}

fn cantilever_law(dmin: f64, law: LawOptions) -> Scenario {
    let t = rect_tess(V2::new(0.0, 0.0), V2::new(8.0, 1.2), dmin, 7);
    let mut m = Model::from_tessellation(&t, Material::concrete(0.3), law, &[]);
    let g = m.ground();
    m.attach_boundary(&t, &BoundaryAttach { edge: 3, range: None, body: g, shear: true, tag: 1 });
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((7.4, 8.0)), body: g, shear: true, tag: 2 });
    m.add_gravity(G);
    let mut q = m.q.clone();
    let r = newton(&m, &mut q, 0.0, None, NewtonOpts { tol: 1e-6, max_iter: 50 });
    assert!(r.converged);
    m.commit(&q);
    m.q = q;
    let tip = (0..m.n_cells()).max_by(|&a, &b| m.bodies[a].x0.x.partial_cmp(&m.bodies[b].x0.x).unwrap()).unwrap();
    Scenario {
        m,
        events: vec![(0, Box::new(|m: &mut Model| m.facets.iter_mut().filter(|f| f.tag == 2).for_each(|f| f.active = false)))],
        tip_body: tip,
        react_tag: 1,
    }
}


/// Reference driver: run the whole trajectory at h; if it is invalid, restart from
/// t = 0 at constant h/2 (BDF2 history is never carried across a step change), down
/// to h/8. `max_iter(attempt)` sets the Newton iteration limit per attempt.
fn simulate_ref(make: &dyn Fn() -> Scenario, scheme: Scheme, h: f64, t_end: f64, solver: Solver, csv: &mut Csv, id: f64, max_iter: &dyn Fn(usize) -> usize) -> (Value, Rec) {
    let mut attempts = Vec::new();
    let mut hk = h;
    for a in 0..4 {
        let (mut r, rec) = simulate_opt(make(), scheme, hk, t_end, solver, csv, id + 0.01 * a as f64, max_iter(a));
        let valid = r["valid"].as_bool().unwrap_or(false);
        attempts.push(json!({"h": hk, "valid": valid, "invalid": r["invalid"].clone()}));
        if valid || a == 3 {
            r["attempts"] = json!(attempts);
            r["h_requested"] = json!(h);
            return (r, rec);
        }
        hk *= 0.5;
    }
    unreachable!()
}

/// Short checks of the corrected paths (Phase 0b / 2 §3); writes to
/// results/phase0b/two-level/fix-checks, never to results/E7.
fn checks() {
    const CX: &str = "phase0b/two-level/fix-checks";
    let mut csv = Csv::new(CX, "e7_series", &["run", "t", "reaction_y", "tip_x", "tip_y", "T", "U", "Vg", "D", "Dnum_IE", "balance", "max_force_over_fail", "broken_facets"]);
    let fresh = cantilever(0.3, None);
    let kappa0: f64 = fresh.m.facets.iter().flat_map(|f| f.pts.iter()).map(|p| p.kappa_c).sum();
    let d0 = fresh.m.dissipated;
    // (a) forced Newton failure (max_iter = 1) on S0 and S0Event: invalid at step 0, nothing committed
    let mut forced = Vec::new();
    for (s, sc) in [(Solver::S0, Scheme::Bdf2), (Solver::S0Event, Scheme::ImplicitEuler)] {
        let (r, rec) = simulate_opt(cantilever(0.3, None), sc, 1.0 / 480.0, 0.02, s, &mut csv, 0.0, 1);
        let ok = r["valid"] == json!(false) && r["steps_completed"] == json!(0) && rec.t.is_empty() && r["integrator_steps"] == json!(0)
            && (r["committed_kappa_sum"].as_f64().unwrap() - kappa0).abs() == 0.0 && r["D_final_J"].as_f64().unwrap() == d0;
        eprintln!("forced failure {}: valid={} steps={} pass={ok}", sname(s), r["valid"], r["steps_completed"]);
        forced.push(json!({"solver": sname(s), "result": r, "pass": ok}));
    }
    // (b) restart driver: the first attempt is forced to fail, the second (h/2) runs normally
    let (rd, _) = simulate_ref(&|| cantilever(0.3, None), Scheme::Bdf2, 1.0 / 480.0, 0.02, Solver::S0, &mut csv, 1.0, &|a| if a == 0 { 1 } else { 60 });
    // pass = the driver restarted from t = 0 at halved constant steps and returned the
    // first valid attempt; whether h/2 itself is valid is a property of the scenario
    let att = rd["attempts"].as_array().cloned().unwrap_or_default();
    let restart_ok = rd["valid"] == json!(true)
        && att.len() >= 2
        && att[0]["valid"] == json!(false)
        && att.iter().enumerate().all(|(k, a)| a["h"].as_f64() == Some(1.0 / 480.0 / f64::powi(2.0, k as i32)))
        && att[..att.len() - 1].iter().all(|a| a["valid"] == json!(false));
    eprintln!("restart driver: valid={} h={} pass={restart_ok}", rd["valid"], rd["h"]);
    // (c) partial support / load: exact clipping vs the midpoint rule on 3 tessellations
    let mut partial = Vec::new();
    for seed in [7u64, 8, 9] {
        let t = rect_tess(V2::new(0.0, 0.0), V2::new(8.0, 1.2), 0.3, seed);
        let mut row = serde_json::Map::new();
        row.insert("seed".into(), json!(seed));
        for exact in [false, true] {
            let mut m = Model::from_tessellation(&t, Material::concrete(0.3), LawOptions::elastic(), &[]);
            let g = m.ground();
            let att = BoundaryAttach { edge: 0, range: Some((7.4, 8.0)), body: g, shear: true, tag: 2 };
            if exact {
                m.attach_boundary_exact(&t, &att);
            } else {
                m.attach_boundary(&t, &att);
            }
            let len: f64 = m.facets.iter().filter(|f| f.tag == 2).map(|f| f.len).sum();
            // partial load: −10 kPa on the top edge over x ∈ [2.3, 5.1]
            let (a, b, ty) = (2.3, 5.1, -1.0e4);
            let tot = if exact {
                experiments::edge_traction_exact(&mut m, &t, 2, Some((a, b)), V2::new(0.0, ty), false)
            } else {
                experiments::edge_traction(&mut m, &t, 2, Some((a, b)), V2::new(0.0, ty), false)
            };
            // moment about the origin of the applied loads (force at centroid + couple)
            let mo: f64 = m.bodies.iter().zip(&m.loads).map(|(bd, l)| bd.x0.cross(V2::new(l[0], l[1])) + l[2]).sum();
            let f_ex = ty * (b - a) * 0.3;
            let m_ex = ty * 0.3 * 0.5 * (b * b - a * a);
            let key = if exact { "exact" } else { "midpoint" };
            row.insert(key.into(), json!({"support_length": len, "support_length_err": (len - 0.6).abs() / 0.6,
                "load_force_err": (tot.y - f_ex).abs() / f_ex.abs(), "load_moment_err": (mo - m_ex).abs() / m_ex.abs()}));
        }
        eprintln!("partial seed {seed}: {}", serde_json::to_string(&row).unwrap());
        partial.push(Value::Object(row));
    }
    let pass_partial = partial.iter().all(|r| {
        let e = &r["exact"];
        e["support_length_err"].as_f64().unwrap() < 1e-12 && e["load_force_err"].as_f64().unwrap() < 1e-12 && e["load_moment_err"].as_f64().unwrap() < 1e-12
    });
    write_json(CX, "e7_checks", &json!({"meta": meta(CX, "e7 checks", Some(7), "forced Newton failure (max_iter=1), restart driver, partial support/load clipping", json!({})),
        "forced_failure": forced, "restart_driver": {"result": rd, "pass": restart_ok}, "partial_support_and_load": partial, "partial_pass": pass_partial}));
}

/// Newton history of the first BDF2/IE step after prop removal (diagnosis of the
/// reference failures found by `checks`): one Newton iteration per call, from the
/// same predictor as `simulate_opt`.
fn newton_diag(h: f64, iters: usize) -> Value {
    let mut sc = cantilever(0.3, None);
    for (ks, ev) in &sc.events {
        if *ks == 0 {
            ev(&mut sc.m);
        }
    }
    let m = &sc.m;
    // onset indicator of the committed propped static state: max κ_c / δ0 per class
    let chi = |root: bool| m.facets.iter().filter(|f| f.active && f.tag == 0 && ((f.pc.x < 1.0) == root)).flat_map(|f| f.pts.iter()).map(|p| p.kappa_c / p.law.delta0).fold(0.0, f64::max);
    let chi_static = json!({"root_x_lt_1m": chi(true), "elsewhere": chi(false)});
    let dynst = Dyn::new(Scheme::ImplicitEuler, h, m);
    let ine = dynst.inertia(&m.q);
    let mut q = m.q.clone();
    for i in 0..q.len() {
        for c in 0..3 {
            if !m.bodies[i].fixed[c] {
                let mm = if c < 2 { m.bodies[i].mass } else { m.bodies[i].inertia };
                q[i][c] = ine.qhat[i][c] + if mm > 0.0 { m.loads[i][c] / (mm * ine.alpha_h2) } else { 0.0 };
            }
        }
    }
    let fscale = m.mat.sigma_t * 0.3 * m.mat.thickness;
    let mut hist = Vec::new();
    for _ in 0..iters {
        let r = newton(m, &mut q, 0.0, Some(&ine), NewtonOpts { tol: 1e-9 * fscale, max_iter: 1 });
        let dmax = m.facets.iter().filter(|f| f.active).flat_map(|f| (0..2).map(move |p| (f, p))).map(|(f, p)| m.eval_point(&q, f, p).1.d).fold(0.0, f64::max);
        let loading = m.facets.iter().filter(|f| f.active).flat_map(|f| (0..2).map(move |p| (f, p))).filter(|(f, p)| m.eval_point(&q, f, *p).1.loading && m.eval_point(&q, f, *p).1.d > 0.0).count();
        let worst = m.facets.iter().enumerate().filter(|(_, f)| f.active).max_by(|a, b| {
            let da = (0..2).map(|p| m.eval_point(&q, a.1, p).1.d).fold(0.0, f64::max);
            let db = (0..2).map(|p| m.eval_point(&q, b.1, p).1.d).fold(0.0, f64::max);
            da.partial_cmp(&db).unwrap()
        }).map(|(i, f)| json!({"facet": i, "tag": f.tag, "x": f.pc.x, "y": f.pc.y, "len": f.len, "h": f.h, "a": f.a, "b": f.b})).unwrap();
        hist.push(json!({"res": r.res, "lu": r.lu_fallbacks, "secant": r.secant_fallbacks, "converged": r.converged, "d_max": dmax, "softening_points": loading, "worst": worst}));
        if r.converged {
            break;
        }
    }
    json!({"h": h, "chi_static": chi_static, "history": hist})
}

/// Relative RMS deviation of a time series from a reference series.
fn series_err(a: &[f64], r: &[f64]) -> f64 {
    let n = a.len().min(r.len());
    let num: f64 = (0..n).map(|i| (a[i] - r[i]).powi(2)).sum();
    let den: f64 = (0..n).map(|i| r[i].powi(2)).sum();
    (num / den.max(1e-300)).sqrt()
}

fn main() {
    if std::env::var("SB_E7_NEWTON_DIAG").is_ok() {
        let mut out = Vec::new();
        for &h in &[1.0 / 1920.0, 1.0 / 7680.0] {
            let d = newton_diag(h, 400);
            let hs = d["history"].as_array().unwrap();
            eprintln!("h = 1/{:.0}: {} iterations, static χ {}, last {}", 1.0 / h, hs.len(), d["chi_static"], hs.last().unwrap());
            for (k, x) in hs.iter().enumerate().filter(|(k, _)| *k < 12 || k % 40 == 0) {
                eprintln!("  it {k}: {x}");
            }
            out.push(d);
        }
        write_json("phase0b/two-level/fix-checks", "e7_newton_diag", &json!(out));
        return;
    }
    if std::env::var("SB_E7_CHECK").is_ok() {
        checks();
        return;
    }
    if std::env::var("SB_E7_DIAG").is_ok() {
        // S1 stability under inexact solves: integrator × sweeps on the elastic prop-removal case
        let mut csv = Csv::new(EXP, "diag", &["run", "t", "reaction_y", "tip_x", "tip_y", "T", "U", "Vg", "D", "Dnum_IE", "balance", "max_force_over_fail", "broken_facets"]);
        let mut out = Vec::new();
        let mut id = 0.0;
        let mut cases: Vec<(Scheme, Solver)> = Vec::new();
        let _ = RhoPolicy::Shared(0.0);
        for &sc in &[Scheme::ImplicitEuler, Scheme::Bdf2, Scheme::Newmark] {
            for &n in &[1usize, 4, 16, 64, 256] {
                cases.push((sc, Solver::S1 { sweeps: n }));
            }
        }
        for &n in &[4usize, 16, 64, 256] {
            cases.push((Scheme::ImplicitEuler, Solver::S1P { sweeps: n, policy: RhoPolicy::Penalty }));
            cases.push((Scheme::ImplicitEuler, Solver::S1P { sweeps: n, policy: RhoPolicy::PerSpring(0.1) }));
            cases.push((Scheme::Bdf2, Solver::S1P { sweeps: n, policy: RhoPolicy::Penalty }));
        }
        for (sc, solver) in cases {
            {
                let n = match solver { Solver::S1 { sweeps } | Solver::S1P { sweeps, .. } => sweeps, _ => 0 };
                let (r, rec) = simulate(cantilever_law(0.3, LawOptions::elastic()), sc, 1.0 / 480.0, 0.2, solver, &mut csv, id);
                let peak = rec.max_force_over_fail.iter().cloned().fold(0.0, f64::max);
                let first_bad = rec.max_force_over_fail.iter().position(|&x| !(x < 100.0)).map(|k| rec.t[k]);
                eprintln!("{} {} | {} sweeps: peak F/Ff {:.3e}, first >100x at {:?}, balance {:.3e}", sc.name(), sname(solver), n, peak, first_bad, r["energy_balance_final_J"].as_f64().unwrap_or(f64::NAN));
                out.push(json!({"scheme": sc.name(), "solver": sname(solver), "sweeps": n, "peak_force_over_failure": peak, "diverged_at_s": first_bad, "balance_J": r["energy_balance_final_J"]}));
                id += 1.0;
            }
        }
        write_json(EXP, "diag", &json!({"meta": meta(EXP, "diag", Some(7), "S1 dyn-rho, elastic prop removal, 0.2 s", json!({})), "runs": out}));
        return;
    }
    let head = ["run", "t", "reaction_y", "tip_x", "tip_y", "T", "U", "Vg", "D", "Dnum_IE", "balance", "max_force_over_fail", "broken_facets"];
    let mut csv = Csv::new(EXP, "series", &head);
    let mut runs = Vec::new();
    let mut id = 0.0;
    let mut info = serde_json::Map::new();
    {
        let sc = cantilever(0.3, None);
        let (lo, hi) = sc.m.band_ratio();
        info.insert("A_cells".into(), json!(sc.m.n_cells()));
        info.insert("A_ell_over_lch".into(), json!([lo, hi]));
    }
    let push = |r: &mut Value, scen: &str, id: &mut f64, runs: &mut Vec<Value>| {
        r["scenario"] = json!(scen);
        r["run"] = json!(*id);
        eprintln!("{r}");
        runs.push(r.clone());
        *id += 1.0;
    };

    // A-elastic: same prop removal with an elastic law: isolates solver/coupling error
    // from fracture. Reaction-history error vs S0.
    let mut group = Vec::new();
    for &s in &[Solver::S0, Solver::S1 { sweeps: 16 }, Solver::S1P { sweeps: 16, policy: RhoPolicy::PerSpring(0.1) }, Solver::S1P { sweeps: 64, policy: RhoPolicy::PerSpring(0.1) }, Solver::S1P { sweeps: 256, policy: RhoPolicy::PerSpring(0.1) },
                Solver::S1S2A { sweeps: 16, policy: RhoPolicy::PerSpring(0.1) }, Solver::S1S2 { pre: 8, post: 8, policy: RhoPolicy::PerSpring(0.1) }] {
        let (mut r, rec) = if matches!(s, Solver::S0) {
            simulate_ref(&|| cantilever_law(0.3, LawOptions::elastic()), Scheme::Bdf2, 1.0 / 480.0, 0.4, s, &mut csv, id, &|_| 60)
        } else {
            simulate(cantilever_law(0.3, LawOptions::elastic()), Scheme::Bdf2, 1.0 / 480.0, 0.4, s, &mut csv, id)
        };
        group.push((r.clone(), rec));
        push(&mut r, "A-elastic cantilever prop removal", &mut id, &mut runs);
    }
    let r0 = group[0].1.react.clone();
    let t0 = group[0].1.tip.clone();
    let mut elastic_cmp = Vec::new();
    for (r, rec) in &group {
        elastic_cmp.push(json!({"solver": r["solver"], "reaction_history_rel_rms_err": series_err(&rec.react, &r0),
            "tip_history_rel_rms_err": series_err(&rec.tip, &t0), "s2A_max_force_over_failure": r["s2A_max_force_over_failure"]}));
    }
    info.insert("A_elastic_vs_S0".into(), json!(elastic_cmp));

    // A: prop removal with fracture
    let solvers_list = [Solver::S0, Solver::S1 { sweeps: 16 }, Solver::S1P { sweeps: 16, policy: RhoPolicy::PerSpring(0.1) }, Solver::S1P { sweeps: 64, policy: RhoPolicy::PerSpring(0.1) }, Solver::S1P { sweeps: 256, policy: RhoPolicy::PerSpring(0.1) }, Solver::S1S2 { pre: 8, post: 8, policy: RhoPolicy::PerSpring(0.1) }];
    for &(scheme, h) in &[(Scheme::Bdf2, 1.0 / 480.0), (Scheme::Bdf2, 1.0 / 960.0), (Scheme::Bdf2, 1.0 / 1920.0)] {
        for &s in &solvers_list {
            if h < 1.0 / 500.0 && !matches!(s, Solver::S0 | Solver::S1P { sweeps: 16, .. }) {
                continue;
            }
            let (mut r, _) = if matches!(s, Solver::S0) {
                simulate_ref(&|| cantilever(0.3, None), scheme, h, 0.4, s, &mut csv, id, &|_| 60)
            } else {
                simulate(cantilever(0.3, None), scheme, h, 0.4, s, &mut csv, id)
            };
            push(&mut r, "A cantilever prop removal, dmin 0.3", &mut id, &mut runs);
        }
    }
    // A with implicit Euler: plain S0 (one-step breakage) vs event sub-stepping; ledger with D_num
    for &s in &[Solver::S0, Solver::S0Event, Solver::S1P { sweeps: 16, policy: RhoPolicy::PerSpring(0.1) }] {
        let (mut r, _) = if matches!(s, Solver::S0 | Solver::S0Event) {
            simulate_ref(&|| cantilever(0.3, None), Scheme::ImplicitEuler, 1.0 / 480.0, 0.4, s, &mut csv, id, &|_| 60)
        } else {
            simulate(cantilever(0.3, None), Scheme::ImplicitEuler, 1.0 / 480.0, 0.4, s, &mut csv, id)
        };
        push(&mut r, "A cantilever prop removal, dmin 0.3 (IE ledger)", &mut id, &mut runs);
    }
    // A, mesh / crack-band sensitivity (S0 only)
    for &(d, eta) in &[(0.2, None), (0.5, Some(0.95))] {
        let (mut r, _) = simulate_ref(&|| cantilever(d, eta), Scheme::Bdf2, 1.0 / 480.0, 0.4, Solver::S0, &mut csv, id, &|_| 60);
        push(&mut r, &format!("A cantilever prop removal, dmin {d}, crack band {:?}", eta), &mut id, &mut runs);
    }
    // B: lateral push
    for &s in &[Solver::S0, Solver::S1P { sweeps: 16, policy: RhoPolicy::PerSpring(0.1) }, Solver::S1P { sweeps: 64, policy: RhoPolicy::PerSpring(0.1) }, Solver::S1S2 { pre: 8, post: 8, policy: RhoPolicy::PerSpring(0.1) }] {
        let (mut r, _) = if matches!(s, Solver::S0) {
            simulate_ref(&column, Scheme::Bdf2, 1.0 / 480.0, 0.3, s, &mut csv, id, &|_| 60)
        } else {
            simulate(column(), Scheme::Bdf2, 1.0 / 480.0, 0.3, s, &mut csv, id)
        };
        push(&mut r, "B column lateral push 60 kN", &mut id, &mut runs);
    }
    // C1 / C2
    for pre in [false, true] {
        for &s in &[Solver::S0, Solver::S1 { sweeps: 16 }, Solver::S1P { sweeps: 16, policy: RhoPolicy::PerSpring(0.1) }, Solver::S1S2A { sweeps: 16, policy: RhoPolicy::PerSpring(0.1) }, Solver::S1S2 { pre: 8, post: 8, policy: RhoPolicy::PerSpring(0.1) }] {
            let (mut r, rec) = if matches!(s, Solver::S0) {
                simulate_ref(&|| block(pre), Scheme::Bdf2, 1.0 / 480.0, 0.25, s, &mut csv, id, &|_| 60)
            } else {
                simulate(block(pre), Scheme::Bdf2, 1.0 / 480.0, 0.25, s, &mut csv, id)
            };
            r["max_force_over_failure_any_step"] = json!(rec.max_force_over_fail.iter().cloned().fold(0.0, f64::max));
            push(&mut r, if pre { "C2 pre-loaded block released" } else { "C1 unloaded block free fall" }, &mut id, &mut runs);
        }
    }
    let res = json!({
        "meta": meta(EXP, "all", Some(7), "per step: prediction q̂ + (h²/α)M⁻¹f as initial guess; S0 Newton | S0+events (IE, halve while Δd>0.2, ≥h/256) | S1 (dynamic rho = m_eff/h²) n sweeps (sweep+dual) | S1+S2(A): forces reported only | S1+S2(B): pre sweeps, S2 with d'Alembert loads, inject λ, post sweeps; commit history at step end",
            json!({"material": Material::concrete(0.3), "gravity": G, "s2_broken_threshold": "1-d < 1e-3", "s2_sanity": "|u| <= 10 |motion| + 1 cm"})),
        "info": Value::Object(info),
        "runs": runs,
    });
    write_json(EXP, "summary", &res);
}
