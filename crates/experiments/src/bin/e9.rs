//! E9 — impact: rigid disk projectile on a supported concrete block. Damage outcome vs
//! time step and contact stiffness; resolved reference; event-triggered local time
//! refinement during contact (a deterministic, state-triggered substep policy).
//!
//! Contact: per step, unilateral frictionless point facets between the projectile and
//! each nearby cell at the current closest points (anchors fixed for the step: manifold
//! once per step). Integrator: implicit Euler throughout (isolates the step-size effect;
//! E8's damping caveat applies).

use experiments::{meta, rect_tess, write_json, Csv};
use geometry::polygon::closest_on_segment;
use geometry::V2;
use model::{BodyKind, BoundaryAttach, LawKind, LawOptions, Material, Model, PointLaw};
use serde_json::json;
use solvers::integrator::{Dyn, Scheme};
use solvers::s0::{newton, NewtonOpts};

const EXP: &str = "E9";
const R: f64 = 0.05;
const CONTACT_TAG: u32 = 99;

fn setup(v0: f64) -> (Model, usize) {
    let t = rect_tess(V2::new(0.0, 0.0), V2::new(1.0, 0.5), 0.05, 9);
    let mut m = Model::from_tessellation(&t, Material::concrete(0.2), LawOptions::damage(), &[]);
    let g = m.ground();
    m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: None, body: g, shear: true, tag: 1 });
    for f in m.facets.iter_mut().filter(|f| f.tag == 1) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Elastic;
        }
    }
    let mp = 7850.0 * std::f64::consts::PI * R * R * 0.2;
    let proj = m.add_plate(V2::new(0.5, 0.5 + R + 0.002), mp, 0.5 * mp * R * R, [false, false, true]);
    m.add_gravity(9.81);
    m.loads[proj][1] -= mp * 9.81;
    let _ = v0;
    (m, proj)
}

/// Replace contact facets by fresh ones at the current closest points.
fn update_contacts(m: &mut Model, q: &[[f64; 3]], proj: usize, margin: f64, kc: f64) -> usize {
    m.facets.retain(|f| f.tag != CONTACT_TAG);
    let c = V2::new(q[proj][0], q[proj][1]);
    let mut n = 0;
    for i in 0..m.bodies.len() {
        if m.bodies[i].kind != BodyKind::Cell {
            continue;
        }
        let xi = V2::new(q[i][0], q[i][1]);
        let th = q[i][2];
        let poly: Vec<V2> = m.bodies[i].poly.iter().map(|&p| xi + (p - m.bodies[i].x0).rot(th)).collect();
        if (xi - c).norm() > R + margin + 0.2 {
            continue;
        }
        let k = poly.len();
        let (mut dbest, mut cp) = (f64::INFINITY, V2::ZERO);
        for e in 0..k {
            let (d, p) = closest_on_segment(c, poly[e], poly[(e + 1) % k]);
            if d < dbest {
                dbest = d;
                cp = p;
            }
        }
        if dbest - R > margin || dbest < 1e-12 {
            continue;
        }
        let nrm = (c - cp) / dbest; // from cell surface towards projectile centre
        let sb = c - nrm * R;
        // reference-frame arms
        let arm_a = (cp - xi).rot(-th);
        let arm_b = sb - c; // projectile does not rotate
        let law = PointLaw { kind: LawKind::Damage, kn: 0.5 * kc, kt: 0.0, gamma: 0.0, delta0: 1e-12, wc: 2e-12, d_max: 1.0 };
        let pt = model::structure::FacetPoint { arm_a, arm_b, s: 0.0, law, kappa_c: 0.0, rest: V2::ZERO };
        // normal from cell a into the projectile b; the projectile frame (w = 1)
        m.facets.push(model::structure::Facet {
            a: i,
            b: proj,
            pc: cp,
            n0: nrm,
            t0: nrm.perp(),
            len: 1.0,
            h: 1.0,
            pts: [pt, pt],
            active: true,
            tag: CONTACT_TAG,
            ends: [cp, cp],
            frame_w: 1.0,
        });
        n += 1;
    }
    n
}

fn contact_force(m: &Model, q: &[[f64; 3]]) -> f64 {
    let mut f = 0.0;
    for fa in m.facets.iter().filter(|f| f.tag == CONTACT_TAG) {
        for p in 0..2 {
            let (_, e) = m.eval_point(q, fa, p);
            f += (-e.fnorm).max(0.0);
        }
    }
    f
}

fn material_dissipation(m: &Model) -> (f64, usize) {
    let mut d = 0.0;
    let mut broken = 0;
    for f in m.facets.iter().filter(|f| f.tag == 0 && f.active) {
        for p in &f.pts {
            d += p.law.dissipation(p.kappa_c);
        }
        if f.pts.iter().any(|p| p.law.damage(p.kappa_c) > 0.99) {
            broken += 1;
        }
    }
    (d, broken)
}

/// Run to t_end with outer step h; if `refine` is Some(hs), steps that may touch are
/// taken in substeps of hs (event-triggered by the predicted gap).
fn run(v0: f64, kc: f64, h: f64, refine: Option<f64>, t_end: f64, csv: &mut Csv, id: f64) -> serde_json::Value {
    run_opt(v0, kc, h, refine, t_end, csv, id, 60)
}

/// One run. A (sub)step whose Newton solve fails is not committed; the run stops and
/// is reported invalid. Coarse runs stay invalid (h is the studied variable); the
/// reference is restarted at h/2 by the caller.
fn run_opt(v0: f64, kc: f64, h: f64, refine: Option<f64>, t_end: f64, csv: &mut Csv, id: f64, newton_max_iter: usize) -> serde_json::Value {
    let (mut m, proj) = setup(v0);
    let fscale = m.mat.sigma_t * 0.05 * m.mat.thickness;
    let mut dynst = Dyn::new(Scheme::ImplicitEuler, h, &m);
    dynst.v[proj] = [0.0, -v0, 0.0];
    let mut q = m.q.clone();
    let mut t = 0.0;
    let mut peak_f: f64 = 0.0;
    let mut impulse = 0.0;
    let (mut t_first, mut t_last) = (None, None);
    let mut steps = 0usize;
    let mut substeps = 0usize;
    let ke0 = 0.5 * m.bodies[proj].mass * v0 * v0;
    while t < t_end - 1e-15 {
        // predicted gap: is contact possible within the outer step?
        let gap_pred = {
            let c = V2::new(q[proj][0], q[proj][1]);
            let mut gmin = f64::INFINITY;
            for i in 0..m.n_cells() {
                let xi = V2::new(q[i][0], q[i][1]);
                if (xi - c).norm() < 0.3 {
                    let th = q[i][2];
                    let poly: Vec<V2> = m.bodies[i].poly.iter().map(|&p| xi + (p - m.bodies[i].x0).rot(th)).collect();
                    for e in 0..poly.len() {
                        gmin = gmin.min(closest_on_segment(c, poly[e], poly[(e + 1) % poly.len()]).0 - R);
                    }
                }
            }
            gmin
        };
        let vrel = dynst.v[proj][1].abs() + dynst.v[proj][0].abs();
        let near = gap_pred < vrel * h + 0.005;
        let hs = match refine {
            Some(hr) if near => hr,
            _ => h,
        };
        let n_sub = ((h / hs).round() as usize).max(1);
        for _ in 0..n_sub {
            dynst.h = h / n_sub as f64;
            let margin = vrel * dynst.h + 0.005;
            update_contacts(&mut m, &q, proj, margin, kc);
            let q_old = q.clone();
            let ine = dynst.inertia(&q_old);
            for i in 0..q.len() {
                for c in 0..3 {
                    if !m.bodies[i].fixed[c] {
                        let mm = if c < 2 { m.bodies[i].mass } else { m.bodies[i].inertia };
                        q[i][c] = ine.qhat[i][c] + m.loads[i][c] / (mm * ine.alpha_h2);
                    }
                }
            }
            let rep = newton(&m, &mut q, 0.0, Some(&ine), NewtonOpts { tol: 1e-9 * fscale, max_iter: newton_max_iter });
            if !rep.converged {
                // not a reference step: nothing committed, the run is reported invalid
                return json!({"v0": v0, "kc": kc, "h": h, "refine_substep": refine, "valid": false,
                    "invalid": {"t": t, "outer_step": steps, "substeps": substeps, "reason": "Newton not converged", "residual": rep.res}});
            }
            let fc = contact_force(&m, &q);
            m.commit(&q);
            dynst.finish(&q_old, &q);
            t += dynst.h;
            substeps += 1;
            peak_f = peak_f.max(fc);
            impulse += fc * dynst.h;
            if fc > 0.0 {
                if t_first.is_none() {
                    t_first = Some(t);
                }
                t_last = Some(t);
            }
            let (dmat, broken) = material_dissipation(&m);
            csv.row(&[id, t, fc, q[proj][1], dynst.v[proj][1], dmat, broken as f64]);
        }
        steps += 1;
    }
    let (dmat, broken) = material_dissipation(&m);
    let vend = dynst.v[proj][1];
    json!({"v0": v0, "kc": kc, "h": h, "refine_substep": refine, "valid": true, "outer_steps": steps, "substeps": substeps,
        "t_c_estimate_s": std::f64::consts::PI * (m.bodies[proj].mass / kc).sqrt(),
        "D_material_J": dmat, "broken_facets": broken, "KE0_J": ke0, "D_over_KE0": dmat / ke0,
        "peak_contact_force_N": peak_f, "impulse_Ns": impulse, "contact_first_s": t_first, "contact_last_s": t_last,
        "projectile_v_end": vend, "restitution": -vend / v0})
}

fn main() {
    if std::env::var("SB_E9_CHECK").is_ok() {
        // forced Newton failure (max_iter = 1): the run must stop invalid at its first step
        const CX: &str = "phase0b/two-level/fix-checks";
        let mut csv = Csv::new(CX, "e9_series", &["run", "t", "contact_force", "proj_y", "proj_vy", "D_material", "broken"]);
        let forced = run_opt(10.0, 1e8, 1.0 / 960.0, None, 0.004, &mut csv, 0.0, 1);
        let normal = run_opt(10.0, 1e8, 1.0 / 960.0, None, 0.004, &mut csv, 1.0, 60);
        let pass = forced["valid"] == json!(false) && forced["invalid"]["outer_step"] == json!(0) && normal["valid"] == json!(true);
        eprintln!("E9 forced failure: {} / normal: {} pass={pass}", forced["valid"], normal["valid"]);
        write_json(CX, "e9_checks", &json!({"meta": meta(CX, "e9 checks", Some(9), "forced Newton failure (max_iter=1) vs normal run, 4 ms at 1/960", json!({})),
            "forced": forced, "normal": normal, "pass": pass}));
        return;
    }
    let mut csv = Csv::new(EXP, "series", &["run", "t", "contact_force", "proj_y", "proj_vy", "D_material", "broken"]);
    let mut runs = Vec::new();
    let mut id = 0.0;
    let t_end = 0.012;
    for &v0 in &[10.0, 20.0] {
        for &kc in &[1e8, 1e9] {
            let mp = 7850.0 * std::f64::consts::PI * R * R * 0.2;
            let tc = std::f64::consts::PI * (mp / kc).sqrt();
            let href = (tc / 100.0).min(5e-6);
            let mut cases: Vec<(f64, Option<f64>, &str)> = vec![(href, None, "reference")];
            for &hi in &[240.0, 480.0, 960.0, 1920.0] {
                cases.push((1.0 / hi, None, "coarse"));
            }
            cases.push((1.0 / 480.0, Some(tc / 20.0), "event-refined 1/480 (t_c/20 during contact)"));
            cases.push((1.0 / 480.0, Some(tc / 5.0), "event-refined 1/480 (t_c/5 during contact)"));
            for (h, refine, label) in cases {
                let mut r = run(v0, kc, h, refine, t_end, &mut csv, id);
                if label == "reference" {
                    let mut attempts = vec![json!({"h": h, "valid": r["valid"].clone(), "invalid": r["invalid"].clone()})];
                    let mut hk = h;
                    while r["valid"] != json!(true) && attempts.len() < 4 {
                        hk *= 0.5;
                        r = run(v0, kc, hk, refine, t_end, &mut csv, id);
                        attempts.push(json!({"h": hk, "valid": r["valid"].clone(), "invalid": r["invalid"].clone()}));
                    }
                    r["attempts"] = json!(attempts);
                }
                r["label"] = json!(label);
                r["run"] = json!(id);
                eprintln!("{r}");
                runs.push(r);
                id += 1.0;
            }
        }
    }
    let res = json!({
        "meta": meta(EXP, "all", Some(9), "implicit Euler, S0 Newton per (sub)step; contact facets rebuilt per step at closest points (unilateral, frictionless)",
            json!({"block": "1.0 x 0.5 m, b=0.2, dmin 0.05, clamped base", "projectile": {"radius": R, "density": 7850.0}, "t_end": t_end})),
        "runs": runs,
    });
    write_json(EXP, "summary", &res);
}
