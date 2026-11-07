//! Q5 range survey: observed magnitudes on representative scenes (lab-scale static,
//! game-scale dynamic with fracture), for fixed-point / Flock IR design. Observed,
//! not proven, bounds; the scene contracts are stated in the output.

use experiments::{meta, rect_tess, write_json};
use geometry::V2;
use model::range;
use model::{BoundaryAttach, LawOptions, Material, Model};
use serde_json::json;
use solvers::integrator::{Dyn, Scheme};
use solvers::s0::{newton, NewtonOpts};
use solvers::s1::{Alm, RhoPolicy};

fn record_state(tag: &str, m: &Model, q: &[[f64; 3]], q_prev: Option<&[[f64; 3]]>) {
    for (i, b) in m.bodies.iter().enumerate() {
        if b.kind != model::BodyKind::Cell {
            continue;
        }
        range::rec(&format!("{tag}.position_abs_m"), q[i][0].abs().max(q[i][1].abs()));
        range::rec(&format!("{tag}.rotation_rad"), q[i][2]);
        if let Some(p) = q_prev {
            range::rec(&format!("{tag}.step_delta_translation_m"), (q[i][0] - p[i][0]).hypot(q[i][1] - p[i][1]));
            range::rec(&format!("{tag}.step_delta_rotation_rad"), q[i][2] - p[i][2]);
        }
    }
    for f in m.facets.iter().filter(|f| f.active && f.tag == 0) {
        for p in 0..2 {
            let (j, e) = m.eval_point(q, f, p);
            range::rec(&format!("{tag}.jump_normal_m"), j.dn);
            range::rec(&format!("{tag}.jump_tangential_m"), j.dt);
            range::rec(&format!("{tag}.point_force_N"), e.fnorm.abs().max(e.ftan.abs()));
            range::rec(&format!("{tag}.damage"), e.d);
            let l = &f.pts[p].law;
            range::rec(&format!("{tag}.law_delta0_m"), l.delta0);
            range::rec(&format!("{tag}.law_wc_m"), l.wc);
            range::rec(&format!("{tag}.law_kn_N_per_m"), l.kn);
            range::rec(&format!("{tag}.strain_dn_over_h"), j.dn / f.h);
            range::rec(&format!("{tag}.arm_m"), f.pts[p].arm_a.norm());
        }
    }
}

fn main() {
    range::enable();
    // Lab scale: E2-like block, static, S0 and S1 (per-spring 0.1) 200 iterations
    {
        let t = rect_tess(V2::new(0.0, 0.0), V2::new(0.2, 0.1), 0.005, 1);
        let mut m = Model::from_tessellation(&t, Material::concrete(0.05), LawOptions::damage(), &[]);
        let g = m.ground();
        m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: None, body: g, shear: true, tag: 1 });
        experiments::edge_traction(&mut m, &t, 2, Some((0.0, 0.08)), V2::new(0.5e6, -2.0e6), false);
        let mut q = m.q.clone();
        newton(&m, &mut q, 0.0, None, NewtonOpts { tol: 1e-6, max_iter: 30 });
        record_state("lab_static_S0", &m, &q, Some(&m.q));
        let mut alm = Alm::new(&m, RhoPolicy::PerSpring(0.1), None);
        let mut q1 = m.q.clone();
        for _ in 0..200 {
            alm.iterate(&m, &mut q1, 0.0, None);
        }
        record_state("lab_static_S1", &m, &q1, None);
    }
    // Game scale: E7 scenario A with fracture, BDF2 1/480, S0 and S1 per-spring 64
    for (tag, sweeps) in [("game_dyn_S0", 0usize), ("game_dyn_S1", 64)] {
        let t = rect_tess(V2::new(0.0, 0.0), V2::new(8.0, 1.2), 0.3, 7);
        let mut m = Model::from_tessellation(&t, Material::concrete(0.3), LawOptions::damage(), &[]);
        let g = m.ground();
        m.attach_boundary(&t, &BoundaryAttach { edge: 3, range: None, body: g, shear: true, tag: 1 });
        m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: Some((7.4, 8.0)), body: g, shear: true, tag: 2 });
        m.add_gravity(9.81);
        let mut q = m.q.clone();
        newton(&m, &mut q, 0.0, None, NewtonOpts { tol: 1e-6, max_iter: 50 });
        m.commit(&q);
        m.q = q.clone();
        m.facets.iter_mut().filter(|f| f.tag == 2).for_each(|f| f.active = false);
        let h = 1.0 / 480.0;
        let mut d = Dyn::new(Scheme::Bdf2, h, &m);
        let mut alm = Alm::new(&m, RhoPolicy::PerSpring(0.1), Some(&d.inertia(&q)));
        for _step in 0..48 {
            let q_old = q.clone();
            let ine = d.inertia(&q_old);
            for i in 0..q.len() {
                for c in 0..3 {
                    if !m.bodies[i].fixed[c] {
                        let mm = if c < 2 { m.bodies[i].mass } else { m.bodies[i].inertia };
                        q[i][c] = ine.qhat[i][c] + m.loads[i][c] / (mm * ine.alpha_h2);
                    }
                }
            }
            if sweeps == 0 {
                newton(&m, &mut q, 0.0, Some(&ine), NewtonOpts { tol: 1e-6, max_iter: 60 });
            } else {
                alm.set_rho(&m, Some(&ine));
                for _ in 0..sweeps {
                    alm.iterate(&m, &mut q, 0.0, Some(&ine));
                }
            }
            m.commit(&q);
            d.finish(&q_old, &q);
            record_state(tag, &m, &q, Some(&q_old));
            for (i, b) in m.bodies.iter().enumerate() {
                if b.kind == model::BodyKind::Cell {
                    range::rec(&format!("{tag}.velocity_m_per_s"), d.v[i][0].hypot(d.v[i][1]));
                }
            }
        }
    }
    let r = experiments::ranges_json();
    let res = json!({
        "meta": meta("Q5", "ranges", Some(1), "observed magnitudes; S1 channels (s, 1-s, k_eff, lambda) recorded inside the solver",
            json!({"lab": "0.2x0.1 m block, cells 5 mm, top patch load (0.5,-2) MPa, S0 + S1 perspring 0.1 x200",
                   "game": "E7 scenario A (8x1.2 m, cells 0.3 m), BDF2 1/480, 0.1 s after prop removal, S0 and S1 perspring 0.1 x64"})),
        "ranges": r,
    });
    write_json("Q5", "ranges", &res);
    println!("{}", serde_json::to_string_pretty(&res["ranges"]).unwrap());
}
