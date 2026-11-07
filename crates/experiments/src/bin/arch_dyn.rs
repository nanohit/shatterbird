//! Phase 0b / 3 §4 — equal-error dynamics (design: docs/phase0b/architecture-decision.md §2.5).
//!
//! E  = leapfrog (kick–drift, v at half steps, history committed every substep);
//! I  = BDF2 / Newmark (control) / generalized-α ρ∞ = 0.5 (force averaging), each step a
//!      minimisation of Φ(q) = α/(2h²)‖q − q̂‖²_M + Π(q; κ_c) solved either by direct Newton
//!      (S0) or by the nonlinear bridge (secant SPD direction + two-level PCG under the RD
//!      policy + Armijo line search). History is committed only after an accepted (sub)step.
//! Problems: A1 one-element energy cycle, A2 R1-elastic vs the modal reference, R1 prop
//! removal, R2 notched beam under prescribed slow displacement, R3 impact.
//! Usage: `arch_dyn a1 | a2 | r1 [seeds] | r2 [cases] | r3 [seeds] | bridge <prob> [seeds]`.

use experiments::{meta, support_reaction, write_json};
use geometry::polygon::closest_on_segment;
use geometry::rng::Rng;
use geometry::rsa::rsa_rect_with;
use geometry::voronoi2d::{rect, voronoi_convex, Tessellation};
use geometry::V2;
use model::law::{LawKind, PointLaw};
use model::structure::{Facet, FacetPoint};
use model::{BodyKind, BoundaryAttach, LawOptions, Material, Model};
use serde_json::{json, Value};
use solvers::assembly::{assemble, axpy_q, energy, DofMap, Inertia, Tangent};
use solvers::s0::{body_reaction, newton, NewtonOpts};
use solvers::twolevel::{aggregate, pcg, rigid_p, smooth_p, AggInput, Coarse, Csr, Precond, Smoother, System, Work};
use std::time::Instant;

const EXP: &str = "phase0b/architecture-selection/dynamics";
const G: f64 = 9.81;
const CONTACT_TAG: u32 = 99;
const R_PROJ: f64 = 0.05;
// R2 (E6 geometry)
const LEN: f64 = 0.5;
const DEPTH: f64 = 0.1;
const SPAN: f64 = 0.4;
const NOTCH: f64 = 0.05;
const BEAM_B: f64 = 0.1;
const PAD: f64 = 0.01;
const CENTRAL: f64 = 0.08;

// ======================================================================= problems

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    A1,
    R1,
    R2,
    R3,
}

/// R2 plate programme: C¹ ramp to d_final with acceleration/deceleration phases t_a.
#[derive(Clone, Copy, Debug)]
struct Ramp {
    v: f64,
    ta: f64,
    d_final: f64,
}

impl Ramp {
    fn t_total(&self) -> f64 {
        self.d_final / self.v + self.ta
    }
    fn d(&self, t: f64) -> f64 {
        let tt = self.t_total();
        if t <= 0.0 {
            0.0
        } else if t < self.ta {
            self.v * t * t / (2.0 * self.ta)
        } else if t < tt - self.ta {
            self.v * (t - 0.5 * self.ta)
        } else if t < tt {
            self.d_final - self.v * (tt - t) * (tt - t) / (2.0 * self.ta)
        } else {
            self.d_final
        }
    }
}

#[derive(Clone)]
struct Prob {
    name: String,
    kind: Kind,
    m: Model,
    /// sample grid [s]; every method's step is an integer number of ticks (I) or a tick is
    /// an integer number of steps (E)
    grid: f64,
    end_ticks: u64,
    plate: Option<usize>,
    y0: f64,
    ramp: Option<Ramp>,
    proj: Option<usize>,
    kc: f64,
    v0: f64,
    /// R1: reaction tag; facets of `remove_tag` are deactivated at t = 0
    react_tag: u32,
    remove_tag: Option<u32>,
    /// criteria force scale and runtime force scale (bridge stop)
    f_scale: f64,
    f_rt: f64,
    e_scale: f64,
    f_min: f64,
    h_safe: f64,
    v_init: Vec<[f64; 3]>,
    /// damageable material points (facet, point)
    mat_pts: Vec<(usize, usize)>,
    /// R1′: the prop is replaced by its reaction (m.pattern) released as s(t) over t_r
    ramp_tr: Option<f64>,
    info: Value,
}

impl Prob {
    /// load factor of m.pattern at time t: ½(1 + cos(πt/t_r)) on [0, t_r], then 0
    fn lf(&self, t: f64) -> f64 {
        match self.ramp_tr {
            Some(tr) if t < tr => 0.5 * (1.0 + (std::f64::consts::PI * t / tr).cos()),
            _ => 0.0,
        }
    }
}

fn contract_tess(lo: V2, hi: V2, dmin: f64, seed: u64, keep: &dyn Fn(V2) -> bool) -> Tessellation {
    let seeds = rsa_rect_with(lo, hi, dmin, &mut Rng::new(seed), 20_000, keep, Vec::new());
    let t = voronoi_convex(&seeds, &rect(lo, hi));
    assert!(t.max_mismatch < 1e-9 * (hi - lo).norm());
    t
}

fn mat_points(m: &Model, include_elastic: bool) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for (fi, f) in m.facets.iter().enumerate() {
        if f.tag == 0 && f.active {
            for p in 0..2 {
                if include_elastic || f.pts[p].law.kind == LawKind::Damage {
                    v.push((fi, p));
                }
            }
        }
    }
    v
}

fn f_min(m: &Model, pts: &[(usize, usize)]) -> f64 {
    pts.iter().map(|&(f, _)| 0.5 * m.facets[f].len * m.mat.thickness * m.mat.sigma_t).fold(f64::INFINITY, f64::min)
}

fn static_solve(m: &Model, q: &mut Vec<[f64; 3]>) {
    let fs = m.mat.sigma_t * 0.01 * m.mat.thickness;
    let r = newton(m, q, 0.0, None, NewtonOpts { tol: 1e-9 * fs, max_iter: 60 });
    assert!(r.converged, "static solve failed: {r:?}");
}

fn potential(m: &Model, q: &[[f64; 3]]) -> f64 {
    m.elastic_energy(q) + m.load_potential(q, 0.0)
}

/// R1: cantilever 8 × 1.2 × 0.3 m, contract seeding, clamped left edge + exact prop on
/// the bottom x ∈ [7.4, 8]; fixture facets elastic; gravity; propped equilibrium.
fn r1(seed: u64, damage: bool, t_r: Option<f64>) -> Prob {
    let dmin = 0.3;
    let (lo, hi) = (V2::new(0.0, 0.0), V2::new(8.0, 1.2));
    let keep = |p: V2| !(p.x < 0.5 * dmin) && !(p.y < 0.5 * dmin && p.x >= 7.4 - dmin);
    let t = contract_tess(lo, hi, dmin, seed, &keep);
    let law = if damage { LawOptions::damage() } else { LawOptions::elastic() };
    let mut m = Model::from_tessellation(&t, Material::concrete(0.3), law, &[]);
    let g = m.ground();
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 3, range: None, body: g, shear: true, tag: 1 });
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((7.4, 8.0)), body: g, shear: true, tag: 2 });
    for f in m.facets.iter_mut().filter(|f| f.tag != 0) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Elastic;
        }
    }
    m.add_gravity(G);
    let mut q = m.q.clone();
    static_solve(&m, &mut q);
    m.commit(&q);
    assert!(m.max_damage() == 0.0, "propped state damaged");
    m.q = q.clone();
    let prop = support_reaction(&m, &q, 2).y;
    // energy released by the removal, elastic model
    let mut el = m.clone();
    for f in el.facets.iter_mut() {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Elastic;
        }
    }
    for f in el.facets.iter_mut().filter(|f| f.tag == 2) {
        f.active = false;
    }
    let mut qs = q.clone();
    static_solve(&el, &mut qs);
    let e_rel = potential(&el, &q) - potential(&el, &qs);
    let c = m.mat.wave_speed();
    let pts = mat_points(&m, !damage);
    let fm = f_min(&m, &pts);
    let n = m.bodies.len();
    if t_r.is_some() {
        // generalised forces of the prop facets on their cells at q0
        for f in m.facets.iter().filter(|f| f.tag == 2) {
            for pt in 0..2 {
                let (j, e) = m.eval_point(&q, f, pt);
                for r in 0..3 {
                    let v = -(j.jna[r] * e.fnorm + j.jta[r] * e.ftan);
                    m.pattern[f.a][r] += v;
                }
            }
        }
    }
    let mut pr = Prob {
        name: format!("R1{} {} s{seed}", if t_r.is_some() { "'" } else { "" }, if damage { "SB-IL1" } else { "elastic" }),
        kind: Kind::R1,
        grid: 1.0 / 7680.0,
        end_ticks: (0.2 * 7680.0) as u64,
        plate: None,
        y0: 0.0,
        ramp: None,
        proj: None,
        kc: 0.0,
        v0: 0.0,
        react_tag: 1,
        remove_tag: Some(2),
        f_scale: prop.abs(),
        f_rt: prop.abs(),
        e_scale: e_rel,
        f_min: fm,
        h_safe: 0.38 * dmin / c,
        v_init: vec![[0.0; 3]; n],
        mat_pts: pts,
        ramp_tr: None,
        info: json!({"cells": m.n_cells(), "facets": m.facets.len(), "dmin": dmin, "prop_reaction_N": prop, "E_released_J": e_rel, "prop_release_s": t_r,
            "static_tip_drop_m": qs.iter().zip(&q).map(|(a, b)| (a[1] - b[1]).abs()).fold(0.0, f64::max)}),
        m,
    };
    pr.ramp_tr = t_r;
    pr
}

/// E6 tessellation with contract rule 2 near the supports and the pad.
fn r2_tess(dmin: f64, seed: u64) -> Tessellation {
    let xc = LEN / 2.0;
    let x1 = (LEN - SPAN) / 2.0;
    let x2 = x1 + SPAN;
    let keep = move |p: V2| {
        let near = |x0: f64| (p.x - x0).abs() <= PAD + dmin;
        !((p.y < 0.5 * dmin && (near(x1) || near(x2))) || (p.y > DEPTH - 0.5 * dmin && near(xc)))
    };
    let mut rng = Rng::new(seed);
    let band = 2.0 * dmin;
    let ytop = NOTCH + dmin;
    let all = rsa_rect_with(V2::new(0.0, 0.0), V2::new(LEN, DEPTH), dmin, &mut rng, 20_000, &keep, Vec::new());
    let mut seeds: Vec<V2> = all.into_iter().filter(|p| !((p.x - xc).abs() < band && p.y < ytop)).collect();
    let mut fails = 0;
    let mut half = Vec::new();
    while fails < 5000 {
        let p = V2::new(rng.range(xc - band, xc - 0.5 * dmin), rng.range(0.0, ytop));
        let mirror = V2::new(2.0 * xc - p.x, p.y);
        let ok = keep(p) && seeds.iter().chain(half.iter()).all(|s: &V2| (*s - p).norm() >= dmin && (*s - mirror).norm() >= dmin);
        if ok {
            half.push(p);
            fails = 0;
        } else {
            fails += 1;
        }
    }
    for p in &half {
        seeds.push(*p);
        seeds.push(V2::new(2.0 * xc - p.x, p.y));
    }
    let t = voronoi_convex(&seeds, &rect(V2::new(0.0, 0.0), V2::new(LEN, DEPTH)));
    assert!(t.max_mismatch < 1e-12);
    t
}

/// R2: E6 notched beam, plate displacement prescribed by a slow C¹ ramp, then a hold.
fn r2(dmin: f64, seed: u64, hold: f64) -> Prob {
    let t = r2_tess(dmin, seed);
    let xc = LEN / 2.0;
    let mut m = Model::from_tessellation(&t, Material::concrete(BEAM_B), LawOptions::damage(), &[]);
    for f in m.facets.iter_mut() {
        let on_plane = (f.ends[0].x - xc).abs() < 1e-12 && (f.ends[1].x - xc).abs() < 1e-12;
        if on_plane && f.pc.y < NOTCH {
            f.active = false;
        }
    }
    let g = m.ground();
    let x1 = (LEN - SPAN) / 2.0;
    let x2 = x1 + SPAN;
    m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: Some((x1 - PAD, x1 + PAD)), body: g, shear: false, tag: 1 });
    m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: Some((x2 - PAD, x2 + PAD)), body: g, shear: false, tag: 2 });
    let plate = m.add_plate(V2::new(xc, DEPTH), 0.0, 0.0, [true, true, true]);
    m.attach_boundary(&t, &BoundaryAttach { edge: 2, range: Some((xc - PAD, xc + PAD)), body: plate, shear: false, tag: 3 });
    for f in m.facets.iter_mut().filter(|f| f.tag != 0) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Damage;
            p.law.delta0 *= 1e-4;
            p.law.wc *= 1e-4;
        }
    }
    for f in m.facets.iter_mut().filter(|f| f.tag == 0) {
        if (f.pc.x - xc).abs() >= CENTRAL {
            for p in f.pts.iter_mut() {
                p.law.kind = LawKind::Elastic;
            }
        }
    }
    let ramp = Ramp { v: 1e-3, ta: 2e-3, d_final: 0.25e-3 };
    let c = m.mat.wave_speed();
    let pts = mat_points(&m, false);
    let fm = f_min(&m, &pts);
    let n = m.bodies.len();
    let y0 = m.bodies[plate].x0.y;
    let t_end = ramp.t_total() + hold;
    Prob {
        name: format!("R2 d{} s{seed}", dmin * 1e3),
        kind: Kind::R2,
        grid: 1.0 / 7680.0,
        end_ticks: (t_end * 7680.0).round() as u64,
        plate: Some(plate),
        y0,
        ramp: Some(ramp),
        proj: None,
        kc: 0.0,
        v0: 0.0,
        react_tag: 0,
        remove_tag: None,
        f_scale: f64::NAN,
        f_rt: m.mat.sigma_t * BEAM_B * (DEPTH - NOTCH),
        e_scale: f64::NAN,
        f_min: fm,
        h_safe: 0.38 * dmin / c,
        v_init: vec![[0.0; 3]; n],
        mat_pts: pts,
        ramp_tr: None,
        info: json!({"cells": m.n_cells(), "facets": m.facets.len(), "dmin": dmin, "ramp": {"v": ramp.v, "ta": ramp.ta, "d_final": ramp.d_final, "t_total": ramp.t_total()}, "hold_s": hold}),
        m,
    }
}

/// R3: E9 block on the contract mesh, projectile 12.3 kg at v0, contact k_c.
fn r3(seed: u64, v0: f64, kc: f64, t_end: f64) -> Prob {
    let dmin = 0.05;
    let (lo, hi) = (V2::new(0.0, 0.0), V2::new(1.0, 0.5));
    let keep = |p: V2| p.y >= 0.5 * dmin;
    let t = contract_tess(lo, hi, dmin, seed, &keep);
    let mut m = Model::from_tessellation(&t, Material::concrete(0.2), LawOptions::damage(), &[]);
    let g = m.ground();
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: None, body: g, shear: true, tag: 1 });
    for f in m.facets.iter_mut().filter(|f| f.tag == 1) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Elastic;
        }
    }
    let mp = 7850.0 * std::f64::consts::PI * R_PROJ * R_PROJ * 0.2;
    m.add_gravity(G);
    // static gravity equilibrium of the block with the projectile held
    let proj = m.add_plate(V2::new(0.5, 0.5 + R_PROJ + 0.002), mp, 0.5 * mp * R_PROJ * R_PROJ, [true, true, true]);
    let mut q = m.q.clone();
    static_solve(&m, &mut q);
    m.commit(&q);
    m.q = q;
    m.bodies[proj].fixed = [false, false, false];
    m.loads[proj][1] -= mp * G;
    let c = m.mat.wave_speed();
    let pts = mat_points(&m, false);
    let fm = f_min(&m, &pts);
    let n = m.bodies.len();
    let mut v_init = vec![[0.0; 3]; n];
    v_init[proj] = [0.0, -v0, 0.0];
    let tc = std::f64::consts::PI * (mp / kc).sqrt();
    Prob {
        name: format!("R3 v{v0} kc{kc:.0e} s{seed}"),
        kind: Kind::R3,
        grid: 1.0 / 192_000.0,
        end_ticks: (t_end * 192_000.0).round() as u64,
        plate: None,
        y0: 0.0,
        ramp: None,
        proj: Some(proj),
        kc,
        v0,
        react_tag: 0,
        remove_tag: None,
        f_scale: f64::NAN,
        f_rt: mp * v0 / tc,
        e_scale: 0.5 * mp * v0 * v0,
        f_min: fm,
        h_safe: 0.38 * dmin / c,
        v_init,
        mat_pts: pts,
        ramp_tr: None,
        info: json!({"cells": m.n_cells(), "facets": m.facets.len(), "dmin": dmin, "projectile_kg": mp, "t_c_s": tc, "KE0_J": 0.5 * mp * v0 * v0}),
        m,
    }
}

/// A1: one 0.1 × 0.1 m cell on one damageable facet to the ground, pulled upward.
fn a1(ke_over_gca: f64) -> Prob {
    let (lo, hi) = (V2::new(0.0, 0.0), V2::new(0.1, 0.1));
    let t = voronoi_convex(&[V2::new(0.05, 0.05)], &rect(lo, hi));
    let mut m = Model::from_tessellation(&t, Material::concrete(0.3), LawOptions::damage(), &[]);
    let g = m.ground();
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: None, body: g, shear: true, tag: 0 });
    let pts = mat_points(&m, false);
    assert_eq!(pts.len(), 2);
    let gca: f64 = pts.iter().map(|&(f, p)| m.facets[f].pts[p].law.full_dissipation()).sum();
    let mass = m.bodies[0].mass;
    let ke0 = ke_over_gca * gca;
    let v0 = (2.0 * ke0 / mass).sqrt();
    let kn: f64 = pts.iter().map(|&(f, p)| m.facets[f].pts[p].law.kn).sum();
    let period = 2.0 * std::f64::consts::PI * (mass / kn).sqrt();
    let mut v_init = vec![[0.0; 3]; m.bodies.len()];
    v_init[0] = [0.0, v0, 0.0];
    let fm = f_min(&m, &pts);
    Prob {
        name: format!("A1 KE0={ke_over_gca}GcA"),
        kind: Kind::A1,
        grid: period / 640.0,
        end_ticks: 640 * 10,
        plate: None,
        y0: 0.0,
        ramp: None,
        proj: None,
        kc: 0.0,
        v0,
        react_tag: 0,
        remove_tag: None,
        f_scale: kn * pts.iter().map(|&(f, p)| m.facets[f].pts[p].law.delta0).sum::<f64>() * 0.5,
        f_rt: fm,
        e_scale: ke0,
        f_min: fm,
        h_safe: period / 10.0,
        v_init,
        mat_pts: pts,
        ramp_tr: None,
        info: json!({"mass_kg": mass, "kn_total": kn, "period_s": period, "GcA_J": gca, "KE0_J": ke0, "v0": v0}),
        m,
    }
}

/// Exact A1 outcome from energy conservation: (κ_max, D) for a monotone pull-out.
fn a1_exact(p: &Prob) -> (f64, f64) {
    let pts: Vec<PointLaw> = p.mat_pts.iter().map(|&(f, q)| p.m.facets[f].pts[q].law).collect();
    let l = pts[0];
    let ke0 = p.e_scale;
    let e_of = |k: f64| -> f64 { pts.iter().map(|l| 0.5 * (1.0 - l.damage(k)) * l.kn * k * k + l.dissipation(k)).sum() };
    if e_of(l.wc) <= ke0 {
        return (f64::INFINITY, pts.iter().map(|l| l.full_dissipation()).sum());
    }
    let (mut a, mut b) = (0.0, l.wc);
    for _ in 0..200 {
        let mid = 0.5 * (a + b);
        if e_of(mid) < ke0 {
            a = mid;
        } else {
            b = mid;
        }
    }
    let k = 0.5 * (a + b);
    (k, pts.iter().map(|l| l.dissipation(k)).sum())
}

// ======================================================================= contacts (R3)

fn update_contacts(m: &mut Model, q: &[[f64; 3]], proj: usize, margin: f64, kc: f64) -> usize {
    m.facets.retain(|f| f.tag != CONTACT_TAG);
    let c = V2::new(q[proj][0], q[proj][1]);
    let mut n = 0;
    for i in 0..m.bodies.len() {
        if m.bodies[i].kind != BodyKind::Cell {
            continue;
        }
        let xi = V2::new(q[i][0], q[i][1]);
        if (xi - c).norm() > R_PROJ + margin + 0.2 {
            continue;
        }
        let th = q[i][2];
        let poly: Vec<V2> = m.bodies[i].poly.iter().map(|&p| xi + (p - m.bodies[i].x0).rot(th)).collect();
        let k = poly.len();
        let (mut dbest, mut cp) = (f64::INFINITY, V2::ZERO);
        for e in 0..k {
            let (d, p) = closest_on_segment(c, poly[e], poly[(e + 1) % k]);
            if d < dbest {
                dbest = d;
                cp = p;
            }
        }
        if dbest - R_PROJ > margin || dbest < 1e-12 {
            continue;
        }
        let nrm = (c - cp) / dbest;
        let sb = c - nrm * R_PROJ;
        let arm_a = (cp - xi).rot(-th);
        let arm_b = (sb - c).rot(-q[proj][2]);
        let law = PointLaw { kind: LawKind::Damage, kn: 0.5 * kc, kt: 0.0, gamma: 0.0, delta0: 1e-12, wc: 2e-12, d_max: 1.0 };
        let pt = FacetPoint { arm_a, arm_b, s: 0.0, law, kappa_c: 0.0, rest: V2::ZERO };
        // normal from the cell into the projectile, measured in the projectile frame (w = 1)
        let th_b = q[proj][2];
        m.facets.push(Facet { a: i, b: proj, pc: cp, n0: nrm.rot(-th_b), t0: nrm.perp().rot(-th_b), len: 1.0, h: 1.0, pts: [pt, pt], active: true, tag: CONTACT_TAG, ends: [cp, cp], frame_w: 1.0 });
        n += 1;
    }
    n
}

fn contact_force(m: &Model, q: &[[f64; 3]]) -> f64 {
    let mut f = 0.0;
    for fa in m.facets.iter().filter(|f| f.tag == CONTACT_TAG && f.active) {
        for p in 0..2 {
            let (_, e) = m.eval_point(q, fa, p);
            f += (-e.fnorm).max(0.0);
        }
    }
    f
}

fn predicted_gap(m: &Model, q: &[[f64; 3]], proj: usize) -> f64 {
    let c = V2::new(q[proj][0], q[proj][1]);
    let mut gmin = f64::INFINITY;
    for i in 0..m.bodies.len() {
        if m.bodies[i].kind != BodyKind::Cell {
            continue;
        }
        let xi = V2::new(q[i][0], q[i][1]);
        if (xi - c).norm() < 0.3 {
            let th = q[i][2];
            let poly: Vec<V2> = m.bodies[i].poly.iter().map(|&p| xi + (p - m.bodies[i].x0).rot(th)).collect();
            for e in 0..poly.len() {
                gmin = gmin.min(closest_on_segment(c, poly[e], poly[(e + 1) % poly.len()]).0 - R_PROJ);
            }
        }
    }
    gmin
}

// ======================================================================= recording

#[derive(Default, Clone)]
struct Traj {
    ticks: Vec<u64>,
    force: Vec<f64>,
    chi: Vec<Vec<f32>>,
    dmat: Vec<f64>,
    enum_: Vec<f64>,
    onset: Option<f64>,
    first_broken: Option<f64>,
    peak: f64,
    t_peak: f64,
    impulse: f64,
    final_d: Vec<f32>,
    valid: bool,
    invalid: Option<Value>,
    steps: u64,
    wall: f64,
    work: Value,
    /// per (sub)step stats of the bridge (outer, pcg, force passes, refreshes)
    step_stats: Vec<[f64; 5]>,
    /// external work at the end (R2 plate, R1′ released prop force)
    work_ext: f64,
}

struct Ledger {
    e0: f64,
    w_ext: f64,
    d0: f64,
}

fn kinetic(m: &Model, v: &[[f64; 3]]) -> f64 {
    let mut t = 0.0;
    for (i, b) in m.bodies.iter().enumerate() {
        if b.fixed.iter().all(|&f| f) {
            continue;
        }
        t += 0.5 * b.mass * (v[i][0].powi(2) + v[i][1].powi(2)) + 0.5 * b.inertia * v[i][2].powi(2);
    }
    t
}

fn dmat(m: &Model) -> f64 {
    m.facets.iter().filter(|f| f.tag == 0).flat_map(|f| f.pts.iter()).map(|p| p.law.dissipation(p.kappa_c)).sum()
}

/// χ = 1 on damaged points (κ_c > δ0), else min(δ_eq/δ0, 1).
fn chi(m: &Model, q: &[[f64; 3]], pts: &[(usize, usize)]) -> Vec<f32> {
    pts.iter()
        .map(|&(fi, p)| {
            let f = &m.facets[fi];
            let law = &f.pts[p].law;
            if f.pts[p].kappa_c > law.delta0 {
                return 1.0;
            }
            let (j, _) = m.eval_point(q, f, p);
            let dn = j.dn.max(0.0);
            (((dn * dn + law.gamma * j.dt * j.dt).sqrt() / law.delta0).min(1.0)) as f32
        })
        .collect()
}

/// The problem's reported force at q (R1 clamp reaction y, R2 plate load, R3 contact force,
/// A1 facet normal force).
fn reported_force(p: &Prob, m: &Model, q: &[[f64; 3]]) -> f64 {
    match p.kind {
        Kind::R1 => support_reaction(m, q, p.react_tag).y,
        Kind::R2 => body_reaction(m, q, p.plate.unwrap())[1],
        Kind::R3 => contact_force(m, q),
        Kind::A1 => -body_reaction(m, q, 0)[1],
    }
}

/// Commit κ ← max(κ, δ_eq(q)) on all active facets; track onset / full damage of material
/// points. Returns the dissipation increment.
fn commit_track(m: &mut Model, q: &[[f64; 3]], mat: &[bool], t: f64, tr: &mut Traj) -> f64 {
    let mut dd = 0.0;
    for fi in 0..m.facets.len() {
        if !m.facets[fi].active {
            continue;
        }
        for p in 0..2 {
            let (_, e) = m.eval_point(q, &m.facets[fi], p);
            let pt = &mut m.facets[fi].pts[p];
            if e.kappa > pt.kappa_c {
                if fi < mat.len() && mat[fi] && pt.law.kind == LawKind::Damage {
                    if pt.kappa_c <= pt.law.delta0 && e.kappa > pt.law.delta0 && tr.onset.is_none() {
                        tr.onset = Some(t);
                    }
                    if pt.law.damage(e.kappa) >= 0.99 && pt.law.damage(pt.kappa_c) < 0.99 && tr.first_broken.is_none() {
                        tr.first_broken = Some(t);
                    }
                }
                dd += pt.law.dissipation(e.kappa) - pt.law.dissipation(pt.kappa_c);
                pt.kappa_c = e.kappa;
            }
        }
    }
    m.dissipated += dd;
    dd
}

fn record(p: &Prob, m: &Model, q: &[[f64; 3]], v: &[[f64; 3]], tick: u64, led: &Ledger, tr: &mut Traj) {
    tr.ticks.push(tick);
    tr.force.push(reported_force(p, m, q));
    tr.chi.push(chi(m, q, &p.mat_pts));
    tr.dmat.push(dmat(m));
    let e = kinetic(m, v) + potential(m, q) + (m.dissipated - led.d0);
    tr.enum_.push(led.w_ext - (e - led.e0));
}

fn finish_traj(p: &Prob, m: &Model, tr: &mut Traj) {
    let mut fd = vec![0.0f32; m.facets.len()];
    for &(fi, pt) in &p.mat_pts {
        fd[fi] = fd[fi].max(m.facets[fi].pts[pt].law.damage(m.facets[fi].pts[pt].kappa_c) as f32);
    }
    tr.final_d = p.mat_pts.iter().map(|&(f, _)| fd[f]).collect();
}

fn mat_mask(p: &Prob) -> Vec<bool> {
    let mut v = vec![false; p.m.facets.len()];
    for &(f, _) in &p.mat_pts {
        v[f] = true;
    }
    v
}

// ======================================================================= E: leapfrog

/// Explicit leapfrog with `div` substeps per sample tick.
fn run_e(p0: &Prob, div: u64) -> Traj {
    let t0 = Instant::now();
    let p = p0;
    let h = p.grid / div as f64;
    let mut tr = Traj { valid: true, ..Default::default() };
    let mat = mat_mask(p);
    let mut mm = p.m.clone();
    let m = &mut mm;
    if let Some(tag) = p.remove_tag {
        m.facets.iter_mut().filter(|f| f.tag == tag).for_each(|f| f.active = false);
    }
    let mut q = m.q.clone();
    let nb = m.bodies.len();
    let mut vh = p.v_init.clone(); // v_{n-1/2}; at n = 0 the initial v
    let free: Vec<[bool; 3]> = m.bodies.iter().map(|b| [!b.fixed[0], !b.fixed[1], !b.fixed[2]]).collect();
    let minv: Vec<[f64; 3]> = m.bodies.iter().map(|b| [1.0 / b.mass, 1.0 / b.mass, 1.0 / b.inertia]).collect();
    let mut led = Ledger { e0: kinetic(m, &vh) + potential(m, &q), w_ext: 0.0, d0: m.dissipated };
    let nsteps = p.end_ticks * div;
    let mut f = vec![[0.0f64; 3]; nb];
    let mut prev_load = 0.0;
    let mut prev_d = 0.0;
    let mut pk = 0.0f64;
    for n in 0..=nsteps {
        let t = n as f64 * h;
        if let (Some(pl), Some(r)) = (p.plate, p.ramp) {
            q[pl][1] = p.y0 - r.d(t);
        }
        if let Some(pj) = p.proj {
            let vrel = vh[pj][0].abs() + vh[pj][1].abs();
            update_contacts(m, &q, pj, vrel * h + 0.005, p.kc);
        }
        // force at q_n with history committed at q_n
        commit_track(m, &q, &mat, t, &mut tr);
        let lf = p.lf(t);
        for i in 0..nb {
            for r in 0..3 {
                f[i][r] = m.loads[i][r] + lf * m.pattern[i][r];
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
        }
        // kick: v_{n+1/2}
        let mut vn = vh.clone();
        let dt_kick = if n == 0 { 0.5 * h } else { h };
        let mut vnew = vh.clone();
        for i in 0..nb {
            for r in 0..3 {
                if free[i][r] {
                    vnew[i][r] = vh[i][r] + dt_kick * minv[i][r] * f[i][r];
                    if n > 0 {
                        vn[i][r] = 0.5 * (vh[i][r] + vnew[i][r]);
                    }
                }
            }
        }
        let load = reported_force(p, m, &q);
        if p.kind == Kind::R2 {
            let d = p.ramp.unwrap().d(t);
            led.w_ext += 0.5 * (load + prev_load) * (d - prev_d);
            prev_d = d;
        }
        prev_load = load;
        if load.abs() > pk {
            pk = load.abs();
            tr.t_peak = t;
        }
        if n % div == 0 {
            record(p, m, &q, &vn, n / div, &led, &mut tr);
        }
        if n == nsteps {
            break;
        }
        vh = vnew;
        for i in 0..nb {
            for r in 0..3 {
                if free[i][r] {
                    led.w_ext += lf * m.pattern[i][r] * h * vh[i][r];
                    q[i][r] += h * vh[i][r];
                }
            }
        }
    }
    tr.peak = pk;
    tr.work_ext = led.w_ext;
    if let Some(pj) = p.proj {
        // contact impulse from the projectile's momentum balance (gravity removed)
        let mp = m.bodies[pj].mass;
        tr.impulse = mp * (vh[pj][1] + p.v0) + mp * G * (nsteps as f64 * h);
    }
    tr.steps = nsteps;
    finish_traj(p, m, &mut tr);
    tr.wall = t0.elapsed().as_secs_f64();
    tr.work = json!({"force_passes": nsteps + 1, "substeps_per_frame": (1.0 / 60.0) / h});
    tr
}

// ======================================================================= I: implicit

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sch {
    Bdf2,
    Nm,
    Ga,
}

impl Sch {
    fn name(&self) -> &'static str {
        match self {
            Sch::Bdf2 => "BDF2",
            Sch::Nm => "Newmark",
            Sch::Ga => "GA(0.5)",
        }
    }
}

const GA_AF: f64 = 1.0 / 3.0;
const GA_BETA: f64 = 4.0 / 9.0;
const GA_GAMMA: f64 = 5.0 / 6.0;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Solv {
    Direct,
    Bridge { eps: f64, consistent: bool },
}

struct ImpState {
    q_prev: Vec<[f64; 3]>,
    v: Vec<[f64; 3]>,
    v_prev: Vec<[f64; 3]>,
    a: Vec<[f64; 3]>,
    /// r_n = ∇Π(q_n) − f on free DOFs (GA)
    r_n: Vec<f64>,
    restart: bool,
}

fn mass_of(m: &Model, i: usize, k: usize) -> f64 {
    if k < 2 {
        m.bodies[i].mass
    } else {
        m.bodies[i].inertia
    }
}

/// Inertial term for the step from q_n with step h; also returns q̃ (Newmark predictor
/// without the GA force shift) for the velocity update.
fn inertia_of(sch: Sch, st: &ImpState, m: &Model, map: &DofMap, q: &[[f64; 3]], h: f64) -> (Inertia, Vec<[f64; 3]>) {
    let n = q.len();
    let mut qhat = vec![[0.0; 3]; n];
    let mut qt = vec![[0.0; 3]; n];
    let alpha_h2 = match sch {
        Sch::Bdf2 if st.restart => 1.0 / (h * h),
        Sch::Bdf2 => 9.0 / (4.0 * h * h),
        Sch::Nm => 4.0 / (h * h),
        Sch::Ga => 1.0 / ((1.0 - GA_AF) * GA_BETA * h * h),
    };
    for i in 0..n {
        for k in 0..3 {
            qhat[i][k] = match sch {
                Sch::Bdf2 if st.restart => q[i][k] + h * st.v[i][k],
                Sch::Bdf2 => 4.0 / 3.0 * q[i][k] - 1.0 / 3.0 * st.q_prev[i][k] + 2.0 * h / 3.0 * (4.0 / 3.0 * st.v[i][k] - 1.0 / 3.0 * st.v_prev[i][k]),
                Sch::Nm => q[i][k] + h * st.v[i][k] + 0.25 * h * h * st.a[i][k],
                Sch::Ga => q[i][k] + h * st.v[i][k] + h * h * (0.5 - GA_BETA) * st.a[i][k],
            };
            qt[i][k] = qhat[i][k];
            if sch == Sch::Ga {
                if let Some(g) = map.idx[i][k] {
                    qhat[i][k] -= GA_AF / (1.0 - GA_AF) * st.r_n[g] / (alpha_h2 * mass_of(m, i, k));
                }
            }
        }
    }
    (Inertia { alpha_h2, qhat }, qt)
}

fn residual_r(m: &Model, q: &[[f64; 3]], map: &DofMap, lf: f64) -> Vec<f64> {
    assemble(m, q, map, lf, None, false, Tangent::Consistent).grad
}

/// Consistent initial acceleration a0 = −M⁻¹ r(q0).
fn consistent_a0(m: &Model, q: &[[f64; 3]], map: &DofMap, lf: f64) -> Vec<[f64; 3]> {
    let r = residual_r(m, q, map, lf);
    let mut a = vec![[0.0; 3]; q.len()];
    for i in 0..q.len() {
        for k in 0..3 {
            if let Some(g) = map.idx[i][k] {
                a[i][k] = -r[g] / mass_of(m, i, k);
            }
        }
    }
    a
}

fn finish_step(sch: Sch, st: &mut ImpState, q_old: &[[f64; 3]], q_new: &[[f64; 3]], qt: &[[f64; 3]], h: f64, free: &[[bool; 3]]) {
    let n = q_new.len();
    let mut v_new = vec![[0.0; 3]; n];
    let mut a_new = vec![[0.0; 3]; n];
    for i in 0..n {
        for k in 0..3 {
            if !free[i][k] {
                continue;
            }
            match sch {
                Sch::Bdf2 if st.restart => {
                    v_new[i][k] = (q_new[i][k] - q_old[i][k]) / h;
                    a_new[i][k] = (v_new[i][k] - st.v[i][k]) / h;
                }
                Sch::Bdf2 => {
                    v_new[i][k] = 1.5 / h * (q_new[i][k] - 4.0 / 3.0 * q_old[i][k] + 1.0 / 3.0 * st.q_prev[i][k]);
                    a_new[i][k] = 1.5 / h * (v_new[i][k] - 4.0 / 3.0 * st.v[i][k] + 1.0 / 3.0 * st.v_prev[i][k]);
                }
                Sch::Nm => {
                    a_new[i][k] = (q_new[i][k] - qt[i][k]) / (0.25 * h * h);
                    v_new[i][k] = st.v[i][k] + 0.5 * h * (st.a[i][k] + a_new[i][k]);
                }
                Sch::Ga => {
                    a_new[i][k] = (q_new[i][k] - qt[i][k]) / (GA_BETA * h * h);
                    v_new[i][k] = st.v[i][k] + h * ((1.0 - GA_GAMMA) * st.a[i][k] + GA_GAMMA * a_new[i][k]);
                }
            }
        }
    }
    st.v_prev = std::mem::replace(&mut st.v, v_new);
    st.a = a_new;
    st.q_prev = q_old.to_vec();
    st.restart = false;
}

// ----------------------------------------------------------------------- bridge

struct Bridge {
    eps: f64,
    ls_fallbacks: u64,
    last_alpha: f64,
    /// direction: secant (Picard) or consistent tangent with secant fallback (declared alternative)
    consistent: bool,
    fallbacks: u64,
    b_sys: Option<System>,
    b_co: Option<Coarse>,
    refreshes: u64,
    retries: u64,
    outer: u64,
    pcg_its: u64,
    force_passes: u64,
    work: Work,
    t_refresh: f64,
    pos: Vec<[f64; 3]>,
    mass: Vec<f64>,
    ell: f64,
}

const CH: Smoother = Smoother::Cheb { deg: 2, lo: 0.25, hi: 2.0 };
const PCG_BUDGET: usize = 8;
const PCG_TOL: f64 = 0.05;
const PHI_NOISE: f64 = 1e-10;

fn to_csr(n: usize, trip: Vec<(usize, usize, f64)>) -> Csr {
    Csr::from_triplets(n, n, trip)
}

impl Bridge {
    fn new(m: &Model, eps: f64) -> Bridge {
        let map = DofMap::new(m);
        let mut pos = Vec::new();
        let mut mass = Vec::new();
        for (i, b) in m.bodies.iter().enumerate() {
            if map.idx[i].iter().any(|x| x.is_some()) {
                assert!(map.idx[i].iter().all(|x| x.is_some()), "bridge needs 3 free DOFs per free body");
                let x0 = b.x0;
                pos.push([x0.x, x0.y, 0.0]);
                mass.push(b.mass);
            }
        }
        let area: f64 = m.bodies.iter().filter(|b| b.kind == BodyKind::Cell).map(|b| b.mass / (m.mat.density * m.mat.thickness)).sum();
        let ell = (area / m.n_cells() as f64).sqrt();
        Bridge { eps, ls_fallbacks: 0, last_alpha: 1.0, consistent: false, fallbacks: 0, b_sys: None, b_co: None, refreshes: 0, retries: 0, outer: 0, pcg_its: 0, force_passes: 0, work: Work::default(), t_refresh: 0.0, pos, mass, ell }
    }

    /// Rebuild B from the current secant operator: strength-aware SA-s8 coarse space.
    fn refresh(&mut self, m: &Model, s: &Csr) {
        let t0 = Instant::now();
        let map = DofMap::new(m);
        let node = |i: usize| map.idx[i][0].map(|g| g / 3);
        let mut edges = Vec::new();
        for f in m.facets.iter().filter(|f| f.active && f.tag == 0) {
            if let (Some(a), Some(b)) = (node(f.a), node(f.b)) {
                let eta = 1.0 - f.pts.iter().map(|p| p.law.damage(p.kappa_c)).fold(0.0, f64::max);
                edges.push((a, b, eta >= 0.1));
            }
        }
        let sys = System::new(s.clone(), 3);
        let agg = aggregate(&AggInput { pos: &self.pos, mass: &self.mass, edges: &edges, target: 8, dim: 2, ell: self.ell });
        let ppr = rigid_p(&agg, &self.pos, &self.mass, 3, self.ell);
        let lam = (1.05 * sys.lambda_dinv_a()).min(2.0);
        let psa = smooth_p(&sys, &ppr, 4.0 / (3.0 * lam));
        let co = Coarse::new(&sys, psa).expect("A_c SPD");
        self.b_sys = Some(sys);
        self.b_co = Some(co);
        self.refreshes += 1;
        self.t_refresh += t0.elapsed().as_secs_f64();
    }

    /// One implicit step: minimise Φ from the initial guess in q.
    fn solve(&mut self, p: &Prob, m: &Model, q: &mut Vec<[f64; 3]>, ine: &Inertia, lf: f64) -> Result<[f64; 5], String> {
        let map = DofMap::new(m);
        let n = map.n;
        let (o0, p0, f0, r0) = (self.outer, self.pcg_its, self.force_passes, self.refreshes);
        let tol_r = self.eps * p.f_min;
        let mut prev: Option<(Vec<f32>, f64)> = None;
        let mut secant_next = false;
        for it in 0..30 {
            let a = assemble(m, q, &map, lf, Some(ine), true, if self.consistent { Tangent::Consistent } else { Tangent::Secant });
            self.force_passes += 1;
            let res = a.grad.iter().fold(0.0f64, |s, &x| s.max(x.abs()));
            let ch = chi(m, q, &p.mat_pts);
            let fr = reported_force(p, m, q);
            let stop = match &prev {
                None => res <= 1e-2 * tol_r,
                Some((c0, f0)) => {
                    let dchi = ch.iter().zip(c0).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) as f64;
                    let df = (fr - f0).abs() / p.f_rt;
                    if it >= 20 && std::env::var("SB_BRIDGE_DEBUG").is_ok() {
                        eprintln!("   it {it}: res/tol {:.3e} dchi {dchi:.3e} df {df:.3e} last_alpha {:.3e} fallbacks {}", res / tol_r, self.last_alpha, self.fallbacks);
                    }
                    res <= tol_r && dchi <= self.eps && df <= self.eps
                }
            };
            if stop {
                return Ok([(self.outer - o0) as f64, (self.pcg_its - p0) as f64, (self.force_passes - f0) as f64, (self.refreshes - r0) as f64, it as f64]);
            }
            prev = Some((ch, fr));
            self.outer += 1;
            let b: Vec<f64> = a.grad.iter().map(|x| -x).collect();
            let mut x = vec![0.0; n];
            let mut use_secant = !self.consistent || secant_next;
            let was_secant_retry = secant_next;
            secant_next = false;
            let mut s_sec: Option<Csr> = None;
            let mut mat = to_csr(n, a.trip);
            loop {
                if use_secant && self.consistent {
                    // fallback: secant matrix at the same state
                    let sa = assemble(m, q, &map, lf, Some(ine), true, Tangent::Secant);
                    self.force_passes += 1;
                    self.fallbacks += 1;
                    mat = to_csr(n, sa.trip);
                }
                if self.b_sys.is_none() {
                    // B is always built from the SPD secant operator
                    let sec = if use_secant { mat.clone() } else {
                        let sa = assemble(m, q, &map, lf, Some(ine), true, Tangent::Secant);
                        self.force_passes += 1;
                        to_csr(n, sa.trip)
                    };
                    self.refresh(m, &sec);
                    s_sec = Some(sec);
                }
                let sys = System::operator(mat.clone(), 3);
                let mut attempt = 0;
                loop {
                    x.iter_mut().for_each(|v| *v = 0.0);
                    let (its, rel) = {
                        let bs = self.b_sys.as_ref().unwrap();
                        let co = self.b_co.as_mut().unwrap();
                        let mut pre = Precond::Stale(bs, co, CH);
                        pcg(&sys, &mut pre, &b, &mut x, PCG_BUDGET, Some(PCG_TOL), &mut self.work, &mut |_, _, _| false)
                    };
                    self.pcg_its += its as u64;
                    if rel <= PCG_TOL || attempt == 1 {
                        break;
                    }
                    // budget failed: refresh B (from the secant) and repeat; first attempt counted
                    self.retries += 1;
                    let sec = match &s_sec {
                        Some(s) => s.clone(),
                        None => {
                            let sa = assemble(m, q, &map, lf, Some(ine), true, Tangent::Secant);
                            self.force_passes += 1;
                            to_csr(n, sa.trip)
                        }
                    };
                    self.refresh(m, &sec);
                    s_sec = Some(sec);
                    attempt += 1;
                }
                let gp: f64 = a.grad.iter().zip(&x).map(|(g, d)| g * d).sum();
                if gp < 0.0 && x.iter().all(|v| v.is_finite()) {
                    break;
                }
                if use_secant {
                    return Err(format!("not a descent direction at outer {it} (g·p = {gp:e})"));
                }
                use_secant = true;
            }
            // Armijo on Φ, residual-decrease fallback
            let gp: f64 = a.grad.iter().zip(&x).map(|(g, d)| g * d).sum();
            let e0 = a.energy;
            let mut alpha = 1.0;
            let mut ok = false;
            // as S0: a predicted decrease below the round-off of Φ cannot be arbitrated by Φ.
            // Φ is built from absolute coordinates, so its relative noise is ≈ ε·L/δ ≈ 1e-10
            // (L ~ 0.5 m, δ ~ 1e-6 m), not 1e3·ε.
            if gp.abs() <= PHI_NOISE * e0.abs() {
                axpy_q(q, &map, 1.0, &x);
                self.last_alpha = 0.0;
                continue;
            }
            for _ in 0..12 {
                let mut qt = q.clone();
                axpy_q(&mut qt, &map, alpha, &x);
                let e1 = energy(m, &qt, lf, Some(ine));
                self.force_passes += 1;
                if e1 <= e0 + 1e-4 * alpha * gp {
                    *q = qt;
                    ok = true;
                    self.last_alpha = alpha;
                    break;
                }
                let at = assemble(m, &qt, &map, lf, Some(ine), false, Tangent::Consistent);
                self.force_passes += 1;
                let rt = at.grad.iter().fold(0.0f64, |s, &v| s.max(v.abs()));
                // residual-decrease acceptance only within the round-off of Φ (its purpose in S0);
                // accepting a real increase of Φ let the iteration cycle at a unilateral kink
                if rt < (1.0 - 1e-4 * alpha) * res && e1 <= e0 + 1e3 * f64::EPSILON * e0.abs() {
                    *q = qt;
                    ok = true;
                    self.last_alpha = -alpha;
                    break;
                }
                alpha *= 0.5;
            }
            if !ok && self.consistent && !was_secant_retry {
                // declared fallback, extended to a failed line search: same state, secant direction
                secant_next = true;
                self.ls_fallbacks += 1;
                continue;
            }
            if !ok {
                return Err(format!("line search failed at outer {it} (Φ {e0:.6e}, g·p {gp:.3e}, res/tol {:.3e})", res / tol_r));
            }
        }
        Err("30 outer iterations".into())
    }
}

/// Implicit run with outer step `h_ticks`; R3 takes `sub_ticks` substeps while the
/// pre-declared event policy is active.
fn run_i(p0: &Prob, sch: Sch, h_ticks: u64, sub_ticks: Option<u64>, solv: Solv) -> Traj {
    let t0 = Instant::now();
    let p = p0;
    let mut tr = Traj { valid: true, ..Default::default() };
    let mat = mat_mask(p);
    let mut mm = p.m.clone();
    let m = &mut mm;
    if let Some(tag) = p.remove_tag {
        m.facets.iter_mut().filter(|f| f.tag == tag).for_each(|f| f.active = false);
    }
    let mut q = m.q.clone();
    let free: Vec<[bool; 3]> = m.bodies.iter().map(|b| [!b.fixed[0], !b.fixed[1], !b.fixed[2]]).collect();
    if let Some(pj) = p.proj {
        update_contacts(m, &q, pj, 0.005, p.kc);
    }
    let map0 = DofMap::new(m);
    let a0 = if sch == Sch::Bdf2 { vec![[0.0; 3]; q.len()] } else { consistent_a0(m, &q, &map0, p.lf(0.0)) };
    let mut st = ImpState { q_prev: q.clone(), v: p.v_init.clone(), v_prev: p.v_init.clone(), a: a0, r_n: Vec::new(), restart: true };
    let mut led = Ledger { e0: kinetic(m, &st.v) + potential(m, &q), w_ext: 0.0, d0: m.dissipated };
    let mut bridge = match solv {
        Solv::Bridge { eps, consistent } => Some({ let mut b = Bridge::new(m, eps); b.consistent = consistent; b }),
        _ => None,
    };
    let mut newton_iters = 0usize;
    let mut roundoff = 0usize;
    let mut force_passes = 0u64;
    record(p, m, &q, &st.v, 0, &led, &mut tr);
    let mut prev_load = tr.force[0];
    let mut pk = prev_load.abs();
    let mut prev_d = 0.0;
    let mut tick = 0u64;
    let mut h_last = 0u64;
    let mut last_contact_t = f64::NEG_INFINITY;
    let mut substeps = 0u64;
    let mut event_steps = 0u64;
    'outer: while tick < p.end_ticks {
        // event policy (R3): decided at outer-step boundaries
        let mut hs = h_ticks.min(p.end_ticks - tick);
        if let (Some(pj), Some(sub)) = (p.proj, sub_ticks) {
            let vrel = st.v[pj][0].abs() + st.v[pj][1].abs();
            let ho = hs as f64 * p.grid;
            let near = predicted_gap(m, &q, pj) < vrel * ho + 0.005;
            let active = contact_force(m, &q) > 0.0;
            let post = tick as f64 * p.grid - last_contact_t < 10.0 * 1.0 / m.mat.wave_speed();
            if near || active || post {
                hs = sub;
                event_steps += 1;
            }
        }
        let n_sub = if hs < h_ticks.min(p.end_ticks - tick) { h_ticks.min(p.end_ticks - tick) / hs } else { 1 };
        for _ in 0..n_sub {
            if hs != h_last && h_last != 0 {
                st.restart = true; // BDF2 restarts with one IE step on a step change
            }
            h_last = hs;
            let h = hs as f64 * p.grid;
            let t_new = (tick + hs) as f64 * p.grid;
            if let Some(pj) = p.proj {
                let vrel = st.v[pj][0].abs() + st.v[pj][1].abs();
                update_contacts(m, &q, pj, vrel * h + 0.005, p.kc);
            }
            let map = DofMap::new(m);
            if sch == Sch::Ga {
                st.r_n = residual_r(m, &q, &map, p.lf(tick as f64 * p.grid));
                force_passes += 1;
            }
            let q_old = q.clone();
            let lf_old = p.lf(tick as f64 * p.grid);
            let lf_new = p.lf(t_new);
            let (ine, qt) = inertia_of(sch, &st, m, &map, &q_old, h);
            // net-force predictor: q̂ − M⁻¹ r(q_n; t_{n+1}) / (α/h²). (E7's free-fall predictor
            // q̂ + M⁻¹f/(α/h²) ignores the internal forces and drove the first step of a
            // structure in equilibrium into a spuriously damaged branch.)
            let r_pred = residual_r(m, &q_old, &map, lf_new);
            force_passes += 1;
            for i in 0..q.len() {
                for c in 0..3 {
                    if let Some(g) = map.idx[i][c] {
                        q[i][c] = ine.qhat[i][c] - r_pred[g] / (mass_of(m, i, c) * ine.alpha_h2);
                    }
                }
            }
            if let (Some(pl), Some(r)) = (p.plate, p.ramp) {
                q[pl][1] = p.y0 - r.d(t_new);
            }
            match solv {
                Solv::Direct => {
                    let rep = newton(m, &mut q, lf_new, Some(&ine), NewtonOpts { tol: 1e-6 * p.f_min, max_iter: 60 });
                    newton_iters += rep.iters;
                    roundoff += rep.roundoff_limited as usize;
                    if !rep.converged {
                        tr.valid = false;
                        tr.invalid = Some(json!({"t": t_new, "reason": "Newton not converged", "res": rep.res, "iters": rep.iters}));
                        break 'outer;
                    }
                }
                Solv::Bridge { .. } => {
                    let br = bridge.as_mut().unwrap();
                    match br.solve(p, m, &mut q, &ine, lf_new) {
                        Ok(s) => tr.step_stats.push(s),
                        Err(e) => {
                            tr.valid = false;
                            tr.invalid = Some(json!({"t": t_new, "reason": e}));
                            break 'outer;
                        }
                    }
                }
            }
            commit_track(m, &q, &mat, t_new, &mut tr);
            for i in 0..q.len() {
                for c in 0..3 {
                    led.w_ext += 0.5 * (lf_old + lf_new) * m.pattern[i][c] * (q[i][c] - q_old[i][c]);
                }
            }
            finish_step(sch, &mut st, &q_old, &q, &qt, h, &free);
            tick += hs;
            substeps += 1;
            let load = reported_force(p, m, &q);
            if p.kind == Kind::R2 {
                let d = p.ramp.unwrap().d(t_new);
                led.w_ext += 0.5 * (load + prev_load) * (d - prev_d);
                prev_d = d;
            }
            if p.kind == Kind::R3 && load > 0.0 {
                last_contact_t = t_new;
            }
            prev_load = load;
            if load.abs() > pk {
                pk = load.abs();
                tr.t_peak = t_new;
            }
            record(p, m, &q, &st.v, tick, &led, &mut tr);
        }
    }
    tr.peak = pk;
    tr.work_ext = led.w_ext;
    if let Some(pj) = p.proj {
        let mp = m.bodies[pj].mass;
        tr.impulse = mp * (st.v[pj][1] + p.v0) + mp * G * (tick as f64 * p.grid);
    }
    tr.steps = substeps;
    finish_traj(p, m, &mut tr);
    tr.wall = t0.elapsed().as_secs_f64();
    let h = h_ticks as f64 * p.grid;
    tr.work = match &bridge {
        Some(b) => json!({"consistent": b.consistent, "secant_fallbacks": b.fallbacks, "line_search_fallbacks": b.ls_fallbacks, "outer": b.outer, "pcg_its": b.pcg_its, "force_passes": b.force_passes + force_passes, "refreshes": b.refreshes, "retries": b.retries,
            "a_apps": b.work.a_apps, "phases": b.work.phases, "reductions": b.work.reductions, "coarse_solves": b.work.coarse_solves, "vec_ops": b.work.vec_ops,
            "t_refresh_s": b.t_refresh, "steps": substeps, "event_steps": event_steps, "steps_per_frame": (1.0 / 60.0) / h}),
        None => json!({"newton_iters": newton_iters, "roundoff_limited": roundoff, "ga_force_passes": force_passes, "steps": substeps, "event_steps": event_steps, "steps_per_frame": (1.0 / 60.0) / h}),
    };
    tr
}

// ======================================================================= comparison

fn criteria(p: &Prob, r: &Traj, c: &Traj, t_load: f64) -> Value {
    criteria_w(p, r, c, t_load, false)
}

/// `force_full`: force criterion on the whole window (used when the reference pair converges
/// there); otherwise up to the reference's first fully damaged facet.
fn criteria_w(p: &Prob, r: &Traj, c: &Traj, t_load: f64, force_full: bool) -> Value {
    let grid = p.grid;
    let idx = |tick: u64| -> Option<usize> { r.ticks.binary_search(&tick).ok() };
    let mut ferr: f64 = 0.0;
    let mut chi_err: f64 = 0.0;
    let mut cross = [[0usize; 2]; 3]; // [θ][false, missed]
    let ths = [0.8f32, 0.9, 1.0];
    let mut eerr: f64 = 0.0;
    let cutoff = r.first_broken.unwrap_or(f64::INFINITY);
    let np = p.mat_pts.len();
    let mut fal = vec![[false; 3]; np];
    let mut mis = vec![[false; 3]; np];
    let mut ferr_all: f64 = 0.0;
    for (k, &tk) in c.ticks.iter().enumerate() {
        let Some(j) = idx(tk) else { continue };
        let fe = (c.force[k] - r.force[j]).abs() / p.f_scale;
        ferr_all = ferr_all.max(fe);
        if force_full || (tk as f64) * grid <= cutoff {
            ferr = ferr.max(fe);
        }
        eerr = eerr.max(c.enum_[k].abs() / p.e_scale);
        if (tk as f64) * grid < cutoff {
            for i in 0..np {
                let (a, b) = (r.chi[j][i], c.chi[k][i]);
                if a >= 0.8 || b >= 0.8 {
                    chi_err = chi_err.max((a - b).abs() as f64);
                }
                for (ti, &th) in ths.iter().enumerate() {
                    if b >= th && a < th {
                        fal[i][ti] = true;
                    }
                    if a >= th && b < th {
                        mis[i][ti] = true;
                    }
                }
            }
        }
    }
    for i in 0..np {
        for t in 0..3 {
            cross[t][0] += fal[i][t] as usize;
            cross[t][1] += mis[i][t] as usize;
        }
    }
    let d_ref = *r.dmat.last().unwrap_or(&0.0);
    let d_c = *c.dmat.last().unwrap_or(&0.0);
    let d_rel = if d_ref > 1e-9 * p.e_scale { (d_c - d_ref).abs() / d_ref } else { f64::NAN };
    let onset_tol = (1e-4f64).max(0.02 * t_load);
    let onset = match (r.onset, c.onset) {
        (Some(a), Some(b)) => Some((b - a).abs()),
        (None, None) => None,
        _ => Some(f64::INFINITY),
    };
    let jac = {
        let (mut i, mut u) = (0, 0);
        for k in 0..np.min(c.final_d.len()).min(r.final_d.len()) {
            let (a, b) = (r.final_d[k] > 0.5, c.final_d[k] > 0.5);
            i += (a && b) as usize;
            u += (a || b) as usize;
        }
        if u == 0 { 1.0 } else { i as f64 / u as f64 }
    };
    let peak_rel = (c.peak - r.peak).abs() / r.peak.max(1e-30);
    let imp_rel = if p.kind == Kind::R3 { (c.impulse - r.impulse).abs() / r.impulse.abs().max(1e-30) } else { f64::NAN };
    let mut fails = Vec::new();
    if !c.valid {
        fails.push("invalid".to_string());
    }
    let force_hist = matches!(p.kind, Kind::R1 | Kind::R2 | Kind::A1);
    if force_hist && ferr > 0.05 {
        fails.push("force".into());
    }
    if matches!(p.kind, Kind::R2 | Kind::R3) && peak_rel > 0.05 {
        fails.push("peak".into());
    }
    if p.kind == Kind::R3 && imp_rel > 0.05 {
        fails.push("impulse".into());
    }
    if let Some(o) = onset {
        if o > onset_tol {
            fails.push("onset".into());
        }
    }
    if chi_err > 0.01 {
        fails.push("chi".into());
    }
    if d_rel.is_finite() && d_rel > 0.05 {
        fails.push("D".into());
    }
    if eerr > 0.01 {
        fails.push("energy".into());
    }
    json!({"pass": fails.is_empty(), "fails": fails, "force_err": ferr, "force_err_full_window": ferr_all, "force_window_end_s": if cutoff.is_finite() && !force_full { json!(cutoff) } else { Value::Null }, "peak_rel": peak_rel, "impulse_rel": imp_rel,
        "onset_err_s": onset, "onset_tol_s": onset_tol, "chi_err": chi_err,
        "crossings": {"false_0.8": cross[0][0], "missed_0.8": cross[0][1], "false_0.9": cross[1][0], "missed_0.9": cross[1][1], "false_1.0": cross[2][0], "missed_1.0": cross[2][1]},
        "D": d_c, "D_ref": d_ref, "D_rel": d_rel, "jaccard_d05": jac, "energy_err": eerr, "E_num_end": c.enum_.last(), "peak": c.peak, "t_peak": c.t_peak, "impulse": c.impulse,
        "onset": c.onset, "first_broken": c.first_broken})
}

fn summary(tr: &Traj) -> Value {
    json!({"valid": tr.valid, "invalid": tr.invalid, "steps": tr.steps, "wall_s": tr.wall, "work": tr.work, "D_end": tr.dmat.last(), "E_num_end": tr.enum_.last(),
        "onset": tr.onset, "first_broken": tr.first_broken, "peak": tr.peak, "t_peak": tr.t_peak, "impulse": tr.impulse, "broken_d05": tr.final_d.iter().filter(|&&d| d > 0.5).count()})
}

fn with_grid(p: &Prob, grid: f64, end_ticks: u64) -> Prob {
    let mut q = p.clone();
    q.grid = grid;
    q.end_ticks = end_ticks;
    q
}

/// A1: one-element energy cycle for every scheme (GA admission gate).
fn cmd_a1() -> Value {
    let mut out = Vec::new();
    for &ke in &[0.05, 0.5, 3.0] {
        let base = a1(ke);
        let (kmax, dex) = if ke < 0.075 { (f64::NAN, 0.0) } else { a1_exact(&base) };
        let period = base.info["period_s"].as_f64().unwrap();
        let mut rows = Vec::new();
        for &n in &[5u64, 10, 20, 40, 80, 160, 320] {
            let p = with_grid(&base, period / n as f64, 10 * n);
            let mut runs: Vec<(String, Traj)> = Vec::new();
            if n >= 20 {
                runs.push(("E".into(), run_e(&p, 1)));
            }
            for sch in [Sch::Bdf2, Sch::Nm, Sch::Ga] {
                runs.push((sch.name().into(), run_i(&p, sch, 1, None, Solv::Direct)));
            }
            for (name, tr) in runs {
                let d = *tr.dmat.last().unwrap();
                let creation = tr.enum_.iter().cloned().fold(0.0f64, f64::min) / base.e_scale;
                let osc = tr.enum_.iter().cloned().fold(0.0f64, f64::max) / base.e_scale;
                let row = json!({"scheme": name, "max_energy_loss_rel": osc, "h_over_T": 1.0 / n as f64, "valid": tr.valid, "D": d, "D_exact": dex, "D_rel_err": if dex > 0.0 { (d - dex) / dex } else { f64::NAN },
                    "max_energy_creation_rel": -creation, "E_num_end_rel": tr.enum_.last().unwrap() / base.e_scale, "steps": tr.steps});
                eprintln!("A1 KE={ke} {name:8} h=T/{n:<4} D {d:.5} (exact {dex:.5}, err {:+.2e}) creation {:.2e} E_num_end {:+.3e}", (d - dex) / dex, -creation, tr.enum_.last().unwrap() / base.e_scale);
                rows.push(row);
            }
        }
        out.push(json!({"case": base.name, "info": base.info, "kappa_max_exact": kmax, "D_exact": dex, "runs": rows}));
    }
    json!(out)
}

/// Run jobs on up to `nt` threads, results in job order.
fn par<T: Send>(nt: usize, jobs: Vec<Box<dyn FnOnce() -> T + Send + '_>>) -> Vec<T> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let n = jobs.len();
    let slots: Vec<Mutex<Option<Box<dyn FnOnce() -> T + Send + '_>>>> = jobs.into_iter().map(|j| Mutex::new(Some(j))).collect();
    let out: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..nt.min(n) {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::SeqCst);
                if k >= n {
                    break;
                }
                let job = slots[k].lock().unwrap().take().unwrap();
                let r = job();
                *out[k].lock().unwrap() = Some(r);
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

const NT: usize = 10;

/// Exact solution of the linearised semi-discrete R1 (prop removed) by dense modes.
fn modal_reference(p: &Prob, end_ticks: u64) -> (Traj, Value) {
    let t0 = Instant::now();
    let mut m = p.m.clone();
    if let Some(tag) = p.remove_tag {
        m.facets.iter_mut().filter(|f| f.tag == tag).for_each(|f| f.active = false);
    }
    let map = DofMap::new(&m);
    let n = map.n;
    let qref: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
    let a = assemble(&m, &qref, &map, 0.0, None, true, Tangent::Consistent);
    let f: Vec<f64> = a.grad.iter().map(|g| -g).collect();
    let mut kd = vec![0.0; n * n];
    for &(r, c, v) in &a.trip {
        kd[r * n + c] += v;
    }
    let mut md = vec![0.0; n];
    for (i, row) in map.idx.iter().enumerate() {
        for k in 0..3 {
            if let Some(g) = row[k] {
                md[g] = mass_of(&m, i, k);
            }
        }
    }
    let bm: Vec<f64> = (0..n * n).map(|p| kd[p] / (md[p / n] * md[p % n]).sqrt()).collect();
    let (lam, y) = solvers::spectrum::jacobi_eig(&bm, n);
    // φ_i = M^{-1/2} y_i (columns of y); residuals
    let phi = |i: usize, r: usize| y[r * n + i] / md[r].sqrt();
    let mut res_max: f64 = 0.0;
    for i in 0..n {
        let mut num: f64 = 0.0;
        let mut den: f64 = 0.0;
        for r in 0..n {
            let mut kv = 0.0;
            for c in 0..n {
                kv += kd[r * n + c] * phi(i, c);
            }
            num = num.max((kv - lam[i] * md[r] * phi(i, r)).abs());
            den = den.max((lam[i] * md[r] * phi(i, r)).abs());
        }
        res_max = res_max.max(num / den);
    }
    // modal coordinates: η̈ + ω²η = g + s(t) p, η(0) = φᵀMu0, η̇(0) = 0
    let mut u0 = vec![0.0; n];
    let mut fp = vec![0.0; n];
    for (i, row) in map.idx.iter().enumerate() {
        for k in 0..3 {
            if let Some(g) = row[k] {
                u0[g] = p.m.q[i][k] - qref[i][k];
                fp[g] = m.pattern[i][k];
            }
        }
    }
    let proj = |v: &[f64], i: usize| -> f64 { (0..n).map(|r| phi(i, r) * v[r]).sum() };
    let gm: Vec<f64> = (0..n).map(|i| proj(&f, i)).collect();
    let pm: Vec<f64> = (0..n).map(|i| proj(&fp, i)).collect();
    let mu0: Vec<f64> = (0..n).map(|r| md[r] * u0[r]).collect();
    let eta0: Vec<f64> = (0..n).map(|i| proj(&mu0, i)).collect();
    let recon: f64 = (0..n).map(|r| ((0..n).map(|i| phi(i, r) * eta0[i]).sum::<f64>() - u0[r]).abs()).fold(0.0, f64::max);
    let omega: Vec<f64> = lam.iter().map(|l| l.max(0.0).sqrt()).collect();
    let tr_r = p.ramp_tr;
    let big_om = tr_r.map(|t| std::f64::consts::PI / t).unwrap_or(0.0);
    let res_gap = if tr_r.is_some() { omega.iter().map(|w| (w * w - big_om * big_om).abs() / (big_om * big_om)).fold(f64::INFINITY, f64::min) } else { f64::NAN };
    let eta_at = |i: usize, t: f64| -> f64 {
        let w2 = omega[i] * omega[i];
        let w = omega[i];
        match tr_r {
            None => gm[i] / w2 + (eta0[i] - gm[i] / w2) * (w * t).cos(),
            Some(tr) => {
                let d = w2 - big_om * big_om;
                let c1 = eta0[i] - (gm[i] + 0.5 * pm[i]) / w2 - 0.5 * pm[i] / d;
                if t <= tr {
                    (gm[i] + 0.5 * pm[i]) / w2 + 0.5 * pm[i] * (big_om * t).cos() / d + c1 * (w * t).cos()
                } else {
                    let er = (gm[i] + 0.5 * pm[i]) / w2 - 0.5 * pm[i] / d + c1 * (w * tr).cos();
                    let vr = -c1 * w * (w * tr).sin();
                    let tau = t - tr;
                    gm[i] / w2 + (er - gm[i] / w2) * (w * tau).cos() + vr / w * (w * tau).sin()
                }
            }
        }
    };
    let mut tr = Traj { valid: true, ..Default::default() };
    let mut q = qref.clone();
    for tick in 0..=end_ticks {
        let t = tick as f64 * p.grid;
        let w: Vec<f64> = (0..n).map(|i| eta_at(i, t)).collect();
        for (bi, row) in map.idx.iter().enumerate() {
            for k in 0..3 {
                if let Some(g) = row[k] {
                    q[bi][k] = qref[bi][k] + (0..n).map(|i| phi(i, g) * w[i]).sum::<f64>();
                }
            }
        }
        tr.ticks.push(tick);
        tr.force.push(reported_force(p, &m, &q));
        tr.chi.push(chi(&m, &q, &p.mat_pts));
        tr.dmat.push(0.0);
        tr.enum_.push(0.0);
    }
    tr.peak = tr.force.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    tr.final_d = vec![0.0; p.mat_pts.len()];
    tr.wall = t0.elapsed().as_secs_f64();
    let wmin = omega.iter().cloned().filter(|w| *w > 0.0).fold(f64::INFINITY, f64::min);
    let wmax = omega.iter().cloned().fold(0.0, f64::max);
    (tr, json!({"dofs": n, "eig_residual_max": res_max, "u0_reconstruction_max_m": recon, "ramp_resonance_gap_min": res_gap, "omega_min": wmin, "omega_max": wmax,
        "f1_Hz": wmin / (2.0 * std::f64::consts::PI), "h_crit_s": 2.0 / wmax, "wall_s": t0.elapsed().as_secs_f64()}))
}

/// E divisor: substeps per sample tick so that h_E ≤ h_safe.
fn e_div(p: &Prob) -> u64 {
    (p.grid / p.h_safe).ceil() as u64
}

fn imp_ticks(p: &Prob, inv_h: f64) -> u64 {
    let t = (1.0 / inv_h / p.grid).round() as u64;
    assert!(((t as f64) * p.grid * inv_h - 1.0).abs() < 1e-9, "step not on the grid");
    t
}

/// A2: R1 elastic, E / BDF2 / Newmark / GA against the modal reference.
fn cmd_a2(seeds: &[u64], long: bool, tr_opt: Option<f64>) -> Value {
    let mut out = Vec::new();
    for &seed in seeds {
        let base = r1(seed, false, tr_opt);
        let p = if long { with_grid(&base, base.grid, 5 * 7680) } else { base.clone() };
        let (rf, minfo) = modal_reference(&p, p.end_ticks);
        eprintln!("A2 {} modal: {minfo}", p.name);
        let nd = e_div(&p);
        let mut jobs: Vec<(String, Box<dyn FnOnce() -> Traj + Send + '_>)> = Vec::new();
        for k in [1u64, 2, 4] {
            let pp = &p;
            jobs.push((format!("E h_E/{k}"), Box::new(move || run_e(pp, nd * k))));
        }
        for sch in [Sch::Bdf2, Sch::Nm, Sch::Ga] {
            for inv in [240.0, 480.0, 960.0, 1920.0, 3840.0] {
                let pp = &p;
                let ht = imp_ticks(&p, inv);
                jobs.push((format!("{} 1/{inv}", sch.name()), Box::new(move || run_i(pp, sch, ht, None, Solv::Direct))));
            }
        }
        let names: Vec<String> = jobs.iter().map(|j| j.0.clone()).collect();
        let trs = par(NT, jobs.into_iter().map(|j| j.1).collect());
        if std::env::var("SB_DUMP").is_ok() {
            let mut csv = experiments::Csv::new(EXP, &format!("a2_series_s{seed}"), &["method", "t", "force"]);
            for (k, &tk) in rf.ticks.iter().enumerate() {
                csv.row(&[-1.0, tk as f64 * p.grid, rf.force[k]]);
            }
            for (mi, tr) in trs.iter().enumerate() {
                for (k, &tk) in tr.ticks.iter().enumerate() {
                    csv.row(&[mi as f64, tk as f64 * p.grid, tr.force[k]]);
                }
            }
        }
        let mut rows = Vec::new();
        for (name, tr) in names.iter().zip(&trs) {
            let c = criteria(&p, &rf, tr, 0.0);
            eprintln!("A2 {} {:16} pass {} force {:.3e} chi {:.3e} energy {:.3e} fails {}", p.name, name, c["pass"], c["force_err"].as_f64().unwrap(), c["chi_err"].as_f64().unwrap(), c["energy_err"].as_f64().unwrap(), c["fails"]);
            rows.push(json!({"method": name, "criteria": c, "run": summary(tr)}));
        }
        out.push(json!({"problem": p.name, "info": p.info, "window_s": p.end_ticks as f64 * p.grid, "h_E_s": p.grid / nd as f64, "modal": minfo, "runs": rows}));
    }
    json!(out)
}

/// Convergence of the reference pair (E h_E/2 against E h_E/4): every main metric < 1 %.
fn converged(p: &Prob, c: &Value) -> bool {
    let f = |k: &str| c[k].as_f64().unwrap_or(0.0);
    let fin = |k: &str| { let v = f(k); if v.is_finite() { v } else { 0.0 } };
    let force_ok = !matches!(p.kind, Kind::R1 | Kind::R2) || f("force_err") <= 0.01;
    let onset_ok = c["onset_err_s"].as_f64().map(|o| o <= 0.2 * f("onset_tol_s")).unwrap_or(true);
    force_ok && fin("peak_rel") <= 0.01 && fin("impulse_rel") <= 0.01 && fin("D_rel") <= 0.01 && f("chi_err") <= 0.002 && onset_ok
}

struct Cand {
    name: String,
    family: String,
    h: f64,
    tr: Traj,
}

/// Reference (E h_E/4), its convergence partner (E h_E/2) and the candidates.
fn sweep(p: &Prob, imp: &[(Sch, f64)], sub_ticks: Option<u64>, extra_e: bool) -> (Traj, Vec<Cand>, Value) {
    let nd = e_div(p);
    let mut jobs: Vec<(String, String, f64, Box<dyn FnOnce() -> Traj + Send + '_>)> = Vec::new();
    let ks: &[u64] = if extra_e { &[8, 4, 2, 1] } else { &[4, 2, 1] };
    for &k in ks {
        jobs.push((format!("E h_E/{k}"), "E".into(), p.grid / (nd * k) as f64, Box::new(move || run_e(p, nd * k))));
    }
    for &(sch, inv) in imp {
        let ht = imp_ticks(p, inv);
        jobs.push((format!("{} 1/{inv}", sch.name()), sch.name().into(), 1.0 / inv, Box::new(move || run_i(p, sch, ht, sub_ticks, Solv::Direct))));
    }
    let meta_: Vec<(String, String, f64)> = jobs.iter().map(|j| (j.0.clone(), j.1.clone(), j.2)).collect();
    let trs = par(NT, jobs.into_iter().map(|j| j.3).collect());
    let mut cands: Vec<Cand> = meta_.into_iter().zip(trs).map(|((name, family, h), tr)| Cand { name, family, h, tr }).collect();
    let rf = cands.remove(0).tr;
    let info = json!({"h_E_s": p.grid / nd as f64, "e_div": nd, "ref_wall_s": rf.wall});
    (rf, cands, info)
}

/// Run a problem: reference, convergence check, candidates, minimal passing h per family.
fn cmd_problem(p0: &Prob, imp: &[(Sch, f64)], sub_ticks: Option<u64>, fine_ref: bool) -> Value {
    let mut p = p0.clone();
    let (rf, cands, info) = sweep(&p, imp, sub_ticks, fine_ref);
    let partner = if fine_ref { "E h_E/4" } else { "E h_E/2" };
    if p.kind != Kind::R1 {
        p.f_scale = rf.peak;
    }
    if p.kind == Kind::R2 {
        p.e_scale = (*rf.enum_.last().unwrap()).abs().max(1e-30); // replaced below by the plate work
    }
    let t_load = match p.kind {
        Kind::R1 => p.ramp_tr.unwrap_or(0.0),
        Kind::R2 => rf.t_peak,
        _ => 0.0,
    };
    if p.kind == Kind::R2 {
        p.e_scale = rf_plate_work(&rf);
    }
    if std::env::var("SB_DUMP").is_ok() {
        let tag: String = p.name.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect();
        let mut csv = experiments::Csv::new(EXP, &format!("series_{tag}"), &["method", "t", "force", "D"]);
        for (k, &tk) in rf.ticks.iter().enumerate() {
            csv.row(&[-1.0, tk as f64 * p.grid, rf.force[k], rf.dmat[k]]);
        }
        for (mi, c) in cands.iter().enumerate() {
            for (k, &tk) in c.tr.ticks.iter().enumerate() {
                csv.row(&[mi as f64, tk as f64 * p.grid, c.tr.force[k], c.tr.dmat[k]]);
            }
        }
    }
    let mut rows = Vec::new();
    let pc = &cands.iter().find(|c| c.name == partner).unwrap().tr;
    let pcr = criteria(&p, &rf, pc, t_load);
    let force_full = pcr["force_err_full_window"].as_f64().unwrap() <= 0.01;
    let conv = json!({"converged": converged(&p, &pcr), "partner": partner, "pair": pcr, "force_full_window_converged": force_full});
    for c in &cands {
        let cr = criteria_w(&p, &rf, &c.tr, t_load, force_full);
        eprintln!("{} | {:16} pass {:5} force {:.2e} (full {:.2e}) peak {:.2e} imp {:.2e} onset {:?} chi {:.2e} D {:.4} ({:+.2e}) E {:.2e} fails {} | wall {:.1}s",
            p.name, c.name, cr["pass"], cr["force_err"].as_f64().unwrap(), cr["force_err_full_window"].as_f64().unwrap(), cr["peak_rel"].as_f64().unwrap(), cr["impulse_rel"].as_f64().unwrap_or(f64::NAN),
            cr["onset_err_s"].as_f64(), cr["chi_err"].as_f64().unwrap(), cr["D"].as_f64().unwrap(), cr["D_rel"].as_f64().unwrap_or(f64::NAN), cr["energy_err"].as_f64().unwrap(), cr["fails"], c.tr.wall);
        rows.push(json!({"method": c.name, "family": c.family, "h": c.h, "criteria": cr, "run": summary(&c.tr)}));
    }
    eprintln!("{} reference: {} convergence pair: {}", p.name, summary(&rf), conv["converged"]);
    let mut minimal = serde_json::Map::new();
    for fam in ["E", "BDF2", "Newmark", "GA(0.5)"] {
        let best = rows.iter().filter(|r| r["family"] == fam && r["criteria"]["pass"] == json!(true)).map(|r| r["h"].as_f64().unwrap()).fold(0.0f64, f64::max);
        minimal.insert(fam.into(), if best > 0.0 { json!(best) } else { Value::Null });
    }
    json!({"problem": p.name, "info": p.info, "sweep": info, "f_scale": p.f_scale, "e_scale": p.e_scale, "t_load": t_load, "reference": summary(&rf), "convergence": conv,
        "largest_passing_h": minimal, "runs": rows})
}

/// Plate work of a reference trajectory (R2): W_ext = E_num + ΔK + ΔΠ + D; recorded in
/// the ledger, so recover it from the last sample (E_num is the residual of W_ext).
fn rf_plate_work(rf: &Traj) -> f64 {
    rf.work_ext.max(1e-30)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let t0 = Instant::now();
    match cmd {
        "a1" => {
            let r = cmd_a1();
            write_json(EXP, "a1_energy_cycle", &json!({"meta": meta(EXP, "a1", None, "one-element energy cycle: E leapfrog, BDF2, Newmark, GA(0.5) with direct Newton; h = T/n", json!({})), "cases": r, "seconds": t0.elapsed().as_secs_f64()}));
        }
        "r1" => {
            let seeds: Vec<u64> = args.iter().skip(2).map(|s| s.parse().unwrap()).collect();
            let seeds = if seeds.is_empty() { vec![7, 8] } else { seeds };
            let mut out = Vec::new();
            for &s in &seeds {
                let p = r1(s, true, Some(0.01));
                let imp: Vec<(Sch, f64)> = [240.0, 480.0, 960.0, 1920.0, 3840.0].iter().flat_map(|&h| [(Sch::Bdf2, h), (Sch::Nm, h)]).collect();
                out.push(cmd_problem(&p, &imp, None, true));
            }
            let name = format!("r1_{}", seeds.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("_"));
            write_json(EXP, &name, &json!({"meta": meta(EXP, "r1", None, "R1' (prop released over 10 ms) with SB-IL1: E leapfrog vs BDF2 / Newmark (direct Newton)", json!({"seeds": seeds})), "problems": out, "seconds": t0.elapsed().as_secs_f64()}));
        }
        "r2" => {
            let cases: Vec<(f64, u64)> = match args.get(2).map(|s| s.as_str()) {
                Some("heldout") => vec![(0.008, 2)],
                Some("refine") => vec![(0.0056, 1)],
                _ => vec![(0.008, 1)],
            };
            let mut out = Vec::new();
            for &(d, s) in &cases {
                let p = r2(d, s, 0.25);
                let imp: Vec<(Sch, f64)> = [240.0, 480.0, 960.0, 1920.0, 3840.0].iter().flat_map(|&h| [(Sch::Bdf2, h), (Sch::Nm, h)]).collect();
                out.push(cmd_problem(&p, &imp, None, false));
            }
            let name = format!("r2_{}", args.get(2).map(|s| s.as_str()).unwrap_or("dev"));
            write_json(EXP, &name, &json!({"meta": meta(EXP, "r2", None, "R2 notched beam, prescribed C1 plate ramp 1 mm/s to 0.25 mm + 0.25 s hold: E leapfrog vs BDF2 / Newmark (direct Newton)", json!({"cases": cases})), "problems": out, "seconds": t0.elapsed().as_secs_f64()}));
        }
        "r3" => {
            let seeds: Vec<u64> = args.iter().skip(2).map(|s| s.parse().unwrap()).collect();
            let seeds = if seeds.is_empty() { vec![9, 10] } else { seeds };
            let mut out = Vec::new();
            for &s in &seeds {
                let p = r3(s, 20.0, 1e9, 0.006);
                let sub: u64 = std::env::var("SB_SUB").ok().map(|v| v.parse().unwrap()).unwrap_or(4);
                let imp: Vec<(Sch, f64)> = [240.0, 480.0, 960.0, 1920.0].iter().flat_map(|&h| [(Sch::Bdf2, h), (Sch::Nm, h)]).collect();
                let mut r = cmd_problem(&p, &imp, Some(sub), false);
                r["event_substep_ticks"] = json!(sub);
                out.push(r);
            }
            let sub_tag = std::env::var("SB_SUB").map(|v| format!("_sub{v}")).unwrap_or_default();
            let name = format!("r3{sub_tag}_{}", seeds.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("_"));
            write_json(EXP, &name, &json!({"meta": meta(EXP, "r3", None, "R3 impact 20 m/s, k_c 1e9: E leapfrog vs BDF2 / Newmark (direct Newton) with the declared event substeps (4 ticks = 20.8 us)", json!({"seeds": seeds})), "problems": out, "seconds": t0.elapsed().as_secs_f64()}));
        }
        "bridge" => {
            // closed-loop bridge vs direct Newton at the same h (algebraic error) + work
            let which = args.get(2).map(|s| s.as_str()).unwrap_or("r1e");
            let inv: f64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(480.0);
            let eps: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1e-3);
            let consistent = args.get(5).map(|s| s == "consistent").unwrap_or(false);
            let mut p = match which {
                "r1e" => r1(7, false, Some(0.01)),
                "r1d" => r1(7, true, Some(0.01)),
                "r2" => r2(0.008, 1, 0.25),
                "r2s" => { let q = r2(0.008, 1, 0.25); let e = (0.12 * 7680.0) as u64; with_grid(&q, q.grid, e) }
                _ => panic!("problem"),
            };
            let ht = imp_ticks(&p, inv);
            let jobs: Vec<Box<dyn FnOnce() -> Traj + Send + '_>> = vec![Box::new(|| run_i(&p, Sch::Bdf2, ht, None, Solv::Direct)), Box::new(|| run_i(&p, Sch::Bdf2, ht, None, Solv::Bridge { eps, consistent }))];
            let mut trs = par(2, jobs);
            let br = trs.pop().unwrap();
            let dr = trs.pop().unwrap();
            if p.kind != Kind::R1 {
                p.f_scale = dr.peak;
                p.e_scale = dr.work_ext.max(1e-30);
            }
            let cr = criteria_w(&p, &dr, &br, if p.kind == Kind::R2 { dr.t_peak } else { 0.01 }, true);
            let st = &br.step_stats;
            let mx = |k: usize| st.iter().map(|s| s[k]).fold(0.0, f64::max);
            let mean = |k: usize| st.iter().map(|s| s[k]).sum::<f64>() / st.len().max(1) as f64;
            eprintln!("bridge {} BDF2 1/{inv} eps {eps:e}: valid {} | algebraic: force {:.2e} chi {:.2e} D {:?} onset {:?} | outer mean {:.2} max {} | pcg mean {:.2} max {} | passes mean {:.1} | work {}",
                p.name, br.valid, cr["force_err"].as_f64().unwrap(), cr["chi_err"].as_f64().unwrap(), cr["D_rel"].as_f64(), cr["onset_err_s"].as_f64(), mean(0), mx(0), mean(1), mx(1), mean(2), br.work);
            if let Some(iv) = &br.invalid { eprintln!("  invalid: {iv}"); }
            if let Some(iv) = &dr.invalid { eprintln!("  direct invalid: {iv}"); }
            write_json(EXP, &format!("bridge_{which}_{inv}_{eps:e}{}", if consistent { "_consistent" } else { "" }), &json!({"meta": meta(EXP, "bridge", None, "BDF2 closed loop: nonlinear bridge (secant + two-level PCG, RD) vs direct Newton at the same h", json!({"problem": which, "inv_h": inv, "eps": eps})),
                "algebraic": cr, "bridge": summary(&br), "direct": summary(&dr), "step_stats": br.step_stats}));
        }
        "r2probe" => {
            // timing probe: E and BDF2 on a short window
            let p = r2(0.008, 1, 0.0);
            let pp = with_grid(&p, p.grid, 77);
            let nd = e_div(&pp);
            let te = run_e(&pp, nd);
            let ti = run_i(&pp, Sch::Bdf2, 32, None, Solv::Direct);
            eprintln!("{} info {} | E div {nd}: {} steps in {:.2}s ({:.1} us/step) | BDF2 1/240: {} steps {:.2}s work {}", pp.name, pp.info, te.steps, te.wall, 1e6 * te.wall / te.steps as f64, ti.steps, ti.wall, ti.work);
            eprintln!("load at 10 ms: E {:.1} N  BDF2 {:.1} N; D {:.3e} {:.3e}", te.force.last().unwrap(), ti.force.last().unwrap(), te.dmat.last().unwrap(), ti.dmat.last().unwrap());
        }
        "a2" | "a2long" => {
            let long = cmd == "a2long";
            let seeds: Vec<u64> = args.iter().skip(2).map(|s| s.parse().unwrap()).collect();
            let seeds = if seeds.is_empty() { vec![7, 8] } else { seeds };
            let tr_opt = std::env::var("SB_TR").ok().map(|v| v.parse::<f64>().unwrap());
            let r = cmd_a2(&seeds, long, tr_opt);
            let name = format!("{cmd}{}_{}", if tr_opt.is_some() { "_ramp" } else { "" }, seeds.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("_"));
            write_json(EXP, &name, &json!({"meta": meta(EXP, cmd, None, "R1 elastic against the modal reference (dense M-orthonormal modes of the linearised prop-free system)", json!({"seeds": seeds})), "problems": r, "seconds": t0.elapsed().as_secs_f64()}));
        }
        _ => {
            eprintln!("usage: arch_dyn a1 | a2 | r1 | r2 | r3 | bridge");
            let _ = (r1 as fn(u64, bool, Option<f64>) -> Prob, r2 as fn(f64, u64, f64) -> Prob, r3 as fn(u64, f64, f64, f64) -> Prob);
            let _ = (criteria as fn(&Prob, &Traj, &Traj, f64) -> Value, summary as fn(&Traj) -> Value, Solv::Bridge { eps: 1e-3, consistent: false });
        }
    }
}
