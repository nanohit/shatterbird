//! Phase 0c Exp 2 — support removal (docs/phase0c/reduced-fracture.md §3–4).
//!
//! Propped concrete cantilever under gravity; the prop force is released over t_r.
//! Reference: full RBSN, explicit leapfrog at h = 0.3 d_min/c. Candidate: detailed
//! zone + floating-frame reduced regions, BDF2 with event control (Δd ≤ 0.1).
//!
//! A region moves as x_i = c + R(φ)(P_i + U_i η), θ_i = φ + W_i η, where U, W are the
//! rows of an M-orthonormal Ritz basis free of rigid modes (static boundary modes,
//! body-force modes, fixed-interface vibration modes). Its internal facets carry
//! ½ ηᵀ K̂ η (exact under any rigid motion); every other facet is evaluated with the
//! full nonlinear law. The region's inertia in the BDF2 incremental potential is
//! evaluated through mass invariants: O(m²) per Newton iteration, O(n_R m) per step.

use experiments::{meta, write_json, Csv};
use geometry::rng::Rng;
use geometry::voronoi2d::{rect, voronoi_convex};
use geometry::V2;
use model::{BoundaryAttach, LawKind, LawOptions, Material, Model};
use serde::Serialize;
use serde_json::{json, Value};
use solvers::assembly::Tangent;
use solvers::s0::{newton, NewtonOpts};
use solvers::sparse::{csc_from_triplets, lu_solve, Spd};
use solvers::spectrum::jacobi_eig;
use std::time::Instant;

const EXP: &str = "phase0c/exp2";
const G: f64 = 9.81;
const C_PT_TANGENT: f64 = 200.0;
const C_PT_FORCE: f64 = 60.0;
/// Step bound while an impactor is in (or about to be in) contact [s].
const H_CONTACT_DEFAULT: f64 = 1e-4;

fn h_contact() -> f64 {
    std::env::var("SB_HC").ok().and_then(|v| v.parse().ok()).unwrap_or(H_CONTACT_DEFAULT)
}

/// Post-hoc fracture-window policy (Exp 2 diagnosis): step bound while damage evolves or
/// an undamaged evaluated facet is within 5 % of onset. Off unless SB_HFRAC is set.
fn h_frac() -> Option<f64> {
    std::env::var("SB_HFRAC").ok().and_then(|v| v.parse().ok())
}

// ---------------------------------------------------------------- scene

#[derive(Clone)]
struct Scene {
    m: Model,
    t_r: f64,
    dmin: f64,
    len: f64,
    depth: f64,
    clamp_cells: Vec<usize>,
    tip_cells: Vec<usize>,
    name: String,
    /// impact scene: the impactor body (moves in y only) and the initial velocities
    impactor: Option<usize>,
    v_init: Vec<[f64; 3]>,
    /// start implicit runs from the model's own static equilibrium (else from the fit)
    static_presolve: bool,
}

fn release(t: f64, t_r: f64) -> f64 {
    if t >= t_r {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * t / t_r).cos())
    }
}

/// Propped cantilever 4 × 0.4 m (b = 0.3 m), clamped on the left edge, propped on the
/// last 0.2 m of the bottom edge; clamp and prop facets elastic. Returns the propped
/// static equilibrium with the prop facets replaced by their forces (m.pattern).
fn cantilever(seed: u64, dmin: f64, damage: bool, t_r: f64, gscale: f64) -> Scene {
    let (len, depth) = (4.0, 0.4);
    let mut rng = Rng::new(seed);
    // contract seeding at the fixtures (as R1 in Phase 0b/3): no generator within d_min/2
    // of the clamp edge or above the prop, so fixture cells are not slivers
    let keep = |p: V2| !(p.x < 0.5 * dmin) && !(p.y < 0.5 * dmin && p.x >= len - 0.2 - dmin);
    let seeds = geometry::rsa::rsa_rect_with(V2::new(0.0, 0.0), V2::new(len, depth), dmin, &mut rng, 20_000, &keep, Vec::new());
    let t = voronoi_convex(&seeds, &rect(V2::new(0.0, 0.0), V2::new(len, depth)));
    let law = if damage { LawOptions::damage() } else { LawOptions::elastic() };
    let mut m = Model::from_tessellation(&t, Material::concrete(0.3), law, &[]);
    let g = m.ground();
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 3, range: None, body: g, shear: true, tag: 1 });
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((len - 0.2, len)), body: g, shear: true, tag: 2 });
    for f in m.facets.iter_mut().filter(|f| f.tag != 0) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Elastic;
        }
    }
    m.add_gravity(G * gscale);
    let mut q = m.q.clone();
    let mg: f64 = m.bodies.iter().map(|b| b.mass).sum::<f64>() * G;
    let rep = newton(&m, &mut q, 0.0, None, NewtonOpts { tol: 1e-10 * mg, max_iter: 40 });
    assert!(rep.converged, "propped static state");
    m.commit(&q);
    assert!(m.max_damage() == 0.0, "propped state damaged");
    m.q = q.clone();
    for fi in 0..m.facets.len() {
        if m.facets[fi].tag != 2 {
            continue;
        }
        for pt in 0..2 {
            let (j, e) = m.eval_point(&q, &m.facets[fi], pt);
            let a = m.facets[fi].a;
            for r in 0..3 {
                m.pattern[a][r] -= j.jna[r] * e.fnorm + j.jta[r] * e.ftan;
            }
        }
        m.facets[fi].active = false;
    }
    let clamp_cells: Vec<usize> = m.facets.iter().filter(|f| f.tag == 1).map(|f| f.a).collect();
    let tip_cells: Vec<usize> = (0..m.n_cells()).filter(|&i| m.bodies[i].x0.x > len - 0.2).collect();
    let nb = m.bodies.len();
    Scene { m, t_r, dmin, len, depth, clamp_cells, tip_cells, name: format!("cantilever_{}_s{seed}_g{gscale:.4}", if damage { "SBIL1" } else { "elastic" }), impactor: None, v_init: vec![[0.0; 3]; nb], static_presolve: true }
}

/// Simply supported concrete beam 2.0 × 0.2 m (b = 0.2 m, span 1.8 m) on unilateral
/// support pads, hit at mid-span by a rigid flat impactor (pad 0.1 m, mass M_p,
/// moving in y only) at speed v0. Impactor pad facets are unilateral (no tension,
/// normal only). Gravity on the beam; the impactor is weightless and starts touching.
fn beam_impact(seed: u64, dmin: f64, v0: f64, mp: f64) -> Scene {
    let (len, depth, span) = (2.0, 0.2, 1.8);
    let xc = 0.5 * len;
    let (x1, x2) = (0.5 * (len - span), 0.5 * (len + span));
    let mut rng = Rng::new(seed);
    let keep = |p: V2| !(p.y < 0.5 * dmin && ((p.x - x1).abs() < 0.05 + dmin || (p.x - x2).abs() < 0.05 + dmin)) && !(p.y > depth - 0.5 * dmin && (p.x - xc).abs() < 0.05 + dmin);
    let seeds = geometry::rsa::rsa_rect_with(V2::new(0.0, 0.0), V2::new(len, depth), dmin, &mut rng, 20_000, &keep, Vec::new());
    let t = voronoi_convex(&seeds, &rect(V2::new(0.0, 0.0), V2::new(len, depth)));
    let mut m = Model::from_tessellation(&t, Material::concrete(0.2), LawOptions::damage(), &[]);
    let g = m.ground();
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((x1 - 0.05, x1 + 0.05)), body: g, shear: false, tag: 1 });
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((x2 - 0.05, x2 + 0.05)), body: g, shear: false, tag: 2 });
    let plate = m.add_plate(V2::new(xc, depth), mp, 1.0, [true, false, true]);
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 2, range: Some((xc - 0.05, xc + 0.05)), body: plate, shear: false, tag: 3 });
    for f in m.facets.iter_mut().filter(|f| f.tag != 0) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Damage;
            p.law.delta0 *= 1e-4;
            p.law.wc *= 1e-4;
        }
    }
    m.add_gravity(G);
    // static state on the supports with a temporary horizontal gauge (no impactor load)
    let gauge = (0..m.n_cells()).min_by(|&a, &b| (m.bodies[a].x0 - V2::new(xc, 0.5 * depth)).norm().partial_cmp(&(m.bodies[b].x0 - V2::new(xc, 0.5 * depth)).norm()).unwrap()).unwrap();
    m.bodies[gauge].fixed[0] = true;
    m.bodies[plate].fixed[1] = true;
    let mut q = m.q.clone();
    let mg: f64 = m.bodies.iter().map(|b| if b.kind == model::BodyKind::Cell { b.mass } else { 0.0 }).sum::<f64>() * G;
    // elastic (bilateral) pads first, then the unilateral pads from that state
    let mut me = m.clone();
    for f in me.facets.iter_mut().filter(|f| f.tag != 0) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Elastic;
        }
    }
    let rep = newton(&me, &mut q, 0.0, None, NewtonOpts { tol: 1e-9 * mg, max_iter: 60 });
    assert!(rep.converged, "impact static state (bilateral pads)");
    let mut qu = q.clone();
    let rep = newton(&m, &mut qu, 0.0, None, NewtonOpts { tol: 1e-9 * mg, max_iter: 100 });
    if rep.converged {
        q = qu;
    } else {
        eprintln!("impact static state: unilateral re-solve did not converge (res {:.3e}); starting from the bilateral state", rep.res);
    }
    m.bodies[gauge].fixed[0] = false;
    m.bodies[plate].fixed[1] = false;
    // impactor just touching: shift it to the deflected top surface (zero pad force)
    let dy = q[plate][1] - m.bodies[plate].x0.y;
    let _ = dy;
    m.commit(&q);
    m.dissipated = 0.0;
    for f in m.facets.iter_mut().filter(|f| f.tag == 3) {
        for p in f.pts.iter_mut() {
            p.kappa_c = 0.0;
        }
    }
    m.q = q;
    let nb = m.bodies.len();
    let mut v_init = vec![[0.0; 3]; nb];
    v_init[plate] = [0.0, -v0, 0.0];
    let support_cells: Vec<usize> = m.facets.iter().filter(|f| f.tag == 1 || f.tag == 2).map(|f| f.a).collect();
    Scene { m, t_r: 0.0, dmin, len, depth, clamp_cells: support_cells, tip_cells: vec![gauge], name: format!("impact_s{seed}_v{v0:.3}"), impactor: Some(plate), v_init, static_presolve: false }
}

// ---------------------------------------------------------------- work

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct Work {
    newton_iters: u64,
    pts_tangent: u64,
    pts_force: u64,
    coupling_flops: f64,
    region_flops: f64,
    fact_flops: f64,
    solve_flops: f64,
    step_flops: f64,
    monitor_flops: f64,
    rebuild_flops: f64,
}

impl Work {
    fn total(&self) -> f64 {
        self.pts_tangent as f64 * C_PT_TANGENT
            + self.pts_force as f64 * C_PT_FORCE
            + self.coupling_flops
            + self.region_flops
            + self.fact_flops
            + self.solve_flops
            + self.step_flops
            + self.monitor_flops
            + self.rebuild_flops
    }
}

// ---------------------------------------------------------------- floating-frame region

struct FReg {
    cells: Vec<usize>,
    loc: Vec<usize>,
    p: Vec<V2>,
    mi: Vec<f64>,
    ii: Vec<f64>,
    mass: f64,
    it: f64,
    xc: V2,
    me: usize,
    /// (3 ncell) × me, row-major; T_eᵀ M T_e = I, M-orthogonal to rigid modes
    te: Vec<f64>,
    khat: Vec<f64>,
    s1: Vec<f64>,
    w: Vec<f64>,
    internal: Vec<usize>,
    bnd: Vec<usize>,
    off: usize,
}

impl FReg {
    fn ny(&self) -> usize {
        3 + self.me
    }
    /// Body-frame offset r_i = P_i + U_i η and θ-part W_i η of local cell li.
    fn local(&self, li: usize, eta: &[f64]) -> (V2, f64) {
        let me = self.me;
        let row = |k: usize| -> f64 { (0..me).map(|j| self.te[(3 * li + k) * me + j] * eta[j]).sum() };
        (self.p[li] + V2::new(row(0), row(1)), row(2))
    }
}

#[derive(Clone, Debug)]
enum Rep {
    Direct([Option<usize>; 3]),
    Bnd(usize),
    Int(usize),
}

struct Sys {
    rep: Vec<Rep>,
    regs: Vec<FReg>,
    eval_facets: Vec<usize>,
    nd: usize,
    n: usize,
    fact_flops: f64,
    nnz_l: usize,
    detailed_cells: usize,
    q_ref: Vec<[f64; 3]>,
}

struct RegStep {
    xh: V2,
    a0: f64,
    b0: f64,
    a: Vec<f64>,
    b: Vec<f64>,
    th: f64,
    tth: Vec<f64>,
}

struct StepData {
    /// α/h² (0 = static)
    ah: f64,
    qhat: Vec<[f64; 3]>,
    reg: Vec<RegStep>,
    lf: f64,
}

fn rot(v: V2, phi: f64) -> V2 {
    v.rot(phi)
}

fn jperp(v: V2) -> V2 {
    V2::new(-v.y, v.x)
}

impl Sys {
    /// Partition: `detailed[i]` cells are direct; the remaining cells form one
    /// floating-frame region per connected component (through active material facets).
    fn build(m: &Model, detailed: &[bool], nvib: usize, w: &mut Work) -> Sys {
        let nb = m.bodies.len();
        let nc = m.n_cells();
        let is_cell = |i: usize| i < nc;
        // components of non-detailed cells
        let mut comp = vec![usize::MAX; nb];
        let mut ncomp = 0;
        let adj = m.neighbours();
        for s in 0..nc {
            if detailed[s] || comp[s] != usize::MAX {
                continue;
            }
            let mut stack = vec![s];
            comp[s] = ncomp;
            while let Some(i) = stack.pop() {
                for &fi in &adj[i] {
                    let f = &m.facets[fi];
                    let o = if f.a == i { f.b } else { f.a };
                    if is_cell(o) && !detailed[o] && comp[o] == usize::MAX && f.tag == 0 {
                        comp[o] = ncomp;
                        stack.push(o);
                    }
                }
            }
            ncomp += 1;
        }
        let mut eval_facets = Vec::new();
        let mut internal = vec![Vec::new(); ncomp];
        for (fi, f) in m.facets.iter().enumerate() {
            if !f.active {
                continue;
            }
            let (ca, cb) = (if is_cell(f.a) { comp[f.a] } else { usize::MAX }, if is_cell(f.b) { comp[f.b] } else { usize::MAX });
            if ca != usize::MAX && ca == cb {
                internal[ca].push(fi);
            } else {
                eval_facets.push(fi);
            }
        }
        let mut bnd = vec![false; nb];
        for &fi in &eval_facets {
            let f = &m.facets[fi];
            for s in [f.a, f.b] {
                if is_cell(s) && comp[s] != usize::MAX {
                    bnd[s] = true;
                }
            }
        }
        for i in 0..nc {
            if comp[i] != usize::MAX && m.pattern[i].iter().any(|&v| v != 0.0) {
                bnd[i] = true;
            }
        }
        let mut rep = vec![Rep::Direct([None; 3]); nb];
        let mut n = 0;
        for i in 0..nb {
            if !is_cell(i) || comp[i] == usize::MAX {
                let mut idx = [None; 3];
                for k in 0..3 {
                    if !m.bodies[i].fixed[k] {
                        idx[k] = Some(n);
                        n += 1;
                    }
                }
                rep[i] = Rep::Direct(idx);
            } else {
                rep[i] = if bnd[i] { Rep::Bnd(comp[i]) } else { Rep::Int(comp[i]) };
            }
        }
        let nd = n;
        let detailed_cells = (0..nc).filter(|&i| detailed[i]).count();
        let mut regs = Vec::new();
        for r in 0..ncomp {
            let mut cells: Vec<usize> = (0..nc).filter(|&i| comp[i] == r && bnd[i]).collect();
            let nbc = cells.len();
            cells.extend((0..nc).filter(|&i| comp[i] == r && !bnd[i]));
            assert!(nbc > 0, "region without boundary (free component): not supported");
            let nr = cells.len();
            let mut loc = vec![usize::MAX; nb];
            for (li, &i) in cells.iter().enumerate() {
                assert!(m.bodies[i].fixed.iter().all(|&f| !f), "fixed DOF inside a region");
                loc[i] = li;
            }
            let mi: Vec<f64> = cells.iter().map(|&i| m.bodies[i].mass).collect();
            let ii: Vec<f64> = cells.iter().map(|&i| m.bodies[i].inertia).collect();
            let mass: f64 = mi.iter().sum();
            let xc = cells.iter().zip(&mi).fold(V2::ZERO, |s, (&i, &mm)| s + m.bodies[i].x0 * mm) * (1.0 / mass);
            let p: Vec<V2> = cells.iter().map(|&i| m.bodies[i].x0 - xc).collect();
            let it: f64 = ii.iter().sum();
            // linear stiffness of internal facets (reference state), local DOFs 3 li + k
            let mut trip = Vec::new();
            let q0: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
            for &fi in &internal[r] {
                let f = &m.facets[fi];
                for pt in 0..2 {
                    let (j, _) = m.eval_point(&q0, f, pt);
                    let (kn, kt) = (f.pts[pt].law.kn, f.pts[pt].law.kt);
                    let jn = [j.jna[0], j.jna[1], j.jna[2], j.jnb[0], j.jnb[1], j.jnb[2]];
                    let jt = [j.jta[0], j.jta[1], j.jta[2], j.jtb[0], j.jtb[1], j.jtb[2]];
                    let ids = [3 * loc[f.a], 3 * loc[f.a] + 1, 3 * loc[f.a] + 2, 3 * loc[f.b], 3 * loc[f.b] + 1, 3 * loc[f.b] + 2];
                    for a in 0..6 {
                        for c in 0..6 {
                            let v = kn * jn[a] * jn[c] + kt * jt[a] * jt[c];
                            if v != 0.0 {
                                trip.push((ids[a], ids[c], v));
                            }
                        }
                    }
                }
            }
            let nbd = 3 * nbc;
            let ndof = 3 * nr;
            let ni = ndof - nbd;
            let mdiag: Vec<f64> = (0..ndof).map(|d| if d % 3 == 2 { ii[d / 3] } else { mi[d / 3] }).collect();
            let mut kii_t = Vec::new();
            let mut kib = Vec::new();
            for &(a, c, v) in &trip {
                if a >= nbd && c >= nbd {
                    kii_t.push((a - nbd, c - nbd, v));
                } else if a >= nbd {
                    kib.push((a - nbd, c, v));
                }
            }
            // Ritz columns over all region DOFs
            let mut cols: Vec<Vec<f64>> = Vec::new();
            if ni > 0 {
                let kii = csc_from_triplets(ni, kii_t.clone());
                let mut spd = Spd::analyze(&kii);
                spd.factor(&kii).expect("K_II not SPD");
                w.rebuild_flops += spd.stats.flops_factor;
                let solve = |spd: &mut Spd, rhs: &mut Vec<f64>, w: &mut Work| {
                    spd.solve(rhs);
                    w.rebuild_flops += 4.0 * spd.stats.nnz_l as f64;
                };
                // static boundary modes
                for d in 0..nbd {
                    let mut rhs = vec![0.0; ni];
                    for &(a, c, v) in &kib {
                        if c == d {
                            rhs[a] -= v;
                        }
                    }
                    solve(&mut spd, &mut rhs, w);
                    let mut col = vec![0.0; ndof];
                    col[d] = 1.0;
                    col[nbd..].copy_from_slice(&rhs);
                    cols.push(col);
                }
                // body-force modes (boundary fixed)
                for k in 0..2 {
                    let mut rhs: Vec<f64> = (0..ni).map(|a| if (nbd + a) % 3 == k { mdiag[nbd + a] } else { 0.0 }).collect();
                    solve(&mut spd, &mut rhs, w);
                    let mut col = vec![0.0; ndof];
                    col[nbd..].copy_from_slice(&rhs);
                    cols.push(col);
                }
                // fixed-interface vibration modes by subspace iteration
                if nvib > 0 && ni > 3 * nvib {
                    let pb = (nvib + 4).min(ni);
                    let mut x: Vec<Vec<f64>> = (0..pb).map(|k| (0..ni).map(|a| ((a as f64 + 1.0) * (k as f64 + 1.0) * 0.7123).sin()).collect()).collect();
                    let kmul = |v: &[f64]| -> Vec<f64> {
                        let mut o = vec![0.0; ni];
                        for &(a, c, val) in &kii_t {
                            o[a] += val * v[c];
                        }
                        o
                    };
                    let mut vals = vec![0.0; pb];
                    for _ in 0..40 {
                        for v in x.iter_mut() {
                            let mut y: Vec<f64> = v.iter().enumerate().map(|(a, &s)| s * mdiag[nbd + a]).collect();
                            solve(&mut spd, &mut y, w);
                            *v = y;
                        }
                        // M-orthonormalise
                        for k in 0..pb {
                            for l in 0..k {
                                let d: f64 = (0..ni).map(|a| x[k][a] * mdiag[nbd + a] * x[l][a]).sum();
                                let xl = x[l].clone();
                                for a in 0..ni {
                                    x[k][a] -= d * xl[a];
                                }
                            }
                            let nn: f64 = (0..ni).map(|a| x[k][a] * x[k][a] * mdiag[nbd + a]).sum::<f64>().sqrt();
                            for a in 0..ni {
                                x[k][a] /= nn;
                            }
                        }
                        let kx: Vec<Vec<f64>> = x.iter().map(|v| kmul(v)).collect();
                        let mut kr = vec![0.0; pb * pb];
                        for k in 0..pb {
                            for l in 0..pb {
                                kr[k * pb + l] = (0..ni).map(|a| x[k][a] * kx[l][a]).sum();
                            }
                        }
                        w.rebuild_flops += (pb * pb * ni * 2 + pb * ni * 4) as f64;
                        let (ev, vec) = jacobi_eig(&kr, pb);
                        let mut order: Vec<usize> = (0..pb).collect();
                        order.sort_by(|&a, &b| ev[a].partial_cmp(&ev[b]).unwrap());
                        let xn: Vec<Vec<f64>> = order.iter().map(|&c| (0..ni).map(|a| (0..pb).map(|k| x[k][a] * vec[k * pb + c]).sum()).collect()).collect();
                        vals = order.iter().map(|&c| ev[c]).collect();
                        x = xn;
                    }
                    let _ = vals;
                    for v in x.into_iter().take(nvib) {
                        let mut col = vec![0.0; ndof];
                        col[nbd..].copy_from_slice(&v);
                        cols.push(col);
                    }
                }
            } else {
                for d in 0..nbd {
                    let mut col = vec![0.0; ndof];
                    col[d] = 1.0;
                    cols.push(col);
                }
            }
            // M-orthonormalise against the rigid modes, then among themselves
            let rig: Vec<Vec<f64>> = (0..3)
                .map(|k| {
                    (0..ndof)
                        .map(|d| {
                            let (li, kk) = (d / 3, d % 3);
                            match k {
                                0 => (kk == 0) as u8 as f64,
                                1 => (kk == 1) as u8 as f64,
                                _ => [-p[li].y, p[li].x, 1.0][kk],
                            }
                        })
                        .collect()
                })
                .collect();
            let mdot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).zip(&mdiag).map(|((x, y), mm)| x * y * mm).sum() };
            let mut basis: Vec<Vec<f64>> = Vec::new();
            for c in rig.into_iter().chain(cols) {
                let n0 = mdot(&c, &c).sqrt();
                let mut v = c;
                for _ in 0..2 {
                    for u in &basis {
                        let d = mdot(&v, u);
                        for (a, b) in v.iter_mut().zip(u) {
                            *a -= d * b;
                        }
                    }
                }
                let n1 = mdot(&v, &v).sqrt();
                if n1 > 1e-8 * n0 {
                    basis.push(v.iter().map(|a| a / n1).collect());
                }
            }
            w.rebuild_flops += (basis.len() * basis.len() * ndof * 4) as f64;
            let te_cols: Vec<Vec<f64>> = basis.into_iter().skip(3).collect();
            let me = te_cols.len();
            let mut te = vec![0.0; ndof * me];
            for (j, col) in te_cols.iter().enumerate() {
                for d in 0..ndof {
                    te[d * me + j] = col[d];
                }
            }
            let mut kt_ = vec![0.0; ndof * me];
            for &(a, c, v) in &trip {
                for j in 0..me {
                    kt_[a * me + j] += v * te[c * me + j];
                }
            }
            let mut khat = vec![0.0; me * me];
            for a in 0..ndof {
                for i in 0..me {
                    let ta = te[a * me + i];
                    if ta != 0.0 {
                        for j in 0..me {
                            khat[i * me + j] += ta * kt_[a * me + j];
                        }
                    }
                }
            }
            for i in 0..me {
                for j in 0..i {
                    let s = 0.5 * (khat[i * me + j] + khat[j * me + i]);
                    khat[i * me + j] = s;
                    khat[j * me + i] = s;
                }
            }
            w.rebuild_flops += 2.0 * (trip.len() * me) as f64 + 2.0 * (ndof * me * me) as f64;
            let s1: Vec<f64> = (0..me).map(|j| (0..nr).map(|li| mi[li] * (p[li].x * te[(3 * li) * me + j] + p[li].y * te[(3 * li + 1) * me + j])).sum()).collect();
            let wv: Vec<f64> = (0..me).map(|j| (0..nr).map(|li| ii[li] * te[(3 * li + 2) * me + j]).sum()).collect();
            let bndc: Vec<usize> = cells[..nbc].to_vec();
            regs.push(FReg { cells, loc, p, mi, ii, mass, it, xc, me, te, khat, s1, w: wv, internal: internal[r].clone(), bnd: bndc, off: n });
            n += 3 + me;
        }
        let q_ref: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
        let mut s = Sys { rep, regs, eval_facets, nd, n, fact_flops: 0.0, nnz_l: 0, detailed_cells, q_ref };
        // factorisation estimate of the Newton matrix pattern
        let z = vec![0.0; n];
        let sd = StepData { ah: 1.0, qhat: m.q.clone(), reg: s.regs.iter().map(|rg| RegStep { xh: V2::ZERO, a0: 0.0, b0: 0.0, a: vec![0.0; rg.me], b: vec![0.0; rg.me], th: 0.0, tth: vec![0.0; rg.me] }).collect(), lf: 0.0 };
        let mut wd = Work::default();
        let mut zz = z.clone();
        for rg in &s.regs {
            zz[rg.off..rg.off + 2].copy_from_slice(&[0.0, 0.0]);
        }
        let (_, trip, _) = s.assemble(m, &zz, &sd, true, Tangent::Consistent, &mut wd);
        let sp = Spd::analyze(&csc_from_triplets(n, trip));
        s.fact_flops = sp.stats.flops_factor;
        s.nnz_l = sp.stats.nnz_l;
        s
    }

    /// Configuration of every body from z (region interiors recovered).
    fn q_all(&self, z: &[f64]) -> Vec<[f64; 3]> {
        let mut q = self.q_ref.clone();
        for (i, r) in self.rep.iter().enumerate() {
            match r {
                Rep::Direct(idx) => {
                    for k in 0..3 {
                        if let Some(g) = idx[k] {
                            q[i][k] += z[g];
                        }
                    }
                }
                Rep::Bnd(ri) | Rep::Int(ri) => q[i] = self.cell_q(*ri, i, z),
            }
        }
        q
    }

    fn cell_q(&self, ri: usize, i: usize, z: &[f64]) -> [f64; 3] {
        let rg = &self.regs[ri];
        let y = &z[rg.off..rg.off + rg.ny()];
        let (r, wth) = rg.local(rg.loc[i], &y[3..]);
        let s = rot(r, y[2]);
        [rg.xc.x + y[0] + s.x, rg.xc.y + y[1] + s.y, y[2] + wth]
    }

    /// Configuration for facet evaluation (direct + region boundary cells only).
    fn q_eval(&self, z: &[f64]) -> Vec<[f64; 3]> {
        let mut q = self.q_ref.clone();
        for (i, r) in self.rep.iter().enumerate() {
            match r {
                Rep::Direct(idx) => {
                    for k in 0..3 {
                        if let Some(g) = idx[k] {
                            q[i][k] += z[g];
                        }
                    }
                }
                Rep::Bnd(ri) => q[i] = self.cell_q(*ri, i, z),
                Rep::Int(_) => {}
            }
        }
        q
    }

    /// d q_i / d y (3 × ny) of a region boundary cell and the body-frame vectors needed
    /// for the second derivatives.
    fn cell_jac(&self, ri: usize, i: usize, z: &[f64]) -> (Vec<[f64; 3]>, V2, Vec<V2>) {
        let rg = &self.regs[ri];
        let y = &z[rg.off..rg.off + rg.ny()];
        let li = rg.loc[i];
        let (r, _) = rg.local(li, &y[3..]);
        let phi = y[2];
        let s = rot(r, phi);
        let mut d = vec![[0.0; 3]; rg.ny()];
        d[0] = [1.0, 0.0, 0.0];
        d[1] = [0.0, 1.0, 0.0];
        let js = jperp(s);
        d[2] = [js.x, js.y, 1.0];
        let mut ru = Vec::with_capacity(rg.me);
        for j in 0..rg.me {
            let u = V2::new(rg.te[(3 * li) * rg.me + j], rg.te[(3 * li + 1) * rg.me + j]);
            let rr = rot(u, phi);
            d[3 + j] = [rr.x, rr.y, rg.te[(3 * li + 2) * rg.me + j]];
            ru.push(rr);
        }
        (d, s, ru)
    }

    fn step_data(&self, qhat: &[[f64; 3]], ah: f64, lf: f64, w: &mut Work) -> StepData {
        let mut reg = Vec::new();
        for rg in &self.regs {
            let me = rg.me;
            let (mut xh, mut a0, mut b0, mut th) = (V2::ZERO, 0.0, 0.0, 0.0);
            let (mut a, mut b, mut tth) = (vec![0.0; me], vec![0.0; me], vec![0.0; me]);
            for (li, &i) in rg.cells.iter().enumerate() {
                let x = V2::new(qhat[i][0], qhat[i][1]);
                let jx = jperp(x);
                let mm = rg.mi[li];
                xh = xh + x * mm;
                a0 += mm * x.dot(rg.p[li]);
                b0 += mm * jx.dot(rg.p[li]);
                th += rg.ii[li] * qhat[i][2];
                for j in 0..me {
                    let (ux, uy, uw) = (rg.te[(3 * li) * me + j], rg.te[(3 * li + 1) * me + j], rg.te[(3 * li + 2) * me + j]);
                    a[j] += mm * (x.x * ux + x.y * uy);
                    b[j] += mm * (jx.x * ux + jx.y * uy);
                    tth[j] += rg.ii[li] * qhat[i][2] * uw;
                }
            }
            w.step_flops += 12.0 * (rg.cells.len() * me) as f64;
            reg.push(RegStep { xh, a0, b0, a, b, th, tth });
        }
        StepData { ah, qhat: qhat.to_vec(), reg, lf }
    }

    fn assemble(&self, m: &Model, z: &[f64], sd: &StepData, want_h: bool, tang: Tangent, w: &mut Work) -> (Vec<f64>, Vec<(usize, usize, f64)>, f64) {
        let n = self.n;
        let q = self.q_eval(z);
        let mut g = vec![0.0; n];
        let mut trip = Vec::new();
        let mut energy = 0.0;
        // direct bodies: inertia, dead and pattern loads
        for (i, r) in self.rep.iter().enumerate() {
            let Rep::Direct(idx) = r else { continue };
            for k in 0..3 {
                let Some(gi) = idx[k] else { continue };
                let f = m.loads[i][k] + sd.lf * m.pattern[i][k];
                let u = q[i][k] - self.q_ref[i][k];
                g[gi] -= f;
                energy -= f * u;
                if sd.ah > 0.0 {
                    let wgt = sd.ah * if k < 2 { m.bodies[i].mass } else { m.bodies[i].inertia };
                    let d = q[i][k] - sd.qhat[i][k];
                    g[gi] += wgt * d;
                    energy += 0.5 * wgt * d * d;
                    if want_h {
                        trip.push((gi, gi, wgt));
                    }
                }
            }
        }
        // region boundary cells: pattern loads (gravity acts on c only, below)
        let mut jac: Vec<Option<(Vec<[f64; 3]>, V2, Vec<V2>)>> = vec![None; self.rep.len()];
        for (i, r) in self.rep.iter().enumerate() {
            if let Rep::Bnd(ri) = r {
                let jc = self.cell_jac(*ri, i, z);
                let off = self.regs[*ri].off;
                let f = [sd.lf * m.pattern[i][0], sd.lf * m.pattern[i][1], sd.lf * m.pattern[i][2]];
                if f.iter().any(|&v| v != 0.0) {
                    for (j, dj) in jc.0.iter().enumerate() {
                        g[off + j] -= dj[0] * f[0] + dj[1] * f[1] + dj[2] * f[2];
                    }
                    let u = [q[i][0] - self.q_ref[i][0], q[i][1] - self.q_ref[i][1], q[i][2]];
                    energy -= f[0] * u[0] + f[1] * u[1] + f[2] * u[2];
                    if want_h {
                        // second derivative of x_i along φ: −s (φφ) and J R U (φη)
                        let (s, ru) = (jc.1, &jc.2);
                        let fxy = V2::new(f[0], f[1]);
                        trip.push((off + 2, off + 2, fxy.dot(s)));
                        for j in 0..ru.len() {
                            let v = -fxy.dot(jperp(ru[j]));
                            trip.push((off + 2, off + 3 + j, v));
                            trip.push((off + 3 + j, off + 2, v));
                        }
                    }
                }
                jac[i] = Some(jc);
            }
        }
        // facets
        let mut dblk: Vec<[[f64; 3]; 3]> = vec![[[0.0; 3]; 3]; self.rep.len()];
        let mut dgeo: Vec<V2> = vec![V2::ZERO; self.rep.len()];
        let mut touched: Vec<usize> = Vec::new();
        for &fi in &self.eval_facets {
            let f = &m.facets[fi];
            let bodies = [f.a, f.b];
            for pt in 0..2 {
                let (j, e) = m.eval_point(&q, f, pt);
                energy += f.pts[pt].law.potential(j.dn, j.dt, f.pts[pt].kappa_c);
                let jn = [j.jna[0], j.jna[1], j.jna[2], j.jnb[0], j.jnb[1], j.jnb[2]];
                let jt = [j.jta[0], j.jta[1], j.jta[2], j.jtb[0], j.jtb[1], j.jtb[2]];
                let gl: Vec<f64> = (0..6).map(|r| jn[r] * e.fnorm + jt[r] * e.ftan).collect();
                for s in 0..2 {
                    let b = bodies[s];
                    match &self.rep[b] {
                        Rep::Direct(idx) => {
                            for k in 0..3 {
                                if let Some(gi) = idx[k] {
                                    g[gi] += gl[3 * s + k];
                                }
                            }
                        }
                        Rep::Bnd(ri) => {
                            let off = self.regs[*ri].off;
                            let (d, _, _) = jac[b].as_ref().unwrap();
                            for (jj, dj) in d.iter().enumerate() {
                                g[off + jj] += dj[0] * gl[3 * s] + dj[1] * gl[3 * s + 1] + dj[2] * gl[3 * s + 2];
                            }
                            dgeo[b] = dgeo[b] + V2::new(gl[3 * s], gl[3 * s + 1]);
                            w.coupling_flops += 6.0 * d.len() as f64;
                        }
                        Rep::Int(_) => unreachable!(),
                    }
                }
                if !want_h {
                    w.pts_force += 1;
                    continue;
                }
                w.pts_tangent += 1;
                let dd = match tang {
                    Tangent::Consistent => e.tan,
                    Tangent::Secant => {
                        let (kn, kt) = f.pts[pt].law.secant(j.dn, e.d);
                        [kn, 0.0, kt]
                    }
                };
                let mut h = [[0.0; 6]; 6];
                for r in 0..6 {
                    let ar = dd[0] * jn[r] + dd[1] * jt[r];
                    let br = dd[1] * jn[r] + dd[2] * jt[r];
                    for c in 0..6 {
                        h[r][c] = ar * jn[c] + br * jt[c];
                    }
                }
                for sa in 0..2 {
                    for sb in 0..2 {
                        let (ba, bb) = (bodies[sa], bodies[sb]);
                        let blk = |k: usize, l: usize| h[3 * sa + k][3 * sb + l];
                        match (&self.rep[ba], &self.rep[bb]) {
                            (Rep::Direct(ia), Rep::Direct(ib)) => {
                                for k in 0..3 {
                                    for l in 0..3 {
                                        if let (Some(r), Some(c)) = (ia[k], ib[l]) {
                                            let v = blk(k, l);
                                            if v != 0.0 {
                                                trip.push((r, c, v));
                                            }
                                        }
                                    }
                                }
                            }
                            (Rep::Bnd(_), Rep::Bnd(_)) if ba == bb => {
                                if dblk[ba] == [[0.0; 3]; 3] && !touched.contains(&ba) {
                                    touched.push(ba);
                                }
                                for k in 0..3 {
                                    for l in 0..3 {
                                        dblk[ba][k][l] += blk(k, l);
                                    }
                                }
                            }
                            (Rep::Bnd(ra), Rep::Direct(ib)) => {
                                let off = self.regs[*ra].off;
                                let (d, _, _) = jac[ba].as_ref().unwrap();
                                for l in 0..3 {
                                    let Some(c) = ib[l] else { continue };
                                    for (jj, dj) in d.iter().enumerate() {
                                        let v = dj[0] * blk(0, l) + dj[1] * blk(1, l) + dj[2] * blk(2, l);
                                        trip.push((off + jj, c, v));
                                    }
                                }
                                w.coupling_flops += 18.0 * d.len() as f64;
                            }
                            (Rep::Direct(ia), Rep::Bnd(rb)) => {
                                let off = self.regs[*rb].off;
                                let (d, _, _) = jac[bb].as_ref().unwrap();
                                for k in 0..3 {
                                    let Some(r) = ia[k] else { continue };
                                    for (jj, dj) in d.iter().enumerate() {
                                        let v = blk(k, 0) * dj[0] + blk(k, 1) * dj[1] + blk(k, 2) * dj[2];
                                        trip.push((r, off + jj, v));
                                    }
                                }
                                w.coupling_flops += 18.0 * d.len() as f64;
                            }
                            (Rep::Bnd(ra), Rep::Bnd(rb)) => {
                                let (oa, ob) = (self.regs[*ra].off, self.regs[*rb].off);
                                let (da, _, _) = jac[ba].as_ref().unwrap();
                                let (db, _, _) = jac[bb].as_ref().unwrap();
                                for (ii, di) in da.iter().enumerate() {
                                    for (jj, dj) in db.iter().enumerate() {
                                        let mut v = 0.0;
                                        for k in 0..3 {
                                            for l in 0..3 {
                                                v += di[k] * blk(k, l) * dj[l];
                                            }
                                        }
                                        trip.push((oa + ii, ob + jj, v));
                                    }
                                }
                                w.coupling_flops += 18.0 * (da.len() * db.len()) as f64;
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }
        if want_h {
            // projected diagonal blocks + geometric terms of the cells' rotation kinematics
            for (i, r) in self.rep.iter().enumerate() {
                let Rep::Bnd(ri) = r else { continue };
                let off = self.regs[*ri].off;
                let (d, s, ru) = jac[i].as_ref().unwrap();
                let blk = &dblk[i];
                let ny = d.len();
                if *blk != [[0.0; 3]; 3] {
                    let hd: Vec<[f64; 3]> = d.iter().map(|dj| std::array::from_fn(|k| blk[k][0] * dj[0] + blk[k][1] * dj[1] + blk[k][2] * dj[2])).collect();
                    for a in 0..ny {
                        for b in 0..ny {
                            let v = d[a][0] * hd[b][0] + d[a][1] * hd[b][1] + d[a][2] * hd[b][2];
                            if v != 0.0 {
                                trip.push((off + a, off + b, v));
                            }
                        }
                    }
                    w.coupling_flops += 18.0 * ny as f64 + 6.0 * (ny * ny) as f64;
                }
                let gxy = dgeo[i];
                if gxy != V2::ZERO {
                    // Σ g·∂²x/∂y²: φφ → −g·s ; φη_j → g·J R U_j
                    trip.push((off + 2, off + 2, -gxy.dot(*s)));
                    for j in 0..ru.len() {
                        let v = gxy.dot(jperp(ru[j]));
                        trip.push((off + 2, off + 3 + j, v));
                        trip.push((off + 3 + j, off + 2, v));
                    }
                }
            }
        }
        // regions: elastic K̂, gravity on c, inertia invariants
        for (ri, rg) in self.regs.iter().enumerate() {
            let me = rg.me;
            let o = rg.off;
            let y = &z[o..o + rg.ny()];
            let eta = &y[3..];
            for a in 0..me {
                let ka: f64 = (0..me).map(|b| rg.khat[a * me + b] * eta[b]).sum();
                g[o + 3 + a] += ka;
                energy += 0.5 * eta[a] * ka;
            }
            if want_h {
                for a in 0..me {
                    for b in 0..me {
                        let v = rg.khat[a * me + b];
                        if v != 0.0 {
                            trip.push((o + 3 + a, o + 3 + b, v));
                        }
                    }
                }
            }
            // gravity (Σ m_i U_i = 0, Σ m_i r_i = 0 → only c)
            let fg: V2 = rg.cells.iter().fold(V2::ZERO, |s, &i| s + V2::new(m.loads[i][0], m.loads[i][1]));
            g[o] -= fg.x;
            g[o + 1] -= fg.y;
            energy -= fg.x * y[0] + fg.y * y[1];
            if sd.ah > 0.0 {
                let rs = &sd.reg[ri];
                let ah = sd.ah;
                let c = V2::new(rg.xc.x + y[0], rg.xc.y + y[1]);
                let phi = y[2];
                let (cs, sn) = (phi.cos(), phi.sin());
                let ae = rs.a0 + (0..me).map(|j| rs.a[j] * eta[j]).sum::<f64>();
                let be = rs.b0 + (0..me).map(|j| rs.b[j] * eta[j]).sum::<f64>();
                let weta: f64 = (0..me).map(|j| rg.w[j] * eta[j]).sum();
                let s1eta: f64 = (0..me).map(|j| rg.s1[j] * eta[j]).sum();
                let tteta: f64 = (0..me).map(|j| rs.tth[j] * eta[j]).sum();
                let ee: f64 = eta.iter().map(|v| v * v).sum();
                energy += 0.5 * ah * (rg.mass * c.dot(c) - 2.0 * c.dot(rs.xh) - 2.0 * (cs * ae - sn * be) + 2.0 * s1eta + ee + rg.it * phi * phi + 2.0 * phi * weta - 2.0 * phi * rs.th - 2.0 * tteta);
                g[o] += ah * (rg.mass * c.x - rs.xh.x);
                g[o + 1] += ah * (rg.mass * c.y - rs.xh.y);
                g[o + 2] += ah * (sn * ae + cs * be + rg.it * phi + weta - rs.th);
                for j in 0..me {
                    g[o + 3 + j] += ah * (-cs * rs.a[j] + sn * rs.b[j] + rg.s1[j] + eta[j] + phi * rg.w[j] - rs.tth[j]);
                }
                if want_h {
                    trip.push((o, o, ah * rg.mass));
                    trip.push((o + 1, o + 1, ah * rg.mass));
                    trip.push((o + 2, o + 2, ah * (cs * ae - sn * be + rg.it)));
                    for j in 0..me {
                        let v = ah * (sn * rs.a[j] + cs * rs.b[j] + rg.w[j]);
                        trip.push((o + 2, o + 3 + j, v));
                        trip.push((o + 3 + j, o + 2, v));
                        trip.push((o + 3 + j, o + 3 + j, ah));
                    }
                }
            }
            w.region_flops += 2.0 * (me * me) as f64 * if want_h { 2.0 } else { 1.0 } + 20.0 * me as f64;
        }
        (g, trip, energy)
    }

    /// Reduced coordinates from per-body positions (M-orthogonal decomposition).
    fn fit(&self, q: &[[f64; 3]], w: &mut Work) -> Vec<f64> {
        let mut z = vec![0.0; self.n];
        for (i, r) in self.rep.iter().enumerate() {
            if let Rep::Direct(idx) = r {
                for k in 0..3 {
                    if let Some(g) = idx[k] {
                        z[g] = q[i][k] - self.q_ref[i][k];
                    }
                }
            }
        }
        for rg in &self.regs {
            let o = rg.off;
            let c = rg.cells.iter().zip(&rg.mi).fold(V2::ZERO, |s, (&i, &mm)| s + V2::new(q[i][0], q[i][1]) * mm) * (1.0 / rg.mass);
            // rotation: rigid-rotation component of the residual vanishes (mean axis)
            let mut phi = {
                let (mut sn, mut cs) = (0.0, 0.0);
                for (li, &i) in rg.cells.iter().enumerate() {
                    let d = V2::new(q[i][0], q[i][1]) - c;
                    cs += rg.mi[li] * rg.p[li].dot(d);
                    sn += rg.mi[li] * (rg.p[li].x * d.y - rg.p[li].y * d.x);
                }
                sn.atan2(cs)
            };
            let resid = |phi: f64| -> Vec<[f64; 3]> {
                rg.cells
                    .iter()
                    .enumerate()
                    .map(|(li, &i)| {
                        let d = rot(V2::new(q[i][0], q[i][1]) - c, -phi) - rg.p[li];
                        [d.x, d.y, q[i][2] - phi]
                    })
                    .collect()
            };
            for _ in 0..20 {
                let d = resid(phi);
                let (mut num, mut den) = (0.0, 0.0);
                for li in 0..rg.cells.len() {
                    num += rg.mi[li] * (rg.p[li].x * d[li][1] - rg.p[li].y * d[li][0]) + rg.ii[li] * d[li][2];
                    den += rg.mi[li] * rg.p[li].dot(rg.p[li]) + rg.ii[li];
                }
                let dphi = num / den;
                phi += dphi;
                if dphi.abs() < 1e-15 {
                    break;
                }
            }
            let d = resid(phi);
            z[o] = c.x - rg.xc.x;
            z[o + 1] = c.y - rg.xc.y;
            z[o + 2] = phi;
            for j in 0..rg.me {
                z[o + 3 + j] = (0..rg.cells.len())
                    .map(|li| rg.mi[li] * (d[li][0] * rg.te[(3 * li) * rg.me + j] + d[li][1] * rg.te[(3 * li + 1) * rg.me + j]) + rg.ii[li] * d[li][2] * rg.te[(3 * li + 2) * rg.me + j])
                    .sum();
            }
            w.step_flops += 8.0 * (rg.cells.len() * rg.me) as f64;
        }
        z
    }

    fn elastic_energy(&self, m: &Model, z: &[f64]) -> f64 {
        let q = self.q_eval(z);
        let mut u = 0.0;
        for &fi in &self.eval_facets {
            for pt in 0..2 {
                u += m.eval_point(&q, &m.facets[fi], pt).1.psi;
            }
        }
        for rg in &self.regs {
            let eta = &z[rg.off + 3..rg.off + rg.ny()];
            for a in 0..rg.me {
                for b in 0..rg.me {
                    u += 0.5 * eta[a] * rg.khat[a * rg.me + b] * eta[b];
                }
            }
        }
        u
    }

    /// Max χ over region-internal facet points (material facets), with the hot facets.
    fn monitor(&self, m: &Model, q: &[[f64; 3]], trig: f64, w: &mut Work) -> (f64, Vec<usize>) {
        let mut cmax: f64 = 0.0;
        let mut hot = Vec::new();
        for rg in &self.regs {
            for &fi in &rg.internal {
                let f = &m.facets[fi];
                if f.tag != 0 {
                    continue;
                }
                let mut cf: f64 = 0.0;
                for pt in 0..2 {
                    let p = &f.pts[pt];
                    if p.law.kind != LawKind::Damage {
                        continue;
                    }
                    let j = model::kin::jump(q[f.a], q[f.b], p.arm_a, p.arm_b, f.n0, f.t0, p.rest, f.frame_w);
                    let dn = j.dn.max(0.0);
                    cf = cf.max((dn * dn + p.law.gamma * j.dt * j.dt).sqrt() / p.law.delta0);
                    w.monitor_flops += C_PT_FORCE;
                }
                cmax = cmax.max(cf);
                if cf >= trig {
                    hot.push(fi);
                }
            }
        }
        (cmax, hot)
    }
}

// ---------------------------------------------------------------- Newton for one step

fn newton_step(sys: &Sys, m: &Model, z: &mut Vec<f64>, sd: &StepData, tol: f64, max_iter: usize, w: &mut Work) -> bool {
    let mut res0 = f64::NAN;
    let mut best = f64::INFINITY;
    let mut stall = 0;
    for it in 0..max_iter {
        let (g, trip, e0) = sys.assemble(m, z, sd, true, Tangent::Consistent, w);
        let res = g.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        if std::env::var("SB_DEBUG").is_ok() {
            let imax = (0..g.len()).max_by(|&a, &b| g[a].abs().partial_cmp(&g[b].abs()).unwrap()).unwrap_or(0);
            eprintln!("    it {it} res {res:.3e} at {imax} (nd {}) e {e0:.6e}", sys.nd);
        }
        if res <= tol {
            return true;
        }
        if it == 0 {
            res0 = res;
        }
        if res < 0.5 * best {
            best = res;
            stall = 0;
        } else {
            stall += 1;
            if stall >= 3 && best <= 1e-9 * res0 {
                return true;
            }
        }
        let rhs: Vec<f64> = g.iter().map(|v| -v).collect();
        let mat = csc_from_triplets(sys.n, trip);
        let mut p = rhs.clone();
        let mut ok = false;
        let mut spd = Spd::analyze(&mat);
        if spd.factor(&mat).is_ok() {
            spd.solve(&mut p);
            ok = p.iter().all(|v| v.is_finite());
        }
        if !ok {
            if let Ok(x) = lu_solve(&mat, &rhs) {
                p = x;
                ok = p.iter().all(|v| v.is_finite());
            }
        }
        w.newton_iters += 1;
        w.fact_flops += sys.fact_flops;
        w.solve_flops += 4.0 * sys.nnz_l as f64;
        let mut gp: f64 = g.iter().zip(&p).map(|(a, b)| a * b).sum();
        if !ok || gp >= 0.0 {
            let (_, ts, _) = sys.assemble(m, z, sd, true, Tangent::Secant, w);
            let mat = csc_from_triplets(sys.n, ts);
            p = rhs.clone();
            let mut spd = Spd::analyze(&mat);
            if spd.factor(&mat).is_ok() {
                spd.solve(&mut p);
            } else if let Ok(x) = lu_solve(&mat, &rhs) {
                p = x;
            } else {
                return false;
            }
            w.fact_flops += sys.fact_flops;
            w.solve_flops += 4.0 * sys.nnz_l as f64;
            gp = g.iter().zip(&p).map(|(a, b)| a * b).sum();
            if gp >= 0.0 {
                return false;
            }
        }
        let pmax = p.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        if pmax <= 1e-13 {
            // step at the round-off of absolute positions (~10 m): residual floor reached
            return true;
        }
        if gp.abs() <= 1e3 * f64::EPSILON * e0.abs() {
            for (a, b) in z.iter_mut().zip(&p) {
                *a += b;
            }
            continue;
        }
        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let zt: Vec<f64> = z.iter().zip(&p).map(|(a, b)| a + alpha * b).collect();
            let (gt, _, e1) = sys.assemble(m, &zt, sd, false, Tangent::Consistent, w);
            let rt = gt.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            if e1 <= e0 + 1e-4 * alpha * gp || rt < (1.0 - 1e-4 * alpha) * res {
                *z = zt;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            return false;
        }
    }
    let (g, _, _) = sys.assemble(m, z, sd, false, Tangent::Consistent, w);
    g.iter().fold(0.0f64, |a, &v| a.max(v.abs())) <= tol
}

// ---------------------------------------------------------------- diagnostics

#[derive(Clone, Default, Serialize)]
struct Fragment {
    separated: bool,
    mass: f64,
    cm: [f64; 2],
    v: [f64; 2],
    omega: f64,
    cells: usize,
}

fn fragment(sc: &Scene, m: &Model, q: &[[f64; 3]], v: &[[f64; 3]]) -> Fragment {
    let nc = m.n_cells();
    let mut parent: Vec<usize> = (0..nc).collect();
    fn find(p: &mut Vec<usize>, i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            p[r] = p[p[r]];
            r = p[r];
        }
        r
    }
    for f in m.facets.iter().filter(|f| f.active && f.tag == 0) {
        if f.pts.iter().all(|p| p.law.damage(p.kappa_c) < 0.99) {
            let (a, b) = (find(&mut parent, f.a), find(&mut parent, f.b));
            if a != b {
                parent[a] = b;
            }
        }
    }
    let root = find(&mut parent, sc.tip_cells[0]);
    let cells: Vec<usize> = (0..nc).filter(|&i| find(&mut parent, i) == root).collect();
    let separated = !sc.clamp_cells.iter().any(|&c| find(&mut parent, c) == root);
    let mass: f64 = cells.iter().map(|&i| m.bodies[i].mass).sum();
    let cm = cells.iter().fold(V2::ZERO, |s, &i| s + V2::new(q[i][0], q[i][1]) * m.bodies[i].mass) * (1.0 / mass);
    let vc = cells.iter().fold(V2::ZERO, |s, &i| s + V2::new(v[i][0], v[i][1]) * m.bodies[i].mass) * (1.0 / mass);
    let (mut l, mut j) = (0.0, 0.0);
    for &i in &cells {
        let r = V2::new(q[i][0], q[i][1]) - cm;
        let dv = V2::new(v[i][0], v[i][1]) - vc;
        l += m.bodies[i].mass * (r.x * dv.y - r.y * dv.x) + m.bodies[i].inertia * v[i][2];
        j += m.bodies[i].mass * r.dot(r) + m.bodies[i].inertia;
    }
    Fragment { separated, mass, cm: [cm.x, cm.y], v: [vc.x, vc.y], omega: l / j, cells: cells.len() }
}

/// Probe 1: tip deflection (cantilever) or impactor velocity (impact).
fn probe1(sc: &Scene, m: &Model, q: &[[f64; 3]], v: &[[f64; 3]]) -> f64 {
    match sc.impactor {
        Some(p) => v[p][1],
        None => tip_dy(sc, m, q),
    }
}

/// Probe 2: clamp root moment (cantilever) or impactor contact force (impact).
fn probe2(sc: &Scene, m: &Model, q: &[[f64; 3]]) -> f64 {
    match sc.impactor {
        Some(_) => {
            let mut f = 0.0;
            for fa in m.facets.iter().filter(|f| f.active && f.tag == 3) {
                for pt in 0..2 {
                    f -= m.eval_point(q, fa, pt).1.fnorm;
                }
            }
            f
        }
        None => root_moment(sc, m, q),
    }
}

/// Minimum opening over the impactor pad points (contact gap; < 0 = in contact).
fn pad_gap(m: &Model, q: &[[f64; 3]]) -> f64 {
    let mut g = f64::INFINITY;
    for fa in m.facets.iter().filter(|f| f.active && f.tag == 3) {
        for pt in 0..2 {
            g = g.min(m.eval_point(q, fa, pt).0.dn);
        }
    }
    g
}

/// All fragments (components through facets with d < 0.99) above `min_frac` of the
/// cell mass, sorted by centre-of-mass x.
fn fragments(m: &Model, q: &[[f64; 3]], v: &[[f64; 3]], min_frac: f64) -> Vec<Fragment> {
    let nc = m.n_cells();
    let mut parent: Vec<usize> = (0..nc).collect();
    fn find(p: &mut Vec<usize>, i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            p[r] = p[p[r]];
            r = p[r];
        }
        r
    }
    for f in m.facets.iter().filter(|f| f.active && f.tag == 0) {
        if f.pts.iter().all(|p| p.law.damage(p.kappa_c) < 0.99) {
            let (a, b) = (find(&mut parent, f.a), find(&mut parent, f.b));
            if a != b {
                parent[a] = b;
            }
        }
    }
    let total: f64 = (0..nc).map(|i| m.bodies[i].mass).sum();
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for i in 0..nc {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut out = Vec::new();
    for (_, cells) in groups {
        let mass: f64 = cells.iter().map(|&i| m.bodies[i].mass).sum();
        if mass < min_frac * total {
            continue;
        }
        let cm = cells.iter().fold(V2::ZERO, |s, &i| s + V2::new(q[i][0], q[i][1]) * m.bodies[i].mass) * (1.0 / mass);
        let vc = cells.iter().fold(V2::ZERO, |s, &i| s + V2::new(v[i][0], v[i][1]) * m.bodies[i].mass) * (1.0 / mass);
        let (mut l, mut j) = (0.0, 0.0);
        for &i in &cells {
            let r = V2::new(q[i][0], q[i][1]) - cm;
            let dv = V2::new(v[i][0], v[i][1]) - vc;
            l += m.bodies[i].mass * (r.x * dv.y - r.y * dv.x) + m.bodies[i].inertia * v[i][2];
            j += m.bodies[i].mass * r.dot(r) + m.bodies[i].inertia;
        }
        out.push(Fragment { separated: true, mass, cm: [cm.x, cm.y], v: [vc.x, vc.y], omega: l / j, cells: cells.len() });
    }
    out.sort_by(|a, b| a.cm[0].partial_cmp(&b.cm[0]).unwrap());
    out
}

fn tip_dy(sc: &Scene, m: &Model, q: &[[f64; 3]]) -> f64 {
    let (mut s, mut w) = (0.0, 0.0);
    for &i in &sc.tip_cells {
        s += m.bodies[i].mass * (q[i][1] - m.bodies[i].x0.y);
        w += m.bodies[i].mass;
    }
    s / w
}

/// Clamp reaction moment about the section centre (0, depth/2), from the tag-1 facets.
fn root_moment(sc: &Scene, m: &Model, q: &[[f64; 3]]) -> f64 {
    let (mut fx, mut m0) = (0.0, 0.0);
    for f in m.facets.iter().filter(|f| f.active && f.tag == 1) {
        for pt in 0..2 {
            let (j, e) = m.eval_point(q, f, pt);
            fx -= j.jnb[0] * e.fnorm + j.jtb[0] * e.ftan;
            m0 -= j.jnb[2] * e.fnorm + j.jtb[2] * e.ftan;
        }
    }
    m0 + 0.5 * sc.depth * fx
}

fn kinetic(m: &Model, v: &[[f64; 3]]) -> f64 {
    m.bodies.iter().zip(v).map(|(b, v)| 0.5 * b.mass * (v[0] * v[0] + v[1] * v[1]) + 0.5 * b.inertia * v[2] * v[2]).sum()
}

fn grav(m: &Model, q: &[[f64; 3]]) -> f64 {
    m.bodies.iter().enumerate().map(|(i, b)| -(m.loads[i][0] * (q[i][0] - b.x0.x) + m.loads[i][1] * (q[i][1] - b.x0.y))).sum()
}

#[derive(Clone, Default, Serialize)]
struct Traj {
    label: String,
    scene: String,
    h: f64,
    /// t, tip_dy, KE, U, Vg, D, W_ext, E_tot, detailed_cells, M_root
    #[serde(skip)]
    rows: Vec<[f64; 10]>,
    onset_t: Option<f64>,
    sep_t: Option<f64>,
    d_end: f64,
    crack: [f64; 6],
    fragment: Fragment,
    e0: f64,
    e_end: f64,
    e_max_rise: f64,
    work: Work,
    work_total: f64,
    wall_s: f64,
    steps: usize,
    rejected: usize,
    min_h: f64,
    expansions: Vec<[f64; 4]>,
    frags: Vec<Fragment>,
    v_impactor: Option<f64>,
    max_detailed: usize,
    cells: usize,
    chi_r_max: f64,
    n_unknowns_end: usize,
    failed_at: Option<f64>,
}

fn crack_stats(m: &Model) -> [f64; 6] {
    let (mut sw, mut sx, mut sy) = (0.0, 0.0, 0.0);
    let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for f in m.facets.iter().filter(|f| f.active && f.tag == 0) {
        let d: f64 = f.pts.iter().map(|p| p.law.dissipation(p.kappa_c)).sum();
        if d > 0.0 {
            sw += d;
            sx += d * f.pc.x;
            sy += d * f.pc.y;
            x0 = x0.min(f.pc.x);
            x1 = x1.max(f.pc.x);
            y0 = y0.min(f.pc.y);
            y1 = y1.max(f.pc.y);
        }
    }
    if sw == 0.0 {
        return [f64::NAN; 6];
    }
    [sx / sw, sy / sw, x0, x1, y0, y1]
}

// ---------------------------------------------------------------- explicit reference

/// Largest eigenvalue of M⁻¹K (linear stiffness at the reference state, all active
/// facets) by power iteration on M^{-1/2} K M^{-1/2}.
fn lambda_max(m: &Model) -> f64 {
    let nb = m.bodies.len();
    let q0: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
    let sq: Vec<[f64; 3]> = m.bodies.iter().map(|b| if b.mass > 0.0 { [b.mass.sqrt(), b.mass.sqrt(), b.inertia.sqrt()] } else { [0.0; 3] }).collect();
    let mut x: Vec<[f64; 3]> = (0..nb).map(|i| std::array::from_fn(|k| if sq[i][k] > 0.0 { ((i * 3 + k) as f64 * 0.618).sin() } else { 0.0 })).collect();
    let mut lam = 0.0;
    for _ in 0..300 {
        let u: Vec<[f64; 3]> = (0..nb).map(|i| std::array::from_fn(|k| if sq[i][k] > 0.0 { x[i][k] / sq[i][k] } else { 0.0 })).collect();
        let mut y = vec![[0.0f64; 3]; nb];
        for f in m.facets.iter().filter(|f| f.active) {
            for pt in 0..2 {
                let (j, _) = m.eval_point(&q0, f, pt);
                let (kn, kt) = (f.pts[pt].law.kn, f.pts[pt].law.kt);
                let dn: f64 = (0..3).map(|k| j.jna[k] * u[f.a][k] + j.jnb[k] * u[f.b][k]).sum();
                let dt: f64 = (0..3).map(|k| j.jta[k] * u[f.a][k] + j.jtb[k] * u[f.b][k]).sum();
                for k in 0..3 {
                    y[f.a][k] += j.jna[k] * kn * dn + j.jta[k] * kt * dt;
                    y[f.b][k] += j.jnb[k] * kn * dn + j.jtb[k] * kt * dt;
                }
            }
        }
        for i in 0..nb {
            for k in 0..3 {
                y[i][k] = if sq[i][k] > 0.0 { y[i][k] / sq[i][k] } else { 0.0 };
            }
        }
        let ny: f64 = y.iter().flat_map(|r| r.iter()).map(|v| v * v).sum::<f64>().sqrt();
        let nx: f64 = x.iter().flat_map(|r| r.iter()).map(|v| v * v).sum::<f64>().sqrt();
        lam = ny / nx;
        x = y.iter().map(|r| std::array::from_fn(|k| r[k] / ny)).collect();
    }
    lam
}

fn run_explicit(sc: &Scene, h: f64, t_end: f64, rec: f64) -> Traj {
    let t0 = Instant::now();
    let mut m = sc.m.clone();
    let nb = m.bodies.len();
    let mut q = m.q.clone();
    let mut vh = sc.v_init.clone();
    let minv: Vec<[f64; 3]> = m.bodies.iter().map(|b| if b.mass > 0.0 { [1.0 / b.mass, 1.0 / b.mass, 1.0 / b.inertia] } else { [0.0; 3] }).collect();
    let mut tr = Traj { label: "full_explicit".into(), scene: sc.name.clone(), h, cells: m.n_cells(), ..Default::default() };
    let nsteps = (t_end / h).round() as usize;
    let mut f = vec![[0.0f64; 3]; nb];
    let mut wext = 0.0;
    let mut next_rec = 0.0;
    let mut vn = vh.clone();
    let mut e_max: f64 = f64::NEG_INFINITY;
    for n in 0..=nsteps {
        let t = n as f64 * h;
        let lf = release(t, sc.t_r);
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
                tr.work.pts_force += 1;
            }
        }
        m.commit(&q);
        if tr.onset_t.is_none() && m.dissipated > 0.0 {
            tr.onset_t = Some(t);
        }
        let dt = if n == 0 { 0.5 * h } else { h };
        for i in 0..nb {
            for r in 0..3 {
                if !m.bodies[i].fixed[r] {
                    let vnew = vh[i][r] + dt * minv[i][r] * f[i][r];
                    vn[i][r] = if n == 0 { vh[i][r] } else { 0.5 * (vh[i][r] + vnew) };
                    vh[i][r] = vnew;
                }
            }
        }
        if t >= next_rec - 1e-12 {
            let u = m.elastic_energy(&q);
            let ke = kinetic(&m, &vn);
            let vg = grav(&m, &q);
            let e = ke + u + vg + m.dissipated - wext;
            if n == 0 {
                tr.e0 = e;
            }
            e_max = e_max.max(e);
            tr.rows.push([t, probe1(sc, &m, &q, &vn), ke, u, vg, m.dissipated, wext, e, 0.0, probe2(sc, &m, &q)]);
            if sc.impactor.is_none() && tr.sep_t.is_none() && m.dissipated > 0.0 && fragment(sc, &m, &q, &vn).separated {
                tr.sep_t = Some(t);
            }
            next_rec += rec;
        }
        if n == nsteps {
            if sc.impactor.is_none() {
                tr.fragment = fragment(sc, &m, &q, &vn);
            }
            tr.frags = fragments(&m, &q, &vn, 0.05);
            tr.v_impactor = sc.impactor.map(|p| vn[p][1]);
            break;
        }
        let lf1 = release(t + h, sc.t_r);
        for i in 0..nb {
            for r in 0..3 {
                if !m.bodies[i].fixed[r] {
                    let dq = h * vh[i][r];
                    wext += 0.5 * (lf + lf1) * m.pattern[i][r] * dq;
                    q[i][r] += dq;
                }
            }
        }
        tr.steps += 1;
    }
    tr.d_end = m.dissipated;
    tr.crack = crack_stats(&m);
    tr.e_end = tr.rows.last().unwrap()[7];
    tr.e_max_rise = e_max - tr.e0;
    tr.work_total = tr.work.total();
    tr.wall_s = t0.elapsed().as_secs_f64();
    tr.min_h = h;
    tr
}

// ---------------------------------------------------------------- implicit (full or reduced)

#[derive(Clone, Copy)]
enum Zone {
    Full,
    /// a priori detailed zone x < w
    Root(f64),
    /// a priori detailed zone |x − x_c| < w
    Mid(f64, f64),
    /// automatic: (χ_trig, r_e)
    Auto(f64, f64),
}

fn run_implicit(sc: &Scene, zone: Zone, h_max: f64, t_end: f64, rec: f64, nvib: usize, label: &str) -> Traj {
    let t0 = Instant::now();
    let mut m = sc.m.clone();
    let nb = m.bodies.len();
    let nc = m.n_cells();
    let mut w = Work::default();
    let mut detailed: Vec<bool> = (0..nc)
        .map(|i| match zone {
            Zone::Full => true,
            Zone::Root(x) => m.bodies[i].x0.x < x,
            Zone::Mid(xc, w) => (m.bodies[i].x0.x - xc).abs() < w,
            Zone::Auto(..) => false,
        })
        .collect();
    let mut sys = Sys::build(&m, &detailed, nvib, &mut w);
    let mg: f64 = m.bodies.iter().map(|b| b.mass).sum::<f64>() * G;
    let tol = 1e-9 * mg;
    // initial state: the model's own static equilibrium at the propped load
    let mut z = sys.fit(&m.q, &mut w);
    let sd0 = StepData { ah: 0.0, qhat: m.q.clone(), reg: Vec::new(), lf: 1.0 };
    let sd0 = StepData { reg: sys.regs.iter().map(|rg| RegStep { xh: V2::ZERO, a0: 0.0, b0: 0.0, a: vec![0.0; rg.me], b: vec![0.0; rg.me], th: 0.0, tth: vec![0.0; rg.me] }).collect(), ..sd0 };
    if sc.static_presolve {
        assert!(newton_step(&sys, &m, &mut z, &sd0, tol, 50, &mut w), "reduced static equilibrium");
    }
    let mut q = sys.q_all(&z);
    let mut qp = q.clone();
    let mut v = sc.v_init.clone();
    let mut vp = v.clone();
    let mut gap_prev = pad_gap(&m, &q);
    let mut tr = Traj { label: label.into(), scene: sc.name.clone(), h: h_max, cells: nc, min_h: h_max, ..Default::default() };
    let static_tip = tip_dy(sc, &m, &q);
    let mut t = 0.0;
    let mut h = h_max;
    let mut restart = true;
    let mut wext = 0.0;
    let mut next_rec = 0.0;
    let mut e_max: f64 = f64::NEG_INFINITY;
    let rec_row = |tr: &mut Traj, t: f64, m: &Model, q: &[[f64; 3]], v: &[[f64; 3]], u: f64, wext: f64, det: usize, e_max: &mut f64| {
        let ke = kinetic(m, v);
        let vg = grav(m, q);
        let e = ke + u + vg + m.dissipated - wext;
        if tr.rows.is_empty() {
            tr.e0 = e;
        }
        *e_max = e_max.max(e);
        tr.rows.push([t, probe1(sc, m, q, v), ke, u, vg, m.dissipated, wext, e, det as f64, probe2(sc, m, q)]);
    };
    let u0 = sys.elastic_energy(&m, &z);
    rec_row(&mut tr, 0.0, &m, &q, &v, u0, 0.0, sys.detailed_cells, &mut e_max);
    next_rec += rec;
    tr.max_detailed = sys.detailed_cells;
    let mut chi_max: f64 = 0.0;
    while t < t_end - 1e-12 {
        // contact-adaptive step (impact scene): h ≤ H_CONTACT while the impactor pad is in
        // contact or will close within one game step
        let mut active = false;
        if sc.impactor.is_some() {
            let gap = pad_gap(&m, &q);
            let closing = (gap_prev - gap).max(0.0) / h.max(1e-12);
            active = gap < 0.0 || gap - closing * h_max < 0.0 || v[sc.impactor.unwrap()][1] < 0.0 && gap < 1e-3;
            gap_prev = gap;
            if active && h > h_contact() {
                h = h_contact();
                restart = true;
            }
        }
        let hs = h.min(t_end - t);
        let (alpha, qhat): (f64, Vec<[f64; 3]>) = if restart {
            (1.0, (0..nb).map(|i| std::array::from_fn(|k| q[i][k] + hs * v[i][k])).collect())
        } else {
            (9.0 / 4.0, (0..nb).map(|i| std::array::from_fn(|k| (4.0 * q[i][k] - qp[i][k]) / 3.0 + (2.0 * hs / 9.0) * (4.0 * v[i][k] - vp[i][k]))).collect())
        };
        w.step_flops += 10.0 * nb as f64;
        let lf1 = release(t + hs, sc.t_r);
        let sd = sys.step_data(&qhat, alpha / (hs * hs), lf1, &mut w);
        let mut zt = z.clone();
        let conv = newton_step(&sys, &m, &mut zt, &sd, tol, 30, &mut w);
        let qt = sys.q_all(&zt);
        w.step_flops += sys.regs.iter().map(|rg| 6.0 * (rg.cells.len() * rg.me) as f64).sum::<f64>();
        let mut near_onset = false;
        let dd = if conv {
            let mut mx: f64 = 0.0;
            for &fi in &sys.eval_facets {
                let f = &m.facets[fi];
                if f.tag != 0 {
                    continue;
                }
                for pt in 0..2 {
                    let (_, e) = m.eval_point(&qt, f, pt);
                    mx = mx.max(e.d - f.pts[pt].law.damage(f.pts[pt].kappa_c));
                    let p = &f.pts[pt];
                    if p.law.kind == LawKind::Damage && p.kappa_c <= p.law.delta0 && e.kappa >= 0.95 * p.law.delta0 {
                        near_onset = true;
                    }
                }
                w.pts_force += 2;
            }
            mx
        } else {
            f64::INFINITY
        };
        if !conv || dd > 0.1 {
            h = 0.5 * hs;
            restart = true;
            tr.rejected += 1;
            tr.min_h = tr.min_h.min(h);
            if h < 1e-8 {
                tr.failed_at = Some(t);
                break;
            }
            continue;
        }
        // validity monitor / automatic zone
        let (chi, hot) = sys.monitor(&m, &qt, match zone {
            Zone::Auto(trig, _) => trig,
            _ => 1.0,
        }, &mut w);
        if let Zone::Auto(_, re) = zone {
            if !hot.is_empty() {
                let mut added = 0;
                for i in 0..nc {
                    if !detailed[i] && hot.iter().any(|&fi| (m.facets[fi].pc - m.bodies[i].x0).norm() < re) {
                        detailed[i] = true;
                        added += 1;
                    }
                }
                let u_before = sys.elastic_energy(&m, &z);
                sys = Sys::build(&m, &detailed, nvib, &mut w);
                z = sys.fit(&q, &mut w);
                let q_fit = sys.q_all(&z);
                let mis = q_fit.iter().zip(&q).map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()).fold(0.0, f64::max);
                let u_after = sys.elastic_energy(&m, &z);
                tr.expansions.push([t, added as f64, sys.detailed_cells as f64, (u_after - u_before) / u_before.abs().max(1e-300) + 0.0 * mis]);
                tr.max_detailed = tr.max_detailed.max(sys.detailed_cells);
                eprintln!("  {label} t={t:.5} expand +{added} → {} detailed, regions {}, position misfit {mis:.2e} m, ΔU/U {:.2e}", sys.detailed_cells, sys.regs.len(), (u_after - u_before) / u_before.abs().max(1e-300));
                continue;
            }
        }
        chi_max = chi_max.max(chi);
        // accept
        let vt: Vec<[f64; 3]> = if restart {
            (0..nb).map(|i| std::array::from_fn(|k| (qt[i][k] - q[i][k]) / hs)).collect()
        } else {
            (0..nb).map(|i| std::array::from_fn(|k| (3.0 * qt[i][k] - 4.0 * q[i][k] + qp[i][k]) / (2.0 * hs))).collect()
        };
        let lf0 = release(t, sc.t_r);
        for i in 0..nb {
            for k in 0..3 {
                wext += 0.5 * (lf0 + lf1) * m.pattern[i][k] * (qt[i][k] - q[i][k]);
            }
        }
        // commit on evaluated facets
        for &fi in &sys.eval_facets {
            for pt in 0..2 {
                let (_, e) = m.eval_point(&qt, &m.facets[fi], pt);
                let p = &mut m.facets[fi].pts[pt];
                if e.kappa > p.kappa_c {
                    m.dissipated += p.law.dissipation(e.kappa) - p.law.dissipation(p.kappa_c);
                    p.kappa_c = e.kappa;
                }
            }
        }
        if tr.onset_t.is_none() && m.dissipated > 0.0 {
            tr.onset_t = Some(t + hs);
        }
        qp = q;
        q = qt;
        vp = v;
        v = vt;
        z = zt;
        t += hs;
        tr.steps += 1;
        let hfrac = h_frac();
        let frac_now = hfrac.is_some() && (dd > 1e-4 || near_onset);
        if frac_now && h > hfrac.unwrap() {
            h = hfrac.unwrap();
            restart = true;
        }
        let mut cap = if active { h_contact().min(h_max) } else { h_max };
        if frac_now {
            cap = cap.min(hfrac.unwrap());
        }
        if !restart && h < cap && dd < 0.025 {
            h = (2.0 * h).min(cap);
            restart = true;
        } else {
            restart = false;
        }
        if t >= next_rec - 1e-12 || t >= t_end - 1e-12 {
            let u = sys.elastic_energy(&m, &z);
            rec_row(&mut tr, t, &m, &q, &v, u, wext, sys.detailed_cells, &mut e_max);
            if sc.impactor.is_none() && tr.sep_t.is_none() && m.dissipated > 0.0 && fragment(sc, &m, &q, &v).separated {
                tr.sep_t = Some(t);
            }
            while next_rec <= t + 1e-12 {
                next_rec += rec;
            }
        }
    }
    if sc.impactor.is_none() {
        tr.fragment = fragment(sc, &m, &q, &v);
    }
    tr.frags = fragments(&m, &q, &v, 0.05);
    tr.v_impactor = sc.impactor.map(|p| v[p][1]);
    tr.d_end = m.dissipated;
    tr.crack = crack_stats(&m);
    tr.e_end = tr.rows.last().unwrap()[7];
    tr.e_max_rise = e_max - tr.e0;
    tr.chi_r_max = chi_max;
    tr.n_unknowns_end = sys.n;
    tr.work_total = w.total();
    tr.work = w;
    tr.wall_s = t0.elapsed().as_secs_f64();
    let _ = static_tip;
    tr
}

// ---------------------------------------------------------------- comparison

fn first_peak(tr: &Traj, t_from: f64) -> (f64, f64) {
    // first local minimum of tip_dy (largest downward deflection) after t_from
    let r = &tr.rows;
    for k in 1..r.len() - 1 {
        if r[k][0] > t_from && r[k][1] <= r[k - 1][1] && r[k][1] < r[k + 1][1] {
            return (r[k][0], r[k][1]);
        }
    }
    (f64::NAN, f64::NAN)
}

fn mean_tip(tr: &Traj, t_from: f64) -> f64 {
    let v: Vec<f64> = tr.rows.iter().filter(|r| r[0] >= t_from).map(|r| r[1]).collect();
    v.iter().sum::<f64>() / v.len() as f64
}

fn period(tr: &Traj, t_from: f64) -> f64 {
    // mean spacing of downward crossings of the mean
    let mu = mean_tip(tr, t_from);
    let r: Vec<&[f64; 10]> = tr.rows.iter().filter(|r| r[0] >= t_from).collect();
    let mut xs = Vec::new();
    for k in 1..r.len() {
        if r[k - 1][1] > mu && r[k][1] <= mu {
            let a = (r[k - 1][1] - mu) / (r[k - 1][1] - r[k][1]);
            xs.push(r[k - 1][0] + a * (r[k][0] - r[k - 1][0]));
        }
    }
    if xs.len() < 2 {
        return f64::NAN;
    }
    (xs[xs.len() - 1] - xs[0]) / (xs.len() - 1) as f64
}

fn amp_ratio(tr: &Traj, t_from: f64) -> f64 {
    // amplitude of the last full period over the first (decay over the run)
    let mu = mean_tip(tr, t_from);
    let r: Vec<&[f64; 10]> = tr.rows.iter().filter(|r| r[0] >= t_from).collect();
    let n = r.len();
    let p = period(tr, t_from);
    if !p.is_finite() {
        return f64::NAN;
    }
    let win = |a: f64, b: f64| r.iter().filter(|x| x[0] >= a && x[0] < b).map(|x| (x[1] - mu).abs()).fold(0.0, f64::max);
    let (t0, t1) = (r[0][0], r[n - 1][0]);
    win(t1 - p, t1 + 1.0) / win(t0, t0 + p)
}

fn compare_elastic(rf: &Traj, c: &Traj, t_r: f64) -> Value {
    let rel = |a: f64, b: f64| (a - b) / b;
    let (tp_r, yp_r) = first_peak(rf, t_r);
    let (tp_c, yp_c) = first_peak(c, t_r);
    let (pr, pc) = (period(rf, t_r), period(c, t_r));
    let (mr, mc) = (mean_tip(rf, t_r), mean_tip(c, t_r));
    let (s_r, s_c) = (rf.rows[0][1], c.rows[0][1]);
    let pass = json!({
        "first_peak": rel(yp_c - s_c, yp_r - s_r).abs() <= 0.10,
        "period": rel(pc, pr).abs() <= 0.10,
        "mean": rel(mc - s_c, mr - s_r).abs() <= 0.10,
        "no_energy_creation": c.e_max_rise <= 1e-3 * (rf.rows.iter().map(|r| r[2]).fold(0.0, f64::max)),
    });
    json!({
        "label": c.label, "h": c.h,
        "first_peak_t": [tp_r, tp_c], "first_peak_drop_rel": rel(yp_c - s_c, yp_r - s_r), "static_tip": [s_r, s_c],
        "period_s": [pr, pc], "period_rel": rel(pc, pr),
        "root_moment_static_propped": [rf.rows[0][9], c.rows[0][9]],
        "root_moment_peak": [rf.rows.iter().map(|r| r[9].abs()).fold(0.0, f64::max), c.rows.iter().map(|r| r[9].abs()).fold(0.0, f64::max)],
        "root_moment_mean_after_release": [rf.rows.iter().filter(|r| r[0] > t_r).map(|r| r[9]).sum::<f64>() / rf.rows.iter().filter(|r| r[0] > t_r).count() as f64, c.rows.iter().filter(|r| r[0] > t_r).map(|r| r[9]).sum::<f64>() / c.rows.iter().filter(|r| r[0] > t_r).count() as f64],
        "mean_drop_rel": rel(mc - s_c, mr - s_r),
        "amp_ratio_last_over_first": [amp_ratio(rf, t_r), amp_ratio(c, t_r)],
        "energy_loss_over_KEmax": [(rf.e0 - rf.e_end) / rf.rows.iter().map(|r| r[2]).fold(0.0, f64::max), (c.e0 - c.e_end) / c.rows.iter().map(|r| r[2]).fold(0.0, f64::max)],
        "e_max_rise": [rf.e_max_rise, c.e_max_rise],
        "work_ratio_vs_ref": rf.work_total / c.work_total, "wall_ratio_vs_ref": rf.wall_s / c.wall_s,
        "criteria": pass,
    })
}

fn compare_fracture(rf: &Traj, c: &Traj, frame: f64, ke_scale: f64) -> Value {
    let rel = |a: f64, b: f64| (a - b) / b;
    let fr = &rf.fragment;
    let fc = &c.fragment;
    let vmag = (fr.v[0].powi(2) + fr.v[1].powi(2)).sqrt();
    let dv = ((fc.v[0] - fr.v[0]).powi(2) + (fc.v[1] - fr.v[1]).powi(2)).sqrt();
    let cm0 = rf.rows[0][1];
    let _ = cm0;
    let ke_r = rf.rows.last().unwrap()[2];
    let ke_c = c.rows.last().unwrap()[2];
    let dt_on = match (rf.onset_t, c.onset_t) {
        (Some(a), Some(b)) => b - a,
        _ => f64::NAN,
    };
    let dt_sep = match (rf.sep_t, c.sep_t) {
        (Some(a), Some(b)) => b - a,
        _ => f64::NAN,
    };
    let pass = json!({
        "fracture_and_separation": fr.separated == fc.separated,
        "event_times_within_frame": dt_on.abs() <= frame && (dt_sep.abs() <= frame || (rf.sep_t.is_none() && c.sep_t.is_none())),
        "crack_location": (c.crack[0] - rf.crack[0]).abs() <= 0.1,
        "fracture_energy": rel(c.d_end, rf.d_end).abs() <= 0.10,
        "fragment_velocity": dv <= 0.10 * vmag,
        "fragment_omega": (fc.omega - fr.omega).abs() <= 0.10 * fr.omega.abs(),
        "fragment_KE": rel(ke_c, ke_r).abs() <= 0.10,
        "no_energy_creation": c.e_max_rise <= 1e-3 * ke_scale,
    });
    json!({
        "label": c.label, "h": c.h,
        "onset_t": [rf.onset_t, c.onset_t], "sep_t": [rf.sep_t, c.sep_t],
        "crack_xy_mean": [[rf.crack[0], rf.crack[1]], [c.crack[0], c.crack[1]]],
        "crack_box": [rf.crack, c.crack],
        "D": [rf.d_end, c.d_end], "D_rel": rel(c.d_end, rf.d_end),
        "fragment": [fr, fc], "fragment_dv_over_v": dv / vmag, "omega_rel": rel(fc.omega, fr.omega),
        "KE_end": [ke_r, ke_c],
        "numerical_dissipation_end": [rf.e0 - rf.e_end, c.e0 - c.e_end],
        "numerical_dissipation_over_KEend": (c.e0 - c.e_end) / ke_r,
        "e_max_rise": [rf.e_max_rise, c.e_max_rise],
        "detailed_cells_max": c.max_detailed, "cells": c.cells, "expansions": c.expansions.len(),
        "chi_r_max": c.chi_r_max,
        "steps": c.steps, "rejected": c.rejected, "min_h": c.min_h,
        "work_ratio_vs_ref": rf.work_total / c.work_total, "wall_ratio_vs_ref": rf.wall_s / c.wall_s,
        "criteria": pass,
        "all_pass": pass.as_object().unwrap().values().all(|v| v.as_bool() == Some(true)),
    })
}

// ---------------------------------------------------------------- Exp 2 threshold + consequences

#[derive(Clone, Copy)]
enum Method {
    Explicit,
    Implicit(Zone, f64),
}

fn methods() -> Vec<(&'static str, Method)> {
    if h_frac().is_some() {
        return vec![
            ("full_explicit", Method::Explicit),
            ("full_bdf2_h240_frac", Method::Implicit(Zone::Full, 1.0 / 240.0)),
            ("red_root0.4_h240_frac", Method::Implicit(Zone::Root(0.4), 1.0 / 240.0)),
            ("auto_chi0.8_h240_frac", Method::Implicit(Zone::Auto(0.8, 0.15), 1.0 / 240.0)),
            ("auto_chi1.0_h240_frac", Method::Implicit(Zone::Auto(1.0, 0.15), 1.0 / 240.0)),
        ];
    }
    vec![
        ("full_explicit", Method::Explicit),
        ("full_bdf2_h240", Method::Implicit(Zone::Full, 1.0 / 240.0)),
        ("red_root0.4_h240", Method::Implicit(Zone::Root(0.4), 1.0 / 240.0)),
        ("red_root0.4_h480", Method::Implicit(Zone::Root(0.4), 1.0 / 480.0)),
        ("auto_chi0.8_h240", Method::Implicit(Zone::Auto(0.8, 0.15), 1.0 / 240.0)),
        ("auto_chi0.8_h480", Method::Implicit(Zone::Auto(0.8, 0.15), 1.0 / 480.0)),
    ]
}

fn run_method(sc: &Scene, meth: Method, label: &str, t_end: f64, rec: f64, nvib: usize) -> Traj {
    match meth {
        Method::Explicit => {
            let lam = lambda_max(&sc.m);
            let h = (0.3 * sc.dmin / sc.m.mat.wave_speed()).min(0.8 * 2.0 / lam.sqrt());
            run_explicit(sc, h, t_end, rec)
        }
        Method::Implicit(zone, h) => run_implicit(sc, zone, h, t_end, rec, nvib, label),
    }
}

fn exp2(seeds: &[u64], dmin: f64, t_end: f64, t_r: f64, rec: f64, nvib: usize) {
    let meths = methods();
    let mut out = Vec::new();
    for &seed in seeds {
        // 1. thresholds by bisection, one thread per method
        let thr = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for (k, (label, meth)) in meths.iter().enumerate() {
                let thr = &thr;
                s.spawn(move || {
                    let sep = |g: f64| -> (bool, f64) {
                        let sc = cantilever(seed, dmin, true, t_r, g);
                        let tr = run_method(&sc, *meth, label, t_end, rec, nvib);
                        (tr.fragment.separated, tr.work_total)
                    };
                    let (mut lo, mut hi) = (0.6, 1.6);
                    let mut log = Vec::new();
                    let (slo, w0) = sep(lo);
                    let (shi, w1) = sep(hi);
                    log.push(json!([lo, slo, w0]));
                    log.push(json!([hi, shi, w1]));
                    let crit = if slo || !shi {
                        None
                    } else {
                        for _ in 0..6 {
                            let mid = 0.5 * (lo + hi);
                            let (sm, wm) = sep(mid);
                            log.push(json!([mid, sm, wm]));
                            if sm {
                                hi = mid;
                            } else {
                                lo = mid;
                            }
                        }
                        Some(0.5 * (lo + hi))
                    };
                    eprintln!("seed {seed} {label}: lambda_crit = {crit:?} bracket [{lo:.4}, {hi:.4}]");
                    thr.lock().unwrap().push((k, json!({"label": label, "lambda_crit": crit, "bracket": [lo, hi], "runs": log})));
                });
            }
        });
        let mut thr = thr.into_inner().unwrap();
        thr.sort_by_key(|t| t.0);
        let thr: Vec<Value> = thr.into_iter().map(|t| t.1).collect();
        let lref = thr[0]["lambda_crit"].as_f64().expect("reference threshold");
        let tcmp: Vec<Value> = thr[1..]
            .iter()
            .map(|t| {
                let l = t["lambda_crit"].as_f64();
                json!({"label": t["label"], "lambda_crit": l, "rel": l.map(|l| (l - lref) / lref), "pass": l.map(|l| ((l - lref) / lref).abs() <= 0.10).unwrap_or(false)})
            })
            .collect();
        // 2. consequences at 1.25 λ_crit,ref
        let gc = 1.25 * lref;
        let sc = cantilever(seed, dmin, true, t_r, gc);
        let results = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for (k, (label, meth)) in meths.iter().enumerate() {
                let (sc, results) = (&sc, &results);
                s.spawn(move || {
                    let tr = run_method(sc, *meth, label, t_end, rec, nvib);
                    eprintln!(
                        "  {} {}: D={:.3} onset={:?} sep={:?} frag v=({:.3},{:.3}) ω={:.3} E0-Eend={:.3e} rise={:.2e} steps={} rej={} det={} work={:.3e} wall={:.1}s",
                        sc.name, tr.label, tr.d_end, tr.onset_t, tr.sep_t, tr.fragment.v[0], tr.fragment.v[1], tr.fragment.omega, tr.e0 - tr.e_end, tr.e_max_rise, tr.steps, tr.rejected, tr.max_detailed, tr.work_total, tr.wall_s
                    );
                    results.lock().unwrap().push((k, tr));
                });
            }
        });
        let mut res = results.into_inner().unwrap();
        res.sort_by_key(|r| r.0);
        let trs: Vec<Traj> = res.into_iter().map(|r| r.1).collect();
        let ke_scale = trs[0].rows.iter().map(|r| r[2]).fold(0.0, f64::max);
        let mut csv = Csv::new(EXP, &format!("traj_{}", sc.name), &["run", "t", "tip_dy", "KE", "U", "Vg", "D", "W_ext", "E_tot", "detailed_cells", "M_root"]);
        for (k, tr) in trs.iter().enumerate() {
            for r in &tr.rows {
                let mut row = vec![k as f64];
                row.extend_from_slice(r);
                csv.row(&row);
            }
        }
        let cmp: Vec<Value> = trs[1..].iter().map(|c| compare_fracture(&trs[0], c, 1.0 / 60.0, ke_scale)).collect();
        for c in &cmp {
            eprintln!("{} {}", sc.name, c);
        }
        out.push(json!({"seed": seed, "thresholds": thr, "threshold_comparison": tcmp, "consequence_gscale": gc, "consequence_scene": sc.name, "runs": trs, "run_labels_in_csv": trs.iter().map(|t| t.label.clone()).collect::<Vec<_>>(), "comparison": cmp}));
    }
    let res = json!({
        "meta": meta(EXP, "exp2", None, "threshold by bisection on the gravity scale λ ∈ [0.6, 1.6] (6 steps, separation by t_end), then consequences at 1.25 λ_crit,ref; reference leapfrog, candidates BDF2 with event control",
            json!({"scene": "propped cantilever 4 x 0.4 m, b = 0.3, d_min 0.04, SB-IL1, clamp/prop elastic, prop force released over t_r", "t_r": t_r, "t_end": t_end, "nvib": nvib, "auto": "chi_trig 0.8, r_e 0.15 m, reject-and-redo", "prereg": "docs/phase0c/reduced-fracture.md §4"})),
        "seeds": out,
    });
    write_json(EXP, if h_frac().is_some() { "summary_exp2_posthoc_hfrac" } else { "summary_exp2" }, &res);
}

// ---------------------------------------------------------------- Exp 3 impact

const MP: f64 = 50.0;
const T_END_IMP: f64 = 0.05;

fn dmin_imp() -> f64 {
    std::env::var("SB_DMIN_IMP").ok().and_then(|v| v.parse().ok()).unwrap_or(0.025)
}

fn imp_methods() -> Vec<(&'static str, Method)> {
    vec![
        ("full_explicit", Method::Explicit),
        ("full_bdf2_h240", Method::Implicit(Zone::Full, 1.0 / 240.0)),
        ("red_mid0.2_h240", Method::Implicit(Zone::Mid(1.0, 0.2), 1.0 / 240.0)),
        ("auto_chi0.8_h240", Method::Implicit(Zone::Auto(0.8, 0.075), 1.0 / 240.0)),
    ]
}

fn compare_impact(rf: &Traj, c: &Traj, v0: f64, ke0: f64) -> Value {
    let (vr, vc) = (rf.v_impactor.unwrap(), c.v_impactor.unwrap());
    let (jr, jc) = (MP * (vr + v0), MP * (vc + v0));
    let nf_ok = rf.frags.len() == c.frags.len();
    let frag_ok = nf_ok
        && rf.frags.iter().zip(&c.frags).all(|(a, b)| ((b.mass - a.mass) / a.mass).abs() <= 0.10 && ((b.v[0] - a.v[0]).powi(2) + (b.v[1] - a.v[1]).powi(2)).sqrt() <= 0.10 * v0);
    let pass = json!({
        "impulse": (jc - jr).abs() <= 0.10 * jr.abs(),
        "rebound": (vc - vr).abs() <= 0.10 * v0,
        "fragments": frag_ok,
        "crack_location": rf.crack[0].is_nan() && c.crack[0].is_nan() || (c.crack[0] - rf.crack[0]).abs() <= 0.1,
        "fracture_energy": (rf.d_end == 0.0 && c.d_end == 0.0) || ((c.d_end - rf.d_end) / rf.d_end).abs() <= 0.10,
        "no_energy_creation": c.e_max_rise <= 1e-3 * ke0,
    });
    json!({
        "label": c.label, "v0": v0,
        "v_impactor_end": [vr, vc], "impulse": [jr, jc],
        "fragments": [&rf.frags, &c.frags],
        "D": [rf.d_end, c.d_end], "crack_xy": [[rf.crack[0], rf.crack[1]], [c.crack[0], c.crack[1]]],
        "numerical_dissipation_over_KE0": [(rf.e0 - rf.e_end) / ke0, (c.e0 - c.e_end) / ke0],
        "e_max_rise_over_KE0": [rf.e_max_rise / ke0, c.e_max_rise / ke0],
        "detailed_cells_max": c.max_detailed, "cells": c.cells, "expansions": c.expansions.len(),
        "steps": c.steps, "rejected": c.rejected, "min_h": c.min_h,
        "work_ratio_vs_ref": rf.work_total / c.work_total, "wall_ratio_vs_ref": rf.wall_s / c.wall_s,
        "criteria": pass,
        "all_pass": pass.as_object().unwrap().values().all(|v| v.as_bool() == Some(true)),
    })
}

fn exp3(seeds: &[u64], dmin: f64, rec: f64) {
    let meths = imp_methods();
    let mut out = Vec::new();
    for &seed in seeds {
        let thr = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for (k, (label, meth)) in meths.iter().enumerate() {
                let thr = &thr;
                s.spawn(move || {
                    let broken = |v0: f64| -> (bool, usize) {
                        let sc = beam_impact(seed, dmin, v0, MP);
                        let tr = run_method(&sc, *meth, label, T_END_IMP, rec, 4);
                        (tr.frags.len() > 1, tr.frags.len())
                    };
                    let (mut lo, mut hi) = (0.05f64, 10.0f64);
                    let mut log = Vec::new();
                    let (blo, nlo) = broken(lo);
                    let (bhi, nhi) = broken(hi);
                    log.push(json!([lo, blo, nlo]));
                    log.push(json!([hi, bhi, nhi]));
                    let crit = if blo || !bhi {
                        None
                    } else {
                        // bisection in log v (7 steps: ±2.1 % in v)
                        for _ in 0..7 {
                            let mid = (lo * hi).sqrt();
                            let (bm, nm) = broken(mid);
                            log.push(json!([mid, bm, nm]));
                            if bm {
                                hi = mid;
                            } else {
                                lo = mid;
                            }
                        }
                        Some((lo * hi).sqrt())
                    };
                    eprintln!("seed {seed} {label}: v_crit = {crit:?} bracket [{lo:.3}, {hi:.3}]");
                    thr.lock().unwrap().push((k, json!({"label": label, "v_crit": crit, "bracket": [lo, hi], "runs": log})));
                });
            }
        });
        let mut thr = thr.into_inner().unwrap();
        thr.sort_by_key(|t| t.0);
        let thr: Vec<Value> = thr.into_iter().map(|t| t.1).collect();
        let vref = thr[0]["v_crit"].as_f64().expect("reference threshold");
        let tcmp: Vec<Value> = thr[1..]
            .iter()
            .map(|t| {
                let v = t["v_crit"].as_f64();
                json!({"label": t["label"], "v_crit": v, "rel": v.map(|v| (v - vref) / vref), "pass": v.map(|v| ((v - vref) / vref).abs() <= 0.10).unwrap_or(false)})
            })
            .collect();
        let mut cons = Vec::new();
        for fac in [0.75, 1.25, 2.0] {
            let v0 = fac * vref;
            let sc = beam_impact(seed, dmin, v0, MP);
            let ke0 = 0.5 * MP * v0 * v0;
            let results = std::sync::Mutex::new(Vec::new());
            std::thread::scope(|s| {
                for (k, (label, meth)) in meths.iter().enumerate() {
                    let (sc, results) = (&sc, &results);
                    s.spawn(move || {
                        let tr = run_method(sc, *meth, label, T_END_IMP, rec, 4);
                        eprintln!("  {} {}: v_out {:?} frags {} D {:.3} E0-E {:.3e} rise {:.2e} steps {} rej {} det {} work {:.3e} wall {:.1}s", sc.name, tr.label, tr.v_impactor, tr.frags.len(), tr.d_end, tr.e0 - tr.e_end, tr.e_max_rise, tr.steps, tr.rejected, tr.max_detailed, tr.work_total, tr.wall_s);
                        results.lock().unwrap().push((k, tr));
                    });
                }
            });
            let mut res = results.into_inner().unwrap();
            res.sort_by_key(|r| r.0);
            let trs: Vec<Traj> = res.into_iter().map(|r| r.1).collect();
            let mut csv = Csv::new(EXP, &format!("traj_{}", sc.name), &["run", "t", "v_impactor", "KE", "U", "Vg", "D", "W_ext", "E_tot", "detailed_cells", "F_contact"]);
            for (k, tr) in trs.iter().enumerate() {
                for r in &tr.rows {
                    let mut row = vec![k as f64];
                    row.extend_from_slice(r);
                    csv.row(&row);
                }
            }
            let cmp: Vec<Value> = trs[1..].iter().map(|c| compare_impact(&trs[0], c, v0, ke0)).collect();
            for c in &cmp {
                eprintln!("{} {}", sc.name, c);
            }
            cons.push(json!({"factor": fac, "v0": v0, "scene": sc.name, "runs": trs, "comparison": cmp}));
        }
        out.push(json!({"seed": seed, "thresholds": thr, "threshold_comparison": tcmp, "consequences": cons}));
    }
    let res = json!({
        "meta": meta(EXP, "exp3", None, "threshold by log-bisection on v0 ∈ [0.05, 10] m/s (7 steps, broken = more than one fragment with ≥ 5 % mass at t_end), consequences at 0.75 / 1.25 / 2.0 v_crit,ref; implicit runs: BDF2 with step ≤ 1e-4 s while the impactor pad is in or closing to contact, game step otherwise, Δd ≤ 0.1",
            json!({"scene": "simply supported beam 2.0 x 0.2 m, b 0.2, span 1.8, unilateral support pads 0.1 m, rigid flat impactor pad 0.1 m at mid-span, mass 50 kg, y only", "dmin": dmin, "t_end": T_END_IMP, "h_contact": h_contact(), "auto": "chi_trig 0.8, r_e 0.075 m", "prereg": "docs/phase0c/reduced-fracture.md §5"})),
        "seeds": out,
    });
    write_json("phase0c/exp3", "summary_exp3", &res);
}

// ---------------------------------------------------------------- checks

fn fdcheck(dmin: f64) {
    let sc = cantilever(1, dmin, false, 0.005, 1.0);
    let m = &sc.m;
    // rigidly rotated configuration (φ = 0.3) so that the check exercises large rotation
    let qr: Vec<[f64; 3]> = m.q.iter().map(|r| {
        let p = V2::new(r[0], r[1]).rot(0.3);
        [p.x, p.y, r[2] + 0.3]
    }).collect();
    let nc = m.n_cells();
    let mut w = Work::default();
    for zone in [Zone::Root(0.4), Zone::Full] {
        let detailed: Vec<bool> = (0..nc).map(|i| match zone { Zone::Root(x) => m.bodies[i].x0.x < x, _ => true }).collect();
        let sys = Sys::build(m, &detailed, 4, &mut w);
        let mut z = sys.fit(&qr, &mut w);
        let q_back = sys.q_all(&z);
        let fit_err = q_back.iter().zip(&qr).take(nc).map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs())).fold(0.0, f64::max);
        for rg in &sys.regs {
            for j in 0..rg.me {
                z[rg.off + 3 + j] += 1e-6 * ((j as f64) * 1.3).sin();
            }
        }
        for i in 0..sys.nd {
            z[i] += 1e-6 * ((i as f64) * 0.7).cos();
        }
        let qh: Vec<[f64; 3]> = sys.q_all(&z).iter().map(|r| [r[0] + 1e-4, r[1] - 2e-4, r[2] + 1e-3]).collect();
        let sd = sys.step_data(&qh, 1.0 / (1e-3f64).powi(2), 0.7, &mut w);
        let (g, trip, e) = sys.assemble(m, &z, &sd, true, Tangent::Consistent, &mut w);
        let n = sys.n;
        let mut hd = vec![0.0; n * n];
        if n <= 800 {
            for &(a, b, v) in &trip {
                hd[a * n + b] += v;
            }
        }
        let (mut eg, mut eh, mut gs, mut hs) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut worst = (0, 0.0);
        let idx: Vec<usize> = if n <= 800 { (0..n).collect() } else { (0..n).step_by(n / 50).collect() };
        for &i in &idx {
            let hstep = if sys.regs.iter().any(|rg| i == rg.off + 2) { 1e-7 } else { 1e-9 };
            let mut zp = z.clone();
            zp[i] += hstep;
            let mut zm = z.clone();
            zm[i] -= hstep;
            let (gp, _, ep) = sys.assemble(m, &zp, &sd, false, Tangent::Consistent, &mut w);
            let (gm, _, em) = sys.assemble(m, &zm, &sd, false, Tangent::Consistent, &mut w);
            let fd = (ep - em) / (2.0 * hstep);
            eg = eg.max((fd - g[i]).abs());
            gs = gs.max(g[i].abs());
            if (fd - g[i]).abs() > worst.1 {
                worst = (i, (fd - g[i]).abs());
            }
            if n <= 800 {
                for j in 0..n {
                    let fdh = (gp[j] - gm[j]) / (2.0 * hstep);
                    if (fdh - hd[j * n + i]).abs() > eh && (fdh - hd[j * n + i]).abs() > 1e-4 * hd[j * n + i].abs().max(1.0) {
                        eprintln!("    H[{j},{i}] analytic {:.6e} fd {:.6e}", hd[j * n + i], fdh);
                    }
                    eh = eh.max((fdh - hd[j * n + i]).abs());
                    hs = hs.max(hd[j * n + i].abs());
                }
            }
        }
        let _ = e;
        eprintln!("fdcheck {:?}: n={n} regions={} me={:?} fit_err={fit_err:.2e} grad err {eg:.3e} / {gs:.3e} (worst idx {} {:.3e}), hess err {eh:.3e} / {hs:.3e}", match zone { Zone::Full => "full", _ => "root0.4" }, sys.regs.len(), sys.regs.iter().map(|r| r.me).collect::<Vec<_>>(), worst.0, worst.1);
        for rg in &sys.regs {
            eprintln!("   region off {} : c {} φ {} η {}..", rg.off, rg.off, rg.off + 2, rg.off + 3);
        }
    }
}

// ---------------------------------------------------------------- driver

fn main() {
    faer::set_global_parallelism(faer::Par::Seq);
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("exp2");
    let seeds: Vec<u64> = args.iter().skip(2).filter_map(|s| s.parse().ok()).collect();
    let seeds = if seeds.is_empty() { vec![1] } else { seeds };
    let dmin: f64 = std::env::var("SB_DMIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.04);
    let t_end: f64 = std::env::var("SB_TEND").ok().and_then(|v| v.parse().ok()).unwrap_or(0.3);
    let t_r = 0.005;
    let rec = 2.5e-4;
    let nvib = 4;
    let c_wave = Material::concrete(0.3).wave_speed();
    let h_geo = 0.3 * dmin / c_wave;
    if mode == "one" {
        let sc = cantilever(seeds[0], dmin, std::env::var("SB_DAMAGE").is_ok(), t_r, std::env::var("SB_G").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0));
        let zone = match std::env::var("SB_ZONE").as_deref() { Ok("full") => Zone::Full, Ok("auto") => Zone::Auto(0.8, 0.15), _ => Zone::Root(0.4) };
        let tr = run_implicit(&sc, zone, 1.0 / 240.0, t_end, rec, nvib, "one");
        eprintln!("one: steps {} rej {} failed {:?} D {:.3} E0-E {:.3e} rise {:.3e}", tr.steps, tr.rejected, tr.failed_at, tr.d_end, tr.e0 - tr.e_end, tr.e_max_rise);
        for r in tr.rows.iter().step_by(8) { eprintln!("  t {:.4} tip {:.4e} KE {:.3e} U {:.3e} Vg {:.3e} D {:.3e} W {:.3e} E {:.6e}", r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]); }
        return;
    }
    if mode == "conv" {
        let g: f64 = std::env::var("SB_G").ok().and_then(|v| v.parse().ok()).unwrap_or(1.1504);
        let sc = cantilever(seeds[0], dmin, true, t_r, g);
        let lam = lambda_max(&sc.m);
        let he = (0.3 * dmin / sc.m.mat.wave_speed()).min(0.8 * 2.0 / lam.sqrt());
        let jobs: Vec<(String, Option<f64>)> = vec![("explicit_hE".into(), None), ("explicit_hE/2".into(), Some(-0.5)), ("bdf2_960".into(), Some(1.0 / 960.0)), ("bdf2_3840".into(), Some(1.0 / 3840.0)), ("bdf2_15360".into(), Some(1.0 / 15360.0))];
        std::thread::scope(|s| {
            for (label, h) in &jobs {
                let sc = &sc;
                s.spawn(move || {
                    let tr = match h {
                        None => run_explicit(sc, he, t_end, rec),
                        Some(x) if *x < 0.0 => run_explicit(sc, he * 0.5, t_end, rec),
                        Some(x) => run_implicit(sc, Zone::Full, *x, t_end, rec, nvib, label),
                    };
                    let w = &tr.work;
                    eprintln!("{label}: sep {:?} onset {:?} v ({:.3},{:.3}) ω {:.3} D {:.3} steps {} rej {} minh {:.2e} numdiss {:.3e} work {:.3e} (rebuild {:.2e} monitor {:.2e})", tr.sep_t, tr.onset_t, tr.fragment.v[0], tr.fragment.v[1], tr.fragment.omega, tr.d_end, tr.steps, tr.rejected, tr.min_h, tr.e0 - tr.e_end, tr.work_total, w.rebuild_flops, w.monitor_flops);
                });
            }
        });
        return;
    }
    if mode == "exp3" {
        exp3(&seeds, dmin_imp(), rec);
        return;
    }
    if mode == "impact_one" {
        let v0: f64 = std::env::var("SB_V0").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0);
        let sc = beam_impact(seeds[0], dmin_imp(), v0, MP);
        eprintln!("{}: {} cells", sc.name, sc.m.n_cells());
        for (label, meth) in imp_methods() {
            let tr = run_method(&sc, meth, label, T_END_IMP, rec, 4);
            eprintln!("  {label}: v_out {:?} frags {:?} D {:.3} E0-E {:.3e} rise {:.3e} steps {} rej {} det {} work {:.3e} wall {:.1}s", tr.v_impactor, tr.frags.iter().map(|f| (f.mass.round(), (f.v[0] * 100.0).round() / 100.0, (f.v[1] * 100.0).round() / 100.0)).collect::<Vec<_>>(), tr.d_end, tr.e0 - tr.e_end, tr.e_max_rise, tr.steps, tr.rejected, tr.max_detailed, tr.work_total, tr.wall_s);
        }
        return;
    }
    if mode == "exp2" {
        exp2(&seeds, dmin, t_end, t_r, rec, nvib);
        return;
    }
    if mode == "fdcheck" {
        fdcheck(dmin);
        return;
    }
    let mut out = Vec::new();
    for &seed in &seeds {
        for damage in [false, true] {
            if (mode == "fracture" && !damage) || (mode == "elastic" && damage) {
                continue;
            }
            let sc = cantilever(seed, dmin, damage, t_r, 1.0);
            let lam = lambda_max(&sc.m);
            let h_crit = 2.0 / lam.sqrt();
            let h_e = h_geo.min(0.8 * h_crit);
            eprintln!("{}: {} cells, {} facets, h_crit = {:.3e}, h_E = {:.3e} (0.3 d/c = {:.3e})", sc.name, sc.m.n_cells(), sc.m.facets.len(), h_crit, h_e, h_geo);
            // runs: (label, zone, h)
            let mut jobs: Vec<(String, Option<(Zone, f64)>)> = vec![("full_explicit".into(), None)];
            jobs.push(("full_bdf2_h240".into(), Some((Zone::Full, 1.0 / 240.0))));
            jobs.push(("full_bdf2_h480".into(), Some((Zone::Full, 1.0 / 480.0))));
            jobs.push(("full_bdf2_h960".into(), Some((Zone::Full, 1.0 / 960.0))));
            jobs.push(("red_root0.4_h240".into(), Some((Zone::Root(0.4), 1.0 / 240.0))));
            jobs.push(("red_root0.4_h120".into(), Some((Zone::Root(0.4), 1.0 / 120.0))));
            jobs.push(("red_root0.4_h480".into(), Some((Zone::Root(0.4), 1.0 / 480.0))));
            if damage {
                jobs.push(("auto_chi0.8_h240".into(), Some((Zone::Auto(0.8, 0.15), 1.0 / 240.0))));
                jobs.push(("auto_chi0.5_h240".into(), Some((Zone::Auto(0.5, 0.15), 1.0 / 240.0))));
            }
            let results = std::sync::Mutex::new(Vec::new());
            std::thread::scope(|s| {
                for (k, (label, cfg)) in jobs.iter().enumerate() {
                    let sc = &sc;
                    let results = &results;
                    s.spawn(move || {
                        let tr = match cfg {
                            None => run_explicit(sc, h_e, t_end, rec),
                            Some((zone, h)) => run_implicit(sc, *zone, *h, t_end, rec, nvib, label),
                        };
                        eprintln!(
                            "  {} {}: D={:.3} onset={:?} sep={:?} frag v=({:.3},{:.3}) ω={:.3} E0-Eend={:.3e} rise={:.2e} steps={} rej={} work={:.3e} wall={:.1}s",
                            sc.name, tr.label, tr.d_end, tr.onset_t, tr.sep_t, tr.fragment.v[0], tr.fragment.v[1], tr.fragment.omega, tr.e0 - tr.e_end, tr.e_max_rise, tr.steps, tr.rejected, tr.work_total, tr.wall_s
                        );
                        results.lock().unwrap().push((k, tr));
                    });
                }
            });
            let mut res = results.into_inner().unwrap();
            res.sort_by_key(|r| r.0);
            let trs: Vec<Traj> = res.into_iter().map(|r| r.1).collect();
            let rf = &trs[0];
            let ke_scale = rf.rows.iter().map(|r| r[2]).fold(0.0, f64::max);
            let mut csv = Csv::new(EXP, &format!("traj_{}", sc.name), &["run", "t", "tip_dy", "KE", "U", "Vg", "D", "W_ext", "E_tot", "detailed_cells", "M_root"]);
            for (k, tr) in trs.iter().enumerate() {
                for r in &tr.rows {
                    let mut row = vec![k as f64];
                    row.extend_from_slice(r);
                    csv.row(&row);
                }
            }
            let cmp: Vec<Value> = trs[1..].iter().map(|c| if damage { compare_fracture(rf, c, 1.0 / 60.0, ke_scale) } else { compare_elastic(rf, c, t_r) }).collect();
            for c in &cmp {
                eprintln!("{} {}", sc.name, c);
            }
            out.push(json!({"scene": sc.name, "cells": sc.m.n_cells(), "h_explicit": h_e, "runs": trs, "run_labels_in_csv": trs.iter().map(|t| t.label.clone()).collect::<Vec<_>>(), "comparison": cmp}));
        }
    }
    let res = json!({
        "meta": meta(EXP, mode, None, "reference: full RBSN, leapfrog at h = 0.3 d_min/c, history committed every substep; candidates: BDF2 (BE restart on step change) with Newton + sparse Cholesky/LU, accept if converged and Δd ≤ 0.1 else halve",
            json!({"scene": "propped cantilever 4 x 0.4 m, b = 0.3 m, concrete SB-IL1 (or elastic), clamp and prop facets elastic, gravity 9.81, prop force released by a cosine ramp over t_r", "t_r": t_r, "dmin": dmin, "t_end": t_end, "nvib": nvib,
                   "regions": "floating frame: x = c + R(φ)(P + Uη), θ = φ + Wη; basis = static boundary modes (M0) + 2 body-force modes + nvib fixed-interface modes, rigid modes removed, M-orthonormal",
                   "prereg": "docs/phase0c/reduced-fracture.md §4"})),
        "scenes": out,
    });
    write_json(EXP, &format!("summary_{mode}"), &res);
}
