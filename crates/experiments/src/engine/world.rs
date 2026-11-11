//! The engine state and the macro step: activation, impulse-matched multirate
//! coupling, monitor, re-reduction and the energy ledger.

use super::contact::{CItem, ContactModel, GROUND};
use super::patch::{build_patch, chol, chol_solve, k_apply, partition, Patch};
use super::rsys::{fit_patch, Extra, RSys, Rep};
use super::scenes::Scene;
use super::{jperp, Work, C_PT_FORCE};
use geometry::V2;
use model::{BodyKind, LawKind, Model};
use solvers::sparse::{csc_from_triplets, Spd};
use serde::Serialize;
use std::time::Instant;

#[path = "implicit.rs"]
mod implicit;
#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub enum Open0 {
    /// every patch closed, activation on
    None,
    /// every patch open, never closed (the explicit reference)
    All,
    /// cells with x < a open at t = 0
    XLess(f64),
    /// every patch closed and never opened (pure reduced model)
    Never,
}

#[derive(Clone, Debug, Serialize)]
pub struct Params {
    pub h_macro: f64,
    pub lp: f64,
    pub deg: usize,
    pub nvib: usize,
    pub min_cells: usize,
    pub chi_trig: f64,
    /// activation radius around a hot point (patch centroids)
    pub r_e: f64,
    pub chi_rr: f64,
    pub eps_rr: f64,
    pub rereduce: bool,
    /// coupling tolerance as a fraction of the smallest interface δ₀
    pub coupling_tol: f64,
    /// interface force residual tolerance relative to the interface force
    pub coupling_ftol: f64,
    /// per-step budget for the interface energy jump, as a fraction of the scene energy scale
    pub coupling_ebudget: f64,
    pub max_coupling: usize,
    /// below this macro step a coupling failure opens the interface patches instead of halving
    pub h_min_coupling: f64,
    pub open0: Open0,
    pub quiet_steps: usize,
    /// fixed explicit sub-step (S0a); None = 0.8·2/√λ_max of the active zone
    pub h_e_fixed: Option<f64>,
    pub h_e_cap_cfl: bool,
    /// scale on the stable explicit step (1 = 0.8·2/√λ_max)
    pub he_scale: f64,
    /// implicit adaptive mode (one Newton system, split/merge, local-error steps)
    pub implicit: bool,
    /// local-error tolerance: atol (× δ₀_ref) + rtol·|Δx|
    pub lte_atol: f64,
    pub lte_rtol: f64,
    pub h_min: f64,
    /// material points enter the local-error control from this χ_rel on
    pub chi_lte: f64,
    /// Generalised force residual relative to cell weight. Kept explicit because
    /// 1e-9 was below the attainable facet-force roundoff on the large 2D scenes.
    pub newton_force_rtol: f64,
    /// Velocity tolerance for leaving a global explicit contact window. This is a
    /// scheduling guard, never a reason to exclude contacts from error control.
    pub contact_quiet_speed: f64,
    /// switch the implicit mode to an all-explicit window when it costs more per
    /// simulated second than explicit on the whole model (ratio), and back when quiet
    pub cost_switch: Option<f64>,
    /// absorbing E–R interface (Lysmer dashpots ρ c A on interface facets): the radiation
    /// of waves the reduced side cannot carry, booked as D_rad
    pub absorbing: bool,
}

impl Params {
    pub fn new(h_macro: f64, lp: f64) -> Params {
        Params {
            h_macro,
            lp,
            deg: 2,
            nvib: 0,
            min_cells: 4,
            chi_trig: 0.9,
            r_e: lp,
            chi_rr: 0.7,
            eps_rr: 0.05,
            rereduce: true,
            coupling_tol: 0.1,
            coupling_ftol: 1e-5,
            coupling_ebudget: 1e-6,
            max_coupling: 4,
            h_min_coupling: h_macro / 16.0,
            open0: Open0::None,
            quiet_steps: 3,
            h_e_fixed: None,
            h_e_cap_cfl: true,
            he_scale: 1.0,
            implicit: false,
            lte_atol: 0.5,
            lte_rtol: 0.05,
            h_min: 1e-7,
            chi_lte: 0.5,
            newton_force_rtol: 1e-8,
            contact_quiet_speed: 0.02,
            cost_switch: Some(1.0),
            absorbing: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Ledger {
    pub w_ext: f64,
    pub d_red: f64,
    pub d_fric: f64,
    /// R's BDF2 dissipation (positive = lost), from R's own per-step balance
    pub d_num_r: f64,
    /// interface work on R and on E (should cancel)
    pub w_iface_r: f64,
    pub w_iface_e: f64,
    /// energy removed (+) or added (−) by the interface jump at accepted coupling steps
    pub d_coupling: f64,
    /// wave energy radiated through an absorbing interface
    pub d_rad: f64,
    /// Signed energy removed by a change of the elastic representation (e.g.
    /// Khat -> full law on split). This is model discrepancy, NOT dissipation.
    pub d_model: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FrameCost {
    pub frame: usize,
    pub work: f64,
    pub wall_s: f64,
    pub accepted: u64,
    pub rejected: u64,
    pub rebuild_work: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Fragment {
    pub separated: bool,
    pub mass: f64,
    pub cm: [f64; 2],
    pub v: [f64; 2],
    pub omega: f64,
    pub cells: usize,
}

/// A recorded row.
pub const ROW: [&str; 16] = ["t", "probe1", "ke", "u", "vg", "d", "w_ext", "e_tot", "e_cells", "probe2", "d_red", "d_fric", "d_num_r", "closed_patches", "coupling_iters", "macro_h"];

#[derive(Clone, Default, Serialize)]
pub struct Traj {
    pub label: String,
    pub scene: String,
    #[serde(skip)]
    pub rows: Vec<[f64; 16]>,
    pub onset_t: Option<f64>,
    pub sep_t: Option<f64>,
    pub d_end: f64,
    pub crack: [f64; 6],
    pub fragment: Fragment,
    pub frags: Vec<Fragment>,
    pub e0: f64,
    pub e_end: f64,
    pub e_max_rise: f64,
    pub ke_max: f64,
    pub ledger: Ledger,
    pub work: Work,
    pub work_total: f64,
    pub wall_s: f64,
    pub cells: usize,
    pub patches: usize,
    pub max_e_cells: usize,
    /// ∫ e_cells dt / (T · cells)
    pub mean_e_frac: f64,
    pub v_impactor: Option<f64>,
    pub failed_at: Option<f64>,
    pub h_e: f64,
    pub chi_r_max: f64,
    pub events: Vec<String>,
    pub momentum_drift: f64,
    /// All work caused by an accepted advance, including rejected attempts,
    /// switches and merges, charged to the 60-Hz frame where it starts.
    /// This deliberately does not amortise a costly event over later frames.
    pub frames: Vec<FrameCost>,
    pub startup_work: f64,
    /// All cohesive components, including those below the visual 5% filter.
    pub fragment_masses: Vec<f64>,
}

struct Snap {
    q: Vec<[f64; 3]>,
    v: Vec<[f64; 3]>,
    vh: Vec<[f64; 3]>,
    fresh: Vec<bool>,
    kappa: Vec<(usize, [f64; 2])>,
    lg: Vec<(usize, f64)>,
    dissipated: f64,
    sub_t0: f64,
    sub_count: u64,
    led: Ledger,
}

pub struct World {
    /// Research-only constitutive mask; independent of the E/R DOF partition.
    pub carrier: Option<super::carrier::Carrier>,
    pub sc: Scene,
    pub p: Params,
    pub adj: Vec<Vec<usize>>,
    pub patches: Vec<Patch>,
    pub alive: Vec<bool>,
    pub open: Vec<bool>,
    pub open_since: Vec<f64>,
    /// times a patch was reopened after a re-reduction (quiet time doubles each time)
    pub reopen: Vec<u32>,
    closed_once: Vec<bool>,
    pub cell_patch: Vec<usize>,
    pub loaded: Vec<bool>,
    pub q: Vec<[f64; 3]>,
    pub v: Vec<[f64; 3]>,
    pub qp: Vec<[f64; 3]>,
    pub vp: Vec<[f64; 3]>,
    vh: Vec<[f64; 3]>,
    fresh: Vec<bool>,
    minv: Vec<[f64; 3]>,
    pub t: f64,
    pub h: f64,
    restart: bool,
    pub rs: RSys,
    pub e_bodies: Vec<usize>,
    pub e_facets: Vec<usize>,
    pub he: f64,
    material_spectrum_cache: Option<(Vec<usize>, Vec<usize>, f64)>,
    sub_t0: f64,
    sub_count: u64,
    pub last_growth: Vec<f64>,
    pub led: Ledger,
    pub w: Work,
    j_prev: Option<(u64, Vec<f64>)>,
    zm_prev: Option<(u64, Vec<f64>)>,
    easy_steps: usize,
    last_iters: usize,
    parent: Vec<usize>,
    h_prev: f64,
    /// all-explicit window (implicit mode): active and earliest exit time
    pub window: Option<f64>,
    h_before_window: f64,
    window_len: f64,
    over: usize,
    d_hist: Vec<(f64, f64)>,
    rate_expl: f64,
    rate_hist: Vec<(f64, f64)>,
    pub windows: Vec<[f64; 2]>,
    pub cm: Option<ContactModel>,
    pub cpairs: Vec<(usize, usize)>,
    fric_power: f64,
    removed_tags: Vec<u32>,
    /// R interface-cell velocities over the current macro step (absorbing interface)
    // Velocities belong to stable body IDs, not to the current iface_cells order.
    // This cache is valid only while a particular subcycle is running.
    vr_iface: std::collections::BTreeMap<usize, [f64; 3]>,
    rad_power: f64,
    version: u64,
    tol: f64,
    delta0_ref: f64,
    iface_d0: f64,
    pub events: Vec<String>,
    pub chi_r_max: f64,
    pub verbose: bool,
}

fn find(p: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while p[r] != r {
        p[r] = p[p[r]];
        r = p[r];
    }
    r
}

impl World {
    pub fn new(sc: Scene, p: Params) -> World {
        let m = &sc.m;
        let nb = m.bodies.len();
        let adj = m.neighbours();
        let loaded = sc.loaded();
        let mut w = Work::default();
        let groups = partition(m, &adj, p.lp, p.min_cells);
        let patches: Vec<Patch> = groups.iter().map(|g| build_patch(m, &adj, g, p.deg, p.nvib, &loaded, &mut w)).collect();
        let np = patches.len();
        let mut cell_patch = vec![usize::MAX; nb];
        for (pid, pa) in patches.iter().enumerate() {
            for &i in &pa.cells {
                cell_patch[i] = pid;
            }
        }
        let open: Vec<bool> = (0..np)
            .map(|pid| match p.open0 {
                Open0::All => true,
                Open0::XLess(x) => patches[pid].cells.iter().any(|&i| m.bodies[i].x0.x < x),
                _ => false,
            })
            .collect();
        let minv: Vec<[f64; 3]> = m.bodies.iter().map(|b| if b.mass > 0.0 { [1.0 / b.mass, 1.0 / b.mass, 1.0 / b.inertia] } else { [0.0; 3] }).collect();
        let mg: f64 = m.bodies.iter().map(|b| if b.kind == BodyKind::Cell { b.mass } else { 0.0 }).sum::<f64>() * super::G;
        let q = m.q.clone();
        let v = sc.v_init.clone();
        let delta0_ref = sc.delta0_ref();
        let h = p.h_macro;
        let nf = m.facets.len();
        let mut wd = World {
            carrier: None,
            adj,
            patches,
            alive: vec![true; np],
            open,
            open_since: vec![0.0; np],
            reopen: vec![0; np],
            closed_once: vec![false; np],
            cell_patch,
            loaded,
            qp: q.clone(),
            vp: v.clone(),
            vh: v.clone(),
            fresh: vec![true; nb],
            q,
            v,
            minv,
            t: 0.0,
            h,
            restart: true,
            rs: RSys::empty(nb),
            e_bodies: Vec::new(),
            e_facets: Vec::new(),
            he: 0.0,
            material_spectrum_cache: None,
            sub_t0: 0.0,
            sub_count: 0,
            last_growth: vec![f64::NEG_INFINITY; nf],
            led: Ledger::default(),
            w,
            j_prev: None,
            zm_prev: None,
            easy_steps: 0,
            last_iters: 0,
            parent: vec![usize::MAX; np],
            h_prev: 0.0,
            window: None,
            h_before_window: 0.0,
            window_len: 2e-3,
            over: 0,
            d_hist: Vec::new(),
            rate_expl: 0.0,
            rate_hist: Vec::new(),
            windows: Vec::new(),
            cm: None,
            cpairs: Vec::new(),
            fric_power: 0.0,
            removed_tags: Vec::new(),
            vr_iface: Default::default(),
            rad_power: 0.0,
            version: 0,
            tol: p.newton_force_rtol * mg,
            delta0_ref,
            iface_d0: delta0_ref,
            events: Vec::new(),
            chi_r_max: 0.0,
            verbose: std::env::var("SB_VERBOSE").is_ok(),
            sc,
            p,
        };
        // reduced static equilibrium of the initial state with every patch closed, then
        // the a priori open set (its cells start from the exact prolongation)
        if let Some(g) = wd.sc.ground {
            let cm = ContactModel::new(&wd.sc.m, Some(g.y0), g.mu);
            wd.cpairs = cm.near_pairs(&wd.q, wd.sc.dmin);
            wd.cm = Some(cm);
        }
        let open0 = std::mem::replace(&mut wd.open, vec![false; np]);
        if wd.p.implicit {
            wd.add_plate_patches();
        }
        wd.rebuild();
        if wd.sc.static_presolve && wd.rs.n > 0 && wd.p.open0 != Open0::All {
            let mut z = wd.rs.fit(&wd.patches, &wd.q, &mut wd.w);
            let sd = wd.rs.step_data(&wd.patches, &wd.q, 0.0, &mut wd.w);
            let ext = wd.sc.ext(0.0);
            let ex = Extra { ext: &ext, fj: None, robin: None, contacts: None };
            let ok = wd.rs.solve(&wd.sc.m, &wd.patches, &mut z, &sd, &ex, wd.tol, 60, &mut wd.w);
            assert!(ok, "reduced static equilibrium");
            let mut q = wd.q.clone();
            wd.rs.q_all(&wd.patches, &z, &mut q, &mut wd.w);
            wd.q = q.clone();
            wd.qp = q;
        }
        if wd.p.implicit {
            // no history yet: start small and let the controller grow the step
            wd.h = wd.p.h_macro / 16.0;
        }
        if open0.iter().any(|&o| o) {
            if wd.p.implicit {
                let set: Vec<usize> = (0..np).filter(|&p| open0[p]).collect();
                wd.split(&set, "a priori");
            } else {
                let mut o = open0;
                o.resize(wd.patches.len(), false);
                wd.open = o;
                wd.rebuild();
            }
        }
        wd
    }

    pub fn enable_carrier(&mut self, config: super::carrier::Config) {
        assert!(self.p.open0 == Open0::All && !self.p.implicit && !self.p.rereduce);
        assert_eq!(self.rs.n,0,"carrier requires all wave DOFs; E/R rollback is unsupported");
        self.carrier=Some(super::carrier::Carrier::new(&self.sc.m,&self.q,config));
    }

    fn is_cell(&self, i: usize) -> bool {
        self.sc.m.bodies[i].kind == BodyKind::Cell
    }

    /// Rebuild the R system and the active-zone lists after the open set changed.
    fn rebuild(&mut self) {
        self.vr_iface.clear();
        self.version += 1;
        let closed: Vec<bool> = (0..self.patches.len()).map(|pid| self.alive[pid] && !self.open[pid]).collect();
        self.rs = RSys::build(&self.sc.m, &self.patches, &closed, &self.loaded, &self.cpairs, self.version, &mut self.w);
        let m = &self.sc.m;
        self.e_bodies = (0..m.bodies.len()).filter(|&i| self.rs.rep[i] == Rep::E).collect();
        let is_e: Vec<bool> = self.rs.rep.iter().map(|r| *r == Rep::E).collect();
        self.e_facets = (0..m.facets.len()).filter(|&fi| m.facets[fi].active && (is_e[m.facets[fi].a] || is_e[m.facets[fi].b])).collect();
        // smallest δ₀ over interface facet points (coupling tolerance scale)
        let mut d0 = f64::INFINITY;
        for &fi in &self.rs.iface_facets {
            for p in &m.facets[fi].pts {
                if m.facets[fi].tag == 0 {
                    d0 = d0.min(p.law.delta0);
                }
            }
        }
        self.iface_d0 = if d0.is_finite() { d0 } else { self.delta0_ref };
        let he_old = self.he;
        self.he = self.stable_he();
        if self.he != he_old {
            // re-synchronise the leapfrog: every active body restarts from its synchronised velocity
            for &i in &self.e_bodies {
                self.fresh[i] = true;
            }
            self.sub_t0 = self.t;
            self.sub_count = 0;
        }
        // cell histories persist through a rebuild; only the multirate mode restarts BDF
        // here (the implicit mode restarts explicitly after a projection)
        if !self.p.implicit {
            self.restart = true;
        }
    }

    /// Stable explicit step of the active zone: 0.8·2/√λ_max(M⁻¹K) with the R side
    /// held fixed, capped by 0.3 d_min/c as in the reference.
    fn stable_he(&mut self) -> f64 {
        if let Some(h) = self.p.h_e_fixed {
            return h;
        }
        if self.e_bodies.is_empty() {
            return 0.0;
        }
        let m = &self.sc.m;
        let nb = m.bodies.len();
        let q0: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
        let free = |i: usize, k: usize| self.rs.rep[i] == Rep::E && !m.bodies[i].fixed[k] && m.bodies[i].mass > 0.0;
        let sq: Vec<[f64; 3]> = (0..nb).map(|i| std::array::from_fn(|k| if free(i, k) { if k < 2 { m.bodies[i].mass.sqrt() } else { m.bodies[i].inertia.sqrt() } } else { 0.0 })).collect();
        let mut x: Vec<[f64; 3]> = (0..nb).map(|i| std::array::from_fn(|k| if sq[i][k] > 0.0 { ((i * 3 + k) as f64 * 0.618).sin() } else { 0.0 })).collect();
        let mut lam = 0.0;
        let mut flops = 0.0;
        if let Some((_, _, cached)) = self.material_spectrum_cache.as_ref().filter(|(b,f,_)| *b == self.e_bodies && *f == self.e_facets) {
            lam = *cached;
        } else { for _ in 0..200 {
            let u: Vec<[f64; 3]> = (0..nb).map(|i| std::array::from_fn(|k| if sq[i][k] > 0.0 { x[i][k] / sq[i][k] } else { 0.0 })).collect();
            let mut y = vec![[0.0f64; 3]; nb];
            for &fi in &self.e_facets {
                let f = &m.facets[fi];
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
                flops += 2.0 * C_PT_FORCE;
            }
            for i in 0..nb {
                for k in 0..3 {
                    y[i][k] = if sq[i][k] > 0.0 { y[i][k] / sq[i][k] } else { 0.0 };
                }
            }
            let ny: f64 = y.iter().flat_map(|r| r.iter()).map(|v| v * v).sum::<f64>().sqrt();
            let nx: f64 = x.iter().flat_map(|r| r.iter()).map(|v| v * v).sum::<f64>().sqrt();
            if ny == 0.0 || nx == 0.0 {
                break;
            }
            lam = ny / nx;
            x = y.iter().map(|r| std::array::from_fn(|k| r[k] / ny)).collect();
        }
            self.material_spectrum_cache = Some((self.e_bodies.clone(), self.e_facets.clone(), lam));
        }
        self.w.rebuild_flops += flops;
        // Normal contact stiffness is absent from the material spectrum. In
        // particular polygon vertices against the floor can be much stiffer than
        // a cell facet. A trace bound on each mass-scaled contact block is an upper
        // bound at every orientation; sum it with the material estimate.
        if let Some(cm) = &self.cm {
            let mut cb = vec![0.0; nb];
            for &(a, b) in &self.cpairs {
                if b == GROUND {
                    if self.rs.rep[a] == Rep::E {
                        for &(r, k) in &cm.verts[a] {
                            cb[a] += k * (1.0 / m.bodies[a].mass + r.dot(r) / m.bodies[a].inertia);
                        }
                    }
                } else {
                    let k = cm.k_pair(a, b);
                    if self.rs.rep[a] == Rep::E { cb[a] += 2.0 * k / m.bodies[a].mass; }
                    if self.rs.rep[b] == Rep::E { cb[b] += 2.0 * k / m.bodies[b].mass; }
                }
            }
            lam += cb.into_iter().fold(0.0f64, f64::max);
        }
        let c = m.mat.wave_speed();
        let mut h = if lam > 0.0 { 0.8 * 2.0 / lam.sqrt() } else { f64::INFINITY };
        if self.p.h_e_cap_cfl {
            h = h.min(0.3 * self.sc.dmin / c);
        }
        h * self.p.he_scale
    }

    fn snapshot(&self) -> Snap {
        let m = &self.sc.m;
        Snap {
            q: self.q.clone(),
            v: self.v.clone(),
            vh: self.vh.clone(),
            fresh: self.fresh.clone(),
            kappa: self.e_facets.iter().map(|&fi| (fi, [m.facets[fi].pts[0].kappa_c, m.facets[fi].pts[1].kappa_c])).collect(),
            lg: self.e_facets.iter().map(|&fi| (fi, self.last_growth[fi])).collect(),
            dissipated: m.dissipated,
            sub_t0: self.sub_t0,
            sub_count: self.sub_count,
            led: self.led,
        }
    }

    fn restore(&mut self, s: &Snap) {
        self.q.clone_from(&s.q);
        self.v.clone_from(&s.v);
        self.vh.clone_from(&s.vh);
        self.fresh.clone_from(&s.fresh);
        for &(fi, k) in &s.kappa {
            self.sc.m.facets[fi].pts[0].kappa_c = k[0];
            self.sc.m.facets[fi].pts[1].kappa_c = k[1];
        }
        for &(fi, t) in &s.lg {
            self.last_growth[fi] = t;
        }
        self.sc.m.dissipated = s.dissipated;
        self.sub_t0 = s.sub_t0;
        self.sub_count = s.sub_count;
        self.led = s.led;
    }

    /// Interpolated reduced state z_n + s (z_e − z_n) restricted to what the
    /// interface needs (the whole vector is cheap at these sizes).
    fn lerp(z0: &[f64], z1: &[f64], s: f64) -> Vec<f64> {
        z0.iter().zip(z1).map(|(a, b)| a + s * (b - a)).collect()
    }

    fn rq(&self, i: usize, z: &[f64]) -> [f64; 3] {
        let Rep::R { k, li, .. } = self.rs.rep[i] else { unreachable!() };
        self.rs.cell_q(&self.patches, k, li, z)
    }

    /// Forces on active bodies at the working configuration `qw` (R interface cells
    /// already written into it). Returns the force on each R interface cell too.
    fn e_forces(&mut self, qw: &[[f64; 3]], t: f64, f: &mut [[f64; 3]], fr: &mut [[f64; 3]], commit: bool) {
        let m = &self.sc.m;
        let mut updates = Vec::new();
        for &i in &self.e_bodies {
            f[i] = m.loads[i];
        }
        for (i, fe) in self.sc.ext(t) {
            if self.rs.rep[i] == Rep::E {
                for r in 0..3 {
                    f[i][r] += fe[r];
                }
            }
        }
        for &i in &self.rs.iface_cells {
            fr[i] = [0.0; 3];
        }
        for &fi in &self.e_facets {
            let fa = &m.facets[fi];
            let evals = if let Some(c)=self.carrier.as_mut() {
                if c.hot[fi] {c.stats.full_points+=2;None} else {
                    let (evals,de)=c.evaluate(m,fi,qw);self.led.d_model-=de;Some(evals)
                }
            } else {None};
            for pt in 0..2 {
                let (j, e) = if let Some(evals)=evals.as_ref(){evals[pt]}else{m.eval_point(qw,fa,pt)};
                if commit && e.kappa > fa.pts[pt].kappa_c {
                    updates.push((fi,pt,e.kappa,fa.pts[pt].law.dissipation(e.kappa)-fa.pts[pt].law.dissipation(fa.pts[pt].kappa_c)));
                }
                for r in 0..3 {
                    let ga = j.jna[r] * e.fnorm + j.jta[r] * e.ftan;
                    let gb = j.jnb[r] * e.fnorm + j.jtb[r] * e.ftan;
                    match self.rs.rep[fa.a] {
                        Rep::E => f[fa.a][r] -= ga,
                        Rep::R { .. } => fr[fa.a][r] -= ga,
                        Rep::Fixed => {}
                    }
                    match self.rs.rep[fa.b] {
                        Rep::E => f[fa.b][r] -= gb,
                        Rep::R { .. } => fr[fa.b][r] -= gb,
                        Rep::Fixed => {}
                    }
                }
            }
        }
        self.w.e_pts += 2 * self.e_facets.len() as u64;
        if self.p.absorbing && !self.vr_iface.is_empty() {
            // Lysmer dashpots on the interface facets, between the active cell (half-step
            // velocity) and the R cell (mean velocity over the macro step); paired forces
            let rc = m.mat.density * m.mat.wave_speed() * m.mat.thickness;
            let mut pw = 0.0;
            for &fi in &self.rs.iface_facets {
                let fa = &m.facets[fi];
                let (ie, ir) = if self.rs.rep[fa.a] == Rep::E { (fa.a, fa.b) } else { (fa.b, fa.a) };
                let Some(vr) = self.vr_iface.get(&ir) else { continue };
                if self.rs.rep[ie] != Rep::E {
                    continue;
                }
                let z = rc * fa.len;
                let dv = [self.vh[ie][0] - vr[0], self.vh[ie][1] - vr[1]];
                for r in 0..2 {
                    f[ie][r] -= z * dv[r];
                    fr[ir][r] += z * dv[r];
                }
                pw += z * (dv[0] * dv[0] + dv[1] * dv[1]);
            }
            if commit {
                self.rad_power = pw;
            }
        }
        if self.cm.is_some() {
            let pw = self.e_contact_forces(qw, f);
            if commit {
                self.fric_power = pw;
            }
        }
        if commit {
            let t_now = t;
            let m = &mut self.sc.m;
            for (fi,pt,kappa,dd) in updates {
                m.facets[fi].pts[pt].kappa_c=kappa;
                m.dissipated+=dd;
                if dd>0.0 {self.last_growth[fi]=t_now;}
            }
        }
    }

    /// All-fine contact path: symmetric friction half-impulses around a complete
    /// velocity-Verlet step. Book impulses at synchronised velocities. Applying
    /// and booking them on the leapfrog half-grid produced a first-order energy
    /// defect when normal force and tangential torque were strongly coupled.
    fn subcycle_contact(&mut self, ns:usize) -> f64 {
        let nb=self.sc.m.bodies.len();let he=self.he;let bodies=self.e_bodies.clone();
        let mut qw=self.q.clone();let mut f=vec![[0.0;3];nb];let mut fr=f.clone();let mut work=0.0;
        for _ in 0..ns {
            let t=self.sub_t0+self.sub_count as f64*he;
            self.e_forces(&qw,t,&mut f,&mut fr,false);
            self.vh.clone_from(&self.v);self.contact_friction(&qw,0.5*he);
            for &i in &bodies {for k in 0..3 {if !self.sc.m.bodies[i].fixed[k] {self.vh[i][k]+=0.5*he*self.minv[i][k]*f[i][k];}}}
            let ext0=self.sc.ext(t);let ext1=self.sc.ext(t+he);
            for &(i,f0) in &ext0 {
                let f1=ext1.iter().find(|x|x.0==i).map(|x|x.1).unwrap_or([0.0;3]);
                for k in 0..3 {if !self.sc.m.bodies[i].fixed[k] {work+=0.5*(f0[k]+f1[k])*he*self.vh[i][k];}}
            }
            for &i in &bodies {for k in 0..3 {if !self.sc.m.bodies[i].fixed[k] {self.q[i][k]+=he*self.vh[i][k];}}qw[i]=self.q[i];}
            self.e_forces(&qw,t+he,&mut f,&mut fr,true);
            for &i in &bodies {for k in 0..3 {if !self.sc.m.bodies[i].fixed[k] {self.vh[i][k]+=0.5*he*self.minv[i][k]*f[i][k];}}}
            self.contact_friction(&qw,0.5*he);self.v.clone_from(&self.vh);
            for &i in &bodies {self.fresh[i]=true;}
            self.sub_count+=1;self.w.e_substeps+=1;self.w.e_body_flops+=(bodies.len()*36) as f64;
        }
        work
    }

    /// Sub-cycle the active zone over the macro step with the R side moving linearly
    /// in z from z_n to z_e. Returns the generalised impulse on R and the interface
    /// work done on E. `ns` sub-steps of `he`.
    fn subcycle(&mut self, z_n: &[f64], z_e: &[f64], ns: usize) -> (Vec<f64>, f64, f64) {
        let nb = self.sc.m.bodies.len();
        let he = self.he;
        let mut qw = self.q.clone();
        let mut f = vec![[0.0f64; 3]; nb];
        let mut fr = vec![[0.0f64; 3]; nb];
        let mut jimp = vec![0.0; self.rs.n];
        let mut w_e = 0.0;
        let mut w_ext = 0.0;
        let iface = self.rs.iface_cells.clone();
        let ebod = self.e_bodies.clone();
        let mut q_r_prev: Vec<[f64; 3]> = iface.iter().map(|&i| self.rq(i, z_n)).collect();
        let hmac = he * ns as f64;
        self.vr_iface = iface.iter().zip(&q_r_prev).map(|(&i, q0)| {
            let q1 = self.rq(i, z_e);
            (i, std::array::from_fn(|k| (q1[k] - q0[k]) / hmac))
        }).collect();
        for s in 0..ns {
            let t_s = self.sub_t0 + self.sub_count as f64 * he;
            // the boundary jumps b_s → b_{s+1} during the sub-step; the interface force
            // is taken at the jump's midpoint (the same force for E's kick and R's
            // impulse), so momentum is exact and the work matches to O(Δb³)
            let zs = Self::lerp(z_n, z_e, (s as f64 + 0.5) / ns as f64);
            for &i in &iface {
                qw[i] = self.rq(i, &zs);
            }
            self.w.iface_flops += (iface.len() * 12) as f64 * 3.0;
            self.e_forces(&qw, t_s, &mut f, &mut fr, true);
            // kick + drift (leapfrog, as the explicit reference)
            let t_s1 = self.sub_t0 + (self.sub_count + 1) as f64 * he;
            let ext0 = self.sc.ext(t_s);
            let ext1 = self.sc.ext(t_s1);
            for &i in &ebod {
                let b = &self.sc.m.bodies[i];
                for r in 0..3 {
                    if b.fixed[r] {
                        continue;
                    }
                    let dt = if self.fresh[i] { 0.5 * he } else { he };
                    self.vh[i][r] = if self.fresh[i] { self.v[i][r] + dt * self.minv[i][r] * f[i][r] } else { self.vh[i][r] + dt * self.minv[i][r] * f[i][r] };
                }
                self.fresh[i] = false;
            }
            self.contact_friction(&qw, he);
            for &(i, fe) in &ext0 {
                if self.rs.rep[i] == Rep::E {
                    let fe1 = ext1.iter().find(|x| x.0 == i).map(|x| x.1).unwrap_or([0.0; 3]);
                    for r in 0..3 {
                        if !self.sc.m.bodies[i].fixed[r] {
                            w_ext += 0.5 * (fe[r] + fe1[r]) * he * self.vh[i][r];
                        }
                    }
                }
            }
            for &i in &ebod {
                let b = &self.sc.m.bodies[i];
                for r in 0..3 {
                    if !b.fixed[r] {
                        let dq = he * self.vh[i][r];
                        self.q[i][r] += dq;
                        qw[i][r] = self.q[i][r];
                    }
                }
            }
            self.w.e_body_flops += (ebod.len() * 12) as f64;
            self.led.d_fric += self.fric_power * he;
            self.fric_power = 0.0;
            self.led.d_rad += self.rad_power * he;
            self.rad_power = 0.0;
            // generalised impulse on R and interface work on E
            let z1 = Self::lerp(z_n, z_e, (s + 1) as f64 / ns as f64);
            for (c, &i) in iface.iter().enumerate() {
                let Rep::R { k, li, .. } = self.rs.rep[i] else { unreachable!() };
                let (d, _, _) = self.rs.cell_jac(&self.patches, k, li, &zs);
                let off = self.rs.off[k];
                for (jj, dj) in d.iter().enumerate() {
                    jimp[off + jj] += he * (dj[0] * fr[i][0] + dj[1] * fr[i][1] + dj[2] * fr[i][2]);
                }
                let qn = self.rq(i, &z1);
                for r in 0..3 {
                    w_e -= fr[i][r] * (qn[r] - q_r_prev[c][r]);
                }
                q_r_prev[c] = qn;
                self.w.iface_flops += (d.len() * 6 + 12 * d.len()) as f64;
            }
            self.sub_count += 1;
            self.w.e_substeps += 1;
        }
        // synchronised velocities at the end (diagnostic, transitions) and the commit at t_{n+1}
        for (c, &i) in iface.iter().enumerate() {
            qw[i] = q_r_prev[c];
        }
        let t_e = self.sub_t0 + self.sub_count as f64 * he;
        self.e_forces(&qw, t_e, &mut f, &mut fr, true);
        for &i in &ebod {
            let b = &self.sc.m.bodies[i];
            for r in 0..3 {
                if !b.fixed[r] {
                    self.v[i][r] = self.vh[i][r] + 0.5 * he * self.minv[i][r] * f[i][r];
                }
            }
        }
        self.vr_iface.clear();
        self.rad_power = 0.0;
        (jimp, w_e, w_ext)
    }

    /// E's impedance seen by R over a macro step: the Schur complement of the active
    /// zone (secant stiffness + 4M/H², the midpoint's dynamic stiffness) onto the
    /// interface R cells, projected on their patch coordinates. z-space triplets.
    fn robin(&mut self, z_n: &[f64], hs: f64) -> Vec<(usize, usize, f64)> {
        let m = &self.sc.m;
        let nb = m.bodies.len();
        let mut eidx = vec![[usize::MAX; 3]; nb];
        let mut ne = 0;
        for &i in &self.e_bodies {
            for k in 0..3 {
                if !m.bodies[i].fixed[k] {
                    eidx[i][k] = ne;
                    ne += 1;
                }
            }
        }
        let mut iidx = vec![usize::MAX; nb];
        for (c, &i) in self.rs.iface_cells.iter().enumerate() {
            iidx[i] = c;
        }
        let ni = 3 * self.rs.iface_cells.len();
        if ni == 0 || ne == 0 {
            return Vec::new();
        }
        let gidx = |b: usize, k: usize| -> (u8, usize) {
            if eidx[b][k] != usize::MAX {
                (0, eidx[b][k])
            } else if iidx[b] != usize::MAX {
                (1, 3 * iidx[b] + k)
            } else {
                (2, 0)
            }
        };
        let mut kee = Vec::new();
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); ni];
        let mut kii = vec![0.0; ni * ni];
        for &fi in &self.e_facets {
            let fa = &m.facets[fi];
            for pt in 0..2 {
                let (j, e) = m.eval_point(&self.q, fa, pt);
                let (kn, kt) = fa.pts[pt].law.secant(j.dn, e.d);
                let jn = [j.jna[0], j.jna[1], j.jna[2], j.jnb[0], j.jnb[1], j.jnb[2]];
                let jt = [j.jta[0], j.jta[1], j.jta[2], j.jtb[0], j.jtb[1], j.jtb[2]];
                let ids: [(u8, usize); 6] = [gidx(fa.a, 0), gidx(fa.a, 1), gidx(fa.a, 2), gidx(fa.b, 0), gidx(fa.b, 1), gidx(fa.b, 2)];
                for r in 0..6 {
                    for c in 0..6 {
                        let v = kn * jn[r] * jn[c] + kt * jt[r] * jt[c];
                        if v == 0.0 {
                            continue;
                        }
                        match (ids[r].0, ids[c].0) {
                            (0, 0) => kee.push((ids[r].1, ids[c].1, v)),
                            (0, 1) => cols[ids[c].1].push((ids[r].1, v)),
                            (1, 1) => kii[ids[r].1 * ni + ids[c].1] += v,
                            _ => {}
                        }
                    }
                }
            }
        }
        let a = 4.0 / (hs * hs);
        for &i in &self.e_bodies {
            for k in 0..3 {
                if eidx[i][k] != usize::MAX {
                    let mm = if k < 2 { m.bodies[i].mass } else { m.bodies[i].inertia };
                    kee.push((eidx[i][k], eidx[i][k], a * mm));
                }
            }
        }
        let mat = csc_from_triplets(ne, kee);
        let mut spd = Spd::analyze(&mat);
        if spd.factor(&mat).is_err() {
            return Vec::new();
        }
        let mut flops = spd.stats.flops_factor;
        let mut s = kii;
        for c in 0..ni {
            if cols[c].is_empty() {
                continue;
            }
            let mut x = vec![0.0; ne];
            for &(e, v) in &cols[c] {
                x[e] += v;
            }
            spd.solve(&mut x);
            flops += 4.0 * spd.stats.nnz_l as f64;
            for r in 0..ni {
                let dot: f64 = cols[r].iter().map(|&(e, v)| v * x[e]).sum();
                s[r * ni + c] -= dot;
            }
        }
        // project: S_z = Gᵀ S G over the interface cells
        let iface = self.rs.iface_cells.clone();
        let gs: Vec<(usize, Vec<[f64; 3]>)> = iface
            .iter()
            .map(|&i| {
                let Rep::R { k, li, .. } = self.rs.rep[i] else { unreachable!() };
                (self.rs.off[k], self.rs.cell_jac(&self.patches, k, li, z_n).0)
            })
            .collect();
        let mut acc: std::collections::BTreeMap<(usize, usize), f64> = Default::default();
        for (ca, (oa, ga)) in gs.iter().enumerate() {
            for (cb, (ob, gb)) in gs.iter().enumerate() {
                // 3×3 block S_ab
                let blk: [[f64; 3]; 3] = std::array::from_fn(|r| std::array::from_fn(|t| s[(3 * ca + r) * ni + 3 * cb + t]));
                if blk.iter().all(|row| row.iter().all(|&v| v == 0.0)) {
                    continue;
                }
                let sg: Vec<[f64; 3]> = gb.iter().map(|dj| std::array::from_fn(|r| blk[r][0] * dj[0] + blk[r][1] * dj[1] + blk[r][2] * dj[2])).collect();
                for (aa, da) in ga.iter().enumerate() {
                    for (bb, sb) in sg.iter().enumerate() {
                        let v = da[0] * sb[0] + da[1] * sb[1] + da[2] * sb[2];
                        *acc.entry((oa + aa, ob + bb)).or_insert(0.0) += v;
                    }
                }
                flops += (9 * gb.len() + 3 * ga.len() * gb.len()) as f64;
            }
        }
        self.w.rebuild_flops += flops;
        acc.into_iter().filter(|(_, v)| *v != 0.0).map(|((r, c), v)| (r, c, v)).collect()
    }


    /// Energy of the active zone's own terms: KE, E facets (incl. interface), gravity.
    pub fn e_energy(&self, q: &[[f64; 3]], v: &[[f64; 3]]) -> f64 {
        let m = &self.sc.m;
        let mut u = 0.0;
        for &fi in &self.e_facets {
            for pt in 0..2 {
                u += if let Some(c)=self.carrier.as_ref(){c.energy(m,fi,pt,q)}else{m.eval_point(q, &m.facets[fi], pt).1.psi};
            }
        }
        self.kinetic_of(&self.e_bodies, v) + u + self.vg_of(&self.e_bodies, q)
    }

    /// Scale turning reduced coordinates into metres: c → 1, φ → patch radius,
    /// η → 1/√(patch mass) (the basis is M-orthonormal).
    fn z_scale(&self) -> Vec<f64> {
        let mut sc = vec![1.0; self.rs.n];
        for (k, &pid) in self.rs.closed.iter().enumerate() {
            let pa = &self.patches[pid];
            let o = self.rs.off[k];
            sc[o + 2] = pa.radius.max(self.sc.dmin);
            for j in 0..pa.me {
                sc[o + 3 + j] = 1.0 / pa.mass.sqrt();
            }
        }
        sc
    }

    /// Interface energy jump: facet energy of the interface with the active zone at
    /// its end state and the R side at `z_b` minus the same with the R side at `z_a`.
    fn iface_jump(&self, z_a: &[f64], z_b: &[f64]) -> f64 {
        let m = &self.sc.m;
        let mut qa = self.q.clone();
        let mut qb = self.q.clone();
        for &i in &self.rs.iface_cells {
            qa[i] = self.rq(i, z_a);
            qb[i] = self.rq(i, z_b);
        }
        let mut du = 0.0;
        for &fi in &self.rs.iface_facets {
            for pt in 0..2 {
                du += m.eval_point(&qb, &m.facets[fi], pt).1.psi - m.eval_point(&qa, &m.facets[fi], pt).1.psi;
            }
        }
        du
    }

    /// Broken material facets (κ ≥ w_c) whose cells have separated (beyond half the
    /// facet length), slid (a quarter) or turned (0.2 rad) against each other are
    /// dropped; contact between those cells goes to the general contact.
    pub fn drop_open_cracks(&mut self) -> usize {
        let m = &self.sc.m;
        let nc = m.n_cells();
        let mut drop = Vec::new();
        for (fi, f) in m.facets.iter().enumerate() {
            if !f.active || f.tag != 0 || f.a >= nc || f.b >= nc {
                continue;
            }
            if !f.pts.iter().all(|p| p.law.kind == LawKind::Damage && p.kappa_c >= p.law.wc) {
                continue;
            }
            // the two-point corotated geometry of a broken facet is meaningless once its
            // cells have separated or turned against each other: hand over to contact
            let (j, _) = m.eval_point(&self.q, f, 0);
            let (j2, _) = m.eval_point(&self.q, f, 1);
            let sep = j.dn.min(j2.dn) > 0.5 * f.len;
            let slid = j.dn.abs().max(j.dt.abs()).max(j2.dn.abs()).max(j2.dt.abs()) > 0.25 * f.len;
            let turned = (self.q[f.a][2] - self.q[f.b][2]).abs() > 0.2;
            // energy-neutral hand-over only: the facet carries no force (open) and the
            // cells' contact circles do not overlap
            let open = j.dn >= 0.0 && j2.dn >= 0.0;
            let clear = self.cm.as_ref().map(|cm| cm.geometry(f.a, f.b, &self.q).0 >= 0.0).unwrap_or(true);
            if (sep || slid || turned) && open && clear {
                drop.push(fi);
            }
        }
        if drop.is_empty() {
            return 0;
        }
        for &fi in &drop {
            self.sc.m.facets[fi].active = false;
            let (a, b) = (self.sc.m.facets[fi].a, self.sc.m.facets[fi].b);
            if let Some(cm) = self.cm.as_mut() {
                cm.excluded.remove(&(a.min(b), a.max(b)));
            }
        }
        self.adj = self.sc.m.neighbours();
        let n = drop.len();
        self.events.push(format!("t={:.5} drop {n} open cracks", self.t));
        self.rebuild();
        n
    }

    /// Generalised force the active zone exerts on R at the current state.
    fn iface_force(&mut self, z_n: &[f64]) -> Vec<f64> {
        let nb = self.sc.m.bodies.len();
        let mut f = vec![[0.0f64; 3]; nb];
        let mut fr = vec![[0.0f64; 3]; nb];
        let qw = self.q.clone();
        self.e_forces(&qw, self.t, &mut f, &mut fr, false);
        let mut g = vec![0.0; self.rs.n];
        for &i in &self.rs.iface_cells.clone() {
            let Rep::R { k, li, .. } = self.rs.rep[i] else { unreachable!() };
            let (d, _, _) = self.rs.cell_jac(&self.patches, k, li, z_n);
            let off = self.rs.off[k];
            for (jj, dj) in d.iter().enumerate() {
                g[off + jj] += dj[0] * fr[i][0] + dj[1] * fr[i][1] + dj[2] * fr[i][2];
            }
        }
        g
    }

    /// Refresh the contact candidate list; rebuild if a pair entered the close range
    /// without being a candidate. Returns true if the system was rebuilt.
    pub fn refresh_contacts(&mut self) -> bool {
        let Some(cm) = &self.cm else { return false };
        // margins cover the relative motion over a macro step
        let vmax = (0..self.v.len()).map(|i| cm.surface_speed(i, &self.v)).fold(0.0f64, f64::max);
        let reach = 2.0 * vmax * self.p.h_macro;
        let near = cm.near_pairs(&self.q, 0.25 * self.sc.dmin + reach);
        if near == self.cpairs {
            return false;
        }
        // Remove obsolete candidates too. Retaining an old wide list forever
        // made a window entered from a large H several times dearer than E alone.
        self.cpairs = near;
        self.w.contact_flops += (self.sc.m.bodies.len() * 40) as f64;
        self.rebuild();
        true
    }

    /// Contact penalty energy at q.
    pub fn contact_energy(&self, q: &[[f64; 3]]) -> f64 {
        let Some(cm) = &self.cm else { return 0.0 };
        self.cpairs.iter().map(|&(a, b)| cm.pair_energy(a, b, q)).sum()
    }

    /// Frozen contact items for an implicit step (pairs with both sides in R).
    pub fn contact_items(&self) -> Vec<CItem> {
        let Some(cm) = &self.cm else { return Vec::new() };
        self.cpairs
            .iter()
            .filter(|&&(a, b)| matches!(self.rs.rep[a], Rep::R { .. }) && (b == GROUND || matches!(self.rs.rep[b], Rep::R { .. })))
            .map(|&(a, b)| {
                let (_, n) = cm.geometry(a, b, &self.q);
                CItem { a, b, n0: n, lam: cm.lagged(a, b, &self.q), q0a: self.q[a], q0b: if b == GROUND { [0.0; 3] } else { self.q[b] } }
            })
            .collect()
    }

    /// Conservative normal contact forces. Friction is a separate passive,
    /// canonically ordered impulse update after the conservative kick.
    fn e_contact_forces(&mut self, qw: &[[f64; 3]], f: &mut [[f64; 3]]) -> f64 {
        let Some(cm) = &self.cm else { return 0.0 };
        for &(a,b) in &self.cpairs {
            let ea=self.rs.rep[a]==Rep::E;
            let eb=b!=GROUND && self.rs.rep[b]==Rep::E;
            if !ea && !eb {continue;}
            if b==GROUND {
                for (gap,w,k) in cm.ground_points(a,qw) {
                    if gap<0.0 {let force=-k*gap; f[a][1]+=force; f[a][2]+=w.x*force;}
                }
            } else {
                let (gap,n)=cm.geometry(a,b,qw);let force=cm.normal(a,b,gap).0;
                if ea {f[a][0]-=force*n.x;f[a][1]-=force*n.y;}
                if eb {f[b][0]+=force*n.x;f[b][1]+=force*n.y;}
            }
        }
        self.w.contact_flops+=40.0*self.cpairs.len() as f64;
        0.0
    }

    /// Sequential impulses see the velocity updated by earlier contact points.
    /// Each impulse removes J*s - 1/2 J^2*(J M^-1 J^T) >= 0 kinetic energy;
    /// unlike simultaneous independently capped forces, their cross terms cannot
    /// over-brake or disappear from the friction ledger. Same Huber slip law as I.
    fn contact_friction(&mut self, qw: &[[f64;3]], h: f64) {
        let Some(cm)=&self.cm else {return};
        if cm.mu==0.0 {return;}
        let m=&self.sc.m;
        for &(a,b) in &self.cpairs {
            let ea=self.rs.rep[a]==Rep::E;
            let eb=b!=GROUND && self.rs.rep[b]==Rep::E;
            if !ea && !eb {continue;}
            if b==GROUND {
                for (gap,w,k) in cm.ground_points(a,qw) {
                    if gap>=0.0 {continue;}
                    let s=self.vh[a][0]-w.y*self.vh[a][2];
                    let inv=1.0/m.bodies[a].mass+w.y*w.y/m.bodies[a].inertia;
                    let cap=cm.mu*(-k*gap)*h*(s.abs()/cm.ev).min(1.0);
                    let impulse=(s.abs()/inv).min(cap)*s.signum();
                    self.vh[a][0]-=impulse/m.bodies[a].mass;
                    self.vh[a][2]+=w.y*impulse/m.bodies[a].inertia;
                    self.led.d_fric+=impulse*s-0.5*inv*impulse*impulse;
                }
            } else {
                let (gap,n)=cm.geometry(a,b,qw);let lam=cm.normal(a,b,gap).0;
                if lam<=0.0 {continue;}
                let (ja,jb)=cm.slip_jac(a,b,n);
                let s=cm.slip_rate(a,b,n,&self.vh);
                let mut inv=0.0;
                for (i,j,free) in [(a,ja,ea),(b,jb,eb)] {
                    if free {for k in 0..3 {if !m.bodies[i].fixed[k] {inv+=j[k]*j[k]*self.minv[i][k];}}}
                }
                if inv==0.0 {continue;}
                let cap=cm.mu*lam*h*(s.abs()/cm.ev).min(1.0);
                let impulse=(s.abs()/inv).min(cap)*s.signum();
                for (i,j,free) in [(a,ja,ea),(b,jb,eb)] {if free {for k in 0..3 {self.vh[i][k]-=impulse*j[k]*self.minv[i][k];}}}
                self.led.d_fric+=impulse*s-0.5*inv*impulse*impulse;
            }
        }
        self.w.contact_flops+=40.0*self.cpairs.len() as f64;
    }

    /// Remove support facets whose time has come (Scene.removals). Their elastic energy
    /// leaves the system as work of the external agent. Returns true if any was removed.
    pub fn apply_removals(&mut self) -> bool {
        let mut any = false;
        let rem = self.sc.removals.clone();
        for (tag, tr) in rem {
            if self.t + 1e-12 < tr || self.removed_tags.contains(&tag) {
                continue;
            }
            let mut u = 0.0;
            for fi in 0..self.sc.m.facets.len() {
                if self.sc.m.facets[fi].tag == tag && self.sc.m.facets[fi].active {
                    for pt in 0..2 {
                        u += self.sc.m.eval_point(&self.q, &self.sc.m.facets[fi], pt).1.psi;
                    }
                    self.sc.m.facets[fi].active = false;
                }
            }
            self.led.w_ext -= u;
            self.removed_tags.push(tag);
            self.events.push(format!("t={:.5} remove support tag {tag} (U = {u:.3e} J)", self.t));
            any = true;
        }
        if any {
            self.adj = self.sc.m.neighbours();
            self.rebuild();
            self.restart = true;
        }
        any
    }

    /// The next removal time after t, if any (macro steps land on it).
    pub fn next_removal(&self) -> Option<f64> {
        self.sc.removals.iter().filter(|(tag, tr)| *tr > self.t + 1e-12 && !self.removed_tags.contains(tag)).map(|x| x.1).fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.min(b))))
    }

    fn kinetic_of(&self, bodies: &[usize], v: &[[f64; 3]]) -> f64 {
        bodies.iter().map(|&i| {
            let b = &self.sc.m.bodies[i];
            0.5 * b.mass * (v[i][0] * v[i][0] + v[i][1] * v[i][1]) + 0.5 * b.inertia * v[i][2] * v[i][2]
        }).sum()
    }

    fn vg_of(&self, bodies: &[usize], q: &[[f64; 3]]) -> f64 {
        let m = &self.sc.m;
        bodies.iter().map(|&i| -(m.loads[i][0] * (q[i][0] - m.bodies[i].x0.x) + m.loads[i][1] * (q[i][1] - m.bodies[i].x0.y))).sum()
    }

    fn r_cells(&self) -> Vec<usize> {
        (0..self.sc.m.bodies.len()).filter(|&i| matches!(self.rs.rep[i], Rep::R { .. })).collect()
    }

    /// Energy of R's own terms: KE (cell-level velocities), eval facets + K̂, gravity.
    fn r_energy(&self, q: &[[f64; 3]], v: &[[f64; 3]], z: &[f64]) -> f64 {
        let rc = self.r_cells();
        let mut ec = 0.0;
        if let Some(cm) = &self.cm {
            for &(a, b) in &self.cpairs {
                if matches!(self.rs.rep[a], Rep::R { .. }) && (b == GROUND || matches!(self.rs.rep[b], Rep::R { .. })) {
                    ec += cm.pair_energy(a, b, q);
                }
            }
        }
        self.kinetic_of(&rc, v) + self.rs.elastic_energy(&self.sc.m, &self.patches, q, z) + self.vg_of(&rc, q) + ec
    }

    /// Contact-type points touching R cells predicted to close by more than ½δ₀ within
    /// 2H. Returns the R cells concerned.
    fn contact_events(&self, h: f64) -> Vec<usize> {
        let m = &self.sc.m;
        let mut out = Vec::new();
        let lists = [&self.rs.eval_facets, &self.rs.iface_facets];
        for list in lists {
            for &fi in list.iter() {
                let fa = &m.facets[fi];
                let contact_like = (fa.tag != 0 && fa.pts[0].law.kind == LawKind::Damage) || (fa.tag == 0 && fa.pts.iter().any(|p| p.law.damage(p.kappa_c) >= 0.99));
                if !contact_like {
                    continue;
                }
                let d0 = if fa.tag == 0 { fa.pts[0].law.delta0 } else { self.delta0_ref };
                for pt in 0..2 {
                    let (j, _) = m.eval_point(&self.q, fa, pt);
                    let s: f64 = -(0..3).map(|k| j.jna[k] * self.v[fa.a][k] + j.jnb[k] * self.v[fa.b][k]).sum::<f64>();
                    let pen_now = (-j.dn).max(0.0);
                    let pen_pred = (-(j.dn - 2.0 * h * s)).max(0.0);
                    if pen_pred - pen_now > 0.5 * d0 {
                        for b in [fa.a, fa.b] {
                            if matches!(self.rs.rep[b], Rep::R { .. }) {
                                out.push(b);
                            }
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Open the patches of the given cells and of hot facets, plus every closed patch
    /// whose centroid lies within r_e of a hot point.
    fn open_around(&mut self, cells: &[usize], points: &[V2], why: &str) -> usize {
        let mut set = std::collections::BTreeSet::new();
        for &i in cells {
            if self.is_cell(i) {
                set.insert(self.cell_patch[i]);
            }
        }
        for (pid, pa) in self.patches.iter().enumerate() {
            if !self.alive[pid] || self.open[pid] {
                continue;
            }
            let c = pa.xc;
            if points.iter().any(|&x| (x - c).norm() < self.p.r_e + pa.radius) {
                set.insert(pid);
            }
        }
        let mut n = 0;
        for pid in set {
            if self.alive[pid] && !self.open[pid] {
                self.open[pid] = true;
                self.open_since[pid] = self.t;
                if self.closed_once[pid] {
                    self.reopen[pid] += 1;
                }
                for &i in &self.patches[pid].cells {
                    self.fresh[i] = true;
                }
                n += 1;
            }
        }
        if n > 0 {
            self.w.activations += n as u64;
            if self.verbose {
                eprintln!("  t={:.5} open {n} patches ({why})", self.t);
            }
            self.events.push(format!("t={:.5} open {n} ({why})", self.t));
        }
        n
    }

    /// One macro step. Returns false on failure (step below 1e-7 s).
    pub fn macro_step(&mut self, t_end: f64) -> bool {
        assert!(self.carrier.is_none() || (self.rs.n==0 && !self.p.implicit),"carrier has no reduced-state/rollback integration");
        self.refresh_contacts();
        loop {
            let hs = self.h.min(t_end - self.t);
            // General E/R contacts are not part of the legacy Robin interface.
            // Promote their R side instead of silently dropping its reaction.
            if let Some(cm)=&self.cm {
                let cells: Vec<usize> = self.cpairs.iter().filter(|&&(a,b)| b!=GROUND && (self.rs.rep[a]==Rep::E || self.rs.rep[b]==Rep::E) && cm.may_contact(a,b,&self.q,&self.v,hs))
                    .flat_map(|&(a,b)|[a,b]).filter(|&i|matches!(self.rs.rep[i],Rep::R{..})).collect();
                if !cells.is_empty() && self.open_around(&cells,&[],"general E/R contact")>0 {self.rebuild();continue;}
            }
            // contact prediction on R cells
            if self.p.open0 != Open0::Never && self.rs.n > 0 {
                let ev = self.contact_events(hs);
                if !ev.is_empty() {
                    let pts: Vec<V2> = ev.iter().map(|&i| self.sc.m.bodies[i].x0).collect();
                    if self.open_around(&ev, &pts, "contact") > 0 {
                        self.rebuild();
                        continue;
                    }
                }
            }
            let snap = self.snapshot();
            match self.try_step(hs) {
                StepOutcome::Accepted => return true,
                StepOutcome::Activate(cells, pts, why) => {
                    self.restore(&snap);
                    self.w.rejected += 1;
                    if self.open_around(&cells, &pts, why) == 0 {
                        // nothing left to open: accept is impossible; treat as failure of the model
                        self.events.push(format!("t={:.5} monitor fired with nothing to open", self.t));
                        return false;
                    }
                    self.rebuild();
                }
                StepOutcome::Retry(hn) => {
                    self.restore(&snap);
                    self.w.rejected += 1;
                    self.h = hn;
                }
                StepOutcome::Failed => {
                    self.restore(&snap);
                    self.w.rejected += 1;
                    self.h = 0.5 * hs;
                    self.restart = true;
                    if self.h < 1e-7 {
                        return false;
                    }
                }
            }
        }
    }

    /// One macro step. R: implicit midpoint (unknown z at t_n + H/2, predictor
    /// x̂ = x_n + (H/2) v_n, α = 4), the partner of the impulse-matched coupling: the
    /// interface force enters as the step's mean, the same as E's sub-cycled impulse.
    fn try_step(&mut self, hs: f64) -> StepOutcome {
        let nb = self.sc.m.bodies.len();
        let mut qhat = self.q.clone();
        for i in 0..nb {
            if let Rep::R { .. } = self.rs.rep[i] {
                qhat[i] = std::array::from_fn(|k| self.q[i][k] + 0.5 * hs * self.v[i][k]);
            }
        }
        let mut w = self.w;
        let sd = self.rs.step_data(&self.patches, &qhat, 4.0 / (hs * hs), &mut w);
        let z_n = self.rs.fit(&self.patches, &self.q, &mut w);
        self.w = w;
        let ext_mid = self.sc.ext(self.t + 0.5 * hs);
        let have_e = !self.e_bodies.is_empty();
        let have_r = self.rs.n > 0;
        let e_r0 = if have_r { self.r_energy(&self.q, &self.v, &z_n) } else { 0.0 };
        let dbg_e = std::env::var("SB_DEBUGE").is_ok();
        let e_e0 = if dbg_e { self.e_energy(&self.q, &self.v) } else { 0.0 };
        let ns = if have_e { ((hs / self.he) * (1.0 - 1e-12)).ceil().max(1.0) as usize } else { 0 };
        if have_e && self.p.h_e_fixed.is_none() && (hs / ns as f64 - self.he).abs() > 1e-15 * self.he {
            // align the sub-step with the macro step (leapfrog re-synchronised)
            self.he = hs / ns as f64;
            for &i in &self.e_bodies {
                self.fresh[i] = true;
            }
            self.sub_t0 = self.t;
            self.sub_count = 0;
        }
        let mut zm = if have_r { self.rs.fit(&self.patches, &qhat, &mut self.w) } else { Vec::new() };
        let mut fj_used: Vec<f64> = vec![0.0; self.rs.n];
        let mut w_e = 0.0;
        let mut w_ext_e = 0.0;
        let tol = self.tol;
        if have_r && !have_e {
            let ex = Extra { ext: &ext_mid, fj: None, robin: None, contacts: None };
            if !self.rs.solve(&self.sc.m, &self.patches, &mut zm, &sd, &ex, tol, 40, &mut self.w) {
                return StepOutcome::Failed;
            }
        } else if have_e && !have_r {
            if self.cm.is_some() {
                w_ext_e=self.subcycle_contact(ns);
            } else {
                let (_, we, wx) = self.subcycle(&z_n, &z_n, ns);
                w_e = we;
                w_ext_e = wx;
            }
        } else {
            // impulse-matched multirate coupling
            let e_snap = (
                self.q.clone(),
                self.v.clone(),
                self.vh.clone(),
                self.fresh.clone(),
                self.e_facets.iter().map(|&fi| (fi, [self.sc.m.facets[fi].pts[0].kappa_c, self.sc.m.facets[fi].pts[1].kappa_c])).collect::<Vec<_>>(),
                self.sc.m.dissipated,
                self.sub_count,
                self.e_facets.iter().map(|&fi| (fi, self.last_growth[fi])).collect::<Vec<_>>(),
                self.led,
            );
            let rob = self.robin(&z_n, hs);
            // predictor: E's force on R linearised about the start state,
            // F(z) ≈ F(t_n) − S (z − z_n), with F(t_n) evaluated exactly (or, if the
            // system is unchanged, last step's mean force about last step's midpoint)
            let (fj0, zc) = match (&self.j_prev, &self.zm_prev) {
                (Some((v1, fj)), Some((v2, zp))) if *v1 == self.rs.version && *v2 == self.rs.version && fj.len() == self.rs.n => (fj.clone(), zp.clone()),
                _ => (self.iface_force(&z_n), z_n.clone()),
            };
            {
                let ex = Extra { ext: &ext_mid, fj: Some((&fj0[..], &z_n[..])), robin: Some((&rob[..], &zc[..])), contacts: None };
                if !self.rs.solve(&self.sc.m, &self.patches, &mut zm, &sd, &ex, tol, 40, &mut self.w) {
                    return StepOutcome::Failed;
                }
            }
            let mut zmk = zm.clone();
            // IQN-ILS history (scaled residuals and outputs of this step)
            let scale = self.z_scale();
            let mut hist: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
            let omega = 1.0;
            let mut converged = false;
            let mut err_prev = f64::INFINITY;
            for it in 0..self.p.max_coupling {
                self.q.clone_from(&e_snap.0);
                self.v.clone_from(&e_snap.1);
                self.vh.clone_from(&e_snap.2);
                self.fresh.clone_from(&e_snap.3);
                for &(fi, k) in &e_snap.4 {
                    self.sc.m.facets[fi].pts[0].kappa_c = k[0];
                    self.sc.m.facets[fi].pts[1].kappa_c = k[1];
                }
                self.sc.m.dissipated = e_snap.5;
                self.sub_count = e_snap.6;
                for &(fi, t) in &e_snap.7 {
                    self.last_growth[fi] = t;
                }
                self.led = e_snap.8;
                let zk: Vec<f64> = zmk.iter().zip(&z_n).map(|(a, b)| 2.0 * a - b).collect();
                let (jimp, we, wx) = self.subcycle(&z_n, &zk, ns);
                w_e = we;
                w_ext_e = wx;
                self.w.coupling_iters += 1;
                let fj: Vec<f64> = jimp.iter().map(|v| v / hs).collect();
                let mut zmn = zmk.clone();
                {
                    let ex = Extra { ext: &ext_mid, fj: Some((&fj[..], &z_n[..])), robin: Some((&rob[..], &zmk[..])), contacts: None };
                    if !self.rs.solve(&self.sc.m, &self.patches, &mut zmn, &sd, &ex, tol, 40, &mut self.w) {
                        return StepOutcome::Failed;
                    }
                }
                let z1: Vec<f64> = zmn.iter().zip(&z_n).map(|(a, b)| 2.0 * a - b).collect();
                let mut err: f64 = 0.0;
                for &i in &self.rs.iface_cells {
                    let (a, b) = (self.rq(i, &z1), self.rq(i, &zk));
                    err = err.max((a[0] - b[0]).abs()).max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs() * 0.5 * self.sc.dmin);
                }
                if std::env::var("SB_DEBUGC").is_ok() {
                    let jn: f64 = jimp.iter().map(|v| v * v).sum::<f64>().sqrt();
                    eprintln!("    t={:.5} it {it} err {err:.3e} tol {:.3e} omega {omega:.3} |J| {jn:.3e} we {we:.3e}", self.t, self.p.coupling_tol * self.iface_d0);
                }
                // force residual of the Robin term: the interface force must be consistent,
                // not only the positions (a stiff active zone turns 1e-8 m into kN)
                let mut rf = vec![0.0; self.rs.n];
                for &(r, c, v) in &rob {
                    rf[r] += v * (zmn[c] - zmk[c]);
                }
                let rf_max = rf.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
                let fj_max = fj.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
                if std::env::var("SB_DEBUGC").is_ok() {
                    eprintln!("      rf {rf_max:.3e} fj {fj_max:.3e}");
                }
                let _ = (rf_max, fj_max);
                if err <= self.p.coupling_tol * self.iface_d0 && rf_max <= self.p.coupling_ftol * fj_max + tol {
                    // candidate: R without the Robin term (receives exactly −E's impulse); accept
                    // if the interface energy jump between what E saw (zk) and where R ends is
                    // within the per-step budget, otherwise keep iterating
                    let mut zf = zmn.clone();
                    let ex = Extra { ext: &ext_mid, fj: Some((&fj[..], &z_n[..])), robin: None, contacts: None };
                    if !self.rs.solve(&self.sc.m, &self.patches, &mut zf, &sd, &ex, tol, 40, &mut self.w) {
                        return StepOutcome::Failed;
                    }
                    let zf1: Vec<f64> = zf.iter().zip(&z_n).map(|(a, b)| 2.0 * a - b).collect();
                    let du = self.iface_jump(&zk, &zf1);
                    if std::env::var("SB_DEBUGC").is_ok() {
                        eprintln!("      jump dU {du:.3e} budget {:.3e}", self.p.coupling_ebudget * self.sc.e_scale);
                    }
                    if du <= self.p.coupling_ebudget * self.sc.e_scale {
                        self.led.d_coupling -= du;
                        zm = zf;
                        fj_used = fj;
                        converged = true;
                        self.last_iters = it + 1;
                        break;
                    }
                }
                // poor contraction: the macro step does not resolve the interface dynamics
                if it >= 2 && err > 0.5 * err_prev {
                    break;
                }
                err_prev = err;
                self.last_iters = it + 2;
                // interface quasi-Newton (IQN-ILS): least-squares inverse Jacobian of the
                // fixed-point map from this step's iterates, in scaled (metre) coordinates
                let r: Vec<f64> = zmn.iter().zip(&zmk).zip(&scale).map(|((a, b), s)| (a - b) * s).collect();
                let xt: Vec<f64> = zmn.iter().zip(&scale).map(|(a, s)| a * s).collect();
                let mut next: Vec<f64> = xt.clone();
                if !hist.is_empty() {
                    let (rl, xl) = hist.last().unwrap();
                    let mut v_cols: Vec<Vec<f64>> = Vec::new();
                    let mut w_cols: Vec<Vec<f64>> = Vec::new();
                    v_cols.push(r.iter().zip(rl).map(|(a, b)| a - b).collect());
                    w_cols.push(xt.iter().zip(xl).map(|(a, b)| a - b).collect());
                    for k in (1..hist.len()).rev() {
                        let (ra, xa) = &hist[k];
                        let (rb, xb) = &hist[k - 1];
                        v_cols.push(ra.iter().zip(rb).map(|(a, b)| a - b).collect());
                        w_cols.push(xa.iter().zip(xb).map(|(a, b)| a - b).collect());
                    }
                    // min ‖V c + r‖ by modified Gram–Schmidt QR, dropping dependent columns
                    let mut q: Vec<Vec<f64>> = Vec::new();
                    let mut rr: Vec<Vec<f64>> = Vec::new();
                    let mut keep: Vec<usize> = Vec::new();
                    for (ci, v) in v_cols.iter().enumerate() {
                        let n0 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                        let mut u = v.clone();
                        let mut coef = vec![0.0; q.len()];
                        for (j, qj) in q.iter().enumerate() {
                            let d: f64 = u.iter().zip(qj).map(|(a, b)| a * b).sum();
                            coef[j] = d;
                            for (a, b) in u.iter_mut().zip(qj) {
                                *a -= d * b;
                            }
                        }
                        let n1 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
                        if n0 > 0.0 && n1 > 1e-10 * n0 {
                            for a in u.iter_mut() {
                                *a /= n1;
                            }
                            coef.push(n1);
                            q.push(u);
                            rr.push(coef);
                            keep.push(ci);
                        }
                    }
                    if !q.is_empty() {
                        let k = q.len();
                        let qtr: Vec<f64> = q.iter().map(|qj| -qj.iter().zip(&r).map(|(a, b)| a * b).sum::<f64>()).collect();
                        let mut c = vec![0.0; k];
                        for i in (0..k).rev() {
                            let mut sum = qtr[i];
                            for j in i + 1..k {
                                sum -= rr[j][i] * c[j];
                            }
                            c[i] = sum / rr[i][i];
                        }
                        for (j, &ci) in keep.iter().enumerate() {
                            for (a, b) in next.iter_mut().zip(&w_cols[ci]) {
                                *a += c[j] * b;
                            }
                        }
                    }
                }
                hist.push((r, xt));
                for ((z, x), s) in zmk.iter_mut().zip(&next).zip(&scale) {
                    *z = x / s;
                }
                let _ = omega;
            }
            if !converged {
                // the patches at the interface cannot follow the active zone at this macro
                // step: they join it (the interface moves away from the strong coupling)
                if self.verbose {
                    eprintln!("  t={:.5} coupling not converged (h={hs:.3e})", self.t);
                }
                // an E–R interface cutting through material faster than the macro step does
                // not converge: the interface patches join the active zone
                if self.p.open0 == Open0::Never || self.rs.iface_cells.is_empty() {
                    return StepOutcome::Failed;
                }
                let cells = self.rs.iface_cells.clone();
                let pts: Vec<V2> = Vec::new();
                return StepOutcome::Activate(cells, pts, "coupling");
            }
            self.j_prev = Some((self.rs.version, fj_used.clone()));
            self.zm_prev = Some((self.rs.version, zm.clone()));
        }
        let z1: Vec<f64> = zm.iter().zip(&z_n).map(|(a, b)| 2.0 * a - b).collect();
        let mut qn = self.q.clone();
        let mut vn = self.v.clone();
        if have_r {
            self.rs.q_all(&self.patches, &z1, &mut qn, &mut self.w);
            for i in 0..nb {
                if let Rep::R { .. } = self.rs.rep[i] {
                    vn[i] = std::array::from_fn(|k| 2.0 * (qn[i][k] - self.q[i][k]) / hs - self.v[i][k]);
                }
            }
            let trig = if self.p.open0 != Open0::Never { self.p.chi_trig } else { f64::INFINITY };
            let (chi, hot) = self.rs.monitor(&self.sc.m, &self.patches, &qn, trig, &mut self.w);
            if !hot.is_empty() {
                let m = &self.sc.m;
                let cells: Vec<usize> = hot.iter().flat_map(|&fi| [m.facets[fi].a, m.facets[fi].b]).filter(|&i| matches!(self.rs.rep[i], Rep::R { .. })).collect();
                let pts: Vec<V2> = hot.iter().map(|&fi| m.facets[fi].pc).collect();
                return StepOutcome::Activate(cells, pts, "monitor");
            }
            self.chi_r_max = self.chi_r_max.max(chi);
        }
        // accept: R work terms and ledger
        let mut w_ext_r = 0.0;
        for &(i, f) in &ext_mid {
            if let Rep::R { .. } = self.rs.rep[i] {
                for k in 0..3 {
                    w_ext_r += f[k] * (qn[i][k] - self.q[i][k]);
                }
            }
        }
        if have_r {
            let w_on_r: f64 = fj_used.iter().zip(z1.iter().zip(&z_n)).map(|(f, (a, b))| f * (a - b)).sum();
            let e_r1 = self.r_energy(&qn, &vn, &z1);
            self.led.d_num_r += e_r0 + w_on_r + w_ext_r - e_r1;
            self.led.w_iface_r += w_on_r;
        }
        if dbg_e {
            let mut qq = self.q.clone();
            for i in 0..nb {
                if let Rep::R { .. } = self.rs.rep[i] {
                    qq[i] = qn[i];
                }
            }
            let e_e1 = self.e_energy(&qq, &self.v);
            eprintln!("  t={:.5} E-balance {:+.4e} (dE_E {:+.4e} w_e {:+.4e} w_ext_e {:+.4e})  R-balance {:+.4e}", self.t, e_e1 - e_e0 - w_e - w_ext_e, e_e1 - e_e0, w_e, w_ext_e, 0.0);
        }
        self.led.w_iface_e += w_e;
        self.led.w_ext += w_ext_r + w_ext_e;
        if have_r {
            let m = &mut self.sc.m;
            for &fi in &self.rs.eval_facets {
                for pt in 0..2 {
                    let (_, e) = m.eval_point(&qn, &m.facets[fi], pt);
                    let p = &mut m.facets[fi].pts[pt];
                    if e.kappa > p.kappa_c {
                        let dd = p.law.dissipation(e.kappa) - p.law.dissipation(p.kappa_c);
                        m.dissipated += dd;
                        p.kappa_c = e.kappa;
                        if dd > 0.0 {
                            self.last_growth[fi] = self.t + hs;
                        }
                    }
                }
            }
            self.w.pts_force += 2 * self.rs.eval_facets.len() as u64;
        }
        for i in 0..nb {
            if let Rep::R { .. } = self.rs.rep[i] {
                self.qp[i] = self.q[i];
                self.vp[i] = self.v[i];
                self.q[i] = qn[i];
                self.v[i] = vn[i];
            }
        }
        self.t += hs;
        self.w.macro_steps += 1;
        // grow back only after a few easy steps (no step oscillation)
        if have_e && have_r {
            if self.last_iters <= 2 {
                self.easy_steps += 1;
            } else {
                self.easy_steps = 0;
            }
        } else {
            self.easy_steps += 1;
        }
        if self.h < self.p.h_macro && self.easy_steps >= 3 {
            self.h = (2.0 * self.h).min(self.p.h_macro);
            self.easy_steps = 0;
        }
        StepOutcome::Accepted
    }

    /// Energy terms at the current macro boundary: (KE, U, V_g).
    pub fn energies(&self) -> (f64, f64, f64) {
        let m = &self.sc.m;
        let nb = m.bodies.len();
        let all: Vec<usize> = (0..nb).collect();
        let ke = self.kinetic_of(&all, &self.v);
        let vg = self.vg_of(&all, &self.q);
        let mut u = 0.0;
        for &fi in &self.e_facets {
            for pt in 0..2 {
                u += if let Some(c)=self.carrier.as_ref() {c.energy(m,fi,pt,&self.q)} else {m.eval_point(&self.q, &m.facets[fi], pt).1.psi};
            }
        }
        if self.rs.n > 0 {
            let z = self.rs.fit(&self.patches, &self.q, &mut Work::default());
            u += self.rs.elastic_energy(m, &self.patches, &self.q, &z);
        }
        u += self.contact_energy(&self.q);
        (ke, u, vg)
    }

    /// Cells with their own DOFs: active-zone cells, or split (single-cell) patches.
    pub fn e_cells(&self) -> usize {
        if self.p.implicit {
            (0..self.patches.len()).filter(|&p| self.alive[p] && self.patches[p].cells.len() == 1 && self.is_cell(self.patches[p].cells[0])).count()
        } else {
            self.e_bodies.iter().filter(|&&i| self.is_cell(i)).count()
        }
    }

    // ------------------------------------------------------------ re-reduction

    /// Close quiet open patches (§1.5). Returns the number of patches closed.
    pub fn rereduce(&mut self) -> usize {
        if !self.p.rereduce || self.p.open0 == Open0::All {
            return 0;
        }
        let t = self.t;
        let quiet_t = self.p.quiet_steps as f64 * self.p.h_macro - 1e-12;
        let cands: Vec<usize> = (0..self.patches.len()).filter(|&pid| self.alive[pid] && self.open[pid] && t - self.open_since[pid] >= quiet_t * f64::powi(2.0, self.reopen[pid].min(3) as i32)).collect();
        if std::env::var("SB_DEBUGRR").is_ok() && (self.t * 1000.0).fract() < 0.3 && ((self.t * 1000.0) as i64) % 10 == 0 {
            let opens: Vec<String> = (0..self.patches.len()).filter(|&p| self.alive[p] && self.open[p]).map(|p| format!("{}:({:.2},{:.2}) re{} since{:.3}", p, self.patches[p].xc.x, self.patches[p].xc.y, self.reopen[p], self.open_since[p])).collect();
            eprintln!("  t={:.4} open patches: {}", self.t, opens.join(" "));
        }
        if cands.is_empty() {
            return 0;
        }
        let ncand = cands.len();
        let m = &self.sc.m;
        let nb = m.bodies.len();
        // contact events touching active cells (closing contact-type points)
        let mut busy = vec![false; nb];
        let dbg_rr = std::env::var("SB_DEBUGRR").is_ok();
        let mut why_busy = [0usize; 3];
        let mut why_block = [0usize; 4];
        for &fi in &self.e_facets {
            let fa = &m.facets[fi];
            let contact_like = (fa.tag != 0 && fa.pts[0].law.kind == LawKind::Damage) || (fa.tag == 0 && fa.pts.iter().any(|p| p.law.damage(p.kappa_c) >= 0.99));
            if contact_like {
                let d0 = if fa.tag == 0 { fa.pts[0].law.delta0 } else { self.delta0_ref };
                for pt in 0..2 {
                    let (j, e) = m.eval_point(&self.q, fa, pt);
                    let s: f64 = -(0..3).map(|k| j.jna[k] * self.v[fa.a][k] + j.jnb[k] * self.v[fa.b][k]).sum::<f64>();
                    let pen_now = (-j.dn).max(0.0);
                    let pen_pred = (-(j.dn - 2.0 * self.p.h_macro * s)).max(0.0);
                    // closing contact, or a loaded impactor pad
                    if pen_pred - pen_now > 0.5 * d0 || (fa.tag == 3 && e.fnorm < 0.0) {
                        busy[fa.a] = true;
                        busy[fa.b] = true;
                        if dbg_rr {
                            why_busy[0] += 1;
                        }
                    }
                }
            } else if fa.tag == 0 {
                // stress state and recent growth
                let mut hot = t - self.last_growth[fi] < quiet_t + self.p.h_macro;
                for p in 0..2 {
                    let pt = &fa.pts[p];
                    if pt.law.kind != LawKind::Damage || pt.kappa_c >= pt.law.wc {
                        continue;
                    }
                    let (j, _) = m.eval_point(&self.q, fa, p);
                    let dn = j.dn.max(0.0);
                    let chi = (dn * dn + pt.law.gamma * j.dt * j.dt).sqrt() / pt.kappa_c.max(pt.law.delta0);
                    // undamaged: margin to onset; damaged: margin to its own history
                    let lim = if pt.kappa_c > pt.law.delta0 { 0.9 } else { self.p.chi_rr };
                    if chi >= lim {
                        hot = true;
                    }
                }
                if hot {
                    busy[fa.a] = true;
                    busy[fa.b] = true;
                    if dbg_rr {
                        why_busy[if t - self.last_growth[fi] < quiet_t + self.p.h_macro { 1 } else { 2 }] += 1;
                    }
                }
            }
        }
        // a patch closes only if its whole neighbourhood (one facet ring) is quiet
        let mut busy2 = busy.clone();
        for (i, &b) in busy.iter().enumerate() {
            if b {
                for &fi in &self.adj[i] {
                    busy2[m.facets[fi].a] = true;
                    busy2[m.facets[fi].b] = true;
                }
            }
        }
        // and its neighbour patches are quiet too (the new interface must see quiet material)
        let mut pbusy = vec![false; self.patches.len()];
        for (i, &b) in busy2.iter().enumerate() {
            if b && self.is_cell(i) {
                pbusy[self.cell_patch[i]] = true;
            }
        }
        let mut busy = busy2.clone();
        for (pid, pa) in self.patches.iter().enumerate() {
            if !self.alive[pid] || !pbusy[pid] {
                continue;
            }
            for &i in &pa.cells {
                for &fi in &self.adj[i] {
                    let f = &m.facets[fi];
                    let intact = f.tag == 0 && f.pts.iter().any(|p| p.law.damage(p.kappa_c) < 0.99);
                    if !intact {
                        continue;
                    }
                    for c in [f.a, f.b] {
                        if self.is_cell(c) {
                            for &j in &self.patches[self.cell_patch[c]].cells {
                                busy[j] = true;
                            }
                        }
                    }
                }
            }
        }
        let mut closed_n = 0;
        let mut new_patches: Vec<(usize, Vec<Patch>, Vec<[f64; 3]>, Vec<[f64; 3]>, f64)> = Vec::new();
        for pid in cands {
            let cells = self.patches[pid].cells.clone();
            if cells.iter().any(|&i| busy[i]) {
                why_block[0] += 1;
                continue;
            }
            // components through undamaged material facets
            let mut loc = vec![usize::MAX; nb];
            for (li, &i) in cells.iter().enumerate() {
                loc[i] = li;
            }
            let mut par: Vec<usize> = (0..cells.len()).collect();
            for &i in &cells {
                for &fi in &self.adj[i] {
                    let f = &m.facets[fi];
                    if !f.active || f.tag != 0 || loc[f.a] == usize::MAX || loc[f.b] == usize::MAX {
                        continue;
                    }
                    if f.pts.iter().all(|p| p.law.damage(p.kappa_c) == 0.0) {
                        let (a, b) = (find(&mut par, loc[f.a]), find(&mut par, loc[f.b]));
                        if a != b {
                            par[a.max(b)] = a.min(b);
                        }
                    }
                }
            }
            let mut comps: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
            for (li, &i) in cells.iter().enumerate() {
                comps.entry(find(&mut par, li)).or_default().push(i);
            }
            let same = comps.len() == 1 && self.patches[pid].internal.iter().all(|&fi| m.facets[fi].pts.iter().all(|p| p.law.damage(p.kappa_c) == 0.0));
            let mut w = Work::default();
            let pats: Vec<Patch> = if same {
                vec![self.patches[pid].clone()]
            } else {
                comps.values().map(|c| build_patch(m, &self.adj, c, self.p.deg, self.p.nvib, &self.loaded, &mut w)).collect()
            };
            self.w.rebuild_flops += w.rebuild_flops;
            // project positions (K-orthogonal deformation) and velocities (M-orthogonal)
            let mut qn = self.q.clone();
            let mut vn = self.v.clone();
            for pa in &pats {
                let kop = |d: &[f64]| k_apply(m, pa, d);
                let y = fit_patch(pa, &self.q, true, Some(&kop as &dyn Fn(&[f64]) -> Vec<f64>));
                let ny = pa.ny();
                let mut gmg = vec![0.0; ny * ny];
                let mut gmv = vec![0.0; ny];
                let mut gs: Vec<Vec<[f64; 3]>> = Vec::with_capacity(pa.cells.len());
                for (li, &i) in pa.cells.iter().enumerate() {
                    let (r, wth) = pa.local(li, &y[3..]);
                    let s = r.rot(y[2]);
                    qn[i] = [pa.xc.x + y[0] + s.x, pa.xc.y + y[1] + s.y, y[2] + wth];
                    let mut d = vec![[0.0; 3]; ny];
                    d[0] = [1.0, 0.0, 0.0];
                    d[1] = [0.0, 1.0, 0.0];
                    let js = jperp(s);
                    d[2] = [js.x, js.y, 1.0];
                    for j in 0..pa.me {
                        let u = V2::new(pa.te[(3 * li) * pa.me + j], pa.te[(3 * li + 1) * pa.me + j]).rot(y[2]);
                        d[3 + j] = [u.x, u.y, pa.te[(3 * li + 2) * pa.me + j]];
                    }
                    let mm = [pa.mi[li], pa.mi[li], pa.ii[li]];
                    for a in 0..ny {
                        for b in 0..ny {
                            gmg[a * ny + b] += (0..3).map(|k| d[a][k] * mm[k] * d[b][k]).sum::<f64>();
                        }
                        gmv[a] += (0..3).map(|k| d[a][k] * mm[k] * self.v[i][k]).sum::<f64>();
                    }
                    gs.push(d);
                }
                let l = chol(&gmg, ny).expect("patch mass matrix");
                chol_solve(&l, ny, &mut gmv);
                for (li, &i) in pa.cells.iter().enumerate() {
                    vn[i] = std::array::from_fn(|k| (0..ny).map(|a| gs[li][a][k] * gmv[a]).sum());
                }
                self.w.rebuild_flops += (pa.cells.len() * ny * ny * 6) as f64;
            }
            new_patches.push((pid, pats, qn, vn, 0.0));
        }
        // close by connected open components (through intact facets): a coherent body is
        // closed whole or not at all, so no interface cuts through moving material; the
        // energy check is done on the whole component with every patch projected
        {
            let m = &self.sc.m;
            let np = self.patches.len();
            let mut par: Vec<usize> = (0..np).collect();
            for f in m.facets.iter() {
                if !f.active || f.tag != 0 || !self.is_cell(f.a) || !self.is_cell(f.b) {
                    continue;
                }
                if f.pts.iter().any(|p| p.law.damage(p.kappa_c) >= 0.99) {
                    continue;
                }
                let (pa, pb) = (self.cell_patch[f.a], self.cell_patch[f.b]);
                if pa != pb && self.open[pa] && self.open[pb] {
                    let (ra, rb) = (find(&mut par, pa), find(&mut par, pb));
                    if ra != rb {
                        par[ra.max(rb)] = ra.min(rb);
                    }
                }
            }
            let mut ok = vec![false; np];
            for (pid, ..) in &new_patches {
                ok[*pid] = true;
            }
            let mut bad = vec![false; np];
            for pid in 0..np {
                if self.alive[pid] && self.open[pid] && !ok[pid] {
                    let r = find(&mut par, pid);
                    bad[r] = true;
                }
            }
            // joint energy check per surviving component
            let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
            for (k, (pid, ..)) in new_patches.iter().enumerate() {
                let r = find(&mut par, *pid);
                if !bad[r] {
                    groups.entry(r).or_default().push(k);
                }
            }
            let mut keep = vec![false; new_patches.len()];
            for (_, ks) in groups {
                let mut cells: Vec<usize> = Vec::new();
                let mut qn = self.q.clone();
                let mut vn = self.v.clone();
                for &k in &ks {
                    let pid = new_patches[k].0;
                    for &i in &self.patches[pid].cells {
                        cells.push(i);
                        qn[i] = new_patches[k].2[i];
                        vn[i] = new_patches[k].3[i];
                    }
                }
                let mut fs: Vec<usize> = cells.iter().flat_map(|&i| self.adj[i].iter().copied()).filter(|&fi| m.facets[fi].active).collect();
                fs.sort_unstable();
                fs.dedup();
                let energy = |q: &[[f64; 3]], v: &[[f64; 3]]| -> (f64, f64) {
                    let (mut e, mut own) = (0.0, 0.0);
                    for &i in &cells {
                        let b = &m.bodies[i];
                        let ke = 0.5 * b.mass * (v[i][0] * v[i][0] + v[i][1] * v[i][1]) + 0.5 * b.inertia * v[i][2] * v[i][2];
                        e += ke - m.loads[i][0] * q[i][0] - m.loads[i][1] * q[i][1];
                        own += ke;
                    }
                    for &fi in &fs {
                        for pt in 0..2 {
                            let psi = m.eval_point(q, &m.facets[fi], pt).1.psi;
                            e += psi;
                            own += psi;
                        }
                    }
                    (e, own)
                };
                let (e0, own) = energy(&self.q, &self.v);
                let (e1, _) = energy(&qn, &vn);
                let loss = e0 - e1;
                if loss < -1e-6 * own - 1e-12 * self.sc.e_scale {
                    why_block[1] += ks.len();
                    continue;
                }
                if loss > (self.p.eps_rr * own).max(1e-4 * self.sc.e_scale) {
                    why_block[2] += ks.len();
                    if dbg_rr {
                        eprintln!("    rr component of {} patches: loss {loss:.3e} own {own:.3e}", ks.len());
                    }
                    continue;
                }
                // book the component's loss on its first patch
                new_patches[ks[0]].4 = loss;
                for &k in &ks {
                    keep[k] = true;
                }
            }
            let mut k = 0;
            new_patches.retain(|_| {
                let r = keep[k];
                k += 1;
                r
            });
        }
        if dbg_rr {
            eprintln!("  t={:.4} rr cands {} busy(contact {}, growth {}, chi {}) blocked(busy {}, creates {}, loss {}) pass {}", self.t, ncand, why_busy[0], why_busy[1], why_busy[2], why_block[0], why_block[1], why_block[2], new_patches.len());
        }
        if new_patches.is_empty() {
            return 0;
        }
        for (pid, pats, qn, vn, loss) in new_patches {
            let cells = self.patches[pid].cells.clone();
            for &i in &cells {
                self.q[i] = qn[i];
                self.v[i] = vn[i];
                self.qp[i] = qn[i];
                self.vp[i] = vn[i];
            }
            self.led.d_red += loss;
            if pats.len() == 1 && pats[0].cells.len() == cells.len() && {
                let mut a = pats[0].cells.clone();
                a.sort_unstable();
                let mut b = cells.clone();
                b.sort_unstable();
                a == b
            } {
                self.patches[pid] = pats.into_iter().next().unwrap();
                self.open[pid] = false;
                self.closed_once[pid] = true;
            } else {
                self.alive[pid] = false;
                self.open[pid] = false;
                for pa in pats {
                    let nid = self.patches.len();
                    for &i in &pa.cells {
                        self.cell_patch[i] = nid;
                    }
                    self.patches.push(pa);
                    self.alive.push(true);
                    self.open.push(false);
                    self.open_since.push(self.t);
                    self.reopen.push(self.reopen[pid]);
                    self.closed_once.push(true);
                }
            }
            closed_n += 1;
        }
        self.w.rereductions += closed_n as u64;
        if self.verbose {
            eprintln!("  t={:.5} re-reduced {closed_n} patches", self.t);
        }
        self.events.push(format!("t={:.5} close {closed_n}", self.t));
        self.rebuild();
        closed_n
    }
}

pub enum StepOutcome {
    Accepted,
    Activate(Vec<usize>, Vec<V2>, &'static str),
    Failed,
    /// redo with this macro step (local-error or event control)
    Retry(f64),
}

// ---------------------------------------------------------------- diagnostics

pub fn fragments(m: &Model, q: &[[f64; 3]], v: &[[f64; 3]], min_frac: f64) -> Vec<Fragment> {
    let nc = m.n_cells();
    let mut parent: Vec<usize> = (0..nc).collect();
    for f in m.facets.iter().filter(|f| f.active && f.tag == 0) {
        if f.pts.iter().any(|p| p.law.damage(p.kappa_c) < 0.99) {
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
        out.push(frag_of(m, &cells, q, v, true));
    }
    out.sort_by(|a, b| a.cm[0].partial_cmp(&b.cm[0]).unwrap());
    out
}

fn frag_of(m: &Model, cells: &[usize], q: &[[f64; 3]], v: &[[f64; 3]], separated: bool) -> Fragment {
    let mass: f64 = cells.iter().map(|&i| m.bodies[i].mass).sum();
    let cm = cells.iter().fold(V2::ZERO, |s, &i| s + V2::new(q[i][0], q[i][1]) * m.bodies[i].mass) * (1.0 / mass);
    let vc = cells.iter().fold(V2::ZERO, |s, &i| s + V2::new(v[i][0], v[i][1]) * m.bodies[i].mass) * (1.0 / mass);
    let (mut l, mut j) = (0.0, 0.0);
    for &i in cells {
        let r = V2::new(q[i][0], q[i][1]) - cm;
        let dv = V2::new(v[i][0], v[i][1]) - vc;
        l += m.bodies[i].mass * (r.x * dv.y - r.y * dv.x) + m.bodies[i].inertia * v[i][2];
        j += m.bodies[i].mass * r.dot(r) + m.bodies[i].inertia;
    }
    Fragment { separated, mass, cm: [cm.x, cm.y], v: [vc.x, vc.y], omega: l / j, cells: cells.len() }
}

/// The cantilever's tip fragment and whether it is separated from the clamp.
pub fn tip_fragment(sc: &Scene, m: &Model, q: &[[f64; 3]], v: &[[f64; 3]]) -> Fragment {
    let nc = m.n_cells();
    let mut parent: Vec<usize> = (0..nc).collect();
    for f in m.facets.iter().filter(|f| f.active && f.tag == 0) {
        if f.pts.iter().any(|p| p.law.damage(p.kappa_c) < 0.99) {
            let (a, b) = (find(&mut parent, f.a), find(&mut parent, f.b));
            if a != b {
                parent[a] = b;
            }
        }
    }
    let root = find(&mut parent, sc.tip_cells[0]);
    let cells: Vec<usize> = (0..nc).filter(|&i| find(&mut parent, i) == root).collect();
    let separated = !sc.clamp_cells.iter().any(|&c| find(&mut parent, c) == root);
    frag_of(m, &cells, q, v, separated)
}

pub fn probe1(sc: &Scene, m: &Model, q: &[[f64; 3]], v: &[[f64; 3]]) -> f64 {
    match sc.impactor {
        Some(p) => v[p][1],
        None => {
            let (mut s, mut w) = (0.0, 0.0);
            for &i in &sc.tip_cells {
                s += m.bodies[i].mass * (q[i][1] - m.bodies[i].x0.y);
                w += m.bodies[i].mass;
            }
            s / w
        }
    }
}

pub fn probe2(sc: &Scene, m: &Model, q: &[[f64; 3]]) -> f64 {
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
        None => {
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
    }
}

pub fn crack_stats(m: &Model) -> [f64; 6] {
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

/// Run to t_end recording every `rec` seconds.
pub fn run(mut wd: World, label: &str, t_end: f64, rec: f64) -> (Traj, World) {
    let t0 = Instant::now();
    let nc = wd.sc.m.n_cells();
    let mut tr = Traj { label: label.into(), scene: wd.sc.name.clone(), cells: nc, patches: wd.patches.len(), h_e: wd.he, ..Default::default() };
    let mut e_max: f64 = f64::NEG_INFINITY;
    let mut next_rec = 0.0;
    let mut e_frac_int = 0.0;
    let p0 = momentum(&wd);
    let record = |wd: &World, tr: &mut Traj, e_max: &mut f64| {
        let (ke, u, vg) = wd.energies();
        let m = &wd.sc.m;
        let e = ke + u + vg + m.dissipated + wd.led.d_red + wd.led.d_fric + wd.led.d_rad + wd.led.d_model - wd.led.w_ext;
        if tr.rows.is_empty() {
            tr.e0 = e;
        }
        *e_max = e_max.max(e);
        tr.ke_max = tr.ke_max.max(ke);
        let closed = (0..wd.patches.len()).filter(|&p| wd.alive[p] && !wd.open[p] && wd.patches[p].cells.len() > 1).count();
        tr.rows.push([wd.t, probe1(&wd.sc, m, &wd.q, &wd.v), ke, u, vg, m.dissipated, wd.led.w_ext, e, wd.e_cells() as f64, probe2(&wd.sc, m, &wd.q), wd.led.d_red, wd.led.d_fric, wd.led.d_num_r, closed as f64, wd.w.coupling_iters as f64, wd.h]);
    };
    record(&wd, &mut tr, &mut e_max);
    next_rec += rec;
    tr.max_e_cells = wd.e_cells();
    tr.startup_work = wd.w.total();
    let run_start = wd.t;
    while wd.t < t_end - 1e-12 {
        let t_before = wd.t;
        let work_before = wd.w;
        let step_clock = Instant::now();
        let ec = wd.e_cells();
        wd.apply_removals();
        let t_stop = wd.next_removal().map_or(t_end, |tr| tr.min(t_end));
        let ok = if wd.p.implicit && wd.window.is_none() { wd.macro_step_implicit(t_stop) } else { wd.macro_step(t_stop) };
        if !ok {
            tr.failed_at = Some(wd.t);
            eprintln!("  {label}: macro step failed at t={:.6}", wd.t);
            break;
        }
        let dt = wd.t - t_before;
        e_frac_int += ec as f64 * dt;
        if tr.onset_t.is_none() && wd.sc.m.dissipated > 0.0 {
            tr.onset_t = Some(wd.t);
        }
        wd.drop_open_cracks();
        if wd.p.implicit {
            wd.cost_switch();
            if wd.window.is_none() {
                wd.merge();
            }
        } else {
            wd.rereduce();
        }
        let frame = (t_before * 60.0).floor() as usize;
        if tr.frames.last().map(|f| f.frame) != Some(frame) {
            tr.frames.push(FrameCost { frame, ..Default::default() });
        }
        let fc = tr.frames.last_mut().unwrap();
        fc.work += wd.w.total() - work_before.total();
        fc.wall_s += step_clock.elapsed().as_secs_f64();
        fc.accepted += 1;
        fc.rejected += wd.w.rejected - work_before.rejected;
        fc.rebuild_work += wd.w.rebuild_flops - work_before.rebuild_flops;
        tr.max_e_cells = tr.max_e_cells.max(wd.e_cells());
        if wd.t >= next_rec - 1e-12 || wd.t >= t_end - 1e-12 {
            record(&wd, &mut tr, &mut e_max);
            if wd.sc.impactor.is_none() && !wd.sc.tip_cells.is_empty() && tr.sep_t.is_none() && wd.sc.m.dissipated > 0.0 && tip_fragment(&wd.sc, &wd.sc.m, &wd.q, &wd.v).separated {
                tr.sep_t = Some(wd.t);
            }
            while next_rec <= wd.t + 1e-12 {
                next_rec += rec;
            }
        }
    }
    let m = &wd.sc.m;
    if wd.sc.impactor.is_none() && !wd.sc.tip_cells.is_empty() {
        tr.fragment = tip_fragment(&wd.sc, m, &wd.q, &wd.v);
    }
    tr.frags = fragments(m, &wd.q, &wd.v, 0.05);
    tr.fragment_masses = fragments(m, &wd.q, &wd.v, 0.0).iter().map(|f|f.mass).collect();
    tr.v_impactor = wd.sc.impactor.map(|p| wd.v[p][1]);
    tr.d_end = m.dissipated;
    tr.crack = crack_stats(m);
    tr.e_end = tr.rows.last().unwrap()[7];
    tr.e_max_rise = e_max - tr.e0;
    tr.ledger = wd.led;
    tr.work = wd.w;
    tr.work_total = wd.w.total();
    tr.wall_s = t0.elapsed().as_secs_f64();
    tr.mean_e_frac = e_frac_int / ((wd.t - run_start).max(1e-30) * nc as f64);
    tr.chi_r_max = wd.chi_r_max;
    tr.events = wd.events.clone();
    let p1 = momentum(&wd);
    tr.momentum_drift = (p1 - p0).norm();
    (tr, wd)
}

/// Total linear momentum (cell-level velocities).
pub fn momentum(wd: &World) -> V2 {
    wd.sc.m.bodies.iter().zip(&wd.v).fold(V2::ZERO, |s, (b, v)| s + V2::new(v[0], v[1]) * b.mass)
}
