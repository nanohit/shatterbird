//! Implicit adaptive mode (docs/phase0d/adaptive-engine.md §5): one Newton system
//! over coarse patches and fine single-body patches, BDF2 with a local-error step
//! controller. Opening a coarse patch splits it into its cells; re-reduction merges
//! a quiet group of cells back into coarse patches. There is no partitioned coupling.

use super::super::patch::{build_patch, chol, chol_solve, k_apply, Patch};
use super::super::contact::{implicit_pair, GROUND};
use super::super::rsys::{fit_patch, Extra, Rep};
use super::super::{jperp, Work, C_PT_FORCE};
use super::{find, Open0, StepOutcome, World};
use geometry::V2;
use model::LawKind;

impl World {
    pub(super) fn is_coarse(&self, pid: usize) -> bool {
        self.patches[pid].cells.len() > 1
    }

    /// Plates (impactors) become single-body patches of the implicit system.
    pub(super) fn add_plate_patches(&mut self) {
        let m = &self.sc.m;
        let mut w = Work::default();
        for i in 0..m.bodies.len() {
            if m.bodies[i].kind == model::BodyKind::Plate && m.bodies[i].fixed.iter().any(|&f| !f) {
                let pa = build_patch(m, &self.adj, &[i], 0, 0, &self.loaded, &mut w);
                self.cell_patch[i] = self.patches.len();
                self.patches.push(pa);
                self.alive.push(true);
                self.open.push(false);
                self.open_since.push(0.0);
                self.reopen.push(0);
                self.closed_once.push(false);
                self.parent.push(usize::MAX);
            }
        }
        self.w.rebuild_flops += w.rebuild_flops;
    }

    /// Split coarse patches into single-cell patches (the implicit "open").
    pub(super) fn split(&mut self, pids: &[usize], why: &str) -> usize {
        let mut n = 0;
        let mut w = Work::default();
        for &pid in pids {
            if !self.alive[pid] || !self.is_coarse(pid) {
                continue;
            }
            if self.closed_once[pid] {
                self.reopen[pid] += 1;
            }
            self.alive[pid] = false;
            let cells = self.patches[pid].cells.clone();
            for i in cells {
                let pa = build_patch(&self.sc.m, &self.adj, &[i], 0, 0, &self.loaded, &mut w);
                self.cell_patch[i] = self.patches.len();
                self.patches.push(pa);
                self.alive.push(true);
                self.open.push(false);
                self.open_since.push(self.t);
                self.reopen.push(self.reopen[pid]);
                self.closed_once.push(false);
                self.parent.push(pid);
            }
            n += 1;
        }
        self.w.rebuild_flops += w.rebuild_flops;
        if n > 0 {
            self.w.activations += n as u64;
            if self.verbose {
                eprintln!("  t={:.5} split {n} patches ({why})", self.t);
            }
            self.events.push(format!("t={:.5} split {n} ({why})", self.t));
            self.rebuild();
            // cell histories persist through a split: BDF2 continues
            self.restart = false;
        }
        n
    }

    /// Coarse patches containing the cells, plus those whose centroid is within r_e of a point.
    pub(super) fn coarse_around(&self, cells: &[usize], points: &[V2]) -> Vec<usize> {
        let mut set = std::collections::BTreeSet::new();
        for &i in cells {
            let pid = self.cell_patch[i];
            if pid != usize::MAX && self.alive[pid] && self.is_coarse(pid) {
                set.insert(pid);
            }
        }
        for (pid, pa) in self.patches.iter().enumerate() {
            if self.alive[pid] && self.is_coarse(pid) && points.iter().any(|&x| (x - pa.xc).norm() < self.p.r_e + pa.radius) {
                set.insert(pid);
            }
        }
        set.into_iter().collect()
    }

    /// χ monitor over facets touching coarse-patch cells: undamaged points at `trig`,
    /// damaged points when they would damage further.
    fn monitor_coarse(&mut self, q: &[[f64; 3]], trig: f64) -> (f64, Vec<usize>) {
        let m = &self.sc.m;
        let coarse_cell = |i: usize| -> bool {
            let pid = self.cell_patch[i];
            pid != usize::MAX && self.patches[pid].cells.len() > 1
        };
        let mut cmax: f64 = 0.0;
        let mut hot = Vec::new();
        let mut flops = 0.0;
        let mut check = |fi: usize, cmax: &mut f64, hot: &mut Vec<usize>| {
            let f = &m.facets[fi];
            if f.tag != 0 {
                return;
            }
            let mut cf: f64 = 0.0;
            for p in &f.pts {
                if p.law.kind != LawKind::Damage || p.kappa_c >= p.law.wc {
                    continue;
                }
                let j = model::kin::jump(q[f.a], q[f.b], p.arm_a, p.arm_b, f.n0, f.t0, p.rest, f.frame_w);
                let dn = j.dn.max(0.0);
                let c = (dn * dn + p.law.gamma * j.dt * j.dt).sqrt() / p.kappa_c.max(p.law.delta0);
                cf = cf.max(if p.kappa_c > p.law.delta0 { c * trig.min(1.0) } else { c });
                flops += C_PT_FORCE;
            }
            *cmax = cmax.max(cf);
            if cf >= trig {
                hot.push(fi);
            }
        };
        for &pid in &self.rs.closed {
            if self.patches[pid].cells.len() > 1 {
                for &fi in &self.patches[pid].internal {
                    check(fi, &mut cmax, &mut hot);
                }
            }
        }
        for &fi in &self.rs.eval_facets {
            let f = &m.facets[fi];
            if (f.a < m.bodies.len() && coarse_cell(f.a)) || (f.b < m.bodies.len() && coarse_cell(f.b)) {
                check(fi, &mut cmax, &mut hot);
            }
        }
        self.w.monitor_flops += flops;
        (cmax, hot)
    }

    /// Contact-type points touching coarse-patch cells predicted to close by more than
    /// ½δ₀ within 2h.
    fn contact_events_coarse(&self, h: f64) -> Vec<usize> {
        let m = &self.sc.m;
        let mut out = Vec::new();
        for &fi in &self.rs.eval_facets {
            let fa = &m.facets[fi];
            let coarse = [fa.a, fa.b].iter().any(|&b| {
                let pid = self.cell_patch[b];
                pid != usize::MAX && self.patches[pid].cells.len() > 1
            });
            if !coarse {
                continue;
            }
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
                        if self.cell_patch[b] != usize::MAX && self.patches[self.cell_patch[b]].cells.len() > 1 {
                            out.push(b);
                        }
                    }
                }
            }
        }
        if let Some(cm) = &self.cm {
            let v_imp = self.p.chi_lte * m.mat.sigma_t / (m.mat.density * m.mat.wave_speed());
            for &(a, b) in &self.cpairs {
                let coarse = [a, b].iter().any(|&c| c != GROUND && self.cell_patch[c] != usize::MAX && self.patches[self.cell_patch[c]].cells.len() > 1);
                if !coarse {
                    continue;
                }
                let (g, n) = cm.geometry(a, b, &self.q);
                let vb = if b == GROUND { V2::ZERO } else { V2::new(self.v[b][0], self.v[b][1]) };
                let closing = -(vb - V2::new(self.v[a][0], self.v[a][1])).dot(n);
                if closing >= v_imp && g - 2.0 * h * closing < 0.0 {
                    for c in [a, b] {
                        if c != GROUND && self.patches[self.cell_patch[c]].cells.len() > 1 {
                            out.push(c);
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub(super) fn macro_step_implicit(&mut self, t_end: f64) -> bool {
        loop {
            let hs = self.h.min(t_end - self.t);
            if self.p.open0 != Open0::Never {
                let ev = self.contact_events_coarse(hs);
                if !ev.is_empty() {
                    let pts: Vec<V2> = ev.iter().map(|&i| self.sc.m.bodies[i].x0).collect();
                    let set = self.coarse_around(&ev, &pts);
                    if self.split(&set, "contact") > 0 {
                        continue;
                    }
                }
            }
            match self.try_step_implicit(hs) {
                StepOutcome::Accepted => return true,
                StepOutcome::Activate(cells, pts, why) => {
                    self.w.rejected += 1;
                    let set = self.coarse_around(&cells, &pts);
                    if self.split(&set, why) == 0 {
                        self.events.push(format!("t={:.5} monitor fired with nothing to split", self.t));
                        return false;
                    }
                }
                StepOutcome::Failed => {
                    self.w.rejected += 1;
                    self.h = 0.5 * hs;
                    self.easy_steps = 0;
                    if self.h < self.p.h_min {
                        return false;
                    }
                }
                StepOutcome::Retry(hn) => {
                    self.w.rejected += 1;
                    self.h = hn;
                    self.easy_steps = 0;
                    if self.h < self.p.h_min {
                        return false;
                    }
                }
            }
        }
    }

    /// BDF2 step (backward-Euler restart after a step change or a system change) with
    /// local-error control: the error of the step against the quadratic Taylor
    /// predictor, per body, against atol·δ₀ + rtol·|Δx|.
    fn try_step_implicit(&mut self, hs: f64) -> StepOutcome {
        let nb = self.sc.m.bodies.len();
        self.refresh_contacts();
        let citems = self.contact_items();
        // variable-step BDF2 (ω = h_n/h_{n−1}); backward Euler only without history
        let restart = self.restart || self.h_prev <= 0.0;
        let om = if restart { 1.0 } else { hs / self.h_prev };
        let beta = if restart { 1.0 } else { (1.0 + om) / (1.0 + 2.0 * om) };
        let c2 = if restart { 1.0 } else { (1.0 + om) * (1.0 + om) / (1.0 + 2.0 * om) };
        let c3 = if restart { 0.0 } else { om * om / (1.0 + 2.0 * om) };
        let alpha = 1.0 / (beta * beta);
        let mut qhat = self.q.clone();
        for i in 0..nb {
            if let Rep::R { .. } = self.rs.rep[i] {
                qhat[i] = std::array::from_fn(|k| c2 * self.q[i][k] - c3 * self.qp[i][k] + beta * hs * (c2 * self.v[i][k] - c3 * self.vp[i][k]));
            }
        }
        let mut w = self.w;
        let sd = self.rs.step_data(&self.patches, &qhat, alpha / (hs * hs), &mut w);
        let z_n = self.rs.fit(&self.patches, &self.q, &mut w);
        self.w = w;
        let e0 = self.r_energy(&self.q, &self.v, &z_n);
        let ext1 = self.sc.ext(self.t + hs);
        let mut z = self.rs.fit(&self.patches, &qhat, &mut self.w);
        let tol = self.tol;
        {
            let cm = self.cm.take();
            let ex = Extra { ext: &ext1, fj: None, robin: None, contacts: cm.as_ref().map(|c| (c, &citems[..], hs)) };
            let ok = self.rs.solve(&self.sc.m, &self.patches, &mut z, &sd, &ex, tol, 40, &mut self.w);
            self.cm = cm;
            if !ok {
                return StepOutcome::Failed;
            }
        }
        let mut qn = self.q.clone();
        self.rs.q_all(&self.patches, &z, &mut qn, &mut self.w);
        let mut vn = self.v.clone();
        for i in 0..nb {
            if let Rep::R { .. } = self.rs.rep[i] {
                vn[i] = std::array::from_fn(|k| (qn[i][k] - c2 * self.q[i][k] + c3 * self.qp[i][k]) / (beta * hs));
            }
        }
        // monitor: coarse material must stay elastic
        if self.p.open0 != Open0::Never {
            let (chi, hot) = self.monitor_coarse(&qn, self.p.chi_trig);
            if !hot.is_empty() {
                let m = &self.sc.m;
                let cells: Vec<usize> = hot.iter().flat_map(|&fi| [m.facets[fi].a, m.facets[fi].b]).filter(|&i| i < nb && m.bodies[i].kind == model::BodyKind::Cell).collect();
                let pts: Vec<V2> = hot.iter().map(|&fi| m.facets[fi].pc).collect();
                return StepOutcome::Activate(cells, pts, "monitor");
            }
            self.chi_r_max = self.chi_r_max.max(chi);
        }
        // event control on damage (Δd ≤ 0.1 per step)
        let m = &self.sc.m;
        let mut dd_max: f64 = 0.0;
        for &fi in &self.rs.eval_facets {
            let f = &m.facets[fi];
            if f.tag != 0 {
                continue;
            }
            for pt in 0..2 {
                let (_, e) = m.eval_point(&qn, f, pt);
                dd_max = dd_max.max(e.d - f.pts[pt].law.damage(f.pts[pt].kappa_c));
            }
        }
        self.w.pts_force += 2 * self.rs.eval_facets.len() as u64;
        if dd_max > 0.1 {
            return StepOutcome::Retry(0.5 * hs);
        }
        // local error in what drives forces and fracture: facet and contact openings.
        // Predictor: x + h v (+ ½h²a for BDF2 steps); its difference to the implicit
        // solution estimates the local error (O(h²) after a restart, O(h³) otherwise),
        // measured against atol·δ₀ + rtol·|Δδ| for every evaluated facet point.
        let mut err: f64 = 0.0;
        {
            let use_a = !restart && self.h_prev > 0.0;
            let mut xp = self.q.clone();
            for i in 0..nb {
                if !matches!(self.rs.rep[i], Rep::R { .. }) {
                    continue;
                }
                for k in 0..3 {
                    let a = if use_a { (self.v[i][k] - self.vp[i][k]) / self.h_prev } else { 0.0 };
                    xp[i][k] = self.q[i][k] + hs * self.v[i][k] + 0.5 * hs * hs * a;
                }
            }
            let m = &self.sc.m;
            let v_imp = self.p.chi_lte * m.mat.sigma_t / (m.mat.density * m.mat.wave_speed());
            for &fi in &self.rs.eval_facets {
                let f = &m.facets[fi];
                for pt in 0..2 {
                    let p = &f.pts[pt];
                    let jn = model::kin::jump(qn[f.a], qn[f.b], p.arm_a, p.arm_b, f.n0, f.t0, p.rest, f.frame_w);
                    let jp = model::kin::jump(xp[f.a], xp[f.b], p.arm_a, p.arm_b, f.n0, f.t0, p.rest, f.frame_w);
                    let j0 = model::kin::jump(self.q[f.a], self.q[f.b], p.arm_a, p.arm_b, f.n0, f.t0, p.rest, f.frame_w);
                    let d0 = if f.tag == 0 { p.law.delta0 } else { self.delta0_ref };
                    // resolve in time only what can break or collide: contact-type points in
                    // (or entering) compression, and material points near their damage surface
                    let contact_like = (f.tag != 0 && p.law.kind == LawKind::Damage) || (f.tag == 0 && p.law.damage(p.kappa_c) >= 0.99);
                    let relevant = if contact_like {
                        // an impact matters through the stress it makes, ρ c Δv: only contacts
                        // closing faster than χ_lte σ_t/(ρ c) are resolved in time
                        let vrel: f64 = (0..3).map(|k| jn.jna[k] * vn[f.a][k] + jn.jnb[k] * vn[f.b][k]).sum::<f64>().abs();
                        let vrel0: f64 = (0..3).map(|k| j0.jna[k] * self.v[f.a][k] + j0.jnb[k] * self.v[f.b][k]).sum::<f64>().abs();
                        (jn.dn < 0.0 || jp.dn < 0.0 || j0.dn < 0.0) && vrel.max(vrel0) >= v_imp
                    } else if p.law.kind == LawKind::Damage {
                        // near the damage surface at the end state *or* at the predicted state:
                        // a damped implicit step must not hide an overshoot the predictor sees
                        let chi_of = |j: &model::kin::Jump| {
                            let dnp = j.dn.max(0.0);
                            (dnp * dnp + p.law.gamma * j.dt * j.dt).sqrt() / p.kappa_c.max(p.law.delta0)
                        };
                        chi_of(&jn).max(chi_of(&jp)) >= self.p.chi_lte
                    } else {
                        false
                    };
                    if !relevant {
                        continue;
                    }
                    let e = ((jn.dn - jp.dn).powi(2) + (jn.dt - jp.dt).powi(2)).sqrt();
                    let dd = ((jn.dn - j0.dn).powi(2) + (jn.dt - j0.dt).powi(2)).sqrt();
                    err = err.max(e / (self.p.lte_atol * d0 + self.p.lte_rtol * dd));
                }
            }
            if let Some(cm) = &self.cm {
                for it in &citems {
                    let (g1, n1) = cm.geometry(it.a, it.b, &qn);
                    let (gp, _) = cm.geometry(it.a, it.b, &xp);
                    let (g0, _) = cm.geometry(it.a, it.b, &self.q);
                    // closing speed along the normal (a → b)
                    let vb = if it.b == GROUND { geometry::V2::ZERO } else { geometry::V2::new(vn[it.b][0], vn[it.b][1]) };
                    let vrel = (vb - geometry::V2::new(vn[it.a][0], vn[it.a][1])).dot(n1).abs();
                    if (g1 < 0.0 || gp < 0.0 || g0 < 0.0) && vrel >= v_imp {
                        err = err.max((g1 - gp).abs() / (self.p.lte_atol * self.delta0_ref + self.p.lte_rtol * (g1 - g0).abs()));
                    }
                }
            }
            self.w.monitor_flops += (self.rs.eval_facets.len() * 6) as f64 * C_PT_FORCE;
            if err > 1.0 && hs > self.p.h_min * 1.0001 {
                let order = if use_a { 3.0 } else { 2.0 };
                let target = hs * (0.9 * err.powf(-1.0 / order)).max(0.1);
                let k = (self.p.h_macro / target).log2().ceil().max(0.0) as i32;
                return StepOutcome::Retry((self.p.h_macro / f64::powi(2.0, k)).max(self.p.h_min).min(0.5 * hs));
            }
        }
        // accept
        let ext0 = self.sc.ext(self.t);
        let mut w_ext = 0.0;
        for &(i, f0) in &ext0 {
            let f1 = ext1.iter().find(|x| x.0 == i).map(|x| x.1).unwrap_or([0.0; 3]);
            for k in 0..3 {
                w_ext += 0.5 * (f0[k] + f1[k]) * (qn[i][k] - self.q[i][k]);
            }
        }
        for &(i, f1) in &ext1 {
            if !ext0.iter().any(|x| x.0 == i) {
                for k in 0..3 {
                    w_ext += 0.5 * f1[k] * (qn[i][k] - self.q[i][k]);
                }
            }
        }
        let e1 = self.r_energy(&qn, &vn, &z);
        let m = &mut self.sc.m;
        let mut d_inc = 0.0;
        for &fi in &self.rs.eval_facets {
            for pt in 0..2 {
                let (_, e) = m.eval_point(&qn, &m.facets[fi], pt);
                let p = &mut m.facets[fi].pts[pt];
                if e.kappa > p.kappa_c {
                    let dd = p.law.dissipation(e.kappa) - p.law.dissipation(p.kappa_c);
                    m.dissipated += dd;
                    d_inc += dd;
                    p.kappa_c = e.kappa;
                    if dd > 0.0 {
                        self.last_growth[fi] = self.t + hs;
                    }
                }
            }
        }
        let mut wf = 0.0;
        if let Some(cm) = &self.cm {
            for it in &citems {
                wf += implicit_pair(cm, it, &qn, hs).3;
            }
        }
        self.led.d_fric += wf;
        // numerical dissipation of the step (positive = lost): what the energy balance misses
        self.led.d_num_r += e0 + w_ext - e1 - d_inc - wf;
        self.led.w_ext += w_ext;
        for i in 0..nb {
            if let Rep::R { .. } = self.rs.rep[i] {
                self.qp[i] = self.q[i];
                self.vp[i] = self.v[i];
                self.q[i] = qn[i];
                self.v[i] = vn[i];
            }
        }
        self.t += hs;
        self.h_prev = hs;
        self.w.macro_steps += 1;
        self.restart = false;
        // grow after a few comfortable steps
        if err < 0.3 && dd_max < 0.025 {
            self.easy_steps += 1;
        } else {
            self.easy_steps = 0;
        }
        if self.h < self.p.h_macro && self.easy_steps >= 3 {
            self.h = (2.0 * self.h).min(self.p.h_macro);
            self.easy_steps = 0;
        }
        StepOutcome::Accepted
    }

    /// Merge quiet groups of split cells back into coarse patches.
    pub(super) fn merge(&mut self) -> usize {
        if !self.p.rereduce {
            return 0;
        }
        let t = self.t;
        let quiet_t = self.p.quiet_steps as f64 * self.p.h_macro - 1e-12;
        // groups of alive fine patches by their (dead) coarse parent
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for pid in 0..self.patches.len() {
            if self.alive[pid] && self.parent[pid] != usize::MAX && self.patches[pid].cells.len() == 1 {
                groups.entry(self.parent[pid]).or_default().push(pid);
            }
        }
        let m = &self.sc.m;
        let nb = m.bodies.len();
        let mut cands: Vec<(usize, Vec<usize>)> = Vec::new();
        for (par, kids) in groups {
            let pcells = self.patches[par].cells.len();
            if kids.len() != pcells {
                continue;
            }
            let since = kids.iter().map(|&k| self.open_since[k]).fold(f64::NEG_INFINITY, f64::max);
            if t - since < quiet_t * f64::powi(2.0, self.reopen[par].min(3) as i32) {
                continue;
            }
            cands.push((par, kids));
        }
        if cands.is_empty() {
            return 0;
        }
        // busy cells: recent growth, stress near onset / history, closing contacts
        let mut busy = vec![false; nb];
        for (fi, fa) in m.facets.iter().enumerate() {
            if !fa.active {
                continue;
            }
            let contact_like = (fa.tag != 0 && fa.pts[0].law.kind == LawKind::Damage) || (fa.tag == 0 && fa.pts.iter().any(|p| p.law.damage(p.kappa_c) >= 0.99));
            let mut hot = false;
            if contact_like {
                let d0 = if fa.tag == 0 { fa.pts[0].law.delta0 } else { self.delta0_ref };
                for pt in 0..2 {
                    let (j, e) = m.eval_point(&self.q, fa, pt);
                    let s: f64 = -(0..3).map(|k| j.jna[k] * self.v[fa.a][k] + j.jnb[k] * self.v[fa.b][k]).sum::<f64>();
                    let pen_now = (-j.dn).max(0.0);
                    let pen_pred = (-(j.dn - 2.0 * self.p.h_macro * s)).max(0.0);
                    if pen_pred - pen_now > 0.5 * d0 || (fa.tag == 3 && e.fnorm < 0.0) {
                        hot = true;
                    }
                }
            } else if fa.tag == 0 {
                hot = t - self.last_growth[fi] < quiet_t + self.p.h_macro;
                for p in 0..2 {
                    let pt = &fa.pts[p];
                    if pt.law.kind != LawKind::Damage || pt.kappa_c >= pt.law.wc {
                        continue;
                    }
                    let (j, _) = m.eval_point(&self.q, fa, p);
                    let dn = j.dn.max(0.0);
                    let chi = (dn * dn + pt.law.gamma * j.dt * j.dt).sqrt() / pt.kappa_c.max(pt.law.delta0);
                    let lim = if pt.kappa_c > pt.law.delta0 { 0.9 } else { self.p.chi_rr };
                    if chi >= lim {
                        hot = true;
                    }
                }
            }
            if hot {
                busy[fa.a] = true;
                busy[fa.b] = true;
            }
        }
        if let Some(cm) = &self.cm {
            let v_imp = self.p.chi_lte * m.mat.sigma_t / (m.mat.density * m.mat.wave_speed());
            for &(a, b) in &self.cpairs {
                let (g, n) = cm.geometry(a, b, &self.q);
                let vb = if b == GROUND { V2::ZERO } else { V2::new(self.v[b][0], self.v[b][1]) };
                let closing = -(vb - V2::new(self.v[a][0], self.v[a][1])).dot(n);
                if closing >= v_imp && g - 2.0 * self.p.h_macro * closing < 0.0 {
                    busy[a] = true;
                    if b != GROUND {
                        busy[b] = true;
                    }
                }
            }
        }
        let mut merged = 0;
        let mut applied: Vec<(usize, Vec<usize>, Vec<Patch>, Vec<(usize, [f64; 3], [f64; 3])>, f64)> = Vec::new();
        for (par, kids) in cands {
            let cells = self.patches[par].cells.clone();
            if cells.iter().any(|&i| busy[i]) {
                continue;
            }
            // components through undamaged material facets
            let mut loc = vec![usize::MAX; nb];
            for (li, &i) in cells.iter().enumerate() {
                loc[i] = li;
            }
            let mut pr: Vec<usize> = (0..cells.len()).collect();
            for &i in &cells {
                for &fi in &self.adj[i] {
                    let f = &m.facets[fi];
                    if !f.active || f.tag != 0 || loc[f.a] == usize::MAX || loc[f.b] == usize::MAX {
                        continue;
                    }
                    if f.pts.iter().all(|p| p.law.damage(p.kappa_c) == 0.0) {
                        let (a, b) = (find(&mut pr, loc[f.a]), find(&mut pr, loc[f.b]));
                        if a != b {
                            pr[a.max(b)] = a.min(b);
                        }
                    }
                }
            }
            let mut comps: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
            for (li, &i) in cells.iter().enumerate() {
                comps.entry(find(&mut pr, li)).or_default().push(i);
            }
            let same = comps.len() == 1 && self.patches[par].internal.iter().all(|&fi| m.facets[fi].pts.iter().all(|p| p.law.damage(p.kappa_c) == 0.0));
            let mut w = Work::default();
            let pats: Vec<Patch> = if same {
                vec![self.patches[par].clone()]
            } else {
                comps.values().filter(|c| c.len() > 1).map(|c| build_patch(m, &self.adj, c, self.p.deg, self.p.nvib, &self.loaded, &mut w)).collect()
            };
            self.w.rebuild_flops += w.rebuild_flops;
            if pats.is_empty() {
                continue;
            }
            // project positions (K-orthogonal deformation) and velocities (M-orthogonal)
            let mut proj: Vec<(usize, [f64; 3], [f64; 3])> = Vec::new();
            for pa in &pats {
                let kop = |d: &[f64]| k_apply(m, pa, d);
                let y = fit_patch(pa, &self.q, true, Some(&kop as &dyn Fn(&[f64]) -> Vec<f64>));
                let ny = pa.ny();
                let mut gmg = vec![0.0; ny * ny];
                let mut gmv = vec![0.0; ny];
                let mut gs: Vec<Vec<[f64; 3]>> = Vec::with_capacity(pa.cells.len());
                let mut qs: Vec<[f64; 3]> = Vec::with_capacity(pa.cells.len());
                for (li, &i) in pa.cells.iter().enumerate() {
                    let (r, wth) = pa.local(li, &y[3..]);
                    let s = r.rot(y[2]);
                    qs.push([pa.xc.x + y[0] + s.x, pa.xc.y + y[1] + s.y, y[2] + wth]);
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
                    let vv: [f64; 3] = std::array::from_fn(|k| (0..ny).map(|a| gs[li][a][k] * gmv[a]).sum());
                    proj.push((i, qs[li], vv));
                }
                self.w.rebuild_flops += (pa.cells.len() * ny * ny * 6) as f64;
            }
            // energy check over the group's cells and facets
            let mut qn = self.q.clone();
            let mut vn = self.v.clone();
            for &(i, qq, vv) in &proj {
                qn[i] = qq;
                vn[i] = vv;
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
                continue;
            }
            if loss > (self.p.eps_rr * own).max(1e-4 * self.sc.e_scale) {
                continue;
            }
            applied.push((par, kids, pats, proj, loss));
        }
        for (par, kids, pats, proj, loss) in applied {
            for &(i, qq, vv) in &proj {
                self.q[i] = qq;
                self.v[i] = vv;
                self.qp[i] = qq;
                self.vp[i] = vv;
            }
            self.led.d_red += loss;
            let mut covered = std::collections::BTreeSet::new();
            for pa in pats {
                let nid = self.patches.len();
                for &i in &pa.cells {
                    self.cell_patch[i] = nid;
                    covered.insert(i);
                }
                self.patches.push(pa);
                self.alive.push(true);
                self.open.push(false);
                self.open_since.push(self.t);
                self.reopen.push(self.reopen[par]);
                self.closed_once.push(true);
                self.parent.push(usize::MAX);
            }
            for &k in &kids {
                let i = self.patches[k].cells[0];
                if covered.contains(&i) {
                    self.alive[k] = false;
                } else {
                    // a cell left alone by damage stays a fine patch, no longer tied to the parent
                    self.parent[k] = usize::MAX;
                }
            }
            merged += 1;
        }
        if merged > 0 {
            self.w.rereductions += merged as u64;
            if self.verbose {
                eprintln!("  t={:.5} merged {merged} patches", self.t);
            }
            self.events.push(format!("t={:.5} merge {merged}", self.t));
            self.rebuild();
            self.restart = true;
        }
        merged
    }

    /// Cost-driven integrator choice. In the implicit mode, the work per simulated
    /// second over the last steps is compared with explicit on the whole model; when
    /// implicit is dearer the whole model runs explicitly (a window), and returns to
    /// the implicit mode (every cell fine, merged as it quiets) when damage has stopped
    /// growing for 1 ms.
    pub(super) fn cost_switch(&mut self) {
        let Some(ratio) = self.p.cost_switch else { return };
        let m = &self.sc.m;
        match self.window {
            None => {
                if self.rate_expl == 0.0 {
                    // explicit cost per second on the whole model at its stable step
                    let saved = std::mem::replace(&mut self.p.h_e_fixed, None);
                    let (eb, ef) = (std::mem::take(&mut self.e_bodies), std::mem::take(&mut self.e_facets));
                    let rep = self.rs.rep.clone();
                    for r in self.rs.rep.iter_mut() {
                        if matches!(r, Rep::R { .. }) {
                            *r = Rep::E;
                        }
                    }
                    self.e_bodies = (0..m.bodies.len()).filter(|&i| self.rs.rep[i] == Rep::E).collect();
                    self.e_facets = (0..m.facets.len()).filter(|&fi| m.facets[fi].active).collect();
                    let he = self.stable_he();
                    let per_sub = (4 * self.e_facets.len()) as f64 * C_PT_FORCE + (12 * self.e_bodies.len()) as f64;
                    self.rate_expl = per_sub / he;
                    self.rs.rep = rep;
                    self.e_bodies = eb;
                    self.e_facets = ef;
                    self.p.h_e_fixed = saved;
                }
                // implicit cost rate: mean work per step over the last 12 steps divided by
                // the current step (responsive to the dynamics, robust to single costly steps)
                self.rate_hist.push((self.t, self.w.total()));
                if self.rate_hist.len() > 13 {
                    self.rate_hist.remove(0);
                }
                if self.rate_hist.len() >= 5 {
                    let n = (self.rate_hist.len() - 1) as f64;
                    let per_step = (self.rate_hist.last().unwrap().1 - self.rate_hist[0].1) / n;
                    let rate = per_step / self.h.max(1e-12);
                    self.over = if rate > ratio * self.rate_expl { self.over + 1 } else { 0 };
                    if self.over >= 2 {
                        self.over = 0;
                        // enter the window: every cell fine, every patch open (all explicit)
                        let coarse: Vec<usize> = (0..self.patches.len()).filter(|&p| self.alive[p] && self.is_coarse(p)).collect();
                        self.split(&coarse, "window");
                        for p in 0..self.patches.len() {
                            if self.alive[p] {
                                self.open[p] = true;
                            }
                        }
                        self.h_before_window = self.h;
                        self.h = self.p.h_macro.min(5e-4);
                        // a window reopened soon after the last one lasts twice as long
                        let reentry = self.windows.last().map(|w| self.t - w[1] < 5e-3).unwrap_or(false);
                        self.window_len = if reentry { (2.0 * self.window_len).min(0.02) } else { 2e-3 };
                        self.window = Some(self.t + self.window_len);
                        self.windows.push([self.t, f64::NAN]);
                        self.rate_hist.clear();
                        if self.verbose {
                            eprintln!("  t={:.5} explicit window (implicit {:.3e} > explicit {:.3e} flop/s)", self.t, rate, self.rate_expl);
                        }
                        self.events.push(format!("t={:.5} window open", self.t));
                        self.rebuild();
                    }
                }
            }
            Some(t_min) => {
                // fracture has stopped evolving: dissipation over the last 1 ms below 1 % of
                // the total (or 10⁻⁵ of the scene energy scale)
                self.d_hist.push((self.t, m.dissipated));
                while self.d_hist.len() > 2 && self.d_hist[1].0 <= self.t - 1e-3 {
                    self.d_hist.remove(0);
                }
                let (t_old, d_old) = self.d_hist[0];
                let quiet = self.t - t_old >= 1e-3 - 1e-12 && m.dissipated - d_old <= (1e-2 * m.dissipated).max(1e-5 * self.sc.e_scale);
                if self.t >= t_min && quiet {
                    self.d_hist.clear();
                    for p in 0..self.patches.len() {
                        self.open[p] = false;
                    }
                    self.window = None;
                    self.h = self.h_before_window.max(self.p.h_min);
                    self.h_prev = 0.0;
                    if let Some(w) = self.windows.last_mut() {
                        w[1] = self.t;
                    }
                    if self.verbose {
                        eprintln!("  t={:.5} explicit window closed", self.t);
                    }
                    self.events.push(format!("t={:.5} window close", self.t));
                    self.rebuild();
                    self.restart = true;
                    self.rate_hist.clear();
                }
            }
        }
    }
}
