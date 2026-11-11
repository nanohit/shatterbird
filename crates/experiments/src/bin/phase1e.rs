//! Phase 1c: members with ports, end to end (docs/phase1/prereg-1c.md).
//! cargo run --release -p experiments --bin phase1e -- validate|hybrid|one ...
use experiments::engine::scenes::{beam_impact, Scene};
use experiments::engine::world::{crack_stats, fragments};
use experiments::{meta, write_json};
use model::{BodyKind, LawKind, Model};
use serde_json::{json, Value};
use solvers::spectrum::Spectral;

const DT: f64 = 2e-6;
const C_PT: f64 = 120.0;
const NONE: usize = usize::MAX;
/// interior damage is evaluated every NS steps
const NS: u64 = 8;

/// A facet may live inside a member's operator while it is not broken (d <= 0.99).
fn undamaged(m: &Model, fi: usize) -> bool {
    m.facets[fi].pts.iter().all(|p| p.law.kind == LawKind::Elastic || p.law.damage(p.kappa_c) <= 0.99)
}
fn envf(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}


struct Member {
    cells: Vec<usize>,
    k: usize,
    lam: Vec<f64>,
    /// physical mode shapes in the body frame: phi[(local * 3 + d) * k + j]
    phi: Vec<f64>,
    r: Vec<[f64; 2]>,
    mass: f64,
    inertia: f64,
    x: [f64; 2],
    th: f64,
    v: [f64; 2],
    om: f64,
    eta: Vec<f64>,
    etad: Vec<f64>,
    port_local: Vec<usize>,
    /// undamaged interior points: opening per unit eta, sn[p * k + j]
    sn: Vec<f64>,
    st: Vec<f64>,
    gam: Vec<f64>,
    d0: Vec<f64>,
    /// modal projections of the cell loads (cos and sin parts) and their resultant
    pg: Vec<f64>,
    sg: Vec<f64>,
    load: [f64; 2],
    /// interior facets (in the linear operator) and the held damage pseudo-force
    int_f: Vec<usize>,
    qd: Vec<f64>,
    /// built on a damaged set: secant operator, damage frozen, watched against history
    frozen: bool,
}

impl Member {
    fn pos(&self, c: usize) -> [f64; 3] {
        let k = self.k;
        let (mut ex, mut ey, mut et) = (self.r[c][0], self.r[c][1], 0.0);
        for j in 0..k {
            ex += self.phi[(c * 3) * k + j] * self.eta[j];
            ey += self.phi[(c * 3 + 1) * k + j] * self.eta[j];
            et += self.phi[(c * 3 + 2) * k + j] * self.eta[j];
        }
        let (s, co) = self.th.sin_cos();
        [self.x[0] + co * ex - s * ey, self.x[1] + s * ex + co * ey, self.th + et]
    }
    fn vel(&self, c: usize, q: [f64; 3]) -> [f64; 3] {
        let k = self.k;
        let (mut ex, mut ey, mut et) = (0.0, 0.0, 0.0);
        for j in 0..k {
            ex += self.phi[(c * 3) * k + j] * self.etad[j];
            ey += self.phi[(c * 3 + 1) * k + j] * self.etad[j];
            et += self.phi[(c * 3 + 2) * k + j] * self.etad[j];
        }
        let (s, co) = self.th.sin_cos();
        let (rx, ry) = (q[0] - self.x[0], q[1] - self.x[1]);
        [self.v[0] - self.om * ry + co * ex - s * ey, self.v[1] + self.om * rx + s * ex + co * ey, self.om + et]
    }
    fn exact(&self) -> f64 {
        let k = self.k;
        let mut mx = 0.0f64;
        for p in 0..self.d0.len() {
            let (mut dn, mut dt) = (0.0, 0.0);
            for j in 0..k {
                dn += self.sn[p * k + j] * self.eta[j];
                dt += self.st[p * k + j] * self.eta[j];
            }
            let dp = dn.max(0.0);
            mx = mx.max((dp * dp + self.gam[p] * dt * dt).sqrt() / self.d0[p]);
        }
        mx
    }
}

struct Sim {
    sc: Scene,
    q: Vec<[f64; 3]>,
    v: Vec<[f64; 3]>,
    f: Vec<[f64; 3]>,
    owner: Vec<usize>,
    local: Vec<usize>,
    members: Vec<Option<Member>>,
    ports: Vec<usize>,
    t: f64,
    step: u64,
    last_damage: f64,
    last_try: f64,
    hybrid: bool,
    kmax: usize,
    chi: f64,
    tq: f64,
    flops: f64,
    single_steps: f64,
    events: Vec<Value>,
    esc: usize,
    red: usize,
    basis_wall: f64,
    check_flops: f64,
    dirty: bool,
    /// start of the current quiet window and the dissipation at that time
    win_t: f64,
    win_d: f64,
    epsd: f64,
    desc: f64,
    chipost: f64,
    /// rings of single cells kept around every natural port (support, contact, damaged facet)
    collar: usize,
}

impl Sim {
    fn new(sc: Scene, hybrid: bool, kmax: usize, chi: f64, tq: f64, collar: usize) -> Sim {
        let nb = sc.m.bodies.len();
        let q = sc.m.q.clone();
        let v = sc.v_init.clone();
        let mut s = Sim { sc, q, v, f: vec![[0.0; 3]; nb], owner: vec![NONE; nb], local: vec![0; nb], members: Vec::new(), ports: Vec::new(), t: 0.0, step: 0, last_damage: -1.0, last_try: -1.0, hybrid, kmax, chi, tq, flops: 0.0, single_steps: 0.0, events: Vec::new(), esc: 0, red: 0, basis_wall: 0.0, check_flops: 0.0, dirty: true, win_t: 0.0, win_d: 0.0, epsd: envf("SB_EPSD", 0.005), desc: envf("SB_DESC", 0.3), chipost: envf("SB_CHIPOST", 1.5), collar };
        s.last_damage = -1.0;
        s.rebuild_ports();
        s
    }

    fn rebuild_ports(&mut self) {
        let m = &self.sc.m;
        self.ports = (0..m.facets.len()).filter(|&fi| {
            let f = &m.facets[fi];
            f.active && !(f.tag == 0 && self.owner[f.a] != NONE && self.owner[f.a] == self.owner[f.b] && undamaged(m, fi))
        }).collect();
        for mem in self.members.iter_mut().flatten() {
            mem.port_local.clear();
        }
        let mut seen = vec![false; m.bodies.len()];
        for &fi in &self.ports {
            for b in [m.facets[fi].a, m.facets[fi].b] {
                if self.owner[b] != NONE && !seen[b] {
                    seen[b] = true;
                    self.members[self.owner[b]].as_mut().unwrap().port_local.push(self.local[b]);
                }
            }
        }
    }

    /// Free member from a connected set of cells joined by undamaged facets.
    fn build_member(&self, cells: &[usize]) -> Option<Member> {
        let m0 = &self.sc.m;
        let mut inset = vec![false; m0.bodies.len()];
        for &c in cells {
            inset[c] = true;
        }
        let mut mm = m0.clone();
        for (i, b) in mm.bodies.iter_mut().enumerate() {
            b.fixed = [!inset[i]; 3];
        }
        let mut interior = Vec::new();
        let mut frozen = false;
        for fi in 0..mm.facets.len() {
            let f = &m0.facets[fi];
            let keep = f.active && f.tag == 0 && inset[f.a] && inset[f.b] && undamaged(m0, fi);
            mm.facets[fi].active = keep;
            if keep {
                interior.push(fi);
                for p in 0..2 {
                    // pristine set: intact stiffness (damage will enter as a pseudo-force);
                    // damaged set: secant stiffness of the committed damage at the current sign
                    let (j, _) = m0.eval_point(&self.q, f, p);
                    let pt = &mut mm.facets[fi].pts[p];
                    let om = 1.0 - pt.law.damage(pt.kappa_c);
                    if om < 1.0 {
                        frozen = true;
                        if j.dn > 0.0 {
                            pt.law.kn *= om;
                        }
                        pt.law.kt *= om;
                    }
                    pt.law.kind = LawKind::Elastic;
                }
            }
        }
        let n = 3 * cells.len();
        let k = self.kmax.min(n.saturating_sub(3));
        let sp = Spectral::from_model2d(&mm);
        let pr = sp.low_shift(k + 3, 3.55e4, 7, 1e-10, n).ok()?;
        let mut ord: Vec<usize> = (0..pr.vals.len()).collect();
        ord.sort_by(|&a, &b| pr.vals[a].partial_cmp(&pr.vals[b]).unwrap());
        // exactly three rigid modes: anything else is a mechanism or a numerical failure
        if ord.len() < k + 3 || pr.vals[ord[2]].abs() > 1e3 || pr.vals[ord[3]] < 50.0 {
            eprintln!("basis rejected: {} cells, {} pairs, lowest {:?}", cells.len(), ord.len(), ord.iter().take(6).map(|&i| pr.vals[i]).collect::<Vec<_>>());
            return None;
        }
        let (idx, _) = mm.dof_map();
        let nc = cells.len();
        let mut phi = vec![0.0; nc * 3 * k];
        let lam: Vec<f64> = (0..k).map(|j| pr.vals[ord[3 + j]]).collect();
        for (c, &b) in cells.iter().enumerate() {
            for d in 0..3 {
                let g = idx[b][d].unwrap();
                let sq = (if d < 2 { m0.bodies[b].mass } else { m0.bodies[b].inertia }).sqrt();
                for j in 0..k {
                    phi[(c * 3 + d) * k + j] = pr.vecs[ord[3 + j]][g] / sq;
                }
            }
        }
        let mass: f64 = cells.iter().map(|&b| m0.bodies[b].mass).sum();
        let (mut cx, mut cy) = (0.0, 0.0);
        for &b in cells {
            cx += m0.bodies[b].mass * m0.bodies[b].x0.x;
            cy += m0.bodies[b].mass * m0.bodies[b].x0.y;
        }
        cx /= mass;
        cy /= mass;
        let r: Vec<[f64; 2]> = cells.iter().map(|&b| [m0.bodies[b].x0.x - cx, m0.bodies[b].x0.y - cy]).collect();
        let inertia: f64 = cells.iter().zip(&r).map(|(&b, r)| m0.bodies[b].mass * (r[0] * r[0] + r[1] * r[1]) + m0.bodies[b].inertia).sum();
        // openings of undamaged interior points per unit modal amplitude (finite difference at x0)
        let q0: Vec<[f64; 3]> = m0.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
        let np = 2 * interior.len();
        let (mut sn, mut st) = (vec![0.0; np * k], vec![0.0; np * k]);
        let (mut gam, mut d0) = (Vec::with_capacity(np), Vec::with_capacity(np));
        for &fi in interior.iter() {
            for p in 0..2 {
                let pt = &m0.facets[fi].pts[p];
                gam.push(pt.law.gamma);
                d0.push(pt.kappa_c.max(pt.law.delta0));
            }
        }
        let eps = 1e-6;
        let mut qe = q0.clone();
        for j in 0..k {
            for (c, &b) in cells.iter().enumerate() {
                for d in 0..3 {
                    qe[b][d] = q0[b][d] + eps * phi[(c * 3 + d) * k + j];
                }
            }
            for (i, &fi) in interior.iter().enumerate() {
                for p in 0..2 {
                    let (jp, _) = m0.eval_point(&qe, &m0.facets[fi], p);
                    sn[(2 * i + p) * k + j] = jp.dn / eps;
                    st[(2 * i + p) * k + j] = jp.dt / eps;
                }
            }
        }
        let (mut pg, mut sg) = (vec![0.0; k], vec![0.0; k]);
        let mut load = [0.0; 2];
        for (c, &b) in cells.iter().enumerate() {
            let l = m0.loads[b];
            load[0] += l[0];
            load[1] += l[1];
            for j in 0..k {
                let (px, py) = (phi[(c * 3) * k + j], phi[(c * 3 + 1) * k + j]);
                pg[j] += px * l[0] + py * l[1];
                sg[j] += px * l[1] - py * l[0];
            }
        }
        // state: rigid fit (2D Procrustes) and modal projection of the remainder
        let (q, v) = (&self.q, &self.v);
        let (mut x, mut vv) = ([0.0; 2], [0.0; 2]);
        for &b in cells {
            let mb = m0.bodies[b].mass;
            x[0] += mb * q[b][0];
            x[1] += mb * q[b][1];
            vv[0] += mb * v[b][0];
            vv[1] += mb * v[b][1];
        }
        x = [x[0] / mass, x[1] / mass];
        vv = [vv[0] / mass, vv[1] / mass];
        let (mut sc_, mut ss_) = (0.0, 0.0);
        for (c, &b) in cells.iter().enumerate() {
            let mb = m0.bodies[b].mass;
            let (dx, dy) = (q[b][0] - x[0], q[b][1] - x[1]);
            sc_ += mb * (r[c][0] * dx + r[c][1] * dy);
            ss_ += mb * (r[c][0] * dy - r[c][1] * dx);
        }
        let th = ss_.atan2(sc_);
        let mut l = 0.0;
        for &b in cells {
            let mb = m0.bodies[b].mass;
            let (dx, dy) = (q[b][0] - x[0], q[b][1] - x[1]);
            l += mb * (dx * (v[b][1] - vv[1]) - dy * (v[b][0] - vv[0])) + m0.bodies[b].inertia * v[b][2];
        }
        let om = l / inertia;
        let (s, co) = th.sin_cos();
        let (mut eta, mut etad) = (vec![0.0; k], vec![0.0; k]);
        for (c, &b) in cells.iter().enumerate() {
            let (mb, ib) = (m0.bodies[b].mass, m0.bodies[b].inertia);
            let (dx, dy) = (q[b][0] - x[0], q[b][1] - x[1]);
            let e = [co * dx + s * dy - r[c][0], -s * dx + co * dy - r[c][1], q[b][2] - th];
            let (wx, wy) = (v[b][0] - vv[0] + om * dy, v[b][1] - vv[1] - om * dx);
            let ed = [co * wx + s * wy, -s * wx + co * wy, v[b][2] - om];
            for j in 0..k {
                let (px, py, pt) = (phi[(c * 3) * k + j], phi[(c * 3 + 1) * k + j], phi[(c * 3 + 2) * k + j]);
                eta[j] += mb * (px * e[0] + py * e[1]) + ib * pt * e[2];
                etad[j] += mb * (px * ed[0] + py * ed[1]) + ib * pt * ed[2];
            }
        }
        Some(Member { cells: cells.to_vec(), k, lam, phi, r, mass, inertia, x, th, v: vv, om, eta, etad, port_local: Vec::new(), sn, st, gam, d0, pg, sg, load, int_f: interior, qd: vec![0.0; k], frozen })
    }

    fn sync_member(&mut self, mi: usize) {
        let mem = self.members[mi].as_ref().unwrap();
        for (c, &b) in mem.cells.iter().enumerate() {
            let q = mem.pos(c);
            self.q[b] = q;
            self.v[b] = mem.vel(c, q);
        }
    }

    fn escalate(&mut self, mi: usize, ratio: f64) {
        self.sync_member(mi);
        let mem = self.members[mi].take().unwrap();
        for &b in &mem.cells {
            self.owner[b] = NONE;
        }
        self.esc += 1;
        self.last_damage = self.t;
        self.dirty = true;
        self.events.push(json!({"t": self.t, "event": "escalate", "cells": mem.cells.len(), "ratio": ratio}));
        self.rebuild_ports();
    }

    fn try_reduce(&mut self) {
        let m = &self.sc.m;
        let nb = m.bodies.len();
        let mut parent: Vec<usize> = (0..nb).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        // collar: cells within `collar` rings of a natural port stay single, so local contact
        // compliance and stress concentration are carried by real cells, not by truncated modes
        let cd = std::env::var("SB_CD").map_or(false, |v| v == "1");
        let mut ring = vec![usize::MAX; nb];
        for fi in 0..m.facets.len() {
            let f = &m.facets[fi];
            if f.active && (f.tag != 0 || (cd && !undamaged(m, fi))) {
                for b in [f.a, f.b] {
                    if m.bodies[b].kind == BodyKind::Cell {
                        ring[b] = 0;
                    }
                }
            }
        }
        for h in 0..self.collar {
            for fi in 0..m.facets.len() {
                let f = &m.facets[fi];
                if f.active && f.tag == 0 {
                    if ring[f.a] == h && ring[f.b] == usize::MAX {
                        ring[f.b] = h + 1;
                    } else if ring[f.b] == h && ring[f.a] == usize::MAX {
                        ring[f.a] = h + 1;
                    }
                }
            }
        }
        let single = |i: usize| self.owner[i] == NONE && m.bodies[i].kind == BodyKind::Cell && (self.collar == 0 || ring[i] >= self.collar);
        for fi in 0..m.facets.len() {
            let f = &m.facets[fi];
            if f.active && f.tag == 0 && single(f.a) && single(f.b) && undamaged(m, fi) {
                let (a, b) = (find(&mut parent, f.a), find(&mut parent, f.b));
                if a != b {
                    parent[a] = b;
                }
            }
        }
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for i in 0..nb {
            if single(i) {
                let r = find(&mut parent, i);
                groups.entry(r).or_default().push(i);
            }
        }
        let t0 = std::time::Instant::now();
        let mut made = Vec::new();
        let mut rejected = Vec::new();
        for (_, cells) in groups {
            if cells.len() < 10 {
                continue;
            }
            match self.build_member(&cells) {
                Some(mem) => made.push(mem),
                None => rejected.push(json!([cells.len(), "basis"])),
            }
        }
        self.basis_wall += t0.elapsed().as_secs_f64();
        if !made.is_empty() {
            self.red += 1;
        }
        if !rejected.is_empty() {
            self.dirty = true;
        }
        self.events.push(json!({"t": self.t, "event": "reduce", "members": made.iter().map(|m| json!([m.cells.len(), m.frozen])).collect::<Vec<_>>(), "rejected": rejected}));
        for mem in made {
            let mi = self.members.len();
            for (c, &b) in mem.cells.iter().enumerate() {
                self.owner[b] = mi;
                self.local[b] = c;
            }
            self.members.push(Some(mem));
        }
        self.rebuild_ports();
    }

    fn drop_open_cracks(&mut self) {
        let mut any = false;
        for &fi in &self.ports.clone() {
            let m = &self.sc.m;
            let f = &m.facets[fi];
            if f.tag != 0 || !f.pts.iter().all(|p| p.law.kind == LawKind::Damage && p.kappa_c >= p.law.wc) {
                continue;
            }
            let (j, _) = m.eval_point(&self.q, f, 0);
            let (j2, _) = m.eval_point(&self.q, f, 1);
            let sep = j.dn.min(j2.dn) > 0.5 * f.len;
            let slid = j.dn.abs().max(j.dt.abs()).max(j2.dn.abs()).max(j2.dt.abs()) > 0.25 * f.len;
            let turned = (self.q[f.a][2] - self.q[f.b][2]).abs() > 0.2;
            if (sep || slid || turned) && j.dn >= 0.0 && j2.dn >= 0.0 {
                self.sc.m.facets[fi].active = false;
                any = true;
            }
        }
        if any {
            self.rebuild_ports();
        }
    }

    fn advance(&mut self) {
        if self.hybrid && self.dirty && self.t - self.last_damage >= self.tq && self.t - self.last_try >= self.tq {
            self.last_try = self.t;
            self.dirty = false;
            self.try_reduce();
        }
        // positions of port cells
        for mem in self.members.iter().flatten() {
            for &c in &mem.port_local {
                self.q[mem.cells[c]] = mem.pos(c);
            }
            self.flops += mem.port_local.len() as f64 * (6.0 * mem.k as f64 + 12.0) + 5.0 * mem.k as f64 + 20.0;
        }
        let nb = self.sc.m.bodies.len();
        for i in 0..nb {
            self.f[i] = if self.owner[i] == NONE { self.sc.m.loads[i] } else { [0.0; 3] };
        }
        let mut dd = 0.0;
        let mut grew = false;
        for &fi in &self.ports {
            for p in 0..2 {
                let (j, e) = self.sc.m.eval_point(&self.q, &self.sc.m.facets[fi], p);
                let (a, b) = (self.sc.m.facets[fi].a, self.sc.m.facets[fi].b);
                for d in 0..3 {
                    self.f[a][d] -= j.jna[d] * e.fnorm + j.jta[d] * e.ftan;
                    self.f[b][d] -= j.jnb[d] * e.fnorm + j.jtb[d] * e.ftan;
                }
                let tag = self.sc.m.facets[fi].tag;
                let pt = &mut self.sc.m.facets[fi].pts[p];
                if e.kappa > pt.kappa_c {
                    let g = pt.law.dissipation(e.kappa) - pt.law.dissipation(pt.kappa_c);
                    dd += g;
                    if g > 0.0 && tag == 0 {
                        grew = true;
                    }
                    pt.kappa_c = e.kappa;
                }
            }
        }
        self.sc.m.dissipated += dd;
        if grew {
            self.dirty = true;
            // quiet means: over a window of tq the interior dissipation grew by less than epsd of its total
            if self.sc.m.dissipated - self.win_d > self.epsd * self.sc.m.dissipated {
                self.last_damage = self.t;
                self.win_t = self.t;
                self.win_d = self.sc.m.dissipated;
            }
        }
        if self.t - self.win_t >= self.tq {
            self.win_t = self.t;
            self.win_d = self.sc.m.dissipated;
        }
        self.flops += 2.0 * C_PT * self.ports.len() as f64;
        // single bodies
        let mut ns = 0.0;
        for i in 0..nb {
            if self.owner[i] != NONE {
                continue;
            }
            let b = &self.sc.m.bodies[i];
            ns += 1.0;
            for d in 0..3 {
                if !b.fixed[d] {
                    self.v[i][d] += DT * self.f[i][d] / if d < 2 { b.mass } else { b.inertia };
                    self.q[i][d] += DT * self.v[i][d];
                }
            }
        }
        self.flops += 9.0 * ns;
        self.single_steps += ns;
        // members
        let mut esc = Vec::new();
        let mut dd_mem = 0.0;
        let (dtol, chipost) = (self.desc, self.chipost);
        for (mi, slot) in self.members.iter_mut().enumerate() {
            let Some(mem) = slot else { continue };
            let k = mem.k;
            let (s, co) = mem.th.sin_cos();
            let (mut fx, mut fy, mut tq) = (mem.load[0], mem.load[1], 0.0);
            if self.step % NS == 0 && mem.frozen {
                // frozen member: only watch the openings against the committed history
                let r = mem.exact();
                let c = mem.d0.len() as f64 * (2.0 * k as f64 + 10.0);
                self.flops += c;
                self.check_flops += c;
                if r >= chipost {
                    esc.push((mi, r));
                }
            } else if self.step % NS == 0 {
                // interior damage from modal openings, every NS steps; held pseudo-force
                let mut dmax = 0.0f64;
                mem.qd.iter_mut().for_each(|x| *x = 0.0);
                let mut nd = 0.0;
                for (i, &fi) in mem.int_f.iter().enumerate() {
                    for p in 0..2 {
                        let ip = 2 * i + p;
                        let (mut dn, mut dt) = (0.0, 0.0);
                        for j in 0..k {
                            dn += mem.sn[ip * k + j] * mem.eta[j];
                            dt += mem.st[ip * k + j] * mem.eta[j];
                        }
                        let pt = &mut self.sc.m.facets[fi].pts[p];
                        let e = pt.law.eval(dn, dt, pt.kappa_c);
                        if e.kappa > pt.kappa_c {
                            let g = pt.law.dissipation(e.kappa) - pt.law.dissipation(pt.kappa_c);
                            dd_mem += g;
                            pt.kappa_c = e.kappa;
                        }
                        dmax = dmax.max(e.d);
                        let (gn, gt) = (e.fnorm - pt.law.kn * dn, e.ftan - pt.law.kt * dt);
                        if gn != 0.0 || gt != 0.0 {
                            nd += 1.0;
                            for j in 0..k {
                                mem.qd[j] -= gn * mem.sn[ip * k + j] + gt * mem.st[ip * k + j];
                            }
                        }
                    }
                }
                let c = (2 * mem.int_f.len()) as f64 * (2.0 * k as f64 + 40.0) + nd * 2.0 * k as f64;
                self.flops += c;
                self.check_flops += c;
                if dmax >= dtol {
                    esc.push((mi, dmax));
                }
            }
            let mut qk: Vec<f64> = (0..k).map(|j| co * mem.pg[j] + s * mem.sg[j] - mem.lam[j] * mem.eta[j] + mem.qd[j]).collect();
            for &c in &mem.port_local {
                let b = mem.cells[c];
                let f = self.f[b];
                fx += f[0];
                fy += f[1];
                tq += (self.q[b][0] - mem.x[0]) * f[1] - (self.q[b][1] - mem.x[1]) * f[0] + f[2];
                let (bx, by) = (co * f[0] + s * f[1], -s * f[0] + co * f[1]);
                for j in 0..k {
                    qk[j] += mem.phi[(c * 3) * k + j] * bx + mem.phi[(c * 3 + 1) * k + j] * by + mem.phi[(c * 3 + 2) * k + j] * f[2];
                }
            }
            mem.v[0] += DT * fx / mem.mass;
            mem.v[1] += DT * fy / mem.mass;
            mem.om += DT * tq / mem.inertia;
            mem.x[0] += DT * mem.v[0];
            mem.x[1] += DT * mem.v[1];
            mem.th += DT * mem.om;
            for j in 0..k {
                mem.etad[j] += DT * qk[j];
                mem.eta[j] += DT * mem.etad[j];
            }
        }
        if dd_mem > 0.0 {
            self.sc.m.dissipated += dd_mem;
            self.dirty = true;
            if self.sc.m.dissipated - self.win_d > self.epsd * self.sc.m.dissipated {
                self.last_damage = self.t;
                self.win_t = self.t;
                self.win_d = self.sc.m.dissipated;
            }
        }
        self.t += DT;
        self.step += 1;
        for (mi, r) in esc {
            self.escalate(mi, r);
        }
        if self.step % 2083 == 0 {
            self.drop_open_cracks();
        }
    }

    fn sync_all(&mut self) {
        for mi in 0..self.members.len() {
            if self.members[mi].is_some() {
                self.sync_member(mi);
            }
        }
    }
}

fn imp_scene(seed: u64, v0: f64) -> Scene {
    let nspan: usize = std::env::var("SB_NSPAN").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    beam_impact(seed, 0.025, v0, 50.0, nspan)
}

fn run(sc: Scene, hybrid: bool, kmax: usize, chi: f64, tq: f64, t_end: f64) -> Value {
    let collar: usize = std::env::var("SB_COLLAR").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let t0 = std::time::Instant::now();
    let plate = sc.impactor.unwrap();
    let v0 = -sc.v_init[plate][1];
    let nc = sc.m.n_cells();
    let mut s = Sim::new(sc, hybrid, kmax, chi, tq, collar);
    let steps = (t_end / DT).round() as u64;
    for _ in 0..steps {
        s.advance();
    }
    s.sync_all();
    let mut hist = [0usize; 5];
    for f in s.sc.m.facets.iter().filter(|f| f.tag == 0) {
        let d = f.pts.iter().map(|p| p.law.damage(p.kappa_c)).fold(0.0, f64::max);
        hist[if d == 0.0 { 0 } else if d <= 0.1 { 1 } else if d <= 0.5 { 2 } else if d < 0.99 { 3 } else { 4 }] += 1;
    }
    let fr = fragments(&s.sc.m, &s.q, &s.v, 0.05);
    let cr = crack_stats(&s.sc.m);
    json!({"v0": v0, "hybrid": hybrid, "k": kmax, "broken": fr.len() >= 2, "frag_mass": fr.iter().map(|f| f.mass).collect::<Vec<_>>(), "frag_v": fr.iter().map(|f| f.v).collect::<Vec<_>>(),
        "cells": nc, "D": s.sc.m.dissipated, "crack_x": cr[0], "v_imp": s.v[plate][1], "flops": s.flops, "wall_s": t0.elapsed().as_secs_f64(), "basis_wall_s": s.basis_wall,
        "collar": collar, "damage_hist_0_0.1_0.5_0.99_1": hist, "check_flops": s.check_flops, "escalations": s.esc, "reductions": s.red, "single_fraction": s.single_steps / (steps as f64 * (nc + 2) as f64), "events": s.events})
}

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

fn vcrit(seed: u64, hybrid: bool, kmax: usize, chi: f64, tq: f64) -> f64 {
    let (mut lo, mut hi) = (0.2f64.ln(), 1.5f64.ln());
    for _ in 0..10 {
        let mid = 0.5 * (lo + hi);
        if run(imp_scene(seed, mid.exp()), hybrid, kmax, chi, tq, 0.05)["broken"].as_bool().unwrap() {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    (0.5 * (lo + hi)).exp()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let envf = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (kmax, chi, tq) = (envf("SB_K", 40.0) as usize, envf("SB_CHI", 0.8), envf("SB_TQ", 1e-3));
    let a1: Value = serde_json::from_str(&std::fs::read_to_string("results/phase1/a1_contact_thresholds.json").unwrap()).unwrap();
    let vref = |s: u64| a1["rows"].as_array().unwrap().iter().find(|r| r["seed"] == s && r["mp"] == 50.0 && r["ks"] == 1.0).unwrap()["v_crit"].as_f64().unwrap();
    match args.get(1).map(|s| s.as_str()) {
        Some("one") => {
            let (seed, v0, hybrid) = (args[2].parse().unwrap(), args[3].parse().unwrap(), args[4] == "h");
            let t_end = args.get(5).map_or(0.05, |x| x.parse().unwrap());
            println!("{}", run(imp_scene(seed, v0), hybrid, kmax, chi, tq, t_end));
        }
        Some(mode @ ("validate" | "hybrid")) => {
            let hybrid = mode == "hybrid";
            let mut jobs: Vec<(u64, f64)> = Vec::new();
            for s in [1u64, 2] {
                jobs.push((s, f64::NAN));
                for fac in [1.25, 2.0] {
                    jobs.push((s, fac * vref(s)));
                }
                if hybrid {
                    jobs.push((s, 0.15));
                }
            }
            let res = par_map(jobs.clone(), |(s, v0)| if v0.is_nan() { json!({"v_crit": vcrit(s, hybrid, kmax, chi, tq)}) } else { run(imp_scene(s, v0), hybrid, kmax, chi, tq, 0.05) });
            let rows: Vec<Value> = jobs.iter().zip(res).map(|(&(s, v0), mut r)| {
                r["seed"] = json!(s);
                r["factor"] = if v0.is_nan() { Value::Null } else { json!(v0 / vref(s)) };
                r
            }).collect();
            let name = if hybrid { format!("e_v3_k{kmax}_collar{}_tq{}", std::env::var("SB_COLLAR").unwrap_or("2".into()), tq * 1e3) } else { "e_validate_all_single".to_string() };
            write_json("phase1", &name, &json!({"meta": meta("phase1", &name, None, "member stepper, dt 2 us, beam impact 50 kg", json!({"k": kmax, "chi": chi, "tq": tq, "hybrid": hybrid})), "rows": rows}));
        }
        _ => eprintln!("usage: phase1e validate | hybrid | one seed v0 h|f [t_end]"),
    }
}
