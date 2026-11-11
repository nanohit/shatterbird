//! The R system: every closed patch as a floating-frame body, one BDF2 Newton
//! system, lagged factorisation. Facets between closed patches (or a patch and a
//! support) are evaluated with the full law on recovered cell positions; facets
//! touching the active zone are not part of R (they enter as the interface impulse).

use super::contact::{implicit_pair, CItem, ContactModel, GROUND};
use super::patch::Patch;
use super::{jperp, rot, Work, C_PT_FORCE};
use geometry::V2;
use model::{LawKind, Model};
use solvers::sparse::{Pattern, Spd};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rep {
    /// cell of closed patch number k (index into `closed`), local index li
    R { k: usize, li: usize, bnd: bool },
    E,
    Fixed,
}

pub struct RegStep {
    pub qhat: Vec<[f64; 3]>,
    pub a0: f64,
    pub b0: f64,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}

pub struct StepData {
    /// α/h² (0 = static)
    pub ah: f64,
    pub reg: Vec<RegStep>,
}

pub struct Extra<'a> {
    /// external forces on bodies at the end of the step (only R cells are used)
    pub ext: &'a [(usize, [f64; 3])],
    /// constant generalised interface force and the start state it does work from
    pub fj: Option<(&'a [f64], &'a [f64])>,
    /// Robin stabilisation: projected interface stiffness and its centre
    pub robin: Option<(&'a [(usize, usize, f64)], &'a [f64])>,
    /// contact pairs frozen at the step start and the step length (friction regularisation)
    pub contacts: Option<(&'a ContactModel, &'a [CItem], f64)>,
}

struct Fact {
    spd: Spd,
    ah: f64,
    valid: bool,
    roundoff: Vec<f64>,
}

pub struct RSys {
    pub closed: Vec<usize>,
    pub off: Vec<usize>,
    pub n: usize,
    pub rep: Vec<Rep>,
    pub eval_facets: Vec<usize>,
    pub iface_facets: Vec<usize>,
    pub iface_cells: Vec<usize>,
    pub bnd_cells: Vec<usize>,
    /// body → position in bnd_cells
    bidx: Vec<usize>,
    pattern: Vec<(usize, usize, f64)>,
    pat: Option<Pattern>,
    pub fact_est: f64,
    pub nnz_l: usize,
    fact: [Option<Fact>; 2],
    pub version: u64,
    /// generalised DOFs held fixed (single-body patches of partly fixed bodies)
    pub fixed_z: Vec<usize>,
}

type Jac = (Vec<[f64; 3]>, V2, Vec<V2>);

impl RSys {
    pub fn empty(nb: usize) -> RSys {
        RSys {
            closed: Vec::new(),
            off: Vec::new(),
            n: 0,
            rep: vec![Rep::E; nb],
            eval_facets: Vec::new(),
            iface_facets: Vec::new(),
            iface_cells: Vec::new(),
            bnd_cells: Vec::new(),
            bidx: vec![usize::MAX; nb],
            pattern: Vec::new(),
            pat: None,
            fact_est: 0.0,
            nnz_l: 0,
            fact: [None, None],
            version: 0,
            fixed_z: Vec::new(),
        }
    }

    pub fn build(m: &Model, patches: &[Patch], closed_flag: &[bool], loaded: &[bool], contacts: &[(usize, usize)], version: u64, w: &mut Work) -> RSys {
        let nb = m.bodies.len();
        let mut rs = RSys::empty(nb);
        rs.version = version;
        for i in 0..nb {
            if m.bodies[i].fixed.iter().all(|&f| f) {
                rs.rep[i] = Rep::Fixed;
            }
        }
        let mut in_khat = vec![false; m.facets.len()];
        for (pid, pa) in patches.iter().enumerate() {
            if !closed_flag[pid] {
                continue;
            }
            let k = rs.closed.len();
            rs.closed.push(pid);
            for (li, &i) in pa.cells.iter().enumerate() {
                rs.rep[i] = Rep::R { k, li, bnd: false };
            }
            for &fi in &pa.internal {
                in_khat[fi] = true;
            }
        }
        let is_r = |r: &Rep| matches!(r, Rep::R { .. });
        let mut bnd = vec![false; nb];
        let mut iface_c = vec![false; nb];
        for (fi, f) in m.facets.iter().enumerate() {
            if !f.active {
                continue;
            }
            let (ra, rb) = (rs.rep[f.a], rs.rep[f.b]);
            if !is_r(&ra) && !is_r(&rb) {
                continue;
            }
            if in_khat[fi] {
                if let (Rep::R { k: ka, .. }, Rep::R { k: kb, .. }) = (ra, rb) {
                    if ka == kb {
                        continue;
                    }
                }
            }
            if ra == Rep::E || rb == Rep::E {
                rs.iface_facets.push(fi);
                for s in [f.a, f.b] {
                    if is_r(&rs.rep[s]) {
                        bnd[s] = true;
                        iface_c[s] = true;
                    }
                }
            } else {
                rs.eval_facets.push(fi);
                for s in [f.a, f.b] {
                    if is_r(&rs.rep[s]) {
                        bnd[s] = true;
                    }
                }
            }
        }
        for &(a, b) in contacts {
            for c in [a, b] {
                if c != GROUND && is_r(&rs.rep[c]) {
                    bnd[c] = true;
                }
            }
        }
        for i in 0..nb {
            if loaded[i] && is_r(&rs.rep[i]) {
                bnd[i] = true;
            }
            if let Rep::R { k, li, .. } = rs.rep[i] {
                rs.rep[i] = Rep::R { k, li, bnd: bnd[i] };
                if bnd[i] {
                    rs.bidx[i] = rs.bnd_cells.len();
                    rs.bnd_cells.push(i);
                }
                if iface_c[i] {
                    rs.iface_cells.push(i);
                }
            }
        }
        let mut n = 0;
        for &pid in &rs.closed {
            rs.off.push(n);
            let pa = &patches[pid];
            if pa.cells.len() == 1 {
                let b = &m.bodies[pa.cells[0]];
                for k in 0..3 {
                    if b.fixed[k] {
                        rs.fixed_z.push(n + k);
                    }
                }
            }
            n += pa.ny();
        }
        rs.n = n;
        // block sparsity pattern: patch blocks + blocks of patch pairs joined by eval facets
        let mut pairs = std::collections::BTreeSet::new();
        for &fi in &rs.eval_facets {
            let f = &m.facets[fi];
            if let (Rep::R { k: ka, .. }, Rep::R { k: kb, .. }) = (rs.rep[f.a], rs.rep[f.b]) {
                if ka != kb {
                    pairs.insert((ka.min(kb), ka.max(kb)));
                }
            }
        }
        for &(a, b) in contacts {
            if b == GROUND {
                continue;
            }
            if let (Rep::R { k: ka, .. }, Rep::R { k: kb, .. }) = (rs.rep[a], rs.rep[b]) {
                if ka != kb {
                    pairs.insert((ka.min(kb), ka.max(kb)));
                }
            }
        }
        // the Robin term of the multirate coupling couples every pair of interface patches
        let mut ip: Vec<usize> = rs.iface_cells.iter().filter_map(|&i| if let Rep::R { k, .. } = rs.rep[i] { Some(k) } else { None }).collect();
        ip.sort_unstable();
        ip.dedup();
        for a in 0..ip.len() {
            for b in a + 1..ip.len() {
                pairs.insert((ip[a], ip[b]));
            }
        }
        let ny = |k: usize| patches[rs.closed[k]].ny();
        for k in 0..rs.closed.len() {
            let o = rs.off[k];
            for a in 0..ny(k) {
                for b in 0..ny(k) {
                    rs.pattern.push((o + a, o + b, 0.0));
                }
            }
        }
        for &(ka, kb) in &pairs {
            let (oa, ob) = (rs.off[ka], rs.off[kb]);
            for a in 0..ny(ka) {
                for b in 0..ny(kb) {
                    rs.pattern.push((oa + a, ob + b, 0.0));
                    rs.pattern.push((ob + b, oa + a, 0.0));
                }
            }
        }
        if n > 0 {
            let ent: Vec<(usize, usize)> = rs.pattern.iter().map(|&(r, c, _)| (r, c)).collect();
            let pat = Pattern::from_entries(n, &ent);
            let sp = Spd::analyze(&pat.matrix(vec![1.0; pat.nnz()]));
            rs.fact_est = sp.stats.flops_factor;
            rs.nnz_l = sp.stats.nnz_l;
            rs.fact[0] = Some(Fact { spd: sp, ah: f64::NAN, valid: false, roundoff: vec![0.0; rs.n] });
            rs.pat = Some(pat);
        }
        w.rebuild_flops += (m.facets.len() * 4) as f64;
        rs
    }

    pub fn invalidate(&mut self) {
        self.mark_stale(0);
        self.mark_stale(1);
    }

    fn y<'a>(&self, patches: &[Patch], k: usize, z: &'a [f64]) -> &'a [f64] {
        let o = self.off[k];
        &z[o..o + patches[self.closed[k]].ny()]
    }

    pub fn cell_q(&self, patches: &[Patch], k: usize, li: usize, z: &[f64]) -> [f64; 3] {
        let pa = &patches[self.closed[k]];
        let y = self.y(patches, k, z);
        let (r, wth) = pa.local(li, &y[3..]);
        let s = rot(r, y[2]);
        [pa.xc.x + y[0] + s.x, pa.xc.y + y[1] + s.y, y[2] + wth]
    }

    /// d q_i / d y (ny rows of [dx, dy, dθ]), s = R(φ) r_i, and R(φ) U_j.
    pub fn cell_jac(&self, patches: &[Patch], k: usize, li: usize, z: &[f64]) -> Jac {
        let pa = &patches[self.closed[k]];
        let y = self.y(patches, k, z);
        let (r, _) = pa.local(li, &y[3..]);
        let phi = y[2];
        let s = rot(r, phi);
        let me = pa.me;
        let mut d = vec![[0.0; 3]; pa.ny()];
        d[0] = [1.0, 0.0, 0.0];
        d[1] = [0.0, 1.0, 0.0];
        let js = jperp(s);
        d[2] = [js.x, js.y, 1.0];
        let mut ru = Vec::with_capacity(me);
        let (cs, sn) = (phi.cos(), phi.sin());
        for j in 0..me {
            let u = V2::new(pa.te[(3 * li) * me + j], pa.te[(3 * li + 1) * me + j]);
            let rr = u.rot_cs(cs, sn);
            d[3 + j] = [rr.x, rr.y, pa.te[(3 * li + 2) * me + j]];
            ru.push(rr);
        }
        (d, s, ru)
    }

    /// Write every R cell's configuration into q.
    pub fn q_all(&self, patches: &[Patch], z: &[f64], q: &mut [[f64; 3]], w: &mut Work) {
        for (k, &pid) in self.closed.iter().enumerate() {
            let pa = &patches[pid];
            for (li, &i) in pa.cells.iter().enumerate() {
                q[i] = self.cell_q(patches, k, li, z);
            }
            w.step_flops += (6 * pa.cells.len() * (pa.me + 2)) as f64;
        }
    }

    /// Reduced coordinates from cell positions (M-orthogonal mean-axis fit).
    pub fn fit(&self, patches: &[Patch], q: &[[f64; 3]], w: &mut Work) -> Vec<f64> {
        let mut z = vec![0.0; self.n];
        for (k, &pid) in self.closed.iter().enumerate() {
            let y = fit_patch(&patches[pid], q, false, None);
            z[self.off[k]..self.off[k] + y.len()].copy_from_slice(&y);
            w.step_flops += 8.0 * (patches[pid].cells.len() * patches[pid].ny()) as f64;
        }
        z
    }

    pub fn step_data(&self, patches: &[Patch], qhat: &[[f64; 3]], ah: f64, w: &mut Work) -> StepData {
        let mut reg = Vec::new();
        for &pid in &self.closed {
            let rg = &patches[pid];
            let me = rg.me;
            let (mut a0, mut b0) = (0.0, 0.0);
            let (mut a, mut b) = (vec![0.0; me], vec![0.0; me]);
            for (li, &i) in rg.cells.iter().enumerate() {
                let x = V2::new(qhat[i][0], qhat[i][1]) - rg.xc;
                let jx = jperp(x);
                let mm = rg.mi[li];
                a0 += mm * x.dot(rg.p[li]);
                b0 += mm * jx.dot(rg.p[li]);
                for j in 0..me {
                    let (ux, uy) = (rg.te[(3 * li) * me + j], rg.te[(3 * li + 1) * me + j]);
                    a[j] += mm * (x.x * ux + x.y * uy);
                    b[j] += mm * (jx.x * ux + jx.y * uy);
                }
            }
            w.step_flops += 12.0 * (rg.cells.len() * me.max(1)) as f64;
            reg.push(RegStep { qhat: rg.cells.iter().map(|&i| qhat[i]).collect(), a0, b0, a, b });
        }
        StepData { ah, reg }
    }

    /// Gradient, Hessian triplets (if want_h) and incremental potential.
    pub fn assemble(&self, m: &Model, patches: &[Patch], z: &[f64], sd: &StepData, ex: &Extra, want_h: bool, w: &mut Work) -> (Vec<f64>, Vec<(usize, usize, f64)>, f64) {
        self.assemble_impl(m, patches, z, sd, ex, want_h, false, w)
    }

    fn assemble_impl(&self, m: &Model, patches: &[Patch], z: &[f64], sd: &StepData, ex: &Extra, want_h: bool, secant: bool, w: &mut Work) -> (Vec<f64>, Vec<(usize, usize, f64)>, f64) {
        let n = self.n;
        let mut g = vec![0.0; n];
        let mut trip = Vec::new();
        let mut energy = 0.0;
        // recover boundary cells
        let mut q: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
        let mut jac: Vec<Jac> = Vec::with_capacity(self.bnd_cells.len());
        for &i in &self.bnd_cells {
            let Rep::R { k, li, .. } = self.rep[i] else { unreachable!() };
            q[i] = self.cell_q(patches, k, li, z);
            jac.push(self.cell_jac(patches, k, li, z));
            w.coupling_flops += (12 * patches[self.closed[k]].ny()) as f64;
        }
        for (i, b) in m.bodies.iter().enumerate() {
            if self.rep[i] == Rep::Fixed {
                q[i] = [b.x0.x, b.x0.y, 0.0];
            }
        }
        // cell-level gradient and Hessian blocks
        let nbc = self.bnd_cells.len();
        let mut gc = vec![[0.0f64; 3]; nbc];
        let mut dblk = vec![[[0.0f64; 3]; 3]; nbc];
        // external forces on R cells
        for &(i, f) in ex.ext {
            if let Rep::R { .. } = self.rep[i] {
                let bi = self.bidx[i];
                debug_assert!(bi != usize::MAX, "loaded R cell not on boundary");
                let u = [q[i][0] - m.bodies[i].x0.x, q[i][1] - m.bodies[i].x0.y, q[i][2]];
                for r in 0..3 {
                    gc[bi][r] -= f[r];
                    energy -= f[r] * u[r];
                }
            }
        }
        for &fi in &self.eval_facets {
            let f = &m.facets[fi];
            let bodies = [f.a, f.b];
            let mut h6 = [[0.0f64; 6]; 6];
            for pt in 0..2 {
                let (j, e) = m.eval_point(&q, f, pt);
                energy += f.pts[pt].law.potential(j.dn, j.dt, f.pts[pt].kappa_c);
                let jn = [j.jna[0], j.jna[1], j.jna[2], j.jnb[0], j.jnb[1], j.jnb[2]];
                let jt = [j.jta[0], j.jta[1], j.jta[2], j.jtb[0], j.jtb[1], j.jtb[2]];
                for s in 0..2 {
                    let bi = self.bidx[bodies[s]];
                    if bi == usize::MAX {
                        continue;
                    }
                    for r in 0..3 {
                        gc[bi][r] += jn[3 * s + r] * e.fnorm + jt[3 * s + r] * e.ftan;
                    }
                }
                if want_h {
                    w.pts_tangent += 1;
                    // Change only the material Hessian, never the physical force or
                    // potential. In compression even a broken cohesive point retains kn.
                    let dd = if secant {
                        let (kn, kt) = f.pts[pt].law.secant(j.dn, e.d);
                        [kn, 0.0, kt]
                    } else { e.tan };
                    for r in 0..6 {
                        let ar = dd[0] * jn[r] + dd[1] * jt[r];
                        let br = dd[1] * jn[r] + dd[2] * jt[r];
                        for c in 0..6 {
                            h6[r][c] += ar * jn[c] + br * jt[c];
                        }
                    }
                } else {
                    w.pts_force += 1;
                }
            }
            if !want_h {
                continue;
            }
            for sa in 0..2 {
                let ba = self.bidx[bodies[sa]];
                if ba == usize::MAX {
                    continue;
                }
                for k in 0..3 {
                    for l in 0..3 {
                        dblk[ba][k][l] += h6[3 * sa + k][3 * sa + l];
                    }
                }
            }
            let (ba, bb) = (self.bidx[bodies[0]], self.bidx[bodies[1]]);
            if ba != usize::MAX && bb != usize::MAX && bodies[0] != bodies[1] {
                let (Rep::R { k: ka, .. }, Rep::R { k: kb, .. }) = (self.rep[bodies[0]], self.rep[bodies[1]]) else { unreachable!() };
                let (oa, ob) = (self.off[ka], self.off[kb]);
                let (da, db) = (&jac[ba].0, &jac[bb].0);
                // H_ab G_b (3 × ny_b), then G_aᵀ (·)
                let hb: Vec<[f64; 3]> = db.iter().map(|dj| std::array::from_fn(|kk| (0..3).map(|l| h6[kk][3 + l] * dj[l]).sum())).collect();
                for (ii, di) in da.iter().enumerate() {
                    for (jj, hj) in hb.iter().enumerate() {
                        let v = di[0] * hj[0] + di[1] * hj[1] + di[2] * hj[2];
                        trip.push((oa + ii, ob + jj, v));
                        trip.push((ob + jj, oa + ii, v));
                    }
                }
                w.coupling_flops += (9 * db.len() + 6 * da.len() * db.len()) as f64;
            }
        }
        // contacts: penalty normal force and lagged-normal regularised friction
        if let Some((cm, items, hstep)) = ex.contacts {
            for it in items {
                let (e, g6, h6, _) = implicit_pair(cm, it, &q, hstep);
                energy += e;
                let bodies = [it.a, it.b];
                for s_ in 0..2 {
                    if bodies[s_] == GROUND {
                        continue;
                    }
                    let bi = self.bidx[bodies[s_]];
                    if bi == usize::MAX {
                        continue;
                    }
                    for r in 0..3 {
                        gc[bi][r] += g6[3 * s_ + r];
                    }
                    if want_h {
                        for k in 0..3 {
                            for l in 0..3 {
                                dblk[bi][k][l] += h6[3 * s_ + k][3 * s_ + l];
                            }
                        }
                    }
                }
                w.contact_flops += 200.0;
                if want_h && it.b != GROUND {
                    let (ba, bb) = (self.bidx[it.a], self.bidx[it.b]);
                    if ba != usize::MAX && bb != usize::MAX {
                        let (Rep::R { k: ka, .. }, Rep::R { k: kb, .. }) = (self.rep[it.a], self.rep[it.b]) else { continue };
                        let (oa, ob) = (self.off[ka], self.off[kb]);
                        let (da, db) = (&jac[ba].0, &jac[bb].0);
                        let hb: Vec<[f64; 3]> = db.iter().map(|dj| std::array::from_fn(|kk| (0..3).map(|l| h6[kk][3 + l] * dj[l]).sum())).collect();
                        for (ii, di) in da.iter().enumerate() {
                            for (jj, hj) in hb.iter().enumerate() {
                                let v = di[0] * hj[0] + di[1] * hj[1] + di[2] * hj[2];
                                trip.push((oa + ii, ob + jj, v));
                                trip.push((ob + jj, oa + ii, v));
                            }
                        }
                    }
                }
            }
        }
        // project cell gradients and diagonal blocks; geometric terms
        for (bi, &i) in self.bnd_cells.iter().enumerate() {
            let Rep::R { k, .. } = self.rep[i] else { unreachable!() };
            let off = self.off[k];
            let (d, s, ru) = &jac[bi];
            let gcell = gc[bi];
            for (jj, dj) in d.iter().enumerate() {
                g[off + jj] += dj[0] * gcell[0] + dj[1] * gcell[1] + dj[2] * gcell[2];
            }
            w.coupling_flops += (6 * d.len()) as f64;
            if !want_h {
                continue;
            }
            let blk = &dblk[bi];
            let ny = d.len();
            if *blk != [[0.0; 3]; 3] {
                let hd: Vec<[f64; 3]> = d.iter().map(|dj| std::array::from_fn(|kk| blk[kk][0] * dj[0] + blk[kk][1] * dj[1] + blk[kk][2] * dj[2])).collect();
                for a in 0..ny {
                    for b in 0..ny {
                        let v = d[a][0] * hd[b][0] + d[a][1] * hd[b][1] + d[a][2] * hd[b][2];
                        trip.push((off + a, off + b, v));
                    }
                }
                w.coupling_flops += (18 * ny + 6 * ny * ny) as f64;
            }
            let gxy = V2::new(gcell[0], gcell[1]);
            if gxy != V2::ZERO {
                trip.push((off + 2, off + 2, -gxy.dot(*s)));
                for j in 0..ru.len() {
                    let v = gxy.dot(jperp(ru[j]));
                    trip.push((off + 2, off + 3 + j, v));
                    trip.push((off + 3 + j, off + 2, v));
                }
            }
        }
        // patches: K̂, gravity on c, inertia invariants
        for (k, &pid) in self.closed.iter().enumerate() {
            let rg = &patches[pid];
            let me = rg.me;
            let o = self.off[k];
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
                        trip.push((o + 3 + a, o + 3 + b, rg.khat[a * me + b]));
                    }
                }
            }
            let fg: V2 = rg.cells.iter().fold(V2::ZERO, |s, &i| s + V2::new(m.loads[i][0], m.loads[i][1]));
            g[o] -= fg.x;
            g[o + 1] -= fg.y;
            energy -= fg.x * y[0] + fg.y * y[1];
            if sd.ah > 0.0 {
                let rs = &sd.reg[k];
                let ah = sd.ah;
                let phi = y[2];
                let (cs, sn) = (phi.cos(), phi.sin());
                let ae = rs.a0 + (0..me).map(|j| rs.a[j] * eta[j]).sum::<f64>();
                let be = rs.b0 + (0..me).map(|j| rs.b[j] * eta[j]).sum::<f64>();
                // Evaluate the inertial residual as short differences, rather than
                // subtracting O(m*x/h^2) absolute-coordinate moments. The latter
                // falsely stalled Newton and hid behind the small-step escape.
                for li in 0..rg.cells.len() {
                    let qi = self.cell_q(patches, k, li, z);
                    let dx = [qi[0]-rs.qhat[li][0], qi[1]-rs.qhat[li][1], qi[2]-rs.qhat[li][2]];
                    let mm = [rg.mi[li], rg.mi[li], rg.ii[li]];
                    let dj = self.cell_jac(patches, k, li, z).0;
                    energy += 0.5 * ah * (0..3).map(|r| mm[r]*dx[r]*dx[r]).sum::<f64>();
                    for j in 0..rg.ny() {
                        g[o+j] += ah * (0..3).map(|r| dj[j][r]*mm[r]*dx[r]).sum::<f64>();
                    }
                }
                w.region_flops += (rg.cells.len() * (18 + 12*rg.ny())) as f64;
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
        if let Some((fj, z0)) = ex.fj {
            for a in 0..n {
                g[a] -= fj[a];
                energy -= fj[a] * (z[a] - z0[a]);
            }
        }
        if !self.fixed_z.is_empty() {
            let mut fx = vec![false; n];
            for &d in &self.fixed_z {
                fx[d] = true;
                g[d] = 0.0;
            }
            if want_h {
                trip.retain(|&(r, c, _)| !fx[r] && !fx[c]);
                for &d in &self.fixed_z {
                    trip.push((d, d, 1.0));
                }
            }
        }
        if let Some((kt, zc)) = ex.robin {
            let mut kd = vec![0.0; n];
            for &(r, c, v) in kt {
                kd[r] += v * (z[c] - zc[c]);
            }
            for a in 0..n {
                g[a] += kd[a];
                energy += 0.5 * kd[a] * (z[a] - zc[a]);
            }
            if want_h {
                trip.extend_from_slice(kt);
            }
            w.region_flops += 2.0 * kt.len() as f64;
        }
        (g, trip, energy)
    }

    fn factorize(&mut self, trip: Vec<(usize, usize, f64)>, slot: usize, ah: f64, w: &mut Work) -> bool {
        // scatter into the fixed pattern (extend it once if an entry falls outside)
        let vals = match self.pat.as_ref().and_then(|p| p.scatter(&trip)) {
            Some(v) => v,
            None => {
                let mut ent: Vec<(usize, usize)> = self.pattern.iter().map(|&(r, c, _)| (r, c)).collect();
                ent.extend(trip.iter().map(|&(r, c, _)| (r, c)));
                self.pattern = ent.iter().map(|&(r, c)| (r, c, 0.0)).collect();
                let pat = Pattern::from_entries(self.n, &ent);
                self.fact = [None, None];
                let v = pat.scatter(&trip).expect("pattern");
                self.pat = Some(pat);
                v
            }
        };
        let mat = self.pat.as_ref().unwrap().matrix(vals);
        let mut spd = match self.fact[slot].take() {
            Some(f) => f.spd,
            None => Spd::analyze(&mat),
        };
        w.fact_flops += self.fact_est;
        w.factorizations += 1;
        let ok = spd.factor(&mat).is_ok();
        self.fact[slot] = Some(Fact { spd, ah, valid: ok, roundoff: vec![0.0; self.n] });
        ok
    }

    fn fact_valid(&self, slot: usize) -> bool {
        self.fact[slot].as_ref().map(|f| f.valid).unwrap_or(false)
    }

    fn mark_stale(&mut self, slot: usize) {
        if let Some(f) = self.fact[slot].as_mut() {
            f.valid = false;
        }
    }

    /// Newton with a lagged factorisation (slot 0 plain, slot 1 with Robin term).
    pub fn solve(&mut self, m: &Model, patches: &[Patch], z: &mut Vec<f64>, sd: &StepData, ex: &Extra, tol: f64, max_iter: usize, w: &mut Work) -> bool {
        if self.n == 0 {
            return true;
        }
        let slot = ex.robin.is_some() as usize;
        if let Some(f) = &self.fact[slot] {
            if f.ah != sd.ah {
                self.mark_stale(slot);
            }
        }
        let mut fresh = false;
        let mut secant = false;
        // At a microsecond step a few ULPs of position can mean a force residual
        // above the static tolerance after multiplication by M/h^2. Bound that
        // attainable residual with |H| times 64 ULPs in physical coordinates.
        // Static force tests retain their original force-only criterion.
        let mut ulp = vec![0.0; self.n];
        if sd.ah > 0.0 {
            for (k,&pid) in self.closed.iter().enumerate() {
                let pa=&patches[pid];let o=self.off[k];
                let dx=64.0*f64::EPSILON*(1.0+pa.xc.x.abs()+pa.xc.y.abs()+pa.radius);
                ulp[o]=dx;ulp[o+1]=dx;ulp[o+2]=dx/pa.radius.max((pa.it/pa.mass).sqrt()).max(1e-3);
                for j in 0..pa.me {ulp[o+3+j]=dx*pa.mass.sqrt();}
            }
        }
        let (mut g, _, mut e0) = self.assemble(m, patches, z, sd, ex, false, w);
        for iteration in 0..max_iter {
            let res = g.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            if std::env::var_os("SB_NEWTON_TRACE").is_some() {
                eprintln!("Newton it={iteration} ah={} res={res:.8e} tol={tol:.8e}", sd.ah);
            }
            if res <= tol {
                return true;
            }
            if self.fact_valid(slot) && g.iter().zip(&self.fact[slot].as_ref().unwrap().roundoff).all(|(g,r)|g.abs()<=tol+r) {
                w.roundoff_accepts+=1;
                return true;
            }
            if !self.fact_valid(slot) {
                let (_, mut trip, _) = self.assemble(m, patches, z, sd, ex, true, w);
                if secant {
                    // drop the softening part: Gauss–Newton with the elastic facet stiffness
                    trip = self.secant_trip(m, patches, z, sd, ex, w);
                }
                let mut roundoff=vec![0.0;self.n];
                for &(r,c,v) in &trip {roundoff[r]+=v.abs()*ulp[c];}
                if !self.factorize(trip, slot, sd.ah, w) {
                    if secant {
                        return false;
                    }
                    secant = true;
                    continue;
                }
                self.fact[slot].as_mut().unwrap().roundoff=roundoff;
                if g.iter().zip(&self.fact[slot].as_ref().unwrap().roundoff).all(|(g,r)|g.abs()<=tol+r) {
                    w.roundoff_accepts+=1;
                    return true;
                }
                fresh = true;
            }
            let mut p: Vec<f64> = g.iter().map(|v| -v).collect();
            self.fact[slot].as_mut().unwrap().spd.solve(&mut p);
            w.solve_flops += 4.0 * self.nnz_l as f64;
            w.newton_iters += 1;
            let gp: f64 = g.iter().zip(&p).map(|(a, b)| a * b).sum();
            if !(gp < 0.0) || p.iter().any(|v| !v.is_finite()) {
                if fresh {
                    if secant {
                        return false;
                    }
                    secant = true;
                }
                self.mark_stale(slot);
                continue;
            }
            let pmax = p.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            if pmax == 0.0 {
                // A tiny displacement is not a force residual certificate.
                return res <= tol;
            }
            let mut alpha = 1.0;
            let mut accepted = false;
            for _ in 0..20 {
                let zt: Vec<f64> = z.iter().zip(&p).map(|(a, b)| a + alpha * b).collect();
                let (gt, _, e1) = self.assemble(m, patches, &zt, sd, ex, false, w);
                let rt = gt.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
                let energy_noise = 64.0 * f64::EPSILON * (1.0 + e0.abs());
                if e1 <= e0 + 1e-4 * alpha * gp || (e1 <= e0 + energy_noise && rt < (1.0 - 1e-4 * alpha) * res) {
                    *z = zt;
                    // lagged Jacobian: refresh when the contraction is poor
                    if rt > 0.5 * res {
                        self.mark_stale(slot);
                    }
                    g = gt;
                    e0 = e1;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                if fresh {
                    return false;
                }
                self.mark_stale(slot);
            }
            fresh = false;
        }
        g.iter().fold(0.0f64, |a, &v| a.max(v.abs())) <= tol
    }

    fn secant_trip(&self, m: &Model, patches: &[Patch], z: &[f64], sd: &StepData, ex: &Extra, w: &mut Work) -> Vec<(usize, usize, f64)> {
        self.assemble_impl(m, patches, z, sd, ex, true, true, w).1
    }

    /// Elastic energy of the R part: eval facets + K̂.
    pub fn elastic_energy(&self, m: &Model, patches: &[Patch], q: &[[f64; 3]], z: &[f64]) -> f64 {
        let mut u = 0.0;
        for &fi in &self.eval_facets {
            for pt in 0..2 {
                u += m.eval_point(q, &m.facets[fi], pt).1.psi;
            }
        }
        for (k, &pid) in self.closed.iter().enumerate() {
            let pa = &patches[pid];
            let eta = &z[self.off[k] + 3..self.off[k] + pa.ny()];
            for a in 0..pa.me {
                for b in 0..pa.me {
                    u += 0.5 * eta[a] * pa.khat[a * pa.me + b] * eta[b];
                }
            }
        }
        u
    }

    /// χ_rel = δ_eq / max(κ_c, δ₀) over closed-patch internal facets and R eval
    /// material facets that can still damage. Returns (max, hot facets ≥ trig).
    pub fn monitor(&self, m: &Model, patches: &[Patch], q: &[[f64; 3]], trig: f64, w: &mut Work) -> (f64, Vec<usize>) {
        let mut cmax: f64 = 0.0;
        let mut hot = Vec::new();
        let check = |fi: usize, cmax: &mut f64, hot: &mut Vec<usize>, w: &mut Work| {
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
                // undamaged points fire at `trig` (margin to onset); damaged points only when
                // they would damage further (χ_rel ≥ 1: reloading past their history)
                let c = (dn * dn + p.law.gamma * j.dt * j.dt).sqrt() / p.kappa_c.max(p.law.delta0);
                cf = cf.max(if p.kappa_c > p.law.delta0 { c * trig.min(1.0) } else { c });
                w.monitor_flops += C_PT_FORCE;
            }
            *cmax = cmax.max(cf);
            if cf >= trig {
                hot.push(fi);
            }
        };
        for &pid in &self.closed {
            for &fi in &patches[pid].internal {
                check(fi, &mut cmax, &mut hot, w);
            }
        }
        for &fi in &self.eval_facets {
            check(fi, &mut cmax, &mut hot, w);
        }
        (cmax, hot)
    }
}

/// Fit one patch to cell positions: centroid, mean-axis rotation, then η either
/// M-orthogonally (k_fit = false) or K-orthogonally in the body frame.
/// `kd` (optional) receives the residual's strain-energy direction (unused when None).
pub fn fit_patch(pa: &Patch, q: &[[f64; 3]], k_fit: bool, kmat: Option<&dyn Fn(&[f64]) -> Vec<f64>>) -> Vec<f64> {
    let nr = pa.cells.len();
    let c = pa.cells.iter().zip(&pa.mi).fold(V2::ZERO, |s, (&i, &mm)| s + V2::new(q[i][0], q[i][1]) * mm) * (1.0 / pa.mass);
    let mut phi = {
        let (mut sn, mut cs) = (0.0, 0.0);
        for (li, &i) in pa.cells.iter().enumerate() {
            let d = V2::new(q[i][0], q[i][1]) - c;
            cs += pa.mi[li] * pa.p[li].dot(d);
            sn += pa.mi[li] * (pa.p[li].x * d.y - pa.p[li].y * d.x);
        }
        if nr == 1 {
            q[pa.cells[0]][2]
        } else {
            sn.atan2(cs)
        }
    };
    // keep φ on the branch of the cells' own rotations
    let th_mean = pa.cells.iter().zip(&pa.ii).fold(0.0, |s, (&i, &ii)| s + ii * q[i][2]) / pa.it;
    let tau = 2.0 * std::f64::consts::PI;
    phi += ((th_mean - phi) / tau).round() * tau;
    let resid = |phi: f64| -> Vec<[f64; 3]> {
        pa.cells
            .iter()
            .enumerate()
            .map(|(li, &i)| {
                let d = rot(V2::new(q[i][0], q[i][1]) - c, -phi) - pa.p[li];
                [d.x, d.y, q[i][2] - phi]
            })
            .collect()
    };
    for _ in 0..30 {
        let d = resid(phi);
        let (mut num, mut den) = (0.0, 0.0);
        for li in 0..nr {
            num += pa.mi[li] * (pa.p[li].x * d[li][1] - pa.p[li].y * d[li][0]) + pa.ii[li] * d[li][2];
            den += pa.mi[li] * pa.p[li].dot(pa.p[li]) + pa.ii[li];
        }
        let dphi = num / den;
        phi += dphi;
        if dphi.abs() < 1e-15 {
            break;
        }
    }
    let d = resid(phi);
    let me = pa.me;
    let mut y = vec![0.0; 3 + me];
    y[0] = c.x - pa.xc.x;
    y[1] = c.y - pa.xc.y;
    y[2] = phi;
    if me == 0 {
        return y;
    }
    if k_fit {
        let kf = kmat.expect("K operator");
        let flat: Vec<f64> = d.iter().flat_map(|r| r.iter().copied()).collect();
        let kd = kf(&flat);
        let mut rhs: Vec<f64> = (0..me).map(|j| (0..3 * nr).map(|a| pa.te[a * me + j] * kd[a]).sum()).collect();
        super::patch::chol_solve(&pa.khat_l, me, &mut rhs);
        y[3..].copy_from_slice(&rhs);
    } else {
        for j in 0..me {
            y[3 + j] = (0..nr).map(|li| pa.mi[li] * (d[li][0] * pa.te[(3 * li) * me + j] + d[li][1] * pa.te[(3 * li + 1) * me + j]) + pa.ii[li] * d[li][2] * pa.te[(3 * li + 2) * me + j]).sum();
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{rect_tess,engine::patch::build_patch};
    use model::{Material,LawOptions};
    #[test]
    fn secant_fallback_keeps_broken_point_compression_and_real_gradient() {
        let tess=rect_tess(V2::ZERO,V2::new(0.5,0.3),0.15,5);
        let mut m=Model::from_tessellation(&tess,Material::concrete(0.2),LawOptions::elastic(),&[]);
        let fi=m.facets.iter().position(|f|f.tag==0).unwrap();
        let f=&mut m.facets[fi];f.pts[0].law.kind=LawKind::Damage;f.pts[0].kappa_c=f.pts[0].law.wc;
        let (b,n)=(f.b,f.n0);let mut q=m.q.clone();q[b][0]-=1e-5*n.x;q[b][1]-=1e-5*n.y;
        let adj=m.neighbours();let loaded=vec![false;m.bodies.len()];let mut w=Work::default();
        let p:Vec<_>=(0..m.n_cells()).map(|i|build_patch(&m,&adj,&[i],0,0,&loaded,&mut w)).collect();
        let rs=RSys::build(&m,&p,&vec![true;p.len()],&loaded,&[],1,&mut w);
        let z=rs.fit(&p,&q,&mut w);let sd=rs.step_data(&p,&q,1e6,&mut w);
        let ex=Extra{ext:&[],fj:None,robin:None,contacts:None};
        let (g,h,e)=rs.assemble(&m,&p,&z,&sd,&ex,true,&mut w);
        let (gs,hs,es)=rs.assemble_impl(&m,&p,&z,&sd,&ex,true,true,&mut w);
        assert_eq!(g,gs);assert_eq!(e,es);
        assert_eq!(h,hs,"fully broken compression must not lose kn in the fallback");
    }
}
