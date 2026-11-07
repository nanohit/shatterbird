//! Contact between cells and the ground (circles against the half-plane y ≥ y₀) and
//! between cells not joined by a facet (circle pairs). Normal penalty k_c = E b,
//! regularised Coulomb friction (μ, slip regularisation ε); friction work is booked
//! as D_fric. The same pair functions serve the explicit forces and the implicit
//! incremental potential (lagged normal force for friction).

use geometry::V2;
use model::{BodyKind, Model};
use std::collections::{BTreeMap, BTreeSet};

pub const GROUND: usize = usize::MAX;

#[derive(Clone, Debug)]
pub struct ContactModel {
    /// circle radius per body (0 = no contact)
    pub radius: Vec<f64>,
    pub kc: f64,
    /// reference size: the pair stiffness is k_c·min(1, 2 min(r_a, r_b)/d_ref)
    pub d_ref: f64,
    pub mu: f64,
    /// slip-velocity regularisation of friction [m/s]
    pub ev: f64,
    pub ground_y: Option<f64>,
    /// pairs never in circle contact (joined by a facet at build, or overlapping at rest)
    pub excluded: BTreeSet<(usize, usize)>,
    /// ground contact acts at the cells' polygon vertices: body-frame arms and stiffness
    /// (E b × tributary edge length / cell radius, like a support facet)
    pub verts: Vec<Vec<(V2, f64)>>,
}

impl ContactModel {
    pub fn new(m: &Model, ground_y: Option<f64>, mu: f64) -> ContactModel {
        let nb = m.bodies.len();
        // half the smallest centroid distance to a neighbour: neighbours' circles just
        // touch in the reference packing, so re-approaching pieces meet without overlap
        let mut radius: Vec<f64> = m.bodies.iter().map(|b| if b.kind == BodyKind::Cell { f64::INFINITY } else { 0.0 }).collect();
        for f in &m.facets {
            if f.a < nb && f.b < nb && m.bodies[f.a].kind == BodyKind::Cell && m.bodies[f.b].kind == BodyKind::Cell {
                // contact acts on the DOF points (centroids), so the radius comes from them
                let d = (m.bodies[f.b].x0 - m.bodies[f.a].x0).norm();
                radius[f.a] = radius[f.a].min(0.5 * d);
                radius[f.b] = radius[f.b].min(0.5 * d);
            }
        }
        for (i, r) in radius.iter_mut().enumerate() {
            if !r.is_finite() {
                *r = if m.bodies[i].kind == BodyKind::Cell { (m.bodies[i].mass / (m.mat.density * m.mat.thickness) / std::f64::consts::PI).sqrt() } else { 0.0 };
            }
        }
        let mut excluded = BTreeSet::new();
        for f in &m.facets {
            if f.a < nb && f.b < nb {
                excluded.insert((f.a.min(f.b), f.a.max(f.b)));
            }
        }
        let rs: Vec<f64> = radius.iter().copied().filter(|&r| r > 0.0).collect();
        let mut sorted = rs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let d_ref = if sorted.is_empty() { 1.0 } else { 2.0 * sorted[sorted.len() / 2] };
        let verts: Vec<Vec<(V2, f64)>> = m
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| {
                if b.kind != BodyKind::Cell || b.poly.is_empty() {
                    return Vec::new();
                }
                let k = b.poly.len();
                (0..k)
                    .map(|v| {
                        let l = 0.5 * ((b.poly[v] - b.poly[(v + k - 1) % k]).norm() + (b.poly[(v + 1) % k] - b.poly[v]).norm());
                        (b.poly[v] - b.x0, m.mat.e * m.mat.thickness * l / radius[i].max(1e-9))
                    })
                    .collect()
            })
            .collect();
        let mut cm = ContactModel { radius, kc: m.mat.e * m.mat.thickness, d_ref, mu, ev: 0.05, ground_y, excluded, verts };
        // pairs overlapping in the reference state interact through their neighbours' facets
        let q0: Vec<[f64; 3]> = m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
        for (a, b) in cm.near_pairs(&q0, 0.0) {
            // (ground pairs are never excluded)
            if b != GROUND {
                cm.excluded.insert((a.min(b), a.max(b)));
            }
        }
        cm
    }

    /// Cell pairs (and cell–ground) whose circles are closer than `margin`.
    pub fn near_pairs(&self, q: &[[f64; 3]], margin: f64) -> Vec<(usize, usize)> {
        let rmax = self.radius.iter().fold(0.0f64, |a, &r| a.max(r));
        if rmax == 0.0 {
            return Vec::new();
        }
        let cell = 2.0 * rmax + margin;
        let mut grid: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
        for (i, &r) in self.radius.iter().enumerate() {
            if r > 0.0 {
                grid.entry(((q[i][0] / cell).floor() as i64, (q[i][1] / cell).floor() as i64)).or_default().push(i);
            }
        }
        let mut out = Vec::new();
        for (&(gx, gy), list) in &grid {
            for &i in list {
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        if let Some(other) = grid.get(&(gx + dx, gy + dy)) {
                            for &j in other {
                                if j <= i || self.excluded.contains(&(i, j)) {
                                    continue;
                                }
                                let d = V2::new(q[j][0] - q[i][0], q[j][1] - q[i][1]).norm();
                                if d < self.radius[i] + self.radius[j] + margin {
                                    out.push((i, j));
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(yg) = self.ground_y {
            for (i, &r) in self.radius.iter().enumerate() {
                // the polygon reaches further than the circle: use its farthest vertex
                let ext = self.verts[i].iter().fold(r, |a, v| a.max(v.0.norm()));
                if r > 0.0 && q[i][1] - ext - yg < margin {
                    out.push((i, GROUND));
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Gap, unit normal (a → b; ground: +y) and the contact arms of a pair.
    pub fn geometry(&self, a: usize, b: usize, q: &[[f64; 3]]) -> (f64, V2) {
        if b == GROUND {
            let yg = self.ground_y.unwrap();
            (q[a][1] - self.radius[a] - yg, V2::new(0.0, -1.0))
        } else {
            let d = V2::new(q[b][0] - q[a][0], q[b][1] - q[a][1]);
            let dn = d.norm().max(1e-12);
            (dn - self.radius[a] - self.radius[b], d * (1.0 / dn))
        }
    }

    /// Ground contact points of a cell: (gap, world arm R(θ) r, stiffness) per vertex.
    pub fn ground_points(&self, a: usize, q: &[[f64; 3]]) -> Vec<(f64, V2, f64)> {
        let yg = self.ground_y.unwrap();
        let (c, s) = (q[a][2].cos(), q[a][2].sin());
        self.verts[a].iter().map(|&(r, k)| {
            let w = r.rot_cs(c, s);
            (q[a][1] + w.y - yg, w, k)
        }).collect()
    }

    /// Contact (penalty) energy of a pair at q.
    pub fn pair_energy(&self, a: usize, b: usize, q: &[[f64; 3]]) -> f64 {
        if b == GROUND {
            self.ground_points(a, q).iter().map(|&(g, _, k)| if g < 0.0 { 0.5 * k * g * g } else { 0.0 }).sum()
        } else {
            self.normal(a, b, self.geometry(a, b, q).0).1
        }
    }

    /// Lagged normal forces of a pair at q (per vertex for the ground, one for a pair).
    pub fn lagged(&self, a: usize, b: usize, q: &[[f64; 3]]) -> Vec<f64> {
        if b == GROUND {
            self.ground_points(a, q).iter().map(|&(g, _, k)| (-k * g).max(0.0)).collect()
        } else {
            vec![self.normal(a, b, self.geometry(a, b, q).0).0]
        }
    }

    /// Penalty stiffness of a pair, scaled with the smaller cell (contact area grows
    /// with size; a sliver must not be stiffer than its own facets).
    pub fn k_pair(&self, a: usize, b: usize) -> f64 {
        let r = if b == GROUND { self.radius[a] } else { self.radius[a].min(self.radius[b]) };
        self.kc * (2.0 * r / self.d_ref).min(1.0)
    }

    /// Normal force magnitude (compression ≥ 0) and energy at a gap.
    pub fn normal_k(&self, k: f64, g: f64) -> (f64, f64) {
        if g < 0.0 {
            (-k * g, 0.5 * k * g * g)
        } else {
            (0.0, 0.0)
        }
    }

    pub fn normal(&self, a: usize, b: usize, g: f64) -> (f64, f64) {
        self.normal_k(self.k_pair(a, b), g)
    }

    /// Relative tangential slip rate at the contact point of a pair (b relative to a),
    /// along t = perp(n), from body velocities.
    pub fn slip_rate(&self, a: usize, b: usize, n: V2, v: &[[f64; 3]]) -> f64 {
        let t = n.perp();
        if b == GROUND {
            // contact point of a at its bottom: x_a + r_a n (n points into the ground)
            let ra = self.radius[a];
            -(t.dot(V2::new(v[a][0], v[a][1])) + ra * v[a][2] * t.dot(n.perp()))
        } else {
            let (ra, rb) = (self.radius[a], self.radius[b]);
            let va = V2::new(v[a][0], v[a][1]) + n.perp() * (ra * v[a][2]);
            let vb = V2::new(v[b][0], v[b][1]) - n.perp() * (rb * v[b][2]);
            t.dot(vb - va)
        }
    }

    /// Jacobian of the slip (b relative to a, along t) w.r.t. (x, y, θ) of a and b.
    pub fn slip_jac(&self, a: usize, b: usize, n: V2) -> ([f64; 3], [f64; 3]) {
        let t = n.perp();
        if b == GROUND {
            let ra = self.radius[a];
            ([-t.x, -t.y, -ra], [0.0; 3])
        } else {
            let (ra, rb) = (self.radius[a], self.radius[b]);
            ([-t.x, -t.y, -ra], [t.x, t.y, -rb])
        }
    }
}

/// A contact pair frozen at the start of an implicit step: lagged normal force for
/// friction, the tangent and the start configuration of the slip.
#[derive(Clone, Debug)]
pub struct CItem {
    pub a: usize,
    pub b: usize,
    pub n0: V2,
    /// lagged normal force(s): one for a pair, one per vertex for the ground
    pub lam: Vec<f64>,
    pub q0a: [f64; 3],
    pub q0b: [f64; 3],
}

/// Contact energy, generalised gradient on (a, b) and the 6×6 Hessian (a, b order)
/// of one pair for the implicit potential, with friction regularised over ε = ev·h.
/// Returns also the friction work of the increment (for D_fric).
pub fn implicit_pair(cm: &ContactModel, it: &CItem, q: &[[f64; 3]], h: f64) -> (f64, [f64; 6], [[f64; 6]; 6], f64) {
    let (a, b) = (it.a, it.b);
    let mut g6 = [0.0; 6];
    let mut h6 = [[0.0; 6]; 6];
    let mut e = 0.0;
    let mut wf = 0.0;
    let eps = (cm.ev * h).max(1e-12);
    if b == GROUND {
        // polygon vertices against y = y₀: normal penalty and lagged-normal friction per vertex
        let (c0, s0) = (it.q0a[2].cos(), it.q0a[2].sin());
        for (vi, (gap, w, k)) in cm.ground_points(a, q).into_iter().enumerate() {
            if gap < 0.0 {
                // ∂gap/∂(x, y, θ) = (0, 1, w.x); ∂²gap/∂θ² = −w.y
                let jg = [0.0, 1.0, w.x];
                e += 0.5 * k * gap * gap;
                for r in 0..3 {
                    g6[r] += k * gap * jg[r];
                    for cc in 0..3 {
                        h6[r][cc] += k * jg[r] * jg[cc];
                    }
                }
                h6[2][2] += k * gap * (-w.y);
            }
            let lam = it.lam.get(vi).copied().unwrap_or(0.0);
            if lam > 0.0 && cm.mu > 0.0 {
                // horizontal slip of the vertex since the step start
                let r = cm.verts[a][vi].0;
                let w0 = r.rot_cs(c0, s0);
                let du = (q[a][0] - it.q0a[0]) + (w.x - w0.x);
                let js = [1.0, 0.0, -w.y];
                let fmax = cm.mu * lam;
                let (pot, f, kk) = if du.abs() < eps { (0.5 * fmax * du * du / eps, fmax * du / eps, fmax / eps) } else { (fmax * (du.abs() - 0.5 * eps), fmax * du.signum(), 0.0) };
                e += pot;
                wf += f * du;
                for r2 in 0..3 {
                    g6[r2] += f * js[r2];
                    for cc in 0..3 {
                        h6[r2][cc] += kk * js[r2] * js[cc];
                    }
                }
                h6[2][2] += f * (-w.x);
            }
        }
        return (e, g6, h6, wf);
    }
    let (gap, n) = cm.geometry(a, b, q);
    let kc = cm.k_pair(a, b);
    if gap < 0.0 {
        let jg = [-n.x, -n.y, 0.0, n.x, n.y, 0.0];
        e += 0.5 * kc * gap * gap;
        for r in 0..6 {
            g6[r] += kc * gap * jg[r];
            for c in 0..6 {
                h6[r][c] += kc * jg[r] * jg[c];
            }
        }
        // curvature of the distance: k g (I − n nᵀ)/d on the translations
        let d = V2::new(q[b][0] - q[a][0], q[b][1] - q[a][1]).norm().max(1e-12);
        let c0 = kc * gap / d;
        let p = [[1.0 - n.x * n.x, -n.x * n.y], [-n.x * n.y, 1.0 - n.y * n.y]];
        for r in 0..2 {
            for c in 0..2 {
                let v = c0 * p[r][c];
                h6[r][c] += v;
                h6[3 + r][3 + c] += v;
                h6[r][3 + c] -= v;
                h6[3 + r][c] -= v;
            }
        }
    }
    let lam = it.lam.first().copied().unwrap_or(0.0);
    if lam > 0.0 && cm.mu > 0.0 {
        let (ja, jb) = cm.slip_jac(a, b, it.n0);
        let js = [ja[0], ja[1], ja[2], jb[0], jb[1], jb[2]];
        let du: f64 = (0..3).map(|k| ja[k] * (q[a][k] - it.q0a[k]) + jb[k] * (q[b][k] - it.q0b[k])).sum();
        let fmax = cm.mu * lam;
        let (pot, f, k) = if du.abs() < eps { (0.5 * fmax * du * du / eps, fmax * du / eps, fmax / eps) } else { (fmax * (du.abs() - 0.5 * eps), fmax * du.signum(), 0.0) };
        e += pot;
        wf += f * du;
        for r in 0..6 {
            g6[r] += f * js[r];
            for c in 0..6 {
                h6[r][c] += k * js[r] * js[c];
            }
        }
    }
    (e, g6, h6, wf)
}
