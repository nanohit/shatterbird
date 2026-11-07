//! Scenes: the Phase 0c support-removal cantilever and impact beam (unchanged
//! geometry and loading), plus the Phase 0d scaling family and the collapse.

use super::{release, G};
use geometry::rng::Rng;
use geometry::voronoi2d::{rect, voronoi_convex};
use geometry::V2;
use model::{BoundaryAttach, BodyKind, LawKind, LawOptions, Material, Model};
use solvers::s0::{newton, NewtonOpts};

/// A force pattern held until t0, then released over tr by a cosine ramp.
#[derive(Clone, Debug)]
pub struct Release {
    pub forces: Vec<(usize, [f64; 3])>,
    pub t0: f64,
    pub tr: f64,
}

/// Rigid half-plane y ≥ y0 with frictional circle contact.
#[derive(Clone, Copy, Debug)]
pub struct Ground {
    pub y0: f64,
    pub mu: f64,
}

#[derive(Clone)]
pub struct Scene {
    pub m: Model,
    pub name: String,
    pub dmin: f64,
    pub len: f64,
    pub depth: f64,
    pub releases: Vec<Release>,
    /// facets (by tag) removed at a given time: (tag, t)
    pub removals: Vec<(u32, f64)>,
    pub clamp_cells: Vec<usize>,
    pub tip_cells: Vec<usize>,
    pub impactor: Option<usize>,
    pub v_init: Vec<[f64; 3]>,
    pub static_presolve: bool,
    pub ground: Option<Ground>,
    /// energy scale for relative tolerances [J]
    pub e_scale: f64,
}

impl Scene {
    pub fn ext(&self, t: f64) -> Vec<(usize, [f64; 3])> {
        let mut out: Vec<(usize, [f64; 3])> = Vec::new();
        for r in &self.releases {
            let lf = release(t - r.t0, r.tr);
            if lf == 0.0 {
                continue;
            }
            for &(i, f) in &r.forces {
                out.push((i, [lf * f[0], lf * f[1], lf * f[2]]));
            }
        }
        out
    }
    pub fn loaded(&self) -> Vec<bool> {
        let mut l = vec![false; self.m.bodies.len()];
        for r in &self.releases {
            for &(i, _) in &r.forces {
                l[i] = true;
            }
        }
        l
    }
    pub fn delta0_ref(&self) -> f64 {
        self.m.mat.sigma_t * self.dmin / self.m.mat.e
    }
}

/// Propped cantilever 4 × 0.4 m (b = 0.3 m) of Phase 0c Exp 2: clamped on the left
/// edge, propped on the last 0.2 m of the bottom edge (both elastic); the propped
/// static state with the prop facets replaced by their forces, released over t_r.
pub fn cantilever(seed: u64, dmin: f64, damage: bool, t_r: f64, gscale: f64) -> Scene {
    cantilever_len(seed, dmin, damage, t_r, gscale, 4.0)
}

/// The cantilever of `cantilever` with another length (same depth, prop and clamp).
pub fn cantilever_len(seed: u64, dmin: f64, damage: bool, t_r: f64, gscale: f64, len: f64) -> Scene {
    let depth = 0.4;
    let mut rng = Rng::new(seed);
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
    let nb = m.bodies.len();
    let mut pat = vec![[0.0f64; 3]; nb];
    for fi in 0..m.facets.len() {
        if m.facets[fi].tag != 2 {
            continue;
        }
        for pt in 0..2 {
            let (j, e) = m.eval_point(&q, &m.facets[fi], pt);
            let a = m.facets[fi].a;
            for r in 0..3 {
                pat[a][r] -= j.jna[r] * e.fnorm + j.jta[r] * e.ftan;
            }
        }
        m.facets[fi].active = false;
    }
    let forces: Vec<(usize, [f64; 3])> = (0..nb).filter(|&i| pat[i] != [0.0; 3]).map(|i| (i, pat[i])).collect();
    let clamp_cells: Vec<usize> = m.facets.iter().filter(|f| f.tag == 1).map(|f| f.a).collect();
    let tip_cells: Vec<usize> = (0..m.n_cells()).filter(|&i| m.bodies[i].x0.x > len - 0.2).collect();
    let mass: f64 = m.bodies.iter().map(|b| b.mass).sum();
    Scene {
        m,
        name: if len == 4.0 { format!("cantilever_{}_s{seed}_g{gscale:.4}", if damage { "SBIL1" } else { "elastic" }) } else { format!("cantilever{len:.0}_{}_s{seed}_g{gscale:.4}", if damage { "SBIL1" } else { "elastic" }) },
        dmin,
        len,
        depth,
        releases: vec![Release { forces, t0: 0.0, tr: t_r }],
        removals: Vec::new(),
        clamp_cells,
        tip_cells,
        impactor: None,
        v_init: vec![[0.0; 3]; nb],
        static_presolve: true,
        ground: None,
        e_scale: mass * G * gscale * 0.01,
    }
}

/// Concrete beam of `nspan` spans of 1.8 m (depth 0.2 m, b = 0.2 m, 0.1 m overhang
/// at each end) on unilateral support pads 0.1 m wide at every support, hit at the
/// middle of the middle span by a rigid flat impactor (pad 0.1 m, mass mp, y only)
/// at speed v0. nspan = 1 is the Phase 0c Exp 3 scene.
pub fn beam_impact(seed: u64, dmin: f64, v0: f64, mp: f64, nspan: usize) -> Scene {
    let span = 1.8;
    let depth = 0.2;
    let len = span * nspan as f64 + 0.2;
    let sup: Vec<f64> = (0..=nspan).map(|k| 0.1 + span * k as f64).collect();
    let xc = 0.1 + span * (nspan / 2) as f64 + 0.5 * span;
    let mut rng = Rng::new(seed);
    let keep = |p: V2| !(p.y < 0.5 * dmin && sup.iter().any(|&xs| (p.x - xs).abs() < 0.05 + dmin)) && !(p.y > depth - 0.5 * dmin && (p.x - xc).abs() < 0.05 + dmin);
    let seeds = geometry::rsa::rsa_rect_with(V2::new(0.0, 0.0), V2::new(len, depth), dmin, &mut rng, 20_000, &keep, Vec::new());
    let t = voronoi_convex(&seeds, &rect(V2::new(0.0, 0.0), V2::new(len, depth)));
    let mut m = Model::from_tessellation(&t, Material::concrete(0.2), LawOptions::damage(), &[]);
    let g = m.ground();
    for (k, &xs) in sup.iter().enumerate() {
        m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((xs - 0.05, xs + 0.05)), body: g, shear: false, tag: 10 + k as u32 });
    }
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
    let gauge = (0..m.n_cells()).min_by(|&a, &b| (m.bodies[a].x0 - V2::new(xc, 0.5 * depth)).norm().partial_cmp(&(m.bodies[b].x0 - V2::new(xc, 0.5 * depth)).norm()).unwrap()).unwrap();
    m.bodies[gauge].fixed[0] = true;
    m.bodies[plate].fixed[1] = true;
    let mut q = m.q.clone();
    let mg: f64 = m.bodies.iter().map(|b| if b.kind == BodyKind::Cell { b.mass } else { 0.0 }).sum::<f64>() * G;
    let mut me = m.clone();
    for f in me.facets.iter_mut().filter(|f| f.tag != 0) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Elastic;
        }
    }
    let rep = newton(&me, &mut q, 0.0, None, NewtonOpts { tol: 1e-8 * mg, max_iter: 200 });
    assert!(rep.converged, "impact static state (bilateral pads): res {:.3e} vs tol {:.3e}", rep.res, 1e-9 * mg);
    let mut qu = q.clone();
    let rep = newton(&m, &mut qu, 0.0, None, NewtonOpts { tol: 1e-8 * mg, max_iter: 100 });
    if rep.converged {
        q = qu;
    } else {
        eprintln!("impact static state: unilateral re-solve did not converge (res {:.3e}); starting from the bilateral state", rep.res);
    }
    m.bodies[gauge].fixed[0] = false;
    m.bodies[plate].fixed[1] = false;
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
    let support_cells: Vec<usize> = m.facets.iter().filter(|f| f.tag >= 10).map(|f| f.a).collect();
    let name = if nspan == 1 { format!("impact_s{seed}_v{v0:.3}") } else { format!("impact{nspan}_s{seed}_v{v0:.3}") };
    Scene {
        m,
        name,
        dmin,
        len,
        depth,
        releases: Vec::new(),
        removals: Vec::new(),
        clamp_cells: support_cells,
        tip_cells: vec![gauge],
        impactor: Some(plate),
        v_init,
        static_presolve: false,
        ground: None,
        e_scale: 0.5 * mp * v0 * v0,
    }
}

/// Collapse (S4): a concrete beam len × depth (b = depth) on two elastic support pads
/// 0.2 m wide at its ends, `height` above a rigid frictional ground. The left pad is
/// released at t = 0: the beam swings about the right pad and its left end strikes the
/// ground; the right pad is released at `t2`, and the right end drops (2 ms ramps).
pub fn collapse_beam(seed: u64, dmin: f64, len: f64, depth: f64, height: f64, mu: f64, t2: f64) -> Scene {
    let mut rng = Rng::new(seed);
    let keep = |p: V2| !(p.y < 0.5 * dmin && (p.x < 0.2 + dmin || p.x > len - 0.2 - dmin));
    let seeds = geometry::rsa::rsa_rect_with(V2::new(0.0, 0.0), V2::new(len, depth), dmin, &mut rng, 20_000, &keep, Vec::new());
    let t = voronoi_convex(&seeds, &rect(V2::new(0.0, 0.0), V2::new(len, depth)));
    let mut m = Model::from_tessellation(&t, Material::concrete(depth), LawOptions::damage(), &[]);
    let g = m.ground();
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((0.0, 0.2)), body: g, shear: true, tag: 1 });
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((len - 0.2, len)), body: g, shear: true, tag: 2 });
    for f in m.facets.iter_mut().filter(|f| f.tag != 0) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Elastic;
        }
    }
    m.add_gravity(G);
    let mut q = m.q.clone();
    let mg: f64 = m.bodies.iter().map(|b| b.mass).sum::<f64>() * G;
    let rep = newton(&m, &mut q, 0.0, None, NewtonOpts { tol: 1e-8 * mg, max_iter: 60 });
    assert!(rep.converged, "collapse static state: res {:.3e}", rep.res);
    m.commit(&q);
    assert!(m.max_damage() == 0.0, "collapse static state damaged");
    m.q = q.clone();
    let nb = m.bodies.len();
    let mut releases = Vec::new();
    // the left pad is replaced by its force and released; the right pad stays a real
    // support (facets) until it is removed at t2
    for (tag, t0) in [(1u32, 0.0)] {
        let mut pat = vec![[0.0f64; 3]; nb];
        for fi in 0..m.facets.len() {
            if m.facets[fi].tag != tag {
                continue;
            }
            for pt in 0..2 {
                let (j, e) = m.eval_point(&q, &m.facets[fi], pt);
                let a = m.facets[fi].a;
                for r in 0..3 {
                    pat[a][r] -= j.jna[r] * e.fnorm + j.jta[r] * e.ftan;
                }
            }
            m.facets[fi].active = false;
        }
        let forces: Vec<(usize, [f64; 3])> = (0..nb).filter(|&i| pat[i] != [0.0; 3]).map(|i| (i, pat[i])).collect();
        releases.push(Release { forces, t0, tr: 0.002 });
    }
    let mass: f64 = m.bodies.iter().map(|b| b.mass).sum();
    Scene {
        m,
        name: format!("collapse_s{seed}_L{len:.1}"),
        dmin,
        len,
        depth,
        releases,
        removals: vec![(2, t2)],
        clamp_cells: Vec::new(),
        tip_cells: Vec::new(),
        impactor: None,
        v_init: vec![[0.0; 3]; nb],
        static_presolve: false,
        ground: Some(Ground { y0: -height, mu }),
        e_scale: mass * G * height,
    }
}

/// Local-event scaling (S3b, post-hoc): a concrete wall `width` × `height` (b = 0.2 m)
/// standing on a foundation (elastic bilateral facets along the bottom edge), hit at the
/// top centre by the rigid flat impactor (pad 0.1 m, mass mp, y only) at speed v0.
pub fn wall_impact(seed: u64, dmin: f64, width: f64, height: f64, v0: f64, mp: f64) -> Scene {
    let xc = 0.5 * width;
    let mut rng = Rng::new(seed);
    let keep = |p: V2| !(p.y > height - 0.5 * dmin && (p.x - xc).abs() < 0.05 + dmin) && !(p.y < 0.5 * dmin);
    let seeds = geometry::rsa::rsa_rect_with(V2::new(0.0, 0.0), V2::new(width, height), dmin, &mut rng, 20_000, &keep, Vec::new());
    let t = voronoi_convex(&seeds, &rect(V2::new(0.0, 0.0), V2::new(width, height)));
    let mut m = Model::from_tessellation(&t, Material::concrete(0.2), LawOptions::damage(), &[]);
    let g = m.ground();
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: None, body: g, shear: true, tag: 10 });
    let plate = m.add_plate(V2::new(xc, height), mp, 1.0, [true, false, true]);
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 2, range: Some((xc - 0.05, xc + 0.05)), body: plate, shear: false, tag: 3 });
    for f in m.facets.iter_mut().filter(|f| f.tag != 0) {
        for p in f.pts.iter_mut() {
            if f.tag == 3 {
                p.law.kind = LawKind::Damage;
                p.law.delta0 *= 1e-4;
                p.law.wc *= 1e-4;
            } else {
                p.law.kind = LawKind::Elastic;
            }
        }
    }
    m.add_gravity(G);
    m.bodies[plate].fixed[1] = true;
    let mut q = m.q.clone();
    let mg: f64 = m.bodies.iter().map(|b| if b.kind == BodyKind::Cell { b.mass } else { 0.0 }).sum::<f64>() * G;
    let rep = newton(&m, &mut q, 0.0, None, NewtonOpts { tol: 1e-8 * mg, max_iter: 100 });
    assert!(rep.converged, "wall static state: res {:.3e}", rep.res);
    m.bodies[plate].fixed[1] = false;
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
    Scene {
        m,
        name: format!("wall_W{width:.0}_s{seed}_v{v0:.2}"),
        dmin,
        len: width,
        depth: height,
        releases: Vec::new(),
        removals: Vec::new(),
        clamp_cells: Vec::new(),
        tip_cells: Vec::new(),
        impactor: Some(plate),
        v_init,
        static_presolve: false,
        ground: None,
        e_scale: 0.5 * mp * v0 * v0,
    }
}

/// Mass failure without rubble (S4b, post-hoc): a continuous concrete beam len × depth
/// (b = 0.3 m) on elastic pads at both ends and at `nsup` evenly spaced intermediate
/// supports; at t = 0 every intermediate support is released (2 ms ramp) and the beam
/// fails at several sections at once. No ground: the pieces fall freely.
pub fn span_collapse(seed: u64, dmin: f64, len: f64, depth: f64, nsup: usize) -> Scene {
    let mut rng = Rng::new(seed);
    let xs: Vec<f64> = (1..=nsup).map(|k| len * k as f64 / (nsup + 1) as f64).collect();
    let keep = |p: V2| !(p.y < 0.5 * dmin && (p.x < 0.2 + dmin || p.x > len - 0.2 - dmin || xs.iter().any(|&x| (p.x - x).abs() < 0.1 + dmin)));
    let seeds = geometry::rsa::rsa_rect_with(V2::new(0.0, 0.0), V2::new(len, depth), dmin, &mut rng, 20_000, &keep, Vec::new());
    let t = voronoi_convex(&seeds, &rect(V2::new(0.0, 0.0), V2::new(len, depth)));
    let mut m = Model::from_tessellation(&t, Material::concrete(0.3), LawOptions::damage(), &[]);
    let g = m.ground();
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((0.0, 0.2)), body: g, shear: true, tag: 1 });
    m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((len - 0.2, len)), body: g, shear: true, tag: 2 });
    for (k, &x) in xs.iter().enumerate() {
        m.attach_boundary_exact(&t, &BoundaryAttach { edge: 0, range: Some((x - 0.1, x + 0.1)), body: g, shear: true, tag: 20 + k as u32 });
    }
    for f in m.facets.iter_mut().filter(|f| f.tag != 0) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Elastic;
        }
    }
    m.add_gravity(G);
    let mut q = m.q.clone();
    let mg: f64 = m.bodies.iter().map(|b| b.mass).sum::<f64>() * G;
    let rep = newton(&m, &mut q, 0.0, None, NewtonOpts { tol: 1e-8 * mg, max_iter: 60 });
    assert!(rep.converged, "span static state: res {:.3e}", rep.res);
    m.commit(&q);
    assert!(m.max_damage() == 0.0, "span static state damaged");
    m.q = q.clone();
    let nb = m.bodies.len();
    let mut pat = vec![[0.0f64; 3]; nb];
    for fi in 0..m.facets.len() {
        if m.facets[fi].tag < 20 {
            continue;
        }
        for pt in 0..2 {
            let (j, e) = m.eval_point(&q, &m.facets[fi], pt);
            let a = m.facets[fi].a;
            for r in 0..3 {
                pat[a][r] -= j.jna[r] * e.fnorm + j.jta[r] * e.ftan;
            }
        }
        m.facets[fi].active = false;
    }
    let forces: Vec<(usize, [f64; 3])> = (0..nb).filter(|&i| pat[i] != [0.0; 3]).map(|i| (i, pat[i])).collect();
    let mass: f64 = m.bodies.iter().map(|b| b.mass).sum();
    Scene {
        m,
        name: format!("span_s{seed}_L{len:.0}_n{nsup}"),
        dmin,
        len,
        depth,
        releases: vec![Release { forces, t0: 0.0, tr: 0.002 }],
        removals: Vec::new(),
        clamp_cells: Vec::new(),
        tip_cells: Vec::new(),
        impactor: None,
        v_init: vec![[0.0; 3]; nb],
        static_presolve: true,
        ground: None,
        e_scale: mass * G * 0.01,
    }
}
