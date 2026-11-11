//! A separate, same-microstep rigid-island control on S4b (no general contacts).
//! Frozen poses include per-cell angular offsets, not just a common angle. The
//! projection preserves P,L; lost deformation KE is explicitly dissipated.
//! Stored elastic energy and damage history are retained. This is not a finished
//! GPU/contact engine and is intentionally kept out of production World.
use experiments::engine::{
    scenes::*,
    world::{fragments, Open0, Params, World},
};
use geometry::V2;
use model::{LawKind, Model};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Instant;
fn xy(x: [f64; 3]) -> V2 {
    V2::new(x[0], x[1])
}
fn cross(a: V2, b: V2) -> f64 {
    a.x * b.y - a.y * b.x
}
#[derive(Clone)]
struct Rigid {
    cells: Vec<usize>,
    arms: Vec<V2>,
    angles: Vec<f64>,
    x: V2,
    phi: f64,
    v: V2,
    w: f64,
    m: f64,
    i: f64,
}
impl Rigid {
    fn new(m: &Model, cells: Vec<usize>, q: &[[f64; 3]], v: &[[f64; 3]]) -> (Self, f64) {
        let mass = cells.iter().map(|&j| m.bodies[j].mass).sum::<f64>();
        let x = cells
            .iter()
            .fold(V2::ZERO, |a, &j| a + xy(q[j]) * m.bodies[j].mass)
            * (1. / mass);
        let vel = cells
            .iter()
            .fold(V2::ZERO, |a, &j| a + xy(v[j]) * m.bodies[j].mass)
            * (1. / mass);
        let arms: Vec<_> = cells.iter().map(|&j| xy(q[j]) - x).collect();
        let inertia = cells
            .iter()
            .zip(&arms)
            .map(|(&j, r)| m.bodies[j].inertia + m.bodies[j].mass * r.dot(*r))
            .sum::<f64>();
        let l = cells
            .iter()
            .zip(&arms)
            .map(|(&j, r)| {
                m.bodies[j].mass * cross(*r, xy(v[j]) - vel) + m.bodies[j].inertia * v[j][2]
            })
            .sum::<f64>();
        let omega = l / inertia;
        let ke = cells
            .iter()
            .map(|&j| {
                0.5 * m.bodies[j].mass * xy(v[j]).dot(xy(v[j]))
                    + 0.5 * m.bodies[j].inertia * v[j][2] * v[j][2]
            })
            .sum::<f64>();
        let loss = ke - 0.5 * mass * vel.dot(vel) - 0.5 * inertia * omega * omega;
        let angles = cells.iter().map(|&j| q[j][2]).collect();
        (
            Self {
                cells,
                arms,
                angles,
                x,
                phi: 0.,
                v: vel,
                w: omega,
                m: mass,
                i: inertia,
            },
            loss.max(0.),
        )
    }
    fn sync(&self, q: &mut [[f64; 3]], v: &mut [[f64; 3]]) {
        for (k, &j) in self.cells.iter().enumerate() {
            let r = self.arms[k].rot(self.phi);
            let p = self.x + r;
            let vel = self.v + r.perp() * self.w;
            q[j] = [p.x, p.y, self.angles[k] + self.phi];
            v[j] = [vel.x, vel.y, self.w];
        }
    }
    fn kick(&mut self, f: &[[f64; 3]], q: &[[f64; 3]], h: f64) {
        let mut ff = V2::ZERO;
        let mut tau = 0.;
        for &j in &self.cells {
            ff = ff + xy(f[j]);
            tau += cross(xy(q[j]) - self.x, xy(f[j])) + f[j][2];
        }
        self.v = self.v + ff * (h / self.m);
        self.w += h * tau / self.i;
    }
}
fn find(p: &mut [usize], a: usize) -> usize {
    if p[a] != a {
        p[a] = find(p, p[a]);
    }
    p[a]
}
fn groups(m: &Model) -> Vec<Vec<usize>> {
    let nc = m.n_cells();
    let mut p: Vec<_> = (0..nc).collect();
    for f in &m.facets {
        if f.active
            && f.tag == 0
            && f.a < nc
            && f.b < nc
            && f.pts.iter().any(|pt| pt.law.damage(pt.kappa_c) < 0.999999)
        {
            let a = find(&mut p, f.a);
            let b = find(&mut p, f.b);
            p[a.min(b)] = a.max(b);
        }
    }
    let mut out: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for j in 0..nc {
        out.entry(find(&mut p, j)).or_default().push(j);
    }
    out.into_values().collect()
}
fn ke(m: &Model, v: &[[f64; 3]]) -> f64 {
    m.bodies
        .iter()
        .zip(v)
        .map(|(b, v)| 0.5 * b.mass * (v[0] * v[0] + v[1] * v[1]) + 0.5 * b.inertia * v[2] * v[2])
        .sum()
}
fn energy(m: &Model, q: &[[f64; 3]], v: &[[f64; 3]]) -> f64 {
    ke(m, v) + m.elastic_energy(q)
        - m.loads
            .iter()
            .zip(q)
            .map(|(f, q)| f.iter().zip(q).map(|(f, x)| f * x).sum::<f64>())
            .sum::<f64>()
        + m.dissipated
}
fn forces(
    sc: &mut Scene,
    q: &[[f64; 3]],
    v: &[[f64; 3]],
    t: f64,
    edges: &[usize],
    last: &mut [f64],
    motion: &mut [f64],
    monitor: bool,
    ops: &mut f64,
) -> Vec<[f64; 3]> {
    let mut out = sc.m.loads.clone();
    for (i, f) in sc.ext(t) {
        for k in 0..3 {
            out[i][k] += f[k];
        }
    }
    let mut updates = vec![];
    for &fi in edges {
        let f = &sc.m.facets[fi];
        for pt in 0..2 {
            let (j, e) = sc.m.eval_point(q, f, pt);
            for k in 0..3 {
                out[f.a][k] -= j.jna[k] * e.fnorm + j.jta[k] * e.ftan;
                out[f.b][k] -= j.jnb[k] * e.fnorm + j.jtb[k] * e.ftan;
            }
            if e.kappa > f.pts[pt].kappa_c {
                updates.push((fi, pt, e.kappa));
            }
            if monitor && f.tag == 0 {
                let rn = (0..3)
                    .map(|d| j.jna[d] * v[f.a][d] + j.jnb[d] * v[f.b][d])
                    .sum::<f64>();
                let rt = (0..3)
                    .map(|d| j.jta[d] * v[f.a][d] + j.jtb[d] * v[f.b][d])
                    .sum::<f64>();
                if rn.abs().max(rt.abs()) > 0.1 {
                    motion[fi] = t;
                }
            }
        }
    }
    for (fi, pt, k) in updates {
        let p = &mut sc.m.facets[fi].pts[pt];
        let dd = p.law.dissipation(k) - p.law.dissipation(p.kappa_c);
        if dd > 1e-14 {
            last[fi] = t;
        }
        sc.m.dissipated += dd;
        p.kappa_c = k;
    }
    *ops += edges.len() as f64 * if monitor { 288. } else { 240. };
    out
}
fn damp(
    m: &Model,
    groups: &[Vec<usize>],
    owner: &[usize],
    q: &[[f64; 3]],
    v: &mut [[f64; 3]],
    h: f64,
    ops: &mut f64,
) -> f64 {
    let alpha = (-h / 0.01).exp();
    let mut lost = 0.;
    for cells in groups {
        if cells.len() < 2 || cells.iter().any(|&i| owner[i] != usize::MAX) {
            continue;
        }
        let (r, l) = Rigid::new(m, cells.clone(), q, v);
        lost += (1. - alpha * alpha) * l;
        for (k, &i) in r.cells.iter().enumerate() {
            let rv = r.v + r.arms[k].perp() * r.w;
            let old = v[i];
            v[i] = [
                rv.x + alpha * (old[0] - rv.x),
                rv.y + alpha * (old[1] - rv.y),
                r.w + alpha * (old[2] - r.w),
            ];
        }
        *ops += 100. * cells.len() as f64;
    }
    lost
}
fn advance(
    sc: &mut Scene,
    q: &mut [[f64; 3]],
    v: &mut [[f64; 3]],
    rigid: &mut [Option<Rigid>],
    owner: &[usize],
    f: &[[f64; 3]],
    h: f64,
    drift: bool,
) {
    for i in 0..q.len() {
        if owner[i] != usize::MAX {
            continue;
        }
        let b = &sc.m.bodies[i];
        if b.mass == 0. {
            continue;
        }
        for k in 0..3 {
            if !b.fixed[k] {
                v[i][k] += h * f[i][k] / if k == 2 { b.inertia } else { b.mass };
            }
        }
    }
    for r in rigid.iter_mut().flatten() {
        r.kick(f, q, h);
    }
    if drift {
        for i in 0..q.len() {
            if owner[i] != usize::MAX {
                continue;
            }
            for k in 0..3 {
                if !sc.m.bodies[i].fixed[k] {
                    q[i][k] += 2. * h * v[i][k];
                }
            }
        }
        for r in rigid.iter_mut().flatten() {
            r.x = r.x + r.v * (2. * h);
            r.phi += 2. * h * r.w;
        }
    }
    for r in rigid.iter().flatten() {
        r.sync(q, v);
    }
}
fn simulate(mut sc: Scene, h: f64, end: f64, mode: &str) -> Value {
    assert!(
        sc.ground.is_none(),
        "This control is S4b, without rubble contacts"
    );
    let start = Instant::now();
    let nc = sc.m.n_cells();
    let nb = sc.m.bodies.len();
    let nf = sc.m.facets.len();
    let mut q = sc.m.q.clone();
    let mut v = sc.v_init.clone();
    let e0 = energy(&sc.m, &q, &v);
    let mut owner = vec![usize::MAX; nb];
    let mut rigid: Vec<Option<Rigid>> = vec![];
    let mut last = vec![f64::NEG_INFINITY; nf];
    let mut motion = last.clone();
    let damping = mode.starts_with("damped");
    let monitor = damping || mode == "windowed";
    let mut damp_groups = vec![];
    let mut damping_loss = 0.;
    let mut edges: Vec<_> =
        sc.m.facets
            .iter()
            .enumerate()
            .filter(|(_, f)| f.active)
            .map(|(i, _)| i)
            .collect();
    let mut ops = 0.;
    let mut f = forces(
        &mut sc,
        &q,
        &v,
        0.,
        &edges,
        &mut last,
        &mut motion,
        monitor,
        &mut ops,
    );
    let (mut loss, mut extwork, mut freezes, mut wakes) = (0., 0., 0, 0);
    let steps = (end / h).ceil() as usize;
    let check = (0.001 / h).round().max(1.) as usize;
    let (mut sumfine, mut minfine, mut peakfine, mut peak_after_50ms) =
        (0., 1.0f64, 0.0f64, 0.0f64);
    let mut frames = vec![0.; (end * 60.).ceil() as usize + 1];
    let mut timeline = vec![];
    let permissive = mode == "aggressive" || mode == "windowed" || damping;
    let quiet = if permissive { 0.002 } else { 0.01 };
    let speed = if permissive { 0.1 } else { 0.01 };
    for s in 0..steps {
        let t = s as f64 * h;
        let q0 = q.clone();
        let before = ops;
        if damping {
            damping_loss += damp(&sc.m, &damp_groups, &owner, &q, &mut v, h / 2., &mut ops);
        }
        advance(
            &mut sc,
            &mut q,
            &mut v,
            &mut rigid,
            &owner,
            &f,
            h / 2.,
            true,
        );
        f = forces(
            &mut sc,
            &q,
            &v,
            t + h,
            &edges,
            &mut last,
            &mut motion,
            monitor,
            &mut ops,
        );
        advance(
            &mut sc,
            &mut q,
            &mut v,
            &mut rigid,
            &owner,
            &f,
            h / 2.,
            false,
        );
        if damping {
            damping_loss += damp(&sc.m, &damp_groups, &owner, &q, &mut v, h / 2., &mut ops);
        }
        let ex0 = sc.ext(t);
        let ex1 = sc.ext(t + h);
        for (i, a) in ex0 {
            let b = ex1
                .iter()
                .find(|x| x.0 == i)
                .map(|x| x.1)
                .unwrap_or([0.; 3]);
            for k in 0..3 {
                extwork += 0.5 * (a[k] + b[k]) * (q[i][k] - q0[i][k]);
            }
        }
        let fine = (0..nc).filter(|&i| owner[i] == usize::MAX).count();
        let macros = rigid.iter().flatten().count();
        ops += 24. * (fine + macros) as f64 + 12. * (nc - fine) as f64; // lifting is charged too
        if s % check == 0 {
            // A load applied to a constituent must wake BEFORE it changes the
            // macro response. S4b has only gravity on detached islands, hence no
            // impact to fabricate here. A nonuniform new external load wakes it.
            for rid in 0..rigid.len() {
                if let Some(r) = &rigid[rid] {
                    let nonuniform = r
                        .cells
                        .iter()
                        .any(|&i| sc.ext(t + h).iter().any(|(j, _)| *j == i));
                    if nonuniform {
                        for &i in &r.cells {
                            owner[i] = usize::MAX;
                        }
                        rigid[rid] = None;
                        wakes += 1;
                    }
                }
            }
            // Drop only open, force-free broken facets, exactly as the engine.
            for fi in 0..nf {
                let ff = &sc.m.facets[fi];
                if !ff.active
                    || ff.tag != 0
                    || !ff
                        .pts
                        .iter()
                        .all(|p| p.law.kind == LawKind::Damage && p.kappa_c >= p.law.wc)
                {
                    continue;
                }
                let a = sc.m.eval_point(&q, ff, 0).0;
                let b = sc.m.eval_point(&q, ff, 1).0;
                if a.dn >= 0.
                    && b.dn >= 0.
                    && (a.dn.min(b.dn) > 0.5 * ff.len
                        || a.dn.abs().max(a.dt.abs()).max(b.dn.abs()).max(b.dt.abs())
                            > 0.25 * ff.len
                        || (q[ff.a][2] - q[ff.b][2]).abs() > 0.2)
                {
                    sc.m.facets[fi].active = false;
                }
            }
            if damping {
                damp_groups = groups(&sc.m)
                    .into_iter()
                    .filter(|cells| {
                        let mut inside = vec![false; nb];
                        for &i in cells {
                            inside[i] = true;
                        }
                        !sc.m
                            .facets
                            .iter()
                            .filter(|f| f.active && f.tag != 0)
                            .any(|f| inside[f.a] || inside[f.b])
                    })
                    .collect();
            }
            if mode != "fine" && mode != "damped_fine" {
                for cells in groups(&sc.m) {
                    if cells.len() < 2 || cells.iter().any(|&i| owner[i] != usize::MAX) {
                        continue;
                    }
                    ops += 16. * nf as f64 + 80. * cells.len() as f64;
                    let mut inside = vec![false; nb];
                    for &i in &cells {
                        inside[i] = true;
                    }
                    let mut supported = false;
                    let (mut growth, mut u, mut capacity) = (f64::NEG_INFINITY, 0., 0.);
                    for (fi, ff) in sc.m.facets.iter().enumerate().filter(|(_, f)| f.active) {
                        if inside[ff.a] && inside[ff.b] {
                            growth = growth.max(last[fi]);
                            if monitor {
                                growth = growth.max(motion[fi]);
                            }
                            for pt in 0..2 {
                                u += sc.m.eval_point(&q, ff, pt).1.psi;
                                capacity += ff.pts[pt].law.full_dissipation();
                            }
                        } else if (inside[ff.a] || inside[ff.b])
                            && ff.pts.iter().any(|p| p.law.damage(p.kappa_c) < 0.999999)
                        {
                            supported = true;
                        }
                    }
                    if supported || t - growth < quiet {
                        continue;
                    }
                    let (r, l) = Rigid::new(&sc.m, cells, &q, &v);
                    let rms = (2. * l / r.m).sqrt();
                    if rms > speed
                        || ((mode == "conservative" || mode == "damped_freeze")
                            && u > 0.01 * capacity)
                    {
                        continue;
                    }
                    let id = rigid.len();
                    for &i in &r.cells {
                        owner[i] = id;
                    }
                    loss += l;
                    freezes += 1;
                    r.sync(&mut q, &mut v);
                    rigid.push(Some(r));
                }
            }
            edges = (0..nf)
                .filter(|&i| {
                    let e = &sc.m.facets[i];
                    e.active && !(owner[e.a] != usize::MAX && owner[e.a] == owner[e.b])
                })
                .collect();
            // Charge complete topology/eligibility scans. This is an intentionally
            // conservative implementation, not a hidden free oracle.
            ops += 120. * nf as f64 + 80. * nc as f64;
            f = forces(
                &mut sc,
                &q,
                &v,
                t + h,
                &edges,
                &mut last,
                &mut motion,
                monitor,
                &mut ops,
            );
            timeline.push(json!({"t":t+h,"fine_cells":(0..nc).filter(|&i|owner[i]==usize::MAX).count(),"rigid_components":rigid.iter().flatten().count(),"D":sc.m.dissipated}));
        }
        let frac = fine as f64 / nc as f64;
        sumfine += frac;
        minfine = minfine.min(frac);
        peakfine = peakfine.max(frac);
        if t > 0.05 {
            peak_after_50ms = peak_after_50ms.max(frac);
        }
        frames[(t * 60.).floor() as usize] += ops - before;
    }
    let fr = fragments(&sc.m, &q, &v, 0.);
    let mut masses: Vec<_> = fr.iter().map(|f| f.mass).collect();
    masses.sort_by(f64::total_cmp);
    let e1 = energy(&sc.m, &q, &v);
    let p =
        sc.m.bodies
            .iter()
            .zip(&v)
            .fold(V2::ZERO, |p, (b, v)| p + xy(*v) * b.mass);
    json!({"mode":mode,"cells":nc,"h":h,"T":end,"D":sc.m.dissipated,"fragment_masses":masses,"large_fragments":fr.iter().filter(|f|f.mass>0.01*masses.iter().sum::<f64>()).collect::<Vec<_>>(),"mean_fine_fraction":sumfine/steps as f64,"min_fine_fraction":minfine,"peak_fine_fraction":peakfine,"peak_fine_after_50ms":peak_after_50ms,"freezes":freezes,"wakes":wakes,"projection_KE_loss":loss,"internal_damping_loss":damping_loss,"damping_tau":damping.then_some(0.01),"energy_balance":e1-e0+loss+damping_loss-extwork,"work":ops,"peak_frame_work":frames.iter().copied().fold(0.,f64::max),"wall_s":start.elapsed().as_secs_f64(),"momentum":[p.x,p.y],"timeline":timeline,"scope":"S4b, no rubble or new impact contacts. Wake-on-impact is NOT validated by this scene. Internal frozen strain energy and kappa retained; scan/lift cost charged. Damped modes share explicit 10ms mass-projected INTERNAL velocity damping, preserving P,L; this is a named new material approximation, not the original undamped law."})
}
pub fn experiment() -> Value {
    let mut rows = vec![];
    for seed in [1, 2] {
        let sc = span_collapse(seed, 0.08, 12., 0.4, 5);
        let mut p = Params::new(0.001, 1e9);
        p.open0 = Open0::All;
        p.rereduce = false;
        let h = World::new(sc.clone(), p).he;
        for mode in ["fine", "conservative", "aggressive"] {
            let row = simulate(sc.clone(), h, 0.5, mode);
            eprintln!(
                "freeze seed{seed} {mode}: D={} active={} freeze={} work={}",
                row["D"], row["mean_fine_fraction"], row["freezes"], row["work"]
            );
            rows.push(json!({"seed":seed,"run":row}));
        }
        rows.push(json!({"seed":seed,"run":simulate(sc,h/2.,0.5,"fine")}));
    }
    json!({"scope":"Rigid-freeze specialization of detached coherent fragments; all runs use the same separate velocity-Verlet control and geometry. Not yet production World or contact validation.","rows":rows})
}
pub fn refinement() -> Value {
    let mut rows = vec![];
    for seed in [1, 2] {
        let sc = span_collapse(seed, 0.08, 12., 0.4, 5);
        let mut p = Params::new(0.001, 1e9);
        p.open0 = Open0::All;
        p.rereduce = false;
        let h = World::new(sc.clone(), p).he;
        for (mode, scale) in [("fine", 0.25), ("aggressive", 0.5)] {
            let r = simulate(sc.clone(), h * scale, 0.5, mode);
            eprintln!(
                "freeze refinement seed{seed} {mode} h/{}, D={}",
                1. / scale,
                r["D"]
            );
            rows.push(json!({"seed":seed,"run":r}));
        }
    }
    json!({"rows":rows,"scope":"Time refinement of the same S4b rigid-freeze control, no contact validation."})
}
pub fn quiet_window() -> Value {
    let mut rows = vec![];
    for seed in [1, 2] {
        let sc = span_collapse(seed, 0.08, 12., 0.4, 5);
        let mut p = Params::new(0.001, 1e9);
        p.open0 = Open0::All;
        p.rereduce = false;
        let h = World::new(sc.clone(), p).he / 2.;
        for mode in ["windowed", "damped_fine", "damped_freeze"] {
            let r = simulate(sc.clone(), h, 0.5, mode);
            eprintln!(
                "freeze quiet seed{seed} {mode}, D={} active={} freezes={}",
                r["D"], r["mean_fine_fraction"], r["freezes"]
            );
            rows.push(json!({"seed":seed,"run":r}));
        }
    }
    json!({"rows":rows,"scope":"Whole quiet-window strain-rate monitor; separately, matched full/frozen controls with explicitly declared internal damping. No contact validation."})
}
#[test]
fn projection_preserves_momenta_and_angular_offsets() {
    let sc = span_collapse(2, 0.2, 4., 0.4, 1);
    let cells: Vec<_> = (0..sc.m.n_cells()).collect();
    let q = sc.m.q.clone();
    let mut v = sc.v_init.clone();
    for i in 0..cells.len() {
        v[i] = [0.1 * (i as f64).sin(), -1., 0.3];
    }
    let (r, loss) = Rigid::new(&sc.m, cells.clone(), &q, &v);
    let mut q1 = q.clone();
    let mut v1 = v.clone();
    r.sync(&mut q1, &mut v1);
    let moment = |v: &[[f64; 3]]| {
        let p = cells
            .iter()
            .fold(V2::ZERO, |a, &i| a + xy(v[i]) * sc.m.bodies[i].mass);
        let l = cells
            .iter()
            .map(|&i| {
                sc.m.bodies[i].mass * cross(xy(q[i]), xy(v[i])) + sc.m.bodies[i].inertia * v[i][2]
            })
            .sum::<f64>();
        (p, l)
    };
    let (a, l) = moment(&v);
    let (b, m) = moment(&v1);
    assert!((a - b).norm() < 1e-9);
    assert!((l - m).abs() < 1e-9);
    assert!(q
        .iter()
        .zip(&q1)
        .all(|(a, b)| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-14)));
    assert!((ke(&sc.m, &v) - ke(&sc.m, &v1) - loss).abs() < 1e-9);
}
#[test]
fn internal_damping_preserves_both_momenta_and_books_only_deformation_ke() {
    let sc = span_collapse(2, 0.2, 4., 0.4, 1);
    let cells: Vec<_> = (0..sc.m.n_cells()).collect();
    let q = sc.m.q.clone();
    let mut v = sc.v_init.clone();
    for i in 0..cells.len() {
        v[i] = [0.1 * (i as f64).sin(), -1., 0.3];
    }
    let total = |v: &[[f64; 3]]| {
        let p = cells
            .iter()
            .fold(V2::ZERO, |a, &i| a + xy(v[i]) * sc.m.bodies[i].mass);
        let l = cells
            .iter()
            .map(|&i| {
                sc.m.bodies[i].mass * cross(xy(q[i]), xy(v[i])) + sc.m.bodies[i].inertia * v[i][2]
            })
            .sum::<f64>();
        (p, l)
    };
    let (a, l) = total(&v);
    let e0 = ke(&sc.m, &v);
    let lost = damp(
        &sc.m,
        &[cells.clone()],
        &vec![usize::MAX; q.len()],
        &q,
        &mut v,
        0.001,
        &mut 0.,
    );
    let (b, m) = total(&v);
    assert!((a - b).norm() < 1e-9);
    assert!((l - m).abs() < 1e-9);
    assert!((e0 - ke(&sc.m, &v) - lost).abs() < 1e-9);
}
#[test]
fn quiet_window_remembers_fast_motion_at_a_turning_point() {
    let mut sc = span_collapse(2, 0.2, 4., 0.4, 1);
    let q = sc.m.q.clone();
    let mut v = sc.v_init.clone();
    v[0] = [2., 3., 1.];
    let edges: Vec<_> = (0..sc.m.facets.len())
        .filter(|&i| sc.m.facets[i].active)
        .collect();
    let mut last = vec![f64::NEG_INFINITY; sc.m.facets.len()];
    let mut motion = last.clone();
    forces(
        &mut sc,
        &q,
        &v,
        0.001,
        &edges,
        &mut last,
        &mut motion,
        true,
        &mut 0.,
    );
    assert!(motion.iter().any(|&t| t == 0.001));
    let saved = motion.clone();
    v.fill([0.; 3]);
    forces(
        &mut sc,
        &q,
        &v,
        0.002,
        &edges,
        &mut last,
        &mut motion,
        true,
        &mut 0.,
    );
    assert_eq!(saved, motion);
}
