//! S3b spatial-reduction control: actual nonlinear RBSN in a central strip,
//! linear prestressed far field, full interface traces plus fixed-interface CMS.
//! One shared fine time grid removes multirate coupling as a confounder.
//! Far-field modes replace its full dynamics (never a parallel energy reservoir).
use experiments::engine::{
    scenes::{wall_impact, Scene},
    world::{Open0, Params, World},
};
use model::Model;
use serde_json::{json, Value};
use std::time::Instant;
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
const FULL_LINEAR: usize = usize::MAX - 1;
struct Port {
    ids: Vec<(usize, usize)>,
    lookup: Vec<[usize; 3]>,
    cols: Vec<Vec<f64>>,
    gamma: Vec<usize>,
    lambda: Vec<f64>,
    g: Vec<f64>,
    im: Vec<f64>,
    sparse: Option<Vec<Vec<(usize, f64)>>>,
    frequencies: Vec<f64>,
    build_s: f64,
    build_work: f64,
}
impl Port {
    fn ndof(&self) -> usize {
        if self.sparse.is_some() {
            self.ids.len()
        } else {
            self.cols.len()
        }
    }
    fn displacement(&self, j: usize, z: &[f64]) -> f64 {
        if self.sparse.is_some() {
            self.im[j] * z[j]
        } else {
            self.cols.iter().zip(z).map(|(c, z)| c[j] * z).sum()
        }
    }
    fn elastic_energy(&self, z: &[f64]) -> f64 {
        0.5 * if let Some(rows) = &self.sparse {
            rows.iter()
                .zip(z)
                .map(|(row, x)| x * row.iter().map(|&(j, k)| k * z[j]).sum::<f64>())
                .sum::<f64>()
        } else {
            z.iter()
                .zip(&self.lambda)
                .map(|(x, l)| x * x * l)
                .sum::<f64>()
        }
    }
}
fn make_port(m: &Model, fine: &[bool], modes: usize) -> Port {
    let clock = Instant::now();
    let nb = m.bodies.len();
    let mut ids = vec![];
    let mut map = vec![[usize::MAX; 3]; nb];
    let mut im = vec![];
    for i in 0..nb {
        if fine[i] || m.bodies[i].mass == 0. {
            continue;
        }
        for d in 0..3 {
            if !m.bodies[i].fixed[d] {
                map[i][d] = ids.len();
                ids.push((i, d));
                im.push(
                    1. / if d == 2 {
                        m.bodies[i].inertia
                    } else {
                        m.bodies[i].mass
                    }
                    .sqrt(),
                );
            }
        }
    }
    let n = ids.len();
    let sparse_mode = modes == FULL_LINEAR;
    let mut sparse_rows: Vec<std::collections::BTreeMap<usize, f64>> = if sparse_mode {
        vec![Default::default(); n]
    } else {
        vec![]
    };
    let mut k = if sparse_mode {
        vec![]
    } else {
        vec![vec![0.; n]; n]
    };
    let mut g = vec![0.; n];
    let mut boundary = vec![false; n];
    for (j, &(i, d)) in ids.iter().enumerate() {
        g[j] = m.loads[i][d] * im[j];
    }
    for f in m.facets.iter().filter(|f| f.active) {
        if fine[f.a] || fine[f.b] {
            for i in [f.a, f.b] {
                if !fine[i] {
                    for d in 0..3 {
                        let j = map[i][d];
                        if j != usize::MAX {
                            boundary[j] = true;
                        }
                    }
                }
            }
            continue;
        }
        for pt in 0..2 {
            let (j, e) = m.eval_point(&m.q, f, pt);
            let mut ent = vec![];
            for (body, jn, jt) in [(f.a, j.jna, j.jta), (f.b, j.jnb, j.jtb)] {
                for d in 0..3 {
                    let a = map[body][d];
                    if a != usize::MAX {
                        ent.push((a, jn[d] * im[a], jt[d] * im[a]));
                        g[a] -= (jn[d] * e.fnorm + jt[d] * e.ftan) * im[a];
                    }
                }
            }
            let (kn, kt) = f.pts[pt].law.secant(j.dn, e.d);
            for &(a, an, at) in &ent {
                for &(b, bn, bt) in &ent {
                    let v = kn * an * bn + kt * at * bt;
                    if sparse_mode {
                        *sparse_rows[a].entry(b).or_default() += v;
                    } else {
                        k[a][b] += v;
                    }
                }
            }
        }
    }
    let gamma: Vec<_> = (0..n).filter(|&j| boundary[j]).collect();
    let interior: Vec<_> = (0..n).filter(|&j| !boundary[j]).collect();
    let ni = interior.len();
    if sparse_mode {
        let sparse: Vec<Vec<(usize, f64)>> = sparse_rows
            .into_iter()
            .map(|row| row.into_iter().filter(|(_, x)| *x != 0.).collect())
            .collect();
        let nnz = sparse.iter().map(|r| r.len()).sum::<usize>();
        return Port {
            ids,
            lookup: map,
            cols: vec![],
            gamma,
            lambda: vec![],
            g,
            im,
            sparse: Some(sparse),
            frequencies: vec![],
            build_s: clock.elapsed().as_secs_f64(),
            build_work: nnz as f64 + 240. * m.facets.len() as f64,
        };
    }
    let a = faer::Mat::<f64>::from_fn(ni, ni, |i, j| k[interior[i]][interior[j]]);
    let eig = a.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let u = eig.U();
    let l = eig.S();
    assert!(
        ni == 0 || l[0] > 0.,
        "fixed-interface far field must be positive definite: {}",
        l[0]
    );
    let mut cols = vec![];
    for &b in &gamma {
        let mut c = vec![0.; n];
        c[b] = 1.;
        let modal: Vec<_> = (0..ni)
            .map(|p| -(0..ni).map(|i| u[(i, p)] * k[interior[i]][b]).sum::<f64>() / l[p])
            .collect();
        for (i, &a) in interior.iter().enumerate() {
            c[a] = (0..ni).map(|p| u[(i, p)] * modal[p]).sum();
        }
        cols.push(c);
    }
    let count = modes.min(ni);
    let frequencies = (0..count)
        .map(|j| l[j].sqrt() / (2. * std::f64::consts::PI))
        .collect();
    for p in 0..count {
        let mut c = vec![0.; n];
        for (i, &a) in interior.iter().enumerate() {
            c[a] = u[(i, p)];
        }
        cols.push(c);
    }
    // Whitened coordinates already absorb physical M. Double orthogonalisation
    // retains the complete trace span while making the microstep mass identity.
    for j in 0..cols.len() {
        for _ in 0..2 {
            for i in 0..j {
                let d = dot(&cols[j], &cols[i]);
                for p in 0..n {
                    cols[j][p] -= d * cols[i][p];
                }
            }
        }
        let norm = dot(&cols[j], &cols[j]).sqrt();
        assert!(norm > 1e-10);
        for x in &mut cols[j] {
            *x /= norm;
        }
    }
    let bc: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| k.iter().map(|r| dot(r, c)).collect())
        .collect();
    let kr: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| bc.iter().map(|b| dot(c, b)).collect())
        .collect();
    let gr: Vec<f64> = cols.iter().map(|c| dot(c, &g)).collect();
    // Exact change of reduced coordinates, NOT another reduction: simultaneously
    // diagonalise M_r=I and K_r once. No dense K_r multiply in the microstep.
    let d = cols.len();
    let a = faer::Mat::<f64>::from_fn(d, d, |i, j| 0.5 * (kr[i][j] + kr[j][i]));
    let eig = a.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let u = eig.U();
    let lambda = (0..d).map(|i| eig.S()[i]).collect();
    let gr = (0..d)
        .map(|j| (0..d).map(|i| u[(i, j)] * gr[i]).sum())
        .collect();
    let old = cols;
    let mut cols: Vec<Vec<f64>> = (0..d)
        .map(|j| {
            (0..n)
                .map(|p| (0..d).map(|i| old[i][p] * u[(i, j)]).sum())
                .collect()
        })
        .collect();
    // Physical displacement lift. Force projection is its transpose.
    for c in &mut cols {
        for i in 0..n {
            c[i] *= im[i];
        }
    }
    Port {
        ids,
        lookup: map,
        cols,
        gamma,
        lambda,
        g: gr,
        im,
        sparse: None,
        frequencies,
        build_s: clock.elapsed().as_secs_f64(),
        build_work: (4. * (ni as f64).powi(3)
            + 4. * (n * n * d) as f64
            + 4. * (d as f64).powi(3)
            + 2. * (n * d * d) as f64),
    }
}
struct ResultRow {
    value: Value,
    local_force: Vec<Vec<f64>>,
    ds: Vec<f64>,
}
struct WaveBranch {
    fi: usize,
    pt: usize,
    tangent: bool,
    z: f64,
    ring: Vec<f64>,
    head: usize,
    reflection: f64,
    stored: f64,
}
fn bridge_kick(
    branches: &mut [WaveBranch],
    sc: &Scene,
    q: &[[f64; 3]],
    v: &mut [[f64; 3]],
    vz: &mut [f64],
    fine: &[bool],
    p: &Port,
    h: f64,
    ops: &mut f64,
) -> f64 {
    let mut absorbed = 0.;
    for b in branches {
        let f = &sc.m.facets[b.fi];
        let j = sc.m.eval_point(q, f, b.pt).0;
        let (a, c) = if b.tangent {
            (j.jta, j.jtb)
        } else {
            (j.jna, j.jnb)
        };
        let mut gr = vec![0.; vz.len()];
        let mut vr = 0.;
        let mut inv = 0.;
        for (body, j) in [(f.a, a), (f.b, c)] {
            if fine[body] {
                for d in 0..3 {
                    if !sc.m.bodies[body].fixed[d] {
                        let mass = if d == 2 {
                            sc.m.bodies[body].inertia
                        } else {
                            sc.m.bodies[body].mass
                        };
                        vr += j[d] * v[body][d];
                        inv += j[d] * j[d] / mass;
                    }
                }
            } else {
                for d in 0..3 {
                    let k = p.lookup[body][d];
                    if k != usize::MAX {
                        if p.sparse.is_some() {
                            gr[k] += j[d] * p.im[k];
                        } else {
                            for r in 0..vz.len() {
                                gr[r] += j[d] * p.cols[r][k];
                            }
                        }
                    }
                }
            }
        }
        vr += dot(&gr, vz);
        inv += dot(&gr, &gr);
        let incoming = b.ring[b.head];
        // Cayley/midpoint scattering update. The exact mechanical KE increment
        // equals h*Z*(incoming^2-outgoing^2), including the reduced-side reaction.
        let impulse = h * b.z * (2. * incoming - vr) / (1. + 0.5 * h * b.z * inv);
        let vbar = vr + 0.5 * inv * impulse;
        let outgoing = vbar - incoming;
        for (body, j) in [(f.a, a), (f.b, c)] {
            if fine[body] {
                for d in 0..3 {
                    if !sc.m.bodies[body].fixed[d] {
                        v[body][d] += impulse * j[d]
                            / if d == 2 {
                                sc.m.bodies[body].inertia
                            } else {
                                sc.m.bodies[body].mass
                            };
                    }
                }
            }
        }
        for r in 0..vz.len() {
            vz[r] += impulse * gr[r];
        }
        let flow = h * b.z * (outgoing * outgoing - incoming * incoming);
        if b.reflection == 0. {
            absorbed += flow;
        } else {
            b.stored += flow;
        }
        b.ring[b.head] = b.reflection * outgoing;
        b.head = (b.head + 1) % b.ring.len();
        *ops += 120. + 20. * vz.len() as f64;
    }
    absorbed
}
fn run(
    sc: Scene,
    modes: Option<usize>,
    h: f64,
    end: f64,
    half_width: f64,
    bridge: &str,
) -> ResultRow {
    run_with_extra(sc, modes, h, end, half_width, bridge, &[])
}
fn run_with_extra(
    mut sc: Scene,
    modes: Option<usize>,
    h: f64,
    end: f64,
    half_width: f64,
    bridge: &str,
    extra: &[usize],
) -> ResultRow {
    let clock = Instant::now();
    let nb = sc.m.bodies.len();
    let nc = sc.m.n_cells();
    let mut fine = vec![false; nb];
    for i in 0..nb {
        fine[i] = modes.is_none()
            || sc.m.bodies[i].mass > 0.
                && ((sc.m.bodies[i].x0.x - sc.len / 2.).abs() < half_width || i >= nc);
    }
    for &i in extra {
        fine[i] = true;
    }
    let port = modes.map(|m| make_port(&sc.m, &fine, m));
    let nr = port.as_ref().map(|p| p.ndof()).unwrap_or(0);
    let q0 = sc.m.q.clone();
    let mut q = q0.clone();
    let mut v = sc.v_init.clone();
    let mut z = vec![0.; nr];
    let mut vz = vec![0.; nr];
    let edges: Vec<_> =
        sc.m.facets
            .iter()
            .enumerate()
            .filter(|(_, f)| f.active && (fine[f.a] || fine[f.b]))
            .map(|(j, _)| j)
            .collect();
    let mut waves = vec![];
    if bridge != "none" {
        assert!(port.is_some());
        for &fi in &edges {
            let fa = &sc.m.facets[fi];
            if fine[fa.a] == fine[fa.b] || fa.a >= nc || fa.b >= nc {
                continue;
            }
            let distance = fa.pc.x.min(sc.len - fa.pc.x);
            let delay = (2. * distance / sc.m.mat.wave_speed() / (h / 2.))
                .round()
                .max(1.) as usize;
            for pt in 0..2 {
                for tangent in [false, true] {
                    waves.push(WaveBranch {
                        fi,
                        pt,
                        tangent,
                        z: 0.5
                            * sc.m.mat.thickness
                            * fa.len
                            * sc.m.mat.density
                            * sc.m.mat.wave_speed(),
                        ring: vec![0.; if bridge == "dashpot" { 1 } else { delay }],
                        head: 0,
                        reflection: match bridge {
                            "wave_free" => 1.,
                            "wave_fixed" => -1.,
                            "dashpot" => 0.,
                            _ => panic!("bridge"),
                        },
                        stored: 0.,
                    });
                }
            }
        }
    }
    // Fixed physical probe for ALL zone widths; comparing two differently ordered
    // sets of springs would be an invalid error metric.
    let local: Vec<_> =
        sc.m.facets
            .iter()
            .enumerate()
            .filter(|(_, f)| f.tag == 0 && (f.pc.x - sc.len / 2.).abs() < 0.30)
            .map(|(j, _)| j)
            .collect();
    let mut ops = 0.;
    let mut onset = None;
    let mut remote_chi = 0.0f64;
    let mut local_force = vec![];
    let mut ds = vec![];
    let mut energy = vec![];
    let mut radiated = 0.;
    let lift = |q: &mut Vec<[f64; 3]>, z: &[f64], all: bool| {
        if let Some(p) = &port {
            if all {
                for (j, &(i, d)) in p.ids.iter().enumerate() {
                    q[i][d] = q0[i][d] + p.displacement(j, z);
                }
            } else {
                for &j in &p.gamma {
                    let (i, d) = p.ids[j];
                    q[i][d] = q0[i][d] + p.displacement(j, z);
                }
            }
        }
    };
    let forces = |sc: &mut Scene, q: &[[f64; 3]], z: &[f64], ops: &mut f64| {
        let mut f = sc.m.loads.clone();
        let mut updates = vec![];
        for &fi in &edges {
            let fa = &sc.m.facets[fi];
            for pt in 0..2 {
                let (j, e) = sc.m.eval_point(q, fa, pt);
                for d in 0..3 {
                    f[fa.a][d] -= j.jna[d] * e.fnorm + j.jta[d] * e.ftan;
                    f[fa.b][d] -= j.jnb[d] * e.fnorm + j.jtb[d] * e.ftan;
                }
                if e.kappa > fa.pts[pt].kappa_c {
                    updates.push((fi, pt, e.kappa));
                }
            }
        }
        for (fi, pt, k) in updates {
            let p = &mut sc.m.facets[fi].pts[pt];
            sc.m.dissipated += p.law.dissipation(k) - p.law.dissipation(p.kappa_c);
            p.kappa_c = k;
        }
        let mut ar = vec![0.; nr];
        if let Some(p) = &port {
            let fi: Vec<_> = p
                .gamma
                .iter()
                .map(|&j| {
                    let (i, d) = p.ids[j];
                    f[i][d] - sc.m.loads[i][d]
                })
                .collect();
            if let Some(rows) = &p.sparse {
                for a in 0..nr {
                    ar[a] = p.g[a] - rows[a].iter().map(|&(j, k)| k * z[j]).sum::<f64>();
                }
                for (c, &j) in p.gamma.iter().enumerate() {
                    ar[j] += p.im[j] * fi[c];
                }
                *ops += 2. * rows.iter().map(|r| r.len()).sum::<usize>() as f64
                    + nr as f64
                    + 5. * p.gamma.len() as f64;
            } else {
                for a in 0..nr {
                    ar[a] = p.g[a] - p.lambda[a] * z[a];
                    for (c, &j) in p.gamma.iter().enumerate() {
                        ar[a] += p.cols[a][j] * fi[c];
                    }
                }
                *ops += 2. * nr as f64 + 4. * (nr * p.gamma.len()) as f64 + p.gamma.len() as f64;
            }
        }
        *ops += 240. * edges.len() as f64;
        (f, ar)
    };
    let (mut f, mut ar) = forces(&mut sc, &q, &z, &mut ops);
    let sample = (0.0001 / h).round().max(1.) as usize;
    let steps = (end / h).ceil() as usize;
    for s in 0..=steps {
        if s % sample == 0 || s == steps {
            lift(&mut q, &z, true);
            let mut ff = vec![];
            for &fi in &local {
                for pt in 0..2 {
                    let e = sc.m.eval_point(&q, &sc.m.facets[fi], pt).1;
                    ff.extend([e.fnorm, e.ftan]);
                }
            }
            local_force.push(ff);
            ds.push(sc.m.dissipated);
            let material_d: f64 =
                sc.m.facets
                    .iter()
                    .filter(|f| f.tag == 0)
                    .flat_map(|f| f.pts.iter().map(|p| p.law.dissipation(p.kappa_c)))
                    .sum();
            if onset.is_none() && material_d > 1e-8 {
                onset = Some(s as f64 * h);
            }
            for fa in
                sc.m.facets
                    .iter()
                    .filter(|fa| fa.active && fa.tag == 0 && !fine[fa.a] && !fine[fa.b])
            {
                for pt in 0..2 {
                    let e = sc.m.eval_point(&q, fa, pt).1;
                    remote_chi = remote_chi.max(e.kappa / fa.pts[pt].law.delta0);
                }
            }
            let kfine: f64 = (0..nb)
                .filter(|&i| fine[i])
                .map(|i| {
                    let b = &sc.m.bodies[i];
                    0.5 * b.mass * (v[i][0] * v[i][0] + v[i][1] * v[i][1])
                        + 0.5 * b.inertia * v[i][2] * v[i][2]
                })
                .sum();
            let u: f64 = edges
                .iter()
                .map(|&fi| {
                    (0..2)
                        .map(|pt| sc.m.eval_point(&q, &sc.m.facets[fi], pt).1.psi)
                        .sum::<f64>()
                })
                .sum();
            let vg: f64 = (0..nb)
                .filter(|&i| fine[i])
                .map(|i| -dot(&sc.m.loads[i], &q[i]))
                .sum();
            let er = port
                .as_ref()
                .map(|p| 0.5 * dot(&vz, &vz) + p.elastic_energy(&z) - dot(&p.g, &z))
                .unwrap_or(0.);
            energy.push(
                kfine
                    + u
                    + vg
                    + er
                    + sc.m.dissipated
                    + radiated
                    + waves.iter().map(|b| b.stored).sum::<f64>(),
            );
            ops += 120. * (local.len() + sc.m.facets.len()) as f64
                + port
                    .as_ref()
                    .map(|p| {
                        if p.sparse.is_some() {
                            2. * p.ids.len() as f64
                        } else {
                            2. * (p.ids.len() * nr) as f64
                        }
                    })
                    .unwrap_or(0.);
        }
        if s == steps {
            break;
        }
        if let Some(p) = &port {
            radiated += bridge_kick(
                &mut waves,
                &sc,
                &q,
                &mut v,
                &mut vz,
                &fine,
                p,
                h / 2.,
                &mut ops,
            );
        }
        for i in 0..nb {
            if !fine[i] || sc.m.bodies[i].mass == 0. {
                continue;
            }
            for d in 0..3 {
                if !sc.m.bodies[i].fixed[d] {
                    let mass = if d == 2 {
                        sc.m.bodies[i].inertia
                    } else {
                        sc.m.bodies[i].mass
                    };
                    v[i][d] += 0.5 * h * f[i][d] / mass;
                    q[i][d] += h * v[i][d];
                }
            }
        }
        for j in 0..nr {
            vz[j] += 0.5 * h * ar[j];
            z[j] += h * vz[j];
        }
        lift(&mut q, &z, false);
        (f, ar) = forces(&mut sc, &q, &z, &mut ops);
        for i in 0..nb {
            if !fine[i] || sc.m.bodies[i].mass == 0. {
                continue;
            }
            for d in 0..3 {
                if !sc.m.bodies[i].fixed[d] {
                    v[i][d] += 0.5 * h * f[i][d]
                        / if d == 2 {
                            sc.m.bodies[i].inertia
                        } else {
                            sc.m.bodies[i].mass
                        };
                }
            }
        }
        for j in 0..nr {
            vz[j] += 0.5 * h * ar[j];
        }
        ops += 24. * (0..nc).filter(|&i| fine[i]).count() as f64 + 6. * nr as f64 + 8.;
        if let Some(p) = &port {
            radiated += bridge_kick(
                &mut waves,
                &sc,
                &q,
                &mut v,
                &mut vz,
                &fine,
                p,
                h / 2.,
                &mut ops,
            );
        }
    }
    let plate = sc.impactor.unwrap();
    let damaged: Vec<_> =
        sc.m.facets
            .iter()
            .enumerate()
            .filter(|(_, f)| f.tag == 0 && f.pts.iter().any(|p| p.law.damage(p.kappa_c) > 1e-6))
            .map(|(i, _)| i)
            .collect();
    let outside_d: f64 =
        sc.m.facets
            .iter()
            .filter(|f| f.tag == 0 && (f.pc.x - sc.len / 2.).abs() > half_width)
            .flat_map(|f| f.pts.iter().map(|p| p.law.dissipation(p.kappa_c)))
            .sum();
    let value = json!({"modes":modes,"bridge":bridge,"half_width":half_width,"cells":nc,"fine_cells":(0..nc).filter(|&i|fine[i]).count(),"dofs":3*(0..nc).filter(|&i|fine[i]).count()+1+nr,"port_dofs":nr,"trace_dofs":port.as_ref().map(|p|p.gamma.len()).unwrap_or(0),"h":h,"T":steps as f64*h,"D":sc.m.dissipated,"onset":onset,"impactor_v":v[plate][1],"damaged_facets":damaged,"max_far_trial_chi":remote_chi,"far_linear_contract_valid":remote_chi<=1.,"work":ops,"build_work":port.as_ref().map(|p|p.build_work).unwrap_or(0.),"build_s":port.as_ref().map(|p|p.build_s).unwrap_or(0.),"wall_s":clock.elapsed().as_secs_f64(),"retained_fixed_interface_frequencies":port.as_ref().map(|p|&p.frequencies),"energy_end_minus_initial":energy.last().unwrap()-energy[0],"energy_max_deviation":energy.iter().map(|e|(e-energy[0]).abs()).fold(0.,f64::max),"energy_scale":sc.e_scale,"absorbed_energy":radiated,"wave_stored_energy":waves.iter().map(|b|b.stored).sum::<f64>(),"wave_buffer_samples":waves.iter().map(|b|b.ring.len()).sum::<usize>(),"bridge_note":"Parallel auxiliary branches on a CMS far field: passive by scattering, but NOT an exact impedance correction. This deliberately tests the proposed bridge without assuming that passivity or correct echo time implies accuracy."});
    let mut value = value;
    value["D_outside_central_strip"] = json!(outside_d);
    value["far_representation"] = json!(if modes == Some(FULL_LINEAR) {
        "sparse_full_state"
    } else if modes.is_some() {
        "CMS_normal_coordinates"
    } else {
        "fine_nonlinear"
    });
    if modes == Some(FULL_LINEAR) {
        value["modes"] = Value::Null;
    }
    value["hot_wall_s"] =
        json!(clock.elapsed().as_secs_f64() - port.as_ref().map(|p| p.build_s).unwrap_or(0.));
    value["basis_or_sparse_coefficients_bytes"] = json!(port
        .as_ref()
        .map(|p| if let Some(rows) = &p.sparse {
            16 * rows.iter().map(|r| r.len()).sum::<usize>()
        } else {
            8 * p.cols.iter().map(|c| c.len()).sum::<usize>()
        })
        .unwrap_or(0));
    ResultRow {
        value,
        local_force,
        ds,
    }
}
fn compare(x: &ResultRow, r: &ResultRow) -> Value {
    assert_eq!(x.local_force.len(), r.local_force.len());
    let (mut num, mut den) = (0., 0.);
    for (a, b) in x.local_force.iter().zip(&r.local_force) {
        assert_eq!(a.len(), b.len());
        for (x, r) in a.iter().zip(b) {
            num += (x - r) * (x - r);
            den += r * r;
        }
    }
    let d = x.value["D"].as_f64().unwrap();
    let dr = r.value["D"].as_f64().unwrap();
    let dd =
        x.ds.iter()
            .zip(&r.ds)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
    let error = |window: usize, limit: usize| {
        let (mut a, mut b) = (0., 0.);
        for s in 0..limit.min(x.local_force.len()) {
            let first = s.saturating_sub(window - 1);
            for j in 0..x.local_force[0].len() {
                let fx =
                    (first..=s).map(|k| x.local_force[k][j]).sum::<f64>() / (s - first + 1) as f64;
                let fr =
                    (first..=s).map(|k| r.local_force[k][j]).sum::<f64>() / (s - first + 1) as f64;
                a += (fx - fr).powi(2);
                b += fr * fr;
            }
        }
        (a / b.max(1e-30)).sqrt()
    };
    let mut peaks = 0.;
    let mut peakden = 0.;
    for j in 0..x.local_force[0].len() / 2 {
        let peak = |f: &[Vec<f64>]| {
            f.iter()
                .map(|row| (row[2 * j].powi(2) + row[2 * j + 1].powi(2)).sqrt())
                .fold(0., f64::max)
        };
        let a = peak(&x.local_force);
        let b = peak(&r.local_force);
        peaks += (a - b).powi(2);
        peakden += b * b;
    }
    let a: std::collections::BTreeSet<_> = x.value["damaged_facets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j.as_u64().unwrap())
        .collect();
    let b: std::collections::BTreeSet<_> = r.value["damaged_facets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j.as_u64().unwrap())
        .collect();
    json!({"local_force_time_l2":(num/den.max(1e-30)).sqrt(),"local_force_first_0p5ms_l2":error(1,5),"local_force_first_2ms_l2":error(1,20),"local_force_smoothed_0p5ms_l2":error(5,usize::MAX),"spring_peak_force_l2":(peaks/peakden.max(1e-30)).sqrt(),"damaged_facets_jaccard":a.intersection(&b).count() as f64/a.union(&b).count().max(1) as f64,"D_relative_error":(d-dr)/dr.max(1e-20),"max_D_history_error_over_final_D":dd/dr.max(1e-20),"work_gain":r.value["work"].as_f64().unwrap()/x.value["work"].as_f64().unwrap(),"wall_gain_including_build":r.value["wall_s"].as_f64().unwrap()/x.value["wall_s"].as_f64().unwrap()})
}
pub fn experiment() -> Value {
    let mut rows = vec![];
    for seed in [1, 2] {
        let sc = wall_impact(seed, 0.1, 8., 1., 1.5, 50.);
        let mut p = Params::new(0.001, 1e9);
        p.open0 = Open0::All;
        p.rereduce = false;
        let h = World::new(sc.clone(), p).he;
        let r = run(sc.clone(), None, h, 0.03, 0.45, "none");
        eprintln!("wall seed{seed} fine: {}", r.value);
        rows.push(json!({"seed":seed,"run":r.value,"reference":true}));
        let rh = run(sc.clone(), None, h / 2., 0.03, 0.45, "none");
        let rc = compare(&rh, &r);
        rows.push(
            json!({"seed":seed,"run":rh.value,"half_step_reference":true,"comparison_to_h":rc}),
        );
        let carrier = run(sc.clone(), Some(FULL_LINEAR), h, 0.03, 0.45, "none");
        let cc = compare(&carrier, &r);
        eprintln!("wall seed{seed} linear carrier: {} {}", carrier.value, cc);
        rows.push(json!({"seed":seed,"run":carrier.value,"comparison":cc}));
        for (modes, width, bridge) in [
            (0, 0.45, "none"),
            (8, 0.45, "none"),
            (32, 0.45, "none"),
            (96, 0.45, "none"),
            (192, 0.45, "none"),
            (32, 1., "none"),
            (96, 1., "none"),
            (0, 0.45, "dashpot"),
            (0, 0.45, "wave_free"),
            (0, 0.45, "wave_fixed"),
        ] {
            let x = run(sc.clone(), Some(modes), h, 0.03, width, bridge);
            let c = compare(&x, &r);
            let projection = if width == 0.45 {
                Some(compare(&x, &carrier))
            } else {
                None
            };
            eprintln!(
                "wall seed{seed} cms{modes} w{width} {bridge}: {} {}",
                x.value, c
            );
            rows.push(json!({"seed":seed,"run":x.value,"comparison":c,"comparison_to_same_linear_far_physics":projection}));
        }
    }
    json!({"scope":"S3b with dmin=0.1m (not original 0.04m). Central strip stays nonlinear, complete interface traces retained, remote undamaged secant K linearised at the prestressed initial state. Shared stable microstep, no multirate or contact reduction. Offline dense eigen build is charged separately and in wall time. Gains are CPU work estimates, not GPU timings.","rows":rows})
}
pub fn original_geometry() -> Value {
    let sc = wall_impact(1, 0.04, 8., 1., 1.5, 50.);
    let mut p = Params::new(0.001, 1e9);
    p.open0 = Open0::All;
    p.rereduce = false;
    let h = World::new(sc.clone(), p).he;
    let r = run(sc.clone(), None, h, 0.03, 0.45, "none");
    let mut rows = vec![json!({"run":r.value,"reference":true})];
    for scale in [1., 0.5] {
        let x = run(sc.clone(), Some(FULL_LINEAR), h * scale, 0.03, 0.45, "none");
        let ref_row = if scale == 1. {
            None
        } else {
            Some(run(sc.clone(), None, h * scale, 0.03, 0.45, "none"))
        };
        let reference = ref_row.as_ref().unwrap_or(&r);
        let c = compare(&x, reference);
        eprintln!(
            "original sparse carrier h/{}, {} {}",
            1. / scale,
            x.value,
            c
        );
        rows.push(json!({"run":x.value,"comparison":c}));
        // Deliberately noncausal diagnostic: does a cheap linear wave carrier
        // suffice if we already know which remote material will damage? This
        // isolates the attainable benefit BEFORE designing an online activator.
        let mut extra = std::collections::BTreeSet::new();
        for fi in reference.value["damaged_facets"].as_array().unwrap() {
            let f = &sc.m.facets[fi.as_u64().unwrap() as usize];
            extra.insert(f.a);
            extra.insert(f.b);
        }
        let extra: Vec<_> = extra.into_iter().collect();
        let oracle = run_with_extra(
            sc.clone(),
            Some(FULL_LINEAR),
            h * scale,
            0.03,
            0.45,
            "none",
            &extra,
        );
        let oc = compare(&oracle, reference);
        eprintln!(
            "original future-mask oracle h/{}, {} {}",
            1. / scale,
            oracle.value,
            oc
        );
        rows.push(json!({"run":oracle.value,"comparison":oc,"oracle_future_damage_mask":true,"scope":"Uses the full reference's future damaged facets. Tests localization potential, NOT an implementable online scheduler."}));
        if let Some(ref_row) = ref_row {
            rows.push(json!({"run":ref_row.value,"half_step_reference":true}));
        }
    }
    json!({"scope":"Original S3b geometry (.04m, 3329 cells). Same separate velocity-Verlet integrator for full nonlinear and sparse affine far field, no multirate. Far state dimension is NOT reduced. h and h/2 controls.","rows":rows})
}
#[test]
fn all_modes_span_far_dofs() {
    let sc = wall_impact(3, 0.3, 2., 1., 0.1, 50.);
    let fine: Vec<_> =
        sc.m.bodies
            .iter()
            .enumerate()
            .map(|(i, b)| i >= sc.m.n_cells() || (b.x0.x - 1.).abs() < 0.4)
            .collect();
    let p = make_port(&sc.m, &fine, usize::MAX);
    assert_eq!(p.cols.len(), p.ids.len());
}
#[test]
fn bridge_kick_books_exact_kinetic_exchange() {
    let sc = wall_impact(3, 0.3, 2., 1., 0.1, 50.);
    let fine: Vec<_> =
        sc.m.bodies
            .iter()
            .enumerate()
            .map(|(i, b)| i >= sc.m.n_cells() || (b.x0.x - 1.).abs() < 0.4)
            .collect();
    let p = make_port(&sc.m, &fine, 2);
    let fi =
        sc.m.facets
            .iter()
            .enumerate()
            .find(|(_, f)| f.tag == 0 && fine[f.a] != fine[f.b])
            .unwrap()
            .0;
    let mut b = vec![WaveBranch {
        fi,
        pt: 0,
        tangent: false,
        z: 1e5,
        ring: vec![0.03, 0.],
        head: 0,
        reflection: -1.,
        stored: 0.,
    }];
    let mut v = sc.v_init.clone();
    for i in 0..v.len() {
        if sc.m.bodies[i].mass > 0. {
            v[i] = [0.1, 0.2, 0.05];
        }
    }
    let mut vz = vec![0.01; p.cols.len()];
    let energy = |v: &[[f64; 3]], vz: &[f64]| {
        0.5 * dot(vz, vz)
            + (0..v.len())
                .filter(|&i| fine[i])
                .map(|i| {
                    let m = &sc.m.bodies[i];
                    0.5 * m.mass * (v[i][0] * v[i][0] + v[i][1] * v[i][1])
                        + 0.5 * m.inertia * v[i][2] * v[i][2]
                })
                .sum::<f64>()
    };
    let e0 = energy(&v, &vz);
    bridge_kick(
        &mut b, &sc, &sc.m.q, &mut v, &mut vz, &fine, &p, 1e-5, &mut 0.,
    );
    let e1 = energy(&v, &vz);
    assert!((e1 - e0 + b[0].stored).abs() < 1e-11);
}
#[test]
fn normal_coordinate_stiffness_matches_physical_facet_energy() {
    let sc = wall_impact(4, 0.2, 2., 1., 0.1, 50.);
    let fine: Vec<_> =
        sc.m.bodies
            .iter()
            .enumerate()
            .map(|(i, b)| i >= sc.m.n_cells() || (b.x0.x - 1.).abs() < 0.4)
            .collect();
    let p = make_port(&sc.m, &fine, 3);
    let z: Vec<_> = (0..p.cols.len())
        .map(|j| 1e-5 * (j as f64 * 0.73).sin())
        .collect();
    let mut delta = vec![[0.; 3]; sc.m.bodies.len()];
    for (j, &(i, d)) in p.ids.iter().enumerate() {
        delta[i][d] = p.cols.iter().zip(&z).map(|(c, x)| c[j] * x).sum();
    }
    let mut physical = 0.;
    for f in
        sc.m.facets
            .iter()
            .filter(|f| f.active && !fine[f.a] && !fine[f.b])
    {
        for pt in 0..2 {
            let (j, e) = sc.m.eval_point(&sc.m.q, f, pt);
            let dn = dot(&j.jna, &delta[f.a]) + dot(&j.jnb, &delta[f.b]);
            let dt = dot(&j.jta, &delta[f.a]) + dot(&j.jtb, &delta[f.b]);
            let (kn, kt) = f.pts[pt].law.secant(j.dn, e.d);
            physical += 0.5 * (kn * dn * dn + kt * dt * dt);
        }
    }
    let reduced = 0.5 * z.iter().zip(&p.lambda).map(|(x, l)| x * x * l).sum::<f64>();
    assert!(
        (physical - reduced).abs() < 1e-9 * physical.max(1e-20),
        "{physical} vs {reduced}"
    );
}
