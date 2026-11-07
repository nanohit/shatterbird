//! E6 — notched beam in three-point bending (lab scale), S0 under CMOD control:
//! 3 resolutions × 3 tessellations, load-step refinement, and S1 under budgets.
//!
//! Notch: seeds within 2·d_min of the notch plane and below the notch tip (+d_min)
//! are mirror pairs, so the notch faces are exact Voronoi edges; above, the
//! tessellation is random (no pre-formed crack path).

use experiments::{meta, write_json, Csv};
use geometry::rng::Rng;
use geometry::rsa::rsa_rect;
use geometry::voronoi2d::{rect, voronoi_convex, Tessellation};
use geometry::V2;
use model::{BoundaryAttach, LawKind, LawOptions, Material, Model};
use serde_json::{json, Value};
use solvers::assembly::Tangent;
use solvers::s0::{body_reaction, newton, newton_controlled, pattern_work, Control, NewtonOpts};
use solvers::s1::{Alm, RhoPolicy};

const EXP: &str = "E6";
const LEN: f64 = 0.5;
const DEPTH: f64 = 0.1;
const SPAN: f64 = 0.4;
const NOTCH: f64 = 0.05;
const B: f64 = 0.1;
const PAD: f64 = 0.01;
const CENTRAL: f64 = 0.08;

fn tess(dmin: f64, seed: u64) -> Tessellation {
    let xc = LEN / 2.0;
    let mut rng = Rng::new(seed);
    let band = 2.0 * dmin;
    let ytop = NOTCH + dmin;
    let all = rsa_rect(V2::new(0.0, 0.0), V2::new(LEN, DEPTH), dmin, &mut rng, 20_000);
    let mut seeds: Vec<V2> = all.into_iter().filter(|p| !((p.x - xc).abs() < band && p.y < ytop)).collect();
    // RSA in the left half of the band, keeping d_min to existing seeds and to the plane
    let mut fails = 0;
    let mut half = Vec::new();
    while fails < 5000 {
        let p = V2::new(rng.range(xc - band, xc - 0.5 * dmin), rng.range(0.0, ytop));
        let mirror = V2::new(2.0 * xc - p.x, p.y);
        let ok = seeds.iter().chain(half.iter()).all(|s: &V2| (*s - p).norm() >= dmin && (*s - mirror).norm() >= dmin);
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

struct Beam {
    m: Model,
    plate: usize,
    cmod: Control,
    lig_area: f64,
}

fn beam(dmin: f64, seed: u64) -> Beam {
    let t = tess(dmin, seed);
    let xc = LEN / 2.0;
    let mut m = Model::from_tessellation(&t, Material::concrete(B), LawOptions::damage(), &[]);
    // notch: facets on the plane below the tip
    for f in m.facets.iter_mut() {
        let on_plane = (f.ends[0].x - xc).abs() < 1e-12 && (f.ends[1].x - xc).abs() < 1e-12;
        if on_plane && f.pc.y < NOTCH {
            f.active = false;
        }
    }
    let g = m.ground();
    let x1 = (LEN - SPAN) / 2.0;
    let x2 = x1 + SPAN;
    // simply supported: both supports are normal-only rollers (elastic, always in compression)
    m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: Some((x1 - PAD, x1 + PAD)), body: g, shear: false, tag: 1 });
    m.attach_boundary(&t, &BoundaryAttach { edge: 0, range: Some((x2 - PAD, x2 + PAD)), body: g, shear: false, tag: 2 });
    // loading plate: free only in y. x and θ fixed, its pad facets normal-only in the plate's
    // frame: it transmits vertical force only. With θ free its normals would tilt and, with x
    // fixed, a second horizontal constraint would form an arch with the gauge (seen in a
    // first run: load rising again after the peak). Pad facets are unilateral (no tensile
    // strength) so the plate cannot bridge the crack.
    let plate = m.add_plate(V2::new(xc, DEPTH), 0.0, 0.0, [true, false, true]);
    m.attach_boundary(&t, &BoundaryAttach { edge: 2, range: Some((xc - PAD, xc + PAD)), body: plate, shear: false, tag: 3 });
    // pad and supports are unilateral contacts (no tensile strength, shear-free)
    for f in m.facets.iter_mut().filter(|f| f.tag != 0) {
        for p in f.pts.iter_mut() {
            p.law.kind = LawKind::Damage;
            p.law.delta0 *= 1e-4;
            p.law.wc *= 1e-4;
        }
    }
    // Damage allowed only in the central zone |x − x_c| < 0.8 D (declared specimen choice,
    // standard for 3PB analyses): SB-IL1 has no pressure-dependent strength, so concentrated
    // support bearing otherwise produces splitting cracks that a real specimen does not show
    // (seen in development runs: cracks around both supports, load rising after the peak).
    for f in m.facets.iter_mut().filter(|f| f.tag == 0) {
        if (f.pc.x - xc).abs() >= CENTRAL {
            for p in f.pts.iter_mut() {
                p.law.kind = LawKind::Elastic;
            }
        }
    }
    // horizontal gauge: one x-DOF of a cell near the left end (zero reaction by equilibrium)
    let gauge = (0..t.cells.len())
        .min_by(|&a, &b| (m.bodies[a].x0 - V2::new(0.05, 0.05)).norm().partial_cmp(&(m.bodies[b].x0 - V2::new(0.05, 0.05)).norm()).unwrap())
        .unwrap();
    m.bodies[gauge].fixed[0] = true;
    m.pattern[plate][1] = -1.0;
    // CMOD: horizontal relative displacement of the two mouth points at (xc, 0)
    let mouth = V2::new(xc, 0.0);
    let find = |left: bool| -> usize {
        (0..t.cells.len())
            .filter(|&i| {
                let c = &t.cells[i];
                (if left { c.seed.x < xc } else { c.seed.x > xc }) && c.verts.iter().any(|v| (*v - mouth).norm() < 1e-12)
            })
            .next()
            .expect("mouth cell")
    };
    let (l, r) = (find(true), find(false));
    let mut cmod = Control::default();
    cmod.terms.push((r, mouth - m.bodies[r].x0, V2::new(1.0, 0.0), 1.0));
    cmod.terms.push((l, mouth - m.bodies[l].x0, V2::new(1.0, 0.0), -1.0));
    Beam { m, plate, cmod, lig_area: (DEPTH - NOTCH) * B }
}

/// Event size over material facets only (the unilateral pad "breaks" by design on lift-off).
fn material_dd(m: &Model, q: &[[f64; 3]]) -> f64 {
    let mut mx: f64 = 0.0;
    for f in m.facets.iter().filter(|f| f.active && f.tag == 0) {
        for p in 0..2 {
            let (_, e) = m.eval_point(q, f, p);
            mx = mx.max(e.d - f.pts[p].law.damage(f.pts[p].kappa_c));
        }
    }
    mx
}

struct Curve {
    cmod: Vec<f64>,
    disp: Vec<f64>,
    load: Vec<f64>,
}

fn run_s0(bm: &mut Beam, dd_max: f64, cmod_end: f64, csv: &mut Csv, id: f64) -> (Value, Curve) {
    let m = &mut bm.m;
    let fscale = m.mat.sigma_t * DEPTH * B;
    let opts = NewtonOpts { tol: 1e-9 * fscale, max_iter: 25 };
    let c0 = bm.cmod.value(&m.q);
    let mut q = m.q.clone();
    let mut lf = 0.0;
    let mut s = 0.0;
    let ds_max = 4e-6;
    let mut ds = 0.5e-6;
    let mut w = 0.0;
    let (mut steps, mut cuts, mut secant_used, mut newton_its) = (0usize, 0usize, 0usize, 0usize);
    let mut peak: f64 = 0.0;
    let mut curve = Curve { cmod: vec![0.0], disp: vec![0.0], load: vec![0.0] };
    let mut failed = None;
    let mut max_eps: f64 = 0.0;
    let t0 = std::time::Instant::now();
    while s < cmod_end {
        let tgt = (s + ds).min(cmod_end);
        let mut ok = false;
        let mut qt = q.clone();
        let mut lft = lf;
        for tang in [Tangent::Consistent, Tangent::Secant] {
            qt = q.clone();
            lft = lf;
            let rep = newton_controlled(m, &mut qt, &mut lft, &bm.cmod, c0 + tgt, if tang == Tangent::Secant { NewtonOpts { max_iter: 200, ..opts } } else { opts }, 1e-12, tang);
            newton_its += rep.iters;
            if rep.converged && material_dd(m, &qt) <= dd_max {
                ok = true;
                if tang == Tangent::Secant {
                    secant_used += 1;
                }
                break;
            }
        }
        if !ok {
            ds *= 0.5;
            cuts += 1;
            if ds < 1e-11 {
                failed = Some(s);
                break;
            }
            continue;
        }
        w += pattern_work(m, &q, &qt, lf, lft);
        m.commit(&qt);
        q = qt;
        lf = lft;
        s = tgt;
        steps += 1;
        peak = peak.max(lf);
        let disp = -(q[bm.plate][1] - m.bodies[bm.plate].x0.y);
        let u = m.elastic_energy(&q);
        let eps = w - u - m.dissipated;
        max_eps = max_eps.max(eps.abs());
        curve.cmod.push(s);
        curve.disp.push(disp);
        curve.load.push(lf);
        csv.row(&[id, steps as f64, s, disp, lf, w, u, m.dissipated, eps]);
        ds = (ds * 1.3).min(ds_max);
    }
    m.q = q.clone();
    // dissipation-weighted process-zone width around the notch plane
    let xc = LEN / 2.0;
    let (mut sw, mut sx2) = (0.0, 0.0);
    let mut dfacets = 0;
    for f in m.facets.iter().filter(|f| f.active && f.tag == 0) {
        let dsum: f64 = f.pts.iter().map(|p| p.law.dissipation(p.kappa_c)).sum();
        if dsum > 0.0 {
            dfacets += 1;
            sw += dsum;
            sx2 += dsum * (f.pc.x - xc).powi(2);
        }
    }
    // damage map: facets with any committed history
    let mut dm = Csv::new(EXP, &format!("damage_run{}", id as usize), &["x0", "y0", "x1", "y1", "d_max", "D_J", "dn_end", "dt_end"]);
    for f in m.facets.iter().filter(|f| f.active && f.tag == 0) {
        let dsum: f64 = f.pts.iter().map(|p| p.law.dissipation(p.kappa_c)).sum();
        if dsum > 0.0 {
            let dmx = f.pts.iter().map(|p| p.law.damage(p.kappa_c)).fold(0.0, f64::max);
            let (j, _) = m.eval_point(&q, f, 0);
            dm.row(&[f.ends[0].x, f.ends[0].y, f.ends[1].x, f.ends[1].y, dmx, dsum, j.dn, j.dt]);
        }
    }
    // final configuration (for deformed-shape plots)
    let mut qc = Csv::new(EXP, &format!("final_q_run{}", id as usize), &["x0", "y0", "x", "y", "theta"]);
    for i in 0..m.n_cells() {
        qc.row(&[m.bodies[i].x0.x, m.bodies[i].x0.y, q[i][0], q[i][1], q[i][2]]);
    }
    // split dissipation: facets that opened (δn > 0 at end) vs closed (shear in compression)
    let (mut d_open, mut d_closed) = (0.0, 0.0);
    for f in m.facets.iter().filter(|f| f.active && f.tag == 0) {
        for p in 0..2 {
            let dd = f.pts[p].law.dissipation(f.pts[p].kappa_c);
            if dd > 0.0 {
                let (j, _) = m.eval_point(&q, f, p);
                if j.dn > 0.0 {
                    d_open += dd;
                } else {
                    d_closed += dd;
                }
            }
        }
    }
    let gc_lig = m.mat.gc * bm.lig_area;
    let last_load = *curve.load.last().unwrap();
    (
        json!({"steps": steps, "cutbacks": cuts, "secant_steps": secant_used, "newton_iters": newton_its, "failed_at_cmod": failed,
            "peak_load_N": peak, "D_J": m.dissipated, "W_ext_J": w, "D_over_Gc_Alig": m.dissipated / gc_lig, "W_over_Gc_Alig": w / gc_lig,
            "final_load_over_peak": last_load / peak, "max_abs_ledger_residual_over_D": max_eps / m.dissipated.max(1e-300),
            "dissipating_facets": dfacets, "process_zone_rms_halfwidth_m": (sx2 / sw.max(1e-300)).sqrt(), "wall_s": t0.elapsed().as_secs_f64(),
            "D_open_points_J": d_open, "D_closed_points_J": d_closed}),
        curve,
    )
}

/// Displacement control of the loading plate with S0 or S1 (budgeted sweeps per step).
fn run_disp(bm: &mut Beam, s1_budget: Option<usize>, d_end: f64, nsteps: usize, csv: &mut Csv, id: f64) -> Value {
    let m = &mut bm.m;
    m.bodies[bm.plate].fixed[1] = true;
    m.pattern[bm.plate][1] = 0.0;
    let fscale = m.mat.sigma_t * DEPTH * B;
    let y0 = m.bodies[bm.plate].x0.y;
    let mut q = m.q.clone();
    let mut alm = s1_budget.map(|_| Alm::new(m, RhoPolicy::PerSpring(0.1), None));
    let mut peak: f64 = 0.0;
    let mut w = 0.0;
    let mut prev_p = 0.0;
    let mut prev_d = 0.0;
    let mut sweeps = 0u64;
    for k in 1..=nsteps {
        let d = d_end * k as f64 / nsteps as f64;
        q[bm.plate][1] = y0 - d;
        match (&mut alm, s1_budget) {
            (Some(a), Some(n)) => {
                for _ in 0..n {
                    a.iterate(m, &mut q, 0.0, None);
                }
                sweeps = a.work.sweeps;
            }
            _ => {
                newton(m, &mut q, 0.0, None, NewtonOpts { tol: 1e-9 * fscale, max_iter: 60 });
            }
        }
        m.commit(&q);
        let p = body_reaction(m, &q, bm.plate)[1];
        w += 0.5 * (p + prev_p) * (d - prev_d);
        prev_p = p;
        prev_d = d;
        peak = peak.max(p);
        csv.row(&[id, k as f64, d, p, m.dissipated]);
    }
    json!({"solver": match s1_budget { Some(n) => format!("S1 perspring c=0.1, {n} sweeps/step"), None => "S0 Newton".into() },
        "steps": nsteps, "peak_load_N": peak, "D_J": m.dissipated, "W_ext_J": w, "total_sweeps": sweeps})
}

fn main() {
    let dmins = [0.008, 0.0056, 0.004];
    let seeds = [1u64, 2, 3];
    let cmod_end = 1.0e-3;
    let mut csv = Csv::new(EXP, "s0_curves", &["run", "step", "cmod", "disp", "load", "W_ext", "U", "D", "residual"]);
    let mut runs = Vec::new();
    let mut id = 0.0;
    let quick = std::env::var("SB_E6_QUICK").is_ok();
    for &d in &dmins {
        for &seed in &seeds {
            if quick && (d != dmins[0] || seed != 1) {
                continue;
            }
            let mut bm = beam(d, seed);
            let (lo, hi) = bm.m.band_ratio();
            let cells = bm.m.n_cells();
            let (mut r, _) = run_s0(&mut bm, 0.1, cmod_end, &mut csv, id);
            r["run"] = json!(id);
            r["dmin"] = json!(d);
            r["seed"] = json!(seed);
            r["cells"] = json!(cells);
            r["ell_cb_over_lch"] = json!([lo, hi]);
            eprintln!("{r}");
            runs.push(r);
            id += 1.0;
        }
    }
    if quick {
        write_json(EXP, "quick", &json!({"runs": runs}));
        return;
    }
    // load-step (event-size) refinement on the middle resolution, seed 1
    let mut refine = Vec::new();
    for &ddm in &[0.2, 0.05] {
        let mut bm = beam(dmins[1], 1);
        let (mut r, _) = run_s0(&mut bm, ddm, cmod_end, &mut csv, id);
        r["run"] = json!(id);
        r["dd_max"] = json!(ddm);
        eprintln!("{r}");
        refine.push(r);
        id += 1.0;
    }
    // S1 under budgets vs S0, displacement control, coarsest mesh
    let mut csv2 = Csv::new(EXP, "disp_control", &["run", "step", "disp", "load", "D"]);
    let mut budget = Vec::new();
    for (k, b) in [None, Some(16usize), Some(128), Some(1024)].into_iter().enumerate() {
        let mut bm = beam(dmins[0], 1);
        let r = run_disp(&mut bm, b, 0.25e-3, 250, &mut csv2, k as f64);
        eprintln!("{r}");
        budget.push(r);
    }
    let m = beam(dmins[0], 1).m;
    let res = json!({
        "meta": meta(EXP, "all", None, "S0: bordered Newton under CMOD control (consistent tangent, secant fallback), accept if converged and max Δd ≤ dd_max (0.1), else halve; history committed on acceptance only",
            json!({"beam": {"length": LEN, "depth": DEPTH, "span": SPAN, "notch": NOTCH, "thickness": B, "pads": 2.0 * PAD},
                   "material": m.mat, "lch": m.mat.lch(), "dmins": dmins, "seeds": seeds, "cmod_end": cmod_end,
                   "law": "SB-IL1 geometric-area normalisation, crack-band option off; damage only for |x - x_c| < 0.08 m (elsewhere elastic)", "supports": "both rollers: normal-only, unilateral (zero tensile strength), support frame; loading pad likewise; plate moves in y only; horizontal gauge on one cell x-DOF (only horizontal constraint, zero reaction)"})),
        "s0_runs": runs,
        "step_refinement": refine,
        "budget": budget,
    });
    write_json(EXP, "summary", &res);
}
