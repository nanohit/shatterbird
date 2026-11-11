//! Causal nonlinear mask on an unreduced linear wave carrier. No oracle inputs.
use experiments::{
    engine::{
        carrier::{Config, Stats},
        scenes::{self, Scene},
        world::{fragments, Open0, Params, World},
    },
    meta, write_json,
};
use serde::Serialize;
use serde_json::json;
use std::time::Instant;

struct Sample {
    t: f64,
    force: Vec<[f64; 4]>,
    near: Vec<bool>,
}
#[derive(Serialize)]
struct ResultRow {
    mode: String,
    completed: bool,
    cells: usize,
    h: f64,
    end: f64,
    d: f64,
    onset: Option<f64>,
    runtime_s: f64,
    energy_defect: f64,
    model_energy_removed: f64,
    mean_hot_cells: f64,
    peak_hot_cells: f64,
    mean_hot_facets: f64,
    cold_chi_max_audit: f64,
    cold_over_onset_audit: usize,
    stats: Option<Stats>,
    fragment_masses: Vec<f64>,
    damage: Vec<usize>,
    rebound: Option<f64>,
    rows: Vec<[f64; 7]>,
}
fn sample(w: &World) -> Sample {
    let m = &w.sc.m;
    let mut force = vec![[0.; 4]; m.facets.len()];
    let mut near = vec![false; m.facets.len()];
    for (fi, f) in m.facets.iter().enumerate() {
        if !f.active || f.tag != 0 {
            continue;
        }
        for p in 0..2 {
            let (_, e) = if let Some(c) = w.carrier.as_ref() {
                if c.hot[fi] {
                    m.eval_point(&w.q, f, p)
                } else {
                    c.linear(m, fi, p, &w.q)
                }
            } else {
                m.eval_point(&w.q, f, p)
            };
            force[fi][2 * p] = e.fnorm;
            force[fi][2 * p + 1] = e.ftan;
            near[fi] |= e.kappa >= 0.8 * f.pts[p].law.delta0;
        }
    }
    Sample {
        t: w.t,
        force,
        near,
    }
}
fn run(
    sc: Scene,
    mode: &str,
    config: Option<Config>,
    scale: f64,
    end: f64,
    rec: f64,
) -> (ResultRow, Vec<Sample>) {
    let mut p = Params::new(0.001, 1e9);
    p.open0 = Open0::All;
    p.rereduce = false;
    p.he_scale = scale;
    let mut w = World::new(sc, p);
    if let Some(c) = config {
        w.enable_carrier(c);
    }
    let cells = w.sc.m.n_cells();
    let h = w.he;
    let (ke, u, vg) = w.energies();
    let e0 = ke + u + vg + w.sc.m.dissipated;
    let mut samples = vec![sample(&w)];
    let mut next = rec;
    let mut rows = vec![];
    let mut runtime = 0.;
    let mut completed = true;
    let mut hot_integral = 0.;
    let mut facet_integral = 0.;
    let mut peak: f64 = 0.;
    let mut onset = None;
    let mut maxchi: f64 = 0.;
    let mut missed = 0;
    while w.t < end - 1e-12 {
        let t = w.t;
        let clock = Instant::now();
        w.apply_removals();
        let stop = w.next_removal().map_or(end, |x| x.min(end));
        if !w.macro_step(stop) {
            completed = false;
            break;
        }
        w.drop_open_cracks();
        runtime += clock.elapsed().as_secs_f64();
        if onset.is_none() && w.sc.m.dissipated > 0. {
            onset = Some(w.t);
        }
        let (hot, hf) = if let Some(c) = w.carrier.as_ref() {
            let count =
                w.sc.m
                    .facets
                    .iter()
                    .enumerate()
                    .filter(|(i, f)| f.tag == 0 && f.active && c.hot[*i])
                    .count();
            let all =
                w.sc.m
                    .facets
                    .iter()
                    .filter(|f| f.tag == 0 && f.active)
                    .count();
            (
                c.hot_cells(&w.sc.m) as f64 / cells as f64,
                count as f64 / all.max(1) as f64,
            )
        } else {
            (1., 1.)
        };
        hot_integral += hot * (w.t - t);
        facet_integral += hf * (w.t - t);
        peak = peak.max(hot);
        if w.t >= next - 1e-12 || w.t >= end - 1e-12 {
            if let Some(c) = w.carrier.as_ref() {
                for (fi, f) in w.sc.m.facets.iter().enumerate() {
                    if !f.active || f.tag != 0 || c.hot[fi] {
                        continue;
                    }
                    for p in 0..2 {
                        let (j, _) = w.sc.m.eval_point(&w.q, f, p);
                        let l = &f.pts[p].law;
                        let chi = (j.dn.max(0.).powi(2) + l.gamma * j.dt * j.dt).sqrt() / l.delta0;
                        maxchi = maxchi.max(chi);
                        missed += (chi > 1.) as usize;
                    }
                }
            }
            let (ke, u, vg) = w.energies();
            let defect =
                ke + u + vg + w.sc.m.dissipated + w.led.d_fric + w.led.d_model - w.led.w_ext - e0;
            rows.push([w.t, w.sc.m.dissipated, hot, hf, defect, ke, u]);
            samples.push(sample(&w));
            while next <= w.t + 1e-12 {
                next += rec;
            }
        }
    }
    let (ke, u, vg) = w.energies();
    let damage =
        w.sc.m
            .facets
            .iter()
            .enumerate()
            .filter(|(_, f)| f.tag == 0 && f.pts.iter().any(|p| p.law.damage(p.kappa_c) > 1e-8))
            .map(|(i, _)| i)
            .collect();
    let masses = fragments(&w.sc.m, &w.q, &w.v, 0.)
        .iter()
        .map(|f| f.mass)
        .collect();
    let row = ResultRow {
        mode: mode.into(),
        completed,
        cells,
        h,
        end: w.t,
        d: w.sc.m.dissipated,
        onset,
        runtime_s: runtime,
        energy_defect: ke + u + vg + w.sc.m.dissipated + w.led.d_fric + w.led.d_model
            - w.led.w_ext
            - e0,
        model_energy_removed: w.led.d_model,
        mean_hot_cells: hot_integral / w.t,
        peak_hot_cells: peak,
        mean_hot_facets: facet_integral / w.t,
        cold_chi_max_audit: maxchi,
        cold_over_onset_audit: missed,
        stats: w.carrier.as_ref().map(|c| c.stats.clone()),
        fragment_masses: masses,
        damage,
        rebound: w.sc.impactor.map(|i| w.v[i][1]),
        rows,
    };
    (row, samples)
}
fn compare(a: &ResultRow, at: &[Sample], b: &ResultRow, bt: &[Sample]) -> serde_json::Value {
    assert_eq!(at.len(), bt.len());
    let mut num = 0.;
    let mut den = 0.;
    let mut lnum = 0.;
    let mut lden = 0.;
    let mut pre_num = 0.;
    let mut pre_den = 0.;
    let mut initial_num = 0.;
    let mut initial_den = 0.;
    for (x, y) in at.iter().zip(bt) {
        assert!((x.t - y.t).abs() < 1e-10);
        for fi in 0..x.force.len() {
            for c in 0..4 {
                let e = (x.force[fi][c] - y.force[fi][c]).powi(2);
                let r = y.force[fi][c].powi(2);
                num += e;
                den += r;
                if y.near[fi] {
                    lnum += e;
                    lden += r;
                }
                if b.onset.is_none_or(|t| y.t < t) {
                    pre_num += e;
                    pre_den += r;
                }
                if b.onset.is_none_or(|t| y.t < t + 0.01) {
                    initial_num += e;
                    initial_den += r;
                }
            }
        }
    }
    let inter = a.damage.iter().filter(|x| b.damage.contains(x)).count();
    let union = a.damage.len() + b.damage.len() - inter;
    json!({"D_relative_error":(a.d-b.d)/b.d.max(1e-30),"force_time_L2":(num/den.max(1e-30)).sqrt(),"near_onset_force_time_L2":(lnum/lden.max(1e-30)).sqrt(),"pre_onset_force_time_L2":(pre_num/pre_den.max(1e-30)).sqrt(),"first_fracture_force_time_L2":(initial_num/initial_den.max(1e-30)).sqrt(),"damage_Jaccard":if union==0{1.}else{inter as f64/union as f64},"runtime_gain":b.runtime_s/a.runtime_s,"reference_D":b.d,"reference_runtime_s":b.runtime_s,"reference_energy_defect":b.energy_defect})
}
fn suite(case: &str, seed: u64, scale: f64) {
    let (sc, end, rec) = match case {
        "wall" => (
            scenes::wall_impact(seed, 0.04, 8., 1., 1.5, 50.),
            0.03,
            0.0005,
        ),
        "wall_coarse" => (
            scenes::wall_impact(seed, 0.1, 8., 1., 1.5, 50.),
            0.03,
            0.0005,
        ),
        "wall_fast" => (
            scenes::wall_impact(seed, 0.08, 8., 1., 3., 50.),
            0.03,
            0.0005,
        ),
        "cantilever" => (
            scenes::cantilever(seed, 0.08, true, 0.005, 1.25 * 0.9046875),
            0.3,
            0.005,
        ),
        "cantilever_pulse" | "cantilever_midpulse" => {
            let mut sc = scenes::cantilever(seed, 0.08, true, 0.005, 1.25 * 0.9046875);
            // Held-out second event with 2 ms ramps after the first fracture.
            // Tip pulse: 5 kN / 25 Ns (often hits an already detached fragment).
            // Midspan pulse: 25 kN / 125 Ns, loading the remaining structure.
            // Opposite held patterns cancel before 120 ms; the mask never sees
            // this schedule, only the ensuing current displacement field.
            let mid = case == "cantilever_midpulse";
            let targets: Vec<usize> = if mid {
                sc.m.bodies
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| {
                        b.kind == model::BodyKind::Cell
                            && b.x0.x > 0.45 * sc.len
                            && b.x0.x < 0.55 * sc.len
                    })
                    .map(|(i, _)| i)
                    .collect()
            } else {
                sc.tip_cells.clone()
            };
            let mass: f64 = targets.iter().map(|&i| sc.m.bodies[i].mass).sum();
            let force = if mid { 25000.0 } else { 5000.0 };
            let forces: Vec<_> = targets
                .iter()
                .map(|&i| (i, [0.0, force * sc.m.bodies[i].mass / mass, 0.0]))
                .collect();
            sc.releases.push(scenes::Release {
                forces: forces.clone(),
                t0: 0.12,
                tr: 0.002,
            });
            sc.releases.push(scenes::Release {
                forces: forces
                    .iter()
                    .map(|(i, f)| (*i, [-f[0], -f[1], -f[2]]))
                    .collect(),
                t0: 0.125,
                tr: 0.002,
            });
            sc.name
                .push_str(if mid { "_mid_pulse" } else { "_tip_pulse" });
            (sc, 0.3, 0.005)
        }
        "span" => (scenes::span_collapse(seed, 0.08, 12., 0.4, 5), 0.5, 0.005),
        "contact" => (
            scenes::collapse_beam(seed, 0.06, 6., 0.3, 1., 0.5, 0.5),
            1.2,
            0.01,
        ),
        _ => panic!("unknown scene"),
    };
    let preset = std::env::args().nth(4).unwrap_or("standard".into());
    let name = format!(
        "{case}_s{seed}_h{scale}{}",
        if preset == "standard" {
            String::new()
        } else {
            format!("_{preset}")
        }
    );
    eprintln!("{name}: reference, {} cells", sc.m.n_cells());
    let (reference, rt) = run(sc.clone(), "full", None, scale, end, rec);
    eprintln!("reference D={} CPU={}s", reference.d, reference.runtime_s);
    let mut runs = vec![];
    let configs = match preset.as_str() {
        "reference" => vec![],
        "thresholds" => vec![
            (
                "early",
                Config {
                    chi: 0.4,
                    ..Config::default()
                },
            ),
            (
                "tight_geometry",
                Config {
                    geometry: Some(0.005),
                    ..Config::default()
                },
            ),
            (
                "strict",
                Config {
                    chi: 0.4,
                    geometry: Some(0.001),
                    ..Config::default()
                },
            ),
        ],
        "rebase" => vec![
            (
                "rebase",
                Config {
                    rebase: true,
                    ..Config::default()
                },
            ),
            (
                "rebase_tight",
                Config {
                    geometry: Some(0.005),
                    rebase: true,
                    ..Config::default()
                },
            ),
        ],
        _ => vec![
            ("causal", Config::default()),
            (
                "strain_only",
                Config {
                    geometry: None,
                    ..Config::default()
                },
            ),
        ],
    };
    write_json(
        "phase0g",
        &name,
        &json!({"meta":meta("phase0g",&name,Some(seed),"all-E production explicit reference",json!({"case":case,"scale":scale})),"reference":reference,"runs":[],"scope":"CPU f64 reference; no mask. Runtime excludes startup and diagnostics."}),
    );
    for (mode, config) in configs {
        let (row, tr) = run(sc.clone(), mode, Some(config), scale, end, rec);
        let cmp = compare(&row, &tr, &reference, &rt);
        eprintln!(
            "{mode}: D={} hot={:.3} gain={} force={} coldchi={}",
            row.d,
            row.mean_hot_cells,
            cmp["runtime_gain"],
            cmp["force_time_L2"],
            row.cold_chi_max_audit
        );
        runs.push(json!({"config":config,"result":row,"comparison":cmp}));
        write_json(
            "phase0g",
            &name,
            &json!({"meta":meta("phase0g",&name,Some(seed),"all-E production clock, all wave DOFs retained; guard precedes constitutive force",json!({"case":case,"scale":scale})),"reference":reference,"runs":runs,"scope":"CPU f64; runtime excludes output/diagnostic sampling and startup for every mode; full-point audits never feed activation. Work.total is deliberately NOT a speedup claim: it still charges cold evaluations as full points and does not model the new guards. No reduction of explicit substep count, wave memory, contacts, or GPU dispatches."}),
        );
    }
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    suite(
        args.get(1).map(String::as_str).unwrap_or("wall_coarse"),
        args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1),
        args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1.),
    );
}
