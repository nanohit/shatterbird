//! E1 — two parallel springs, compliant ALM with exact scalar primal solve.
//! Validates the implementation against the closed forms of docs §1, then
//! measures errors vs iterations for the four prescribed regimes.

use experiments::{meta, write_json, Csv};
use serde_json::json;

const EXP: &str = "E1";

#[derive(Clone, Copy, Debug)]
struct Sys {
    k: [f64; 2],
    rho: [f64; 2],
    p: f64,
    /// Inertial weight μ = m/h² (0 = quasi-static) and prediction x̂.
    mu: f64,
    xhat: f64,
}

impl Sys {
    fn s(&self, i: usize) -> f64 {
        self.k[i] / (self.k[i] + self.rho[i])
    }
    fn w(&self, i: usize) -> f64 {
        self.rho[i] * self.s(i)
    }
    /// Exact minimiser of Φ(x; λ) = μ/2 (x-x̂)² − P x + Σ (s λ x + ½ k_eff x²).
    fn primal(&self, lam: [f64; 2]) -> f64 {
        (self.p + self.mu * self.xhat - self.s(0) * lam[0] - self.s(1) * lam[1]) / (self.w(0) + self.w(1) + self.mu)
    }
    fn dual(&self, lam: [f64; 2], x: f64) -> [f64; 2] {
        [self.s(0) * (lam[0] + self.rho[0] * x), self.s(1) * (lam[1] + self.rho[1] * x)]
    }
    fn iterate(&self, lam: [f64; 2]) -> (f64, [f64; 2]) {
        let x = self.primal(lam);
        model::range::rec("E1.s", self.s(0));
        model::range::rec("E1.s", self.s(1));
        model::range::rec("E1.1-s", 1.0 - self.s(0));
        model::range::rec("E1.1-s", 1.0 - self.s(1));
        model::range::rec("E1.k_eff", self.w(0));
        model::range::rec("E1.k_eff", self.w(1));
        let l = self.dual(lam, x);
        model::range::rec("E1.lambda", l[0]);
        model::range::rec("E1.lambda", l[1]);
        (x, l)
    }
    fn exact(&self) -> (f64, [f64; 2]) {
        let x = (self.p + self.mu * self.xhat) / (self.k[0] + self.k[1] + self.mu);
        (x, [self.k[0] * x, self.k[1] * x])
    }
    /// Quasi-static closed form r = (s1 w2 + s2 w1)/(w1 + w2) and left eigenvector ℓ.
    fn r(&self) -> f64 {
        (self.s(0) * self.w(1) + self.s(1) * self.w(0)) / (self.w(0) + self.w(1))
    }
    fn ell(&self) -> [f64; 2] {
        let t = self.rho[0] + self.rho[1];
        [self.rho[1] / t, -self.rho[0] / t]
    }
    /// Iteration matrix M = S − w sᵀ/(W + μ).
    fn mat(&self) -> [[f64; 2]; 2] {
        let d = self.w(0) + self.w(1) + self.mu;
        let mut m = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                m[i][j] = if i == j { self.s(i) } else { 0.0 } - self.w(i) * self.s(j) / d;
            }
        }
        m
    }
}

fn spectral_radius(m: [[f64; 2]; 2]) -> f64 {
    let tr = m[0][0] + m[1][1];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let disc = tr * tr / 4.0 - det;
    if disc >= 0.0 {
        let a = tr / 2.0 + disc.sqrt();
        let b = tr / 2.0 - disc.sqrt();
        a.abs().max(b.abs())
    } else {
        det.abs().sqrt()
    }
}

/// Predicted λ error after n ≥ 1 quasi-static iterations from initial error e0.
fn predicted(sys: &Sys, e0: [f64; 2], n: u32) -> [f64; 2] {
    let l = sys.ell();
    let c = sys.r().powi(n as i32) * (l[0] * e0[0] + l[1] * e0[1]);
    [c, -c]
}

struct Track {
    /// Worst deviation from the closed form while the predicted error > 1e-6·max|λ*|.
    max_rel_dev: f64,
    /// Error plateau reached when the prediction is far below round-off (relative to max|λ*|).
    floor: f64,
    rows: usize,
}

/// Iterate from λ0 and log; returns the worst relative deviation from the closed form
/// over iterations where the predicted error exceeds `floor`.
fn run_iters(sys: &Sys, lam0: [f64; 2], nmax: u32, csv: &mut Csv, cfg_id: f64, floor: f64) -> Track {
    let (xs, ls) = sys.exact();
    let e0 = [lam0[0] - ls[0], lam0[1] - ls[1]];
    let ustar = 0.5 * sys.k[0] * xs * xs + 0.5 * sys.k[1] * xs * xs;
    let mut lam = lam0;
    let mut tr = Track { max_rel_dev: 0.0, floor: 0.0, rows: 0 };
    let lmax = ls[0].abs().max(ls[1].abs());
    for n in 1..=nmax {
        let (x, l) = sys.iterate(lam);
        lam = l;
        let e = [lam[0] - ls[0], lam[1] - ls[1]];
        let pr = predicted(sys, e0, n);
        let pmag = pr[0].abs();
        if pmag > floor * lmax {
            let dev = ((e[0] - pr[0]).abs().max((e[1] - pr[1]).abs())) / pmag;
            tr.max_rel_dev = tr.max_rel_dev.max(dev);
        }
        let log_it = n <= 64 || n.is_power_of_two() || n % (nmax / 400).max(1) == 0 || n == nmax;
        if log_it {
            let ul = lam[0] * lam[0] / (2.0 * sys.k[0]) + lam[1] * lam[1] / (2.0 * sys.k[1]);
            let ux = 0.5 * (sys.k[0] + sys.k[1]) * x * x;
            csv.row(&[
                cfg_id,
                n as f64,
                (x - xs) / xs,
                e[0] / sys.p,
                e[1] / sys.p,
                (sys.k[0] * x - ls[0]) / sys.p,
                (sys.k[1] * x - ls[1]) / sys.p,
                (lam[0] + lam[1] - sys.p) / sys.p,
                (lam[0] - sys.k[0] * x) / sys.p,
                (lam[1] - sys.k[1] * x) / sys.p,
                (ul - ustar) / ustar,
                (ux - ustar) / ustar,
                pr[0] / sys.p,
                e[0].abs().max(e[1].abs()) / lmax,
            ]);
            tr.rows += 1;
        }
        if pmag < 1e-18 * lmax && n > 64 {
            tr.floor = e[0].abs().max(e[1].abs()) / lmax;
            break;
        }
        tr.floor = e[0].abs().max(e[1].abs()) / lmax;
    }
    tr
}

const HEAD: [&str; 14] = [
    "cfg", "iter", "x_rel_err", "e_lam1_overP", "e_lam2_overP", "e_kx1_overP", "e_kx2_overP", "sum_lam_minus_P_overP",
    "constit1_overP", "constit2_overP", "U_lambda_rel_err", "U_x_rel_err", "pred_e_lam1_overP", "force_err_rel_max",
];

fn main() {
    model::range::enable();
    let p = 1.0e3; // N
    let rho0 = 1.0e6; // N/m
    let ratios = [2.0, 10.0];
    let levels = [1.0, 10.0, 1000.0];
    let mut summary = Vec::new();

    // ---------- Regime 1: cold start, shared ρ ----------
    let mut csv = Csv::new(EXP, "cold_shared", &HEAD);
    let mut cfgs = Vec::new();
    let mut id = 0.0;
    for &ratio in &ratios {
        for &lev in &levels {
            let k1 = lev * rho0;
            let sys = Sys { k: [k1, ratio * k1], rho: [rho0, rho0], p, mu: 0.0, xhat: 0.0 };
            let r = sys.r();
            let r_shared = 2.0 / (2.0 + rho0 / sys.k[0] + rho0 / sys.k[1]);
            let r_eig = spectral_radius(sys.mat());
            let n100 = (100f64.ln() / -r.ln()).ceil();
            let nmax = ((45.0 / -r.ln()).ceil() as u32).clamp(200, 400_000);
            let tr = run_iters(&sys, [0.0, 0.0], nmax, &mut csv, id, 1e-6);
            // first-iterate split
            let (_, l1) = sys.iterate([0.0, 0.0]);
            cfgs.push(json!({
                "cfg": id, "ratio": ratio, "k1_over_rho": lev, "r_formula": r, "r_shared_formula": r_shared,
                "r_eig": r_eig, "iters_per_100x": n100, "max_rel_dev_from_closed_form": tr.max_rel_dev,
                "roundoff_floor_rel": tr.floor, "eps_over_1_minus_r": f64::EPSILON / (1.0 - r),
                "split_after_first_update": [l1[0] / p, l1[1] / p], "exact_split": [sys.k[0] / (sys.k[0] + sys.k[1]), sys.k[1] / (sys.k[0] + sys.k[1])],
            }));
            id += 1.0;
        }
    }
    summary.push(json!({"regime": "cold_shared", "configs": cfgs}));

    // ---------- Per-spring ρ: closed form vs eigenvalue vs measured ----------
    let mut rng = geometry::rng::Rng::new(11);
    let mut worst_formula = 0.0f64;
    let mut worst_measured = 0.0f64;
    for _ in 0..200 {
        let k1 = 10f64.powf(rng.range(5.0, 9.0));
        let k2 = k1 * 10f64.powf(rng.range(0.0, 2.0));
        let c1 = 10f64.powf(rng.range(-4.0, 2.0));
        let c2 = 10f64.powf(rng.range(-4.0, 2.0));
        let sys = Sys { k: [k1, k2], rho: [c1 * k1, c2 * k2], p, mu: 0.0, xhat: 0.0 };
        let r = sys.r();
        worst_formula = worst_formula.max((r - spectral_radius(sys.mat())).abs() / r);
        // measured: ratio of successive errors
        let (_, ls) = sys.exact();
        let mut lam = [0.0, 0.0];
        let mut prev = f64::NAN;
        for n in 1..=6 {
            lam = sys.iterate(lam).1;
            let e = (lam[0] - ls[0]).abs();
            if n >= 3 && e > 1e-9 * ls[0].abs() && prev > 0.0 {
                worst_measured = worst_measured.max((e / prev - r).abs() / r.max(1e-300));
            }
            prev = e;
        }
    }
    let mut csv = Csv::new(EXP, "cold_perspring", &HEAD);
    let mut cfgs = Vec::new();
    let mut id = 0.0;
    for &ratio in &ratios {
        for &c in &[1e-3, 1e-2, 0.1, 1.0, 10.0] {
            let k1 = 1000.0 * rho0;
            let sys = Sys { k: [k1, ratio * k1], rho: [c * k1, c * ratio * k1], p, mu: 0.0, xhat: 0.0 };
            let tr = run_iters(&sys, [0.0, 0.0], 100_000, &mut csv, id, 1e-6);
            cfgs.push(json!({"cfg": id, "ratio": ratio, "rho_over_k": c, "r": sys.r(), "r_expected_1_over_1pc": 1.0/(1.0+c),
                "iters_per_100x": (100f64.ln() / -sys.r().ln()).ceil(), "max_rel_dev_from_closed_form": tr.max_rel_dev}));
            id += 1.0;
        }
    }
    summary.push(json!({"regime": "cold_perspring", "random_pairs_formula_vs_eig_max_rel": worst_formula,
        "random_pairs_measured_vs_formula_max_rel": worst_measured, "configs": cfgs}));

    // ---------- Regime 2: slow ramp, warm start ----------
    let mut csv = Csv::new(EXP, "ramp", &["ratio", "k1_over_rho", "steps", "m", "rel_err_final", "rel_err_pred", "x_rel_err_final"]);
    let mut ramp_dev = 0.0f64;
    let mut ramps = Vec::new();
    for &ratio in &ratios {
        for &lev in &levels {
            let k1 = lev * rho0;
            for &nsteps in &[60u32, 600, 6000] {
                for &mm in &[1u32, 4, 8, 16] {
                    let mut sys = Sys { k: [k1, ratio * k1], rho: [rho0, rho0], p: 0.0, mu: 0.0, xhat: 0.0 };
                    let dp = p / nsteps as f64;
                    let mut lam = [0.0, 0.0];
                    let mut x = 0.0;
                    for n in 1..=nsteps {
                        sys.p = n as f64 * dp;
                        for _ in 0..mm {
                            let r = sys.iterate(lam);
                            x = r.0;
                            lam = r.1;
                        }
                    }
                    let (xs, ls) = sys.exact();
                    let l = sys.ell();
                    let kap = [k1 / (k1 + ratio * k1), ratio * k1 / (k1 + ratio * k1)];
                    let delta = (l[0] * kap[0] + l[1] * kap[1]) * dp;
                    let rm = sys.r().powi(mm as i32);
                    let c = -delta * rm * (1.0 - rm.powi(nsteps as i32)) / (1.0 - rm);
                    let lmax = ls[0].abs().max(ls[1].abs());
                    let err = (lam[0] - ls[0]).abs().max((lam[1] - ls[1]).abs()) / lmax;
                    let pred = c.abs() / lmax;
                    if pred > 1e-10 {
                        ramp_dev = ramp_dev.max((err - pred).abs() / pred);
                    }
                    csv.row(&[ratio, lev, nsteps as f64, mm as f64, err, pred, (x - xs) / xs]);
                    if ratio == 2.0 && lev == 1000.0 {
                        ramps.push(json!({"steps": nsteps, "m": mm, "rel_force_err": err, "pred": pred}));
                    }
                }
            }
        }
    }
    summary.push(json!({"regime": "ramp_warm", "max_rel_dev_from_closed_form": ramp_dev, "ratio2_k1000": ramps}));

    // ---------- Regime 3: step change in P ----------
    let mut csv = Csv::new(EXP, "stepP", &HEAD);
    let mut cfgs = Vec::new();
    let mut id = 0.0;
    for &ratio in &ratios {
        for &lev in &levels {
            let k1 = lev * rho0;
            for &fac in &[2.0, 0.5] {
                let sys_a = Sys { k: [k1, ratio * k1], rho: [rho0, rho0], p, mu: 0.0, xhat: 0.0 };
                let (_, la) = sys_a.exact();
                let sys_b = Sys { p: fac * p, ..sys_a };
                let tr = run_iters(&sys_b, la, 100_000, &mut csv, id, 1e-6);
                cfgs.push(json!({"cfg": id, "ratio": ratio, "k1_over_rho": lev, "P_factor": fac, "r": sys_b.r(), "max_rel_dev_from_closed_form": tr.max_rel_dev}));
                id += 1.0;
            }
        }
    }
    summary.push(json!({"regime": "stepP", "configs": cfgs}));

    // ---------- Regime 4: step change in k2 (damage stand-in), three λ policies ----------
    let mut csv = Csv::new(EXP, "stepK", &HEAD);
    let mut cfgs = Vec::new();
    let mut id = 0.0;
    for &ratio in &ratios {
        for &lev in &levels {
            let k1 = lev * rho0;
            for &fac in &[0.3, 0.05] {
                let sys_a = Sys { k: [k1, ratio * k1], rho: [rho0, rho0], p, mu: 0.0, xhat: 0.0 };
                let (_, la) = sys_a.exact();
                let sys_b = Sys { k: [k1, fac * ratio * k1], ..sys_a };
                for (pi, pol) in ["keep", "rescale", "reset"].iter().enumerate() {
                    let lam0 = match pi {
                        0 => la,
                        1 => [la[0], la[1] * fac],
                        _ => [0.0, 0.0],
                    };
                    let tr = run_iters(&sys_b, lam0, 100_000, &mut csv, id, 1e-6);
                    let (_, lb) = sys_b.exact();
                    let e0 = [lam0[0] - lb[0], lam0[1] - lb[1]];
                    let first = predicted(&sys_b, e0, 1);
                    cfgs.push(json!({"cfg": id, "ratio": ratio, "k1_over_rho": lev, "k2_factor": fac, "policy": pol,
                        "r_new": sys_b.r(), "first_iter_rel_force_err": first[0].abs() / lb[0].abs().max(lb[1].abs()),
                        "max_rel_dev_from_closed_form": tr.max_rel_dev}));
                    id += 1.0;
                }
            }
        }
    }
    summary.push(json!({"regime": "stepK", "configs": cfgs}));

    // ---------- S2 in E1 (quasi-static: S2 = S0) ----------
    let mut s2 = Vec::new();
    for &ratio in &ratios {
        for &lev in &levels {
            let k1 = lev * rho0;
            let sys = Sys { k: [k1, ratio * k1], rho: [rho0, rho0], p, mu: 0.0, xhat: 0.0 };
            let (xs, ls) = sys.exact();
            let u = p / (sys.k[0] + sys.k[1]); // 1×1 "factorization"
            let f = [sys.k[0] * u, sys.k[1] * u];
            // coupling B: inject and run one ALM iteration
            let (x1, l1) = sys.iterate(f);
            s2.push(json!({"ratio": ratio, "k1_over_rho": lev,
                "A_force_rel_err": ((f[0]-ls[0]).abs().max((f[1]-ls[1]).abs()))/ls[1].abs(),
                "B_after_one_iter_force_rel_err": ((l1[0]-ls[0]).abs().max((l1[1]-ls[1]).abs()))/ls[1].abs(),
                "B_after_one_iter_x_rel_err": (x1 - xs).abs() / xs, "factorizations": 1}));
        }
    }
    summary.push(json!({"regime": "S2", "note": "quasi-static linear scalar: S2 is S0; carries no cost/consistency information", "results": s2}));

    // ---------- E1-dyn: dynamic policy ρ = μ = m/h², step load from rest ----------
    let h = 1.0 / 480.0;
    let mut csv = Csv::new(EXP, "dyn", &["k1_over_mu", "m_iters", "step", "t", "x_exact", "x_s1", "split_err_s1", "onestep_split_err"]);
    let mut dyn_sum = Vec::new();
    for &lev in &[10.0, 1000.0] {
        for &mm in &[1u32, 8, 16] {
            let k1 = 1.0e8;
            let mu = k1 / lev;
            let mass = mu * h * h;
            let mk = |x: f64, v: f64, pp: f64| Sys { k: [k1, 2.0 * k1], rho: [mu, mu], p: pp, mu, xhat: x + h * v };
            let (mut xe, mut ve) = (0.0f64, 0.0f64);
            let (mut xs1, mut vs1) = (0.0f64, 0.0f64);
            let mut lam = [0.0, 0.0];
            let sys0 = mk(0.0, 0.0, p);
            let rad = spectral_radius(sys0.mat());
            let mut worst_late = 0.0f64;
            for n in 1..=480u32 {
                // exact IE trajectory
                let se = mk(xe, ve, p);
                let (xn, _) = se.exact();
                ve = (xn - xe) / h;
                xe = xn;
                // S1 trajectory
                let ss = mk(xs1, vs1, p);
                let (x_ref_step, l_ref_step) = ss.exact();
                let mut x = xs1;
                for _ in 0..mm {
                    let r = ss.iterate(lam);
                    x = r.0;
                    lam = r.1;
                }
                vs1 = (x - xs1) / h;
                xs1 = x;
                let _ = x_ref_step;
                let tot = lam[0] + lam[1];
                let split = if tot.abs() > 0.0 { (lam[0] / tot - 1.0 / 3.0).abs() / (1.0 / 3.0) } else { 0.0 };
                let one = ((lam[0] - l_ref_step[0]).abs().max((lam[1] - l_ref_step[1]).abs())) / l_ref_step[1].abs().max(1e-300);
                if n > 240 {
                    worst_late = worst_late.max(one);
                }
                if n <= 48 || n % 8 == 0 {
                    csv.row(&[lev, mm as f64, n as f64, n as f64 * h, xe, xs1, split, one]);
                }
            }
            dyn_sum.push(json!({"k1_over_mu": lev, "m": mm, "mass": mass, "spectral_radius_M_mu": rad,
                "worst_onestep_force_err_t_gt_0.5s": worst_late}));
        }
    }
    summary.push(json!({"regime": "E1-dyn", "h": h, "results": dyn_sum}));

    let out = json!({
        "meta": meta(EXP, "all", None, "exact scalar primal minimisation, then simultaneous dual update; multipliers frozen during primal",
            json!({"P_N": p, "rho0_N_per_m": rho0, "ratios_k2_over_k1": ratios, "k1_over_rho_levels": levels,
                   "assumption": "k_i/rho levels apply to k1; k2 = ratio*k1"})),
        "summary": summary,
        "ranges": experiments::ranges_json(),
    });
    write_json(EXP, "summary", &out);
    println!("{}", serde_json::to_string_pretty(&out["summary"]).unwrap());
}
