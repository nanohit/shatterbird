//! Causal discrete Dirichlet-to-Neumann control. The eliminated remote chain
//! uses exactly the central-difference recurrence of the unreduced chain.
//! A continuum waveguide is a different (nondispersive) impedance, even when
//! lossless. This deliberately distinguishes passivity from port accuracy.
use experiments::engine::patch::{chol, chol_solve};
use model::{LawKind, PointLaw};
use serde_json::{json, Value};
const N: usize = 64;
const A: usize = 8;
fn initial_velocity(i: usize, width: f64, amp: f64) -> f64 {
    // Trace and remote domain start at rest: the identical initial state belongs
    // to every CMS space. Initial projection error cannot pollute this control.
    if i < A - 1 {
        amp * (-0.5 * ((i as f64 - 2.) / width).powi(2)).exp()
    } else {
        0.
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn solve(l: &[f64], b: &[f64]) -> Vec<f64> {
    let mut x = b.to_vec();
    chol_solve(l, b.len(), &mut x);
    x
}
fn law() -> PointLaw {
    PointLaw {
        kind: LawKind::Damage,
        kn: 1.,
        kt: 1.,
        gamma: 1.,
        delta0: 0.12,
        wc: 0.6,
        d_max: 1.,
    }
}
fn kernel(h: f64, steps: usize) -> Vec<f64> {
    let r = N - A;
    let mut g = vec![0.; steps + 1];
    for k in 1..=r {
        let p = std::f64::consts::PI * k as f64 / (r + 1) as f64;
        let theta = (1. - 0.5 * h * h * (2. - 2. * p.cos())).acos();
        let w = 2. * p.sin().powi(2) / (r + 1) as f64;
        for (l, x) in g.iter_mut().enumerate().skip(1) {
            *x += h * h * w * (l as f64 * theta).sin() / theta.sin();
        }
    }
    g
}
#[derive(Clone)]
struct Trace {
    q: Vec<Vec<f64>>,
    f: Vec<f64>,
    d: Vec<f64>,
    work: f64,
    energy_balance: f64,
    stored: f64,
}
// Internal active forces, without the remote port force. The left end is fixed.
fn force(q: &[f64], kc: &mut f64, damage: bool) -> Vec<f64> {
    let n = q.len();
    let mut f = vec![0.; n];
    f[0] -= q[0];
    for i in 1..n {
        let jump = q[i] - q[i - 1];
        let ff = if damage && i == 4 {
            let e = law().eval(jump, 0., *kc);
            *kc = e.kappa;
            e.fnorm
        } else {
            jump
        };
        f[i - 1] += ff;
        f[i] -= ff;
    }
    f
}
fn exact_or_port(kind: &str, width: f64, amp: f64, damage: bool, h: f64, end: f64) -> Trace {
    let steps = (end / h).round() as usize;
    let full = kind == "fine";
    let n = if full { N } else { A };
    let mut q = vec![0.; n];
    let mut prev: Vec<f64> = (0..n)
        .map(|i| -h * initial_velocity(i, width, amp))
        .collect();
    let e0 = 0.5 * dot(&prev, &prev) / (h * h);
    let mut kc = 0.;
    let g = kernel(h, steps);
    let mut history = Vec::new();
    // Incoming/outgoing travelling VELOCITY waves. Fixed far end: reflection -1.
    let delay = (2. * (N - A + 1) as f64 / h).round() as usize;
    let mut waves = vec![0.; steps + delay + 1];
    let mut balance = 0.;
    let mut wave_energy = 0.;
    let mut out = Trace {
        q: Vec::new(),
        f: Vec::new(),
        d: Vec::new(),
        work: 0.,
        energy_balance: 0.,
        stored: 0.,
    };
    for s in 0..steps {
        let mut f = force(&q, &mut kc, damage);
        let mut next = vec![0.; n];
        let port = match kind {
            "fine" => {
                f[n - 1] -= q[n - 1];
                q[A] - q[A - 1]
            }
            "static" => -q[A - 1] / (N - A + 1) as f64,
            "irf" => {
                out.work += 2. * s as f64;
                history
                    .iter()
                    .enumerate()
                    .map(|(j, x)| g[s - j] * x)
                    .sum::<f64>()
                    - q[A - 1]
            }
            "dashpot" | "waveguide" => 0.,
            _ => panic!("unknown port"),
        };
        for i in 0..n {
            next[i] = 2. * q[i] - prev[i] + h * h * f[i];
        }
        let mut actual_port = port;
        if kind == "dashpot" || kind == "waveguide" {
            let incoming = if kind == "waveguide" { waves[s] } else { 0. };
            // Implicit centred velocity avoids an explicit-dashpot stability bug.
            let i = n - 1;
            next[i] = (2. * q[i] - (1. - h / 2.) * prev[i] + h * h * f[i] + 2. * h * h * incoming)
                / (1. + h / 2.);
            let v = (next[i] - prev[i]) / (2. * h);
            let outgoing = v - incoming;
            actual_port = 2. * incoming - v;
            let flow = h * (outgoing * outgoing - incoming * incoming);
            if kind == "waveguide" {
                waves[s + delay] = -outgoing;
                wave_energy += flow;
            } else {
                balance += flow;
            }
        } else if !full {
            next[n - 1] += h * h * port;
        }
        out.q.push(q[..A].to_vec());
        out.f.push(actual_port);
        out.d.push(if damage { law().dissipation(kc) } else { 0. });
        history.push(q[A - 1]);
        prev = q;
        q = next;
        out.work += 12. * n as f64;
    }
    // Leapfrog's exact quadratic invariant (linear runs only), not naive KE+U.
    let vv: Vec<_> = q.iter().zip(&prev).map(|(x, p)| (x - p) / h).collect();
    let mut fp = force(&prev, &mut 0., false);
    if full {
        fp[n - 1] -= prev[n - 1];
    } else if kind == "static" {
        fp[n - 1] -= prev[n - 1] / (N - A + 1) as f64;
    }
    let inv = 0.5 * dot(&vv, &vv) - 0.5 * dot(&q, &fp);
    out.energy_balance = (inv + balance + wave_energy - e0) / e0;
    out.stored = wave_energy / e0;
    out
}
fn cms(modes: usize, width: f64, amp: f64, damage: bool, h: f64, end: f64) -> Trace {
    let r = N - A;
    let d = A + modes;
    let mut t = vec![vec![0.; N]; d];
    for i in 0..A {
        t[i][i] = 1.;
    }
    for j in 0..r {
        t[A - 1][A + j] = (r - j) as f64 / (r + 1) as f64;
    }
    for k in 0..modes {
        for j in 0..r {
            t[A + k][A + j] =
                (std::f64::consts::PI * (k + 1) as f64 * (j + 1) as f64 / (r + 1) as f64).sin()
                    * (2. / (r + 1) as f64).sqrt();
        }
    }
    let mass: Vec<_> = (0..d)
        .flat_map(|i| {
            let t = &t;
            (0..d).map(move |j| dot(&t[i], &t[j]))
        })
        .collect();
    let l = chol(&mass, d).unwrap();
    // Remote K including the interface spring. Physical active nonlinear springs
    // are evaluated separately. This avoids duplicating the interface stiffness.
    let mut k = vec![0.; d * d];
    for e in A..=N {
        let v: Vec<_> = t
            .iter()
            .map(|c| if e == N { -c[N - 1] } else { c[e] - c[e - 1] })
            .collect();
        for i in 0..d {
            for j in 0..d {
                k[i * d + j] += v[i] * v[j];
            }
        }
    }
    let mut q = vec![0.; d];
    let mut v: Vec<_> = (0..d)
        .map(|i| {
            (0..A)
                .map(|j| t[i][j] * initial_velocity(j, width, amp))
                .sum::<f64>()
        })
        .collect();
    v = solve(&l, &v);
    let mut kc = 0.;
    let mut out = Trace {
        q: vec![],
        f: vec![],
        d: vec![],
        work: 0.,
        energy_balance: 0.,
        stored: 0.,
    };
    let calc = |q: &[f64], kc: &mut f64| {
        let mut ff = vec![0.; d];
        let af = force(&q[..A], kc, damage);
        ff[..A].copy_from_slice(&af);
        for i in 0..d {
            ff[i] -= dot(&k[i * d..(i + 1) * d], q);
        }
        solve(&l, &ff)
    };
    let mut acc = calc(&q, &mut kc);
    for _ in 0..(end / h).round() as usize {
        out.q.push(q[..A].to_vec());
        out.f
            .push(t.iter().zip(&q).map(|(c, q)| c[A] * q).sum::<f64>() - q[A - 1]);
        out.d.push(if damage { law().dissipation(kc) } else { 0. });
        for i in 0..d {
            v[i] += 0.5 * h * acc[i];
            q[i] += h * v[i];
        }
        acc = calc(&q, &mut kc);
        for i in 0..d {
            v[i] += 0.5 * h * acc[i];
        }
        out.work += 6. * (d * d) as f64 + 12. * A as f64;
    }
    out
}
fn error(x: &Trace, r: &Trace, end: usize) -> f64 {
    let (mut num, mut den) = (0., 0.);
    for s in 0..end {
        for i in 0..A {
            num += (x.q[s][i] - r.q[s][i]).powi(2);
            den += r.q[s][i].powi(2);
        }
    }
    (num / den.max(1e-30)).sqrt()
}
pub fn experiment() -> Value {
    let mut rows = vec![];
    for h in [0.2, 0.1] {
        for width in [0.65, 2.5] {
            for damage in [false, true] {
                for amp in if damage {
                    vec![0.15, 0.22, 0.3]
                } else {
                    vec![0.03]
                } {
                    let end = 160.;
                    let r = exact_or_port("fine", width, amp, damage, h, end);
                    for name in [
                        "static",
                        "dashpot",
                        "waveguide",
                        "irf",
                        "cms0",
                        "cms4",
                        "cms16",
                        "cms48",
                    ] {
                        let x = if let Some(k) = name.strip_prefix("cms") {
                            cms(k.parse().unwrap(), width, amp, damage, h, end)
                        } else {
                            exact_or_port(name, width, amp, damage, h, end)
                        };
                        let early = (40. / h) as usize;
                        let d = *x.d.last().unwrap();
                        let dr = *r.d.last().unwrap();
                        let fnumer =
                            x.f.iter()
                                .zip(&r.f)
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>();
                        let fden = r.f.iter().map(|b| b * b).sum::<f64>();
                        rows.push(json!({"port":name,"h":h,"width_cells":width,"amplitude":amp,"damage":damage,"early_local_q_l2":error(&x,&r,early),"full_local_q_l2":error(&x,&r,x.q.len()),"port_force_time_l2":(fnumer/fden.max(1e-30)).sqrt(),"port_impulse_error":h*x.f.iter().zip(&r.f).map(|(a,b)|a-b).sum::<f64>(),"D":d,"reference_D":dr,"false_positive_damage":dr==0.&&d>1e-10,"D_relative_error":if dr>0.{Some((d-dr)/dr)}else{None},"linear_energy_balance":if !damage&&["fine","static","dashpot","waveguide"].contains(&name){Some(x.energy_balance)}else{None},"wave_storage_over_initial_energy":x.stored,"work":x.work,"fine_work":r.work,"work_gain":r.work/x.work}));
                    }
                }
            }
        }
    }
    json!({"scope":"same-step nonlinear active chain, finite linear remote chain; not a 2D fracture validation. IRF exactly eliminates the discrete remote recurrence. Waveguide replaces the remote impedance rather than being added in parallel. CMS uses consistent mass. Work counts the implemented dense algebra / full-history convolution.","N":N,"active":A,"T":160.,"continuum_return_time":2*(N-A+1),"rows":rows})
}
#[test]
fn irf_matches_discrete_chain_even_with_damage() {
    for damage in [false, true] {
        let r = exact_or_port("fine", 0.65, 0.3, damage, 0.2, 160.);
        let x = exact_or_port("irf", 0.65, 0.3, damage, 0.2, 160.);
        assert!(error(&x, &r, r.q.len()) < 1e-10);
        assert!((x.d.last().unwrap() - r.d.last().unwrap()).abs() < 1e-12);
    }
}
#[test]
fn full_cms_reproduces_fine_discrete_chain() {
    let r = exact_or_port("fine", 0.65, 0.3, false, 0.2, 80.);
    let x = cms(N - A, 0.65, 0.3, false, 0.2, 80.);
    assert!(error(&x, &r, r.q.len()) < 1e-10);
}
#[test]
fn scattering_port_is_passive() {
    for name in ["static", "dashpot", "waveguide"] {
        let x = exact_or_port(name, 0.65, 0.03, false, 0.2, 160.);
        assert!(
            x.energy_balance.abs() < 1e-10,
            "{name} {}",
            x.energy_balance
        );
    }
}
