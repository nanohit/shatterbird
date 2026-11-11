//! A bounded scheduler plus a constitutive control. Not a coupled simulation and
//! emphatically not a claim to a GPU frame-time guarantee.
use model::{LawKind, PointLaw};
use serde_json::{json, Value};
pub fn experiment() -> Value {
    let mut queue = Vec::new();
    for duration in [1usize, 5, 20] {
        let jobs = 10_000usize;
        let cap = 2048;
        let mut start = vec![usize::MAX; jobs];
        let mut finish = vec![0usize; jobs];
        let mut max_active = 0;
        for frame in 0..200 {
            let mut active = (0..jobs)
                .filter(|&i| start[i] != usize::MAX && finish[i] > frame)
                .count();
            // All jobs have the same overload: canonical ID breaks the tie.
            for i in 0..jobs {
                if start[i] == usize::MAX && active < cap {
                    start[i] = frame;
                    finish[i] = frame + duration;
                    active += 1;
                }
            }
            max_active = max_active.max(active);
            if start.iter().all(|&s| s != usize::MAX) {
                break;
            }
        }
        queue.push(json!({"jobs":jobs,"cap":cap,"job_duration_frames":duration,"peak_active":max_active,"max_wait_frames":start.iter().max(),"max_wait_ms":*start.iter().max().unwrap() as f64*1000./60.,"mean_wait_frames":start.iter().sum::<usize>() as f64/jobs as f64,"last_completion_frame":finish.iter().max()}));
    }
    let law = PointLaw {
        kind: LawKind::Damage,
        kn: 1e9,
        kt: 1e9,
        gamma: 1.,
        delta0: 3e-5,
        wc: 2e-4,
        d_max: 1.,
    };
    let mut material = Vec::new();
    for wait_frames in [0, 1, 4, 20] {
        for speed in [0.001, 0.01] {
            let dt = 1e-5;
            let t_end = 0.12;
            let onset = law.delta0 / speed;
            let eligible = onset + wait_frames as f64 / 60.;
            let (
                mut kc,
                mut kr,
                mut work,
                mut wr,
                mut last_f,
                mut last_r,
                mut maxf,
                mut maxr,
                mut delta,
            ) = (0.0f64, 0., 0., 0., 0., 0., 0.0f64, 0.0f64, 0.);
            for step in 0..=(t_end / dt) as usize {
                let t = step as f64 * dt;
                let d = (speed * t).min(law.wc * 1.2);
                let er = law.eval(d, 0., kr);
                kr = er.kappa;
                // Waiting cells keep the previous damage; their loading is still
                // carried. Admission catches up history, with no hidden energy reset.
                if t >= eligible {
                    kc = kc.max(d);
                }
                let f = (1. - law.damage(kc)) * law.kn * d;
                work += 0.5 * (last_f + f) * (d - delta);
                wr += 0.5 * (last_r + er.fnorm) * (d - delta);
                maxf = maxf.max(f);
                maxr = maxr.max(er.fnorm);
                last_f = f;
                last_r = er.fnorm;
                delta = d;
            }
            let psi = 0.5 * (1. - law.damage(kc)) * law.kn * delta * delta;
            let d = law.dissipation(kc);
            material.push(json!({"wait_frames":wait_frames,"opening_speed":speed,"peak_force_ratio":maxf/maxr,"work":work,"reference_work":wr,"work_ratio":work/wr,"D":d,"elastic_end":psi,"unbooked_work":work-d-psi,"unbooked_fraction_of_GcA":(work-d-psi)/law.full_dissipation()}));
        }
    }
    let mut split = vec![];
    for ratio in [1., 1.25, 2., 4.] {
        let opening = ratio * law.delta0;
        let stored = 0.5 * law.kn * opening * opening;
        let required = law.full_dissipation();
        split.push(json!({"opening_over_onset":ratio,"undamaged_elastic_energy_before":stored,"required_fracture_energy":required,"energy_deficit":(required-stored).max(0.),"surplus_requires_kinetic_transfer_or_explicit_loss":(stored-required).max(0.),"fraction_of_Gc_paid_by_deleting_spring":stored/required}));
    }
    // Same net impulse, different damage risk: a force couple has P=0 and M!=0.
    let bundle = ["all", "hold", "fast_split"]
        .iter()
        .map(|mode| bundle(mode))
        .collect::<Vec<_>>();
    json!({"scope":"canonical capacity scheduler, prescribed-opening PointLaw control, an energy feasibility check, and a dynamically coupled 10,000-link bundle. No RBSN contact workload or GPU timing included.","queue":queue,"constitutive_hold":material,"instant_split":split,"coupled_bundle":bundle,"wake_counterexample":{"forces":[[0.,1000.],[0.,-1000.]],"arms_x":[-1.,1.],"net_force":0.,"torque":-2000.,"claim":"net impulse alone cannot safely wake a frozen component"},"zero_wrench_wake_counterexample":{"forces":[[1000.,0.],[-1000.,0.]],"arms_x":[-1.,1.],"net_force":[0.,0.],"torque":0.,"claim":"even net force AND net torque miss a self-balanced compressive/tensile load; retain spatial boundary tractions"},"fast_split_requirement":"Compare stored energy release with Gc*A. Preserve linear/angular momentum and book any excess as an explicitly named loss. A threshold crossing by itself does not close this energy budget."})
}
fn bundle(mode: &str) -> Value {
    // One moving platen shares load among 10,000 cohesive links. Thus delays
    // feed back into motion and load redistribution, unlike prescribed opening.
    let n = 10_000usize;
    let cap = 2048usize;
    let dt = 0.0001;
    let end = 0.12;
    let mass = 10. * n as f64;
    let laws: Vec<_> = (0..n)
        .map(|i| PointLaw {
            kind: LawKind::Damage,
            kn: 1e6,
            kt: 1e6,
            gamma: 1.,
            delta0: 0.001 * (0.9 + 0.2 * ((i * 7919) % n) as f64 / n as f64),
            wc: 0.004,
            d_max: 1.,
        })
        .collect();
    let mut kc = vec![0.; n];
    let mut admitted = vec![false; n];
    let mut born = vec![None; n];
    let mut start = vec![None; n];
    let mut cheap = vec![false; n];
    let mut fallback = vec![false; n];
    let (mut q, mut v, mut acc, mut d, mut w, mut peak_force, mut max_active, mut max_wait) =
        (0.0f64, 0.0f64, 0., 0., 0., 0.0f64, 0usize, 0usize);
    let mut trace = vec![];
    let mut frame_prev = usize::MAX;
    let ext = |t: f64| 1.2e3 * n as f64 * (t / 0.02).min(1.);
    for step in 0..(end / dt) as usize {
        let t = step as f64 * dt;
        let frame = (t * 60.).floor() as usize;
        let old = q;
        v += 0.5 * dt * acc;
        q += dt * v;
        if frame != frame_prev {
            for i in 0..n {
                if kc[i] >= laws[i].wc {
                    admitted[i] = false;
                }
                if q >= laws[i].delta0 && born[i].is_none() {
                    born[i] = Some(frame);
                }
            }
            let active = admitted.iter().filter(|&&a| a).count();
            let mut pending: Vec<_> = (0..n)
                .filter(|&i| !admitted[i] && kc[i] < laws[i].wc && q >= laws[i].delta0)
                .collect();
            pending.sort_by(|&i, &j| {
                (q / laws[j].delta0)
                    .total_cmp(&(q / laws[i].delta0))
                    .then(i.cmp(&j))
            });
            let available = if mode == "all" {
                n
            } else {
                cap.saturating_sub(active)
            };
            for (rank, i) in pending.into_iter().enumerate() {
                if rank < available {
                    admitted[i] = true;
                    if start[i].is_none() {
                        start[i] = Some(frame);
                        max_wait = max_wait.max(frame - born[i].unwrap());
                    }
                } else if mode == "fast_split" {
                    fallback[i] = true;
                }
            }
            max_active = max_active.max(admitted.iter().filter(|&&a| a).count());
            frame_prev = frame;
        }
        let mut force = 0.;
        let mut u = 0.;
        for i in 0..n {
            let law = &laws[i];
            let oldk = kc[i];
            if mode == "all" || admitted[i] {
                let e = law.eval(q, 0., oldk);
                kc[i] = e.kappa;
                d += law.dissipation(kc[i]) - law.dissipation(oldk);
                force += e.fnorm;
                u += e.psi;
            } else if fallback[i] && kc[i] < law.wc {
                // Explicitly energy-limited brittle fallback: it pays only the
                // energy available NOW. It is passive, but changes effective Gc.
                d += 0.5 * (1. - law.damage(oldk)) * law.kn * q * q;
                kc[i] = law.wc;
                cheap[i] = true;
            } else {
                let om = 1. - law.damage(oldk);
                force += law.kn * (q.min(0.) + om * q.max(0.));
                u += 0.5 * law.kn * (q.min(0.).powi(2) + om * q.max(0.).powi(2));
            }
        }
        peak_force = peak_force.max(force);
        w += 0.5 * (ext(t) + ext(t + dt)) * (q - old);
        acc = (ext(t + dt) - force) / mass;
        v += 0.5 * dt * acc;
        if step % 50 == 0 {
            trace.push(json!({"t":t+dt,"q":q,"D":d,"force":force,"balance":0.5*mass*v*v+u+d-w}));
        }
    }
    let u = laws
        .iter()
        .zip(&kc)
        .map(|(l, k)| 0.5 * (1. - l.damage(*k)) * l.kn * q * q)
        .sum::<f64>();
    json!({"mode":mode,"links":n,"cap":cap,"max_active":if mode=="all"{n}else{max_active},"max_admitted_wait_frames":max_wait,"still_waiting":(0..n).filter(|&i|born[i].is_some()&&start[i].is_none()&&!cheap[i]).count(),"cheap_splits":cheap.iter().filter(|&&x|x).count(),"D":d,"Gc_capacity":laws.iter().map(|l|l.full_dissipation()).sum::<f64>(),"final_q":q,"final_v":v,"peak_force":peak_force,"energy_balance":0.5*mass*v*v+u+d-w,"trace":trace,"note":"Same platen mass and link laws. Frame-based admission. The fast fallback is declared energy-limited brittle fracture, not an equivalent cohesive update. CPU f64 priority is a research control, not cross-GPU determinism."})
}
#[test]
fn queue_lower_bound() {
    let x = experiment();
    assert_eq!(x["queue"][0]["max_wait_frames"], 4);
    assert_eq!(x["queue"][1]["max_wait_frames"], 20);
}
