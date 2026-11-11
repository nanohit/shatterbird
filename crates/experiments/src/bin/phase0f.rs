//! Dynamic ports, rigid aggregation, and a bounded fracture scheduler.
//! Research controls write new results; the Phase 0d/e records are immutable.
use experiments::{meta, write_json};
use serde_json::{json, Value};
#[path = "../research/budget.rs"]
mod budget;
#[path = "../research/freeze.rs"]
mod freeze;
#[path = "../research/ports.rs"]
mod ports;
#[path = "../research/wall_ports.rs"]
mod wall_ports;
fn save(name: &str, data: Value) {
    println!("{name}: {}", serde_json::to_string(&data).unwrap());
    write_json(
        "phase0f",
        name,
        &json!({"meta":meta("phase0f",name,None,"CPU f64 research controls; no GPU timing claim",json!({})),"data":data}),
    );
}
fn legacy_wall() {
    use experiments::engine::{
        scenes::wall_impact,
        world::{run, Open0, Params, World},
    };
    let sc = wall_impact(1, 0.04, 8., 1., 1.5, 50.);
    let mut rows = vec![];
    for mode in [
        "fine",
        "multirate",
        "multirate_dashpot",
        "multirate_modes12",
    ] {
        let mut p = Params::new(1. / 240., 0.2);
        p.rereduce = false;
        if mode == "fine" {
            p.open0 = Open0::All;
            p.lp = 1e9;
        }
        p.absorbing = mode == "multirate_dashpot";
        if mode == "multirate_modes12" {
            p.nvib = 12;
        }
        let (tr, _) = run(World::new(sc.clone(), p), mode, 0.03, 0.0001);
        experiments::write_json("phase0f", &format!("legacy_wall_{mode}"), &tr);
        let value = json!({"mode":mode,"completed":tr.failed_at.is_none(),"failed_at":tr.failed_at,"cells":tr.cells,"D":tr.d_end,"rebound":tr.v_impactor,"work":tr.work_total,"startup_work":tr.startup_work,"mean_fine":tr.mean_e_frac,"max_fine":tr.max_e_cells,"wall_s":tr.wall_s,"energy_end_minus_initial":tr.e_end-tr.e0,"ledger":tr.ledger});
        eprintln!("legacy_wall {value}");
        rows.push(value);
        save("legacy_wall_progress", json!({"rows":rows}));
    }
    save(
        "legacy_wall",
        json!({"rows":rows,"scope":"Original S3b geometry, dmin=.04m, current legacy E/R World coupler. No claim that its current tolerances/automatic promotion reproduce the historical 3.3x development run."}),
    );
}
fn legacy_relaxed() {
    use experiments::engine::{
        scenes::wall_impact,
        world::{run, Params, World},
    };
    let sc = wall_impact(1, 0.04, 8., 1., 1.5, 50.);
    let mut rows = vec![];
    for absorbing in [false, true] {
        let mut p = Params::new(1. / 240., 0.2);
        p.rereduce = false;
        p.coupling_ftol = 1e9;
        p.absorbing = absorbing;
        let (tr, _) = run(
            World::new(sc.clone(), p),
            "disabled force residual gate",
            0.03,
            0.0001,
        );
        let value = json!({"absorbing":absorbing,"completed":tr.failed_at.is_none(),"failed_at":tr.failed_at,"D":tr.d_end,"rebound":tr.v_impactor,"work":tr.work_total,"mean_fine":tr.mean_e_frac,"max_fine":tr.max_e_cells,"wall_s":tr.wall_s,"energy_end_minus_initial":tr.e_end-tr.e0,"ledger":tr.ledger});
        experiments::write_json(
            "phase0f",
            if absorbing {
                "legacy_relaxed_dashpot"
            } else {
                "legacy_relaxed_plain"
            },
            &tr,
        );
        eprintln!("legacy_relaxed {value}");
        rows.push(value);
    }
    save(
        "legacy_relaxed",
        json!({"rows":rows,"scope":"Original geometry. Explicit ablation disables the interface FORCE residual gate only (ftol=1e9), emulating the ignored check of the historical prototype. Other coupling tolerances unchanged. Not an accepted solver configuration."}),
    );
}
fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("ports") => save("ports", ports::experiment()),
        Some("budget") => save("budget", budget::experiment()),
        Some("freeze") => save("freeze", freeze::experiment()),
        Some("freeze_refinement") => save("freeze_refinement", freeze::refinement()),
        Some("freeze_quiet") => save("freeze_quiet", freeze::quiet_window()),
        Some("wall") => save("wall_ports", wall_ports::experiment()),
        Some("wall_original") => save("wall_original", wall_ports::original_geometry()),
        Some("legacy_wall") => legacy_wall(),
        Some("legacy_relaxed") => legacy_relaxed(),
        _ => panic!("ports|wall|legacy_wall|freeze|freeze_refinement|freeze_quiet|budget"),
    }
}
