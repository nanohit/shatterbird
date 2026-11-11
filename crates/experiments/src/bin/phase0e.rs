//! Remediation audit. Writes fresh results only; historical Phase 0d is immutable.
//! cargo run --release -p experiments --bin phase0e -- regression|collapse|scale|pulse|numeric3d
use experiments::{engine::{scenes::*, world::*}, meta, write_json};
use serde_json::{json, Value};

const OUT: &str = "phase0e";
fn engine(lp: f64) -> Params {
    let mut p=Params::new(1.0/240.0,lp); p.implicit=true; p.lte_atol=0.1; p.lte_rtol=0.02; p
}
fn reference() -> Params { let mut p=Params::new(0.001,1e9);p.open0=Open0::All;p.rereduce=false;p }
fn save(name: &str, data: Value) {
    write_json(OUT,name,&json!({"meta":meta(OUT,name,None,"CPU f64; work is an operation estimate, not GPU timing",json!({})),"data":data}));
}
fn brief(tr:&Traj)->Value {
    let mut f:Vec<f64>=tr.frames.iter().map(|f|f.work).collect();f.sort_by(f64::total_cmp);
    let peak=f.last().copied().unwrap_or(0.0);
    json!({"completed":tr.failed_at.is_none(),"failed_at":tr.failed_at,"D":tr.d_end,"fragment":tr.fragment,"fragments":tr.frags,
        "wall_s":tr.wall_s,"work":tr.work_total,"startup_work":tr.startup_work,"mean_active":tr.mean_e_frac,
        "peak_frame_work":peak,"p95_frame_work":f.get((f.len() as f64*0.95).floor() as usize).copied(),"peak_frame_wall_s":tr.frames.iter().map(|f|f.wall_s).fold(0.0,f64::max),
        "ledger":tr.ledger,"energy_rise":tr.e_max_rise,"ke_max":tr.ke_max,"steps":tr.work.macro_steps,"rejected":tr.work.rejected,"roundoff_accepts":tr.work.roundoff_accepts})
}
fn run_save(sc:Scene,p:Params,label:&str,end:f64)->Traj {
    let (tr,_)=run(World::new(sc,p),label,end,0.001);
    write_json(OUT,label,&tr);println!("{label}: {}",brief(&tr));tr
}
fn regression() {
    let mut rows=Vec::new();
    for seed in [1,2] {
        let sc=cantilever(seed,0.04,true,0.005,1.25*0.9046875);
        let r=run_save(sc.clone(),reference(),&format!("s1_seed{seed}_reference"),0.3);
        let e=run_save(sc,engine(0.2),&format!("s1_seed{seed}_engine"),0.3);
        rows.push(json!({"seed":seed,"reference":brief(&r),"engine":brief(&e),"valid_comparison":r.failed_at.is_none()&&e.failed_at.is_none()}));
    }
    save("regression",json!(rows));
}
fn s1variants() {
    let mut rows=Vec::new();
    for mode in ["base","no_merge","fine_explicit","strict_lte"] {
        let sc=cantilever(1,0.04,true,0.005,1.25*0.9046875);
        let (_,mut wd)=run(World::new(sc,engine(0.2)),"prefix",0.03,0.001);
        match mode {
            "no_merge"=>wd.p.rereduce=false,
            "fine_explicit"=>wd.use_fine_explicit(),
            "strict_lte"=>{wd.p.lte_atol=0.02;wd.p.lte_rtol=0.005;},_=>{}
        }
        let label=format!("s1_seed1_after30ms_{mode}");
        let (tr,wd)=run(wd,&label,0.3,0.001);write_json(OUT,&label,&tr);
        let d:Vec<_>=wd.sc.m.facets.iter().flat_map(|f|f.pts.iter().map(|p|p.law.dissipation(p.kappa_c))).collect();
        println!("{label}: {}",brief(&tr));
        rows.push(json!({"mode":mode,"result":brief(&tr),"dissipation_by_point":d}));
    }
    save("s1variants",json!(rows));
}
fn collapse() {
    let mut rows=Vec::new();
    // Same original scene, including its registered support-release timing.
    // h, h/2 and mu=0 isolate contact time error from frictional loss.
    for mu in [0.0,0.5] {
        let sc=collapse_beam(1,0.06,6.0,0.3,1.0,mu,0.5);
        for (tag,mut p) in [("reference",reference()),("reference_half",reference()),("engine",engine(0.2))] {
            if tag=="reference_half" {p.he_scale=0.5;}
            let label=format!("collapse_mu{mu}_{tag}");
            let tr=run_save(sc.clone(),p,&label,1.2);
            rows.push(json!({"mu":mu,"mode":tag,"dmin":0.06,"run":brief(&tr)}));
        }
    }
    save("collapse",json!(rows));
}
fn scale() {
    let mut rows=Vec::new();
    // Fixed material, cross-section, gravity and physical duration; no T~L^2.
    for len in [2.0,4.0,8.0] {
        let sc=cantilever_len(1,0.06,true,0.005,0.5,len);
        for (tag,p) in [("reference",reference()),("engine",engine(0.2))] {
            let tr=run_save(sc.clone(),p,&format!("scale_L{len}_{tag}"),0.12);
            rows.push(json!({"L":len,"cells":tr.cells,"T":0.12,"run":brief(&tr)}));
        }
    }
    save("scale",json!(rows));
}
fn pulse_engine() {
    use geometry::V2;
    use model::{Model,Material,LawOptions};
    let tess=experiments::rect_tess(V2::ZERO,V2::new(2.0,0.6),0.1,4);
    let m=Model::from_tessellation(&tess,Material::concrete(0.2),LawOptions::elastic(),&[]);
    let nc=m.n_cells();let nb=m.bodies.len();
    let vi:Vec<_>=m.bodies.iter().map(|b|[0.0,0.01*(-0.5*((b.x0.x-0.15)/0.07).powi(2)).exp(),0.0]).collect();
    let probe:Vec<_>=(0..nc).filter(|&i|m.bodies[i].x0.x<0.4).collect();
    let sc=Scene{m,name:"elastic pulse control".into(),dmin:0.1,len:2.0,depth:0.6,releases:vec![],removals:vec![],clamp_cells:vec![],tip_cells:probe.clone(),impactor:None,v_init:vi,static_presolve:false,ground:None,e_scale:1.0};
    let mut p=reference();let wr=World::new(sc.clone(),p.clone());let he=wr.he;let end=256.0*he;
    p.h_e_fixed=Some(he/4.0);p.h_macro=4.0*he;
    let (rf,rw)=run(World::new(sc.clone(),p),"pulse reference",end,4.0*he);
    write_json(OUT,"pulse_engine_reference",&rf);
    let mut rows=Vec::new();
    for modes in [0,12] {for ratio in [1,8,32] {for refinement in [1,4] {
        let h=he/refinement as f64;
        let mut p=Params::new(ratio as f64*h,0.6);p.open0=Open0::XLess(0.15);p.rereduce=false;p.nvib=modes;p.h_e_fixed=Some(h);p.max_coupling=8;
        let label=format!("pulse_engine_v{modes}_r{ratio}_h{refinement}");
        let (tr,wd)=run(World::new(sc.clone(),p),&label,end,4.0*he);write_json(OUT,&label,&tr);
        let (mut num,mut den)=(0.0,0.0);
        for &i in &probe {for k in 0..3 {num+=(wd.q[i][k]-rw.q[i][k]).powi(2);den+=(rw.q[i][k]-sc.m.q[i][k]).powi(2);}}
        let (mut fnumer,mut fden)=(0.0,0.0);
        for (fi,_) in sc.m.facets.iter().enumerate().filter(|(_,f)|f.pc.x<0.4) {for pt in 0..2 {
            let e=wd.sc.m.eval_point(&wd.q,&wd.sc.m.facets[fi],pt).1;let r=rw.sc.m.eval_point(&rw.q,&rw.sc.m.facets[fi],pt).1;
            fnumer+=(e.fnorm-r.fnorm).powi(2)+(e.ftan-r.ftan).powi(2);fden+=r.fnorm*r.fnorm+r.ftan*r.ftan;
        }}
        let valid=rf.failed_at.is_none()&&tr.failed_at.is_none();
        let (mut pn,mut pd)=(0.0,0.0);
        for row in &tr.rows {
            let j=rf.rows.partition_point(|r|r[0]<row[0]).min(rf.rows.len()-1);
            let ref_value=if j==0 {rf.rows[0][1]}else {let a=rf.rows[j-1];let b=rf.rows[j];a[1]+(b[1]-a[1])*(row[0]-a[0])/(b[0]-a[0])};
            pn+=(row[1]-ref_value).powi(2);pd+=ref_value*ref_value;
        }
        rows.push(json!({"nvib":modes,"requested_ratio":ratio,"refinement":refinement,"T":end,"h_fine":h,"probe_time_l2":valid.then(||(pn/pd.max(1e-30)).sqrt()),"local_final_displacement_l2":valid.then(||(num/den.max(1e-30)).sqrt()),"local_final_force_l2":valid.then(||(fnumer/fden.max(1e-30)).sqrt()),"reference":brief(&rf),"engine":brief(&tr),"body_slots":nb}));
    }}}
    save("pulse_engine",json!({"scope":"actual legacy E/R coupler; auto-promotion remains enabled and must be checked in mean_active","rows":rows}));
}

#[path="../research/pulse.rs"] mod pulse;
#[path="../research/numeric3d.rs"] mod numeric3d;
#[path="../research/pulse2d.rs"] mod pulse2d;
fn main(){match std::env::args().nth(1).as_deref(){
    Some("regression")=>regression(),Some("collapse")=>collapse(),Some("scale")=>scale(),
    Some("s1variants")=>s1variants(),
    Some("pulse")=>save("pulse",pulse::experiment()),Some("numeric3d")=>save("numeric3d",numeric3d::experiment()),
    Some("pulse2d")=>save("pulse2d",pulse2d::experiment()),
    Some("pulse_engine")=>pulse_engine(),
    Some("smoke")=>{run_save(cantilever(2,0.04,true,0.005,1.25*0.9046875),engine(0.2),"smoke",0.3);},
    Some("collapse_engine")=>{for mu in [0.0,0.5] {run_save(collapse_beam(1,0.06,6.0,0.3,1.0,mu,0.5),engine(0.2),&format!("collapse_hold_mu{mu}_engine"),1.2);}},
    Some("collapse_final")=>{for mu in [0.0,0.5] {let sc=collapse_beam(1,0.06,6.0,0.3,1.0,mu,0.5);for (tag,mut p) in [("reference",reference()),("reference_half",reference()),("engine",engine(0.2))] {if tag=="reference_half"{p.he_scale=0.5;}run_save(sc.clone(),p,&format!("collapse_passive_mu{mu}_{tag}"),1.2);}}},
    Some("collapse_refresh")=>{for mu in [0.0,0.5] {run_save(collapse_beam(1,0.06,6.0,0.3,1.0,mu,0.5),engine(0.2),&format!("collapse_passive_mu{mu}_engine"),1.2);}},
    _=>panic!("regression|collapse|scale|pulse|numeric3d"),
}}
