use super::*;

fn free_world(seed: u64) -> World {
    let mut sc = super::super::scenes::cantilever(seed, 0.12, false, 0.005, 1.0);
    for f in &mut sc.m.facets { if f.tag != 0 { f.active = false; } }
    sc.m.loads.fill([0.0; 3]);
    sc.m.q = sc.m.bodies.iter().map(|b| [b.x0.x, b.x0.y, 0.0]).collect();
    sc.releases.clear(); sc.removals.clear(); sc.static_presolve = false;
    let mut p = Params::new(1.0 / 240.0, 0.5);
    p.implicit = true; p.cost_switch = None; p.eps_rr = 0.5;
    World::new(sc, p)
}

fn angular(w: &World) -> f64 {
    w.sc.m.bodies.iter().enumerate().map(|(i,b)| b.mass * (w.q[i][0]*w.v[i][1]-w.q[i][1]*w.v[i][0]) + b.inertia*w.v[i][2]).sum()
}

#[test]
fn adjacent_merges_close_actual_energy_and_both_momenta() {
    for seed in 1..=4 {
        let mut w = free_world(seed);
        let pids: Vec<usize> = (0..w.patches.len()).collect();
        w.split(&pids, "test"); w.t = 0.1;
        for i in 0..w.sc.m.n_cells() {
            w.q[i][0] += 2e-7 * (i as f64 * 0.973).sin();
            w.q[i][1] += 2e-7 * (i as f64 * 0.531).cos();
            w.v[i] = [0.01 - 0.3*w.q[i][1] + 1e-4*(i as f64).sin(), 0.02 + 0.3*w.q[i][0] + 1e-4*(i as f64).cos(), 0.3];
        }
        let hist: Vec<_> = w.sc.m.facets.iter().map(|f| f.pts.map(|p| p.kappa_c)).collect();
        let p0 = momentum(&w); let l0 = angular(&w);
        let e = w.energies(); let e0 = e.0+e.1+e.2; let d0 = w.led.d_red;
        assert!(w.merge() > 1, "fixture must exercise adjacent candidates");
        let e = w.energies(); let e1 = e.0+e.1+e.2;
        assert!((e1-e0+w.led.d_red-d0).abs() < 1e-9, "seed {seed}");
        assert!((momentum(&w)-p0).norm() < 1e-9);
        assert!((angular(&w)-l0).abs() < 1e-9);
        assert_eq!(hist, w.sc.m.facets.iter().map(|f| f.pts.map(|p| p.kappa_c)).collect::<Vec<_>>());
    }
}

#[test]
fn one_surviving_cohesive_point_keeps_component_connected() {
    let mut w = free_world(1);
    // Keep one internal edge only. One integration point is broken, the other intact.
    let fi = w.sc.m.facets.iter().position(|f| f.tag == 0).unwrap();
    for (i,f) in w.sc.m.facets.iter_mut().enumerate() { f.active = i == fi; }
    let f = &mut w.sc.m.facets[fi];
    for p in &mut f.pts { p.law.kind = LawKind::Damage; }
    f.pts[0].kappa_c = f.pts[0].law.wc;
    let nc = w.sc.m.n_cells();
    assert_eq!(fragments(&w.sc.m, &w.q, &w.v, 0.0).len(), nc-1);
    w.sc.m.facets[fi].pts[1].kappa_c = w.sc.m.facets[fi].pts[1].law.wc;
    assert_eq!(fragments(&w.sc.m, &w.q, &w.v, 0.0).len(), nc);
}

#[test]
fn rotating_ground_contact_uses_polygon_surface() {
    let w = free_world(1);
    let cm = ContactModel::new(&w.sc.m, Some(0.0), 0.0);
    let a = 0; let mut q = w.q.clone(); let mut v = w.v.clone();
    let ext = cm.verts[a].iter().map(|p| p.0.norm()).fold(0.0, f64::max);
    q[a] = [0.0, ext+1e-5, 0.4]; v[a] = [0.0, 0.0, 1.0];
    let g = cm.ground_points(a, &q).iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    assert_eq!(cm.geometry(a, GROUND, &q).0, g);
    assert!(cm.may_contact(a, GROUND, &q, &v, 2.0));
}

#[test]
fn multiple_ground_friction_impulses_are_passive_and_fully_booked() {
    let mut w=free_world(1);
    let pids:Vec<_>=(0..w.patches.len()).collect();w.split(&pids,"friction test");
    for (i,b) in w.sc.m.bodies.iter().enumerate() {
        if b.kind==BodyKind::Cell {w.q[i]=[i as f64,10.0,0.0];}
    }
    let a=0;
    let cm=ContactModel::new(&w.sc.m,Some(0.0),0.5);
    w.q[a]=[0.0,-1e-3,0.1];w.vh[a]=[1.0,0.0,2.0];
    w.cm=Some(cm);w.cpairs=vec![(a,GROUND)];
    for i in 0..w.patches.len() {if w.alive[i] {w.open[i]=true;}}
    w.rebuild();
    let kinetic=|v:[f64;3],m:f64,i:f64|0.5*m*(v[0]*v[0]+v[1]*v[1])+0.5*i*v[2]*v[2];
    let body=&w.sc.m.bodies[a];let (mass,inertia)=(body.mass,body.inertia);
    let e0=kinetic(w.vh[a],mass,inertia);let d0=w.led.d_fric;
    w.contact_friction(&w.q.clone(),0.01);
    let e1=kinetic(w.vh[a],mass,inertia);
    assert!(e1<e0);
    assert!((e1-e0+w.led.d_fric-d0).abs()<1e-12);
}

#[test]
fn short_difference_inertia_gradient_matches_its_energy() {
    let w=free_world(2);
    let mut z=w.rs.fit(&w.patches,&w.q,&mut Work::default());
    for (i,x) in z.iter_mut().enumerate(){*x+=0.001*(i as f64*0.37).sin();}
    let sd=w.rs.step_data(&w.patches,&w.q,1e6,&mut Work::default());
    let zero=w.rs.step_data(&w.patches,&w.q,0.0,&mut Work::default());
    let ex=Extra{ext:&[],fj:None,robin:None,contacts:None};
    let eval=|z:&[f64],sd:&super::super::rsys::StepData|w.rs.assemble(&w.sc.m,&w.patches,z,sd,&ex,false,&mut Work::default());
    let g=eval(&z,&sd).0;let g0=eval(&z,&zero).0;
    let d:Vec<f64>=(0..z.len()).map(|i|(i as f64*0.61).cos()).collect();
    let expected:f64=g.iter().zip(g0).zip(&d).map(|((g,g0),d)|(g-g0)*d).sum();
    let h=1e-6;
    let zp:Vec<_>=z.iter().zip(&d).map(|(z,d)|z+h*d).collect();
    let zm:Vec<_>=z.iter().zip(&d).map(|(z,d)|z-h*d).collect();
    let fd=(eval(&zp,&sd).2-eval(&zp,&zero).2-eval(&zm,&sd).2+eval(&zm,&zero).2)/(2.0*h);
    assert!((fd-expected).abs()<1e-5*expected.abs().max(1.0),"{fd} vs {expected}");
}

#[test]
fn synchronised_contact_split_has_converging_global_energy_balance() {
    let mut errors=Vec::new();
    for h in [0.001,0.0005] {
        let mut w=free_world(1);w.use_fine_explicit();
        for f in &mut w.sc.m.facets {f.active=false;}
        w.p.h_e_fixed=Some(h);w.h=0.005;
        for i in 0..w.sc.m.n_cells(){w.q[i]=[i as f64,10.0,0.0];}
        let mut cm=ContactModel::new(&w.sc.m,Some(0.0),0.5);
        for verts in &mut cm.verts {for (_,k) in verts {*k*=1e-6;}}
        let a=0;let bottom=cm.verts[a].iter().map(|v|v.0.rot(0.3).y).fold(f64::INFINITY,f64::min);
        w.q[a]=[0.0,-bottom-0.001,0.3];w.v[a]=[0.2,-0.3,1.0];
        w.cm=Some(cm);w.cpairs=vec![(a,GROUND)];w.rebuild();
        let (tr,_)=run(w,"one polygon friction",0.05,0.001);
        assert!(tr.failed_at.is_none());
        errors.push((tr.e_end-tr.e0).abs()/tr.e0);
    }
    assert!(errors[1]<0.5*errors[0],"{errors:?}");
    assert!(errors[1]<0.002,"{errors:?}");
}

#[test]
fn rebuilding_interface_invalidates_subcycle_velocity_cache() {
    let mut w=free_world(1);w.p.absorbing=true;
    w.vr_iface.insert(usize::MAX,[1.0,2.0,0.0]);
    let pids=vec![0];w.split(&pids,"interface-cache regression");
    assert!(w.vr_iface.is_empty());
    let mut f=vec![[0.0;3];w.q.len()];let mut fr=f.clone();
    // This is exactly the force predictor that used a previous, shorter
    // iface_cells-indexed vector in the original S3b absorbing run.
    w.e_forces(&w.q.clone(),w.t,&mut f,&mut fr,false);
    assert!(f.iter().flatten().all(|x|x.is_finite()));
}
