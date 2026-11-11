//! Projection-only test on actual 2D RBSN and engine patch bases. Exact linear
//! modal evolution removes time integration / partitioned coupling as confounders.
use experiments::{rect_tess,engine::{patch::{build_patch,partition},Work}};
use geometry::V2;
use model::{Model,Material,LawOptions};
use solvers::spectrum::Spectral;
use serde_json::{json,Value};
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(a,b)|a*b).sum()}
fn basis(m:&Model,lp:f64,active_x:f64,nvib:usize,deg:usize)->(Vec<Vec<f64>>,usize,usize){
    let n=3*m.n_cells();let adj=m.neighbours();let groups=partition(m,&adj,lp,4);let loaded=vec![false;m.bodies.len()];
    let mut t=Vec::new();let mut fine=0;let mut extra=0;
    for cells in groups{
        if cells.iter().any(|&i|m.bodies[i].x0.x<active_x){
            fine+=cells.len();for &i in &cells{for k in 0..3{let mut c=vec![0.;n];c[3*i+k]=1.;t.push(c);}}continue
        }
        let mut w=Work::default();let p=build_patch(m,&adj,&cells,deg,nvib,&loaded,&mut w);extra+=p.me;
        let ir=(p.it+p.p.iter().zip(&p.mi).map(|(r,m)|m*r.dot(*r)).sum::<f64>()).sqrt();
        for j in 0..p.ny(){
            let mut c=vec![0.;n];
            for (li,&i) in p.cells.iter().enumerate(){
                let mm=[p.mi[li].sqrt(),p.mi[li].sqrt(),p.ii[li].sqrt()];
                for k in 0..3{
                    let v=if j==0{if k==0{1./p.mass.sqrt()}else{0.}}else if j==1{if k==1{1./p.mass.sqrt()}else{0.}}else if j==2{[-p.p[li].y,p.p[li].x,1.][k]/ir}else{p.te[(3*li+k)*p.me+j-3]};
                    c[3*i+k]=v*mm[k];
                }
            }t.push(c);
        }
    }
    let orth=(0..t.len()).flat_map(|i|(0..=i).map(move|j|(i,j))).map(|(i,j)|(dot(&t[i],&t[j])-if i==j{1.}else{0.}).abs()).fold(0.,f64::max);
    assert!(orth<1e-8,"M orthogonality {orth}");(t,fine,extra)
}
fn evolve(s:&Spectral,t:&[Vec<f64>],initial:&[f64],times:&[f64])->Vec<Vec<f64>>{
    let k=t.len();let mut bt=Vec::new();for c in t{let mut y=vec![0.;s.n];s.apply_b(c,&mut y);bt.push(y);}
    let a=faer::Mat::<f64>::from_fn(k,k,|i,j|0.5*(dot(&t[i],&bt[j])+dot(&t[j],&bt[i])));
    let eig=a.self_adjoint_eigen(faer::Side::Lower).unwrap();let u=eig.U();let l=eig.S();
    let vr:Vec<f64>=t.iter().map(|c|dot(c,initial)).collect();
    let coeff:Vec<f64>=(0..k).map(|j|(0..k).map(|i|u[(i,j)]*vr[i]).sum()).collect();
    times.iter().map(|&time|{
        let modal:Vec<f64>=(0..k).map(|j|{let w=l[j].max(0.).sqrt();coeff[j]*if w<1e-3{time}else{(w*time).sin()/w}}).collect();
        let z:Vec<f64>=(0..k).map(|i|(0..k).map(|j|u[(i,j)]*modal[j]).sum()).collect();
        (0..s.n).map(|i|t.iter().zip(&z).map(|(c,z)|c[i]*z).sum()).collect()
    }).collect()
}
pub fn experiment()->Value{
    let tess=rect_tess(V2::ZERO,V2::new(2.,0.6),0.10,4);
    let m=Model::from_tessellation(&tess,Material::concrete(0.2),LawOptions::elastic(),&[]);
    let s=Spectral::from_model2d(&m);let nc=m.n_cells();assert_eq!(s.n,3*nc);
    let full:Vec<Vec<f64>>=(0..s.n).map(|j|{let mut c=vec![0.;s.n];c[j]=1.;c}).collect();
    let times:Vec<f64>=(0..=120).map(|j|j as f64*0.0008/120.).collect();
    let mut rows=Vec::new();
    for direction in [0,1] {for width in [0.07,0.20]{
        let initial:Vec<f64>=(0..s.n).map(|j|if j%3==direction{m.bodies[j/3].mass.sqrt()*0.01*(-0.5*((m.bodies[j/3].x0.x-0.15)/width).powi(2)).exp()}else{0.}).collect();
        let reference=evolve(&s,&full,&initial,&times);
        for active_x in [0.15,0.75]{for nvib in [0,12]{for deg in [1,2,3]{
            let (t,fine,extra)=basis(&m,0.6,active_x,nvib,deg);let q=evolve(&s,&t,&initial,&times);
            let (mut local,mut lden,mut global,mut gden,mut force,mut fden)=(0.,0.,0.,0.,0.,0.);
            for (x,r) in q.iter().zip(&reference){
                for j in 0..s.n{let e=(x[j]-r[j]).powi(2);global+=e;gden+=r[j]*r[j];if m.bodies[j/3].x0.x<0.4{local+=e;lden+=r[j]*r[j];}}
                for (comp,subs) in s.comps.iter().zip(&s.wsub){
                    if m.facets[comp.facet].pc.x>0.4{continue}
                    let extension=|v:&[f64]|subs.iter().flat_map(|(bi,c)|s.blocks[*bi].dofs.iter().zip(c)).map(|(&j,c)|v[j]*c).sum::<f64>();
                    let fr=comp.k*extension(r);let fx=comp.k*extension(x);force+=(fx-fr).powi(2);fden+=fr*fr;
                }
            }
            rows.push(json!({"direction":direction,"width_m":width,"active_x":active_x,"degree":deg,"cells":nc,"fine_cells":fine,"nvib_requested":nvib,"deformation_modes":extra,"dofs":t.len(),"full_dofs":s.n,
                "local_displacement_l2":(local/lden).sqrt(),"global_displacement_l2":(global/gden).sqrt(),"local_spring_force_l2":(force/fden).sqrt()}));
        }}}
    }}
    json!({"scope":"actual 2D RBSN and patch basis; exact linear modal evolution. Projection error only, no multirate, damage or contact","T":0.0008,"rows":rows})
}
