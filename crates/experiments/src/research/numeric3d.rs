//! Small 3D linear RBSN and integer local-block feasibility probe. This is not a
//! 3D fracture/contact engine and does not claim a deterministic sparse factorisation.
use geometry::{V3,rng::Rng,voronoi3d::voronoi_box};
use model::rbsn3::{Mat3D,Model3};
use solvers::{sparse::{csc_from_triplets,Spd},spectrum::Spectral};
#[cfg(test)]
use experiments::engine::patch::{chol,chol_solve};
use serde_json::{json,Value};
const Q:i64=1<<30;
fn mul(a:i64,b:i64)->i64{((a as i128*b as i128+(1i128<<29))>>30).try_into().expect("wide fixed product overflow")}
fn div(a:i64,b:i64)->i64{((a as i128*Q as i128)/b as i128).try_into().expect("fixed quotient overflow")}
fn fixed_solve(a:&[f64],b:&[f64],n:usize)->Option<(Vec<f64>,i64,i64)>{
    let aa:Vec<i64>=a.iter().map(|v|(v*Q as f64).round() as i64).collect();
    let mut l=vec![0i64;n*n];let mut d=vec![0i64;n];let mut max=0i64;
    for j in 0..n{
        let mut dj=aa[j*n+j];for k in 0..j{dj-=mul(mul(l[j*n+k],l[j*n+k]),d[k]);}if dj<=0{return None}d[j]=dj;l[j*n+j]=Q;
        for i in j+1..n{let mut r=aa[i*n+j];for k in 0..j{r-=mul(mul(l[i*n+k],l[j*n+k]),d[k]);}l[i*n+j]=div(r,dj);max=max.max(l[i*n+j].abs());}
    }
    let mut x:Vec<i64>=b.iter().map(|v|(v*Q as f64).round() as i64).collect();
    for i in 0..n{for j in 0..i{x[i]-=mul(l[i*n+j],x[j]);}}
    for i in 0..n{x[i]=div(x[i],d[i]);}
    for i in (0..n).rev(){for j in i+1..n{x[i]-=mul(l[j*n+i],x[j]);}}
    Some((x.iter().map(|v|*v as f64/Q as f64).collect(),max,*d.iter().min().unwrap()))
}
fn mass(m:&Model3,x:&[f64])->Vec<f64>{
    let mut y=vec![0.;x.len()];for (i,c) in m.cells.iter().enumerate(){for k in 0..3{y[6*i+k]=c.mass*x[6*i+k];for j in 0..3{y[6*i+3+k]+=c.inertia[k][j]*x[6*i+3+j];}}}y
}
fn kv(m:&Model3,x:&[f64])->Vec<f64>{
    let mut y=vec![0.;x.len()];for c in &m.comps{let f=c.k*m.comp_value(c,x);for j in 0..6{y[6*c.a+j]+=f*c.ja[j];if let Some(b)=c.b{y[6*b+j]+=f*c.jb[j];}}}y
}
fn energy(m:&Model3,q:&[f64],v:&[f64])->f64{0.5*(v.iter().zip(mass(m,v)).map(|(x,y)|x*y).sum::<f64>()+q.iter().zip(kv(m,q)).map(|(x,y)|x*y).sum::<f64>())}
fn midpoint(m:&Model3,h:f64)->Value{
    let n=m.ndof();let mut trip:Vec<_>=m.stiffness_triplets().into_iter().map(|(i,j,v)|(i,j,0.25*h*h*v)).collect();
    for (i,c) in m.cells.iter().enumerate(){for k in 0..3{trip.push((6*i+k,6*i+k,c.mass));for j in 0..3{trip.push((6*i+3+k,6*i+3+j,c.inertia[k][j]));}}}
    let a=csc_from_triplets(n,trip);let mut solver=Spd::analyze(&a);solver.factor(&a).unwrap();
    let mut q=vec![0.;n];let mut v=vec![0.;n];v[n/2+1]=0.01;let e0=energy(m,&q,&v);let mut max_error=0f64;
    let clock=std::time::Instant::now();
    for _ in 0..100{
        let mq=mass(m,&q);let mv=mass(m,&v);let kq=kv(m,&q);
        let mut qn:Vec<f64>=(0..n).map(|i|mq[i]+h*mv[i]-0.25*h*h*kq[i]).collect();solver.solve(&mut qn);
        for i in 0..n{v[i]=2.*(qn[i]-q[i])/h-v[i];}q=qn;
        max_error=max_error.max((energy(m,&q,&v)-e0).abs()/e0);
    }
    json!({"h":h,"steps":100,"relative_energy_error":max_error,"wall_s":clock.elapsed().as_secs_f64(),"factor_flops":solver.stats.flops_factor,"nnz_l":solver.stats.nnz_l})
}
pub fn experiment()->Value{
    let mut rows=Vec::new();let mut blocks=Vec::new();
    for (nx,ny,nz) in [(4,2,2),(8,3,3)]{
        let mut rng=Rng::new(7);let mut seeds=Vec::new();let ell=0.25;
        for i in 0..nx{for j in 0..ny{for k in 0..nz{seeds.push(V3::new((i as f64+0.5+rng.range(-0.08,0.08))*ell,(j as f64+0.5+rng.range(-0.08,0.08))*ell,(k as f64+0.5+rng.range(-0.08,0.08))*ell));}}}
        let cells=voronoi_box(&seeds,V3::ZERO,V3::new(nx as f64*ell,ny as f64*ell,nz as f64*ell));
        let m=Model3::build(&cells,&[0],Mat3D::concrete());
        let spectrum=Spectral::from_model3(&m);let eig=spectrum.dense_eigs();
        rows.push(json!({"cells":m.cells.len(),"dofs":m.ndof(),"lambda_min":eig.iter().copied().fold(f64::INFINITY,f64::min),"lambda_max":eig.iter().copied().fold(0.,f64::max),"midpoint":[midpoint(&m,1./480.),midpoint(&m,1e-5)]}));
        let kt=m.stiffness_triplets();
        for h in [1./60.,1./480.,1e-5]{for (cell,c) in m.cells.iter().enumerate(){
            let mut a=vec![0.;36];for &(i,j,v) in &kt{if i/6==cell&&j/6==cell{a[(i%6)*6+j%6]+=h*h*v;}}
            for k in 0..3{a[k*6+k]+=c.mass;for j in 0..3{a[(3+k)*6+3+j]+=c.inertia[k][j];}}
            let scale:Vec<f64>=(0..6).map(|i|1./a[i*6+i].sqrt()).collect();
            for i in 0..6{for j in 0..6{a[i*6+j]*=scale[i]*scale[j];}}
            let truth:Vec<f64>=(0..6).map(|i|0.1*((i+1) as f64).sin()).collect();
            let b:Vec<f64>=(0..6).map(|i|(0..6).map(|j|a[i*6+j]*truth[j]).sum()).collect();
            let ev=faer::Mat::<f64>::from_fn(6,6,|i,j|a[i*6+j]).self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
            let fixed=fixed_solve(&a,&b,6).map(|(x,max,pivot)|json!({"max_solution_error":x.iter().zip(&truth).map(|(x,y)|(x-y).abs()).fold(0.,f64::max),"max_L":max as f64/Q as f64,"min_pivot":pivot as f64/Q as f64,"L_fits_i32":max<=i32::MAX as i64}));
            blocks.push(json!({"cells":m.cells.len(),"cell":cell,"h":h,"condition":ev.iter().copied().fold(0.,f64::max)/ev.iter().copied().fold(f64::INFINITY,f64::min),"fixed":fixed}));
        }}
    }
    json!({"scope":"linear 3D RBSN, consistent full inertia; no 3D damage/contact. Q30 LDL uses i128 products/divisions as CPU oracle; not a WGSL performance claim","models":rows,"local_blocks":blocks})
}
#[test]
fn fixed_ldl_solves_spd_block(){
    let a=vec![1.,0.2,0.1,0.2,1.,0.3,0.1,0.3,1.];let b=vec![0.1,0.2,0.3];
    let mut r=b.clone();chol_solve(&chol(&a,3).unwrap(),3,&mut r);
    let (x,_,_)=fixed_solve(&a,&b,3).unwrap();assert!(x.iter().zip(r).all(|(x,y)|(x-y).abs()<1e-8));
}
