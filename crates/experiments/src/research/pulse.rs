//! Controlled elastic port experiment, not a fracture validation. A finite chain
//! has exactly the same M,K in every run. CMS retains the interface displacement,
//! a static constraint mode and fixed-interface vibration modes. No dashpots,
//! mass changes, or fitted energy sinks. Orthonormalising T makes M_r=I.
use serde_json::{json,Value};
const N:usize=128;
const ACTIVE:usize=8;
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(x,y)|x*y).sum()}
fn mv(a:&[Vec<f64>],x:&[f64])->Vec<f64>{a.iter().map(|r|dot(r,x)).collect()}
fn lift(t:&[Vec<f64>],x:&[f64])->Vec<f64>{(0..N).map(|i|t.iter().zip(x).map(|(c,z)|c[i]*z).sum()).collect()}
fn basis(a:usize,modes:usize,full:bool)->Vec<Vec<f64>>{
    if full{return (0..N).map(|i|{let mut c=vec![0.;N];c[i]=1.;c}).collect()}
    let mut cols=Vec::new();
    for i in 0..a{let mut c=vec![0.;N];c[i]=1.;cols.push(c);}
    // Port is node a; right boundary is a fixed node N.
    cols.push((0..N).map(|j|if j>=a{(N-j) as f64/(N-a) as f64}else{0.}).collect());
    for k in 1..=modes.min(N-a-1){cols.push((0..N).map(|j|if j>a{(std::f64::consts::PI*k as f64*(j-a) as f64/(N-a) as f64).sin()}else{0.}).collect());}
    for j in 0..cols.len(){
        for _ in 0..2{for k in 0..j{let d=dot(&cols[j],&cols[k]);for i in 0..N{cols[j][i]-=d*cols[k][i];}}}
        let norm=dot(&cols[j],&cols[j]).sqrt();assert!(norm>1e-12);
        for v in &mut cols[j]{*v/=norm;}
    }cols
}
fn stiffness(t:&[Vec<f64>],a:usize)->(Vec<Vec<f64>>,Vec<Vec<f64>>){
    let n=t.len();let mut fast=vec![vec![0.;n];n];let mut slow=fast.clone();
    // Edge e connects e-1 and e, including grounded endpoints -1 and N.
    for e in 0..=N{
        let d:Vec<f64>=t.iter().map(|c|if e==0{c[0]}else if e==N{-c[N-1]}else{c[e]-c[e-1]}).collect();
        let k=if e<a{&mut fast}else{&mut slow};
        for i in 0..n{for j in 0..n{k[i][j]+=d[i]*d[j];}}
    }(fast,slow)
}
fn energy(q:&[f64],v:&[f64])->f64{
    0.5*(dot(v,v)+q[0]*q[0]+q[N-1]*q[N-1]+q.windows(2).map(|r|(r[1]-r[0]).powi(2)).sum::<f64>())
}
struct Trace{q:Vec<Vec<f64>>,v:Vec<Vec<f64>>,energy:Vec<f64>,work:f64}
fn integrate(t:&[Vec<f64>],a:usize,width:f64,ratio:usize,h:f64)->Trace{
    let (kf,ks)=stiffness(t,a);let n=t.len();
    let initial:Vec<f64>=(0..N).map(|i|if i<ACTIVE{(-0.5*((i as f64-3.)/width).powi(2)).exp()*(std::f64::consts::PI*(i+1) as f64/(ACTIVE+1) as f64).sin()}else{0.}).collect();
    let mut q=vec![0.;n];let mut v:Vec<f64>=t.iter().map(|c|dot(c,&initial)).collect();
    let mut out=Trace{q:Vec::new(),v:Vec::new(),energy:Vec::new(),work:0.};
    let steps=(60./h).round() as usize;let hs=h*ratio as f64;
    for step in 0..=steps/ratio{
        if step>0{
            if ratio==1{
                let k:Vec<Vec<f64>>=kf.iter().zip(&ks).map(|(r,s)|r.iter().zip(s).map(|(x,y)|x+y).collect()).collect();
                let f=mv(&k,&q);for i in 0..n{v[i]-=0.5*h*f[i];q[i]+=h*v[i];}
                let f=mv(&k,&q);for i in 0..n{v[i]-=0.5*h*f[i];}
                out.work+=4.*(n*n) as f64;
            }else{
                // Symmetric impulse/r-RESPA: slow environmental + interface force
                // only at coarse endpoints. Same projected mechanics, different time coupling.
                let f=mv(&ks,&q);for i in 0..n{v[i]-=0.5*hs*f[i];}
                for _ in 0..ratio{
                    let f=mv(&kf,&q);for i in 0..n{v[i]-=0.5*h*f[i];q[i]+=h*v[i];}
                    let f=mv(&kf,&q);for i in 0..n{v[i]-=0.5*h*f[i];}
                }
                let f=mv(&ks,&q);for i in 0..n{v[i]-=0.5*hs*f[i];}
                // Explicit count of the implemented dense operations, not an O(N) fine cost.
                out.work+=4.*(n*n) as f64*(ratio+1) as f64;
            }
        }
        let qq=lift(t,&q);let vv=lift(t,&v);
        out.energy.push(energy(&qq,&vv));out.q.push(qq);out.v.push(vv);
    }out
}
fn compare(x:&Trace,r:&Trace,ratio:usize,range:std::ops::Range<usize>)->f64{
    let(mut num,mut den)=(0.,0.);
    for s in 0..x.q.len(){for i in range.clone(){num+=(x.q[s][i]-r.q[s*ratio][i]).powi(2);den+=r.q[s*ratio][i].powi(2);}}
    (num/den.max(1e-30)).sqrt()
}
pub fn experiment()->Value{
    let mut rows=Vec::new();
    for width in [0.75,2.0]{
        let full=basis(0,0,true);let reference=integrate(&full,ACTIVE,width,1,0.025);
        for buffer in [0,8,24]{for modes in [0,4,16,48]{
            let a=ACTIVE+buffer;let t=basis(a,modes,false);
            let common=integrate(&t,a,width,1,0.05);
            for ratio in [1,4,16]{
                let run=if ratio==1{integrate(&t,a,width,1,0.05)}else{integrate(&t,a,width,ratio,0.05)};
                let e0=run.energy[0];let drift=run.energy.iter().map(|e|(e-e0).abs()/e0).fold(0.,f64::max);
                rows.push(json!({"width_cells":width,"N":N,"active":ACTIVE,"buffer":buffer,"modes":modes,"dofs":t.len(),"ratio":ratio,
                    "local_relative_l2":compare(&run,&reference,2*ratio,0..ACTIVE),
                    "global_relative_l2":compare(&run,&reference,2*ratio,0..N),
                    "time_coupling_local_l2":compare(&run,&common,ratio,0..ACTIVE),
                    "max_energy_drift":drift,"energy_creation":run.energy.iter().map(|e|(e-e0)/e0).fold(0.,f64::max),
                    "implemented_dense_work":run.work,"completed_finite":run.energy.iter().all(|e|e.is_finite())}));
            }
        }}
    }
    json!({"scope":"linear 1D finite-chain CMS control; not evidence of fracture correctness or GPU speed","reference_h":0.025,"common_h":0.05,"T":60.,"damage":false,"dashpots":false,"rows":rows})
}

#[test]
fn full_cms_span_recovers_chain_and_energy(){
    let r=integrate(&basis(0,0,true),ACTIVE,0.75,1,0.1);
    let x=integrate(&basis(ACTIVE,N-ACTIVE-1,false),ACTIVE,0.75,1,0.1);
    assert!(compare(&x,&r,1,0..N)<1e-10);
    let e0=x.energy[0];assert!(x.energy.iter().all(|e|(e-e0).abs()/e0<0.01));
}
