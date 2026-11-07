//! Shared experiment plumbing: provenance, output files, common geometries and
//! error metrics.

pub mod engine;

use geometry::rng::Rng;
use geometry::rsa::rsa_rect;
use geometry::voronoi2d::{rect, voronoi_convex, Tessellation};
use geometry::V2;
use model::{FacetForce, LawOptions, Material, Model};
use serde::Serialize;
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

pub const SOURCE_HASH: &str = env!("SB_SOURCE_HASH");

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn out_dir(exp: &str) -> PathBuf {
    let d = root().join("results").join(exp);
    fs::create_dir_all(&d).unwrap();
    d
}

pub fn meta(exp: &str, run: &str, seed: Option<u64>, schedule: &str, config: Value) -> Value {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    json!({
        "experiment": exp,
        "run": run,
        "source_hash": SOURCE_HASH,
        "law_version": model::LAW_VERSION,
        "seed": seed,
        "solver_schedule": schedule,
        "unix_time": t,
        "config": config,
    })
}

pub fn write_json<T: Serialize>(exp: &str, name: &str, v: &T) {
    let p = out_dir(exp).join(format!("{name}.json"));
    fs::write(&p, serde_json::to_string_pretty(v).unwrap()).unwrap();
}

pub struct Csv {
    f: fs::File,
}

impl Csv {
    pub fn new(exp: &str, name: &str, header: &[&str]) -> Self {
        let p = out_dir(exp).join(format!("{name}.csv"));
        let mut f = fs::File::create(p).unwrap();
        writeln!(f, "{}", header.join(",")).unwrap();
        Csv { f }
    }
    pub fn row(&mut self, v: &[f64]) {
        let s: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
        writeln!(self.f, "{}", s.join(",")).unwrap();
    }
    pub fn row_s(&mut self, v: &[String]) {
        writeln!(self.f, "{}", v.join(",")).unwrap();
    }
}

/// RSA seeds + Voronoi in an axis-aligned rectangle.
pub fn rect_tess(lo: V2, hi: V2, dmin: f64, seed: u64) -> Tessellation {
    let mut rng = Rng::new(seed);
    let s = rsa_rect(lo, hi, dmin, &mut rng, 20_000);
    let t = voronoi_convex(&s, &rect(lo, hi));
    assert!(t.max_mismatch < 1e-9 * (hi - lo).norm(), "tessellation mismatch {}", t.max_mismatch);
    t
}

pub fn build(t: &Tessellation, mat: Material, law: LawOptions, notches: &[(V2, V2)]) -> Model {
    Model::from_tessellation(t, mat, law, notches)
}

/// Relative errors of facet resultants against a reference, per component:
/// (L2 relative, max abs / max |ref|) for N, V, M.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ForceErr {
    pub l2_n: f64,
    pub l2_v: f64,
    pub l2_m: f64,
    pub max_n: f64,
    pub max_v: f64,
    pub max_m: f64,
    /// Combined: max over the three L2 values.
    pub l2: f64,
    /// Combined: max over the three max-norm values.
    pub max: f64,
}

pub fn force_err(f: &[FacetForce], r: &[FacetForce], mask: &[bool]) -> ForceErr {
    let mut e = ForceErr::default();
    let comp = |sel: &dyn Fn(&FacetForce) -> f64| -> (f64, f64) {
        let (mut num, mut den, mut mx, mut mref) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 0..r.len() {
            if !mask[i] {
                continue;
            }
            let d = sel(&f[i]) - sel(&r[i]);
            num += d * d;
            den += sel(&r[i]).powi(2);
            mx = mx.max(d.abs());
            mref = mref.max(sel(&r[i]).abs());
        }
        let l2 = if den > 0.0 { (num / den).sqrt() } else { num.sqrt() };
        let m = if mref > 0.0 { mx / mref } else { mx };
        (l2, m)
    };
    (e.l2_n, e.max_n) = comp(&|x| x.n);
    (e.l2_v, e.max_v) = comp(&|x| x.v);
    (e.l2_m, e.max_m) = comp(&|x| x.m);
    e.l2 = e.l2_n.max(e.l2_v).max(e.l2_m);
    e.max = e.max_n.max(e.max_v).max(e.max_m);
    e
}

/// Add a uniform traction (force per area, global frame) on the cells' boundary
/// edges of a domain edge, as equivalent force + moment at each cell centroid.
/// Returns the total force applied.
pub fn edge_traction(m: &mut Model, t: &Tessellation, edge: usize, range: Option<(f64, f64)>, tr: V2, pattern: bool) -> V2 {
    use geometry::voronoi2d::EdgeTag;
    let mut tot = V2::ZERO;
    for (i, c) in t.cells.iter().enumerate() {
        let k = c.verts.len();
        for e in 0..k {
            if c.tags[e] != EdgeTag::Boundary(edge) {
                continue;
            }
            let p0 = c.verts[e];
            let p1 = c.verts[(e + 1) % k];
            let mid = (p0 + p1) * 0.5;
            let d = p1 - p0;
            if let Some((lo, hi)) = range {
                let s = if d.x.abs() > d.y.abs() { mid.x } else { mid.y };
                if s < lo || s > hi {
                    continue;
                }
            }
            let f = tr * (d.norm() * m.mat.thickness);
            let arm = mid - m.bodies[i].x0;
            let target = if pattern { &mut m.pattern[i] } else { &mut m.loads[i] };
            target[0] += f.x;
            target[1] += f.y;
            target[2] += arm.cross(f);
            tot += f;
        }
    }
    tot
}

/// `edge_traction` with the load interval applied exactly: each boundary edge is
/// clipped to `range`, so the resultant force and moment do not depend on the
/// tessellation (the midpoint rule of `edge_traction` does).
pub fn edge_traction_exact(m: &mut Model, t: &Tessellation, edge: usize, range: Option<(f64, f64)>, tr: V2, pattern: bool) -> V2 {
    use geometry::voronoi2d::EdgeTag;
    let d0 = t.domain[edge];
    let dir = (t.domain[(edge + 1) % t.domain.len()] - d0).normalized();
    let mut tot = V2::ZERO;
    for (i, c) in t.cells.iter().enumerate() {
        let k = c.verts.len();
        for e in 0..k {
            if c.tags[e] != EdgeTag::Boundary(edge) {
                continue;
            }
            let Some((p0, p1)) = model::structure::clip_to_range(c.verts[e], c.verts[(e + 1) % k], dir, range) else { continue };
            let f = tr * ((p1 - p0).norm() * m.mat.thickness);
            let arm = (p0 + p1) * 0.5 - m.bodies[i].x0;
            let target = if pattern { &mut m.pattern[i] } else { &mut m.loads[i] };
            target[0] += f.x;
            target[1] += f.y;
            target[2] += arm.cross(f);
            tot += f;
        }
    }
    tot
}

/// Sum of forces the facets with `tag` exert on their support body (global frame).
pub fn support_reaction(m: &Model, q: &[[f64; 3]], tag: u32) -> V2 {
    let mut r = V2::ZERO;
    for f in m.facets.iter().filter(|f| f.active && f.tag == tag) {
        for p in 0..2 {
            let (j, e) = m.eval_point(q, f, p);
            // force on body b = −(Jᵀf) restricted to translations of b
            r.x -= j.jnb[0] * e.fnorm + j.jtb[0] * e.ftan;
            r.y -= j.jnb[1] * e.fnorm + j.jtb[1] * e.ftan;
        }
    }
    r
}

pub fn ranges_json() -> Value {
    let r = model::range::take();
    let mut o = serde_json::Map::new();
    for (k, v) in r {
        o.insert(
            k,
            json!({"min_abs_nonzero": v.min_abs_nonzero, "max_abs": v.max_abs, "count": v.count, "zeros": v.zeros, "dyn_bits": v.dyn_bits()}),
        );
    }
    Value::Object(o)
}
