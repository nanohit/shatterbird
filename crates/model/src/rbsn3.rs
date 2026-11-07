//! Linearised 3D RBSN on Voronoi polyhedra (the E10 model, as a library for the
//! Phase 0b spectral work): 6 DOFs per cell at the centroid (translation, rotation
//! vector); per facet 3 translational components at the facet centroid (k_n = EA/ℓ_e,
//! k_t = γk_n along the principal in-plane axes) and 3 rotational ones (bending
//! k_n·I/A about each principal axis, torsion k_t·I_p/A). Support facets connect a
//! cell to fixed ground with ℓ_e = distance from the generator to the support plane.

use geometry::voronoi3d::{planar_props, FaceTag, VCell3};
use geometry::V3;

#[derive(Clone, Copy, Debug)]
pub struct Mat3D {
    pub e: f64,
    pub density: f64,
    pub gamma: f64,
}

impl Mat3D {
    pub fn concrete() -> Self {
        Self { e: 30e9, density: 2400.0, gamma: 1.0 }
    }
    pub fn wave_speed(&self) -> f64 {
        (self.e / self.density).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct Cell3 {
    pub centroid: V3,
    pub generator: V3,
    pub volume: f64,
    pub mass: f64,
    /// Inertia tensor about the centroid [kg m²].
    pub inertia: [[f64; 3]; 3],
}

#[derive(Clone, Debug)]
pub struct Facet3 {
    pub a: usize,
    /// None = ground (support facet).
    pub b: Option<usize>,
    pub area: f64,
    pub centroid: V3,
    /// Unit normal from a to b (into the support for ground facets).
    pub normal: V3,
    pub ell: f64,
    /// Domain face id for support facets.
    pub face: Option<usize>,
}

/// One scalar spring component: C = j·[u_a ω_a u_b ω_b], energy ½ k C².
#[derive(Clone, Debug)]
pub struct Comp3 {
    pub a: usize,
    pub b: Option<usize>,
    pub ja: [f64; 6],
    pub jb: [f64; 6],
    pub k: f64,
    pub facet: usize,
}

#[derive(Clone, Debug)]
pub struct Model3 {
    pub mat: Mat3D,
    pub cells: Vec<Cell3>,
    pub facets: Vec<Facet3>,
    pub comps: Vec<Comp3>,
    pub loads: Vec<[f64; 6]>,
}

fn facet_comps(a: usize, b: Option<usize>, ca: V3, cb: Option<V3>, pc: V3, area: f64, e1: V3, e2: V3, i2: [f64; 3], n: V3, kn: f64, kt: f64, fi: usize, out: &mut Vec<Comp3>) {
    // principal axes of the in-plane second moments
    let (iuu, ivv, iuv) = (i2[0], i2[1], i2[2]);
    let th = 0.5 * (2.0 * iuv).atan2(iuu - ivv);
    let (s, c) = th.sin_cos();
    let p1 = e1 * c + e2 * s;
    let p2 = e2 * c - e1 * s;
    let i_p1 = iuu * c * c + 2.0 * iuv * s * c + ivv * s * s; // ∫ (coord along p1)²
    let i_p2 = iuu * s * s - 2.0 * iuv * s * c + ivv * c * c; // ∫ (coord along p2)²
    let ra = pc - ca;
    let rb = cb.map(|x| pc - x).unwrap_or(V3::ZERO);
    let tr = |d: V3, k: f64, out: &mut Vec<Comp3>| {
        let rad = ra.cross(d);
        let rbd = rb.cross(d);
        out.push(Comp3 { a, b, ja: [-d.x, -d.y, -d.z, -rad.x, -rad.y, -rad.z], jb: [d.x, d.y, d.z, rbd.x, rbd.y, rbd.z], k, facet: fi });
    };
    tr(n, kn, out);
    tr(p1, kt, out);
    tr(p2, kt, out);
    let rot = |e: V3, k: f64, out: &mut Vec<Comp3>| {
        out.push(Comp3 { a, b, ja: [0.0, 0.0, 0.0, -e.x, -e.y, -e.z], jb: [0.0, 0.0, 0.0, e.x, e.y, e.z], k, facet: fi });
    };
    // rotation about p1 makes the normal jump vary along p2: stiffness k_n ∫(p2)²/A
    rot(p1, kn * i_p2 / area, out);
    rot(p2, kn * i_p1 / area, out);
    rot(n, kt * (i_p1 + i_p2) / area, out);
}

impl Model3 {
    /// Cells from a Voronoi tessellation; boundary faces listed in `supports` are
    /// connected to ground.
    pub fn build(cells3: &[VCell3], supports: &[usize], mat: Mat3D) -> Model3 {
        let cells: Vec<Cell3> = cells3
            .iter()
            .map(|c| Cell3 {
                centroid: c.props.centroid,
                generator: c.seed,
                volume: c.props.volume,
                mass: mat.density * c.props.volume,
                inertia: c.props.inertia.map(|r| r.map(|x| x * mat.density)),
            })
            .collect();
        let mut m = Model3 { mat, cells, facets: Vec::new(), comps: Vec::new(), loads: vec![[0.0; 6]; cells3.len()] };
        for (i, c) in cells3.iter().enumerate() {
            for f in &c.poly.faces {
                let (b, n, ell, face) = match f.tag {
                    FaceTag::Cell(j) if i < j => {
                        let d = cells3[j].seed - c.seed;
                        (Some(j), d.normalized(), d.norm(), None)
                    }
                    FaceTag::Boundary(k) if supports.contains(&k) => (None, f.normal, (f.verts[0] - c.seed).dot(f.normal).abs(), Some(k)),
                    _ => continue,
                };
                let (area, pc, e1, e2, i2) = planar_props(&f.verts, n);
                let kn = mat.e * area / ell;
                let fi = m.facets.len();
                let cb = b.map(|j| m.cells[j].centroid);
                facet_comps(i, b, m.cells[i].centroid, cb, pc, area, e1, e2, i2, n, kn, mat.gamma * kn, fi, &mut m.comps);
                m.facets.push(Facet3 { a: i, b, area, centroid: pc, normal: n, ell, face });
            }
        }
        m
    }

    /// Uniform traction on a boundary face, as force + moment at the centroids.
    pub fn traction(&mut self, cells3: &[VCell3], face: usize, t: V3) -> V3 {
        let mut tot = V3::ZERO;
        for (i, c) in cells3.iter().enumerate() {
            for f in &c.poly.faces {
                if f.tag == FaceTag::Boundary(face) {
                    let (area, pc, _, _, _) = planar_props(&f.verts, f.normal);
                    let fv = t * area;
                    let mo = (pc - self.cells[i].centroid).cross(fv);
                    for k in 0..3 {
                        self.loads[i][k] += fv[k];
                        self.loads[i][3 + k] += mo[k];
                    }
                    tot += fv;
                }
            }
        }
        tot
    }

    pub fn ndof(&self) -> usize {
        6 * self.cells.len()
    }

    /// C = j·u for a global DOF vector u (6 per cell).
    pub fn comp_value(&self, c: &Comp3, u: &[f64]) -> f64 {
        let mut v = 0.0;
        for k in 0..6 {
            v += c.ja[k] * u[6 * c.a + k];
            if let Some(b) = c.b {
                v += c.jb[k] * u[6 * b + k];
            }
        }
        v
    }

    pub fn stiffness_triplets(&self) -> Vec<(usize, usize, f64)> {
        let mut t = Vec::new();
        for c in &self.comps {
            let mut ent: Vec<(usize, f64)> = (0..6).map(|k| (6 * c.a + k, c.ja[k])).collect();
            if let Some(b) = c.b {
                ent.extend((0..6).map(|k| (6 * b + k, c.jb[k])));
            }
            for &(r, vr) in &ent {
                for &(cc, vc) in &ent {
                    if vr * vc != 0.0 {
                        t.push((r, cc, c.k * vr * vc));
                    }
                }
            }
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geometry::voronoi3d::voronoi_box;

    #[test]
    fn free_pair_rigid_modes_and_twist() {
        let seeds = [V3::new(0.25, 0.5, 0.5), V3::new(0.75, 0.5, 0.5)];
        let cells3 = voronoi_box(&seeds, V3::ZERO, V3::new(1.0, 1.0, 1.0));
        let m = Model3::build(&cells3, &[], Mat3D::concrete());
        let cen = V3::new(0.5, 0.5, 0.5);
        // rigid translations and rotations produce no spring stretch
        for k in 0..6 {
            let mut u = vec![0.0; 12];
            for (i, c) in m.cells.iter().enumerate() {
                if k < 3 {
                    u[6 * i + k] = 1.0;
                } else {
                    let mut w = V3::ZERO;
                    w[k - 3] = 1.0;
                    let t = w.cross(c.centroid - cen);
                    u[6 * i..6 * i + 3].copy_from_slice(&[t.x, t.y, t.z]);
                    u[6 * i + k] = 1.0;
                }
            }
            for c in &m.comps {
                assert!(m.comp_value(c, &u).abs() < 1e-14, "mode {k}");
            }
        }
        // unit-square facet at x = 0.5, ℓ_e = 0.5: torsion k_t I_p/A = k_n/6
        let kn = m.mat.e / 0.5;
        let tw = m.comps.iter().find(|c| (c.ja[3] + 1.0).abs() < 1e-12).map(|c| c.k).unwrap();
        assert!((tw / kn - 1.0 / 6.0).abs() < 1e-12);
    }
}
