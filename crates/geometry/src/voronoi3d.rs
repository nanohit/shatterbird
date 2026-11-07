//! 3D Voronoi of a box by plane clipping of convex polyhedra (E10 only).

use crate::V3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceTag {
    Cell(usize),
    Boundary(usize),
}

#[derive(Clone, Debug)]
pub struct Face {
    /// Counter-clockwise seen from outside (along `normal`).
    pub verts: Vec<V3>,
    pub normal: V3,
    pub tag: FaceTag,
}

#[derive(Clone, Debug)]
pub struct Polyhedron {
    pub faces: Vec<Face>,
}

#[derive(Clone, Copy, Debug)]
pub struct MassProps {
    pub volume: f64,
    pub centroid: V3,
    /// ∫ (|r|² I - r rᵀ) dV about the centroid (unit density), row-major.
    pub inertia: [[f64; 3]; 3],
}

pub fn box_poly(lo: V3, hi: V3) -> Polyhedron {
    let p = |x: usize, y: usize, z: usize| {
        V3::new(if x == 0 { lo.x } else { hi.x }, if y == 0 { lo.y } else { hi.y }, if z == 0 { lo.z } else { hi.z })
    };
    let f = |v: [V3; 4], n: V3, k: usize| Face { verts: v.to_vec(), normal: n, tag: FaceTag::Boundary(k) };
    Polyhedron {
        faces: vec![
            f([p(0, 0, 0), p(0, 0, 1), p(0, 1, 1), p(0, 1, 0)], V3::new(-1.0, 0.0, 0.0), 0),
            f([p(1, 0, 0), p(1, 1, 0), p(1, 1, 1), p(1, 0, 1)], V3::new(1.0, 0.0, 0.0), 1),
            f([p(0, 0, 0), p(1, 0, 0), p(1, 0, 1), p(0, 0, 1)], V3::new(0.0, -1.0, 0.0), 2),
            f([p(0, 1, 0), p(0, 1, 1), p(1, 1, 1), p(1, 1, 0)], V3::new(0.0, 1.0, 0.0), 3),
            f([p(0, 0, 0), p(0, 1, 0), p(1, 1, 0), p(1, 0, 0)], V3::new(0.0, 0.0, -1.0), 4),
            f([p(0, 0, 1), p(1, 0, 1), p(1, 1, 1), p(0, 1, 1)], V3::new(0.0, 0.0, 1.0), 5),
        ],
    }
}

impl Polyhedron {
    /// Keep the half-space (x - m)·n <= 0.
    pub fn clip(&self, m: V3, n: V3, tag: FaceTag, eps: f64) -> Polyhedron {
        let mut faces = Vec::with_capacity(self.faces.len() + 1);
        let mut cap: Vec<V3> = Vec::new();
        let mut any_out = false;
        for face in &self.faces {
            let f: Vec<f64> = face.verts.iter().map(|&v| (v - m).dot(n)).collect();
            if f.iter().any(|&x| x > eps) {
                any_out = true;
            }
            let k = face.verts.len();
            let mut out = Vec::with_capacity(k + 1);
            for i in 0..k {
                let j = (i + 1) % k;
                let (a, b) = (face.verts[i], face.verts[j]);
                let ina = f[i] <= eps;
                let inb = f[j] <= eps;
                if ina {
                    out.push(a);
                    if f[i].abs() <= eps {
                        cap.push(a);
                    }
                }
                if ina != inb {
                    let t = f[i] / (f[i] - f[j]);
                    let p = a + (b - a) * t;
                    out.push(p);
                    cap.push(p);
                }
            }
            dedup_loop(&mut out, eps);
            if out.len() >= 3 && poly_area(&out, face.normal) > eps * eps {
                faces.push(Face { verts: out, normal: face.normal, tag: face.tag });
            }
        }
        if !any_out {
            return self.clone();
        }
        // Order the cap points by angle around their mean in the plane.
        let mut uniq: Vec<V3> = Vec::new();
        for p in cap {
            if !uniq.iter().any(|q| (*q - p).norm() < eps * 10.0) {
                uniq.push(p);
            }
        }
        if uniq.len() >= 3 {
            let c = uniq.iter().fold(V3::ZERO, |s, &p| s + p) / uniq.len() as f64;
            let e1 = n.any_orthonormal();
            let e2 = n.cross(e1);
            uniq.sort_by(|a, b| {
                let ta = (*a - c).dot(e2).atan2((*a - c).dot(e1));
                let tb = (*b - c).dot(e2).atan2((*b - c).dot(e1));
                ta.partial_cmp(&tb).unwrap()
            });
            if poly_area(&uniq, n) > eps * eps {
                faces.push(Face { verts: uniq, normal: n, tag });
            }
        }
        Polyhedron { faces }
    }

    pub fn max_radius(&self, c: V3) -> f64 {
        self.faces.iter().flat_map(|f| f.verts.iter()).map(|&v| (v - c).norm()).fold(0.0, f64::max)
    }

    pub fn mass_props(&self) -> MassProps {
        let o = self.faces[0].verts[0];
        let mut vol = 0.0;
        let mut first = V3::ZERO;
        let mut second = [[0.0; 3]; 3];
        for f in &self.faces {
            let a = f.verts[0] - o;
            for k in 1..f.verts.len() - 1 {
                let b = f.verts[k] - o;
                let c = f.verts[k + 1] - o;
                let v = a.dot(b.cross(c)) / 6.0;
                vol += v;
                let s = a + b + c;
                first += s * (v / 4.0);
                // ∫ x xᵀ over tet (0,a,b,c) = V/20 (Σ v_i v_iᵀ + s sᵀ)
                for i in 0..3 {
                    for j in 0..3 {
                        second[i][j] += v / 20.0 * (a[i] * a[j] + b[i] * b[j] + c[i] * c[j] + s[i] * s[j]);
                    }
                }
            }
        }
        let cl = first / vol;
        // shift second moment to centroid
        let mut cov = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] = second[i][j] - vol * cl[i] * cl[j];
            }
        }
        let tr = cov[0][0] + cov[1][1] + cov[2][2];
        let mut inertia = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                inertia[i][j] = if i == j { tr } else { 0.0 } - cov[i][j];
            }
        }
        MassProps { volume: vol, centroid: cl + o, inertia }
    }
}

fn dedup_loop(v: &mut Vec<V3>, eps: f64) {
    let mut k = 0;
    while v.len() > 1 && k < v.len() {
        let j = (k + 1) % v.len();
        if (v[j] - v[k]).norm() < eps {
            v.remove(j);
        } else {
            k += 1;
        }
    }
}

pub fn poly_area(v: &[V3], n: V3) -> f64 {
    let mut s = V3::ZERO;
    for k in 1..v.len().saturating_sub(1) {
        s += (v[k] - v[0]).cross(v[k + 1] - v[0]);
    }
    0.5 * s.dot(n)
}

/// Planar polygon properties in an in-plane frame (e1, e2): area, centroid and
/// second moments about the centroid (∫u², ∫v², ∫uv).
pub fn planar_props(v: &[V3], n: V3) -> (f64, V3, V3, V3, [f64; 3]) {
    let e1 = n.any_orthonormal();
    let e2 = n.cross(e1);
    let o = v[0];
    let p: Vec<(f64, f64)> = v.iter().map(|&x| ((x - o).dot(e1), (x - o).dot(e2))).collect();
    let k = p.len();
    let (mut a2, mut cx, mut cy, mut iuu, mut ivv, mut iuv) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..k {
        let (x0, y0) = p[i];
        let (x1, y1) = p[(i + 1) % k];
        let c = x0 * y1 - x1 * y0;
        a2 += c;
        cx += (x0 + x1) * c;
        cy += (y0 + y1) * c;
        ivv += (y0 * y0 + y0 * y1 + y1 * y1) * c; // ∫ y²
        iuu += (x0 * x0 + x0 * x1 + x1 * x1) * c; // ∫ x²
        iuv += (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * c; // 2∫xy*12/...
    }
    let area = 0.5 * a2;
    let (ux, uy) = (cx / (3.0 * a2), cy / (3.0 * a2));
    let iuu_c = iuu / 12.0 - area * ux * ux;
    let ivv_c = ivv / 12.0 - area * uy * uy;
    let iuv_c = iuv / 24.0 - area * ux * uy;
    let centroid = o + e1 * ux + e2 * uy;
    (area, centroid, e1, e2, [iuu_c, ivv_c, iuv_c])
}

#[derive(Clone, Debug)]
pub struct VCell3 {
    pub seed: V3,
    pub poly: Polyhedron,
    pub props: MassProps,
}

/// Brute-force-ordered clipping; adequate for the ~10²–10³ cells of E10.
pub fn voronoi_box(seeds: &[V3], lo: V3, hi: V3) -> Vec<VCell3> {
    let eps = 1e-12 * (hi - lo).norm();
    let base = box_poly(lo, hi);
    seeds
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let mut order: Vec<(f64, usize)> =
                seeds.iter().enumerate().filter(|&(j, _)| j != i).map(|(j, &t)| ((t - s).norm(), j)).collect();
            order.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut poly = base.clone();
            for &(d, j) in &order {
                if d > 2.0 * poly.max_radius(s) {
                    break;
                }
                let m = (s + seeds[j]) * 0.5;
                let n = (seeds[j] - s).normalized();
                poly = poly.clip(m, n, FaceTag::Cell(j), eps);
            }
            let props = poly.mass_props();
            VCell3 { seed: s, poly, props }
        })
        .collect()
}

/// Same cells as `voronoi_box`, with candidate seeds from a uniform grid: for each cell
/// the candidates within radius R are taken in the same (distance, index) order and the
/// same stop rule is applied; R grows until the stop rule is reached inside it, so the
/// clipping sequence — and the result — is identical to the brute-force version.
pub fn voronoi_box_grid(seeds: &[V3], lo: V3, hi: V3, spacing: f64) -> Vec<VCell3> {
    let eps = 1e-12 * (hi - lo).norm();
    let base = box_poly(lo, hi);
    let cs = spacing.max(1e-12);
    let dims = [
        (((hi.x - lo.x) / cs).ceil() as usize).max(1),
        (((hi.y - lo.y) / cs).ceil() as usize).max(1),
        (((hi.z - lo.z) / cs).ceil() as usize).max(1),
    ];
    let cell_of = |p: V3| {
        [
            (((p.x - lo.x) / cs) as usize).min(dims[0] - 1),
            (((p.y - lo.y) / cs) as usize).min(dims[1] - 1),
            (((p.z - lo.z) / cs) as usize).min(dims[2] - 1),
        ]
    };
    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); dims[0] * dims[1] * dims[2]];
    for (i, &s) in seeds.iter().enumerate() {
        let c = cell_of(s);
        grid[(c[2] * dims[1] + c[1]) * dims[0] + c[0]].push(i);
    }
    seeds
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let c = cell_of(s);
            let mut ring = 2usize;
            loop {
                let r = ring as f64 * cs;
                let mut order: Vec<(f64, usize)> = Vec::new();
                for z in c[2].saturating_sub(ring)..(c[2] + ring + 1).min(dims[2]) {
                    for y in c[1].saturating_sub(ring)..(c[1] + ring + 1).min(dims[1]) {
                        for x in c[0].saturating_sub(ring)..(c[0] + ring + 1).min(dims[0]) {
                            for &j in &grid[(z * dims[1] + y) * dims[0] + x] {
                                if j != i {
                                    let d = (seeds[j] - s).norm();
                                    if d <= r {
                                        order.push((d, j));
                                    }
                                }
                            }
                        }
                    }
                }
                order.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mut poly = base.clone();
                let mut stopped = false;
                for &(d, j) in &order {
                    if d > 2.0 * poly.max_radius(s) {
                        stopped = true;
                        break;
                    }
                    let m = (s + seeds[j]) * 0.5;
                    let n = (seeds[j] - s).normalized();
                    poly = poly.clip(m, n, FaceTag::Cell(j), eps);
                }
                // all seeds beyond r are farther than r: complete if 2·rmax ≤ r
                let covers_domain = ring >= dims[0].max(dims[1]).max(dims[2]);
                if stopped || 2.0 * poly.max_radius(s) <= r || covers_domain {
                    let props = poly.mass_props();
                    return VCell3 { seed: s, poly, props };
                }
                ring *= 2;
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::rsa::rsa_box;

    #[test]
    fn volumes_sum() {
        let mut rng = Rng::new(3);
        let lo = V3::ZERO;
        let hi = V3::new(1.0, 1.0, 0.6);
        let s = rsa_box(lo, hi, 0.2, &mut rng, 2000);
        let cells = voronoi_box(&s, lo, hi);
        let v: f64 = cells.iter().map(|c| c.props.volume).sum();
        assert!((v - 0.6).abs() < 1e-10, "{v}");
    }

    #[test]
    fn grid_version_identical() {
        let mut rng = Rng::new(5);
        let lo = V3::ZERO;
        let hi = V3::new(1.2, 1.0, 0.8);
        let s = rsa_box(lo, hi, 0.15, &mut rng, 2000);
        let a = voronoi_box(&s, lo, hi);
        let b = voronoi_box_grid(&s, lo, hi, 0.15);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.props.volume.to_bits(), y.props.volume.to_bits());
            assert_eq!(x.poly.faces.len(), y.poly.faces.len());
        }
    }

    #[test]
    fn cube_inertia() {
        let p = box_poly(V3::ZERO, V3::new(1.0, 2.0, 3.0));
        let m = p.mass_props();
        assert!((m.volume - 6.0).abs() < 1e-12);
        // I_xx = V (b² + c²)/12
        assert!((m.inertia[0][0] - 6.0 * (4.0 + 9.0) / 12.0).abs() < 1e-10);
        assert!(m.inertia[0][1].abs() < 1e-10);
    }

    #[test]
    fn planar_square() {
        let v = vec![V3::new(0.0, 0.0, 0.0), V3::new(2.0, 0.0, 0.0), V3::new(2.0, 1.0, 0.0), V3::new(0.0, 1.0, 0.0)];
        let (a, c, e1, _e2, i) = planar_props(&v, V3::new(0.0, 0.0, 1.0));
        assert!((a - 2.0).abs() < 1e-12);
        assert!((c - V3::new(1.0, 0.5, 0.0)).norm() < 1e-12);
        // polar = (b h³ + h b³)/12 = (2 + 8)/12
        assert!((i[0] + i[1] - 10.0 / 12.0).abs() < 1e-12, "{:?} {:?}", i, e1);
    }
}
