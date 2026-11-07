use crate::V2;

/// Area, centroid and polar second moment about the centroid (per unit thickness).
#[derive(Clone, Copy, Debug)]
pub struct PolyProps {
    pub area: f64,
    pub centroid: V2,
    /// ∫ |r - c|² dA
    pub polar: f64,
}

pub fn poly_props(v: &[V2]) -> PolyProps {
    let n = v.len();
    // Shift to the first vertex to limit cancellation far from the origin.
    let o = v[0];
    let (mut a2, mut cx, mut cy, mut ixx, mut iyy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for k in 0..n {
        let p = v[k] - o;
        let q = v[(k + 1) % n] - o;
        let c = p.cross(q);
        a2 += c;
        cx += (p.x + q.x) * c;
        cy += (p.y + q.y) * c;
        ixx += (p.y * p.y + p.y * q.y + q.y * q.y) * c;
        iyy += (p.x * p.x + p.x * q.x + q.x * q.x) * c;
    }
    let area = 0.5 * a2;
    let c = V2::new(cx / (3.0 * a2), cy / (3.0 * a2));
    let polar_o = (ixx + iyy) / 12.0;
    PolyProps { area, centroid: c + o, polar: polar_o - area * c.norm2() }
}

/// Distance from a point to a segment and the closest point.
pub fn closest_on_segment(p: V2, a: V2, b: V2) -> (f64, V2) {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.norm2()).clamp(0.0, 1.0);
    let c = a + ab * t;
    ((p - c).norm(), c)
}

/// Proper intersection test of two closed segments.
pub fn segments_intersect(a: V2, b: V2, c: V2, d: V2) -> bool {
    let o1 = (b - a).cross(c - a);
    let o2 = (b - a).cross(d - a);
    let o3 = (d - c).cross(a - c);
    let o4 = (d - c).cross(b - c);
    (o1 * o2 <= 0.0) && (o3 * o4 <= 0.0)
}

pub fn point_in_convex(p: V2, poly: &[V2]) -> bool {
    let n = poly.len();
    (0..n).all(|k| (poly[(k + 1) % n] - poly[k]).cross(p - poly[k]) >= 0.0)
}
