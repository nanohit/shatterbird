//! Voronoi tessellation of a convex domain by half-plane clipping.
//!
//! Each cell starts as the domain polygon and is clipped by the bisectors of
//! seeds in order of distance. Clipping stops once the next candidate is farther
//! than twice the current maximum vertex radius: no farther seed can cut the cell.
//! Every edge keeps a tag saying what lies across it (another cell or a domain edge).

use crate::polygon::{poly_props, PolyProps};
use crate::V2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeTag {
    Cell(usize),
    Boundary(usize),
}

#[derive(Clone, Debug)]
pub struct VCell {
    pub seed: V2,
    /// CCW vertices; edge k goes from verts[k] to verts[k+1] and has tags[k].
    pub verts: Vec<V2>,
    pub tags: Vec<EdgeTag>,
    pub props: PolyProps,
}

#[derive(Clone, Debug)]
pub struct VFacet {
    pub a: usize,
    pub b: usize,
    pub p0: V2,
    pub p1: V2,
}

#[derive(Clone, Debug)]
pub struct Tessellation {
    pub domain: Vec<V2>,
    pub cells: Vec<VCell>,
    /// Interior facets, a < b, oriented with cell a's edge direction.
    pub facets: Vec<VFacet>,
    /// Largest length mismatch between the two sides of a facet (diagnostic).
    pub max_mismatch: f64,
}

fn clip(verts: &[V2], tags: &[EdgeTag], m: V2, nrm: V2, new_tag: EdgeTag, eps: f64) -> (Vec<V2>, Vec<EdgeTag>) {
    let n = verts.len();
    let f: Vec<f64> = verts.iter().map(|&v| (v - m).dot(nrm)).collect();
    if f.iter().all(|&x| x <= eps) {
        return (verts.to_vec(), tags.to_vec());
    }
    let mut ov = Vec::with_capacity(n + 1);
    let mut ot = Vec::with_capacity(n + 1);
    for k in 0..n {
        let k1 = (k + 1) % n;
        let (a, b) = (verts[k], verts[k1]);
        let (fa, fb) = (f[k], f[k1]);
        let ina = fa <= eps;
        let inb = fb <= eps;
        if ina {
            ov.push(a);
            ot.push(tags[k]);
        }
        if ina != inb {
            let t = fa / (fa - fb);
            let p = a + (b - a) * t;
            if ina {
                ov.push(p);
                ot.push(new_tag);
            } else {
                ov.push(p);
                ot.push(tags[k]);
            }
        }
    }
    // Remove degenerate edges (coincident consecutive vertices).
    let mut changed = true;
    while changed && ov.len() > 2 {
        changed = false;
        let n = ov.len();
        for k in 0..n {
            let k1 = (k + 1) % n;
            if (ov[k1] - ov[k]).norm() < eps {
                ot[k] = ot[k1];
                ov.remove(k1);
                ot.remove(k1);
                changed = true;
                break;
            }
        }
    }
    (ov, ot)
}

/// `domain` must be convex and counter-clockwise.
pub fn voronoi_convex(seeds: &[V2], domain: &[V2]) -> Tessellation {
    let n = seeds.len();
    let (mut lo, mut hi) = (domain[0], domain[0]);
    for &d in domain {
        lo = V2::new(lo.x.min(d.x), lo.y.min(d.y));
        hi = V2::new(hi.x.max(d.x), hi.y.max(d.y));
    }
    let diag = (hi - lo).norm();
    let eps = 1e-12 * diag;
    // Uniform grid for candidate lookup.
    let mean_spacing = ((hi.x - lo.x) * (hi.y - lo.y) / n.max(1) as f64).sqrt();
    let cs = mean_spacing.max(1e-12);
    let nx = ((hi.x - lo.x) / cs).ceil().max(1.0) as usize;
    let ny = ((hi.y - lo.y) / cs).ceil().max(1.0) as usize;
    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); nx * ny];
    let gidx = |p: V2| {
        let gx = (((p.x - lo.x) / cs) as usize).min(nx - 1);
        let gy = (((p.y - lo.y) / cs) as usize).min(ny - 1);
        (gx, gy)
    };
    for (i, &s) in seeds.iter().enumerate() {
        let (gx, gy) = gidx(s);
        grid[gy * nx + gx].push(i);
    }

    let dom_tags: Vec<EdgeTag> = (0..domain.len()).map(EdgeTag::Boundary).collect();
    let mut cells = Vec::with_capacity(n);
    for i in 0..n {
        let s = seeds[i];
        let (gx, gy) = gidx(s);
        let mut verts = domain.to_vec();
        let mut tags = dom_tags.clone();
        let mut ring = 0usize;
        let mut done: Vec<usize> = Vec::new();
        loop {
            // Candidates in the square ring at Chebyshev distance `ring`.
            let mut cand: Vec<(f64, usize)> = Vec::new();
            let r = ring as i64;
            for iy in (gy as i64 - r)..=(gy as i64 + r) {
                for ix in (gx as i64 - r)..=(gx as i64 + r) {
                    if ix < 0 || iy < 0 || ix >= nx as i64 || iy >= ny as i64 {
                        continue;
                    }
                    if (ix - gx as i64).abs() != r && (iy - gy as i64).abs() != r {
                        continue;
                    }
                    for &j in &grid[iy as usize * nx + ix as usize] {
                        if j != i {
                            cand.push(((seeds[j] - s).norm(), j));
                        }
                    }
                }
            }
            cand.sort_by(|a, b| a.partial_cmp(b).unwrap());
            done.extend(cand.iter().map(|c| c.1));
            for &(_, j) in &cand {
                let m = (s + seeds[j]) * 0.5;
                let nrm = (seeds[j] - s).normalized();
                let (v, t) = clip(&verts, &tags, m, nrm, EdgeTag::Cell(j), eps);
                verts = v;
                tags = t;
            }
            // Seeds outside the rings searched so far are at least ring*cs away.
            let rmax = verts.iter().map(|&v| (v - s).norm()).fold(0.0, f64::max);
            let searched = ring as f64 * cs;
            let covers_all = ring >= nx.max(ny);
            if 2.0 * rmax <= searched || covers_all {
                break;
            }
            ring += 1;
        }
        // Clip order within a ring is by distance; across rings a later clip by a
        // nearer seed cannot occur because rings are processed in order and the
        // result of convex clipping is order-independent anyway.
        let props = poly_props(&verts);
        cells.push(VCell { seed: s, verts, tags, props });
    }

    // Facets from cell a's side; check the matching edge on b's side.
    let mut facets = Vec::new();
    let mut max_mismatch: f64 = 0.0;
    for a in 0..n {
        let c = &cells[a];
        let m = c.verts.len();
        for k in 0..m {
            if let EdgeTag::Cell(b) = c.tags[k] {
                if a < b {
                    let p0 = c.verts[k];
                    let p1 = c.verts[(k + 1) % m];
                    let la = (p1 - p0).norm();
                    let cb = &cells[b];
                    let mb = cb.verts.len();
                    let lb = (0..mb)
                        .filter(|&q| cb.tags[q] == EdgeTag::Cell(a))
                        .map(|q| (cb.verts[(q + 1) % mb] - cb.verts[q]).norm())
                        .fold(0.0, f64::max);
                    max_mismatch = max_mismatch.max((la - lb).abs());
                    facets.push(VFacet { a, b, p0, p1 });
                }
            }
        }
        // Edges tagged with a cell that does not list `a` (numerically degenerate,
        // length ~eps) are dropped by construction above only if a > b; record them.
        for k in 0..m {
            if let EdgeTag::Cell(b) = c.tags[k] {
                if b < a && !cells[b].tags.contains(&EdgeTag::Cell(a)) {
                    let l = (c.verts[(k + 1) % m] - c.verts[k]).norm();
                    max_mismatch = max_mismatch.max(l);
                }
            }
        }
    }
    Tessellation { domain: domain.to_vec(), cells, facets, max_mismatch }
}

pub fn rect(lo: V2, hi: V2) -> Vec<V2> {
    vec![lo, V2::new(hi.x, lo.y), hi, V2::new(lo.x, hi.y)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::rsa::rsa_rect;

    #[test]
    fn areas_sum_and_facets_match() {
        let mut rng = Rng::new(7);
        let lo = V2::new(0.0, 0.0);
        let hi = V2::new(1.0, 0.5);
        let seeds = rsa_rect(lo, hi, 0.05, &mut rng, 5000);
        let t = voronoi_convex(&seeds, &rect(lo, hi));
        let a: f64 = t.cells.iter().map(|c| c.props.area).sum();
        assert!((a - 0.5).abs() < 1e-12, "area {a}");
        assert!(t.max_mismatch < 1e-10, "mismatch {}", t.max_mismatch);
        // Each facet is equidistant from both seeds.
        for f in &t.facets {
            let sa = t.cells[f.a].seed;
            let sb = t.cells[f.b].seed;
            for p in [f.p0, f.p1] {
                assert!(((p - sa).norm() - (p - sb).norm()).abs() < 1e-10);
            }
        }
    }
}
