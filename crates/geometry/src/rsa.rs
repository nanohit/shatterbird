//! Random sequential addition with a minimum distance (hard-core point process).
//! Stops after `max_fail` consecutive rejected trials, which approaches the jamming
//! density (~0.547 area fraction of disks of diameter `dmin` in 2D) from below.

use crate::rng::Rng;
use crate::{V2, V3};

pub fn rsa_rect(lo: V2, hi: V2, dmin: f64, rng: &mut Rng, max_fail: usize) -> Vec<V2> {
    rsa_rect_with(lo, hi, dmin, rng, max_fail, &|_| true, Vec::new())
}

/// RSA continued from `init` (assumed to respect `dmin` and `keep`), where a trial is
/// also rejected unless `keep(p)`. Each trial draws the same random numbers as
/// `rsa_rect`, so `keep ≡ true` with empty `init` reproduces it exactly.
pub fn rsa_rect_with(lo: V2, hi: V2, dmin: f64, rng: &mut Rng, max_fail: usize, keep: &dyn Fn(V2) -> bool, init: Vec<V2>) -> Vec<V2> {
    let cs = dmin / std::f64::consts::SQRT_2;
    let nx = ((hi.x - lo.x) / cs).ceil().max(1.0) as usize;
    let ny = ((hi.y - lo.y) / cs).ceil().max(1.0) as usize;
    let mut grid: Vec<i64> = vec![-1; nx * ny];
    let cell = |p: V2| {
        let gx = (((p.x - lo.x) / cs) as usize).min(nx - 1);
        let gy = (((p.y - lo.y) / cs) as usize).min(ny - 1);
        (gx, gy)
    };
    let mut pts: Vec<V2> = init;
    for (k, &p) in pts.iter().enumerate() {
        let (gx, gy) = cell(p);
        grid[gy * nx + gx] = k as i64;
    }
    let d2 = dmin * dmin;
    let mut fails = 0;
    while fails < max_fail {
        let p = V2::new(rng.range(lo.x, hi.x), rng.range(lo.y, hi.y));
        let (gx, gy) = cell(p);
        let mut ok = keep(p);
        'outer: for iy in gy.saturating_sub(2)..(gy + 3).min(ny) {
            if !ok {
                break;
            }
            for ix in gx.saturating_sub(2)..(gx + 3).min(nx) {
                let k = grid[iy * nx + ix];
                if k >= 0 && (pts[k as usize] - p).norm2() < d2 {
                    ok = false;
                    break 'outer;
                }
            }
        }
        if ok {
            grid[gy * nx + gx] = pts.len() as i64;
            pts.push(p);
            fails = 0;
        } else {
            fails += 1;
        }
    }
    pts
}

pub fn rsa_box(lo: V3, hi: V3, dmin: f64, rng: &mut Rng, max_fail: usize) -> Vec<V3> {
    rsa_box_with(lo, hi, dmin, rng, max_fail, &|_| true, Vec::new())
}

/// 3D counterpart of `rsa_rect_with`.
pub fn rsa_box_with(lo: V3, hi: V3, dmin: f64, rng: &mut Rng, max_fail: usize, keep: &dyn Fn(V3) -> bool, init: Vec<V3>) -> Vec<V3> {
    let cs = dmin / 3f64.sqrt();
    let n = [
        ((hi.x - lo.x) / cs).ceil().max(1.0) as usize,
        ((hi.y - lo.y) / cs).ceil().max(1.0) as usize,
        ((hi.z - lo.z) / cs).ceil().max(1.0) as usize,
    ];
    let mut grid: Vec<i64> = vec![-1; n[0] * n[1] * n[2]];
    let cell = |p: V3| {
        [
            (((p.x - lo.x) / cs) as usize).min(n[0] - 1),
            (((p.y - lo.y) / cs) as usize).min(n[1] - 1),
            (((p.z - lo.z) / cs) as usize).min(n[2] - 1),
        ]
    };
    let mut pts: Vec<V3> = init;
    for (k, &p) in pts.iter().enumerate() {
        let g = cell(p);
        grid[(g[2] * n[1] + g[1]) * n[0] + g[0]] = k as i64;
    }
    let d2 = dmin * dmin;
    let mut fails = 0;
    while fails < max_fail {
        let p = V3::new(rng.range(lo.x, hi.x), rng.range(lo.y, hi.y), rng.range(lo.z, hi.z));
        let g = cell(p);
        let mut ok = keep(p);
        'outer: for iz in g[2].saturating_sub(2)..(g[2] + 3).min(n[2]) {
            if !ok {
                break;
            }
            for iy in g[1].saturating_sub(2)..(g[1] + 3).min(n[1]) {
                for ix in g[0].saturating_sub(2)..(g[0] + 3).min(n[0]) {
                    let k = grid[(iz * n[1] + iy) * n[0] + ix];
                    if k >= 0 && (pts[k as usize] - p).norm2() < d2 {
                        ok = false;
                        break 'outer;
                    }
                }
            }
        }
        if ok {
            grid[(g[2] * n[1] + g[1]) * n[0] + g[0]] = pts.len() as i64;
            pts.push(p);
            fails = 0;
        } else {
            fails += 1;
        }
    }
    pts
}
