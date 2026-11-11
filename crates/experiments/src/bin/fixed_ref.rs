//! Phase 0b / 3 §7 — CPU reference of the fixed-point E-B-H substep (gpu-probe/fixed.js).
//!
//! Reads the mesh binary, the integer constants the probe page computed (so both sides use
//! identical bits) and the GPU state after N substeps; repeats the integer arithmetic with
//! native i64 (wrapping where the WGSL emulation wraps, the same round-half-up narrowing and
//! saturation), compares bit for bit, and measures accuracy against an f64 run of the same
//! law and the headroom of every format.
//! Usage: `fixed_ref <dir> <label> <steps> <tag>`.

use experiments::write_json;
use serde_json::json;

const EXP: &str = "phase0b/architecture-selection/numeric";

fn rd_u32(b: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap());
    *o += 4;
    v
}

struct Mesh {
    nc: usize,
    nf: usize,
    cells: Vec<f32>,
    fidx: Vec<u32>,
    fdata: Vec<f32>,
    ptr: Vec<u32>,
    idx: Vec<u32>,
}

fn mesh(path: &str) -> Mesh {
    let b = std::fs::read(path).unwrap();
    let mut o = 0;
    let h: Vec<u32> = (0..5).map(|_| rd_u32(&b, &mut o)).collect();
    assert_eq!(h[0], 0x3153_4253);
    let (nc, nf, ncsr) = (h[1] as usize, h[2] as usize, h[4] as usize);
    let f = |o: &mut usize, n: usize| -> Vec<f32> { (0..n).map(|_| f32::from_bits(rd_u32(&b, o))).collect() };
    let cells = f(&mut o, 25 * nc);
    let fidx: Vec<u32> = (0..2 * nf).map(|_| rd_u32(&b, &mut o)).collect();
    let fdata = f(&mut o, 20 * nf);
    let ptr: Vec<u32> = (0..nc + 1).map(|_| rd_u32(&b, &mut o)).collect();
    let idx: Vec<u32> = (0..ncsr).map(|_| rd_u32(&b, &mut o)).collect();
    Mesh { nc, nf, cells, fidx, fdata, ptr, idx }
}

#[derive(Default)]
struct Head {
    flag: bool,
    /// first overflow: (shift, value before narrowing)
    first: Option<(u32, i64)>,
    /// max |value| per quantity, in units of its own LSB
    mx: [i64; 8],
}

impl Head {
    /// floor(x·2^-s + ½) with saturation to i32 (as `rs` in fixed.js)
    fn rs(&mut self, x: i64, s: u32) -> i32 {
        let y = x.wrapping_add(1i64 << (s - 1)) >> s;
        if y > i32::MAX as i64 || y < i32::MIN as i64 {
            self.flag = true;
            if self.first.is_none() {
                self.first = Some((s, y));
            }
            return if y < 0 { -2147483647 } else { 2147483647 };
        }
        y as i32
    }
}

fn m(a: i32, b: i32) -> i64 {
    a as i64 * b as i64
}

fn isqrt(q0: u32) -> u32 {
    let (mut q, mut r, mut bit) = (q0, 0u32, 1u32 << 30);
    while bit > q {
        bit >>= 2;
    }
    while bit != 0 {
        if q >= r + bit {
            q -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    r
}

type V = [i32; 3];

fn sub(a: V, b: V) -> V {
    [a[0].wrapping_sub(b[0]), a[1].wrapping_sub(b[1]), a[2].wrapping_sub(b[2])]
}
fn add(a: V, b: V) -> V {
    [a[0].wrapping_add(b[0]), a[1].wrapping_add(b[1]), a[2].wrapping_add(b[2])]
}
fn dot3(a: V, b: V) -> i64 {
    m(a[0], b[0]).wrapping_add(m(a[1], b[1])).wrapping_add(m(a[2], b[2]))
}

fn crs(h: &mut Head, a: V, b: V, s: u32) -> V {
    [
        h.rs(m(a[1], b[2]).wrapping_sub(m(a[2], b[1])), s),
        h.rs(m(a[2], b[0]).wrapping_sub(m(a[0], b[2])), s),
        h.rs(m(a[0], b[1]).wrapping_sub(m(a[1], b[0])), s),
    ]
}

fn comb(h: &mut Head, c: V, x: V, y: V, z: V) -> V {
    let mut o = [0; 3];
    for k in 0..3 {
        o[k] = h.rs(m(c[0], x[k]).wrapping_add(m(c[1], y[k])).wrapping_add(m(c[2], z[k])), 30);
    }
    o
}

struct Consts {
    g: Vec<i32>,
    fc: Vec<i32>,
    cc: Vec<i32>,
    init: Vec<i32>,
}

/// One integer substep; returns (state, κ).
fn step_fix(ms: &Mesh, k: &Consts, s: &[i32], kin: &[u32], h: &mut Head) -> (Vec<i32>, Vec<u32>) {
    let (sk, sr, sm, si, sh, wc, ch) = (k.g[0] as u32, k.g[1] as u32, k.g[2] as u32, k.g[3] as u32, k.g[4] as u32, k.g[5] as u32, k.g[6]);
    let ld = |i: usize, o: usize| -> V { [s[12 * i + o], s[12 * i + o + 1], s[12 * i + o + 2]] };
    let mut out = vec![0i32; 12 * ms.nc];
    let mut kout = kin.to_vec();
    for i in 0..ms.nc {
        let mut ff: V = [0; 3];
        let mut tt: V = [0; 3];
        for q in ms.ptr[i] as usize..ms.ptr[i + 1] as usize {
            let e = ms.idx[q];
            let fi = (e & 0x7fff_ffff) as usize;
            let side_b = e >> 31 == 1;
            let (a, bb) = (ms.fidx[2 * fi] as usize, ms.fidx[2 * fi + 1]);
            let (ub, wb) = if bb != u32::MAX { (ld(bb as usize, 0), ld(bb as usize, 3)) } else { ([0; 3], [0; 3]) };
            let (ua, wa) = (ld(a, 0), ld(a, 3));
            let b = 24 * fi;
            let fv = |o: usize| -> V { [k.fc[b + o], k.fc[b + o + 1], k.fc[b + o + 2]] };
            let (n, p1, p2, ra, rb) = (fv(0), fv(3), fv(6), fv(9), fv(12));
            let (kn, kt, kr1, kr2, ktor) = (k.fc[b + 15], k.fc[b + 16], k.fc[b + 17], k.fc[b + 18], k.fc[b + 19]);
            let (d0, gam) = (k.fc[b + 20] as u32, k.fc[b + 21] as u32);
            let cb = crs(h, wb, rb, 30);
            let ca = crs(h, wa, ra, 30);
            let jump = sub(add(ub, cb), add(ua, ca));
            let dw = sub(wb, wa);
            let dn = h.rs(dot3(n, jump), 30);
            let t1 = h.rs(dot3(p1, jump), 30);
            let t2 = h.rs(dot3(p2, jump), 30);
            h.mx[0] = h.mx[0].max(dn.unsigned_abs() as i64).max(t1.unsigned_abs() as i64).max(t2.unsigned_abs() as i64);
            let dp = dn.max(0);
            let (c0, c1, c2) = (dp.min(32767) as u32, t1.unsigned_abs().min(32767), t2.unsigned_abs().min(32767));
            let tq = ((c1 * c1 + c2 * c2) as u64 * gam as u64) as i64;
            let qv = c0 * c0 + h.rs(tq, 16) as u32;
            let kap = kin[fi].max(isqrt(qv));
            let om: i32 = if kap >= wc {
                0
            } else if kap > d0 {
                ((d0 as u64 * (wc - kap) as u64 * 65536) / (kap as u64 * (wc - d0) as u64)) as i32
            } else {
                65536
            };
            let sf = 22 - sk;
            let fnt = h.rs(m(kn, dp), sf);
            let fnn = h.rs(m(om, fnt), 16).wrapping_add(h.rs(m(kn, dn.min(0)), sf));
            let ft1 = { let x = h.rs(m(kt, t1), sf); h.rs(m(om, x), 16) };
            let ft2 = { let x = h.rs(m(kt, t2), sf); h.rs(m(om, x), 16) };
            let f = comb(h, [fnn, ft1, ft2], n, p1, p2);
            let smr = 24 - sr;
            let mut mm = [0i32; 3];
            for (j, (kr, p)) in [(kr1, p1), (kr2, p2), (ktor, n)].into_iter().enumerate() {
                let th = h.rs(dot3(p, dw), 30);
                let x = h.rs(m(kr, th), smr);
                mm[j] = h.rs(m(om, x), 16);
            }
            let mv = comb(h, mm, p1, p2, n);
            for c in 0..3 {
                h.mx[1] = h.mx[1].max(f[c].unsigned_abs() as i64);
            }
            if side_b {
                ff = sub(ff, f);
                let t = crs(h, rb, f, 28);
                tt = sub(sub(tt, t), mv);
            } else {
                ff = add(ff, f);
                let t = crs(h, ra, f, 28);
                tt = add(add(tt, t), mv);
                kout[fi] = kap;
            }
        }
        let c = 16 * i;
        let cm = k.cc[c];
        let ci = |r: usize| -> V { [k.cc[c + 1 + 3 * r], k.cc[c + 2 + 3 * r], k.cc[c + 3 + 3 * r]] };
        let mut v = ld(i, 6);
        let mut w = ld(i, 9);
        let mut u = ld(i, 0);
        let mut th = ld(i, 3);
        for r in 0..3 {
            v[r] = v[r].wrapping_add(h.rs(m(ff[r], cm), sm));
            w[r] = w[r].wrapping_add(h.rs(dot3(ci(r), tt), si));
        }
        for r in 0..3 {
            u[r] = u[r].wrapping_add(h.rs(m(v[r], ch), sh));
            th[r] = th[r].wrapping_add(h.rs(m(w[r], ch), sh));
            h.mx[2] = h.mx[2].max(u[r].unsigned_abs() as i64);
            h.mx[3] = h.mx[3].max(v[r].unsigned_abs() as i64);
            h.mx[4] = h.mx[4].max(th[r].unsigned_abs() as i64);
            h.mx[5] = h.mx[5].max(w[r].unsigned_abs() as i64);
            h.mx[6] = h.mx[6].max(ff[r].unsigned_abs() as i64);
            h.mx[7] = h.mx[7].max(tt[r].unsigned_abs() as i64);
        }
        out[12 * i..12 * i + 12].copy_from_slice(&[u[0], u[1], u[2], th[0], th[1], th[2], v[0], v[1], v[2], w[0], w[1], w[2]]);
    }
    (out, kout)
}

/// f64 run of the same law from the f32 mesh constants (as cpuLeapfrogH in probe.js).
fn run_f64(ms: &Mesh, dt: f64, steps: usize) -> (Vec<f64>, Vec<f64>) {
    let mut s = vec![0.0f64; 12 * ms.nc];
    for i in 0..ms.nc {
        for k in 0..3 {
            s[12 * i + 6 + k] = ms.cells[25 * i + 22 + k] as f64;
        }
    }
    let mut kap = vec![0.0f64; ms.nf];
    let mut ff = vec![0.0f64; 12 * ms.nf];
    let cr = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let d3 = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for _ in 0..steps {
        for f in 0..ms.nf {
            let (a, b) = (ms.fidx[2 * f] as usize, ms.fidx[2 * f + 1]);
            let d = 20 * f;
            let g = |k: usize| [ms.fdata[d + k] as f64, ms.fdata[d + k + 1] as f64, ms.fdata[d + k + 2] as f64];
            let (n, p1, p2, ra, rb) = (g(0), g(3), g(6), g(9), g(12));
            let (kn, kt) = (ms.fdata[d + 15] as f64, ms.fdata[d + 16] as f64);
            let st = |i: usize, o: usize| [s[12 * i + o], s[12 * i + o + 1], s[12 * i + o + 2]];
            let (ua, wa) = (st(a, 0), st(a, 3));
            let (ub, wb) = if b != u32::MAX { (st(b as usize, 0), st(b as usize, 3)) } else { ([0.0; 3], [0.0; 3]) };
            let (cb, ca) = (cr(wb, rb), cr(wa, ra));
            let j = [ub[0] + cb[0] - ua[0] - ca[0], ub[1] + cb[1] - ua[1] - ca[1], ub[2] + cb[2] - ua[2] - ca[2]];
            let dw = [wb[0] - wa[0], wb[1] - wa[1], wb[2] - wa[2]];
            let (dn, t1, t2) = (d3(n, j), d3(p1, j), d3(p2, j));
            let ell = ((ra[0] - rb[0]).powi(2) + (ra[1] - rb[1]).powi(2) + (ra[2] - rb[2]).powi(2)).sqrt().max(1e-6);
            let (d0, wc, gam, dp) = (1e-4 * ell, 6.6666667e-5, kt / kn, dn.max(0.0));
            let deq = (dp * dp + gam * (t1 * t1 + t2 * t2)).sqrt();
            let kp = kap[f].max(deq);
            kap[f] = kp;
            let dd = if kp >= wc { 1.0 } else if kp > d0 { (kp - d0) / (kp * (1.0 - d0 / wc)) } else { 0.0 };
            let om = 1.0 - dd;
            let fnn = om * kn * dp + kn * dn.min(0.0);
            let fv: [f64; 3] = std::array::from_fn(|k| fnn * n[k] + om * kt * (t1 * p1[k] + t2 * p2[k]));
            let (m1, m2, m3) = (ms.fdata[d + 17] as f64 * d3(p1, dw), ms.fdata[d + 18] as f64 * d3(p2, dw), ms.fdata[d + 19] as f64 * d3(n, dw));
            let mv: [f64; 3] = std::array::from_fn(|k| om * (m1 * p1[k] + m2 * p2[k] + m3 * n[k]));
            let (ta, tb) = (cr(ra, fv), cr(rb, fv));
            for k in 0..3 {
                ff[12 * f + k] = fv[k];
                ff[12 * f + 3 + k] = ta[k] + mv[k];
                ff[12 * f + 6 + k] = -fv[k];
                ff[12 * f + 9 + k] = -tb[k] - mv[k];
            }
        }
        for i in 0..ms.nc {
            let (mut fx, mut tx) = ([0.0; 3], [0.0; 3]);
            for q in ms.ptr[i] as usize..ms.ptr[i + 1] as usize {
                let e = ms.idx[q];
                let o = 12 * (e & 0x7fff_ffff) as usize + if e >> 31 == 1 { 6 } else { 0 };
                for k in 0..3 {
                    fx[k] += ff[o + k];
                    tx[k] += ff[o + 3 + k];
                }
            }
            let c = 25 * i;
            let mass = ms.cells[c + 3] as f64;
            for k in 0..3 {
                s[12 * i + 6 + k] += dt * fx[k] / mass;
            }
            for k in 0..3 {
                let w: f64 = (0..3).map(|l| ms.cells[c + 13 + 3 * k + l] as f64 * tx[l]).sum();
                s[12 * i + 9 + k] += dt * w;
            }
            for k in 0..6 {
                s[12 * i + k] += dt * s[12 * i + 6 + k];
            }
        }
    }
    (s, kap)
}

fn fnv(st: &[i32], kp: &[u32]) -> String {
    let mut h: u32 = 2_166_136_261;
    for &x in st {
        h = (h ^ x as u32).wrapping_mul(16_777_619);
    }
    for &x in kp {
        h = (h ^ x).wrapping_mul(16_777_619);
    }
    format!("{h:x}")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, label, steps, tag) = (&a[1], &a[2], a[3].parse::<usize>().unwrap(), &a[4]);
    let ms = mesh(&format!("{dir}/mesh_{label}.bin"));
    let cb = std::fs::read(format!("{dir}/out/fixed_consts_{label}_{tag}.bin")).unwrap();
    let ints: Vec<i32> = cb.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let (nc, nf) = (ms.nc, ms.nf);
    let k = Consts { g: ints[0..16].to_vec(), fc: ints[16..16 + 24 * nf].to_vec(), cc: ints[16 + 24 * nf..16 + 24 * nf + 16 * nc].to_vec(), init: ints[16 + 24 * nf + 16 * nc..].to_vec() };
    assert_eq!(k.init.len(), 12 * nc);
    let gb = std::fs::read(format!("{dir}/out/fixed_state_{label}_{tag}.bin")).unwrap();
    let gi: Vec<u32> = gb.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let gst: Vec<i32> = gi[..12 * nc].iter().map(|&x| x as i32).collect();
    let gkp = gi[12 * nc..].to_vec();
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/index.json")).unwrap()).unwrap();
    let dt = idx["dt_contract_s"].as_f64().unwrap();
    let t0 = std::time::Instant::now();
    let mut h = Head::default();
    let mut s = k.init.clone();
    let mut kp = vec![0u32; nf];
    for _ in 0..steps {
        let (s2, k2) = step_fix(&ms, &k, &s, &kp, &mut h);
        s = s2;
        kp = k2;
    }
    let t_fix = t0.elapsed().as_secs_f64();
    let state_diff = s.iter().zip(&gst).filter(|(a, b)| a != b).count();
    let kap_diff = kp.iter().zip(&gkp).filter(|(a, b)| a != b).count();
    let (sf, kf) = run_f64(&ms, dt, steps);
    // accuracy against f64: displacement field, velocities, damage
    let (mut eu, mut mu, mut ev, mut mv) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..nc {
        for c in 0..3 {
            let u = s[12 * i + c] as f64 / 2f64.powi(k.g[9]);
            eu = eu.max((u - sf[12 * i + c]).abs());
            mu = mu.max(sf[12 * i + c].abs());
            let v = s[12 * i + 6 + c] as f64 / 2f64.powi(k.g[10]);
            ev = ev.max((v - sf[12 * i + 6 + c]).abs());
            mv = mv.max(sf[12 * i + 6 + c].abs());
        }
    }
    let dmg = |kp: f64, f: usize| {
        let d = 20 * f;
        let ell = ((ms.fdata[d + 9] - ms.fdata[d + 12]).powi(2) + (ms.fdata[d + 10] - ms.fdata[d + 13]).powi(2) + (ms.fdata[d + 11] - ms.fdata[d + 14]).powi(2)).sqrt().max(1e-6) as f64;
        let (d0, wc) = (1e-4 * ell, 6.6666667e-5);
        if kp >= wc { 1.0 } else if kp > d0 { (kp - d0) / (kp * (1.0 - d0 / wc)) } else { 0.0 }
    };
    let (mut ddmax, mut on_x, mut on_f, mut br_x, mut br_f, mut flip) = (0.0f64, 0, 0, 0, 0, 0);
    for f in 0..nf {
        let a = dmg(kp[f] as f64 / 2f64.powi(28), f);
        let b = dmg(kf[f], f);
        ddmax = ddmax.max((a - b).abs());
        on_x += (a > 0.0) as usize;
        on_f += (b > 0.0) as usize;
        br_x += (a >= 1.0) as usize;
        br_f += (b >= 1.0) as usize;
        flip += ((a > 0.0) != (b > 0.0)) as usize;
    }
    let bits = |x: i64| if x > 0 { (x as f64).log2() } else { 0.0 };
    let names = ["jump", "facet force", "u", "v", "theta", "w", "cell force", "cell torque"];
    let headroom: serde_json::Map<String, serde_json::Value> = names.iter().enumerate().map(|(j, n)| (n.to_string(), json!({"max_abs_lsb": h.mx[j], "bits_used": bits(h.mx[j]), "headroom_bits": 31.0 - bits(h.mx[j])}))).collect();
    let out = json!({"label": label, "cells": nc, "facets": nf, "steps": steps, "scales_global": k.g[..13].to_vec(),
        "bitwise": {"state_words_differing": state_diff, "kappa_words_differing": kap_diff, "fnv1a32_cpu": fnv(&s, &kp), "fnv1a32_gpu": fnv(&gst, &gkp), "cpu_overflow_flag": h.flag, "first_overflow_shift_value": h.first.map(|(s, v)| json!([s, v]))},
        "accuracy_vs_f64": {"u_max_abs_err_m": eu, "u_max_m": mu, "u_rel": eu / mu, "v_rel": ev / mv, "damage_max_abs_diff": ddmax,
            "damaged_fixed": on_x, "damaged_f64": on_f, "broken_fixed": br_x, "broken_f64": br_f, "onset_disagreements": flip},
        "headroom": headroom, "cpu_fixed_s": t_fix});
    eprintln!("{label}: {}", serde_json::to_string(&out).unwrap());
    write_json(if a.iter().any(|s| s == "--phase0e") { "phase0e/gpu" } else { EXP }, &format!("fixed_ref_{label}_{tag}"), &out);
}
