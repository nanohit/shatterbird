// Shatterbird phase 1 probe: determinism tax by kernel class, float divergence census,
// memory bandwidth, dispatch overhead and a block-cellular-automaton workload.
// Self-contained: every input is generated from an integer RNG; every integer kernel has a
// JS reference with identical semantics. Works from any https/localhost origin.
'use strict';
const D0 = 1024, DF = 16384, GAMMA = 128;
const out = document.getElementById('o');
const log = s => { out.textContent += s + '\n'; };
function rng(seed) { let s = seed >>> 0; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s; }; }
function fnv(a) { const u = new Uint32Array(a.buffer, a.byteOffset, a.byteLength >> 2); let h = 0x811c9dc5; for (let i = 0; i < u.length; i++) { h ^= u[i]; h = Math.imul(h, 0x01000193); } return (h >>> 0).toString(16).padStart(8, '0'); }

// ---------------------------------------------------------------- facet kernel data
function facetData(nf, seed) {
  const r = rng(seed), nc = nf >> 1;
  const cells = new Int32Array(4 * nc), fac = new Int32Array(12 * nf);
  const sym = b => (r() % (2 << b)) - (1 << b);
  for (let c = 0; c < nc; c++) { cells[4 * c] = sym(12); cells[4 * c + 1] = sym(12); cells[4 * c + 2] = sym(7); }
  for (let f = 0; f < nf; f++) {
    const i = r() % nc, j = (i + 1 + r() % 61) % nc, a = (r() % 65536) / 65536 * 2 * Math.PI, b = 12 * f;
    fac[b] = i; fac[b + 1] = j; fac[b + 2] = Math.round(8192 * Math.cos(a)); fac[b + 3] = Math.round(8192 * Math.sin(a));
    fac[b + 4] = sym(11); fac[b + 5] = sym(11); fac[b + 6] = sym(11); fac[b + 7] = sym(11);
    fac[b + 8] = 256 + r() % 3841; fac[b + 9] = r() % 4096;
  }
  return { nc, cells, fac };
}
// JS reference of the integer specification (all products fit i32 by construction).
function kernRef(du, dv, thi, thj, nx, ny, rix, riy, rjx, rjy, kn, kap, o) {
  const jx = du + ((Math.imul(thi, riy) - Math.imul(thj, rjy) + 16) >> 5);
  const jy = dv + ((Math.imul(thj, rjx) - Math.imul(thi, rix) + 16) >> 5);
  const cl = v => Math.max(-32768, Math.min(32767, v));
  const dn = cl((Math.imul(jx, nx) + Math.imul(jy, ny) + 4096) >> 13), dt = cl((Math.imul(jy, nx) - Math.imul(jx, ny) + 4096) >> 13);
  const dp = Math.max(dn, 0), e2 = dp * dp + ((dt * dt) >>> 8) * GAMMA;
  let s = Math.floor(Math.sqrt(e2)); while (s * s > e2) s--; while ((s + 1) * (s + 1) <= e2) s++;
  const kp = Math.max(kap, s);
  let w = 32768; if (kp > D0) w = kp >= DF ? 0 : Math.trunc(Math.trunc((D0 << 15) / kp) * (DF - kp) / (DF - D0));
  const fn = dn < 0 ? kn * dn : kn * ((w * dn) >> 15), ft = kn * ((w * dt) >> 15);
  const mq = (f, n) => (f >> 13) * n + (((f & 8191) * n) >> 13);
  const fx = mq(fn, nx) - mq(ft, ny), fy = mq(fn, ny) + mq(ft, nx);
  o[0] = fx; o[1] = fy; o[2] = Math.imul(rix, fy >> 9) - Math.imul(riy, fx >> 9); o[3] = -(Math.imul(rjx, fy >> 9) - Math.imul(rjy, fx >> 9)); o[4] = kp;
}
function facetRef(d, nf, rep) {
  const res = new Int32Array(8 * nf), o = new Int32Array(5), { cells, fac } = d;
  for (let f = 0; f < nf; f++) {
    const b = 12 * f, i = fac[b], j = fac[b + 1];
    const du = cells[4 * j] - cells[4 * i], dv = cells[4 * j + 1] - cells[4 * i + 1];
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0;
    for (let r = 0; r < rep; r++) {
      kernRef(du + r * 37, dv - r * 53, cells[4 * i + 2], cells[4 * j + 2], fac[b + 2], fac[b + 3], fac[b + 4], fac[b + 5], fac[b + 6], fac[b + 7], fac[b + 8], fac[b + 9], o);
      a0 = (a0 + o[0]) | 0; a1 = (a1 + o[1]) | 0; a2 = (a2 + o[2]) | 0; a3 = (a3 + o[3]) | 0; a4 = (a4 + o[4]) | 0;
    }
    res[8 * f] = a0; res[8 * f + 1] = a1; res[8 * f + 2] = a2; res[8 * f + 3] = a3; res[8 * f + 4] = a4;
  }
  return res;
}

// ---------------------------------------------------------------- WGSL
const HEAD = `
struct P { nf: u32, rep: u32, a: u32, b: u32 }
@group(0) @binding(0) var<storage, read> cells: array<i32>;
@group(0) @binding(1) var<storage, read> fac: array<i32>;
@group(0) @binding(2) var<storage, read_write> outb: array<i32>;
@group(0) @binding(3) var<uniform> p: P;
const D0: i32 = ${D0}; const DF: i32 = ${DF}; const GAMMA: u32 = ${GAMMA}u;
struct O { v: vec4i, k: i32 }
struct OF { v: vec4f, k: f32 }
`;
const ISQRT = `
fn isqrt(e2: u32) -> u32 { // float proposes, integer verifies: exact floor(sqrt)
  var s = u32(sqrt(f32(e2)));
  if (s * s > e2) { s -= 1u; } if (s * s > e2) { s -= 1u; }
  if ((s + 1u) * (s + 1u) <= e2) { s += 1u; } if ((s + 1u) * (s + 1u) <= e2) { s += 1u; }
  return s;
}`;
const KERN_N = ISQRT + `
fn mq(f: i32, n: i32) -> i32 { return (f >> 13u) * n + (((f & 8191) * n) >> 13u); }
fn kern(du: i32, dv: i32, thi: i32, thj: i32, nx: i32, ny: i32, rix: i32, riy: i32, rjx: i32, rjy: i32, kn: i32, kap: i32) -> O {
  let jx = du + ((thi * riy - thj * rjy + 16) >> 5u);
  let jy = dv + ((thj * rjx - thi * rix + 16) >> 5u);
  let dn = clamp((jx * nx + jy * ny + 4096) >> 13u, -32768, 32767);
  let dt = clamp((jy * nx - jx * ny + 4096) >> 13u, -32768, 32767);
  let dp = max(dn, 0);
  let e2 = u32(dp * dp) + (u32(dt * dt) >> 8u) * GAMMA;
  let kp = max(kap, i32(isqrt(e2)));
  var w = 32768;
  if (kp > D0) { if (kp >= DF) { w = 0; } else { w = (((D0 << 15u) / kp) * (DF - kp)) / (DF - D0); } }
  var fnn = kn * dn; if (dn >= 0) { fnn = kn * ((w * dn) >> 15u); }
  let ft = kn * ((w * dt) >> 15u);
  let fx = mq(fnn, nx) - mq(ft, ny); let fy = mq(fnn, ny) + mq(ft, nx);
  var o: O; o.v = vec4i(fx, fy, rix * (fy >> 9u) - riy * (fx >> 9u), -(rjx * (fy >> 9u) - rjy * (fx >> 9u))); o.k = kp; return o;
}`;
// Same specification, written in the generic "every product is 32x32->64" style.
const KERN_W = ISQRT + `
struct L { hi: i32, lo: u32 }
fn mul64(a: i32, b: i32) -> L {
  let ua = u32(abs(a)); let ub = u32(abs(b));
  let al = ua & 0xffffu; let ah = ua >> 16u; let bl = ub & 0xffffu; let bh = ub >> 16u;
  let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
  let mid = (ll >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
  var lo = (ll & 0xffffu) | (mid << 16u); var hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
  if ((a < 0) != (b < 0)) { lo = ~lo + 1u; hi = ~hi + select(0u, 1u, lo == 0u); }
  return L(bitcast<i32>(hi), lo);
}
fn add64(a: L, b: L) -> L { let lo = a.lo + b.lo; return L(a.hi + b.hi + select(0, 1, lo < a.lo), lo); }
fn neg64(a: L) -> L { let lo = ~a.lo + 1u; return L(~a.hi + select(0, 1, lo == 0u), lo); }
fn shr64(a: L, s: u32) -> i32 { return bitcast<i32>((a.lo >> s) | (bitcast<u32>(a.hi) << (32u - s))); }
fn c64(v: i32) -> L { return L(select(0, -1, v < 0), bitcast<u32>(v)); }
fn low(a: L) -> i32 { return bitcast<i32>(a.lo); }
fn kern(du: i32, dv: i32, thi: i32, thj: i32, nx: i32, ny: i32, rix: i32, riy: i32, rjx: i32, rjy: i32, kn: i32, kap: i32) -> O {
  let jx = du + shr64(add64(add64(mul64(thi, riy), neg64(mul64(thj, rjy))), c64(16)), 5u);
  let jy = dv + shr64(add64(add64(mul64(thj, rjx), neg64(mul64(thi, rix))), c64(16)), 5u);
  let dn = clamp(shr64(add64(add64(mul64(jx, nx), mul64(jy, ny)), c64(4096)), 13u), -32768, 32767);
  let dt = clamp(shr64(add64(add64(mul64(jy, nx), neg64(mul64(jx, ny))), c64(4096)), 13u), -32768, 32767);
  let dp = max(dn, 0);
  let e2 = mul64(dp, dp).lo + bitcast<u32>(low(mul64(i32(mul64(dt, dt).lo >> 8u), i32(GAMMA))));
  let kp = max(kap, i32(isqrt(e2)));
  var w = 32768;
  if (kp > D0) { if (kp >= DF) { w = 0; } else { w = low(mul64((D0 << 15u) / kp, DF - kp)) / (DF - D0); } }
  var fnn = low(mul64(kn, dn)); if (dn >= 0) { fnn = low(mul64(kn, shr64(mul64(w, dn), 15u))); }
  let ft = low(mul64(kn, shr64(mul64(w, dt), 15u)));
  let fx = shr64(mul64(fnn, nx), 13u) - shr64(mul64(ft, ny), 13u); let fy = shr64(mul64(fnn, ny), 13u) + shr64(mul64(ft, nx), 13u);
  var o: O;
  o.v = vec4i(fx, fy, low(add64(mul64(rix, fy >> 9u), neg64(mul64(riy, fx >> 9u)))), -low(add64(mul64(rjx, fy >> 9u), neg64(mul64(rjy, fx >> 9u))))); o.k = kp; return o;
}`;
// Float versions of the same real-valued function (no intermediate rounding), three legal evaluation orders.
const kernF = v => `
fn sop(a: f32, b: f32, c: f32, d: f32) -> f32 { ${v === 'fma' ? 'return fma(a, b, c * d);' : v === 'alt' ? 'return c * d + a * b;' : 'return a * b + c * d;'} }
fn kern(du: f32, dv: f32, thi: f32, thj: f32, nx: f32, ny: f32, rix: f32, riy: f32, rjx: f32, rjy: f32, kn: f32, kap: f32) -> OF {
  let jx = du + sop(thi, riy, -thj, rjy) * 0.03125;
  let jy = dv + sop(thj, rjx, -thi, rix) * 0.03125;
  let dn = clamp(sop(jx, nx, jy, ny) * (1.0 / 8192.0), -32768.0, 32767.0);
  let dt = clamp(sop(jy, nx, -jx, ny) * (1.0 / 8192.0), -32768.0, 32767.0);
  let dp = max(dn, 0.0);
  let kp = max(kap, sqrt(sop(dp, dp, ${GAMMA / 256} * dt, dt)));
  var w = 1.0;
  if (kp > ${D0}.0) { if (kp >= ${DF}.0) { w = 0.0; } else { w = ${v === 'alt' ? `${D0}.0 * (${DF}.0 - kp) / (kp * ${DF - D0}.0)` : `(${D0}.0 / kp) * (${DF}.0 - kp) / ${DF - D0}.0`}; } }
  var fnn = kn * dn; if (dn >= 0.0) { fnn = ${v === 'alt' ? 'kn * (w * dn)' : 'kn * w * dn'}; }
  let ft = ${v === 'alt' ? 'kn * (w * dt)' : 'kn * w * dt'};
  let fx = sop(fnn, nx, -ft, ny) * (1.0 / 8192.0); let fy = sop(fnn, ny, ft, nx) * (1.0 / 8192.0);
  var o: OF; o.v = vec4f(fx, fy, sop(rix, fy, -riy, fx) * (1.0 / 512.0), -sop(rjx, fy, -rjy, fx) * (1.0 / 512.0)); o.k = kp; return o;
}`;
const MAIN_I = `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let f = g.x; if (f >= p.nf) { return; }
  let b = f * 12u; let i = u32(fac[b]); let j = u32(fac[b + 1u]);
  let du = cells[4u * j] - cells[4u * i]; let dv = cells[4u * j + 1u] - cells[4u * i + 1u];
  let thi = cells[4u * i + 2u]; let thj = cells[4u * j + 2u];
  var acc = vec4i(0); var ka = 0;
  for (var r = 0; r < i32(p.rep); r++) {
    let o = kern(du + r * 37, dv - r * 53, thi, thj, fac[b + 2u], fac[b + 3u], fac[b + 4u], fac[b + 5u], fac[b + 6u], fac[b + 7u], fac[b + 8u], fac[b + 9u]);
    acc += o.v; ka += o.k;
  }
  let k = f * 8u; outb[k] = acc.x; outb[k + 1u] = acc.y; outb[k + 2u] = acc.z; outb[k + 3u] = acc.w; outb[k + 4u] = ka;
}`;
const MAIN_F = `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let f = g.x; if (f >= p.nf) { return; }
  let b = f * 12u; let i = u32(fac[b]); let j = u32(fac[b + 1u]);
  let du = cells[4u * j] - cells[4u * i]; let dv = cells[4u * j + 1u] - cells[4u * i + 1u];
  let thi = f32(cells[4u * i + 2u]); let thj = f32(cells[4u * j + 2u]);
  var acc = vec4f(0.0); var ka = 0.0;
  for (var r = 0; r < i32(p.rep); r++) {
    let o = kern(f32(du + r * 37), f32(dv - r * 53), thi, thj, f32(fac[b + 2u]), f32(fac[b + 3u]), f32(fac[b + 4u]), f32(fac[b + 5u]), f32(fac[b + 6u]), f32(fac[b + 7u]), f32(fac[b + 8u]), f32(fac[b + 9u]));
    acc += o.v; ka += o.k;
  }
  let k = f * 8u; outb[k] = bitcast<i32>(acc.x); outb[k + 1u] = bitcast<i32>(acc.y); outb[k + 2u] = bitcast<i32>(acc.z); outb[k + 3u] = bitcast<i32>(acc.w); outb[k + 4u] = bitcast<i32>(ka);
}`;
const VARIANTS = { N: HEAD + KERN_N + MAIN_I, W: HEAD + KERN_W + MAIN_I, F: HEAD + kernF('plain') + MAIN_F, Ffma: HEAD + kernF('fma') + MAIN_F, Falt: HEAD + kernF('alt') + MAIN_F };

// Margolus block automaton: 0 empty, 1 sand, 2 water, 3 wall. Each 2x2 block is a pure function of
// its four cells and an integer hash; blocks tile the torus, so the update is race free and conserves cells.
const CA_RULE_JS = (a, b, c, d, h) => { // a b / c d (c, d below)
  const mv = x => x !== 3, hv = (x, y) => mv(x) && mv(y) && x !== 0 && (y === 0 || (x === 1 && y === 2));
  if (hv(a, c)) { const t = a; a = c; c = t; } if (hv(b, d)) { const t = b; b = d; d = t; }
  if (hv(a, d) && c !== 0 && (h & 1)) { const t = a; a = d; d = t; } else if (hv(b, c) && d !== 0 && (h & 1)) { const t = b; b = c; c = t; }
  if ((h & 2) && ((a === 2 && b === 0) || (a === 0 && b === 2)) && c !== 0 && d !== 0) { const t = a; a = b; b = t; }
  if ((h & 4) && ((c === 2 && d === 0) || (c === 0 && d === 2))) { const t = c; c = d; d = t; }
  return [a, b, c, d];
};
const CA_WGSL = `
struct P { w: u32, h: u32, ox: u32, tick: u32 }
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
fn hv(x: u32, y: u32) -> bool { return x != 3u && y != 3u && x != 0u && (y == 0u || (x == 1u && y == 2u)); }
fn hash(x: u32, y: u32, t: u32) -> u32 { var h = (x * 0x9E3779B1u) ^ (y * 0x85EBCA77u) ^ (t * 0xC2B2AE3Du); h ^= h >> 15u; h *= 0x2C1B3C6Du; h ^= h >> 12u; return h; }
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.w / 2u || g.y >= p.h / 2u) { return; }
  let x0 = (2u * g.x + p.ox) % p.w; let y0 = (2u * g.y + p.ox) % p.h; let x1 = (x0 + 1u) % p.w; let y1 = (y0 + 1u) % p.h;
  let ia = y0 * p.w + x0; let ib = y0 * p.w + x1; let ic = y1 * p.w + x0; let id = y1 * p.w + x1;
  var a = src[ia]; var b = src[ib]; var c = src[ic]; var d = src[id]; var t = 0u;
  let h = hash(g.x, g.y, p.tick);
  if (hv(a, c)) { t = a; a = c; c = t; } if (hv(b, d)) { t = b; b = d; d = t; }
  if (hv(a, d) && c != 0u && (h & 1u) != 0u) { t = a; a = d; d = t; } else if (hv(b, c) && d != 0u && (h & 1u) != 0u) { t = b; b = c; c = t; }
  if ((h & 2u) != 0u && ((a == 2u && b == 0u) || (a == 0u && b == 2u)) && c != 0u && d != 0u) { t = a; a = b; b = t; }
  if ((h & 4u) != 0u && ((c == 2u && d == 0u) || (c == 0u && d == 2u))) { t = c; c = d; d = t; }
  dst[ia] = a; dst[ib] = b; dst[ic] = c; dst[id] = d;
}`;
function caInit(w, h, seed) { const r = rng(seed), g = new Uint32Array(w * h); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const q = r() % 16; g[y * w + x] = (y === h - 1 || x === 0 || x === w - 1 || (y % 64 === 40 && (x % 96) < 60)) ? 3 : q < 5 ? 1 : q < 9 ? 2 : 0; } return g; }
function caHash(x, y, t) { let h = (Math.imul(x, 0x9E3779B1) ^ Math.imul(y, 0x85EBCA77) ^ Math.imul(t, 0xC2B2AE3D)) >>> 0; h ^= h >>> 15; h = Math.imul(h, 0x2C1B3C6D) >>> 0; h ^= h >>> 12; return h >>> 0; }
function caRef(g, w, h, steps) {
  let s = g.slice(), d = new Uint32Array(w * h);
  for (let t = 0; t < steps; t++) {
    const ox = t & 1;
    for (let by = 0; by < h / 2; by++) for (let bx = 0; bx < w / 2; bx++) {
      const x0 = (2 * bx + ox) % w, y0 = (2 * by + ox) % h, x1 = (x0 + 1) % w, y1 = (y0 + 1) % h;
      const r = CA_RULE_JS(s[y0 * w + x0], s[y0 * w + x1], s[y1 * w + x0], s[y1 * w + x1], caHash(bx, by, t));
      d[y0 * w + x0] = r[0]; d[y0 * w + x1] = r[1]; d[y1 * w + x0] = r[2]; d[y1 * w + x1] = r[3];
    }
    const q = s; s = d; d = q;
  }
  return s;
}
const COPY_WGSL = `
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) { let i = g.x + g.y * 4194240u; if (i < arrayLength(&src)) { dst[i] = src[i] ^ 0x5bd1e995u; } }`;

// ---------------------------------------------------------------- harness
async function probe(opts = {}) {
  const R = { ua: navigator.userAgent, time: new Date().toISOString(), tag: opts.tag || '' };
  if (!navigator.gpu) { R.error = 'no WebGPU'; return R; }
  const adapter = await navigator.gpu.requestAdapter(opts.fallback ? { forceFallbackAdapter: true } : { powerPreference: 'high-performance' });
  if (!adapter) { R.error = 'no adapter'; return R; }
  const inf = adapter.info || {}; R.adapter = { vendor: inf.vendor, architecture: inf.architecture, device: inf.device, description: inf.description, fallback: !!opts.fallback };
  const want = { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 28), maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 28) };
  const dev = await adapter.requestDevice({ requiredLimits: want });
  R.limits = { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize, maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension, maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage };
  const errs = []; dev.addEventListener('uncapturederror', e => errs.push(e.error.message));
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const mk = (data, usage = S) => { const b = dev.createBuffer({ size: data.byteLength, usage }); dev.queue.writeBuffer(b, 0, data); return b; };
  const pipe = async code => { const module = dev.createShaderModule({ code }); const ci = await module.getCompilationInfo(); const er = ci.messages.filter(m => m.type === 'error'); if (er.length) throw new Error(er.map(m => m.lineNum + ':' + m.linePos + ' ' + m.message).join(' | ')); return dev.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } }); };
  const read = async (buf, bytes) => { const r = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); const e = dev.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, r, 0, bytes); dev.queue.submit([e.finish()]); await r.mapAsync(GPUMapMode.READ); const a = new Int32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return a; };
  const timed = async (enc, reps = 5) => { let best = 1e99; for (let k = 0; k < reps; k++) { const cmds = enc(); await dev.queue.onSubmittedWorkDone(); const t0 = performance.now(); dev.queue.submit(cmds); await dev.queue.onSubmittedWorkDone(); best = Math.min(best, performance.now() - t0); } return best; };
  const small = opts.small, NF = small ? 1 << 16 : 1 << 20, NV = 1 << 16;

  R.stage_errors = []; const mark = n => { try { localStorage.setItem('sb_stage', n); } catch (e) { } };
  let closing = false; dev.lost.then(i => { if (!closing) R.stage_errors.push('device lost: ' + i.message); });
  try { mark('facet');
  // --- 1. facet kernel: correctness (small problem, rep 1 and 3) and float census
  log('facet kernel: correctness');
  const dv = facetData(NV, 12345), uni = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const cb = mk(dv.cells), fb = mk(dv.fac), ob = dev.createBuffer({ size: 32 * NV, usage: S });
  const pipes = {}; R.facet = { hashes: {}, mismatch_vs_cpu: {}, float_census: {} };
  for (const v of Object.keys(VARIANTS)) { try { pipes[v] = await pipe(VARIANTS[v]); } catch (e) { R.facet.hashes[v] = 'compile error: ' + e.message; } }
  const runFacet = (v, b0, b1, b2, nf, rep, n = 1) => { dev.queue.writeBuffer(uni, 0, new Uint32Array([nf, rep, 0, 0])); const bg = dev.createBindGroup({ layout: pipes[v].getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: b0 } }, { binding: 1, resource: { buffer: b1 } }, { binding: 2, resource: { buffer: b2 } }, { binding: 3, resource: { buffer: uni } }] }); const e = dev.createCommandEncoder(), ps = e.beginComputePass(); ps.setPipeline(pipes[v]); ps.setBindGroup(0, bg); for (let k = 0; k < n; k++) ps.dispatchWorkgroups(Math.ceil(nf / 64)); ps.end(); return [e.finish()]; };
  const outs = {};
  for (const rep of [1, 3]) {
    const ref = facetRef(dv, NV, rep); R.facet.hashes['cpu_rep' + rep] = fnv(ref);
    for (const v of Object.keys(pipes)) {
      dev.queue.submit(runFacet(v, cb, fb, ob, NV, rep)); const o = await read(ob, 32 * NV); outs[v + rep] = o; R.facet.hashes[v + '_rep' + rep] = fnv(o);
      if (v === 'N' || v === 'W') { let bad = 0; for (let i = 0; i < o.length; i++) if (o[i] !== ref[i]) bad++; R.facet.mismatch_vs_cpu[v + '_rep' + rep] = bad; }
    }
    const cen = (x, y) => { const a = outs[x + rep], b = outs[y + rep]; if (!a || !b) return null; let w = 0, el = 0; for (let f = 0; f < NV; f++) { let d = 0; for (let k = 0; k < 5; k++) if (a[8 * f + k] !== b[8 * f + k]) d++; w += d; if (d) el++; } return { words: w / (5 * NV), elements: el / NV }; };
    R.facet.float_census['rep' + rep] = { F_vs_Ffma: cen('F', 'Ffma'), F_vs_Falt: cen('F', 'Falt') };
    // float vs integer specification, in output units (how far the real-valued function is from the spec)
    const fo = outs['F' + rep], io = outs['N' + rep]; if (fo && io) { const ff = new Float32Array(fo.buffer); let mx = 0, rnd = 0; for (let f = 0; f < NV; f++) for (let k = 0; k < 2; k++) { const d = Math.abs(ff[8 * f + k] - io[8 * f + k]); mx = Math.max(mx, d); if (Math.round(ff[8 * f + k]) !== io[8 * f + k]) rnd++; } R.facet.float_vs_spec = { max_abs_force_lsb: mx, rounded_float_differs_fraction: rnd / (2 * NV) }; }
  }
  log(JSON.stringify(R.facet));
  // --- 2. facet kernel: timing versus compute intensity
  log('facet kernel: timing');
  const dt = facetData(NF, 777), cb2 = mk(dt.cells), fb2 = mk(dt.fac), ob2 = dev.createBuffer({ size: 32 * NF, usage: S });
  const REPS = opts.reps || [1, 8, 32, 128];
  R.facet.timing = { nf: NF, bytes_per_facet: 48 + 32 + 32, ns_per_eval: {} };
  for (const v of ['F', 'N', 'W']) { if (!pipes[v]) continue; R.facet.timing.ns_per_eval[v] = {}; for (const rep of REPS) { const t1 = await timed(() => runFacet(v, cb2, fb2, ob2, NF, rep, 1), 2); const n = Math.max(1, Math.min(64, Math.ceil(40 / t1))); const ms = await timed(() => runFacet(v, cb2, fb2, ob2, NF, rep, n)); R.facet.timing.ns_per_eval[v][rep] = ms * 1e6 / (NF * n); } }
  { const t = R.facet.timing.ns_per_eval, hi = REPS[REPS.length - 1], lo = REPS[REPS.length - 2], sl = v => (t[v][hi] - t[v][lo]) / (hi - lo); if (t.F && t.N && t.W) R.facet.timing.summary = { ns_per_facet_memory_bound: t.F[1], compute_ns_per_eval: { F: sl('F'), N: sl('N'), W: sl('W') }, tax_compute_bound: { N: sl('N') / sl('F'), W: sl('W') / sl('F') }, tax_at_rep: Object.fromEntries(REPS.map(r => [r, { N: t.N[r] / t.F[r], W: t.W[r] / t.F[r] }])) }; }
  log(JSON.stringify(R.facet.timing));
  } catch (e) { R.stage_errors.push('facet: ' + (e.message || e)); log('facet stage failed: ' + (e.message || e)); }
  try { mark('bandwidth');
  // --- 3. bandwidth
  log('bandwidth');
  const NB = small ? 1 << 22 : 1 << 24, src = mk(new Uint32Array(NB)), dst = dev.createBuffer({ size: 4 * NB, usage: S }), cp = await pipe(COPY_WGSL);
  const bgc = dev.createBindGroup({ layout: cp.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: dst } }] });
  const nrep = 8, msb = await timed(() => { const e = dev.createCommandEncoder(), ps = e.beginComputePass(); ps.setPipeline(cp); ps.setBindGroup(0, bgc); for (let k = 0; k < nrep; k++) ps.dispatchWorkgroups(Math.min(65535, NB / 64), Math.ceil(NB / 64 / 65535)); ps.end(); return [e.finish()]; });
  R.bandwidth = { bytes: 4 * NB, GB_per_s_read_plus_write: 8 * NB * nrep / (msb * 1e6) };
  // --- 4. dispatch overhead (tiny kernel)
  const tiny = dev.createBuffer({ size: 256, usage: S }), tiny2 = dev.createBuffer({ size: 256, usage: S });
  const bgt = dev.createBindGroup({ layout: cp.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: tiny } }, { binding: 1, resource: { buffer: tiny2 } }] });
  const nd = 2000;
  const same = await timed(() => { const e = dev.createCommandEncoder(), ps = e.beginComputePass(); ps.setPipeline(cp); ps.setBindGroup(0, bgt); for (let k = 0; k < nd; k++) ps.dispatchWorkgroups(1); ps.end(); return [e.finish()]; });
  const sep = await timed(() => { const e = dev.createCommandEncoder(); for (let k = 0; k < nd; k++) { const ps = e.beginComputePass(); ps.setPipeline(cp); ps.setBindGroup(0, bgt); ps.dispatchWorkgroups(1); ps.end(); } return [e.finish()]; });
  R.dispatch = { us_per_dispatch_same_pass: same * 1e3 / nd, us_per_dispatch_separate_passes: sep * 1e3 / nd };
  log(JSON.stringify({ bandwidth: R.bandwidth, dispatch: R.dispatch }));
  } catch (e) { R.stage_errors.push('bandwidth: ' + (e.message || e)); log('bandwidth stage failed: ' + (e.message || e)); }
  try { mark('automaton');
  // --- 5. block automaton: correctness (256x256, 64 steps) and throughput
  log('block automaton');
  const cap = await pipe(CA_WGSL), cu = [0, 1].map(() => dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  const caRun = async (w, h, steps, init, timing) => {
    const bufs = [mk(init), dev.createBuffer({ size: 4 * w * h, usage: S })];
    const us = []; for (let t = 0; t < steps; t++) { const u = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(u, 0, new Uint32Array([w, h, t & 1, t])); us.push(u); }
    const bgs = us.map((u, t) => dev.createBindGroup({ layout: cap.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bufs[t & 1] } }, { binding: 1, resource: { buffer: bufs[1 - (t & 1)] } }, { binding: 2, resource: { buffer: u } }] }));
    const enc = () => { const e = dev.createCommandEncoder(); for (let t = 0; t < steps; t++) { const ps = e.beginComputePass(); ps.setPipeline(cap); ps.setBindGroup(0, bgs[t]); ps.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16)); ps.end(); } return [e.finish()]; };
    if (timing) return timed(enc, 3);
    dev.queue.submit(enc()); return read(bufs[steps & 1], 4 * w * h);
  };
  const ci = caInit(256, 256, 99), cg = await caRun(256, 256, 64, ci, false), cr = caRef(ci, 256, 256, 64);
  let cbad = 0, moved = 0; for (let i = 0; i < cr.length; i++) { if ((cg[i] >>> 0) !== cr[i]) cbad++; if (cr[i] !== ci[i]) moved++; }
  R.ca = { hash_gpu: fnv(cg), hash_cpu: fnv(cr), mismatch_cells: cbad, changed_fraction_after_64: moved / cr.length, throughput: {} };
  for (const w of small ? [1024] : [1024, 2048, 4096]) { const steps = 16, ms = await caRun(w, w, steps, caInit(w, w, 5), true); R.ca.throughput[w] = { cells: w * w, ms_per_step: ms / steps, cell_updates_per_s: w * w * steps / (ms * 1e-3), GB_per_s: 8 * w * w * steps / (ms * 1e6) }; }
  log(JSON.stringify(R.ca));
  } catch (e) { R.stage_errors.push('automaton: ' + (e.message || e)); log('automaton stage failed: ' + (e.message || e)); }
  mark('done'); R.errors = errs; closing = true; dev.destroy(); return R;
}
// Compact, human-checkable summary (about 1 KB) that fits into a prefilled GitHub issue.
const PROBE_VERSION = 1, REPO = 'nanohit/shatterbird';
function compact(res) {
  const g3 = x => (typeof x === 'number' && isFinite(x)) ? +x.toPrecision(3) : x, c = { v: PROBE_VERSION };
  const one = (R, full) => {
    if (!R) return null; const o = {}; if (R.error) o.error = R.error; if (R.adapter) o.ad = [R.adapter.vendor, R.adapter.architecture, R.adapter.device, R.adapter.description].map(x => x || '').join('|');
    if (R.limits) o.lim = [R.limits.maxStorageBufferBindingSize, R.limits.maxBufferSize];
    if (R.facet) { o.h = R.facet.hashes; o.mm = R.facet.mismatch_vs_cpu; if (full) { o.census = R.facet.float_census; const t = R.facet.timing; if (t) { o.nf = t.nf; o.t = {}; for (const v in t.ns_per_eval) o.t[v] = Object.fromEntries(Object.entries(t.ns_per_eval[v]).map(([k, x]) => [k, g3(x)])); } } }
    if (R.bandwidth && full) o.bw = g3(R.bandwidth.GB_per_s_read_plus_write);
    if (R.dispatch && full) o.disp = [g3(R.dispatch.us_per_dispatch_same_pass), g3(R.dispatch.us_per_dispatch_separate_passes)];
    if (R.ca) { o.ca = { g: R.ca.hash_gpu, c: R.ca.hash_cpu, mm: R.ca.mismatch_cells }; if (full) o.ca.ms = Object.fromEntries(Object.entries(R.ca.throughput).map(([k, x]) => [k, g3(x.ms_per_step)])); }
    const er = [...(R.stage_errors || []), ...(R.errors || [])]; if (er.length) o.err = er.map(e => String(e).slice(0, 160)).slice(0, 4);
    return o;
  };
  c.tag = res.tag; c.small = !!res.small; c.ua = (res.main && res.main.ua || navigator.userAgent).slice(0, 200); c.prev_crash = res.prev_crash || undefined;
  c.main = one(res.main, true); c.fb = one(res.fallback, false); return c;
}
async function runAll(opts = {}) {
  out.textContent = ''; let prev = null; try { prev = localStorage.getItem('sb_stage'); } catch (e) { }
  const res = { tag: opts.tag || 'run', small: !!opts.small, prev_crash: prev && prev !== 'done' ? prev : null };
  try { res.main = await probe(opts); } catch (e) { res.main = { error: String(e && e.message || e), ua: navigator.userAgent }; }
  if (!opts.nofallback) { try { res.fallback = await probe({ ...opts, fallback: true, small: true }); } catch (e) { res.fallback = { error: String(e && e.message || e) }; } }
  try { localStorage.setItem('sb_stage', 'done'); } catch (e) { }
  window.__det = res; window.__compact = compact(res);
  out.textContent += '\n=== COMPACT RESULT (this is what gets sent) ===\n' + JSON.stringify(window.__compact, null, 1);
  try { if (location.hostname === '127.0.0.1' || location.hostname === 'localhost') await fetch('/save/det_' + (opts.tag || 'run') + '.json', { method: 'POST', body: JSON.stringify(res, null, 1) }); } catch (e) { }
  const st = document.getElementById('st'); if (st) st.textContent = 'done - now press "Send result" (or "Copy result" and paste it to Pavel)';
  return res;
}
const Q = new URLSearchParams(location.search), mobile = /Mobi|Android|iPhone|iPad/.test(navigator.userAgent);
const start = small => { const st = document.getElementById('st'); if (st) st.textContent = 'running, keep this tab in the foreground (1-3 min)...'; return runAll({ tag: Q.get('tag') || 'manual', small }); };
document.getElementById('go').onclick = () => start(false);
document.getElementById('gs').onclick = () => start(true);
document.getElementById('cp').onclick = async () => { const t = JSON.stringify(window.__compact || {}); try { await navigator.clipboard.writeText(t); document.getElementById('st').textContent = 'copied'; } catch (e) { const r = document.createRange(); r.selectNodeContents(out); getSelection().removeAllRanges(); getSelection().addRange(r); } };
document.getElementById('gh').onclick = () => { const c = window.__compact; if (!c) return; const ad = (c.main && c.main.ad || 'unknown').split('|').filter(x => x).join(' ').slice(0, 60); const body = 'Probe v' + c.v + (Q.get('who') ? ' from ' + Q.get('who') : '') + '\n\n```json\n' + JSON.stringify(c) + '\n```\n'; window.open('https://github.com/' + REPO + '/issues/new?title=' + encodeURIComponent('probe: ' + ad) + '&body=' + encodeURIComponent(body), '_blank'); };
if (mobile) document.getElementById('go').textContent = 'Run full (may crash on phones)';
if (!navigator.gpu) document.getElementById('st').textContent = 'WebGPU is not available in this browser (needs Chrome/Edge 113+, Safari 26+, on https).';
if (Q.get('auto')) runAll({ tag: Q.get('tag') || 'auto', small: !!Q.get('small') });
