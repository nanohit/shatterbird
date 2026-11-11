// Phase 0b / 3 §6.1 — WebGPU probe of the linear rbsn3 network (f32, no damage history).
// E-A: facet pass + cell pass per substep; E-B: one cell pass per substep (each cell
// evaluates its facets in canonical orientation and applies its own Jacobian row,
// ping-pong state); C: empty dispatches; I: matvec, 6×6 block solve, two-pass dot,
// PR restriction / prolongation. Chains are recorded in one compute pass without
// readback; GPU time from timestamp queries, wall time from onSubmittedWorkDone.

const WG = 64;

function parseMesh(buf) {
  const u = new Uint32Array(buf), f = new Float32Array(buf);
  const [magic, nc, nf, nagg, ncsr] = u.slice(0, 5);
  if (magic !== 0x31534253) throw new Error('bad magic');
  let o = 5;
  const take = (arr, n) => { const v = arr.slice(o, o + n); o += n; return v; };
  const cells = take(f, 25 * nc), fidx = take(u, 2 * nf), fdata = take(f, 20 * nf);
  const csrPtr = take(u, nc + 1), csrIdx = take(u, ncsr), dinv = take(f, 36 * nc);
  const aggOf = take(u, nc), aggData = take(f, 4 * nagg), aggPtr = take(u, nagg + 1), aggIdx = take(u, nc);
  return { nc, nf, nagg, ncsr, cells, fidx, fdata, csrPtr, csrIdx, dinv, aggOf, aggData, aggPtr, aggIdx, bytes: buf.byteLength };
}

const FORCE_FN = /* wgsl */ `
struct PairOut { fa: vec3f, ta: vec3f, fb: vec3f, tb: vec3f }
fn v3(a: ptr<storage, array<f32>, read>, i: u32) -> vec3f { return vec3f((*a)[i], (*a)[i + 1u], (*a)[i + 2u]); }
// canonical pair: inputs always (state of a, state of b, facet constants)
fn pair(fi: u32, ua: vec3f, wa: vec3f, ub: vec3f, wb: vec3f) -> PairOut {
  let b = fi * 20u;
  let n = v3(&fdata, b); let p1 = v3(&fdata, b + 3u); let p2 = v3(&fdata, b + 6u);
  let ra = v3(&fdata, b + 9u); let rb = v3(&fdata, b + 12u);
  let kn = fdata[b + 15u]; let kt = fdata[b + 16u]; let kr1 = fdata[b + 17u]; let kr2 = fdata[b + 18u]; let ktor = fdata[b + 19u];
  let jump = (ub + cross(wb, rb)) - (ua + cross(wa, ra));
  let dw = wb - wa;
  let f = kn * dot(n, jump) * n + kt * (dot(p1, jump) * p1 + dot(p2, jump) * p2);
  let m = kr1 * dot(p1, dw) * p1 + kr2 * dot(p2, dw) * p2 + ktor * dot(n, dw) * n;
  var o: PairOut;
  o.fa = f; o.ta = cross(ra, f) + m; o.fb = -f; o.tb = -cross(rb, f) - m;
  return o;
}
`;

const SHADERS = {
  // ---- E-A facet pass: forces of every facet into fforce (12 f32)
  eaFacet: /* wgsl */ `
@group(0) @binding(0) var<storage, read> fdata: array<f32>;
@group(0) @binding(1) var<storage, read> fidx: array<u32>;
@group(0) @binding(2) var<storage, read> state: array<f32>;
@group(0) @binding(3) var<storage, read_write> fforce: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4u;
${FORCE_FN}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let fi = g.x; if (fi >= P.y) { return; }
  let a = fidx[2u * fi]; let bb = fidx[2u * fi + 1u];
  let ua = vec3f(state[12u*a], state[12u*a+1u], state[12u*a+2u]); let wa = vec3f(state[12u*a+3u], state[12u*a+4u], state[12u*a+5u]);
  var ub = vec3f(0.0); var wb = vec3f(0.0);
  if (bb != 0xffffffffu) { ub = vec3f(state[12u*bb], state[12u*bb+1u], state[12u*bb+2u]); wb = vec3f(state[12u*bb+3u], state[12u*bb+4u], state[12u*bb+5u]); }
  let o = pair(fi, ua, wa, ub, wb);
  let k = 12u * fi;
  fforce[k] = o.fa.x; fforce[k+1u] = o.fa.y; fforce[k+2u] = o.fa.z; fforce[k+3u] = o.ta.x; fforce[k+4u] = o.ta.y; fforce[k+5u] = o.ta.z;
  fforce[k+6u] = o.fb.x; fforce[k+7u] = o.fb.y; fforce[k+8u] = o.fb.z; fforce[k+9u] = o.tb.x; fforce[k+10u] = o.tb.y; fforce[k+11u] = o.tb.z;
}`,
  // ---- E-A cell pass: gather + symplectic Euler (in place: only own state is touched)
  eaCell: /* wgsl */ `
@group(0) @binding(0) var<storage, read> cells: array<f32>;
@group(0) @binding(1) var<storage, read> csrPtr: array<u32>;
@group(0) @binding(2) var<storage, read> csrIdx: array<u32>;
@group(0) @binding(3) var<storage, read> fforce: array<f32>;
@group(0) @binding(4) var<storage, read_write> state: array<f32>;
@group(0) @binding(5) var<uniform> P: vec4f;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; let nc = u32(P.y); if (i >= nc) { return; }
  var F = vec3f(0.0); var T = vec3f(0.0);
  for (var q = csrPtr[i]; q < csrPtr[i + 1u]; q++) {
    let e = csrIdx[q]; let fi = e & 0x7fffffffu; let k = 12u * fi + select(0u, 6u, (e >> 31u) == 1u);
    F += vec3f(fforce[k], fforce[k+1u], fforce[k+2u]); T += vec3f(fforce[k+3u], fforce[k+4u], fforce[k+5u]);
  }
  let c = 25u * i; let dt = P.x;
  let mass = cells[c + 3u];
  let inv = mat3x3f(cells[c+13u], cells[c+16u], cells[c+19u], cells[c+14u], cells[c+17u], cells[c+20u], cells[c+15u], cells[c+18u], cells[c+21u]);
  let s = 12u * i;
  var v = vec3f(state[s+6u], state[s+7u], state[s+8u]) + dt * F / mass;
  var w = vec3f(state[s+9u], state[s+10u], state[s+11u]) + dt * (inv * T);
  state[s+6u] = v.x; state[s+7u] = v.y; state[s+8u] = v.z; state[s+9u] = w.x; state[s+10u] = w.y; state[s+11u] = w.z;
  state[s] += dt * v.x; state[s+1u] += dt * v.y; state[s+2u] += dt * v.z; state[s+3u] += dt * w.x; state[s+4u] += dt * w.y; state[s+5u] += dt * w.z;
}`,
  // ---- E-B: one dispatch per substep (ping-pong state; optional pair debug output)
  eb: /* wgsl */ `
@group(0) @binding(0) var<storage, read> fdata: array<f32>;
@group(0) @binding(1) var<storage, read> fidx: array<u32>;
@group(0) @binding(2) var<storage, read> cells: array<f32>;
@group(0) @binding(3) var<storage, read> csrPtr: array<u32>;
@group(0) @binding(4) var<storage, read> csrIdx: array<u32>;
@group(0) @binding(5) var<storage, read> sin: array<f32>;
@group(0) @binding(6) var<storage, read_write> sout: array<f32>;
@group(0) @binding(7) var<storage, read_write> dbg: array<f32>;
@group(0) @binding(8) var<uniform> P: vec4f;
${FORCE_FN}
fn ld(i: u32, o: u32) -> vec3f { return vec3f(sin[12u*i+o], sin[12u*i+o+1u], sin[12u*i+o+2u]); }
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; let nc = u32(P.y); if (i >= nc) { return; }
  var F = vec3f(0.0); var T = vec3f(0.0);
  for (var q = csrPtr[i]; q < csrPtr[i + 1u]; q++) {
    let e = csrIdx[q]; let fi = e & 0x7fffffffu; let side_b = (e >> 31u) == 1u;
    let a = fidx[2u * fi]; let bb = fidx[2u * fi + 1u];
    var ub = vec3f(0.0); var wb = vec3f(0.0);
    if (bb != 0xffffffffu) { ub = ld(bb, 0u); wb = ld(bb, 3u); }
    let o = pair(fi, ld(a, 0u), ld(a, 3u), ub, wb);
    var f6 = array<f32, 6>(o.fa.x, o.fa.y, o.fa.z, o.ta.x, o.ta.y, o.ta.z);
    if (side_b) { F += o.fb; T += o.tb; f6 = array<f32, 6>(o.fb.x, o.fb.y, o.fb.z, o.tb.x, o.tb.y, o.tb.z); } else { F += o.fa; T += o.ta; }
    if (P.z > 0.5) { let k = 12u * fi + select(0u, 6u, side_b); for (var j = 0u; j < 6u; j++) { dbg[k + j] = f6[j]; } }
  }
  let c = 25u * i; let dt = P.x; let mass = cells[c + 3u];
  let inv = mat3x3f(cells[c+13u], cells[c+16u], cells[c+19u], cells[c+14u], cells[c+17u], cells[c+20u], cells[c+15u], cells[c+18u], cells[c+21u]);
  let v = ld(i, 6u) + dt * F / mass; let w = ld(i, 9u) + dt * (inv * T);
  let u = ld(i, 0u) + dt * v; let th = ld(i, 3u) + dt * w;
  let s = 12u * i;
  sout[s] = u.x; sout[s+1u] = u.y; sout[s+2u] = u.z; sout[s+3u] = th.x; sout[s+4u] = th.y; sout[s+5u] = th.z;
  sout[s+6u] = v.x; sout[s+7u] = v.y; sout[s+8u] = v.z; sout[s+9u] = w.x; sout[s+10u] = w.y; sout[s+11u] = w.z;
}`,
  // ---- E-B-H: E-B with a synthetic SB-IL1 history per facet (κ ping-pong; the canonical
  // owner — side a — writes κ_new, both sides compute it identically). §6.2.
  ebh: /* wgsl */ `
@group(0) @binding(0) var<storage, read> fdata: array<f32>;
@group(0) @binding(1) var<storage, read> fidx: array<u32>;
@group(0) @binding(2) var<storage, read> cells: array<f32>;
@group(0) @binding(3) var<storage, read> csrPtr: array<u32>;
@group(0) @binding(4) var<storage, read> csrIdx: array<u32>;
@group(0) @binding(5) var<storage, read> sin: array<f32>;
@group(0) @binding(6) var<storage, read_write> sout: array<f32>;
@group(0) @binding(7) var<storage, read> kin: array<f32>;
@group(0) @binding(8) var<storage, read_write> kout: array<f32>;
@group(0) @binding(9) var<storage, read_write> dbg: array<f32>;
@group(0) @binding(10) var<uniform> P: vec4f;
fn v3(a: ptr<storage, array<f32>, read>, i: u32) -> vec3f { return vec3f((*a)[i], (*a)[i + 1u], (*a)[i + 2u]); }
fn ld(i: u32, o: u32) -> vec3f { return vec3f(sin[12u*i+o], sin[12u*i+o+1u], sin[12u*i+o+2u]); }
struct PairH { fa: vec3f, ta: vec3f, fb: vec3f, tb: vec3f, kap: f32 }
// canonical pair with history: δ_eq = sqrt(<δn>² + γ|δt|²), κ = max(κ_old, δ_eq), SB-IL1 d(κ)
fn pairh(fi: u32, ua: vec3f, wa: vec3f, ub: vec3f, wb: vec3f, kold: f32) -> PairH {
  let b = fi * 20u;
  let n = v3(&fdata, b); let p1 = v3(&fdata, b + 3u); let p2 = v3(&fdata, b + 6u);
  let ra = v3(&fdata, b + 9u); let rb = v3(&fdata, b + 12u);
  let kn = fdata[b + 15u]; let kt = fdata[b + 16u]; let kr1 = fdata[b + 17u]; let kr2 = fdata[b + 18u]; let ktor = fdata[b + 19u];
  let jump = (ub + cross(wb, rb)) - (ua + cross(wa, ra));
  let dw = wb - wa;
  let dn = dot(n, jump); let t1 = dot(p1, jump); let t2 = dot(p2, jump);
  let ell = max(length(ra - rb), 1e-6);
  let d0 = 1.0e-4 * ell; let wc = 6.6666667e-5; let gam = kt / kn;
  let dp = max(dn, 0.0);
  let deq = sqrt(dp * dp + gam * (t1 * t1 + t2 * t2));
  let kap = max(kold, deq);
  var d = 0.0;
  if (kap >= wc) { d = 1.0; } else if (kap > d0) { d = (kap - d0) / (kap * (1.0 - d0 / wc)); }
  let om = 1.0 - d;
  let fnn = om * kn * dp + kn * min(dn, 0.0);
  let f = fnn * n + om * kt * (t1 * p1 + t2 * p2);
  let m = om * (kr1 * dot(p1, dw) * p1 + kr2 * dot(p2, dw) * p2 + ktor * dot(n, dw) * n);
  var o: PairH;
  o.fa = f; o.ta = cross(ra, f) + m; o.fb = -f; o.tb = -cross(rb, f) - m; o.kap = kap;
  return o;
}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; let nc = u32(P.y); if (i >= nc) { return; }
  var F = vec3f(0.0); var T = vec3f(0.0);
  for (var q = csrPtr[i]; q < csrPtr[i + 1u]; q++) {
    let e = csrIdx[q]; let fi = e & 0x7fffffffu; let side_b = (e >> 31u) == 1u;
    let a = fidx[2u * fi]; let bb = fidx[2u * fi + 1u];
    var ub = vec3f(0.0); var wb = vec3f(0.0);
    if (bb != 0xffffffffu) { ub = ld(bb, 0u); wb = ld(bb, 3u); }
    let o = pairh(fi, ld(a, 0u), ld(a, 3u), ub, wb, kin[fi]);
    if (side_b) { F += o.fb; T += o.tb; } else { F += o.fa; T += o.ta; kout[fi] = o.kap; }
    if (P.z > 0.5) {
      let k = 14u * fi + select(0u, 7u, side_b);
      var f7 = array<f32, 7>(o.fa.x, o.fa.y, o.fa.z, o.ta.x, o.ta.y, o.ta.z, o.kap);
      if (side_b) { f7 = array<f32, 7>(o.fb.x, o.fb.y, o.fb.z, o.tb.x, o.tb.y, o.tb.z, o.kap); }
      for (var j = 0u; j < 7u; j++) { dbg[k + j] = f7[j]; }
    }
  }
  let c = 25u * i; let dt = P.x; let mass = cells[c + 3u];
  let inv = mat3x3f(cells[c+13u], cells[c+16u], cells[c+19u], cells[c+14u], cells[c+17u], cells[c+20u], cells[c+15u], cells[c+18u], cells[c+21u]);
  let v = ld(i, 6u) + dt * F / mass; let w = ld(i, 9u) + dt * (inv * T);
  let u = ld(i, 0u) + dt * v; let th = ld(i, 3u) + dt * w;
  let s = 12u * i;
  sout[s] = u.x; sout[s+1u] = u.y; sout[s+2u] = u.z; sout[s+3u] = th.x; sout[s+4u] = th.y; sout[s+5u] = th.z;
  sout[s+6u] = v.x; sout[s+7u] = v.y; sout[s+8u] = v.z; sout[s+9u] = w.x; sout[s+10u] = w.y; sout[s+11u] = w.z;
}`,
  empty: /* wgsl */ `@compute @workgroup_size(${WG}) fn main() {}`,
  // ---- I: y = μ M x + K x (cell-gather form, canonical pairs); x, y 6 f32 per cell
  matvec: /* wgsl */ `
@group(0) @binding(0) var<storage, read> fdata: array<f32>;
@group(0) @binding(1) var<storage, read> fidx: array<u32>;
@group(0) @binding(2) var<storage, read> cells: array<f32>;
@group(0) @binding(3) var<storage, read> csrPtr: array<u32>;
@group(0) @binding(4) var<storage, read> csrIdx: array<u32>;
@group(0) @binding(5) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(7) var<uniform> P: vec4f;
${FORCE_FN}
fn lx(i: u32, o: u32) -> vec3f { return vec3f(x[6u*i+o], x[6u*i+o+1u], x[6u*i+o+2u]); }
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; let nc = u32(P.y); if (i >= nc) { return; }
  var F = vec3f(0.0); var T = vec3f(0.0);
  for (var q = csrPtr[i]; q < csrPtr[i + 1u]; q++) {
    let e = csrIdx[q]; let fi = e & 0x7fffffffu; let side_b = (e >> 31u) == 1u;
    let a = fidx[2u * fi]; let bb = fidx[2u * fi + 1u];
    var ub = vec3f(0.0); var wb = vec3f(0.0);
    if (bb != 0xffffffffu) { ub = lx(bb, 0u); wb = lx(bb, 3u); }
    let o = pair(fi, lx(a, 0u), lx(a, 3u), ub, wb);
    if (side_b) { F += o.fb; T += o.tb; } else { F += o.fa; T += o.ta; }
  }
  let c = 25u * i; let mu = P.x; let mass = cells[c + 3u];
  let I = mat3x3f(cells[c+4u], cells[c+7u], cells[c+10u], cells[c+5u], cells[c+8u], cells[c+11u], cells[c+6u], cells[c+9u], cells[c+12u]);
  let yu = mu * mass * lx(i, 0u) - F; let yw = mu * (I * lx(i, 3u)) - T;
  y[6u*i] = yu.x; y[6u*i+1u] = yu.y; y[6u*i+2u] = yu.z; y[6u*i+3u] = yw.x; y[6u*i+4u] = yw.y; y[6u*i+5u] = yw.z;
}`,
  // ---- I: z = D⁻¹ r (6×6 per cell)
  bsolve: /* wgsl */ `
@group(0) @binding(0) var<storage, read> dinv: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
@group(0) @binding(3) var<uniform> P: vec4f;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= u32(P.y)) { return; }
  for (var a = 0u; a < 6u; a++) { var s = 0.0; for (var b = 0u; b < 6u; b++) { s += dinv[36u*i + 6u*a + b] * r[6u*i + b]; } z[6u*i + a] = s; }
}`,
  // ---- I: dot product, pass 1 (workgroup partial sums) and pass 2 (single workgroup)
  dot1: /* wgsl */ `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
@group(0) @binding(3) var<uniform> P: vec4f;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) wg: vec3u) {
  let n = u32(P.y); var s = 0.0;
  for (var i = g.x; i < n; i += u32(P.z) * 256u) { s += x[i] * y[i]; }
  sh[l.x] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (l.x < k) { sh[l.x] += sh[l.x + k]; } workgroupBarrier(); }
  if (l.x == 0u) { part[wg.x] = sh[0]; }
}`,
  dot2: /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> res: array<f32>;
@group(0) @binding(2) var<uniform> P: vec4f;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3u) {
  var s = 0.0; for (var i = l.x; i < u32(P.z); i += 256u) { s += part[i]; }
  sh[l.x] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (l.x < k) { sh[l.x] += sh[l.x + k]; } workgroupBarrier(); }
  if (l.x == 0u) { res[0] = sh[0]; }
}`,
  // ---- I: PR restriction rc = Pᵀ r (thread per aggregate) and prolongation x += P z
  restrict: /* wgsl */ `
@group(0) @binding(0) var<storage, read> cells: array<f32>;
@group(0) @binding(1) var<storage, read> aggData: array<f32>;
@group(0) @binding(2) var<storage, read> aggPtr: array<u32>;
@group(0) @binding(3) var<storage, read> aggIdx: array<u32>;
@group(0) @binding(4) var<storage, read> r: array<f32>;
@group(0) @binding(5) var<storage, read_write> rc: array<f32>;
@group(0) @binding(6) var<uniform> P: vec4f;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let a = g.x; if (a >= u32(P.z)) { return; }
  let cen = vec3f(aggData[4u*a], aggData[4u*a+1u], aggData[4u*a+2u]); let rho = aggData[4u*a+3u];
  var t = vec3f(0.0); var w = vec3f(0.0);
  for (var q = aggPtr[a]; q < aggPtr[a + 1u]; q++) {
    let i = aggIdx[q]; let x = vec3f(cells[25u*i], cells[25u*i+1u], cells[25u*i+2u]) - cen;
    let ru = vec3f(r[6u*i], r[6u*i+1u], r[6u*i+2u]); let rt = vec3f(r[6u*i+3u], r[6u*i+4u], r[6u*i+5u]);
    t += ru; w += (cross(x, ru) + rt) / rho;
  }
  rc[6u*a] = t.x; rc[6u*a+1u] = t.y; rc[6u*a+2u] = t.z; rc[6u*a+3u] = w.x; rc[6u*a+4u] = w.y; rc[6u*a+5u] = w.z;
}`,
  prolong: /* wgsl */ `
@group(0) @binding(0) var<storage, read> cells: array<f32>;
@group(0) @binding(1) var<storage, read> aggData: array<f32>;
@group(0) @binding(2) var<storage, read> aggOf: array<u32>;
@group(0) @binding(3) var<storage, read> zc: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<uniform> P: vec4f;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= u32(P.y)) { return; }
  let a = aggOf[i]; let cen = vec3f(aggData[4u*a], aggData[4u*a+1u], aggData[4u*a+2u]); let rho = aggData[4u*a+3u];
  let r = vec3f(cells[25u*i], cells[25u*i+1u], cells[25u*i+2u]) - cen;
  let t = vec3f(zc[6u*a], zc[6u*a+1u], zc[6u*a+2u]); let w = vec3f(zc[6u*a+3u], zc[6u*a+4u], zc[6u*a+5u]) / rho;
  let du = t + cross(w, r);
  x[6u*i] += du.x; x[6u*i+1u] += du.y; x[6u*i+2u] += du.z; x[6u*i+3u] += w.x; x[6u*i+4u] += w.y; x[6u*i+5u] += w.z;
}`,
};

// ---- CPU f64 reference of the same symplectic Euler
function cpuLeapfrog(m, dt, steps) {
  const { nc, nf, cells, fidx, fdata, csrPtr, csrIdx } = m;
  const s = new Float64Array(12 * nc);
  for (let i = 0; i < nc; i++) for (let k = 0; k < 3; k++) s[12 * i + 6 + k] = cells[25 * i + 22 + k];
  const ff = new Float64Array(12 * nf);
  const cr = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dt3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  for (let st = 0; st < steps; st++) {
    for (let f = 0; f < nf; f++) {
      const a = fidx[2 * f], b = fidx[2 * f + 1], d = 20 * f, g = (k) => [fdata[d + k], fdata[d + k + 1], fdata[d + k + 2]];
      const n = g(0), p1 = g(3), p2 = g(6), ra = g(9), rb = g(12);
      const ua = [s[12 * a], s[12 * a + 1], s[12 * a + 2]], wa = [s[12 * a + 3], s[12 * a + 4], s[12 * a + 5]];
      let ub = [0, 0, 0], wb = [0, 0, 0];
      if (b !== 0xffffffff) { ub = [s[12 * b], s[12 * b + 1], s[12 * b + 2]]; wb = [s[12 * b + 3], s[12 * b + 4], s[12 * b + 5]]; }
      const cb = cr(wb, rb), ca = cr(wa, ra);
      const j = [ub[0] + cb[0] - ua[0] - ca[0], ub[1] + cb[1] - ua[1] - ca[1], ub[2] + cb[2] - ua[2] - ca[2]];
      const dw = [wb[0] - wa[0], wb[1] - wa[1], wb[2] - wa[2]];
      const cn = fdata[d + 15] * dt3(n, j), c1 = fdata[d + 16] * dt3(p1, j), c2 = fdata[d + 16] * dt3(p2, j);
      const fv = [0, 1, 2].map((k) => cn * n[k] + c1 * p1[k] + c2 * p2[k]);
      const m1 = fdata[d + 17] * dt3(p1, dw), m2 = fdata[d + 18] * dt3(p2, dw), m3 = fdata[d + 19] * dt3(n, dw);
      const mv = [0, 1, 2].map((k) => m1 * p1[k] + m2 * p2[k] + m3 * n[k]);
      const ta = cr(ra, fv), tb = cr(rb, fv);
      for (let k = 0; k < 3; k++) { ff[12 * f + k] = fv[k]; ff[12 * f + 3 + k] = ta[k] + mv[k]; ff[12 * f + 6 + k] = -fv[k]; ff[12 * f + 9 + k] = -tb[k] - mv[k]; }
    }
    for (let i = 0; i < nc; i++) {
      const F = [0, 0, 0], T = [0, 0, 0];
      for (let q = csrPtr[i]; q < csrPtr[i + 1]; q++) {
        const e = csrIdx[q], fi = e & 0x7fffffff, o = 12 * fi + ((e >>> 31) ? 6 : 0);
        for (let k = 0; k < 3; k++) { F[k] += ff[o + k]; T[k] += ff[o + 3 + k]; }
      }
      const c = 25 * i, mass = cells[c + 3];
      for (let k = 0; k < 3; k++) {
        s[12 * i + 6 + k] += dt * F[k] / mass;
        let w = 0; for (let l = 0; l < 3; l++) w += cells[c + 13 + 3 * k + l] * T[l];
        s[12 * i + 9 + k] += dt * w;
      }
      for (let k = 0; k < 6; k++) s[12 * i + k] += dt * s[12 * i + 6 + k];
    }
  }
  return s;
}

async function probe(labels, chains, reps, warm, verifySteps) {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const ts = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({ requiredFeatures: ts ? ['timestamp-query'] : [], requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize, maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage } });
  const index = await (await fetch('index.json')).json();
  const dt = index.dt_contract_s;
  const out = { adapter: adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture } : null, ua: navigator.userAgent, timestamp_query: ts, dt, meshes: [] };
  const pipes = {};
  for (const [k, code] of Object.entries(SHADERS)) {
    const mod = device.createShaderModule({ code });
    const info = await mod.getCompilationInfo();
    const errs = info.messages.filter((x) => x.type === 'error');
    if (errs.length) throw new Error(k + ': ' + errs.map((e) => e.lineNum + ':' + e.message).join('; '));
    pipes[k] = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
  }
  const S = GPUBufferUsage.STORAGE, C = GPUBufferUsage.COPY_SRC, D = GPUBufferUsage.COPY_DST;
  const mk = (arr, usage = S) => { const b = device.createBuffer({ size: Math.max(16, arr.byteLength), usage: usage | D }); device.queue.writeBuffer(b, 0, arr); return b; };
  const zeros = (n) => device.createBuffer({ size: Math.max(16, 4 * n), usage: S | C | D });
  const uni = (v) => { const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | D }); device.queue.writeBuffer(b, 0, v); return b; };
  const bg = (pipe, bufs) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const readback = async (buf, n) => { const r = device.createBuffer({ size: 4 * n, usage: GPUBufferUsage.MAP_READ | D }); const e = device.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, r, 0, 4 * n); device.queue.submit([e.finish()]); await r.mapAsync(GPUMapMode.READ); const v = new Float32Array(r.getMappedRange().slice(0)); r.unmap(); return v; };
  const qs = ts ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  const qres = ts ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | C }) : null;
  // run a chain: steps = array of [pipeline, bindGroup, workgroups]
  async function chain(steps) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(ts ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    for (const [p, g, n] of steps) { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(n); }
    pass.end();
    let rb = null;
    if (ts) { enc.resolveQuerySet(qs, 0, 2, qres, 0); rb = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | D }); enc.copyBufferToBuffer(qres, 0, rb, 0, 16); }
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const wall = performance.now() - t0;
    let gpu = NaN;
    if (rb) { await rb.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(rb.getMappedRange()); gpu = Number(t[1] - t[0]) / 1e6; rb.unmap(); }
    return { gpu, wall };
  }
  const stats = (v) => { const s = [...v].sort((a, b) => a - b); return { median: s[Math.floor(s.length / 2)], p95: s[Math.min(s.length - 1, Math.ceil(0.95 * s.length) - 1)], min: s[0], max: s[s.length - 1] }; };
  for (const label of labels) {
    const meta = index.meshes.find((x) => x.label === label);
    const m = parseMesh(await (await fetch(meta.file)).arrayBuffer());
    const res = { label, cells: m.nc, facets: m.nf, aggregates: m.nagg, file_bytes: m.bytes, timings: [], checks: {} };
    const bFd = mk(m.fdata), bFi = mk(m.fidx), bCells = mk(m.cells), bPtr = mk(m.csrPtr), bIdx = mk(m.csrIdx), bDinv = mk(m.dinv);
    const bAggData = mk(m.aggData), bAggPtr = mk(m.aggPtr), bAggIdx = mk(m.aggIdx), bAggOf = mk(m.aggOf);
    const init = new Float32Array(12 * m.nc); for (let i = 0; i < m.nc; i++) for (let k = 0; k < 3; k++) init[12 * i + 6 + k] = m.cells[25 * i + 22 + k];
    const st0 = zeros(12 * m.nc), st1 = zeros(12 * m.nc), stA = zeros(12 * m.nc), ffor = zeros(12 * m.nf), dbg = zeros(12 * m.nf);
    const reset = () => { device.queue.writeBuffer(st0, 0, init); device.queue.writeBuffer(st1, 0, init); device.queue.writeBuffer(stA, 0, init); };
    const wc = Math.ceil(m.nc / WG), wf = Math.ceil(m.nf / WG);
    const uA = uni(new Uint32Array([0, m.nf, 0, 0])), uC = uni(new Float32Array([dt, m.nc, 0, 0])), uB = uni(new Float32Array([dt, m.nc, 0, 0])), uBd = uni(new Float32Array([dt, m.nc, 1, 0]));
    const gFac = bg(pipes.eaFacet, [bFd, bFi, stA, ffor, uA]), gCell = bg(pipes.eaCell, [bCells, bPtr, bIdx, ffor, stA, uC]);
    const gB01 = bg(pipes.eb, [bFd, bFi, bCells, bPtr, bIdx, st0, st1, dbg, uB]), gB10 = bg(pipes.eb, [bFd, bFi, bCells, bPtr, bIdx, st1, st0, dbg, uB]);
    const gBdbg = bg(pipes.eb, [bFd, bFi, bCells, bPtr, bIdx, st0, st1, dbg, uBd]);
    const gEmpty = device.createBindGroup({ layout: pipes.empty.getBindGroupLayout(0), entries: [] });
    // I-branch buffers
    const n6 = 6 * m.nc, xv = zeros(n6), yv = zeros(n6), zv = zeros(n6), rc = zeros(6 * m.nagg);
    const xinit = new Float32Array(n6); for (let i = 0; i < n6; i++) xinit[i] = Math.sin(0.37 * i);
    device.queue.writeBuffer(xv, 0, xinit);
    const nparts = Math.min(1024, Math.ceil(n6 / 256)), part = zeros(nparts), dres = zeros(4);
    const uM = uni(new Float32Array([2.25 * 480 * 480, m.nc, m.nagg, 0])), uD = uni(new Float32Array([0, n6, nparts, 0]));
    const gMv = bg(pipes.matvec, [bFd, bFi, bCells, bPtr, bIdx, xv, yv, uM]), gBs = bg(pipes.bsolve, [bDinv, yv, zv, uM]);
    const gD1 = bg(pipes.dot1, [xv, yv, part, uD]), gD2 = bg(pipes.dot2, [part, dres, uD]);
    const gR = bg(pipes.restrict, [bCells, bAggData, bAggPtr, bAggIdx, yv, rc, uM]), gP = bg(pipes.prolong, [bCells, bAggData, bAggOf, rc, zv, uM]);
    const wa = Math.ceil(m.nagg / WG);
    const variants = {
      'E-A (2 dispatch)': (n) => { const s = []; for (let k = 0; k < n; k++) { s.push([pipes.eaFacet, gFac, wf], [pipes.eaCell, gCell, wc]); } return s; },
      'E-B (1 dispatch)': (n) => { const s = []; for (let k = 0; k < n; k++) s.push([pipes.eb, k % 2 ? gB10 : gB01, wc]); return s; },
      'C empty': (n) => Array.from({ length: n }, () => [pipes.empty, gEmpty, wc]),
      'I matvec': (n) => Array.from({ length: n }, () => [pipes.matvec, gMv, wc]),
      'I block solve': (n) => Array.from({ length: n }, () => [pipes.bsolve, gBs, wc]),
      'I dot (2 passes)': (n) => { const s = []; for (let k = 0; k < n; k++) s.push([pipes.dot1, gD1, nparts], [pipes.dot2, gD2, 1]); return s; },
      'I restrict+prolong': (n) => { const s = []; for (let k = 0; k < n; k++) s.push([pipes.restrict, gR, wa], [pipes.prolong, gP, wc]); return s; },
    };
    for (const [name, build] of Object.entries(variants)) {
      for (const n of (chains.length ? chains : [])) {
        const steps = build(n);
        for (let k = 0; k < warm; k++) { reset(); await chain(steps); }
        const g = [], w = [];
        for (let k = 0; k < reps; k++) { reset(); const r = await chain(steps); g.push(r.gpu); w.push(r.wall); }
        res.timings.push({ variant: name, substeps: n, dispatches: steps.length, gpu_ms: stats(g), wall_ms: stats(w) });
      }
    }
    // ---- verification: E-A and E-B against CPU f64 after verifySteps substeps
    reset(); await chain(variants['E-A (2 dispatch)'](verifySteps));
    const sA = await readback(stA, 12 * m.nc);
    reset(); await chain(variants['E-B (1 dispatch)'](verifySteps));
    const sB = await readback(verifySteps % 2 ? st1 : st0, 12 * m.nc);
    const t0 = performance.now(); const ref = cpuLeapfrog(m, dt, verifySteps); const tcpu = performance.now() - t0;
    const err = (s) => { let e = 0, mx = 0; for (let i = 0; i < m.nc; i++) for (let k = 0; k < 3; k++) { e = Math.max(e, Math.abs(s[12 * i + k] - ref[12 * i + k])); mx = Math.max(mx, Math.abs(ref[12 * i + k])); } return e / mx; };
    // ---- E-B pair check: one debug substep after 100 substeps (non-zero forces)
    reset(); await chain(variants['E-B (1 dispatch)'](100)); await chain([[pipes.eb, gBdbg, wc]]);
    const d = await readback(dbg, 12 * m.nf);
    let bitwise = 0, interior = 0, momRel = 0, ulp = 0;
    const cr = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dv = new DataView(d.buffer);
    for (let f = 0; f < m.nf; f++) {
      if (m.fidx[2 * f + 1] === 0xffffffff) continue;
      interior++;
      let same = true;
      for (let k = 0; k < 3; k++) { const x = dv.getUint32(4 * (12 * f + k), true), y = dv.getUint32(4 * (12 * f + 6 + k), true); if (((x ^ y) >>> 0) !== 0x80000000 && !(((x & 0x7fffffff) >>> 0) === 0 && ((y & 0x7fffffff) >>> 0) === 0)) { same = false; ulp = Math.max(ulp, Math.abs((x & 0x7fffffff) - (y & 0x7fffffff))); } }
      if (same) bitwise++;
      const a = m.fidx[2 * f], b = m.fidx[2 * f + 1];
      const xa = [m.cells[25 * a], m.cells[25 * a + 1], m.cells[25 * a + 2]], xb = [m.cells[25 * b], m.cells[25 * b + 1], m.cells[25 * b + 2]];
      const Fa = [d[12 * f], d[12 * f + 1], d[12 * f + 2]], Ta = [d[12 * f + 3], d[12 * f + 4], d[12 * f + 5]], Fb = [d[12 * f + 6], d[12 * f + 7], d[12 * f + 8]], Tb = [d[12 * f + 9], d[12 * f + 10], d[12 * f + 11]];
      const c1 = cr(xa, Fa), c2 = cr(xb, Fb);
      const res3 = [0, 1, 2].map((k) => Ta[k] + Tb[k] + c1[k] + c2[k]);
      const scale = Math.hypot(...Ta) + Math.hypot(...c1) + 1e-30;
      momRel = Math.max(momRel, Math.hypot(...res3) / scale);
    }
    let fmax = 0; for (let k = 0; k < d.length; k++) fmax = Math.max(fmax, Math.abs(d[k]));
    res.checks = { pair_check_max_abs_force: fmax, verify_substeps: verifySteps, EA_rel_err: err(sA), EB_rel_err: err(sB), cpu_f64_ms: tcpu, pair_force_bitwise_opposite: `${bitwise}/${interior}`, pair_max_ulp_mismatch: ulp, moment_balance_max_rel: momRel };
    out.meshes.push(res);
    for (const b of [bFd, bFi, bCells, bPtr, bIdx, bDinv, bAggData, bAggPtr, bAggIdx, bAggOf, st0, st1, stA, ffor, dbg, xv, yv, zv, rc, part, dres]) b.destroy();
  }
  return out;
}

// ---- CPU f64 reference of E-B-H (same law, same order: κ at q_n, then forces, then update)
function cpuLeapfrogH(m, dt, steps) {
  const { nc, nf, cells, fidx, fdata, csrPtr, csrIdx } = m;
  const s = new Float64Array(12 * nc);
  for (let i = 0; i < nc; i++) for (let k = 0; k < 3; k++) s[12 * i + 6 + k] = cells[25 * i + 22 + k];
  const kap = new Float64Array(nf);
  const ff = new Float64Array(12 * nf);
  const cr = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dt3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  for (let st = 0; st < steps; st++) {
    for (let f = 0; f < nf; f++) {
      const a = fidx[2 * f], b = fidx[2 * f + 1], d = 20 * f, g = (k) => [fdata[d + k], fdata[d + k + 1], fdata[d + k + 2]];
      const n = g(0), p1 = g(3), p2 = g(6), ra = g(9), rb = g(12);
      const kn = fdata[d + 15], kt = fdata[d + 16];
      const ua = [s[12 * a], s[12 * a + 1], s[12 * a + 2]], wa = [s[12 * a + 3], s[12 * a + 4], s[12 * a + 5]];
      let ub = [0, 0, 0], wb = [0, 0, 0];
      if (b !== 0xffffffff) { ub = [s[12 * b], s[12 * b + 1], s[12 * b + 2]]; wb = [s[12 * b + 3], s[12 * b + 4], s[12 * b + 5]]; }
      const cb = cr(wb, rb), ca = cr(wa, ra);
      const j = [ub[0] + cb[0] - ua[0] - ca[0], ub[1] + cb[1] - ua[1] - ca[1], ub[2] + cb[2] - ua[2] - ca[2]];
      const dw = [wb[0] - wa[0], wb[1] - wa[1], wb[2] - wa[2]];
      const dn = dt3(n, j), t1 = dt3(p1, j), t2 = dt3(p2, j);
      const ell = Math.max(Math.hypot(ra[0] - rb[0], ra[1] - rb[1], ra[2] - rb[2]), 1e-6);
      const d0 = 1e-4 * ell, wc = 6.6666667e-5, gam = kt / kn, dp = Math.max(dn, 0);
      const deq = Math.sqrt(dp * dp + gam * (t1 * t1 + t2 * t2));
      const kp = Math.max(kap[f], deq); kap[f] = kp;
      let dd = 0; if (kp >= wc) dd = 1; else if (kp > d0) dd = (kp - d0) / (kp * (1 - d0 / wc));
      const om = 1 - dd, fnn = om * kn * dp + kn * Math.min(dn, 0);
      const fv = [0, 1, 2].map((k) => fnn * n[k] + om * kt * (t1 * p1[k] + t2 * p2[k]));
      const m1 = fdata[d + 17] * dt3(p1, dw), m2 = fdata[d + 18] * dt3(p2, dw), m3 = fdata[d + 19] * dt3(n, dw);
      const mv = [0, 1, 2].map((k) => om * (m1 * p1[k] + m2 * p2[k] + m3 * n[k]));
      const ta = cr(ra, fv), tb = cr(rb, fv);
      for (let k = 0; k < 3; k++) { ff[12 * f + k] = fv[k]; ff[12 * f + 3 + k] = ta[k] + mv[k]; ff[12 * f + 6 + k] = -fv[k]; ff[12 * f + 9 + k] = -tb[k] - mv[k]; }
    }
    for (let i = 0; i < nc; i++) {
      const F = [0, 0, 0], T = [0, 0, 0];
      for (let q = csrPtr[i]; q < csrPtr[i + 1]; q++) {
        const e = csrIdx[q], fi = e & 0x7fffffff, o = 12 * fi + ((e >>> 31) ? 6 : 0);
        for (let k = 0; k < 3; k++) { F[k] += ff[o + k]; T[k] += ff[o + 3 + k]; }
      }
      const c = 25 * i, mass = cells[c + 3];
      for (let k = 0; k < 3; k++) s[12 * i + 6 + k] += dt * F[k] / mass;
      for (let k = 0; k < 3; k++) { let w = 0; for (let l = 0; l < 3; l++) w += cells[c + 13 + 3 * k + l] * T[l]; s[12 * i + 9 + k] += dt * w; }
      for (let k = 0; k < 6; k++) s[12 * i + k] += dt * s[12 * i + 6 + k];
    }
  }
  return { s, kap };
}

async function makeDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const ts = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({ requiredFeatures: ts ? ['timestamp-query'] : [], requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize, maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage } });
  const pipes = {};
  for (const [k, code] of Object.entries(SHADERS)) {
    const mod = device.createShaderModule({ code });
    const info = await mod.getCompilationInfo();
    const errs = info.messages.filter((x) => x.type === 'error');
    if (errs.length) throw new Error(k + ': ' + errs.map((e) => e.lineNum + ':' + e.message).join('; '));
    pipes[k] = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
  }
  const S = GPUBufferUsage.STORAGE, C = GPUBufferUsage.COPY_SRC, D = GPUBufferUsage.COPY_DST;
  const mk = (arr) => { const b = device.createBuffer({ size: Math.max(16, arr.byteLength), usage: S | D }); device.queue.writeBuffer(b, 0, arr); return b; };
  const zeros = (n) => device.createBuffer({ size: Math.max(16, 4 * n), usage: S | C | D });
  const uni = (v) => { const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | D }); device.queue.writeBuffer(b, 0, v); return b; };
  const bg = (pipe, bufs) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const readback = async (buf, n) => { const r = device.createBuffer({ size: 4 * n, usage: GPUBufferUsage.MAP_READ | D }); const e = device.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, r, 0, 4 * n); device.queue.submit([e.finish()]); await r.mapAsync(GPUMapMode.READ); const v = new Float32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return v; };
  const qs = ts ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  const qres = ts ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | C }) : null;
  async function chain(steps) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(ts ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    for (const [p, g, n] of steps) { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(n); }
    pass.end();
    let rb = null;
    if (ts) { enc.resolveQuerySet(qs, 0, 2, qres, 0); rb = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | D }); enc.copyBufferToBuffer(qres, 0, rb, 0, 16); }
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const wall = performance.now() - t0;
    let gpu = NaN;
    if (rb) { await rb.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(rb.getMappedRange()); gpu = Number(t[1] - t[0]) / 1e6; rb.unmap(); rb.destroy(); }
    return { gpu, wall };
  }
  return { adapter, device, ts, pipes, mk, zeros, uni, bg, readback, chain };
}

const statsOf = (v) => { const s = [...v].sort((a, b) => a - b); return { median: s[Math.floor(s.length / 2)], p95: s[Math.min(s.length - 1, Math.ceil(0.95 * s.length) - 1)], min: s[0], max: s[s.length - 1], n: s.length }; };

// E-B vs E-B-H on one mesh: timings, CPU f64 verification with damage, pair check.
async function setupH(G, label, index) {
  const meta = index.meshes.find((x) => x.label === label);
  const m = parseMesh(await (await fetch(meta.file)).arrayBuffer());
  const dt = index.dt_contract_s;
  const bFd = G.mk(m.fdata), bFi = G.mk(m.fidx), bCells = G.mk(m.cells), bPtr = G.mk(m.csrPtr), bIdx = G.mk(m.csrIdx);
  const init = new Float32Array(12 * m.nc); for (let i = 0; i < m.nc; i++) for (let k = 0; k < 3; k++) init[12 * i + 6 + k] = m.cells[25 * i + 22 + k];
  const st0 = G.zeros(12 * m.nc), st1 = G.zeros(12 * m.nc), k0 = G.zeros(m.nf), k1 = G.zeros(m.nf), dbg = G.zeros(14 * m.nf);
  const reset = () => { G.device.queue.writeBuffer(st0, 0, init); G.device.queue.writeBuffer(st1, 0, init); G.device.queue.writeBuffer(k0, 0, new Float32Array(m.nf)); G.device.queue.writeBuffer(k1, 0, new Float32Array(m.nf)); };
  const uB = G.uni(new Float32Array([dt, m.nc, 0, 0])), uBd = G.uni(new Float32Array([dt, m.nc, 1, 0]));
  const wc = Math.ceil(m.nc / WG);
  const gB01 = G.bg(G.pipes.eb, [bFd, bFi, bCells, bPtr, bIdx, st0, st1, dbg, uB]), gB10 = G.bg(G.pipes.eb, [bFd, bFi, bCells, bPtr, bIdx, st1, st0, dbg, uB]);
  const gH01 = G.bg(G.pipes.ebh, [bFd, bFi, bCells, bPtr, bIdx, st0, st1, k0, k1, dbg, uB]), gH10 = G.bg(G.pipes.ebh, [bFd, bFi, bCells, bPtr, bIdx, st1, st0, k1, k0, dbg, uB]);
  const gHd = G.bg(G.pipes.ebh, [bFd, bFi, bCells, bPtr, bIdx, st0, st1, k0, k1, dbg, uBd]);
  const variants = {
    'E-B linear': (n) => { const s = []; for (let k = 0; k < n; k++) s.push([G.pipes.eb, k % 2 ? gB10 : gB01, wc]); return s; },
    'E-B-H history': (n) => { const s = []; for (let k = 0; k < n; k++) s.push([G.pipes.ebh, k % 2 ? gH10 : gH01, wc]); return s; },
  };
  const destroy = () => { for (const b of [bFd, bFi, bCells, bPtr, bIdx, st0, st1, k0, k1, dbg]) b.destroy(); };
  return { m, dt, reset, variants, gHd, wc, st0, st1, k0, k1, dbg, destroy };
}

async function probeH(labels, chains, reps, warm, verifySteps) {
  const G = await makeDevice();
  const index = await (await fetch('index.json')).json();
  const out = { adapter: G.adapter.info ? { vendor: G.adapter.info.vendor, architecture: G.adapter.info.architecture } : null, ua: navigator.userAgent, meshes: [] };
  for (const label of labels) {
    const H = await setupH(G, label, index);
    const m = H.m;
    const res = { label, cells: m.nc, facets: m.nf, timings: [], checks: {} };
    for (const [name, build] of Object.entries(H.variants)) {
      for (const n of chains) {
        const steps = build(n);
        for (let k = 0; k < warm; k++) { H.reset(); await G.chain(steps); }
        const g = [], w = [];
        for (let k = 0; k < reps; k++) { H.reset(); const r = await G.chain(steps); g.push(r.gpu); w.push(r.wall); }
        res.timings.push({ variant: name, substeps: n, gpu_ms: statsOf(g), wall_ms: statsOf(w) });
      }
    }
    // verification with damage against CPU f64
    H.reset(); await G.chain(H.variants['E-B-H history'](verifySteps));
    const sG = await G.readback(verifySteps % 2 ? H.st1 : H.st0, 12 * m.nc);
    const kG = await G.readback(verifySteps % 2 ? H.k1 : H.k0, m.nf);
    const t0 = performance.now(); const ref = cpuLeapfrogH(m, H.dt, verifySteps); const tcpu = performance.now() - t0;
    let e = 0, mx = 0; for (let i = 0; i < m.nc; i++) for (let k = 0; k < 3; k++) { e = Math.max(e, Math.abs(sG[12 * i + k] - ref.s[12 * i + k])); mx = Math.max(mx, Math.abs(ref.s[12 * i + k])); }
    const dmg = (kp, f) => { const d = m.fdata, b = 20 * f; const ra = [d[b + 9], d[b + 10], d[b + 11]], rb = [d[b + 12], d[b + 13], d[b + 14]]; const ell = Math.max(Math.hypot(ra[0] - rb[0], ra[1] - rb[1], ra[2] - rb[2]), 1e-6); const d0 = 1e-4 * ell, wc = 6.6666667e-5; return kp >= wc ? 1 : kp > d0 ? (kp - d0) / (kp * (1 - d0 / wc)) : 0; };
    let dmax = 0, onG = 0, onC = 0, brG = 0, brC = 0, flip = 0;
    for (let f = 0; f < m.nf; f++) {
      const a = dmg(kG[f], f), b = dmg(ref.kap[f], f);
      dmax = Math.max(dmax, Math.abs(a - b)); onG += a > 0; onC += b > 0; brG += a >= 1; brC += b >= 1; flip += (a > 0) !== (b > 0);
    }
    // pair check with history after 100 substeps
    H.reset(); await G.chain(H.variants['E-B-H history'](100)); await G.chain([[G.pipes.ebh, H.gHd, H.wc]]);
    const d = await G.readback(H.dbg, 14 * m.nf); const dv = new DataView(d.buffer);
    let bit = 0, inter = 0, kapSame = 0;
    for (let f = 0; f < m.nf; f++) {
      if (m.fidx[2 * f + 1] === 0xffffffff) continue;
      inter++; let same = true;
      for (let k = 0; k < 3; k++) { const x = dv.getUint32(4 * (14 * f + k), true), y = dv.getUint32(4 * (14 * f + 7 + k), true); if (((x ^ y) >>> 0) !== 0x80000000 && !(((x & 0x7fffffff) >>> 0) === 0 && ((y & 0x7fffffff) >>> 0) === 0)) same = false; }
      if (same) bit++;
      if (dv.getUint32(4 * (14 * f + 6), true) === dv.getUint32(4 * (14 * f + 13), true)) kapSame++;
    }
    res.checks = { verify_substeps: verifySteps, pos_rel_err: e / mx, cpu_f64_ms: tcpu, damage_max_abs_diff: dmax, damaged_facets_gpu: onG, damaged_facets_cpu: onC, broken_gpu: brG, broken_cpu: brC, damage_onset_disagreements: flip,
      pair_force_bitwise_opposite: `${bit}/${inter}`, pair_kappa_bitwise_equal: `${kapSame}/${inter}` };
    out.meshes.push(res);
    H.destroy();
  }
  return out;
}

// Sustained load: E-B-H chains of n substeps back to back for `minutes`, per-chain GPU time.
async function sustain(label, n, minutes) {
  const G = await makeDevice();
  const index = await (await fetch('index.json')).json();
  const H = await setupH(G, label, index);
  const steps = H.variants['E-B-H history'](n);
  const rec = [];
  const tEnd = performance.now() + minutes * 60000;
  const t0 = performance.now();
  while (performance.now() < tEnd) { H.reset(); const r = await G.chain(steps); rec.push([(performance.now() - t0) / 1000, r.gpu, r.wall]); }
  const win = (a, b) => statsOf(rec.filter((x) => x[0] >= a && x[0] < b).map((x) => x[1]));
  const out = { label, substeps_per_chain: n, minutes, chains: rec.length, gpu_all: statsOf(rec.map((x) => x[1])), first_minute: win(0, 60), last_minute: win(minutes * 60 - 60, minutes * 60 + 1), series_s_gpu_wall: rec.filter((_, i) => i % 10 === 0) };
  H.destroy();
  return out;
}

window.runH = async (labels = ['1k', '10k', '30k'], chains = [800], reps = 10, warm = 3, verifySteps = 800, saveAs = 'probe_h.json') => {
  const o = document.getElementById('o'); o.textContent = 'running H…';
  try { window.__h = await probeH(labels, chains, reps, warm, verifySteps); o.textContent = JSON.stringify(window.__h, null, 1); await fetch(`save/${saveAs}`, { method: 'POST', body: JSON.stringify(window.__h, null, 1) }); }
  catch (e) { window.__h = { error: String(e), stack: e.stack }; o.textContent = String(e); }
  return 'done';
};
window.runSustain = async (label = '30k', n = 800, minutes = 10, saveAs = 'sustain.json') => {
  const o = document.getElementById('o'); o.textContent = 'sustaining…';
  try { window.__sus = await sustain(label, n, minutes); o.textContent = JSON.stringify(window.__sus, null, 1).slice(0, 4000); await fetch(`save/${saveAs}`, { method: 'POST', body: JSON.stringify(window.__sus, null, 1) }); }
  catch (e) { window.__sus = { error: String(e), stack: e.stack }; o.textContent = String(e); }
  return 'done';
};

window.runProbe = async (labels = ['1k', '10k', '30k'], chains = [250, 400, 700, 800], reps = 10, warm = 3, verifySteps = 800, saveAs = 'probe.json') => {
  const o = document.getElementById('o');
  o.textContent = 'running…';
  try {
    window.__probe = await probe(labels, chains, reps, warm, verifySteps); o.textContent = JSON.stringify(window.__probe, null, 1);
    await fetch(`save/${saveAs}`, { method: 'POST', body: JSON.stringify(window.__probe, null, 1) });
  }
  catch (e) { window.__probe = { error: String(e), stack: e.stack }; o.textContent = String(e); }
  return 'done';
};
