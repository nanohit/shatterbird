// Phase 0b / 3 §7 — numeric-contract prototype for E: the E-B-H substep in fixed point.
//
// Formats (i32 unless stated; LSB in brackets):
//   u [2^-28 m], θ [2^-30 rad], v [2^-20 m/s], w [2^-22 rad/s], force [2^-6 N], moment [2^-6 N·m],
//   unit vectors n, p1, p2 [2^-30], arms r [2^-28 m], κ, δ0, w_c [2^-28 m] (u32),
//   k_n, k_t [2^SK N/m], rotational springs [2^SR N·m/rad], γ = k_t/k_n [2^-16], ω = 1 − d [2^-16].
// Products are exact 64-bit (emulated with 16-bit limbs in WGSL); every narrowing is
// round-half-up: floor(x·2^-s + ½); a narrowing that does not fit i32 saturates and raises a flag.
// Accumulation of facet forces per cell is integer addition, hence independent of order.
// Constants are computed once here (f64 from the f32 mesh, Math.round) and shipped to both the
// GPU and the CPU reference (crates/experiments/src/bin/fixed_ref.rs), so both use identical bits.

const FIX = { U: 28, TH: 30, V: 20, W: 22, F: 6, Q: 30, R: 28, G: 16 };

function fixConsts(m, dt, FX = FIX) {
  if (FX.U - FX.V !== FX.TH - FX.W) throw new Error('U-V must equal TH-W (one c_h)');
  const { nc, nf, cells, fdata } = m;
  let knMax = 0, krMax = 0, gMax = 0;
  for (let f = 0; f < nf; f++) {
    const b = 20 * f;
    knMax = Math.max(knMax, Math.abs(fdata[b + 15]), Math.abs(fdata[b + 16]));
    krMax = Math.max(krMax, Math.abs(fdata[b + 17]), Math.abs(fdata[b + 18]), Math.abs(fdata[b + 19]));
    gMax = Math.max(gMax, fdata[b + 16] / fdata[b + 15]);
  }
  if (gMax > 1) throw new Error('gamma > 1 breaks the u32 bound of δ_eq²');
  const SK = Math.max(0, Math.ceil(Math.log2(knMax / 2 ** 30)));
  const SR = Math.max(0, Math.ceil(Math.log2(krMax / 2 ** 30)));
  if (SK >= 22 || SR >= 24) throw new Error('stiffness out of range');
  let cmMax = 0, ciMax = 0;
  for (let i = 0; i < nc; i++) {
    const c = 25 * i;
    cmMax = Math.max(cmMax, dt / cells[c + 3] * 2 ** (FX.V - FX.F));
    for (let k = 0; k < 9; k++) ciMax = Math.max(ciMax, Math.abs(dt * cells[c + 13 + k]) * 2 ** (FX.W - FX.F));
  }
  const SM = Math.floor(Math.log2(2 ** 30 / cmMax)), SI = Math.floor(Math.log2(2 ** 30 / ciMax));
  const SH = Math.floor(Math.log2(2 ** 30 / (dt * 2 ** (FX.U - FX.V))));
  const wc = Math.round(6.6666667e-5 * 2 ** FIX.U);
  const fc = new Int32Array(24 * nf);
  for (let f = 0; f < nf; f++) {
    const b = 20 * f, o = 24 * f;
    for (let k = 0; k < 9; k++) fc[o + k] = Math.round(fdata[b + k] * 2 ** FIX.Q);
    for (let k = 9; k < 15; k++) fc[o + k] = Math.round(fdata[b + k] * 2 ** FIX.R);
    fc[o + 15] = Math.round(fdata[b + 15] / 2 ** SK); fc[o + 16] = Math.round(fdata[b + 16] / 2 ** SK);
    for (let k = 17; k < 20; k++) fc[o + k] = Math.round(fdata[b + k] / 2 ** SR);
    const ell = Math.max(Math.hypot(fdata[b + 9] - fdata[b + 12], fdata[b + 10] - fdata[b + 13], fdata[b + 11] - fdata[b + 14]), 1e-6);
    fc[o + 20] = Math.round(1e-4 * ell * 2 ** FIX.U);
    fc[o + 21] = Math.round(fdata[b + 16] / fdata[b + 15] * 2 ** FIX.G);
  }
  const cc = new Int32Array(16 * nc);
  for (let i = 0; i < nc; i++) {
    const c = 25 * i, o = 16 * i;
    cc[o] = Math.round(dt / cells[c + 3] * 2 ** (FX.V - FX.F + SM));
    for (let k = 0; k < 9; k++) cc[o + 1 + k] = Math.round(dt * cells[c + 13 + k] * 2 ** (FX.W - FX.F + SI));
  }
  const ch = Math.round(dt * 2 ** (FX.U - FX.V + SH));
  const glob = new Int32Array([SK, SR, SM, SI, SH, wc, ch, nc, nf, FX.U, FX.V, FX.W, FX.TH, 0, 0, 0]);
  const init = new Int32Array(12 * nc);
  for (let i = 0; i < nc; i++) for (let k = 0; k < 3; k++) init[12 * i + 6 + k] = Math.round(cells[25 * i + 22 + k] * 2 ** FX.V);
  return { fc, cc, glob, init, scales: { SK, SR, SM, SI, SH, wc, ch, knMax, krMax, gMax, cmMax, ciMax } };
}

const FIX_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> fc: array<i32>;
@group(0) @binding(1) var<storage, read> fidx: array<u32>;
@group(0) @binding(2) var<storage, read> cc: array<i32>;
@group(0) @binding(3) var<storage, read> csr: array<u32>;
@group(0) @binding(4) var<storage, read> sin: array<i32>;
@group(0) @binding(5) var<storage, read_write> sout: array<i32>;
@group(0) @binding(6) var<storage, read> kin: array<u32>;
@group(0) @binding(7) var<storage, read_write> kout: array<u32>;
@group(0) @binding(8) var<storage, read_write> flag: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> dbg: array<i32>;
struct Uni { p: vec4u, g: array<vec4i, 4> }
@group(0) @binding(10) var<uniform> U: Uni;
fn gl(k: u32) -> i32 { return U.g[k / 4u][k % 4u]; }

// ---- 64-bit two's complement as vec2u(lo, hi)
fn umul(a: u32, b: u32) -> vec2u {
  let a0 = a & 0xffffu; let a1 = a >> 16u; let b0 = b & 0xffffu; let b1 = b >> 16u;
  let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
  let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
  return vec2u((p00 & 0xffffu) | (mid << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}
fn neg64(x: vec2u) -> vec2u { let lo = ~x.x + 1u; return vec2u(lo, ~x.y + select(0u, 1u, lo == 0u)); }
fn add64(x: vec2u, y: vec2u) -> vec2u { let lo = x.x + y.x; return vec2u(lo, x.y + y.y + select(0u, 1u, lo < x.x)); }
fn smul(a: i32, b: i32) -> vec2u {
  let p = umul(u32(abs(a)), u32(abs(b)));
  return select(p, neg64(p), (a < 0) != (b < 0));
}
fn i64of(a: i32) -> vec2u { return vec2u(u32(a), select(0u, 0xffffffffu, a < 0)); }
// floor(x·2^-s + ½), s in 1..62, saturated to i32
fn rs(x0: vec2u, s: u32) -> i32 {
  var h = vec2u(0u, 0u);
  if (s - 1u < 32u) { h = vec2u(1u << (s - 1u), 0u); } else { h = vec2u(0u, 1u << (s - 33u)); }
  let x = add64(x0, h);
  var lo: u32; var hi: i32;
  if (s < 32u) { lo = (x.x >> s) | (x.y << (32u - s)); hi = i32(x.y) >> s; }
  else { lo = u32(i32(x.y) >> (s - 32u)); hi = i32(x.y) >> 31u; }
  if (hi != (i32(lo) >> 31u)) {
    atomicOr(&flag[0], 1u);
    return select(2147483647, -2147483647, hi < 0);
  }
  return i32(lo);
}
fn dot3(a: vec3i, b: vec3i) -> vec2u { return add64(add64(smul(a.x, b.x), smul(a.y, b.y)), smul(a.z, b.z)); }
fn crs(a: vec3i, b: vec3i, s: u32) -> vec3i {
  return vec3i(rs(add64(smul(a.y, b.z), neg64(smul(a.z, b.y))), s), rs(add64(smul(a.z, b.x), neg64(smul(a.x, b.z))), s), rs(add64(smul(a.x, b.y), neg64(smul(a.y, b.x))), s));
}
fn comb(c: vec3i, x: vec3i, y: vec3i, z: vec3i) -> vec3i {
  // c.x·x + c.y·y + c.z·z, per component, narrowed by 2^30
  return vec3i(rs(add64(add64(smul(c.x, x.x), smul(c.y, y.x)), smul(c.z, z.x)), 30u),
               rs(add64(add64(smul(c.x, x.y), smul(c.y, y.y)), smul(c.z, z.y)), 30u),
               rs(add64(add64(smul(c.x, x.z), smul(c.y, y.z)), smul(c.z, z.z)), 30u));
}
fn isqrt(q0: u32) -> u32 {
  var q = q0; var r = 0u; var bit = 1u << 30u;
  while (bit > q) { bit >>= 2u; }
  while (bit != 0u) { if (q >= r + bit) { q -= r + bit; r = (r >> 1u) + bit; } else { r >>= 1u; } bit >>= 2u; }
  return r;
}
fn fv(base: u32) -> vec3i { return vec3i(fc[base], fc[base + 1u], fc[base + 2u]); }
fn ld(i: u32, o: u32) -> vec3i { return vec3i(sin[12u*i+o], sin[12u*i+o+1u], sin[12u*i+o+2u]); }
struct PairX { f: vec3i, ta: vec3i, tb: vec3i, kap: u32 }
fn pairx(fi: u32, ua: vec3i, wa: vec3i, ub: vec3i, wb: vec3i, kold: u32) -> PairX {
  let b = 24u * fi;
  let n = fv(b); let p1 = fv(b + 3u); let p2 = fv(b + 6u); let ra = fv(b + 9u); let rb = fv(b + 12u);
  let kn = fc[b + 15u]; let kt = fc[b + 16u]; let kr1 = fc[b + 17u]; let kr2 = fc[b + 18u]; let ktor = fc[b + 19u];
  let d0 = u32(fc[b + 20u]); let gam = u32(fc[b + 21u]);
  let SK = u32(gl(0u)); let SR = u32(gl(1u)); let wc = u32(gl(5u));
  let jump = (ub + crs(wb, rb, 30u)) - (ua + crs(wa, ra, 30u));
  let dw = wb - wa;
  let dn = rs(dot3(n, jump), 30u); let t1 = rs(dot3(p1, jump), 30u); let t2 = rs(dot3(p2, jump), 30u);
  let dp = max(dn, 0);
  let c0 = u32(min(dp, 32767)); let c1 = u32(min(abs(t1), 32767)); let c2 = u32(min(abs(t2), 32767));
  let tq = umul(c1 * c1 + c2 * c2, gam);
  let q = c0 * c0 + u32(rs(tq, 16u));
  let kap = max(kold, isqrt(q));
  var om = 65536;
  if (kap >= wc) { om = 0; } else if (kap > d0) {
    let num = d0 * (wc - kap); let den = kap * (wc - d0);
    var r = num; var qq = 0u;
    for (var k = 0u; k < 16u; k++) { r = r << 1u; qq = qq << 1u; if (r >= den) { r -= den; qq |= 1u; } }
    om = i32(qq);
  }
  let sf = 22u - SK;
  let fnt = rs(smul(kn, dp), sf);
  let fnn = rs(smul(om, fnt), 16u) + rs(smul(kn, min(dn, 0)), sf);
  let ft1 = rs(smul(om, rs(smul(kt, t1), sf)), 16u);
  let ft2 = rs(smul(om, rs(smul(kt, t2), sf)), 16u);
  let f = comb(vec3i(fnn, ft1, ft2), n, p1, p2);
  let sm = 24u - SR;
  let m1 = rs(smul(om, rs(smul(kr1, rs(dot3(p1, dw), 30u)), sm)), 16u);
  let m2 = rs(smul(om, rs(smul(kr2, rs(dot3(p2, dw), 30u)), sm)), 16u);
  let m3 = rs(smul(om, rs(smul(ktor, rs(dot3(n, dw), 30u)), sm)), 16u);
  let mv = comb(vec3i(m1, m2, m3), p1, p2, n);
  var o: PairX;
  o.f = f; o.ta = crs(ra, f, 28u) + mv; o.tb = -crs(rb, f, 28u) - mv; o.kap = kap;
  return o;
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; let nc = u32(gl(7u)); if (i >= nc) { return; }
  var F = vec3i(0); var T = vec3i(0);
  for (var q = csr[i]; q < csr[i + 1u]; q++) {
    let e = csr[nc + 1u + q]; let fi = e & 0x7fffffffu; let side_b = (e >> 31u) == 1u;
    let a = fidx[2u * fi]; let bb = fidx[2u * fi + 1u];
    var ub = vec3i(0); var wb = vec3i(0);
    if (bb != 0xffffffffu) { ub = ld(bb, 0u); wb = ld(bb, 3u); }
    let o = pairx(fi, ld(a, 0u), ld(a, 3u), ub, wb, kin[fi]);
    if (side_b) { F -= o.f; T += o.tb; } else { F += o.f; T += o.ta; kout[fi] = o.kap; }
    if (U.p.x == 1u) {
      let k = 7u * (2u * fi + select(0u, 1u, side_b));
      let ff = select(o.f, -o.f, side_b); let tt = select(o.ta, o.tb, side_b);
      dbg[k] = ff.x; dbg[k+1u] = ff.y; dbg[k+2u] = ff.z; dbg[k+3u] = tt.x; dbg[k+4u] = tt.y; dbg[k+5u] = tt.z; dbg[k+6u] = i32(o.kap);
    }
  }
  let c = 16u * i; let SM = u32(gl(2u)); let SI = u32(gl(3u)); let SH = u32(gl(4u)); let ch = gl(6u);
  let cm = cc[c];
  let ci0 = vec3i(cc[c+1u], cc[c+2u], cc[c+3u]); let ci1 = vec3i(cc[c+4u], cc[c+5u], cc[c+6u]); let ci2 = vec3i(cc[c+7u], cc[c+8u], cc[c+9u]);
  let v = ld(i, 6u) + vec3i(rs(smul(F.x, cm), SM), rs(smul(F.y, cm), SM), rs(smul(F.z, cm), SM));
  let w = ld(i, 9u) + vec3i(rs(dot3(ci0, T), SI), rs(dot3(ci1, T), SI), rs(dot3(ci2, T), SI));
  let u = ld(i, 0u) + vec3i(rs(smul(v.x, ch), SH), rs(smul(v.y, ch), SH), rs(smul(v.z, ch), SH));
  let th = ld(i, 3u) + vec3i(rs(smul(w.x, ch), SH), rs(smul(w.y, ch), SH), rs(smul(w.z, ch), SH));
  let s = 12u * i;
  sout[s] = u.x; sout[s+1u] = u.y; sout[s+2u] = u.z; sout[s+3u] = th.x; sout[s+4u] = th.y; sout[s+5u] = th.z;
  sout[s+6u] = v.x; sout[s+7u] = v.y; sout[s+8u] = v.z; sout[s+9u] = w.x; sout[s+10u] = w.y; sout[s+11u] = w.z;
}`;

async function fixedRun(labels, steps, reps, warm, saveTag, FX = FIX, chains = [250, 800]) {
  const G = await makeDevice();
  const mod = G.device.createShaderModule({ code: FIX_WGSL });
  const info = await mod.getCompilationInfo();
  const errs = info.messages.filter((x) => x.type === 'error');
  if (errs.length) throw new Error('fixed: ' + errs.map((e) => e.lineNum + ':' + e.message).join('; '));
  const pipe = await G.device.createComputePipelineAsync({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
  const index = await (await fetch('index.json')).json();
  const out = { adapter: G.adapter.info ? { vendor: G.adapter.info.vendor, architecture: G.adapter.info.architecture } : null, ua: navigator.userAgent, steps, meshes: [] };
  const S = GPUBufferUsage.STORAGE, C = GPUBufferUsage.COPY_SRC, D = GPUBufferUsage.COPY_DST;
  const rbU = async (buf, n) => { const r = G.device.createBuffer({ size: 4 * n, usage: GPUBufferUsage.MAP_READ | D }); const e = G.device.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, r, 0, 4 * n); G.device.queue.submit([e.finish()]); await r.mapAsync(GPUMapMode.READ); const v = new Int32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return v; };
  for (const label of labels) {
    const meta = index.meshes.find((x) => x.label === label);
    const m = parseMesh(await (await fetch(meta.file)).arrayBuffer());
    const k = fixConsts(m, index.dt_contract_s, FX);
    const csr = new Uint32Array(m.csrPtr.length + m.csrIdx.length); csr.set(m.csrPtr, 0); csr.set(m.csrIdx, m.csrPtr.length);
    const bFc = G.mk(k.fc), bFi = G.mk(m.fidx), bCc = G.mk(k.cc), bCsr = G.mk(csr);
    const uniOf = (dbgFlag) => { const b = G.device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); const a = new Int32Array(20); a[0] = dbgFlag; a.set(k.glob, 4); G.device.queue.writeBuffer(b, 0, a); return b; };
    const s0 = G.zeros(12 * m.nc), s1 = G.zeros(12 * m.nc), k0 = G.zeros(m.nf), k1 = G.zeros(m.nf), flag = G.zeros(4), dbg = G.zeros(14 * m.nf);
    const reset = () => { for (const b of [s0, s1]) G.device.queue.writeBuffer(b, 0, k.init); G.device.queue.writeBuffer(k0, 0, new Uint32Array(m.nf)); G.device.queue.writeBuffer(k1, 0, new Uint32Array(m.nf)); G.device.queue.writeBuffer(flag, 0, new Uint32Array(4)); };
    const u0 = uniOf(0), u1 = uniOf(1);
    G.device.pushErrorScope('validation');
    const g01 = G.bg(pipe, [bFc, bFi, bCc, bCsr, s0, s1, k0, k1, flag, dbg, u0]), g10 = G.bg(pipe, [bFc, bFi, bCc, bCsr, s1, s0, k1, k0, flag, dbg, u0]);
    const gD = G.bg(pipe, [bFc, bFi, bCc, bCsr, s0, s1, k0, k1, flag, dbg, u1]);
    const verr = await G.device.popErrorScope();
    if (verr) throw new Error('bind groups: ' + verr.message);
    const wcnt = Math.ceil(m.nc / 64);
    const chainOf = (n) => { const s = []; for (let q = 0; q < n; q++) s.push([pipe, q % 2 ? g10 : g01, wcnt]); return s; };
    const res = { label, cells: m.nc, facets: m.nf, formats: FX, scales: k.scales, timings: [] };
    for (const n of chains) {
      const ch = chainOf(n);
      for (let q = 0; q < warm; q++) { reset(); await G.chain(ch); }
      const gg = [];
      for (let q = 0; q < reps; q++) { reset(); gg.push((await G.chain(ch)).gpu); }
      res.timings.push({ substeps: n, gpu_ms: statsOf(gg) });
    }
    // determinism run: `steps` substeps, then state, κ and overflow flag
    if (!(res.timings[0].gpu_ms.median > 0)) throw new Error('zero GPU time: kernel did not run');
    reset(); await G.chain(chainOf(steps));
    const st = await rbU(steps % 2 ? s1 : s0, 12 * m.nc), kp = await rbU(steps % 2 ? k1 : k0, m.nf), fl = await rbU(flag, 1);
    // pair check: one debug substep after 100
    reset(); await G.chain(chainOf(100)); await G.chain([[pipe, gD, wcnt]]);
    const d = await rbU(dbg, 14 * m.nf);
    let opp = 0, kapEq = 0, inter = 0;
    for (let f = 0; f < m.nf; f++) {
      if (m.fidx[2 * f + 1] === 0xffffffff) continue;
      inter++;
      if (d[14 * f] === -d[14 * f + 7] && d[14 * f + 1] === -d[14 * f + 8] && d[14 * f + 2] === -d[14 * f + 9]) opp++;
      if (d[14 * f + 6] === d[14 * f + 13]) kapEq++;
    }
    let h = 2166136261 >>> 0; for (const x of st) { h = Math.imul(h ^ (x >>> 0), 16777619) >>> 0; } for (const x of kp) { h = Math.imul(h ^ (x >>> 0), 16777619) >>> 0; }
    res.run = { steps, overflow_flag: fl[0], fnv1a32: h.toString(16), pair_force_exact_opposite: `${opp}/${inter}`, pair_kappa_equal: `${kapEq}/${inter}` };
    await fetch(`save/fixed_consts_${label}_${saveTag}.bin`, { method: 'POST', body: new Blob([k.glob, k.fc, k.cc, k.init]) });
    await fetch(`save/fixed_state_${label}_${saveTag}.bin`, { method: 'POST', body: new Blob([st, kp]) });
    out.meshes.push(res);
    for (const b of [bFc, bFi, bCc, bCsr, u0, u1, s0, s1, k0, k1, flag, dbg]) b.destroy();
  }
  return out;
}

window.runFixed = async (labels = ['1k', '10k', '30k'], steps = 800, reps = 10, warm = 3, saveTag = 's800', FX = FIX, chains = [250, 800]) => {
  const o = document.getElementById('o'); o.textContent = 'fixed…';
  try { window.__fx = await fixedRun(labels, steps, reps, warm, saveTag, FX, chains); o.textContent = JSON.stringify(window.__fx, null, 1); await fetch(`save/fixed_${saveTag}.json`, { method: 'POST', body: JSON.stringify(window.__fx, null, 1) }); }
  catch (e) { window.__fx = { error: String(e), stack: e.stack }; o.textContent = String(e); }
  return 'done';
};
