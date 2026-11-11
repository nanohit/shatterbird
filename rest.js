// Shatterbird phase 1d: contact solver with persistent impulses for equal spheres.
// Integer specification (JS reference). Positions in 1/256 of a length unit, diameter 960 units,
// grid cell 1024 units. Velocities and impulses share one unit: 1/F of a position step per substep
// (F = prm.f), so impulses add to velocities without rounding. Each body keeps up to K contacts;
// a contact carries an impulse vector P, antisymmetric between the two bodies. Per substep:
//   rho = v - g + sum P                               (stored impulses act as forces)
//   P   = cone(P - trunc((rho_i - rho_j - gap) / (c_i + c_j)))   (Jacobi, mass splitting, stick)
//   v   = v - g + sum P;  a supported body with |v| <= eps is set to rest (momentum to a ledger)
//   x  += trunc(v / F) + push-out of penetration above the slop (positions only, whole 1/cq units)
// Every reduction is an integer sum over the contacts of one body: slot order cannot matter.
'use strict';
const REST = (() => {
  const DIAM = 960, R = 480, KS = 8, K = 16, MARGIN = 16, VMAX = 25600, WALL = 0xFFFFFFF0 >>> 0, NONE = 0xFFFFFFFF >>> 0;
  function rng(seed) { let s = seed >>> 0; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s; }; }
  function isqrt(v) { let s = Math.floor(Math.sqrt(v)); while (s * s > v) s--; while ((s + 1) * (s + 1) <= v) s++; return s; }
  function make(N, gx, gy, gz) {
    return { N, gx, gy, gz, pos: new Int32Array(4 * N), vel: new Int32Array(4 * N), sid: new Uint32Array(K * N).fill(NONE), sP: new Int32Array(3 * K * N), sn: new Uint8Array(N),
      nid: new Uint32Array(K * N), Pl: [], rl: [], cn: new Uint8Array(N), mark: new Uint8Array(gx * gy * gz), seed: null, act: new Uint32Array(N), evals: 0,
      cnt: new Uint32Array(gx * gy * gz), cslot: new Uint32Array(gx * gy * gz * KS), corr: new Int32Array(4 * N), ledger: new Float64Array(4), chg: new Uint8Array(N), setc: new Uint8Array(N), t: 0 };
  }
  // nx*ny*nz lattice, spacing (sx, sy, sx) units, jitter +-jit units, lowest layer resting on the floor.
  // vr: random initial velocity, position units per substep (scaled by f).
  function lattice(nx, ny, nz, sx, sy, jit, seed, pad = 2, head = 4, vr = 0, f = 32) {
    const r = rng(seed), N = nx * ny * nz, gx = Math.ceil(nx * sx / 1024) + 2 * pad, gz = Math.ceil(nz * sx / 1024) + 2 * pad, gy = Math.ceil(ny * sy / 1024) + head;
    const S = make(N, gx, gy, gz); let i = 0;
    const j = () => jit ? (r() % (2 * jit + 1)) - jit : 0, jv = () => vr ? ((r() % (2 * vr + 1)) - vr) * f : 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++, i++) {
      S.pos[4 * i] = (pad * 1024 + x * sx + (sx >> 1) + j()) << 8; S.pos[4 * i + 1] = (y * sy + R + (y ? j() : 0)) << 8; S.pos[4 * i + 2] = (pad * 1024 + z * sx + (sx >> 1) + j()) << 8;
      S.vel[4 * i] = jv(); S.vel[4 * i + 1] = jv(); S.vel[4 * i + 2] = jv();
    }
    return S;
  }
  const tr = Math.trunc;
  // Geometry of slot `id` seen from body i: e (points from the partner to i), len, pen (units).
  const geo = { ex: 0, ey: 0, ez: 0, len: 1, pen: 0 };
  function geom(S, i, id) {
    const p = S.pos, px = p[4 * i] >> 8, py = p[4 * i + 1] >> 8, pz = p[4 * i + 2] >> 8;
    if (id >= WALL) {
      const a = id - WALL, ax = a >> 1, hi = a & 1, c = ax === 0 ? px : ax === 1 ? py : pz, H = (ax === 0 ? S.gx : ax === 1 ? S.gy : S.gz) * 1024;
      geo.ex = geo.ey = geo.ez = 0; const s = hi ? -1 : 1; if (ax === 0) geo.ex = s; else if (ax === 1) geo.ey = s; else geo.ez = s;
      geo.len = 1; geo.pen = hi ? c - (H - R) : R - c; return;
    }
    const ex = px - (p[4 * id] >> 8), ey = py - (p[4 * id + 1] >> 8), ez = pz - (p[4 * id + 2] >> 8), d2 = ex * ex + ey * ey + ez * ez;
    if (d2 === 0) { geo.ex = i < id ? 1 : -1; geo.ey = geo.ez = 0; geo.len = 1; geo.pen = DIAM; return; }
    geo.ex = ex; geo.ey = ey; geo.ez = ez; geo.len = isqrt(d2); geo.pen = DIAM - geo.len;
  }
  // Contacts of body i from the grid, impulses carried over by partner id, rho from stored impulses.
  function match(S, i, prm) {
    const { pos, vel, gx, gy, gz, cnt, cslot, sid, sP, sn, nid } = S, nP = S.Pl[0], vs = S.rl[0];
    const px = pos[4 * i] >> 8, py = pos[4 * i + 1] >> 8, pz = pos[4 * i + 2] >> 8, cx = px >> 10, cy = py >> 10, cz = pz >> 10, MG = prm.margin, lim = (DIAM + MG) * (DIAM + MG);
    let n = 0, over = 0; const b = i * K;
    for (let dz = -1; dz <= 1; dz++) { const z = cz + dz; if (z < 0 || z >= gz) continue;
      for (let dy = -1; dy <= 1; dy++) { const y = cy + dy; if (y < 0 || y >= gy) continue;
        for (let dx = -1; dx <= 1; dx++) { const x = cx + dx; if (x < 0 || x >= gx) continue;
          const c = x + gx * (y + gy * z), m = Math.min(cnt[c], KS);
          for (let s = 0; s < m; s++) {
            const j = cslot[c * KS + s]; if (j === i) continue;
            const ex = px - (pos[4 * j] >> 8), ey = py - (pos[4 * j + 1] >> 8), ez = pz - (pos[4 * j + 2] >> 8);
            if (ex * ex + ey * ey + ez * ez >= lim) continue;
            if (n < K) nid[b + n++] = j; else over++;
          } } } }
    const wl = (c, H, a) => { if (R - c > -MG) { if (n < K) nid[b + n++] = WALL + 2 * a; else over++; } if (c - (H - R) > -MG) { if (n < K) nid[b + n++] = WALL + 2 * a + 1; else over++; } };
    wl(px, gx * 1024, 0); wl(py, gy * 1024, 1); wl(pz, gz * 1024, 2);
    const on = sn[i]; let same = n === on, ax = 0, ay = 0, az = 0;
    for (let s = 0; s < n; s++) {
      const id = nid[b + s]; let f = -1;
      for (let q = 0; q < on; q++) if (sid[b + q] === id) { f = q; break; }
      const o = 3 * (b + s);
      if (f < 0) { same = false; nP[o] = nP[o + 1] = nP[o + 2] = 0; } else { const q = 3 * (b + f); nP[o] = sP[q]; nP[o + 1] = sP[q + 1]; nP[o + 2] = sP[q + 2]; }
      ax += nP[o]; ay += nP[o + 1]; az += nP[o + 2];
    }
    S.cn[i] = n; S.setc[i] = same ? 0 : 1;
    vs[4 * i] = vel[4 * i] + ax; vs[4 * i + 1] = vel[4 * i + 1] - prm.grav * prm.f + ay; vs[4 * i + 2] = vel[4 * i + 2] + az; vs[4 * i + 3] = (vel[4 * i] | vel[4 * i + 1] | vel[4 * i + 2]) ? 1 : 0;
    return over;
  }
  // One Jacobi iteration for body i: new impulses of its contacts and its new rho. corr: push-out.
  function solve(S, i, prm, src, dst, vin, vout, corr) {
    const { vel, nid, cn } = S, b = i * K, n = cn[i], F = prm.f, PM = 16384 * F, TQ = 4 * F, cl = q => q > PM ? PM : q < -PM ? -PM : q; let ax = 0, ay = 0, az = 0, sup = 0;
    const vx = vin[4 * i], vy = vin[4 * i + 1], vz = vin[4 * i + 2];
    for (let s = 0; s < n; s++) {
      const id = nid[b + s], o = 3 * (b + s); geom(S, i, id);
      const { ex, ey, ez, len, pen } = geo;
      let rx = vx, ry = vy, rz = vz, den = n;
      if (id < WALL) { rx -= vin[4 * id]; ry -= vin[4 * id + 1]; rz -= vin[4 * id + 2]; den += cn[id]; }
      if (pen < -prm.g0) { if (corr && S.dbg && (src[o] | src[o + 1] | src[o + 2])) S.dbg.release++; const g = (pen + prm.g0) * 256 * F; rx -= tr(g * ex / len); ry -= tr(g * ey / len); rz -= tr(g * ez / len); }   // speculative: may close the gap, not more
      else if (prm.bg && pen > prm.slop) { const g = tr(Math.min(pen - prm.slop, 64) * 256 * F / prm.bg); rx -= tr(g * ex / len); ry -= tr(g * ey / len); rz -= tr(g * ez / len); }   // slow velocity-level push-out
      else if (prm.cq && corr && pen > prm.slop && (pen > prm.hi || vin[4 * i + 3] || (id < WALL && vin[4 * id + 3]))) { const q = Math.min(pen - prm.slop, 64) * prm.cq, u = 256 / prm.cq; corr[4 * i] += tr(q * ex / (len * den)) * u; corr[4 * i + 1] += tr(q * ey / (len * den)) * u; corr[4 * i + 2] += tr(q * ez / (len * den)) * u; }
      const th = prm.th, live = !th || rx >= th || rx <= -th || ry >= th || ry <= -th || rz >= th || rz <= -th;   // impulse quantum: smaller relative velocities are left to the rest quantum
      let Px = src[o], Py = src[o + 1], Pz = src[o + 2];
      if (live) { Px -= cl(tr(rx / den)); Py -= cl(tr(ry / den)); Pz -= cl(tr(rz / den)); }
      Px = cl(Px); Py = cl(Py); Pz = cl(Pz);   // keeps every product below 2^31 for the i32 implementation
      const pn = tr((Px * ex + Py * ey + Pz * ez) / len);
      if (pn <= 0) { if (corr && S.dbg && (src[o] | src[o + 1] | src[o + 2])) S.dbg.open++; Px = Py = Pz = 0; }
      else {
        const nx = tr(pn * ex / len), ny = tr(pn * ey / len), nz = tr(pn * ez / len), tx = tr((Px - nx) / TQ), ty = tr((Py - ny) / TQ), tz = tr((Pz - nz) / TQ);
        const t2 = tx * tx + ty * ty + tz * tz, lm = tr(prm.mu8 * pn / (8 * TQ));
        if (t2 > lm * lm) { if (corr && S.dbg) S.dbg.slip++; const tl = isqrt(t2), lk = tr(prm.muk8 * pn / (8 * TQ)); Px = nx + tr(tx * lk / tl) * TQ; Py = ny + tr(ty * lk / tl) * TQ; Pz = nz + tr(tz * lk / tl) * TQ; }   // a slipping contact drops to kinetic friction
      }
      Px = cl(Px); Py = cl(Py); Pz = cl(Pz);
      if (Px | Py | Pz) sup = 1;
      dst[o] = Px; dst[o + 1] = Py; dst[o + 2] = Pz;
      ax += Px; ay += Py; az += Pz;
    }
    vout[4 * i] = vel[4 * i] + ax; vout[4 * i + 1] = vel[4 * i + 1] - prm.grav * F + ay; vout[4 * i + 2] = vel[4 * i + 2] + az; vout[4 * i + 3] = sup;
  }
  function finish(S, i, prm) {
    const { pos, vel, sid, sP, sn, nid, cn } = S, vs = S.rl[prm.iters], nP = S.Pl[prm.iters], cP = S.Pl[0], b = i * K, n = cn[i], F = prm.f, VM = VMAX * F, cl = (v, a, c) => v < a ? a : v > c ? c : v;
    let vx = cl(vs[4 * i], -VM, VM), vy = cl(vs[4 * i + 1], -VM, VM), vz = cl(vs[4 * i + 2], -VM, VM);
    const e = prm.eps;
    if (e && vs[4 * i + 3] && vx <= e && vx >= -e && vy <= e && vy >= -e && vz <= e && vz >= -e) { S.ledger[0] += vx; S.ledger[1] += vy; S.ledger[2] += vz; S.ledger[3] += Math.abs(vx) + Math.abs(vy) + Math.abs(vz); vx = vy = vz = 0; }
    const mx = tr(vx / F) + S.corr[4 * i], my = tr(vy / F) + S.corr[4 * i + 1], mz = tr(vz / F) + S.corr[4 * i + 2];
    let ch = S.setc[i] || mx !== 0 || my !== 0 || mz !== 0 || vx !== vel[4 * i] || vy !== vel[4 * i + 1] || vz !== vel[4 * i + 2];
    for (let s = 0; s < n; s++) { const o = 3 * (b + s); if (nP[o] !== cP[o] || nP[o + 1] !== cP[o + 1] || nP[o + 2] !== cP[o + 2]) ch = true; sid[b + s] = nid[b + s]; sP[o] = nP[o]; sP[o + 1] = nP[o + 1]; sP[o + 2] = nP[o + 2]; }
    sn[i] = n; vel[4 * i] = vx; vel[4 * i + 1] = vy; vel[4 * i + 2] = vz;
    const lo = R << 8;
    pos[4 * i] = cl(pos[4 * i] + mx, lo, ((S.gx * 1024 - R) << 8) - 1); pos[4 * i + 1] = cl(pos[4 * i + 1] + my, lo, ((S.gy * 1024 - R) << 8) - 1); pos[4 * i + 2] = cl(pos[4 * i + 2] + mz, lo, ((S.gz * 1024 - R) << 8) - 1);
    S.chg[i] = ch ? 1 : 0; return ch ? 1 : 0;
  }
  function buildGrid(S) {
    const { N, gx, gy, pos, cnt, cslot } = S; let over = 0; cnt.fill(0);
    for (let i = 0; i < N; i++) { const c = (pos[4 * i] >> 18) + gx * ((pos[4 * i + 1] >> 18) + gy * (pos[4 * i + 2] >> 18)), s = cnt[c]++; if (s < KS) cslot[c * KS + s] = i; else over++; }
    return over;
  }
  function cellOf(S, i) { const p = S.pos; return (p[4 * i] >> 18) + S.gx * ((p[4 * i + 1] >> 18) + S.gy * (p[4 * i + 2] >> 18)); }
  // One substep over the bodies in `list` (null: all). Records the cells touched by changed bodies.
  function run(S, prm, list) {
    const I = prm.iters, n = list ? list.length : S.N, at = list ? (q => list[q]) : (q => q);
    while (S.Pl.length <= I) { S.Pl.push(new Int32Array(3 * K * S.N)); S.rl.push(new Int32Array(4 * S.N)); }
    let over = 0, changed = 0; const seed = [];
    for (let q = 0; q < n; q++) { const i = at(q); over += match(S, i, prm); S.corr[4 * i] = S.corr[4 * i + 1] = S.corr[4 * i + 2] = 0; }
    for (let it = 0; it < I; it++) for (let q = 0; q < n; q++) solve(S, at(q), prm, S.Pl[it], S.Pl[it + 1], S.rl[it], S.rl[it + 1], it === 0 ? S.corr : null);
    for (let q = 0; q < n; q++) { const i = at(q), c0 = cellOf(S, i); if (finish(S, i, prm)) { changed++; seed.push(c0); const c1 = cellOf(S, i); if (c1 !== c0) seed.push(c1); } }
    S.seed = seed; S.evals += n; S.t++; return { changed, over, evaluated: n };
  }
  // Full step: every body.
  function step(S, prm) { const over = buildGrid(S), r = run(S, prm, null); r.over += over; return r; }
  // Change-driven step: only bodies in cells within iters + 1 cells of a cell touched by a body that
  // changed in the previous step. A body's update reads state at most iters + 1 hops away and one hop
  // never crosses more than one cell, so every skipped body would have reproduced its own state.
  function stepActive(S, prm) {
    const over = buildGrid(S); if (S.seed === null) { const r = run(S, prm, null); r.over += over; return r; }
    const { gx, gy, gz, mark, cnt, cslot, act } = S, r = prm.iters + 1, cells = []; let n = 0;
    for (const c of S.seed) { const cz = Math.floor(c / (gx * gy)), cy = Math.floor((c - cz * gx * gy) / gx), cx = c - gx * (cy + gy * cz);
      for (let z = Math.max(0, cz - r); z <= Math.min(gz - 1, cz + r); z++) for (let y = Math.max(0, cy - r); y <= Math.min(gy - 1, cy + r); y++) for (let x = Math.max(0, cx - r); x <= Math.min(gx - 1, cx + r); x++) { const d = x + gx * (y + gy * z); if (!mark[d]) { mark[d] = 1; cells.push(d); } } }
    for (const d of cells) { mark[d] = 0; const m = Math.min(cnt[d], KS); for (let q = 0; q < m; q++) act[n++] = cslot[d * KS + q]; }
    const res = run(S, prm, act.subarray(0, n)); res.over += over; return res;
  }
  function stats(S) {
    const { N, pos, vel, sid, sP, sn } = S; let vmax = 0, pmax = -1e9, top = 0, nc = 0, cmax = 0, Pmax = 0;
    for (let i = 0; i < N; i++) {
      vmax = Math.max(vmax, Math.abs(vel[4 * i]), Math.abs(vel[4 * i + 1]), Math.abs(vel[4 * i + 2])); top = Math.max(top, pos[4 * i + 1]);
      nc += sn[i]; cmax = Math.max(cmax, sn[i]);
      for (let s = 0; s < sn[i]; s++) { geom(S, i, sid[i * K + s]); if (geo.pen > pmax) pmax = geo.pen; const o = 3 * (i * K + s); Pmax = Math.max(Pmax, Math.abs(sP[o]), Math.abs(sP[o + 1]), Math.abs(sP[o + 2])); }
    }
    return { vmax, pen_max: pmax, top: top / 256, contacts: nc / N, cmax, Pmax };
  }
  function fnv(S) { let h = 0x811c9dc5; const f = a => { for (let i = 0; i < a.length; i++) { h ^= a[i] & 0xffff; h = Math.imul(h, 0x01000193); h ^= a[i] >>> 16; h = Math.imul(h, 0x01000193); } }; f(S.pos); f(S.vel); return (h >>> 0).toString(16); }

  // ---- WebGPU implementation of the same specification (i32 only). ----
  const HEAD = `
struct P { n: u32, gx: u32, gy: u32, gz: u32, grav: i32, f: i32, mu8: i32, slop: i32, cq: i32, eps: i32, g0: i32, margin: i32, hi: i32, th: i32, first: i32, muk8: i32 }
@group(0) @binding(0) var<uniform> p: P;
const DIAM = 960; const RAD = 480; const KS = 8u; const K = 16u; const VMAX = 25600; const WALL = 0xFFFFFFF0u;
fn dims() -> vec3i { return vec3i(i32(p.gx), i32(p.gy), i32(p.gz)); }
fn isqrt(v: u32) -> u32 { var s = u32(sqrt(f32(v))); if (s * s > v) { s -= 1u; } if (s * s > v) { s -= 1u; } if ((s + 1u) * (s + 1u) <= v) { s += 1u; } if ((s + 1u) * (s + 1u) <= v) { s += 1u; } return s; }
struct Geo { e: vec3i, len: i32, pen: i32 }
fn geo_wall(pu: vec3i, a: u32) -> Geo { let ax = a >> 1u; let hi = (a & 1u) == 1u; let c = pu[ax]; let H = dims()[ax] * 1024; var e = vec3i(0); e[ax] = select(1, -1, hi); return Geo(e, 1, select(RAD - c, c - (H - RAD), hi)); }
fn geo_sph(pu: vec3i, qu: vec3i, i: u32, j: u32) -> Geo { let e = pu - qu; let d2 = dot(e, e); if (d2 == 0) { return Geo(vec3i(select(-1, 1, i < j), 0, 0), 1, DIAM); } let l = i32(isqrt(u32(d2))); return Geo(e, l, DIAM - l); }
`;
  const WGSL = {
    clear: HEAD + `
@group(0) @binding(1) var<storage, read_write> cnt: array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x + g.y * 4194240u; if (c >= p.gx * p.gy * p.gz) { return; } atomicStore(&cnt[c], 0u);
}`,
    insert: HEAD + `
@group(0) @binding(1) var<storage, read> pos: array<vec4i>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cslot: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; }
  let q = vec3u(pos[i].xyz >> vec3u(18u)); let c = q.x + p.gx * (q.y + p.gy * q.z);
  let s = atomicAdd(&cnt[c], 1u); if (s < KS) { cslot[c * KS + s] = i; }
}`,
    match: HEAD + `
@group(0) @binding(1) var<storage, read> pos: array<vec4i>;
@group(0) @binding(2) var<storage, read> vel: array<vec4i>;
@group(0) @binding(3) var<storage, read> cnt: array<u32>;
@group(0) @binding(4) var<storage, read> cslot: array<u32>;
@group(0) @binding(5) var<storage, read> slot: array<vec4i>;
@group(0) @binding(6) var<storage, read_write> l0: array<vec4i>;
@group(0) @binding(7) var<storage, read_write> r0: array<vec4i>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; }
  let pu = pos[i].xyz >> vec3u(8u); let cc = pu >> vec3u(10u); let d = dims(); let mg = DIAM + p.margin; let lim = mg * mg;
  var ids: array<u32, 16>; var n = 0u;
  for (var dz = -1; dz <= 1; dz++) { let z = cc.z + dz; if (z < 0 || z >= d.z) { continue; }
    for (var dy = -1; dy <= 1; dy++) { let y = cc.y + dy; if (y < 0 || y >= d.y) { continue; }
      for (var dx = -1; dx <= 1; dx++) { let x = cc.x + dx; if (x < 0 || x >= d.x) { continue; }
        let c = u32(x) + p.gx * (u32(y) + p.gy * u32(z)); let m = min(cnt[c], KS);
        for (var s = 0u; s < m; s++) {
          let j = cslot[c * KS + s]; if (j == i) { continue; }
          let e = pu - (pos[j].xyz >> vec3u(8u)); if (dot(e, e) >= lim) { continue; }
          if (n < K) { ids[n] = j; n++; }
        } } } }
  for (var a = 0u; a < 3u; a++) { let c = pu[a]; let H = d[a] * 1024;
    if (RAD - c > -p.margin) { if (n < K) { ids[n] = WALL + 2u * a; n++; } }
    if (c - (H - RAD) > -p.margin) { if (n < K) { ids[n] = WALL + 2u * a + 1u; n++; } } }
  let v = vel[i]; let on = u32(v.w); var acc = vec3i(0);
  for (var s = 0u; s < n; s++) { let id = ids[s]; var q = vec3i(0);
    for (var k = 0u; k < on; k++) { let o = slot[i * K + k]; if (u32(o.w) == id) { q = o.xyz; break; } }
    l0[i * K + s] = vec4i(q, i32(id)); acc += q; }
  r0[i] = vec4i(v.xyz + acc - vec3i(0, p.grav * p.f, 0), i32(n << 8u) | select(0, 1, any(v.xyz != vec3i(0))));
}`,
    solve: HEAD + `
@group(0) @binding(1) var<storage, read> pos: array<vec4i>;
@group(0) @binding(2) var<storage, read> vel: array<vec4i>;
@group(0) @binding(3) var<storage, read> lk: array<vec4i>;
@group(0) @binding(4) var<storage, read> rk: array<vec4i>;
@group(0) @binding(5) var<storage, read_write> ln: array<vec4i>;
@group(0) @binding(6) var<storage, read_write> rn: array<vec4i>;
@group(0) @binding(7) var<storage, read_write> corr: array<vec4i>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; }
  let me = rk[i]; let n = u32(me.w) >> 8u; let pu = pos[i].xyz >> vec3u(8u); let F = p.f; let PM = 16384 * F; let TQ = 4 * F;
  var acc = vec3i(0); var sup = 0; var cr = vec3i(0);
  for (var s = 0u; s < n; s++) {
    let L = lk[i * K + s]; let id = u32(L.w); var ge: Geo; var r = me.xyz; var den = i32(n); var aw = (me.w & 1) != 0;
    if (id >= WALL) { ge = geo_wall(pu, id - WALL); } else { let q = rk[id]; ge = geo_sph(pu, pos[id].xyz >> vec3u(8u), i, id); r -= q.xyz; den += q.w >> 8u; aw = aw || ((q.w & 1) != 0); }
    if (ge.pen < -p.g0) { let gg = (ge.pen + p.g0) * 256 * F; r -= (gg * ge.e) / ge.len; }
    else if (p.first != 0 && p.cq != 0 && ge.pen > p.slop && (ge.pen > p.hi || aw)) { let q = min(ge.pen - p.slop, 64) * p.cq; cr += ((q * ge.e) / (ge.len * den)) * (256 / p.cq); }
    var Q = L.xyz;
    if (p.th == 0 || any(abs(r) >= vec3i(p.th))) { Q -= clamp(r / den, vec3i(-PM), vec3i(PM)); }
    Q = clamp(Q, vec3i(-PM), vec3i(PM));
    let pn = dot(Q, ge.e) / ge.len;
    if (pn <= 0) { Q = vec3i(0); }
    else { let nv = (pn * ge.e) / ge.len; let t = (Q - nv) / TQ; let t2 = dot(t, t); let lm = (p.mu8 * pn) / (8 * TQ);
      if (t2 > lm * lm) { let tl = i32(isqrt(u32(t2))); let lk = (p.muk8 * pn) / (8 * TQ); Q = nv + ((t * lk) / tl) * TQ; } }
    Q = clamp(Q, vec3i(-PM), vec3i(PM));
    if (any(Q != vec3i(0))) { sup = 1; }
    ln[i * K + s] = vec4i(Q, L.w); acc += Q;
  }
  rn[i] = vec4i(vel[i].xyz + acc - vec3i(0, p.grav * F, 0), i32(n << 8u) | sup);
  if (p.first != 0) { corr[i] = vec4i(cr, 0); }
}`,
    finish: HEAD + `
@group(0) @binding(1) var<storage, read> rk: array<vec4i>;
@group(0) @binding(2) var<storage, read> lk: array<vec4i>;
@group(0) @binding(3) var<storage, read> corr: array<vec4i>;
@group(0) @binding(4) var<storage, read_write> pos: array<vec4i>;
@group(0) @binding(5) var<storage, read_write> vel: array<vec4i>;
@group(0) @binding(6) var<storage, read_write> slot: array<vec4i>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; }
  let me = rk[i]; let n = u32(me.w) >> 8u; let F = p.f; let VM = VMAX * F;
  var v = clamp(me.xyz, vec3i(-VM), vec3i(VM));
  if (p.eps != 0 && (me.w & 1) != 0 && all(abs(v) <= vec3i(p.eps))) { v = vec3i(0); }
  let m = v / F + corr[i].xyz;
  for (var s = 0u; s < n; s++) { slot[i * K + s] = lk[i * K + s]; }
  vel[i] = vec4i(v, i32(n));
  let lo = vec3i(RAD * 256); let hi = (dims() * 1024 - vec3i(RAD)) * 256 - vec3i(1);
  pos[i] = vec4i(clamp(pos[i].xyz + m, lo, hi), 0);
}`,
  };
  // h = { dev, mk, pipe }. Returns buffers and an encoder of `steps` substeps.
  async function gpu(h, S0, prm) {
    const { dev, mk, pipe } = h, U = GPUBufferUsage, SU = U.STORAGE | U.COPY_SRC | U.COPY_DST, N = S0.N, cells = S0.gx * S0.gy * S0.gz, I = prm.iters;
    const buf = sz => dev.createBuffer({ size: sz, usage: SU }), vel0 = S0.vel.slice(); for (let i = 0; i < N; i++) vel0[4 * i + 3] = 0;
    const b = { pos: mk(S0.pos), vel: mk(vel0), slot: buf(16 * K * N), corr: buf(16 * N), cnt: buf(4 * cells), cslot: buf(4 * cells * KS) }, L = [], Rr = [], unis = [];
    for (let k = 0; k <= I; k++) { L.push(buf(16 * K * N)); Rr.push(buf(16 * N)); }
    for (const first of [1, 0]) { const u = new ArrayBuffer(64), dv = new DataView(u); [N, S0.gx, S0.gy, S0.gz].forEach((x, k) => dv.setUint32(4 * k, x, true)); [prm.grav, prm.f, prm.mu8, prm.slop, prm.cq, prm.eps, prm.g0, prm.margin, prm.hi, prm.th, first, prm.muk8].forEach((x, k) => dv.setInt32(16 + 4 * k, x, true)); const ub = dev.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST }); dev.queue.writeBuffer(ub, 0, u); unis.push(ub); }
    const pl = {}; for (const k of Object.keys(WGSL)) pl[k] = await pipe(WGSL[k]);
    const bg = (k, uni, bufs) => dev.createBindGroup({ layout: pl[k].getBindGroupLayout(0), entries: [uni, ...bufs].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const wn = [Math.ceil(N / 64)], passes = [['clear', bg('clear', unis[0], [b.cnt]), [Math.min(65535, Math.ceil(cells / 64)), Math.ceil(Math.ceil(cells / 64) / 65535)]], ['insert', bg('insert', unis[0], [b.pos, b.cnt, b.cslot]), wn], ['match', bg('match', unis[0], [b.pos, b.vel, b.cnt, b.cslot, b.slot, L[0], Rr[0]]), wn]];
    for (let k = 0; k < I; k++) passes.push(['solve', bg('solve', unis[k === 0 ? 0 : 1], [b.pos, b.vel, L[k], Rr[k], L[k + 1], Rr[k + 1], b.corr]), wn]);
    passes.push(['finish', bg('finish', unis[0], [Rr[I], L[I], b.corr, b.pos, b.vel, b.slot]), wn]);
    const enc = (steps, only) => { const e = dev.createCommandEncoder(); for (let t = 0; t < steps; t++) for (const [k, g, w] of passes) { if (only && !only.includes(k)) continue; const ps = e.beginComputePass(); ps.setPipeline(pl[k]); ps.setBindGroup(0, g); ps.dispatchWorkgroups(...w); ps.end(); } return [e.finish()]; };
    const all = [...Object.values(b), ...L, ...Rr, ...unis];
    return { b, enc, cells, bytes_per_body: (16 * 2 + 16 + 16 * K + (I + 1) * (16 * K + 16)), free: () => all.forEach(x => x.destroy()) };
  }
  // Canonical hash of positions, velocities and impulses from raw buffers (slots sorted by partner id).
  function canon(N, pos, vel, slot, cntOf) {
    let h = 0x811c9dc5; const u = x => { h ^= x & 0xffff; h = Math.imul(h, 0x01000193); h ^= x >>> 16; h = Math.imul(h, 0x01000193); };
    for (let i = 0; i < N; i++) { for (let k = 0; k < 3; k++) { u(pos[4 * i + k]); u(vel[4 * i + k]); } const n = cntOf(i), ix = []; for (let q = 0; q < n; q++) ix.push(q); ix.sort((a, c) => (slot[4 * (i * K + a) + 3] >>> 0) - (slot[4 * (i * K + c) + 3] >>> 0)); for (const q of ix) { const o = 4 * (i * K + q); u(slot[o + 3]); u(slot[o]); u(slot[o + 1]); u(slot[o + 2]); } }
    return (h >>> 0).toString(16).padStart(8, '0');
  }
  function canonRef(S) { const sl = new Int32Array(4 * K * S.N); for (let i = 0; i < S.N; i++) for (let q = 0; q < S.sn[i]; q++) { const o = 4 * (i * K + q), s3 = 3 * (i * K + q); sl[o] = S.sP[s3]; sl[o + 1] = S.sP[s3 + 1]; sl[o + 2] = S.sP[s3 + 2]; sl[o + 3] = S.sid[i * K + q] | 0; } return canon(S.N, S.pos, S.vel, sl, i => S.sn[i]); }
  async function stage(h, opts = {}) {
    const { dev, read, timed, log } = h, prm = { ...PRM, ...(opts.prm || {}) }, Rz = { spec: prm };
    // correctness: a loose cloud collapsing into a pile (impacts, sliding, push-out, rest), GPU against the JS specification
    const nC = opts.small ? 12 : 20, steps = opts.small ? 400 : 600, A = lattice(nC, nC, nC, 1400, 1400, 200, 11, 1, 2, 4096, prm.f), B = lattice(nC, nC, nC, 1400, 1400, 200, 11, 1, 2, 4096, prm.f);
    const t0 = performance.now(); let over = 0, chg = 0; for (let t = 0; t < steps; t++) { const r = step(B, prm); over += r.over; chg = r.changed; } Rz.cpu_ms_per_substep = (performance.now() - t0) / steps;
    const G = await gpu(h, A, prm); for (let t = 0; t < steps; t += 50) { dev.queue.submit(G.enc(Math.min(50, steps - t))); await dev.queue.onSubmittedWorkDone(); }
    const gp = await read(G.b.pos, 16 * A.N), gv = await read(G.b.vel, 16 * A.N), gs = await read(G.b.slot, 16 * K * A.N);
    let bad = 0; for (let i = 0; i < A.N; i++) for (let k = 0; k < 3; k++) { if (gp[4 * i + k] !== B.pos[4 * i + k]) bad++; if (gv[4 * i + k] !== B.vel[4 * i + k]) bad++; }
    const st = stats(B);
    Rz.correct = { n: A.N, substeps: steps, hash_gpu: canon(A.N, gp, gv, gs, i => gv[4 * i + 3]), hash_cpu: canonRef(B), mismatch_words: bad, cpu_overflow: over, cpu_changed_last: chg, contacts: st.contacts, cmax: st.cmax, pen_max: st.pen_max };
    G.free(); log('rest correctness: ' + JSON.stringify(Rz.correct));
    // throughput on a settled pile produced by the GPU itself
    Rz.throughput = {};
    for (const n of opts.sizes || (opts.small ? [24] : [50])) {
      const T = lattice(n, n, n, 1400, 1400, 200, 99, 1, 2, 4096, prm.f), H = await gpu(h, T, prm), warm = opts.warm || 3000;
      for (let t = 0; t < warm; t += 100) { dev.queue.submit(H.enc(100)); await dev.queue.onSubmittedWorkDone(); }
      const t1 = await timed(() => H.enc(4), 2), sub = Math.max(4, Math.min(128, Math.ceil(4 * 40 / Math.max(t1, 0.05))));
      const ms = await timed(() => H.enc(sub), 5), row = { n: T.N, cells: H.cells, bytes_per_body: H.bytes_per_body, warm_substeps: warm, substeps_timed: sub, ms_per_substep: ms / sub, ns_per_sphere_substep: ms * 1e6 / (sub * T.N), pass_ns: {} };
      const per = async ks => (await timed(() => H.enc(sub, ks), 3)) * 1e6 / (sub * T.N);
      row.pass_ns.clear_insert = await per(['clear', 'insert']); row.pass_ns.match = await per(['match']); row.pass_ns.solve_all = await per(['solve']); row.pass_ns.finish = await per(['finish']);
      dev.queue.submit(H.enc(1)); await dev.queue.onSubmittedWorkDone();
      const pv = await read(H.b.vel, 16 * T.N), pp = await read(H.b.pos, 16 * T.N); let nc = 0, cm = 0, mv = 0, top = 0; for (let i = 0; i < T.N; i++) { nc += pv[4 * i + 3]; if (pv[4 * i + 3] > cm) cm = pv[4 * i + 3]; if (pv[4 * i] | pv[4 * i + 1] | pv[4 * i + 2]) mv++; if (pp[4 * i + 1] > top) top = pp[4 * i + 1]; }
      row.contacts_per_sphere = nc / T.N; row.cmax = cm; row.moving = mv; row.depth_D = top / 256 / 960;
      Rz.throughput[n] = row; H.free(); log('rest ' + T.N + ': ' + JSON.stringify(row));
    }
    return Rz;
  }
  function fnvFull(S) { let h = 0x811c9dc5; const u = x => { h ^= x & 0xffff; h = Math.imul(h, 0x01000193); h ^= x >>> 16; h = Math.imul(h, 0x01000193); };
    for (let i = 0; i < S.N; i++) { for (let k = 0; k < 3; k++) { u(S.pos[4 * i + k]); u(S.vel[4 * i + k]); } const ix = []; for (let q = 0; q < S.sn[i]; q++) ix.push(q); ix.sort((a, b) => S.sid[i * K + a] - S.sid[i * K + b]); for (const q of ix) { u(S.sid[i * K + q]); for (let k = 0; k < 3; k++) u(S.sP[3 * (i * K + q) + k]); } }
    return (h >>> 0).toString(16); }
  const PRM = { grav: 111, f: 32, mu8: 4, muk8: 3, iters: 4, slop: 4, cq: 8, eps: 32, g0: 2, bg: 0, margin: 16, hi: 24, th: 0 };
  return { DIAM, R, K, KS, MARGIN, WALL, NONE, make, lattice, step, stepActive, run, stats, fnv, fnvFull, WGSL, gpu, stage, canon, canonRef, geom, geo, match, solve, finish, buildGrid, rng, PRM };
})();
if (typeof module !== 'undefined') module.exports = REST;
