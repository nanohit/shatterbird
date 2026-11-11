// Shatterbird phase 1b probe, workload C: a deterministic contact pipeline for N equal spheres.
// Integer state (positions and velocities in 1/256 of a length unit, diameter 1024 units),
// uniform grid built with atomic counters, position-based contact with antisymmetric integer
// corrections summed per sphere. Every reduction over neighbours is an integer sum, so the
// unordered slot order inside a grid cell cannot change the result. Runs in browsers and node.
'use strict';
const SPH = (() => {
  const DIAM = 1024, KSLOT = 8, VMAX = 100 << 8;
  function rng(seed) { let s = seed >>> 0; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s; }; }
  // Lattice of side n with spacing `sp` units and +-jit jitter, resting on the floor of a box with a
  // margin of `pad` cells on the sides and `head` cells above.
  function init(n, sp, jit, seed, pad = 2, head = 4, ny = n) {
    const r = rng(seed), N = n * ny * n, g = Math.ceil(n * sp / DIAM) + 2 * pad, gy = Math.ceil(ny * sp / DIAM) + head;
    const pos = new Int32Array(4 * N), vel = new Int32Array(4 * N);
    let i = 0;
    for (let z = 0; z < n; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < n; x++, i++) {
      const j = () => (r() % (2 * jit + 1)) - jit;
      pos[4 * i] = (pad * DIAM + x * sp + (sp >> 1) + j()) << 8; pos[4 * i + 1] = (y * sp + (DIAM >> 1) + j()) << 8; pos[4 * i + 2] = (pad * DIAM + z * sp + (sp >> 1) + j()) << 8;
      vel[4 * i] = (r() % 2049) - 1024; vel[4 * i + 1] = (r() % 2049) - 1024; vel[4 * i + 2] = (r() % 2049) - 1024;
    }
    return { N, gx: g, gy, gz: g, pos, vel };
  }
  // Random permutation of the sphere order: memory order no longer follows space, so every
  // neighbour read is an incoherent gather (the layout of a world after long mixing).
  function shuffle(S, seed) {
    const r = rng(seed), N = S.N, p = new Int32Array(4 * N), v = new Int32Array(4 * N), ix = new Uint32Array(N);
    for (let i = 0; i < N; i++) ix[i] = i;
    for (let i = N - 1; i > 0; i--) { const j = r() % (i + 1), t = ix[i]; ix[i] = ix[j]; ix[j] = t; }
    for (let i = 0; i < N; i++) for (let k = 0; k < 4; k++) { p[4 * i + k] = S.pos[4 * ix[i] + k]; v[4 * i + k] = S.vel[4 * ix[i] + k]; }
    S.pos = p; S.vel = v; return S;
  }
  function isqrt(v) { let s = Math.floor(Math.sqrt(v)); while (s * s > v) s--; while ((s + 1) * (s + 1) <= v) s++; return s; }
  // One substep of the integer specification. Returns statistics of the step.
  function refStep(S, grav, w) {
    const { N, gx, gy, gz, pos, vel } = S, lo = (DIAM >> 1) << 8, hx = ((gx * DIAM - (DIAM >> 1)) << 8) - 1, hy = ((gy * DIAM - (DIAM >> 1)) << 8) - 1, hz = ((gz * DIAM - (DIAM >> 1)) << 8) - 1;
    const cl = (v, a, b) => v < a ? a : v > b ? b : v;
    const pred = S.pred || (S.pred = new Int32Array(4 * N)), cnt = S.cnt || (S.cnt = new Uint32Array(gx * gy * gz)), slots = S.slots || (S.slots = new Uint32Array(gx * gy * gz * KSLOT));
    cnt.fill(0);
    for (let i = 0; i < N; i++) {
      const vx = cl(vel[4 * i], -VMAX, VMAX), vy = cl(vel[4 * i + 1] - grav, -VMAX, VMAX), vz = cl(vel[4 * i + 2], -VMAX, VMAX);
      pred[4 * i] = cl(pos[4 * i] + vx, lo, hx); pred[4 * i + 1] = cl(pos[4 * i + 1] + vy, lo, hy); pred[4 * i + 2] = cl(pos[4 * i + 2] + vz, lo, hz);
    }
    let over = 0, maxc = 0;
    for (let i = 0; i < N; i++) {
      const c = (pred[4 * i] >> 18) + gx * ((pred[4 * i + 1] >> 18) + gy * (pred[4 * i + 2] >> 18)), s = cnt[c]++;
      if (s < KSLOT) slots[c * KSLOT + s] = i; else over++;
      if (cnt[c] > maxc) maxc = cnt[c];
    }
    let contacts = 0, maxpen = 0, sumpen = 0;
    for (let i = 0; i < N; i++) {
      const px = pred[4 * i] >> 8, py = pred[4 * i + 1] >> 8, pz = pred[4 * i + 2] >> 8, cx = px >> 10, cy = py >> 10, cz = pz >> 10;
      let ax = 0, ay = 0, az = 0;
      for (let dz = -1; dz <= 1; dz++) { const z = cz + dz; if (z < 0 || z >= gz) continue;
        for (let dy = -1; dy <= 1; dy++) { const y = cy + dy; if (y < 0 || y >= gy) continue;
          for (let dx = -1; dx <= 1; dx++) { const x = cx + dx; if (x < 0 || x >= gx) continue;
            const c = x + gx * (y + gy * z), n = Math.min(cnt[c], KSLOT);
            for (let s = 0; s < n; s++) {
              const j = slots[c * KSLOT + s]; if (j === i) continue;
              const ex = px - (pred[4 * j] >> 8), ey = py - (pred[4 * j + 1] >> 8), ez = pz - (pred[4 * j + 2] >> 8), d2 = ex * ex + ey * ey + ez * ez;
              if (d2 >= DIAM * DIAM) continue;
              contacts++;
              if (d2 === 0) { ax += (i < j ? 1 : -1) * ((DIAM * w) >> 4); continue; }
              const len = isqrt(d2), pen = DIAM - len, den = 16 * len;
              sumpen += pen; if (pen > maxpen) maxpen = pen;
              ax += Math.trunc(pen * w * ex / den); ay += Math.trunc(pen * w * ey / den); az += Math.trunc(pen * w * ez / den);
            } } } }
      const nx = cl(pred[4 * i] + ax * 256, lo, hx), ny = cl(pred[4 * i + 1] + ay * 256, lo, hy), nz = cl(pred[4 * i + 2] + az * 256, lo, hz);
      S.nv[4 * i] = cl(nx - pos[4 * i], -VMAX, VMAX); S.nv[4 * i + 1] = cl(ny - pos[4 * i + 1], -VMAX, VMAX); S.nv[4 * i + 2] = cl(nz - pos[4 * i + 2], -VMAX, VMAX);
      S.np[4 * i] = nx; S.np[4 * i + 1] = ny; S.np[4 * i + 2] = nz;
    }
    pos.set(S.np); vel.set(S.nv);
    return { over, maxc, contacts_per_sphere: contacts / N, max_pen: maxpen, mean_pen: contacts ? sumpen / contacts : 0 };
  }
  function ref(S, steps, grav, w) { S.np = new Int32Array(4 * S.N); S.nv = new Int32Array(4 * S.N); let st; for (let t = 0; t < steps; t++) st = refStep(S, grav, w); return st; }

  const HEAD = `
struct P { n: u32, gx: u32, gy: u32, gz: u32, grav: i32, w: i32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
const DIAM = 1024; const KSLOT = 8u; const VMAX = 25600;
fn lo() -> vec3i { return vec3i(131072); }
fn hi() -> vec3i { return (vec3i(i32(p.gx), i32(p.gy), i32(p.gz)) * DIAM - vec3i(512)) * 256 - vec3i(1); }
`;
  const WGSL = {
    integrate: HEAD + `
@group(0) @binding(1) var<storage, read> pos: array<vec4i>;
@group(0) @binding(2) var<storage, read> vel: array<vec4i>;
@group(0) @binding(3) var<storage, read_write> pred: array<vec4i>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; }
  let v = clamp(vel[i].xyz - vec3i(0, p.grav, 0), vec3i(-VMAX), vec3i(VMAX));
  pred[i] = vec4i(clamp(pos[i].xyz + v, lo(), hi()), 0);
}`,
    clear: HEAD + `
@group(0) @binding(1) var<storage, read_write> cnt: array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x + g.y * 4194240u; if (c >= p.gx * p.gy * p.gz) { return; } atomicStore(&cnt[c], 0u);
}`,
    insert: HEAD + `
@group(0) @binding(1) var<storage, read> pred: array<vec4i>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> slots: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; }
  let q = vec3u(pred[i].xyz >> vec3u(18u)); let c = q.x + p.gx * (q.y + p.gy * q.z);
  let s = atomicAdd(&cnt[c], 1u); if (s < KSLOT) { slots[c * KSLOT + s] = i; }
}`,
    solve: HEAD + `
@group(0) @binding(1) var<storage, read> pred: array<vec4i>;
@group(0) @binding(2) var<storage, read> cnt: array<u32>;
@group(0) @binding(3) var<storage, read> slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> pos: array<vec4i>;
@group(0) @binding(5) var<storage, read_write> vel: array<vec4i>;
fn isqrt(v: u32) -> u32 { var s = u32(sqrt(f32(v))); if (s * s > v) { s -= 1u; } if (s * s > v) { s -= 1u; } if ((s + 1u) * (s + 1u) <= v) { s += 1u; } if ((s + 1u) * (s + 1u) <= v) { s += 1u; } return s; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; }
  let pr = pred[i].xyz; let pc = pr >> vec3u(8u); let cc = pc >> vec3u(10u);
  var acc = vec3i(0);
  for (var dz = -1; dz <= 1; dz++) { let z = cc.z + dz; if (z < 0 || z >= i32(p.gz)) { continue; }
    for (var dy = -1; dy <= 1; dy++) { let y = cc.y + dy; if (y < 0 || y >= i32(p.gy)) { continue; }
      for (var dx = -1; dx <= 1; dx++) { let x = cc.x + dx; if (x < 0 || x >= i32(p.gx)) { continue; }
        let c = u32(x) + p.gx * (u32(y) + p.gy * u32(z)); let n = min(cnt[c], KSLOT);
        for (var s = 0u; s < n; s++) {
          let j = slots[c * KSLOT + s]; if (j == i) { continue; }
          let e = pc - (pred[j].xyz >> vec3u(8u)); let d2 = dot(e, e);
          if (d2 >= DIAM * DIAM) { continue; }
          if (d2 == 0) { acc.x += select(-1, 1, i < j) * ((DIAM * p.w) >> 4u); continue; }
          let len = i32(isqrt(u32(d2))); let pen = DIAM - len;
          acc += (pen * p.w * e) / (16 * len);
        } } } }
  let nx = clamp(pr + acc * 256, lo(), hi());
  vel[i] = vec4i(clamp(nx - pos[i].xyz, vec3i(-VMAX), vec3i(VMAX)), 0);
  pos[i] = vec4i(nx, 0);
}`,
  };

  // GPU run. h = { dev, mk, pipe, read, timed, fnv, log }.
  async function gpu(h, S0, grav, w) {
    const { dev, mk, pipe } = h, U = GPUBufferUsage, SU = U.STORAGE | U.COPY_SRC | U.COPY_DST, cells = S0.gx * S0.gy * S0.gz;
    const b = { pos: mk(S0.pos), vel: mk(S0.vel), pred: dev.createBuffer({ size: 16 * S0.N, usage: SU }), cnt: dev.createBuffer({ size: 4 * cells, usage: SU }), slots: dev.createBuffer({ size: 4 * cells * KSLOT, usage: SU }), uni: dev.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST }) };
    const u = new ArrayBuffer(32), dv = new DataView(u); [S0.N, S0.gx, S0.gy, S0.gz].forEach((x, k) => dv.setUint32(4 * k, x, true)); dv.setInt32(16, grav, true); dv.setInt32(20, w, true); dev.queue.writeBuffer(b.uni, 0, u);
    const pl = {}; for (const k of Object.keys(WGSL)) pl[k] = await pipe(WGSL[k]);
    const bg = (k, bufs) => dev.createBindGroup({ layout: pl[k].getBindGroupLayout(0), entries: [b.uni, ...bufs].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const g = { integrate: bg('integrate', [b.pos, b.vel, b.pred]), clear: bg('clear', [b.cnt]), insert: bg('insert', [b.pred, b.cnt, b.slots]), solve: bg('solve', [b.pred, b.cnt, b.slots, b.pos, b.vel]) };
    const wg = { integrate: [Math.ceil(S0.N / 64)], insert: [Math.ceil(S0.N / 64)], solve: [Math.ceil(S0.N / 64)], clear: [Math.min(65535, Math.ceil(cells / 64)), Math.ceil(Math.ceil(cells / 64) / 65535)] };
    const enc = (steps, only) => { const e = dev.createCommandEncoder(); for (let t = 0; t < steps; t++) for (const k of only || ['integrate', 'clear', 'insert', 'solve']) { const ps = e.beginComputePass(); ps.setPipeline(pl[k]); ps.setBindGroup(0, g[k]); ps.dispatchWorkgroups(...wg[k]); ps.end(); } return [e.finish()]; };
    return { b, enc, cells, free: () => Object.values(b).forEach(x => x.destroy()) };
  }

  async function stage(h, opts = {}) {
    const { dev, read, timed, fnv, log } = h, R = { spec: { diam: DIAM, slots: KSLOT, grav: 111, w: 2 } }, grav = 111, w = 2;
    // correctness: dense jittered lattice, many contacts from the first substep
    const nC = opts.small ? 16 : 28, steps = opts.small ? 32 : 64, A = init(nC, 900, 60, 4242), B = init(nC, 900, 60, 4242);
    const t0 = performance.now(), st = ref(B, steps, grav, w); R.cpu_ms_per_substep = (performance.now() - t0) / steps;
    const G = await gpu(h, A, grav, w); dev.queue.submit(G.enc(steps));
    const gp = await read(G.b.pos, 16 * A.N), gv = await read(G.b.vel, 16 * A.N), gc = await read(G.b.cnt, 4 * G.cells);
    let bad = 0; for (let i = 0; i < gp.length; i++) { if (gp[i] !== B.pos[i]) bad++; if (gv[i] !== B.vel[i]) bad++; }
    let maxc = 0; for (let i = 0; i < gc.length; i++) if (gc[i] > maxc) maxc = gc[i];
    R.correct = { n: A.N, substeps: steps, hash_gpu: fnv(gp) + fnv(gv), hash_cpu: fnv(B.pos) + fnv(B.vel), mismatch_words: bad, max_cell_count_gpu: maxc, cpu: st };
    G.free(); log('spheres correctness: ' + JSON.stringify(R.correct));
    // throughput: dense state after a warm-up; total and per pass; memory order following space
    // ("coherent") and randomly permuted ("shuffled")
    R.throughput = {};
    for (const n of opts.small ? [32, 48] : [50, 100]) for (const layout of ['coherent', 'shuffled']) {
      const T = init(n, 1000, 40, 99); if (layout === 'shuffled') shuffle(T, 31337);
      const H = await gpu(h, T, grav, w);
      for (let k = 0; k < 6; k++) { dev.queue.submit(H.enc(100)); await dev.queue.onSubmittedWorkDone(); }
      const t1 = await timed(() => H.enc(4), 2), sub = Math.max(4, Math.min(256, Math.ceil(4 * 40 / Math.max(t1, 0.05))));
      const ms = await timed(() => H.enc(sub), 5), row = { n: T.N, layout, cells: H.cells, substeps_timed: sub, ms_per_substep: ms / sub, ns_per_sphere_substep: ms * 1e6 / (sub * T.N), pass_ns: {} };
      // per pass, on a valid grid (clear + insert once before the passes that read it)
      const per = async ks => (await timed(() => H.enc(sub, ks), 3)) * 1e6 / (sub * T.N);
      row.pass_ns.integrate = await per(['integrate']); row.pass_ns.clear = await per(['clear']); row.pass_ns.clear_insert = await per(['clear', 'insert']);
      dev.queue.submit(H.enc(1)); await dev.queue.onSubmittedWorkDone(); row.pass_ns.solve = await per(['solve']);
      // contact statistics of the timed state, from the integer reference on the read-back state
      const Q = { N: T.N, gx: T.gx, gy: T.gy, gz: T.gz, pos: await read(H.b.pos, 16 * T.N), vel: await read(H.b.vel, 16 * T.N), np: new Int32Array(4 * T.N), nv: new Int32Array(4 * T.N) }, qs = refStep(Q, grav, w);
      row.contacts_per_sphere = qs.contacts_per_sphere; row.max_cell_count = qs.maxc; row.overflow = qs.over; row.max_pen = qs.max_pen; row.mean_pen = qs.mean_pen;
      R.throughput[n + '_' + layout] = row; H.free(); log('spheres ' + T.N + ' ' + layout + ': ' + JSON.stringify(row));
    }
    return R;
  }
  return { init, shuffle, ref, refStep, stage, WGSL, KSLOT, DIAM };
})();
if (typeof module !== 'undefined') module.exports = SPH;
