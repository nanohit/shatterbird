// Cross-device harness for gpu-probe/rest.js (specification and pipeline untouched): signatures with
// full state dumps, cell and contact overflow, invariance under a permutation of the body order, and
// a per-stage GPU-time profile from timestamp queries. window.xdev(opts) -> result object.
'use strict';
const XD = (() => {
  // GPU time per stage: every compute pass begun while `on` gets a pair of timestamp queries; the
  // stage name comes from the pipeline set on the pass.
  const prof = { on: false, qs: null, cap: 0, qi: 0, labels: [], names: new Map() };
  function install() {
    if (install.done) return; install.done = true;
    const bp = GPUCommandEncoder.prototype.beginComputePass, sp = GPUComputePassEncoder.prototype.setPipeline;
    GPUCommandEncoder.prototype.beginComputePass = function (d = {}) {
      if (prof.on && prof.qi + 2 <= prof.cap) { const p = bp.call(this, { ...d, timestampWrites: { querySet: prof.qs, beginningOfPassWriteIndex: prof.qi, endOfPassWriteIndex: prof.qi + 1 } }); p.__q = prof.qi >> 1; prof.qi += 2; return p; }
      return bp.call(this, d);
    };
    GPUComputePassEncoder.prototype.setPipeline = function (pl) { if (this.__q !== undefined) prof.labels[this.__q] = prof.names.get(pl) || 'other'; return sp.call(this, pl); };
  }
  async function setup(R) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); if (!adapter) throw new Error('no adapter');
    const inf = adapter.info || {}; R.adapter = { vendor: inf.vendor, architecture: inf.architecture, device: inf.device, description: inf.description, isFallbackAdapter: inf.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null };
    const ts = adapter.features.has('timestamp-query'); R.timestamp_query = ts;
    const dev = await adapter.requestDevice({ requiredFeatures: ts ? ['timestamp-query'] : [], requiredLimits: { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30), maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30), maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 10) } });
    dev.addEventListener('uncapturederror', e => R.errors.push(String(e.error.message).slice(0, 300)));
    const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const mk = (data, usage = SU) => { const b = dev.createBuffer({ size: data.byteLength, usage }); dev.queue.writeBuffer(b, 0, data); return b; };
    let pnames = null, pk = 0;
    const pipe = async code => { const module = dev.createShaderModule({ code }); const ci = await module.getCompilationInfo(); const er = ci.messages.filter(m => m.type === 'error'); if (er.length) throw new Error(er.map(m => m.lineNum + ':' + m.linePos + ' ' + m.message).join(' | ')); const p = await dev.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } }); if (pnames) prof.names.set(p, pnames[pk++ % pnames.length]); return p; };
    const read = async (buf, bytes) => { const r = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); const e = dev.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, r, 0, bytes); dev.queue.submit([e.finish()]); await r.mapAsync(GPUMapMode.READ); const a = new Int32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return a; };
    const timed = async (enc, reps = 5) => { let best = 1e99; for (let k = 0; k < reps; k++) { const cmds = enc(); await dev.queue.onSubmittedWorkDone(); const t0 = performance.now(); dev.queue.submit(cmds); await dev.queue.onSubmittedWorkDone(); best = Math.min(best, performance.now() - t0); } return best; };
    const names = n => { pnames = n; pk = 0; };
    // GPU time of the passes encoded by enc(): { total_ms (first begin to last end), stage_ms: {name: sum} }
    const gputime = async enc => {
      if (!ts) return null; install();
      prof.cap = 4096; prof.qs = dev.createQuerySet({ type: 'timestamp', count: prof.cap }); prof.qi = 0; prof.labels = []; prof.on = true;
      const cmds = enc(); prof.on = false; const n = prof.qi, rb = dev.createBuffer({ size: 8 * n, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }), e = dev.createCommandEncoder(); e.resolveQuerySet(prof.qs, 0, n, rb, 0);
      await dev.queue.onSubmittedWorkDone(); dev.queue.submit([...cmds, e.finish()]);
      const raw = await read(rb, 8 * n), t = new BigUint64Array(raw.buffer); rb.destroy(); prof.qs.destroy();
      const st = {}; for (let q = 0; q < n / 2; q++) { const d = Number(t[2 * q + 1] - t[2 * q]) / 1e6; st[prof.labels[q]] = (st[prof.labels[q]] || 0) + d; }
      return { passes: n / 2, total_ms: Number(t[n - 1] - t[0]) / 1e6, stage_ms: st };
    };
    return { dev, mk, pipe, read, timed, names, gputime, log: m => { console.log(m); document.getElementById('o').textContent += '\n' + m; } };
  }
  const save = async (name, arr) => { const r = await fetch('/save/' + name, { method: 'POST', body: new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength) }); if (!r.ok) throw new Error('save failed ' + name); };
  return { setup, save, prof };
})();

async function xdev(opts = {}) {
  const R = { tag: opts.tag || 'dev', ua: navigator.userAgent, when: new Date().toISOString(), errors: [] }, K = REST.K;
  try {
    const h = await XD.setup(R), { dev, read, log } = h, prm = { ...REST.PRM, ...(opts.prm || {}) }; R.spec = prm;
    const scene = n => REST.lattice(n, n, n, 1400, 1400, 200, 11, 1, 2, 4096, prm.f);
    const run = async (S0, steps, watch) => { h.names(Object.keys(REST.WGSL)); const G = await REST.gpu(h, S0, prm); let maxcell = 0;
      for (let t = 0; t < steps; t += 50) { dev.queue.submit(G.enc(Math.min(50, steps - t))); await dev.queue.onSubmittedWorkDone(); if (watch) { const c = await read(G.b.cnt, 4 * G.cells); for (let i = 0; i < c.length; i++) if (c[i] > maxcell) maxcell = c[i]; } }
      const o = { pos: await read(G.b.pos, 16 * S0.N), vel: await read(G.b.vel, 16 * S0.N), slot: await read(G.b.slot, 16 * K * S0.N), maxcell }; G.free(); return o; };
    // 1. signatures, full state, overflow
    R.signature = {};
    for (const [label, n, steps] of opts.sig || [['small', 12, 400], ['full', 20, 600]]) {
      const A = scene(n), B = scene(n); let over = 0, cellmax = 0, cmax = 0;
      for (let t = 0; t < steps; t++) { const r = REST.step(B, prm); over += r.over; for (let c = 0; c < B.cnt.length; c++) if (B.cnt[c] > cellmax) cellmax = B.cnt[c]; for (let i = 0; i < B.N; i++) if (B.cn[i] > cmax) cmax = B.cn[i]; }
      const g = await run(A, steps, true); let bad = 0, gcmax = 0;
      for (let i = 0; i < A.N; i++) { for (let k = 0; k < 3; k++) { if (g.pos[4 * i + k] !== B.pos[4 * i + k]) bad++; if (g.vel[4 * i + k] !== B.vel[4 * i + k]) bad++; } if (g.vel[4 * i + 3] > gcmax) gcmax = g.vel[4 * i + 3]; }
      R.signature[label] = { n: A.N, substeps: steps, hash_gpu: REST.canon(A.N, g.pos, g.vel, g.slot, i => g.vel[4 * i + 3]), hash_cpu: REST.canonRef(B), mismatch_words: bad,
        cell_capacity: REST.KS, max_cell_count_gpu_sampled: g.maxcell, max_cell_count_cpu_every_step: cellmax, contact_capacity: K, max_contacts_gpu: gcmax, max_contacts_cpu_every_step: cmax, cpu_overflow_events: over };
      if (opts.dump !== false) for (const k of ['pos', 'vel', 'slot']) await XD.save(`${R.tag}_${label}_${k}.bin`, g[k]);
      log('signature ' + label + ': ' + JSON.stringify(R.signature[label]));
    }
    // 2. the same world with its bodies stored in a different order
    if (opts.perm !== false) {
      const n = opts.perm_n || 20, steps = opts.perm_steps || 600, A = scene(n), N = A.N, P = scene(n), rnd = REST.rng(777), pi = new Uint32Array(N), inv = new Uint32Array(N);
      for (let i = 0; i < N; i++) pi[i] = i; for (let i = N - 1; i > 0; i--) { const j = rnd() % (i + 1), t = pi[i]; pi[i] = pi[j]; pi[j] = t; }
      for (let k = 0; k < N; k++) { inv[pi[k]] = k; for (let c = 0; c < 4; c++) { P.pos[4 * k + c] = A.pos[4 * pi[k] + c]; P.vel[4 * k + c] = A.vel[4 * pi[k] + c]; } }
      const a = await run(A, steps), p = await run(P, steps); let bad = 0, sbad = 0, cbad = 0;
      const key = (st, i, map) => { const m = st.vel[4 * i + 3], o = []; for (let q = 0; q < m; q++) { const b = 4 * (i * K + q), id = st.slot[b + 3] >>> 0; o.push([id >= REST.WALL ? id : map(id), st.slot[b], st.slot[b + 1], st.slot[b + 2]].join(',')); } return o.sort().join(';'); };
      for (let k = 0; k < N; k++) { const i = pi[k]; for (let c = 0; c < 3; c++) { if (p.pos[4 * k + c] !== a.pos[4 * i + c]) bad++; if (p.vel[4 * k + c] !== a.vel[4 * i + c]) bad++; } if (p.vel[4 * k + 3] !== a.vel[4 * i + 3]) cbad++; if (key(p, k, j => pi[j]) !== key(a, i, j => j)) sbad++; }
      R.permutation = { n: N, substeps: steps, mismatch_words_pos_vel: bad, bodies_with_different_contact_count: cbad, bodies_with_different_contact_sets_or_impulses: sbad };
      log('permutation: ' + JSON.stringify(R.permutation));
    }
    // 3. GPU time per stage on the throughput scene of the probe
    if (opts.profile !== false) {
      R.profile = {};
      for (const n of opts.sizes || [50]) {
        const T = REST.lattice(n, n, n, 1400, 1400, 200, 99, 1, 2, 4096, prm.f); h.names(Object.keys(REST.WGSL)); const H = await REST.gpu(h, T, prm), warm = opts.warm || 3000;
        for (let t = 0; t < warm; t += 100) { dev.queue.submit(H.enc(100)); await dev.queue.onSubmittedWorkDone(); }
        const sub = opts.sub || 32, wall = await h.timed(() => H.enc(sub), 5), row = { n: T.N, bytes_per_body: H.bytes_per_body, substeps: sub, wall_ns_per_sphere_substep: wall * 1e6 / (sub * T.N), runs: [] };
        for (let k = 0; k < 3; k++) { const g = await h.gputime(() => H.enc(sub)); if (g) row.runs.push({ gpu_ns_per_sphere_substep: g.total_ms * 1e6 / (sub * T.N), stage_ns: Object.fromEntries(Object.entries(g.stage_ms).map(([s, ms]) => [s, ms * 1e6 / (sub * T.N)])) }); }
        const v = await read(H.b.vel, 16 * T.N); let nc = 0, mv = 0; for (let i = 0; i < T.N; i++) { nc += v[4 * i + 3]; if (v[4 * i] | v[4 * i + 1] | v[4 * i + 2]) mv++; } row.contacts_per_sphere = nc / T.N; row.moving = mv;
        R.profile[n] = row; H.free(); log('profile ' + T.N + ': ' + JSON.stringify(row));
      }
    }
    dev.destroy();
  } catch (e) { R.error = String(e && e.message || e); }
  return R;
}
