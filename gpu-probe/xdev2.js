// Cross-device harness for specification v2 (rest2.js): full pipeline against the JS specification,
// incremental against full, permutation of the body order, forced overflow, GPU time per stage.
// window.xdev2(opts) -> result object. opts.tests: subset of ['full','inc','perm','overflow','time'].
'use strict';
async function xdev2(opts = {}) {
  const V = REST2, K = V.K, R = { tag: opts.tag || 'dev', ua: navigator.userAgent, when: new Date().toISOString(), errors: [] }, tests = opts.tests || ['full', 'inc', 'perm', 'overflow'];
  try {
    const h = await XD.setup(R), { dev, read, log } = h, prm = { ...V.PRM, ...(opts.prm || {}) }; R.spec = prm;
    const cloud = (n, ny = n, seed = 11) => V.prime(V.lattice(n, ny, n, 1400, 1400, 200, seed, 1, 2, 4096, prm.f), prm);
    // a GPU run: steps, state read-back, canonical hash, counters, host edits
    const start = async (S0, o) => { h.names(Object.keys(V.WGSL)); const G = await V.gpu(h, S0, prm, o), N = S0.N;
      const run = { G, N, steps: async n => { for (let t = 0; t < n; t += 40) { dev.queue.submit(G.enc(Math.min(40, n - t))); await dev.queue.onSubmittedWorkDone(); } },
        state: async () => ({ pos: await read(G.b.pos, 16 * N), vel: await read(G.b.vel, 16 * N), slot: await read(G.b.la, 16 * K * N) }),
        hash: async () => { const s = await run.state(); return V.canon(N, s.pos, s.vel, s.slot); },
        counters: async () => V.counters(await read(G.b.at, 4 * 128)),
        // host edit of one body's velocity; the next substep runs full (the header flag "primed" is cleared)
        // host edit of one body's velocity. With `cell` (the body's cell, world at rest) the edit is announced as a
        // seed and the next substep stays incremental; without it the next substep runs full.
        kick: (i, v, cell) => { const a = Math.max(Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2])); dev.queue.writeBuffer(G.b.vel, 16 * i, new Int32Array(v)); dev.queue.writeBuffer(G.b.pos, 16 * i + 12, new Int32Array([a <= prm.vr ? 1 : 0]));
          if (cell === undefined) dev.queue.writeBuffer(G.b.bl, 28, new Uint32Array([0])); else { const s = i & 63; dev.queue.writeBuffer(G.b.seeds, 4 * s * G.seg, new Uint32Array([cell])); dev.queue.writeBuffer(G.b.at, 4 * (V.AT.SEED + s), new Uint32Array([1])); } } };
      return run; };
    const jsKick = (S, i, v, seed) => { S.vel[4 * i] = v[0]; S.vel[4 * i + 1] = v[1]; S.vel[4 * i + 2] = v[2]; const a = Math.max(Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2])); S.pos[4 * i + 3] = a <= prm.vr ? 1 : 0; S.seed = seed ? [V.cellOf(S, i)] : null; };
    const surface = S => { let k = 0, best = -1e18; const mx = S.gx * 512 * 256, mz = S.gz * 512 * 256; for (let i = 0; i < S.N; i++) { const d = Math.hypot(S.pos[4 * i] - mx, S.pos[4 * i + 2] - mz), sc = S.pos[4 * i + 1] - 2 * d; if (sc > best) { best = sc; k = i; } } return k; };
    const diffState = (a, b, N) => { let pv = 0, sl = 0; for (let i = 0; i < N; i++) { for (let k = 0; k < 4; k++) { if (a.pos[4 * i + k] !== b.pos[4 * i + k]) pv++; if (a.vel[4 * i + k] !== b.vel[4 * i + k]) pv++; } } return pv; };

    if (tests.includes('full')) { R.full = {};
      for (const [label, n, steps] of opts.sig || [['small', 12, 400], ['full', 20, 600]]) {
        const A = cloud(n), B = cloud(n); for (let t = 0; t < steps; t++) V.step(B, prm);
        const g = await start(A, { force: true }); await g.steps(steps); const s = await g.state(), c = await g.counters();
        R.full[label] = { n: A.N, substeps: steps, hash_gpu: V.canon(A.N, s.pos, s.vel, s.slot), hash_cpu: V.hash(B), pos_vel_words_different: diffState(s, B, A.N), faults_gpu: c.faults, faults_cpu: Array.from(B.faults), full_steps: c.full_steps, inc_steps: c.inc_steps, bytes_per_body: Math.round(g.G.bytes_per_body), passes_per_substep: g.G.passes };
        if (opts.dump !== false) for (const k of ['pos', 'vel', 'slot']) await XD.save(`${R.tag}_v2${label}_${k}.bin`, s[k]);
        g.G.free(); log('full ' + label + ': ' + JSON.stringify(R.full[label]));
      } }

    if (tests.includes('inc')) {
      // one world three ways: JS incremental, GPU forced full, GPU incremental; settle, rest, kick, wake, rest
      const n = opts.inc_n || 16, A = cloud(n), J = cloud(n), N = A.N, gf = await start(A, { force: true }), gi = await start(A, { dense: opts.dense === undefined ? 0xFFFFFFFF : opts.dense }), rows = []; let jwork = 0, bad = 0, jrest = -1;
      const check = async (label) => { const hf = await gf.hash(), hi = await gi.hash(), hj = V.hash(J), ci = await gi.counters(), cf = await gf.counters(); const ok = hf === hi && hi === hj; if (!ok) bad++;
        rows.push({ at: label, js: hj, gpu_full: hf, gpu_inc: hi, identical: ok, work_js: jwork, work_gpu_inc: ci.work, work_gpu_full: cf.work, faults_inc: ci.faults.join('/'), faults_full: cf.faults.join('/'), faults_js: Array.from(J.faults).join('/'), inc_full_steps: ci.full_steps, inc_inc_steps: ci.inc_steps }); };
      const adv = async (steps, every, label) => { for (let t = 0; t < steps; t += every) { for (let q = 0; q < every; q++) { const r = V.stepInc(J, prm); jwork += r.evaluated; if (r.changed === 0 && jrest < 0) jrest = J.t; } await gf.steps(every); await gi.steps(every); await check(label + '+' + (t + every)); } };
      await adv(opts.inc_settle || 1000, 100, 'settle'); const k = surface(J), v = [Math.round(0.5 * 5436) * prm.f, -5436 * prm.f, 0];
      jsKick(J, k, v, jrest > 0); gf.kick(k, v); gi.kick(k, v, jrest > 0 ? V.cellOf(J, k) : undefined); await adv(300, 25, 'kick');
      R.inc = { n: N, js_rest_at: jrest, checkpoints: rows.length, mismatching_checkpoints: bad, rows: opts.verbose ? rows : rows.filter((r, i) => !r.identical || i === rows.length - 1 || i === 9) };
      gf.G.free(); gi.G.free(); log('inc: ' + JSON.stringify({ ...R.inc, rows: undefined }) + ' last ' + JSON.stringify(rows[rows.length - 1]));
    }

    if (tests.includes('fuzz')) {
      // random host edits on a settled pile: single kicks, bursts of many kicks (the substep turns full and
      // back), and a run with tiny list capacities so that the overflow path (full substep, global reset
      // of the marks) is taken. JS full stepping against GPU incremental, every 40 substeps.
      R.fuzz = [];
      for (const cfg of opts.fuzz || [{ label: 'default capacities' }, { label: 'cell lists of 256 entries, seed stripes of 8', capc: 256, seg: 8 }, { label: 'never switch to full by seed count', dense: 0xFFFFFFFF }]) {
        const n = opts.fuzz_n || 14, A = cloud(n), J = cloud(n), N = A.N, gi = await start(A, cfg), rnd = V.rng(4242), rows = []; let bad = 0, edits = 0;
        const sync = async label => { const hi = await gi.hash(), hj = V.hash(J); if (hi !== hj) bad++; rows.push({ at: label, js: hj, gpu_inc: hi, ok: hi === hj }); };
        for (let t = 0; t < 800; t += 100) { for (let q = 0; q < 100; q++) V.step(J, prm); await gi.steps(100); } await sync('settled');
        for (let round = 0; round < (opts.fuzz_rounds || 24); round++) {
          const burst = round % 6 === 5 ? 300 : 1 + rnd() % 3;
          for (let b = 0; b < burst; b++) { const i = rnd() % N, sp = 1000 + rnd() % 20000, v = [((rnd() % 2001) - 1000) * sp / 1000 * prm.f | 0, -sp * prm.f, ((rnd() % 2001) - 1000) * sp / 1000 * prm.f | 0]; jsKick(J, i, v); gi.kick(i, v); edits++; }
          for (let q = 0; q < 40; q++) V.step(J, prm); await gi.steps(40); await sync('round ' + round + ', ' + burst + ' kicks');
        }
        const c = await gi.counters(); R.fuzz.push({ config: cfg, n: N, host_edits: edits, checkpoints: rows.length, mismatching_checkpoints: bad, full_substeps: c.full_steps, incremental_substeps: c.inc_steps, faults_gpu: c.faults.join('/'), faults_js: Array.from(J.faults).join('/'), first_mismatch: rows.find(r => !r.ok) || null, last: rows[rows.length - 1] });
        gi.G.free(); log('fuzz ' + cfg.label + ': ' + JSON.stringify(R.fuzz[R.fuzz.length - 1]));
      }
    }

    if (tests.includes('perm')) {
      const n = opts.perm_n || 20, steps = opts.perm_steps || 600, A = cloud(n), N = A.N, P = cloud(n), rnd = V.rng(777), pi = new Uint32Array(N);
      for (let i = 0; i < N; i++) pi[i] = i; for (let i = N - 1; i > 0; i--) { const j = rnd() % (i + 1), t = pi[i]; pi[i] = pi[j]; pi[j] = t; }
      for (let k = 0; k < N; k++) for (let c = 0; c < 4; c++) { P.pos[4 * k + c] = A.pos[4 * pi[k] + c]; P.vel[4 * k + c] = A.vel[4 * pi[k] + c]; }
      const ga = await start(A, { force: true }), gp = await start(P, { force: true }); await ga.steps(steps); await gp.steps(steps); const a = await ga.state(), p = await gp.state(); let bad = 0, sbad = 0;
      const key = (st, i, map) => { const m = st.vel[4 * i + 3] & 31, o = []; for (let q = 0; q < m; q++) { const b = 4 * (i * K + q), id = st.slot[b + 3] >>> 0; o.push([id >= V.WALL ? id : map(id), st.slot[b], st.slot[b + 1], st.slot[b + 2]].join(',')); } return o.sort().join(';'); };
      for (let k = 0; k < N; k++) { const i = pi[k]; for (let c = 0; c < 4; c++) { if (p.pos[4 * k + c] !== a.pos[4 * i + c]) bad++; if (p.vel[4 * k + c] !== a.vel[4 * i + c]) bad++; } if (key(p, k, j => pi[j]) !== key(a, i, j => j)) sbad++; }
      R.permutation = { n: N, substeps: steps, mismatch_words_pos_vel: bad, bodies_with_different_contacts_or_impulses: sbad, faults: (await ga.counters()).faults.join('/') + ' | ' + (await gp.counters()).faults.join('/') };
      ga.G.free(); gp.G.free(); log('permutation: ' + JSON.stringify(R.permutation));
    }

    if (tests.includes('overflow')) {
      // a clump of bodies squeezed into a few cells inside a cloud: more than 8 bodies per cell and more than K contacts per body
      const mk = () => { const S = cloud(10), r = V.rng(5), c0 = [S.gx * 512, 3000, S.gz * 512]; for (let i = 0; i < 120; i++) { S.pos[4 * i] = (c0[0] + (r() % 700) - 350) << 8; S.pos[4 * i + 1] = (c0[1] + (r() % 700) - 350) << 8; S.pos[4 * i + 2] = (c0[2] + (r() % 700) - 350) << 8; } return V.prime(S, prm); };
      const A = mk(), B = mk(), N = A.N, steps = 60; let maxcell = 0; V.buildGrid(B); { const cnt = new Map(); for (let i = 0; i < N; i++) { const c = V.cellOf(B, i); cnt.set(c, (cnt.get(c) || 0) + 1); } for (const x of cnt.values()) if (x > maxcell) maxcell = x; }
      for (let t = 0; t < steps; t++) V.step(B, prm);
      const g = await start(A, { force: true }); await g.steps(steps); const s = await g.state(), c = await g.counters();
      // the same world with its bodies renumbered
      const P = mk(), rnd = V.rng(99), pi = new Uint32Array(N); for (let i = 0; i < N; i++) pi[i] = i; for (let i = N - 1; i > 0; i--) { const j = rnd() % (i + 1), t = pi[i]; pi[i] = pi[j]; pi[j] = t; }
      const Q = mk(); for (let k = 0; k < N; k++) for (let q = 0; q < 4; q++) { Q.pos[4 * k + q] = P.pos[4 * pi[k] + q]; Q.vel[4 * k + q] = P.vel[4 * pi[k] + q]; }
      const gq = await start(Q, { force: true }); await gq.steps(steps); const sq = await gq.state(), cq = await gq.counters(); let pbad = 0; for (let k = 0; k < N; k++) for (let q = 0; q < 4; q++) { if (sq.pos[4 * k + q] !== s.pos[4 * pi[k] + q]) pbad++; if (sq.vel[4 * k + q] !== s.vel[4 * pi[k] + q]) pbad++; }
      R.overflow = { n: N, substeps: steps, max_bodies_in_one_cell_at_start: maxcell, hash_gpu: V.canon(N, s.pos, s.vel, s.slot), hash_cpu: V.hash(B), pos_vel_words_different: diffState(s, B, N), faults_gpu: c.faults, faults_cpu: Array.from(B.faults), renumbered_pos_vel_words_different: pbad, renumbered_faults: cq.faults };
      if (opts.dump !== false) for (const k of ['pos', 'vel', 'slot']) await XD.save(`${R.tag}_v2overflow_${k}.bin`, s[k]);
      g.G.free(); gq.G.free(); log('overflow: ' + JSON.stringify(R.overflow));
    }
    if (tests.includes('time')) {
      // GPU time (timestamp queries) of full and incremental stepping of the same world: all bodies falling,
      // settling, rest, the wake of a kick. Wide shallow clouds: [nx, ny] -> nx * ny * nx bodies.
      R.time = [];
      for (const [nx, ny] of opts.sizes || [[64, 8]]) {
        const A = cloud(nx, ny, 99), N = A.N, gf = await start(A, { force: true }), gi = await start(A, {}), row = { n: N, layers: ny, cells: gi.G.cells, bytes_per_body: Math.round(gi.G.bytes_per_body), passes_per_substep: gi.G.passes };
        const tm = async (g, sub, split) => { const t = await h.gputime(() => g.G.enc(sub, split)); return t ? { ms: t.total_ms / sub, st: split ? Object.fromEntries(Object.entries(t.stage_ms).map(([k, v]) => [k, +(v / sub).toFixed(4)])) : undefined } : null; };
        const pair = async (sub, label, split) => { const f = await tm(gf, sub, split), i = await tm(gi, sub, split); return { at: label, substeps: sub, full_ms: +f.ms.toFixed(4), inc_ms: +i.ms.toFixed(4), ratio: +(i.ms / f.ms).toFixed(4), full_stages: f.st, inc_stages: i.st }; };
        row.falling = await pair(40, 'first 40 substeps, every body moving'); log('time ' + N + ' falling: ' + JSON.stringify({ ...row.falling, full_stages: undefined, inc_stages: undefined }));
        // settle: chunks of 100 substeps until the incremental run evaluates nothing
        let t = 40, w0 = (await gi.counters()).work, rest = -1; const smax = opts.settle_max || 4000; row.settle = [];
        while (t < smax) { const p = await pair(50, 'settle'); await gf.steps(50); await gi.steps(50); t += 100; const c = await gi.counters(); p.t = t; p.evaluated_fraction = +((c.work - w0) / (100 * N)).toFixed(4); w0 = c.work; delete p.full_stages; delete p.inc_stages; row.settle.push(p); if (p.evaluated_fraction === 0) { rest = t; break; } }
        row.rest_by_substep = rest; row.at_rest = await pair(40, 'rest'); row.at_rest_split_passes = await pair(40, 'rest, one pass per stage', true); log('time ' + N + ' rest by ' + rest + ': ' + JSON.stringify(row.at_rest));
        const hf0 = await gf.hash(), hi0 = await gi.hash(); row.rest_hash = hi0; row.rest_identical = hf0 === hi0;
        // kick one surface sphere, 1 m/s down and 0.5 m/s sideways; windows of 20 substeps
        const pos = await read(gi.G.b.pos, 16 * N), k = surface({ N, gx: A.gx, gz: A.gz, pos }), v = [Math.round(0.5 * 5436) * prm.f, -5436 * prm.f, 0]; gf.kick(k, v); gi.kick(k, v, rest > 0 ? (pos[4 * k] >> 18) + A.gx * ((pos[4 * k + 1] >> 18) + A.gy * (pos[4 * k + 2] >> 18)) : undefined); row.wake = []; let wf = 0, wi = 0, quiet = -1; w0 = (await gi.counters()).work;
        for (let q = 0; q < 20; q++) { const p = await pair(20, 'wake'); const c = await gi.counters(); p.t = (q + 1) * 20; p.evaluated_fraction = +((c.work - w0) / (20 * N)).toFixed(5); w0 = c.work; wf += p.full_ms * 20; wi += p.inc_ms * 20; delete p.full_stages; delete p.inc_stages; row.wake.push(p); if (p.evaluated_fraction === 0 && q > 0) { quiet = p.t; break; } }
        row.wake_quiet_by = quiet; row.wake_total = { full_ms: +wf.toFixed(3), inc_ms: +wi.toFixed(3), ratio: +(wi / wf).toFixed(4) };
        const hf1 = await gf.hash(), hi1 = await gi.hash(), ci = await gi.counters(), cf = await gf.counters(); row.wake_hash = hi1; row.wake_identical = hf1 === hi1; row.faults = { inc: ci.faults.join('/'), full: cf.faults.join('/') }; row.inc_steps = [ci.full_steps, ci.inc_steps];
        const st = await gi.state(); let an = 0, na = 0, nc = 0; for (let i = 0; i < N; i++) { const w = st.vel[4 * i + 3]; nc += w & 31; const aw = w >> 8; if (aw) { na++; for (let c = 0; c < 3; c++) an += Math.abs(((aw >> (7 * c)) & 127) << 25 >> 25); } }
        row.contacts_per_body = +(nc / N).toFixed(2); row.anchors_pct_weight = +(an / (N * prm.grav * prm.f) * 100).toFixed(4); row.anchored_bodies = na;
        R.time.push(row); gf.G.free(); gi.G.free(); log('time ' + N + ': ' + JSON.stringify({ rest_by: rest, rest: row.at_rest.ratio, wake: row.wake_total, quiet: quiet, identical: [row.rest_identical, row.wake_identical], falling: row.falling.ratio, bytes: row.bytes_per_body }));
      }
    }
    dev.destroy();
  } catch (e) { R.error = String(e && e.message || e) + ' ' + String(e && e.stack || '').slice(0, 300); }
  return R;
}
