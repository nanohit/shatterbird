// node gpu-probe/lab2.cjs <mode> ...   lab for specification v2 (rest2.js). Parameters: SB_<NAME>=value.
const V = require('./rest2.js'), fs = require('fs');
const mode = process.argv[2], prm = { ...V.PRM }; for (const k of Object.keys(prm)) if (process.env['SB_' + k.toUpperCase()] !== undefined) prm[k] = +process.env['SB_' + k.toUpperCase()];
const W = N => N * prm.grav * prm.f, lsum = S => V.stats(S).anchor_abs;
// settle with incremental stepping until no body changes; returns { rest_at, ... }
function settle(S, Tmax, log) { let rest = -1, work = 0; for (let t = 1; t <= Tmax; t++) { const r = V.stepInc(S, prm); work += r.work; if (log && t % log === 0) console.log(`  t=${t} changed=${r.changed}`); if (r.changed === 0) { rest = t; break; } } return { rest_at: rest, work }; }
function restState(S) { const h = V.hash(S); V.step(S, prm); const st = V.stats(S), w = W(S.N); return { anchors_pct_weight: +(st.anchor_abs / w * 100).toFixed(4), anchors_y_pct_weight: +(st.anchor_y / w * 100).toFixed(4), anchored_bodies: st.anchored_bodies, floor_pct_weight: +(st.floor_impulse_y / w * 100).toFixed(3), walls_pct_weight: +(st.wall_impulse_y / w * 100).toFixed(3), still_fixed: V.hash(S) === h, pen_max: st.pen_max, contacts: +st.contacts.toFixed(2), cmax: st.cmax, depth_D: +(st.top / 960).toFixed(1) }; }
if (mode === 'column') {
  const rows = [];
  for (const L of (process.argv[3] || '8,16,32,64').split(',').map(Number)) {
    const S = V.prime(V.lattice(1, L, 1, 1200, 960, 0, 7, 1, 4, 0, prm.f), prm), ideal = (L - 1) * 960 + 480, H = (L - 1) * 960; let sag = 0, rest = -1, T = +(process.argv[4] || 60000);
    for (let t = 1; t <= T; t++) { const r = V.stepInc(S, prm); let top = 0; for (let i = 0; i < S.N; i++) top = Math.max(top, S.pos[4 * i + 1]); sag = Math.max(sag, (ideal - top / 256) / H * 100); if (r.changed === 0) { rest = t; break; } }
    let top = 0; for (let i = 0; i < S.N; i++) top = Math.max(top, S.pos[4 * i + 1]);
    const row = { layers: L, iters: prm.iters, route: prm.route, rest_at_substep: rest, sag_transient_pct: +sag.toFixed(3), sag_final_pct: +((ideal - top / 256) / H * 100).toFixed(3), ...restState(S) }; rows.push(row); console.log(JSON.stringify(row));
  }
  if (process.env.SB_OUT) fs.writeFileSync(process.env.SB_OUT, JSON.stringify({ prm, rows }, null, 1));
}
if (mode === 'pile') {
  const n = +(process.argv[3] || 12), ny = +(process.env.SB_NY || n), S = V.prime(V.lattice(n, ny, n, 1400, 1400, 200, +(process.env.SB_SEED || 11), 1, 2, 4096, prm.f), prm), t0 = Date.now();
  const s = settle(S, +(process.argv[4] || 12000), +(process.env.SB_LOG || 0)), row = { N: S.N, route: prm.route, iters: prm.iters, eps: prm.eps, th: prm.th, ...s, work_full: S.N * (prm.iters + 2) * (s.rest_at > 0 ? s.rest_at : +(process.argv[4] || 12000)), fault: Array.from(S.faults).join('/'), ...restState(S), wall_s: +((Date.now() - t0) / 1000).toFixed(1) };
  console.log(JSON.stringify(row)); if (process.env.SB_OUT) fs.writeFileSync(process.env.SB_OUT, JSON.stringify({ prm, row }, null, 1));
}
module.exports = { V, prm, settle, restState, W, lsum };
// physics: settle a pile, then (a) kick one surface sphere, (b) move the +x wall 20 cells away (slump).
// Observables in diameters. SB_OUT gets the full record.
if (mode === 'physics') {
  const n = +(process.argv[3] || 16), seed = +(process.env.SB_SEED || 11), Tset = +(process.env.SB_TSET || 20000), K = V.K, D = 960 * 256, t0 = Date.now();
  const S = V.prime(V.lattice(n, n, n, 1400, 1400, 200, seed, 1, 2, 4096, prm.f), prm), N = S.N, out = { prm, n, seed, N };
  out.settle = { ...settle(S, Tset), ...restState(S) }; console.log('settle ' + JSON.stringify(out.settle));
  // SB_AFTER='{"route":1}': parameters switched after the pile has formed (same pile, different response)
  if (process.env.SB_AFTER) { Object.assign(prm, JSON.parse(process.env.SB_AFTER)); out.prm_after = { ...prm }; }
  const snap = () => ({ pos: S.pos.slice(), vel: S.vel.slice(), slot: S.slot.slice() }), base = snap();
  const load = (T, st) => { T.pos.set(st.pos); T.vel.set(st.vel); T.slot.set(st.slot); T.seed = null; T.evals = 0; return T; };
  const disp = (T, i) => Math.hypot(T.pos[4 * i] - base.pos[4 * i], T.pos[4 * i + 1] - base.pos[4 * i + 1], T.pos[4 * i + 2] - base.pos[4 * i + 2]) / D;
  // (a) kicks
  let kick = 0, best = -1e18; const mx = S.gx * 512 * 256, mz = S.gz * 512 * 256;
  for (let i = 0; i < N; i++) { const d = Math.hypot(S.pos[4 * i] - mx, S.pos[4 * i + 2] - mz), sc = S.pos[4 * i + 1] - 2 * d; if (sc > best) { best = sc; kick = i; } }
  out.kicks = [];
  for (const v of [0.25, 1, 4]) {
    load(S, base); const U = Math.round(v * 5436) * prm.f; S.vel[4 * kick] = U >> 1; S.vel[4 * kick + 1] = -U; V.prime(S, prm);
    const ever = new Uint8Array(N); let nev = 0, wake = 0, quiet = -1;
    for (let t = 1; t <= 4000; t++) { const r = V.stepInc(S, prm); wake += r.changed; for (let i = 0; i < N; i++) if (!S.quiet[i] && !ever[i]) { ever[i] = 1; nev++; } if (r.changed === 0) { quiet = t; break; } }
    let sum = 0, mxd = 0, n001 = 0, n01 = 0; for (let i = 0; i < N; i++) { const d = disp(S, i); sum += d; if (d > mxd) mxd = d; if (d > 0.01) n001++; if (d > 0.1) n01++; }
    const k = { v_m_s: v, quiet_after: quiet, kicked_disp_D: +disp(S, kick).toFixed(4), sum_disp_D: +sum.toFixed(3), max_disp_D: +mxd.toFixed(4), moved_over_0p01D: n001, moved_over_0p1D: n01, ever_changed: nev, wake_body_steps: wake, work: S.evals };
    out.kicks.push(k); console.log('kick ' + JSON.stringify(k));
  }
  // (b) slump
  const T = V.make(N, S.gx + 20, S.gy, S.gz); load(T, base); V.prime(T, prm); out.slump = [];
  const obs = t => { const xs = [], nb = Math.ceil(T.gx * 1024 / 1920), prof = new Float64Array(nb); let cx = 0, cy = 0, moved = 0;
    for (let i = 0; i < N; i++) { const x = T.pos[4 * i] / D, y = T.pos[4 * i + 1] / D; xs.push(x); cx += x / N; cy += y / N; if (disp(T, i) > 1) moved++; const b = Math.floor(x / 2); if (y > prof[b]) prof[b] = y; }
    xs.sort((a, b) => a - b); return { t, x95_D: +xs[Math.floor(0.95 * N)].toFixed(3), xmax_D: +xs[N - 1].toFixed(3), com_x_D: +cx.toFixed(4), com_y_D: +cy.toFixed(4), moved_over_1D: moved, profile_D: Array.from(prof).map(v => +v.toFixed(2)) }; };
  out.slump.push(obs(0)); let rest = -1, l0 = 0;
  for (let t = 1; t <= 6000; t++) { const r = V.stepInc(T, prm); if (t === 240 || t === 480 || t === 960 || t === 1920 || t === 3840) { const o = obs(t); o.changed = r.changed; o.anchors_pct_weight = +(lsum(T) / W(N) * 100).toFixed(4); out.slump.push(o); console.log('slump ' + JSON.stringify({ ...o, profile_D: undefined })); } if (r.changed === 0) { rest = t; break; } }
  const fin = obs(rest > 0 ? rest : 6000); fin.rest_at = rest; fin.fault = Array.from(T.faults).map((x, k) => x + S.faults[k]).join('/'); out.slump.push(fin); console.log('slump end ' + JSON.stringify({ ...fin, profile_D: undefined }));
  out.wall_s = (Date.now() - t0) / 1000; if (process.env.SB_OUT) fs.writeFileSync(process.env.SB_OUT, JSON.stringify(out, null, 1));
}
// depth: clouds of fixed footprint and growing height; time to exact rest and what keeps bodies changing.
// Per window of 100 substeps: bodies not quiet, of which moving (position changed), pushed out, impulses only.
if (mode === 'depth') {
  const nx = +(process.argv[3] || 12), rows = [], K = V.K;
  for (const ny of (process.argv[4] || '6,12,24,48').split(',').map(Number)) {
    const S = V.prime(V.lattice(nx, ny, nx, 1400, 1400, 200, 11, 1, 2, 4096, prm.f), prm), N = S.N, Tmax = +(process.env.SB_TMAX || 30000), t0 = Date.now(); let rest = -1, hist = [], acc = [0, 0, 0, 0], last = null;
    for (let t = 1; t <= Tmax; t++) {
      const p0 = t % 100 === 0 ? S.pos.slice() : null; const r = V.stepInc(S, prm);
      if (p0) { let mv = 0, po = 0, im = 0; for (let i = 0; i < N; i++) if (!S.quiet[i]) { if (S.corr[4 * i] | S.corr[4 * i + 1] | S.corr[4 * i + 2]) po++; else if (S.pos[4 * i] !== p0[4 * i] || S.pos[4 * i + 1] !== p0[4 * i + 1] || S.pos[4 * i + 2] !== p0[4 * i + 2]) mv++; else im++; } hist.push([t, r.changed, mv, po, im]); }
      if (r.changed === 0) { rest = t; break; }
    }
    const st = V.stats(S), row = { layers_dropped: ny, N, depth_D: +(st.top / 960).toFixed(1), rest_at: rest, route: prm.route, iters: prm.iters, anchors_pct_weight: +(st.anchor_abs / W(N) * 100).toFixed(4), pen_max: st.pen_max, contacts: +st.contacts.toFixed(2), faults: Array.from(S.faults).join('/'), wall_s: +((Date.now() - t0) / 1000).toFixed(0), history_t_changed_moving_pushed_impulse_only: hist.filter((h, i) => i % Math.max(1, Math.floor(hist.length / 12)) === 0 || i === hist.length - 1) };
    rows.push(row); console.log(JSON.stringify(row));
  }
  if (process.env.SB_OUT) fs.writeFileSync(process.env.SB_OUT, JSON.stringify({ prm, rows }, null, 1));
}
