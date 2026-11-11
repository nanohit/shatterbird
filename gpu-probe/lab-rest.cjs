// node gpu-probe/lab-rest.cjs stack|pile [args]  — solver lab for phase 1d.
const REST = require('./rest.js');
const mode = process.argv[2] || 'stack', env = k => process.env[k] !== undefined ? +process.env[k] : undefined;
const prm = { ...REST.PRM }; for (const k of Object.keys(prm)) if (env('SB_' + k.toUpperCase()) !== undefined) prm[k] = env('SB_' + k.toUpperCase());
if (mode === 'stack') {
  const L = +(process.argv[3] || 64), steps = +(process.argv[4] || 9600), jit = +(process.argv[5] || 20);
  const S = REST.lattice(4, L, 4, 1200, 960, jit, 7, 1, 4), ideal = (L - 1) * 960 + 480, H = (L - 1) * 960;
  let sagmax = 0, rest_at = -1, run0 = 0;
  for (let t = 1; t <= steps; t++) {
    const r = REST.step(S, prm);
    if (r.changed === 0) { if (!run0) rest_at = t; run0++; } else { run0 = 0; rest_at = -1; }
    if (t % 8 === 0 || t === steps) { let top = 0; for (let i = 0; i < S.N; i++) top = Math.max(top, S.pos[4 * i + 1]); const sag = (ideal - top / 256) / H; if (sag > sagmax) sagmax = sag; }
    if (t % (steps / 12 | 0) === 0 || t === steps) { const s = REST.stats(S); console.log(`t=${t} changed=${r.changed}/${S.N} over=${r.over} vmax=${s.vmax} pen=${s.pen_max} sag=${((ideal - s.top) / H * 100).toFixed(3)}% Pmax=${s.Pmax} c=${s.contacts.toFixed(2)}`); }
  }
  console.log(JSON.stringify({ L, prm, sag_transient_pct: +(sagmax * 100).toFixed(3), rest_since_step: rest_at, rest_run: run0 }));
}
if (mode === 'pile') {
  const n = +(process.argv[3] || 28), steps = +(process.argv[4] || 4800), ev = +(process.argv[5] || 480);
  const ny = +(process.env.SB_NY || n);
  const S = REST.lattice(n, ny, n, 1400, 1400, 200, 11, 1, 2, 4096, prm.f);
  let t0 = Date.now(), rest = -1, hist = [];
  for (let t = 1; t <= steps; t++) {
    const r = REST.step(S, prm);
    if (r.changed === 0) { if (rest < 0) rest = t; } else rest = -1;
    if (t % ev === 0 || t === steps) { const s = REST.stats(S); let nv = 0, mv = 0; for (let i = 0; i < S.N; i++) { if (S.vel[4 * i] | S.vel[4 * i + 1] | S.vel[4 * i + 2]) nv++; }
      hist.push({ t, changed: r.changed, vnz: nv, vmax: s.vmax, pen: s.pen_max, c: +s.contacts.toFixed(2), cmax: s.cmax, Pmax: s.Pmax, over: r.over });
      console.log(`t=${t} changed=${r.changed} v!=0:${nv} vmax=${s.vmax} pen=${s.pen_max} c=${s.contacts.toFixed(2)} cmax=${s.cmax} Pmax=${s.Pmax} over=${r.over} top=${(s.top / 960).toFixed(1)}D sink=${((S.ledger[3] - (S.l0 || 0)) / ev / (S.N * prm.grav * prm.f) * 100).toFixed(3)}%w ${((Date.now() - t0) / t).toFixed(1)}ms/step`); S.l0 = S.ledger[3]; }
    if (rest > 0 && t - rest > 1000) break;
  }
  console.log(JSON.stringify({ N: S.N, prm, rest_since_step: rest, hash: REST.fnv(S) }));
  if (process.env.SB_OUT) require('fs').writeFileSync(process.env.SB_OUT, JSON.stringify({ N: S.N, prm, rest_since_step: rest, hash: REST.fnv(S), hist }, null, 1));
}
// wake: settle a pile to an exact fixed point with change-driven stepping, then kick one surface
// sphere and record how far and how long the change spreads (D2, D3).
if (mode === 'wake') {
  const n = +(process.argv[3] || 28), ny = +(process.env.SB_NY || n), Tmax = +(process.argv[4] || 20000), fs = require('fs');
  const S = REST.lattice(n, ny, n, 1400, 1400, 200, 11, 1, 2, 4096, prm.f), N = S.N, t0 = Date.now();
  let rest = -1, hist = [], l0 = 0;
  for (let t = 1; t <= Tmax; t++) {
    const r = REST.stepActive(S, prm);
    if (t % 480 === 0) { const st = REST.stats(S); hist.push({ t, changed: r.changed, evaluated: r.evaluated, pen: st.pen_max, sink_pct_w: +((S.ledger[3] - l0) / 480 / (N * prm.grav * prm.f) * 100).toFixed(3) }); l0 = S.ledger[3]; console.log(`settle t=${t} changed=${r.changed} evaluated=${r.evaluated} pen=${st.pen_max} top=${(st.top / 960).toFixed(1)}D c=${st.contacts.toFixed(2)} over=${r.over} ${((Date.now() - t0) / 1000).toFixed(0)}s`); }
    if (r.changed === 0) { rest = t; break; }
  }
  const st = REST.stats(S), settle = { N, rest_at_step: rest, evals: S.evals, evals_full: N * S.t, pen_max: st.pen_max, depth_D: +(st.top / 960).toFixed(1), contacts: +st.contacts.toFixed(2), cmax: st.cmax, hash: REST.fnvFull(S) };
  let f0 = S.ledger[3]; REST.step(S, prm); settle.sink_pct_weight_at_rest = +((S.ledger[3] - f0) / (N * prm.grav * prm.f) * 100).toFixed(3); settle.full_step_still_fixed = REST.fnvFull(S) === settle.hash;
  console.log(JSON.stringify(settle));
  const snap = {}; for (const k of ['pos', 'vel', 'sid', 'sP', 'sn', 'nid', 'cn']) snap[k] = S[k].slice(); snap.Pl = S.Pl.map(a => a.slice()); snap.rl = S.rl.map(a => a.slice());
  // surface sphere nearest the middle of the box
  let kick = 0, best = -1e18; const mx = S.gx * 512 * 256, mz = S.gz * 512 * 256;
  for (let i = 0; i < N; i++) { const d = Math.hypot(S.pos[4 * i] - mx, S.pos[4 * i + 2] - mz), sc = S.pos[4 * i + 1] - 2 * d; if (sc > best) { best = sc; kick = i; } }
  const kicks = [];
  for (const V of (process.env.SB_KICKS || '2718,5436,10872,21744').split(',').map(Number)) {
    for (const k of ['pos', 'vel', 'sid', 'sP', 'sn', 'nid', 'cn']) S[k].set(snap[k]); S.Pl.forEach((a, q) => a.set(snap.Pl[q])); S.rl.forEach((a, q) => a.set(snap.rl[q])); S.evals = 0;
    S.vel[4 * kick] = (V >> 1) * prm.f; S.vel[4 * kick + 1] = -V * prm.f; S.seed = [(S.pos[4 * kick] >> 18) + S.gx * ((S.pos[4 * kick + 1] >> 18) + S.gy * (S.pos[4 * kick + 2] >> 18))];
    const ever = new Uint8Array(N); let nev = 0, sumc = 0, peak = 0, quiet = -1, rmax = 0, t1 = Date.now(), moved = 0;
    const kx = snap.pos[4 * kick], ky = snap.pos[4 * kick + 1], kz = snap.pos[4 * kick + 2];
    for (let t = 1; t <= Tmax; t++) {
      const r = REST.stepActive(S, prm); sumc += r.changed; if (r.changed > peak) peak = r.changed;
      for (let i = 0; i < N; i++) if (S.chg[i] && !ever[i]) { ever[i] = 1; nev++; const d = Math.hypot(snap.pos[4 * i] - kx, snap.pos[4 * i + 1] - ky, snap.pos[4 * i + 2] - kz) / 256 / 960; if (d > rmax) rmax = d; }
      if (r.changed === 0) { quiet = t; break; }
    }
    for (let i = 0; i < N; i++) if (Math.hypot(S.pos[4 * i] - snap.pos[4 * i], S.pos[4 * i + 1] - snap.pos[4 * i + 1], S.pos[4 * i + 2] - snap.pos[4 * i + 2]) > 256 * 96) moved++;
    const k = { kick_m_per_s: +(V / 5436).toFixed(2), quiet_after_steps: quiet, ever_changed: nev, ever_changed_pct: +(nev / N * 100).toFixed(2), wake_volume_body_steps: sumc, peak_changed: peak, reach_D: +rmax.toFixed(1), moved_over_0p1D: moved, evaluated_body_steps: S.evals, full_body_steps: N * (quiet > 0 ? quiet : Tmax), wall_s: +((Date.now() - t1) / 1000).toFixed(1) };
    kicks.push(k); console.log(JSON.stringify(k));
  }
  if (process.env.SB_OUT) fs.writeFileSync(process.env.SB_OUT, JSON.stringify({ prm, settle, settle_history: hist, kicks }, null, 1));
}
