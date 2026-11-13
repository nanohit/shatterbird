// Phase 2a, experiment R: is the impulse network of the rigid law (v3) a state or a cache?
// node gpu-probe/lab4.cjs <n> <seed> <iters> <outdir>      (SB_CACHE as lab3)
// One geometry (a pile at rest), four impulse networks on it:
//   A  exact integer equilibrium from the float solve warm-started with the stored impulses
//   A1 the same float solution plus noise of one unit per component, projected again (noise control)
//   B  exact integer equilibrium from a cold float solve (a different admissible network)
//   Z  all impulses zero (amnesia)
// Same loads from each, same substeps: control, kicks, removal of a buried sphere, slow tilt to failure, slump.
'use strict';
process.argv.splice(2, 0, '__lib');
const L = require('./lab3.cjs'), { V, E, basePile, snap, load, D } = L, fs = require('fs'), path = require('path');
const n = +process.argv[3], seed = +process.argv[4], iters = +process.argv[5], dir = process.argv[6];
const prm = { ...V.PRM3, iters }, K = V.K;
const S = basePile(n, seed), N = S.N, b0 = snap(S), { C } = E.contacts(S, V.PRM);
const fw = E.solveFloat(S, V.PRM, C, { warm: true, mu: 0.45 }), fc = E.solveFloat(S, V.PRM, C, { warm: false, mu: 0.45, sweeps: 400000 });
let rs = 12345 + seed; const rnd = () => { rs = (Math.imul(rs, 1664525) + 1013904223) | 0; return (rs >>> 8) / 16777216; };
const Pn = fw.P.slice(); for (let c = 0; c < C.length; c++) { const nn = C[c].n, o = 3 * c, pn = Pn[o] * nn[0] + Pn[o + 1] * nn[1] + Pn[o + 2] * nn[2]; if (pn > 64) for (let k = 0; k < 3; k++) Pn[o + k] += 2 * rnd() - 1; }
const nets = { A: E.project(S, V.PRM, C, fw.P, {}), A1: E.project(S, V.PRM, C, Pn, {}), B: E.project(S, V.PRM, C, fc.P, {}), Z: { Pi: new Int32Array(3 * C.length), unbalanced: -1 } };
const l1 = (x, y) => { let d = 0, s = 0; for (let k = 0; k < x.length; k++) { d += Math.abs(x[k] - y[k]); s += Math.abs(x[k]); } return +(d / s).toFixed(5); };
const out = { n, seed, N, iters, prm, contacts: C.length, float_res: { warm: fw.res, cold: fc.res, cold_sweeps: fc.sweeps }, network_L1_vs_A: { A1: l1(nets.A.Pi, nets.A1.Pi), B: l1(nets.A.Pi, nets.B.Pi) }, unbalanced: { A: nets.A.unbalanced, A1: nets.A1.unbalanced, B: nets.B.unbalanced }, states: {} };
const targets = L.surfaceSpheres(S), CHK = [12, 50, 200, 800];
const field = (T, b) => { const f = new Float32Array(3 * N); for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) f[3 * i + k] = (T.pos[4 * i + k] - b.pos[4 * i + k]) / D; return f; };
for (const name of Object.keys(nets)) {
  load(S, b0); E.write(S, C, nets[name].Pi); V.prime(S, prm); const b = snap(S), runs = {}, fields = {};
  const events = ['control', 'down:1:0', 'down:1:1', 'down:1:2', 'down:4:0', 'side:1:0', 'remove:0', 'remove:1', 'ramp', 'slump'];
  for (const ev of events) {
    const [kind, a1, a2] = ev.split(':'), p = { ...prm }; let T = S, kick = -1;
    if (kind === 'slump') T = V.make(N, S.gx + 20, S.gy, S.gz);
    load(T, b); T.ghost = -1;
    if (kind === 'down' || kind === 'side') { kick = targets[+a2]; const U = Math.round(+a1 * 5436) * p.f; if (kind === 'down') { T.vel[4 * kick] = U >> 1; T.vel[4 * kick + 1] = -U; } else T.vel[4 * kick] = U; }
    if (kind === 'remove') T.ghost = L.deepBody(T, targets[+a1]);
    V.prime(T, p); const rows = []; let quiet = -1;
    if (kind === 'ramp') {      // gravity tilts by 1/111 every 24 substeps up to tan 0.7
      for (let g = 1; g <= 78; g++) { p.gx = g; V.prime(T, p); let ch = 0; for (let t = 0; t < 24; t++) ch = V.stepInc(T, p).changed;
        let cx = 0, m01 = 0, m1 = 0; for (let i = 0; i < N; i++) { cx += (T.pos[4 * i] - b.pos[4 * i]) / D / N; const d = Math.hypot(T.pos[4 * i] - b.pos[4 * i], T.pos[4 * i + 1] - b.pos[4 * i + 1], T.pos[4 * i + 2] - b.pos[4 * i + 2]) / D; if (d > 0.1) m01++; if (d > 1) m1++; }
        rows.push({ tan: +(g / 111).toFixed(4), com_dx_D: +cx.toExponential(4), over_0p1D: m01, over_1D: m1, changed: ch }); if (g % 11 === 0) fields[ev + '@' + g] = field(T, b); }
    } else {
      const tm = kind === 'slump' ? 1920 : 800, chk = kind === 'slump' ? [240, 960, 1920] : CHK;
      for (let t = 1; t <= tm; t++) { const r = V.stepInc(T, p); if (r.changed === 0 && quiet < 0) quiet = t;
        const rec = c => { const o = kind === 'slump' ? L.slumpObs(T, b, c, {}) : L.observe(T, b, kick, c, {}); delete o.profile_D; delete o.ke; rows.push(o); };
        if (chk.includes(t)) rec(t); if (r.changed === 0) { for (const c of chk) if (c > t) rec(c); break; } }
      fields[ev] = field(T, b);
    }
    runs[ev] = { quiet_at: quiet, rows };
    const l = rows[rows.length - 1]; console.log(`i${iters} ${name} ${ev}: quiet=${quiet} ` + JSON.stringify(l).slice(0, 150));
  }
  out.states[name] = runs;
  const keys = Object.keys(fields); fs.writeFileSync(path.join(dir, `ab_${n}_${seed}_i${iters}_${name}.f32`), Buffer.concat(keys.map(k => Buffer.from(fields[k].buffer)))); out.field_keys = keys;
}
fs.writeFileSync(path.join(dir, `ab_${n}_${seed}_i${iters}.json`), JSON.stringify(out));
