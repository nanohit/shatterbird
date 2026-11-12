// node gpu-probe/lab3.cjs <mode> ...   phase 1g lab: equilibrium of impulses, responses at equal horizons.
// Parameters of the solver: SB_<NAME>=value (as lab2). SB_CACHE: directory for settled base piles.
'use strict';
const V = require('./rest2.js'), E = require('./equil.cjs'), fs = require('fs'), path = require('path');
const mode = process.argv[2], base = { ...V.PRM }; for (const k of Object.keys(base)) if (process.env['SB_' + k.toUpperCase()] !== undefined) base[k] = +process.env['SB_' + k.toUpperCase()];
const CACHE = process.env.SB_CACHE || path.join(__dirname, 'out'), D = 960 * 256, K = V.K;
const snap = S => ({ pos: S.pos.slice(), vel: S.vel.slice(), slot: S.slot.slice() });
const load = (T, st) => { T.pos.set(st.pos); T.vel.set(st.vel); T.slot.set(st.slot); T.seed = null; T.evals = 0; T.faults.fill(0); return T; };
// A pile at exact rest under the default law (4 iterations, eps 32): the common geometry of every comparison.
function basePile(n, seed, prm = V.PRM) {
  const f = path.join(CACHE, `base_${n}_${seed}.bin`), S = V.prime(V.lattice(n, n, n, 1400, 1400, 200, seed, 1, 2, 4096, prm.f), prm);
  if (fs.existsSync(f)) { const b = fs.readFileSync(f), a = new Int32Array(b.buffer, b.byteOffset, b.length / 4), N = S.N; S.pos.set(a.subarray(0, 4 * N)); S.vel.set(a.subarray(4 * N, 8 * N)); S.slot.set(a.subarray(8 * N)); S.seed = null; return S; }
  let rest = -1; for (let t = 1; t <= 20000; t++) if (V.stepInc(S, prm).changed === 0) { rest = t; break; }
  if (rest < 0) throw new Error('base pile did not rest');
  fs.mkdirSync(CACHE, { recursive: true }); const a = new Int32Array(8 * S.N + S.slot.length); a.set(S.pos, 0); a.set(S.vel, 4 * S.N); a.set(S.slot, 8 * S.N); fs.writeFileSync(f, Buffer.from(a.buffer)); return S;
}
// Exact integer equilibrium on the geometry of S (in place). Returns the record of the solve.
function equilibrate(S, prm, opt = {}) {
  const t0 = Date.now(), { C, asym } = E.contacts(S, prm), f = E.solveFloat(S, prm, C, opt), p = E.project(S, prm, C, f.P, opt);
  const rec = { contacts: C.length, touching: C.filter(c => c.touch).length, asymmetric: asym, warm: !!opt.warm, mu: opt.mu ?? 0.45, sweeps: f.sweeps, float_imbalance_max: f.res, hist: f.hist, float_cone: E.coneStats(prm, C, f.P), ...p, Pi: undefined, int_cone: E.coneStats(prm, C, p.Pi), ms: Date.now() - t0 };
  E.write(S, C, p.Pi); V.prime(S, prm); rec.C = C; rec.Pf = f.P; rec.Pint = p.Pi; return rec;
}
const pub = r => ({ ...r, C: undefined, Pf: undefined, Pint: undefined });
module.exports = { V, E, basePile, equilibrate, snap, load, base, D, pub };

if (mode === 'equil') {
  const n = +(process.argv[3] || 12), seed = +(process.env.SB_SEED || 11), S = basePile(n, seed), N = S.N, W = N * base.grav * base.f, out = { n, seed, N };
  const st0 = V.stats(S); out.base = { anchors_pct_weight: +(st0.anchor_abs / W * 100).toFixed(4), anchored_bodies: st0.anchored_bodies, floor_pct: +(st0.floor_impulse_y / W * 100).toFixed(3), walls_pct: +(st0.wall_impulse_y / W * 100).toFixed(3), contacts: +st0.contacts.toFixed(2), pen_max: st0.pen_max };
  console.log('base ' + JSON.stringify(out.base));
  const b0 = snap(S); out.solves = [];
  let keep = null;
  for (const o of (process.env.SB_COLD === '0' ? [{ warm: true, mu: 0.45 }, { warm: true, mu: 0.5 }] : [{ warm: true, mu: 0.45 }, { warm: false, mu: 0.45 }, { warm: true, mu: 0.5 }, { warm: false, mu: 0.3 }])) {
    load(S, b0); const r = equilibrate(S, base, { ...o, sweeps: +(process.env.SB_SWEEPS || 200000), minKeep: +(process.env.SB_MINKEEP || 0.5) }); r.min_keep = +(process.env.SB_MINKEEP || 0.5);
    // is it a fixed point of the integer law, with no rest quantum at all?
    const chk = {}; for (const [name, q] of [['iters4_eps0', { iters: 4, eps: 0 }], ['iters32_eps0', { iters: 32, eps: 0 }], ['iters4_eps32', { iters: 4, eps: 32 }], ['v3_iters4', { ...V.PRM3 }], ['v3_iters4_fine_cone_eps0', { eps: 0, cone: 1 }]]) { const T = V.prime(load(V.make(N, S.gx, S.gy, S.gz), snap(S)), { ...base, ...q }); const h = V.hash(T); let ch = 0; for (let t = 0; t < 3; t++) ch += V.step(T, { ...base, ...q }).changed; chk[name] = { changed_in_3_steps: ch, hash_same: V.hash(T) === h }; }
    r.fixed_point = chk; if (!keep) keep = { C: r.C, Pf: r.Pf }; else if (o.mu === 0.45) { let d = 0, s = 0; for (let k = 0; k < r.Pf.length; k++) { d += Math.abs(r.Pf[k] - keep.Pf[k]); s += Math.abs(keep.Pf[k]); } r.diff_to_warm_L1_rel = +(d / s).toFixed(4); }
    out.solves.push(pub(r)); console.log(JSON.stringify(pub(r)));
  }
  if (process.env.SB_OUT) fs.writeFileSync(process.env.SB_OUT, JSON.stringify(out, null, 1));
}

// ---- responses at equal horizons from one common exact equilibrium ----
// Every variant starts from the same positions, zero velocities and the same exactly balanced impulses,
// so its control run (no kick) is checked to stay put; observables are taken at the same substeps.
const CHK = [6, 12, 25, 50, 100, 200, 400, 800, 1600, 3200];
// Exposed bodies (no partner above within 60 degrees) nearest to six target points: the centre and five points on a ring.
function surfaceSpheres(S) {
  const N = S.N, ex = new Uint8Array(N).fill(1), mx = S.gx * 512 * 256, mz = S.gz * 512 * 256, rad = 0.2 * Math.min(S.gx, S.gz) * 1024 * 256, out = [];
  for (let i = 0; i < N; i++) { const n = S.vel[4 * i + 3] & 31; for (let s = 0; s < n; s++) { const id = S.slot[4 * (i * K + s) + 3] >>> 0; if (id >= V.WALL) continue; V.geom(S, i, id); if (-V.geo.ey * 2 >= V.geo.len) { ex[i] = 0; break; } } }
  for (let q = 0; q < 6; q++) { const tx = mx + (q ? rad * Math.cos(q * 2 * Math.PI / 5) : 0), tz = mz + (q ? rad * Math.sin(q * 2 * Math.PI / 5) : 0); let best = -1e18, kick = -1;
    for (let i = 0; i < N; i++) if (ex[i] && !out.includes(i)) { const sc = S.pos[4 * i + 1] - 2 * Math.hypot(S.pos[4 * i] - tx, S.pos[4 * i + 2] - tz); if (sc > best) { best = sc; kick = i; } } out.push(kick); }
  return out;
}
function anchorAbs(S) { let a = 0; for (let i = 0; i < S.N; i++) { const aw = S.vel[4 * i + 3] >> 8; if (aw) for (let k = 0; k < 3; k++) a += Math.abs(((aw >> (7 * k)) & 127) << 25 >> 25); } return a; }
function observe(S, b, kick, t, extra) {
  const N = S.N; let sum = 0, mxd = 0, n001 = 0, n01 = 0, ke = 0, cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < N; i++) { const dx = (S.pos[4 * i] - b.pos[4 * i]) / D, dy = (S.pos[4 * i + 1] - b.pos[4 * i + 1]) / D, dz = (S.pos[4 * i + 2] - b.pos[4 * i + 2]) / D, d = Math.hypot(dx, dy, dz); sum += d; cx += dx; cy += dy; cz += dz; if (d > mxd) mxd = d; if (d > 0.01) n001++; if (d > 0.1) n01++; ke += S.vel[4 * i] ** 2 + S.vel[4 * i + 1] ** 2 + S.vel[4 * i + 2] ** 2; }
  const kd = i => [0, 1, 2].map(k => +((S.pos[4 * i + k] - b.pos[4 * i + k]) / D).toFixed(5));
  return { t, kicked_D: kick >= 0 ? kd(kick) : null, kicked_abs_D: kick >= 0 ? +Math.hypot(...kd(kick)).toFixed(5) : null, sum_D: +sum.toFixed(4), max_D: +mxd.toFixed(5), over_0p01D: n001, over_0p1D: n01, com_shift_D: [cx / N, cy / N, cz / N].map(v => +v.toExponential(3)), ke, ...extra };
}
function field(S, b) { const f = new Float32Array(3 * S.N); for (let i = 0; i < S.N; i++) for (let k = 0; k < 3; k++) f[3 * i + k] = (S.pos[4 * i + k] - b.pos[4 * i + k]) / D; return f; }
if (mode === 'resp') {
  // node lab3.cjs resp <n> <name> <iters> <eps> [Tmax]   (SB_SEED, SB_OUT directory, SB_KICKS='down:1,down:4,side:1')
  const n = +process.argv[3], name = process.argv[4], prm = { ...base, iters: +process.argv[5], eps: +process.argv[6] }, Tmax = +(process.argv[7] || 1600), seed = +(process.env.SB_SEED || 11), t0 = Date.now();
  const S = basePile(n, seed), N = S.N, eq = equilibrate(S, base, { warm: true, mu: 0.45 }), b = snap(S), targets = surfaceSpheres(S), dir = process.env.SB_OUT || CACHE;
  const out = { name, n, seed, N, prm, kick_bodies: targets, equilibrium: { sweeps: eq.sweeps, float_imbalance_max: eq.float_imbalance_max, carried_max: eq.carried_max, unbalanced: eq.unbalanced }, runs: [] };
  const run = (label, v, d, kick = targets[0]) => {
    load(S, b); if (v) { const U = Math.round(v * 5436) * prm.f; if (d === 'down') { S.vel[4 * kick] = U >> 1; S.vel[4 * kick + 1] = -U; } else if (d === 'side') { S.vel[4 * kick] = U; } else if (d === 'up') { S.vel[4 * kick + 1] = U; } } V.prime(S, prm);
    const rows = [], fields = []; let work = 0, anch = 0, quietAt = -1, ke0 = 0; for (let k = 0; k < 3; k++) ke0 += S.vel[4 * kick + k] ** 2;
    for (let t = 1; t <= Tmax; t++) { const r = V.stepInc(S, prm); work += r.work; anch += anchorAbs(S); if (r.changed === 0 && quietAt < 0) quietAt = t;
      if (CHK.includes(t)) { const o = observe(S, b, v ? kick : -1, t, { changed: r.changed, work, anchor_impulse_cum: anch }); o.ke_frac = ke0 ? +(o.ke / ke0).toExponential(3) : o.ke; delete o.ke; rows.push(o); fields.push(field(S, b)); }
      if (r.changed === 0 && !v) { for (const c of CHK) if (c > t && c <= Tmax) { rows.push(observe(S, b, -1, c, { changed: 0, work, anchor_impulse_cum: anch })); fields.push(field(S, b)); } break; } }
    const f = Buffer.concat(fields.map(a => Buffer.from(a.buffer))); fs.writeFileSync(path.join(dir, `resp_${n}_${seed}_${name}_${label}.f32`), f);
    out.runs.push({ label, v_m_s: v, dir: d, quiet_at: quietAt, faults: Array.from(S.faults), rows }); const l = rows[rows.length - 1]; console.log(`${name} ${label}: quiet_at=${quietAt} t=${l.t} kicked=${l.kicked_abs_D} sum=${l.sum_D} max=${l.max_D} n001=${l.over_0p01D} changed=${l.changed}`);
  };
  run('control', 0, '');
  for (const k of (process.env.SB_KICKS || 'down:0.25,down:1,down:4,side:1').split(',')) { const [d, v] = k.split(':'); for (let q = 0; q < +(process.env.SB_NK || 6); q++) run(`${d}${v}_k${q}`, +v, d, targets[q]); }
  out.wall_s = (Date.now() - t0) / 1000; fs.writeFileSync(path.join(dir, `resp_${n}_${seed}_${name}.json`), JSON.stringify(out, null, 1));
}

// ---- resp2: any variant (JSON of parameters), any event, same initial state, same substeps ----
// node lab3.cjs resp2 <n> <name> '<json prm>' [Tmax]   SB_SEED, SB_OUT, SB_EQ=float|int,
//   SB_EVENTS='down:1:0,side:1:3,slump,tilt:0.3,remove:2'  (kick dir:speed:target, slump, tilt:gx/grav, remove:target)
// Exact integer equilibrium reached by the handover law itself from the rest state of the default law.
// parameters as JSON or as 'eps=0+hand=32' ('-' for none)
function parsePrm(a) { if (!a || a === '-') return {}; if (a[0] === '{') return JSON.parse(a); const o = {}; for (const kv of a.split('+')) { const [k, v] = kv.split('='); o[k] = +v; } return o; }
function equilibrateInt(S) { const p = { ...V.PRM, eps: 0, hand: 32 }; V.prime(S, p); for (let t = 1; t <= 4000; t++) if (V.stepInc(S, p).changed === 0) return t; throw new Error('handover did not reach an exact balance'); }
function slumpObs(T, b, t, extra) { const N = T.N, xs = [], nb = Math.ceil(T.gx * 1024 / 1920), prof = new Float64Array(nb); let cx = 0, cy = 0, moved = 0, m01 = 0;
  for (let i = 0; i < N; i++) { const x = T.pos[4 * i] / D, y = T.pos[4 * i + 1] / D; xs.push(x); cx += (T.pos[4 * i] - b.pos[4 * i]) / D / N; cy += (T.pos[4 * i + 1] - b.pos[4 * i + 1]) / D / N; const d = Math.hypot(T.pos[4 * i] - b.pos[4 * i], T.pos[4 * i + 1] - b.pos[4 * i + 1], T.pos[4 * i + 2] - b.pos[4 * i + 2]) / D; if (d > 1) moved++; if (d > 0.1) m01++; const q = Math.floor(x / 2); if (y > prof[q]) prof[q] = y; }
  xs.sort((a, c) => a - c); return { t, x95_D: +xs[Math.floor(0.95 * N)].toFixed(3), xmax_D: +xs[N - 1].toFixed(3), com_dx_D: +cx.toFixed(5), com_dy_D: +cy.toFixed(5), moved_over_1D: moved, moved_over_0p1D: m01, profile_D: Array.from(prof).map(v => +v.toFixed(2)), ...extra }; }
function deepBody(S, target) { let top = 0; for (let i = 0; i < S.N; i++) top = Math.max(top, S.pos[4 * i + 1]); let best = 1e18, q = -1; for (let i = 0; i < S.N; i++) { const d = Math.hypot(S.pos[4 * i] - S.pos[4 * target], S.pos[4 * i + 2] - S.pos[4 * target + 2]) + 2 * Math.abs(S.pos[4 * i + 1] - top / 3); if (d < best) { best = d; q = i; } } return q; }
if (mode === 'resp2') {
  const n = +process.argv[3], name = process.argv[4], prm = { ...base, ...parsePrm(process.argv[5]) }, Tmax = +(process.argv[6] || 800), seed = +(process.env.SB_SEED || 11), t0 = Date.now(), dir = process.env.SB_OUT || CACHE;
  const S = basePile(n, seed), N = S.N; let eqrec;
  if ((process.env.SB_EQ || 'float') === 'int') eqrec = { kind: 'int', substeps: equilibrateInt(S) }; else { const eq = equilibrate(S, base, { warm: true, mu: 0.45 }); eqrec = { kind: 'float', sweeps: eq.sweeps, float_imbalance_max: eq.float_imbalance_max, unbalanced: eq.unbalanced }; }
  const b = snap(S), targets = surfaceSpheres(S), out = { name, n, seed, N, prm, kick_bodies: targets, equilibrium: eqrec, runs: [] };
  const anchorSum = T => anchorAbs(T);
  for (const ev of (process.env.SB_EVENTS || 'control').split(',')) {
    const [kind, a1, a2] = ev.split(':'), p = { ...prm }; let T = S, chk = CHK, ghost = -1, kick = -1, v = 0;
    if (kind === 'slump') { T = V.make(N, S.gx + 20, S.gy, S.gz); chk = [60, 120, 240, 480, 960, 1920, 3840]; }
    load(T, b); T.ghost = -1; T.lazy = 0;
    if (kind === 'down' || kind === 'side' || kind === 'up') { v = +a1; kick = targets[+a2 || 0]; const U = Math.round(v * 5436) * p.f; if (kind === 'down') { T.vel[4 * kick] = U >> 1; T.vel[4 * kick + 1] = -U; } else if (kind === 'side') T.vel[4 * kick] = U; else T.vel[4 * kick + 1] = U; }
    if (kind === 'tilt') p.gx = Math.round(+a1 * p.grav);
    if (kind === 'remove') { ghost = deepBody(T, targets[+a1 || 0]); T.ghost = ghost; }
    V.prime(T, p); const rows = [], fields = []; let work = 0, anch = 0, quietAt = -1, ke0 = 0; if (kick >= 0) for (let k = 0; k < 3; k++) ke0 += T.vel[4 * kick + k] ** 2;
    const tm = kind === 'slump' ? Math.max(Tmax, 3840) : Tmax;
    for (let t = 1; t <= tm; t++) { const r = V.stepInc(T, p); work += r.work; anch += anchorSum(T); if (r.changed === 0 && quietAt < 0) quietAt = t;
      const rec = c => { const ex = { changed: r.changed, work, anchor_impulse_cum: anch, lazy_work: T.lazy || 0 }; if (kind === 'slump') rows.push(slumpObs(T, b, c, ex)); else { const o = observe(T, b, kick, c, ex); o.ke_frac = ke0 ? +(o.ke / ke0).toExponential(3) : o.ke; delete o.ke; if (kind === 'tilt') { const so = slumpObs(T, b, c, {}); o.moved_over_1D = so.moved_over_1D; } rows.push(o); } fields.push(field(T, b)); };
      if (chk.includes(t)) rec(t);
      if (r.changed === 0) { for (const c of chk) if (c > t && c <= tm) rec(c); break; } }
    fs.writeFileSync(path.join(dir, `r2_${n}_${seed}_${name}_${ev.replace(/:/g, '_')}.f32`), Buffer.concat(fields.map(a => Buffer.from(a.buffer))));
    const l = rows[rows.length - 1]; out.runs.push({ event: ev, kind, quiet_at: quietAt, ghost, kick, faults: Array.from(T.faults), rows });
    console.log(`${name} ${ev}: quiet_at=${quietAt} t=${l.t} ` + (kind === 'slump' ? `x95=${l.x95_D} com_dx=${l.com_dx_D} moved1D=${l.moved_over_1D}` : `kicked=${l.kicked_abs_D} sum=${l.sum_D} max=${l.max_D} n01=${l.over_0p1D}`) + ` changed=${l.changed}`);
  }
  out.wall_s = (Date.now() - t0) / 1000; fs.writeFileSync(path.join(dir, `r2_${n}_${seed}_${name}.json`), JSON.stringify(out, null, 1));
}

// ---- analytic checks of the contact law: one body or a short stack on the floor, gravity tilted ----
// node lab3.cjs analytic '<json prm>'
if (mode === 'analytic') {
  const prm = { ...base, ...JSON.parse(process.argv[3] || '{}') }, F = prm.f, G = prm.grav, mus = prm.mu8 / 8, muk = prm.muk8 / 8, out = { prm, incline: [], stack: [], stop: [] };
  const mk = layers => { const S = V.prime(V.lattice(1, layers, 1, 1200, 960, 0, 7, 2, 4, 0, F, 40), prm); for (let t = 0; t < 400; t++) if (V.stepInc(S, prm).changed === 0) break; return S; };
  for (const layers of [1, 3]) for (const ratio of [0.3, 0.4, 0.45, 0.48, 0.52, 0.55, 0.6, 0.75, 1]) {
    const S = mk(layers), x0 = S.pos.slice(), T = 240, p = { ...prm, gx: Math.round(ratio * G) }, real = p.gx / G; V.prime(S, p); let rest = -1; for (let t = 1; t <= T; t++) if (V.stepInc(S, p).changed === 0) { rest = t; break; }
    const dx = []; for (let i = 0; i < S.N; i++) dx.push(S.pos[4 * i] - x0[4 * i]); const ideal = real > mus ? 0.5 * (p.gx - muk * G) * T * T : 0, spread = Math.max(...dx) - Math.min(...dx);
    const row = { layers, tan_theta: +real.toFixed(4), moved_fine_units: dx[0], ideal_fine_units: Math.round(ideal), ratio_to_ideal: ideal ? +(dx[0] / ideal).toFixed(4) : null, stack_spread_fine_units: spread, rest_at: rest, verdict: ideal ? (Math.abs(dx[0] / ideal - 1) <= 0.05 ? 'ok' : 'off') : (dx[0] === 0 ? 'ok' : 'moved') };
    (layers === 1 ? out.incline : out.stack).push(row); console.log(JSON.stringify(row)); }
  for (const Vf of [8, 64, 512, 2048, 4096, 8192]) { const S = mk(1), x0 = S.pos[0]; S.vel[0] = Vf * F; V.prime(S, prm); let rest = -1; for (let t = 1; t <= 4000; t++) if (V.stepInc(S, prm).changed === 0) { rest = t; break; }
    const d = S.pos[0] - x0, ideal = Vf * Vf / (2 * muk * G), row = { v0_fine_units_per_substep: Vf, v0_m_s: +(Vf * F / (5436 * F)).toFixed(3), travelled_D: +(d / D).toFixed(4), ideal_D: +(ideal / D).toFixed(4), ratio_to_ideal: +(d / ideal).toFixed(4), rest_at: rest }; out.stop.push(row); console.log(JSON.stringify(row)); }
  if (process.env.SB_OUT) fs.writeFileSync(process.env.SB_OUT, JSON.stringify(out, null, 1));
}

// ---- contact capacity: how much overlap does it take to exceed K = 16 contacts? ----
// A settled pile is compressed uniformly towards its centre (all distances scaled by s); for each s the largest pair
// overlap and the largest number of partners within the search margin. Theory (Tammes problem): n points on a sphere
// are separated by at most 57.14 degrees for n = 13 and 51.09 degrees for n = 17, so 13 (17) partners within
// (1 + margin) D force two of them to overlap by at least 2.8 % (12.3 %) of the diameter.
if (mode === 'overflow') {
  const n = +(process.argv[3] || 12), S = basePile(n, +(process.env.SB_SEED || 11)), N = S.N, lim = 960 + base.margin, rows = []; let first = null;
  const px = i => [S.pos[4 * i] / 256, S.pos[4 * i + 1] / 256, S.pos[4 * i + 2] / 256]; let c = [0, 0, 0]; for (let i = 0; i < N; i++) { const p = px(i); for (let k = 0; k < 3; k++) c[k] += p[k] / N; }
  const d2 = []; for (let i = 0; i < N; i++) { const a = px(i), l = []; for (let j = 0; j < N; j++) if (j !== i) { const b = px(j), q = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); if (q < 3 * 960) l.push(q); } l.sort((x, y) => x - y); d2.push(l); }
  for (let s = 1; s >= 0.5; s -= 0.01) { let mx = 0, dmin = 1e9; for (let i = 0; i < N; i++) { const l = d2[i]; let k = 0; while (k < l.length && l[k] * s < lim) k++; if (k > mx) mx = k; if (l[0] * s < dmin) dmin = l[0] * s; }
    const row = { scale: +s.toFixed(2), max_partners_in_margin: mx, max_pair_overlap_pct_D: +((960 - dmin) / 9.6).toFixed(1) }; rows.push(row); if (mx > K && !first) first = row; if (mx > K + 4) break; }
  const out = { n, N, K, margin_units: base.margin, tammes_bound_pct_D: { partners_13: 2.8, partners_17: 12.3 }, first_overflow: first, rows: rows.filter((r, i) => i % 5 === 0 || r === first) };
  console.log(JSON.stringify(out)); if (process.env.SB_OUT) fs.writeFileSync(process.env.SB_OUT, JSON.stringify(out, null, 1));
}

// ---- balance: from the rest state of the default law (anchors) to an exact balance under another law ----
// node lab3.cjs balance <n> '<json prm>'   (three seeds)
if (mode === 'balance') {
  const n = +process.argv[3], prm = { ...V.PRM3, ...parsePrm(process.argv[4]) }, rows = [];
  for (const seed of [11, 12, 13]) { const S = basePile(n, seed), b = snap(S), W = S.N * prm.grav * prm.f, a0 = V.stats(S); V.prime(S, prm); let rest = -1, work = 0;
    for (let t = 1; t <= 4000; t++) { const r = V.stepInc(S, prm); work += r.work; if (r.changed === 0) { rest = t; break; } }
    let sum = 0, mx = 0; for (let i = 0; i < S.N; i++) { const d = Math.hypot(S.pos[4 * i] - b.pos[4 * i], S.pos[4 * i + 1] - b.pos[4 * i + 1], S.pos[4 * i + 2] - b.pos[4 * i + 2]) / D; sum += d; if (d > mx) mx = d; }
    const st = V.stats(S), row = { n, seed, N: S.N, anchors_before_pct_weight: +(a0.anchor_abs / W * 100).toFixed(4), anchored_before: a0.anchored_bodies, exact_balance_after_substeps: rest, work_in_full_substeps_of_the_default_law: +(work / (S.N * 6)).toFixed(1), moved_max_D: +mx.toExponential(2), moved_sum_D: +sum.toExponential(2),
      floor_plus_walls_pct_weight: +((st.floor_impulse_y + st.wall_impulse_y) / W * 100).toFixed(6), anchors_after_pct_weight: +(st.anchor_abs / W * 100).toFixed(5), anchored_after: st.anchored_bodies };
    rows.push(row); console.log(JSON.stringify(row)); }
  if (process.env.SB_OUT) fs.writeFileSync(process.env.SB_OUT, JSON.stringify({ prm, rows }, null, 1));
}
// ---- stack: a column of L spheres at rest, gravity tilted suddenly; hold below the static limit, slide as a block above ----
// node lab3.cjs stack '<json prm>' [L1,L2,..] [tan1,tan2,..]
if (mode === 'stack') {
  const prm = { ...base, ...parsePrm(process.argv[3]) }, Ls = (process.argv[4] || '3,8,16').split(',').map(Number), rs = (process.argv[5] || '0.3,0.45,0.6').split(',').map(Number), T = 480, rows = [];
  for (const L of Ls) for (const ratio of rs) { const S = V.prime(V.lattice(1, L, 1, 1200, 960, 0, 7, 2, 4, 0, 32, 60), prm); let ok = 0; for (let t = 0; t < 60000; t++) if (V.stepInc(S, prm).changed === 0) { ok = 1; break; }
    const x0 = S.pos.slice(), p = { ...prm, gx: Math.round(ratio * prm.grav) }; V.prime(S, p); let rest = -1; for (let t = 1; t <= T; t++) if (V.stepInc(S, p).changed === 0) { rest = t; break; }
    let mn = 1e18, mx = -1e18; for (let i = 0; i < S.N; i++) { const d = S.pos[4 * i] - x0[4 * i]; mn = Math.min(mn, d); mx = Math.max(mx, d); }
    const tan = p.gx / prm.grav, ideal = tan > prm.mu8 / 8 ? 0.5 * (p.gx - prm.muk8 / 8 * prm.grav) * T * T : 0;
    const row = { layers: L, tan_theta: +tan.toFixed(3), settled_before_tilt: !!ok, lowest_body_moved_D: +(mn / D).toFixed(4), ideal_D: +(ideal / D).toFixed(3), spread_of_the_stack_D: +((mx - mn) / D).toFixed(4), rest_at: rest,
      verdict: ideal ? ((mx - mn) / D < 0.05 && Math.abs(mn / ideal - 1) < 0.05 ? 'slides as a block' : 'falls apart') : ((mx - mn) / D < 0.05 && Math.abs(mn / D) < 0.05 ? 'holds' : 'shears') };
    rows.push(row); console.log(JSON.stringify(row)); }
  if (process.env.SB_OUT) fs.writeFileSync(process.env.SB_OUT, JSON.stringify({ prm, rows }, null, 1));
}
