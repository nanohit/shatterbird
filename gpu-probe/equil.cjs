// Shatterbird phase 1g: static equilibrium of the contact impulses on a frozen geometry.
// Given a state of rest2.js (positions and contact lists), find impulses per substep that balance the
// weight of every body exactly: sum of the contact impulses on a body = (0, +grav * f, 0), every
// impulse inside its friction cone, zero on contacts that do not touch.
//   solveFloat: projected Gauss-Seidel in float64 on the frozen geometry (optionally warm-started
//               from the impulses of the state), friction cone narrowed to `mu` for margin;
//   project:    rounding to integers, then the integer imbalance of every body is carried along a
//               spanning forest of loaded contacts to the floor and walls (the balance equations are
//               an incidence system, so an integer solution next to a real one always exists);
//   write:      impulses back into the slots of both partners.
'use strict';
const V = require('./rest2.js');
const { K, WALL, FLOOR, geo } = V;

// unique contacts of the state: a < b for sphere pairs, b = -1 for walls. n: unit normal from b to a.
function contacts(S, prm) {
  const N = S.N, C = [], sl = S.slot; let asym = 0;
  for (let i = 0; i < N; i++) { const n = S.vel[4 * i + 3] & 31;
    for (let s = 0; s < n; s++) { const o = 4 * (i * K + s), id = sl[o + 3] >>> 0;
      if (id < WALL) { if (id < i) continue; const m = S.vel[4 * id + 3] & 31; let f = -1; for (let q = 0; q < m; q++) if ((sl[4 * (id * K + q) + 3] >>> 0) === i) { f = q; break; } if (f < 0) { asym++; continue; }
        V.geom(S, i, id); const l = Math.hypot(geo.ex, geo.ey, geo.ez) || 1; C.push({ a: i, b: id, sa: s, sb: f, wall: 0, n: [geo.ex / l, geo.ey / l, geo.ez / l], pen: geo.pen, touch: geo.pen >= -prm.g0, P0: [sl[o], sl[o + 1], sl[o + 2]] }); }
      else { V.geom(S, i, id); C.push({ a: i, b: -1, sa: s, sb: -1, wall: id, n: [geo.ex, geo.ey, geo.ez], pen: geo.pen, touch: geo.pen >= -prm.g0, P0: [sl[o], sl[o + 1], sl[o + 2]] }); }
    } }
  return { C, asym };
}

function residual(S, prm, C, P, g) {
  const N = S.N, G = prm.grav * prm.f, r = new Float64Array(3 * N), gv = g || [0, -G, 0];
  for (let i = 0; i < N; i++) { r[3 * i] = gv[0]; r[3 * i + 1] = gv[1]; r[3 * i + 2] = gv[2]; }
  for (let c = 0; c < C.length; c++) { const q = C[c]; for (let k = 0; k < 3; k++) { r[3 * q.a + k] += P[3 * c + k]; if (q.b >= 0) r[3 * q.b + k] -= P[3 * c + k]; } }
  return r;
}
const maxabs = a => { let m = 0; for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > m) m = v; } return m; };

// Projected Gauss-Seidel on the frozen geometry. Returns { P, sweeps, res } (res: max |imbalance| over bodies and axes).
function solveFloat(S, prm, C, opt = {}) {
  const mu = opt.mu ?? 0.45, tol = opt.tol ?? 1e-7, maxSweeps = opt.sweeps ?? 200000, om = opt.omega ?? 1, P = new Float64Array(3 * C.length);
  if (opt.warm) for (let c = 0; c < C.length; c++) if (C[c].touch) for (let k = 0; k < 3; k++) P[3 * c + k] = C[c].P0[k];
  const r = residual(S, prm, C, P, opt.g), hist = []; let sweeps = 0, res = maxabs(r);
  const proj = (c, n) => { const o = 3 * c, pn = P[o] * n[0] + P[o + 1] * n[1] + P[o + 2] * n[2]; if (pn <= 0) { P[o] = P[o + 1] = P[o + 2] = 0; return; }
    const tx = P[o] - pn * n[0], ty = P[o + 1] - pn * n[1], tz = P[o + 2] - pn * n[2], t = Math.hypot(tx, ty, tz), lim = mu * pn;
    if (t > lim) { const s = lim / t; P[o] = pn * n[0] + tx * s; P[o + 1] = pn * n[1] + ty * s; P[o + 2] = pn * n[2] + tz * s; } };
  while (res > tol && sweeps < maxSweeps) {
    for (let c = 0; c < C.length; c++) { const q = C[c]; if (!q.touch) continue; const o = 3 * c, a = 3 * q.a, b = 3 * q.b, w = q.b >= 0 ? 0.5 * om : om;
      const x0 = P[o], y0 = P[o + 1], z0 = P[o + 2];
      if (q.b >= 0) { P[o] -= w * (r[a] - r[b]); P[o + 1] -= w * (r[a + 1] - r[b + 1]); P[o + 2] -= w * (r[a + 2] - r[b + 2]); } else { P[o] -= w * r[a]; P[o + 1] -= w * r[a + 1]; P[o + 2] -= w * r[a + 2]; }
      proj(c, q.n); const dx = P[o] - x0, dy = P[o + 1] - y0, dz = P[o + 2] - z0;
      r[a] += dx; r[a + 1] += dy; r[a + 2] += dz; if (q.b >= 0) { r[b] -= dx; r[b + 1] -= dy; r[b + 2] -= dz; } }
    sweeps++; if ((sweeps & 15) === 0 || sweeps < 16) { res = maxabs(r); if ((sweeps & (sweeps - 1)) === 0) hist.push([sweeps, +res.toPrecision(3)]); }
  }
  res = maxabs(residual(S, prm, C, P, opt.g)); hist.push([sweeps, +res.toPrecision(3)]);
  return { P, sweeps, res, hist };
}

// Integer projection with exact balance. Returns { Pi (Int32Array), unbalanced, moved_max, ... }.
function project(S, prm, C, Pf, opt = {}) {
  const N = S.N, G = prm.grav * prm.f, Pi = new Int32Array(3 * C.length), minLoad = opt.minLoad ?? 64, gv = opt.g || [0, -G, 0];
  // contacts carrying less than `minKeep` along the normal are dropped before rounding (their load joins the integer
  // imbalance that the forest carries): after rounding, a contact that small can sit anywhere relative to its cone
  const minKeep = opt.minKeep ?? 0.5;
  for (let c = 0; c < C.length; c++) { const n = C[c].n, o = 3 * c, pn = Pf[o] * n[0] + Pf[o + 1] * n[1] + Pf[o + 2] * n[2]; if (pn < minKeep) continue; for (let k = 0; k < 3; k++) Pi[o + k] = Math.round(Pf[o + k]); }
  const rho = new Int32Array(3 * N); for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) rho[3 * i + k] = gv[k];
  for (let c = 0; c < C.length; c++) { const q = C[c]; for (let k = 0; k < 3; k++) { rho[3 * q.a + k] += Pi[3 * c + k]; if (q.b >= 0) rho[3 * q.b + k] -= Pi[3 * c + k]; } }
  const before = maxabs(rho);
  // maximum-load spanning forest (Kruskal by normal load, the world frame as one more vertex N): the integer
  // imbalance is carried towards the world along the most heavily loaded contacts, whose cones have the most room
  const load = c => { const n = C[c].n, o = 3 * c; return Pi[o] * n[0] + Pi[o + 1] * n[1] + Pi[o + 2] * n[2]; };
  const uf = new Int32Array(N + 1); for (let i = 0; i <= N; i++) uf[i] = i; const find = x => { while (uf[x] !== x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; };
  const ord = []; for (let c = 0; c < C.length; c++) if (load(c) >= minLoad) ord.push(c); ord.sort((x, y) => load(y) - load(x) || x - y);
  const adj = Array.from({ length: N + 1 }, () => []);
  for (const c of ord) { const a = C[c].a, b = C[c].b < 0 ? N : C[c].b, ra = find(a), rb = find(b); if (ra === rb) continue; uf[ra] = rb; adj[a].push(c); adj[b].push(c); }
  const par = new Int32Array(N).fill(-2), order = []; const seen = new Uint8Array(N + 1); seen[N] = 1; const q = [N];
  for (let h = 0; h < q.length; h++) { const u = q[h]; for (const c of adj[u]) { const a = C[c].a, b = C[c].b < 0 ? N : C[c].b, v = u === a ? b : a; if (seen[v]) continue; seen[v] = 1; par[v] = c; order.push(v); q.push(v); } }
  let unreached = 0, movedMax = 0; for (let i = 0; i < N; i++) if (par[i] === -2) unreached++;
  for (let h = order.length - 1; h >= 0; h--) { const i = order[h], c = par[i], q = C[c], o = 3 * c;
    for (let k = 0; k < 3; k++) { const d = rho[3 * i + k]; if (!d) continue; if (Math.abs(d) > movedMax) movedMax = Math.abs(d);
      if (q.a === i) { Pi[o + k] -= d; if (q.b >= 0) rho[3 * q.b + k] += d; } else { Pi[o + k] += d; rho[3 * q.a + k] += d; } rho[3 * i + k] = 0; } }
  return { Pi, rounding_imbalance_max: before, unreached, carried_max: movedMax, unbalanced: maxabs(rho) };
}

function write(S, C, Pi) {
  const sl = S.slot; for (let i = 0; i < S.N; i++) { const n = S.vel[4 * i + 3] & 31; for (let s = 0; s < n; s++) { const o = 4 * (i * K + s); sl[o] = sl[o + 1] = sl[o + 2] = 0; } S.vel[4 * i] = S.vel[4 * i + 1] = S.vel[4 * i + 2] = 0; S.vel[4 * i + 3] &= 31; }
  for (let c = 0; c < C.length; c++) { const q = C[c], oa = 4 * (q.a * K + q.sa); for (let k = 0; k < 3; k++) sl[oa + k] = Pi[3 * c + k]; if (q.b >= 0) { const ob = 4 * (q.b * K + q.sb); for (let k = 0; k < 3; k++) sl[ob + k] = -Pi[3 * c + k]; } }
  S.seed = null;
}

// cone statistics of a float or integer solution: fraction of loaded contacts by friction mobilisation |t| / (mu_s * pn)
function coneStats(prm, C, P) {
  const mus = prm.mu8 / 8; let loaded = 0, over = 0, near = 0, sum = 0, floorY = 0, wallY = 0, tens = 0;
  for (let c = 0; c < C.length; c++) { const n = C[c].n, o = 3 * c, pn = P[o] * n[0] + P[o + 1] * n[1] + P[o + 2] * n[2];
    if (C[c].wall === FLOOR) floorY += P[o + 1]; else if (C[c].wall) wallY += P[o + 1];
    if (!(P[o] || P[o + 1] || P[o + 2])) continue; if (pn <= 0) { tens++; continue; } loaded++;
    const t = Math.hypot(P[o] - pn * n[0], P[o + 1] - pn * n[1], P[o + 2] - pn * n[2]), m = t / (mus * pn); sum += m; if (m > 1) over++; else if (m > 0.9) near++; }
  return { loaded, tension: tens, mobilisation_mean: +(sum / Math.max(1, loaded)).toFixed(3), over_cone: over, within_10pct_of_cone: near, floor_y: floorY, wall_y: wallY };
}
module.exports = { contacts, solveFloat, project, write, residual, coneStats, maxabs };
