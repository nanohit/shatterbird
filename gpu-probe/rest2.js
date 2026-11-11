// Shatterbird phase 1f: contact solver with persistent impulses, specification v2 (JS reference).
// Same units and contact law as rest.js (v1). What changed:
//  * grid: per-cell linked lists, no capacity, nothing is ever dropped from the neighbour search;
//  * contacts: at most K per body; if more are in range the K deepest are kept (ties: smaller id) -
//    a choice that does not depend on traversal order - and every dropped one counts in `fault`;
//  * routing: a slow body resting on slow supports hands its whole residual momentum to those
//    supports (exact integer split, the remainder to one designated support) instead of solving the
//    contact by Jacobi; the lower body sees the reaction one iteration later. Momentum is conserved
//    exactly; the ledger only receives what a friction cone or a lost support leaves over;
//  * the ledger is state, not a running sum: what a supported body still cannot balance after the
//    last iteration (at most eps) is taken by an impulse towards the world frame and the body is set
//    to rest; that impulse, the anchor, is recomputed every substep and stored with the body (7 bits
//    per axis next to the contact count in vel.w, so eps <= 63). The unbalanced share of the weight is
//    then a quantity of the current state (the sum of the anchors), the same whether the world is
//    stepped fully or incrementally;
//  * every clamp is counted (`faults`): contacts dropped, bodies with a saturated impulse, velocity
//    clamps, position clamps at the box;
//  * bookkeeping for incremental execution: residual buffers ping-pong between two copies, every
//    level carries the same flag word for a body at rest, and `quiet` marks a strong fixed point
//    (nothing changed in any iteration), so both copies hold the rest value of a quiet body.
// Every reduction is an integer sum or an order-free selection: traversal order cannot matter.
'use strict';
const REST2 = (() => {
  const DIAM = 960, R = 480, K = 16, VMAX = 25600, WALL = 0xFFFFFFF0 >>> 0, FLOOR = WALL + 2;
  const W_AWAKE = 1, W_SUP = 2, W_DIRTY = 4, W_ELIG = 8, W_HASDES = 16, NS_SH = 5, CN_SH = 8, W_CLAMP = 1 << 14, W_DESFLOOR = 1 << 15, DES_SH = 16;
  function rng(seed) { let s = seed >>> 0; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s; }; }
  function isqrt(v) { let s = Math.floor(Math.sqrt(v)); while (s * s > v) s--; while ((s + 1) * (s + 1) <= v) s++; return s; }
  const tr = Math.trunc;
  function make(N, gx, gy, gz) {
    return { N, gx, gy, gz, pos: new Int32Array(4 * N), vel: new Int32Array(4 * N), slot: null, L: [new Int32Array(4 * K * N), new Int32Array(4 * K * N)], r: [new Int32Array(4 * N), new Int32Array(4 * N)], corr: new Int32Array(4 * N),
      head: new Int32Array(gx * gy * gz), next: new Int32Array(N), quiet: new Uint8Array(N), faults: new Float64Array(4), evals: 0, t: 0, seed: null, dist: new Int8Array(gx * gy * gz).fill(-1) };
  }
  // The impulses of the state live in L[0] (the level the last iteration writes when iters is even).
  function mk(N, gx, gy, gz) { const S = make(N, gx, gy, gz); S.slot = S.L[0]; return S; }
  // nx*ny*nz lattice, spacing (sx, sy, sx) units, jitter +-jit units, lowest layer resting on the floor.
  function lattice(nx, ny, nz, sx, sy, jit, seed, pad = 2, head = 4, vr = 0, f = 32, padx = 0) {
    const r = rng(seed), N = nx * ny * nz, gx = Math.ceil(nx * sx / 1024) + 2 * pad + padx, gz = Math.ceil(nz * sx / 1024) + 2 * pad, gy = Math.ceil(ny * sy / 1024) + head;
    const S = mk(N, gx, gy, gz); let i = 0;
    const j = () => jit ? (r() % (2 * jit + 1)) - jit : 0, jv = () => vr ? ((r() % (2 * vr + 1)) - vr) * f : 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++, i++) {
      S.pos[4 * i] = (pad * 1024 + x * sx + (sx >> 1) + j()) << 8; S.pos[4 * i + 1] = (y * sy + R + (y ? j() : 0)) << 8; S.pos[4 * i + 2] = (pad * 1024 + z * sx + (sx >> 1) + j()) << 8;
      S.vel[4 * i] = jv(); S.vel[4 * i + 1] = jv(); S.vel[4 * i + 2] = jv();
    }
    return S;
  }
  // The flag in pos.w (slow) is part of the state; call once after filling pos and vel by hand.
  function prime(S, prm) { for (let i = 0; i < S.N; i++) { const a = Math.max(Math.abs(S.vel[4 * i]), Math.abs(S.vel[4 * i + 1]), Math.abs(S.vel[4 * i + 2])); S.pos[4 * i + 3] = a <= prm.vr ? 1 : 0; } S.seed = null; return S; }
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
  function cellOf(S, i) { const p = S.pos; return (p[4 * i] >> 18) + S.gx * ((p[4 * i + 1] >> 18) + S.gy * (p[4 * i + 2] >> 18)); }
  function buildGrid(S) { const { N, head, next } = S; head.fill(-1); for (let i = 0; i < N; i++) { const c = cellOf(S, i); next[i] = head[c]; head[c] = i; } }
  // Order of bodies that does not depend on their numbering: by position, the index only for exact coincidence; walls last.
  function before(S, a, b) { if (a >= WALL || b >= WALL) return a < b; const p = S.pos; for (let k = 0; k < 3; k++) { const d = p[4 * a + k] - p[4 * b + k]; if (d) return d < 0; } return a < b; }
  const pkey = (S, j) => ((Math.imul(S.pos[4 * j], 73856093) ^ Math.imul(S.pos[4 * j + 1], 19349663) ^ Math.imul(S.pos[4 * j + 2], 83492791)) >>> 16);
  const cid = new Uint32Array(K), cpen = new Int32Array(K), oid = new Uint32Array(K), oP = new Int32Array(3 * K);
  // Level 0: contacts of body i, impulses carried over by partner id, flags, residual.
  function match(S, i, prm) {
    const { pos, vel, gx, gy, gz, head, next } = S, L0 = S.L[0], r0 = S.r[0];
    const px = pos[4 * i] >> 8, py = pos[4 * i + 1] >> 8, pz = pos[4 * i + 2] >> 8, cx = px >> 10, cy = py >> 10, cz = pz >> 10, MG = prm.margin, lim = (DIAM + MG) * (DIAM + MG);
    let n = 0; const b = i * K;
    const add = (id, pen) => {
      if (n < K) { cid[n] = id; cpen[n++] = pen; return; }
      S.faults[0]++; let w = 0; for (let q = 1; q < K; q++) if (cpen[q] < cpen[w] || (cpen[q] === cpen[w] && before(S, cid[w], cid[q]))) w = q;
      if (pen > cpen[w] || (pen === cpen[w] && before(S, id, cid[w]))) { cid[w] = id; cpen[w] = pen; }
    };
    for (let dz = -1; dz <= 1; dz++) { const z = cz + dz; if (z < 0 || z >= gz) continue;
      for (let dy = -1; dy <= 1; dy++) { const y = cy + dy; if (y < 0 || y >= gy) continue;
        for (let dx = -1; dx <= 1; dx++) { const x = cx + dx; if (x < 0 || x >= gx) continue;
          for (let j = head[x + gx * (y + gy * z)]; j >= 0; j = next[j]) {
            if (j === i) continue;
            const ex = px - (pos[4 * j] >> 8), ey = py - (pos[4 * j + 1] >> 8), ez = pz - (pos[4 * j + 2] >> 8), d2 = ex * ex + ey * ey + ez * ez;
            if (d2 >= lim) continue;
            add(j, d2 === 0 ? DIAM : DIAM - isqrt(d2));
          } } } }
    const wl = (c, H, a) => { if (R - c > -MG) add(WALL + 2 * a, R - c); if (c - (H - R) > -MG) add(WALL + 2 * a + 1, c - (H - R)); };
    wl(px, gx * 1024, 0); wl(py, gy * 1024, 1); wl(pz, gz * 1024, 2);
    const awake = (vel[4 * i] | vel[4 * i + 1] | vel[4 * i + 2]) ? 1 : 0;
    const on = vel[4 * i + 3] & 31; for (let q = 0; q < on; q++) { const o = 4 * (b + q); oid[q] = L0[o + 3] >>> 0; oP[3 * q] = L0[o]; oP[3 * q + 1] = L0[o + 1]; oP[3 * q + 2] = L0[o + 2]; }   // the list is rewritten in place
    let same = n === on, ax = 0, ay = 0, az = 0, sup = 0, nr = 0, des = 0, dey = 0, dlen = 1, has = 0;
    for (let s = 0; s < n; s++) {
      const id = cid[s], o = 4 * (b + s); let f = -1;
      for (let q = 0; q < on; q++) if (oid[q] === id) { f = q; break; }
      if (f < 0) { same = false; L0[o] = L0[o + 1] = L0[o + 2] = 0; } else { L0[o] = oP[3 * f]; L0[o + 1] = oP[3 * f + 1]; L0[o + 2] = oP[3 * f + 2]; }
      L0[o + 3] = id | 0; ax += L0[o]; ay += L0[o + 1]; az += L0[o + 2]; if (L0[o] | L0[o + 1] | L0[o + 2]) sup = 1;
      // supports that can take routed momentum: partner below within 60 degrees of vertical, touching, slow
      geom(S, i, id);
      if (id === FLOOR || (id < WALL && geo.ey * 2 >= geo.len && geo.pen >= -prm.g0 && (pos[4 * id + 3] & 1))) {
        nr++;
        if (!has || geo.ey * dlen > dey * geo.len || (geo.ey * dlen === dey * geo.len && des !== FLOOR && (id === FLOOR || before(S, id, des)))) { has = 1; des = id; dey = geo.ey; dlen = geo.len; }
      }
    }
    // the designated support is published as a 16-bit hash of its position; two supports with the same hash: no designated one
    const dk = has && des !== FLOOR ? pkey(S, des) : 0;
    if (has && des !== FLOOR) for (let s = 0; s < n; s++) { const id = cid[s]; if (id !== des && id < WALL && pkey(S, id) === dk) { geom(S, i, id); if (geo.ey * 2 >= geo.len && geo.pen >= -prm.g0 && (pos[4 * id + 3] & 1)) has = 0; } }
    const elig = prm.route && (pos[4 * i + 3] & 1) && nr >= 1 && nr <= 7 ? 1 : 0;
    r0[4 * i] = vel[4 * i] + ax; r0[4 * i + 1] = vel[4 * i + 1] - prm.grav * prm.f + ay; r0[4 * i + 2] = vel[4 * i + 2] + az;
    r0[4 * i + 3] = (awake * W_AWAKE | sup * W_SUP | (same ? 0 : W_DIRTY) | elig * W_ELIG | has * W_HASDES | (elig ? nr : 0) << NS_SH | n << CN_SH | (has && des === FLOOR ? W_DESFLOOR : 0) | dk << DES_SH) | 0;
  }
  // Share of the residual (x, y, z) of an eligible body that goes to one of its ns supports.
  const sh = [0, 0, 0];
  function share(S, rr, o, w, pid) { const ns = (w >> NS_SH) & 7, d = (w & W_HASDES) && (pid === FLOOR ? (w & W_DESFLOOR) : (!(w & W_DESFLOOR) && (w >>> DES_SH) === pkey(S, pid))); for (let k = 0; k < 3; k++) { const q = tr(rr[o + k] / ns); sh[k] = d ? rr[o + k] - (ns - 1) * q : q; } }
  // One iteration for body i: level k (src, rin) -> level k+1 (dst, rout). corr: push-out, iteration 0 only.
  function solve(S, i, prm, src, dst, rin, rout, corr) {
    const { pos, vel } = S, b = i * K, mw = rin[4 * i + 3], n = (mw >> CN_SH) & 31, F = prm.f, PM = 16384 * F, TQ = 4 * F, cl = q => q > PM ? PM : q < -PM ? -PM : q;
    let ax = 0, ay = 0, az = 0, sup = 0, dirty = 0, clamp = 0, kx = 0, ky = 0, kz = 0;
    const vx = rin[4 * i], vy = rin[4 * i + 1], vz = rin[4 * i + 2], slowi = pos[4 * i + 3] & 1;
    for (let s = 0; s < n; s++) {
      const o = 4 * (b + s), id = src[o + 3] >>> 0; geom(S, i, id);
      const { ex, ey, ez, len, pen } = geo, sph = id < WALL, wj = sph ? rin[4 * id + 3] : 0, den = n + ((wj >> CN_SH) & 31);
      let Px = src[o], Py = src[o + 1], Pz = src[o + 2];
      if ((mw & W_ELIG) && (id === FLOOR || (sph && ey * 2 >= len && pen >= -prm.g0 && (pos[4 * id + 3] & 1)))) { share(S, rin, 4 * i, mw, id); Px -= sh[0]; Py -= sh[1]; Pz -= sh[2]; }
      else if (sph && (wj & W_ELIG) && slowi && -ey * 2 >= len && pen >= -prm.g0) { share(S, rin, 4 * id, wj, i); Px += sh[0]; Py += sh[1]; Pz += sh[2]; }
      else {
        let rx = vx, ry = vy, rz = vz;
        if (sph) { rx -= rin[4 * id]; ry -= rin[4 * id + 1]; rz -= rin[4 * id + 2]; }
        if (pen < -prm.g0) { const g = (pen + prm.g0) * 256 * F; rx -= tr(g * ex / len); ry -= tr(g * ey / len); rz -= tr(g * ez / len); }
        const th = prm.th;
        if (!th || rx >= th || rx <= -th || ry >= th || ry <= -th || rz >= th || rz <= -th) { Px -= cl(tr(rx / den)); Py -= cl(tr(ry / den)); Pz -= cl(tr(rz / den)); }
      }
      if (corr && pen > prm.slop && (pen > prm.hi || (mw & W_AWAKE) || (wj & W_AWAKE))) { const q = Math.min(pen - prm.slop, 64) * prm.cq, u = 256 / prm.cq; kx += tr(q * ex / (len * den)) * u; ky += tr(q * ey / (len * den)) * u; kz += tr(q * ez / (len * den)) * u; }
      { const ax0 = Px, ay0 = Py, az0 = Pz; Px = cl(Px); Py = cl(Py); Pz = cl(Pz); if (Px !== ax0 || Py !== ay0 || Pz !== az0) clamp = 1; }
      const pn = tr((Px * ex + Py * ey + Pz * ez) / len);
      if (pn <= 0) { Px = Py = Pz = 0; }
      else {
        const nx = tr(pn * ex / len), ny = tr(pn * ey / len), nz = tr(pn * ez / len), tx = tr((Px - nx) / TQ), ty = tr((Py - ny) / TQ), tz = tr((Pz - nz) / TQ);
        const t2 = tx * tx + ty * ty + tz * tz, lm = tr(prm.mu8 * pn / (8 * TQ));
        if (t2 > lm * lm) { const tl = isqrt(t2), lk = tr(prm.muk8 * pn / (8 * TQ)); Px = nx + tr(tx * lk / tl) * TQ; Py = ny + tr(ty * lk / tl) * TQ; Pz = nz + tr(tz * lk / tl) * TQ; }
      }
      Px = cl(Px); Py = cl(Py); Pz = cl(Pz);
      if (Px | Py | Pz) sup = 1;
      if (Px !== src[o] || Py !== src[o + 1] || Pz !== src[o + 2]) dirty = 1;
      dst[o] = Px; dst[o + 1] = Py; dst[o + 2] = Pz; dst[o + 3] = src[o + 3];
      ax += Px; ay += Py; az += Pz;
    }
    const qx = vel[4 * i] + ax, qy = vel[4 * i + 1] - prm.grav * F + ay, qz = vel[4 * i + 2] + az;
    rout[4 * i] = qx; rout[4 * i + 1] = qy; rout[4 * i + 2] = qz;
    rout[4 * i + 3] = (mw & ~W_SUP) | sup * W_SUP | (dirty | clamp) * W_DIRTY | clamp * W_CLAMP;
    if (corr) { corr[4 * i] = kx; corr[4 * i + 1] = ky; corr[4 * i + 2] = kz; }
  }
  // End of the substep for body i. Returns 1 if the body is not quiet.
  function finish(S, i, prm) {
    const { pos, vel, corr } = S, I = prm.iters, rr = S.r[I & 1], b = i * K, w = rr[4 * i + 3], n = (w >> CN_SH) & 31, F = prm.f, VM = VMAX * F, cl = (v, a, c) => v < a ? a : v > c ? c : v;
    let vx = cl(rr[4 * i], -VM, VM), vy = cl(rr[4 * i + 1], -VM, VM), vz = cl(rr[4 * i + 2], -VM, VM);
    if (vx !== rr[4 * i] || vy !== rr[4 * i + 1] || vz !== rr[4 * i + 2]) S.faults[2]++;
    if (w & W_CLAMP) S.faults[1]++;
    let A = 0; const e = prm.eps;   // anchor: the small leftover of a supported body goes to the world frame, as state
    if (e && (w & W_SUP) && vx <= e && vx >= -e && vy <= e && vy >= -e && vz <= e && vz >= -e) { A = ((-vx) & 127) | ((-vy) & 127) << 7 | ((-vz) & 127) << 14; vx = vy = vz = 0; }
    const wv = n | A << 8;
    const mx = tr(vx / F) + corr[4 * i], my = tr(vy / F) + corr[4 * i + 1], mz = tr(vz / F) + corr[4 * i + 2];
    const nq = (w & W_DIRTY) || corr[4 * i] || corr[4 * i + 1] || corr[4 * i + 2] || mx || my || mz || vx !== vel[4 * i] || vy !== vel[4 * i + 1] || vz !== vel[4 * i + 2] || wv !== vel[4 * i + 3] ? 1 : 0;
    vel[4 * i] = vx; vel[4 * i + 1] = vy; vel[4 * i + 2] = vz; vel[4 * i + 3] = wv;
    const lo = R << 8, ux = pos[4 * i] + mx, uy = pos[4 * i + 1] + my, uz = pos[4 * i + 2] + mz;
    pos[4 * i] = cl(ux, lo, ((S.gx * 1024 - R) << 8) - 1); pos[4 * i + 1] = cl(uy, lo, ((S.gy * 1024 - R) << 8) - 1); pos[4 * i + 2] = cl(uz, lo, ((S.gz * 1024 - R) << 8) - 1);
    if (pos[4 * i] !== ux || pos[4 * i + 1] !== uy || pos[4 * i + 2] !== uz) S.faults[3]++;
    pos[4 * i + 3] = Math.max(Math.abs(vx), Math.abs(vy), Math.abs(vz)) <= prm.vr ? 1 : 0;
    S.quiet[i] = nq ? 0 : 1; return nq;
  }
  // Full step: every body. Returns the number of bodies that are not quiet.
  function step(S, prm) {
    const N = S.N, I = prm.iters; if (I & 1) throw new Error('iters must be even'); buildGrid(S); const seed = [];
    for (let i = 0; i < N; i++) match(S, i, prm);
    for (let k = 0; k < I; k++) for (let i = 0; i < N; i++) solve(S, i, prm, S.L[k & 1], S.L[(k + 1) & 1], S.r[k & 1], S.r[(k + 1) & 1], k === 0 ? S.corr : null);
    let nq = 0; for (let i = 0; i < N; i++) { const c0 = cellOf(S, i); if (finish(S, i, prm)) { nq++; seed.push(c0); const c1 = cellOf(S, i); if (c1 !== c0) seed.push(c1); } }
    S.seed = seed; S.evals += N * (I + 2); S.t++; return { changed: nq, evaluated: N, work: N * (I + 2) };
  }
  // Incremental step, the scheme the GPU uses: cells get their Chebyshev distance d to the nearest cell
  // touched by a non-quiet body of the previous step (old or new cell); level 0 is recomputed for
  // d <= 1, iteration k for d <= k + 2, the end of the step for d <= iters + 1. A change travels one
  // contact per level and a contact never spans more than one cell, so every body left out would have
  // reproduced the value it already holds - in both residual copies, because it is quiet.
  function stepInc(S, prm) {
    if (S.seed === null) return step(S, prm);
    const { gx, gy, gz, dist, head, next } = S, I = prm.iters, rmax = I + 1; buildGrid(S);
    const cells = [[]], bod = [];
    for (const c of S.seed) if (dist[c] < 0) { dist[c] = 0; cells[0].push(c); }
    for (let d = 1; d <= rmax; d++) { const cur = []; for (const c of cells[d - 1]) { const cz = Math.floor(c / (gx * gy)), cy = Math.floor((c - cz * gx * gy) / gx), cx = c - gx * (cy + gy * cz);
      for (let z = Math.max(0, cz - 1); z <= Math.min(gz - 1, cz + 1); z++) for (let y = Math.max(0, cy - 1); y <= Math.min(gy - 1, cy + 1); y++) for (let x = Math.max(0, cx - 1); x <= Math.min(gx - 1, cx + 1); x++) { const q = x + gx * (y + gy * z); if (dist[q] < 0) { dist[q] = d; cur.push(q); } } } cells.push(cur); }
    for (let d = 0; d <= rmax; d++) { const l = []; for (const c of cells[d]) { dist[c] = -1; for (let j = head[c]; j >= 0; j = next[j]) l.push(j); } bod.push(l); }
    let work = 0; const upto = (d, f) => { for (let q = 0; q <= Math.min(d, rmax); q++) { const l = bod[q]; work += l.length; for (let t = 0; t < l.length; t++) f(l[t]); } };
    upto(1, i => match(S, i, prm));
    for (let k = 0; k < I; k++) upto(k + 2, i => solve(S, i, prm, S.L[k & 1], S.L[(k + 1) & 1], S.r[k & 1], S.r[(k + 1) & 1], k === 0 ? S.corr : null));
    const seed = []; let nq = 0, ev = 0;
    upto(rmax, i => { ev++; const c0 = cellOf(S, i); if (finish(S, i, prm)) { nq++; seed.push(c0); const c1 = cellOf(S, i); if (c1 !== c0) seed.push(c1); } });
    S.seed = seed; S.evals += work; S.t++; return { changed: nq, evaluated: ev, work };
  }
  // Canonical hash of positions, velocities, flags and impulses (slots sorted by partner id).
  function canon(N, pos, vel, slot) {
    let h = 0x811c9dc5; const u = x => { h ^= x & 0xffff; h = Math.imul(h, 0x01000193); h ^= x >>> 16; h = Math.imul(h, 0x01000193); };
    for (let i = 0; i < N; i++) { for (let k = 0; k < 4; k++) { u(pos[4 * i + k]); u(vel[4 * i + k]); } const n = vel[4 * i + 3] & 31, ix = []; for (let q = 0; q < n; q++) ix.push(q); ix.sort((a, c) => (slot[4 * (i * K + a) + 3] >>> 0) - (slot[4 * (i * K + c) + 3] >>> 0)); for (const q of ix) { const o = 4 * (i * K + q); u(slot[o + 3]); u(slot[o]); u(slot[o + 1]); u(slot[o + 2]); } }
    return (h >>> 0).toString(16).padStart(8, '0');
  }
  const hash = S => canon(S.N, S.pos, S.vel, S.slot);
  function stats(S) {
    const { N, pos, vel, slot } = S; let vmax = 0, pmax = -1e9, top = 0, nc = 0, cmax = 0, fy = 0, wy = 0, an = 0, ay = 0, na = 0;
    for (let i = 0; i < N; i++) {
      vmax = Math.max(vmax, Math.abs(vel[4 * i]), Math.abs(vel[4 * i + 1]), Math.abs(vel[4 * i + 2])); top = Math.max(top, pos[4 * i + 1]);
      const n = vel[4 * i + 3] & 31, aw = vel[4 * i + 3] >> 8; nc += n; cmax = Math.max(cmax, n); if (aw) { na++; for (let k = 0; k < 3; k++) { const c = ((aw >> (7 * k)) & 127) << 25 >> 25; an += Math.abs(c); if (k === 1) ay += c; } }
      for (let s = 0; s < n; s++) { const o = 4 * (i * K + s), id = slot[o + 3] >>> 0; geom(S, i, id); if (geo.pen > pmax) pmax = geo.pen; if (id === FLOOR) fy += slot[o + 1]; else if (id >= WALL) wy += slot[o + 1]; }
    }
    return { vmax, pen_max: pmax, top: top / 256, contacts: nc / N, cmax, floor_impulse_y: fy, wall_impulse_y: wy, anchor_abs: an, anchor_y: ay, anchored_bodies: na };
  }

  // ---- WebGPU implementation (i32 / u32 only), full or incremental. ----
  // Buffers: pos (xyz, flags), vel (xyz, contact count), la / lb (K slots per body: impulse, partner id;
  // la also holds the state between substeps), ra / rb (residual and flag word per level), corr,
  // grid (cell heads, then one `next` link per body), fa / fb (per-cell distance marks), bl (header of
  // 128 words, then the list of active bodies), lx / ly / lz (cell lists of the distance transform),
  // seeds (64 stripes of cells touched by non-quiet bodies), at (atomic counters), ind (indirect args).
  const HDR = 128, AT = { NX: 0, NY: 1, NZ: 2, NB: 3, SEED: 8, LED: 96, FAULT: 108, WORK: 112, FULLSTEPS: 114, INCSTEPS: 115 };
  const HEAD = `
struct P { n: u32, gx: u32, gy: u32, gz: u32, grav: i32, f: i32, mu8: i32, muk8: i32, slop: i32, cq: i32, eps: i32, g0: i32, margin: i32, hi: i32, th: i32, route: i32,
  vr: i32, iters: u32, first: u32, thr: u32, capc: u32, seg: u32, dense: u32, force: u32 }
@group(0) @binding(0) var<uniform> p: P;
const DIAM = 960; const RAD = 480; const K = 16u; const VMAX = 25600; const WALL = 0xFFFFFFF0u; const FLOOR = 0xFFFFFFF2u; const EMPTY = 0xFFFFFFFFu; const UNM = 0xFFFFFFFFu; const HDR = 128u;
const AT_NX = 0u; const AT_NY = 1u; const AT_NZ = 2u; const AT_NB = 3u; const AT_SEED = 8u; const AT_LED = 96u; const AT_FAULT = 108u; const AT_WORK = 112u;
fn dims() -> vec3i { return vec3i(i32(p.gx), i32(p.gy), i32(p.gz)); }
fn cells() -> u32 { return p.gx * p.gy * p.gz; }
fn cidx(c: vec3i) -> u32 { return u32(c.x) + p.gx * (u32(c.y) + p.gy * u32(c.z)); }
fn ccoord(c: u32) -> vec3i { let z = c / (p.gx * p.gy); let y = (c - z * p.gx * p.gy) / p.gx; return vec3i(i32(c - p.gx * (y + p.gy * z)), i32(y), i32(z)); }
fn isqrt(v: u32) -> u32 { var s = u32(sqrt(f32(v))); if (s * s > v) { s -= 1u; } if (s * s > v) { s -= 1u; } if ((s + 1u) * (s + 1u) <= v) { s += 1u; } if ((s + 1u) * (s + 1u) <= v) { s += 1u; } return s; }
struct Geo { e: vec3i, len: i32, pen: i32 }
fn geo_wall(pu: vec3i, a: u32) -> Geo { let ax = a >> 1u; let hi = (a & 1u) == 1u; let c = pu[ax]; let H = dims()[ax] * 1024; var e = vec3i(0); e[ax] = select(1, -1, hi); return Geo(e, 1, select(RAD - c, c - (H - RAD), hi)); }
fn geo_sph(pu: vec3i, qu: vec3i, i: u32, j: u32) -> Geo { let e = pu - qu; let d2 = dot(e, e); if (d2 == 0) { return Geo(vec3i(select(-1, 1, i < j), 0, 0), 1, DIAM); } let l = i32(isqrt(u32(d2))); return Geo(e, l, DIAM - l); }
fn pkey(x: vec3i) -> u32 { let u = vec3u(x); return ((u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u)) >> 16u; }
// share of the residual r of an eligible body (flag word w) for one support; isdes: that support is the designated one
fn share(r: vec3i, w: u32, isdes: bool) -> vec3i { let ns = i32((w >> 5u) & 7u); let q = r / ns; if (isdes) { return r - (ns - 1) * q; } return q; }
fn isdes(w: u32, floor: bool, key: u32) -> bool { if ((w & 16u) == 0u) { return false; } if (floor) { return (w & 0x8000u) != 0u; } return (w & 0x8000u) == 0u && (w >> 16u) == key; }
`;
  const BODY = `
fn body(k: u32) -> u32 { if (bl[0] != 0u) { return k; } return bl[HDR + k]; }
fn skip(c: u32) -> bool { return bl[0] == 0u && fa[c] > p.thr; }
`;
  const WGSL = {
    clear: HEAD + `
@group(0) @binding(1) var<storage, read_write> grid: array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x + g.y * 4194240u; if (c >= cells()) { return; } atomicStore(&grid[c], EMPTY);
}`,
    insert: HEAD + `
@group(0) @binding(1) var<storage, read> pos: array<vec4i>;
@group(0) @binding(2) var<storage, read_write> grid: array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; }
  let c = cidx(pos[i].xyz >> vec3u(18u)); atomicStore(&grid[cells() + i], atomicExchange(&grid[c], i));
}`,
    // one invocation: reads the seed counters of the previous substep, decides full or incremental
    args0: HEAD + `
@group(0) @binding(1) var<storage, read_write> at: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> bl: array<u32>;
@group(0) @binding(3) var<storage, read_write> ind: array<u32>;
@compute @workgroup_size(1) fn main() {
  var m = 0u; var tot = 0u; var ovf = false;
  for (var s = 0u; s < 64u; s++) { let c = atomicExchange(&at[AT_SEED + s], 0u); let cc = min(c, p.seg); bl[8u + s] = cc; m = max(m, cc); tot += c; if (c > p.seg) { ovf = true; } }
  let full = ovf || tot > p.dense || bl[7] == 0u || p.force != 0u; bl[7] = 1u;
  for (var q = 0u; q < 4u; q++) { atomicStore(&at[q], 0u); }
  bl[0] = select(0u, 1u, full); bl[5] = m; bl[6] = 0u; bl[2] = 0u; bl[3] = 0u; bl[4] = 0u;
  for (var q = 0u; q < 18u; q++) { ind[q] = 0u; }
  if (full) { bl[1] = p.n; ind[12] = (p.n + 63u) / 64u; ind[13] = 1u; ind[14] = 1u; }
  else { bl[1] = 0u; ind[0] = (m + 63u) / 64u; ind[1] = 64u; ind[2] = 1u; }
}`,
    // distance transform along x: seeds get 16, their row neighbours 16 + |dx| (the offset marks "x stage")
    markx: HEAD + `
@group(0) @binding(1) var<storage, read> bl: array<u32>;
@group(0) @binding(2) var<storage, read> seeds: array<u32>;
@group(0) @binding(3) var<storage, read_write> fa: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> lx: array<u32>;
@group(0) @binding(5) var<storage, read_write> at: array<atomic<u32>>;
fn app(c: u32) { let k = atomicAdd(&at[AT_NX], 1u); if (k < p.capc) { lx[k] = c; } }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let s = g.y; if (g.x >= bl[8u + s]) { return; }
  let c = seeds[s * p.seg + g.x]; let old = atomicMin(&fa[c], 16u); if (old == 16u) { return; } if (old == UNM) { app(c); }
  let cc = ccoord(c); let R = i32(p.iters) + 1;
  for (var d = 1; d <= R; d++) {
    if (cc.x - d >= 0) { let q = c - u32(d); if (atomicMin(&fa[q], 16u + u32(d)) == UNM) { app(q); } }
    if (cc.x + d < i32(p.gx)) { let q = c + u32(d); if (atomicMin(&fa[q], 16u + u32(d)) == UNM) { app(q); } }
  }
}`,
    args1: HEAD + `
@group(0) @binding(1) var<storage, read_write> at: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> bl: array<u32>;
@group(0) @binding(3) var<storage, read_write> ind: array<u32>;
@compute @workgroup_size(1) fn main() {
  // p.first selects the stage: 1 after x, 2 after y, 3 after z, 4 after the body list
  if (bl[0] != 0u) { if (p.first == 4u) { let w = atomicAdd(&at[AT_WORK], p.n); if (w + p.n < w) { atomicAdd(&at[AT_WORK + 1u], 1u); } atomicAdd(&at[AT_WORK + 2u], 1u); } return; }
  let c = atomicLoad(&at[p.first - 1u]);
  if (p.first < 4u && c > p.capc) { bl[0] = 1u; bl[6] = 1u; bl[1] = p.n; ind[12] = (p.n + 63u) / 64u; ind[13] = 1u; ind[14] = 1u; ind[15] = min(65535u, (cells() + 63u) / 64u); ind[16] = ((cells() + 63u) / 64u + 65534u) / 65535u; ind[17] = 1u; return; }
  let gcount = (c + 63u) / 64u;
  if (p.first == 1u) { bl[2] = c; ind[3] = gcount; ind[4] = 1u; ind[5] = 1u; }
  if (p.first == 2u) { bl[3] = c; ind[6] = gcount; ind[7] = 1u; ind[8] = 1u; }
  if (p.first == 3u) { bl[4] = c; ind[9] = gcount; ind[10] = 1u; ind[11] = 1u; }
  if (p.first == 4u) { bl[1] = c; ind[12] = gcount; ind[13] = 1u; ind[14] = 1u; let w = atomicAdd(&at[AT_WORK], c); if (w + c < w) { atomicAdd(&at[AT_WORK + 1u], 1u); } atomicAdd(&at[AT_WORK + 3u], 1u); }
}`,
    marky: HEAD + `
@group(0) @binding(1) var<storage, read> bl: array<u32>;
@group(0) @binding(2) var<storage, read> lx: array<u32>;
@group(0) @binding(3) var<storage, read> fa: array<u32>;
@group(0) @binding(4) var<storage, read_write> fb: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> ly: array<u32>;
@group(0) @binding(6) var<storage, read_write> at: array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (bl[0] != 0u || g.x >= bl[2]) { return; }
  let c = lx[g.x]; let v = i32(fa[c]) - 16; let cc = ccoord(c); let R = i32(p.iters) + 1;
  for (var d = -R; d <= R; d++) { let y = cc.y + d; if (y < 0 || y >= i32(p.gy)) { continue; } let val = max(v, abs(d)); if (val > R) { continue; }
    let q = u32(i32(c) + d * i32(p.gx)); if (atomicMin(&fb[q], u32(val)) == UNM) { let k = atomicAdd(&at[AT_NY], 1u); if (k < p.capc) { ly[k] = q; } } }
}`,
    markz: HEAD + `
@group(0) @binding(1) var<storage, read> bl: array<u32>;
@group(0) @binding(2) var<storage, read> ly: array<u32>;
@group(0) @binding(3) var<storage, read> fb: array<u32>;
@group(0) @binding(4) var<storage, read_write> fa: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> lz: array<u32>;
@group(0) @binding(6) var<storage, read_write> at: array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (bl[0] != 0u || g.x >= bl[3]) { return; }
  let c = ly[g.x]; let v = i32(fb[c]); let cc = ccoord(c); let R = i32(p.iters) + 1;
  for (var d = -R; d <= R; d++) { let z = cc.z + d; if (z < 0 || z >= i32(p.gz)) { continue; } let val = max(v, abs(d)); if (val > R) { continue; }
    let q = u32(i32(c) + d * i32(p.gx * p.gy)); if (atomicMin(&fa[q], u32(val)) >= 16u) { let k = atomicAdd(&at[AT_NZ], 1u); if (k < p.capc) { lz[k] = q; } } }
}`,
    collect: HEAD + `
@group(0) @binding(1) var<storage, read_write> bl: array<u32>;
@group(0) @binding(2) var<storage, read> lz: array<u32>;
@group(0) @binding(3) var<storage, read> grid: array<u32>;
@group(0) @binding(4) var<storage, read_write> at: array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (bl[0] != 0u || g.x >= bl[4]) { return; }
  var j = grid[lz[g.x]]; let nc = cells();
  while (j != EMPTY) { let k = atomicAdd(&at[AT_NB], 1u); bl[HDR + k] = j; j = grid[nc + j]; }
}`,
    match: HEAD + `
@group(0) @binding(1) var<storage, read> bl: array<u32>;
@group(0) @binding(2) var<storage, read> pos: array<vec4i>;
@group(0) @binding(3) var<storage, read> vel: array<vec4i>;
@group(0) @binding(4) var<storage, read> grid: array<u32>;
@group(0) @binding(5) var<storage, read> fa: array<u32>;
@group(0) @binding(6) var<storage, read_write> la: array<vec4i>;
@group(0) @binding(7) var<storage, read_write> ra: array<vec4i>;
@group(0) @binding(8) var<storage, read_write> at: array<atomic<u32>>;
` + BODY + `
fn before(a: u32, b: u32) -> bool { if (a >= WALL || b >= WALL) { return a < b; } let x = pos[a].xyz; let y = pos[b].xyz;
  if (x.x != y.x) { return x.x < y.x; } if (x.y != y.y) { return x.y < y.y; } if (x.z != y.z) { return x.z < y.z; } return a < b; }
fn addc(ids: ptr<function, array<u32, 16>>, pens: ptr<function, array<i32, 16>>, n: ptr<function, u32>, dropped: ptr<function, u32>, id: u32, pen: i32) {
  if (*n < K) { (*ids)[*n] = id; (*pens)[*n] = pen; *n += 1u; return; }
  *dropped += 1u; var w = 0u;
  for (var q = 1u; q < K; q++) { if ((*pens)[q] < (*pens)[w] || ((*pens)[q] == (*pens)[w] && before((*ids)[w], (*ids)[q]))) { w = q; } }
  if (pen > (*pens)[w] || (pen == (*pens)[w] && before(id, (*ids)[w]))) { (*ids)[w] = id; (*pens)[w] = pen; }
}
struct Sup { ok: bool, ey: i32, len: i32 }
fn support(pu: vec3i, i: u32, id: u32) -> Sup {
  if (id == FLOOR) { return Sup(true, 1, 1); } if (id >= WALL) { return Sup(false, 0, 1); }
  let pj = pos[id]; let ge = geo_sph(pu, pj.xyz >> vec3u(8u), i, id);
  return Sup(ge.e.y * 2 >= ge.len && ge.pen >= -p.g0 && (pj.w & 1) != 0, ge.e.y, ge.len);
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= bl[1]) { return; } let i = body(g.x);
  let pw = pos[i]; let pu = pw.xyz >> vec3u(8u); let cc = pu >> vec3u(10u); if (skip(cidx(cc))) { return; }
  let d = dims(); let mg = DIAM + p.margin; let lim = mg * mg; let nc = cells();
  var ids: array<u32, 16>; var pens: array<i32, 16>; var n = 0u; var dropped = 0u;
  for (var dz = -1; dz <= 1; dz++) { let z = cc.z + dz; if (z < 0 || z >= d.z) { continue; }
    for (var dy = -1; dy <= 1; dy++) { let y = cc.y + dy; if (y < 0 || y >= d.y) { continue; }
      for (var dx = -1; dx <= 1; dx++) { let x = cc.x + dx; if (x < 0 || x >= d.x) { continue; }
        var j = grid[cidx(vec3i(x, y, z))];
        while (j != EMPTY) {
          if (j != i) { let e = pu - (pos[j].xyz >> vec3u(8u)); let d2 = dot(e, e);
            if (d2 < lim) { var pen = DIAM; if (d2 != 0) { pen = DIAM - i32(isqrt(u32(d2))); } addc(&ids, &pens, &n, &dropped, j, pen); } }
          j = grid[nc + j];
        } } } }
  for (var a = 0u; a < 3u; a++) { let c = pu[a]; let H = d[a] * 1024;
    if (RAD - c > -p.margin) { addc(&ids, &pens, &n, &dropped, WALL + 2u * a, RAD - c); }
    if (c - (H - RAD) > -p.margin) { addc(&ids, &pens, &n, &dropped, WALL + 2u * a + 1u, c - (H - RAD)); } }
  if (dropped != 0u) { atomicAdd(&at[AT_FAULT], dropped); }
  let v = vel[i]; let on = u32(v.w) & 31u; var oid: array<u32, 16>; var op: array<vec3i, 16>;
  for (var q = 0u; q < on; q++) { let o = la[i * K + q]; oid[q] = u32(o.w); op[q] = o.xyz; }
  var acc = vec3i(0); var same = n == on; var sup = false; var nr = 0u; var has = false; var des = 0u; var dey = 0; var dlen = 1;
  for (var s = 0u; s < n; s++) { let id = ids[s]; var q3 = vec3i(0); var found = false;
    for (var q = 0u; q < on; q++) { if (oid[q] == id) { q3 = op[q]; found = true; break; } }
    if (!found) { same = false; }
    la[i * K + s] = vec4i(q3, bitcast<i32>(id)); acc += q3; if (any(q3 != vec3i(0))) { sup = true; }
    let su = support(pu, i, id);
    if (su.ok) { nr += 1u; if (!has || su.ey * dlen > dey * su.len || (su.ey * dlen == dey * su.len && des != FLOOR && (id == FLOOR || before(id, des)))) { has = true; des = id; dey = su.ey; dlen = su.len; } }
  }
  var dk = 0u;
  if (has && des != FLOOR) { dk = pkey(pos[des].xyz); for (var s = 0u; s < n; s++) { let id = ids[s]; if (id != des && id < WALL && pkey(pos[id].xyz) == dk) { if (support(pu, i, id).ok) { has = false; } } } }
  let elig = p.route != 0 && (pw.w & 1) != 0 && nr >= 1u && nr <= 7u;
  var w = (n << 8u) | (dk << 16u); if (has && des == FLOOR) { w |= 0x8000u; }
  if (any(v.xyz != vec3i(0))) { w |= 1u; } if (sup) { w |= 2u; } if (!same) { w |= 4u; } if (elig) { w |= 8u | (nr << 5u); } if (has) { w |= 16u; }
  ra[i] = vec4i(v.xyz + acc - vec3i(0, p.grav * p.f, 0), bitcast<i32>(w));
}`,
    solve: HEAD + `
@group(0) @binding(1) var<storage, read> bl: array<u32>;
@group(0) @binding(2) var<storage, read> pos: array<vec4i>;
@group(0) @binding(3) var<storage, read> fa: array<u32>;
@group(0) @binding(4) var<storage, read> lk: array<vec4i>;
@group(0) @binding(5) var<storage, read> rk: array<vec4i>;
@group(0) @binding(6) var<storage, read_write> ln: array<vec4i>;
@group(0) @binding(7) var<storage, read_write> rn: array<vec4i>;
@group(0) @binding(8) var<storage, read_write> corr: array<vec4i>;
` + BODY + `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= bl[1]) { return; } let i = body(g.x);
  let pw = pos[i]; let pu = pw.xyz >> vec3u(8u); if (skip(cidx(pu >> vec3u(10u)))) { return; }
  let me = rk[i]; let mw = bitcast<u32>(me.w); let n = (mw >> 8u) & 31u; let F = p.f; let PM = 16384 * F; let TQ = 4 * F; let slowi = (pw.w & 1) != 0;
  var dacc = vec3i(0); var sup = false; var dirty = false; var cr = vec3i(0); var clamps = 0u;
  for (var s = 0u; s < n; s++) {
    let L = lk[i * K + s]; let id = bitcast<u32>(L.w); var ge: Geo; var wj = 0u; var rj = vec3i(0); var slowj = true; let sph = id < WALL; var kj = 0u;
    if (sph) { let pj = pos[id]; let q = rk[id]; ge = geo_sph(pu, pj.xyz >> vec3u(8u), i, id); wj = bitcast<u32>(q.w); rj = q.xyz; slowj = (pj.w & 1) != 0; kj = pkey(pj.xyz); } else { ge = geo_wall(pu, id - WALL); }
    let den = i32(n + ((wj >> 8u) & 31u)); var Q = L.xyz;
    if ((mw & 8u) != 0u && (id == FLOOR || (sph && ge.e.y * 2 >= ge.len && ge.pen >= -p.g0 && slowj))) { Q -= share(me.xyz, mw, isdes(mw, id == FLOOR, kj)); }
    else if (sph && (wj & 8u) != 0u && slowi && -ge.e.y * 2 >= ge.len && ge.pen >= -p.g0) { Q += share(rj, wj, isdes(wj, false, pkey(pw.xyz))); }
    else {
      var r = me.xyz - rj;
      if (ge.pen < -p.g0) { let gg = (ge.pen + p.g0) * 256 * F; r -= (gg * ge.e) / ge.len; }
      if (p.th == 0 || any(abs(r) >= vec3i(p.th))) { Q -= clamp(r / den, vec3i(-PM), vec3i(PM)); }
    }
    if (p.first != 0u && ge.pen > p.slop && (ge.pen > p.hi || (mw & 1u) != 0u || (wj & 1u) != 0u)) { let q = min(ge.pen - p.slop, 64) * p.cq; cr += ((q * ge.e) / (ge.len * den)) * (256 / p.cq); }
    let Q0 = Q; Q = clamp(Q, vec3i(-PM), vec3i(PM)); if (any(Q != Q0)) { clamps += 1u; }
    let pn = dot(Q, ge.e) / ge.len;
    if (pn <= 0) { Q = vec3i(0); }
    else { let nv = (pn * ge.e) / ge.len; let t = (Q - nv) / TQ; let t2 = dot(t, t); let lm = (p.mu8 * pn) / (8 * TQ);
      if (t2 > lm * lm) { let tl = i32(isqrt(u32(t2))); let lkk = (p.muk8 * pn) / (8 * TQ); Q = nv + ((t * lkk) / tl) * TQ; } }
    Q = clamp(Q, vec3i(-PM), vec3i(PM));
    if (any(Q != vec3i(0))) { sup = true; } if (any(Q != L.xyz)) { dirty = true; }
    ln[i * K + s] = vec4i(Q, L.w); dacc += Q - L.xyz;
  }
  var w = mw & ~2u; if (sup) { w |= 2u; } if (dirty || clamps != 0u) { w |= 4u; } if (clamps != 0u) { w |= 0x4000u; }
  rn[i] = vec4i(me.xyz + dacc, bitcast<i32>(w));
  if (p.first != 0u) { corr[i] = vec4i(cr, 0); }
}`,
    finish: HEAD + `
@group(0) @binding(1) var<storage, read> bl: array<u32>;
@group(0) @binding(2) var<storage, read_write> pos: array<vec4i>;
@group(0) @binding(3) var<storage, read_write> vel: array<vec4i>;
@group(0) @binding(4) var<storage, read> fa: array<u32>;
@group(0) @binding(5) var<storage, read> rk: array<vec4i>;
@group(0) @binding(6) var<storage, read> corr: array<vec4i>;
@group(0) @binding(7) var<storage, read_write> seeds: array<u32>;
@group(0) @binding(8) var<storage, read_write> at: array<atomic<u32>>;
` + BODY + `
fn seed(i: u32, c: u32) { let s = i & 63u; let k = atomicAdd(&at[AT_SEED + s], 1u); if (k < p.seg) { seeds[s * p.seg + k] = c; } }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= bl[1]) { return; } let i = body(g.x);
  let pw = pos[i]; let c0 = cidx(pw.xyz >> vec3u(18u)); if (skip(c0)) { return; }
  let me = rk[i]; let w = bitcast<u32>(me.w); let n = (w >> 8u) & 31u; let F = p.f; let VM = VMAX * F; let vo = vel[i];
  var v = clamp(me.xyz, vec3i(-VM), vec3i(VM)); if (any(v != me.xyz)) { atomicAdd(&at[AT_FAULT + 2u], 1u); }
  if ((w & 0x4000u) != 0u) { atomicAdd(&at[AT_FAULT + 1u], 1u); }
  var A = 0u;
  if (p.eps != 0 && (w & 2u) != 0u && all(abs(v) <= vec3i(p.eps))) { let a = vec3u(-v) & vec3u(127u); A = a.x | (a.y << 7u) | (a.z << 14u); v = vec3i(0); }
  let wv = i32(n | (A << 8u)); let k3 = corr[i].xyz; let m = v / F + k3;
  let nq = (w & 4u) != 0u || any(k3 != vec3i(0)) || any(m != vec3i(0)) || any(v != vo.xyz) || wv != vo.w;
  let lo = vec3i(RAD * 256); let hi = (dims() * 1024 - vec3i(RAD)) * 256 - vec3i(1); let u = pw.xyz + m; let x = clamp(u, lo, hi);
  if (any(x != u)) { atomicAdd(&at[AT_FAULT + 3u], 1u); }
  vel[i] = vec4i(v, wv); pos[i] = vec4i(x, select(0, 1, all(abs(v) <= vec3i(p.vr))));
  if (nq) { seed(i, c0); let c1 = cidx(x >> vec3u(18u)); if (c1 != c0) { seed(i, c1); } }
}`,
    reset: HEAD + `
@group(0) @binding(1) var<storage, read> bl: array<u32>;
@group(0) @binding(2) var<storage, read> lz: array<u32>;
@group(0) @binding(3) var<storage, read_write> fa: array<u32>;
@group(0) @binding(4) var<storage, read_write> fb: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (bl[0] != 0u || g.x >= bl[4] || g.x >= p.capc) { return; } let c = lz[g.x]; fa[c] = UNM; fb[c] = UNM;
}`,
    resetall: HEAD + `
@group(0) @binding(1) var<storage, read_write> fa: array<u32>;
@group(0) @binding(2) var<storage, read_write> fb: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x + g.y * 4194240u; if (c >= cells()) { return; } fa[c] = UNM; fb[c] = UNM;
}`,
  };
  // h = { dev, mk, pipe }. opt: { force: always full, dense: seed count above which a substep runs full,
  // capc: capacity of the cell lists, seg: capacity of one seed stripe }.
  async function gpu(h, S0, prm, opt = {}) {
    const { dev, mk, pipe } = h, U = GPUBufferUsage, SU = U.STORAGE | U.COPY_SRC | U.COPY_DST, N = S0.N, cells = S0.gx * S0.gy * S0.gz, I = prm.iters;
    if (I < 2 || (I & 1)) throw new Error('iters must be even');
    const capc = opt.capc || Math.max(4096, Math.min(cells, 1 << 20)), seg = opt.seg || Math.max(256, Math.ceil(N / 128)), dense = opt.dense === undefined ? Math.ceil(N / 32) : opt.dense, force = opt.force ? 1 : 0;
    const buf = (sz, fill) => { const b = dev.createBuffer({ size: sz, usage: SU, mappedAtCreation: fill !== undefined }); if (fill !== undefined) { new Uint32Array(b.getMappedRange()).fill(fill); b.unmap(); } return b; };
    const la0 = new Int32Array(4 * K * N); la0.set(S0.slot);
    const b = { pos: mk(S0.pos), vel: mk(S0.vel), la: mk(la0), lb: buf(16 * K * N), ra: buf(16 * N), rb: buf(16 * N), corr: buf(16 * N), grid: buf(4 * (cells + N)), fa: buf(4 * cells, 0xFFFFFFFF), fb: buf(4 * cells, 0xFFFFFFFF),
      bl: buf(4 * (HDR + N)), lx: buf(4 * capc), ly: buf(4 * capc), lz: buf(4 * capc), seeds: buf(4 * 64 * seg), at: buf(4 * 128), ind: dev.createBuffer({ size: 4 * 18, usage: SU | U.INDIRECT }) };
    const uni = (first, thr) => { const u = new ArrayBuffer(96), dv = new DataView(u); [N, S0.gx, S0.gy, S0.gz].forEach((x, k) => dv.setUint32(4 * k, x, true));
      [prm.grav, prm.f, prm.mu8, prm.muk8, prm.slop, prm.cq, prm.eps, prm.g0, prm.margin, prm.hi, prm.th, prm.route, prm.vr].forEach((x, k) => dv.setInt32(16 + 4 * k, x, true));
      [I, first, thr, capc, seg, dense, force].forEach((x, k) => dv.setUint32(68 + 4 * k, x, true)); const ub = dev.createBuffer({ size: 96, usage: U.UNIFORM | U.COPY_DST }); dev.queue.writeBuffer(ub, 0, u); return ub; };
    const pl = {}; for (const k of Object.keys(WGSL)) pl[k] = await pipe(WGSL[k]);
    const bg = (k, u, bufs) => dev.createBindGroup({ layout: pl[k].getBindGroupLayout(0), entries: [u, ...bufs].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const u0 = uni(0, 0), unis = [u0], cg = [Math.min(65535, Math.ceil(cells / 64)), Math.ceil(Math.ceil(cells / 64) / 65535)], P = [];
    const add = (k, u, bufs, disp) => P.push([k, bg(k, u, bufs), disp]); const U1 = (f, t) => { const x = uni(f, t); unis.push(x); return x; };
    add('clear', u0, [b.grid], cg); add('insert', u0, [b.pos, b.grid], [Math.ceil(N / 64)]);
    add('args0', u0, [b.at, b.bl, b.ind], [1]); add('markx', u0, [b.bl, b.seeds, b.fa, b.lx, b.at], 0);
    add('args1', U1(1, 0), [b.at, b.bl, b.ind], [1]); add('marky', u0, [b.bl, b.lx, b.fa, b.fb, b.ly, b.at], 12);
    add('args1', U1(2, 0), [b.at, b.bl, b.ind], [1]); add('markz', u0, [b.bl, b.ly, b.fb, b.fa, b.lz, b.at], 24);
    add('args1', U1(3, 0), [b.at, b.bl, b.ind], [1]); add('collect', u0, [b.bl, b.lz, b.grid, b.at], 36);
    add('args1', U1(4, 0), [b.at, b.bl, b.ind], [1]);
    add('match', U1(0, 1), [b.bl, b.pos, b.vel, b.grid, b.fa, b.la, b.ra, b.at], 48);
    for (let k = 0; k < I; k++) add('solve', U1(k === 0 ? 1 : 0, k + 2), k & 1 ? [b.bl, b.pos, b.fa, b.lb, b.rb, b.la, b.ra, b.corr] : [b.bl, b.pos, b.fa, b.la, b.ra, b.lb, b.rb, b.corr], 48);
    add('finish', U1(0, I + 1), [b.bl, b.pos, b.vel, b.fa, b.ra, b.corr, b.seeds, b.at], 48);
    add('reset', u0, [b.bl, b.lz, b.fa, b.fb], 36); add('resetall', u0, [b.fa, b.fb], 60);
    const enc = (steps, only) => { const e = dev.createCommandEncoder(); for (let t = 0; t < steps; t++) for (const [k, g, d] of P) { if (only && !only.includes(k)) continue; const ps = e.beginComputePass(); ps.setPipeline(pl[k]); ps.setBindGroup(0, g); if (typeof d === 'number') ps.dispatchWorkgroupsIndirect(b.ind, d); else ps.dispatchWorkgroups(...d); ps.end(); } return [e.finish()]; };
    const all = [...Object.values(b), ...unis]; let bytes = 0; for (const x of Object.values(b)) bytes += x.size;
    return { b, enc, cells, passes: P.length, bytes_per_body: bytes / N, free: () => all.forEach(x => x.destroy()) };
  }
  // Counters of a GPU run as numbers: faults (4), evaluated body-substeps, full and incremental substeps.
  function counters(at) { const u = new Uint32Array(at.buffer, at.byteOffset, at.length), two = k => u[k] + u[k + 1] * 4294967296; return { faults: [0, 1, 2, 3].map(k => u[AT.FAULT + k]), work: two(AT.WORK), full_steps: u[AT.FULLSTEPS], inc_steps: u[AT.INCSTEPS] }; }
  const PRM = { grav: 111, f: 32, mu8: 4, muk8: 3, iters: 4, slop: 4, cq: 8, eps: 32, g0: 2, margin: 16, hi: 24, th: 0, route: 1, vr: 28416 };
  return { DIAM, R, K, WALL, FLOOR, make: mk, lattice, prime, step, stepInc, stats, hash, canon, geom, geo, match, solve, finish, buildGrid, cellOf, rng, PRM, W_ELIG, W_DIRTY, CN_SH, WGSL, gpu, counters, AT, HDR };
})();
if (typeof module !== 'undefined') module.exports = REST2;
