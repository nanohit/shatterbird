// Phase 2a's modal body + unilateral port force, with a brittle seam between two members.
// The authoritative tick stores Q24 in i32. Velocity is displacement per microstep.
export const H = {
  age: 0, fired: 1, spent: 2, touch: 3, broken: 4, separated: 5,
  peak: 6, dissipated: 7, fault: 8, strikers: 9, modes: 13,
};
export const TICK = 60;
export const loadBeam = async () => {
  const response = await fetch(new URL('./beam.json', import.meta.url));
  if (!response.ok) throw new Error('Не удалось загрузить пролёт');
  return fixture(await response.json());
};

export function fixture(asset) {
  const a = asset;
  const n = a.modes.length, links = a.bonds.length, Q = 2 ** a.bits;
  const micro = a.hz / TICK, all = (1 << links) - 1;
  const end = a.hz * 6, fall = a.hz * 0.75;
  const state = new Int32Array(H.modes + 2 * n);
  for (let i = 0; i < n; i++) state[H.modes + 2 * i] = a.modes[i][3];
  for (let i = 0; i < 2; i++) state[H.strikers + i * 2] = a.launch[i];
  const finished = s => s[H.age] >= end || (s[H.separated] > 0 && s[H.age] - s[H.separated] >= fall);

  function cpu() {
    const s = state.slice(), force = new Int32Array(n), openings = new Int32Array(links);
    let failed = false;
    const checked = x => {
      if (x > 2147483647 || x < -2147483647) { failed = true; return 0; }
      return x;
    };
    const mul = (x, y) => {
      // Splitting x at 16 bits preserves exact JS arithmetic, even up to the i32 limits.
      const u = Math.abs(x), v = Math.abs(y);
      const hi = (u >>> 16) * v + Math.floor(((u & 65535) * v + Q / 2) / 65536);
      const mag = Math.floor(hi / (Q / 65536));
      return checked((x < 0) !== (y < 0) ? -mag : mag);
    };
    const at = i => H.modes + 2 * i;
    function dot(row) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum = checked(sum + mul(row[i], s[at(i)]));
      return sum;
    }
    function scatter(row, f) {
      for (let i = 0; i < n; i++) force[i] = checked(force[i] + mul(row[i], f));
    }
    function step(mask, reset = false, fracture = true) {
      if (reset) { s.set(state); failed = false; return s; }
      if (finished(s) || failed) return s;
      for (let j = 0; j < 2; j++) {
        const bit = 1 << j;
        if ((mask & bit) && !(s[H.fired] & bit)) {
          s[H.fired] |= bit;
          s[H.strikers + 2 * j + 1] = a.speed;
        }
      }
      if (!s[H.fired]) return s;
      for (let sub = 0; sub < micro && !finished(s) && !failed; sub++) {
        for (let i = 0; i < n; i++) {
          const [k, d, g] = a.modes[i], p = at(i);
          force[i] = checked(checked(g - mul(k, s[p])) - mul(d, s[p + 1]));
        }
        if (!s[H.broken]) {
          let breaks = false;
          for (let j = 0; j < links; j++) {
            const x = openings[j] = dot(a.bonds[j]);
            s[H.peak] = Math.max(s[H.peak], x);
            if (fracture && x > a.limits[j]) breaks = true;
          }
          // The seam is one brittle interface with several integration points.
          // Its shared damage variable changes the operator, leaving q and v intact.
          for (let j = 0; j < links; j++) {
            const x = openings[j], f = mul(a.seam_stiffness, x);
            if (breaks) s[H.dissipated] = checked(s[H.dissipated] + Math.trunc(mul(f, x) / 2));
            else scatter(a.bonds[j], -f);
          }
          if (breaks) {
            s[H.broken] = all;
            s[H.separated] = s[H.age] + 1;
          }
        }
        for (let j = 0; j < 2; j++) {
          const bit = 1 << j, p = H.strikers + 2 * j;
          if (!(s[H.fired] & bit) || (s[H.spent] & bit)) continue;
          const x = checked(dot(a.ports[j]) - s[p]);
          if (x <= 0 && (s[H.touch] & bit)) {
            s[H.spent] |= bit; // Guided striker is withdrawn by its tool on separation.
            continue;
          }
          const f = mul(a.contact_stiffness, Math.max(0, x));
          if (x > 0) s[H.touch] |= bit;
          scatter(a.ports[j], -f);
          s[p + 1] = checked(s[p + 1] + Math.trunc(f / a.striker_mass));
          s[p] = checked(s[p] + s[p + 1]);
        }
        for (let i = 0; i < n; i++) {
          const p = at(i);
          s[p + 1] = checked(s[p + 1] + force[i]);
          s[p] = checked(s[p] + s[p + 1]);
        }
        s[H.age]++;
      }
      if (failed) s[H.fault] = 1;
      return s;
    }
    return { step };
  }

  function coordinates(s) {
    const xy = new Float32Array(a.shape.length / n);
    for (let p = 0; p < xy.length; p++) {
      let u = 0;
      for (let i = 0; i < n; i++) u += (a.shape[p * n + i] / Q) * (s[H.modes + i * 2] / Q);
      xy[p] = a.mesh[p >> 1][p & 1] + u;
    }
    return xy;
  }
  return { asset: a, n, links, Q, micro, all, state, end, fall, finished, cpu, coordinates };
}

export function signature(s) {
  let h = 0x811c9dc5;
  for (const word of s) h = Math.imul(h ^ word, 0x01000193);
  return (h >>> 0).toString(16).padStart(8, '0');
}
