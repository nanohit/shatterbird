// node gpu-probe/cmp-dumps.mjs <tagA> <tagB> <label>  — compares full state dumps in gpu-probe/out/.
// Positions and velocities must match word for word; contact slots are compared as sets per body
// (the order inside a body's slot list follows the race in the grid and carries no meaning).
import { readFileSync } from 'node:fs';
const [a, b, label] = process.argv.slice(2), K = 16, ld = (t, k) => { const u = readFileSync(new URL(`./out/${t}_${label}_${k}.bin`, import.meta.url)); return new Int32Array(u.buffer, u.byteOffset, u.byteLength >> 2); };
const A = { pos: ld(a, 'pos'), vel: ld(a, 'vel'), slot: ld(a, 'slot') }, B = { pos: ld(b, 'pos'), vel: ld(b, 'vel'), slot: ld(b, 'slot') }, N = A.pos.length / 4;
let pv = 0, cnt = 0, sets = 0, order = 0, contacts = 0;
const key = (S, i) => { const o = []; for (let q = 0; q < (S.vel[4 * i + 3] & 31); q++) { const p = 4 * (i * K + q); o.push((S.slot[p + 3] >>> 0) + ':' + S.slot[p] + ',' + S.slot[p + 1] + ',' + S.slot[p + 2]); } return o; };
for (let i = 0; i < N; i++) { for (let k = 0; k < 3; k++) { if (A.pos[4 * i + k] !== B.pos[4 * i + k]) pv++; if (A.vel[4 * i + k] !== B.vel[4 * i + k]) pv++; } if (A.vel[4 * i + 3] !== B.vel[4 * i + 3]) cnt++; const ka = key(A, i), kb = key(B, i); contacts += ka.length; if (ka.join(';') !== kb.join(';')) order++; if (ka.sort().join(';') !== kb.sort().join(';')) sets++; }
console.log(JSON.stringify({ a, b, label, bodies: N, contacts, pos_vel_words_different: pv, bodies_with_different_contact_count: cnt, bodies_with_different_contact_set_or_impulse: sets, bodies_whose_slot_order_differs: order }));
