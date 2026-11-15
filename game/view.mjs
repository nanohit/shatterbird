import { H } from './world.mjs?v=2';

const colours = ['#a2daf5', '#ffbd8a'];

export function view(canvas, w) {
  const ctx = canvas.getContext('2d', { alpha: false });
  const { nx, ny, cut } = w.asset;
  return (s, side) => {
    const W = canvas.clientWidth, Y = canvas.clientHeight;
    if (!W || !Y) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(Y * dpr)) {
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(Y * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const faces = [], labels = [], xy = w.coordinates(s);
    let low = -8;
    if (s[H.broken]) {
      for (let i = 1; i < xy.length; i += 2) low = Math.min(low, xy[i] - 1.2);
    }
    const middle = (low + 8) / 2;
    const units = Math.min(W / 33, Y / (12 - low)), angle = side === 1 ? -1 : 1;
    const project = ([x, y, z]) => [
      W * .48 + units * (x - 11.5 + z * .47 * angle),
      Y * .54 + units * ((x - 11.5) * .10 * angle - .9 * (y - middle) + z * .29),
    ];
    const face = (v, fill, stroke = '#16242d') => {
      const depth = v.reduce((sum, [x, y, z]) => sum + z - x * .35 * angle + y * .23, 0) / v.length;
      faces.push({ v, fill, stroke, depth });
    };
    function prism(l, r, back, front, c, caps = 3) {
      const [lb, lt] = l, [rb, rt] = r;
      const p = ([x, y], z) => [x, y, z];
      face([p(lt, back), p(rt, back), p(rt, front), p(lt, front)], c[0]);
      face([p(lb, front), p(rb, front), p(rt, front), p(lt, front)], c[1]);
      const [u, v] = angle > 0 ? l : r;
      if (caps & (angle > 0 ? 1 : 2)) face([p(u, back), p(v, back), p(v, front), p(u, front)], c[2]);
    }
    function box(x, y, z, a, b, c, style) {
      prism([[x, y], [x, y + b]], [[x + a, y], [x + a, y + b]], z, z + c, style);
    }
    const bg = ctx.createLinearGradient(0, 0, 0, Y);
    bg.addColorStop(0, '#2c3d49');
    bg.addColorStop(1, '#131a20');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, Y);
    ctx.strokeStyle = '#455762';
    ctx.lineWidth = 1;
    for (let i = -10; i <= 35; i += 2) {
      line(project([i, -32, -8]), project([i, 12, -8]));
    }
    for (let i = -32; i <= 12; i += 2) line(project([-10, i, -8]), project([35, i, -8]));
    box(-3.5, -32, -3, 4.0, 37, 6, ['#77838a', '#505d66', '#63737e']);
    box(-2.7, -.6, -3.15, 2.9, 4.25, 6.3, ['#a6afb1', '#63717c', '#85939a']);

    const point = (i, j) => xy.subarray(2 * (i + j * nx), 2 * (i + j * nx) + 2);
    const layer = i => [point(i, 0), point(i, ny - 1)];
    const slab = ['#d7d1be', '#9e9d8e', '#b6b3a4'];
    const seam = ['#ecb67b', '#af764d', '#df9f69'];
    for (let i = 0; i < nx - 1; i++) {
      const l = layer(i), r = layer(i + 1);
      if (i !== cut - 1) {
        prism(l, r, -1.68, 1.68, slab, (i === 0 ? 1 : 0) | (i === nx - 2 ? 2 : 0));
        continue;
      }
      const edge = (a, shift) => a.map(([x,y], j) => [x + shift, y]);
      const left = edge(l, .42), right = edge(r, -.42);
      prism(l, left, -1.68, 1.68, slab, s[H.broken] ? 2 : 0);
      prism(right, r, -1.68, 1.68, slab, s[H.broken] ? 1 : 0);
      if (!s[H.broken]) prism(left, right, -1.69, 1.69, seam, 0);
    }
    for (let i = 0; i < 2; i++) {
      const x = (i === 0 ? 18 : 23) + .5;
      const p = H.strikers + i * 2;
      const visible = !(s[H.spent] & (1 << i));
      const y = 3 * Math.sqrt(3) / 2 + (visible ? s[p] : w.state[p]) / w.Q;
      const col = colours[i];
      box(x - .57, 6.7, -1.06, 1.14, .52, 2.12, ['#788b97', '#526674', '#394b57']);
      if (visible) {
        box(x - .14, y + .79, -.14, .28, Math.max(0, 6.7 - y - .79), .28, ['#617888', '#455a6a', '#4d6070']);
        box(x - .52, y, -.9, 1.04, .82, 1.8, [col, i ? '#bb794d' : '#5a9ab6', col]);
      }
      labels.push([project([x, 7.8, 0]), i === 0 ? 'A' : 'B', col]);
    }
    faces.sort((a, b) => a.depth - b.depth);
    for (const { v, fill, stroke } of faces) {
      ctx.beginPath();
      for (let i = 0; i < v.length; i++) {
        const p = project(v[i]);
        if (i === 0) ctx.moveTo(...p);
        else ctx.lineTo(...p);
      }
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = .7;
      ctx.stroke();
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = 'bold 12px ui-monospace, monospace';
    for (const [[x, y], label, col] of labels) {
      ctx.fillStyle = '#19242c';
      ctx.fillRect(x - 10, y - 9, 20, 18);
      ctx.fillStyle = col;
      ctx.fillText(label, x, y);
    }
    function line(a, b) {
      ctx.beginPath();
      ctx.moveTo(...a);
      ctx.lineTo(...b);
      ctx.stroke();
    }
  };
}
