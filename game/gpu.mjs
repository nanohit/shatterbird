import { H } from './world.mjs?v=2';

// The same limb product as fixed.js, with round-to-nearest, ties away from zero.
function source(w) {
  const a = w.asset, n = w.n, rows = w.links + 2;
  return `
@group(0) @binding(0) var<storage,read_write> s: array<i32>;
@group(0) @binding(1) var<storage,read> data: array<i32>;
@group(0) @binding(2) var<storage,read> input: array<i32>;
var<private> force: array<i32, ${n}>;
fn add(a: i32, b: i32) -> i32 {
  let r = a + b;
  if (((a ^ r) & (b ^ r)) < 0 || r == -2147483647 - 1) { s[${H.fault}] = 1; return 0; }
  return r;
}
fn mul(a: i32, b: i32) -> i32 {
  let u = u32(abs(a)); let v = u32(abs(b));
  let a0 = u & 65535u; let a1 = u >> 16u; let b0 = v & 65535u; let b1 = v >> 16u;
  let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0;
  let mid = (p00 >> 16u) + (p01 & 65535u) + (p10 & 65535u);
  let lo = (p00 & 65535u) | (mid << 16u);
  let rounded = lo + ${w.Q / 2}u;
  let hi = a1 * b1 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u) + select(0u, 1u, rounded < lo);
  if (hi >= (1u << ${a.bits - 1}u)) { s[${H.fault}] = 1; return 0; }
  let m = i32((hi << ${32 - a.bits}u) | (rounded >> ${a.bits}u));
  return select(m, -m, (a < 0) != (b < 0));
}
fn dot(row: u32) -> i32 {
  var x = 0;
  for (var i = 0u; i < ${n}u; i++) {
    x = add(x, mul(data[row * ${n}u + i], s[${H.modes}u + 2u * i]));
  }
  return x;
}
fn scatter(row: u32, f: i32) {
  for (var i = 0u; i < ${n}u; i++) {
    force[i] = add(force[i], mul(data[row * ${n}u + i], f));
  }
}
fn finished() -> bool {
  return s[${H.age}] >= ${w.end} || (s[${H.separated}] > 0 && s[${H.age}] - s[${H.separated}] >= ${w.fall});
}
@compute @workgroup_size(1) fn main() {
  if (finished() || s[${H.fault}] != 0) { return; }
  for (var j = 0u; j < 2u; j++) {
    let bit = 1 << j;
    if (s[${H.broken}] == 0 && (input[0] & bit) != 0 && (s[${H.fired}] & bit) == 0) {
      s[${H.fired}] |= bit;
      s[${H.strikers}u + 2u * j + 1u] = ${a.speed};
    }
  }
  if (s[${H.fired}] == 0) { return; }
  for (var step = 0u; step < ${w.micro}u && !finished() && s[${H.fault}] == 0; step++) {
    for (var i = 0u; i < ${n}u; i++) {
      let p = ${H.modes}u + 2u * i; let c = ${rows * n}u + 3u * i;
      force[i] = add(add(data[c + 2u], -mul(data[c], s[p])), -mul(data[c + 1u], s[p + 1u]));
    }
    if (s[${H.broken}] == 0) {
      var openings: array<i32, ${w.links}>;
      var breaks = false;
      for (var j = 0u; j < ${w.links}u; j++) {
        let x = dot(j);
        openings[j] = x;
        s[${H.peak}] = max(s[${H.peak}], x);
        if (input[1] != 0 && x > data[${rows * n + n * 3}u + j]) { breaks = true; }
      }
      for (var j = 0u; j < ${w.links}u; j++) {
        let x = openings[j]; let f = mul(${a.seam_stiffness}, x);
        if (breaks) { s[${H.dissipated}] = add(s[${H.dissipated}], mul(f, x) / 2); }
        else { scatter(j, -f); }
      }
      if (breaks) {
        s[${H.broken}] = ${w.all};
        s[${H.separated}] = s[${H.age}] + 1;
        s[${H.splitRotation}] = s[${H.modes + a.rotation.mode * 2}];
      }
    }
    for (var j = 0u; j < 2u; j++) {
      let bit = 1 << j; let p = ${H.strikers}u + 2u * j; let row = ${w.links}u + j;
      if ((s[${H.fired}] & bit) == 0 || (s[${H.spent}] & bit) != 0) { continue; }
      let x = add(dot(row), -s[p]);
      if (x <= 0 && (s[${H.touch}] & bit) != 0) { s[${H.spent}] |= bit; continue; }
      let f = mul(${a.contact_stiffness}, max(0, x));
      if (x > 0) { s[${H.touch}] |= bit; }
      scatter(row, -f);
      s[p + 1u] = add(s[p + 1u], f / ${a.striker_mass});
      s[p] = add(s[p], s[p + 1u]);
    }
    for (var i = 0u; i < ${n}u; i++) {
      let p = ${H.modes}u + 2u * i;
      s[p + 1u] = add(s[p + 1u], force[i]);
      s[p] = add(s[p], s[p + 1u]);
    }
    s[${H.age}] += 1;
  }
}
`;
}

export async function gpu(w, device) {
  const { asset: a } = w;
  const module = device.createShaderModule({ code: source(w) });
  const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
  if (errors.length) throw new Error(errors.map(m => m.message).join('\n'));
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto', compute: { module, entryPoint: 'main' },
  });
  const U = GPUBufferUsage;
  const upload = (v, usage) => {
    const b = device.createBuffer({ size: v.byteLength, usage: usage | U.COPY_DST });
    device.queue.writeBuffer(b, 0, v);
    return b;
  };
  const state = upload(w.state, U.STORAGE | U.COPY_SRC);
  const data = upload(Int32Array.from([...a.bonds.flat(), ...a.ports.flat(),
    ...a.modes.flatMap(m => m.slice(0, 3)), ...a.limits]), U.STORAGE);
  const input = upload(new Int32Array(2), U.STORAGE);
  const read = device.createBuffer({ size: w.state.byteLength, usage: U.COPY_DST | U.MAP_READ });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: [state, data, input].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  return {
    async step(mask, reset = false, fracture = true) {
      if (reset) {
        device.queue.writeBuffer(state, 0, w.state);
        return w.state.slice();
      }
      device.queue.writeBuffer(input, 0, new Int32Array([mask, Number(fracture)]));
      const cmd = device.createCommandEncoder();
      const pass = cmd.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
      pass.end();
      cmd.copyBufferToBuffer(state, 0, read, 0, w.state.byteLength);
      device.queue.submit([cmd.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const snapshot = new Int32Array(read.getMappedRange()).slice();
      read.unmap();
      return snapshot;
    },
    destroy() { for (const b of [state, data, input, read]) b.destroy(); },
  };
}
