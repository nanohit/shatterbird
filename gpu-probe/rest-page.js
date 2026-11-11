// Minimal harness for the phase 1d rest solver: window.runRest(opts) -> result object.
'use strict';
async function runRest(opts = {}) {
  const out = document.getElementById('o'), log = m => { out.textContent += '\n' + m; };
  const R = { ua: navigator.userAgent, when: new Date().toISOString(), errors: [] };
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); if (!adapter) { R.error = 'no adapter'; return R; }
    const inf = adapter.info || {}; R.adapter = { vendor: inf.vendor, architecture: inf.architecture, device: inf.device, description: inf.description };
    const dev = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30), maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30), maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 10) } });
    dev.addEventListener('uncapturederror', e => R.errors.push(e.error.message));
    const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const mk = (data, usage = SU) => { const b = dev.createBuffer({ size: data.byteLength, usage }); dev.queue.writeBuffer(b, 0, data); return b; };
    const pipe = async code => { const module = dev.createShaderModule({ code }); const ci = await module.getCompilationInfo(); const er = ci.messages.filter(m => m.type === 'error'); if (er.length) throw new Error(er.map(m => m.lineNum + ':' + m.linePos + ' ' + m.message).join(' | ')); return dev.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } }); };
    const read = async (buf, bytes) => { const r = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); const e = dev.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, r, 0, bytes); dev.queue.submit([e.finish()]); await r.mapAsync(GPUMapMode.READ); const a = new Int32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return a; };
    const timed = async (enc, reps = 5) => { let best = 1e99; for (let k = 0; k < reps; k++) { const cmds = enc(); await dev.queue.onSubmittedWorkDone(); const t0 = performance.now(); dev.queue.submit(cmds); await dev.queue.onSubmittedWorkDone(); best = Math.min(best, performance.now() - t0); } return best; };
    R.rest = await REST.stage({ dev, mk, pipe, read, timed, log }, opts);
    dev.destroy();
  } catch (e) { R.error = String(e && e.message || e); }
  return R;
}
