// node gpu-probe/worker.mjs <local|win> <page.html> <function> <out.json> [json opts]
// Runs window.<function>(opts) in headless Chrome with hardware WebGPU on this Mac ("local") or on the
// NVIDIA worker reached as `ssh shatterbird-win` ("win"). The page is always served from this
// checkout: for "win" the static server and Chrome's debugging port travel through SSH tunnels, so
// the worker needs nothing but Chrome. Console output of the page is echoed as progress; binary
// dumps the page POSTs to /save/<name> land in gpu-probe/out/. The result gets a `worker` block with
// the Chrome version and the GPU driver as the operating system reports it.
import { spawn, execFileSync } from 'node:child_process';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const here = dirname(fileURLToPath(import.meta.url)), [target, page, fn, outp, popts = '{}'] = process.argv.slice(2);
if (!['local', 'win'].includes(target) || !fn) { console.error('usage: worker.mjs <local|win> <page.html> <function> <out.json> [json opts]'); process.exit(2); }
const HTTP = target === 'win' ? 9363 : 9373, DBG = target === 'win' ? 9364 : 9374, DBG_LOCAL = target === 'win' ? 9464 : 9374, HOST = 'shatterbird-win', MARK = 'sb-worker-profile', STAMP = Date.now().toString(36);
const flags = ['--headless=new', '--no-first-run', '--no-default-browser-check', '--enable-unsafe-webgpu', '--enable-webgpu-developer-features', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', `--remote-debugging-port=${DBG}`];
const sh = (cmd, args) => { try { return execFileSync(cmd, args, { encoding: 'utf8', timeout: 30000 }).trim(); } catch (e) { return 'error: ' + String(e.message).slice(0, 200); } };
const ssh = cmd => sh('ssh', ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST, cmd]);
const killRemote = () => ssh(`powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \\"Name='chrome.exe'\\" | Where-Object { $_.CommandLine -like '*${MARK}*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"`);
const worker = { target };
const server = spawn('python3', [join(here, 'serve.py'), String(HTTP)], { cwd: here, stdio: 'ignore' });
let browser, profile, ws;
try {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const launch = () => {
    if (target === 'win') {
      const chrome = '"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"';
      return spawn('ssh', ['-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=15', '-L', `${DBG_LOCAL}:127.0.0.1:${DBG}`, '-R', `${HTTP}:127.0.0.1:${HTTP}`, HOST,
        `${chrome} ${flags.join(' ')} --user-data-dir=%TEMP%\\${MARK}-${STAMP}-${attempt} http://127.0.0.1:${HTTP}/${page}`], { stdio: 'ignore' });
    }
    return spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [...flags, '--use-angle=metal', `--user-data-dir=${profile}`, `http://127.0.0.1:${HTTP}/${page}`], { stdio: 'ignore' });
  };
  if (target === 'win') {
    killRemote(); await sleep(1500);
    worker.host = ssh('hostname'); worker.os = ssh('ver');
    worker.gpu_driver = ssh('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader');
    worker.video_controllers = ssh('powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + \' | \' + $_.DriverVersion }"');
  } else {
    profile = await mkdtemp(join(tmpdir(), 'shatterbird-worker-'));
    worker.host = sh('hostname', []); worker.os = sh('sw_vers', ['-productVersion']);
    worker.gpu_driver = sh('sh', ['-c', "system_profiler SPDisplaysDataType | grep -E 'Chipset Model|Metal' | tr -s ' ' | tr '\\n' ';'"]);
  }
  // Chrome on the worker sometimes comes up without the GPU right after a previous instance was killed;
  // the hardware adapter is required, so the browser is relaunched until it is there (never software).
  let attempt = 0, evaluate, call;
  for (;;) {
    attempt++; browser = launch();
    let tabs;
    for (let i = 0; i < 200; i++) { try { tabs = await (await fetch(`http://127.0.0.1:${DBG_LOCAL}/json/list`)).json(); if (tabs.some(t => t.type === 'page')) break; } catch { } await sleep(150); }
    const tab = tabs?.find(t => t.type === 'page'); if (!tab) throw new Error('Chrome did not start on ' + target);
    worker.chrome = (await (await fetch(`http://127.0.0.1:${DBG_LOCAL}/json/version`)).json()).Browser;
    ws = new WebSocket(tab.webSocketDebuggerUrl.replace(/127\.0\.0\.1:\d+/, `127.0.0.1:${DBG_LOCAL}`)); await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
    let id = 0; const pending = new Map();
    ws.onmessage = e => { const m = JSON.parse(e.data);
      if (m.id) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(m.error) : p.resolve(m.result); }
      else if (m.method === 'Runtime.consoleAPICalled') console.error(`[${target}] ` + m.params.args.map(a => a.value ?? a.description ?? '').join(' ').slice(0, 400));
      else if (m.method === 'Runtime.exceptionThrown') console.error(`[${target}] exception: ` + JSON.stringify(m.params.exceptionDetails).slice(0, 400)); };
    call = (method, params = {}) => new Promise((resolve, reject) => { const n = ++id; pending.set(n, { resolve, reject }); ws.send(JSON.stringify({ id: n, method, params })); });
    evaluate = async expression => { const r = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails).slice(0, 800)); return r.result.value; };
    await call('Runtime.enable');
    await evaluate(`new Promise((r,j)=>{const ok=()=>typeof ${fn.match(/^[A-Za-z_$][\w$]*/)?.[0] || 'undefined'}!=='undefined';if(ok())r();else window.addEventListener('load',()=>ok()?r():j('function missing'),{once:true});setTimeout(()=>j('load timeout'),20000)})`);
    const tries = await evaluate(`(async()=>{for(let i=0;i<8;i++){const a=navigator.gpu&&await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(a&&!(a.info&&a.info.isFallbackAdapter))return i;await new Promise(r=>setTimeout(r,500))}return -1})()`);
    if (tries >= 0) { worker.launch_attempts = attempt; break; }
    console.error(`[${target}] no hardware adapter on attempt ${attempt}, relaunching`);
    ws.close(); browser.kill('SIGTERM'); if (target === 'win') killRemote(); await sleep(4000);
    if (attempt >= 6) throw new Error('no hardware WebGPU adapter on ' + target);
  }
  const t0 = Date.now(), result = await evaluate(`(${fn})(${popts})`);
  const outv = (result && typeof result === 'object') ? { worker: { ...worker, wall_s: (Date.now() - t0) / 1000 }, ...result } : { worker, result };
  if (outp && outp !== '-') await writeFile(outp, JSON.stringify(outv, null, 1));
  console.log(JSON.stringify(outv));
} catch (e) { console.error('worker failed: ' + (e.message || JSON.stringify(e))); process.exitCode = 1; }
finally { ws?.close(); browser?.kill('SIGTERM'); server.kill('SIGTERM'); if (target === 'win') { killRemote(); ssh(`cmd /c "for /d %d in (%TEMP%\\${MARK}-*) do rmdir /s /q %d"`); } await new Promise(r => setTimeout(r, 300)); if (profile) await rm(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 }); }
