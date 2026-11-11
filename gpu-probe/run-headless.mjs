// Isolated local Chrome profile; existing user tabs/profile are never touched.
// node gpu-probe/run-headless.mjs <directory containing exported 1k mesh/index>
// Reuses the real 3D face/history, cell integration and coarse-operation kernels.
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,copyFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
const dir=resolve(process.argv[2]);
const profile=await mkdtemp(join(tmpdir(),'shatterbird-gpu-'));
const chrome='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
for(const name of ['probe.html','probe.js','fixed.js'])await copyFile(new URL(name,import.meta.url),join(dir,name));
const server=spawn('python3',[new URL('serve.py',import.meta.url).pathname,'9333'],{cwd:dir,stdio:'ignore'});
const browser=spawn(chrome,['--headless=new','--no-first-run','--no-default-browser-check','--enable-unsafe-webgpu','--use-angle=metal','--remote-debugging-port=9334',`--user-data-dir=${profile}`,'http://127.0.0.1:9333/probe.html'],{stdio:'ignore'});
let ws;
try{
 let tabs;
 for(let i=0;i<100;i++){
  try{tabs=await(await fetch('http://127.0.0.1:9334/json/list')).json();if(tabs.some(t=>t.type==='page'))break;}catch{}
  await new Promise(r=>setTimeout(r,100));
 }
 const tab=tabs?.find(t=>t.type==='page');if(!tab)throw new Error('isolated Chrome did not start');
 ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
 let id=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
 const evaluate=async expression=>{
  const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
  if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;
 };
 await evaluate(`new Promise((r,j)=>{if(window.runProbe)r();else window.addEventListener('load',r,{once:true});setTimeout(()=>j('load timeout'),15000)})`);
 await evaluate(await readFile(new URL('fixed.js',import.meta.url),'utf8'));
 const result=await evaluate(`(async()=>{
  await runProbe(['1k'],[100,500],5,2,100,'audit-probe.json');
  await runH(['1k'],[100,500],5,2,100,'audit-history.json');
  await runFixed(['1k'],100,5,2,'audit',FIX,[100,500]);
  return {scope:'single Apple GPU; existing 3D kernels, not the adaptive engine',probe:window.__probe,float_history:window.__h,fixed:window.__fx};
 })()`);
 result.kernel_sha256={};
 for(const name of ['probe.js','fixed.js'])result.kernel_sha256[name]=createHash('sha256').update(await readFile(join(dir,name))).digest('hex');
 await writeFile(join(dir,'audit.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify(result));
}finally{
 ws?.close();browser.kill('SIGTERM');server.kill('SIGTERM');
 // Give Chrome time to release the dedicated profile before removing it.
 await new Promise(r=>setTimeout(r,300));await rm(profile,{recursive:true,force:true,maxRetries:3,retryDelay:100});
}
