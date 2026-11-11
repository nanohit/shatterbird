// node gpu-probe/run-det.mjs <tag> [extra chrome flags...]  -> results/phase1/det_<tag>.json
// Isolated headless Chrome profile; the user's browser profile and tabs are never touched.
import {spawn} from 'node:child_process';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const here=dirname(fileURLToPath(import.meta.url)),tag=process.argv[2]||'chrome',extra=process.argv.slice(3);
const profile=await mkdtemp(join(tmpdir(),'shatterbird-det-'));
const chrome='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const server=spawn('python3',[join(here,'serve.py'),'9343'],{cwd:here,stdio:'ignore'});
const browser=spawn(chrome,['--headless=new','--no-first-run','--no-default-browser-check','--enable-unsafe-webgpu','--use-angle=metal','--remote-debugging-port=9344',`--user-data-dir=${profile}`,...extra,'http://127.0.0.1:9343/det.html'],{stdio:'ignore'});
let ws;
try{
 let tabs;
 for(let i=0;i<100;i++){try{tabs=await(await fetch('http://127.0.0.1:9344/json/list')).json();if(tabs.some(t=>t.type==='page'))break;}catch{}await new Promise(r=>setTimeout(r,100));}
 const tab=tabs?.find(t=>t.type==='page');if(!tab)throw new Error('isolated Chrome did not start');
 ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
 let id=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
 const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await evaluate(`new Promise((r,j)=>{if(window.runAll)r();else window.addEventListener('load',r,{once:true});setTimeout(()=>j('load timeout'),15000)})`);
 const result=await evaluate(`runAll(${JSON.stringify({tag,...JSON.parse(process.env.DET_OPTS||"{}")})})`);
 await writeFile(join(here,'..','results','phase1',`det_${tag}.json`),JSON.stringify(result,null,1));
 console.log(JSON.stringify(result));
}finally{ws?.close();browser.kill('SIGTERM');server.kill('SIGTERM');await new Promise(r=>setTimeout(r,300));await rm(profile,{recursive:true,force:true,maxRetries:3,retryDelay:100});}
