// node gpu-probe/run-page.mjs <page.html> <function> <out.json> [json opts]  -> runs window.<function>(opts) in headless Chrome
// Isolated headless Chrome profile; the user's browser profile and tabs are never touched.
import {spawn} from 'node:child_process';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const here=dirname(fileURLToPath(import.meta.url)),page=process.argv[2],fn=process.argv[3],outp=process.argv[4],popts=process.argv[5]||'{}',extra=[];
const profile=await mkdtemp(join(tmpdir(),'shatterbird-det-'));
const chrome='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const server=spawn('python3',[join(here,'serve.py'),'9353'],{cwd:here,stdio:'ignore'});
const browser=spawn(chrome,['--headless=new','--no-first-run','--no-default-browser-check','--enable-unsafe-webgpu','--use-angle=metal','--remote-debugging-port=9354',`--user-data-dir=${profile}`,...extra,'http://127.0.0.1:9353/'+page],{stdio:'ignore'});
let ws;
try{
 let tabs;
 for(let i=0;i<100;i++){try{tabs=await(await fetch('http://127.0.0.1:9354/json/list')).json();if(tabs.some(t=>t.type==='page'))break;}catch{}await new Promise(r=>setTimeout(r,100));}
 const tab=tabs?.find(t=>t.type==='page');if(!tab)throw new Error('isolated Chrome did not start');
 ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
 let id=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
 const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await evaluate(`new Promise((r,j)=>{if(window[${JSON.stringify(fn)}])r();else window.addEventListener('load',r,{once:true});setTimeout(()=>j('load timeout'),15000)})`);
 const result=await evaluate(`${fn}(${popts})`);
 await writeFile(outp,JSON.stringify(result,null,1));
 console.log(JSON.stringify(result));
}finally{ws?.close();browser.kill('SIGTERM');server.kill('SIGTERM');await new Promise(r=>setTimeout(r,300));await rm(profile,{recursive:true,force:true,maxRetries:3,retryDelay:100});}
