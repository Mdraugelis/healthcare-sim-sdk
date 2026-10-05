import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {WebSocket} from 'ws';

test('local server authenticates HTTP and websocket, and executes a real project PTY',async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(),'sim-server-'));
  const port = 14317; const origin = `http://127.0.0.1:${port}`;
  const child = spawn(process.execPath,['server.js'],{cwd:new URL('..',import.meta.url),env:{...process.env,PORT:String(port),SIM_PROJECT:root}});
  t.after(async () => {child.kill(); await fs.rm(root,{recursive:true,force:true});});
  let output = ''; let stderr = ''; child.stderr.on('data',d => stderr += d);
  const token = await new Promise((resolve,reject) => {
    const timer = setTimeout(() => reject(new Error(`Server start timed out: ${stderr}`)),10000);
    child.stdout.on('data',data => {output += data; const match = output.match(/#([a-f0-9]{64})/); if (match) {clearTimeout(timer); resolve(match[1]);}});
    child.on('error',reject);
  });
  assert.equal((await fetch(origin+'/api/studies')).status,401);
  assert.equal((await fetch(origin+'/api/login',{method:'POST',headers:{'Content-Type':'application/json',Origin:'http://evil.example'},body:JSON.stringify({token})})).status,403);
  const login = await fetch(origin+'/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});
  const cookie = login.headers.get('set-cookie').split(';')[0]; const headers = {Cookie:cookie,'Content-Type':'application/json'};
  const study = await (await fetch(origin+'/api/studies',{method:'POST',headers,body:JSON.stringify({title:'PTY smoke'})})).json();
  assert.equal((await fetch(origin+'/api/studies',{headers})).status,200);
  const reportDir = path.join(root,'local','reports'); await fs.mkdir(reportDir,{recursive:true});
  await fs.writeFile(path.join(reportDir,'example.html'),'<h1>Capacity report</h1><script>document.body.dataset.ready="yes"</script>');
  assert.equal((await fetch(origin+'/api/report?path=local/reports/example.html')).status,401);
  const preview = await fetch(origin+'/api/report?path=local/reports/example.html',{headers});
  assert.equal(preview.status,200);
  assert.match(preview.headers.get('content-type'),/text\/html/);
  assert.match(preview.headers.get('content-security-policy'),/sandbox allow-scripts;/);
  assert.doesNotMatch(preview.headers.get('content-security-policy'),/allow-same-origin/);
  assert.match(await preview.text(),/Capacity report/);
  const reports = await (await fetch(origin+'/api/reports',{headers})).json();
  assert.equal(reports.reports[0].path,'local/reports/example.html');
  const unauth = new WebSocket(origin.replace('http','ws')+'/terminal?study='+study.id+'&provider=shell',{origin});
  await new Promise(resolve => {unauth.on('error',resolve);});
  const ws = new WebSocket(origin.replace('http','ws')+'/terminal?study='+study.id+'&provider=shell',{origin,headers:{Cookie:cookie}});
  let terminal = '';
  const executed = new Promise((resolve,reject) => {
    const timeout = setTimeout(() => reject(new Error(`PTY did not execute: ${terminal}`)),10000);
    ws.on('message',raw => {
      const msg = JSON.parse(raw); if (msg.type === 'error') {clearTimeout(timeout); reject(new Error(msg.error));}
      if (msg.type === 'connected') ws.send(JSON.stringify({type:'input',data:'pwd\r'}));
      if (msg.type === 'output') {terminal += msg.data; if (terminal.includes(root)) {clearTimeout(timeout); resolve();}}
    }); ws.on('error',reject);
  });
  await executed;
  ws.send(JSON.stringify({type:'resize',cols:82,rows:24}));
  ws.send(JSON.stringify({type:'stop'})); ws.close();
});
