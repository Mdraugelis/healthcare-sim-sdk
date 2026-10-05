import express from 'express';
import { WebSocketServer } from 'ws';
import * as pty from 'node-pty';
import path from 'node:path';
import fs from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { randomBytes } from 'node:crypto';
import { Store } from './store.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = await fs.realpath(process.env.SIM_PROJECT || path.join(here, '..'));
const store = new Store(root);
const token = randomBytes(32).toString('hex');
const port = Number(process.env.PORT || 4317);
const origin = `http://127.0.0.1:${port}`;
const app = express();
app.disable('x-powered-by');
app.use((req,res,next) => {
  if (req.headers.host !== `127.0.0.1:${port}` || (req.headers.origin && req.headers.origin !== origin)) return res.sendStatus(403);
  res.set('Cache-Control','no-store');
  res.set('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' ws://127.0.0.1:" + port + "; object-src 'none'; frame-ancestors 'none'");
  next();
});
app.use(express.json({limit:'1mb'}));
const authenticated = req => (req.headers.cookie || '').split(';').some(c => c.trim() === `sim_session=${token}`);
app.post('/api/login', (req,res) => {
  if (req.body.token !== token) return res.sendStatus(403);
  res.setHeader('Set-Cookie', `sim_session=${token}; HttpOnly; SameSite=Strict; Path=/`); res.json({ok:true});
});
app.use('/api', (req,res,next) => authenticated(req) ? next() : res.sendStatus(401));
app.get('/api/project', async (req,res) => res.json({root, providers:['codex','claude','shell']}));
app.get('/api/studies', async (req,res) => res.json(await store.list()));
app.post('/api/studies', async (req,res) => res.json(await store.create(req.body)));
app.get('/api/studies/:id', async (req,res) => res.json(await store.read(req.params.id)));
app.put('/api/studies/:id', async (req,res) => res.json(await store.save(req.params.id, req.body, req.body.revision)));
app.get('/api/runs', async (req,res) => res.json(await store.runs()));
app.get('/api/reports', async (req,res) => res.json(await store.reports()));
app.get('/api/report', async (req,res) => {
  const file = await store.safeFile(req.query.path);
  if (!/\.html?$/i.test(file)) throw new Error('Expected an HTML report');
  // Separate response policy: opaque sandbox origin with inline report interactivity.
  res.set('Content-Security-Policy', "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; form-action 'none'; frame-ancestors 'self'");
  res.type('html').send(await fs.readFile(file,'utf8'));
});
app.get('/api/artifact', async (req,res) => {
  const file = await store.safeFile(req.query.path);
  res.type('text/plain').send(await fs.readFile(file,'utf8'));
});
app.use('/vendor/xterm',express.static(path.join(here,'node_modules/@xterm/xterm')));
app.use('/vendor/fit',express.static(path.join(here,'node_modules/@xterm/addon-fit')));
app.use(express.static(path.join(here,'public')));
app.use((error,req,res,next) => res.status(error.status || (error.code === 'ENOENT' ? 404 : 400)).json({error:error.message}));

const sessions = new Map();
const server = app.listen(port,'127.0.0.1', () => console.log(`Simulation Workbench\n${origin}/#${token}\nProject: ${root}`));
const wss = new WebSocketServer({noServer:true, maxPayload: 128 * 1024});
server.on('upgrade',(req,socket,head) => {
  if (!authenticated(req) || req.headers.origin !== origin || req.headers.host !== `127.0.0.1:${port}`) { socket.destroy(); return; }
  wss.handleUpgrade(req,socket,head, ws => wss.emit('connection',ws,req));
});
wss.on('connection', async (ws,req) => {
  let session;
  const send = message => { if (ws.readyState === 1) ws.send(JSON.stringify(message)); };
  try {
    const url = new URL(req.url,origin);
    if (url.pathname !== '/terminal') throw new Error('Unknown terminal endpoint');
    const id = url.searchParams.get('study'); const provider = url.searchParams.get('provider');
    await store.read(id);
    if (!['codex','claude','shell'].includes(provider)) throw new Error('Unknown agent');
    const key = `${id}:${provider}`;
    session = sessions.get(key);
    if (!session) {
      const brief = path.join(store.studyPath(id),'brief.md');
      const prompt = `Work with me on the study in ${brief}. Read CLAUDE.md and .claude/agents/sim-guide.md for the SDK protocol. You have the full project workspace and your normal tools. Reason freely about the question; use analytical methods first when appropriate, and extend SDK scenarios when needed. The brief is editable context, not a rigid workflow. Discuss missing assumptions with me. Keep institutional information and study outputs under local/. Generate self-contained HTML reports with inline styles, scripts, and embedded images (no external assets) in this study's outputs folder so the workbench Reports panel can render them. Put this study's experiments under ${path.join(store.studyPath(id),'outputs')}. Use finalize_experiment and disclose verification scope; saving results is not validation. Preserve code, resolved configs, seeds, SDK source version, and environment information with the study. Do not infer clinical effectiveness from simulated results.`;
      const executable = provider === 'shell' ? (process.env.SHELL || '/bin/zsh') : (provider === 'codex' ? process.env.CODEX_BIN || path.join(here,'node_modules/.bin/codex') : process.env.CLAUDE_BIN || 'claude');
      const args = provider === 'shell' ? ['-l'] : [prompt];
      const env = Object.fromEntries(Object.entries(process.env).filter(([k,v]) => typeof v === 'string' && !['CLAUDECODE','CLAUDE_CODE_ENTRYPOINT','npm_config_prefix','NPM_CONFIG_PREFIX'].includes(k)));
      // Make the project's Python environment available to the unmodified agent.
      env.PATH = `${path.join(root,'.venv/bin')}:${env.PATH}`;
      const proc = pty.spawn(executable,args,{name:'xterm-256color',cols:100,rows:30,cwd:root,env});
      session = {proc,clients:new Set(),buffer:'',exited:false}; sessions.set(key,session);
      proc.onData(data => {
        session.buffer = (session.buffer + data).slice(-1024 * 1024);
        for (const client of session.clients) if (client.readyState === 1) client.send(JSON.stringify({type:'output',data}));
      });
      proc.onExit(({exitCode}) => {
        session.exited = true;
        for (const client of session.clients) if (client.readyState === 1) client.send(JSON.stringify({type:'exit',exitCode}));
        sessions.delete(key);
      });
    }
    session.clients.add(ws); send({type:'output',data:session.buffer}); send({type:'connected',provider});
    ws.on('message', raw => {
      try {
        const msg = JSON.parse(raw);
        if (msg.type === 'input' && typeof msg.data === 'string') session.proc.write(msg.data);
        if (msg.type === 'resize' && Number.isInteger(msg.cols) && Number.isInteger(msg.rows)) session.proc.resize(Math.max(20,Math.min(300,msg.cols)),Math.max(5,Math.min(150,msg.rows)));
        if (msg.type === 'stop') session.proc.kill();
      } catch (error) { send({type:'error',error:error.message}); }
    });
    ws.on('close',() => session.clients.delete(ws));
  } catch (error) { send({type:'error',error:`Cannot start agent: ${error.message}. Check the CLI installation and sign in from a terminal.`}); ws.close(); }
});
function shutdown() { for (const session of sessions.values()) session.proc.kill(); wss.close(); server.close(); }
process.on('SIGINT', () => { shutdown(); process.exit(0); });
process.on('SIGTERM', () => { shutdown(); process.exit(0); });
