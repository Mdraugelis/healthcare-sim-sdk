const $ = id => document.getElementById(id);
let study, studies = [], dirty = false, socket;
const comparisons = new Map();
const form = $('brief-form');
form.inert = true; $('save').disabled = true; $('ask-agent').disabled = true;
const term = new window.Terminal({cursorBlink:true,fontSize:12,fontFamily:'Menlo, Consolas, monospace',scrollback:10000,theme:{background:'#17241f',foreground:'#dae4d8',cursor:'#d9ed92'}});
const fit = new window.FitAddon.FitAddon(); term.loadAddon(fit); term.open($('terminal')); fit.fit();
term.writeln('\x1b[32mYour project. Your agent.\x1b[0m\r\n\r\nCreate or select a study, then start a session.\r\nCodex and Claude Code run here with their native tools and controls.');
function notice(message) { $('notice').textContent = message; }
async function api(url, options = {}) {
  const response = await fetch(url, {headers:{'Content-Type':'application/json'},...options});
  if (!response.ok) { const text = await response.text(); let message = text; try { message = JSON.parse(text).error || text; } catch {} throw new Error(message); }
  return response.json();
}
function fields() { return Object.fromEntries(new FormData(form)); }
function drawStudies() {
  $('studies').replaceChildren(...studies.map(s => {
    const b = document.createElement('button'); b.textContent = s.title || 'Untitled study'; b.className = s.id === study?.id ? 'selected' : ''; b.onclick = () => selectStudy(s.id); return b;
  }));
}
async function selectStudy(id) {
  try {
    if (dirty && !window.confirm('Discard unsaved brief changes?')) return;
    study = await api(`/api/studies/${id}`);
    for (const [key,value] of Object.entries(study)) if (form.elements.namedItem(key)) form.elements.namedItem(key).value = value;
    form.inert = false; $('save').disabled = false; $('ask-agent').disabled = false;
    dirty = false; $('saved-state').textContent = 'Saved in local/workbench/studies/'; $('study-title').textContent = study.title || 'Untitled study'; drawStudies();
    if (socket) { socket.close(); socket = null;  $('connect').disabled = false; $('provider').disabled = false; $('stop').disabled = true; $('connection').textContent = 'Detached'; }
    comparisons.clear(); drawComparison(); await refreshRuns(); await refreshReports();
  } catch (error) { notice(error.message); }
}
async function save() {
  if (!study) throw new Error('Create a study first.');
  study = await api(`/api/studies/${study.id}`, {method:'PUT',body:JSON.stringify({...fields(),revision:study.revision})});
  dirty = false; $('saved-state').textContent = 'Saved'; $('study-title').textContent = study.title || 'Untitled study'; studies = await api('/api/studies'); drawStudies(); notice('Brief saved. Ask the agent to reread it when you want to use the changes.');
}
form.addEventListener('input',() => {dirty = true; $('saved-state').textContent = 'Unsaved changes';});
form.addEventListener('submit',event => event.preventDefault());
$('save').onclick = () => save().catch(error => notice(error.message));
$('new-study').onclick = async () => {
  try {
    if (dirty) await save();
    form.inert = true; $('save').disabled = true;
    const created = await api('/api/studies',{method:'POST',body:JSON.stringify({title:'Untitled study'})});
    studies = await api('/api/studies'); await selectStudy(created.id);
  } catch (error) { notice(error.message); }
};
$('ask-agent').onclick = async () => {
  try { if (dirty) await save(); await navigator.clipboard.writeText(`Please reread local/workbench/studies/${study.id}/brief.md. I updated the study context; incorporate the changes and discuss any implications with me.`); notice('Copied. Paste into the agent terminal when ready.'); }
  catch (error) { notice(error.message); }
};
$('connect').onclick = async () => {
  try {
    if (!study) throw new Error('Create a study first.');
    if (dirty) await save();
    const provider = $('provider').value; 
    term.reset(); $('connection').textContent = 'Connecting…'; $('connect').disabled = true; $('provider').disabled = true;
    const ws = new WebSocket(`ws://${location.host}/terminal?study=${study.id}&provider=${provider}`); socket = ws;
    ws.onopen = () => { fit.fit(); term.focus(); };
    ws.onmessage = event => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'output') term.write(msg.data);
      if (msg.type === 'connected') { ws.send(JSON.stringify({type:'resize',cols:term.cols,rows:term.rows})); $('connection').textContent = `${msg.provider} session`; $('stop').disabled = false; }
      if (msg.type === 'error') notice(msg.error);
      if (msg.type === 'exit') { notice(`Agent exited (${msg.exitCode}). Start a session to continue using the CLI’s resume command.`); ws.close(); refreshRuns(); }
    };
    ws.onclose = () => { if (socket !== ws) return; $('connection').textContent = 'Disconnected'; $('connect').disabled = false; $('provider').disabled = false; $('stop').disabled = true; socket = null;  };
    ws.onerror = () => notice('Terminal connection failed. Check that the workbench server is running.');
  } catch (error) { notice(error.message); $('connect').disabled = false; $('provider').disabled = false; }
};
term.onData(data => { if (socket?.readyState === 1) socket.send(JSON.stringify({type:'input',data})); });
new ResizeObserver(() => { fit.fit(); if (socket?.readyState === 1) socket.send(JSON.stringify({type:'resize',cols:term.cols,rows:term.rows})); }).observe($('terminal'));
$('stop').onclick = () => { if (window.confirm('End the agent process? Running work will be interrupted.')) socket?.send(JSON.stringify({type:'stop'})); };
function tab(view) {
  for (const name of ['brief','results','reports']) {
    const active = name === view;
    $(`${name}-view`).hidden = !active;
    $(`${name}-tab`).classList.toggle('active', active);
    $(`${name}-tab`).setAttribute('aria-selected', String(active));
  }
  if (view === 'results') refreshRuns();
  if (view === 'reports') refreshReports();
}
$('brief-tab').onclick = () => tab('brief'); $('results-tab').onclick = () => tab('results');
$('reports-tab').onclick = () => tab('reports');
function el(tag,text,className) { const e = document.createElement(tag); e.textContent = text; if (className) e.className = className; return e; }
function scalar(value) { return typeof value === 'number' ? (Number.isInteger(value) ? String(value) : value.toPrecision(5)) : typeof value === 'object' ? JSON.stringify(value) : String(value); }
async function refreshRuns() {
  try {
    const data = await api('/api/runs');
    const runs = data.runs.filter(run => $('run-scope').value === 'project' || run.path.startsWith(`local/workbench/studies/${study?.id}/outputs`));
    $('run-warnings').textContent = [...data.warnings, ...(data.truncated ? ['Scan limit reached. Some runs are not shown.'] : [])].join('\n');
    $('runs').replaceChildren();
    if (!runs.length) $('runs').append(el('div','No experiments here yet. Ask your agent to explore the question and save experiments to this study’s outputs folder. Existing runs are available under “Entire project”.','empty'));
    for (const run of runs) {
      const card = el('article','','run');
      const compareLabel = el('label','Compare this run'); const compareBox = document.createElement('input'); compareBox.type = 'checkbox'; compareBox.checked = comparisons.has(run.path);
      compareBox.onchange = () => { if (compareBox.checked) comparisons.set(run.path,run); else comparisons.delete(run.path); drawComparison(); }; compareLabel.prepend(compareBox); card.append(compareLabel); card.append(el('h3',run.config?.experiment_name || run.config?.scenario || 'Experiment'),el('div',run.path,'run-path'));
      const checks = run.validation?.checks;
      const failed = checks?.some(c => c.status === 'FAIL' || (c.status == null && c.passed === false));
      const passed = run.validation?.validated === true && Array.isArray(checks) && checks.length > 0 && checks.every(c => c.status === 'PASS' || c.status === 'NOT_APPLICABLE' || (c.status == null && c.passed === true));
      card.append(el('span',failed ? 'CHECKS FAILED' : passed ? 'RECORDED CHECKS PASSED · SEE SCOPE' : 'VALIDATION INCOMPLETE / NOT RUN',`badge ${failed ? 'fail' : passed ? 'pass' : ''}`));
      const table = document.createElement('table'); table.className = 'metrics';
      for (const [key,value] of Object.entries(run.metrics || {})) { const row = document.createElement('tr'); row.append(el('td',key.replaceAll('_',' ')),el('td',scalar(value))); table.append(row); } card.append(table);
      const details = document.createElement('details'); details.append(el('summary','Configuration and validation scope'),el('pre',JSON.stringify({config:run.config,validation:run.validation},null,2))); card.append(details);
      const artifacts = el('div','','artifacts');
      for (const file of run.artifacts) { const b = el('button',file.split('/').pop()); b.onclick = async () => {
        try { if (/\.html?$/i.test(file)) { $('report-scope').value = 'project'; await refreshReports(file); tab('reports'); return; } const response = await fetch(`/api/artifact?path=${encodeURIComponent(file)}`); if (!response.ok) throw new Error(await response.text()); $('artifact-name').textContent = file; $('artifact-content').textContent = await response.text(); $('artifact').hidden = false; $('artifact').scrollIntoView({behavior:'smooth',block:'nearest'}); } catch (error) { notice(error.message); }
      }; artifacts.append(b); } card.append(artifacts); $('runs').append(card);
    }
  } catch (error) { notice(error.message); }
}
$('refresh').onclick = refreshRuns; $('run-scope').onchange = refreshRuns; $('close-artifact').onclick = () => $('artifact').hidden = true;
window.addEventListener('beforeunload',event => { if (dirty) {event.preventDefault(); event.returnValue = '';} });
async function boot() {
  try {
    const token = location.hash.slice(1);
    if (token) { await api('/api/login',{method:'POST',body:JSON.stringify({token})}); history.replaceState(null,'',location.pathname); }
    const project = await api('/api/project'); $('project-path').textContent = project.root;
    studies = await api('/api/studies'); drawStudies(); if (studies.length) await selectStudy(studies[0].id);
    else notice('Start with a new study. You can leave uncertain inputs blank and work them out with the agent.');
  } catch (error) { notice(`Open the complete URL printed by npm start to connect securely. ${error.message}`); }
}
boot();

function drawComparison() {
  const box = $('comparison'); box.replaceChildren(); box.hidden = !comparisons.size;
  if (!comparisons.size) return;
  const runs = [...comparisons.values()]; box.append(el('h3','Run comparison'),el('p','Raw metric values; check configurations, units, populations, and validation scope before comparing.'));
  const table = el('table','','metrics'); const header = document.createElement('tr'); header.append(el('th','Metric'));
  for (const run of runs) header.append(el('th',run.config?.experiment_name || run.path.split('/').pop())); table.append(header);
  for (const key of new Set(runs.flatMap(r => Object.keys(r.metrics || {})))) {
    const row = document.createElement('tr'); row.append(el('td',key.replaceAll('_',' ')));
    for (const run of runs) row.append(el('td',run.metrics?.[key] === undefined ? 'Missing' : scalar(run.metrics[key]))); table.append(row);
  }
  box.append(table);
}

let reportSelection = null;
let reportRefreshPending = false;
async function refreshReports(preferredPath) {
  if (reportRefreshPending) return;
  reportRefreshPending = true;
  const studyId = study?.id;
  try {
    const data = await api('/api/reports');
    if (studyId !== study?.id) return;
    const prefix = `local/workbench/studies/${studyId}/`;
    const reports = data.reports.filter(report => $('report-scope').value === 'project' || report.path.startsWith(prefix));
    const previous = preferredPath || $('report-select').value;
    $('report-select').replaceChildren(...reports.map(report => {
      const option = document.createElement('option'); option.value = report.path;
      option.textContent = $('report-scope').value === 'project' ? report.path : report.path.slice(prefix.length);
      return option;
    }));
    if (reports.some(r => r.path === previous)) $('report-select').value = previous;
    $('report-warnings').textContent = [...data.warnings, ...(data.truncated ? ['Report scan limit reached.'] : [])].join('\n');
    $('report-empty').hidden = reports.length > 0;
    $('report-select').disabled = reports.length === 0;
    if (!reports.length) {
      $('report-preview').hidden = true; $('report-preview').removeAttribute('src'); reportSelection = null;
      $('report-status').textContent = ''; return;
    }
    const selected = reports.find(r => r.path === $('report-select').value);
    if (selected && (reportSelection?.path !== selected.path || reportSelection?.modified !== selected.modified || preferredPath)) {
      reportSelection = selected; showReport(selected);
    }
  } catch (error) { $('report-status').textContent = error.message; }
  finally { reportRefreshPending = false; }
}
function showReport(report) {
  $('report-preview').src = `/api/report?path=${encodeURIComponent(report.path)}&version=${encodeURIComponent(report.modified)}`;
  $('report-preview').hidden = false;
  $('report-status').textContent = `Updated ${new Date(report.modified).toLocaleString()} · Refreshes automatically`;
}
$('refresh-reports').onclick = () => { reportSelection = null; refreshReports(); };
$('report-scope').onchange = () => { reportSelection = null; refreshReports(); };
$('report-select').onchange = () => refreshReports($('report-select').value);
setInterval(() => { if (!$('reports-view').hidden && !document.hidden) refreshReports(); }, 3000);
