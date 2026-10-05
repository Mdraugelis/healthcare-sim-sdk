import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {Store,fields} from '../store.js';

async function fixture(t) { const root = await fs.mkdtemp(path.join(os.tmpdir(),'sim-workbench-')); t.after(() => fs.rm(root,{recursive:true,force:true})); return {root,store:new Store(root)}; }
test('study persists editable context and rejects stale revisions',async t => {
  const {store} = await fixture(t); const study = await store.create({title:'Capacity decision',question:'Would two nurses help?'});
  const saved = await store.save(study.id,{...study,assumptions:'Treatment halves risk'},study.revision);
  assert.equal((await store.read(study.id)).assumptions,'Treatment halves risk');
  assert.match(await fs.readFile(path.join(store.studyPath(study.id),'brief.md'),'utf8'),/Treatment halves risk/);
  await assert.rejects(store.save(study.id,study,study.revision),/changed on disk/);
  assert.equal(saved.revision,2); assert.equal((await store.list()).length,1);
});
test('artifact access refuses traversal and symlinks outside private output roots',async t => {
  const {root,store} = await fixture(t); await fs.mkdir(path.join(root,'outputs'));
  await fs.writeFile(path.join(root,'secret.json'),'{"secret":true}');
  await fs.symlink(path.join(root,'secret.json'),path.join(root,'outputs/link.json'));
  await assert.rejects(store.safeFile('outputs/../secret.json'),/escapes/);
  await assert.rejects(store.safeFile('outputs/link.json'),/escapes/);
  await assert.rejects(store.safeFile('secret.json'),/must be under/);
});
test('run discovery preserves missing and failed evidence and reports corrupt artifacts',async t => {
  const {root,store} = await fixture(t);
  for (const [name,validation] of [['missing',null],['failed',{validated:false,checks:[{name:'capacity',status:'FAIL'}]}]]) {
    const dir = path.join(root,'outputs',name); await fs.mkdir(dir,{recursive:true});
    await fs.writeFile(path.join(dir,'metrics.json'),'{"patients_seen":12}');
    await fs.writeFile(path.join(dir,'config.json'),'{"scenario":"reminders","seed":42}');
    if (validation) await fs.writeFile(path.join(dir,'validation.json'),JSON.stringify(validation));
  }
  const bad = path.join(root,'outputs/bad'); await fs.mkdir(bad); await fs.writeFile(path.join(bad,'metrics.json'),'broken');
  const {runs,warnings} = await store.runs();
  assert.equal(runs.length,3); assert.equal(runs.find(r => r.path.endsWith('missing')).validation,null);
  assert.equal(runs.find(r => r.path.endsWith('failed')).validation.checks[0].status,'FAIL');
  assert.ok(warnings.some(w => w.includes('bad/metrics.json')));
});

test('HTML reports are discovered without metrics and sorted by update time', async t => {
  const {root,store} = await fixture(t);
  await fs.mkdir(path.join(root,'local/study'),{recursive:true});
  await fs.mkdir(path.join(root,'reports'),{recursive:true});
  await fs.writeFile(path.join(root,'local/study/report.html'),'<h1>Study</h1>');
  await fs.writeFile(path.join(root,'reports/older.htm'),'<h1>Older</h1>');
  await fs.utimes(path.join(root,'reports/older.htm'),new Date(0),new Date(0));
  const {reports} = await store.reports();
  assert.deepEqual(reports.map(r => r.path),['local/study/report.html','reports/older.htm']);
  assert.equal((await store.runs()).runs.length,0);
  await fs.symlink(path.join(root,'package.html'),path.join(root,'local/study/escape.html'));
  await fs.writeFile(path.join(root,'package.html'),'<h1>Outside</h1>');
  await assert.rejects(store.safeFile('local/study/escape.html'),/escapes/);
});
