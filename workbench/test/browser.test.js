import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {chromium} from '@playwright/test';

test('browser saves study, runs project shell, compares evidence, and safely displays artifacts',async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(),'sim-browser-')); const port = 14318;
  for (const [name,count,validation] of [['baseline',10,null],['reminders',14,{validated:false,checks:[{name:'capacity',status:'FAIL'}]}]]) {
    const folder = path.join(root,'outputs',name); await fs.mkdir(folder,{recursive:true});
    await fs.writeFile(path.join(folder,'config.json'),JSON.stringify({experiment_name:name,seed:42}));
    await fs.writeFile(path.join(folder,'metrics.json'),JSON.stringify({patients_seen:count}));
    await fs.writeFile(path.join(folder,'report.md'),'<script>window.artifactExecuted=true</script>');
    if (validation) await fs.writeFile(path.join(folder,'validation.json'),JSON.stringify(validation));
  }
  const server = spawn(process.execPath,['server.js'],{cwd:new URL('..',import.meta.url),env:{...process.env,SIM_PROJECT:root,PORT:String(port)}});
  t.after(async () => {server.kill(); await fs.rm(root,{recursive:true,force:true});});
  const url = await new Promise((resolve,reject) => {
    let text = ''; const timer = setTimeout(() => reject(new Error('Server start timeout')),10000);
    server.stdout.on('data',d => {text += d; const match = text.match(/http:\/\/127.0.0.1:\d+\/#[a-f0-9]+/); if (match) {clearTimeout(timer);resolve(match[0]);}});
  });
  const browser = await chromium.launch({channel:process.env.PLAYWRIGHT_CHANNEL || 'chrome'}); t.after(() => browser.close());
  const page = await browser.newPage({viewport:{width:1512,height:1100}}); const errors = []; page.on('pageerror',error => errors.push(error.message));
  await page.goto(url); await page.getByText('Start with a new study.',{exact:false}).waitFor(); await page.getByRole('button',{name:'New study'}).click();
  await page.getByRole('heading',{name:'Untitled study',exact:true}).waitFor();
  await page.getByLabel('Study title',{exact:true}).fill('Reminder capacity study');
  await page.getByLabel('Decision question').fill('Can reminders improve access within current capacity?');
  await page.getByRole('button',{name:'Save brief'}).click();
  await page.getByRole('heading',{name:'Reminder capacity study',exact:true}).waitFor();
  await page.reload(); await page.getByRole('heading',{name:'Reminder capacity study',exact:true}).waitFor();
  await page.getByLabel('Agent',{exact:true}).selectOption('shell'); await page.getByRole('button',{name:'Start session'}).click();
  await page.getByText('shell session',{exact:true}).waitFor();
  await page.locator('.xterm-helper-textarea').focus(); await page.keyboard.type('pwd'); await page.keyboard.press('Enter');
  await page.waitForFunction(project => document.querySelector('#terminal').textContent.includes(project), root);
  await page.getByRole('tab',{name:'Experiments'}).click(); await page.getByLabel('Show',{exact:true}).selectOption('project');
  await page.getByText('CHECKS FAILED',{exact:true}).waitFor();
  assert.equal(await page.getByText('VALIDATION INCOMPLETE / NOT RUN',{exact:true}).count(),1);
  const boxes = page.getByRole('checkbox'); await boxes.nth(0).check(); await boxes.nth(1).check();
  assert.match(await page.locator('#comparison').innerText(),/10/); assert.match(await page.locator('#comparison').innerText(),/14/);
  await page.getByRole('button',{name:'report.md',exact:true}).first().click();
  await page.locator('#artifact').waitFor({state:'visible'});
  assert.match(await page.locator('#artifact-content').innerText(),/<script>/);
  assert.equal(await page.evaluate(() => window.artifactExecuted),undefined);
  await page.getByRole('tab',{name:'Study brief'}).click();
  await fs.mkdir(new URL('../test-results/',import.meta.url),{recursive:true});
  await page.screenshot({path:new URL('../test-results/workbench.png',import.meta.url).pathname,fullPage:true});
  page.once('dialog',dialog => dialog.accept()); await page.getByRole('button',{name:'End session'}).click();
  // Stop the server in teardown; agent processes are terminated by its shutdown handler.
  assert.deepEqual(errors,[]);
});
