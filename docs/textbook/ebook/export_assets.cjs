/* Export the existing offline textbook's static visual outputs for an EPUB.
 * No source HTML or chapter file is modified. Chrome uses a temporary profile.
 * Run with bundled Node and PLAYWRIGHT_MODULE, or a local Playwright install.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const bundledModule = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || (fs.existsSync(bundledModule) ? bundledModule : 'playwright'));
const book = path.resolve(__dirname, '../index.html');
const assetDir = path.join(__dirname, 'assets');
const manifestPath = path.join(__dirname, 'assets_manifest.json');
const chartWidth = 640, compactChartWidth = 360, deviceScaleFactor = 3, background = '#ffffff';
const browserExecutable = process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const textClean = s => s.replace(/\u00a0/g, ' ').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
function dimensions(buffer) {
  assert.equal(buffer.toString('ascii', 1, 4), 'PNG');
  return {width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20)};
}
const captureCSS = `
  *{box-sizing:border-box}html,body{margin:0;padding:0;background:${background};color:#172a30}
  body{font:16px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
  #asset{display:flow-root;margin:0;background:${background};overflow:visible}
  #asset svg{display:block;margin:0;max-width:none}
  #asset.equation-asset{padding:10px 12px}
  #asset.chart-asset svg text{font:16px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#172a30}
  .hero-mobile-flow{display:grid;grid-template-columns:1fr 1fr;gap:12px;font:17px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
  .hero-mobile-flow>div{background:#dcece6;padding:16px;border-radius:4px}
  .hero-mobile-flow strong{display:block;font-size:18px;margin-bottom:5px}
  .hero-mobile-flow b{display:block;margin-top:8px}
  .hero-mobile-flow .flow-wide{grid-column:1/-1}
  .hero-mobile-flow .flow-comparator{background:#eee5d6}
  .hero-mobile-flow .flow-arrow{grid-column:1/-1;text-align:center;padding:0;background:none;color:#55776e}
`;
(async () => {
  fs.mkdirSync(assetDir, {recursive:true});
  fs.mkdirSync(path.join(assetDir, 'compact'), {recursive:true});
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'healthcare-textbook-assets-'));
  const context = await chromium.launchPersistentContext(profile, {
    executablePath:browserExecutable, headless:true,
    viewport:{width:1000,height:900}, deviceScaleFactor,
    acceptDownloads:false,
    args:['--disable-background-networking','--disable-component-update','--no-first-run'],
  });
  const errors=[], remoteRequests=[];
  await context.route(/^https?:\/\//, route => {remoteRequests.push(route.request().url()); return route.abort();});
  try {
    const page = await context.newPage();
    page.on('pageerror', e=>errors.push(e.message));
    await page.goto(pathToFileURL(book).href, {waitUntil:'load'});
    await page.evaluate(()=>document.fonts.ready);
    await page.addStyleTag({content:'.shell{margin:0!important}.book{width:720px!important;max-width:none!important;padding:0 40px!important;margin:0!important}.lab{padding:0!important;border:0!important}.lab .chart{width:640px!important}'});
    await page.evaluate(()=>window.dispatchEvent(new Event('resize')));
    await page.waitForTimeout(250);
    assert.equal(await page.locator('.equation').count(),47,'Expected 47 source equations');
    assert.equal(await page.locator('.lab .lab-result').count(),8,'Expected eight initialized labs');
    const manifest={
      schema_version:1,
      source_html:{path:'../index.html',sha256:sha256(fs.readFileSync(book)),manuscript_identity:await page.locator('html').getAttribute('data-identity')},
      export_settings:{chart_css_width:compactChartWidth,wide_chart_css_width:chartWidth,device_scale_factor:deviceScaleFactor,background,browser:await context.browser().version(),equation_scale:1.5,equation_max_svg_css_width:616,chart_font_css_px:16,static_default_outputs_only:true},
      equations:[],diagrams:[],labs:[],checks:{},
    };
    const capture = await context.newPage();
    capture.on('pageerror',e=>errors.push(e.message));
    async function writeImage(name, html, cssWidth, cssHeight, kind) {
      await capture.setContent(`<!doctype html><html lang="en"><head><meta charset="utf-8"><style>${captureCSS}</style></head><body><div id="asset" class="${kind}" style="width:${cssWidth}px">${html}</div></body></html>`);
      await capture.evaluate(()=>document.fonts.ready);
      if (cssHeight !== null) await capture.locator('#asset>svg').evaluate((svg,height)=>svg.style.height=height+'px',cssHeight);
      assert.equal(await capture.locator('input,select,button,textarea').count(),0,'No interactive controls may enter a static image');
      const box=await capture.locator('#asset').boundingBox();
      assert.ok(box.width>0&&box.height>0,'Visible nonempty asset');
      const destination=path.join(assetDir,name+'.png');
      const buffer=await capture.locator('#asset').screenshot({path:destination,animations:'disabled',omitBackground:false,scale:'device'});
      const size=dimensions(buffer);
      return {path:'assets/'+name+'.png',...size,css_width:box.width,css_height:box.height,sha256:sha256(buffer),bytes:buffer.length};
    }
    const equationData=await page.locator('.equation').evaluateAll(figures=>figures.map(f=>{
      const s=f.querySelector('svg'), v=s.viewBox.baseVal, b=s.getBoundingClientRect();
      return {id:f.id,label:f.querySelector('figcaption').textContent.trim(),latex:f.querySelector('.equation-source code')?.textContent||s.getAttribute('aria-label'),chapter_id:f.closest('.chapter').id,svg:s.outerHTML,width:b.width,view_width:v.width,view_height:v.height};
    }));
    for(const eq of equationData){
      const w=Math.min(616,eq.width*1.5),h=w*eq.view_height/eq.view_width;
      const html=eq.svg.replace('<svg ',`<svg style="width:${w}px;height:${h}px" `);
      const image=await writeImage(eq.id,html,w+24,null,'equation-asset');
      manifest.equations.push({id:eq.id,label:eq.label,latex:eq.latex,chapter_id:eq.chapter_id,alt:eq.latex,image});
    }
    for(const spec of [
      {key:'hero-desktop',selector:'.hero-figure>svg',title:'Two policy worlds and two evidence questions'},
      {key:'engine-branches',selector:'.chapter-diagram>svg',title:'The engine’s factual and counterfactual paths'},
    ]){
      const d=await page.locator(spec.selector).evaluate(s=>({svg:s.outerHTML,description:s.getAttribute('aria-label'),caption:s.parentElement.querySelector('figcaption')?.textContent.trim(),chapter_id:s.closest('.chapter')?.id||null,view_width:s.viewBox.baseVal.width,view_height:s.viewBox.baseVal.height}));
      const h=chartWidth*d.view_height/d.view_width;
      const html=d.svg.replace('<svg ',`<svg style="width:${chartWidth}px;height:${h}px" `);
      const image=await writeImage('diagram-'+spec.key,html,chartWidth,null,'diagram-asset');
      manifest.diagrams.push({key:spec.key,title:spec.title,description:d.description,caption:d.caption,chapter_id:d.chapter_id,source_selector:spec.selector,image});
    }
    const compact=await page.locator('.hero-mobile-flow').evaluate(el=>({html:el.outerHTML,description:el.getAttribute('aria-label')}));
    manifest.diagrams.push({key:'hero-compact',title:'Two policy worlds: compact reading layout',description:compact.description,caption:'The same policy comparison, with larger labels for a narrow reading screen.',chapter_id:null,source_selector:'.hero-mobile-flow',preferred_for_small_screens:true,image:await writeImage('diagram-hero-compact',compact.html,chartWidth,null,'diagram-asset')});
    const labData=await page.locator('[data-lab]').evaluateAll(labs=>labs.map(el=>{
      const normalize=s=>s.replace(/\u00a0/g,' ').replace(/[ \t]+/g,' ').replace(/\n{3,}/g,'\n\n').trim();
      // Read details too: plain-text evidence must not lose the patient table.
      el.querySelectorAll('details').forEach(d=>d.open=true);
      return {
        key:el.dataset.lab,title:el.querySelector('h4').textContent.trim(),description:el.querySelector('.lab-desc').textContent.trim(),chapter_id:el.closest('.chapter').id,
        default_parameters:[...el.querySelectorAll('input,select')].map(input=>{
          const label=input.labels?.[0]?.cloneNode(true);label?.querySelectorAll('output,input,select').forEach(c=>c.remove());
          const output=el.querySelector('#'+input.id+'-value');
          const record={id:input.id,label:label?.textContent.trim()||input.id,value:input.type==='range'?Number(input.value):input.value,display_value:output?.textContent.trim()||input.selectedOptions?.[0]?.textContent.trim()||input.value};
          if(input.type==='range')Object.assign(record,{min:Number(input.min),max:Number(input.max),step:Number(input.step)});
          else record.options=[...input.options].map(o=>({value:o.value,label:o.textContent.trim()}));
          return record;
        }),
        charts:[...el.querySelectorAll('.chart>svg')].map(s=>({title:s.getAttribute('aria-label')||s.querySelector('title')?.textContent,svg:s.outerHTML,width:s.viewBox.baseVal.width,height:s.viewBox.baseVal.height})),
        results_text:normalize(el.querySelector('.lab-result').innerText),
        notes:[...el.querySelectorAll('.lab-note')].map(n=>normalize(n.innerText)),
        tables:[...el.querySelectorAll('.lab-result table')].map(table=>({headers:[...table.querySelectorAll('thead th')].map(x=>x.textContent.trim()),rows:[...table.querySelectorAll('tbody tr')].map(row=>[...row.children].map(x=>x.textContent.trim()))})),
      };
    }));
    for(const lab of labData){
      const charts=[];
      for(let i=0;i<lab.charts.length;i++){
        const c=lab.charts[i],h=chartWidth*c.height/c.width;
        const html=c.svg.replace('<svg ',`<svg style="width:${chartWidth}px;height:${h}px" `);
        charts.push({title:c.title,alt:c.title,image:await writeImage('lab-'+lab.key+(lab.charts.length>1?'-'+(i+1):''),html,chartWidth,null,'chart-asset')});
      }
      const resultsText = lab.key === 'capacity'
        ? lab.results_text.split('Inspect every patient and gain')[0]
        : lab.results_text;
      manifest.labs.push({...lab,charts,results_text:textClean(resultsText)});
    }
    // Redraw the same default outputs at their native narrow-reader width.
    // Raster shrinking a 640px plot to 324px also halves its label text.
    await page.addStyleTag({content:'.lab .chart{width:360px!important}'});
    await page.evaluate(()=>window.dispatchEvent(new Event('resize')));
    await page.waitForTimeout(250);
    for (const lab of manifest.labs) {
      const narrowCharts = await page.locator(`[data-lab="${lab.key}"] .chart>svg`).evaluateAll(elements=>elements.map(svg=>({html:svg.outerHTML,width:svg.viewBox.baseVal.width,height:svg.viewBox.baseVal.height})));
      assert.equal(narrowCharts.length, lab.charts.length, 'Narrow outputs retain every chart');
      for (let i=0; i<narrowCharts.length; i++) {
        const c=narrowCharts[i], h=compactChartWidth*c.height/c.width;
        const html=c.html.replace('<svg ',`<svg style="width:${compactChartWidth}px;height:${h}px" `);
        const chart=lab.charts[i];
        chart.wide_image=chart.image;
        chart.image=await writeImage('compact/lab-'+lab.key+(lab.charts.length>1?'-'+(i+1):''),html,compactChartWidth,null,'chart-asset');
      }
    }
    assert.equal(new Set(manifest.equations.map(e=>e.id)).size,47,'Unique source equation IDs');
    assert.ok(manifest.equations.every(e=>e.latex&&e.alt&&e.chapter_id),'Every equation has accessible source notation');
    assert.ok(manifest.diagrams.every(d=>d.title&&d.description),'Every diagram has accessible text');
    assert.ok(manifest.labs.every(l=>l.charts.every(c=>c.title&&c.alt)),'Every chart has accessible text');
    const images=[...manifest.equations.map(e=>e.image),...manifest.diagrams.map(d=>d.image),...manifest.labs.flatMap(l=>l.charts.flatMap(c=>[c.image,c.wide_image]))];
    for(const image of images){
      const bytes=fs.readFileSync(path.join(__dirname,image.path));
      assert.deepEqual(dimensions(bytes),{width:image.width,height:image.height});
      assert.equal(sha256(bytes),image.sha256);
    }
    manifest.checks={equation_count:manifest.equations.length,diagram_count:manifest.diagrams.length,lab_count:manifest.labs.length,chart_count:manifest.labs.reduce((n,l)=>n+l.charts.length,0),all_images_have_accessible_text:true,all_png_dimensions_and_hashes_valid:true,controls_in_images:0,page_errors:errors,remote_asset_requests:remoteRequests};
    assert.equal(errors.length,0,'No browser errors');
    assert.equal(remoteRequests.length,0,'No remote assets requested');
    assert.equal(manifest.checks.chart_count,9,'Eight labs include two monitoring charts');
    assert.ok(manifest.labs.every(l=>l.results_text&&l.charts.length&&l.default_parameters.length),'Complete lab descriptions and outputs');
    fs.writeFileSync(manifestPath,JSON.stringify(manifest,null,2)+'\n');
    console.log(JSON.stringify({manifest:manifestPath,...manifest.checks},null,2));
  } finally {
    await context.close();
    fs.rmSync(profile,{recursive:true,force:true});
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
