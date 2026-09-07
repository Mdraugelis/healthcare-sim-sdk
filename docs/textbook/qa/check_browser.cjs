/* Exercise the actual offline artifact in Chromium. See ../README.md. */
'use strict';
const fs = require('fs'), path = require('path'), assert = require('assert/strict');
const {pathToFileURL} = require('url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const book=path.resolve(__dirname,'../index.html'), url=pathToFileURL(book).href;
const results=[];
function check(name,ok,details){assert.ok(ok,name);results.push({name,status:'PASS',details});}
(async()=>{
  const browser=await chromium.launch({headless:true,...(process.env.CHROME_BIN?{executablePath:process.env.CHROME_BIN}:{})});
  try {
    const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
    const page=await context.newPage(),errors=[],remote=[];
    page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))remote.push(r.url());});
    await page.goto(url);await page.evaluate(()=>document.fonts.ready);
    check('All 16 chapters render',await page.locator('.chapter[id^="0"],.chapter[id^="1"]').count()===16);
    check('Eight initialized labs',await page.locator('.lab .lab-result').count()===8);
    check('Forty-seven equations render with accessible labels',await page.locator('.equation svg[aria-label]').count()===47);
    check('No external asset requests',remote.length===0,remote);
    await page.screenshot({path:path.join(__dirname,'desktop.png')});
    await page.locator('[data-lab="bayes"]').screenshot({path:path.join(__dirname,'bayes_lab.png')});
    await page.locator('[data-lab="its"]').screenshot({path:path.join(__dirname,'its_lab.png')});
    await page.locator('.equation').first().screenshot({path:path.join(__dirname,'equation.png')});
    // Every range control must update its named output and leave all labs finite.
    const sliderIds=await page.locator('.lab input[type=range]').evaluateAll(inputs=>inputs.map(i=>i.id));
    for(const id of sliderIds){for(const boundary of ['min','max']){
      await page.locator('#'+id).evaluate((i,b)=>{i.value=i[b];i.dispatchEvent(new Event('input',{bubbles:true}));},boundary);
      const val=await page.locator('#'+id).inputValue(),label=await page.locator('#'+id+'-value').innerText();
      assert.ok(label.startsWith(val),id+' output');
    }}
    check('Every slider responds at both domain boundaries',true,{slider_count:sliderIds.length});
    check('No NaN or Infinity in lab results',!(await page.locator('.lab-result').allTextContents()).some(t=>/\b(?:NaN|Infinity)\b/.test(t)));
    await page.reload();
    await page.locator('#book-search').fill('malnutrition');
    check('Search exposes matching chapters',await page.locator('#search-results a').count()>2);
    await page.locator('#search-results a[href="#12-malnutrition"]').click();
    check('Search navigates to exact chapter',new URL(page.url()).hash==='#12-malnutrition');
    await page.locator('.chapter[id="12-malnutrition"] h2').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(__dirname,'malnutrition.png')});
    // Explicitly save one reviewer, then a second chapter by a different reviewer.
    await page.locator('#review-toggle').click();await page.locator('#review-name').fill('Reviewer A');
    await page.locator('#review-status').selectOption('reviewed');await page.locator('#review-note').fill('Check comparator capacity. <script>alert(1)</script>');await page.locator('#review-save').click();
    await page.locator('#review-chapter').selectOption('12-malnutrition');await page.locator('#review-name').fill('Reviewer B');await page.locator('#review-note').fill('Check horizon denominators.');await page.locator('#review-save').click();
    let saved=await page.evaluate(()=>JSON.parse(localStorage.getItem('healthcare-sdk-textbook-review-v1')));
    check('Reviewers are bound separately to chapter notes',saved.chapters['01-orientation'].reviewer==='Reviewer A'&&saved.chapters['12-malnutrition'].reviewer==='Reviewer B');
    await page.reload();await page.locator('#review-toggle').click();
    check('Notes survive reload as text',await page.locator('#review-note').inputValue()==='Check comparator capacity. <script>alert(1)</script>');
    await page.locator('#review-name').fill('');await page.locator('#review-status').selectOption('reviewed');await page.locator('#review-save').click();
    check('Unnamed reviewed status is rejected',(await page.locator('#review-notice').innerText()).includes('Enter a reviewer name'));
    // Inject an older saved manuscript identity, then export without renewing it.
    await page.evaluate(()=>{const k='healthcare-sdk-textbook-review-v1',d=JSON.parse(localStorage.getItem(k));d.chapters['01-orientation'].manuscript='older-manuscript-fixture';localStorage.setItem(k,JSON.stringify(d));});
    await page.reload();await page.locator('#review-toggle').click();
    check('Older manuscript notes trigger an applicability notice',(await page.locator('#review-notice').innerText()).includes('older manuscript'));
    const downloadWait=page.waitForEvent('download');await page.locator('#review-export').click();const download=await downloadWait;
    const exported=JSON.parse(fs.readFileSync(await download.path(),'utf8'));
    check('Export preserves old identity and named reviewer without renewing',exported.chapters['01-orientation'].manuscript==='older-manuscript-fixture'&&exported.chapters['01-orientation'].reviewer==='Reviewer A'&&exported.current_manuscript!=='older-manuscript-fixture');
    await page.locator('#review-save').click();
    saved=await page.evaluate(()=>JSON.parse(localStorage.getItem('healthcare-sdk-textbook-review-v1')));
    check('Explicit Save renews identity and retains prior identity',saved.chapters['01-orientation'].manuscript===exported.current_manuscript&&saved.chapters['01-orientation'].previous_manuscript==='older-manuscript-fixture');
    await page.locator('#review-close').click();
    check('Review closes with focus restored',await page.locator('#review-toggle').evaluate(e=>e===document.activeElement));
    // Print path reveals disclosures then restores their open/closed state.
    const openBefore=await page.locator('details[open]').count();
    await page.evaluate(()=>window.dispatchEvent(new Event('beforeprint')));
    check('Print opens every disclosure',await page.locator('details:not([open])').count()===0);
    await page.emulateMedia({media:'print'});
    check('Print hides navigation and interactive controls',await page.locator('.sidebar').evaluate(e=>getComputedStyle(e).display==='none'));
    await page.emulateMedia({media:'screen'});await page.evaluate(()=>window.dispatchEvent(new Event('afterprint')));
    check('Print restores disclosure state',await page.locator('details[open]').count()===openBefore);
    // Desktop and narrow viewports: no page-level horizontal overflow.
    for(const width of [1440,1024,768,390,320]){
      await page.setViewportSize({width,height:900});await page.waitForTimeout(200);
      const sizes=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,client:document.documentElement.clientWidth}));
      if(sizes.scroll>sizes.client+1) console.log(JSON.stringify(await page.evaluate(()=>[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+2&&!e.closest('.sidebar')).slice(0,12).map(e=>({tag:e.tagName,cls:e.className,txt:e.textContent.slice(0,90),right:e.getBoundingClientRect().right})))));
      check('No page overflow at '+width+'px',sizes.scroll<=sizes.client+1,sizes);
    }
    await page.setViewportSize({width:390,height:844});await page.goto(url);await page.waitForTimeout(200);
    check('Closed mobile navigation is keyboard-inert',await page.locator('.sidebar').evaluate(e=>e.inert));
    await page.locator('.menu-toggle').click();check('Mobile contents opens',await page.locator('.menu-toggle').getAttribute('aria-expanded')==='true');
    await page.locator('.sidebar a[href="#06-ml-metrics"]').click();check('Mobile chapter selection closes navigation',await page.locator('.sidebar').evaluate(e=>e.inert));
    await page.locator('[data-lab="bayes"]').screenshot({path:path.join(__dirname,'mobile_lab.png')});
    await page.goto(url);await page.screenshot({path:path.join(__dirname,'mobile.png')});
    const unlabeled=await page.locator('input,select,textarea').evaluateAll(inputs=>inputs.filter(i=>!i.labels?.length&&!i.getAttribute('aria-label')).map(i=>i.id));
    check('All form controls have associated labels',unlabeled.length===0,unlabeled);
    check('No browser runtime errors',errors.length===0,errors);
    const nojs=await browser.newContext({javaScriptEnabled:false,viewport:{width:1200,height:900}}),staticPage=await nojs.newPage();await staticPage.goto(url);
    check('Book remains readable without JavaScript',await staticPage.locator('.equation').count()===47&&await staticPage.locator('.chapter-body').count()===16);
    await staticPage.locator('.chapter details').first().click();
    check('Native disclosures work without JavaScript',await staticPage.locator('details[open]').count()===1);
    const report={status:'PASS',html_sha256:require('crypto').createHash('sha256').update(fs.readFileSync(book)).digest('hex'),browser:await browser.version(),checks:results};
    fs.writeFileSync(path.join(__dirname,'browser_results.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({status:'PASS',checks:results.length,browser:report.browser},null,2));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
