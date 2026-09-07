/* Preview extracted EPUB XHTML in isolated Chrome, not a Kindle conversion. */
'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),cp=require('node:child_process');
const {pathToFileURL}=require('node:url');
const modulePath=process.env.PLAYWRIGHT_MODULE||path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {chromium}=require(modulePath);
const epub=path.resolve(__dirname,'../healthcare-simulation-sdk-textbook.epub');
const unpack=path.join(__dirname,'unpacked'),renders=path.join(__dirname,'render');
const extraction=`import sys,zipfile,io,hashlib,json,shutil
from pathlib import Path
import xml.etree.ElementTree as E
source,target=Path(sys.argv[1]),Path(sys.argv[2]);raw=source.read_bytes()
if target.exists():shutil.rmtree(target)
target.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 for name in z.namelist():
  p=(target/name).resolve()
  if not p.is_relative_to(target.resolve()):raise ValueError('Unsafe archive path')
 z.extractall(target)
 container=E.fromstring(z.read('META-INF/container.xml'));opf=container.find('.//{*}rootfile').attrib['full-path'];root=E.fromstring(z.read(opf));base=Path(opf).parent
 manifest={x.attrib['id']:x.attrib for x in root.find('{*}manifest')}
 spine=[]
 for ref in root.find('{*}spine'):
  item=manifest[ref.attrib['idref']];rel=str(base/item['href']);doc=E.fromstring(z.read(rel))
  headings=[''.join(n.itertext()).strip() for n in doc.iter() if n.tag.split('}')[-1] in ('h1','h2','h3')]
  images=[n.attrib.get('alt','') for n in doc.iter() if n.tag.split('}')[-1]=='img']
  spine.append({'path':rel,'headings':headings,'image_alts':images})
 nav=next(v for v in manifest.values() if 'nav' in v.get('properties','').split());navpath=str(base/nav['href']);navdoc=E.fromstring(z.read(navpath))
 navlinks=[{'href':n.get('href'),'text':''.join(n.itertext()).strip()} for n in navdoc.iter() if n.tag.split('}')[-1]=='a']
 print(json.dumps({'epub_sha256':hashlib.sha256(raw).hexdigest(),'spine':spine,'nav_links':navlinks}))`;
const metadata=JSON.parse(cp.execFileSync(process.env.PYTHON_BIN||'python3',['-c',extraction,epub,unpack],{encoding:'utf8'}));
const checks=[],screenshots=[];
const check=(name,ok,details)=>checks.push({name,status:ok?'PASS':'FAIL',details});
(async()=>{
 fs.mkdirSync(renders,{recursive:true});
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'healthcare-epub-preview-'));
 const context=await chromium.launchPersistentContext(profile,{executablePath:process.env.CHROME_BIN||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,viewport:{width:600,height:900},deviceScaleFactor:2,javaScriptEnabled:false,args:['--disable-background-networking','--no-first-run']});
 const errors=[],requests=[];
 await context.route(/^https?:\/\//,r=>{requests.push(r.request().url());return r.abort();});
 try{
  const page=await context.newPage();page.setDefaultTimeout(5000);page.on('pageerror',e=>errors.push(e.message));
  const chapters=fs.readdirSync(path.resolve(__dirname,'../../chapters')).filter(n=>/^\d\d-.*\.json$/.test(n)).map(n=>JSON.parse(fs.readFileSync(path.resolve(__dirname,'../../chapters',n),'utf8')));
  const missingNav=chapters.filter(ch=>!metadata.nav_links.some(a=>a.text.includes(ch.title)));
  check('All sixteen main chapters appear in EPUB navigation',missingNav.length===0,{missing:missingNav.map(c=>c.id),nav_link_count:metadata.nav_links.length,spine_count:metadata.spine.length});
  const layout=[];let imageCount=0;
  for(const width of [600,360]){
   await page.setViewportSize({width,height:900});
   for(const doc of metadata.spine){
    await page.goto(pathToFileURL(path.join(unpack,doc.path)).href,{waitUntil:'load'});
    const reading=await page.evaluate(()=>({
     scroll:document.documentElement.scrollWidth,width:innerWidth,scripts:document.scripts.length,
     failed_images:[...document.images].filter(i=>!i.complete||i.naturalWidth===0).map(i=>i.getAttribute('src')),
     image_count:document.images.length,
     out_of_bounds_images:[...document.images].filter(i=>{const b=i.getBoundingClientRect();return b.left<-.5||b.right>innerWidth+.5}).map(i=>({alt:i.alt,width:i.getBoundingClientRect().width})),
     overflowing:[...document.body.querySelectorAll('*')].filter(e=>{const r=e.getBoundingClientRect();return r.width&&r.right>innerWidth+1}).slice(0,5).map(e=>({tag:e.tagName,cls:e.className,text:e.textContent.slice(0,100),right:e.getBoundingClientRect().right}))
    }));
    if(width===600)imageCount+=reading.image_count;
    layout.push({path:doc.path,viewport:width,...reading});
   }
  }
  check('No page-level horizontal overflow at 600 and 360 pixels',layout.every(r=>r.scroll<=r.width+1),layout.filter(r=>r.scroll>r.width+1));
  check('All packaged images load without scripts',layout.every(r=>!r.failed_images.length),{image_count:imageCount,failures:layout.filter(r=>r.failed_images.length)});
  check('Figures stay within the reading viewport',layout.every(r=>!r.out_of_bounds_images.length),layout.filter(r=>r.out_of_bounds_images.length));
  check('EPUB content contains no scripts',layout.every(r=>r.scripts===0));
  async function openDoc(test,width=360){
   const doc=metadata.spine.find(test);if(!doc)throw new Error('Missing preview target');
   await page.setViewportSize({width,height:900});await page.goto(pathToFileURL(path.join(unpack,doc.path)).href,{waitUntil:'load'});return doc;
  }
  async function captureView(name,doc,selector=null,gray=false){
   if(selector){const target=page.locator(selector).first();await target.scrollIntoViewIfNeeded();await target.evaluate(e=>window.scrollTo(0,window.scrollY+e.getBoundingClientRect().top-15));}
   if(gray)await page.evaluate(()=>document.documentElement.style.filter='grayscale(1)');
   const file=path.join(renders,name+'.png');await page.screenshot({path:file,animations:'disabled'});
   screenshots.push({path:'render/'+name+'.png',source:doc.path,viewport:page.viewportSize(),grayscale:gray});
  }
  let doc=await openDoc(d=>d.path.endsWith('/cover.xhtml'),600);await captureView('cover-600',doc);
  doc=await openDoc(d=>d.headings.some(h=>h.includes('A laboratory for decisions about healthcare AI')));
  check('First chapter remains readable with JavaScript disabled',await page.locator('body').innerText().then(t=>t.includes('The central question')));
  doc=await openDoc(d=>d.image_alts.some(a=>a.includes('P(Y=1')&&a.includes('P(F=1')));
  const eqIndex=await page.locator('img').evaluateAll(imgs=>imgs.findIndex(i=>i.alt.includes('P(Y=1')&&i.alt.includes('P(F=1')));
  const eqLocator=page.locator('img').nth(eqIndex);await eqLocator.scrollIntoViewIfNeeded();await eqLocator.evaluate(e=>window.scrollTo(0,window.scrollY+e.getBoundingClientRect().top-70));await captureView('dense-equation-360',doc);
  doc=await openDoc(d=>d.headings.some(h=>h.includes('Turn percentages into people')));
  let heading=page.getByRole('heading',{name:'Worked lab: Turn percentages into people',exact:true});
  await heading.scrollIntoViewIfNeeded();await heading.evaluate(e=>window.scrollTo(0,window.scrollY+e.getBoundingClientRect().top-15));
  await captureView('bayes-lab-360-grayscale',doc,null,true);
  doc=await openDoc(d=>d.headings.some(h=>h.includes('Compare policies under the same budget')));
  heading=page.getByRole('heading',{name:'Worked lab: Compare policies under the same budget',exact:true});
  await heading.scrollIntoViewIfNeeded();await heading.evaluate(e=>window.scrollTo(0,window.scrollY+e.getBoundingClientRect().top-15));await captureView('capacity-lab-360',doc);
  doc=await openDoc(d=>d.headings.some(h=>/^Malnutrition:/.test(h)));
  await captureView('malnutrition-360',doc,'h1');
  doc=await openDoc(d=>d.headings.some(h=>h.startsWith('Source notes 14:')));
  await captureView('source-notes-360',doc,'pre');
  check('No browser runtime errors',errors.length===0,errors);
  check('No remote resources needed',requests.length===0,requests);
  const report={status:checks.every(c=>c.status==='PASS')?'PASS':'FAIL',scope:'Browser preview of extracted EPUB XHTML/CSS/images; not Kindle conversion, pagination, hardware, or e-ink device validation.',epub_sha256:metadata.epub_sha256,browser:await context.browser().version(),javascript_enabled:false,checks,screenshots,layout};
  fs.writeFileSync(path.join(__dirname,'reading_results.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({status:report.status,checks:checks.length,screenshots:screenshots.length,epub_sha256:report.epub_sha256,failures:checks.filter(c=>c.status==='FAIL')},null,2));
 }finally{await context.close();fs.rmSync(profile,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exitCode=1;});
