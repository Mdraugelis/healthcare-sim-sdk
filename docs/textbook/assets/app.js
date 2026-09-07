/* Offline teaching calculators. These do not run the production simulator. */
(function () {
  'use strict';
  const M = {
    bayes(p, se, sp, n = 10000) {
      const tp = n * p * se, fn = n * p * (1-se), fp = n * (1-p) * (1-sp), tn = n * (1-p) * sp;
      return {tp, fn, fp, tn, ppv: tp+fp ? tp/(tp+fp) : null};
    },
    risk(r, years) { return -Math.expm1(Math.log1p(-r) * years); },
    coupling(sd0, sd1, rho, n) { const variance = sd0**2 + sd1**2 - 2*rho*sd0*sd1; return {variance, mcse: Math.sqrt(variance/n)}; },
    mc(sd, n) { return {mcse: sd/Math.sqrt(n), half: 1.959963984540054*sd/Math.sqrt(n)}; },
    falseAlarm(alpha, checks) { return -Math.expm1(checks*Math.log1p(-alpha)); },
    auc(scores, labels) {
      const pos=scores.filter((_,i)=>labels[i]===1), neg=scores.filter((_,i)=>labels[i]===0);
      if (!pos.length || !neg.length) return null;
      let wins=0; for(const p of pos) for(const n of neg) wins += p>n ? 1 : p===n ? .5 : 0;
      return wins/(pos.length*neg.length);
    },
    calibrate(q, slope, intercept) { return q.map(p => 1/(1+Math.exp(-(intercept+slope*Math.log(p/(1-p)))))); },
    expectedBrier(q, p) { return q.reduce((sum,x,i)=>sum+x*(1-p[i])**2+(1-x)*p[i]**2,0)/q.length; },
    // Six fixed, equal-cost opportunities in arrival order; all numbers are hypothetical.
    patients: [
      {id:'A', risk:.20, contact:1, efficacy:.5}, {id:'B', risk:.50, contact:.2, efficacy:.2},
      {id:'C', risk:.10, contact:1, efficacy:.5}, {id:'D', risk:.40, contact:.9, efficacy:.5},
      {id:'E', risk:.30, contact:.8, efficacy:.6}, {id:'F', risk:.15, contact:1, efficacy:.4}
    ],
    capacity(k, reserved, order) {
      const available=Math.max(0,k-reserved), all=M.patients.map(p=>({...p, gain:p.risk*p.contact*p.efficacy}));
      const baseline=all.slice(0,available), sorted=[...all].sort((a,b)=>order==='benefit' ? b.gain-a.gain : b.risk-a.risk), ranked=sorted.slice(0,available);
      const total=a=>a.reduce((s,p)=>s+p.gain,0);
      return {available, all, baseline, ranked, baselineGain:total(baseline), rankedGain:total(ranked), delta:total(ranked)-total(baseline)};
    },
    solve(a,b) {
      const m=a.map((r,i)=>[...r,b[i]]), n=b.length;
      for(let k=0;k<n;k++) {
        let pivot=k; for(let i=k+1;i<n;i++) if(Math.abs(m[i][k])>Math.abs(m[pivot][k])) pivot=i;
        if(Math.abs(m[pivot][k])<1e-10) throw new Error('Singular teaching design');
        [m[k],m[pivot]]=[m[pivot],m[k]];
        const d=m[k][k]; for(let j=k;j<=n;j++) m[k][j]/=d;
        for(let i=0;i<n;i++) if(i!==k) { const v=m[i][k]; for(let j=k;j<=n;j++) m[i][j]-=v*m[k][j]; }
      }
      return m.map(r=>r[n]);
    },
    its(level, slope, confound, noiseScale=0, rho=0, breakTime=24) {
      const fixture=[.2,-.7,.5,.1,-.4,.9,-.3,.4,-.8,.6,-.2,.1];
      let noise=0;
      const rows=Array.from({length:48},(_,t)=>{ noise=rho*noise+noiseScale*fixture[t%fixture.length]; const post=t>=breakTime?1:0, elapsed=Math.max(0,t-breakTime); return {t,x:[1,t,post,elapsed],base:30+.2*t,y:30+.2*t+(level+confound)*post+slope*elapsed+noise}; });
      const a=Array.from({length:4},()=>Array(4).fill(0)), b=Array(4).fill(0);
      for(const r of rows) for(let i=0;i<4;i++){b[i]+=r.x[i]*r.y;for(let j=0;j<4;j++)a[i][j]+=r.x[i]*r.x[j];}
      const beta=M.solve(a,b); rows.forEach(r=>r.fit=r.x.reduce((s,x,j)=>s+x*beta[j],0));
      return {rows,beta};
    },
    monitor(shift, ramp, k, h) {
      const fixture=[.2,-.6,.7,-.1,.3,-.8,.5,-.2,.9,-.5,.1,-.4]; let c=0;
      return Array.from({length:60},(_,t)=>{const effect=t<30?0:shift*(ramp?Math.min(1,(t-29)/15):1), z=fixture[t%fixture.length]+effect; c=Math.max(0,c+z-k); const value=c, alarm=c>h; if(alarm)c=0; return {t,z,c:value,alarm,point:z>3};});
    }
  };
  if (typeof module !== 'undefined' && module.exports) module.exports=M;
  if (typeof document === 'undefined') return;
  window.TextbookMath=M;
  const $=(s,e=document)=>e.querySelector(s), $$=(s,e=document)=>[...e.querySelectorAll(s)];
  const fmt=(n,d=2)=>n===null?'undefined':Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function control(id,label,min,max,step,value,unit='') {
    return `<label for="${id}"><span>${label}<output id="${id}-value" for="${id}">${value}${unit}</output></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}" data-unit="${unit}"></label>`;
  }
  function frame(el,title,desc,controls,note) {
    el.className='lab'; el.setAttribute('role','region'); el.setAttribute('aria-label',title);
    el.innerHTML=`<div class="lab-kind">Interactive teaching lab · hypothetical inputs</div><h4>${title}</h4><p class="lab-desc">${desc}</p><div class="lab-controls">${controls}</div><div class="chart"></div><div class="lab-result" aria-live="polite"></div><p class="lab-note">${note}</p>`;
  }
  function values(el) { const out={}; $$('input,select',el).forEach(i=>{out[i.id]=i.type==='range'?+i.value:i.value; const o=$('#'+i.id+'-value',el); if(o)o.textContent=i.value+(i.dataset.unit||'');});return out; }
  function width(el){return Math.max(270,Math.round($('.chart',el).clientWidth));}
  function svg(w,h,title,body){return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>${body}</svg>`;}
  function bars(el,items,max,title) {
    const w=width(el), h=items.length*56+20, available=w-2;
    return svg(w,h,title,items.map((d,i)=>`<text x="0" y="${i*56+15}">${esc(d.label)}</text><rect x="0" y="${i*56+24}" width="${available}" height="16" fill="#e9eeea"/><rect x="0" y="${i*56+24}" width="${max?available*d.value/max:0}" height="16" fill="${d.color||'#086b65'}"/><text x="${w-2}" text-anchor="end" y="${i*56+15}">${esc(d.text||fmt(d.value))}</text>`).join(''));
  }
  function lines(el,series,{xmin=0,xmax=1,ymin=0,ymax=1,xlabel='',ylabel='',vline=null,title=''}={}) {
    const w=width(el),h=250,l=45,r=12,top=27,bot=43;
    const x=n=>l+(w-l-r)*(n-xmin)/(xmax-xmin),y=n=>h-bot-(h-top-bot)*(n-ymin)/(ymax-ymin);
    let body=`<text x="${l}" y="14">${esc(ylabel)}</text>`;
    for(let i=0;i<=4;i++){const n=ymin+(ymax-ymin)*i/4;body+=`<line x1="${l}" x2="${w-r}" y1="${y(n)}" y2="${y(n)}" stroke="#d5ddd9"/><text x="${l-6}" y="${y(n)+4}" text-anchor="end">${fmt(n,Math.abs(ymax-ymin)>5?0:1)}</text>`;}
    for(let i=0;i<=4;i++){const n=xmin+(xmax-xmin)*i/4;body+=`<text x="${x(n)}" y="${h-bot+19}" text-anchor="middle">${fmt(n,xmax<=1?1:0)}</text>`;}
    if(vline!==null)body+=`<line x1="${x(vline)}" x2="${x(vline)}" y1="${top}" y2="${h-bot}" stroke="#ad6a19" stroke-dasharray="4 4"/>`;
    for(const s of series) { const pts=s.points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ');body+=`<polyline points="${pts}" fill="none" stroke="${s.color||'#086b65'}" stroke-width="2.5" ${s.dash?'stroke-dasharray="5 4"':''}/>`;if(s.dots)for(const p of s.points)body+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="3.2" fill="${s.color||'#086b65'}"/>`;}
    body+=`<text x="${(l+w-r)/2}" y="${h-3}" text-anchor="middle">${esc(xlabel)}</text>`;
    return svg(w,h,title,body);
  }
  const labs={
    dynamics(el) {
      frame(el,'A year is not 52 identical fractions','Change annual risk and the observation window. The exact conversion assumes a constant hazard within the window.',control('risk-annual','Annual probability',1,90,1,5,'%')+control('risk-weeks','Window length',1,52,1,4,' weeks'),'Analytical conversion only. No patient trajectories are generated. Percentages refer to probability of at least one event.');
      return()=>{const v=values(el),r=v['risk-annual']/100,dt=v['risk-weeks']/52,exact=M.risk(r,dt);$('.chart',el).innerHTML=bars(el,[{label:'Constant hazard',value:exact*100,text:fmt(exact*100,4)+'%'},{label:'Linear scaling',value:r*dt*100,color:'#ad6a19',text:fmt(r*dt*100,4)+'%'}],Math.max(exact,r*dt)*100,'Exact versus linear window probability');$('.lab-result',el).innerHTML=`Window probability: <strong>${fmt(exact*100,4)}%</strong>. Annual hazard: ${fmt(-Math.log1p(-r),5)} per year. Linear scaling differs by ${fmt((exact-r*dt)*100,4)} percentage points.`;};
    },
    coupling(el) {
      frame(el,'Shared worlds change precision','Both branch standard deviations are 10 events. Change their correlation and the number of independent paired replications.',control('crn-rho','Branch correlation',-1,1,.05,.8)+control('crn-n','Replications',5,200,5,25),'Var(L₀−L₁) = 100 + 100 − 200ρ. Correlation is an illustrative input; shared seeds do not guarantee this value. The mean effect is unchanged in this calculator.');
      return()=>{const v=values(el),d=M.coupling(10,10,v['crn-rho'],v['crn-n']),ind=M.coupling(10,10,0,v['crn-n']);$('.chart',el).innerHTML=bars(el,[{label:'Selected correlation',value:d.mcse,text:fmt(d.mcse)+' events'},{label:'Independent branches',value:ind.mcse,color:'#315eac',text:fmt(ind.mcse)+' events'}],Math.max(d.mcse,ind.mcse),'Monte Carlo standard error of paired mean');$('.lab-result',el).innerHTML=`Difference variance: <strong>${fmt(d.variance)} events²</strong>. Mean MCSE: <strong>${fmt(d.mcse)} events</strong>. ${d.variance<200?'Positive covariance reduces variance.':d.variance>200?'Negative covariance increases variance.':'Covariance is zero.'}`;};
    },
    'monte-carlo'(el) {
      frame(el,'More runs sharpen the mean','Assume normally distributed independent run effects with mean 10 events averted. Compare the scale of individual-run variation with the sampling distribution of the mean.',control('mc-sd','Run standard deviation',1,10,.5,4,' events')+control('mc-n','Replications',5,200,5,25),'Uses a known teaching SD and a normal reference factor 1.96. The production helper estimates SD and uses a t interval. This lab does not draw seeds, estimate study power or vary clinical assumptions.');
      return()=>{const v=values(el),sd=v['mc-sd'],d=M.mc(sd,v['mc-n']),normal=(s)=>Array.from({length:401},(_,i)=>{const lo=Math.max(-10,10-5*s),hi=Math.min(30,10+5*s),x=lo+(hi-lo)*i/400;return[x,Math.exp(-.5*((x-10)/s)**2)/(s*Math.sqrt(2*Math.PI))];});$('.chart',el).innerHTML=lines(el,[{points:normal(sd),color:'#ad6a19'},{points:normal(d.mcse)}],{xmin:-10,xmax:30,ymin:0,ymax:1/(d.mcse*Math.sqrt(2*Math.PI))*1.1,xlabel:'Events averted',ylabel:'Probability density',title:'Run-effect distribution in gold; distribution of means in teal'});$('.lab-result',el).innerHTML=`Run SD (gold): <strong>${fmt(sd)} events</strong>. Mean MCSE (teal): <strong>${fmt(d.mcse)} events</strong>. A 95% normal reference range for repeated sample means is ${fmt(10-d.half)} to ${fmt(10+d.half)} events. It narrows while the run distribution stays fixed.`;};
    },
    bayes(el) {
      frame(el,'Turn percentages into people','Expected counts in 10,000 independent opportunities. Each setting describes a hypothetical joint distribution, not an independently achievable model operating point.',control('bayes-p','Prevalence',0,50,1,5,'%')+control('bayes-se','Sensitivity',0,100,1,80,'%')+control('bayes-sp','Specificity',0,100,1,90,'%'),'These are expected counts; a finite realized cohort varies. PPV is undefined when there are no positive flags. Changing prevalence may also change real-world sensitivity and specificity.');
      return()=>{const v=values(el),d=M.bayes(v['bayes-p']/100,v['bayes-se']/100,v['bayes-sp']/100);$('.chart',el).innerHTML=bars(el,[{label:'True positives',value:d.tp},{label:'False positives',value:d.fp,color:'#9b3937'},{label:'False negatives',value:d.fn,color:'#ad6a19'},{label:'True negatives',value:d.tn,color:'#55776e'}],10000,'Expected confusion matrix counts on a shared 10,000-person scale');$('.lab-result',el).innerHTML=`Of ${fmt(d.tp+d.fp,0)} expected positive flags, ${fmt(d.tp,0)} are cases. <strong>PPV: ${d.ppv===null?'undefined':fmt(d.ppv*100)+'%'}</strong>. Total expected count: ${fmt(d.tp+d.fp+d.fn+d.tn,0)}.`;};
    },
    calibration(el) {
      const q=[.05,.10,.20,.30,.40,.55,.65,.75,.85,.95],y=[0,0,1,0,0,1,0,1,1,1];
      frame(el,'Ranking can stay fixed while probabilities move','Apply a strictly increasing logit transformation to ten known toy risks. The points show transformed predictions against those known risks.',control('cal-slope','Logit multiplier',.2,3,.1,1)+control('cal-offset','Logit offset',-2,2,.1,0),'AUC uses one fixed illustrative binary-label vector [0,0,1,0,0,1,0,1,1,1]. Expected Brier integrates over Bernoulli outcomes at the known toy risks. Neither is a held-out clinical validation metric.');
      return()=>{const v=values(el),p=M.calibrate(q,v['cal-slope'],v['cal-offset']);$('.chart',el).innerHTML=lines(el,[{points:[[0,0],[1,1]],color:'#9aa9a3',dash:true},{points:p.map((s,i)=>[s,q[i]]),dots:true}],{xlabel:'Transformed predicted probability',ylabel:'Known toy risk',title:'Probability correspondence: identity line and transformed probabilities'});$('.lab-result',el).innerHTML=`Fixed-label rank AUC: <strong>${fmt(M.auc(p,y),3)}</strong>. Expected Brier: <strong>${fmt(M.expectedBrier(q,p),4)}</strong> (at identity: ${fmt(M.expectedBrier(q,q),4)}). Flags at p ≥ .5: ${p.filter(s=>s>=.5).length} of 10. Monotone transformation preserves ranks; it need not preserve a fixed-threshold decision.`;};
    },
    capacity(el) {
      frame(el,'Compare policies under the same budget','Six hypothetical patients arrive in order A–F. Each optional assessment consumes one slot. Obligatory consult slots are reserved equally in both policies.',control('cap-k','Total slots',0,6,1,3)+control('cap-reserved','Reserved consult slots',0,6,1,1)+'<label for="cap-order">New worklist order<select id="cap-order"><option value="risk">Predicted risk</option><option value="benefit">Expected preventable events</option></select></label>','Gain = risk × contact probability × relative efficacy. All inputs are invented and gains are additive with no further spillovers. Reserved demand can exceed capacity; this toy then leaves zero optional slots. It does not simulate the full malnutrition queue.');
      return()=>{const v=values(el),d=M.capacity(v['cap-k'],v['cap-reserved'],v['cap-order']);$('.chart',el).innerHTML=bars(el,[{label:'Arrival-order baseline',value:d.baselineGain,color:'#315eac'},{label:'Selected new ordering',value:d.rankedGain}],.8,'Expected prevented events under equal available capacity');$('.lab-result',el).innerHTML=`Optional slots: <strong>${d.available}</strong>. Baseline: ${d.baseline.map(p=>p.id).join(', ')||'none'}; new order: ${d.ranked.map(p=>p.id).join(', ')||'none'}. Incremental expected events averted: <strong>${fmt(d.delta,3)}</strong>.<details><summary>Inspect every patient and gain</summary><div class="table-wrap"><table><thead><tr><th>ID</th><th>Risk</th><th>Contact</th><th>Efficacy</th><th>Expected gain</th></tr></thead><tbody>${d.all.map(p=>`<tr><td>${p.id}</td><td>${p.risk}</td><td>${p.contact}</td><td>${p.efficacy}</td><td>${fmt(p.gain,3)}</td></tr>`).join('')}</tbody></table></div></details>`;};
    },
    its(el) {
      frame(el,'An interrupted trend is an assumption','A 48-period toy series has baseline 30 + 0.2t. Fit an intercept, time, post-break indicator and time since break. A coincident change is statistically inseparable from the intervention jump.',control('its-level','Intervention level change',-8,8,.5,-3)+control('its-slope','Post-break slope change',-.4,.4,.05,-.1)+control('its-confound','Coincident level change',-8,8,.5,0)+control('its-break','Break period',12,36,1,24)+control('its-noise','Fixed-noise amplitude',0,2,.25,0)+control('its-rho','Noise persistence',0,.9,.1,0),'OLS is a small independent teaching implementation. Optional noise is a fixed repeating innovation fixture filtered as eₜ = ρeₜ₋₁ + a·fixtureₜ; it is not a random Gaussian sample. No confidence interval, coverage or power is estimated here.');
      return()=>{const v=values(el),d=M.its(v['its-level'],v['its-slope'],v['its-confound'],v['its-noise'],v['its-rho'],v['its-break']),ys=d.rows.flatMap(r=>[r.y,r.base,r.fit]);$('.chart',el).innerHTML=lines(el,[{points:d.rows.map(r=>[r.t,r.base]),color:'#ad6a19',dash:true},{points:d.rows.map(r=>[r.t,r.y]),color:'#8aa59e',dots:true},{points:d.rows.map(r=>[r.t,r.fit])}],{xmin:0,xmax:47,ymin:Math.floor(Math.min(...ys)-2),ymax:Math.ceil(Math.max(...ys)+2),xlabel:'Period t',ylabel:'Outcome units',vline:v['its-break'],title:'Gold: assumed baseline; muted points: observed toy series; teal: OLS fit'});$('.lab-result',el).innerHTML=`Estimated level change: <strong>${fmt(d.beta[2],3)}</strong>; slope change: <strong>${fmt(d.beta[3],3)} per period</strong>. Planted intervention jump: ${fmt(v['its-level'],1)}; coincident jump: ${fmt(v['its-confound'],1)}. The fitted level combines both. Gold is the assumed baseline, teal is the fit.`;};
    },
    monitoring(el) {
      frame(el,'Detect a change without crediting earlier alarms','A fixed standardized noise fixture receives an upward shift beginning at t=30. Compare an immediate z>3 rule with a one-sided CUSUM Cₜ=max(0,Cₜ₋₁+zₜ−k), resetting after Cₜ>h.',control('mon-shift','Adverse shift',0,4,.25,1.5,' SD')+control('mon-k','CUSUM allowance k',0,1,.1,.5)+control('mon-h','CUSUM threshold h',2,8,.5,4)+'<label for="mon-mode">Change pattern<select id="mon-mode"><option value="abrupt">Abrupt</option><option value="ramp">15-period ramp</option></select></label>','This deterministic trace teaches timing only. It does not estimate a monitor’s sensitivity, specificity or false-alarm rate. The independent-check calculation below has its own explicit assumption; correlated surveillance requires repeated null simulations.');
      return()=>{const v=values(el),rows=M.monitor(v['mon-shift'],v['mon-mode']==='ramp',v['mon-k'],v['mon-h']),point=rows.find(r=>r.t>=30&&r.point),alarm=rows.find(r=>r.t>=30&&r.alarm);$('.chart',el).innerHTML=lines(el,[{points:rows.map(r=>[r.t,r.z]),dots:true},{points:[[0,3],[59,3]],color:'#9b3937',dash:true}],{xmin:0,xmax:59,ymin:-1,ymax:5,xlabel:'Period t (change at 30)',ylabel:'Standardized observed value',vline:30,title:'Observed standardized signal and single-observation threshold at 3'})+lines(el,[{points:rows.map(r=>[r.t,r.c]),color:'#315eac'},{points:[[0,v['mon-h']],[59,v['mon-h']]],color:'#9b3937',dash:true}],{xmin:0,xmax:59,ymin:0,ymax:Math.max(10,...rows.map(r=>r.c)),xlabel:'Period t (change at 30)',ylabel:'CUSUM before alarm reset',vline:30,title:'CUSUM path and selected alarm threshold'});$('.lab-result',el).innerHTML=`First qualifying immediate alarm: <strong>${point?'t='+point.t+', delay '+(point.t-30):'none by t=59'}</strong>. CUSUM: <strong>${alarm?'t='+alarm.t+', delay '+(alarm.t-30):'none by t=59'}</strong>. Pre-change CUSUM alarms: ${rows.filter(r=>r.t<30&&r.alarm).length}.<p class="lab-note">Separate probability example: 52 independent checks, each with null alarm probability .0027, yield 1−(1−.0027)⁵² = ${fmt(M.falseAlarm(.0027,52)*100,2)}% chance of at least one false alarm. This is not the CUSUM’s false-alarm rate.</p>`;};
    }
  };
  const redraw=[];
  $$('[data-lab]').forEach(el=>{const factory=labs[el.dataset.lab];if(!factory)return;const update=factory(el);el.addEventListener('input',update);el.addEventListener('change',update);update();redraw.push(update);});
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>redraw.forEach(f=>f()),150);});
  const sidebar=$('.sidebar'), menu=$('.menu-toggle');
  function syncNav(){sidebar.inert=window.innerWidth<=800&&!sidebar.classList.contains('open');}
  syncNav();window.addEventListener('resize',syncNav);
  menu.addEventListener('click',()=>{const open=sidebar.classList.toggle('open');menu.setAttribute('aria-expanded',String(open));syncNav();if(open)$('a',sidebar).focus();});
  $$('.sidebar a').forEach(a=>a.addEventListener('click',()=>{sidebar.classList.remove('open');menu.setAttribute('aria-expanded','false');syncNav();}));
  const chapters=$$('.chapter'), nav=$$('.chapter-link');
  if('IntersectionObserver' in window){
    const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting)nav.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id));},{rootMargin:'-10% 0px -70% 0px'});chapters.forEach(c=>observer.observe(c));}
  window.addEventListener('scroll',()=>{const total=document.documentElement.scrollHeight-window.innerHeight;$('.reading-progress').style.width=(Math.max(0,window.scrollY)/total*(window.innerWidth-(window.innerWidth>800?262:0)))+'px';},{passive:true});
  const search=$('#book-search'),results=$('#search-results');
  search.addEventListener('input',()=>{const q=search.value.trim().toLowerCase();results.replaceChildren();results.hidden=q.length<2;if(results.hidden)return;const matches=chapters.filter(c=>c.innerText.toLowerCase().includes(q));for(const c of matches){const a=document.createElement('a');a.href='#'+c.id;a.textContent=$('h2',c).textContent;const small=document.createElement('small');const text=$('.chapter-body',c)?.innerText||c.innerText;const i=text.toLowerCase().indexOf(q);small.textContent=text.slice(Math.max(0,i-35),Math.max(0,i-35)+160).replace(/\s+/g,' ');a.append(small);a.addEventListener('click',()=>{results.hidden=true;search.value='';});results.append(a);}if(!matches.length)results.textContent='No matching chapters.';});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){results.hidden=true;const navWasOpen=sidebar.classList.contains('open');sidebar.classList.remove('open');menu.setAttribute('aria-expanded','false');syncNav();if(navWasOpen)menu.focus();if(!$('#review-panel').hidden)closeReview();}});
  let printOpened=[];
  window.addEventListener('beforeprint',()=>{printOpened=$$('details:not([open])');printOpened.forEach(d=>d.open=true);});
  window.addEventListener('afterprint',()=>printOpened.forEach(d=>d.open=false));
  $('.print-button').addEventListener('click',()=>window.print());
  const panel=$('#review-panel'),select=$('#review-chapter'),name=$('#review-name'),status=$('#review-status'),note=$('#review-note'),notice=$('#review-notice'),identity=document.documentElement.dataset.identity,key='healthcare-sdk-textbook-review-v1';
  let data={schema:2,default_reviewer:'',chapters:{}},storageOK=true;
  try{const saved=JSON.parse(localStorage.getItem(key)||'null');if(saved&&[1,2].includes(saved.schema)&&saved.chapters&&typeof saved.chapters==='object'){
    data={schema:2,default_reviewer:saved.default_reviewer||saved.reviewer||'',chapters:saved.chapters};
    if(saved.schema===1)Object.values(data.chapters).forEach(d=>{if(d&&typeof d==='object'&&!('reviewer' in d))d.reviewer=saved.reviewer||null;});
  }}catch(_){storageOK=false;}
  function loadNote(){const d=data.chapters[select.value]||{};name.value=typeof d.reviewer==='string'?d.reviewer:data.default_reviewer;note.value=typeof d.note==='string'?d.note:'';status.value=['pending','questions','changes_requested','reviewed'].includes(d.status)?d.status:'pending';notice.textContent=!storageOK?'Browser storage unavailable. Save to memory, then export before closing.':d.manuscript&&d.manuscript!==identity?'This note refers to an older manuscript identity. Export preserves it; Save explicitly renews it for the current manuscript.':'Notes stay in this browser. Save edits before exporting or changing chapters.';}
  function persist(){if(status.value==='reviewed'&&!name.value.trim()){notice.textContent='Enter a reviewer name before recording a reviewed status.';name.focus();return;}
    const previous=data.chapters[select.value];data.default_reviewer=name.value.trim();data.chapters[select.value]={reviewer:name.value.trim()||null,note:note.value,status:status.value,manuscript:identity,previous_manuscript:previous?.manuscript!==identity?previous?.manuscript:(previous?.previous_manuscript||null),updated:new Date().toISOString()};try{localStorage.setItem(key,JSON.stringify(data));notice.textContent='Saved locally for this manuscript. Human review notes do not change machine evidence.';}catch(_){storageOK=false;notice.textContent='Saved in memory only; browser storage unavailable. Export before closing.';}}
  let returnFocus=null;
  function closeReview(){panel.hidden=true;$('#review-toggle').setAttribute('aria-expanded','false');returnFocus?.focus();}
  $('#review-toggle').addEventListener('click',()=>{if(!panel.hidden){closeReview();return;}returnFocus=document.activeElement;panel.hidden=false;$('#review-toggle').setAttribute('aria-expanded','true');loadNote();name.focus();});
  $('#review-close').addEventListener('click',closeReview);select.addEventListener('change',loadNote);$('#review-save').addEventListener('click',persist);
  $('#review-export').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({...data,current_manuscript:identity,exported:new Date().toISOString()},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='healthcare-sdk-textbook-review.json';a.click();notice.textContent='Exported saved notes with their original identities and reviewer names. Unsaved edits are not included.';setTimeout(()=>URL.revokeObjectURL(url),1000);});
  loadNote();
})();
