import {topics} from './help.js';
import {streamSteering, readNeuron, activationHTML, escapeHTML as esc} from './experiments.js';

const $ = s => document.querySelector(s);
const storage = {
  get(key, fallback=null) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } },
  set(key,value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} }
};
const VERSION = 1, KEY = 'explorer.tour.v1';
const bytes = n => n == null ? 'Not measured yet' : `${(n / 1024**3).toFixed(2)} GiB`;
async function request(path, body) {
  const response = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}
export function initOnboarding(bridge) {
  let status = null, actionBusy = false, tour = storage.get(KEY), controller = null, previousFocus = null;
  if (tour?.version !== VERSION) tour = null;
  // Never replay requests after reload. Results completed before reload may be reviewed.
  if (tour) { tour.running = false; tour.active = false; }
  document.body.insertAdjacentHTML('beforeend', `
    <dialog id="welcome" aria-labelledby="welcome-title" class="learn-dialog">
      <h2 id="welcome-title">See what changes inside a language model</h2>
      <p>Explore Qwen’s neurons, inspect clues, and try a real steering experiment. No prior neural-network knowledge needed.</p>
      <p>The map is available now. Live experiments need a local model download. ChatGPT sign-in is optional.</p>
      <div class="learn-actions"><button data-welcome="setup">Set up Qwen</button><button data-welcome="demo">Start guided demo</button><button data-welcome="map">Explore the map</button></div>
    </dialog>
    <dialog id="model-dialog" aria-labelledby="model-title" class="learn-dialog">
      <div class="learn-heading"><h2 id="model-title">Model &amp; setup</h2><button data-close aria-label="Close model setup">×</button></div>
      <p>Runs on the computer hosting this app. Dependencies are installed by the launcher; these controls manage model files and memory.</p>
      <div id="model-cards"></div>
      <label for="model-device">Execution device</label>
      <select id="model-device"><option value="auto">Automatic accelerator (CUDA / MPS)</option><option value="cuda">CUDA</option><option value="mps">Apple MPS</option><option value="cpu">CPU — slow, float32</option></select>
      <div class="learn-actions" id="model-actions"><button data-model="download">Download Qwen</button><button data-model="download/cancel">Cancel download</button><button data-model="start">Start Qwen</button><button data-model="stop">Stop Qwen</button></div>
      <p id="model-error" role="alert"></p><p id="model-notice" role="status" aria-live="polite"></p>
      <button data-help="setup">Setup help and troubleshooting</button>
    </dialog>
    <dialog id="help-dialog" aria-labelledby="help-title" class="learn-dialog">
      <div class="learn-heading"><h2 id="help-title">Help &amp; glossary</h2><button data-close aria-label="Close help">×</button></div>
      <label for="help-search">Search help</label><input id="help-search" type="search" placeholder="Try dose, tokens, or connections">
      <div class="learn-actions"><button id="tour-start">Start / Resume guided demo</button><button id="tour-restart">Restart guided demo</button></div>
      <div id="help-topics"></div>
    </dialog>
    <section id="demo-panel" hidden aria-labelledby="demo-title" class="learn-panel">
      <div class="learn-heading"><span id="demo-count" class="dim"></span><button id="demo-exit">Exit demo</button></div>
      <h2 id="demo-title" tabindex="-1"></h2><div id="demo-body"></div>
      <p id="demo-pause" role="status"></p><p id="demo-error" role="alert"></p>
      <div class="learn-actions"><button id="demo-back">Back</button><button id="demo-skip">Skip</button><button id="demo-next">Next</button><button id="demo-restart">Restart</button></div>
    </section>
    <div id="help-tooltip" role="tooltip" hidden></div>`);

  function openDialog(id) {
    const dialog = $(id);
    if (dialog.open) return;
    dialog.returnFocus = document.activeElement;
    dialog.showModal();
  }
  for (const dialog of document.querySelectorAll('.learn-dialog')) {
    dialog.querySelector('[data-close]')?.addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => { if (dialog.returnFocus?.isConnected) dialog.returnFocus.focus(); });
  }
  function setup() { openDialog('#model-dialog'); refresh(); }
  $('#model-open').onclick = setup;
  document.addEventListener('open-model-setup', setup);
  $('#help-open').onclick = () => help();
  function help(topic) {
    $('#help-search').value = topic ? topics[topic][0] : '';
    renderHelp(); openDialog('#help-dialog');
  }
  function renderHelp() {
    const query = $('#help-search').value.toLowerCase();
    const entries = Object.entries(topics).filter(([,content]) => content.join(' ').toLowerCase().includes(query));
    $('#help-topics').innerHTML = entries.map(([key,[title,body,example,limit]]) => `<article id="help-${key}"><h3>${title}</h3><p>${body}</p><p><b>Try it:</b> ${example}</p><p class="dim"><b>Interpret carefully:</b> ${limit}</p></article>`).join('') || '<p>No matching topic. Try a shorter search.</p>';
  }
  $('#help-search').oninput = renderHelp;
  document.addEventListener('click', e => { const button = e.target.closest('[data-help]'); if (button) help(button.dataset.help); });
  const hints = {'#jump':'layers','#layer':'layers','#layout-mode':'map','#web-mode':'connections','#web-z':'connections','#q-mode':'lenses','#lens-q':'lenses','#c-mode':'steering','#c-dose':'dose','#c-dir':'activations','#show-random':'random','#inf-text':'activations','#views':'embeddings','#a-agent':'assistant'};
  for (const [selector,topic] of Object.entries(hints)) {
    const control = $(selector); if (!control) continue;
    const hint = document.createElement('button'); hint.type='button'; hint.className='help-dot'; hint.dataset.help=topic;
    hint.textContent='?'; hint.setAttribute('aria-label',`Help: ${topics[topic][0]}`);
    control.insertAdjacentElement('afterend',hint);
    const show = () => {
      const tip = $('#help-tooltip'); tip.textContent = topics[topic][1]; tip.hidden=false;
      const r = control.getBoundingClientRect(); tip.style.left=`${Math.max(8,Math.min(r.left,innerWidth-320))}px`;
      tip.style.top=`${Math.min(r.bottom+8,innerHeight-tip.offsetHeight-8)}px`;
      control.setAttribute('aria-describedby','help-tooltip');
    };
    const hide = () => { $('#help-tooltip').hidden=true; control.removeAttribute('aria-describedby'); };
    control.addEventListener('pointerenter',show); control.addEventListener('pointerleave',hide);
    control.addEventListener('focusin',show); control.addEventListener('focusout',hide);
  }
  document.addEventListener('keydown', e => { if (e.key === 'Escape') { $('#help-tooltip').hidden=true; if (!document.querySelector('dialog[open]') && tour?.active) exitTour(); } });

  const liveSelectors = '#c-go, #q-mode [data-q="fires"], #ltabs [data-lt="run"]';
  document.addEventListener('click', e => {
    if (e.target.closest(liveSelectors) && !status?.ready) { e.preventDefault(); e.stopImmediatePropagation(); setup(); }
  },true);
  $('#c-form').addEventListener('submit', e => { if (!status?.ready) { e.preventDefault(); e.stopImmediatePropagation(); setup(); } },true);
  function renderStatus() {
    if (!status) return;
    const s=status, d=s.diagnostics, p=s.download || {};
    const progress = p.total ? `<progress max="${p.total}" value="${p.downloaded||0}" aria-label="Downloaded model bytes"></progress><span>${bytes(p.downloaded||0)} / ${bytes(p.total)} reconstructed${p.transferred ? ` · ${bytes(p.transferred)} transferred this session` : ''}</span>` : `<span>Estimated download: ${bytes(s.estimated_download_bytes)}. Exact size is measured when the download starts.</span>`;
    $('#model-cards').innerHTML = `<article><h3>Qwen <span class="status-pill">${esc(s.state.replaceAll('_',' '))}</span></h3>
      <p>${esc(s.model)} · ${esc(s.state==='downloading' ? p.stage : s.stage)}</p>${progress}
      <dl><dt>Download location</dt><dd>${esc(s.cache_path)}</dd><dt>Free disk space</dt><dd>${bytes(s.disk_free)}</dd><dt>Model revision</dt><dd>${esc(s.revision||p.revision||'Resolved on download; then pinned')}</dd>
      <dt>Device</dt><dd>${esc(s.device||d.gpu||d.device)} · ${esc(s.precision||'precision selected on start')}</dd><dt>Available memory</dt><dd>${bytes(d.memory_free)} / ${bytes(d.memory_total)}</dd><dt>Diagnostics</dt><dd>PyTorch ${esc(d.torch)} · CUDA ${d.cuda?'available':'unavailable'} · MPS ${d.mps?'available':'unavailable'}. ${s.ready?'Inference and neuron hooks passed.':'Start verifies inference and hooks.'}</dd></dl></article>
      <article><h3>Experiment data</h3><p>Precomputed reference map, lenses and connections available. ${s.data?.sweep_layers||0} sweep layers · ${s.data?.embeddings||0} embedding files.</p><p class="dim">${s.data?.sweep_layers?'Original experiments are available.':'Original sweep examples are not installed. Downloading Qwen enables new live experiments, not historical sweep results.'}</p></article>
      <article><h3>AI assistant <span class="status-pill">Optional</span></h3><p>${esc($('#a-auth-status')?.textContent||'ChatGPT sign-in is available in AI assistant.')}</p><p class="dim">Connecting ChatGPT does not load Qwen. No sign-in is needed for the demo.</p></article>`;
    $('#model-error').textContent = s.restriction_reason || s.error || '';
    const busy = ['loading','downloading','stopping'].includes(s.state);
    for (const b of document.querySelectorAll('[data-model]')) {
      const a=b.dataset.model;
      b.disabled=actionBusy || s.restricted || (a==='download' ? busy||s.ready||s.installed : a==='download/cancel' ? s.state!=='downloading' : a==='start' ? busy||s.ready||!s.installed : !['ready','loading'].includes(s.state));
      if (a==='download') b.textContent=s.error||p.downloaded ? 'Retry / Resume download' : 'Download Qwen';
    }
    $('#model-device').disabled=busy||s.ready;
    $('#model-open').textContent=`Model & setup · ${s.ready?'Ready':s.state.replaceAll('_',' ')}`;
    for (const b of document.querySelectorAll(liveSelectors)) { b.setAttribute('aria-disabled',String(!s.ready)); b.title=s.ready?'Run a live Qwen experiment':'Qwen is not ready. Activate to open model setup.'; }
    if (tour?.active) {
      $('#demo-pause').innerHTML=s.ready?'':`Tour paused for live experiments: ${esc(s.state.replaceAll('_',' '))}. <button id="tour-setup">Open model setup</button>`;
      $('#tour-setup')?.addEventListener('click',setup);
      if (!s.ready && controller) controller.abort();
      for (const b of document.querySelectorAll('[data-demo-run]')) b.disabled=!s.ready||!!controller;
      $('#demo-next').disabled=!!controller || (tour.step===0&&!s.ready);
    }
  }
  async function refresh() {
    try { status=await request('/api/model/status'); bridge.onStatus(status); renderStatus(); if ($('#model-notice').textContent.startsWith('Server unavailable')) $('#model-notice').textContent=''; }
    catch { $('#model-notice').textContent='Server unavailable. Launch the app again, then reopen setup.'; }
  }
  async function poll() { await refresh(); setTimeout(poll,2000); }
  poll();
  $('#model-actions').onclick=async e => {
    const b=e.target.closest('[data-model]'); if (!b||actionBusy) return;
    actionBusy=true; renderStatus(); $('#model-notice').textContent='';
    try { await request(`/api/model/${b.dataset.model}`,{device:$('#model-device').value}); }
    catch (error) { $('#model-notice').textContent=error.message; }
    finally { actionBusy=false; await refresh(); }
  };
  $('#welcome').onclick=e => {
    const action=e.target.dataset.welcome; if (!action) return;
    storage.set('explorer.welcomed',VERSION); if (action!=='map') $('#welcome').returnFocus=null; $('#welcome').close();
    if (action==='setup') setup(); else if (action==='demo') startTour();
  };
  if (!storage.get('explorer.welcomed')) openDialog('#welcome');

  const steps = [
    ['Prepare Qwen','The guided demo takes about 5–10 minutes after setup, depending on your computer. Download and start Qwen, then wait for its inference and neuron-hook checks to pass. ChatGPT is optional.'],
    ['Understand the map','Each sheet is a processing layer. Each dot is an MLP neuron. Attention blocks move information between text positions. We selected L22 N2742: layer 22, neuron 2742. These indices start at zero.'],
    ['Inspect the evidence','The Details panel shows token lenses and Connections. This is precomputed reference data. Tokens suggest associations; links are hypotheses about influence. Neither guarantees what this neuron will do in a new experiment.'],
    ['Run a steering experiment','Use the same prompt with a positive push, no intervention, and a negative push. The three columns run sequentially to reduce memory. Each output is limited to 96 new tokens.'],
    ['Compare what actually happened','Compare the generated text: topic, self-description, wording and coherence. Similar answers are a valid outcome. A difference demonstrates an effect for this prompt and dose, not a fixed meaning for the neuron. No particular theme is guaranteed.'],
    ['Explore activation','Run a short text through the same neuron. Yellow highlights positive activation and pink highlights negative activation. Stronger color means larger magnitude within this text. Hover or focus a token to read its values.'],
    ['Continue independently','Try another prompt or dose below, then explore connected neurons. You can use Neuron chat and Run text after the tour. Repeated experiments help distinguish consistent effects from prompt-specific behavior.']
  ];
  function persist() { storage.set(KEY,tour); }
  async function startTour(restart=false) {
    previousFocus=document.activeElement.closest('dialog') ? $('#help-open') : document.activeElement;
    for (const dialog of document.querySelectorAll('.learn-dialog[open]')) { dialog.returnFocus=null; dialog.close(); }
    storage.set('explorer.welcomed',VERSION);
    if (restart && tour?.active) await exitTour();
    if (!tour || restart || tour.finished) tour={version:VERSION,step:0,lastCompleted:-1,saved:bridge.snapshot(),prompt:'Who are you?',dose:.8};
    tour.active=true; tour.running=false; persist(); await showStep();
  }
  async function exitTour() {
    if (controller) controller.abort(); controller=null;
    tour.active=false; tour.running=false; persist(); $('#demo-panel').hidden=true;
    document.querySelectorAll('.tour-target').forEach(el=>el.classList.remove('tour-target'));
    await bridge.restore(tour.saved); previousFocus?.focus();
  }
  async function showStep() {
    $('#demo-panel').hidden=false; $('#demo-count').textContent=`LIVE GUIDED DEMO · ${tour.step+1} / ${steps.length}`;
    $('#demo-title').textContent=steps[tour.step][0]; $('#demo-error').textContent='';
    await bridge.select(tour.step);
    if (!tour.active) return;
    document.querySelectorAll('.tour-target').forEach(el=>el.classList.remove('tour-target'));
    $((['#model-open','#jump','#tab-details','#tab-chat','#tab-chat','#c-dir','#jump'])[tour.step])?.classList.add('tour-target');
    let content=`<p>${steps[tour.step][1]}</p>`;
    if (tour.step===0) content+='<button id="demo-setup">Open model setup</button>';
    if ([3,6].includes(tour.step)) content+=`<label for="demo-prompt">Prompt</label><input id="demo-prompt" value="${esc(tour.prompt)}"><label for="demo-dose">Additive strength</label><input id="demo-dose" type="number" min="0" max="5" step="0.1" value="${tour.dose}"><button data-demo-run="comparison">Run comparison</button><button id="demo-cancel" hidden>Cancel run</button>`;
    if ([3,4,6].includes(tour.step)) content+='<div id="demo-results"></div>';
    if (tour.step===5) content+=`<label for="demo-text">Text to inspect</label><textarea id="demo-text" rows="2">${esc(tour.text||'Hello, who are you? Tell me about the stars.')}</textarea><button data-demo-run="activation">Run activation</button><button id="demo-cancel" hidden>Cancel run</button><div id="demo-activation"></div><p class="dim">Highlighting is relative to this run. It does not prove causation.</p>`;
    if (tour.step===6) content+='<p>Explore further: <button data-help="steering">Steering</button> · <button data-help="connections">Connections</button> · <button data-help="activations">Activations</button> · <button data-help="data">Data limitations</button></p>';
    $('#demo-body').innerHTML=content;
    $('#demo-setup')?.addEventListener('click',setup);
    $('#demo-back').disabled=tour.step===0;
    $('#demo-next').textContent=tour.step===6?'Finish':'Next';
    $('#demo-skip').hidden=tour.step===6;
    $('#demo-prompt')?.addEventListener('input',e=>{tour.prompt=e.target.value;persist();});
    $('#demo-dose')?.addEventListener('change',e=>{tour.dose=Math.max(0,Math.min(5,Number(e.target.value)||0));persist();});
    $('#demo-text')?.addEventListener('input',e=>{tour.text=e.target.value;persist();});
    document.querySelector('[data-demo-run]')?.addEventListener('click',e=>run(e.target.dataset.demoRun));
    $('#demo-cancel')?.addEventListener('click',()=>controller?.abort());
    results(); if (tour.activation && $('#demo-activation')) $('#demo-activation').innerHTML=activationHTML(tour.activation);
    renderStatus(); $('#demo-title').focus();
  }
  function results() {
    if (!$('#demo-results')) return;
    const result=tour.result;
    if (!result) { $('#demo-results').innerHTML='<p class="dim">No live comparison has been run yet.</p>'; return; }
    const words = text => (text.toLowerCase().match(/[\p{L}]{4,}/gu)||[]).filter(w=>!['that','this','with','have','from','your','what','would','about','which','their','there','they','will','been','were','into','some','like','more','than','also'].includes(w));
    const baseline = new Set(words(result.outputs[1]));
    const difference = i => [...new Set(words(result.outputs[i]))].filter(w=>!baseline.has(w)).slice(0,8);
    const observed = result.complete ? `<p><b>Observed comparison:</b> ${[0,2].map(i=> {
      if (result.outputs[i]===result.outputs[1]) return `${i===0?'Positive':'Negative'} steering produced the same text as baseline.`;
      const unique=difference(i);
      return `${i===0?'Positive':'Negative'} steering changed the text${unique.length ? ` and included words absent from baseline: ${esc(unique.join(', '))}` : ', with no new long words after ignoring common words'}.`;
    }).join(' ')} This describes the text difference; it does not establish a general neuron function.</p>` : '';
    $('#demo-results').innerHTML=`<div class="demo-columns">${['Positive steering','Unchanged baseline','Negative steering'].map((label,i)=>`<article><h3>${label}</h3><div>${esc(result.outputs[i]||'Waiting for this column…')}</div></article>`).join('')}</div>${observed}<details><summary>Experiment record · ${result.complete?'completed':'partial / interrupted'}</summary><pre>${esc(JSON.stringify(result.metadata,null,2))}</pre></details><p class="dim">${result.metadata.normalization==='local_reference'?'Normalization estimated locally from a reference passage because sweep statistics were absent.':'Normalization uses installed sweep statistics.'}</p>`;
  }
  async function run(kind) {
    if (!status?.ready || controller) return;
    const ctl=new AbortController(), activeTour=tour; controller=ctl; tour.running=true;
    $('#demo-error').textContent=''; $('#demo-cancel').hidden=false;
    for (const selector of ['#demo-back','#demo-next','#demo-skip','#demo-restart']) $(selector).disabled=true;
    renderStatus();
    try {
      if (kind==='comparison') {
        tour.result={outputs:['','',''],complete:false,metadata:{prompt:tour.prompt||'Who are you?',neuron:'L22 N2742',dose:tour.dose,mode:'add',max_new_tokens:96,revision:status.revision,device:status.device,precision:status.precision,normalization:'local_reference'}};
        results();
        await streamSteering({layer:22,n:2742,prompt:tour.prompt||'Who are you?',dose:tour.dose,mode:'add',max_new:96,sequential:true,history:[]},ctl.signal,event=>{
          if (controller!==ctl || tour!==activeTour) return;
          if (event.meta) Object.assign(tour.result.metadata,event.meta);
          if (event.row!=null) { tour.result.outputs[event.row]+=event.t; results(); }
        });
        if (!ctl.signal.aborted) tour.result.complete=true;
        results();
      } else {
        tour.text=$('#demo-text').value;
        const result=await readNeuron({layer:22,n:2742},tour.text,ctl.signal);
        if (!ctl.signal.aborted) { tour.activation=result; tour.activationRecord={text:tour.text,neuron:'L22 N2742',revision:status.revision,device:status.device,precision:status.precision,chat_template:true}; $('#demo-activation').innerHTML=activationHTML(result); }
      }
    } catch (error) { if (tour.active && controller===ctl) $('#demo-error').textContent=error.name==='AbortError'?'Run cancelled or runtime stopped. Partial output is retained; press Run to try again.':error.message; }
    finally {
      if (controller===ctl) {
        controller=null; tour.running=false; persist();
        $('#demo-cancel').hidden=true; $('#demo-back').disabled=false; $('#demo-skip').disabled=false; $('#demo-restart').disabled=false; renderStatus();
      }
    }
  }
  async function move(delta,skip=false) {
    if (controller) return;
    if (delta>0&&!skip) tour.lastCompleted=Math.max(tour.lastCompleted,tour.step);
    if (tour.step===6&&delta>0) { tour.finished=true; await exitTour(); return; }
    tour.step=Math.max(0,Math.min(6,tour.step+delta)); persist(); await showStep();
  }
  $('#demo-back').onclick=()=>move(-1); $('#demo-next').onclick=()=>move(1); $('#demo-skip').onclick=()=>move(1,true);
  $('#demo-exit').onclick=exitTour; $('#demo-restart').onclick=()=>startTour(true);
  $('#tour-start').onclick=()=>startTour(); $('#tour-restart').onclick=()=>startTour(true);
}
