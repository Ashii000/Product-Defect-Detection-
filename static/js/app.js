/* DeepVision AI — front-end logic (no build step, no external libraries) */
(function () {
  'use strict';

  // ------------------------------------------------------------------ config
  const DEFAULT_THRESHOLD = 1.2;
  const CONF_MIN = 0.6;                       // classifier confidence below this => "review"
  const MAX_BYTES = 16 * 1024 * 1024;
  const HISTORY_MAX = 8;
  const EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];
  const TYPE_TO_EXT = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/bmp': 'bmp' };
  const MODE_HELP = {
    combined: 'Runs both methods and compares their verdicts.',
    classifier: 'Mode A: ResNet18 classifier trained on your labeled images.',
    anomaly: 'Mode B: compares the image with normal reference images and draws a heatmap.'
  };

  const $ = (id) => document.getElementById(id);
  const els = {};
  [
    'pillClassifier', 'pillReference', 'pillDevice', 'btnRefresh', 'dropzone', 'fileInput', 'dzTitle', 'dzSub',
    'modeHelp', 'thresholdRange', 'thresholdValue', 'btnResetThreshold', 'opacityRange', 'opacityValue',
    'btnAnalyze', 'btnReset', 'emptyState', 'workspace', 'errorBox', 'errorText', 'btnDismissError',
    'verdictCard', 'verdictBadge', 'verdictTitle', 'verdictReason', 'verdictMeta', 'toggleHeatmap',
    'imgOriginal', 'imgBase', 'imgHeat', 'heatPlaceholder', 'legend', 'loading', 'loadingText', 'btnCancel',
    'cardA', 'cardAMsg', 'cardABody', 'predLabel', 'predConf', 'probBars',
    'cardB', 'cardBMsg', 'cardBBody', 'scoreNum', 'gaugeFill', 'gaugeMarker',
    'btnDownloadOverlay', 'btnDownloadReport', 'historyPanel', 'historyList', 'btnClearHistory'
  ].forEach((id) => { els[id] = $(id); });
  const modeButtons = Array.prototype.slice.call(document.querySelectorAll('.seg[data-mode]'));

  // ------------------------------------------------------------------- state
  const state = {
    file: null, fileUrl: null,
    mode: 'combined',
    result: null,                // { classifier, anomaly, width, height, elapsed_ms }
    threshold: DEFAULT_THRESHOLD,
    opacity: 0.7,
    showHeat: true,
    busy: false, controller: null, reqId: 0,
    history: [], currentEntry: null, nextId: 1
  };

  // ----------------------------------------------------------------- helpers
  const show = (el, visible) => { el.hidden = !visible; };
  const pct = (x) => (x * 100).toFixed(1) + '%';
  const fmtSize = (b) => (b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(b / 1024)) + ' KB');
  const setSrc = (img, url) => { if (img.getAttribute('src') !== url) img.setAttribute('src', url); };
  const inHistory = (url) => state.history.some((e) => e.url === url);

  function showError(msg) { els.errorText.textContent = msg; show(els.errorBox, true); }
  function clearError() { els.errorText.textContent = ''; show(els.errorBox, false); }

  function setPill(el, st, text) {
    el.setAttribute('data-state', st);
    el.querySelector('.pill-text').textContent = text;
  }

  function usesA() { return state.mode !== 'anomaly'; }
  function usesB() { return state.mode !== 'classifier'; }

  function currentHeatmap() {
    const a = state.result && state.result.anomaly;
    return usesB() && a && a.available && a.heatmap ? a.heatmap : null;
  }

  // ---------------------------------------------------------------- verdict
  function evaluate(result, mode, threshold) {
    if (!result) return null;
    const useA = mode !== 'anomaly';
    const useB = mode !== 'classifier';
    const states = [], reasons = [], problems = [];

    if (useA && result.classifier) {
      const c = result.classifier;
      if (c.available) {
        if (c.confidence < CONF_MIN) {
          states.push('uncertain');
          reasons.push('Classifier is unsure: "' + c.label + '" at only ' + pct(c.confidence));
        } else {
          states.push(c.is_defect ? 'defect' : 'ok');
          reasons.push('Classifier: ' + c.label + ' (' + pct(c.confidence) + ')');
        }
      } else { problems.push(c.message); }
    }
    if (useB && result.anomaly) {
      const a = result.anomaly;
      if (a.available) {
        const over = a.score > threshold;
        states.push(over ? 'defect' : 'ok');
        reasons.push('Anomaly score ' + a.score.toFixed(2) + (over ? ' is above' : ' is within') + ' threshold ' + threshold.toFixed(2));
      } else { problems.push(a.message); }
    }

    if (!states.length) {
      return { state: 'none', badge: 'N/A', title: 'No result available', reason: problems.join(' ') || 'Nothing to evaluate.' };
    }
    let st = 'review';
    if (states.every((s) => s === 'defect')) st = 'fail';
    else if (states.every((s) => s === 'ok')) st = 'pass';
    if (st === 'review' && states.indexOf('uncertain') === -1) reasons.push('The two methods disagree - please inspect manually.');

    const meta = {
      pass: ['PASS', 'No defect detected'],
      fail: ['FAIL', 'Defect suspected'],
      review: ['REVIEW', 'Needs human review']
    }[st];
    return { state: st, badge: meta[0], title: meta[1], reason: reasons.concat(problems).join(' · ') };
  }

  // --------------------------------------------------------------- rendering
  function render() {
    const hasFile = !!state.file;
    show(els.emptyState, !hasFile);
    show(els.workspace, hasFile);
    els.dropzone.classList.toggle('has-file', hasFile);

    els.btnAnalyze.disabled = !hasFile || state.busy;
    els.btnAnalyze.textContent = state.busy ? 'Analyzing…' : (state.result ? 'Re-analyze' : 'Analyze image');
    els.btnReset.disabled = !hasFile;
    show(els.loading, state.busy);
    els.loadingText.textContent = 'Analyzing "' + (state.file ? state.file.name : '') + '"…';

    if (hasFile) renderWorkspace();
    renderHistory();
  }

  function renderWorkspace() {
    const r = state.result;
    setSrc(els.imgOriginal, state.fileUrl);
    setSrc(els.imgBase, state.fileUrl);

    // heatmap overlay
    const heat = currentHeatmap();
    if (heat) setSrc(els.imgHeat, heat);
    els.imgHeat.style.opacity = String(state.opacity);
    show(els.imgHeat, !!heat && state.showHeat);
    show(els.legend, !!heat && state.showHeat);
    show(els.heatPlaceholder, !heat);
    if (!heat) {
      let msg = 'Press "Analyze image" to generate the heatmap.';
      if (!usesB()) msg = 'The heatmap comes from Mode B. Switch to Combined or B · Anomaly to see it.';
      else if (r && r.anomaly && !r.anomaly.available) msg = r.anomaly.message;
      els.heatPlaceholder.textContent = msg;
    }

    renderVerdict();

    // classifier card
    const showA = usesA() && !!(r && r.classifier);
    show(els.cardA, showA);
    if (showA) renderClassifierCard(r.classifier);

    // anomaly card
    const showB = usesB() && !!(r && r.anomaly);
    show(els.cardB, showB);
    if (showB) renderAnomalyCard(r.anomaly);

    els.btnDownloadOverlay.disabled = !heat || state.busy;
    els.btnDownloadReport.disabled = !r || state.busy;
  }

  function renderVerdict() {
    const card = els.verdictCard;
    const r = state.result;
    if (state.busy) {
      card.setAttribute('data-state', 'busy');
      els.verdictBadge.textContent = '…';
      els.verdictTitle.textContent = 'Analyzing…';
      els.verdictReason.textContent = 'Please wait.';
      els.verdictMeta.textContent = '';
      return;
    }
    const ev = evaluate(r, state.mode, state.threshold);
    if (!ev) {
      card.setAttribute('data-state', 'idle');
      els.verdictBadge.textContent = 'READY';
      els.verdictTitle.textContent = 'Ready to analyze';
      els.verdictReason.textContent = 'Press "Analyze image" to start.';
      els.verdictMeta.textContent = '';
      return;
    }
    card.setAttribute('data-state', ev.state);
    els.verdictBadge.textContent = ev.badge;
    els.verdictTitle.textContent = ev.title;
    els.verdictReason.textContent = ev.reason;
    els.verdictMeta.textContent = (r.elapsed_ms / 1000).toFixed(1) + ' s · ' + r.width + '×' + r.height + ' px';
  }

  function renderClassifierCard(c) {
    show(els.cardAMsg, !c.available);
    show(els.cardABody, !!c.available);
    if (!c.available) { els.cardAMsg.textContent = c.message; return; }

    els.predLabel.textContent = c.label;
    els.predConf.textContent = 'Confidence ' + pct(c.confidence) + (c.confidence < CONF_MIN ? ' · low - review manually' : '');
    els.probBars.textContent = '';
    const defects = c.defect_classes || [];
    const entries = Object.keys(c.probabilities).map((k) => [k, c.probabilities[k]]).sort((a, b) => b[1] - a[1]);
    const fills = [];
    entries.forEach((pair, i) => {
      const row = document.createElement('div');
      row.className = 'prob-row' + (i === 0 ? ' top' : '');
      const head = document.createElement('div');
      head.className = 'prob-head';
      const name = document.createElement('span'); name.textContent = pair[0];
      const val = document.createElement('span'); val.textContent = pct(pair[1]);
      head.appendChild(name); head.appendChild(val);
      const track = document.createElement('div'); track.className = 'prob-track';
      const fill = document.createElement('div');
      fill.className = 'prob-fill ' + (defects.indexOf(pair[0]) !== -1 ? 'defect' : 'ok');
      track.appendChild(fill);
      row.appendChild(head); row.appendChild(track);
      els.probBars.appendChild(row);
      fills.push([fill, pair[1]]);
    });
    setTimeout(() => { fills.forEach((f) => { f[0].style.width = (f[1] * 100).toFixed(1) + '%'; }); }, 20);
  }

  function renderAnomalyCard(a) {
    show(els.cardBMsg, !a.available);
    show(els.cardBBody, !!a.available);
    if (!a.available) { els.cardBMsg.textContent = a.message; return; }
    renderGauge();
  }

  function renderGauge() {
    const a = state.result && state.result.anomaly;
    if (!a || !a.available) return;
    const max = Math.max(3, state.threshold * 1.5, a.score * 1.1);
    els.scoreNum.textContent = a.score.toFixed(2);
    els.gaugeFill.style.width = Math.min(100, (a.score / max) * 100) + '%';
    els.gaugeFill.classList.toggle('over', a.score > state.threshold);
    els.gaugeMarker.style.left = Math.min(100, (state.threshold / max) * 100) + '%';
  }

  function renderHistory() {
    show(els.historyPanel, state.history.length > 0);
    els.historyList.textContent = '';
    state.history.forEach((entry) => {
      const ev = evaluate(entry.result, entry.mode, state.threshold);
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'history-item' + (entry === state.currentEntry ? ' current' : '');
      btn.setAttribute('data-id', String(entry.id));
      btn.title = 'Open ' + entry.name;
      const img = document.createElement('img');
      img.className = 'history-thumb'; img.alt = ''; img.src = entry.url;
      const info = document.createElement('div'); info.className = 'history-info';
      const name = document.createElement('div'); name.className = 'history-name'; name.textContent = entry.name;
      const tag = document.createElement('span');
      tag.className = 'history-tag ' + (ev ? ev.state : '');
      tag.textContent = ev ? ev.badge : '—';
      info.appendChild(name); info.appendChild(tag);
      btn.appendChild(img); btn.appendChild(info);
      li.appendChild(btn);
      els.historyList.appendChild(li);
    });
  }

  function updateModeButtons() {
    modeButtons.forEach((b) => {
      const on = b.getAttribute('data-mode') === state.mode;
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    els.modeHelp.textContent = MODE_HELP[state.mode];
  }

  // ---------------------------------------------------------------- status
  async function loadStatus() {
    setPill(els.pillClassifier, 'loading', 'Classifier…');
    setPill(els.pillReference, 'loading', 'References…');
    setPill(els.pillDevice, 'loading', 'Device…');
    try {
      const resp = await fetch('/api/status', { cache: 'no-store' });
      const d = await resp.json();
      if (!resp.ok || !d.ok) throw new Error('bad status');
      if (d.classifier.ready) setPill(els.pillClassifier, 'ok', 'Classifier ready · ' + d.classifier.classes.length + ' classes');
      else setPill(els.pillClassifier, 'warn', 'Classifier not trained');
      const n = d.reference_count;
      if (n >= 2) setPill(els.pillReference, 'ok', n + ' reference images');
      else setPill(els.pillReference, 'warn', n + ' reference image' + (n === 1 ? '' : 's') + ' (need 2+)');
      setPill(els.pillDevice, 'ok', 'Device: ' + d.device);
    } catch (err) {
      setPill(els.pillClassifier, 'bad', 'Server offline');
      setPill(els.pillReference, 'bad', 'References: unknown');
      setPill(els.pillDevice, 'bad', 'Device: unknown');
    }
  }

  // ----------------------------------------------------------------- files
  function normalizeFile(f) {
    const ext = (f.name.split('.').pop() || '').toLowerCase();
    if (f.name.indexOf('.') !== -1 && EXTENSIONS.indexOf(ext) !== -1) return f;
    const mapped = TYPE_TO_EXT[f.type];
    if (!mapped) return null;
    return new File([f], (f.name || 'pasted-image') + '.' + mapped, { type: f.type });
  }

  function handleFiles(list) {
    if (!list || !list.length) return;
    const original = list[0];
    const f = normalizeFile(original);
    if (!f) { showError('Unsupported file type. Please choose a JPG, PNG, BMP or WEBP image.'); return; }
    if (f.size > MAX_BYTES) { showError('File too large (' + fmtSize(f.size) + '). Maximum is 16 MB.'); return; }
    if (f.size === 0) { showError('This file is empty.'); return; }
    setFile(f);
  }

  function releaseUrl() {
    if (state.fileUrl && !inHistory(state.fileUrl)) URL.revokeObjectURL(state.fileUrl);
    state.fileUrl = null;
  }

  function setFile(f) {
    cancelAnalysis();
    releaseUrl();
    state.file = f;
    state.fileUrl = URL.createObjectURL(f);
    state.result = null;
    state.currentEntry = null;
    els.dzTitle.textContent = f.name;
    els.dzSub.textContent = fmtSize(f.size) + ' · click, drop or paste to replace';
    clearError();
    render();
    scrollToResults();
  }

  function scrollToResults() {
    try {
      if (window.matchMedia && window.matchMedia('(max-width: 991px)').matches && els.workspace.scrollIntoView) {
        els.workspace.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } catch (e) { /* ignore */ }
  }

  function resetAll() {
    cancelAnalysis();
    releaseUrl();
    state.file = null;
    state.result = null;
    state.currentEntry = null;
    els.fileInput.value = '';
    els.dzTitle.textContent = 'Drop an image here';
    els.dzSub.innerHTML = 'or click to browse · paste with Ctrl+V<br>JPG, PNG, BMP, WEBP · max 16 MB';
    clearError();
    render();
  }

  // -------------------------------------------------------------- analysis
  function cancelAnalysis() {
    if (state.controller) { state.controller.abort(); state.controller = null; }
    state.reqId += 1;
    state.busy = false;
  }

  function needsRun() {
    const r = state.result;
    if (!r) return true;
    return (usesA() && !r.classifier) || (usesB() && !r.anomaly);
  }

  async function analyze() {
    if (!state.file || state.busy) return;
    clearError();
    const myId = ++state.reqId;
    const controller = new AbortController();
    state.controller = controller;
    state.busy = true;
    render();

    const form = new FormData();
    form.append('image', state.file, state.file.name);
    form.append('mode', state.mode);

    try {
      const resp = await fetch('/api/analyze', { method: 'POST', body: form, signal: controller.signal });
      let data = null;
      try { data = await resp.json(); } catch (e) { data = null; }
      if (myId !== state.reqId) return;                      // cancelled / replaced
      if (!resp.ok || !data || !data.ok) throw new Error((data && data.error) || 'Server error (' + resp.status + ')');
      const prev = state.result;
      state.result = {
        classifier: data.classifier || (prev && prev.classifier) || null,
        anomaly: data.anomaly || (prev && prev.anomaly) || null,
        width: data.width, height: data.height, elapsed_ms: data.elapsed_ms
      };
      upsertHistory();
    } catch (err) {
      if (err && err.name === 'AbortError') return;
      if (myId !== state.reqId) return;
      const msg = (err instanceof TypeError || (err && err.message === 'Failed to fetch'))
        ? 'Cannot reach the server. Is "python server.py" still running?'
        : (err && err.message) || 'Something went wrong.';
      showError(msg);
    } finally {
      if (myId === state.reqId) { state.busy = false; state.controller = null; }
      render();
    }
  }

  function setMode(mode) {
    if (state.mode === mode) return;
    state.mode = mode;
    updateModeButtons();
    if (state.file && (state.busy || (state.result && needsRun()))) {
      cancelAnalysis();
      analyze();
    } else {
      render();
    }
  }

  // --------------------------------------------------------------- history
  function upsertHistory() {
    let entry = state.currentEntry;
    if (!entry || state.history.indexOf(entry) === -1) {
      entry = { id: state.nextId++, file: state.file, url: state.fileUrl, name: state.file.name, mode: state.mode, result: state.result };
      state.history.unshift(entry);
      state.currentEntry = entry;
      while (state.history.length > HISTORY_MAX) {
        const old = state.history.pop();
        if (old.url !== state.fileUrl) URL.revokeObjectURL(old.url);
      }
    } else {
      entry.mode = state.mode;
      entry.result = state.result;
    }
  }

  function restoreEntry(id) {
    const entry = state.history.find((e) => e.id === id);
    if (!entry) return;
    cancelAnalysis();
    if (state.fileUrl !== entry.url) releaseUrl();
    state.file = entry.file;
    state.fileUrl = entry.url;
    state.result = entry.result;
    state.mode = entry.mode;
    state.currentEntry = entry;
    updateModeButtons();
    els.dzTitle.textContent = entry.name;
    els.dzSub.textContent = fmtSize(entry.file.size) + ' · click, drop or paste to replace';
    clearError();
    render();
    scrollToResults();
  }

  function clearHistory() {
    state.history.forEach((e) => { if (e.url !== state.fileUrl) URL.revokeObjectURL(e.url); });
    state.history = [];
    state.currentEntry = null;
    renderHistory();
  }

  // ------------------------------------------------------------- downloads
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function baseName() { return (state.file ? state.file.name : 'image').replace(/\.[^.]+$/, ''); }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Could not load the image for export.'));
      img.src = src;
    });
  }

  async function downloadOverlay() {
    const heatSrc = currentHeatmap();
    if (!heatSrc || !state.fileUrl) return;
    try {
      const [base, heat] = await Promise.all([loadImage(state.fileUrl), loadImage(heatSrc)]);
      const scale = Math.min(1, 2000 / Math.max(base.naturalWidth, base.naturalHeight));
      const w = Math.max(1, Math.round(base.naturalWidth * scale));
      const h = Math.max(1, Math.round(base.naturalHeight * scale));
      const canvas = document.createElement('canvas');
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(base, 0, 0, w, h);
      if (state.showHeat) { ctx.globalAlpha = state.opacity; ctx.drawImage(heat, 0, 0, w, h); }
      canvas.toBlob((blob) => {
        if (!blob) { showError('Could not create the PNG.'); return; }
        downloadBlob(blob, baseName() + '_deepvision.png');
      }, 'image/png');
    } catch (err) {
      showError(err.message || 'Download failed.');
    }
  }

  function downloadReport() {
    const r = state.result;
    if (!r || !state.file) return;
    const ev = evaluate(r, state.mode, state.threshold);
    const report = {
      generated_at: new Date().toISOString(),
      file: state.file.name,
      image_size: { width: r.width, height: r.height },
      mode: state.mode,
      anomaly_threshold: state.threshold,
      verdict: ev ? ev.state : null,
      summary: ev ? ev.reason : null,
      classifier: usesA() ? r.classifier : null,
      anomaly: null,
      processing_ms: r.elapsed_ms,
      note: 'Educational prototype - not a certified inspection system.'
    };
    if (usesB() && r.anomaly) {
      report.anomaly = {};
      Object.keys(r.anomaly).forEach((k) => { if (k !== 'heatmap') report.anomaly[k] = r.anomaly[k]; });
    }
    downloadBlob(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }), baseName() + '_report.json');
  }

  // ---------------------------------------------------------------- events
  function hasFiles(e) {
    const t = e.dataTransfer && e.dataTransfer.types;
    return !!t && Array.prototype.indexOf.call(t, 'Files') !== -1;
  }

  function bindEvents() {
    // upload: click / keyboard / input
    els.dropzone.addEventListener('click', () => els.fileInput.click());
    els.dropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); els.fileInput.click(); }
    });
    els.fileInput.addEventListener('change', () => {
      const files = els.fileInput.files;
      handleFiles(files);
      try { els.fileInput.value = ''; } catch (e) { /* ignore */ }
    });

    // drag & drop anywhere on the page
    window.addEventListener('dragover', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      els.dropzone.classList.add('drag');
    });
    window.addEventListener('dragleave', (e) => {
      if (!e.relatedTarget) els.dropzone.classList.remove('drag');
    });
    window.addEventListener('drop', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      els.dropzone.classList.remove('drag');
      handleFiles(e.dataTransfer.files);
    });

    // paste from clipboard
    document.addEventListener('paste', (e) => {
      const items = e.clipboardData && e.clipboardData.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].kind === 'file' && items[i].type.indexOf('image/') === 0) {
          const f = items[i].getAsFile();
          if (f) { e.preventDefault(); handleFiles([f]); return; }
        }
      }
    });

    // mode buttons
    modeButtons.forEach((b) => b.addEventListener('click', () => setMode(b.getAttribute('data-mode'))));

    // sliders
    els.thresholdRange.addEventListener('input', () => {
      state.threshold = parseFloat(els.thresholdRange.value);
      els.thresholdValue.textContent = state.threshold.toFixed(2);
      if (state.file) { renderVerdict(); renderGauge(); renderHistory(); }
    });
    els.btnResetThreshold.addEventListener('click', () => {
      els.thresholdRange.value = String(DEFAULT_THRESHOLD);
      els.thresholdRange.dispatchEvent(new Event('input'));
    });
    els.opacityRange.addEventListener('input', () => {
      state.opacity = parseInt(els.opacityRange.value, 10) / 100;
      els.opacityValue.textContent = els.opacityRange.value + '%';
      els.imgHeat.style.opacity = String(state.opacity);
    });
    els.toggleHeatmap.addEventListener('change', () => {
      state.showHeat = els.toggleHeatmap.checked;
      if (state.file) renderWorkspace();
    });

    // actions
    els.btnAnalyze.addEventListener('click', analyze);
    els.btnReset.addEventListener('click', resetAll);
    els.btnCancel.addEventListener('click', () => { cancelAnalysis(); render(); });
    els.btnDismissError.addEventListener('click', clearError);
    els.btnRefresh.addEventListener('click', loadStatus);
    els.btnDownloadOverlay.addEventListener('click', downloadOverlay);
    els.btnDownloadReport.addEventListener('click', downloadReport);
    els.btnClearHistory.addEventListener('click', clearHistory);
    els.historyList.addEventListener('click', (e) => {
      const btn = e.target.closest ? e.target.closest('.history-item') : null;
      if (btn) restoreEntry(parseInt(btn.getAttribute('data-id'), 10));
    });
  }

  // ------------------------------------------------------------------ init
  // browsers (e.g. Firefox) may restore slider/switch values after a reload: read them back
  state.threshold = parseFloat(els.thresholdRange.value) || DEFAULT_THRESHOLD;
  state.opacity = (parseInt(els.opacityRange.value, 10) || 0) / 100;
  els.opacityValue.textContent = els.opacityRange.value + '%';
  state.showHeat = els.toggleHeatmap.checked;
  bindEvents();
  updateModeButtons();
  els.thresholdValue.textContent = state.threshold.toFixed(2);
  render();
  loadStatus();

  // tiny hook used by automated tests
  window.__deepvision = { state: state, evaluate: evaluate };
})();