// Live replay (level B): the user picks a stock and the parameters, the WebAssembly engine computes
// the four models in the browser, and the result is played back bar by bar through a forecast
// source (sources.js), with exception counters updated as the playhead moves.
import { fmt, t } from '../i18n.js';
import { load } from '../data.js';
import { DEFAULT_RUN, runComparison } from '../engine.js';
import { Player, ReplaySource } from '../sources.js';
import { BAD, MUTED, axis, baseOption, escapeHtml, makeChart, modelColor, sortModels, wilson } from '../charts.js';

const WINDOW = { synthetic: 60, b3_real: 30 };
const SESSIONS = { synthetic: 150, b3_real: 60 };
const SPEEDS = [5, 20, 60, 240];
const VIEW = 240;  // bars visible while playing

let el, datasets, chart = null, player = null, source = null, run = null;
let params = { ...DEFAULT_RUN };
let lambdaMode = 'estimated';
let speed = 20;
let pos = 0;
let busy = false;
let lastDraw = 0;
let pre = null;  // per-model hits and cumulative counts

export async function init(root) {
  el = root;
  datasets = await load('datasets');
  render();
  compute();
}

export function render() {
  if (chart) { chart.dispose(); chart = null; }
  const ds = params.dataset;
  el.innerHTML = `
    <h1>${t('rp.title')}</h1>
    <p class="lead">${t('rp.lead')}</p>
    <div class="card">
      <div class="controls">
        <label class="field">${t('rp.dataset')}<select id="rp-ds">
          ${['b3_real', 'synthetic'].map((d) => `<option value="${d}" ${d === ds ? 'selected' : ''}>${t(`ds.${d}`)}</option>`).join('')}</select></label>
        <label class="field">${t('rp.ticker')}<select id="rp-ticker">
          ${datasets[ds].map((a) => `<option value="${a.ticker}" ${a.ticker === params.ticker ? 'selected' : ''}>${escapeHtml(a.ticker)} · ${escapeHtml(a.name)}</option>`).join('')}</select></label>
        <label class="field">${t('rp.conf')}<select id="rp-conf">
          ${[0.95, 0.975, 0.99].map((c) => `<option value="${c}" ${c === params.confidence ? 'selected' : ''}>${fmt.pct(c, 1)}</option>`).join('')}</select></label>
        <label class="field">${t('rp.window')}<input id="rp-window" type="number" min="5" max="${SESSIONS[ds] - 1}" value="${params.windowDays}"></label>
        <label class="field">λ (EWMA)<select id="rp-lmode">
          <option value="estimated" ${lambdaMode === 'estimated' ? 'selected' : ''}>${t('rp.lambda.est')}</option>
          <option value="fixed" ${lambdaMode === 'fixed' ? 'selected' : ''}>${t('rp.lambda.fixed')}</option></select></label>
        <label class="field">${t('rp.lambda.value')}<input id="rp-lambda" type="number" min="0.8" max="0.999" step="0.005" value="${params.lambda > 0 ? params.lambda : 0.94}" ${lambdaMode === 'fixed' ? '' : 'disabled'}></label>
        <label class="field check"><input id="rp-diurnal" type="checkbox" ${params.diurnal ? 'checked' : ''}>${t('rp.diurnal')}</label>
        <button class="btn" id="rp-run" ${busy ? 'disabled' : ''}>${t('rp.run')}</button>
      </div>
      <p class="status ${busy ? 'busy' : ''}" id="rp-status" style="margin-top:10px">${statusText()}</p>
    </div>

    <div class="section grid grid-side" id="rp-main">
      <div class="card">
        <div class="player">
          <button class="btn" id="rp-play">${player?.playing ? t('rp.pause') : t('rp.play')}</button>
          <button class="btn secondary" id="rp-end">${t('rp.end')}</button>
          <label class="field" style="flex-direction:row;align-items:center;gap:6px">${t('rp.speed')}
            <select id="rp-speed">${SPEEDS.map((s) => `<option value="${s}" ${s === speed ? 'selected' : ''}>${s} ${t('rp.perSec')}</option>`).join('')}</select></label>
          <input id="rp-pos" type="range" min="0" max="${run ? run.forecasts.t.length - 1 : 0}" value="${pos}" aria-label="${escapeHtml(t('rp.position'))}">
          <span class="small muted" id="rp-clock"></span>
        </div>
        <div id="rp-chart" class="chart tall"></div>
        <p class="small muted">${t('rp.chart.note')}</p>
      </div>
      <div class="card">
        <h3>${t('rp.counters')}</h3>
        <p class="small muted" id="rp-progress"></p>
        <div class="counters" id="rp-counters"></div>
      </div>
    </div>

    <div class="section card">
      <h3>${t('rp.summary')}</h3>
      <p class="small muted">${t('rp.summary.note')}</p>
      <div class="table-wrap" id="rp-table"></div>
    </div>
    <div class="note">${t('rp.how')}</div>`;

  const narrow = window.matchMedia('(max-width: 960px)').matches;
  if (narrow) el.querySelector('#rp-main').style.gridTemplateColumns = '1fr';
  bind();
  if (run) { setupChart(); draw(pos, true); table(); }
}

function statusText() {
  if (busy) return t('rp.computing', { ticker: params.ticker });
  if (!run) return '';
  return t('rp.done', { s: fmt.num(run.elapsed_ms / 1000, 1), n: fmt.int(run.forecasts.t.length), c: run.calibrations.length,
    w: run.window_days, ticker: run.ticker });
}

function bind() {
  const $ = (id) => el.querySelector(id);
  $('#rp-ds').addEventListener('change', (e) => {
    params.dataset = e.target.value;
    params.ticker = datasets[params.dataset][0].ticker;
    params.windowDays = WINDOW[params.dataset];
    render();
  });
  $('#rp-lmode').addEventListener('change', (e) => { lambdaMode = e.target.value; $('#rp-lambda').disabled = lambdaMode !== 'fixed'; });
  $('#rp-run').addEventListener('click', () => {
    params.ticker = $('#rp-ticker').value;
    params.confidence = Number($('#rp-conf').value);
    const w = Math.round(Number($('#rp-window').value));
    params.windowDays = Math.min(SESSIONS[params.dataset] - 1, Math.max(5, Number.isFinite(w) ? w : WINDOW[params.dataset]));
    const lam = Number($('#rp-lambda').value);
    params.lambda = lambdaMode === 'fixed' && lam >= 0.8 && lam < 1 ? lam : 0;
    params.diurnal = $('#rp-diurnal').checked;
    compute();
  });
  $('#rp-play').addEventListener('click', () => {
    if (!player) return;
    if (player.playing) player.pause(); else player.play();
    $('#rp-play').textContent = player.playing ? t('rp.pause') : t('rp.play');
  });
  $('#rp-end').addEventListener('click', () => { if (player) { player.pause(); player.seek(source.size - 1); } });
  $('#rp-speed').addEventListener('change', (e) => { speed = Number(e.target.value); player?.setSpeed(speed); });
  $('#rp-pos').addEventListener('input', (e) => { if (player) { player.pause(); player.seek(Number(e.target.value)); } });
}

async function compute() {
  busy = true;
  player?.pause();
  render();
  try {
    run = await runComparison(params);
    source = new ReplaySource(run);
    precompute();
    player = new Player(source, { onFrame: (i, playing) => { pos = i; draw(i, !playing); } });
    player.setSpeed(speed);
    pos = source.size - 1;
  } catch (e) {
    run = null;
    busy = false;
    render();
    el.querySelector('#rp-status').textContent = `${t('app.error')}: ${e.message}`;
    return;
  }
  busy = false;
  render();
}

function precompute() {
  const f = run.forecasts, n = f.t.length, M = run.models.length;
  const hit = run.models.map((_, m) => f.y.map((y, i) => (f.eval[i] === 1 && y != null && y < f.q[m][i] ? 1 : 0)));
  const cumHit = hit.map((h) => { let s = 0; return h.map((x) => (s += x)); });
  let e = 0;
  const cumEval = f.eval.map((x) => (e += x));
  const anyHit = Array.from({ length: n }, (_, i) => hit.some((h) => h[i]));
  pre = { hit, cumHit, cumEval, anyHit, M };
}

function setupChart() {
  chart = makeChart(el.querySelector('#rp-chart'));
  const f = run.forecasts;
  const models = sortModels(run.models);
  const labels = f.t.map((x) => fmt.time(x + 900));
  chart.setOption(baseOption({
    animation: false,
    legend: { type: 'scroll', top: 0, textStyle: { color: MUTED } },
    grid: { left: 60, right: 24, top: 40, bottom: 70 },
    tooltip: { trigger: 'axis', formatter: tooltip },
    xAxis: { type: 'category', data: labels, ...axis({ axisLabel: { color: MUTED, hideOverlap: true } }) },
    yAxis: { type: 'value', name: t('rp.chart.y'), ...axis({ axisLabel: { formatter: (v) => fmt.num(v, 1) + '%', color: MUTED } }) },
    dataZoom: [{ type: 'slider', bottom: 18, height: 22 }, { type: 'inside' }],
    series: [
      { name: t('rp.chart.ret'), type: 'bar', data: [], barCategoryGap: '15%', large: false },
      ...models.map((m) => ({ name: m, type: 'line', step: 'end', showSymbol: false, data: [],
        lineStyle: { width: m === 'Realized-CAViaR' ? 2.2 : 1.4, color: modelColor(m) }, itemStyle: { color: modelColor(m) } })),
    ],
  }), true);
}

function tooltip(items) {
  if (!items.length) return '';
  const i = items[0].dataIndex, f = run.forecasts;
  const y = f.y[i];
  const rows = sortModels(run.models).map((m) => {
    const k = run.models.indexOf(m), q = f.q[k][i], hit = pre.hit[k][i];
    return `<span style="color:${modelColor(m)}">●</span> ${escapeHtml(m)}: ${fmt.num(100 * q, 3)}%${hit ? ` <b style="color:${BAD}">${t('rp.exception')}</b>` : ''}`;
  });
  return `<b>${fmt.time(f.t[i] + 900)}</b><br>${t('rp.chart.ret')}: ${y == null ? t('rp.pending') : fmt.num(100 * y, 3) + '%'}<br>${rows.join('<br>')}`;
}

function draw(i, force = false) {
  if (!run || !chart) return;
  const now = performance.now();
  if (!force && now - lastDraw < 40) { counters(i); return; }
  lastDraw = now;
  const f = run.forecasts;
  const models = sortModels(run.models);
  const upto = (arr, map) => arr.map((v, k) => (k <= i ? map(v, k) : null));
  const bars = upto(f.y, (v, k) => (v == null ? null : { value: 100 * v, itemStyle: { color: pre.anyHit[k] ? BAD : '#B9C2CF' } }));
  const series = [{ data: bars }, ...models.map((m) => ({ data: upto(f.q[run.models.indexOf(m)], (v) => 100 * v) }))];
  const playing = player?.playing;
  const opt = { series };
  if (playing) opt.dataZoom = [{ startValue: Math.max(0, i - VIEW), endValue: i }];
  chart.setOption(opt);
  const range = el.querySelector('#rp-pos');
  if (range && Number(range.value) !== i) range.value = i;
  const clock = el.querySelector('#rp-clock');
  if (clock) clock.textContent = f.t[i] == null ? '–' : fmt.time(f.t[i] + 900);
  const play = el.querySelector('#rp-play');
  if (play) play.textContent = playing ? t('rp.pause') : t('rp.play');
  counters(i);
}

function counters(i) {
  const box = el.querySelector('#rp-counters');
  if (!box || !pre) return;
  const tau = 1 - run.confidence;
  const nEval = pre.cumEval[i];
  el.querySelector('#rp-progress').textContent = t('rp.progress', { n: fmt.int(nEval), exp: fmt.num(nEval * tau, 1) });
  box.innerHTML = sortModels(run.models).map((m) => {
    const k = run.models.indexOf(m), x = pre.cumHit[k][i], expected = nEval * tau;
    const ratio = expected > 0 ? x / expected : 0;
    const s = run.summaries.find((r) => r.model === m);
    return `<div class="counter">
      <span class="name"><span class="dot" style="background:${modelColor(m)}"></span>${escapeHtml(m)}</span><span></span>
      <span class="val">${x} / ${fmt.num(expected, 1)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, 50 * ratio)}%;background:${modelColor(m)}"></div><div class="bar-mark" style="left:50%"></div></div>
      <span class="meta">${t('rp.counter.meta', { rate: fmt.pct(nEval ? x / nEval : 0), ns: fmt.num(s.mean_ns, 1), bytes: s.state_bytes })}</span>
    </div>`;
  }).join('');
}

function table() {
  const box = el.querySelector('#rp-table');
  const rows = sortModels(run.models).map((m) => {
    const s = run.summaries.find((r) => r.model === m);
    const [lo, hi] = wilson(s.exceptions, s.n);
    return `<tr><td><span class="dot" style="background:${modelColor(m)}"></span> ${escapeHtml(m)}</td>
      <td class="num">${fmt.int(s.n)}</td><td class="num">${s.exceptions}</td>
      <td class="num">${fmt.pct(s.rate)} <span class="muted small">[${fmt.pct(lo)}–${fmt.pct(hi)}]</span></td>
      <td class="num">${fmt.p(s.kupiec_p)}</td><td class="num">${fmt.p(s.cc_p)}</td>
      <td class="num">${fmt.num(1e4 * s.pinball, 3)}</td><td class="num">${fmt.num(s.mean_ns, 1)}</td><td class="num">${fmt.num(s.p99_ns, 1)}</td>
      <td class="num">${fmt.int(s.state_bytes)}</td></tr>`;
  }).join('');
  box.innerHTML = `<table><thead><tr><th>${t('tbl.model')}</th><th class="num">n</th><th class="num">${t('tbl.exc')}</th><th class="num">${t('tbl.rate')}</th>
    <th class="num">Kupiec p</th><th class="num">CC p</th><th class="num">${t('tbl.pinball')}</th><th class="num">ns</th><th class="num">p99 ns</th><th class="num">${t('tbl.state')}</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}
