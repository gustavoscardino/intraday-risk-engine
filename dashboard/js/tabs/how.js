// How the models work: interactive explanations. EWMA weights and the GARCH shock response are
// computed here from their formulas; the Normal x t tail uses the engine's own VaR/ES factors;
// the CAViaR recursion and the intraday profile come from a real run of the WebAssembly engine.
import { fmt, t } from '../i18n.js';
import { DEFAULT_RUN, runComparison, tailFactors } from '../engine.js';
import { ACCENT, INK, MUTED, axis, baseOption, makeChart, modelColor, sortModels } from '../charts.js';

let el;
let charts = [];
let run = null;
let state = { lambda: 0.94, alpha: 0.06, beta: 0.91, nu: 5, conf: 0.99, day: null };

export async function init(root) {
  el = root;
  render();
  runComparison(DEFAULT_RUN).then((r) => { run = r; render(); })
    .catch((e) => { const s = el.querySelector('#how-status'); if (s) s.textContent = `${t('app.error')}: ${e.message}`; });
}

function clear() { charts.forEach((c) => c.dispose()); charts = []; }
const chart = (id) => { const c = makeChart(el.querySelector(id)); charts.push(c); return c; };

export function render() {
  clear();
  el.innerHTML = `
    <h1>${t('how.title')}</h1>
    <p class="lead">${t('how.lead')}</p>

    <div class="section card">
      <h2>1 · EWMA</h2>
      <p>${t('how.ewma.text')}</p>
      <p class="formula">σ²(t) = λ · σ²(t−1) + (1 − λ) · r²(t)</p>
      <div class="slider-row"><label for="how-lambda">λ</label><input id="how-lambda" type="range" min="0.80" max="0.995" step="0.005" value="${state.lambda}"><output id="how-lambda-out"></output></div>
      <div id="how-ewma" class="chart short"></div>
      <p class="small muted" id="how-ewma-note"></p>
    </div>

    <div class="section card">
      <h2>2 · GARCH(1,1)-t</h2>
      <p>${t('how.garch.text')}</p>
      <p class="formula">h(t) = ω + α · ε²(t−1) + β · h(t−1)&nbsp;&nbsp;&nbsp;&nbsp;ε = √h · z,&nbsp;&nbsp;z ~ t(ν)</p>
      <div class="grid grid-2">
        <div class="slider-row"><label for="how-alpha">α</label><input id="how-alpha" type="range" min="0.01" max="0.25" step="0.01" value="${state.alpha}"><output id="how-alpha-out"></output></div>
        <div class="slider-row"><label for="how-beta">β</label><input id="how-beta" type="range" min="0.50" max="0.98" step="0.01" value="${state.beta}"><output id="how-beta-out"></output></div>
      </div>
      <div id="how-garch" class="chart short"></div>
      <p class="small muted" id="how-garch-note"></p>
    </div>

    <div class="section card">
      <h2>3 · ${t('how.tail.title')}</h2>
      <p>${t('how.tail.text')}</p>
      <div class="controls">
        <div class="slider-row" style="flex:1;min-width:260px"><label for="how-nu">ν</label><input id="how-nu" type="range" min="3" max="30" step="1" value="${state.nu}"><output id="how-nu-out"></output></div>
        <label class="field">${t('how.tail.conf')}
          <select id="how-conf">${[0.95, 0.975, 0.99].map((c) => `<option value="${c}" ${c === state.conf ? 'selected' : ''}>${fmt.pct(c, 1)}</option>`).join('')}</select></label>
      </div>
      <div id="how-tail" class="chart"></div>
      <p class="small" id="how-tail-note"></p>
    </div>

    <div class="section card">
      <h2>4 · Realized-CAViaR</h2>
      <p>${t('how.caviar.text')}</p>
      <p class="formula">q(t+1) = β₀ + β₁ · q(t) + β₂ · |R(t)| + β₃ · √RV(t)&nbsp;&nbsp;&nbsp;&nbsp;R = Σ r,&nbsp;&nbsp;RV = Σ r²&nbsp;&nbsp;(${t('how.caviar.window')})</p>
      <p class="status ${run ? '' : 'busy'}" id="how-status">${run ? t('how.example', { ticker: run.ticker, ms: fmt.num(run.elapsed_ms / 1000, 1) }) : t('how.loading')}</p>
      <div class="controls"><label class="field">${t('how.caviar.day')}<select id="how-day"></select></label></div>
      <div id="how-caviar" class="chart tall"></div>
      <p class="small" id="how-caviar-betas"></p>
    </div>

    <div class="section card">
      <h2>5 · ${t('how.diurnal.title')}</h2>
      <p>${t('how.diurnal.text')}</p>
      <div id="how-diurnal" class="chart short"></div>
    </div>`;

  ewma();
  garch();
  tail();
  caviar();
  diurnal();
}

// ---------------------------------------------------------------- EWMA weights
function ewma() {
  const c = chart('#how-ewma');
  const input = el.querySelector('#how-lambda');
  const draw = () => {
    const lam = Number(input.value);
    state.lambda = lam;
    el.querySelector('#how-lambda-out').textContent = fmt.num(lam, 3);
    const k = Array.from({ length: 80 }, (_, i) => i);
    c.setOption(baseOption({
      animation: false,
      tooltip: { trigger: 'axis', valueFormatter: (v) => fmt.pct(v, 2) },
      xAxis: { type: 'category', data: k.map((i) => String(i)), name: t('how.ewma.x'), nameLocation: 'middle', nameGap: 28, ...axis() },
      yAxis: { type: 'value', name: t('how.ewma.y'), ...axis({ axisLabel: { formatter: (v) => fmt.pct(v, 0), color: MUTED } }) },
      series: [{ type: 'bar', data: k.map((i) => (1 - lam) * lam ** i), itemStyle: { color: modelColor('EWMA-N') }, barCategoryGap: '20%' }],
    }), true);
    const half = Math.log(0.5) / Math.log(lam);
    const median = run ? medianOf(run.calibrations.map((x) => x.lambda)) : null;
    el.querySelector('#how-ewma-note').textContent = t('how.ewma.note', { half: fmt.num(half, 1), min: fmt.num(half * 15, 0) })
      + (median ? ' ' + t('how.ewma.est', { lam: fmt.num(median, 3) }) : '');
  };
  input.addEventListener('input', draw);
  draw();
}

// ---------------------------------------------------------------- GARCH shock response
function garch() {
  const c = chart('#how-garch');
  const a = el.querySelector('#how-alpha'), b = el.querySelector('#how-beta');
  const draw = () => {
    let alpha = Number(a.value), beta = Number(b.value);
    if (alpha + beta > 0.995) { beta = 0.995 - alpha; b.value = beta; }
    state.alpha = alpha; state.beta = beta;
    el.querySelector('#how-alpha-out').textContent = fmt.num(alpha, 2);
    el.querySelector('#how-beta-out').textContent = fmt.num(beta, 2);
    // long-run variance 1; a 5-sigma shock at bar 0; expected variance afterwards
    const p = alpha + beta, h1 = (1 - p) + alpha * 25 + beta;
    const k = Array.from({ length: 120 }, (_, i) => i);
    const h = k.map((i) => (i === 0 ? 1 : 1 + p ** (i - 1) * (h1 - 1)));
    c.setOption(baseOption({
      animation: false,
      tooltip: { trigger: 'axis', valueFormatter: (v) => fmt.num(v, 2) + '×' },
      xAxis: { type: 'category', data: k.map(String), name: t('how.garch.x'), nameLocation: 'middle', nameGap: 28, ...axis() },
      yAxis: { type: 'value', name: t('how.garch.y'), min: 0, ...axis() },
      series: [{ type: 'line', data: h, showSymbol: false, lineStyle: { width: 2.5, color: modelColor('GARCH-t') }, areaStyle: { color: 'rgba(27,58,107,.08)' },
        markLine: { silent: true, symbol: 'none', lineStyle: { type: 'dashed', color: MUTED }, label: { formatter: t('how.garch.longrun'), color: MUTED, position: 'insideEndTop' }, data: [{ yAxis: 1 }] } }],
    }), true);
    el.querySelector('#how-garch-note').textContent = t('how.garch.note', { p: fmt.num(p, 3), half: fmt.num(Math.log(0.5) / Math.log(p), 1) });
  };
  a.addEventListener('input', draw);
  b.addEventListener('input', draw);
  draw();
}

// ---------------------------------------------------------------- Normal x t tail
function lgamma(x) {  // Lanczos approximation
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const tt = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(tt) - tt + Math.log(a);
}
const normalPdf = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
function unitTPdf(z, nu) {
  const s = Math.sqrt(nu / (nu - 2)), x = z * s;
  return s * Math.exp(lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * Math.log(nu * Math.PI) - ((nu + 1) / 2) * Math.log1p(x * x / nu));
}

function tail() {
  const c = chart('#how-tail');
  const nuIn = el.querySelector('#how-nu'), confIn = el.querySelector('#how-conf');
  let token = 0;
  const draw = async () => {
    const nu = Number(nuIn.value), conf = Number(confIn.value), my = ++token;
    state.nu = nu; state.conf = conf;
    el.querySelector('#how-nu-out').textContent = String(nu);
    const zs = Array.from({ length: 241 }, (_, i) => -6 + i * 0.05);
    const series = (name, color, f, dash) => ({ name, type: 'line', showSymbol: false, data: zs.map((z) => [z, f(z)]), lineStyle: { width: 2.5, color, type: dash ? 'dashed' : 'solid' }, itemStyle: { color } });
    let n, s;
    try { [n, s] = await Promise.all([tailFactors(0, conf), tailFactors(nu, conf)]); } catch { n = s = null; }
    if (my !== token) return;
    const marks = n && s ? [
      { xAxis: -n.var, lineStyle: { color: modelColor('EWMA-N') }, label: { formatter: `VaR N`, color: INK } },
      { xAxis: -s.var, lineStyle: { color: modelColor('EWMA-t') }, label: { formatter: `VaR t`, color: INK } },
    ] : [];
    c.setOption(baseOption({
      animation: false,
      legend: { top: 0, right: 10, textStyle: { color: MUTED } },
      grid: { left: 60, right: 24, top: 64, bottom: 44 },
      tooltip: { trigger: 'axis', valueFormatter: (v) => (v == null ? '–' : Number(v).toExponential(2)) },
      xAxis: { type: 'value', min: -6, max: 6, name: t('how.tail.x'), nameLocation: 'middle', nameGap: 28, ...axis() },
      yAxis: { type: 'log', min: 1e-7, name: t('how.tail.y'), ...axis({ axisLabel: { formatter: (v) => v.toExponential(0), color: MUTED } }) },
      series: [
        { ...series(t('how.tail.normal'), modelColor('EWMA-N'), normalPdf, false), markLine: { silent: true, symbol: 'none', lineStyle: { type: 'dashed', width: 1.5 }, data: marks } },
        series(t('how.tail.student', { nu }), modelColor('EWMA-t'), (z) => unitTPdf(z, nu), false),
      ],
    }), true);
    el.querySelector('#how-tail-note').innerHTML = n && s ? t('how.tail.note', {
      conf: fmt.pct(conf, 1), vn: fmt.num(n.var, 3), vt: fmt.num(s.var, 3), en: fmt.num(n.es, 3), et: fmt.num(s.es, 3),
      dv: fmt.pct(s.var / n.var - 1, 1), de: fmt.pct(s.es / n.es - 1, 1),
    }) : t('how.loading');
  };
  nuIn.addEventListener('input', draw);
  confIn.addEventListener('change', draw);
  draw();
}

// ---------------------------------------------------------------- CAViaR on a real session
function caviar() {
  const sel = el.querySelector('#how-day');
  if (!run) { sel.disabled = true; return; }
  const f = run.forecasts;
  const days = [...new Set(f.t.map((x) => Math.floor(x / 86400)))];
  if (state.day == null || !days.includes(state.day)) state.day = days[Math.min(days.length - 1, 5)];
  sel.innerHTML = days.map((d) => `<option value="${d}" ${d === state.day ? 'selected' : ''}>${fmt.date(d * 86400)}</option>`).join('');
  const c = chart('#how-caviar');
  const draw = () => {
    state.day = Number(sel.value);
    const idx = f.t.map((x, i) => [x, i]).filter(([x]) => Math.floor(x / 86400) === state.day).map(([, i]) => i);
    const labels = idx.map((i) => fmt.time(f.t[i] + 900, false));
    const models = sortModels(run.models);
    const pct = (v) => (v == null ? null : 100 * v);
    c.setOption(baseOption({
      animation: false,
      legend: { top: 0, textStyle: { color: MUTED } },
      tooltip: { trigger: 'axis', valueFormatter: (v) => (v == null ? '–' : fmt.num(v, 3) + '%') },
      grid: [{ left: 60, right: 24, top: 40, height: '52%' }, { left: 60, right: 24, top: '72%', bottom: 36 }],
      xAxis: [
        { type: 'category', data: labels, gridIndex: 0, ...axis({ axisLabel: { show: false } }) },
        { type: 'category', data: labels, gridIndex: 1, name: t('how.caviar.x'), nameLocation: 'middle', nameGap: 26, ...axis() },
      ],
      yAxis: [
        { type: 'value', gridIndex: 0, name: t('how.caviar.y1'), ...axis() },
        { type: 'value', gridIndex: 1, name: t('how.caviar.y2'), ...axis() },
      ],
      series: [
        { name: t('how.caviar.ret'), type: 'bar', data: idx.map((i) => pct(f.y[i])), itemStyle: { color: '#C9D0DB' }, xAxisIndex: 0, yAxisIndex: 0 },
        ...models.map((m) => ({ name: m, type: 'line', step: 'end', showSymbol: false, data: idx.map((i) => pct(f.q[run.models.indexOf(m)][i])),
          lineStyle: { width: m === 'Realized-CAViaR' ? 3 : 1.5, color: modelColor(m) }, itemStyle: { color: modelColor(m) }, xAxisIndex: 0, yAxisIndex: 0 })),
        { name: '|R|', type: 'bar', data: idx.map((i) => pct(Math.abs(f.R[i]))), itemStyle: { color: '#F2C29A' }, xAxisIndex: 1, yAxisIndex: 1 },
        { name: '√RV', type: 'line', data: idx.map((i) => pct(Math.sqrt(f.RV[i]))), showSymbol: false, lineStyle: { color: ACCENT, width: 2 }, itemStyle: { color: ACCENT }, xAxisIndex: 1, yAxisIndex: 1 },
      ],
    }), true);
    const first = f.t[idx[0]];
    const cal = [...run.calibrations].filter((x) => x.t <= first).pop() || run.calibrations[0];
    const [b0, b1, b2, b3] = cal.caviar;
    el.querySelector('#how-caviar-betas').innerHTML = t('how.caviar.betas', {
      b0: fmt.num(1e4 * b0, 2), b1: fmt.num(b1, 3), b2: fmt.num(b2, 3), b3: fmt.num(b3, 3),
    });
  };
  sel.addEventListener('change', draw);
  draw();
}

// ---------------------------------------------------------------- intraday profile
function diurnal() {
  if (!run || !run.diurnal_profile.length) return;
  const c = chart('#how-diurnal');
  const labels = run.diurnal_profile.map((_, k) => { const m = 600 + 15 * (k + 1); return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`; });
  c.setOption(baseOption({
    tooltip: { trigger: 'axis', valueFormatter: (v) => fmt.num(v, 2) + '×' },
    xAxis: { type: 'category', data: labels, name: t('how.diurnal.x'), nameLocation: 'middle', nameGap: 28, ...axis() },
    yAxis: { type: 'value', name: t('how.diurnal.y'), ...axis() },
    series: [{ type: 'bar', data: run.diurnal_profile, itemStyle: { color: modelColor('GARCH-t') },
      markLine: { silent: true, symbol: 'none', lineStyle: { type: 'dashed', color: MUTED }, label: { formatter: '1×', color: MUTED, position: 'insideEndTop' }, data: [{ yAxis: 1 }] } }],
  }), true);
}

function medianOf(v) {
  const s = v.filter(Number.isFinite).sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : null;
}
