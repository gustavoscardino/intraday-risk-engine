// Architecture and quality: the diagrams, the ATAM evaluation (scorecard, findings, faulty-feed
// robustness) and the backtests of the portfolio engines (ewma, garch_t).
import { fmt, t } from '../i18n.js';
import { load } from '../data.js';
import { MUTED, axis, baseOption, escapeHtml, makeChart } from '../charts.js';

const DIAGRAMS = ['00_visao_geral', '01_fluxo_de_dados', '03_ciclo_15min', '05_realized_caviar', '04_modulos_garch_t', '02_modulos'];
const ENGINES = [['ewma', 'EWMA'], ['garch_t', 'GARCH-t'], ['realized_caviar', 'Realized-CAViaR']];
let el, atam, portfolio, charts = [];
let diagram = '00_visao_geral', kind = 'risk', run = 'synthetic';

export async function init(root) {
  el = root;
  [atam, portfolio] = await Promise.all([load('atam'), load('portfolio')]);
  render();
}

const statusClass = (s) => ({ 'MET': 'met', 'NOT MET': 'notmet', 'PARTIAL': 'partial' }[s] || 'na');

export function render() {
  charts.forEach((c) => c.dispose());
  charts = [];
  el.innerHTML = `
    <h1>${t('q.title')}</h1>
    <p class="lead">${t('q.lead')}</p>

    <div class="section card">
      <h2>${t('q.diagrams')}</h2>
      <div class="seg" style="flex-wrap:wrap">${DIAGRAMS.map((d) => `<button type="button" data-diagram="${d}" aria-pressed="${d === diagram}">${t(`q.d.${d}`)}</button>`).join('')}</div>
      <p class="small muted" style="margin-top:10px">${t(`q.d.${diagram}.text`)}</p>
      <img class="diagram" src="assets/${diagram}.svg" alt="${escapeHtml(t(`q.d.${diagram}`))}">
    </div>

    <div class="section card">
      <h2>${t('q.atam')}</h2>
      <p class="muted">${t('q.atam.lead')}</p>
      <div class="table-wrap">${scorecard()}</div>
      <p class="small muted">${t('q.atam.note')}</p>
    </div>

    <div class="section grid grid-2">
      <div class="card">
        <h3>${t('q.findings')}</h3>
        <div class="seg" style="flex-wrap:wrap">${['risk', 'non-risk', 'sensitivity', 'trade-off'].map((k) => `<button type="button" data-kind="${k}" aria-pressed="${k === kind}">${t(`q.kind.${k}`)}</button>`).join('')}</div>
        <p class="small muted" style="margin-top:8px">${t('q.findings.lang')}</p>
        <div class="findings">${atam.findings.filter((f) => f.kind === kind).map((f) => `<div class="finding"><h4><span class="tag">${escapeHtml(f.id)}</span>${escapeHtml(f.title)}</h4><p>${escapeHtml(f.text)}</p></div>`).join('')}</div>
      </div>
      <div class="card">
        <h3>${t('q.robust')}</h3>
        <p class="small muted">${t('q.robust.note')}</p>
        <div id="q-robust" class="chart"></div>
      </div>
    </div>

    <div class="section card">
      <h2>${t('q.portfolio')}</h2>
      <p class="muted">${t('q.portfolio.lead')}</p>
      <div class="seg">${['synthetic', 'b3_real', 'synthetic_degraded'].map((r) => `<button type="button" data-run="${r}" aria-pressed="${r === run}">${t(`q.run.${r}`)}</button>`).join('')}</div>
      <div class="table-wrap" style="margin-top:12px">${portfolioTable()}</div>
    </div>`;

  el.querySelectorAll('[data-diagram]').forEach((b) => b.addEventListener('click', () => { diagram = b.dataset.diagram; render(); }));
  el.querySelectorAll('[data-kind]').forEach((b) => b.addEventListener('click', () => { kind = b.dataset.kind; render(); }));
  el.querySelectorAll('[data-run]').forEach((b) => b.addEventListener('click', () => { run = b.dataset.run; render(); }));
  robust();
}

function scorecard() {
  const head = `<tr><th>ID</th><th>${t('q.sc.attr')}</th><th>${t('q.sc.threshold')}</th>${ENGINES.map(([, n]) => `<th>${n}</th>`).join('')}</tr>`;
  const body = atam.scorecard.map((r) => {
    const thr = r.op === 'report' ? t('q.sc.report') : `${r.op} ${r.threshold} ${r.unit}`;
    const cells = ENGINES.map(([k]) => {
      const st = r[`${k}_status`] ?? '', v = r[`${k}_value`];
      const val = v == null || v === '' ? '' : ` <span class="small muted">${typeof v === 'number' ? fmt.num(v, Math.abs(v) < 10 ? 2 : 0) : escapeHtml(v)}</span>`;
      return `<td title="${escapeHtml(r[`${k}_detail`] ?? '')}"><span class="chip ${statusClass(st)}">${escapeHtml(st || 'N/A')}</span>${val}</td>`;
    }).join('');
    return `<tr><td><b>${escapeHtml(r.id)}</b></td><td>${escapeHtml(r.attribute)}: ${escapeHtml(r.refinement)}</td><td class="small">${escapeHtml(thr)}</td>${cells}</tr>`;
  }).join('');
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function robust() {
  const c = makeChart(el.querySelector('#q-robust'));
  charts.push(c);
  const rows = atam.degraded_feed;
  const label = (v) => t(`q.variant.${v}`) === `q.variant.${v}` ? v : t(`q.variant.${v}`);
  c.setOption(baseOption({
    legend: { top: 0, textStyle: { color: MUTED } },
    tooltip: { trigger: 'axis', valueFormatter: (v) => fmt.num(v, 2) + '%' },
    grid: { left: 10, right: 30, top: 40, bottom: 30, containLabel: true },
    xAxis: { type: 'value', ...axis({ axisLabel: { formatter: (v) => v + '%', color: MUTED } }) },
    yAxis: { type: 'category', data: rows.map((r) => label(r.variant)), inverse: true, ...axis({ axisLabel: { color: MUTED } }) },
    series: [
      { name: t('q.robust.gt1'), type: 'bar', data: rows.map((r) => r.share_dev_gt_1pct), itemStyle: { color: '#9CC3E8' } },
      { name: t('q.robust.gt5'), type: 'bar', data: rows.map((r) => r.share_dev_gt_5pct), itemStyle: { color: '#B5561A' } },
    ],
  }));
}

function portfolioTable() {
  const rows = [];
  for (const [engine, name] of [['ewma', 'ewma'], ['garch_t', 'garch_t']]) {
    const data = portfolio[engine]?.[run];
    if (!data) continue;
    for (const r of data) {
      if (!r.model || String(r.model).startsWith('overnight') || String(r.model).startsWith('book')) continue;
      rows.push(`<tr><td>${name}</td><td>${escapeHtml(r.model)}</td><td class="num">${fmt.pct(r.confidence, 1)}</td><td class="num">${fmt.int(r.n)}</td>
        <td class="num">${r.exceptions}</td><td class="num">${fmt.pct(r.rate)}</td><td class="num">${fmt.p(r.kupiec_p)}</td>
        <td><span class="chip ${r.traffic_light}">${t(`zone.${r.traffic_light}`)}</span></td><td class="num">${fmt.num(r.es_ratio, 3)}</td><td class="num">${fmt.p(r.es_z2_p)}</td></tr>`);
    }
  }
  return `<table><thead><tr><th>${t('q.pf.engine')}</th><th>${t('tbl.model')}</th><th class="num">${t('q.pf.conf')}</th><th class="num">n</th><th class="num">${t('tbl.exc')}</th>
    <th class="num">${t('tbl.rate')}</th><th class="num">Kupiec p</th><th>${t('q.pf.zone')}</th><th class="num">ES ratio</th><th class="num">ES Z2 p</th></tr></thead>
    <tbody>${rows.join('') || `<tr><td colspan="10" class="muted">${t('q.pf.none')}</td></tr>`}</tbody></table>`;
}
