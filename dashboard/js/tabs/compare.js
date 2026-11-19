// Comparison over every stock (native engine, data/comparison.json): exception rate with its 95%
// interval, per-stock pinball loss, Diebold-Mariano tests, quality x cost, and time of day.
import { fmt, t } from '../i18n.js';
import { load } from '../data.js';
import { MUTED, axis, baseOption, escapeHtml, makeChart, modelColor, sortModels, wilson } from '../charts.js';

let el, comp, charts = [];
let ds = 'b3_real';

export async function init(root) {
  el = root;
  comp = await load('comparison');
  render();
}

const chart = (id) => { const c = makeChart(el.querySelector(id)); charts.push(c); return c; };

export function render() {
  charts.forEach((c) => c.dispose());
  charts = [];
  const sum = comp.summary.filter((s) => s.dataset === ds);
  const models = sortModels(sum.map((s) => s.model));
  const stocks = sum[0]?.stocks ?? 0;
  el.innerHTML = `
    <h1>${t('cmp.title')}</h1>
    <p class="lead">${t('cmp.lead', { w: comp.windows[ds], stocks })}</p>
    <div class="seg" role="group" aria-label="${escapeHtml(t('rp.dataset'))}">
      ${['b3_real', 'synthetic'].map((d) => `<button type="button" data-ds="${d}" aria-pressed="${d === ds}">${t(`ds.${d}`)}</button>`).join('')}
    </div>

    <div class="section grid grid-2">
      <div class="card"><h3>${t('cmp.rate')}</h3><p class="small muted">${t('cmp.rate.note')}</p><div id="c-rate" class="chart"></div></div>
      <div class="card"><h3>${t('cmp.cost')}</h3><p class="small muted">${t('cmp.cost.note')}</p><div id="c-cost" class="chart"></div></div>
    </div>

    <div class="section card">
      <h3>${t('cmp.table')}</h3>
      <div class="table-wrap">${summaryTable(sum, models)}</div>
    </div>

    <div class="section grid grid-2">
      <div class="card"><h3>${t('cmp.dm')}</h3><p class="small muted">${t('cmp.dm.note')}</p><div id="c-dm" class="chart"></div>
        <p class="small">${t('cmp.dm.bonf')}</p></div>
      <div class="card"><h3>${t('cmp.hour')}</h3><p class="small muted">${t('cmp.hour.note')}</p><div id="c-hour" class="chart"></div></div>
    </div>

    <div class="section card">
      <h3>${t('cmp.heat')}</h3><p class="small muted">${t('cmp.heat.note')}</p>
      <div id="c-heat" class="chart" style="height:${Math.max(360, 26 * stocks + 90)}px"></div>
    </div>
    <p class="small muted">${t('ov.source', { code: escapeHtml(String(comp.code).replace(/\.+$/, '')) })}</p>`;

  el.querySelectorAll('[data-ds]').forEach((b) => b.addEventListener('click', () => { ds = b.dataset.ds; render(); }));
  rate(sum, models);
  cost(sum, models);
  dm(models);
  hour(models);
  heat(models);
}

function summaryTable(sum, models) {
  const rows = models.map((m) => {
    const s = sum.find((x) => x.model === m);
    const [lo, hi] = wilson(s.exceptions, s.n);
    return `<tr><td><span class="dot" style="background:${modelColor(m)}"></span> ${escapeHtml(m)}</td>
      <td class="num">${fmt.int(s.n)}</td><td class="num">${fmt.int(s.exceptions)}</td>
      <td class="num">${fmt.pct(s.rate)} <span class="muted small">[${fmt.pct(lo)}–${fmt.pct(hi)}]</span></td>
      <td class="num">${s.kupiec_rejections}/${s.stocks}</td><td class="num">${s.cc_rejections}/${s.stocks}</td>
      <td class="num">${fmt.num(1e4 * s.mean_pinball, 4)}</td><td class="num">${s.lowest_pinball_stocks}/${s.stocks}</td>
      <td class="num">${fmt.num(s.median_latency_ns, 1)}</td><td class="num">${fmt.int(s.state_bytes)}</td></tr>`;
  }).join('');
  return `<table><thead><tr><th>${t('tbl.model')}</th><th class="num">n</th><th class="num">${t('tbl.exc')}</th><th class="num">${t('tbl.rate')}</th>
    <th class="num">${t('tbl.kupiec')}</th><th class="num">${t('tbl.cc')}</th><th class="num">${t('tbl.pinball')}</th><th class="num">${t('tbl.lowest')}</th>
    <th class="num">ns</th><th class="num">${t('tbl.state')}</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function rate(sum, models) {
  const c = chart('#c-rate');
  const data = models.map((m) => { const s = sum.find((x) => x.model === m); const [lo, hi] = wilson(s.exceptions, s.n); return { m, r: s.rate, lo, hi }; });
  const expected = sum[0].expected_rate;
  c.setOption(baseOption({
    tooltip: { trigger: 'item', formatter: (p) => { const d = data[p.dataIndex]; return `<b>${escapeHtml(d.m)}</b><br>${fmt.pct(d.r)} [${fmt.pct(d.lo)}–${fmt.pct(d.hi)}]`; } },
    xAxis: { type: 'category', data: models, ...axis({ axisLabel: { color: MUTED, interval: 0 } }) },
    yAxis: { type: 'value', min: 0, ...axis({ axisLabel: { formatter: (v) => fmt.pct(v, 1), color: MUTED } }) },
    series: [
      { type: 'bar', data: data.map((d) => ({ value: d.r, itemStyle: { color: modelColor(d.m) } })), barWidth: '45%',
        label: { show: true, position: 'insideTop', formatter: (p) => fmt.pct(p.value), color: '#fff', fontWeight: 700 },
        markLine: { silent: true, symbol: 'none', lineStyle: { type: 'dashed', color: '#B5561A', width: 2 },
          label: { formatter: t('cmp.expected', { r: fmt.pct(expected, 0) }), color: '#B5561A', position: 'insideEndTop' }, data: [{ yAxis: expected }] } },
      { type: 'custom', name: 'CI', renderItem: (params, api) => {
        const x = api.value(0), lo = api.coord([x, api.value(1)]), hi = api.coord([x, api.value(2)]), w = api.size([1, 0])[0] * 0.12;
        const style = { stroke: '#14213D', lineWidth: 1.5 };
        return { type: 'group', children: [
          { type: 'line', shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style },
          { type: 'line', shape: { x1: lo[0] - w, y1: lo[1], x2: lo[0] + w, y2: lo[1] }, style },
          { type: 'line', shape: { x1: hi[0] - w, y1: hi[1], x2: hi[0] + w, y2: hi[1] }, style },
        ] };
      }, data: data.map((d, i) => [i, d.lo, d.hi]), z: 10, tooltip: { show: false } },
    ],
  }));
}

function cost(sum, models) {
  const c = chart('#c-cost');
  c.setOption(baseOption({
    tooltip: { trigger: 'item', formatter: (p) => `<b>${escapeHtml(p.seriesName)}</b><br>${fmt.num(p.value[0], 1)} ns · ${fmt.num(p.value[1], 4)} bp` },
    grid: { left: 70, right: 30, top: 30, bottom: 50 },
    xAxis: { type: 'value', name: t('cmp.cost.x'), nameLocation: 'middle', nameGap: 30, scale: true, ...axis() },
    yAxis: { type: 'value', name: t('cmp.cost.y'), scale: true, ...axis({ axisLabel: { formatter: (v) => fmt.num(v, 3), color: MUTED } }) },
    series: models.map((m) => {
      const s = sum.find((x) => x.model === m);
      return { name: m, type: 'scatter', symbolSize: 22, data: [[s.median_latency_ns, 1e4 * s.mean_pinball]], itemStyle: { color: modelColor(m), borderColor: '#fff', borderWidth: 2 },
        label: { show: true, position: 'right', formatter: m, color: '#14213D', fontWeight: 600 } };
    }),
  }));
}

function dm(models) {
  const c = chart('#c-dm');
  const pooled = comp.diebold_mariano.filter((r) => r.dataset === ds && r.scope === 'pooled');
  const pairs = models.length * (models.length - 1) / 2;
  const bonf = 0.05 / pairs;
  const cells = [];
  models.forEach((a, i) => models.forEach((b, j) => {
    if (i === j) { cells.push([j, i, null, '']); return; }
    const r = pooled.find((x) => (x.model_a === a && x.model_b === b) || (x.model_a === b && x.model_b === a));
    const diff = r.model_a === a ? r.mean_loss_diff : -r.mean_loss_diff;  // row model minus column model
    const sig = r.p_value < 0.05;
    cells.push([j, i, sig ? (diff < 0 ? 1 : -1) : 0, `${diff < 0 ? '▲' : '▼'} p ${fmt.p(r.p_value)}${r.p_value < bonf ? ' *' : ''}`, r.p_value, diff]);
  }));
  c.setOption(baseOption({
    tooltip: { trigger: 'item', formatter: (p) => {
      if (p.value[2] == null) return '';
      const a = models[p.value[1]], b = models[p.value[0]];
      return `${escapeHtml(a)} × ${escapeHtml(b)}<br>${t('cmp.dm.diff')}: ${fmt.num(1e4 * p.value[5], 4)} bp<br>p = ${fmt.p(p.value[4])}`;
    } },
    grid: { left: 120, right: 20, top: 20, bottom: 70 },
    xAxis: { type: 'category', data: models, position: 'bottom', ...axis({ axisLabel: { color: MUTED, interval: 0, rotate: 20 } }), splitArea: { show: false } },
    yAxis: { type: 'category', data: models, inverse: true, ...axis({ axisLabel: { color: MUTED } }) },
    visualMap: { type: 'piecewise', show: false, dimension: 2, pieces: [
      { value: 1, color: '#9CC3E8' }, { value: -1, color: '#F2C29A' }, { value: 0, color: '#EEF1F5' }] },
    series: [{ type: 'heatmap', data: cells.map((x) => (x[2] == null ? [x[0], x[1], null] : x)),
      label: { show: true, formatter: (p) => p.value[3] || '', color: '#14213D', fontSize: 12 }, itemStyle: { borderColor: '#fff', borderWidth: 2 } }],
  }));
}

function hour(models) {
  const c = chart('#c-hour');
  const rows = comp.by_hour.filter((r) => r.dataset === ds);
  const hours = [...new Set(rows.map((r) => r.hour))].sort((a, b) => a - b);
  const expected = comp.summary.find((s) => s.dataset === ds).expected_rate;
  c.setOption(baseOption({
    legend: { top: 0, textStyle: { color: MUTED } },
    tooltip: { trigger: 'axis', valueFormatter: (v) => fmt.pct(v) },
    grid: { left: 60, right: 24, top: 44, bottom: 44 },
    xAxis: { type: 'category', data: hours.map((h) => `${h}h–${h + 1}h`), ...axis() },
    yAxis: { type: 'value', min: 0, ...axis({ axisLabel: { formatter: (v) => fmt.pct(v, 1), color: MUTED } }) },
    series: models.map((m, k) => ({ name: m, type: 'line', symbolSize: 7, data: hours.map((h) => rows.find((r) => r.model === m && r.hour === h)?.rate ?? null),
      lineStyle: { color: modelColor(m), width: 2 }, itemStyle: { color: modelColor(m) },
      ...(k === 0 ? { markLine: { silent: true, symbol: 'none', lineStyle: { type: 'dashed', color: '#B5561A' }, label: { formatter: fmt.pct(expected, 0), color: '#B5561A' }, data: [{ yAxis: expected }] } } : {}) })),
  }));
}

function heat(models) {
  const c = chart('#c-heat');
  const rows = comp.by_stock.filter((r) => r.dataset === ds);
  const tickers = [...new Set(rows.map((r) => r.ticker))];
  const data = [];
  let max = 0;
  tickers.forEach((tk, i) => {
    const mine = rows.filter((r) => r.ticker === tk);
    const best = Math.min(...mine.map((r) => r.mean_pinball));
    models.forEach((m, j) => {
      const r = mine.find((x) => x.model === m);
      const excess = 100 * (r.mean_pinball / best - 1);
      max = Math.max(max, excess);
      data.push([j, i, excess, r.exceptions, r.n, r.kupiec_p]);
    });
  });
  c.setOption(baseOption({
    tooltip: { trigger: 'item', formatter: (p) => `<b>${escapeHtml(tickers[p.value[1]])} · ${escapeHtml(models[p.value[0]])}</b><br>
      ${t('cmp.heat.excess')}: +${fmt.num(p.value[2], 1)}%<br>${t('tbl.exc')}: ${p.value[3]}/${p.value[4]} · Kupiec p ${fmt.p(p.value[5])}` },
    grid: { left: 80, right: 30, top: 10, bottom: 70 },
    xAxis: { type: 'category', data: models, ...axis({ axisLabel: { color: MUTED, interval: 0 } }) },
    yAxis: { type: 'category', data: tickers, inverse: true, ...axis({ axisLabel: { color: MUTED } }) },
    visualMap: { dimension: 2, min: 0, max: Math.max(5, Math.ceil(max)), calculable: false, orient: 'horizontal', left: 'center', bottom: 6, itemHeight: 160,
      inRange: { color: ['#FFFFFF', '#FBE3CF', '#F2A65F', '#B5561A'] }, text: [`+${Math.ceil(max)}%`, t('cmp.heat.best')], textStyle: { color: MUTED } },
    series: [{ type: 'heatmap', data, label: { show: true, formatter: (p) => (p.value[2] < 0.05 ? '★' : `+${fmt.num(p.value[2], 1)}%`), color: '#14213D', fontSize: 11 },
      itemStyle: { borderColor: '#fff', borderWidth: 1 } }],
  }));
}
