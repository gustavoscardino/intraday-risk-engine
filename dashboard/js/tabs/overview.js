// Overview: the research question, one card per model and the findings, computed from the
// native comparison over every stock (data/comparison.json).
import { fmt, t } from '../i18n.js';
import { load, modelKey } from '../data.js';
import { escapeHtml, modelColor, sortModels } from '../charts.js';

let el;
let comp;

export async function init(root) {
  el = root;
  comp = await load('comparison');
  render();
}

const row = (ds, m) => comp.summary.find((s) => s.dataset === ds && s.model === m);
function dm(ds, a, b) {
  const r = comp.diebold_mariano.find((x) => x.dataset === ds && x.scope === 'pooled'
    && ((x.model_a === a && x.model_b === b) || (x.model_a === b && x.model_b === a)));
  return r ? { ...r, sign: r.model_a === a ? 1 : -1 } : null;
}

function card(m) {
  const s = row('synthetic', m), b = row('b3_real', m);
  return `<div class="card kpi">
    <div class="model"><span class="dot" style="background:${modelColor(m)}"></span>${escapeHtml(m)}</div>
    <p class="sub">${t(`model.${modelKey(m)}.short`)}</p>
    <div class="big">${fmt.pct(s.rate)} <span class="sub">· ${fmt.pct(b.rate)}</span></div>
    <p class="sub">${t('ov.kpi.rate')}</p>
    <dl>
      <dt>${t('ov.kpi.kupiec')}</dt><dd>${s.kupiec_rejections}/${s.stocks} · ${b.kupiec_rejections}/${b.stocks}</dd>
      <dt>${t('ov.kpi.pinball')}</dt><dd>${fmt.num(1e4 * s.mean_pinball, 3)} · ${fmt.num(1e4 * b.mean_pinball, 3)}</dd>
      <dt>${t('ov.kpi.best')}</dt><dd>${s.lowest_pinball_stocks} · ${b.lowest_pinball_stocks}</dd>
      <dt>${t('ov.kpi.cost')}</dt><dd>${fmt.num(b.median_latency_ns, 0)} ns · ${fmt.int(b.state_bytes)} B</dd>
    </dl>
  </div>`;
}

export function render() {
  const models = sortModels([...new Set(comp.summary.map((s) => s.model))]);
  const en = row('synthetic', 'EWMA-N'), et = row('synthetic', 'EWMA-t'), gs = row('synthetic', 'GARCH-t');
  const cb = row('b3_real', 'Realized-CAViaR');
  const vsN = dm('b3_real', 'Realized-CAViaR', 'EWMA-N'), vsG = dm('b3_real', 'Realized-CAViaR', 'GARCH-t');
  const cheapest = [...comp.summary.filter((s) => s.dataset === 'b3_real')].sort((a, b) => a.median_latency_ns - b.median_latency_ns)[0];
  const best = [...comp.summary.filter((s) => s.dataset === 'b3_real')].sort((a, b) => a.mean_pinball - b.mean_pinball)[0];
  const findings = [
    t('ov.f.tail', { n: en.kupiec_rejections, t: et.kupiec_rejections, stocks: en.stocks, rn: fmt.pct(en.rate), rt: fmt.pct(et.rate) }),
    t('ov.f.garch', { rate: fmt.pct(gs.rate), rej: gs.kupiec_rejections, stocks: gs.stocks }),
    t('ov.f.b3', { best: escapeHtml(best.model), rate: fmt.pct(cb.rate), pN: fmt.p(vsN?.p_value), pG: fmt.p(vsG?.p_value) }),
    t('ov.f.cost', { model: escapeHtml(cheapest.model), ns: fmt.num(cheapest.median_latency_ns, 0), bytes: cheapest.state_bytes }),
    t('ov.f.caveat'),
  ];
  el.innerHTML = `
    <h1>${t('ov.title')}</h1>
    <p class="lead">${t('ov.lead')}</p>
    <div class="note callout">${t('ov.question')}</div>
    <div class="section">
      <h2>${t('ov.models')}</h2>
      <p class="muted">${t('ov.models.lead')}</p>
      <div class="grid grid-4">${models.map(card).join('')}</div>
    </div>
    <div class="section grid grid-2">
      <div class="card"><h3>${t('ov.findings')}</h3><ul>${findings.map((f) => `<li>${f}</li>`).join('')}</ul></div>
      <div class="card"><h3>${t('ov.where')}</h3>
        <ul>
          <li>${t('ov.go.how')}</li>
          <li>${t('ov.go.replay')}</li>
          <li>${t('ov.go.compare')}</li>
          <li>${t('ov.go.quality')}</li>
          <li>${t('ov.go.method')}</li>
        </ul>
        <p class="small muted">${t('ov.source', { code: escapeHtml(String(comp.code).replace(/\.+$/, '')) })}</p>
      </div>
    </div>
    <div class="section">
      <h2>${t('ov.arch')}</h2>
      <p class="muted">${t('ov.arch.lead')}</p>
      <img class="diagram" src="assets/00_visao_geral.svg" alt="${escapeHtml(t('ov.arch.alt'))}">
    </div>`;
}
