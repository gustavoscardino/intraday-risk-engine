// Tabs (hash router), language switch and lazy initialization: a tab is built the first time
// it is shown (its charts need a visible container) and re-rendered when the language changes.
import { applyStatic, onLang, setLang, t } from './i18n.js';
import { load } from './data.js';
import * as overview from './tabs/overview.js';
import * as how from './tabs/how.js';
import * as replay from './tabs/replay.js';
import * as compare from './tabs/compare.js';
import * as quality from './tabs/quality.js';
import * as method from './tabs/method.js';

const TABS = { overview, how, replay, compare, quality, method };
const started = new Map();

function show(name) {
  if (!TABS[name]) name = 'overview';
  for (const key of Object.keys(TABS)) {
    document.getElementById(`tab-${key}`).hidden = key !== name;
    const link = document.querySelector(`[data-tab="${key}"]`);
    if (key === name) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  }
  if (!started.has(name)) {
    const el = document.getElementById(`tab-${name}`);
    started.set(name, Promise.resolve(TABS[name].init(el)).catch((err) => {
      el.innerHTML = `<div class="note warn">${t('app.error')}: ${String(err.message || err)}</div>`;
      console.error(err);
    }));
  }
}

function footer() {
  load('meta').then((m) => {
    document.getElementById('foot-meta').textContent = t('app.footmeta', { date: m.generated, commit: m.commit });
  }).catch(() => {});
}

document.querySelectorAll('[data-lang]').forEach((b) => b.addEventListener('click', () => setLang(b.dataset.lang)));
onLang(() => {
  footer();
  for (const [name, ready] of started) ready.then(() => TABS[name].render?.());
});
window.addEventListener('hashchange', () => show(location.hash.slice(1)));
applyStatic();
footer();
show(location.hash.slice(1) || 'overview');
