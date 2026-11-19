// Method and limitations: data, conventions, evaluation, limitations, how to reproduce, and the
// plan for level C (live market data). Text only, from strings.js.
import { t } from '../i18n.js';

let el;

export function init(root) { el = root; render(); }

export function render() {
  const block = (key) => `<div class="section card"><h2>${t(`m.${key}.title`)}</h2>${t(`m.${key}.body`)}</div>`;
  el.innerHTML = `
    <h1>${t('m.title')}</h1>
    <p class="lead">${t('m.lead')}</p>
    ${['data', 'conv', 'eval', 'engine', 'limits', 'repro', 'levelc'].map(block).join('')}`;
}
