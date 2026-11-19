// Languages (pt / en): t(key, vars) looks a string up in strings.js; static markup uses
// data-i18n="key" (text) or data-i18n-html="key" (our own trusted markup). The choice is kept
// in localStorage; tabs re-render on the 'lang' event.
import { STRINGS } from './strings.js';

const stored = (() => { try { return localStorage.getItem('lang'); } catch { return null; } })();
let lang = stored === 'en' || stored === 'pt' ? stored : 'pt';
const listeners = new Set();

export const getLang = () => lang;
export const locale = () => (lang === 'pt' ? 'pt-BR' : 'en-US');
export const onLang = (fn) => listeners.add(fn);

export function t(key, vars = {}) {
  const s = STRINGS[lang][key] ?? STRINGS.pt[key] ?? key;
  return s.replace(/\{(\w+)\}/g, (_, k) => (k in vars ? vars[k] : `{${k}}`));
}

export function applyStatic(root = document) {
  root.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
  root.querySelectorAll('[data-i18n-html]').forEach((el) => { el.innerHTML = t(el.dataset.i18nHtml); });
  document.querySelectorAll('[data-lang]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.lang === lang)));
  document.documentElement.lang = locale();
  document.title = t('app.doctitle');
}

export function setLang(next) {
  if (next !== 'pt' && next !== 'en') return;
  lang = next;
  try { localStorage.setItem('lang', lang); } catch { /* private mode: keep it in memory */ }
  applyStatic();
  listeners.forEach((fn) => fn(lang));
}

// Number formatting in the current language.
export const fmt = {
  num(x, digits = 2) {
    if (x == null || !Number.isFinite(x)) return '–';
    return new Intl.NumberFormat(locale(), { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(x);
  },
  int(x) { return x == null ? '–' : new Intl.NumberFormat(locale()).format(x); },
  pct(x, digits = 2) {
    if (x == null || !Number.isFinite(x)) return '–';
    return new Intl.NumberFormat(locale(), { style: 'percent', minimumFractionDigits: digits, maximumFractionDigits: digits }).format(x);
  },
  p(x) {
    if (x == null || !Number.isFinite(x)) return '–';
    return x < 0.001 ? `< ${fmt.num(0.001, 3)}` : fmt.num(x, x < 0.01 ? 4 : 3);
  },
  // forecast times are exchange-local wall-clock seconds, formatted as UTC to keep the clock
  time(sec, withDate = true) {
    if (sec == null || !Number.isFinite(sec)) return '–';
    const o = { timeZone: 'UTC', hour: '2-digit', minute: '2-digit' };
    if (withDate) Object.assign(o, { day: '2-digit', month: '2-digit' });
    return new Intl.DateTimeFormat(locale(), o).format(new Date(sec * 1000));
  },
  date(sec) {
    return new Intl.DateTimeFormat(locale(), { timeZone: 'UTC', day: '2-digit', month: '2-digit', year: 'numeric' }).format(new Date(sec * 1000));
  },
};
