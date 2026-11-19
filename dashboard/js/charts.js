// ECharts helpers: one look for every chart, the model colors and automatic resizing.
/* global echarts */

export const MODEL_COLORS = {
  'Realized-CAViaR': '#E07A2E',
  'EWMA-N': '#93B7E3',
  'EWMA-N (ref)': '#A8B1BF',
  'EWMA-t': '#3D7CC9',
  'GARCH-t': '#1B3A6B',
};
export const modelColor = (m) => MODEL_COLORS[m] || '#8A94A6';
export const MODEL_ORDER = ['Realized-CAViaR', 'EWMA-N', 'EWMA-t', 'GARCH-t'];
export const sortModels = (ms) => [...ms].sort((a, b) => (MODEL_ORDER.indexOf(a) + 99 * (MODEL_ORDER.indexOf(a) < 0)) - (MODEL_ORDER.indexOf(b) + 99 * (MODEL_ORDER.indexOf(b) < 0)));

export const INK = '#14213D';
export const INK2 = '#33415C';
export const MUTED = '#5A6478';
export const GRID = '#EEF1F5';
export const ACCENT = '#B5561A';
export const BAD = '#B42318';

export function makeChart(el) {
  const chart = echarts.init(el, null, { renderer: 'canvas' });
  new ResizeObserver(() => chart.resize()).observe(el);
  return chart;
}

export const textStyle = { fontFamily: '"Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif', color: INK2 };

export function axis(extra = {}) {
  return {
    axisLine: { lineStyle: { color: '#C9D0DB' } },
    axisTick: { lineStyle: { color: '#C9D0DB' } },
    axisLabel: { color: MUTED },
    splitLine: { lineStyle: { color: GRID } },
    nameTextStyle: { color: MUTED },
    ...extra,
  };
}

export function baseOption(extra = {}) {
  return {
    textStyle,
    animationDuration: 400,
    grid: { left: 60, right: 24, top: 36, bottom: 44 },
    tooltip: { trigger: 'axis', backgroundColor: '#fff', borderColor: '#D5DAE3', textStyle: { color: INK } },
    ...extra,
  };
}

// Wilson 95% interval of k exceptions in n (as in the engines' backtests).
export function wilson(k, n, z = 1.959963984540054) {
  if (!n) return [NaN, NaN];
  const p = k / n, z2 = z * z, den = 1 + z2 / n;
  const centre = (p + z2 / (2 * n)) / den;
  const half = (z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / den;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

export const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
