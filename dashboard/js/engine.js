// The C++ engine compiled to WebAssembly (engine/rcaviar.*), run in a Web Worker so the page
// stays responsive while a stock is replayed (a few seconds for 150 sessions).
let worker = null;
let seq = 0;
const pending = new Map();
const runs = new Map();

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./engine.worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (e) => {
      const { id, out, error } = e.data;
      const p = pending.get(id);
      pending.delete(id);
      if (!p) return;
      if (error) p.reject(new Error(error)); else p.resolve(out);
    };
    worker.onerror = (e) => {
      for (const p of pending.values()) p.reject(new Error(e.message || 'engine worker failed'));
      pending.clear();
      worker = null;
    };
  }
  return worker;
}

function call(fn, args) {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, fn, args });
  });
}

export const DEFAULT_RUN = {
  dataset: 'b3_real', ticker: 'PETR4', windowDays: 30, confidence: 0.99,
  lambda: 0, diurnal: true, exposure: 100000, benchmarkUpdates: 200000,
};

// One matched replay of a stock (four models). lambda <= 0: estimated at each calibration.
// Results are cached per parameter set for the session.
export function runComparison(params) {
  const p = { ...DEFAULT_RUN, ...params };
  const key = JSON.stringify(p);
  if (!runs.has(key)) {
    const job = call('run', [p.dataset, p.ticker, p.windowDays, p.confidence, p.lambda, p.diurnal, p.exposure, p.benchmarkUpdates])
      .then((r) => { if (r.error) throw new Error(r.error); return r; });
    job.catch(() => runs.delete(key));
    runs.set(key, job);
  }
  return runs.get(key);
}

// VaR / ES factors per unit of sigma: Normal (nu <= 0) or unit-variance Student t.
export const tailFactors = (nu, confidence) => call('tail', [nu, confidence]);
