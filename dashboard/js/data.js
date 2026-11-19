// Precomputed results of the native engines (dashboard/data/*.json, from tools/export_data.py).
const cache = new Map();

export function load(name) {
  if (!cache.has(name)) {
    cache.set(name, fetch(`data/${name}.json`).then((r) => {
      if (!r.ok) throw new Error(`data/${name}.json: HTTP ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(name);
}

// Short key of a model name, for strings and colors: "Realized-CAViaR" -> "realizedcaviar".
export const modelKey = (m) => m.toLowerCase().replace(/[^a-z]/g, '');
