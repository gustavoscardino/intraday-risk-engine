// Web Worker that hosts the WebAssembly engine. Messages: {id, fn, args} -> {id, out | error}.
import createEngine from '../engine/rcaviar.js';

// Emscripten resolves the .data package against the worker's own folder (js/); point every
// engine file to engine/ instead.
const ready = createEngine({ locateFile: (file) => new URL(`../engine/${file}`, import.meta.url).href });

self.onmessage = async (e) => {
  const { id, fn, args } = e.data;
  try {
    const m = await ready;
    if (typeof m[fn] !== 'function') throw new Error(`unknown engine function ${fn}`);
    self.postMessage({ id, out: JSON.parse(m[fn](...args)) });
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err) });
  }
};
